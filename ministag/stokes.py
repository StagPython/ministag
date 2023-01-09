from __future__ import annotations

from dataclasses import dataclass
import typing

from scipy.sparse.linalg import factorized
import numpy as np
import scipy.sparse as sp

if typing.TYPE_CHECKING:
    from typing import Callable, List

    from numpy.typing import NDArray

    from .solver import Grid


@dataclass(frozen=True)
class ViscoStencil:
    """Viscosity values around a given point."""

    ctr: float
    x_m: float
    z_m: float
    xz_c: float
    xz_xp: float
    xz_zp: float

    @staticmethod
    def eval_at(
        visco: NDArray, ix: int, iz: int, periodic: bool
    ) -> ViscoStencil:
        """Viscosity around a grid point."""
        n_x, n_z = visco.shape

        # these won't be used around the boundaries if not periodic
        ixm = (ix - 1) % n_x
        ixp = (ix + 1) % n_x

        etaii_c = visco[ix, iz]
        etaii_xm = visco[ixm, iz] if ix > 0 or periodic else 0
        etaii_zm = visco[ix, iz - 1] if iz > 0 else 0
        if (ix > 0 or periodic) and iz > 0:
            etaxz_c = (visco[ix, iz] * visco[ixm, iz] *
                       visco[ix, iz - 1] * visco[ixm, iz - 1])**0.25
        else:
            etaxz_c = 0
        if (ix > 0 or periodic) and iz < n_z - 1:
            etaxz_zp = (visco[ix, iz + 1] * visco[ixm, iz + 1] *
                        visco[ix, iz] * visco[ixm, iz])**0.25
        else:
            etaxz_zp = 0
        if (ix < n_x - 1 or periodic) and iz > 0:
            etaxz_xp = (visco[ixp, iz] * visco[ix, iz] *
                        visco[ixp, iz - 1] * visco[ix, iz - 1])**0.25
        else:
            etaxz_xp = 0
        return ViscoStencil(ctr=etaii_c, x_m=etaii_xm, z_m=etaii_zm,
                            xz_c=etaxz_c, xz_xp=etaxz_xp, xz_zp=etaxz_zp)


class SparseMatrix:

    """Sparse matrix."""

    def __init__(self, size: int):
        self._rows: List[int] = []
        self._cols: List[int] = []
        self._coefs: List[float] = []
        self._size = size

    def coef(self, irow: int, icol: int, value: float) -> None:
        """Add a new coefficient in the matrix.

        Args:
            irow: row index.
            icol: column index.
            value: value of the coefficient.
        """
        self._rows.append(irow)
        self._cols.append(icol)
        self._coefs.append(value)

    def lu_solver(self) -> Callable[[NDArray], NDArray]:
        """Return a solver based on LU factorization."""
        return factorized(
            sp.csc_matrix((self._coefs, (self._rows, self._cols)),
                          shape=(self._size, self._size)))


@dataclass(frozen=True)
class StokesRHS:
    grid: Grid
    ranum: float
    # FIXME: should handle arbitrary BCs

    def eval(self, temp: NDArray) -> NDArray:
        n_x = self.grid.n_x
        n_z = self.grid.n_z

        # Buoyancy -Ra * T, evaluated at vz points
        # note that rhsz[:, 0] == 0.  This is fine
        # since the equation for these points is
        # vz=0.
        rhsz = np.zeros((n_x, n_z))
        rhsz[:, 1:] = -self.ranum * (
            temp[:, :-1] + temp[:, 1:]) / 2

        # RHS is non-zero only along z (rhsz):
        # - for vx (x-momentum): forcing is 0 (gravity along z), or BC is vx=0
        # - for vz (z-momentum): rhsz forcing, or BC is vz=0
        # - for p (continuity): div v = 0, or p = 0 in one cell (for closure)
        rhs = np.zeros(n_x * n_z * 3)
        for iz in range(1, n_z):
            for ix in range(n_x):
                icell = ix + iz * n_x
                ieqx = icell * 3
                ieqz = ieqx + 1
                rhs[ieqz] = rhsz[ix, iz]

        return rhs


@dataclass(frozen=True)
class StokesMatrix:
    grid: Grid
    periodic: bool  # should be generic over BCs

    def eval(self, viscosity: NDArray) -> SparseMatrix:
        n_x = self.grid.n_x
        n_z = self.grid.n_z
        periodic = self.periodic

        odz = 1 / self.grid.d_z
        odz2 = odz**2

        # indices offset
        idx = 3
        idz = n_x * 3

        nvars = n_x * n_z * 3
        spm = SparseMatrix(nvars)

        for iz in range(n_z):
            for ix in range(n_x):
                # define indices in the matrix
                icell = ix + iz * n_x
                ieqx = icell * 3
                ieqz = ieqx + 1
                ieqc = ieqx + 2
                ieqxp = (ieqx + idx) % nvars if periodic else ieqx + idx
                ieqzp = ieqxp + 1
                ieqxpm = (ieqxp - idz) % nvars if periodic else ieqxp - idz
                ieqxm = (ieqx - idx) % nvars if periodic else ieqx - idx
                ieqzm = ieqxm + 1
                ieqcm = ieqxm + 2

                eta = ViscoStencil.eval_at(viscosity, ix, iz, self.periodic)

                # x-momentum
                if ix > 0 or periodic:
                    spm.coef(ieqx, ieqx, -odz2 * (2 * eta.ctr + 2 * eta.x_m +
                                                  eta.xz_c + eta.xz_zp))
                    spm.coef(ieqx, ieqxm, 2 * odz2 * eta.x_m)
                    spm.coef(ieqx, ieqz, -odz2 * eta.xz_c)
                    spm.coef(ieqx, ieqzm, odz2 * eta.xz_c)
                    spm.coef(ieqx, ieqc, -odz)
                    spm.coef(ieqx, ieqcm, odz)

                    if ix + 1 < n_x or periodic:
                        spm.coef(ieqx, ieqxp, 2 * odz2 * eta.ctr)
                    if iz + 1 < n_z:
                        spm.coef(ieqx, ieqx + idz, odz2 * eta.xz_zp)
                        spm.coef(ieqx, ieqz + idz, odz2 * eta.xz_zp)
                        spm.coef(ieqx, ieqz + idz - idx, -odz2 * eta.xz_zp)
                    if iz > 0:
                        spm.coef(ieqx, ieqx - idz, odz2 * eta.xz_c)
                else:
                    spm.coef(ieqx, ieqx, 1)

                # z-momentum
                if iz > 0:
                    spm.coef(ieqz, ieqz, -odz2 * (2 * eta.ctr + 2 * eta.z_m +
                                                  eta.xz_c + eta.xz_xp))
                    spm.coef(ieqz, ieqz - idz, 2 * odz2 * eta.z_m)
                    spm.coef(ieqz, ieqx, -odz2 * eta.xz_c)
                    spm.coef(ieqz, ieqx - idz, odz2 * eta.xz_c)
                    spm.coef(ieqz, ieqc, -odz)
                    spm.coef(ieqz, ieqc - idz, odz)

                    if iz + 1 < n_z:
                        spm.coef(ieqz, ieqz + idz, 2 * odz2 * eta.ctr)
                    if ix + 1 < n_x or periodic:
                        spm.coef(ieqz, ieqzp, odz2 * eta.xz_xp)
                        spm.coef(ieqz, ieqxp, odz2 * eta.xz_xp)
                        spm.coef(ieqz, ieqxpm, -odz2 * eta.xz_xp)
                    if ix > 0 or periodic:
                        spm.coef(ieqz, ieqzm, odz2 * eta.xz_c)
                else:
                    spm.coef(ieqz, ieqz, 1)

                # continuity
                if ix == 0 and iz == 0:
                    spm.coef(ieqc, ieqc, 1)
                else:
                    spm.coef(ieqc, ieqx, -odz)
                    spm.coef(ieqc, ieqz, -odz)
                    if ix + 1 < n_x or periodic:
                        spm.coef(ieqc, ieqxp, odz)
                    if iz + 1 < n_z:
                        spm.coef(ieqc, ieqz + idz, odz)

        return spm
