from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

    from .solver import Grid


@dataclass(frozen=True)
class Diffusion:
    grid: Grid
    periodic: bool  # should be generalized for any BC, via GC of fed scalar
    cfl_factor: float = 0.4

    @cached_property
    def dt_cfl(self) -> float:
        return self.cfl_factor * self.grid.d_z**2

    def eval(self, scalar: NDArray) -> NDArray:
        """Computes Laplacian of scalar

        zero flux BC on the vertical sides for non-periodic cases
        Dirichlet = 0 at the top
        Dirichlet = 1 at the bottom
        """
        grd = self.grid
        delsqT = np.zeros_like(scalar)
        # should be generalized for non-square grids
        for i in range(grd.n_x):
            if self.periodic:
                im = (i - 1 + grd.n_x) % grd.n_x
                ip = (i + 1) % grd.n_x
            else:
                im = max(i - 1, 0)
                ip = min(i + 1, grd.n_x - 1)

            for j in range(0, grd.n_z):
                T_xm = scalar[im, j]
                T_xp = scalar[ip, j]
                if j == 0:  # enforce bottom BC
                    T_zm = 2 - scalar[i, j]
                else:
                    T_zm = scalar[i, j - 1]
                if j == grd.n_z - 1:
                    T_zp = - scalar[i, j]
                else:
                    T_zp = scalar[i, j + 1]

                delsqT[i, j] = (T_xm + T_xp + T_zm + T_zp -
                                4 * scalar[i, j]) / grd.d_z**2

        return delsqT


@dataclass(frozen=True)
class DonorCellAdvection:
    grid: Grid
    v_x: NDArray
    v_z: NDArray
    periodic: bool  # should be generalized for any BC, via GC of fed scalar
    cfl_factor: float = 0.4

    @cached_property
    def dt_cfl(self) -> float:
        vmax = np.maximum(np.amax(np.abs(self.v_x)), np.amax(np.abs(self.v_z)))
        return self.cfl_factor * self.grid.d_z / vmax

    def eval(self, scalar: NDArray) -> NDArray:
        """Donor cell advection div(v T)."""
        dscalar = np.zeros_like(scalar)
        v_x = self.v_x
        v_z = self.v_z
        grd = self.grid

        for i in range(grd.n_x):
            if self.periodic:
                im = (i - 1 + grd.n_x) % grd.n_x
                ip = (i + 1) % grd.n_x
            else:
                im = max(i - 1, 0)
                ip = min(i + 1, grd.n_x - 1)

            for j in range(grd.n_z):
                if i > 0 or self.periodic:
                    flux_xm = scalar[im, j] * v_x[i, j] if v_x[i, j] > 0 else\
                        scalar[i, j] * v_x[i, j]
                else:
                    flux_xm = 0

                if i < grd.n_x - 1 or self.periodic:
                    flux_xp = scalar[i, j] * v_x[ip, j] if v_x[ip, j] > 0 else\
                        scalar[ip, j] * v_x[ip, j]
                else:
                    flux_xp = 0

                if j > 0:
                    flux_zm = scalar[i, j - 1] * v_z[i, j] \
                        if v_z[i, j] > 0 else scalar[i, j] * v_z[i, j]
                else:
                    flux_zm = 0

                if j < grd.n_z - 1:
                    flux_zp = scalar[i, j] * v_z[i, j + 1] \
                        if v_z[i, j + 1] >= 0 else \
                        scalar[i, j + 1] * v_z[i, j + 1]
                else:
                    flux_zp = 0
                dscalar[i, j] = (
                    flux_xm - flux_xp + flux_zm - flux_zp) / grd.d_z
                # assumes d_x = d_z. To be generalized
        return dscalar


@dataclass(frozen=True)
class TimeEvolEquation:
    diff: Diffusion
    adv: DonorCellAdvection
    source: float

    @cached_property
    def dt_cfl(self) -> float:
        return min(self.diff.dt_cfl, self.adv.dt_cfl)

    def eval(self, scalar: NDArray) -> NDArray:
        """Time derivative of temperature."""
        return self.diff.eval(scalar) + self.adv.eval(scalar) + self.source
