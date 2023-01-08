from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import typing

from scipy.sparse.linalg import factorized
import h5py
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np

from .evol import Diffusion, DonorCellAdvection, TimeEvolEquation
from .init import StartFileIC

if typing.TYPE_CHECKING:
    from typing import Optional, Callable, List
    from numpy.typing import NDArray
    from .config import Config


_NTSERIES = 9


# Many parts of the code assume a grid with constant spacing and the same
# spacing in both directions, this should be fixed.
@dataclass(frozen=True)
class Grid:
    n_x: int
    n_z: int

    @property
    def aspect_ratio(self) -> float:
        return self.n_x / self.n_z

    @property
    def d_x(self) -> float:
        return self.d_z

    @property
    def d_z(self) -> float:
        return 1 / self.n_z

    @cached_property
    def x_walls(self) -> NDArray[np.float64]:
        return np.linspace(0., self.aspect_ratio, self.n_x + 1)

    @cached_property
    def z_walls(self) -> NDArray[np.float64]:
        return np.linspace(0., 1., self.n_z + 1)

    @cached_property
    def x_centers(self) -> NDArray[np.float64]:
        return (self.x_walls[:-1] + self.x_walls[1:]) / 2

    @cached_property
    def z_centers(self) -> NDArray[np.float64]:
        return (self.z_walls[:-1] + self.z_walls[1:]) / 2


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
        visco: np.ndarray, ix: int, iz: int, periodic: bool
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


class StokesState:

    """Represent a consistent physical state.

    Velocity, pressure, and viscosity fields are all consistent with a given
    temperature field.
    """

    def __init__(self, temp: NDArray, grid: Grid, conf: Config):
        self._conf = conf
        self._lumat: Optional[Callable[[NDArray], NDArray]] = None
        self.grid = grid
        self.temp = temp

    @property
    def temp(self) -> NDArray:
        return self._temp

    @temp.setter
    def temp(self, field: NDArray) -> None:
        self._temp = field
        # keep the state self-consistent
        self._solve_stokes()

    @property
    def viscosity(self) -> NDArray:
        """Viscosity field."""
        return self._visco

    @property
    def v_x(self) -> NDArray:
        """Horizontal velocity."""
        return self._v_x

    @property
    def v_z(self) -> NDArray:
        """Vertical velocity."""
        return self._v_z

    @property
    def dynp(self) -> NDArray:
        """Dynamic pressure."""
        return self._dynp

    def _eval_viscosity(self) -> None:
        """Compute viscosity for a given temperature field."""
        grd = self.grid
        if self._conf.physical.var_visc:
            a_visc = np.log(self._conf.physical.var_visc_temp)
            b_visc = np.log(self._conf.physical.var_visc_depth)
            depth = 0.5 - grd.z_centers
            self._visco = np.exp(-a_visc * (self.temp - 0.5) + b_visc * depth)
        else:
            self._visco = np.ones((grd.n_x, grd.n_z))

    def _solve_stokes(self) -> None:
        """Solve the Stokes equation for a given temperature field."""
        self._eval_viscosity()
        n_x = self.grid.n_x
        n_z = self.grid.n_z
        periodic = self._conf.physical.periodic
        rhsz = np.zeros((n_x, n_z))
        rhsz[:, 1:] = -self._conf.physical.ranum * (
            self.temp[:, :-1] + self.temp[:, 1:]) / 2

        odz = 1 / self.grid.d_z
        odz2 = odz**2

        rhs = np.zeros(n_x * n_z * 3)
        # indices offset
        idx = 3
        idz = n_x * 3

        spm = SparseMatrix(rhs.size)

        for iz in range(n_z):
            for ix in range(n_x):
                # define indices in the matrix
                icell = ix + iz * n_x
                ieqx = icell * 3
                ieqz = ieqx + 1
                ieqc = ieqx + 2
                ieqxp = (ieqx + idx) % rhs.size if periodic else ieqx + idx
                ieqzp = ieqxp + 1
                ieqxpm = (ieqxp - idz) % rhs.size if periodic else ieqxp - idz
                ieqxm = (ieqx - idx) % rhs.size if periodic else ieqx - idx
                ieqzm = ieqxm + 1
                ieqcm = ieqxm + 2

                eta = ViscoStencil.eval_at(self.viscosity, ix, iz,
                                           self._conf.physical.periodic)

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
                    rhs[ieqx] = 0
                else:
                    spm.coef(ieqx, ieqx, 1)
                    rhs[ieqx] = 0

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
                    rhs[ieqz] = rhsz[ix, iz]
                else:
                    spm.coef(ieqz, ieqz, 1)
                    rhs[ieqz] = 0

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
                rhs[ieqc] = 0

        if self._conf.physical.var_visc or self._lumat is None:
            self._lumat = spm.lu_solver()
        sol = self._lumat(rhs)
        self._v_x = np.reshape(sol[::3], (n_z, n_x)).T
        # remove drift velocity (unconstrained)
        if periodic:
            self._v_x -= np.mean(self._v_x)
        self._v_z = np.reshape(sol[1::3], (n_z, n_x)).T
        self._dynp = np.reshape(sol[2::3], (n_z, n_x)).T
        self._dynp -= np.mean(self._dynp)

    def step_forward(self) -> float:
        """Update state according to heat equation.

        Note that the Stokes equation is solved as well to keep the state
        self-consistent.

        Returns:
            the dt used to forward the state.
        """
        heat_eq = TimeEvolEquation(
            diff=Diffusion(
                grid=self.grid,
                periodic=self._conf.physical.periodic
            ),
            adv=DonorCellAdvection(
                grid=self.grid,
                v_x=self.v_x,
                v_z=self.v_z,
                periodic=self._conf.physical.periodic,
            ),
            source=self._conf.physical.int_heat,
        )
        self.temp = self.temp + heat_eq.dt_cfl * heat_eq.eval(self.temp)
        return heat_eq.dt_cfl


class RunManager:

    """Simulation of the Rayleigh Bénard convection problem."""

    def __init__(self, conf: Config):
        self._conf = conf

        self._fstart = None
        self.grid = Grid(n_x=conf.numerical.n_x, n_z=conf.numerical.n_z)
        if self.conf.numerical.restart:
            try:
                self._fstart = max(self.conf.inout.outdir.glob('fields*.npz'))
            except ValueError:
                pass
        if self._fstart is not None:
            print(f"restarting from {self._fstart}")
            init_cond = StartFileIC(self._fstart).build_ic(self.grid)
        else:
            init_cond = conf.physical.init_cond.build_ic(self.grid)

        self._istart = init_cond.istart
        self.time = init_cond.time
        temp = init_cond.temperature

        self.state = StokesState(temp, self.grid, conf)

    @property
    def conf(self) -> Config:
        """Configuration of solver."""
        return self._conf

    def _outfile(
        self, name: str, istep: int, ext: Optional[str] = None
    ) -> Path:
        fname = '{}{:08d}'.format(name, istep)
        if ext is not None:
            fname += '.{}'.format(ext)
        return self.conf.inout.outdir / fname

    def _save(self, istep: int) -> None:
        fname = self._outfile('fields', istep, 'npz')
        np.savez(fname, T=self.state.temp, vx=self.state.v_x,
                 vz=self.state.v_z, p=self.state.dynp, time=self.time,
                 step=istep)

        if not self._conf.inout.figures:
            return
        grd = self.grid
        fig, axis = plt.subplots()
        # first plot the temperature field
        surf = axis.pcolormesh(grd.x_walls, grd.z_walls, self.state.temp.T,
                               rasterized=True, cmap='RdBu_r')
        cbar = plt.colorbar(surf, shrink=0.5)
        cbar.set_label('Temperature')
        plt.axis('equal')
        plt.axis('off')
        axis.set_adjustable('box')
        axis.set_xlim(0, grd.aspect_ratio)
        axis.set_ylim(0, 1)
        # interpolate velocities at the same points for streamlines
        u_x = 0.5 * (self.state.v_x + np.roll(self.state.v_x, 1, 0))
        u_z = np.zeros_like(self.state.v_z)
        u_z[1:] = self.state.v_z[:-1]
        u_z += self.state.v_z
        u_z *= 0.5
        speed = np.sqrt(u_x ** 2 + u_z ** 2)
        # plot the streamlines
        lw = 2 * speed / speed.max()
        axis.streamplot(grd.x_centers, grd.z_centers, u_x.T, u_z.T, color='k',
                        linewidth=lw.T)
        fig.savefig(self._outfile('T_v', istep, 'pdf'), bbox_inches='tight')
        plt.close(fig)

    def _timeseries(self, istep: int) -> NDArray:
        """Time series diagnostic for one step."""
        grd = self.grid
        tseries = np.empty(_NTSERIES)
        tseries[0] = istep
        tseries[1] = self.time
        tseries[2] = np.amin(self.state.temp)
        tseries[3] = np.mean(self.state.temp)
        tseries[4] = np.amax(self.state.temp)
        ekin = np.mean(self.state.v_x ** 2 + self.state.v_z ** 2)
        tseries[5] = np.sqrt(ekin)
        tseries[6] = np.sqrt(np.mean(self.state.v_x[:, grd.n_z - 1] ** 2))
        tseries[7] = 2 * (1 - np.mean(self.state.temp[:, 0])) / grd.d_z
        tseries[8] = 2 * np.mean(self.state.temp[:, grd.n_z - 1]) / grd.d_z
        return tseries

    def solve(self, progress: bool = False) -> None:
        """Resolution of asked problem.

        Args:
            progress (bool): output progress.
        """
        self.conf.inout.outdir.mkdir(exist_ok=True)
        if self._fstart is None:
            self._save(0)
        tfilename = self.conf.inout.outdir / 'time.h5'
        if self._fstart is None or not tfilename.exists():
            tfile = h5py.File(tfilename, 'w')
            dset = tfile.create_dataset('series', (1, _NTSERIES),
                                        maxshape=(None, _NTSERIES),
                                        data=self._timeseries(self._istart))
        else:
            tfile = h5py.File(tfilename, 'a')
            dset = tfile['series']

        nsteps = self.conf.numerical.nsteps
        nwrite = self.conf.numerical.nwrite
        step_msg = '\rstep: {{:{}d}}/{}'.format(len(str(nsteps)), nsteps)
        tseries = np.zeros((nwrite, _NTSERIES))

        for irun, istep in enumerate(range(self._istart + 1, nsteps + 1)):
            if progress:
                print(step_msg.format(istep), end='')
            dtime = self.state.step_forward()
            self.time += dtime
            tseries[irun % nwrite] = self._timeseries(istep)
            if (irun + 1) % nwrite == 0:
                self._save(istep)
                dset.resize((len(dset) + nwrite, _NTSERIES))
                dset[-nwrite:] = tseries
        if progress:
            print()
        tfile.close()

    def dump_pars(self, parfile: Path) -> None:
        """Dump configuration.

        Args:
            parfile: path of the par file where the configuration should be
                dumped.
        """
        self.conf.to_file_(parfile)
