from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import typing

import h5py
import matplotlib.pyplot as plt
import numpy as np

from .evol import AdvDiffSource, Diffusion, DonorCellAdvection, EulerExplicit
from .init import StartFileIC
from .rheology import Arrhenius, ConstantVisco, Rheology
from .stokes import StokesMatrix, StokesRHS

if typing.TYPE_CHECKING:
    from typing import Optional, Callable
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


class StokesState:

    """Represent a consistent physical state.

    Velocity, pressure, and viscosity fields are all consistent with a given
    temperature field.
    """

    def __init__(
        self, temp: NDArray, grid: Grid, rheology: Rheology, conf: Config
    ):
        self._conf = conf
        self._lumat: Optional[Callable[[NDArray], NDArray]] = None
        self.grid = grid
        self.rheology = rheology
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

    def _solve_stokes(self) -> None:
        """Solve the Stokes equation for a given temperature field."""
        self._visco = self.rheology.visco(self.temp)

        stokes_rhs = StokesRHS(grid=self.grid, ranum=self._conf.physical.ranum)
        rhs = stokes_rhs.eval(self.temp)

        if self._conf.physical.var_visc or self._lumat is None:
            stokes_mat = StokesMatrix(
                grid=self.grid,
                periodic=self._conf.physical.periodic,
            )
            # FIXME: let a Rheology object handle viscosity
            spm = stokes_mat.eval(self.viscosity)
            self._lumat = spm.lu_solver()

        sol = self._lumat(rhs)

        n_x = self.grid.n_x
        n_z = self.grid.n_z
        self._v_x = np.reshape(sol[::3], (n_z, n_x)).T
        # remove drift velocity (unconstrained)
        if self._conf.physical.periodic:
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
        heat_eq = EulerExplicit(AdvDiffSource(
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
        ))
        self.temp = heat_eq.apply_dt_cfl(self.temp)
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

        rheology = Arrhenius(
            temp_factor=conf.physical.var_visc_temp,
            depth_factor=conf.physical.var_visc_depth,
            grid=self.grid,
        ) if conf.physical.var_visc else ConstantVisco()
        self.state = StokesState(temp, self.grid, rheology, conf)

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
