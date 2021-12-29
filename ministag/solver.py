from __future__ import annotations
from pathlib import Path
import typing

from scipy.sparse.linalg import factorized
import h5py
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np

if typing.TYPE_CHECKING:
    from typing import Optional, Callable, Tuple, List
    from numpy import ndarray
    from .config import Config


_NTSERIES = 9


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

    def lu_solver(self) -> Callable[[ndarray], ndarray]:
        """Return a solver based on LU factorization."""
        return factorized(
            sp.csc_matrix((self._coefs, (self._rows, self._cols)),
                          shape=(self._size, self._size)))


class StokesState:

    """Represent a consistent physical state.

    Velocity, pressure, and viscosity fields are all consistent with a given
    temperature field.
    """

    def __init__(self, temp: ndarray, conf: Config):
        self._conf = conf
        self._lumat: Optional[Callable[[ndarray], ndarray]] = None
        self.temp = temp

    @property
    def temp(self) -> ndarray:
        return self._temp

    @temp.setter
    def temp(self, field: ndarray) -> None:
        self._temp = field
        # keep the state self-consistent
        self._solve_stokes()

    @property
    def viscosity(self) -> ndarray:
        """Viscosity field."""
        return self._visco

    @property
    def v_x(self) -> ndarray:
        """Horizontal velocity."""
        return self._v_x

    @property
    def v_z(self) -> ndarray:
        """Vertical velocity."""
        return self._v_z

    @property
    def dynp(self) -> ndarray:
        """Dynamic pressure."""
        return self._dynp

    def _eval_viscosity(self) -> None:
        """Compute viscosity for a given temperature field."""
        n_x = self._conf.numerical.n_x
        n_z = self._conf.numerical.n_z
        if self._conf.physical.var_visc:
            d_z = 1 / n_z
            a_visc = np.log(self._conf.physical.var_visc_temp)
            b_visc = np.log(self._conf.physical.var_visc_depth)
            depth = 0.5 - np.linspace(0.5 * d_z, 1 - 0.5 * d_z, n_z)
            self._visco = np.exp(-a_visc * (self.temp - 0.5) + b_visc * depth)
        else:
            self._visco = np.ones((n_x, n_z))

    def _visco_around(self, ix: int, iz: int) -> Tuple[float, ...]:
        """Viscosity around a grid point."""
        n_x = self._conf.numerical.n_x
        n_z = self._conf.numerical.n_z
        periodic = self._conf.physical.periodic
        # deal with boundary conditions on vertical planes
        if periodic:
            ixm = (ix - 1 + n_x) % n_x
            ixp = (ix + 1) % n_x
        else:
            ixm = max(ix - 1, 0)
            ixp = min(ix + 1, n_x - 1)

        etaii_c = self.viscosity[ix, iz]
        etaii_xm = self.viscosity[ixm, iz] if ix > 0 or periodic else 0
        etaii_zm = self.viscosity[ix, iz - 1] if iz > 0 else 0
        if (ix > 0 or periodic) and iz > 0:
            etaxz_c = (self.viscosity[ix, iz] * self.viscosity[ixm, iz] *
                       self.viscosity[ix, iz - 1] *
                       self.viscosity[ixm, iz - 1])**0.25
        else:
            etaxz_c = 0
        if (ix > 0 or periodic) and iz < n_z - 1:
            etaxz_zp = (self.viscosity[ix, iz + 1] *
                        self.viscosity[ixm, iz + 1] * self.viscosity[ix, iz] *
                        self.viscosity[ixm, iz])**0.25
        else:
            etaxz_zp = 0
        if (ix < n_x - 1 or periodic) and iz > 0:
            etaxz_xp = (self.viscosity[ixp, iz] * self.viscosity[ix, iz] *
                        self.viscosity[ixp, iz - 1] *
                        self.viscosity[ix, iz - 1])**0.25
        else:
            etaxz_xp = 0
        return etaii_c, etaii_xm, etaii_zm, etaxz_c, etaxz_xp, etaxz_zp

    def _solve_stokes(self) -> None:
        """Solve the Stokes equation for a given temperature field."""
        self._eval_viscosity()
        n_x = self._conf.numerical.n_x
        n_z = self._conf.numerical.n_z
        periodic = self._conf.physical.periodic
        rhsz = np.zeros((n_x, n_z))
        rhsz[:, 1:] = -self._conf.physical.ranum * (
            self.temp[:, :-1] + self.temp[:, 1:]) / 2

        odz = n_z
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

                etaii_c, etaii_xm, etaii_zm, etaxz_c, etaxz_xp, etaxz_zp =\
                    self._visco_around(ix, iz)

                xmom_zero_eta = (etaii_c == 0 and etaii_xm == 0 and
                                 etaxz_c == 0 and etaxz_zp == 0)
                zmom_zero_eta = (etaii_c == 0 and etaii_zm == 0 and
                                 etaxz_c == 0 and etaxz_xp == 0)

                # x-momentum
                if (ix > 0 or periodic) and not xmom_zero_eta:
                    spm.coef(ieqx, ieqx, -odz2 * (2 * etaii_c + 2 * etaii_xm +
                                                  etaxz_c + etaxz_zp))
                    spm.coef(ieqx, ieqxm, 2 * odz2 * etaii_xm)
                    spm.coef(ieqx, ieqz, -odz2 * etaxz_c)
                    spm.coef(ieqx, ieqzm, odz2 * etaxz_c)
                    spm.coef(ieqx, ieqc, -odz)
                    spm.coef(ieqx, ieqcm, odz)

                    if ix + 1 < n_x or periodic:
                        spm.coef(ieqx, ieqxp, 2 * odz2 * etaii_c)
                    if iz + 1 < n_z:
                        spm.coef(ieqx, ieqx + idz, odz2 * etaxz_zp)
                        spm.coef(ieqx, ieqz + idz, odz2 * etaxz_zp)
                        spm.coef(ieqx, ieqz + idz - idx, -odz2 * etaxz_zp)
                    if iz > 0:
                        spm.coef(ieqx, ieqx - idz, odz2 * etaxz_c)
                    rhs[ieqx] = 0
                else:
                    spm.coef(ieqx, ieqx, 1)
                    rhs[ieqx] = 0

                # z-momentum
                if iz > 0 and not zmom_zero_eta:
                    spm.coef(ieqz, ieqz, -odz2 * (2 * etaii_c + 2 * etaii_zm +
                                                  etaxz_c + etaxz_xp))
                    spm.coef(ieqz, ieqz - idz, 2 * odz2 * etaii_zm)
                    spm.coef(ieqz, ieqx, -odz2 * etaxz_c)
                    spm.coef(ieqz, ieqx - idz, odz2 * etaxz_c)
                    spm.coef(ieqz, ieqc, -odz)
                    spm.coef(ieqz, ieqc - idz, odz)

                    if iz + 1 < n_z:
                        spm.coef(ieqz, ieqz + idz, 2 * odz2 * etaii_c)
                    if ix + 1 < n_x or periodic:
                        spm.coef(ieqz, ieqzp, odz2 * etaxz_xp)
                        spm.coef(ieqz, ieqxp, odz2 * etaxz_xp)
                        spm.coef(ieqz, ieqxpm, -odz2 * etaxz_xp)
                    if ix > 0 or periodic:
                        spm.coef(ieqz, ieqzm, odz2 * etaxz_c)
                    rhs[ieqz] = rhsz[ix, iz]
                else:
                    spm.coef(ieqz, ieqz, 1)
                    rhs[ieqz] = 0

                # continuity
                if (ix == 0 and iz == 0) or (xmom_zero_eta and zmom_zero_eta):
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
        # compute stabe timestep
        # assumes n_x=n_z. To be generalized
        dt_diff = 0.1 / self._conf.numerical.n_z**2
        vmax = np.maximum(np.amax(np.abs(self.v_x)), np.amax(np.abs(self.v_z)))
        dt_adv = 0.5 / self._conf.numerical.n_z / vmax
        dt = min(dt_diff, dt_adv)
        # diffusion and internal heating
        self.temp = self.temp + dt * (self._del2temp() +
                                      self._donor_cell_advection() +
                                      self._conf.physical.int_heat)
        return dt

    def _del2temp(self) -> ndarray:
        """Computes Laplacian of temperature

        zero flux BC on the vertical sides for non-periodic cases
        T = 0 at the top
        T = 1 at the bottom
        """
        n_x = self._conf.numerical.n_x
        n_z = self._conf.numerical.n_z
        delsqT = np.zeros_like(self.temp)
        dsq = n_z**2  # inverse of dz ^ 2
        # should be generalized for non-square grids

        for i in range(n_x):
            if self._conf.physical.periodic:
                im = (i - 1 + n_x) % n_x
                ip = (i + 1) % n_x
            else:
                im = max(i - 1, 0)
                ip = min(i + 1, n_x - 1)

            for j in range(0, n_z):
                T_xm = self.temp[im, j]
                T_xp = self.temp[ip, j]
                if j == 0:  # enforce bottom BC
                    T_zm = 2 - self.temp[i, j]
                else:
                    T_zm = self.temp[i, j - 1]
                if j == n_z - 1:
                    T_zp = - self.temp[i, j]
                else:
                    T_zp = self.temp[i, j + 1]

                delsqT[i, j] = (T_xm + T_xp + T_zm + T_zp -
                                4 * self.temp[i, j]) * dsq

        return delsqT

    def _donor_cell_advection(self) -> ndarray:
        """Donor cell advection div(v T)"""
        dtemp = np.zeros_like(self.temp)
        temp = self.temp
        v_x = self.v_x
        v_z = self.v_z
        n_x = self._conf.numerical.n_x
        n_z = self._conf.numerical.n_z

        for i in range(n_x):
            if self._conf.physical.periodic:
                im = (i - 1 + n_x) % n_x
                ip = (i + 1) % n_x
            else:
                im = max(i - 1, 0)
                ip = min(i + 1, n_x - 1)

            for j in range(n_z):
                if i > 0 or self._conf.physical.periodic:
                    flux_xm = temp[im, j] * v_x[i, j] if v_x[i, j] > 0 else\
                        temp[i, j] * v_x[i, j]
                else:
                    flux_xm = 0

                if i < n_x - 1 or self._conf.physical.periodic:
                    flux_xp = temp[i, j] * v_x[ip, j] if v_x[ip, j] > 0 else\
                        temp[ip, j] * v_x[ip, j]
                else:
                    flux_xp = 0

                if j > 0:
                    flux_zm = temp[i, j - 1] * v_z[i, j] \
                        if v_z[i, j] > 0 else temp[i, j] * v_z[i, j]
                else:
                    flux_zm = 0

                if j < n_z - 1:
                    flux_zp = temp[i, j] * v_z[i, j + 1] \
                        if v_z[i, j + 1] >= 0 else \
                        temp[i, j + 1] * v_z[i, j + 1]
                else:
                    flux_zp = 0
                dtemp[i, j] = (flux_xm - flux_xp + flux_zm - flux_zp) * n_z
                # assumes d_x = d_z. To be generalized
        return dtemp


class RunManager:

    """Simulation of the Rayleigh Bénard convection problem."""

    def __init__(self, conf: Config):
        self._conf = conf

        self._fstart = None
        self._istart = -1
        if self.conf.numerical.restart:
            for fname in self.conf.inout.outdir.glob('fields*.npz'):
                ifile = int(fname.name[6:-4])
                if ifile > self._istart:
                    self._istart = ifile
                    self._fstart = fname
        if self._fstart is not None:
            with np.load(self._fstart) as fld:
                self.time = fld['time']
                temp = fld['T']
        else:
            self._istart = 0
            temp = self._init_temp()
            self.time = 0

        self.state = StokesState(temp, conf)

    @property
    def conf(self) -> Config:
        """Configuration of solver."""
        return self._conf

    def _outfile(self, name: str, istep: int, ext: str = None) -> Path:
        fname = '{}{:08d}'.format(name, istep)
        if ext is not None:
            fname += '.{}'.format(ext)
        return self.conf.inout.outdir / fname

    def _init_temp(self) -> ndarray:
        """Compute inital temperature."""
        n_x = self.conf.numerical.n_x
        n_z = self.conf.numerical.n_z
        if self.conf.physical.pert_init == 'sin':
            xgrid = np.linspace(0, n_x / n_z, n_x)
            zgrid = np.linspace(0, 1, n_z)
            temp = self.conf.physical.temp_init + \
                0.01 * np.outer(np.sin(np.pi * xgrid),
                                np.sin(np.pi * zgrid))
        else:
            temp = self.conf.physical.temp_init + \
                0.01 * np.random.uniform(-1, 1, (n_x, n_z))
        return temp

    def _save(self, istep: int) -> None:
        n_x = self.conf.numerical.n_x
        n_z = self.conf.numerical.n_z
        fname = self._outfile('fields', istep, 'npz')
        np.savez(fname, T=self.state.temp, vx=self.state.v_x,
                 vz=self.state.v_z, p=self.state.dynp, time=self.time)

        xgrid = np.linspace(0, n_x / n_z, n_x)
        zgrid = np.linspace(0, 1, n_z)
        fig, axis = plt.subplots()
        # first plot the temperature field
        surf = axis.pcolormesh(xgrid, zgrid, self.state.temp.T, cmap='RdBu_r',
                               shading='gouraud')
        cbar = plt.colorbar(surf, shrink=0.5)
        cbar.set_label('Temperature')
        plt.axis('equal')
        plt.axis('off')
        axis.set_adjustable('box')
        axis.set_xlim(0, n_x / n_z)
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
        axis.streamplot(xgrid, zgrid, u_x.T, u_z.T, color='k',
                        linewidth=lw.T)
        fig.savefig(self._outfile('T_v', istep, 'pdf'), bbox_inches='tight')
        plt.close(fig)

    def _timeseries(self, istep: int) -> ndarray:
        """Time series diagnostic for one step."""
        n_z = self.conf.numerical.n_z
        tseries = np.empty(_NTSERIES)
        tseries[0] = istep
        tseries[1] = self.time
        tseries[2] = np.amin(self.state.temp)
        tseries[3] = np.mean(self.state.temp)
        tseries[4] = np.amax(self.state.temp)
        ekin = np.mean(self.state.v_x ** 2 + self.state.v_z ** 2)
        tseries[5] = np.sqrt(ekin)
        tseries[6] = np.sqrt(np.mean(self.state.v_x[:, n_z - 1] ** 2))
        tseries[7] = 2 * n_z * (1 - np.mean(self.state.temp[:, 0]))
        tseries[8] = 2 * n_z * np.mean(self.state.temp[:, n_z - 1])
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
        self.conf.to_file(parfile)
