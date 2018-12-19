import pathlib
from scipy.sparse.linalg import factorized
import h5py
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
import toml


CONF_DEFAULT = {
    'numerical': {
        'n_x': (32, 'Number of grid points in the horizontal direction.'),
        'n_z': (32, 'Number of grid points in the vertical direction.'),
        'nsteps': (100, 'Number of timesteps to perform.'),
        'nwrite': (10, 'Save data every nwrite timesteps.'),
        'restart': (False, 'Look for a file to restart from.'),
    },
    'physical': {
        'ranum': (3e3, 'Rayleigh number.'),
        'int_heat': (0, 'Internal heating.'),
        'temp_init': (0.5, 'Average initial temperature.'),
        'pert_init': (
            'random',
            "Initial temperature perturbation, either 'random' or 'sin'."),
        'var_visc': (False, 'Whether viscosity is variable.'),
        'var_visc_temp': (1e6, 'Viscosity contrast with temperature.'),
        'var_visc_depth': (1e2, 'Viscosity contrast with depth.'),
        'periodic': (False, 'Controls conditions in the horizontal direction')
    },
}


_NTSERIES = 9
_CHANGEMAT = set(('n_x', 'n_y',
                  'var_visc', 'var_visc_temp', 'var_visc_depth',
                  'periodic'))


# these two helpers are necessary to capture section and opt in the closure
def _getter(section, opt):
    return lambda self: self._conf[section][opt]


def _setter(section, opt):
    def func(self, val):
        if opt in _CHANGEMAT:
            self._lumat = None
        self._conf[section][opt] = val
    return func


class _MetaRBS(type):

    """Dynamically add configuration properties."""

    def __new__(cls, name, bases, dct):
        rbs = super().__new__(cls, name, bases, dct)
        for section, opts in CONF_DEFAULT.items():
            for opt, (_, doc) in opts.items():
                setattr(rbs, opt, property(_getter(section, opt),
                                           _setter(section, opt),
                                           doc=doc))
        return rbs


class RayleighBenardStokes(metaclass=_MetaRBS):

    """Solver of Rayleigh Benard convection at infinite Prandtl number."""

    def __init__(self, outdir='output', parfile=None):
        """Initialization of instance:

        Args:
            outdir (path-like): path to the output directory.
            parfile (path-like): path to the parameters file.
        """
        self.outdir = pathlib.Path(outdir)
        self._conf = {section: {opt: val for opt, (val, _) in opts.items()}
                      for section, opts in CONF_DEFAULT.items()}

        if parfile is not None:
            self.load_pars(parfile)

        self.time = 0
        self.temp = None
        self.v_x = None
        self.v_z = None
        self.dynp = None
        self.eta = None
        self._lumat = None

    def load_pars(self, parfile):
        """Read a TOML par file and update configuration accordingly."""
        pars = toml.load(parfile)
        for section, opts in pars.items():
            if section not in CONF_DEFAULT:
                print('Unknow section {}.'.format(section))
                continue
            for opt, val in opts.items():
                if opt not in CONF_DEFAULT[section]:
                    print('Unknown option {}.{}.'.format(section, opt))
                    continue
                setattr(self, opt, val)

    def _outfile(self, name, istep, ext=None):
        fname = '{}{:08d}'.format(name, istep)
        if ext is not None:
            fname += '.{}'.format(ext)
        return self.outdir / fname

    def _init_temp(self):
        if self.pert_init == 'sin':
            xgrid = np.linspace(0, self.n_x / self.n_z, self.n_x)
            zgrid = np.linspace(0, 1, self.n_z)
            self.temp = self.temp_init + \
                0.01 * np.outer(np.sin(np.pi * xgrid),
                                np.sin(np.pi * zgrid))
        else:
            self.temp = self.temp_init + \
                0.01 * np.random.uniform(-1, 1, (self.n_x, self.n_z))

    def _save(self, istep):
        fname = self._outfile('fields', istep, 'npz')
        np.savez(fname, T=self.temp, vx=self.v_x, vz=self.v_z, p=self.dynp,
                 time=self.time)

        xgrid = np.linspace(0, self.n_x / self.n_z, self.n_x)
        zgrid = np.linspace(0, 1, self.n_z)
        fig, axis = plt.subplots()
        # first plot the temperature field
        surf = axis.pcolormesh(xgrid, zgrid, self.temp.T, cmap='RdBu_r',
                               shading='gouraud')
        cbar = plt.colorbar(surf, shrink=0.5)
        cbar.set_label('Temperature')
        plt.axis('equal')
        plt.axis('off')
        axis.set_adjustable('box')
        axis.set_xlim(0, self.n_x / self.n_z)
        axis.set_ylim(0, 1)
        # interpolate velocities at the same points for streamlines
        u_x = 0.5 * (self.v_x + np.roll(self.v_x, 1, 0))
        u_z = np.zeros_like(self.v_z)
        u_z[1:] = self.v_z[:-1]
        u_z += self.v_z
        u_z *= 0.5
        speed = np.sqrt(u_x ** 2 + u_z ** 2)
        # plot the streamlines
        lw = 2 * speed / speed.max()
        axis.streamplot(xgrid, zgrid, u_x.T, u_z.T, color='k',
                        linewidth=lw.T)
        fig.savefig(self._outfile('T_v', istep, 'pdf'), bbox_inches='tight')
        plt.close(fig)

    def _calc_eta(self):
        if self.var_visc:
            d_z = 1 / self.n_z
            a_visc = np.log(self.var_visc_temp)
            b_visc = np.log(self.var_visc_depth)
            depth = 0.5 - np.linspace(0.5 * d_z, 1 - 0.5 * d_z, self.n_z)
            self.eta = np.exp(-a_visc * (self.temp - 0.5) + b_visc * depth)
        else:
            self.eta = np.ones((self.n_x, self.n_z))

    def _eta_around(self, ix, iz):
        # deal with boundary conditions on vertical planes
        if self.periodic:
            ixm = (ix - 1 + self.n_x) % self.n_x
            ixp = (ix + 1) % self.n_x
        else:
            ixm = max(ix - 1, 0)
            ixp = min(ix + 1, self.n_x - 1)

        etaii_c = self.eta[ix, iz]
        etaii_xm = self.eta[ixm, iz] if ix > 0 or self.periodic else 0
        etaii_zm = self.eta[ix, iz - 1] if iz > 0 else 0
        if (ix > 0 or self.periodic) and iz > 0:
            etaxz_c = (self.eta[ix, iz] * self.eta[ixm, iz] *
                       self.eta[ix, iz - 1] *
                       self.eta[ixm, iz - 1])**0.25
        else:
            etaxz_c = 0
        if (ix > 0 or self.periodic) and iz < self.n_z - 1:
            etaxz_zp = (self.eta[ix, iz + 1] *
                        self.eta[ixm, iz + 1] * self.eta[ix, iz] *
                        self.eta[ixm, iz])**0.25
        else:
            etaxz_zp = 0
        if (ix < self.n_x - 1 or self.periodic) and iz > 0:
            etaxz_xp = (self.eta[ixp, iz] * self.eta[ix, iz] *
                        self.eta[ixp, iz - 1] *
                        self.eta[ix, iz - 1])**0.25
        else:
            etaxz_xp = 0
        return etaii_c, etaii_xm, etaii_zm, etaxz_c, etaxz_xp, etaxz_zp

    def _stokes(self):
        self._calc_eta()
        rhsz = np.zeros((self.n_x, self.n_z))
        rhsz[:, 1:] = -self.ranum * (self.temp[:, :-1] + self.temp[:, 1:]) / 2

        odz = self.n_z
        odz2 = odz**2

        rhs = np.zeros(self.n_x * self.n_z * 3)
        # indices offset
        idx = 3
        idz = self.n_x * 3
        # a priori number of non-zero matrix coefficients
        n_non0 = (11 + 11 + 4) * self.n_x * self.n_z
        rows = np.zeros(n_non0)
        cols = np.zeros(n_non0)
        coefs = np.zeros(n_non0)
        n_non0 = 0  # track number of non zeros coef

        def mcoef(row, col, coef):
            nonlocal n_non0
            rows[n_non0] = row
            cols[n_non0] = col
            coefs[n_non0] = coef
            n_non0 += 1

        for iz in range(self.n_z):
            for ix in range(self.n_x):
                # define indices in the matrix
                icell = ix + iz * self.n_x
                ieqx = icell * 3
                ieqz = ieqx + 1
                ieqc = ieqx + 2
                ieqxp = (ieqx + idx) % rhs.size \
                    if self.periodic else ieqx + idx
                ieqzp = ieqxp + 1
                ieqxpm = (ieqxp - idz) % rhs.size \
                    if self.periodic else ieqxp - idz
                ieqxm = (ieqx - idx) % rhs.size \
                    if self.periodic else ieqx - idx
                ieqzm = ieqxm + 1
                ieqcm = ieqxm + 2

                etaii_c, etaii_xm, etaii_zm, etaxz_c, etaxz_xp, etaxz_zp =\
                    self._eta_around(ix, iz)

                xmom_zero_eta = (etaii_c == 0 and etaii_xm == 0 and
                                 etaxz_c == 0 and etaxz_zp == 0)
                zmom_zero_eta = (etaii_c == 0 and etaii_zm == 0 and
                                 etaxz_c == 0 and etaxz_xp == 0)

                # x-momentum
                if (ix > 0 or self.periodic) and not xmom_zero_eta:
                    mcoef(ieqx, ieqx, -odz2 * (2 * etaii_c + 2 * etaii_xm +
                                               etaxz_c + etaxz_zp))
                    mcoef(ieqx, ieqxm, 2 * odz2 * etaii_xm)
                    mcoef(ieqx, ieqz, -odz2 * etaxz_c)
                    mcoef(ieqx, ieqzm, odz2 * etaxz_c)
                    mcoef(ieqx, ieqc, -odz)
                    mcoef(ieqx, ieqcm, odz)

                    if ix + 1 < self.n_x or self.periodic:
                        mcoef(ieqx, ieqxp, 2 * odz2 * etaii_c)
                    if iz + 1 < self.n_z:
                        mcoef(ieqx, ieqx + idz, odz2 * etaxz_zp)
                        mcoef(ieqx, ieqz + idz, odz2 * etaxz_zp)
                        mcoef(ieqx, ieqz + idz - idx, -odz2 * etaxz_zp)
                    if iz > 0:
                        mcoef(ieqx, ieqx - idz, odz2 * etaxz_c)
                    rhs[ieqx] = 0
                else:
                    mcoef(ieqx, ieqx, 1)
                    rhs[ieqx] = 0

                # z-momentum
                if iz > 0 and not zmom_zero_eta:
                    mcoef(ieqz, ieqz, -odz2 * (2 * etaii_c + 2 * etaii_zm +
                                               etaxz_c + etaxz_xp))
                    mcoef(ieqz, ieqz - idz, 2 * odz2 * etaii_zm)
                    mcoef(ieqz, ieqx, -odz2 * etaxz_c)
                    mcoef(ieqz, ieqx - idz, odz2 * etaxz_c)
                    mcoef(ieqz, ieqc, -odz)
                    mcoef(ieqz, ieqc - idz, odz)

                    if iz + 1 < self.n_z:
                        mcoef(ieqz, ieqz + idz, 2 * odz2 * etaii_c)
                    if ix + 1 < self.n_x or self.periodic:
                        mcoef(ieqz, ieqzp, odz2 * etaxz_xp)
                        mcoef(ieqz, ieqxp, odz2 * etaxz_xp)
                        mcoef(ieqz, ieqxpm, -odz2 * etaxz_xp)
                    if ix > 0 or self.periodic:
                        mcoef(ieqz, ieqzm, odz2 * etaxz_c)
                    rhs[ieqz] = rhsz[ix, iz]
                else:
                    mcoef(ieqz, ieqz, 1)
                    rhs[ieqz] = 0

                # continuity
                if (ix == 0 and iz == 0) or (xmom_zero_eta and zmom_zero_eta):
                    mcoef(ieqc, ieqc, 1)
                else:
                    mcoef(ieqc, ieqx, -odz)
                    mcoef(ieqc, ieqz, -odz)
                    if ix + 1 < self.n_x or self.periodic:
                        mcoef(ieqc, ieqxp, odz)
                    if iz + 1 < self.n_z:
                        mcoef(ieqc, ieqz + idz, odz)
                rhs[ieqc] = 0

        if self.var_visc or self._lumat is None:
            self._lumat = factorized(sp.csc_matrix((coefs, (rows, cols)),
                                                   shape=(rhs.size, rhs.size)))
        sol = self._lumat(rhs)
        self.v_x = np.reshape(sol[::3], (self.n_z, self.n_x)).T
        # remove drift velocity (unconstrained)
        if self.periodic:
            self.v_x -= np.mean(self.v_x)
        self.v_z = np.reshape(sol[1::3], (self.n_z, self.n_x)).T
        self.dynp = np.reshape(sol[2::3], (self.n_z, self.n_x)).T
        self.dynp -= np.mean(self.dynp)

    def _heat(self):
        """Advection diffusion equation for time stepping"""
        # compute stabe timestep
        # assumes n_x=n_z. To be generalized
        dt_diff = 0.1 / self.n_z**2
        vmax = np.maximum(np.amax(np.abs(self.v_x)), np.amax(np.abs(self.v_z)))
        dt_adv = 0.5 / self.n_z / vmax
        dt = np.minimum(dt_diff, dt_adv)
        self.time += dt
        # diffusion and internal heating
        self.temp += dt * (self._del2temp() + self.int_heat)
        # advection
        self.temp += dt * self._donor_cell_advection()

    def _del2temp(self):
        """Computes Laplacian of temperature

        zero flux BC on the vertical sides for non-periodic cases
        T = 0 at the top
        T = 1 at the bottom
        """
        delsqT = np.zeros_like(self.temp)
        dsq = self.n_z**2  # inverse of dz ^ 2
        # should be generalized for non-square grids

        for i in range(self.n_x):
            if self.periodic:
                im = (i - 1 + self.n_x) % self.n_x
                ip = (i + 1) % self.n_x
            else:
                im = max(i - 1, 0)
                ip = min(i + 1, self.n_x - 1)

            for j in range(0, self.n_z):
                T_xm = self.temp[im, j]
                T_xp = self.temp[ip, j]
                if j == 0:  # enforce bottom BC
                    T_zm = 2 - self.temp[i, j]
                else:
                    T_zm = self.temp[i, j - 1]
                if j == self.n_z - 1:
                    T_zp = - self.temp[i, j]
                else:
                    T_zp = self.temp[i, j + 1]

                delsqT[i, j] = (T_xm + T_xp + T_zm + T_zp -
                                4 * self.temp[i, j]) * dsq

        return delsqT

    def _donor_cell_advection(self):
        """Donor cell advection div(v T)"""
        dtemp = np.zeros_like(self.temp)
        temp = self.temp
        v_x = self.v_x
        v_z = self.v_z

        for i in range(self.n_x):
            if self.periodic:
                im = (i - 1 + self.n_x) % self.n_x
                ip = (i + 1) % self.n_x
            else:
                im = max(i - 1, 0)
                ip = min(i + 1, self.n_x - 1)

            for j in range(self.n_z):
                if i > 0 or self.periodic:
                    flux_xm = temp[im, j] * v_x[i, j] if v_x[i, j] > 0 else\
                        temp[i, j] * v_x[i, j]
                else:
                    flux_xm = 0

                if i < self.n_x - 1 or self.periodic:
                    flux_xp = temp[i, j] * v_x[ip, j] if v_x[ip, j] > 0 else\
                        temp[ip, j] * v_x[ip, j]
                else:
                    flux_xp = 0

                if j > 0:
                    flux_zm = temp[i, j - 1] * v_z[i, j] \
                        if v_z[i, j] > 0 else temp[i, j] * v_z[i, j]
                else:
                    flux_zm = 0

                if j < self.n_z - 1:
                    flux_zp = temp[i, j] * v_z[i, j + 1] \
                        if v_z[i, j + 1] >= 0 else \
                        temp[i, j + 1] * v_z[i, j + 1]
                else:
                    flux_zp = 0
                dtemp[i, j] = (flux_xm - flux_xp +
                               flux_zm - flux_zp) * self.n_z
                # assumes d_x = d_z. To be generalized
        return dtemp

    def _timeseries(self, istep):
        """Time series diagnostic for one step."""
        tseries = np.empty(_NTSERIES)
        tseries[0] = istep
        tseries[1] = self.time
        tseries[2] = np.amin(self.temp)
        tseries[3] = np.mean(self.temp)
        tseries[4] = np.amax(self.temp)
        ekin = np.mean(self.v_x ** 2 + self.v_z ** 2)
        tseries[5] = np.sqrt(ekin)
        tseries[6] = np.sqrt(np.mean(self.v_x[:, self.n_z - 1] ** 2))
        tseries[7] = 2 * self.n_z * (1 - np.mean(self.temp[:, 0]))
        tseries[8] = 2 * self.n_z * np.mean(self.temp[:, self.n_z - 1])
        return tseries

    def solve(self, progress=False):
        """Resolution of asked problem.

        Args:
            progress (bool): output progress.
        """
        fstart = None
        istart = -1
        if self.restart:
            for fname in self.outdir.glob('fields*.npz'):
                ifile = int(fname.name[6:-4])
                if ifile > istart:
                    istart = ifile
                    fstart = fname
        self.outdir.mkdir(exist_ok=True)
        if fstart is not None:
            with np.load(fstart) as fld:
                self.temp = fld['T']
                self.time = fld['time']
        else:
            istart = 0
            self._init_temp()
            self.time = 0
        self._stokes()
        if fstart is None:
            self._save(0)

        tfilename = self.outdir / 'time.h5'
        if fstart is None or not tfilename.exists():
            tfile = h5py.File(tfilename, 'w')
            dset = tfile.create_dataset('series', (1, _NTSERIES),
                                        maxshape=(None, _NTSERIES),
                                        data=self._timeseries(istart))
        else:
            tfile = h5py.File(tfilename, 'a')
            dset = tfile['series']

        step_msg = '\rstep: {{:{}d}}/{}'.format(len(str(self.nsteps)),
                                                self.nsteps)
        tseries = np.zeros((self.nwrite, _NTSERIES))

        for irun, istep in enumerate(range(istart + 1, self.nsteps + 1)):
            if progress:
                print(step_msg.format(istep), end='')
            self._heat()
            self._stokes()
            tseries[irun % self.nwrite] = self._timeseries(istep)
            if (irun + 1) % self.nwrite == 0:
                self._save(istep)
                dset.resize((len(dset) + self.nwrite, _NTSERIES))
                dset[-self.nwrite:] = tseries
        if progress:
            print()
        tfile.close()

    def dump_pars(self, parfile):
        """Dump configuration.

        Args:
            parfile (pathlib.Path): path of the par file where the
                configuration should be dumped.
        """
        with parfile.open('w') as pf:
            toml.dump(self._conf, pf)
