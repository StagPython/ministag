import pathlib
from scipy.sparse.linalg import factorized
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
import toml


class RayleighBenardStokes:

    """Solver of Rayleigh Benard convection at infinite Prandtl number."""

    def __init__(self, parfile=None):
        """Initialization of instance:

        Args:
            parfile (path-like): path to the parameters file.
        """
        pars = toml.load(parfile) if parfile is not None else {}
        self.set_numerical(**pars.get('numerical', {}))
        self.set_physical(**pars.get('physical', {}))
        self.temp = None
        self.v_x = None
        self.v_z = None
        self.dynp = None
        self.eta = None
        self._restart = pars.get('restart', {}).get('file', None)

    def _outfile_stem(self, name, istep):
        return 'output/{}{:08d}'.format(name, istep)

    def _init_fields(self):
        if self._restart is not None:
            with np.load(restart) as fld:
                self.temp = fld['T']
        else:
            self.temp = self.temp_init + \
                0.01 * np.random.rand(self.n_x, self.n_z)

    def _save(self, istep):
        pathlib.Path('output').mkdir(exist_ok=True)
        fname = self._outfile_stem('fields', istep) + '.npz'
        np.savez(fname, T=self.temp, vx=self.v_x, vz=self.v_z, p=self.dynp)

        xgrid = np.linspace(0, self.n_x / self.n_z, self.n_x)
        zgrid = np.linspace(0, 1, self.n_z)
        surf = plt.pcolormesh(xgrid, zgrid, self.temp.T, shading='gouraud')
        cbar = plt.colorbar(surf)
        cbar.set_label('Temperature')
        plt.axis('equal')
        plt.axis('off')
        plt.savefig(self._outfile_stem('T', istep) + '.pdf',
                    bbox_inches='tight')
        plt.close()

    def _stokes(self):
        d_z = 1 / self.n_z
        if self.var_visc:
            a_visc = np.log(self.var_visc_temp)
            b_visc = np.log(self.var_visc_depth)
            depth = 0.5 - np.linspace(0.5 * d_z, 1 - 0.5 * d_z, self.n_z)
            self.eta = np.exp(-a_visc * (self.temp - 0.5) + b_visc * depth)
        else:
            self.eta = np.ones((self.n_x, self.n_z))
        rhsz = np.zeros((self.n_x, self.n_z))
        rhsz[:, 1:] = -self.ranum * (self.temp[:, :-1] + self.temp[:, 1:]) / 2

        odz = self.n_z
        odz2 = odz**2

        rhs = np.zeros(self.n_x * self.n_z * 3)
        # indices offset
        idx = 3
        idz = self.n_x * 3
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
                icell = ix + iz * self.n_x
                ieqx = icell * 3
                ieqz = ieqx + 1
                ieqc = ieqx + 2

                etaii_c  = self.eta[ix, iz]
                if ix > 0:
                    etaii_xm = self.eta[ix-1, iz]
                if iz > 0:
                    etaii_zm = self.eta[ix, iz-1]
                if ix > 0 and iz > 0:
                    etaxz_c = (self.eta[ix, iz] * self.eta[ix - 1, iz] *
                               self.eta[ix, iz - 1] *
                               self.eta[ix - 1, iz - 1])**0.25
                else:
                    etaxz_c = 0
                if ix > 0 and iz < self.n_z - 1:
                    etaxz_zp = (self.eta[ix, iz + 1] *
                                self.eta[ix - 1, iz + 1] * self.eta[ix, iz] *
                                self.eta[ix - 1, iz])**0.25
                else:
                    etaxz_zp = 0
                if ix < self.n_x-1 and iz > 0:
                    etaxz_xp = (self.eta[ix + 1, iz] * self.eta[ix, iz] *
                                self.eta[ix + 1, iz - 1] *
                                self.eta[ix, iz - 1])**0.25
                else:
                    etaxz_xp = 0

                xmom_zero_eta = (etaii_c == 0 and etaii_xm == 0 and
                                 etaxz_c == 0 and etaxz_zp == 0)
                zmom_zero_eta = (etaii_c == 0 and etaii_zm == 0 and
                                 etazx_c == 0 and etazx_xp == 0)

                # x-momentum
                if ix > 0 and not xmom_zero_eta:
                    mcoef(ieqx, ieqx, -odz2 * (2 * etaii_c + 2 * etaii_xm +
                                               etaxz_c + etaxz_zp))
                    mcoef(ieqx, ieqx - idx, 2 * odz2 * etaii_xm)
                    mcoef(ieqx, ieqz, -odz2 * etaxz_c)
                    mcoef(ieqx, ieqz - idx, odz2 * etaxz_c)
                    mcoef(ieqx, ieqc, -odz)
                    mcoef(ieqx, ieqc - idx, odz)

                    if ix + 1 < self.n_x:
                        mcoef(ieqx, ieqx + idx, 2 * odz2 * etaii_c)
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
                    mcoef(ieqz, ieqz, -2 * odz2 * (2 * etaii_c + 2 * etaii_zm +
                                                   etaxz_c + etaxz_xp))
                    mcoef(ieqz, ieqz - idz, 2 * odz2 * etaii_zm)
                    mcoef(ieqz, ieqx, -odz2 * etaxz_c)
                    mcoef(ieqz, ieqx - idz, odz2 * etaxz_c)
                    mcoef(ieqz, ieqc, -odz)
                    mcoef(ieqz, ieqc - ieqz, odz)

                    if iz + 1 < self.n_z:
                        mcoef(ieqz, ieqz + idz, 2 * odz2 * etaii_c)
                    if ix + 1 < self.n_x:
                        mcoef(ieqz, ieqz + idx, odz2 * etaxz_xp)
                        mcoef(ieqz, ieqx + idx, odz2 * etaxz_xp)
                        mcoef(ieqz, ieqx + idx - idz, -odz2 * etaxz_xp)
                    if ix > 0:
                        mcoef(ieqz, ieqz - idx, odz2 * etaxz_c)
                    rhs[ieqz] = rhsz[ix, iz]
                else:
                    mcoef(ieqz, ieqz, 1)
                    rhs[ieqz] = 0

                # continuity
                if (ix==0 and iz==0) or (xmom_zero_eta and zmom_zero_eta):
                    mcoef(ieqc, ieqc, 1)
                else:
                    mcoef(ieqc, ieqx, -odz)
                    mcoef(ieqc, ieqz, -odz)
                    if ix + 1 < self.n_x:
                        mcoef(ieqc, ieqx + idx, odz)
                    if iz + 1 < self.n_z:
                        mcoef(ieqc, ieqz + idz, odz)
                rhs[ieqc] = 0

        fsolve = factorized(sp.coo_matrix((coefs, (rows, cols)),
                                          shape=(rhs.size, rhs.size)))
        sol = fsolve(rhs)
        self.v_x = np.reshape(sol[::3], (self.n_x, self.n_z))
        self.v_z = np.reshape(sol[1::3], (self.n_x, self.n_z))
        self.dynp = np.reshape(sol[2::3], (self.n_x, self.n_z))

    def _heat(self):
        pass#self.temp = None

    def solve(self, restart=None):
        """Resolution of asked problem.

        Args:
            restart (path-like): path to npz file defining the temperature
                file.
        """
        if restart is not None:
            self._restart = restart
        self._init_fields()
        restart = self._restart
        if restart is None:
            restart = -1
        for istep in range(restart + 1, self.nsteps + 1):
            self._stokes()
            self._heat()
            # time series
            if istep % self.nwrite == 0:
                self._save(istep)

    def set_numerical(self, n_x=32, n_z=32, nsteps=100, nwrite=10):
        """Set numerical parameters.

        Args:
            n_x (int): number of point in the horizontal direction, default 32.
            n_z (int): number of point in the vertical direction, default 32.
            nsteps (int): number of timesteps to perform, default 100.
            nwrite (int): save every nwrite timesteps, default 10.
        """
        self.n_x = n_x
        self.n_z = n_z
        self.nsteps = nsteps
        self.nwrite = nwrite

    def set_physical(self, ranum=3e3, int_heat=0, temp_init=0.5,
                     var_visc=False, var_visc_temp=1e6, var_visc_depth=1e2):
        """Set physical parameters.

        Args:
            ranum (float): Rayleigh number, default 3000.
            int_heat (float): internal heating, default 0.
            temp_init (float): average initial temperature, default 0.5.
            var_visc (bool): whether viscosity is variable, default False.
            var_visc_temp (float): viscosity contrast with temperature, default
                1e6.  Ignored if var_visc is False.
            var_visc_depth (float): viscosity contrast with depth, default 1e2.
                Ignored if var_visc is False.
        """
        self.ranum = ranum
        self.int_heat = int_heat
        self.temp_init = temp_init
        self.var_visc = var_visc
        self.var_visc_temp = var_visc_temp
        self.var_visc_depth = var_visc_depth
