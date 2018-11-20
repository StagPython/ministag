import pathlib
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
        self.v_x = None
        self.v_z = None

    def _heat(self):
        pass#self.temp = None

    def _del2temp(self):
        """Computes Laplacian of temperature

        zero flux BC on the vertical sides
        T = 0 at the top
        T = 1 at the bottom
        """
        delsqT = np.zeros(self.temp.shape)
        dsq =  (self.n_z -1 ) ** 2 # inverse of dz ^ 2
            # should be generalized for non-square grids
        
        for i in range(0, self.n_x):
            im = max(i-1, 0)
            ip = min(i+1, self.n_x)
            
            for j in range(0, self.n_z):
                T_xm = self.temp[im, j]
                T_xp = self.temp[ip, j]
                if j==0: # enforce bottom BC
                    T_zm = 2 - self.temp[i, j]
                else:
                    T_zm = self.temp[i, j - 1]
                if j==self.n_z:
                    T_zp = - self.temp[i, j + 1]
                else:
                    T_zp = self.temp[i, j + 1]

                delsqT[i, j] = (T_xm + T_xp + T_zm + T_zp - 4 * self.temp[i,j]) * dsq

        return delsqT

    def _donor_cell_advection(self):
        """Donor cell advection div(v T)"""
        Tnew = np.zeros(self.temp.shape)

        for i in range(0, self.n_x):
            for j in range(0, self.n_z):
                pass

        return Tnew

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
