from __future__ import annotations

from dataclasses import dataclass
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
