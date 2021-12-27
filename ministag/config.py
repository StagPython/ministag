from __future__ import annotations
from dataclasses import dataclass, asdict
import typing

import toml

if typing.TYPE_CHECKING:
    from typing import Mapping, Any
    from pathlib import Path


@dataclass(frozen=True)
class NumericalConf:
    """Configuration of numerical domain.

    Attributes:
        n_x: number of grid points in the horizontal direction.
        n_z: number of grid points in the vertical direction.
        nsteps: number of timesteps to perform.
        nwrite: save data every nwrite timesteps.
        restart: look for a file to restart from.
    """
    n_x: int = 32
    n_z: int = 32
    nsteps: int = 100
    nwrite: int = 10
    restart: bool = False


@dataclass(frozen=True)
class PhysicalConf:
    """Configuration of physical domain.

    Attributes:
        ranum: Rayleigh number.
        int_heat: Internal heating.
        temp_init: Average initial temperature.
        pert_init: Initial temperature perturbation, either 'random' or 'sin'.
        var_visc: Whether viscosity is variable.
        var_visc_temp: Viscosity contrast with temperature.
        var_visc_depth: Viscosity contrast with depth.
        periodic: Controls conditions in the horizontal direction
    """
    ranum: float = 3e3
    int_heat: float = 0.
    temp_init: float = 0.5
    pert_init: str = 'random'
    var_visc: bool = False
    var_visc_temp: float = 1e6
    var_visc_depth: float = 1e2
    periodic: bool = False


@dataclass(frozen=True)
class Config:
    numerical: NumericalConf = NumericalConf()
    physical: PhysicalConf = PhysicalConf()

    @staticmethod
    def from_file(parfile: Path) -> Config:
        """Read configuration from toml file."""
        pars = toml.load(parfile)
        return Config.from_dict(pars)

    @staticmethod
    def from_dict(options: Mapping[str, Mapping[str, Any]]) -> Config:
        """Create configuration from a dictionary."""
        num_dict = options.get("numerical", {})
        phy_dict = options.get("physical", {})
        return Config(numerical=NumericalConf(**num_dict),
                      physical=PhysicalConf(**phy_dict))

    def to_file(self, parfile: Path) -> None:
        """Write configuration in toml file.

        Args:
            parfile: path of the toml file.
        """
        with parfile.open('w') as pf:
            toml.dump(asdict(self), pf)
