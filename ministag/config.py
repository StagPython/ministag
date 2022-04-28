from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from loam.base import entry, Section, ConfigBase
from loam.tools import path_entry


@dataclass
class Numerical(Section):
    """Configuration of numerical domain."""
    n_x: int = entry(val=32, doc="grid points in the horizontal direction")
    n_z: int = entry(val=32, doc="grid points in the vertical direction")
    nsteps: int = entry(val=100, doc="number of timesteps to perform")
    nwrite: int = entry(val=10, doc="save data every nwrite timesteps")
    restart: bool = entry(val=False, doc="look for a file to restart from")


@dataclass
class Physical(Section):
    """Configuration of physical domain."""
    ranum: float = entry(val=3e3, doc="Rayleigh number")
    int_heat: float = entry(val=0., doc="internal heating")
    temp_init: float = entry(val=0.5, doc="average initial temperature")
    pert_init: str = entry(
        val="random",
        doc="initial temperature perturbation, either 'random' or 'sin'")
    var_visc: bool = entry(val=False, doc="whether viscosity is variable")
    var_visc_temp: float = entry(
        val=1e6, doc="viscosity contrast with temperature")
    var_visc_depth: float = entry(
        val=1e2, doc="viscosity contrast with depth")
    periodic: bool = entry(
        val=False, doc="controls conditions in the horizontal direction")


@dataclass
class InOut(Section):
    """Configuration of input and output."""
    outdir: Path = path_entry(path="output", doc="output directory")


@dataclass
class Config(ConfigBase):
    numerical: Numerical
    physical: Physical
    inout: InOut
