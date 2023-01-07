from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from os import PathLike
    from typing import Dict, Type, Union, Any

    from numpy.typing import NDArray

    from . config import Config


@dataclass(frozen=True)
class InitialCondition:
    """An initial condition defined from temperature.
Attributes:
        temperature: the temperature field.
        istart: the starting step number.
        time: the starting time.
    """
    temperature: NDArray[np.float64]
    istart: int = 0
    time: float = 0.0


class ICFactory(ABC):
    """Describe an initial condition for temperature."""

    _impls: Dict[str, Type[ICFactory]] = {}

    def __init_subclass__(cls: Type[ICFactory], ic_name: str):
        if ic_name in ICFactory._impls:
            ValueError(f"IC {ic_name!r} defined twice.")
        ICFactory._impls[ic_name] = cls
        cls._ic_name = ic_name  # type: ignore

    @staticmethod
    def from_name_or_dict(val: Union[str, Dict[str, Any]]) -> ICFactory:
        if isinstance(val, dict):
            try:
                ic_name: str = val.pop("name")
            except KeyError:
                raise ValueError("No name is specified for the IC.")
            kwargs = val
        elif isinstance(val, str):
            ic_name = val
            kwargs = {}
        else:
            raise TypeError("Cannot build an IC from a {val.__class__}.")
        try:
            ic_impl = ICFactory._impls[ic_name]
        except KeyError:
            raise ValueError(f"IC {val!r} is not defined.")
        return ic_impl(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        dct = {"name": self._ic_name}  # type: ignore
        dct.update(asdict(self))
        return dct

    @abstractmethod
    def build_ic(self, conf: Config) -> InitialCondition:
        """The initial condition."""


@dataclass(frozen=True)
class RandomIC(ICFactory, ic_name="random"):
    mean_temperature: float = 0.5
    noise_amplitude: float = 1e-2

    def build_ic(self, conf: Config) -> InitialCondition:
        temp = self.noise_amplitude * np.random.uniform(
            -1, 1, (conf.numerical.n_x, conf.numerical.n_z)
        ) + self.mean_temperature
        return InitialCondition(temperature=temp)


@dataclass(frozen=True)
class SineIC(ICFactory, ic_name="sin"):
    mean_temperature: float = 0.5
    amplitude: float = 1e-2

    def build_ic(self, conf: Config) -> InitialCondition:
        n_x = conf.numerical.n_x
        n_z = conf.numerical.n_z
        xgrid = np.linspace(0, n_x / n_z, n_x)
        zgrid = np.linspace(0, 1, n_z)
        temp = self.mean_temperature + self.amplitude * np.outer(
            np.sin(np.pi * xgrid), np.sin(np.pi * zgrid))
        return InitialCondition(temperature=temp)


@dataclass(frozen=True)
class StartFileIC(ICFactory, ic_name="from_file"):
    filename: Union[str, PathLike]

    def build_ic(self, conf: Config) -> InitialCondition:
        with np.load(Path(self.filename)) as f_init:
            step = f_init.get("step", 0)
            time = f_init.get("time", 0.0)
            temp = f_init["T"]
        return InitialCondition(temperature=temp, istart=step, time=time)
