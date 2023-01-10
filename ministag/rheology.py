from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

    from .solver import Grid


class Rheology(ABC):

    @abstractmethod
    def visco(self, temp: NDArray) -> NDArray:
        """Viscosity for a given temperature."""

    @property
    @abstractmethod
    def is_temp_dependent(self) -> bool:
        """Whether the viscosity depends on temperature."""


class ConstantVisco(Rheology):

    def visco(self, temp: NDArray) -> NDArray:
        return np.ones_like(temp)

    @property
    def is_temp_dependent(self) -> bool:
        return False


@dataclass(frozen=True)
class Arrhenius(Rheology):
    temp_factor: float
    depth_factor: float
    grid: Grid

    @cached_property
    def depth(self) -> NDArray:
        return 0.5 - self.grid.z_centers

    @cached_property
    def temp_coef(self) -> np.floating:
        return np.log(self.temp_factor)

    @cached_property
    def depth_coef(self) -> np.floating:
        return np.log(self.depth_factor)

    def visco(self, temp: NDArray) -> NDArray:
        return np.exp(self.temp_coef * (0.5 - temp) +
                      self.depth_coef * self.depth)

    @property
    def is_temp_dependent(self) -> bool:
        return self.temp_factor != 1.
