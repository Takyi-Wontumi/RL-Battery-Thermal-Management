"""
controllers/baseline_controllers.py

Baseline controllers for the battery thermal management environment.

These controllers are used for benchmarking PPO performance in Phase 2. They are not expected to be optimal, but they should be reasonable and represent a range of simple control strategies.

Controllers included:
    - NoCoolingController
    - ConstantCoolingController
    - BangBangController
    - ProportionalController
    - PIController

Expected observation format from BatteryThermalEnv:
    obs = [battery_temp_C, ambient_temp_C, heat_generation_W, time_fraction, previous_action]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ThermalController(Protocol):
    """Protocol for baseline thermal controllers."""

    name: str

    def reset(self) -> None:
        """Reset internal controller state at the beginning of an episode."""
        ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return normalized cooling action in [0, 1]."""
        ...


@dataclass
class NoCoolingController:
    """Always commands minimum cooling."""

    name: str = "No cooling"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


@dataclass
class ConstantCoolingController:
    """Always commands a fixed cooling level."""

    cooling_level: float = 0.5
    name: str = "Constant cooling"

    def __post_init__(self) -> None:
        self.cooling_level = float(np.clip(self.cooling_level, 0.0, 1.0))
        self.name = f"Constant cooling u={self.cooling_level:.2f}"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.array([self.cooling_level], dtype=np.float32)


@dataclass
class BangBangController:
    """
    Thermostat-style controller with hysteresis.

    This prevents the command from rapidly switching on/off around the target.
    """

    target_temp: float = 30.0
    deadband: float = 1.0
    low_action: float = 0.0
    high_action: float = 1.0
    name: str = "Bang-bang thermostat"

    def __post_init__(self) -> None:
        self.low_action = float(np.clip(self.low_action, 0.0, 1.0))
        self.high_action = float(np.clip(self.high_action, 0.0, 1.0))
        self._is_high: bool = False

    def reset(self) -> None:
        self._is_high = False

    def act(self, obs: np.ndarray) -> np.ndarray:
        temp = float(obs[0])

        upper = self.target_temp + self.deadband
        lower = self.target_temp - self.deadband

        if temp >= upper:
            self._is_high = True
        elif temp <= lower:
            self._is_high = False

        action = self.high_action if self._is_high else self.low_action
        return np.array([action], dtype=np.float32)


@dataclass
class ProportionalController:
    """
    Proportional thermal controller.

    action = bias + kp * (T_batt - target_temp)

    If kp is too low, the controller is lazy.
    If kp is too high, it becomes almost bang-bang.
    """

    target_temp: float = 30.0
    kp: float = 0.08
    bias: float = 0.15
    name: str = "Proportional controller"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        temp = float(obs[0])
        error = temp - self.target_temp
        action = self.bias + self.kp * error
        action = float(np.clip(action, 0.0, 1.0))
        return np.array([action], dtype=np.float32)


@dataclass
class PIController:
    """
    Proportional-integral thermal controller with anti-windup.

    The integral term helps eliminate steady-state error, but it can wind up if the controller is saturated for a long time. The anti-windup limits the integral term to prevent this.
    """

    target_temp: float = 30.0
    kp: float = 0.06
    ki: float = 0.002
    bias: float = 0.10
    dt: float = 1.0
    integral_limit: float = 100.0
    name: str = "PI controller"

    def __post_init__(self) -> None:
        self.integral_error: float = 0.0

    def reset(self) -> None:
        self.integral_error = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        temp = float(obs[0])
        error = temp - self.target_temp

        self.integral_error += error * self.dt
        self.integral_error = float(
            np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        )

        action = self.bias + self.kp * error + self.ki * self.integral_error
        action = float(np.clip(action, 0.0, 1.0))
        return np.array([action], dtype=np.float32)


def build_default_controllers(target_temp: float = 30.0, dt: float = 1.0) -> list[ThermalController]:
    """Return the standard controller set for Phase 2 comparisons."""
    return [
        NoCoolingController(),
        ConstantCoolingController(cooling_level=0.5),
        ConstantCoolingController(cooling_level=1.0),
        BangBangController(target_temp=target_temp, deadband=1.0),
        ProportionalController(target_temp=target_temp, kp=0.08, bias=0.15),
        PIController(
    target_temp=target_temp,
    kp=0.20,
    ki=0.006,
    bias=0.25,
    dt=dt,
    integral_limit=75.0,
    name="PI controller tuned",
),
    ]
