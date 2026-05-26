"""
envs/battery_pack_thermal_env.py

Phase 4 battery pack thermal environment.

This upgrades the single lumped battery model into a multi-node battery pack model.
Each node represents a cell or module with its own temperature and heat generation.
Neighboring nodes exchange heat through conductive coupling, while a shared cooling
command removes heat from all nodes.

State idea:
    T_i for each cell/module
    Q_i for each cell/module
    ambient temperature
    time fraction
    previous cooling command

Action:
    u in [0, 1] shared cooling command

Reward penalizes:
    - maximum temperature above target
    - pack temperature imbalance
    - overheating
    - cooling effort
    - action chattering
    - hard thermal failure

This is still not CFD. It is a control-oriented multi-node thermal network model.
That is exactly the right next step before pretending to model a real EV pack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


PackHeatProfile = Callable[[float, np.random.Generator, int], np.ndarray]


@dataclass
class BatteryPackThermalConfig:
    # Simulation
    n_cells: int = 16
    total_time: float = 1800.0
    dt: float = 1.0
    seed: Optional[int] = 7

    # Thermal properties per cell/module
    initial_temp: float = 25.0
    ambient_temp: float = 25.0
    thermal_capacitance: float = 8_000.0  # J/K per cell/module node
    surface_area_per_cell: float = 0.12   # m^2 effective cooling area per node

    # Cooling model
    h_min: float = 5.0
    h_max: float = 95.0
    cooling_nonlinearity: float = 1.0
    direct_cooling_max_per_cell: float = 0.0

    # Coupling between adjacent cells/modules
    conduction_coupling: float = 8.0  # W/K between neighboring nodes

    # Temperature limits
    target_temp: float = 35.0
    soft_max_temp: float = 45.0
    hard_max_temp: float = 60.0
    min_temp: float = -10.0

    # Reward weights
    max_temp_weight: float = 2.0
    mean_temp_weight: float = 0.1
    imbalance_weight: float = 1.0
    over_temp_weight: float = 15.0
    action_weight: float = 0.06
    action_smoothness_weight: float = 0.04
    hard_violation_penalty: float = 150.0

    # Randomization
    initial_temp_randomization: float = 1.0
    ambient_randomization: float = 1.5
    cell_heat_variation: float = 0.12

    # Physical cell spacing (optional).  When set, overrides conduction_coupling
    # using Fourier's law: g = g_ref * (d_ref / d_spacing).
    # Reference point: 2 mm gap ↔ 8.0 W/K.
    cell_spacing_m: Optional[float] = None

    def __post_init__(self) -> None:
        if self.cell_spacing_m is not None and self.cell_spacing_m > 0:
            self.conduction_coupling = 8.0 * (0.002 / self.cell_spacing_m)


# -----------------------------------------------------------------------------
# Heat profiles
# -----------------------------------------------------------------------------

def uniform_constant_pack_heat(q_total: float = 700.0) -> PackHeatProfile:
    """Uniform constant total heat distributed across cells."""

    def profile(t: float, rng: np.random.Generator, n_cells: int) -> np.ndarray:
        return np.full(n_cells, q_total / n_cells, dtype=np.float32)

    return profile


def nonuniform_step_pack_heat(
    q_low_total: float = 400.0,
    q_high_total: float = 1_100.0,
    step_time: float = 500.0,
    hotspot_factor: float = 1.8,
) -> PackHeatProfile:
    """Step heat profile with one hotter region."""

    def profile(t: float, rng: np.random.Generator, n_cells: int) -> np.ndarray:
        q_total = q_low_total if t < step_time else q_high_total
        weights = np.ones(n_cells, dtype=np.float32)
        hotspot_idx = n_cells // 2
        weights[hotspot_idx] *= hotspot_factor
        weights /= np.sum(weights)
        return (q_total * weights).astype(np.float32)

    return profile


def pulsed_hotspot_pack_heat(
    q_low_total: float = 300.0,
    q_high_total: float = 1_200.0,
    period: float = 160.0,
    duty_cycle: float = 0.40,
    hotspot_factor: float = 2.2,
) -> PackHeatProfile:
    """Pulsed total heat with a stronger hotspot cell/module."""

    def profile(t: float, rng: np.random.Generator, n_cells: int) -> np.ndarray:
        phase = (t % period) / period
        q_total = q_high_total if phase < duty_cycle else q_low_total
        weights = np.ones(n_cells, dtype=np.float32)
        weights[n_cells // 2] *= hotspot_factor
        weights[max(0, n_cells // 2 - 1)] *= 1.4
        weights /= np.sum(weights)
        return (q_total * weights).astype(np.float32)

    return profile


def random_nonuniform_pack_heat(
    q_mean_total: float = 650.0,
    q_std_total: float = 18.0,
    smoothing: float = 0.88,
    q_min_total: float = 450.0,
    q_max_total: float = 900.0,
) -> PackHeatProfile:
    """Smooth random total heat with slowly changing cell-to-cell nonuniformity."""
    state: Dict[str, np.ndarray | float] = {
        "q_total": q_mean_total,
        "weights": np.array([], dtype=np.float32),
    }

    def profile(t: float, rng: np.random.Generator, n_cells: int) -> np.ndarray:
        if len(state["weights"]) != n_cells:  # type: ignore[arg-type]
            raw_weights = rng.uniform(0.85, 1.15, size=n_cells).astype(np.float32)
            raw_weights[n_cells // 2] *= 1.5
            raw_weights /= np.sum(raw_weights)
            state["weights"] = raw_weights

        disturbance = rng.normal(0.0, q_std_total)
        state["q_total"] = smoothing * float(state["q_total"]) + (1.0 - smoothing) * q_mean_total + disturbance
        q_total = float(np.clip(float(state["q_total"]), q_min_total, q_max_total))

        weights = np.asarray(state["weights"], dtype=np.float32)
        weight_noise = rng.normal(0.0, 0.003, size=n_cells).astype(np.float32)
        weights = np.clip(weights + weight_noise, 0.01, None)
        weights /= np.sum(weights)
        state["weights"] = weights

        return (q_total * weights).astype(np.float32)

    return profile


def make_pack_profile(profile_name: str) -> PackHeatProfile:
    profiles: Dict[str, PackHeatProfile] = {
        "UniformConstant": uniform_constant_pack_heat(),
        "NonuniformStep": nonuniform_step_pack_heat(),
        "PulsedHotspot": pulsed_hotspot_pack_heat(),
        "RandomNonuniform": random_nonuniform_pack_heat(),
    }
    if profile_name not in profiles:
        raise KeyError(f"Unknown pack profile '{profile_name}'. Choose from {list(profiles.keys())}")
    return profiles[profile_name]


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

class BatteryPackThermalEnv(gym.Env):
    """
    Multi-node battery pack thermal management environment.

    The model is a 1D chain thermal network:
        cell 0 <-> cell 1 <-> ... <-> cell n-1

    Each node has:
        C dT_i/dt = Q_i + conduction_from_neighbors - cooling_i
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[BatteryPackThermalConfig] = None,
        heat_profile: Optional[PackHeatProfile] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.config = config or BatteryPackThermalConfig()
        self.heat_profile = heat_profile or nonuniform_step_pack_heat()
        self.render_mode = render_mode

        self.rng = np.random.default_rng(self.config.seed)

        n = self.config.n_cells

        # Observation:
        # [T_cells normalized raw-ish, Q_cells scaled, ambient, time_fraction, previous_action]
        obs_dim = 2 * n + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.time: float = 0.0
        self.step_count: int = 0
        self.temperatures = np.full(n, self.config.initial_temp, dtype=np.float32)
        self.ambient_temperature: float = self.config.ambient_temp
        self.heat_generation = np.zeros(n, dtype=np.float32)
        self.previous_action: float = 0.0
        self.episode_log: Dict[str, list] = {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        randomize = bool(options.get("randomize", False))

        cfg = self.config
        n = cfg.n_cells

        self.time = 0.0
        self.step_count = 0
        self.previous_action = 0.0

        if randomize:
            base_temp = cfg.initial_temp + self.rng.uniform(
                -cfg.initial_temp_randomization,
                cfg.initial_temp_randomization,
            )
            cell_offsets = self.rng.normal(0.0, 0.35, size=n)
            self.temperatures = (base_temp + cell_offsets).astype(np.float32)
            self.ambient_temperature = float(
                cfg.ambient_temp + self.rng.uniform(-cfg.ambient_randomization, cfg.ambient_randomization)
            )
        else:
            self.temperatures = np.full(n, cfg.initial_temp, dtype=np.float32)
            self.ambient_temperature = cfg.ambient_temp

        self.heat_generation = self.heat_profile(self.time, self.rng, n).astype(np.float32)
        self._reset_log()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        cfg = self.config
        n = cfg.n_cells

        u = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))

        self.heat_generation = self.heat_profile(self.time, self.rng, n).astype(np.float32)

        h = self._cooling_coefficient(u)
        cooling = h * cfg.surface_area_per_cell * (self.temperatures - self.ambient_temperature)
        cooling += cfg.direct_cooling_max_per_cell * u

        conduction = self._compute_conduction_terms(self.temperatures)

        dTdt = (self.heat_generation + conduction - cooling) / cfg.thermal_capacitance
        self.temperatures = self.temperatures + cfg.dt * dTdt.astype(np.float32)

        self.time += cfg.dt
        self.step_count += 1

        reward = self._compute_reward(u)

        terminated = bool(
            np.max(self.temperatures) >= cfg.hard_max_temp
            or np.min(self.temperatures) <= cfg.min_temp
        )
        truncated = bool(self.time >= cfg.total_time)

        self._log_step(u, reward, h, cooling, conduction)
        self.previous_action = u

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _compute_conduction_terms(self, temperatures: np.ndarray) -> np.ndarray:
        """Compute neighbor conduction for a 1D cell/module chain."""
        k = self.config.conduction_coupling
        conduction = np.zeros_like(temperatures, dtype=np.float32)

        for i in range(len(temperatures)):
            if i > 0:
                conduction[i] += k * (temperatures[i - 1] - temperatures[i])
            if i < len(temperatures) - 1:
                conduction[i] += k * (temperatures[i + 1] - temperatures[i])

        return conduction

    def _cooling_coefficient(self, u: float) -> float:
        cfg = self.config
        shaped_u = u ** cfg.cooling_nonlinearity
        return float(cfg.h_min + (cfg.h_max - cfg.h_min) * shaped_u)

    def _compute_reward(self, u: float) -> float:
        cfg = self.config

        max_temp = float(np.max(self.temperatures))
        mean_temp = float(np.mean(self.temperatures))
        temp_std = float(np.std(self.temperatures))

        normalizer = max(1e-6, cfg.soft_max_temp - cfg.target_temp)

        max_temp_error = (max_temp - cfg.target_temp) / normalizer
        mean_temp_error = (mean_temp - cfg.target_temp) / normalizer

        max_temp_cost = cfg.max_temp_weight * max_temp_error**2
        mean_temp_cost = cfg.mean_temp_weight * mean_temp_error**2
        imbalance_cost = cfg.imbalance_weight * (temp_std / normalizer) ** 2

        over_temp = max(0.0, max_temp - cfg.soft_max_temp)
        over_temp_cost = over_temp ** 2

        action_cost = 0.02 * u ** 2
        smoothness_cost = 0.01 * (u - self.previous_action) ** 2

        hard_penalty = 0.0
        if max_temp >= cfg.hard_max_temp or np.min(self.temperatures) <= cfg.min_temp:
            hard_penalty = cfg.hard_violation_penalty

        reward = -(
            max_temp_cost
            + mean_temp_cost
            + imbalance_cost
            + over_temp_cost
            + action_cost
            + smoothness_cost
            + hard_penalty
        )

        return float(reward)

    def _get_obs(self) -> np.ndarray:
        cfg = self.config

        # Keep observation numerically reasonable for RL.
        temp_obs = (self.temperatures - cfg.target_temp) / max(1e-6, cfg.soft_max_temp - cfg.target_temp)
        heat_obs = self.heat_generation / 250.0
        ambient_obs = np.array([(self.ambient_temperature - cfg.target_temp) / 20.0], dtype=np.float32)
        time_obs = np.array([self.time / cfg.total_time], dtype=np.float32)
        prev_action_obs = np.array([self.previous_action], dtype=np.float32)

        return np.concatenate(
            [
                temp_obs.astype(np.float32),
                heat_obs.astype(np.float32),
                ambient_obs,
                time_obs,
                prev_action_obs,
            ]
        ).astype(np.float32)

    def _get_info(self) -> Dict:
        return {
            "time": self.time,
            "temperatures": self.temperatures.copy(),
            "max_temperature": float(np.max(self.temperatures)),
            "mean_temperature": float(np.mean(self.temperatures)),
            "min_temperature": float(np.min(self.temperatures)),
            "temperature_std": float(np.std(self.temperatures)),
            "ambient_temperature": self.ambient_temperature,
            "heat_generation": self.heat_generation.copy(),
            "total_heat_generation": float(np.sum(self.heat_generation)),
            "previous_action": self.previous_action,
        }

    def _reset_log(self) -> None:
        self.episode_log = {
            "time": [],
            "temperatures": [],
            "max_temperature": [],
            "mean_temperature": [],
            "min_temperature": [],
            "temperature_std": [],
            "ambient_temperature": [],
            "heat_generation": [],
            "total_heat_generation": [],
            "action": [],
            "cooling_coefficient": [],
            "cooling_per_cell": [],
            "total_cooling": [],
            "conduction": [],
            "reward": [],
        }

    def _log_step(
        self,
        u: float,
        reward: float,
        h: float,
        cooling: np.ndarray,
        conduction: np.ndarray,
    ) -> None:
        self.episode_log["time"].append(self.time)
        self.episode_log["temperatures"].append(self.temperatures.copy())
        self.episode_log["max_temperature"].append(float(np.max(self.temperatures)))
        self.episode_log["mean_temperature"].append(float(np.mean(self.temperatures)))
        self.episode_log["min_temperature"].append(float(np.min(self.temperatures)))
        self.episode_log["temperature_std"].append(float(np.std(self.temperatures)))
        self.episode_log["ambient_temperature"].append(self.ambient_temperature)
        self.episode_log["heat_generation"].append(self.heat_generation.copy())
        self.episode_log["total_heat_generation"].append(float(np.sum(self.heat_generation)))
        self.episode_log["action"].append(u)
        self.episode_log["cooling_coefficient"].append(h)
        self.episode_log["cooling_per_cell"].append(cooling.copy())
        self.episode_log["total_cooling"].append(float(np.sum(cooling)))
        self.episode_log["conduction"].append(conduction.copy())
        self.episode_log["reward"].append(reward)

    def get_episode_log(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for key, values in self.episode_log.items():
            out[key] = np.asarray(values)
        return out


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    env = BatteryPackThermalEnv(
        config=BatteryPackThermalConfig(n_cells=8),
        heat_profile=nonuniform_step_pack_heat(),
    )

    obs, info = env.reset(seed=7, options={"randomize": False})
    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print("Smoke test complete")
    print(f"Final max temperature: {info['max_temperature']:.2f} C")
    print(f"Final mean temperature: {info['mean_temperature']:.2f} C")
    print(f"Final temperature std: {info['temperature_std']:.3f} C")
    print(f"Total reward: {total_reward:.2f}")
