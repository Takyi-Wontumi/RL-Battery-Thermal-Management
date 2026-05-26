"""
envs/battery_pack_thermal_env_3d.py

Phase 2 — 3D cell-resolved battery pack thermal management environment.

Replaces the 1D chain thermal network with a full 3D lumped-parameter thermal
network.  Each cell [i, j, k] has its own temperature state, heat generation
term, neighbor-conduction interaction with up to 6 adjacent cells, and
geometry-based cooling exposure.

This is NOT CFD.  It is a 3D lumped-parameter thermal network designed for
controller development and battery cooling strategy comparison.

State (7-element observation):
    [T_max_norm, T_avg_norm, T_min_norm,
     T_gradient_norm, T_variance_norm, T_center_norm,
     u_prev]

    All temperatures normalized by (safe_temp - target_temp) so that:
        obs[k] = 0  →  temperature is at the target
        obs[k] = 1  →  temperature is at the safe limit

    T_center is the geometric center cell — typically the hardest to cool
    and an early indicator of hotspot formation that T_max alone misses.

Action:
    u ∈ [0, 1]  — shared pack cooling command

Reward penalizes:
    - Max cell temperature above target
    - Temperature gradient (hotspot spread)
    - Over-temperature violations
    - Cooling effort
    - Action chattering
    - Hard thermal failure
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.pack_config import CellConfig, PackConfig, build_default_configs
from models.thermal_model_3d import BatteryPackThermal3D


# ---------------------------------------------------------------------------
# Heat profile type
# ---------------------------------------------------------------------------

# Callable: (time_s, rng, pack_shape) → np.ndarray of shape pack_shape
Pack3DHeatProfile = Callable[[float, np.random.Generator, Tuple[int, int, int]], np.ndarray]


# ---------------------------------------------------------------------------
# Built-in heat profiles
# ---------------------------------------------------------------------------

def uniform_constant_3d_heat(q_total_w: float = 100.0) -> Pack3DHeatProfile:
    """Uniform heat distributed equally across all cells."""

    def profile(t: float, rng: np.random.Generator, shape: Tuple[int, int, int]) -> np.ndarray:
        n = int(np.prod(shape))
        return np.full(shape, q_total_w / n, dtype=np.float64)

    return profile


def nonuniform_step_3d_heat(
    q_low_w: float = 60.0,
    q_high_w: float = 200.0,
    step_time_s: float = 500.0,
    hotspot_factor: float = 1.4,
) -> Pack3DHeatProfile:
    """Step heat profile — total heat steps up after step_time_s.
    Center cell carries a persistent hotspot."""

    def profile(t: float, rng: np.random.Generator, shape: Tuple[int, int, int]) -> np.ndarray:
        n = int(np.prod(shape))
        q_total = q_low_w if t < step_time_s else q_high_w
        weights = np.ones(shape, dtype=np.float64)
        cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
        weights[cx, cy, cz] *= hotspot_factor
        weights /= weights.sum()
        return (q_total * weights).astype(np.float64)

    return profile


def pulsed_hotspot_3d_heat(
    q_low_w: float = 40.0,
    q_high_w: float = 220.0,
    period_s: float = 160.0,
    duty_cycle: float = 0.40,
    hotspot_factor: float = 1.6,
) -> Pack3DHeatProfile:
    """Pulsed total heat with a center-cluster hotspot."""

    def profile(t: float, rng: np.random.Generator, shape: Tuple[int, int, int]) -> np.ndarray:
        n = int(np.prod(shape))
        phase = (t % period_s) / period_s
        q_total = q_high_w if phase < duty_cycle else q_low_w
        weights = np.ones(shape, dtype=np.float64)
        cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
        weights[cx, cy, cz] *= hotspot_factor
        # Adjacent cells also run hotter
        if cx > 0:
            weights[cx - 1, cy, cz] *= 1.4
        weights /= weights.sum()
        return (q_total * weights).astype(np.float64)

    return profile


def random_nonuniform_3d_heat(
    q_mean_w: float = 100.0,
    q_std_w: float = 6.0,
    smoothing: float = 0.88,
    q_min_w: float = 55.0,
    q_max_w: float = 180.0,
) -> Pack3DHeatProfile:
    """Smoothly varying random total heat with slowly drifting cell distribution."""
    _state: Dict = {"q_total": q_mean_w, "weights": None, "shape": None}

    def profile(t: float, rng: np.random.Generator, shape: Tuple[int, int, int]) -> np.ndarray:
        if _state["weights"] is None or _state["shape"] != shape:
            raw = rng.uniform(0.85, 1.15, size=shape).astype(np.float64)
            cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
            raw[cx, cy, cz] *= 1.5
            raw /= raw.sum()
            _state["weights"] = raw
            _state["shape"] = shape

        disturbance = rng.normal(0.0, q_std_w)
        _state["q_total"] = smoothing * _state["q_total"] + (1.0 - smoothing) * q_mean_w + disturbance
        q_total = float(np.clip(_state["q_total"], q_min_w, q_max_w))

        w = _state["weights"].copy()
        w += rng.normal(0.0, 0.003, size=shape)
        w = np.clip(w, 0.01, None)
        w /= w.sum()
        _state["weights"] = w

        return (q_total * w).astype(np.float64)

    return profile


def make_3d_profile(profile_name: str) -> Pack3DHeatProfile:
    profiles: Dict[str, Pack3DHeatProfile] = {
        "UniformConstant": uniform_constant_3d_heat(),
        "NonuniformStep": nonuniform_step_3d_heat(),
        "PulsedHotspot": pulsed_hotspot_3d_heat(),
        "RandomNonuniform": random_nonuniform_3d_heat(),
    }
    if profile_name not in profiles:
        raise KeyError(
            f"Unknown 3D profile '{profile_name}'. Choose from {list(profiles.keys())}"
        )
    return profiles[profile_name]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BatteryPackThermalEnv3D(gym.Env):
    """
    3D cell-resolved battery pack thermal management environment.

    The thermal plant is a BatteryPackThermal3D instance.  The controller
    sees pack-level aggregated metrics (T_max, T_avg, T_min, T_gradient) so
    it can respond to hotspot formation without needing the full 3D state.

    Observation (7D):
        [T_max_norm, T_avg_norm, T_min_norm,
         T_gradient_norm, T_variance_norm, T_center_norm,
         u_prev]

    Action (1D):
        u ∈ [0, 1] — shared cooling command applied to all boundary cells.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cell_config: Optional[CellConfig] = None,
        pack_config: Optional[PackConfig] = None,
        heat_profile: Optional[Pack3DHeatProfile] = None,
        total_time_s: float = 1800.0,
        dt_s: float = 1.0,
        seed: Optional[int] = 7,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.cell_config = cell_config or CellConfig()
        self.pack_config = pack_config or PackConfig()
        self.heat_profile = heat_profile or nonuniform_step_3d_heat()
        self.total_time_s = total_time_s
        self.dt_s = dt_s
        self.render_mode = render_mode

        self.rng = np.random.default_rng(seed)

        self._normalizer = float(
            max(1e-6, self.pack_config.safe_temp_c - self.pack_config.target_temp_c)
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.thermal_model = BatteryPackThermal3D(
            self.cell_config, self.pack_config, rng=self.rng
        )

        self.time_s: float = 0.0
        self.u_prev: float = 0.0
        self.episode_log: Dict[str, list] = {}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.thermal_model.rng = self.rng

        options = options or {}
        randomize = bool(options.get("randomize", False))

        if randomize:
            spread = self.pack_config.safe_temp_c - self.pack_config.target_temp_c
            base_temp = self.pack_config.initial_temp_c + self.rng.uniform(-1.0, 1.0)
            self.thermal_model.T = np.full(
                self.pack_config.shape,
                base_temp,
                dtype=np.float64,
            ) + self.rng.normal(0.0, 0.3, size=self.pack_config.shape)
        else:
            self.thermal_model.reset()

        self.time_s = 0.0
        self.u_prev = 0.0
        self._reset_log()

        obs = self.thermal_model.get_observation(self.u_prev)
        return obs, self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        u = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))

        q_gen = self.heat_profile(self.time_s, self.rng, self.pack_config.shape)
        T, metrics = self.thermal_model.step(u=u, dt=self.dt_s, q_gen=q_gen)

        self.time_s += self.dt_s

        reward = self._compute_reward(metrics, u)

        terminated = bool(metrics["critical"] or np.min(T) <= -10.0)
        truncated = bool(self.time_s >= self.total_time_s)

        self._log_step(u, reward, metrics, q_gen)
        self.u_prev = u

        obs = self.thermal_model.get_observation(self.u_prev)
        return obs, reward, terminated, truncated, self._get_info(metrics)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, metrics: Dict, u: float) -> float:
        cfg = self.pack_config
        normalizer = self._normalizer

        T_max = metrics["T_max"]
        T_gradient = metrics["T_gradient"]

        # Max-temp tracking cost
        max_temp_error = (T_max - cfg.target_temp_c) / normalizer
        max_temp_cost = 2.0 * max_temp_error ** 2

        # Over-temperature penalty — unnormalized quadratic makes safety the
        # dominant signal; every degree above safe_temp costs over_temp^2.
        over_temp = max(0.0, T_max - cfg.safe_temp_c)
        over_temp_cost = over_temp ** 2

        # Gradient (hotspot spread) penalty
        gradient_cost = 0.5 * (T_gradient / normalizer) ** 2

        # Cooling effort cost (reduced so safety dominates)
        action_cost = 0.02 * u ** 2

        # Action smoothness (reduced so safety dominates)
        smoothness_cost = 0.01 * (u - self.u_prev) ** 2

        # Hard failure
        hard_penalty = 150.0 if metrics["critical"] else 0.0

        reward = -(
            max_temp_cost
            + over_temp_cost
            + gradient_cost
            + action_cost
            + smoothness_cost
            + hard_penalty
        )
        return float(reward)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _get_info(self, metrics: Optional[Dict] = None) -> Dict:
        if metrics is None:
            metrics = self.thermal_model.get_metrics()
        n_above_safe = int(np.sum(self.thermal_model.T > self.pack_config.safe_temp_c))
        return {
            "time_s": self.time_s,
            "T_max": metrics["T_max"],
            "T_avg": metrics["T_avg"],
            "T_min": metrics["T_min"],
            "T_gradient": metrics["T_gradient"],
            "safe": metrics["safe"],
            "critical": metrics["critical"],
            "n_cells_above_safe": n_above_safe,
            "u_prev": self.u_prev,
            "temperatures_3d": self.thermal_model.T.copy(),
        }

    def _reset_log(self) -> None:
        self.episode_log = {
            "time": [],
            "T_max": [],
            "T_avg": [],
            "T_min": [],
            "T_gradient": [],
            "n_cells_above_safe": [],
            "action": [],
            "reward": [],
            "q_gen_total": [],
            "q_gen_max_cell": [],
            "q_cool_total": [],
        }

    def _log_step(self, u: float, reward: float, metrics: Dict, q_gen: np.ndarray) -> None:
        self.episode_log["time"].append(self.time_s)
        self.episode_log["T_max"].append(metrics["T_max"])
        self.episode_log["T_avg"].append(metrics["T_avg"])
        self.episode_log["T_min"].append(metrics["T_min"])
        self.episode_log["T_gradient"].append(metrics["T_gradient"])
        self.episode_log["n_cells_above_safe"].append(
            int(np.sum(self.thermal_model.T > self.pack_config.safe_temp_c))
        )
        self.episode_log["action"].append(u)
        self.episode_log["reward"].append(reward)
        self.episode_log["q_gen_total"].append(float(np.sum(q_gen)))
        self.episode_log["q_gen_max_cell"].append(float(np.max(q_gen)))
        self.episode_log["q_cool_total"].append(metrics.get("q_cool_total", 0.0))

    def get_episode_log(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.episode_log.items()}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = BatteryPackThermalEnv3D(
        pack_config=PackConfig(shape=(4, 3, 2)),
        heat_profile=nonuniform_step_3d_heat(),
        seed=7,
    )

    obs, info = env.reset(seed=7, options={"randomize": False})
    print(f"obs shape: {obs.shape}  obs: {obs}")

    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    log = env.get_episode_log()
    print(f"\nSmoke test complete — 3D pack env")
    print(f"Final T_max:      {log['T_max'][-1]:.2f} °C")
    print(f"Final T_avg:      {log['T_avg'][-1]:.2f} °C")
    print(f"Final T_gradient: {log['T_gradient'][-1]:.3f} °C")
    print(f"Total reward:     {total_reward:.2f}")
