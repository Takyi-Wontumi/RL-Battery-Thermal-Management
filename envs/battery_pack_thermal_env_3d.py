"""
envs/battery_pack_thermal_env_3d.py

3D cell-resolved battery pack thermal management environment.

== What changed and why ==

Multi-zone cooling (action space upgrade):
    Previously: u ∈ [0, 1]  — one command for the whole pack.
    Now:        u ∈ [0, 1]^Nx — one independent command per x-column (zone).

    WHY: With one command the agent is forced to over-cool boundary cells
    to reach interior hot-spots, wasting energy.  With Nx = 4 zones the
    agent can direct high cooling exactly to the column that is hottest
    while leaving cooler columns at lower effort.

    Backward compatibility: step() accepts a scalar or shape-(1,) action and
    broadcasts it to all zones, so existing baseline controllers still run.

Electrochemical heat profile (optional):
    Pass an ElectrochemicalHeatProfile from models/electrochemical_model.py
    instead of a prescribed pattern.  When detected, two SOC features are
    added to the observation so the agent can anticipate the rising resistance
    as the pack discharges.

Observation (multi-zone, no electrochemical):
    [T_max_norm, T_avg_norm, T_min_norm,
     T_gradient_norm, T_variance_norm, T_center_norm,
     u_prev_zone_0, ..., u_prev_zone_{Nx-1}]
    dim = 6 + Nx

Observation (multi-zone + electrochemical):
    [T_max_norm, T_avg_norm, T_min_norm,
     T_gradient_norm, T_variance_norm, T_center_norm,
     SOC_mean, SOC_min,
     u_prev_zone_0, ..., u_prev_zone_{Nx-1}]
    dim = 8 + Nx

Action:
    u ∈ [0, 1]^Nx — Nx independent cooling commands (one per x-column).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

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

# Callable: (time_s, rng, pack_shape, T=None) → np.ndarray of shape pack_shape
Pack3DHeatProfile = Callable[[float, np.random.Generator, Tuple[int, int, int]], np.ndarray]


# ---------------------------------------------------------------------------
# Built-in prescribed heat profiles (unchanged, backward compatible)
# ---------------------------------------------------------------------------

def uniform_constant_3d_heat(q_total_w: float = 12.0) -> Pack3DHeatProfile:
    """Uniform heat distributed equally across all cells."""

    def profile(t: float, rng: np.random.Generator, shape: Tuple[int, int, int], **_) -> np.ndarray:
        return np.full(shape, q_total_w / int(np.prod(shape)), dtype=np.float64)

    return profile


def nonuniform_step_3d_heat(
    q_low_w: float = 8.0,
    q_high_w: float = 20.0,
    step_time_s: float = 500.0,
    hotspot_factor: float = 1.4,
) -> Pack3DHeatProfile:
    """Step heat profile with a center-cell hotspot."""

    def profile(t: float, rng: np.random.Generator, shape: Tuple[int, int, int], **_) -> np.ndarray:
        q_total = q_low_w if t < step_time_s else q_high_w
        weights = np.ones(shape, dtype=np.float64)
        cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
        weights[cx, cy, cz] *= hotspot_factor
        weights /= weights.sum()
        return (q_total * weights).astype(np.float64)

    return profile


def pulsed_hotspot_3d_heat(
    q_low_w: float = 5.0,
    q_high_w: float = 22.0,
    period_s: float = 160.0,
    duty_cycle: float = 0.40,
    hotspot_factor: float = 1.6,
) -> Pack3DHeatProfile:
    """Pulsed total heat with a center-cluster hotspot."""

    def profile(t: float, rng: np.random.Generator, shape: Tuple[int, int, int], **_) -> np.ndarray:
        phase = (t % period_s) / period_s
        q_total = q_high_w if phase < duty_cycle else q_low_w
        weights = np.ones(shape, dtype=np.float64)
        cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
        weights[cx, cy, cz] *= hotspot_factor
        if cx > 0:
            weights[cx - 1, cy, cz] *= 1.4
        weights /= weights.sum()
        return (q_total * weights).astype(np.float64)

    return profile


def random_nonuniform_3d_heat(
    q_mean_w: float = 12.0,
    q_std_w: float = 1.5,
    smoothing: float = 0.88,
    q_min_w: float = 7.0,
    q_max_w: float = 18.0,
) -> Pack3DHeatProfile:
    """Smoothly varying random total heat with slowly drifting cell distribution."""
    _state: Dict = {"q_total": q_mean_w, "weights": None, "shape": None}

    def profile(t: float, rng: np.random.Generator, shape: Tuple[int, int, int], **_) -> np.ndarray:
        if _state["weights"] is None or _state["shape"] != shape:
            raw = rng.uniform(0.85, 1.15, size=shape).astype(np.float64)
            cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
            raw[cx, cy, cz] *= 1.5
            raw /= raw.sum()
            _state["weights"] = raw
            _state["shape"] = shape

        _state["q_total"] = (
            smoothing * _state["q_total"]
            + (1.0 - smoothing) * q_mean_w
            + rng.normal(0.0, q_std_w)
        )
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
# Shared evaluation reward — identical for ALL controllers
#
# Design rationale (each revision addressed):
#   R1  One function called by every controller's env step → no rigging.
#   R3  Asymmetric: overheating hurts 6.7× more per °C than overcooling.
#   R4  Every term normalised before squaring → clear priority order.
#   R5  Safe band [target_low, target_high]: zero penalty inside the band.
#   R6  Returns all components for per-step logging and post-run diagnosis.
# ---------------------------------------------------------------------------

_REWARD_COMPONENT_KEYS = (
    "reward_too_hot",
    "reward_too_cold",
    "reward_safe",
    "reward_critical",
    "reward_spread",
    "reward_energy",
    "reward_smooth",
    "reward_hard_penalty",
)


def compute_reward_components(
    T: np.ndarray,
    u_zones: np.ndarray,
    u_prev: np.ndarray,
    cfg: "PackConfig",
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the scalar reward and its named components.

    This is the ONLY reward function used by every controller during
    both training and evaluation — baselines and RL agents are scored
    identically.

    Priority order (highest → lowest weight):
        critical violation  (-200 × normalised²  + -150 hard)
        safe violation      (-20  × normalised²)
        band tracking       (-1   × hot²  + -0.15 × cold²)
        spatial spread      (-0.50× normalised²)
        cooling energy      (-0.05× mean u²)
        actuator smoothness (-0.02× mean Δu²)

    Args:
        T        : (Nx, Ny, Nz) temperature array from the thermal model (°C)
        u_zones  : current cooling command per zone, clipped to [0, 1]
        u_prev   : previous step's cooling command
        cfg      : PackConfig supplying target_low, target_high, safe, critical

    Returns:
        (total_reward, components_dict)
    """
    T_arr  = np.asarray(T, dtype=np.float64)
    T_max  = float(T_arr.max())
    T_min  = float(T_arr.min())
    T_mean = float(T_arr.mean())

    tl = cfg.target_low_temp_c
    th = cfg.target_high_temp_c
    safe = cfg.safe_temp_c
    crit = cfg.critical_temp_c

    # ── Band tracking (normalised to 10 °C reference) ──
    too_hot  = max(0.0, T_max  - th) / 10.0
    too_cold = max(0.0, tl - T_mean) / 10.0

    # ── Safety violations (normalised to 5 °C reference) ──
    safe_viol = max(0.0, T_max - safe) / 5.0
    crit_viol = max(0.0, T_max - crit) / 5.0

    # ── Spatial uniformity ──
    spread = (T_max - T_min) / 5.0

    # ── Control cost ──
    u = np.asarray(u_zones, dtype=np.float64)
    up = np.asarray(u_prev,  dtype=np.float64)
    energy = float(np.mean(np.square(u)))
    smooth = float(np.mean(np.square(u - up)))

    # ── Component values ──
    r_hot   = -1.00 * too_hot  ** 2
    r_cold  = -0.15 * too_cold ** 2
    r_safe  = -20.0 * safe_viol ** 2
    r_crit  = -200.0 * crit_viol ** 2
    r_sprd  = -0.50 * spread ** 2
    r_ener  = -0.05 * energy
    r_smth  = -0.02 * smooth
    r_hard  = -150.0 if T_max >= crit else 0.0

    total = r_hot + r_cold + r_safe + r_crit + r_sprd + r_ener + r_smth + r_hard

    components: Dict[str, float] = {
        "reward_too_hot":     r_hot,
        "reward_too_cold":    r_cold,
        "reward_safe":        r_safe,
        "reward_critical":    r_crit,
        "reward_spread":      r_sprd,
        "reward_energy":      r_ener,
        "reward_smooth":      r_smth,
        "reward_hard_penalty": r_hard,
    }
    return float(total), components


# ---------------------------------------------------------------------------
# Helper: call heat profile with or without temperature array
# ---------------------------------------------------------------------------

def _call_heat_profile(
    profile: Pack3DHeatProfile,
    t: float,
    rng: np.random.Generator,
    shape: Tuple[int, int, int],
    T: Optional[np.ndarray],
) -> np.ndarray:
    """
    Call a heat profile, passing T only if the profile accepts it.

    ElectrochemicalHeatProfile and built-in profiles accept T via **_.
    External profiles (e.g. randomized_3d_heat_profile) may not — the
    TypeError is caught and the call is retried without T.
    """
    try:
        return profile(t, rng, shape, T=T)
    except TypeError:
        return profile(t, rng, shape)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BatteryPackThermalEnv3D(gym.Env):
    """
    3D cell-resolved battery pack thermal management environment.

    Action (multi-zone):
        u ∈ [0, 1]^Nx — one cooling command per x-column (zone).
        A scalar or shape-(1,) action is broadcast to all zones for backward compat.

    Observation (without electrochemical profile):
        dim = 6 + Nx
        [T_max_norm, T_avg_norm, T_min_norm,
         T_gradient_norm, T_variance_norm, T_center_norm,
         u_prev_zone_0, ..., u_prev_zone_{Nx-1}]

    Observation (with ElectrochemicalHeatProfile):
        dim = 8 + Nx
        [T_max_norm, T_avg_norm, T_min_norm,
         T_gradient_norm, T_variance_norm, T_center_norm,
         SOC_mean, SOC_min,
         u_prev_zone_0, ..., u_prev_zone_{Nx-1}]
    """

    metadata = {"render_modes": ["human"]}

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

        # Number of independent cooling zones = number of x-columns
        self.n_zones: int = self.pack_config.shape[0]

        # Detect electrochemical profile by duck-typing on soc_mean property
        self._has_echem: bool = hasattr(self.heat_profile, "soc_mean")

        # Observation dimension: 6 thermal + 2 SOC (if echem) + n_zones action history
        self._obs_dim: int = 6 + (2 if self._has_echem else 0) + self.n_zones

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.zeros(self.n_zones, dtype=np.float32),
            high=np.ones(self.n_zones, dtype=np.float32),
            dtype=np.float32,
        )

        self.thermal_model = BatteryPackThermal3D(
            self.cell_config, self.pack_config, rng=self.rng
        )

        self.time_s: float = 0.0
        self.u_prev_zones: np.ndarray = np.zeros(self.n_zones, dtype=np.float32)
        self.episode_log: Dict[str, list] = {}
        self._last_reward_components: Dict[str, float] = {k: 0.0 for k in _REWARD_COMPONENT_KEYS}

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
            base_temp = self.pack_config.initial_temp_c + self.rng.uniform(-1.0, 1.0)
            self.thermal_model.T = (
                np.full(self.pack_config.shape, base_temp, dtype=np.float64)
                + self.rng.normal(0.0, 0.3, size=self.pack_config.shape)
            )
        else:
            self.thermal_model.reset()

        self.time_s = 0.0
        self.u_prev_zones = np.zeros(self.n_zones, dtype=np.float32)
        self._last_reward_components = {k: 0.0 for k in _REWARD_COMPONENT_KEYS}
        self._reset_log()

        if self._has_echem:
            self.heat_profile.reset()

        obs = self._build_obs()
        return obs, self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        u_zones = self._parse_action(action)

        q_gen = _call_heat_profile(
            self.heat_profile,
            self.time_s,
            self.rng,
            self.pack_config.shape,
            self.thermal_model.T,
        )

        T, metrics = self.thermal_model.step(u_zones=u_zones, dt=self.dt_s, q_gen=q_gen)

        self.time_s += self.dt_s

        reward, self._last_reward_components = compute_reward_components(
            T=T,
            u_zones=u_zones,
            u_prev=self.u_prev_zones,
            cfg=self.pack_config,
        )

        terminated = bool(metrics["critical"] or np.min(T) <= -10.0)
        truncated = bool(self.time_s >= self.total_time_s)

        self._log_step(u_zones, reward, metrics, q_gen)
        self.u_prev_zones = u_zones.astype(np.float32)

        obs = self._build_obs()
        return obs, reward, terminated, truncated, self._get_info(metrics)

    # ------------------------------------------------------------------
    # Observation assembly
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """Assemble the full observation vector."""
        thermal_obs = self.thermal_model.get_thermal_obs()  # 6 elements

        if self._has_echem:
            soc_obs = np.array([
                self.heat_profile.soc_mean,
                self.heat_profile.soc_min,
            ], dtype=np.float32)
            return np.concatenate([thermal_obs, soc_obs, self.u_prev_zones])

        return np.concatenate([thermal_obs, self.u_prev_zones])

    # ------------------------------------------------------------------
    # Action parsing (backward compatible)
    # ------------------------------------------------------------------

    def _parse_action(self, action: np.ndarray) -> np.ndarray:
        """
        Accept scalar, shape-(1,), or shape-(n_zones,) action.
        Scalar/single-value is broadcast to all zones.
        """
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            arr = np.full(self.n_zones, float(arr[0]))
        elif arr.size != self.n_zones:
            raise ValueError(
                f"Expected action of size 1 or {self.n_zones}, got {arr.size}"
            )
        return np.clip(arr, 0.0, 1.0).astype(np.float64)

    # ------------------------------------------------------------------
    # Reward — delegated to module-level compute_reward_components()
    # All controllers (RL and classical) are scored identically.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Logging and info
    # ------------------------------------------------------------------

    def _get_info(self, metrics: Optional[Dict] = None) -> Dict:
        if metrics is None:
            metrics = self.thermal_model.get_metrics()
        n_above_safe = int(np.sum(self.thermal_model.T > self.pack_config.safe_temp_c))
        info = {
            "time_s": self.time_s,
            "T_max": metrics["T_max"],
            "T_avg": metrics["T_avg"],
            "T_min": metrics["T_min"],
            "T_gradient": metrics["T_gradient"],
            "safe": metrics["safe"],
            "critical": metrics["critical"],
            "n_cells_above_safe": n_above_safe,
            "u_prev_zones": self.u_prev_zones.tolist(),
            "temperatures_3d": self.thermal_model.T.copy(),
            "zone_max_temps": self.thermal_model.get_zone_max_temps().tolist(),
        }
        # Reward components — same keys for every controller (R1, R6)
        info.update(self._last_reward_components)
        if self._has_echem:
            info["soc_mean"] = self.heat_profile.soc_mean
            info["soc_min"] = self.heat_profile.soc_min
        return info

    def _reset_log(self) -> None:
        self.episode_log = {
            "time": [],
            "T_max": [],
            "T_avg": [],
            "T_min": [],
            "T_gradient": [],
            "n_cells_above_safe": [],
            "actions": [],          # shape (n_zones,) per step
            "reward": [],
            "q_gen_total": [],
            "q_gen_max_cell": [],
            "q_cool_total": [],
            "q_cool_per_zone": [],  # shape (n_zones,) per step
            "zone_max_temps": [],   # shape (n_zones,) per step
            # Reward components — logged every step for diagnosis (R6)
            **{k: [] for k in _REWARD_COMPONENT_KEYS},
        }
        if self._has_echem:
            self.episode_log["soc_mean"] = []
            self.episode_log["soc_min"] = []

    def _log_step(self, u_zones: np.ndarray, reward: float, metrics: Dict, q_gen: np.ndarray) -> None:
        self.episode_log["time"].append(self.time_s)
        self.episode_log["T_max"].append(metrics["T_max"])
        self.episode_log["T_avg"].append(metrics["T_avg"])
        self.episode_log["T_min"].append(metrics["T_min"])
        self.episode_log["T_gradient"].append(metrics["T_gradient"])
        self.episode_log["n_cells_above_safe"].append(
            int(np.sum(self.thermal_model.T > self.pack_config.safe_temp_c))
        )
        self.episode_log["actions"].append(u_zones.tolist())
        self.episode_log["reward"].append(reward)
        self.episode_log["q_gen_total"].append(float(np.sum(q_gen)))
        self.episode_log["q_gen_max_cell"].append(float(np.max(q_gen)))
        self.episode_log["q_cool_total"].append(metrics.get("q_cool_total", 0.0))
        self.episode_log["q_cool_per_zone"].append(metrics.get("q_cool_per_zone", []))
        self.episode_log["zone_max_temps"].append(self.thermal_model.get_zone_max_temps().tolist())
        for k in _REWARD_COMPONENT_KEYS:
            self.episode_log[k].append(self._last_reward_components.get(k, 0.0))
        if self._has_echem:
            self.episode_log["soc_mean"].append(self.heat_profile.soc_mean)
            self.episode_log["soc_min"].append(self.heat_profile.soc_min)

    def render(self) -> None:
        if self.render_mode != "human":
            return
        try:
            from scripts.render_pack_3d_pyvista import render_temperature_field
        except ImportError:
            print("PyVista not installed — run: pip install pyvista")
            return
        render_temperature_field(
            T=self.thermal_model.T,
            cell=self.cell_config,
            pack=self.pack_config,
            vmin=self.pack_config.ambient_temp_c,
            vmax=self.pack_config.safe_temp_c,
        )

    def get_episode_log(self) -> Dict[str, np.ndarray]:
        result = {}
        for k, v in self.episode_log.items():
            try:
                result[k] = np.asarray(v)
            except ValueError:
                result[k] = v
        return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = BatteryPackThermalEnv3D(
        pack_config=PackConfig(shape=(4, 4, 1)),
        heat_profile=nonuniform_step_3d_heat(),
        seed=7,
    )

    print(f"Observation space: {env.observation_space.shape}  ({env._obs_dim}D)")
    print(f"Action space:      {env.action_space.shape}  ({env.n_zones} zones)")

    obs, info = env.reset(seed=7)
    print(f"Initial obs: {obs}")

    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        # Zone-proportional controller: each zone gets proportional cooling to its max temp
        zone_max = np.array(info["zone_max_temps"])
        kp = 0.1
        action = np.clip(kp * (zone_max - env.pack_config.target_temp_c), 0.0, 1.0).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    log = env.get_episode_log()
    print(f"\nSmoke test — 3D pack env (multi-zone)")
    print(f"Final T_max:      {log['T_max'][-1]:.2f} °C")
    print(f"Final T_avg:      {log['T_avg'][-1]:.2f} °C")
    print(f"Final T_gradient: {log['T_gradient'][-1]:.3f} °C")
    print(f"Total reward:     {total_reward:.2f}")
    print(f"Final zone actions: {log['actions'][-1]}")
