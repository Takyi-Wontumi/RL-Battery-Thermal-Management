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
from configs.sensor_simulation import (
    SensorConfig,
    ActuatorConfig,
    SensorPacket,
    SensorSimulation,
    CoolingActuatorSimulation,
)


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
# Zone mapping: assigns each cell to a 2×2 spatial cooling zone
# ---------------------------------------------------------------------------

def build_zone_ids(shape: Tuple[int, int, int], num_zones: int = 4) -> np.ndarray:
    """
    Assign each cell to a cooling zone.

    Iteration order matches zone_ids[k*Nx*Ny + j*Nx + i] so that
    _build_zone_ids_3d() can map back to T[i, j, k] cleanly.

    num_zones=4 — 2×2 spatial split in the x-y plane:
        zone 0: x-left  (i < Nx/2), y-front (j < Ny/2)
        zone 1: x-right (i >= Nx/2), y-front (j < Ny/2)
        zone 2: x-left  (i < Nx/2), y-rear  (j >= Ny/2)
        zone 3: x-right (i >= Nx/2), y-rear  (j >= Ny/2)

    z-layers share the same x-y zone assignment (cooling is top-down).
    num_zones=1 collapses all cells into a single global zone.
    """
    Nx, Ny, Nz = shape
    zone_ids = []
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if num_zones == 1:
                    zone = 0
                elif num_zones == 4:
                    x_right = i >= Nx / 2
                    y_rear = j >= Ny / 2
                    if not x_right and not y_rear:
                        zone = 0
                    elif x_right and not y_rear:
                        zone = 1
                    elif not x_right and y_rear:
                        zone = 2
                    else:
                        zone = 3
                else:
                    raise ValueError(
                        f"Unsupported num_zones={num_zones}. Use 1 or 4."
                    )
                zone_ids.append(zone)
    return np.array(zone_ids, dtype=np.int32)


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
    "reward_warn",
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
    u_cmd: Optional[np.ndarray] = None,
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
        T       : (Nx, Ny, Nz) temperature array from the thermal model (°C)
        u_zones : applied cooling command per zone — used for energy penalty
        u_prev  : previous commanded value — used for smoothness penalty
        cfg     : PackConfig supplying target_low, target_high, safe, critical
        u_cmd   : current commanded value (before delay); if provided, the
                  smoothness term uses (u_cmd - u_prev) so delay doesn't
                  artificially inflate the smooth penalty.  If None, falls
                  back to u_zones (backward compatible, no-delay case).

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
    u = np.asarray(u_zones, dtype=np.float64)   # applied (for energy)
    up = np.asarray(u_prev,  dtype=np.float64)
    # Smoothness: penalise command chattering; when delay is active,
    # use the commanded value (not delayed applied) vs previous command.
    u_s = np.asarray(u_cmd, dtype=np.float64) if u_cmd is not None else u
    energy = float(np.mean(np.square(u)))
    smooth = float(np.mean(np.square(u_s - up)))

    # ── Warning zone: quadratic ramp 3°C before the safety limit ──
    # Accounts for actuator + sensor delay (≥5s): the policy must start cooling
    # well before 45°C. Firing at 42°C gives 3°C of early-warning signal.
    warn_start = safe - 3.0  # 42°C for default config
    warn_frac  = max(0.0, T_max - warn_start) / max(1e-6, safe - warn_start)
    r_warn = -5.0 * warn_frac ** 2  # 0 at 42°C → -5.0 at 45°C

    # ── Component values ──
    r_hot   = -1.00 * too_hot  ** 2
    r_cold  = -0.15 * too_cold ** 2
    r_safe  = -20.0 * safe_viol ** 2
    r_crit  = -200.0 * crit_viol ** 2
    r_sprd  = -0.50 * spread ** 2
    r_ener  = -0.05 * energy
    r_smth  = -0.02 * smooth
    r_hard  = -150.0 if T_max >= crit else 0.0

    total = r_hot + r_cold + r_warn + r_safe + r_crit + r_sprd + r_ener + r_smth + r_hard

    components: Dict[str, float] = {
        "reward_too_hot":      r_hot,
        "reward_too_cold":     r_cold,
        "reward_warn":         r_warn,
        "reward_safe":         r_safe,
        "reward_critical":     r_crit,
        "reward_spread":       r_sprd,
        "reward_energy":       r_ener,
        "reward_smooth":       r_smth,
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

    Observation — default (enable_sensor_simulation=False):
        dim = 6 + Nx
        [T_max_norm, T_avg_norm, T_min_norm,
         T_gradient_norm, T_variance_norm, T_center_norm,
         u_prev_zone_0, ..., u_prev_zone_{Nx-1}]

    Observation — sensor simulation (enable_sensor_simulation=True):
        dim = n_zones + 8 + n_series_groups + n_zones + n_zones
        [T_zone_meas_0..3,
         T_pack_max_est, T_pack_mean_est, T_gradient_est,
         pack_current_meas, pack_voltage_meas, soc_est,
         coolant_inlet_meas, coolant_outlet_meas,
         group_voltage_0..3,
         u_actual_0..3, u_prev_cmd_0..3]
        = 24D for a 4-zone, 4S pack.
        Use this mode for RL training with realistic BMS-style observations.

    Sensor simulation disabled by default to preserve backward compatibility.
    Enable with: BatteryPackThermalEnv3D(..., enable_sensor_simulation=True)
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
        sensor_config: Optional[SensorConfig] = None,
        actuator_config: Optional[ActuatorConfig] = None,
        enable_sensor_simulation: bool = False,
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

        # Number of independent cooling zones (2×2 quadrant split by default)
        self.n_zones: int = self.pack_config.num_cooling_zones

        # Detect electrochemical profile by duck-typing on soc_mean property
        self._has_echem: bool = hasattr(self.heat_profile, "soc_mean")

        # Sensor simulation mode (opt-in — backward compatible when False)
        self._use_sensor_sim: bool = enable_sensor_simulation

        # Observation dimension
        if self._use_sensor_sim:
            # Base sensor packet: n_zones + 8 pack summary + n_series_groups + n_zones u_actual + n_zones u_prev
            # + derivative features: dT_zone_dt[n_zones] + dT_gradient_dt[1]
            n_groups = self.pack_config.series_count
            self._obs_dim: int = self.n_zones + 8 + n_groups + self.n_zones + self.n_zones + self.n_zones + 1
            # = 4 + 8 + 4 + 4 + 4 + 4 + 1 = 29  (for n_zones=4, n_groups=4)
        else:
            # Legacy: 6 thermal + 2 SOC (if echem) + n_zones action history
            self._obs_dim = 6 + (2 if self._has_echem else 0) + self.n_zones

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

        # Zone mapping: flat (k-major) IDs and 3D mask matching T[i,j,k]
        self.num_cells: int = int(np.prod(self.pack_config.shape))
        self.zone_ids: np.ndarray = build_zone_ids(self.pack_config.shape, self.n_zones)
        self.zone_ids_3d: np.ndarray = self._build_zone_ids_3d()

        # Hotspot RNG: separate from main RNG so fixed-seed evals are reproducible
        if self.pack_config.hotspot_seed is not None:
            self._hotspot_rng = np.random.default_rng(self.pack_config.hotspot_seed)
        else:
            self._hotspot_rng = self.rng  # share main RNG during training

        # Hotspot state (populated in reset via _sample_hotspot_zone)
        self.hotspot_zone: Optional[int] = None
        self.hotspot_ids: np.ndarray = np.array([], dtype=np.int32)
        self.heat_multiplier_3d: np.ndarray = np.ones(self.pack_config.shape, dtype=np.float64)

        # Cooling actuator delay
        if self.pack_config.enable_cooling_delay:
            self.delay_steps: int = max(1, int(round(self.pack_config.cooling_delay_s / self.dt_s)))
        else:
            self.delay_steps = 0
        self.action_buffer: list = [
            np.zeros(self.n_zones, dtype=np.float32) for _ in range(self.delay_steps)
        ]

        self.time_s: float = 0.0
        self.u_prev_zones: np.ndarray = np.zeros(self.n_zones, dtype=np.float32)  # applied (for obs)
        self.u_prev_cmd: np.ndarray = np.zeros(self.n_zones, dtype=np.float32)    # commanded (for rate limit)
        self._last_u_cmd: np.ndarray = np.zeros(self.n_zones, dtype=np.float32)
        self._last_u_applied: np.ndarray = np.zeros(self.n_zones, dtype=np.float32)
        self.episode_log: Dict[str, list] = {}
        self._last_reward_components: Dict[str, float] = {k: 0.0 for k in _REWARD_COMPONENT_KEYS}

        # Electrical state (estimated from q_gen each step; used by sensor sim)
        self._soc: float = 1.0
        self._last_q_cool_total: float = 0.0
        self._last_sensor_packet: Optional[SensorPacket] = None

        # Previous-step normalized zone temps and gradient for derivative features
        self._prev_T_zone_norm: np.ndarray = np.zeros(self.n_zones, dtype=np.float32)
        self._prev_T_gradient_norm: float = 0.0

        # Sensor and actuator simulation (only active when enable_sensor_simulation=True)
        if self._use_sensor_sim:
            _sensor_cfg = sensor_config or SensorConfig(num_zones=self.n_zones)
            # Sync actuator config with PackConfig defaults when not explicitly provided
            _actuator_cfg = actuator_config or ActuatorConfig(
                num_zones=self.n_zones,
                cooling_delay_s=self.pack_config.cooling_delay_s,
                enable_rate_limit=self.pack_config.enable_cooling_rate_limit,
                max_cooling_rate_per_s=self.pack_config.max_cooling_rate_per_s,
            )
            # zone_ids_3d.flatten() is C-order — matches T.reshape(-1) indexing
            self.sensor_sim = SensorSimulation(
                cfg=_sensor_cfg,
                dt_s=self.dt_s,
                zone_ids=self.zone_ids_3d.flatten(),
                seed=seed,
            )
            self.actuator_sim = CoolingActuatorSimulation(
                cfg=_actuator_cfg,
                dt_s=self.dt_s,
            )
        else:
            self.sensor_sim = None   # type: ignore[assignment]
            self.actuator_sim = None  # type: ignore[assignment]

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
        self.u_prev_cmd = np.zeros(self.n_zones, dtype=np.float32)
        self._last_u_cmd = np.zeros(self.n_zones, dtype=np.float32)
        self._last_u_applied = np.zeros(self.n_zones, dtype=np.float32)
        self.action_buffer = [
            np.zeros(self.n_zones, dtype=np.float32) for _ in range(self.delay_steps)
        ]
        self._last_reward_components = {k: 0.0 for k in _REWARD_COMPONENT_KEYS}
        self._soc = 1.0
        self._last_q_cool_total = 0.0
        self._last_sensor_packet = None

        if self._use_sensor_sim:
            T_flat = self.thermal_model.T.flatten()
            self.sensor_sim.reset(T_flat)
            self.actuator_sim.reset()
            V_nom_cell = self.cell_config.nominal_voltage_v
            self._last_sensor_packet = self.sensor_sim.measure(
                T_cells_true_c=T_flat,
                pack_current_true_a=0.0,
                pack_voltage_true_v=self.pack_config.series_count * V_nom_cell,
                group_voltage_true_v=np.full(self.pack_config.series_count, V_nom_cell, dtype=np.float32),
                soc_true=self._soc,
                coolant_inlet_true_c=self.pack_config.ambient_temp_c,
                coolant_outlet_true_c=self.pack_config.ambient_temp_c,
                u_actual=np.zeros(self.n_zones, dtype=np.float32),
                u_prev_command=np.zeros(self.n_zones, dtype=np.float32),
            )
            # Seed derivative history from initial packet so first-step dT = 0
            _temp_scale = max(1e-6, self.pack_config.safe_temp_c - self.pack_config.target_temp_c)
            _temp_ref = self.pack_config.target_temp_c
            self._prev_T_zone_norm = (
                (self._last_sensor_packet.T_zone_meas_c - _temp_ref) / _temp_scale
            ).astype(np.float32)
            self._prev_T_gradient_norm = float(
                self._last_sensor_packet.T_gradient_est_c / _temp_scale
            )
        else:
            self._prev_T_zone_norm = np.zeros(self.n_zones, dtype=np.float32)
            self._prev_T_gradient_norm = 0.0

        self._sample_hotspot_zone()   # sample before first episode
        self._reset_log()

        if self._has_echem:
            self.heat_profile.reset()

        obs = self._build_obs()
        return obs, self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Capture previous command BEFORE processing (used for smoothness reward)
        u_prev_for_reward = self.u_prev_cmd.copy()

        if self._use_sensor_sim:
            # Actuator simulation handles rate-limit, delay, effectiveness, faults
            u_cmd_f32, u_applied_f32 = self.actuator_sim.step(action)
            u_cmd = u_cmd_f32.astype(np.float64)
            u_applied = u_applied_f32.astype(np.float64)
            self.u_prev_cmd = self.actuator_sim.u_prev_command.copy()
        else:
            u_cmd, u_applied = self._process_cooling_action(action)

        # Optionally resample hotspot at fixed intervals (non-persistent mode)
        if (
            not self.pack_config.hotspot_persistent
            and self.time_s > 0
            and self.pack_config.hotspot_change_interval_s > 0
            and int(self.time_s) % int(self.pack_config.hotspot_change_interval_s) == 0
        ):
            self._sample_hotspot_zone()

        q_gen = _call_heat_profile(
            self.heat_profile,
            self.time_s,
            self.rng,
            self.pack_config.shape,
            self.thermal_model.T,
        )
        # Global scale factor (stress-test knob) then spatial hotspot overlay
        if self.pack_config.q_gen_multiplier != 1.0:
            q_gen = q_gen * self.pack_config.q_gen_multiplier
        q_gen = q_gen * self.heat_multiplier_3d

        # Pass per-cell 3D commands so each quadrant zone cools its own cells
        u_cell_3d = self._zone_action_to_cell_action(u_applied)
        T, metrics = self.thermal_model.step(u_zones=u_cell_3d, dt=self.dt_s, q_gen=q_gen)

        self.time_s += self.dt_s

        reward, self._last_reward_components = compute_reward_components(
            T=T,
            u_zones=u_applied,          # energy penalty: actual cooling applied
            u_prev=u_prev_for_reward,   # smoothness reference: prev command
            cfg=self.pack_config,
            u_cmd=u_cmd,                # smoothness: cmd vs prev cmd (not applied)
        )

        terminated = bool(metrics["critical"] or np.min(T) <= -10.0)
        truncated = bool(self.time_s >= self.total_time_s)

        self._last_u_cmd = u_cmd
        self._last_u_applied = u_applied
        self._last_q_cool_total = metrics.get("q_cool_total", 0.0)

        if self._use_sensor_sim:
            self._last_sensor_packet = self._build_sensor_packet(q_gen)

        self._log_step(u_cmd, u_applied, reward, metrics, q_gen)
        self.u_prev_zones = u_applied.astype(np.float32)  # obs: what was applied

        obs = self._build_obs()
        return obs, reward, terminated, truncated, self._get_info(metrics)

    # ------------------------------------------------------------------
    # Observation assembly
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """Assemble the full observation vector."""
        if self._use_sensor_sim and self._last_sensor_packet is not None:
            return self._normalize_sensor_obs(self._last_sensor_packet)

        # Legacy: normalized thermal features + action history
        thermal_obs = self.thermal_model.get_thermal_obs()  # 6 elements

        if self._has_echem:
            soc_obs = np.array([
                self.heat_profile.soc_mean,
                self.heat_profile.soc_min,
            ], dtype=np.float32)
            return np.concatenate([thermal_obs, soc_obs, self.u_prev_zones])

        return np.concatenate([thermal_obs, self.u_prev_zones])

    def _normalize_sensor_obs(self, packet: SensorPacket) -> np.ndarray:
        """
        Normalize a SensorPacket into a policy-friendly observation vector.

        All features are scaled to approximately [-1, 2]:
          Temperatures → (T - target_temp) / (safe_temp - target_temp)
                          0 at target, 1 at safe limit, <0 if too cold
          Gradient      → gradient / (safe_temp - target_temp)
          Current       → I_pack / I_pack_max  (0–1 under normal operation)
          Voltage       → (V - V_nominal) / V_range  (≈ -0.5 to +0.5)
          SOC           → unchanged [0, 1]
          Coolant temps → (T - ambient) / (safe - ambient)
          Group voltage → (V_cell - V_nom_cell) / V_cell_range
          Cooling cmds  → unchanged [0, 1]
        """
        cfg = self.pack_config
        cell = self.cell_config

        temp_scale = max(1e-6, cfg.safe_temp_c - cfg.target_temp_c)  # 10°C
        temp_ref = cfg.target_temp_c                                    # 35°C

        T_zone_norm = (packet.T_zone_meas_c - temp_ref) / temp_scale

        pack_max_norm  = (packet.T_pack_max_est_c  - temp_ref) / temp_scale
        pack_mean_norm = (packet.T_pack_mean_est_c - temp_ref) / temp_scale
        pack_grad_norm = packet.T_gradient_est_c / temp_scale

        I_max_pack = cell.max_continuous_discharge_a * cfg.parallel_count
        V_nom_pack = cfg.series_count * cell.nominal_voltage_v
        V_range_pack = cfg.series_count * (cell.max_voltage_v - cell.cutoff_voltage_v) / 2.0
        I_norm = packet.pack_current_meas_a / max(1.0, I_max_pack)
        V_norm = (packet.pack_voltage_meas_v - V_nom_pack) / max(1.0, V_range_pack)
        SOC   = float(np.clip(packet.soc_est, 0.0, 1.0))

        coolant_scale = max(1.0, cfg.safe_temp_c - cfg.ambient_temp_c)
        T_in_norm  = (packet.coolant_inlet_temp_meas_c  - cfg.ambient_temp_c) / coolant_scale
        T_out_norm = (packet.coolant_outlet_temp_meas_c - cfg.ambient_temp_c) / coolant_scale

        V_nom_cell   = cell.nominal_voltage_v
        V_range_cell = (cell.max_voltage_v - cell.cutoff_voltage_v) / 2.0
        V_group_norm = (packet.group_voltage_meas_v - V_nom_cell) / max(1e-6, V_range_cell)

        # Derivative features: change in normalized values since previous step.
        # At step 0 (reset), prev values equal current so derivatives are zero.
        dT_zone_norm     = T_zone_norm.astype(np.float32) - self._prev_T_zone_norm
        dT_gradient_norm = float(pack_grad_norm) - self._prev_T_gradient_norm

        # Update history for next call
        self._prev_T_zone_norm    = T_zone_norm.astype(np.float32)
        self._prev_T_gradient_norm = float(pack_grad_norm)

        return np.concatenate([
            T_zone_norm.astype(np.float32),
            np.array([pack_max_norm, pack_mean_norm, pack_grad_norm,
                      I_norm, V_norm, SOC, T_in_norm, T_out_norm], dtype=np.float32),
            V_group_norm.astype(np.float32),
            np.clip(packet.u_actual_feedback, 0.0, 1.0).astype(np.float32),
            np.clip(packet.u_prev_command, 0.0, 1.0).astype(np.float32),
            dT_zone_norm,
            np.array([dT_gradient_norm], dtype=np.float32),
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Curriculum support
    # ------------------------------------------------------------------

    def update_sensor_config(
        self,
        sensor_config: "SensorConfig",
        actuator_config: Optional["ActuatorConfig"] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Replace the active sensor/actuator simulation objects in-place.

        Called by CurriculumCallback at training stage boundaries to
        progressively increase environment difficulty without recreating
        the entire env (which would reset VecNormalize statistics).
        Only has effect when enable_sensor_simulation=True.
        """
        if not self._use_sensor_sim:
            return
        _seed = seed if seed is not None else int(self.rng.integers(0, 2**31))
        T_flat = self.thermal_model.T.flatten()
        self.sensor_sim = SensorSimulation(
            cfg=sensor_config,
            dt_s=self.dt_s,
            zone_ids=self.zone_ids_3d.flatten(),
            seed=_seed,
        )
        self.sensor_sim.reset(T_flat)
        if actuator_config is not None:
            self.actuator_sim = CoolingActuatorSimulation(
                cfg=actuator_config, dt_s=self.dt_s
            )
            self.actuator_sim.reset()

    # ------------------------------------------------------------------
    # Hotspot sampling
    # ------------------------------------------------------------------

    def _sample_hotspot_zone(self) -> None:
        """
        Randomly select a cooling zone and hotspot cells within it.

        Zone is drawn uniformly from [0, n_zones). Cells are drawn without
        replacement from the selected zone's cell pool. A per-cell multiplier
        in [hotspot_multiplier_min, hotspot_multiplier_max] is applied to q_gen
        so one region heats faster than the rest.

        When enable_random_hotspot=False, heat_multiplier_3d stays all-ones
        (no spatial bias — used for ablation/baseline comparison).

        Hotspot location is logged in info as hotspot_zone and hotspot_ids.
        """
        if not self.pack_config.enable_random_hotspot:
            self.hotspot_zone = None
            self.hotspot_ids = np.array([], dtype=np.int32)
            self.heat_multiplier_3d = np.ones(self.pack_config.shape, dtype=np.float64)
            return

        rng = self._hotspot_rng
        Nx, Ny, Nz = self.pack_config.shape

        # Pick a random zone to stress
        hotspot_zone = int(rng.integers(0, self.n_zones))
        cells_in_zone = np.where(self.zone_ids == hotspot_zone)[0]

        n_hot = min(self.pack_config.num_hotspot_cells, len(cells_in_zone))
        hotspot_ids = rng.choice(cells_in_zone, size=n_hot, replace=False).astype(np.int32)

        # Build 3D multiplier array; default = 1.0, elevated at hotspot cells
        mult_3d = np.ones((Nx, Ny, Nz), dtype=np.float64)
        for flat_idx in hotspot_ids:
            # Convert flat k-major index → (i, j, k) for T indexing
            k = int(flat_idx) // (Nx * Ny)
            j = (int(flat_idx) % (Nx * Ny)) // Nx
            i = int(flat_idx) % Nx
            mult_3d[i, j, k] = rng.uniform(
                self.pack_config.hotspot_multiplier_min,
                self.pack_config.hotspot_multiplier_max,
            )

        self.hotspot_zone = hotspot_zone
        self.hotspot_ids = hotspot_ids
        self.heat_multiplier_3d = mult_3d

    # ------------------------------------------------------------------
    # Action processing: broadcast → rate-limit → delay
    # ------------------------------------------------------------------

    def _process_cooling_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a controller action through the full actuator chain.

        Steps:
            1. Broadcast scalar / shape-(1,) to all zones.
            2. Clip to [0, 1].
            3. Rate-limit: constrain per-step change to max_cooling_rate_per_s * dt.
            4. Delay: push into buffer; pop the oldest entry as u_applied.

        Returns:
            u_cmd    : rate-limited command (what the controller requested this step)
            u_applied: delayed command (what the physics actually receives this step)

        Baseline controllers returning scalar or shape-(1,) are broadcast here,
        so they experience the same delay and rate-limit as RL agents — fair comparison.
        """
        u_cmd = np.asarray(action, dtype=np.float64).reshape(-1)
        if u_cmd.size == 1:
            u_cmd = np.full(self.n_zones, float(u_cmd[0]))
        elif u_cmd.size != self.n_zones:
            raise ValueError(
                f"Expected action of size 1 or {self.n_zones}, got {u_cmd.size}"
            )
        u_cmd = np.clip(u_cmd, 0.0, 1.0)

        if self.pack_config.enable_cooling_rate_limit:
            max_delta = self.pack_config.max_cooling_rate_per_s * self.dt_s
            u_cmd = np.clip(
                u_cmd,
                self.u_prev_cmd - max_delta,
                self.u_prev_cmd + max_delta,
            )
            u_cmd = np.clip(u_cmd, 0.0, 1.0)

        if self.delay_steps > 0:
            self.action_buffer.append(u_cmd.copy())
            u_applied = self.action_buffer.pop(0)
        else:
            u_applied = u_cmd.copy()

        self.u_prev_cmd = u_cmd.astype(np.float32)
        return u_cmd.astype(np.float64), u_applied.astype(np.float64)

    def _zone_action_to_cell_action(self, u_applied_zones: np.ndarray) -> np.ndarray:
        """
        Convert per-zone commands, shape (n_zones,), to a per-cell 3D array
        matching T.shape = (Nx, Ny, Nz) using the precomputed zone_ids_3d mask.
        """
        return u_applied_zones[self.zone_ids_3d].astype(np.float64)

    def _build_zone_ids_3d(self) -> np.ndarray:
        """
        Reshape flat zone_ids (k-major order) to 3D array (i, j, k) so that
        zone_ids_3d[i, j, k] == zone of cell T[i, j, k].
        """
        Nx, Ny, Nz = self.pack_config.shape
        z3d = np.zeros((Nx, Ny, Nz), dtype=np.int32)
        c = 0
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    z3d[i, j, k] = self.zone_ids[c]
                    c += 1
        return z3d

    def _get_zone_max_temps(self) -> np.ndarray:
        """Return max temperature per cooling zone, shape (n_zones,)."""
        result = np.zeros(self.n_zones, dtype=np.float64)
        for z in range(self.n_zones):
            mask = self.zone_ids_3d == z
            result[z] = float(self.thermal_model.T[mask].max()) if mask.any() else self.pack_config.ambient_temp_c
        return result

    def _estimate_electrical_state(self, q_gen: np.ndarray) -> Dict:
        """
        Estimate pack-level electrical state from the current q_gen array.

        Used by _build_sensor_packet() to supply approximate current/voltage/SOC
        when no electrochemical model is attached. The estimates are physically
        consistent (I²R heat balance) but not high-fidelity.
        """
        q_total = float(np.sum(q_gen))
        R_dc = self.cell_config.dc_ir_ohm
        # Back-calculate per-cell current from I²R: I_cell = sqrt(q_cell / R_dc)
        I_cell = float(np.sqrt(max(0.0, q_total / max(1, self.num_cells) / max(1e-9, R_dc))))
        I_pack = I_cell * self.pack_config.parallel_count

        V_nom_cell = self.cell_config.nominal_voltage_v
        V_pack = self.pack_config.series_count * V_nom_cell
        V_group = np.full(self.pack_config.series_count, V_nom_cell, dtype=np.float32)

        # Coulomb counting (rough; resets to 1.0 on each episode reset)
        cap_as = self.cell_config.capacity_ah * 3600.0
        self._soc = float(np.clip(self._soc - I_cell * self.dt_s / max(1.0, cap_as), 0.0, 1.0))

        return {
            "pack_current_a": I_pack,
            "pack_voltage_v": V_pack,
            "group_voltage_v": V_group,
            "soc": self._soc,
        }

    def _build_sensor_packet(self, q_gen: np.ndarray) -> SensorPacket:
        """
        Generate a BMS-style sensor measurement from the current thermal/electrical state.
        Called once per step when enable_sensor_simulation=True.
        """
        if self._has_echem:
            # Prefer electrochemical model's electrical state when available
            elec = {
                "pack_current_a": getattr(self.heat_profile, "pack_current_a", 0.0),
                "pack_voltage_v": self.pack_config.series_count * self.cell_config.nominal_voltage_v,
                "group_voltage_v": np.full(
                    self.pack_config.series_count,
                    self.cell_config.nominal_voltage_v,
                    dtype=np.float32,
                ),
                "soc": getattr(self.heat_profile, "soc_mean", self._soc),
            }
        else:
            elec = self._estimate_electrical_state(q_gen)

        T_flat = self.thermal_model.T.flatten()  # C-order — matches zone_ids_3d.flatten()

        coolant_inlet_c = self.pack_config.ambient_temp_c
        # Rough coolant outlet: inlet + heat removed / nominal exchange capacity
        coolant_outlet_c = coolant_inlet_c + self._last_q_cool_total / 5000.0

        return self.sensor_sim.measure(
            T_cells_true_c=T_flat,
            pack_current_true_a=elec["pack_current_a"],
            pack_voltage_true_v=elec["pack_voltage_v"],
            group_voltage_true_v=elec["group_voltage_v"],
            soc_true=elec["soc"],
            coolant_inlet_true_c=coolant_inlet_c,
            coolant_outlet_true_c=coolant_outlet_c,
            u_actual=self._last_u_applied,
            u_prev_command=self.u_prev_cmd,
        )

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

        # True zone max temps (always computed for logging)
        zone_max_true = self._get_zone_max_temps()

        # zone_max_temps in info: measured (noisy) when sensor sim active, true otherwise.
        # Zone controllers use info["zone_max_temps"] — this makes them use measured values.
        if self._use_sensor_sim and self._last_sensor_packet is not None:
            pkt = self._last_sensor_packet
            zone_max_for_info = pkt.T_zone_meas_c
        else:
            zone_max_for_info = zone_max_true

        delay_steps = self.actuator_sim.delay_steps if self._use_sensor_sim else self.delay_steps

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
            "u_cmd_zones": self._last_u_cmd.tolist(),
            "u_applied_zones": self._last_u_applied.tolist(),
            "cooling_delay_steps": delay_steps,
            "temperatures_3d": self.thermal_model.T.copy(),
            "zone_max_temps": zone_max_for_info.tolist(),
            "zone_max_temps_true": zone_max_true.tolist(),
        }

        # Measured/estimated values for classical baseline controllers and diagnostics
        if self._use_sensor_sim and self._last_sensor_packet is not None:
            pkt = self._last_sensor_packet
            info["T_max_meas"] = pkt.T_pack_max_est_c
            info["T_mean_meas"] = pkt.T_pack_mean_est_c
            info["T_grad_meas"] = pkt.T_gradient_est_c
            info["soc_est"] = pkt.soc_est
            info["coolant_inlet_meas"] = pkt.coolant_inlet_temp_meas_c
            info["coolant_outlet_meas"] = pkt.coolant_outlet_temp_meas_c
            # Sensor health — exposed so dropout/fault is never silently hidden
            info["sensor_fault_flags"] = pkt.sensor_fault_flags
            info["sensor_dropout_active"] = bool(pkt.sensor_fault_flags.get("temperature_dropout", False))
            # Actuator fault diagnostic
            if self.actuator_sim is not None:
                info["actuator_fault_active"] = bool(self.actuator_sim.cfg.enable_actuator_fault)
                info["u_cmd_zones"] = self._last_u_cmd.tolist()   # override with actuator sim values
                info["u_actual_zones"] = self.actuator_sim.u_actual.tolist()
            else:
                info["actuator_fault_active"] = False

        # Per-zone applied cooling
        for z in range(self.n_zones):
            info[f"u_zone_{z}"] = float(self._last_u_applied[z])
        # Hotspot metadata
        info["hotspot_zone"] = self.hotspot_zone
        info["hotspot_ids"] = self.hotspot_ids.tolist()
        # Reward components
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
            "actions": [],          # alias for u_applied (backward compat)
            "u_cmd": [],            # commanded (before delay), shape (n_zones,) per step
            "u_applied": [],        # actually applied (after delay), shape (n_zones,) per step
            "reward": [],
            "q_gen_total": [],
            "q_gen_max_cell": [],
            "q_cool_total": [],
            "q_cool_per_zone": [],
            "zone_max_temps": [],        # per-zone measured (or true if no sensor sim)
            "zone_max_temps_true": [],   # per-zone true (always)
            **{f"u_zone_{z}": [] for z in range(self.n_zones)},
            "hotspot_zone": [],
            "hotspot_cell_ids": [],      # flat cell IDs of active hotspot
            "q_hotspot_boost": [],
            # Sensor diagnostics (populated when sensor sim is active, else 0/False)
            "sensor_dropout_active": [],
            "actuator_fault_active": [],
            # Multi-zone targeting: is the highest-cooled zone the hottest measured zone?
            "targeting_correct": [],   # bool per step
            "hot_zone_meas": [],       # argmax of measured zone temps
            "max_cool_zone": [],       # argmax of u_applied
            # Reward components — logged every step for diagnosis (R6)
            **{k: [] for k in _REWARD_COMPONENT_KEYS},
        }
        if self._has_echem:
            self.episode_log["soc_mean"] = []
            self.episode_log["soc_min"] = []

    def _log_step(
        self,
        u_cmd: np.ndarray,
        u_applied: np.ndarray,
        reward: float,
        metrics: Dict,
        q_gen: np.ndarray,
    ) -> None:
        zone_max_true = self._get_zone_max_temps()

        # Measured zone temps (same as true when sensor sim is off)
        if self._use_sensor_sim and self._last_sensor_packet is not None:
            zone_max_meas = self._last_sensor_packet.T_zone_meas_c
            sensor_dropout = bool(self._last_sensor_packet.sensor_fault_flags.get("temperature_dropout", False))
        else:
            zone_max_meas = zone_max_true
            sensor_dropout = False

        actuator_fault = bool(
            self._use_sensor_sim
            and self.actuator_sim is not None
            and self.actuator_sim.cfg.enable_actuator_fault
        )

        self.episode_log["time"].append(self.time_s)
        self.episode_log["T_max"].append(metrics["T_max"])
        self.episode_log["T_avg"].append(metrics["T_avg"])
        self.episode_log["T_min"].append(metrics["T_min"])
        self.episode_log["T_gradient"].append(metrics["T_gradient"])
        self.episode_log["n_cells_above_safe"].append(
            int(np.sum(self.thermal_model.T > self.pack_config.safe_temp_c))
        )
        self.episode_log["actions"].append(u_applied.tolist())
        self.episode_log["u_cmd"].append(u_cmd.tolist())
        self.episode_log["u_applied"].append(u_applied.tolist())
        self.episode_log["reward"].append(reward)
        self.episode_log["q_gen_total"].append(float(np.sum(q_gen)))
        self.episode_log["q_gen_max_cell"].append(float(np.max(q_gen)))
        self.episode_log["q_cool_total"].append(metrics.get("q_cool_total", 0.0))
        self.episode_log["q_cool_per_zone"].append(metrics.get("q_cool_per_zone", []))
        self.episode_log["zone_max_temps"].append(zone_max_meas.tolist())
        self.episode_log["zone_max_temps_true"].append(zone_max_true.tolist())
        for z in range(self.n_zones):
            self.episode_log[f"u_zone_{z}"].append(float(u_applied[z]))
        self.episode_log["hotspot_zone"].append(self.hotspot_zone)
        self.episode_log["hotspot_cell_ids"].append(self.hotspot_ids.tolist())
        boost = float(np.sum(q_gen * (self.heat_multiplier_3d - 1.0)))
        self.episode_log["q_hotspot_boost"].append(boost)
        self.episode_log["sensor_dropout_active"].append(sensor_dropout)
        self.episode_log["actuator_fault_active"].append(actuator_fault)

        # Zone targeting: is the highest-cooled zone the hottest measured zone?
        hot_zone = int(np.argmax(zone_max_meas))
        max_cool_zone = int(np.argmax(u_applied)) if np.any(u_applied > 0) else -1
        self.episode_log["hot_zone_meas"].append(hot_zone)
        self.episode_log["max_cool_zone"].append(max_cool_zone)
        self.episode_log["targeting_correct"].append(int(hot_zone == max_cool_zone))

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
        # Zone-proportional controller: each quadrant gets cooling proportional
        # to its local max temperature (exploits multi-zone control)
        zone_max = np.array(info["zone_max_temps"])
        kp = 0.1
        action = np.clip(kp * (zone_max - env.pack_config.target_temp_c), 0.0, 1.0).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    log = env.get_episode_log()
    print(f"\nSmoke test — 3D pack env (quadrant multi-zone + delay)")
    print(f"Final T_max:        {log['T_max'][-1]:.2f} °C")
    print(f"Final T_avg:        {log['T_avg'][-1]:.2f} °C")
    print(f"Final T_gradient:   {log['T_gradient'][-1]:.3f} °C")
    print(f"Total reward:       {total_reward:.2f}")
    print(f"Delay steps:        {env.delay_steps}")
    print(f"Final u_cmd:        {log['u_cmd'][-1]}")
    print(f"Final u_applied:    {log['u_applied'][-1]}")
