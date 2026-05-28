"""
scripts/compare_pack_baselines_3d.py

Phase 2 baseline benchmark for the 3D cell-resolved battery pack thermal model.

Run from project root:
    python -m scripts.compare_pack_baselines_3d

Expected outputs:
    outputs/phase2_3d_baseline_summary.csv
    outputs/phase2_3d_baseline_temperatures.png
    outputs/phase2_3d_baseline_actions.png
    outputs/phase2_3d_baseline_gradient.png
    outputs/phase2_3d_baseline_heat_profiles.png
    outputs/phase2_3d_layer_heatmaps_<profile>.png  (one per profile, final step)

Controller groups (engineering progression):
    Group 1 — Global baselines:    No cooling, Constant, Bang-bang, Proportional, PI
    Group 2 — Multi-zone classical: Zone proportional, Zone PI, Zone hysteresis, Thermal balancing
    Group 3 — Multi-zone RL:        PPO, SAC  (added post-training)

All Group 1 controllers receive the 7-element normalized observation (pack-level).
All Group 2 controllers additionally use info["zone_max_temps"] for per-zone control.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.pack_config import PackConfig, CellConfig
from envs.battery_pack_thermal_env_3d import (
    BatteryPackThermalEnv3D,
    make_3d_profile,
    Pack3DHeatProfile,
)


# ---------------------------------------------------------------------------
# Observation decoder
# ---------------------------------------------------------------------------

def extract_3d_pack_state(
    obs: np.ndarray,
    pack_config: PackConfig,
    info: Optional[Dict] = None,
) -> Dict:
    """
    Decode pack state for global baseline controllers.

    When sensor simulation is active, info contains pre-measured values
    (T_max_meas, T_mean_meas, T_grad_meas) that include noise and delay.
    These are preferred over perfect obs-decoded values so global baselines
    experience the same observability as zone-wise controllers.

    Falls back to obs decoding for environments without sensor simulation.
    """
    if info is not None and "T_max_meas" in info:
        return {
            "T_max": float(info["T_max_meas"]),
            "T_avg": float(info.get("T_mean_meas", info["T_max_meas"])),
            "T_min": float(info.get("T_max_meas", info["T_max_meas"])),  # not tracked separately
            "T_gradient": float(info.get("T_grad_meas", 0.0)),
            "u_prev": 0.0,
        }
    # Legacy: decode from normalized obs (default behavior without sensor sim)
    normalizer = max(1e-6, pack_config.safe_temp_c - pack_config.target_temp_c)
    return {
        "T_max": float(obs[0]) * normalizer + pack_config.target_temp_c,
        "T_avg": float(obs[1]) * normalizer + pack_config.target_temp_c,
        "T_min": float(obs[2]) * normalizer + pack_config.target_temp_c,
        "T_gradient": float(obs[3]) * normalizer,
        "u_prev": float(obs[4]) if len(obs) > 4 else 0.0,
    }


# ---------------------------------------------------------------------------
# Controller protocol
# ---------------------------------------------------------------------------

class Pack3DController(Protocol):
    name: str
    controller_type: str

    def reset(self) -> None: ...
    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Group 1: Global baseline controllers
# ---------------------------------------------------------------------------

@dataclass
class NoCooling3D:
    name: str = "No cooling"
    controller_type: str = "Global baseline"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


@dataclass
class ConstantCooling3D:
    cooling_level: float = 0.5
    name: str = "Constant cooling"
    controller_type: str = "Global baseline"

    def __post_init__(self) -> None:
        self.cooling_level = float(np.clip(self.cooling_level, 0.0, 1.0))
        self.name = f"Constant cooling u={self.cooling_level:.2f}"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        return np.array([self.cooling_level], dtype=np.float32)


@dataclass
class BangBang3D:
    pack_config: PackConfig
    target_temp_c: float = 35.0
    deadband_c: float = 1.0
    name: str = "Bang-bang (global)"
    controller_type: str = "Global baseline"

    def __post_init__(self) -> None:
        self._is_high: bool = False

    def reset(self) -> None:
        self._is_high = False

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        state = extract_3d_pack_state(obs, self.pack_config, info)
        T_max = state["T_max"]

        if T_max >= self.target_temp_c + self.deadband_c:
            self._is_high = True
        elif T_max <= self.target_temp_c - self.deadband_c:
            self._is_high = False

        u = 1.0 if self._is_high else 0.0
        return np.array([u], dtype=np.float32)


@dataclass
class Proportional3D:
    pack_config: PackConfig
    target_temp_c: float = 35.0
    kp: float = 0.10
    bias: float = 0.15
    imbalance_gain: float = 0.04
    name: str = "Proportional (global)"
    controller_type: str = "Global baseline"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        state = extract_3d_pack_state(obs, self.pack_config, info)
        error = state["T_max"] - self.target_temp_c
        u = self.bias + self.kp * error + self.imbalance_gain * state["T_gradient"]
        u = float(np.clip(u, 0.0, 1.0))
        return np.array([u], dtype=np.float32)


@dataclass
class PI3D:
    pack_config: PackConfig
    target_temp_c: float = 35.0
    kp: float = 0.30
    ki: float = 0.001
    bias: float = 0.30
    imbalance_gain: float = 0.08
    integral_limit: float = 50.0
    dt: float = 1.0
    name: str = "PI (global)"
    controller_type: str = "Global baseline"

    def __post_init__(self) -> None:
        self.integral_error: float = 0.0

    def reset(self) -> None:
        self.integral_error = 0.0

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        state = extract_3d_pack_state(obs, self.pack_config, info)
        error = state["T_max"] - self.target_temp_c

        self.integral_error += error * self.dt
        self.integral_error = float(
            np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        )

        u = (
            self.bias
            + self.kp * error
            + self.ki * self.integral_error
            + self.imbalance_gain * state["T_gradient"]
        )
        u = float(np.clip(u, 0.0, 1.0))
        return np.array([u], dtype=np.float32)


# ---------------------------------------------------------------------------
# Group 2: Multi-zone classical controllers
#
# All zone controllers:
#   - Return a 4-element action array (one command per quadrant zone)
#   - Use info["zone_max_temps"] for per-zone temperature
#   - Fall back to global T_max broadcast if info is not available
# ---------------------------------------------------------------------------

def _zone_temps_from_info(
    obs: np.ndarray,
    info: Optional[Dict],
    pack_config: PackConfig,
    num_zones: int = 4,
) -> np.ndarray:
    """Return per-zone max temperatures [°C], shape (num_zones,)."""
    if info is not None and "zone_max_temps" in info:
        return np.array(info["zone_max_temps"], dtype=np.float64)
    # Fallback: broadcast pack-level T_max to all zones
    state = extract_3d_pack_state(obs, pack_config)
    return np.full(num_zones, state["T_max"], dtype=np.float64)


@dataclass
class ZoneProportional3D:
    """
    Independent proportional controller per quadrant zone.
    Same control law as Proportional3D but applied per-zone.
    """
    pack_config: PackConfig
    target_temp_c: float = 35.0
    kp: float = 0.10
    bias: float = 0.15
    num_zones: int = 4
    name: str = "Zone proportional"
    controller_type: str = "Multi-zone classical"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        zone_temps = _zone_temps_from_info(obs, info, self.pack_config, self.num_zones)
        errors = zone_temps - self.target_temp_c
        u = np.clip(self.bias + self.kp * errors, 0.0, 1.0)
        return u.astype(np.float32)


@dataclass
class ZonePI3D:
    """
    Independent PI controller per quadrant zone.
    Same gains as PI3D (global) — comparison isolates zone-wise vs global actuator.
    """
    pack_config: PackConfig
    target_temp_c: float = 35.0
    kp: float = 0.30
    ki: float = 0.001
    bias: float = 0.30
    integral_limit: float = 50.0
    dt: float = 1.0
    num_zones: int = 4
    name: str = "Zone PI"
    controller_type: str = "Multi-zone classical"

    def __post_init__(self) -> None:
        self.integral_errors: np.ndarray = np.zeros(self.num_zones, dtype=np.float64)

    def reset(self) -> None:
        self.integral_errors = np.zeros(self.num_zones, dtype=np.float64)

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        zone_temps = _zone_temps_from_info(obs, info, self.pack_config, self.num_zones)
        errors = zone_temps - self.target_temp_c

        self.integral_errors += errors * self.dt
        self.integral_errors = np.clip(
            self.integral_errors, -self.integral_limit, self.integral_limit
        )

        u = np.clip(
            self.bias + self.kp * errors + self.ki * self.integral_errors,
            0.0, 1.0,
        )
        return u.astype(np.float32)


@dataclass
class ZoneHysteresis3D:
    """
    Independent bang-bang controller with hysteresis per quadrant zone.
    Same control law as BangBang3D but applied per-zone.
    """
    pack_config: PackConfig
    target_temp_c: float = 35.0
    deadband_c: float = 1.0
    num_zones: int = 4
    name: str = "Zone bang-bang"
    controller_type: str = "Multi-zone classical"

    def __post_init__(self) -> None:
        self._is_high: np.ndarray = np.zeros(self.num_zones, dtype=bool)

    def reset(self) -> None:
        self._is_high = np.zeros(self.num_zones, dtype=bool)

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        zone_temps = _zone_temps_from_info(obs, info, self.pack_config, self.num_zones)

        turn_on  = zone_temps >= self.target_temp_c + self.deadband_c
        turn_off = zone_temps <= self.target_temp_c - self.deadband_c
        self._is_high = np.where(turn_on, True, np.where(turn_off, False, self._is_high))

        u = np.where(self._is_high, 1.0, 0.0)
        return u.astype(np.float32)


@dataclass
class ThermalBalancingZone3D:
    """
    Thermal-balancing zone controller.

    Computes a global base cooling command from T_max (same as Global PI),
    then allocates more cooling to hotter zones and less to cooler zones —
    keeping total cooling effort constant while minimizing inter-zone spread.

    This is the strongest classical multi-zone baseline:
      - Same energy as Global PI on uniform loads
      - Less energy than Global PI when hotspots are localized
      - Always maintains safety by anchoring to T_max like PI
    """
    pack_config: PackConfig
    target_temp_c: float = 35.0
    kp: float = 0.30
    ki: float = 0.001
    bias: float = 0.30
    integral_limit: float = 50.0
    balance_gain: float = 0.5
    dt: float = 1.0
    num_zones: int = 4
    name: str = "Thermal balancing zone"
    controller_type: str = "Multi-zone classical"

    def __post_init__(self) -> None:
        self.integral_error: float = 0.0

    def reset(self) -> None:
        self.integral_error = 0.0

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        state = extract_3d_pack_state(obs, self.pack_config, info)
        zone_temps = _zone_temps_from_info(obs, info, self.pack_config, self.num_zones)

        # Global base command from T_max PI (ensures safety)
        error_max = state["T_max"] - self.target_temp_c
        self.integral_error += error_max * self.dt
        self.integral_error = float(
            np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        )
        u_base = self.bias + self.kp * error_max + self.ki * self.integral_error

        # Per-zone deviation: hotter zones get more cooling, cooler zones less
        zone_errors = zone_temps - np.mean(zone_temps)   # deviation from mean
        u_zone = u_base + self.balance_gain * zone_errors / max(
            1.0, self.pack_config.safe_temp_c - self.target_temp_c
        )

        return np.clip(u_zone, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Controller registry
# ---------------------------------------------------------------------------

def build_3d_baseline_controllers(pack_config: PackConfig) -> List:
    """Return all Group 1 (global) and Group 2 (multi-zone) classical controllers."""
    return [
        # Group 1 — Global baselines
        NoCooling3D(),
        ConstantCooling3D(cooling_level=0.5),
        ConstantCooling3D(cooling_level=1.0),
        BangBang3D(pack_config=pack_config, target_temp_c=pack_config.target_temp_c),
        Proportional3D(
            pack_config=pack_config,
            target_temp_c=pack_config.target_temp_c,
            kp=0.10,
            bias=0.15,
            imbalance_gain=0.04,
        ),
        PI3D(
            pack_config=pack_config,
            target_temp_c=pack_config.target_temp_c,
            kp=0.30,
            ki=0.001,
            bias=0.30,
            imbalance_gain=0.08,
            integral_limit=50.0,
            name="PI tuned (global)",
        ),
        # Group 2 — Multi-zone classical
        ZoneProportional3D(
            pack_config=pack_config,
            target_temp_c=pack_config.target_temp_c,
            kp=0.10,
            bias=0.15,
        ),
        ZonePI3D(
            pack_config=pack_config,
            target_temp_c=pack_config.target_temp_c,
            kp=0.30,
            ki=0.001,
            bias=0.30,
            integral_limit=50.0,
        ),
        ZoneHysteresis3D(
            pack_config=pack_config,
            target_temp_c=pack_config.target_temp_c,
            deadband_c=1.0,
        ),
        ThermalBalancingZone3D(
            pack_config=pack_config,
            target_temp_c=pack_config.target_temp_c,
            kp=0.30,
            ki=0.001,
            bias=0.30,
            integral_limit=50.0,
            balance_gain=0.5,
        ),
    ]


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_controller_case(
    controller,
    profile_name: str,
    cell_config: CellConfig,
    pack_config: PackConfig,
    seed: int = 7,
    enable_sensor_simulation: bool = False,
    sensor_config=None,
    actuator_config=None,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    env = BatteryPackThermalEnv3D(
        cell_config=cell_config,
        pack_config=pack_config,
        heat_profile=make_3d_profile(profile_name),
        seed=seed,
        enable_sensor_simulation=enable_sensor_simulation,
        sensor_config=sensor_config,
        actuator_config=actuator_config,
    )

    controller.reset()
    obs, info = env.reset(seed=seed, options={"randomize": False})

    terminated = truncated = False
    total_reward = 0.0
    final_T3d = None

    while not (terminated or truncated):
        action = controller.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        final_T3d = info["temperatures_3d"]

    log = env.get_episode_log()

    # Derive a scalar action series from multi-zone actions for backward-compatible
    # metrics and plotting (mean across zones each step).
    if "actions" in log and "action" not in log:
        actions_2d = np.asarray(log["actions"])          # shape (T, n_zones)
        log["action"] = np.mean(actions_2d, axis=1)      # shape (T,) mean per step

    failed = bool(terminated and log["T_max"].max() >= pack_config.critical_temp_c)
    time_above_safe = float(np.sum(log["T_max"] > pack_config.safe_temp_c) * env.dt_s)

    # Safety status for two-stage ranking
    is_safe = (time_above_safe == 0.0) and (float(log["T_max"].max()) < pack_config.critical_temp_c)

    ctrl_type = getattr(controller, "controller_type", "Unknown")

    # Mean-action scalar for backward-compatible metrics (mean across zones per step, then mean over time)
    action_mean = float(log["action"].mean())
    action_sum  = float(log["action"].sum() * env.dt_s)

    # Zone targeting accuracy (how often highest-cooled zone == hottest measured zone)
    targeting_arr = np.asarray(log.get("targeting_correct", []))
    zone_targeting_accuracy = float(np.mean(targeting_arr)) if len(targeting_arr) > 0 else float("nan")

    # Sensor error (|T_max_true - T_max_meas|) — only meaningful when sensor sim active
    T_max_true_arr = np.asarray(log["T_max"])
    T_max_meas_arr = np.asarray(log.get("zone_max_temps_true", log["T_max"]))  # fallback = true
    # Use zone_max_temps (measured) if available, otherwise skip
    zone_meas_arr = np.asarray(log.get("zone_max_temps", []))
    if zone_meas_arr.ndim == 2 and len(zone_meas_arr) > 0:
        T_max_meas_per_step = zone_meas_arr.max(axis=1)
        mean_abs_sensor_error = float(np.mean(np.abs(T_max_true_arr - T_max_meas_per_step)))
    else:
        mean_abs_sensor_error = 0.0

    # Actuator saturation time (fraction of steps where any zone is at u=1.0)
    u_applied_arr = np.asarray(log.get("u_applied", []))
    if u_applied_arr.ndim == 2:
        saturated = np.any(u_applied_arr >= 0.99, axis=1)
        actuator_saturation_time = float(np.mean(saturated))
    else:
        actuator_saturation_time = float("nan")

    metrics: Dict = {
        "profile": profile_name,
        "controller": controller.name,
        "type": ctrl_type,
        # Temperature metrics
        "T_max_peak_C": float(log["T_max"].max()),
        "T_max_final_C": float(log["T_max"][-1]),
        "T_avg_peak_C": float(log["T_avg"].max()),
        "T_avg_final_C": float(log["T_avg"][-1]),
        "T_gradient_max_C": float(log["T_gradient"].max()),
        "T_gradient_mean_C": float(log["T_gradient"].mean()),
        # Safety
        "n_cells_above_safe_peak": int(log["n_cells_above_safe"].max()),
        "time_above_safe_s": time_above_safe,
        "is_safe": bool(is_safe),
        "failed": failed,
        # Cooling effort
        "mean_cooling_action": action_mean,
        "total_cooling_effort": action_sum,
        "cooling_energy_total": float(log.get("q_cool_total", np.array([0.0])).sum() * env.dt_s),
        "action_variation": float(np.abs(np.diff(log["action"])).sum()),
        "actuator_saturation_time": actuator_saturation_time,
        # Multi-zone targeting
        "zone_targeting_accuracy": zone_targeting_accuracy,
        # Sensor realism
        "mean_abs_sensor_error_C": mean_abs_sensor_error,
        # Reward
        "total_reward": float(total_reward),
    }

    # Mean reward component values for per-controller diagnosis
    from envs.battery_pack_thermal_env_3d import _REWARD_COMPONENT_KEYS
    for k in _REWARD_COMPONENT_KEYS:
        arr = log.get(k)
        if arr is not None and len(arr) > 0:
            metrics[f"mean_{k}"] = float(np.mean(arr))

    log["final_T3d"] = final_T3d
    return log, metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PROFILE_NAMES = ["UniformConstant", "NonuniformStep", "PulsedHotspot", "RandomNonuniform"]

# Line styles per controller type for visual separation
_TYPE_STYLE: Dict[str, Dict] = {
    "Global baseline":       {"linestyle": "-",  "linewidth": 1.3, "alpha": 0.85},
    "Multi-zone classical":  {"linestyle": "--", "linewidth": 1.6, "alpha": 0.90},
    "Multi-zone RL":         {"linestyle": "-",  "linewidth": 2.3, "alpha": 1.00},
    "Unknown":               {"linestyle": ":",  "linewidth": 1.0, "alpha": 0.70},
}


def _plot_metric_grid(
    all_logs: Dict[Tuple[str, str], Dict],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    hlines: Optional[List[Tuple[float, str, str]]] = None,
    controller_types: Optional[Dict[str, str]] = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, PROFILE_NAMES):
        for (pname, cname), log in all_logs.items():
            if pname != profile_name:
                continue
            ctype = (controller_types or {}).get(cname, "Unknown")
            style = _TYPE_STYLE.get(ctype, _TYPE_STYLE["Unknown"])
            ax.plot(
                log["time"], log[metric_key],
                label=cname,
                **style,
            )

        if hlines:
            for val, ls, label in hlines:
                ax.axhline(val, linestyle=ls, linewidth=1.0, color="gray",
                           label=label if profile_name == PROFILE_NAMES[0] else None)

        ax.set_title(profile_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7)
    plt.tight_layout(rect=[0, 0, 0.80, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_layer_heatmaps(
    log: Dict,
    profile_name: str,
    pack_config: PackConfig,
    output_path: Path,
) -> None:
    """Per-layer temperature heatmap at the final simulation step."""
    T3d = log.get("final_T3d")
    if T3d is None:
        return

    Nz = pack_config.shape[2]
    fig, axes = plt.subplots(1, Nz, figsize=(5 * Nz, 4))
    if Nz == 1:
        axes = [axes]

    vmin = pack_config.ambient_temp_c
    vmax = pack_config.safe_temp_c

    for k, ax in enumerate(axes):
        im = ax.imshow(
            T3d[:, :, k],
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="RdYlBu_r",
        )
        ax.set_title(f"Layer z={k}")
        ax.set_xlabel("y index")
        ax.set_ylabel("x index")
        fig.colorbar(im, ax=ax, label="Temperature (°C)")

    fig.suptitle(
        f"3D pack temperature field — {profile_name}\n(final step, Zone PI controller)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heat_profiles(all_logs: Dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.flatten()
    fig.suptitle("Phase 2 — 3D pack heat generation profiles", fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, PROFILE_NAMES):
        log = next(
            (l for (pn, _), l in all_logs.items() if pn == profile_name), None
        )
        if log is None:
            continue
        ax.plot(log["time"], log["q_gen_total"], linewidth=1.6, label="Total heat (W)")
        ax.plot(log["time"], log["q_gen_max_cell"], linewidth=1.0, linestyle="--", label="Max cell heat (W)")
        ax.set_title(profile_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heat generation (W)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.84, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_GROUP_ORDER = ["Global baseline", "Multi-zone classical", "Multi-zone RL"]


def print_rankings(df: pd.DataFrame, pack_config: Optional[PackConfig] = None) -> None:
    """
    Two-stage ranking:
      Stage 1 — Safety (pass/fail): time_above_safe == 0 AND T_max < critical
      Stage 2 — Among safe controllers: ranked by mean total reward (descending)

    Controllers are grouped by type in Stage 2.
    """
    agg = df.groupby(["controller", "type"]).agg(
        is_safe=("is_safe", "all"),
        total_reward=("total_reward", "mean"),
        T_max_peak=("T_max_peak_C", "mean"),
        time_above_safe=("time_above_safe_s", "mean"),
        T_gradient_mean=("T_gradient_mean_C", "mean"),
        mean_cooling=("mean_cooling_action", "mean"),
        zone_targeting=("zone_targeting_accuracy", "mean"),
    ).reset_index()

    safe_ctrl   = agg[agg["is_safe"]].sort_values("total_reward", ascending=False)
    unsafe_ctrl = agg[~agg["is_safe"]].sort_values("T_max_peak", ascending=True)

    W = 90
    print("\n" + "=" * W)
    print("  STAGE 1 — Safety check  (pass = 0s above safe AND T_max < critical)")
    print("=" * W)

    for _, row in agg.sort_values(["type", "controller"]).iterrows():
        status = "PASS" if row["is_safe"] else f"FAIL  ({row['time_above_safe']:.0f}s above safe)"
        print(f"  {'PASS' if row['is_safe'] else 'FAIL':4s}  [{row['type']:<22}]  {row['controller']}")

    has_targeting = "zone_targeting" in agg.columns and not agg["zone_targeting"].isna().all()

    print("\n" + "=" * W)
    print("  STAGE 2 — Ranking among SAFE controllers  (higher reward = better)")
    hdr = f"  {'Rank':<5} {'Type':<22} {'Controller':<30} {'Reward':>8} {'T_max':>8} {'Grad':>7} {'u_mean':>7}"
    if has_targeting:
        hdr += f" {'Target%':>8}"
    print(hdr)
    print("  " + "-" * (W - 2))

    rank = 1
    for group in _GROUP_ORDER:
        group_rows = safe_ctrl[safe_ctrl["type"] == group]
        if group_rows.empty:
            continue
        print(f"\n  --- {group} ---")
        for _, row in group_rows.iterrows():
            tag = " ← RL" if ("SAC" in row["controller"] or "PPO" in row["controller"]) else ""
            line = (
                f"  {rank:<5} {row['type']:<22} {row['controller']:<30} "
                f"{row['total_reward']:>8.2f} "
                f"{row['T_max_peak']:>7.2f}° {row['T_gradient_mean']:>6.2f}° "
                f"{row['mean_cooling']:>6.3f}"
            )
            if has_targeting:
                tgt = row.get("zone_targeting", float("nan"))
                line += f" {tgt*100:>7.1f}%" if not np.isnan(tgt) else f"{'N/A':>8}"
            print(line + tag)
            rank += 1

    if not unsafe_ctrl.empty:
        print("\n  UNSAFE controllers (excluded from ranking):")
        for _, row in unsafe_ctrl.iterrows():
            print(
                f"  ✗  [{row['type']:<22}]  {row['controller']:<30}  "
                f"T_max={row['T_max_peak']:.2f}°C  "
                f"above_safe={row['time_above_safe']:.0f}s"
            )

    # Reward component breakdown
    comp_cols = [c for c in df.columns if c.startswith("mean_reward_")]
    if comp_cols:
        print("\n" + "=" * W)
        print("  Reward component breakdown (mean per step, higher = better)")
        hdr = f"  {'Controller':<30} " + "  ".join(f"{c.replace('mean_reward_','')[:8]:>8}" for c in comp_cols)
        print(hdr)
        print("  " + "-" * (W - 2))
        all_names = list(safe_ctrl["controller"]) + list(unsafe_ctrl["controller"])
        for name in all_names:
            row_data = df[df["controller"] == name][comp_cols].mean()
            vals = "  ".join(f"{row_data[c]:>8.3f}" for c in comp_cols)
            print(f"  {name:<30} {vals}")

    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_config = CellConfig()
    pack_config = PackConfig(shape=(4, 3, 2))

    controllers = build_3d_baseline_controllers(pack_config)

    all_logs: Dict[Tuple[str, str], Dict] = {}
    summary_rows: List[Dict] = []
    controller_types: Dict[str, str] = {}

    for profile_name in PROFILE_NAMES:
        for controller in controllers:
            log, metrics = run_controller_case(
                controller=controller,
                profile_name=profile_name,
                cell_config=cell_config,
                pack_config=pack_config,
                seed=7,
            )
            all_logs[(profile_name, controller.name)] = log
            summary_rows.append(metrics)
            controller_types[controller.name] = getattr(controller, "controller_type", "Unknown")

            print(
                f"{profile_name:16s} | [{getattr(controller, 'controller_type', '?'):22s}] "
                f"{controller.name:28s} | "
                f"T_max={metrics['T_max_peak_C']:.1f}°C  "
                f"grad={metrics['T_gradient_max_C']:.2f}°C  "
                f"above_safe={metrics['time_above_safe_s']:.0f}s  "
                f"reward={metrics['total_reward']:.1f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "phase2_3d_baseline_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    # Max temperature
    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="T_max",
        title="Phase 2 (3D) — Max cell temperature",
        ylabel="T_max (°C)",
        output_path=output_dir / "phase2_3d_baseline_temperatures.png",
        hlines=[
            (pack_config.target_temp_c, ":", "Target"),
            (pack_config.safe_temp_c, "--", "Safe limit"),
        ],
        controller_types=controller_types,
    )

    # Cooling actions
    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="action",
        title="Phase 2 (3D) — Cooling commands",
        ylabel="Cooling command u",
        output_path=output_dir / "phase2_3d_baseline_actions.png",
        controller_types=controller_types,
    )

    # Temperature gradient
    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="T_gradient",
        title="Phase 2 (3D) — Temperature gradient (T_max - T_min)",
        ylabel="T_gradient (°C)",
        output_path=output_dir / "phase2_3d_baseline_gradient.png",
        controller_types=controller_types,
    )

    # Heat profiles
    plot_heat_profiles(all_logs=all_logs, output_path=output_dir / "phase2_3d_baseline_heat_profiles.png")

    # Layer heatmaps — use Zone PI for each profile (strongest classical zone controller)
    zone_pi_name = "Zone PI"
    for profile_name in PROFILE_NAMES:
        key = (profile_name, zone_pi_name)
        if key in all_logs:
            plot_layer_heatmaps(
                log=all_logs[key],
                profile_name=profile_name,
                pack_config=pack_config,
                output_path=output_dir / f"phase2_3d_layer_heatmaps_{profile_name.lower()}.png",
            )

    print_rankings(summary_df)

    print("\nSaved outputs:")
    print(f"  {csv_path}")
    print(f"  {output_dir / 'phase2_3d_baseline_temperatures.png'}")
    print(f"  {output_dir / 'phase2_3d_baseline_actions.png'}")
    print(f"  {output_dir / 'phase2_3d_baseline_gradient.png'}")
    print(f"  {output_dir / 'phase2_3d_baseline_heat_profiles.png'}")
    for p in PROFILE_NAMES:
        print(f"  {output_dir / f'phase2_3d_layer_heatmaps_{p.lower()}.png'}")


if __name__ == "__main__":
    main()
