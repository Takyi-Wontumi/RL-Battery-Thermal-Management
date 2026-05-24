"""
scripts/compare_pack_baselines.py

Phase 4 baseline benchmark for the multi-node battery pack thermal environment.

Run from project root:
    python -m scripts.compare_pack_baselines

Expected outputs:
    outputs/phase4_pack_baseline_summary.csv
    outputs/phase4_pack_baseline_temperatures.png
    outputs/phase4_pack_baseline_actions.png
    outputs/phase4_pack_baseline_imbalance.png
    outputs/phase4_pack_baseline_heat_profiles.png

This benchmark tests classical controllers on a multi-node pack model.
The controller sees multiple cell/module temperatures and commands one shared cooling input.

Key pack-level metrics:
    - max cell temperature
    - mean pack temperature
    - cell-to-cell temperature spread
    - temperature standard deviation
    - time above safe limit
    - cooling effort
    - total reward
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envs.battery_pack_thermal_env import (
    BatteryPackThermalConfig,
    BatteryPackThermalEnv,
    make_pack_profile,
)


class PackController(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        ...


# -----------------------------------------------------------------------------
# Observation helper
# -----------------------------------------------------------------------------

def extract_pack_state(obs: np.ndarray, config: BatteryPackThermalConfig) -> Dict[str, np.ndarray | float]:
    """
    Convert normalized pack observation back into useful physical values.

    Observation layout from BatteryPackThermalEnv:
        [normalized T_cells, scaled Q_cells, ambient_obs, time_fraction, previous_action]
    """
    n = config.n_cells
    normalizer = max(1e-6, config.soft_max_temp - config.target_temp)

    temp_norm = obs[:n]
    heat_scaled = obs[n : 2 * n]

    temperatures = temp_norm * normalizer + config.target_temp
    heat_generation = heat_scaled * 250.0

    ambient_obs = float(obs[2 * n])
    ambient_temp = ambient_obs * 20.0 + config.target_temp

    time_fraction = float(obs[2 * n + 1])
    previous_action = float(obs[2 * n + 2])

    return {
        "temperatures": temperatures.astype(np.float32),
        "heat_generation": heat_generation.astype(np.float32),
        "ambient_temperature": ambient_temp,
        "time_fraction": time_fraction,
        "previous_action": previous_action,
        "max_temperature": float(np.max(temperatures)),
        "mean_temperature": float(np.mean(temperatures)),
        "temperature_std": float(np.std(temperatures)),
        "temperature_spread": float(np.max(temperatures) - np.min(temperatures)),
    }


# -----------------------------------------------------------------------------
# Pack baseline controllers
# -----------------------------------------------------------------------------

@dataclass
class PackNoCoolingController:
    name: str = "Pack no cooling"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


@dataclass
class PackConstantCoolingController:
    cooling_level: float = 0.5
    name: str = "Pack constant cooling"

    def __post_init__(self) -> None:
        self.cooling_level = float(np.clip(self.cooling_level, 0.0, 1.0))
        self.name = f"Pack constant cooling u={self.cooling_level:.2f}"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.array([self.cooling_level], dtype=np.float32)


@dataclass
class PackBangBangController:
    config: BatteryPackThermalConfig
    target_temp: float = 35.0
    deadband: float = 1.0
    low_action: float = 0.0
    high_action: float = 1.0
    name: str = "Pack max-temp bang-bang"

    def __post_init__(self) -> None:
        self._is_high = False
        self.low_action = float(np.clip(self.low_action, 0.0, 1.0))
        self.high_action = float(np.clip(self.high_action, 0.0, 1.0))

    def reset(self) -> None:
        self._is_high = False

    def act(self, obs: np.ndarray) -> np.ndarray:
        state = extract_pack_state(obs, self.config)
        max_temp = float(state["max_temperature"])

        upper = self.target_temp + self.deadband
        lower = self.target_temp - self.deadband

        if max_temp >= upper:
            self._is_high = True
        elif max_temp <= lower:
            self._is_high = False

        u = self.high_action if self._is_high else self.low_action
        return np.array([u], dtype=np.float32)


@dataclass
class PackProportionalController:
    config: BatteryPackThermalConfig
    target_temp: float = 35.0
    kp: float = 0.10
    bias: float = 0.15
    imbalance_gain: float = 0.04
    name: str = "Pack max-temp proportional"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        state = extract_pack_state(obs, self.config)
        max_temp = float(state["max_temperature"])
        spread = float(state["temperature_spread"])

        error = max_temp - self.target_temp
        u = self.bias + self.kp * error + self.imbalance_gain * spread
        u = float(np.clip(u, 0.0, 1.0))
        return np.array([u], dtype=np.float32)


@dataclass
class PackPIController:
    config: BatteryPackThermalConfig
    target_temp: float = 35.0
    kp: float = 0.16
    ki: float = 0.004
    bias: float = 0.20
    imbalance_gain: float = 0.05
    integral_limit: float = 100.0
    name: str = "Pack max-temp PI"

    def __post_init__(self) -> None:
        self.integral_error = 0.0

    def reset(self) -> None:
        self.integral_error = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        state = extract_pack_state(obs, self.config)
        max_temp = float(state["max_temperature"])
        spread = float(state["temperature_spread"])

        error = max_temp - self.target_temp
        self.integral_error += error * self.config.dt
        self.integral_error = float(np.clip(self.integral_error, -self.integral_limit, self.integral_limit))

        u = (
            self.bias
            + self.kp * error
            + self.ki * self.integral_error
            + self.imbalance_gain * spread
        )
        u = float(np.clip(u, 0.0, 1.0))
        return np.array([u], dtype=np.float32)


def build_pack_baseline_controllers(config: BatteryPackThermalConfig) -> List[PackController]:
    return [
        PackNoCoolingController(),
        PackConstantCoolingController(cooling_level=0.5),
        PackConstantCoolingController(cooling_level=1.0),
        PackBangBangController(config=config, target_temp=config.target_temp, deadband=1.0),
        PackProportionalController(
            config=config,
            target_temp=config.target_temp,
            kp=0.10,
            bias=0.15,
            imbalance_gain=0.04,
        ),
        PackPIController(
            config=config,
            target_temp=config.target_temp,
            kp=0.30,
            ki=0.001,
            bias=0.30,
            imbalance_gain=0.08,
            integral_limit=50.0,
            name="Pack max-temp PI tuned",
        ),
    ]


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

def run_controller_case(
    controller: PackController,
    profile_name: str,
    config: BatteryPackThermalConfig,
    seed: int = 7,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float | str | bool]]:
    env = BatteryPackThermalEnv(
        config=config,
        heat_profile=make_pack_profile(profile_name),
        render_mode=None,
    )

    controller.reset()
    obs, info = env.reset(seed=seed, options={"randomize": False})

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action = controller.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    log = env.get_episode_log()

    max_temp = log["max_temperature"]
    mean_temp = log["mean_temperature"]
    min_temp = log["min_temperature"]
    temp_std = log["temperature_std"]
    spread = max_temp - min_temp
    action = log["action"]
    time = log["time"]

    failed = bool(terminated and np.max(max_temp) >= config.hard_max_temp)

    metrics: Dict[str, float | str | bool] = {
        "profile": profile_name,
        "controller": controller.name,
        "max_cell_temperature_C": float(np.max(max_temp)),
        "final_max_cell_temperature_C": float(max_temp[-1]),
        "mean_pack_temperature_C": float(np.mean(mean_temp)),
        "final_mean_pack_temperature_C": float(mean_temp[-1]),
        "max_temperature_spread_C": float(np.max(spread)),
        "mean_temperature_spread_C": float(np.mean(spread)),
        "max_temperature_std_C": float(np.max(temp_std)),
        "mean_temperature_std_C": float(np.mean(temp_std)),
        "time_above_safe_s": float(np.sum(max_temp > config.soft_max_temp) * config.dt),
        "total_cooling_effort": float(np.sum(action) * config.dt),
        "mean_cooling_action": float(np.mean(action)),
        "action_variation": float(np.sum(np.abs(np.diff(action)))) if len(action) > 1 else 0.0,
        "total_reward": float(total_reward),
        "failed": failed,
        "episode_time_s": float(time[-1]) if len(time) > 0 else 0.0,
    }

    return log, metrics


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_pack_temperatures(
    all_logs: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    config: BatteryPackThermalConfig,
    output_path: Path,
) -> None:
    profile_names = ["UniformConstant", "NonuniformStep", "PulsedHotspot", "RandomNonuniform"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    fig.suptitle("Phase 4 — Pack baseline max-cell temperature comparison", fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, profile_names):
        for (logged_profile, controller_name), log in all_logs.items():
            if logged_profile != profile_name:
                continue
            ax.plot(log["time"], log["max_temperature"], linewidth=1.4, label=controller_name)

        ax.axhline(config.target_temp, linestyle=":", linewidth=1.0, label="Target" if profile_name == profile_names[0] else None)
        ax.axhline(config.soft_max_temp, linestyle="--", linewidth=1.0, label="Safe limit" if profile_name == profile_names[0] else None)
        ax.set_title(profile_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Max cell temperature (°C)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.82, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pack_actions(
    all_logs: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    profile_names = ["UniformConstant", "NonuniformStep", "PulsedHotspot", "RandomNonuniform"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    fig.suptitle("Phase 4 — Pack baseline cooling actions", fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, profile_names):
        for (logged_profile, controller_name), log in all_logs.items():
            if logged_profile != profile_name:
                continue
            ax.plot(log["time"], log["action"], linewidth=1.4, label=controller_name)

        ax.set_title(profile_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cooling command u")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.82, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pack_imbalance(
    all_logs: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    profile_names = ["UniformConstant", "NonuniformStep", "PulsedHotspot", "RandomNonuniform"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    fig.suptitle("Phase 4 — Pack temperature imbalance comparison", fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, profile_names):
        for (logged_profile, controller_name), log in all_logs.items():
            if logged_profile != profile_name:
                continue
            spread = log["max_temperature"] - log["min_temperature"]
            ax.plot(log["time"], spread, linewidth=1.4, label=controller_name)

        ax.set_title(profile_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Max-min temperature spread (°C)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.82, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pack_heat_profiles(
    all_logs: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    profile_names = ["UniformConstant", "NonuniformStep", "PulsedHotspot", "RandomNonuniform"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.flatten()

    fig.suptitle("Phase 4 — Pack benchmark heat-generation profiles", fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, profile_names):
        selected_log = None
        for (logged_profile, _controller_name), log in all_logs.items():
            if logged_profile == profile_name:
                selected_log = log
                break

        if selected_log is None:
            continue

        heat = selected_log["heat_generation"]
        time = selected_log["time"]
        total_heat = selected_log["total_heat_generation"]

        ax.plot(time, total_heat, linewidth=1.6, label="Total pack heat")
        ax.plot(time, np.max(heat, axis=1), linewidth=1.0, linestyle="--", label="Max cell heat")
        ax.plot(time, np.mean(heat, axis=1), linewidth=1.0, linestyle=":", label="Mean cell heat")

        ax.set_title(profile_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heat generation (W)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.84, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def print_rankings(summary_df: pd.DataFrame) -> None:
    print("\n=== Overall ranking by mean total reward: higher is better ===")
    reward_rank = (
        summary_df.groupby("controller", as_index=False)["total_reward"]
        .mean()
        .sort_values("total_reward", ascending=False)
    )
    print(reward_rank.to_string(index=False))

    print("\n=== Safety ranking by total time above safe limit: lower is better ===")
    safety_rank = (
        summary_df.groupby("controller", as_index=False)["time_above_safe_s"]
        .sum()
        .sort_values("time_above_safe_s", ascending=True)
    )
    print(safety_rank.to_string(index=False))

    print("\n=== Pack imbalance ranking by mean temperature spread: lower is better ===")
    imbalance_rank = (
        summary_df.groupby("controller", as_index=False)["mean_temperature_spread_C"]
        .mean()
        .sort_values("mean_temperature_spread_C", ascending=True)
    )
    print(imbalance_rank.to_string(index=False))

    print("\n=== Cooling effort ranking: lower is better only if safe ===")
    effort_rank = (
        summary_df.groupby("controller", as_index=False)["total_cooling_effort"]
        .mean()
        .sort_values("total_cooling_effort", ascending=True)
    )
    print(effort_rank.to_string(index=False))


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BatteryPackThermalConfig(
        n_cells=8,
        total_time=1800.0,
        dt=1.0,
        initial_temp=25.0,
        ambient_temp=25.0,
        thermal_capacitance=8_000.0,
        surface_area_per_cell=0.12,
        h_min=5.0,
        h_max=95.0,
        cooling_nonlinearity=1.0,
        direct_cooling_max_per_cell=0.0,
        conduction_coupling=8.0,
        target_temp=35.0,
        soft_max_temp=45.0,
        hard_max_temp=60.0,
        seed=7,
    )

    profile_names = ["UniformConstant", "NonuniformStep", "PulsedHotspot", "RandomNonuniform"]
    controllers = build_pack_baseline_controllers(config)

    all_logs: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    summary_rows: List[Dict[str, float | str | bool]] = []

    for profile_name in profile_names:
        for controller in controllers:
            log, metrics = run_controller_case(
                controller=controller,
                profile_name=profile_name,
                config=config,
                seed=7,
            )

            all_logs[(profile_name, controller.name)] = log
            summary_rows.append(metrics)

            print(
                f"Finished {profile_name:16s} | {controller.name:30s} | "
                f"max cell T={metrics['max_cell_temperature_C']:.2f} C | "
                f"spread={metrics['max_temperature_spread_C']:.2f} C | "
                f"effort={metrics['total_cooling_effort']:.1f} | "
                f"reward={metrics['total_reward']:.2f}"
            )

    summary_df = pd.DataFrame(summary_rows)

    first_cols = ["profile", "controller"]
    other_cols = [col for col in summary_df.columns if col not in first_cols]
    summary_df = summary_df[first_cols + other_cols]

    csv_path = output_dir / "phase4_pack_baseline_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    plot_pack_temperatures(
        all_logs=all_logs,
        config=config,
        output_path=output_dir / "phase4_pack_baseline_temperatures.png",
    )
    plot_pack_actions(
        all_logs=all_logs,
        output_path=output_dir / "phase4_pack_baseline_actions.png",
    )
    plot_pack_imbalance(
        all_logs=all_logs,
        output_path=output_dir / "phase4_pack_baseline_imbalance.png",
    )
    plot_pack_heat_profiles(
        all_logs=all_logs,
        output_path=output_dir / "phase4_pack_baseline_heat_profiles.png",
    )

    print_rankings(summary_df)

    print("\nSaved outputs:")
    print(f"  {csv_path}")
    print(f"  {output_dir / 'phase4_pack_baseline_temperatures.png'}")
    print(f"  {output_dir / 'phase4_pack_baseline_actions.png'}")
    print(f"  {output_dir / 'phase4_pack_baseline_imbalance.png'}")
    print(f"  {output_dir / 'phase4_pack_baseline_heat_profiles.png'}")


if __name__ == "__main__":
    main()
