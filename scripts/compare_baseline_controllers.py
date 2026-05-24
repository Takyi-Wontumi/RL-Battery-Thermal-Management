"""
scripts/compare_baseline_controllers.py

Phase 2 baseline benchmark:
Compare classical cooling controllers on the battery thermal environment.

Run from project root:
    python -m scripts.compare_baseline_controllers

Expected outputs:
    outputs/phase2_baseline_temperature.png
    outputs/phase2_baseline_actions.png
    outputs/phase2_baseline_summary.csv

This script is intentionally strict: RL must beat these baselines later.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from controllers.baseline_controllers import ThermalController, build_default_controllers
from envs.battery_thermal_env import BatteryThermalConfig, BatteryThermalEnv


HeatProfile = Callable[[float, np.random.Generator], float]


def constant_heat_profile(q_gen: float = 700.0) -> HeatProfile:
    """Constant heat generation profile."""

    def profile(t: float, rng: np.random.Generator) -> float:
        return float(q_gen)

    return profile


def step_heat_profile(
    q_low: float = 400.0,
    q_high: float = 1_100.0,
    step_time: float = 500.0,
) -> HeatProfile:
    """Step increase in heat generation."""

    def profile(t: float, rng: np.random.Generator) -> float:
        return float(q_low if t < step_time else q_high)

    return profile


def pulsed_heat_profile(
    q_low: float = 300.0,
    q_high: float = 1_200.0,
    period: float = 160.0,
    duty_cycle: float = 0.40,
) -> HeatProfile:
    """Repeated high-load pulse profile."""

    def profile(t: float, rng: np.random.Generator) -> float:
        phase = (t % period) / period
        return float(q_high if phase < duty_cycle else q_low)

    return profile


def random_heat_profile(
    q_mean: float = 650.0,
    q_std: float = 18.0,
    smoothing: float = 0.88,
) -> HeatProfile:
    """Smooth random load profile."""
    state = {"q": q_mean}

    def profile(t: float, rng: np.random.Generator) -> float:
        disturbance = rng.normal(0.0, q_std)
        state["q"] = smoothing * state["q"] + (1.0 - smoothing) * q_mean + disturbance
        return float(np.clip(state["q"], 450.0, 900.0))

    return profile


def build_heat_profiles() -> Dict[str, HeatProfile]:
    """Return fresh heat profile functions for one benchmark batch."""
    return {
        "Constant": constant_heat_profile(q_gen=700.0),
        "Step": step_heat_profile(q_low=400.0, q_high=1_100.0, step_time=500.0),
        "Pulsed": pulsed_heat_profile(q_low=300.0, q_high=1_200.0, period=160.0, duty_cycle=0.40),
        "Random": random_heat_profile(q_mean=650.0, q_std=18.0, smoothing=0.88),
    }


def make_fresh_profile(profile_name: str) -> HeatProfile:
    """
    Recreate heat profiles before every simulation.

    This matters because random profiles can be stateful. Reusing one closure across
    controllers would quietly poison the comparison.
    """
    profiles = build_heat_profiles()
    if profile_name not in profiles:
        raise KeyError(f"Unknown heat profile: {profile_name}")
    return profiles[profile_name]


def run_controller_case(
    controller: ThermalController,
    heat_profile: HeatProfile,
    config: BatteryThermalConfig,
    seed: int = 7,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float | str | bool]]:
    """Run one controller on one heat profile and compute performance metrics."""
    env = BatteryThermalEnv(
        config=config,
        heat_profile=heat_profile,
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

    temperature = log["temperature"]
    action = log["action"]
    time = log["time"]

    safe_limit = config.soft_max_temp
    target_temp = config.target_temp

    dt = config.dt
    time_above_safe = float(np.sum(temperature > safe_limit) * dt)
    mean_abs_temp_error = float(np.mean(np.abs(temperature - target_temp)))
    rms_temp_error = float(np.sqrt(np.mean((temperature - target_temp) ** 2)))
    max_temperature = float(np.max(temperature))
    final_temperature = float(temperature[-1])
    total_cooling_effort = float(np.sum(action) * dt)
    mean_cooling_action = float(np.mean(action))
    action_variation = float(np.sum(np.abs(np.diff(action)))) if len(action) > 1 else 0.0
    failed = bool(terminated and max_temperature >= config.hard_max_temp)

    metrics: Dict[str, float | str | bool] = {
        "controller": controller.name,
        "max_temperature_C": max_temperature,
        "final_temperature_C": final_temperature,
        "mean_abs_temp_error_C": mean_abs_temp_error,
        "rms_temp_error_C": rms_temp_error,
        "time_above_safe_s": time_above_safe,
        "total_cooling_effort": total_cooling_effort,
        "mean_cooling_action": mean_cooling_action,
        "action_variation": action_variation,
        "total_reward": float(total_reward),
        "failed": failed,
        "episode_time_s": float(time[-1]) if len(time) > 0 else 0.0,
    }

    return log, metrics


def plot_temperature_results(
    all_logs: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    config: BatteryThermalConfig,
    output_path: Path,
) -> None:
    """Plot battery temperature response for all controllers and profiles."""
    profile_names = ["Constant", "Step", "Pulsed", "Random"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    fig.suptitle(
        "Phase 2 — Baseline controller temperature comparison",
        fontsize=14,
        fontweight="bold",
    )

    for ax, profile_name in zip(axes, profile_names):
        for (logged_profile, controller_name), log in all_logs.items():
            if logged_profile != profile_name:
                continue

            ax.plot(
                log["time"],
                log["temperature"],
                linewidth=1.4,
                label=controller_name,
            )

        ax.axhline(config.target_temp, linestyle=":", linewidth=1.0, label="Target" if profile_name == "Constant" else None)
        ax.axhline(config.soft_max_temp, linestyle="--", linewidth=1.0, label="Safe limit" if profile_name == "Constant" else None)
        ax.set_title(f"{profile_name} heat profile")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Battery temperature (°C)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.84, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_action_results(
    all_logs: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot cooling actions for all controllers and profiles."""
    profile_names = ["Constant", "Step", "Pulsed", "Random"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    fig.suptitle(
        "Phase 2 — Baseline controller cooling action comparison",
        fontsize=14,
        fontweight="bold",
    )

    for ax, profile_name in zip(axes, profile_names):
        for (logged_profile, controller_name), log in all_logs.items():
            if logged_profile != profile_name:
                continue

            ax.plot(
                log["time"],
                log["action"],
                linewidth=1.4,
                label=controller_name,
            )

        ax.set_title(f"{profile_name} heat profile")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cooling command u")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.84, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heat_profiles(
    all_logs: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot one heat-generation trace per profile."""
    profile_names = ["Constant", "Step", "Pulsed", "Random"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), sharex=True)
    axes = axes.flatten()

    fig.suptitle("Phase 2 — Heat-generation profiles used in baseline benchmark", fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, profile_names):
        # Use first available controller trace because heat profile is recreated with same seed.
        selected_log = None
        for (logged_profile, _controller_name), log in all_logs.items():
            if logged_profile == profile_name:
                selected_log = log
                break

        if selected_log is None:
            continue

        ax.plot(selected_log["time"], selected_log["heat_generation"], linewidth=1.5)
        ax.set_title(f"{profile_name} heat profile")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Q_gen (W)")
        ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_rankings(summary_df: pd.DataFrame) -> None:
    """Print compact benchmark rankings."""
    print("\n=== Overall ranking by total reward: higher is better ===")
    reward_ranking = (
        summary_df.groupby("controller", as_index=False)["total_reward"]
        .mean()
        .sort_values("total_reward", ascending=False)
    )
    print(reward_ranking.to_string(index=False))

    print("\n=== Thermal safety ranking by time above safe limit: lower is better ===")
    safety_ranking = (
        summary_df.groupby("controller", as_index=False)["time_above_safe_s"]
        .mean()
        .sort_values("time_above_safe_s", ascending=True)
    )
    print(safety_ranking.to_string(index=False))

    print("\n=== Energy ranking by cooling effort: lower is better, but only meaningful if safe ===")
    effort_ranking = (
        summary_df.groupby("controller", as_index=False)["total_cooling_effort"]
        .mean()
        .sort_values("total_cooling_effort", ascending=True)
    )
    print(effort_ranking.to_string(index=False))


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BatteryThermalConfig(
        total_time=1800.0,
        dt=1.0,
        initial_temp=25.0,
        ambient_temp=25.0,
        thermal_capacitance=40_000.0,
        surface_area=1.0,
        h_min=5.0,
        h_max=95.0,
        direct_cooling_max=0.0,
        target_temp=35.0,
        soft_max_temp=45.0,
        hard_max_temp=60.0,
        seed=7,
    )

    profile_names = ["Constant", "Step", "Pulsed", "Random"]
    controllers = build_default_controllers(target_temp=config.target_temp, dt=config.dt)

    all_logs: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    summary_rows: List[Dict[str, float | str | bool]] = []

    for profile_name in profile_names:
        for controller in controllers:
            heat_profile = make_fresh_profile(profile_name)

            log, metrics = run_controller_case(
                controller=controller,
                heat_profile=heat_profile,
                config=config,
                seed=7,
            )

            metrics["profile"] = profile_name
            summary_rows.append(metrics)
            all_logs[(profile_name, controller.name)] = log

            print(
                f"Finished {profile_name:8s} | {controller.name:28s} | "
                f"max T={metrics['max_temperature_C']:.2f} C | "
                f"reward={metrics['total_reward']:.2f}"
            )

    summary_df = pd.DataFrame(summary_rows)

    # Put profile/controller first for readability.
    first_cols = ["profile", "controller"]
    other_cols = [col for col in summary_df.columns if col not in first_cols]
    summary_df = summary_df[first_cols + other_cols]

    csv_path = output_dir / "phase2_baseline_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    plot_temperature_results(
        all_logs=all_logs,
        config=config,
        output_path=output_dir / "phase2_baseline_temperature.png",
    )
    plot_action_results(
        all_logs=all_logs,
        output_path=output_dir / "phase2_baseline_actions.png",
    )
    plot_heat_profiles(
        all_logs=all_logs,
        output_path=output_dir / "phase2_baseline_heat_profiles.png",
    )

    print_rankings(summary_df)

    print("\nSaved outputs:")
    print(f"  {output_dir / 'phase2_baseline_temperature.png'}")
    print(f"  {output_dir / 'phase2_baseline_actions.png'}")
    print(f"  {output_dir / 'phase2_baseline_heat_profiles.png'}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
