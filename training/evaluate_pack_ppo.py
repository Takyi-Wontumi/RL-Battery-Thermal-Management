"""
training/evaluate_pack_ppo.py

Evaluate trained pack-level PPO against classical pack baseline controllers.

Run from project root:
    python -m training.evaluate_pack_ppo

Expected inputs:
    models/ppo_battery_pack_thermal/best_model.zip
    models/ppo_battery_pack_thermal/final_model.zip
    models/ppo_battery_pack_thermal/vec_normalize.pkl

Expected outputs:
    outputs/phase5_pack_ppo_vs_baselines_summary.csv
    outputs/phase5_pack_ppo_vs_baselines_temperatures.png
    outputs/phase5_pack_ppo_vs_baselines_actions.png
    outputs/phase5_pack_ppo_vs_baselines_imbalance.png
    outputs/phase5_pack_ppo_vs_baselines_heat_profiles.png

Main benchmark to beat:
    Pack max-temp PI tuned mean reward ≈ -395.19
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Protocol, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.battery_pack_thermal_env import (
    BatteryPackThermalConfig,
    BatteryPackThermalEnv,
    make_pack_profile,
    uniform_constant_pack_heat,
)
from scripts.compare_pack_baselines import (
    PackBangBangController,
    PackConstantCoolingController,
    PackNoCoolingController,
    PackPIController,
    PackProportionalController,
)


class PackControllerLike(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        ...


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

def make_eval_config(seed: int = 7) -> BatteryPackThermalConfig:
    return BatteryPackThermalConfig(
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
        max_temp_weight=1.0,
        mean_temp_weight=0.15,
        imbalance_weight=0.40,
        over_temp_weight=10.0,
        action_weight=0.04,
        action_smoothness_weight=0.08,
        hard_violation_penalty=150.0,
        seed=seed,
    )


# -----------------------------------------------------------------------------
# PPO adapter
# -----------------------------------------------------------------------------

class PackPPOController:
    """Adapter that makes a Stable-Baselines3 PPO model look like a pack controller."""

    def __init__(self, model: PPO, vec_normalize: VecNormalize, name: str) -> None:
        self.model = model
        self.vec_normalize = vec_normalize
        self.name = name

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        norm_obs = self.vec_normalize.normalize_obs(obs_batch)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        u = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        return np.array([u], dtype=np.float32)


def make_dummy_vec_env_for_vecnormalize(config: BatteryPackThermalConfig) -> DummyVecEnv:
    """Create dummy env so saved VecNormalize stats can be loaded."""

    def _init():
        env = BatteryPackThermalEnv(
            config=config,
            heat_profile=uniform_constant_pack_heat(),
            render_mode=None,
        )
        return Monitor(env)

    return DummyVecEnv([_init])


def load_pack_ppo_controller(
    model_path: Path,
    vecnorm_path: Path,
    config: BatteryPackThermalConfig,
    name: str,
) -> PackPPOController:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing PPO model: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"Missing VecNormalize stats: {vecnorm_path}")

    dummy_env = make_dummy_vec_env_for_vecnormalize(config)
    vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False

    model = PPO.load(str(model_path), env=None, device="auto")
    return PackPPOController(model=model, vec_normalize=vec_normalize, name=name)


# -----------------------------------------------------------------------------
# Controller construction
# -----------------------------------------------------------------------------

def build_controllers(config: BatteryPackThermalConfig) -> List[PackControllerLike]:
    controllers: List[PackControllerLike] = [
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

    model_dir = PROJECT_ROOT / "models" / "ppo_battery_pack_thermal"
    vecnorm_path = model_dir / "vec_normalize.pkl"
    best_model_path = model_dir / "best_model.zip"
    final_model_path = model_dir / "final_model.zip"

    if best_model_path.exists():
        controllers.append(
            load_pack_ppo_controller(
                model_path=best_model_path,
                vecnorm_path=vecnorm_path,
                config=config,
                name="Pack PPO best model",
            )
        )
    else:
        print(f"Warning: missing {best_model_path}. Skipping Pack PPO best model.")

    if final_model_path.exists():
        controllers.append(
            load_pack_ppo_controller(
                model_path=final_model_path,
                vecnorm_path=vecnorm_path,
                config=config,
                name="Pack PPO final model",
            )
        )
    else:
        print(f"Warning: missing {final_model_path}. Skipping Pack PPO final model.")

    return controllers


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def run_controller_case(
    controller: PackControllerLike,
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

def _plot_metric_grid(
    all_logs: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    profile_names: List[str],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    config: BatteryPackThermalConfig | None = None,
    ylim: Tuple[float, float] | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False)
    axes = axes.flatten()

    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, profile_names):
        for (logged_profile, controller_name), log in all_logs.items():
            if logged_profile != profile_name:
                continue

            linewidth = 2.3 if "PPO" in controller_name else 1.3

            if metric_key == "spread":
                y = log["max_temperature"] - log["min_temperature"]
            else:
                y = log[metric_key]

            ax.plot(log["time"], y, linewidth=linewidth, label=controller_name)

        if config is not None and metric_key == "max_temperature":
            ax.axhline(config.target_temp, linestyle=":", linewidth=1.0)
            ax.axhline(config.soft_max_temp, linestyle="--", linewidth=1.0)

        ax.set_title(profile_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.82, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heat_profiles(
    all_logs: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    profile_names: List[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.flatten()

    fig.suptitle("Phase 5 — Pack PPO benchmark heat-generation profiles", fontsize=14, fontweight="bold")

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

    print("\n=== Pack PPO vs tuned PI check ===")
    grouped = summary_df.groupby("controller", as_index=False).agg(
        mean_reward=("total_reward", "mean"),
        mean_effort=("total_cooling_effort", "mean"),
        total_time_above_safe=("time_above_safe_s", "sum"),
        worst_max_cell_temp=("max_cell_temperature_C", "max"),
        mean_spread=("mean_temperature_spread_C", "mean"),
    )

    ppo_rows = grouped[grouped["controller"].str.contains("PPO", case=False, regex=False)]
    pi_rows = grouped[grouped["controller"] == "Pack max-temp PI tuned"]

    if len(ppo_rows) == 0 or len(pi_rows) == 0:
        print("Could not compare PPO and tuned PI because one is missing.")
        return

    best_ppo = ppo_rows.sort_values("mean_reward", ascending=False).iloc[0]
    tuned_pi = pi_rows.iloc[0]

    print(f"Best PPO controller:      {best_ppo['controller']}")
    print(f"PPO mean reward:          {best_ppo['mean_reward']:.3f}")
    print(f"PI tuned mean reward:     {tuned_pi['mean_reward']:.3f}")
    print(f"Reward delta:             {best_ppo['mean_reward'] - tuned_pi['mean_reward']:.3f}  positive means PPO beat PI")
    print(f"PPO mean effort:          {best_ppo['mean_effort']:.3f}")
    print(f"PI tuned mean effort:     {tuned_pi['mean_effort']:.3f}")
    print(f"Effort delta:             {best_ppo['mean_effort'] - tuned_pi['mean_effort']:.3f}  negative means PPO used less cooling")
    print(f"PPO worst max cell temp:  {best_ppo['worst_max_cell_temp']:.3f} C")
    print(f"PI worst max cell temp:   {tuned_pi['worst_max_cell_temp']:.3f} C")
    print(f"PPO mean spread:          {best_ppo['mean_spread']:.3f} C")
    print(f"PI mean spread:           {tuned_pi['mean_spread']:.3f} C")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_eval_config(seed=7)
    profile_names = ["UniformConstant", "NonuniformStep", "PulsedHotspot", "RandomNonuniform"]
    controllers = build_controllers(config)

    if not any("PPO" in controller.name for controller in controllers):
        raise RuntimeError("No pack PPO model found. Train first with: python -m training.train_pack_ppo")

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

    csv_path = output_dir / "phase5_pack_ppo_vs_baselines_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    _plot_metric_grid(
        all_logs=all_logs,
        profile_names=profile_names,
        metric_key="max_temperature",
        title="Phase 5 — Pack PPO vs baselines: max cell temperature",
        ylabel="Max cell temperature (°C)",
        output_path=output_dir / "phase5_pack_ppo_vs_baselines_temperatures.png",
        config=config,
    )

    _plot_metric_grid(
        all_logs=all_logs,
        profile_names=profile_names,
        metric_key="action",
        title="Phase 5 — Pack PPO vs baselines: cooling action",
        ylabel="Cooling command u",
        output_path=output_dir / "phase5_pack_ppo_vs_baselines_actions.png",
        ylim=(-0.05, 1.05),
    )

    _plot_metric_grid(
        all_logs=all_logs,
        profile_names=profile_names,
        metric_key="spread",
        title="Phase 5 — Pack PPO vs baselines: cell temperature spread",
        ylabel="Max-min temperature spread (°C)",
        output_path=output_dir / "phase5_pack_ppo_vs_baselines_imbalance.png",
    )

    plot_heat_profiles(
        all_logs=all_logs,
        profile_names=profile_names,
        output_path=output_dir / "phase5_pack_ppo_vs_baselines_heat_profiles.png",
    )

    print_rankings(summary_df)

    print("\nSaved outputs:")
    print(f"  {csv_path}")
    print(f"  {output_dir / 'phase5_pack_ppo_vs_baselines_temperatures.png'}")
    print(f"  {output_dir / 'phase5_pack_ppo_vs_baselines_actions.png'}")
    print(f"  {output_dir / 'phase5_pack_ppo_vs_baselines_imbalance.png'}")
    print(f"  {output_dir / 'phase5_pack_ppo_vs_baselines_heat_profiles.png'}")


if __name__ == "__main__":
    main()
