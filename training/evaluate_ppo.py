"""
training/evaluate_ppo.py

Evaluate trained PPO against classical baseline controllers on the exact Phase 2
benchmark heat profiles.

Run from project root:
    python -m training.evaluate_ppo

Expected inputs:
    models/ppo_battery_thermal/best_model.zip
    models/ppo_battery_thermal/final_model.zip
    models/ppo_battery_thermal/vec_normalize.pkl

Expected outputs:
    outputs/phase3_rl_vs_baselines_summary.csv
    outputs/phase3_rl_vs_baselines_temperature.png
    outputs/phase3_rl_vs_baselines_actions.png
    outputs/phase3_rl_vs_baselines_heat_profiles.png

This script answers the only question that matters:
    Did PPO beat the tuned PI controller on the same benchmark?
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from controllers.baseline_controllers import (
    BangBangController,
    ConstantCoolingController,
    NoCoolingController,
    PIController,
    ProportionalController,
    ThermalController,
)
from envs.battery_thermal_env import BatteryThermalConfig, BatteryThermalEnv


HeatProfile = Callable[[float, np.random.Generator], float]


# -----------------------------------------------------------------------------
# Benchmark heat profiles: must match Phase 2 exactly
# -----------------------------------------------------------------------------

def constant_heat_profile(q_gen: float = 700.0) -> HeatProfile:
    def profile(t: float, rng: np.random.Generator) -> float:
        return float(q_gen)

    return profile


def step_heat_profile(
    q_low: float = 400.0,
    q_high: float = 1_100.0,
    step_time: float = 500.0,
) -> HeatProfile:
    def profile(t: float, rng: np.random.Generator) -> float:
        return float(q_low if t < step_time else q_high)

    return profile


def pulsed_heat_profile(
    q_low: float = 300.0,
    q_high: float = 1_200.0,
    period: float = 160.0,
    duty_cycle: float = 0.40,
) -> HeatProfile:
    def profile(t: float, rng: np.random.Generator) -> float:
        phase = (t % period) / period
        return float(q_high if phase < duty_cycle else q_low)

    return profile


def random_heat_profile(
    q_mean: float = 650.0,
    q_std: float = 18.0,
    smoothing: float = 0.88,
) -> HeatProfile:
    state = {"q": q_mean}

    def profile(t: float, rng: np.random.Generator) -> float:
        disturbance = rng.normal(0.0, q_std)
        state["q"] = smoothing * state["q"] + (1.0 - smoothing) * q_mean + disturbance
        return float(np.clip(state["q"], 450.0, 900.0))

    return profile


def build_heat_profiles() -> Dict[str, HeatProfile]:
    """Return fresh benchmark heat profile functions."""
    return {
        "Constant": constant_heat_profile(),
        "Step": step_heat_profile(),
        "Pulsed": pulsed_heat_profile(),
        "Random": random_heat_profile(),
    }


def make_fresh_profile(profile_name: str) -> HeatProfile:
    profiles = build_heat_profiles()
    if profile_name not in profiles:
        raise KeyError(f"Unknown profile: {profile_name}")
    return profiles[profile_name]


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

def make_eval_config(seed: Optional[int] = 7) -> BatteryThermalConfig:
    """Evaluation config aligned with Phase 2 benchmark."""
    return BatteryThermalConfig(
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
        temp_error_weight=1.0,
        over_temp_weight=8.0,
        action_weight=0.03,
        action_smoothness_weight=0.08,
        hard_violation_penalty=100.0,
        seed=seed,
    )


# -----------------------------------------------------------------------------
# PPO policy adapter
# -----------------------------------------------------------------------------

class PPOController:
    """Adapter that makes a Stable-Baselines3 PPO model look like a controller."""

    def __init__(self, model: PPO, vec_normalize: VecNormalize, name: str = "PPO") -> None:
        self.model = model
        self.vec_normalize = vec_normalize
        self.name = name

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Predict action from raw environment observation.

        Important:
            The PPO model was trained with VecNormalize, so raw observations must be
            normalized before calling model.predict().
        """
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        norm_obs = self.vec_normalize.normalize_obs(obs_batch)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return np.array([float(np.clip(action[0], 0.0, 1.0))], dtype=np.float32)


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def build_baseline_controllers(config: BatteryThermalConfig) -> List[ThermalController]:
    """Build the baseline set used for RL comparison."""
    return [
        NoCoolingController(),
        ConstantCoolingController(cooling_level=0.5),
        ConstantCoolingController(cooling_level=1.0),
        BangBangController(target_temp=config.target_temp, deadband=1.0),
        ProportionalController(target_temp=config.target_temp, kp=0.08, bias=0.15),
        PIController(
            target_temp=config.target_temp,
            kp=0.20,
            ki=0.006,
            bias=0.25,
            dt=config.dt,
            integral_limit=75.0,
            name="PI controller tuned",
        ),
    ]


def run_controller_case(
    controller,
    heat_profile: HeatProfile,
    config: BatteryThermalConfig,
    seed: int = 7,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float | str | bool]]:
    """Run one controller on one benchmark profile."""
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
    action_log = log["action"]
    time = log["time"]

    failed = bool(terminated and np.max(temperature) >= config.hard_max_temp)

    metrics: Dict[str, float | str | bool] = {
        "controller": controller.name,
        "max_temperature_C": float(np.max(temperature)),
        "final_temperature_C": float(temperature[-1]),
        "mean_abs_temp_error_C": float(np.mean(np.abs(temperature - config.target_temp))),
        "rms_temp_error_C": float(np.sqrt(np.mean((temperature - config.target_temp) ** 2))),
        "time_above_safe_s": float(np.sum(temperature > config.soft_max_temp) * config.dt),
        "total_cooling_effort": float(np.sum(action_log) * config.dt),
        "mean_cooling_action": float(np.mean(action_log)),
        "action_variation": float(np.sum(np.abs(np.diff(action_log)))) if len(action_log) > 1 else 0.0,
        "total_reward": float(total_reward),
        "failed": failed,
        "episode_time_s": float(time[-1]) if len(time) > 0 else 0.0,
    }

    return log, metrics


def make_dummy_vec_env_for_vecnormalize(config: BatteryThermalConfig) -> DummyVecEnv:
    """
    Create a dummy env only so VecNormalize can be loaded.

    We do not use this env to step evaluation. We only need the saved obs_rms stats.
    """

    def _init():
        env = BatteryThermalEnv(
            config=config,
            heat_profile=constant_heat_profile(),
            render_mode=None,
        )
        return Monitor(env)

    return DummyVecEnv([_init])


def load_ppo_controller(
    model_path: Path,
    vecnorm_path: Path,
    config: BatteryThermalConfig,
    name: str,
) -> PPOController:
    """Load PPO model and VecNormalize stats."""
    if not model_path.exists():
        raise FileNotFoundError(f"Missing PPO model: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"Missing VecNormalize stats: {vecnorm_path}")

    dummy_env = make_dummy_vec_env_for_vecnormalize(config)
    vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False

    model = PPO.load(str(model_path), env=None, device="auto")
    return PPOController(model=model, vec_normalize=vec_normalize, name=name)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_temperature_results(
    all_logs: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    config: BatteryThermalConfig,
    output_path: Path,
) -> None:
    profile_names = ["Constant", "Step", "Pulsed", "Random"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    fig.suptitle("Phase 3 — PPO vs baseline temperature comparison", fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, profile_names):
        for (logged_profile, controller_name), log in all_logs.items():
            if logged_profile != profile_name:
                continue
            linewidth = 2.4 if "PPO" in controller_name else 1.3
            ax.plot(log["time"], log["temperature"], linewidth=linewidth, label=controller_name)

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
    profile_names = ["Constant", "Step", "Pulsed", "Random"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    fig.suptitle("Phase 3 — PPO vs baseline cooling action comparison", fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, profile_names):
        for (logged_profile, controller_name), log in all_logs.items():
            if logged_profile != profile_name:
                continue
            linewidth = 2.4 if "PPO" in controller_name else 1.3
            ax.plot(log["time"], log["action"], linewidth=linewidth, label=controller_name)

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
    profile_names = ["Constant", "Step", "Pulsed", "Random"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), sharex=True)
    axes = axes.flatten()

    fig.suptitle("Phase 3 — Benchmark heat-generation profiles", fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, profile_names):
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

    print("\n=== Thermal safety ranking by total time above safe limit: lower is better ===")
    safety_rank = (
        summary_df.groupby("controller", as_index=False)["time_above_safe_s"]
        .sum()
        .sort_values("time_above_safe_s", ascending=True)
    )
    print(safety_rank.to_string(index=False))

    print("\n=== Cooling effort ranking: lower is better only if safe ===")
    effort_rank = (
        summary_df.groupby("controller", as_index=False)["total_cooling_effort"]
        .mean()
        .sort_values("total_cooling_effort", ascending=True)
    )
    print(effort_rank.to_string(index=False))

    print("\n=== PPO vs tuned PI check ===")
    grouped = summary_df.groupby("controller", as_index=False).agg(
        mean_reward=("total_reward", "mean"),
        mean_effort=("total_cooling_effort", "mean"),
        total_time_above_safe=("time_above_safe_s", "sum"),
        worst_max_temp=("max_temperature_C", "max"),
    )

    ppo_rows = grouped[grouped["controller"].str.contains("PPO", case=False, regex=False)]
    pi_rows = grouped[grouped["controller"] == "PI controller tuned"]

    if len(ppo_rows) == 0 or len(pi_rows) == 0:
        print("Could not compare PPO and tuned PI because one is missing.")
        return

    best_ppo = ppo_rows.sort_values("mean_reward", ascending=False).iloc[0]
    tuned_pi = pi_rows.iloc[0]

    delta_reward = float(best_ppo["mean_reward"] - tuned_pi["mean_reward"])
    delta_effort = float(best_ppo["mean_effort"] - tuned_pi["mean_effort"])

    print(f"Best PPO controller: {best_ppo['controller']}")
    print(f"PPO mean reward:     {best_ppo['mean_reward']:.3f}")
    print(f"PI mean reward:      {tuned_pi['mean_reward']:.3f}")
    print(f"Reward delta:        {delta_reward:.3f}  positive means PPO beat PI")
    print(f"PPO mean effort:     {best_ppo['mean_effort']:.3f}")
    print(f"PI mean effort:      {tuned_pi['mean_effort']:.3f}")
    print(f"Effort delta:        {delta_effort:.3f}  negative means PPO used less cooling")
    print(f"PPO worst max temp:  {best_ppo['worst_max_temp']:.3f} C")
    print(f"PI worst max temp:   {tuned_pi['worst_max_temp']:.3f} C")


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    model_dir = PROJECT_ROOT / "models" / "ppo_battery_thermal"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_eval_config(seed=7)
    profile_names = ["Constant", "Step", "Pulsed", "Random"]

    vecnorm_path = model_dir / "vec_normalize.pkl"
    best_model_path = model_dir / "best_model.zip"
    final_model_path = model_dir / "final_model.zip"

    controllers = build_baseline_controllers(config)

    # Evaluate best model if available. Evaluate final model too, because sometimes
    # the final model is worse than the best checkpoint.
    if best_model_path.exists():
        controllers.append(
            load_ppo_controller(
                model_path=best_model_path,
                vecnorm_path=vecnorm_path,
                config=config,
                name="PPO best model",
            )
        )
    else:
        print(f"Warning: missing {best_model_path}. Skipping PPO best model.")

    if final_model_path.exists():
        controllers.append(
            load_ppo_controller(
                model_path=final_model_path,
                vecnorm_path=vecnorm_path,
                config=config,
                name="PPO final model",
            )
        )
    else:
        print(f"Warning: missing {final_model_path}. Skipping PPO final model.")

    if not any("PPO" in controller.name for controller in controllers):
        raise RuntimeError("No PPO model found. Train PPO first with: python -m training.train_ppo")

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
                f"effort={metrics['total_cooling_effort']:.1f} | "
                f"reward={metrics['total_reward']:.2f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    first_cols = ["profile", "controller"]
    other_cols = [col for col in summary_df.columns if col not in first_cols]
    summary_df = summary_df[first_cols + other_cols]

    csv_path = output_dir / "phase3_rl_vs_baselines_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    plot_temperature_results(
        all_logs=all_logs,
        config=config,
        output_path=output_dir / "phase3_rl_vs_baselines_temperature.png",
    )
    plot_action_results(
        all_logs=all_logs,
        output_path=output_dir / "phase3_rl_vs_baselines_actions.png",
    )
    plot_heat_profiles(
        all_logs=all_logs,
        output_path=output_dir / "phase3_rl_vs_baselines_heat_profiles.png",
    )

    print_rankings(summary_df)

    print("\nSaved outputs:")
    print(f"  {csv_path}")
    print(f"  {output_dir / 'phase3_rl_vs_baselines_temperature.png'}")
    print(f"  {output_dir / 'phase3_rl_vs_baselines_actions.png'}")
    print(f"  {output_dir / 'phase3_rl_vs_baselines_heat_profiles.png'}")


if __name__ == "__main__":
    main()
