"""
scripts/tune_pi_controller.py

Grid-search tuning for the PI baseline controller.

Purpose:
    Find PI gains that create a strong classical baseline before training RL.

Run from project root:
    python -m scripts.tune_pi_controller

Output:
    outputs/phase2_pi_tuning_results.csv

This is not fancy optimization. It is deliberately simple and transparent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from controllers.baseline_controllers import PIController
from envs.battery_thermal_env import BatteryThermalConfig, BatteryThermalEnv


HeatProfile = Callable[[float, np.random.Generator], float]


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


def make_profile(profile_name: str) -> HeatProfile:
    profiles: Dict[str, HeatProfile] = {
        "Constant": constant_heat_profile(),
        "Step": step_heat_profile(),
        "Pulsed": pulsed_heat_profile(),
        "Random": random_heat_profile(),
    }
    return profiles[profile_name]


def run_pi_case(
    profile_name: str,
    kp: float,
    ki: float,
    bias: float,
    integral_limit: float,
    config: BatteryThermalConfig,
    seed: int = 7,
) -> Dict[str, float | str | bool]:
    env = BatteryThermalEnv(
        config=config,
        heat_profile=make_profile(profile_name),
        render_mode=None,
    )

    controller = PIController(
        target_temp=config.target_temp,
        kp=kp,
        ki=ki,
        bias=bias,
        dt=config.dt,
        integral_limit=integral_limit,
        name="PI controller",
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

    failed = bool(terminated and np.max(temperature) >= config.hard_max_temp)

    return {
        "profile": profile_name,
        "kp": kp,
        "ki": ki,
        "bias": bias,
        "integral_limit": integral_limit,
        "max_temperature_C": float(np.max(temperature)),
        "mean_abs_temp_error_C": float(np.mean(np.abs(temperature - config.target_temp))),
        "rms_temp_error_C": float(np.sqrt(np.mean((temperature - config.target_temp) ** 2))),
        "time_above_safe_s": float(np.sum(temperature > config.soft_max_temp) * config.dt),
        "total_cooling_effort": float(np.sum(action) * config.dt),
        "mean_cooling_action": float(np.mean(action)),
        "action_variation": float(np.sum(np.abs(np.diff(action)))) if len(action) > 1 else 0.0,
        "total_reward": float(total_reward),
        "failed": failed,
    }


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

    kp_values = [0.06, 0.08, 0.10, 0.12, 0.16, 0.20]
    ki_values = [0.0005, 0.001, 0.002, 0.004, 0.006]
    bias_values = [0.05, 0.10, 0.15, 0.20, 0.25]
    integral_limits = [40.0, 75.0, 100.0, 150.0]

    rows: List[Dict[str, float | str | bool]] = []

    total_cases = len(profile_names) * len(kp_values) * len(ki_values) * len(bias_values) * len(integral_limits)
    case_idx = 0

    for kp in kp_values:
        for ki in ki_values:
            for bias in bias_values:
                for integral_limit in integral_limits:
                    for profile_name in profile_names:
                        case_idx += 1
                        result = run_pi_case(
                            profile_name=profile_name,
                            kp=kp,
                            ki=ki,
                            bias=bias,
                            integral_limit=integral_limit,
                            config=config,
                            seed=7,
                        )
                        rows.append(result)

                    if case_idx % 80 == 0:
                        print(f"Completed {case_idx}/{total_cases} profile cases...")

    raw_df = pd.DataFrame(rows)

    # Aggregate across all heat profiles.
    grouped = (
        raw_df.groupby(["kp", "ki", "bias", "integral_limit"], as_index=False)
        .agg(
            mean_total_reward=("total_reward", "mean"),
            worst_max_temperature_C=("max_temperature_C", "max"),
            mean_abs_temp_error_C=("mean_abs_temp_error_C", "mean"),
            mean_rms_temp_error_C=("rms_temp_error_C", "mean"),
            total_time_above_safe_s=("time_above_safe_s", "sum"),
            mean_total_cooling_effort=("total_cooling_effort", "mean"),
            mean_action_variation=("action_variation", "mean"),
            any_failure=("failed", "max"),
        )
        .sort_values("mean_total_reward", ascending=False)
    )

    # Safety-first filter. Reward-only tuning can produce stupid controllers.
    safe_df = grouped[
        (grouped["any_failure"] == False)
        & (grouped["total_time_above_safe_s"] == 0.0)
        & (grouped["worst_max_temperature_C"] <= 43.0)
    ].copy()

    raw_path = output_dir / "phase2_pi_tuning_raw_results.csv"
    summary_path = output_dir / "phase2_pi_tuning_results.csv"
    safe_path = output_dir / "phase2_pi_tuning_safe_candidates.csv"

    raw_df.to_csv(raw_path, index=False)
    grouped.to_csv(summary_path, index=False)
    safe_df.to_csv(safe_path, index=False)

    print("\n=== Top 10 PI settings by mean reward ===")
    print(grouped.head(10).to_string(index=False))

    print("\n=== Top 10 safe PI candidates ===")
    if len(safe_df) == 0:
        print("No safe candidates found under the current filter. Loosen the filter or expand the gain grid.")
    else:
        print(safe_df.head(10).to_string(index=False))

    print("\nSaved outputs:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {safe_path}")


if __name__ == "__main__":
    main()
