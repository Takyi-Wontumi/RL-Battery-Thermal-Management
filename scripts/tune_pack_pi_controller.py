"""
scripts/tune_pack_pi_controller.py

Grid-search tuning for the multi-node battery pack PI controller.

Purpose:
    Find a strong classical PI baseline for the Phase 4 battery pack environment
    before training pack-level PPO.

Run from project root:
    python -m scripts.tune_pack_pi_controller

Outputs:
    outputs/phase4_pack_pi_tuning_raw_results.csv
    outputs/phase4_pack_pi_tuning_results.csv
    outputs/phase4_pack_pi_tuning_safe_candidates.csv

This tunes PI on max cell temperature and includes a temperature-spread term.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from envs.battery_pack_thermal_env import (
    BatteryPackThermalConfig,
    BatteryPackThermalEnv,
    make_pack_profile,
)
from scripts.compare_pack_baselines import extract_pack_state


@dataclass
class TunablePackPIController:
    config: BatteryPackThermalConfig
    target_temp: float
    kp: float
    ki: float
    bias: float
    imbalance_gain: float
    integral_limit: float
    name: str = "Pack PI tuned candidate"

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
        self.integral_error = float(
            np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        )

        u = (
            self.bias
            + self.kp * error
            + self.ki * self.integral_error
            + self.imbalance_gain * spread
        )
        u = float(np.clip(u, 0.0, 1.0))
        return np.array([u], dtype=np.float32)


def make_tuning_config(seed: int = 7) -> BatteryPackThermalConfig:
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


def run_pi_case(
    profile_name: str,
    kp: float,
    ki: float,
    bias: float,
    imbalance_gain: float,
    integral_limit: float,
    config: BatteryPackThermalConfig,
    seed: int = 7,
) -> Dict[str, float | str | bool]:
    env = BatteryPackThermalEnv(
        config=config,
        heat_profile=make_pack_profile(profile_name),
        render_mode=None,
    )

    controller = TunablePackPIController(
        config=config,
        target_temp=config.target_temp,
        kp=kp,
        ki=ki,
        bias=bias,
        imbalance_gain=imbalance_gain,
        integral_limit=integral_limit,
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

    failed = bool(terminated and np.max(max_temp) >= config.hard_max_temp)

    return {
        "profile": profile_name,
        "kp": kp,
        "ki": ki,
        "bias": bias,
        "imbalance_gain": imbalance_gain,
        "integral_limit": integral_limit,
        "max_cell_temperature_C": float(np.max(max_temp)),
        "mean_pack_temperature_C": float(np.mean(mean_temp)),
        "final_max_cell_temperature_C": float(max_temp[-1]),
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
    }


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_tuning_config(seed=7)

    profile_names = [
        "UniformConstant",
        "NonuniformStep",
        "PulsedHotspot",
        "RandomNonuniform",
    ]

    # Start around the current Pack PI: kp=0.16, ki=0.004, bias=0.20, imbalance_gain=0.05.
    # This grid is intentionally modest. Huge grids waste time and teach you nothing.
    kp_values = [0.10, 0.12, 0.16, 0.20, 0.24, 0.30]
    ki_values = [0.001, 0.002, 0.004, 0.006, 0.008]
    bias_values = [0.10, 0.15, 0.20, 0.25, 0.30]
    imbalance_gain_values = [0.00, 0.02, 0.05, 0.08, 0.12]
    integral_limits = [50.0, 75.0, 100.0, 150.0]

    rows: List[Dict[str, float | str | bool]] = []

    total_settings = (
        len(kp_values)
        * len(ki_values)
        * len(bias_values)
        * len(imbalance_gain_values)
        * len(integral_limits)
    )
    total_profile_cases = total_settings * len(profile_names)
    completed_profile_cases = 0
    completed_settings = 0

    for kp in kp_values:
        for ki in ki_values:
            for bias in bias_values:
                for imbalance_gain in imbalance_gain_values:
                    for integral_limit in integral_limits:
                        completed_settings += 1

                        for profile_name in profile_names:
                            completed_profile_cases += 1
                            result = run_pi_case(
                                profile_name=profile_name,
                                kp=kp,
                                ki=ki,
                                bias=bias,
                                imbalance_gain=imbalance_gain,
                                integral_limit=integral_limit,
                                config=config,
                                seed=7,
                            )
                            rows.append(result)

                        if completed_settings % 50 == 0:
                            print(
                                f"Completed {completed_settings}/{total_settings} gain settings "
                                f"({completed_profile_cases}/{total_profile_cases} profile cases)..."
                            )

    raw_df = pd.DataFrame(rows)

    grouped = (
        raw_df.groupby(
            ["kp", "ki", "bias", "imbalance_gain", "integral_limit"],
            as_index=False,
        )
        .agg(
            mean_total_reward=("total_reward", "mean"),
            worst_max_cell_temperature_C=("max_cell_temperature_C", "max"),
            mean_pack_temperature_C=("mean_pack_temperature_C", "mean"),
            mean_temperature_spread_C=("mean_temperature_spread_C", "mean"),
            worst_temperature_spread_C=("max_temperature_spread_C", "max"),
            mean_temperature_std_C=("mean_temperature_std_C", "mean"),
            total_time_above_safe_s=("time_above_safe_s", "sum"),
            mean_total_cooling_effort=("total_cooling_effort", "mean"),
            mean_action_variation=("action_variation", "mean"),
            any_failure=("failed", "max"),
        )
        .sort_values("mean_total_reward", ascending=False)
    )

    # Safety-first filter. Do not accept a controller that wins reward by flirting with failure.
    safe_df = grouped[
        (grouped["any_failure"] == False)
        & (grouped["total_time_above_safe_s"] == 0.0)
        & (grouped["worst_max_cell_temperature_C"] <= 42.0)
    ].copy()

    # Engineering preference: among safe candidates, reward first, then effort, then imbalance.
    safe_df = safe_df.sort_values(
        ["mean_total_reward", "mean_total_cooling_effort", "mean_temperature_spread_C"],
        ascending=[False, True, True],
    )

    raw_path = output_dir / "phase4_pack_pi_tuning_raw_results.csv"
    summary_path = output_dir / "phase4_pack_pi_tuning_results.csv"
    safe_path = output_dir / "phase4_pack_pi_tuning_safe_candidates.csv"

    raw_df.to_csv(raw_path, index=False)
    grouped.to_csv(summary_path, index=False)
    safe_df.to_csv(safe_path, index=False)

    print("\n=== Top 15 Pack PI settings by mean reward ===")
    print(grouped.head(15).to_string(index=False))

    print("\n=== Top 15 safe Pack PI candidates ===")
    if len(safe_df) == 0:
        print("No safe candidates found. Loosen the filter or expand the gain grid.")
    else:
        print(safe_df.head(15).to_string(index=False))

    print("\nCurrent hand-tuned Pack PI reference:")
    reference = grouped[
        (grouped["kp"] == 0.16)
        & (grouped["ki"] == 0.004)
        & (grouped["bias"] == 0.20)
        & (grouped["imbalance_gain"] == 0.05)
        & (grouped["integral_limit"] == 100.0)
    ]
    if len(reference) > 0:
        print(reference.to_string(index=False))
    else:
        print("Reference setting was not found in grid. Check the grid values.")

    print("\nSaved outputs:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {safe_path}")


if __name__ == "__main__":
    main()
