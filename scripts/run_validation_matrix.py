"""
scripts/run_validation_matrix.py

9-configuration sensor-realism validation matrix.

Purpose
-------
Before trusting any comparison between RL and classical baselines, verify that
the environment + sensor simulation behaves correctly across isolated failure
modes. Each configuration stresses ONE realism dimension at a time so that
unexpected behavior has a clear cause.

Test matrix
-----------
  A  Perfect sensors, no delay         — sanity baseline, must match old results
  B  Noise only                        — checks noise robustness (controller still works?)
  C  Sensor delay only                 — checks delayed feedback stability
  D  Sparse thermistors only           — checks realistic measurement limitation
  E  Actuator delay + rate limit       — checks physical actuator realism
  F  Random hotspot zone               — checks multi-zone targeting
  G  Sensor dropout (10% probability)  — checks fault tolerance
  H  Actuator degradation (70% eff.)   — checks cooling fault robustness
  I  Combined realism                  — final real-world evaluation

Run:
    python -m scripts.run_validation_matrix

Output:
    outputs/validation_matrix_summary.csv
    outputs/validation_matrix_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from configs.pack_config import PackConfig, CellConfig
from configs.sensor_simulation import SensorConfig, ActuatorConfig
from envs.battery_pack_thermal_env_3d import BatteryPackThermalEnv3D, make_3d_profile
from scripts.compare_pack_baselines_3d import build_3d_baseline_controllers


# ---------------------------------------------------------------------------
# Validation scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class ValidationScenario:
    tag: str
    label: str
    sensor_config: SensorConfig
    actuator_config: Optional[ActuatorConfig]
    enable_sensor_sim: bool
    description: str


def build_scenarios(n_zones: int = 4, dt_s: float = 1.0) -> List[ValidationScenario]:
    """Return the 9 validation scenarios in canonical order."""

    def _base_sensor(**kwargs) -> SensorConfig:
        defaults = dict(
            num_zones=n_zones,
            enabled=True,
            use_sparse_thermistors=False,
            temp_noise_std_c=0.0,
            current_noise_std_a=0.0,
            pack_voltage_noise_std_v=0.0,
            group_voltage_noise_std_v=0.0,
            coolant_temp_noise_std_c=0.0,
            actuator_feedback_noise_std=0.0,
            temp_bias_range_c=0.0,
            current_bias_range_a=0.0,
            voltage_bias_range_v=0.0,
            sensor_delay_s=0.0,
            enable_sensor_dropout=False,
            enable_lowpass_filter=False,
        )
        defaults.update(kwargs)
        return SensorConfig(**defaults)

    def _base_actuator(**kwargs) -> ActuatorConfig:
        defaults = dict(
            num_zones=n_zones,
            cooling_delay_s=0.0,
            enable_rate_limit=False,
            effectiveness=1.0,
            enable_actuator_fault=False,
        )
        defaults.update(kwargs)
        return ActuatorConfig(**defaults)

    _realistic_sensor = _base_sensor(
        use_sparse_thermistors=True,
        temp_noise_std_c=0.20,
        current_noise_std_a=0.20,
        pack_voltage_noise_std_v=0.02,
        group_voltage_noise_std_v=0.005,
        coolant_temp_noise_std_c=0.15,
        actuator_feedback_noise_std=0.01,
        temp_bias_range_c=0.50,
        current_bias_range_a=0.10,
        voltage_bias_range_v=0.01,
        sensor_delay_s=2.0,
        enable_sensor_dropout=False,
        enable_lowpass_filter=True,
        lowpass_alpha=0.35,
    )
    _realistic_actuator = _base_actuator(
        cooling_delay_s=5.0,
        enable_rate_limit=True,
        max_cooling_rate_per_s=0.05,
        effectiveness=1.0,
    )

    return [
        ValidationScenario(
            tag="A",
            label="Perfect sensors",
            sensor_config=_base_sensor(),
            actuator_config=_base_actuator(),
            enable_sensor_sim=True,
            description="Ideal observations. Must reproduce same results as non-sensor env.",
        ),
        ValidationScenario(
            tag="B",
            label="Noise only",
            sensor_config=_base_sensor(
                temp_noise_std_c=0.20,
                temp_bias_range_c=0.50,
                actuator_feedback_noise_std=0.01,
                enable_lowpass_filter=True,
                lowpass_alpha=0.35,
            ),
            actuator_config=_base_actuator(),
            enable_sensor_sim=True,
            description="Temperature noise + bias. Coolant and electrical noise kept for realism.",
        ),
        ValidationScenario(
            tag="C",
            label="Sensor delay only",
            sensor_config=_base_sensor(sensor_delay_s=4.0),
            actuator_config=_base_actuator(),
            enable_sensor_sim=True,
            description="4s sensor delay. Controller sees past temperatures.",
        ),
        ValidationScenario(
            tag="D",
            label="Sparse thermistors",
            sensor_config=_base_sensor(
                use_sparse_thermistors=True,
                thermistors_per_zone=2,
                temp_noise_std_c=0.10,
            ),
            actuator_config=_base_actuator(),
            enable_sensor_sim=True,
            description="2 thermistors per zone (8 total). Zone temp = max of sampled cells.",
        ),
        ValidationScenario(
            tag="E",
            label="Actuator lag",
            sensor_config=_base_sensor(),
            actuator_config=_base_actuator(
                cooling_delay_s=5.0,
                enable_rate_limit=True,
                max_cooling_rate_per_s=0.05,
            ),
            enable_sensor_sim=True,
            description="5s actuator delay + 0.05/s rate limit. Cooling arrives late and ramps slowly.",
        ),
        ValidationScenario(
            tag="F",
            label="Random hotspot",
            sensor_config=_base_sensor(
                use_sparse_thermistors=True,
                thermistors_per_zone=2,
                temp_noise_std_c=0.10,
            ),
            actuator_config=_base_actuator(
                cooling_delay_s=5.0,
                enable_rate_limit=True,
                max_cooling_rate_per_s=0.05,
            ),
            enable_sensor_sim=True,
            description="Sparse sensors + actuator lag. Hotspot zone forces multi-zone targeting.",
        ),
        ValidationScenario(
            tag="G",
            label="Sensor dropout",
            sensor_config=_base_sensor(
                use_sparse_thermistors=True,
                temp_noise_std_c=0.20,
                enable_sensor_dropout=True,
                dropout_probability=0.10,
                dropout_hold_last_value=True,
                enable_lowpass_filter=True,
                lowpass_alpha=0.35,
            ),
            actuator_config=_base_actuator(),
            enable_sensor_sim=True,
            description="10% per-zone per-step dropout. NaN recovery holds last valid reading.",
        ),
        ValidationScenario(
            tag="H",
            label="Actuator degradation",
            sensor_config=_base_sensor(
                temp_noise_std_c=0.20,
                actuator_feedback_noise_std=0.02,
                enable_lowpass_filter=True,
            ),
            actuator_config=_base_actuator(
                cooling_delay_s=3.0,
                enable_rate_limit=True,
                max_cooling_rate_per_s=0.05,
                effectiveness=0.70,  # 30% weaker than commanded
            ),
            enable_sensor_sim=True,
            description="70% actuator effectiveness. Controller must over-command to achieve target cooling.",
        ),
        ValidationScenario(
            tag="I",
            label="Combined realism",
            sensor_config=_realistic_sensor,
            actuator_config=_realistic_actuator,
            enable_sensor_sim=True,
            description="Full realism: noise, bias, delay, sparse thermistors, actuator lag.",
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: ValidationScenario,
    pack_config: PackConfig,
    cell_config: CellConfig,
    profile_name: str = "PulsedHotspot",
    seed: int = 7,
    controllers=None,
) -> pd.DataFrame:
    """Run all controllers under one validation scenario. Return metrics DataFrame."""
    if controllers is None:
        controllers = build_3d_baseline_controllers(pack_config)

    rows = []
    for ctrl in controllers:
        env = BatteryPackThermalEnv3D(
            cell_config=cell_config,
            pack_config=pack_config,
            heat_profile=make_3d_profile(profile_name),
            seed=seed,
            enable_sensor_simulation=scenario.enable_sensor_sim,
            sensor_config=scenario.sensor_config if scenario.enable_sensor_sim else None,
            actuator_config=scenario.actuator_config if scenario.enable_sensor_sim else None,
        )

        ctrl.reset()
        obs, info = env.reset(seed=seed, options={"randomize": False})
        total_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = ctrl.act(obs, info)
            obs, rew, terminated, truncated, info = env.step(action)
            total_reward += rew

        log = env.get_episode_log()

        # Scalar action series (mean across zones)
        if "actions" in log and "action" not in log:
            log["action"] = np.mean(np.asarray(log["actions"]), axis=1)

        T_max_arr = np.asarray(log["T_max"])
        time_above_safe = float(np.sum(T_max_arr > pack_config.safe_temp_c) * env.dt_s)
        is_safe = (time_above_safe == 0.0) and (T_max_arr.max() < pack_config.critical_temp_c)

        targeting_arr = np.asarray(log.get("targeting_correct", []))
        zone_targeting = float(np.mean(targeting_arr)) if len(targeting_arr) > 0 else float("nan")

        dropout_arr = np.asarray(log.get("sensor_dropout_active", []))
        dropout_steps = int(np.sum(dropout_arr)) if len(dropout_arr) > 0 else 0

        u_applied_arr = np.asarray(log.get("u_applied", []))
        if u_applied_arr.ndim == 2:
            saturated = np.any(u_applied_arr >= 0.99, axis=1)
            actuator_sat_frac = float(np.mean(saturated))
        else:
            actuator_sat_frac = float("nan")

        # Sensor error: |zone_max_meas - zone_max_true|
        zone_meas_arr = np.asarray(log.get("zone_max_temps", []))
        zone_true_arr = np.asarray(log.get("zone_max_temps_true", []))
        if zone_meas_arr.ndim == 2 and zone_true_arr.ndim == 2 and len(zone_meas_arr) > 0:
            meas_err = float(np.mean(np.abs(zone_meas_arr - zone_true_arr)))
        else:
            meas_err = 0.0

        rows.append({
            "scenario": scenario.tag,
            "scenario_label": scenario.label,
            "profile": profile_name,
            "controller": ctrl.name,
            "type": getattr(ctrl, "controller_type", "Unknown"),
            "T_max_peak_C": float(T_max_arr.max()),
            "T_gradient_mean_C": float(np.asarray(log["T_gradient"]).mean()),
            "time_above_safe_s": time_above_safe,
            "is_safe": bool(is_safe),
            "mean_cooling_action": float(np.asarray(log.get("action", log.get("actions", [[0.5]]))).mean()),
            "zone_targeting_accuracy": zone_targeting,
            "mean_abs_sensor_error_C": meas_err,
            "sensor_dropout_steps": dropout_steps,
            "actuator_saturation_frac": actuator_sat_frac,
            "total_reward": float(total_reward),
        })

        tag_str = f"{scenario.tag}  [{getattr(ctrl, 'controller_type', '?'):<22}] {ctrl.name:<28}"
        safe_str = "PASS" if is_safe else f"FAIL ({time_above_safe:.0f}s)"
        tgt_str  = f"{zone_targeting*100:.0f}%" if not np.isnan(zone_targeting) else " N/A"
        print(f"  {tag_str}  {safe_str:8s}  T_max={T_max_arr.max():.1f}°C  u_mean={rows[-1]['mean_cooling_action']:.3f}  target={tgt_str}")

    return pd.DataFrame(rows)


def print_scenario_summary(all_df: pd.DataFrame) -> None:
    """Print a compact cross-scenario targeting accuracy table for zone controllers."""
    zone_ctls = all_df[all_df["type"] == "Multi-zone classical"]
    if zone_ctls.empty:
        return

    W = 95
    print("\n" + "=" * W)
    print("  Zone targeting accuracy across scenarios  (% of steps: hottest zone == highest-cooled zone)")
    print(f"  {'Controller':<28}" + "".join(f"  {s:>6}" for s in all_df["scenario"].unique()))
    print("  " + "-" * (W - 2))

    for ctrl in zone_ctls["controller"].unique():
        row_str = f"  {ctrl:<28}"
        for sc in all_df["scenario"].unique():
            val = all_df[(all_df["controller"] == ctrl) & (all_df["scenario"] == sc)]["zone_targeting_accuracy"]
            v = float(val.mean()) if len(val) > 0 else float("nan")
            row_str += f"  {v*100:>5.1f}%" if not np.isnan(v) else "    N/A"
        print(row_str)
    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    pack_config = PackConfig(shape=(4, 3, 2))
    cell_config = CellConfig()
    scenarios = build_scenarios(n_zones=pack_config.num_cooling_zones, dt_s=1.0)

    # Use PulsedHotspot — it has the clearest spatial hotspot signal for targeting tests
    profile_name = "PulsedHotspot"

    # Subset of controllers for the validation matrix (exclude constant cooling for speed)
    all_controllers = build_3d_baseline_controllers(pack_config)
    controllers = [c for c in all_controllers if "Constant" not in c.name]

    all_rows = []
    for sc in scenarios:
        print(f"\nScenario {sc.tag}: {sc.label}")
        print(f"  {sc.description}")
        df_sc = run_scenario(sc, pack_config, cell_config, profile_name, controllers=controllers)
        all_rows.append(df_sc)

    all_df = pd.concat(all_rows, ignore_index=True)
    csv_path = output_dir / "validation_matrix_summary.csv"
    all_df.to_csv(csv_path, index=False)

    print_scenario_summary(all_df)

    # Text report
    report_lines = [
        "Validation matrix report",
        "=" * 70,
        f"Profile: {profile_name}",
        f"Pack: {pack_config.shape}  ({int(np.prod(pack_config.shape))} cells, {pack_config.num_cooling_zones} zones)",
        "",
        "Safety status per scenario (P=pass, F=fail):",
        f"  {'Controller':<28}" + "".join(f"  {s.tag:>4}" for s in scenarios),
        "-" * 70,
    ]
    for ctrl_name in all_df["controller"].unique():
        sub = all_df[all_df["controller"] == ctrl_name]
        row = f"  {ctrl_name:<28}"
        for sc in scenarios:
            val = sub[sub["scenario"] == sc.tag]["is_safe"]
            v = bool(val.values[0]) if len(val) > 0 else None
            row += f"  {'P' if v else 'F':>4}"
        report_lines.append(row)

    report_path = output_dir / "validation_matrix_report.txt"
    report_text = "\n".join(report_lines) + "\n"
    report_path.write_text(report_text)

    print(f"\nSaved:")
    print(f"  {csv_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
