"""
scripts/compare_pack_baselines_1d_50mm.py

Baseline benchmark for the 1D battery pack with 50 mm cell spacing.

At 50 mm spacing the inter-cell conduction conductance drops from 8.0 W/K
(2 mm reference gap) to 0.32 W/K.  Cells are more thermally isolated so
hotspots build faster and spread less — a harder control problem.

Run from project root:
    python -m scripts.compare_pack_baselines_1d_50mm

Outputs:
    outputs/1d_50mm_baseline_summary.csv
    outputs/1d_50mm_baseline_temperatures.png
    outputs/1d_50mm_baseline_actions.png
    outputs/1d_50mm_baseline_spread.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envs.battery_pack_thermal_env import BatteryPackThermalConfig
from scripts.compare_pack_baselines import (
    build_pack_baseline_controllers,
    run_controller_case,
)

PROFILE_NAMES = ["UniformConstant", "NonuniformStep", "PulsedHotspot", "RandomNonuniform"]


def make_50mm_config() -> BatteryPackThermalConfig:
    """8-cell 1D pack with 50 mm inter-cell gap.

    cell_spacing_m=0.050 triggers __post_init__ which sets:
        conduction_coupling = 8.0 * (0.002 / 0.050) = 0.32 W/K
    """
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
        cell_spacing_m=0.050,
        target_temp=35.0,
        soft_max_temp=45.0,
        hard_max_temp=60.0,
        seed=7,
    )


def _plot_metric(
    all_logs: Dict[Tuple[str, str], Dict],
    log_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    hlines: Optional[List[Tuple[float, str, str]]] = None,
    derived_fn=None,
) -> None:
    """Plot a log metric across all 4 profiles.

    If derived_fn is set it is called as derived_fn(log) → array instead of
    reading log[log_key] directly.  Used for computed series like T_spread.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, profile in zip(axes, PROFILE_NAMES):
        for (pname, cname), log in all_logs.items():
            if pname != profile:
                continue
            series = derived_fn(log) if derived_fn else log[log_key]
            ax.plot(log["time"], series, linewidth=1.4, label=cname)

        if hlines:
            for val, ls, label in hlines:
                ax.axhline(val, linestyle=ls, linewidth=1.0, color="gray",
                           label=label if profile == PROFILE_NAMES[0] else None)
        ax.set_title(profile)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.80, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_rankings(df: pd.DataFrame) -> None:
    print("\n=== Ranking by mean reward (higher = better) ===")
    print(df.groupby("controller")["total_reward"].mean().sort_values(ascending=False).to_string())
    print("\n=== Safety: time above safe limit (lower = better) ===")
    print(df.groupby("controller")["time_above_safe_s"].sum().sort_values(ascending=True).to_string())
    print("\n=== Hotspot: mean temperature spread (lower = better) ===")
    print(df.groupby("controller")["mean_temperature_spread_C"].mean().sort_values(ascending=True).to_string())


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_50mm_config()
    print(f"1D pack  |  n_cells={config.n_cells}  "
          f"spacing={config.cell_spacing_m * 1000:.0f} mm  "
          f"conduction_coupling={config.conduction_coupling:.3f} W/K")

    controllers = build_pack_baseline_controllers(config)
    all_logs: Dict[Tuple[str, str], Dict] = {}
    summary_rows: List[Dict] = []

    for profile_name in PROFILE_NAMES:
        for ctrl in controllers:
            log, metrics = run_controller_case(
                controller=ctrl,
                profile_name=profile_name,
                config=config,
                seed=7,
            )
            all_logs[(profile_name, ctrl.name)] = log
            summary_rows.append(metrics)
            print(
                f"{profile_name:16s} | {ctrl.name:32s} | "
                f"T_max={metrics['max_cell_temperature_C']:.1f}°C  "
                f"spread={metrics['max_temperature_spread_C']:.2f}°C  "
                f"above_safe={metrics['time_above_safe_s']:.0f}s  "
                f"reward={metrics['total_reward']:.1f}"
            )

    df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "1d_50mm_baseline_summary.csv"
    df.to_csv(csv_path, index=False)

    _plot_metric(
        all_logs, "max_temperature",
        title="1D pack 50 mm spacing — Max cell temperature",
        ylabel="T_max (°C)",
        output_path=output_dir / "1d_50mm_baseline_temperatures.png",
        hlines=[
            (config.target_temp, ":", "Target"),
            (config.soft_max_temp, "--", "Safe limit"),
        ],
    )
    _plot_metric(
        all_logs, "action",
        title="1D pack 50 mm spacing — Cooling commands",
        ylabel="u",
        output_path=output_dir / "1d_50mm_baseline_actions.png",
    )
    _plot_metric(
        all_logs, "",
        title="1D pack 50 mm spacing — Cell temperature spread (T_max − T_min)",
        ylabel="Spread (°C)",
        output_path=output_dir / "1d_50mm_baseline_spread.png",
        derived_fn=lambda log: np.asarray(log["max_temperature"]) - np.asarray(log["min_temperature"]),
    )

    print_rankings(df)

    print("\nSaved:")
    print(f"  {csv_path}")
    for tag in ["temperatures", "actions", "spread"]:
        print(f"  {output_dir / f'1d_50mm_baseline_{tag}.png'}")


if __name__ == "__main__":
    main()
