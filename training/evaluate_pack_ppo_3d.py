"""
training/evaluate_pack_ppo_3d.py

Evaluate trained 3D pack PPO against classical 3D baseline controllers.

Run from project root:
    python -m training.evaluate_pack_ppo_3d

Requires:
    models/ppo_pack_3d/best_model.zip
    models/ppo_pack_3d/final_model.zip
    models/ppo_pack_3d/vec_normalize.pkl

    Train first with:  python -m training.train_pack_ppo_3d

Outputs:
    outputs/phase2_3d_ppo_vs_baselines_summary.csv
    outputs/phase2_3d_ppo_vs_baselines_temperatures.png
    outputs/phase2_3d_ppo_vs_baselines_actions.png
    outputs/phase2_3d_ppo_vs_baselines_gradient.png
    outputs/phase2_3d_ppo_vs_baselines_heat_profiles.png
    outputs/phase2_3d_ppo_layer_heatmaps_<profile>.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from configs.pack_config import CellConfig, PackConfig
from envs.battery_pack_thermal_env_3d import (
    BatteryPackThermalEnv3D,
    make_3d_profile,
    uniform_constant_3d_heat,
)
from scripts.compare_pack_baselines_3d import (
    NoCooling3D,
    ConstantCooling3D,
    BangBang3D,
    Proportional3D,
    PI3D,
    run_controller_case,
    plot_layer_heatmaps,
    PROFILE_NAMES,
)


# ---------------------------------------------------------------------------
# Controller protocol
# ---------------------------------------------------------------------------

class Pack3DControllerLike(Protocol):
    name: str
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# PPO adapter
# ---------------------------------------------------------------------------

class PackPPO3DController:
    """Wraps a trained SB3 PPO model as a pack controller."""

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


def _make_dummy_vec_env(pack_config: PackConfig) -> DummyVecEnv:
    def _init():
        env = BatteryPackThermalEnv3D(
            cell_config=CellConfig(),
            pack_config=pack_config,
            heat_profile=uniform_constant_3d_heat(),
            seed=0,
        )
        return Monitor(env)
    return DummyVecEnv([_init])


def load_ppo_controller(
    model_path: Path,
    vecnorm_path: Path,
    pack_config: PackConfig,
    name: str,
) -> PackPPO3DController:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing PPO model: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"Missing VecNormalize: {vecnorm_path}")

    dummy_env = _make_dummy_vec_env(pack_config)
    vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False

    model = PPO.load(str(model_path), env=None, device="auto")
    return PackPPO3DController(model=model, vec_normalize=vec_normalize, name=name)


# ---------------------------------------------------------------------------
# Build controller set
# ---------------------------------------------------------------------------

def build_controllers(pack_config: PackConfig) -> List[Pack3DControllerLike]:
    controllers: List[Pack3DControllerLike] = [
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
            name="PI tuned (T_max)",
        ),
    ]

    model_dir = PROJECT_ROOT / "models" / "ppo_pack_3d"
    vecnorm_path = model_dir / "vec_normalize.pkl"

    for model_name, file_name in [("PPO best model (3D)", "best_model.zip"),
                                   ("PPO final model (3D)", "final_model.zip")]:
        path = model_dir / file_name
        if path.exists():
            try:
                controllers.append(
                    load_ppo_controller(path, vecnorm_path, pack_config, model_name)
                )
                print(f"Loaded: {model_name}")
            except Exception as exc:
                print(f"Warning: could not load {model_name}: {exc}")
        else:
            print(f"Warning: {path} not found — skipping {model_name}.")

    return controllers


# ---------------------------------------------------------------------------
# Plotting helpers (same pattern as the 1D scripts)
# ---------------------------------------------------------------------------

def _plot_metric_grid(
    all_logs: Dict[Tuple[str, str], Dict],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    hlines: Optional[List[Tuple[float, str, str]]] = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, PROFILE_NAMES):
        for (pname, cname), log in all_logs.items():
            if pname != profile_name:
                continue
            lw = 2.3 if "PPO" in cname else 1.3
            ax.plot(log["time"], log[metric_key], linewidth=lw, label=cname)

        if hlines:
            for val, ls, label in hlines:
                ax.axhline(
                    val, linestyle=ls, linewidth=1.0, color="gray",
                    label=label if profile_name == PROFILE_NAMES[0] else None,
                )

        ax.set_title(profile_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.80, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heat_profiles(all_logs: Dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.flatten()
    fig.suptitle("Phase 2 — 3D PPO benchmark heat profiles", fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, PROFILE_NAMES):
        log = next((l for (pn, _), l in all_logs.items() if pn == profile_name), None)
        if log is None:
            continue
        ax.plot(log["time"], log["q_gen_total"], linewidth=1.6, label="Total (W)")
        ax.plot(log["time"], log["q_gen_max_cell"], linewidth=1.0, linestyle="--", label="Max cell (W)")
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

def print_rankings(df: pd.DataFrame) -> None:
    print("\n=== Ranking by mean reward (higher = better) ===")
    print(df.groupby("controller")["total_reward"].mean().sort_values(ascending=False).to_string())

    print("\n=== Safety: time above safe limit (lower = better) ===")
    print(df.groupby("controller")["time_above_safe_s"].sum().sort_values(ascending=True).to_string())

    print("\n=== Hotspot: mean T_gradient (lower = better) ===")
    print(df.groupby("controller")["T_gradient_mean_C"].mean().sort_values(ascending=True).to_string())

    ppo_rows = df[df["controller"].str.contains("PPO", case=False, regex=False)]
    pi_rows = df[df["controller"] == "PI tuned (T_max)"]

    if ppo_rows.empty or pi_rows.empty:
        return

    best_ppo = ppo_rows.groupby("controller")["total_reward"].mean().idxmax()
    ppo_mean = ppo_rows[ppo_rows["controller"] == best_ppo]["total_reward"].mean()
    pi_mean = pi_rows["total_reward"].mean()

    print(f"\n=== PPO vs tuned PI ===")
    print(f"Best PPO ({best_ppo}): mean reward = {ppo_mean:.2f}")
    print(f"PI tuned (T_max):      mean reward = {pi_mean:.2f}")
    print(f"Delta (PPO - PI):      {ppo_mean - pi_mean:.2f}  (positive = PPO wins)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_config = CellConfig()
    pack_config = PackConfig(shape=(4, 3, 2))
    controllers = build_controllers(pack_config)

    if not any("PPO" in c.name for c in controllers):
        raise RuntimeError(
            "No trained 3D PPO model found.\n"
            "Train first with:  python -m training.train_pack_ppo_3d"
        )

    all_logs: Dict[Tuple[str, str], Dict] = {}
    summary_rows: List[Dict] = []

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

            print(
                f"{profile_name:16s} | {controller.name:30s} | "
                f"T_max={metrics['T_max_peak_C']:.1f}°C  "
                f"grad={metrics['T_gradient_max_C']:.2f}°C  "
                f"safe={metrics['time_above_safe_s']:.0f}s  "
                f"reward={metrics['total_reward']:.1f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "phase2_3d_ppo_vs_baselines_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="T_max",
        title="Phase 2 (3D) — PPO vs baselines: max cell temperature",
        ylabel="T_max (°C)",
        output_path=output_dir / "phase2_3d_ppo_vs_baselines_temperatures.png",
        hlines=[
            (pack_config.target_temp_c, ":", "Target"),
            (pack_config.safe_temp_c, "--", "Safe limit"),
        ],
    )

    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="action",
        title="Phase 2 (3D) — PPO vs baselines: cooling commands",
        ylabel="Cooling command u",
        output_path=output_dir / "phase2_3d_ppo_vs_baselines_actions.png",
    )

    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="T_gradient",
        title="Phase 2 (3D) — PPO vs baselines: temperature gradient",
        ylabel="T_gradient (°C)",
        output_path=output_dir / "phase2_3d_ppo_vs_baselines_gradient.png",
    )

    plot_heat_profiles(
        all_logs=all_logs,
        output_path=output_dir / "phase2_3d_ppo_vs_baselines_heat_profiles.png",
    )

    ppo_controllers = [c for c in controllers if "PPO" in c.name]
    best_ppo_name = ppo_controllers[-1].name if ppo_controllers else None

    for profile_name in PROFILE_NAMES:
        key = (profile_name, best_ppo_name) if best_ppo_name else None
        if key and key in all_logs:
            plot_layer_heatmaps(
                log=all_logs[key],
                profile_name=profile_name,
                pack_config=pack_config,
                output_path=output_dir / f"phase2_3d_ppo_layer_heatmaps_{profile_name.lower()}.png",
            )

    print_rankings(summary_df)

    print("\nSaved outputs:")
    print(f"  {csv_path}")
    for name in ["temperatures", "actions", "gradient", "heat_profiles"]:
        print(f"  {output_dir / f'phase2_3d_ppo_vs_baselines_{name}.png'}")


if __name__ == "__main__":
    main()
