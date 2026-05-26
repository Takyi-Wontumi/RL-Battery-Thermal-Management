"""
evaluation/compare_controllers.py

Compare all controllers on the 3D battery pack thermal environment.

Controllers evaluated:
    Classical: No cooling, Constant 0.5, Constant 1.0, Bang-bang, Proportional, PI tuned
    RL:        PPO, SAC

Usage:
    python evaluation/compare_controllers.py \\
        --ppo-model /path/to/ppo_pack_final.zip \\
        --sac-model /path/to/sac_pack_final.zip \\
        --results-dir /path/to/results \\
        --episodes 20

    # Baselines only (no RL models required):
    python evaluation/compare_controllers.py --results-dir outputs/comparison

Outputs (in --results-dir):
    comparison_metrics.csv
    temperature_curves.png
    cooling_control_curves.png
    reward_comparison.png
    max_temperature_comparison.png
    cooling_energy_comparison.png
    temperature_uniformity_comparison.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from configs.pack_config import CellConfig, PackConfig
from envs.battery_pack_thermal_env_3d import (
    BatteryPackThermalEnv3D,
    uniform_constant_3d_heat,
)
from scripts.compare_pack_baselines_3d import (
    PROFILE_NAMES,
    build_3d_baseline_controllers,
    run_controller_case,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare PPO, SAC, and classical controllers on 3D pack"
    )
    p.add_argument("--ppo-model",    type=str, default=None,
                   help="Path to PPO model .zip")
    p.add_argument("--ppo-vecnorm",  type=str, default=None,
                   help="Path to PPO vec_normalize.pkl (inferred from --ppo-model dir if omitted)")
    p.add_argument("--sac-model",    type=str, default=None,
                   help="Path to SAC model .zip")
    p.add_argument("--results-dir",  type=str, default=None,
                   help="Output directory (default: outputs/comparison/)")
    p.add_argument("--episodes",     type=int, default=20,
                   help="Total evaluation episodes across all profiles (default: 20)")
    p.add_argument("--seed",         type=int, default=7,
                   help="Base random seed (default: 7)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# RL controller adapters
# ---------------------------------------------------------------------------

class PackSAC3DController:
    """SAC adapter — uses raw observations, no VecNormalize."""

    def __init__(self, model: SAC, name: str = "SAC") -> None:
        self.model = model
        self.name  = name

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_in = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action, _ = self.model.predict(obs_in, deterministic=True)
        u = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        return np.array([u], dtype=np.float32)


class PackPPO3DController:
    """PPO adapter — normalises observations with stored VecNormalize stats."""

    def __init__(self, model: PPO, vec_normalize: VecNormalize, name: str = "PPO") -> None:
        self.model        = model
        self.vec_normalize = vec_normalize
        self.name         = name

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        norm_obs  = self.vec_normalize.normalize_obs(obs_batch)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        u = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        return np.array([u], dtype=np.float32)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _make_dummy_env(pack_config: PackConfig) -> DummyVecEnv:
    def _init():
        env = BatteryPackThermalEnv3D(
            cell_config=CellConfig(),
            pack_config=pack_config,
            heat_profile=uniform_constant_3d_heat(),
            seed=0,
        )
        return Monitor(env)
    return DummyVecEnv([_init])


def load_sac(model_path: str, name: str = "SAC") -> Optional[PackSAC3DController]:
    path = Path(model_path)
    if not path.exists():
        print(f"Warning: SAC model not found at {path} — skipping.")
        return None
    try:
        model = SAC.load(str(path), env=None, device="auto")
        print(f"Loaded SAC: {path.name}")
        return PackSAC3DController(model=model, name=name)
    except Exception as exc:
        print(f"Warning: could not load SAC: {exc}")
        return None


def load_ppo(
    model_path: str,
    vecnorm_path: Optional[str],
    pack_config: PackConfig,
    name: str = "PPO",
) -> Optional[PackPPO3DController]:
    mp = Path(model_path)
    vp = Path(vecnorm_path) if vecnorm_path else mp.parent / "vec_normalize.pkl"

    if not mp.exists():
        print(f"Warning: PPO model not found at {mp} — skipping.")
        return None
    if not vp.exists():
        print(f"Warning: PPO vec_normalize not found at {vp} — skipping.")
        return None
    try:
        dummy = _make_dummy_env(pack_config)
        vn    = VecNormalize.load(str(vp), dummy)
        vn.training    = False
        vn.norm_reward = False
        model = PPO.load(str(mp), env=None, device="auto")
        print(f"Loaded PPO: {mp.name}  (vecnorm: {vp.name})")
        return PackPPO3DController(model=model, vec_normalize=vn, name=name)
    except Exception as exc:
        print(f"Warning: could not load PPO: {exc}")
        return None


# ---------------------------------------------------------------------------
# Build full controller list
# ---------------------------------------------------------------------------

def build_controllers(
    args: argparse.Namespace,
    pack_config: PackConfig,
) -> list:
    controllers = list(build_3d_baseline_controllers(pack_config))

    if args.sac_model:
        ctrl = load_sac(args.sac_model, name="SAC")
        if ctrl:
            controllers.append(ctrl)

    if args.ppo_model:
        ctrl = load_ppo(args.ppo_model, args.ppo_vecnorm, pack_config, name="PPO")
        if ctrl:
            controllers.append(ctrl)

    return controllers


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_all(
    controllers: list,
    pack_config: PackConfig,
    cell_config: CellConfig,
    n_rounds: int,
    seed_base: int,
) -> Tuple[Dict, pd.DataFrame]:
    all_logs: Dict[Tuple, Dict] = {}
    rows: List[Dict] = []

    for round_idx in range(n_rounds):
        seed = seed_base + round_idx * 100
        for profile in PROFILE_NAMES:
            for ctrl in controllers:
                log, metrics = run_controller_case(
                    controller=ctrl,
                    profile_name=profile,
                    cell_config=cell_config,
                    pack_config=pack_config,
                    seed=seed,
                )
                all_logs[(round_idx, profile, ctrl.name)] = log
                metrics.update({"round": round_idx, "seed": seed})
                rows.append(metrics)
                tag = " [RL]" if ("SAC" in ctrl.name or "PPO" in ctrl.name) else ""
                print(
                    f"  r{round_idx} {profile:16s} | {ctrl.name:30s}{tag} | "
                    f"T_max={metrics['T_max_peak_C']:.1f}°C  "
                    f"above_safe={metrics['time_above_safe_s']:.0f}s  "
                    f"reward={metrics['total_reward']:.1f}"
                )
    return all_logs, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _line_style(name: str) -> Dict:
    if "SAC" in name:  return {"lw": 2.6, "ls": "-",  "zorder": 10, "color": "#2ca02c"}
    if "PPO" in name:  return {"lw": 2.0, "ls": "--", "zorder": 9,  "color": "#1f77b4"}
    if "PI"  in name:  return {"lw": 1.8, "ls": "-.", "zorder": 8,  "color": "#ff7f0e"}
    return              {"lw": 1.2, "ls": ":",  "zorder": 5,  "color": None}


def _bar_color(name: str) -> str:
    if "SAC" in name: return "#2ca02c"
    if "PPO" in name: return "#1f77b4"
    if "PI"  in name: return "#ff7f0e"
    return "#aec7e8"


def plot_time_series(
    all_logs: Dict,
    log_key: str,
    title: str,
    ylabel: str,
    path: Path,
    hlines: Optional[List[Tuple]] = None,
    derived_fn=None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Collect one representative log per (profile, controller) — last round
    best_per: Dict[Tuple[str, str], Dict] = {}
    for (r, profile, cname), log in all_logs.items():
        best_per[(profile, cname)] = log

    for ax, profile in zip(axes, PROFILE_NAMES):
        for (pname, cname), log in best_per.items():
            if pname != profile:
                continue
            st = _line_style(cname)
            series = derived_fn(log) if derived_fn else log.get(log_key, [])
            if len(series) == 0:
                continue
            kwargs = {"linewidth": st["lw"], "linestyle": st["ls"], "zorder": st["zorder"],
                      "label": cname}
            if st["color"]:
                kwargs["color"] = st["color"]
            ax.plot(log["time"], series, **kwargs)
        if hlines:
            for val, ls, label in hlines:
                ax.axhline(val, linestyle=ls, linewidth=0.9, color="gray",
                           label=label if profile == PROFILE_NAMES[0] else None)
        ax.set_title(profile, fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7.5, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 0.78, 0.94])
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bar(df: pd.DataFrame, metric: str, title: str, xlabel: str, path: Path,
             ascending: bool = True) -> None:
    ranked = df.groupby("controller")[metric].mean().sort_values(ascending=ascending)
    colors = [_bar_color(n) for n in ranked.index]

    fig, ax = plt.subplots(figsize=(9, max(4, len(ranked) * 0.6)))
    bars = ax.barh(ranked.index, ranked.values, color=colors, edgecolor="white", height=0.65)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.25)

    span = abs(ranked.values.max() - ranked.values.min()) or 1.0
    for bar, val in zip(bars, ranked.values):
        ax.text(bar.get_width() + span * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=8)

    ax.legend(handles=[
        Patch(facecolor="#2ca02c", label="SAC (RL)"),
        Patch(facecolor="#1f77b4", label="PPO (RL)"),
        Patch(facecolor="#ff7f0e", label="PI (classical)"),
        Patch(facecolor="#aec7e8", label="Other baselines"),
    ], loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rankings printout
# ---------------------------------------------------------------------------

def print_rankings(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  Mean total reward (higher = better)")
    print("=" * 65)
    for rank, (name, val) in enumerate(
        df.groupby("controller")["total_reward"].mean()
          .sort_values(ascending=False).items(), 1
    ):
        tag = " ← SAC" if "SAC" in name else (" ← PPO" if "PPO" in name else
              " ← classical best" if "PI tuned" in name else "")
        print(f"  {rank:2d}. {name:34s}  {val:9.2f}{tag}")

    print("\n" + "=" * 65)
    print("  Time above safe limit — s (lower = better)")
    print("=" * 65)
    for name, val in (df.groupby("controller")["time_above_safe_s"]
                        .sum().sort_values().items()):
        print(f"  {name:38s}  {val:6.0f} s")

    print("\n" + "=" * 65)
    print("  Peak temperature spread — °C (lower = better)")
    print("=" * 65)
    for name, val in (df.groupby("controller")["T_gradient_max_C"]
                        .mean().sort_values().items()):
        print(f"  {name:38s}  {val:5.2f} °C")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "outputs" / "comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    cell_config = CellConfig()
    pack_config = PackConfig(
        shape=(4, 3, 2),
        cell_spacing_m=0.002,
        ambient_temp_c=25.0,
        target_temp_c=35.0,
        safe_temp_c=45.0,
        critical_temp_c=55.0,
        h_min_w_per_m2_k=5.0,
        h_max_w_per_m2_k=80.0,
        g_cond_w_per_k=0.25,
        enable_heat_variation=True,
    )

    controllers = build_controllers(args, pack_config)
    n_rounds    = max(1, args.episodes // len(PROFILE_NAMES))

    print(f"\n{len(controllers)} controllers × {len(PROFILE_NAMES)} profiles × "
          f"{n_rounds} rounds ({n_rounds * len(PROFILE_NAMES) * len(controllers)} total episodes)\n")

    all_logs, df = run_all(
        controllers=controllers,
        pack_config=pack_config,
        cell_config=cell_config,
        n_rounds=n_rounds,
        seed_base=args.seed,
    )

    # --- Save CSV ---
    csv_path = results_dir / "comparison_metrics.csv"
    df.to_csv(csv_path, index=False)

    # --- Plots ---
    hlines_temp = [
        (pack_config.target_temp_c, ":", "Target"),
        (pack_config.safe_temp_c,   "--", "Safe limit"),
    ]

    plot_time_series(
        all_logs, "T_max", "Max cell temperature", "T_max (°C)",
        results_dir / "temperature_curves.png", hlines=hlines_temp,
    )
    plot_time_series(
        all_logs, "action", "Cooling command", "u",
        results_dir / "cooling_control_curves.png",
    )
    plot_time_series(
        all_logs, "reward", "Per-step reward", "reward",
        results_dir / "reward_comparison.png",
    )

    plot_bar(df, "T_max_peak_C",       "Peak cell temperature (°C)",
             "T_max peak (°C)",        results_dir / "max_temperature_comparison.png",
             ascending=True)
    plot_bar(df, "total_cooling_effort", "Total cooling energy (u·s)",
             "Cooling effort",         results_dir / "cooling_energy_comparison.png",
             ascending=True)
    plot_bar(df, "T_gradient_mean_C", "Temperature uniformity — mean spread (°C)",
             "Mean T_gradient (°C)",   results_dir / "temperature_uniformity_comparison.png",
             ascending=True)
    plot_bar(df, "total_reward",       "Mean total reward (higher = better)",
             "Mean reward",            results_dir / "reward_bar_comparison.png",
             ascending=False)

    print_rankings(df)

    print(f"\nSaved to {results_dir}/")
    for f in sorted(results_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
