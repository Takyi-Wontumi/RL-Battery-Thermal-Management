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


def _expected_obs_dim(pack_config: PackConfig) -> int:
    """Obs dim produced by the current env for a given pack config (no echem profile)."""
    return 6 + pack_config.shape[0]  # 6 thermal stats + n_zones action history


def load_sac(model_path: str, pack_config: PackConfig, name: str = "SAC") -> Optional[PackSAC3DController]:
    path = Path(model_path)
    if not path.exists():
        print(f"Warning: SAC model not found at {path} — skipping.")
        return None
    try:
        model = SAC.load(str(path), env=None, device="auto")
        model_obs_dim = model.observation_space.shape[0]
        expected = _expected_obs_dim(pack_config)
        if model_obs_dim != expected:
            print(
                f"Warning: SAC obs dim {model_obs_dim} != env obs dim {expected}. "
                f"Model was trained on an old environment — skipping. "
                f"Re-run training to get a compatible model."
            )
            return None
        print(f"Loaded SAC: {path.name}  (obs_dim={model_obs_dim})")
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
        model = PPO.load(str(mp), env=None, device="auto")
        model_obs_dim = model.observation_space.shape[0]
        expected = _expected_obs_dim(pack_config)
        if model_obs_dim != expected:
            print(
                f"Warning: PPO obs dim {model_obs_dim} != env obs dim {expected}. "
                f"Model was trained on an old environment — skipping. "
                f"Re-run training to get a compatible model."
            )
            return None
        dummy = _make_dummy_env(pack_config)
        vn = VecNormalize.load(str(vp), dummy)
        vn.training    = False
        vn.norm_reward = False
        print(f"Loaded PPO: {mp.name}  (obs_dim={model_obs_dim}, vecnorm: {vp.name})")
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
        ctrl = load_sac(args.sac_model, pack_config, name="SAC")
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


def plot_reward_components(df: pd.DataFrame, path: Path) -> None:
    """
    Grouped bar chart showing mean reward component contribution per controller.
    Makes explicit WHY each controller gets the score it does (R6).
    Components with larger magnitude (more negative) are the binding constraints.
    """
    comp_cols = [c for c in df.columns if c.startswith("mean_reward_")]
    if not comp_cols:
        return

    labels = [c.replace("mean_reward_", "").replace("_", " ") for c in comp_cols]
    ctrl_means = df.groupby("controller")[comp_cols].mean()
    controllers = ctrl_means.index.tolist()

    # Sort safe controllers first, then unsafe
    if "is_safe" in df.columns:
        safe_set = set(df[df["is_safe"]]["controller"].unique())
        controllers = (
            [c for c in controllers if c in safe_set] +
            [c for c in controllers if c not in safe_set]
        )
        ctrl_means = ctrl_means.loc[controllers]

    x = np.arange(len(labels))
    n = len(controllers)
    width = 0.8 / n

    colors = []
    for name in controllers:
        if "SAC" in name:   colors.append("#2ca02c")
        elif "PPO" in name: colors.append("#1f77b4")
        elif "PI" in name:  colors.append("#ff7f0e")
        else:               colors.append("#aec7e8")

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 2), 6))
    for i, (ctrl, color) in enumerate(zip(controllers, colors)):
        offset = (i - n / 2 + 0.5) * width
        vals = ctrl_means.loc[ctrl, comp_cols].values
        bars = ax.bar(x + offset, vals, width * 0.9, label=ctrl, color=color, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean reward contribution per step", fontsize=10)
    ax.set_title("Reward component breakdown — what each controller wins and loses on", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(bottom=min(ctrl_means.values.min() * 1.15, -0.05))

    plt.tight_layout()
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
# Rankings printout — two-stage: safety first, then reward (R2)
# ---------------------------------------------------------------------------

def print_rankings(df: pd.DataFrame, pack_config: Optional[PackConfig] = None) -> None:
    """
    Stage 1 — Safety pass/fail: time_above_safe == 0 AND T_max < critical_temp.
    Stage 2 — Among safe controllers: rank by mean total reward.
    Diagnostic footer shows mean reward component per controller (R6).
    """
    agg = df.groupby("controller").agg(
        is_safe=("is_safe", "all") if "is_safe" in df.columns
                else ("time_above_safe_s", lambda x: (x == 0).all()),
        total_reward=("total_reward", "mean"),
        T_max_peak=("T_max_peak_C", "mean"),
        time_above_safe=("time_above_safe_s", "mean"),
        T_gradient_mean=("T_gradient_mean_C", "mean"),
        mean_cooling=("mean_cooling_action", "mean"),
    )
    # Fallback if is_safe column wasn't produced (old data)
    if "is_safe" not in agg.columns:
        agg["is_safe"] = agg["time_above_safe"] == 0

    safe_ctrl   = agg[agg["is_safe"]].sort_values("total_reward", ascending=False)
    unsafe_ctrl = agg[~agg["is_safe"]].sort_values("T_max_peak", ascending=True)

    W = 76
    print("\n" + "=" * W)
    print("  EVALUATION RESULTS — two-stage ranking")
    print("=" * W)

    # Stage 1
    print("\n  STAGE 1 — Safety  (pass = 0 s above safe AND T_max < critical)")
    print(f"  {'Status':<6} {'Controller'}")
    for name, row in agg.sort_values("is_safe", ascending=False).iterrows():
        status = "PASS" if row["is_safe"] else "FAIL"
        detail = "" if row["is_safe"] else f"  ({row['time_above_safe']:.0f}s above safe, peak {row['T_max_peak']:.1f}°C)"
        print(f"  {status:<6} {name}{detail}")

    # Stage 2
    print(f"\n  STAGE 2 — Ranking among safe controllers  (higher = better)")
    print(f"  {'Rank':<5} {'Controller':<36} {'Reward':>8} {'T_max':>7} {'Grad':>7} {'u_mean':>7}")
    print("  " + "-" * (W - 2))
    for rank, (name, row) in enumerate(safe_ctrl.iterrows(), 1):
        tag = " ← SAC" if "SAC" in name else (" ← PPO" if "PPO" in name else "")
        print(f"  {rank:<5} {name:<36} {row['total_reward']:>8.2f} "
              f"{row['T_max_peak']:>6.1f}° {row['T_gradient_mean']:>6.2f}° "
              f"{row['mean_cooling']:>6.3f}{tag}")

    if not unsafe_ctrl.empty:
        print(f"\n  UNSAFE — excluded from ranking:")
        for name, row in unsafe_ctrl.iterrows():
            print(f"  ✗  {name:<36}  T_max={row['T_max_peak']:.1f}°C  "
                  f"above_safe={row['time_above_safe']:.0f}s")

    # Component diagnosis
    comp_cols = [c for c in df.columns if c.startswith("mean_reward_")]
    if comp_cols:
        labels = [c.replace("mean_reward_", "") for c in comp_cols]
        print(f"\n  Reward component diagnosis (mean/step — which term costs most)")
        header = f"  {'Controller':<36} " + " ".join(f"{l[:9]:>9}" for l in labels)
        print(header)
        print("  " + "-" * (W - 2))
        for name in list(safe_ctrl.index) + list(unsafe_ctrl.index):
            row_vals = df[df["controller"] == name][comp_cols].mean()
            vals = " ".join(f"{row_vals[c]:>9.3f}" for c in comp_cols)
            print(f"  {name:<36} {vals}")

    print("=" * W + "\n")


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
        target_low_temp_c=30.0,
        target_high_temp_c=38.0,
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

    plot_reward_components(df, results_dir / "reward_component_breakdown.png")

    print_rankings(df, pack_config)

    print(f"\nSaved to {results_dir}/")
    for f in sorted(results_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
