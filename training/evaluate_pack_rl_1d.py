"""
training/evaluate_pack_rl_1d.py

Unified benchmark: SAC vs PPO vs classical baselines — 1D pack, 50 mm spacing.

Run from project root:
    python -m training.evaluate_pack_rl_1d

Requires at least one trained model:
    models/sac_pack_1d/best_model.zip     (train_pack_sac_1d)
    models/ppo_pack_1d/best_model.zip     (train_pack_ppo_1d)

Outputs:
    outputs/1d_50mm_rl_vs_baselines_summary.csv
    outputs/1d_50mm_rl_vs_baselines_temperatures.png
    outputs/1d_50mm_rl_vs_baselines_actions.png
    outputs/1d_50mm_rl_vs_baselines_spread.png
    outputs/1d_50mm_rl_vs_baselines_heat_balance.png
    outputs/1d_50mm_rl_ranking.png
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
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.battery_pack_thermal_env import BatteryPackThermalEnv, uniform_constant_pack_heat
from scripts.compare_pack_baselines import (
    build_pack_baseline_controllers,
    run_controller_case,
)
from scripts.compare_pack_baselines_1d_50mm import make_50mm_config, PROFILE_NAMES


# ---------------------------------------------------------------------------
# Controller protocol
# ---------------------------------------------------------------------------

class Pack1DControllerLike(Protocol):
    name: str
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# SAC adapter — raw observations, no VecNormalize
# ---------------------------------------------------------------------------

class PackSAC1DController:
    def __init__(self, model: SAC, name: str) -> None:
        self.model = model
        self.name = name

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_input = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action, _ = self.model.predict(obs_input, deterministic=True)
        u = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        return np.array([u], dtype=np.float32)


# ---------------------------------------------------------------------------
# PPO adapter — requires VecNormalize statistics
# ---------------------------------------------------------------------------

class PackPPO1DController:
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


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_sac(model_dir: Path, file_name: str, name: str) -> Optional[PackSAC1DController]:
    path = model_dir / file_name
    if not path.exists():
        print(f"Warning: {path} not found — skipping {name}.")
        return None
    try:
        model = SAC.load(str(path), env=None, device="auto")
        print(f"Loaded: {name}")
        return PackSAC1DController(model=model, name=name)
    except Exception as exc:
        print(f"Warning: could not load {name}: {exc}")
        return None


def _load_ppo(model_dir: Path, file_name: str, vecnorm_path: Path, name: str) -> Optional[PackPPO1DController]:
    model_path = model_dir / file_name
    if not model_path.exists():
        print(f"Warning: {model_path} not found — skipping {name}.")
        return None
    if not vecnorm_path.exists():
        print(f"Warning: {vecnorm_path} not found — skipping {name}.")
        return None
    try:
        config = make_50mm_config()

        def _init():
            env = BatteryPackThermalEnv(config=config,
                                        heat_profile=uniform_constant_pack_heat())
            return Monitor(env)

        dummy_env = DummyVecEnv([_init])
        vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        model = PPO.load(str(model_path), env=None, device="auto")
        print(f"Loaded: {name}")
        return PackPPO1DController(model=model, vec_normalize=vec_normalize, name=name)
    except Exception as exc:
        print(f"Warning: could not load {name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Build full controller set
# ---------------------------------------------------------------------------

def build_controllers() -> List[Pack1DControllerLike]:
    config = make_50mm_config()
    controllers: List[Pack1DControllerLike] = list(build_pack_baseline_controllers(config))

    sac_dir = PROJECT_ROOT / "models" / "sac_pack_1d"
    for name, fname in [("SAC best (1D)", "best_model.zip"),
                         ("SAC final (1D)", "final_model.zip")]:
        ctrl = _load_sac(sac_dir, fname, name)
        if ctrl is not None:
            controllers.append(ctrl)

    ppo_dir = PROJECT_ROOT / "models" / "ppo_pack_1d"
    vecnorm = ppo_dir / "vec_normalize.pkl"
    for name, fname in [("PPO best (1D)", "best_model.zip"),
                         ("PPO final (1D)", "final_model.zip")]:
        ctrl = _load_ppo(ppo_dir, fname, vecnorm, name)
        if ctrl is not None:
            controllers.append(ctrl)

    return controllers


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _line_style(name: str) -> Dict:
    if "SAC" in name:
        return {"lw": 2.6, "ls": "-",  "zorder": 10}
    if "PPO" in name:
        return {"lw": 2.0, "ls": "--", "zorder": 9}
    if "PI" in name:
        return {"lw": 1.8, "ls": "-.", "zorder": 8}
    return {"lw": 1.2, "ls": ":",  "zorder": 5}


def _plot_metric(
    all_logs: Dict[Tuple[str, str], Dict],
    log_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    hlines: Optional[List[Tuple[float, str, str]]] = None,
    derived_fn=None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, profile in zip(axes, PROFILE_NAMES):
        for (pname, cname), log in all_logs.items():
            if pname != profile:
                continue
            st = _line_style(cname)
            series = derived_fn(log) if derived_fn else log[log_key]
            ax.plot(log["time"], series,
                    linewidth=st["lw"], linestyle=st["ls"],
                    zorder=st["zorder"], label=cname)

        if hlines:
            for val, ls, label in hlines:
                ax.axhline(val, linestyle=ls, linewidth=1.0, color="gray",
                           label=label if profile == PROFILE_NAMES[0] else None)
        ax.set_title(profile, fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7.5, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 0.78, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heat_balance(
    all_logs: Dict[Tuple[str, str], Dict],
    title: str,
    output_path: Path,
) -> None:
    """Q_gen_total, Q_cool_total, and net heat per controller per profile."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, profile in zip(axes, PROFILE_NAMES):
        for (pname, cname), log in all_logs.items():
            if pname != profile:
                continue
            st = _line_style(cname)
            q_gen  = np.asarray(log.get("total_heat_generation", []))
            q_cool = np.asarray(log.get("total_cooling", []))
            time   = np.asarray(log["time"])
            if q_gen.size == 0:
                continue
            ax.plot(time, q_gen, linewidth=st["lw"], linestyle=st["ls"],
                    color="#d62728", zorder=st["zorder"],
                    label=f"{cname} Q_gen" if profile == PROFILE_NAMES[0] else None)
            if q_cool.size == q_gen.size:
                ax.plot(time, q_cool, linewidth=st["lw"], linestyle="--",
                        color="#1f77b4", zorder=st["zorder"],
                        label=f"{cname} Q_cool" if profile == PROFILE_NAMES[0] else None)
                net = q_gen - q_cool
                ax.fill_between(time, 0, net, where=(net > 0),
                                alpha=0.12, color="#d62728")
                ax.fill_between(time, 0, net, where=(net <= 0),
                                alpha=0.12, color="#1f77b4")

        ax.axhline(0, linewidth=0.8, color="black")
        ax.set_title(profile, fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heat (W)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7.5, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 0.78, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ranking_bar(df: pd.DataFrame, output_path: Path) -> None:
    ranked = (df.groupby("controller")["total_reward"]
               .mean()
               .sort_values(ascending=True))

    colors = []
    for name in ranked.index:
        if "SAC" in name:      colors.append("#2ca02c")
        elif "PPO" in name:    colors.append("#1f77b4")
        elif "PI" in name:     colors.append("#ff7f0e")
        else:                   colors.append("#aec7e8")

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(ranked.index, ranked.values, color=colors, edgecolor="white", height=0.65)
    ax.set_xlabel("Mean total reward (higher = better)", fontsize=11)
    ax.set_title("1D Pack 50 mm spacing — RL vs Classical Controllers\n"
                 "(all 4 heat profiles averaged)", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.25)

    span = abs(ranked.values.max() - ranked.values.min())
    for bar, val in zip(bars, ranked.values):
        ax.text(bar.get_width() + span * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", ha="left", fontsize=9)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2ca02c", label="SAC (RL)"),
        Patch(facecolor="#1f77b4", label="PPO (RL)"),
        Patch(facecolor="#ff7f0e", label="PI (classical)"),
        Patch(facecolor="#aec7e8", label="Other baselines"),
    ], loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_rankings(df: pd.DataFrame) -> None:
    config = make_50mm_config()
    print("\n" + "=" * 60)
    print("  1D pack (50 mm) — mean total reward (higher = better)")
    print("=" * 60)
    reward_rank = df.groupby("controller")["total_reward"].mean().sort_values(ascending=False)
    for rank, (name, val) in enumerate(reward_rank.items(), 1):
        tag = "  ← SAC" if "SAC" in name else ("  ← PPO" if "PPO" in name else
              "  ← classical best" if "PI tuned" in name else "")
        print(f"  {rank:2d}. {name:38s}  {val:8.2f}{tag}")

    print("\n" + "=" * 60)
    print("  Safety: cumulative time above safe limit (lower = better)")
    print("=" * 60)
    for name, val in df.groupby("controller")["time_above_safe_s"].sum().sort_values().items():
        print(f"  {name:42s}  {val:6.0f} s")

    print("\n" + "=" * 60)
    print("  RL vs tuned PI — mean reward delta")
    print("=" * 60)
    pi_rows = df[df["controller"].str.contains("PI tuned", case=False, regex=False)]
    if pi_rows.empty:
        print("  PI tuned not found in results.")
        return
    pi_mean = pi_rows["total_reward"].mean()
    for name in reward_rank.index:
        if "SAC" in name or "PPO" in name:
            rl_mean = df[df["controller"] == name]["total_reward"].mean()
            delta = rl_mean - pi_mean
            sign = "+" if delta >= 0 else ""
            win = "WINS" if delta >= 0 else "loses"
            print(f"  {name:38s}  {sign}{delta:.2f}  ({win} vs PI)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_50mm_config()
    controllers = build_controllers()

    rl_controllers = [c for c in controllers if "SAC" in c.name or "PPO" in c.name]
    if not rl_controllers:
        raise RuntimeError(
            "No trained 1D RL models found.\n"
            "Train SAC:  python -m training.train_pack_sac_1d\n"
            "Train PPO:  python -m training.train_pack_ppo_1d"
        )

    all_logs: Dict[Tuple[str, str], Dict] = {}
    summary_rows: List[Dict] = []

    print(f"\nRunning {len(controllers)} controllers × {len(PROFILE_NAMES)} profiles "
          f"(50 mm spacing, g={config.conduction_coupling:.3f} W/K) ...\n")

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

            tag = " [SAC]" if "SAC" in ctrl.name else (" [PPO]" if "PPO" in ctrl.name else "")
            print(
                f"{profile_name:16s} | {ctrl.name:38s}{tag}\n"
                f"             T_max={metrics['max_cell_temperature_C']:.1f}°C  "
                f"spread={metrics['max_temperature_spread_C']:.2f}°C  "
                f"above_safe={metrics['time_above_safe_s']:.0f}s  "
                f"reward={metrics['total_reward']:.1f}"
            )

    df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "1d_50mm_rl_vs_baselines_summary.csv"
    df.to_csv(csv_path, index=False)

    _plot_metric(
        all_logs, "max_temperature",
        title="1D pack 50 mm — Max cell temperature: RL vs baselines",
        ylabel="T_max (°C)",
        output_path=output_dir / "1d_50mm_rl_vs_baselines_temperatures.png",
        hlines=[(config.target_temp, ":", "Target"),
                (config.soft_max_temp, "--", "Safe limit")],
    )
    _plot_metric(
        all_logs, "action",
        title="1D pack 50 mm — Cooling commands: RL vs baselines",
        ylabel="u",
        output_path=output_dir / "1d_50mm_rl_vs_baselines_actions.png",
    )
    _plot_metric(
        all_logs, "",
        title="1D pack 50 mm — Cell temperature spread: RL vs baselines",
        ylabel="T_max − T_min (°C)",
        output_path=output_dir / "1d_50mm_rl_vs_baselines_spread.png",
        derived_fn=lambda log: np.asarray(log["max_temperature"]) - np.asarray(log["min_temperature"]),
    )

    plot_heat_balance(
        all_logs,
        title="1D pack 50 mm — Heat balance: Q_gen vs Q_cool",
        output_path=output_dir / "1d_50mm_rl_vs_baselines_heat_balance.png",
    )

    plot_ranking_bar(df, output_dir / "1d_50mm_rl_ranking.png")
    print_rankings(df)

    print("\nSaved:")
    print(f"  {csv_path}")
    for tag in ["temperatures", "actions", "spread", "heat_balance"]:
        print(f"  {output_dir / f'1d_50mm_rl_vs_baselines_{tag}.png'}")
    print(f"  {output_dir / '1d_50mm_rl_ranking.png'}")


if __name__ == "__main__":
    main()
