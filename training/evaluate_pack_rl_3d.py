"""
training/evaluate_pack_rl_3d.py

Unified benchmark: SAC vs PPO vs all classical 3D baseline controllers.

Run from project root:
    python -m training.evaluate_pack_rl_3d

Requires at least one trained model:
    models/sac_pack_3d/best_model.zip     (train_pack_sac_3d)
    models/ppo_pack_3d/best_model.zip     (train_pack_ppo_3d)

Outputs:
    outputs/phase2_3d_rl_vs_baselines_summary.csv
    outputs/phase2_3d_rl_vs_baselines_temperatures.png
    outputs/phase2_3d_rl_vs_baselines_tavg.png
    outputs/phase2_3d_rl_vs_baselines_actions.png
    outputs/phase2_3d_rl_vs_baselines_gradient.png
    outputs/phase2_3d_rl_vs_baselines_heat_balance.png
    outputs/phase2_3d_rl_vs_baselines_reward.png
    outputs/phase2_3d_rl_ranking.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from configs.pack_config import CellConfig, PackConfig
from envs.battery_pack_thermal_env_3d import (
    BatteryPackThermalEnv3D,
    uniform_constant_3d_heat,
)
from scripts.compare_pack_baselines_3d import (
    NoCooling3D,
    ConstantCooling3D,
    BangBang3D,
    Proportional3D,
    PI3D,
    run_controller_case,
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
# SAC adapter — no VecNormalize; uses raw observations
# ---------------------------------------------------------------------------

class PackSAC3DController:
    """Wraps a trained SB3 SAC model as a pack controller."""

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
# PPO adapter — requires VecNormalize statistics from training
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


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

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


def _load_sac(model_dir: Path, file_name: str, name: str) -> Optional[PackSAC3DController]:
    path = model_dir / file_name
    if not path.exists():
        print(f"Warning: {path} not found — skipping {name}.")
        return None
    try:
        model = SAC.load(str(path), env=None, device="auto")
        print(f"Loaded: {name}")
        return PackSAC3DController(model=model, name=name)
    except Exception as exc:
        print(f"Warning: could not load {name}: {exc}")
        return None


def _load_ppo(
    model_dir: Path,
    file_name: str,
    vecnorm_path: Path,
    pack_config: PackConfig,
    name: str,
) -> Optional[PackPPO3DController]:
    model_path = model_dir / file_name
    if not model_path.exists():
        print(f"Warning: {model_path} not found — skipping {name}.")
        return None
    if not vecnorm_path.exists():
        print(f"Warning: {vecnorm_path} not found — skipping {name}.")
        return None
    try:
        dummy_env = _make_dummy_vec_env(pack_config)
        vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        model = PPO.load(str(model_path), env=None, device="auto")
        print(f"Loaded: {name}")
        return PackPPO3DController(model=model, vec_normalize=vec_normalize, name=name)
    except Exception as exc:
        print(f"Warning: could not load {name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Build full controller set
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

    # SAC models (no VecNormalize needed)
    sac_dir = PROJECT_ROOT / "models" / "sac_pack_3d"
    for model_name, file_name in [
        ("SAC best model (3D)", "best_model.zip"),
        ("SAC final model (3D)", "final_model.zip"),
    ]:
        ctrl = _load_sac(sac_dir, file_name, model_name)
        if ctrl is not None:
            controllers.append(ctrl)

    # PPO models (require VecNormalize)
    ppo_dir = PROJECT_ROOT / "models" / "ppo_pack_3d"
    vecnorm_path = ppo_dir / "vec_normalize.pkl"
    for model_name, file_name in [
        ("PPO best model (3D)", "best_model.zip"),
        ("PPO final model (3D)", "final_model.zip"),
    ]:
        ctrl = _load_ppo(ppo_dir, file_name, vecnorm_path, pack_config, model_name)
        if ctrl is not None:
            controllers.append(ctrl)

    return controllers


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_STYLE: Dict[str, Dict] = {
    "SAC": {"lw": 2.6, "ls": "-",  "zorder": 10},
    "PPO": {"lw": 2.0, "ls": "--", "zorder": 9},
    "PI":  {"lw": 1.8, "ls": "-.", "zorder": 8},
    "other": {"lw": 1.2, "ls": ":", "zorder": 5},
}

def _line_style(name: str) -> Dict:
    if "SAC" in name:
        return _STYLE["SAC"]
    if "PPO" in name:
        return _STYLE["PPO"]
    if "PI" in name:
        return _STYLE["PI"]
    return _STYLE["other"]


def _plot_metric_grid(
    all_logs: Dict[Tuple[str, str], Dict],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    hlines: Optional[List[Tuple[float, str, str]]] = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, PROFILE_NAMES):
        for (pname, cname), log in all_logs.items():
            if pname != profile_name:
                continue
            st = _line_style(cname)
            ax.plot(
                log["time"], log[metric_key],
                linewidth=st["lw"], linestyle=st["ls"],
                zorder=st["zorder"], label=cname,
            )

        if hlines:
            for val, ls, label in hlines:
                ax.axhline(
                    val, linestyle=ls, linewidth=1.0, color="gray",
                    label=label if profile_name == PROFILE_NAMES[0] else None,
                )

        ax.set_title(profile_name, fontsize=11)
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
    """Q_gen, Q_cool, and net heat per controller per profile."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, PROFILE_NAMES):
        for (pname, cname), log in all_logs.items():
            if pname != profile_name:
                continue
            st = _line_style(cname)
            q_gen  = np.asarray(log.get("q_gen_total",  []))
            q_cool = np.asarray(log.get("q_cool_total", []))
            time   = np.asarray(log["time"])
            if q_gen.size == 0:
                continue
            ax.plot(time, q_gen,  linewidth=st["lw"], linestyle=st["ls"],
                    color="#d62728", zorder=st["zorder"], label=f"{cname} Q_gen" if pname == PROFILE_NAMES[0] else None)
            if q_cool.size == q_gen.size:
                ax.plot(time, q_cool, linewidth=st["lw"], linestyle="--",
                        color="#1f77b4", zorder=st["zorder"], label=f"{cname} Q_cool" if pname == PROFILE_NAMES[0] else None)
                net = q_gen - q_cool
                ax.fill_between(time, 0, net, where=(net > 0),
                                alpha=0.12, color="#d62728")
                ax.fill_between(time, 0, net, where=(net <= 0),
                                alpha=0.12, color="#1f77b4")

        ax.axhline(0, linewidth=0.8, color="black")
        ax.set_title(profile_name, fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heat (W)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7.5, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 0.78, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ranking_bar(df: pd.DataFrame, pack_config: PackConfig, output_path: Path) -> None:
    """Horizontal bar chart of mean reward across all profiles."""
    ranked = (
        df.groupby("controller")["total_reward"]
        .mean()
        .sort_values(ascending=True)
    )

    colors = []
    for name in ranked.index:
        if "SAC" in name:
            colors.append("#2ca02c")   # green
        elif "PPO" in name:
            colors.append("#1f77b4")   # blue
        elif "PI" in name:
            colors.append("#ff7f0e")   # orange
        else:
            colors.append("#aec7e8")   # light blue

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(ranked.index, ranked.values, color=colors, edgecolor="white", height=0.65)

    ax.set_xlabel("Mean total reward (higher = better)", fontsize=11)
    ax.set_title("3D Pack — RL vs Classical Controllers\n(all 4 heat profiles averaged)", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.25)

    for bar, val in zip(bars, ranked.values):
        ax.text(
            bar.get_width() + (abs(ranked.values.max() - ranked.values.min()) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}",
            va="center", ha="left", fontsize=9,
        )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="SAC (RL)"),
        Patch(facecolor="#1f77b4", label="PPO (RL)"),
        Patch(facecolor="#ff7f0e", label="PI (classical)"),
        Patch(facecolor="#aec7e8", label="Other baselines"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_rankings(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  Ranking by mean total reward (higher = better)")
    print("=" * 60)
    reward_rank = df.groupby("controller")["total_reward"].mean().sort_values(ascending=False)
    for rank, (name, val) in enumerate(reward_rank.items(), 1):
        tag = ""
        if "SAC" in name:
            tag = "  ← SAC"
        elif "PPO" in name:
            tag = "  ← PPO"
        elif "PI tuned" in name:
            tag = "  ← classical best"
        print(f"  {rank:2d}. {name:34s}  {val:8.2f}{tag}")

    print("\n" + "=" * 60)
    print("  Safety: cumulative time above safe limit (lower = better)")
    print("=" * 60)
    safe_rank = df.groupby("controller")["time_above_safe_s"].sum().sort_values(ascending=True)
    for name, val in safe_rank.items():
        print(f"  {name:38s}  {val:6.0f} s")

    print("\n" + "=" * 60)
    print("  Hotspot control: mean T_gradient (lower = better)")
    print("=" * 60)
    grad_rank = df.groupby("controller")["T_gradient_mean_C"].mean().sort_values(ascending=True)
    for name, val in grad_rank.items():
        print(f"  {name:38s}  {val:5.3f} °C")

    # RL vs PI head-to-head
    print("\n" + "=" * 60)
    print("  RL vs tuned PI head-to-head (mean reward delta)")
    print("=" * 60)
    pi_mean = df[df["controller"] == "PI tuned (T_max)"]["total_reward"].mean()
    for name in reward_rank.index:
        if "SAC" in name or "PPO" in name:
            rl_mean = df[df["controller"] == name]["total_reward"].mean()
            delta = rl_mean - pi_mean
            sign = "+" if delta >= 0 else ""
            win = "WINS" if delta >= 0 else "loses"
            print(f"  {name:34s}  {sign}{delta:.2f}  ({win} vs PI)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_config = CellConfig()
    pack_config = PackConfig(shape=(4, 3, 2))
    controllers = build_controllers(pack_config)

    rl_controllers = [c for c in controllers if "SAC" in c.name or "PPO" in c.name]
    if not rl_controllers:
        raise RuntimeError(
            "No trained RL models found.\n"
            "Train SAC:  python -m training.train_pack_sac_3d\n"
            "Train PPO:  python -m training.train_pack_ppo_3d"
        )

    all_logs: Dict[Tuple[str, str], Dict] = {}
    summary_rows: List[Dict] = []

    print(f"\nRunning {len(controllers)} controllers × {len(PROFILE_NAMES)} profiles ...\n")

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

            tag = ""
            if "SAC" in controller.name:
                tag = " [SAC]"
            elif "PPO" in controller.name:
                tag = " [PPO]"

            print(
                f"{profile_name:16s} | {controller.name:34s}{tag}\n"
                f"             T_max={metrics['T_max_peak_C']:.1f}°C  "
                f"grad={metrics['T_gradient_max_C']:.2f}°C  "
                f"above_safe={metrics['time_above_safe_s']:.0f}s  "
                f"reward={metrics['total_reward']:.1f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "phase2_3d_rl_vs_baselines_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="T_max",
        title="Phase 2 (3D) — Max cell temperature: RL vs baselines",
        ylabel="T_max (°C)",
        output_path=output_dir / "phase2_3d_rl_vs_baselines_temperatures.png",
        hlines=[
            (pack_config.target_temp_c, ":", "Target"),
            (pack_config.safe_temp_c, "--", "Safe limit"),
        ],
    )

    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="T_avg",
        title="Phase 2 (3D) — Average cell temperature: RL vs baselines",
        ylabel="T_avg (°C)",
        output_path=output_dir / "phase2_3d_rl_vs_baselines_tavg.png",
        hlines=[
            (pack_config.target_temp_c, ":", "Target"),
            (pack_config.safe_temp_c, "--", "Safe limit"),
        ],
    )

    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="action",
        title="Phase 2 (3D) — Cooling commands: RL vs baselines",
        ylabel="Cooling command u",
        output_path=output_dir / "phase2_3d_rl_vs_baselines_actions.png",
    )

    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="T_gradient",
        title="Phase 2 (3D) — Temperature gradient: RL vs baselines",
        ylabel="T_gradient (°C)",
        output_path=output_dir / "phase2_3d_rl_vs_baselines_gradient.png",
    )

    plot_heat_balance(
        all_logs=all_logs,
        title="Phase 2 (3D) — Heat balance: Q_gen vs Q_cool",
        output_path=output_dir / "phase2_3d_rl_vs_baselines_heat_balance.png",
    )

    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="reward",
        title="Phase 2 (3D) — Per-step reward: RL vs baselines",
        ylabel="Reward",
        output_path=output_dir / "phase2_3d_rl_vs_baselines_reward.png",
    )

    plot_ranking_bar(
        df=summary_df,
        pack_config=pack_config,
        output_path=output_dir / "phase2_3d_rl_ranking.png",
    )

    print_rankings(summary_df)

    print("\nSaved outputs:")
    print(f"  {csv_path}")
    for tag in ["temperatures", "tavg", "actions", "gradient", "heat_balance", "reward"]:
        print(f"  {output_dir / f'phase2_3d_rl_vs_baselines_{tag}.png'}")
    print(f"  {output_dir / 'phase2_3d_rl_ranking.png'}")


if __name__ == "__main__":
    main()
