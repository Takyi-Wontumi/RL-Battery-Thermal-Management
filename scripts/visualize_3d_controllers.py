"""
scripts/visualize_3d_controllers.py

Post-training 3D PyVista visualization: PI, SAC, and PPO side-by-side.

Runs one episode per controller on the NonuniformStep heat profile (8W→20W),
captures 4 temperature-field snapshots, and produces:
  - Individual off-screen PyVista PNGs per snapshot
  - A matplotlib summary figure (time series + embedded 3D images)

Usage (from project root):
    python scripts/visualize_3d_controllers.py
    python scripts/visualize_3d_controllers.py --output-dir outputs/3d_viz
    python scripts/visualize_3d_controllers.py --profile PulsedHotspot

Outputs (all in --output-dir):
    pyvista_<ctrl>_t<time>s.png      — one 3D temperature field image per snapshot
    3d_controller_comparison.png     — combined time-series + 3D image figure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.pack_config import CellConfig, PackConfig
from envs.battery_pack_thermal_env_3d import (
    BatteryPackThermalEnv3D,
    make_3d_profile,
    uniform_constant_3d_heat,
)
from scripts.compare_pack_baselines_3d import PI3D

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    from scripts.render_pack_3d_pyvista import render_temperature_field
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config — mirrors training/train_pack_ppo_3d.py make_pack_config()
# ---------------------------------------------------------------------------

def _make_pack_config() -> PackConfig:
    return PackConfig(
        shape=(4, 3, 2),
        cell_spacing_m=0.002,
        ambient_temp_c=25.0,
        initial_temp_c=25.0,
        h_min_w_per_m2_k=5.0,
        h_max_w_per_m2_k=80.0,
        target_temp_c=35.0,
        safe_temp_c=45.0,
        critical_temp_c=55.0,
        g_cond_w_per_k=0.25,
        enable_heat_variation=True,
        heat_variation_std=0.05,
    )


# ---------------------------------------------------------------------------
# RL model loaders
# ---------------------------------------------------------------------------

def _make_dummy_vec_env(pack_config: PackConfig) -> "DummyVecEnv":
    def _init():
        env = BatteryPackThermalEnv3D(
            cell_config=CellConfig(),
            pack_config=pack_config,
            heat_profile=uniform_constant_3d_heat(),
            seed=0,
        )
        return Monitor(env)
    return DummyVecEnv([_init])


def _load_sac(pack_config: PackConfig) -> Optional[object]:
    if not SB3_AVAILABLE:
        return None
    path = PROJECT_ROOT / "models" / "sac_pack_3d" / "best_model.zip"
    if not path.exists():
        print(f"  SAC model not found: {path}")
        return None
    try:
        from training.evaluate_pack_rl_3d import PackSAC3DController
        model = SAC.load(str(path), env=None, device="auto")
        expected = 6 + pack_config.shape[0]
        if model.observation_space.shape[0] != expected:
            print(f"  SAC obs_dim mismatch ({model.observation_space.shape[0]} != {expected}) — retrain first.")
            return None
        print(f"  Loaded SAC: {path}")
        return PackSAC3DController(model=model, name="SAC (best)")
    except Exception as exc:
        print(f"  SAC load failed: {exc}")
        return None


def _load_ppo(pack_config: PackConfig) -> Optional[object]:
    if not SB3_AVAILABLE:
        return None
    model_dir = PROJECT_ROOT / "models" / "ppo_pack_3d"
    model_path = model_dir / "best_model.zip"
    vecnorm_path = model_dir / "vec_normalize.pkl"
    if not model_path.exists():
        print(f"  PPO model not found: {model_path}")
        return None
    if not vecnorm_path.exists():
        print(f"  VecNormalize not found: {vecnorm_path}")
        return None
    try:
        from training.evaluate_pack_rl_3d import PackPPO3DController
        dummy_env = _make_dummy_vec_env(pack_config)
        vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        model = PPO.load(str(model_path), env=None, device="auto")
        expected = 6 + pack_config.shape[0]
        if model.observation_space.shape[0] != expected:
            print(f"  PPO obs_dim mismatch ({model.observation_space.shape[0]} != {expected}) — retrain first.")
            return None
        print(f"  Loaded PPO: {model_path}")
        return PackPPO3DController(model=model, vec_normalize=vec_normalize, name="PPO (best)")
    except Exception as exc:
        print(f"  PPO load failed: {exc}")
        return None


def load_controllers(pack_config: PackConfig) -> List:
    controllers = []

    controllers.append(PI3D(
        pack_config=pack_config,
        target_temp_c=pack_config.target_temp_c,
        kp=0.30, ki=0.001, bias=0.30,
        imbalance_gain=0.08,
        integral_limit=50.0,
        name="PI (tuned)",
    ))

    sac = _load_sac(pack_config)
    if sac is not None:
        controllers.append(sac)

    ppo = _load_ppo(pack_config)
    if ppo is not None:
        controllers.append(ppo)

    return controllers


# ---------------------------------------------------------------------------
# Episode runner — captures all T3d frames, picks 4 evenly-spaced snapshots
# ---------------------------------------------------------------------------

def run_episode(
    controller,
    pack_config: PackConfig,
    cell_config: CellConfig,
    profile_name: str,
    seed: int = 42,
    n_snapshots: int = 4,
) -> Tuple[pd.DataFrame, List[np.ndarray], List[float]]:
    env = BatteryPackThermalEnv3D(
        cell_config=cell_config,
        pack_config=pack_config,
        heat_profile=make_3d_profile(profile_name),
        seed=seed,
    )

    controller.reset()
    obs, info = env.reset(seed=seed)

    terminated = truncated = False
    rows: List[Dict] = []
    all_T3d: List[np.ndarray] = []
    all_times: List[float] = []

    while not (terminated or truncated):
        action = controller.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        t = float(info["time_s"])
        rows.append({
            "time_s": t,
            "T_max_C": info["T_max"],
            "T_avg_C": info["T_avg"],
            "T_min_C": info["T_min"],
            "T_gradient_C": info["T_gradient"],
            "n_above_safe": info["n_cells_above_safe"],
            "reward": reward,
        })
        all_T3d.append(info["temperatures_3d"].copy())
        all_times.append(t)

    df = pd.DataFrame(rows)
    n = len(all_T3d)

    # Evenly-spaced snapshot indices (always include final frame)
    indices = sorted(set(
        [0] + [int(round(i * (n - 1) / (n_snapshots - 1))) for i in range(n_snapshots)]
    ))[:n_snapshots]

    return df, [all_T3d[i] for i in indices], [all_times[i] for i in indices]


# ---------------------------------------------------------------------------
# PyVista screenshot saver
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_").replace("(", "").replace(")", "")
        .replace("/", "_").replace(".", "").replace("-", "_")
    )


def save_pyvista_screenshots(
    controller_name: str,
    T3d_snaps: List[np.ndarray],
    snap_times: List[float],
    cell_config: CellConfig,
    pack_config: PackConfig,
    output_dir: Path,
) -> List[Path]:
    if not PYVISTA_AVAILABLE:
        print("  PyVista not available — skipping 3D screenshots.")
        return []

    slug = _slug(controller_name)
    vmin = float(pack_config.ambient_temp_c)
    vmax = float(pack_config.safe_temp_c)
    paths: List[Path] = []

    for T3d, t in zip(T3d_snaps, snap_times):
        png = output_dir / f"pyvista_{slug}_t{int(t):04d}s.png"
        render_temperature_field(
            T=T3d,
            cell=cell_config,
            pack=pack_config,
            vmin=vmin,
            vmax=vmax,
            off_screen=True,
            title=f"{controller_name}  t={t:.0f}s   T_max={T3d.max():.1f}°C   ΔT={T3d.max()-T3d.min():.2f}°C",
            screenshot_path=png,
        )
        paths.append(png)

    return paths


# ---------------------------------------------------------------------------
# Summary matplotlib figure
# ---------------------------------------------------------------------------

_CTRL_COLORS     = ["#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
_CTRL_LINESTYLES = ["-.",      "-",       "--",      ":"]
_CTRL_LINEWIDTHS = [1.8,       2.5,       2.2,       1.8]


def build_summary_figure(
    controller_data: Dict[str, Dict],
    pack_config: PackConfig,
    profile_name: str,
    output_dir: Path,
    n_snapshots: int = 4,
) -> Path:
    n_ctrl = len(controller_data)
    fig_height = 4.5 + 3.2 * n_ctrl
    fig = plt.figure(figsize=(5 * n_snapshots, fig_height))
    fig.suptitle(
        f"3D Battery Pack — Controller Comparison  ({profile_name} profile)",
        fontsize=13, fontweight="bold",
    )

    # GridSpec: 2 signal rows (T_max, T_gradient) + n_ctrl image rows
    n_rows = 2 + n_ctrl
    gs = gridspec.GridSpec(
        n_rows, n_snapshots,
        figure=fig,
        hspace=0.50, wspace=0.12,
        height_ratios=[2.2, 1.4] + [2.0] * n_ctrl,
    )

    # ---- Row 0: T_max time series ----
    ax_tmax = fig.add_subplot(gs[0, :])
    for idx, (ctrl_name, data) in enumerate(controller_data.items()):
        df = data["df"]
        c  = _CTRL_COLORS[idx % len(_CTRL_COLORS)]
        ls = _CTRL_LINESTYLES[idx % len(_CTRL_LINESTYLES)]
        lw = _CTRL_LINEWIDTHS[idx % len(_CTRL_LINEWIDTHS)]
        ax_tmax.plot(df["time_s"], df["T_max_C"], color=c, ls=ls, lw=lw,
                     label=f"{ctrl_name} T_max")
        ax_tmax.plot(df["time_s"], df["T_avg_C"], color=c, ls=":", lw=1.0, alpha=0.5)

    ax_tmax.axhline(pack_config.target_temp_c, ls=":", lw=1.2, color="green",
                    label=f"Target {pack_config.target_temp_c:.0f}°C")
    ax_tmax.axhline(pack_config.safe_temp_c, ls="--", lw=1.0, color="orange",
                    label=f"Safe {pack_config.safe_temp_c:.0f}°C")
    ax_tmax.axhline(pack_config.critical_temp_c, ls="-.", lw=0.8, color="red",
                    label=f"Critical {pack_config.critical_temp_c:.0f}°C")
    ax_tmax.set_ylabel("Temperature (°C)", fontsize=10)
    ax_tmax.set_title("Pack max temperature (solid) and average (dotted)", fontsize=10)
    ax_tmax.legend(fontsize=8, loc="upper left", ncol=2)
    ax_tmax.grid(True, alpha=0.25)

    # ---- Row 1: T_gradient time series ----
    ax_grad = fig.add_subplot(gs[1, :])
    for idx, (ctrl_name, data) in enumerate(controller_data.items()):
        df = data["df"]
        c  = _CTRL_COLORS[idx % len(_CTRL_COLORS)]
        ls = _CTRL_LINESTYLES[idx % len(_CTRL_LINESTYLES)]
        lw = _CTRL_LINEWIDTHS[idx % len(_CTRL_LINEWIDTHS)]
        ax_grad.plot(df["time_s"], df["T_gradient_C"], color=c, ls=ls, lw=lw,
                     label=ctrl_name)

    ax_grad.set_ylabel("ΔT pack (°C)", fontsize=10)
    ax_grad.set_xlabel("Time (s)", fontsize=10)
    ax_grad.set_title("Pack temperature gradient (T_max − T_min)", fontsize=10)
    ax_grad.legend(fontsize=8, loc="upper right")
    ax_grad.grid(True, alpha=0.25)

    # ---- Rows 2+: PyVista image grid ----
    for row_idx, (ctrl_name, data) in enumerate(controller_data.items()):
        png_paths = data["png_paths"]
        snap_times = data["snap_times"]

        for col_idx in range(n_snapshots):
            ax = fig.add_subplot(gs[2 + row_idx, col_idx])
            ax.axis("off")

            if col_idx < len(png_paths) and png_paths[col_idx].exists():
                import matplotlib.image as mpimg
                img = mpimg.imread(str(png_paths[col_idx]))
                ax.imshow(img, aspect="auto")
                t_label = f"t={snap_times[col_idx]:.0f}s"
                ax.set_title(t_label, fontsize=8, pad=2)
            else:
                # PyVista unavailable — show temperature heatmap fallback
                if col_idx < len(data.get("T3d_snaps", [])):
                    T3d = data["T3d_snaps"][col_idx]
                    layer = T3d[:, :, 0]  # show z=0 layer
                    im = ax.imshow(
                        layer, origin="lower", cmap="RdYlBu_r",
                        vmin=pack_config.ambient_temp_c,
                        vmax=pack_config.safe_temp_c,
                        aspect="auto",
                    )
                    t_label = f"t={snap_times[col_idx]:.0f}s (z=0 layer)"
                    ax.set_title(t_label, fontsize=7, pad=2)
                    ax.axis("on")
                    ax.tick_params(labelsize=5)
                else:
                    ax.text(0.5, 0.5, "PyVista\nnot available",
                            ha="center", va="center", transform=ax.transAxes, fontsize=8)

            if col_idx == 0:
                ax.set_ylabel(ctrl_name, fontsize=9, rotation=90, labelpad=6)
                ax.yaxis.set_visible(True)
                ax.yaxis.set_label_position("left")

    out_path = output_dir / "3d_controller_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Per-controller summary table
# ---------------------------------------------------------------------------

def print_summary_table(controller_data: Dict[str, Dict], pack_config: PackConfig) -> None:
    print("\n" + "=" * 72)
    print(f"  {'Controller':<24} {'T_max_peak':>10} {'T_grad_max':>10} "
          f"{'Above_safe':>10} {'Reward':>10}")
    print("=" * 72)
    for ctrl_name, data in controller_data.items():
        df = data["df"]
        t_above = int((df["T_max_C"] > pack_config.safe_temp_c).sum())
        print(
            f"  {ctrl_name:<24} "
            f"{df['T_max_C'].max():>10.2f}°C "
            f"{df['T_gradient_C'].max():>10.2f}°C "
            f"{t_above:>9d}s "
            f"{df['reward'].sum():>10.1f}"
        )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D post-training controller visualization")
    p.add_argument("--output-dir", type=str, default="outputs/3d_viz",
                   help="Directory for screenshots and figure (default: outputs/3d_viz)")
    p.add_argument("--profile", type=str, default="NonuniformStep",
                   choices=["UniformConstant", "NonuniformStep", "PulsedHotspot", "RandomNonuniform"],
                   help="Heat profile for visualization (default: NonuniformStep)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-snapshots", type=int, default=4,
                   help="Number of 3D snapshots per controller (default: 4)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_config = CellConfig()
    pack_config = _make_pack_config()

    print("\n" + "=" * 60)
    print("  3D Pack Controller Visualization")
    print(f"  Profile:     {args.profile}")
    print(f"  Pack shape:  {pack_config.shape}  ({int(np.prod(pack_config.shape))} cells)")
    print(f"  Output dir:  {output_dir}")
    print(f"  PyVista:     {'available' if PYVISTA_AVAILABLE else 'NOT installed — using matplotlib fallback'}")
    print("=" * 60)

    print("\nLoading controllers ...")
    controllers = load_controllers(pack_config)
    print(f"  Controllers: {[c.name for c in controllers]}\n")

    controller_data: Dict[str, Dict] = {}

    for ctrl in controllers:
        print(f"Running episode: {ctrl.name} ...")
        df, T3d_snaps, snap_times = run_episode(
            controller=ctrl,
            pack_config=pack_config,
            cell_config=cell_config,
            profile_name=args.profile,
            seed=args.seed,
            n_snapshots=args.n_snapshots,
        )

        print(f"  T_max peak: {df['T_max_C'].max():.2f}°C   "
              f"T_grad max: {df['T_gradient_C'].max():.2f}°C   "
              f"reward: {df['reward'].sum():.1f}")

        png_paths = save_pyvista_screenshots(
            controller_name=ctrl.name,
            T3d_snaps=T3d_snaps,
            snap_times=snap_times,
            cell_config=cell_config,
            pack_config=pack_config,
            output_dir=output_dir,
        )

        controller_data[ctrl.name] = {
            "df": df,
            "T3d_snaps": T3d_snaps,
            "snap_times": snap_times,
            "png_paths": png_paths,
        }

    print("\nBuilding summary figure ...")
    fig_path = build_summary_figure(
        controller_data=controller_data,
        pack_config=pack_config,
        profile_name=args.profile,
        output_dir=output_dir,
        n_snapshots=args.n_snapshots,
    )

    print_summary_table(controller_data, pack_config)

    print("Outputs:")
    if PYVISTA_AVAILABLE:
        for data in controller_data.values():
            for p in data["png_paths"]:
                print(f"  {p}")
    print(f"  {fig_path}")


if __name__ == "__main__":
    main()
