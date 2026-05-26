"""
scripts/visualize_pack_controller_cooling_3d.py

Detailed single-controller visualization for the 3D cell-resolved battery pack.

Run from project root:
    python -m scripts.visualize_pack_controller_cooling_3d

Outputs:
    outputs/3d_visualize_<controller>_<profile>.png
    outputs/3d_visualize_<controller>_<profile>.csv

Supported controllers:
    no_cooling
    constant_05
    constant_10
    bang_bang
    proportional
    pi_tuned
    ppo_best
    ppo_final
    sac_best
    sac_final

Supported profiles:
    UniformConstant
    NonuniformStep
    PulsedHotspot
    RandomNonuniform
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

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
from scripts.compare_pack_baselines_3d import (
    NoCooling3D,
    ConstantCooling3D,
    BangBang3D,
    Proportional3D,
    PI3D,
)

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from training.evaluate_pack_ppo_3d import load_ppo_controller, _make_dummy_vec_env
    from training.evaluate_pack_rl_3d import PackSAC3DController
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class Controller3DLike(Protocol):
    name: str
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------

def make_controller(key: str, pack_config: PackConfig) -> Controller3DLike:
    key = key.lower().strip()
    if key == "no_cooling":
        return NoCooling3D()
    if key == "constant_05":
        return ConstantCooling3D(cooling_level=0.5)
    if key == "constant_10":
        return ConstantCooling3D(cooling_level=1.0)
    if key == "bang_bang":
        return BangBang3D(pack_config=pack_config, target_temp_c=pack_config.target_temp_c)
    if key == "proportional":
        return Proportional3D(
            pack_config=pack_config,
            target_temp_c=pack_config.target_temp_c,
            kp=0.10, bias=0.15, imbalance_gain=0.04,
        )
    if key == "pi_tuned":
        return PI3D(
            pack_config=pack_config,
            target_temp_c=pack_config.target_temp_c,
            kp=0.30, ki=0.001, bias=0.30, imbalance_gain=0.08,
            integral_limit=50.0, name="PI tuned (T_max)",
        )
    if key in ("ppo_best", "ppo_final"):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required for PPO visualization.")
        model_dir = PROJECT_ROOT / "models" / "ppo_pack_3d"
        file_name = "best_model.zip" if key == "ppo_best" else "final_model.zip"
        label = "PPO best model (3D)" if key == "ppo_best" else "PPO final model (3D)"
        return load_ppo_controller(model_dir / file_name, model_dir / "vec_normalize.pkl", pack_config, label)
    if key in ("sac_best", "sac_final"):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required for SAC visualization.")
        model_dir = PROJECT_ROOT / "models" / "sac_pack_3d"
        file_name = "best_model.zip" if key == "sac_best" else "final_model.zip"
        label = "SAC best model (3D)" if key == "sac_best" else "SAC final model (3D)"
        model = SAC.load(str(model_dir / file_name), env=None, device="auto")
        return PackSAC3DController(model=model, name=label)
    raise KeyError(
        f"Unknown controller '{key}'. Choose from: no_cooling, constant_05, constant_10, "
        "bang_bang, proportional, pi_tuned, ppo_best, ppo_final, sac_best, sac_final"
    )


# ---------------------------------------------------------------------------
# Simulation — collects full 3D temperature snapshots
# ---------------------------------------------------------------------------

def run_episode(
    controller: Controller3DLike,
    profile_name: str,
    cell_config: CellConfig,
    pack_config: PackConfig,
    seed: int = 7,
    snapshot_stride: int = 60,
) -> Tuple[pd.DataFrame, List[np.ndarray], List[float]]:
    """
    Run one episode and return:
        df              — step-by-step scalar metrics
        T3d_snapshots   — list of 3D temperature arrays (shape Nx×Ny×Nz),
                          sampled every snapshot_stride seconds
        snapshot_times  — corresponding simulation times
    """
    env = BatteryPackThermalEnv3D(
        cell_config=cell_config,
        pack_config=pack_config,
        heat_profile=make_3d_profile(profile_name),
        seed=seed,
    )

    controller.reset()
    obs, info = env.reset(seed=seed, options={"randomize": False})

    terminated = truncated = False
    total_reward = 0.0
    rows: List[Dict] = []
    T3d_snapshots: List[np.ndarray] = []
    snapshot_times: List[float] = []
    last_snap_t = -snapshot_stride

    while not (terminated or truncated):
        action = controller.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        t = float(info["time_s"])
        rows.append({
            "time_s": t,
            "T_max_C": info["T_max"],
            "T_avg_C": info["T_avg"],
            "T_min_C": info["T_min"],
            "T_gradient_C": info["T_gradient"],
            "n_cells_above_safe": info["n_cells_above_safe"],
            "u": float(action.reshape(-1)[0]),
            "q_total_W": float(np.sum(info.get("q_gen_total", 0))),
            "reward": reward,
            "cumulative_reward": total_reward,
        })

        if t - last_snap_t >= snapshot_stride:
            T3d_snapshots.append(info["temperatures_3d"].copy())
            snapshot_times.append(t)
            last_snap_t = t

    # Always include final frame
    if len(snapshot_times) == 0 or snapshot_times[-1] != rows[-1]["time_s"]:
        T3d_snapshots.append(info["temperatures_3d"].copy())
        snapshot_times.append(rows[-1]["time_s"])

    return pd.DataFrame(rows), T3d_snapshots, snapshot_times


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("=", "")
        .replace(".", "")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def _draw_layer_heatmaps(
    ax_list: List,
    T3d: np.ndarray,
    pack_config: PackConfig,
    title_prefix: str,
    vmin: float,
    vmax: float,
) -> None:
    """Draw one heatmap per z-layer onto the supplied axes."""
    Nz = pack_config.shape[2]
    for k, ax in enumerate(ax_list):
        layer = T3d[:, :, k]
        im = ax.imshow(
            layer, origin="lower", aspect="auto",
            vmin=vmin, vmax=vmax, cmap="hot",
        )
        ax.set_title(f"{title_prefix} z={k}", fontsize=8)
        ax.set_xlabel("y", fontsize=7)
        ax.set_ylabel("x", fontsize=7)
        ax.tick_params(labelsize=6)
        # Annotate each cell with its temperature
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                ax.text(j, i, f"{layer[i, j]:.1f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if layer[i, j] > (vmin + vmax) * 0.6 else "black")
        return im  # return last im for colorbar


# ---------------------------------------------------------------------------
# Visualization — 7-panel static figure
# ---------------------------------------------------------------------------

def plot_3d_controller(
    df: pd.DataFrame,
    T3d_snapshots: List[np.ndarray],
    snapshot_times: List[float],
    pack_config: PackConfig,
    output_path: Path,
) -> None:
    controller_name = str(df["controller"].iloc[0]) if "controller" in df.columns else "Unknown"
    profile_name = str(df["profile"].iloc[0]) if "profile" in df.columns else "Unknown"
    time = df["time_s"].to_numpy()
    Nz = pack_config.shape[2]

    # ---- pick 4 evenly-spaced snapshots for the heatmap panels ----
    n_snaps = len(T3d_snapshots)
    snap_indices = [0,
                    max(1, n_snaps // 3),
                    max(2, 2 * n_snaps // 3),
                    n_snaps - 1]
    snap_indices = sorted(set(snap_indices))

    # ---- global color scale for heatmaps ----
    all_T = np.stack(T3d_snapshots, axis=0)
    vmin_h = float(all_T.min())
    vmax_h = float(all_T.max())
    if vmax_h - vmin_h < 3:
        vmax_h = vmin_h + 3

    # ---- build figure layout ----
    # Top block: 6 signal panels (3 cols × 2 rows)
    # Bottom block: heatmap grid  (len(snap_indices) × Nz)
    n_snap_cols = len(snap_indices)
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"3D pack — {controller_name} on {profile_name}",
        fontsize=13, fontweight="bold",
    )

    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.2, 1.6], hspace=0.45)

    # Signal panels (top half)
    sig = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=outer[0], hspace=0.55, wspace=0.35)
    ax_heat = fig.add_subplot(sig[0, 0])
    ax_cool = fig.add_subplot(sig[0, 1])
    ax_temp = fig.add_subplot(sig[1, 0])
    ax_grad = fig.add_subplot(sig[1, 1])
    ax_safe = fig.add_subplot(sig[2, 0])
    ax_rew  = fig.add_subplot(sig[2, 1])

    # 1. Heat generation
    if "q_total_W" in df.columns and df["q_total_W"].max() > 0:
        ax_heat.plot(time, df["q_total_W"], linewidth=1.6)
    ax_heat.set_ylabel("Heat (W)")
    ax_heat.set_title("Total pack heat generation")
    ax_heat.grid(True, alpha=0.25)

    # 2. Cooling command
    ax_cool.plot(time, df["u"], linewidth=1.6, color="steelblue")
    ax_cool.set_ylim(-0.05, 1.05)
    ax_cool.set_ylabel("u")
    ax_cool.set_title("Cooling command")
    ax_cool.grid(True, alpha=0.25)

    # 3. Temperature envelope
    ax_temp.plot(time, df["T_max_C"], linewidth=1.8, label="T_max")
    ax_temp.plot(time, df["T_avg_C"], linewidth=1.3, label="T_avg")
    ax_temp.plot(time, df["T_min_C"], linewidth=1.0, linestyle="--", label="T_min")
    ax_temp.axhline(pack_config.target_temp_c, linestyle=":", linewidth=1.0, color="green",
                    label=f"Target {pack_config.target_temp_c:.0f}°C")
    ax_temp.axhline(pack_config.safe_temp_c, linestyle="--", linewidth=1.0, color="orange",
                    label=f"Safe {pack_config.safe_temp_c:.0f}°C")
    ax_temp.axhline(pack_config.critical_temp_c, linestyle="-.", linewidth=0.8, color="red",
                    label=f"Critical {pack_config.critical_temp_c:.0f}°C")
    ax_temp.set_ylabel("Temp (°C)")
    ax_temp.set_title("Temperature envelope")
    ax_temp.legend(fontsize=7, loc="upper left")
    ax_temp.grid(True, alpha=0.25)

    # 4. Gradient
    ax_grad.plot(time, df["T_gradient_C"], linewidth=1.6, color="darkorange")
    ax_grad.set_ylabel("ΔT (°C)")
    ax_grad.set_title("Pack temperature gradient (T_max − T_min)")
    ax_grad.grid(True, alpha=0.25)

    # 5. Cells above safe limit
    ax_safe.plot(time, df["n_cells_above_safe"], linewidth=1.6, color="red")
    ax_safe.set_ylabel("# cells")
    ax_safe.set_xlabel("Time (s)")
    ax_safe.set_title("Cells above safe limit")
    ax_safe.grid(True, alpha=0.25)

    # 6. Cumulative reward
    ax_rew.plot(time, df["cumulative_reward"], linewidth=1.6, color="purple")
    ax_rew.set_ylabel("Cumulative reward")
    ax_rew.set_xlabel("Time (s)")
    ax_rew.set_title("Cumulative reward")
    ax_rew.grid(True, alpha=0.25)

    # Heatmap panels (bottom half)
    hmap = gridspec.GridSpecFromSubplotSpec(
        Nz, n_snap_cols,
        subplot_spec=outer[1],
        hspace=0.5, wspace=0.35,
    )

    last_im = None
    for col_idx, snap_idx in enumerate(snap_indices):
        T3d = T3d_snapshots[snap_idx]
        t_label = f"t={snapshot_times[snap_idx]:.0f}s"
        for k in range(Nz):
            ax = fig.add_subplot(hmap[k, col_idx])
            layer = T3d[:, :, k]
            im = ax.imshow(
                layer, origin="lower", aspect="auto",
                vmin=vmin_h, vmax=vmax_h, cmap="hot",
            )
            last_im = im
            ax.set_title(f"{t_label} z={k}", fontsize=8)
            ax.set_xlabel("y", fontsize=7)
            if col_idx == 0:
                ax.set_ylabel("x", fontsize=7)
            ax.tick_params(labelsize=6)
            for i in range(layer.shape[0]):
                for j in range(layer.shape[1]):
                    ax.text(j, i, f"{layer[i,j]:.1f}", ha="center", va="center", fontsize=5.5,
                            color="white" if layer[i,j] > (vmin_h + vmax_h) * 0.6 else "black")

    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.04, 0.012, 0.28])
        fig.colorbar(last_im, cax=cbar_ax, label="Temperature (°C)")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary print
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, pack_config: PackConfig) -> None:
    n_above = int((df["T_max_C"] > pack_config.safe_temp_c).sum())
    print(f"\n3D pack visualization summary")
    print(f"  T_max peak:          {df['T_max_C'].max():.2f} °C")
    print(f"  T_avg mean:          {df['T_avg_C'].mean():.2f} °C")
    print(f"  T_gradient max:      {df['T_gradient_C'].max():.2f} °C")
    print(f"  Time above safe:     {n_above:.0f} s")
    print(f"  Mean cooling u:      {df['u'].mean():.3f}")
    print(f"  Total reward:        {df['cumulative_reward'].iloc[-1]:.2f}")


# ---------------------------------------------------------------------------
# CSV export  (adds controller + profile columns)
# ---------------------------------------------------------------------------

def save_csv(
    df: pd.DataFrame,
    controller_name: str,
    profile_name: str,
    path: Path,
) -> None:
    out = df.copy()
    out.insert(0, "controller", controller_name)
    out.insert(1, "profile", profile_name)
    out.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Change these two to explore different cases ----
    controller_key = "pi_tuned"
    profile_name   = "NonuniformStep"
    # Good comparison cases:
    #   controller_key = "no_cooling"   profile_name = "NonuniformStep"
    #   controller_key = "pi_tuned"     profile_name = "NonuniformStep"
    #   controller_key = "bang_bang"    profile_name = "PulsedHotspot"
    #   controller_key = "proportional" profile_name = "RandomNonuniform"
    #   controller_key = "ppo_best"     profile_name = "NonuniformStep"
    # -------------------------------------------------------

    cell_config = CellConfig()
    pack_config = PackConfig(shape=(4, 3, 2))
    controller = make_controller(controller_key, pack_config)

    print(f"Running 3D pack: {controller.name} on {profile_name} ...")
    df, T3d_snaps, snap_times = run_episode(
        controller=controller,
        profile_name=profile_name,
        cell_config=cell_config,
        pack_config=pack_config,
        seed=7,
        snapshot_stride=60,
    )

    slug_c = sanitize(controller.name)
    slug_p = sanitize(profile_name)

    png_path = output_dir / f"3d_visualize_{slug_c}_{slug_p}.png"
    csv_path = output_dir / f"3d_visualize_{slug_c}_{slug_p}.csv"

    plot_3d_controller(
        df=df,
        T3d_snapshots=T3d_snaps,
        snapshot_times=snap_times,
        pack_config=pack_config,
        output_path=png_path,
    )
    save_csv(df, controller.name, profile_name, csv_path)
    print_summary(df, pack_config)

    print(f"\nSaved:")
    print(f"  {png_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
