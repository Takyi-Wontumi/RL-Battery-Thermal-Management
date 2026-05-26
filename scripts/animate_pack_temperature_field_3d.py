"""
scripts/animate_pack_temperature_field_3d.py

Animated GIF of the 3D battery pack temperature field evolving over time.

Each animation frame shows one heatmap per z-layer (left side) alongside live
signal traces for T_max, T_avg, T_gradient, and the cooling command (right side).

Run from project root — pick one or more controllers and an optional profile:

    python -m scripts.animate_pack_temperature_field_3d --PI
    python -m scripts.animate_pack_temperature_field_3d --SAC --PPO
    python -m scripts.animate_pack_temperature_field_3d --PI --SAC --PPO --bang-bang
    python -m scripts.animate_pack_temperature_field_3d --PI --profile PulsedHotspot

Outputs (one set per controller):
    outputs/3d_animate_<controller>_<profile>.gif
    outputs/3d_animate_<controller>_<profile>_final_frame.png
    outputs/3d_animate_<controller>_<profile>.csv

Supported --profile values:
    UniformConstant  NonuniformStep  PulsedHotspot  RandomNonuniform
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from configs.pack_config import CellConfig, PackConfig
from envs.battery_pack_thermal_env_3d import (
    BatteryPackThermalEnv3D,
    make_3d_profile,
)
from scripts.compare_pack_baselines_3d import (
    NoCooling3D,
    ConstantCooling3D,
    BangBang3D,
    Proportional3D,
    PI3D,
)
from scripts.visualize_pack_controller_cooling_3d import (
    make_controller,
    sanitize,
)

try:
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Simulation — captures 3D frames at every stride-th step
# ---------------------------------------------------------------------------

def run_episode_with_frames(
    controller,
    profile_name: str,
    cell_config: CellConfig,
    pack_config: PackConfig,
    seed: int = 7,
    stride: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[float]]:
    """
    Returns (time, T_max, T_avg, T_gradient, u, q_total,
             T3d_frames, frame_times).

    T3d_frames: list of (Nx, Ny, Nz) arrays sampled every `stride` steps.
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
    step = 0
    times, T_maxs, T_avgs, T_grads, us, qs = [], [], [], [], [], []
    T3d_frames: List[np.ndarray] = []
    frame_times: List[float] = []

    while not (terminated or truncated):
        action = controller.act(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        step += 1

        t = float(info["time_s"])
        times.append(t)
        T_maxs.append(info["T_max"])
        T_avgs.append(info["T_avg"])
        T_grads.append(info["T_gradient"])
        us.append(float(action.reshape(-1)[0]))
        # q_total not in info — use log from env
        qs.append(float(np.sum(env.episode_log["q_gen_total"][-1])
                        if env.episode_log["q_gen_total"] else 0.0))

        if step % stride == 0:
            T3d_frames.append(info["temperatures_3d"].copy())
            frame_times.append(t)

    # Always capture final state
    if not frame_times or frame_times[-1] != times[-1]:
        T3d_frames.append(info["temperatures_3d"].copy())
        frame_times.append(times[-1])

    return (
        np.array(times),
        np.array(T_maxs),
        np.array(T_avgs),
        np.array(T_grads),
        np.array(us),
        np.array(qs),
        T3d_frames,
        frame_times,
    )


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate_3d_pack(
    controller_name: str,
    profile_name: str,
    time: np.ndarray,
    T_max: np.ndarray,
    T_avg: np.ndarray,
    T_gradient: np.ndarray,
    u: np.ndarray,
    T3d_frames: List[np.ndarray],
    frame_times: List[float],
    pack_config: PackConfig,
    output_gif: Path,
    output_png: Path,
    fps: int = 10,
) -> None:
    Nz = pack_config.shape[2]
    n_frames = len(T3d_frames)

    # Fixed color scale anchored to physics: blue=ambient, red=safe limit.
    # This keeps the colormap consistent across controllers and time.
    vmin = pack_config.ambient_temp_c
    vmax = pack_config.safe_temp_c

    # ---- Figure layout ----
    # Left: Nz heatmap panels stacked vertically
    # Right: 3 signal panels (temp, gradient, cooling)
    fig = plt.figure(figsize=(6 + Nz * 3.5, 9))
    fig.suptitle(
        f"3D pack temperature field — {controller_name} on {profile_name}",
        fontsize=12, fontweight="bold",
    )

    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[Nz * 1.5, 2.0], wspace=0.35)

    left_gs = gridspec.GridSpecFromSubplotSpec(Nz, 1, subplot_spec=outer[0], hspace=0.55)
    right_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.55)

    layer_axes = [fig.add_subplot(left_gs[k]) for k in range(Nz)]
    ax_temp = fig.add_subplot(right_gs[0])
    ax_grad = fig.add_subplot(right_gs[1])
    ax_u    = fig.add_subplot(right_gs[2])

    # ---- Static traces on signal panels ----
    ax_temp.plot(time, T_max, linewidth=1.6, label="T_max")
    ax_temp.plot(time, T_avg, linewidth=1.2, label="T_avg")
    ax_temp.axhline(pack_config.target_temp_c, linestyle=":", linewidth=0.9, color="green",
                    label=f"Target {pack_config.target_temp_c:.0f}°C")
    ax_temp.axhline(pack_config.safe_temp_c, linestyle="--", linewidth=0.9, color="orange",
                    label=f"Safe {pack_config.safe_temp_c:.0f}°C")
    ax_temp.set_ylabel("Temp (°C)", fontsize=9)
    ax_temp.legend(fontsize=7, loc="upper left")
    ax_temp.grid(True, alpha=0.25)

    ax_grad.plot(time, T_gradient, linewidth=1.4, color="darkorange")
    ax_grad.set_ylabel("ΔT (°C)", fontsize=9)
    ax_grad.set_title("Gradient T_max − T_min", fontsize=9)
    ax_grad.grid(True, alpha=0.25)

    ax_u.plot(time, u, linewidth=1.4, color="steelblue")
    ax_u.set_ylim(-0.05, 1.05)
    ax_u.set_ylabel("u", fontsize=9)
    ax_u.set_xlabel("Time (s)", fontsize=9)
    ax_u.set_title("Cooling command", fontsize=9)
    ax_u.grid(True, alpha=0.25)

    # Moving vertical time-marker on all signal panels
    vlines = [
        ax_temp.axvline(frame_times[0], linewidth=1.2, color="red", alpha=0.7),
        ax_grad.axvline(frame_times[0], linewidth=1.2, color="red", alpha=0.7),
        ax_u.axvline(frame_times[0], linewidth=1.2, color="red", alpha=0.7),
    ]

    # ---- Initial heatmap images ----
    layer_images = []
    colorbars_added = False

    for k, ax in enumerate(layer_axes):
        layer = T3d_frames[0][:, :, k]
        im = ax.imshow(
            layer, origin="lower", aspect="auto",
            vmin=vmin, vmax=vmax, cmap="RdYlBu_r",
        )
        layer_images.append(im)
        ax.set_title(f"Layer z={k}", fontsize=9)
        ax.set_xlabel("y", fontsize=8)
        ax.set_ylabel("x", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.colorbar(layer_images[0], ax=layer_axes, fraction=0.025, pad=0.04,
                 label="Temperature (°C)")

    # ---- Cell annotation texts (reused across frames) ----
    cell_texts: List[List] = []
    Nx, Ny = pack_config.shape[0], pack_config.shape[1]
    for k, ax in enumerate(layer_axes):
        texts_k = []
        for i in range(Nx):
            for j in range(Ny):
                t_obj = ax.text(
                    j, i, "",
                    ha="center", va="center", fontsize=6,
                    color="white",
                )
                texts_k.append((i, j, t_obj))
        cell_texts.append(texts_k)

    # ---- Update function ----
    def update(frame_idx: int):
        T3d = T3d_frames[frame_idx]
        ft = frame_times[frame_idx]

        for k, (ax, im) in enumerate(zip(layer_axes, layer_images)):
            layer = T3d[:, :, k]
            im.set_data(layer)

            for (i, j, t_obj) in cell_texts[k]:
                val = layer[i, j]
                t_obj.set_text(f"{val:.1f}")
                t_obj.set_color("white" if val > (vmin + vmax) * 0.58 else "black")

            ax.set_title(
                f"Layer z={k} | t={ft:.0f}s | T_max={T3d.max():.1f}°C",
                fontsize=8,
            )

        for vl in vlines:
            vl.set_xdata([ft, ft])

        return layer_images + vlines

    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=int(1000 / fps), blit=False,
    )
    anim.save(output_gif, writer=PillowWriter(fps=fps))

    # Save final frame as PNG
    update(n_frames - 1)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(
    time: np.ndarray,
    T_max: np.ndarray,
    T_avg: np.ndarray,
    T_gradient: np.ndarray,
    u: np.ndarray,
    controller_name: str,
    profile_name: str,
    path: Path,
) -> None:
    pd.DataFrame({
        "controller": controller_name,
        "profile": profile_name,
        "time_s": time,
        "T_max_C": T_max,
        "T_avg_C": T_avg,
        "T_gradient_C": T_gradient,
        "u": u,
    }).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    # Friendly flag → internal key mapping
    _ALIASES = {
        "pi":          "pi_tuned",
        "sac":         "sac_best",
        "sac_best":    "sac_best",
        "sac_final":   "sac_final",
        "ppo":         "ppo_best",
        "ppo_best":    "ppo_best",
        "ppo_final":   "ppo_final",
        "bang_bang":   "bang_bang",
        "bang-bang":   "bang_bang",
        "proportional": "proportional",
        "no_cooling":  "no_cooling",
        "constant_05": "constant_05",
        "constant_10": "constant_10",
    }

    parser = argparse.ArgumentParser(
        description="Animate 3D battery pack temperature field for one or more controllers.",
    )
    parser.add_argument("--PI",        dest="controllers", action="append_const", const="pi_tuned",
                        help="PI tuned controller")
    parser.add_argument("--SAC",       dest="controllers", action="append_const", const="sac_best",
                        help="SAC best model")
    parser.add_argument("--PPO",       dest="controllers", action="append_const", const="ppo_best",
                        help="PPO best model")
    parser.add_argument("--bang-bang", dest="controllers", action="append_const", const="bang_bang",
                        help="Bang-bang controller")
    parser.add_argument("--proportional", dest="controllers", action="append_const", const="proportional",
                        help="Proportional controller")
    parser.add_argument("--no-cooling", dest="controllers", action="append_const", const="no_cooling",
                        help="No cooling baseline")
    parser.add_argument("--constant-1", dest="controllers", action="append_const", const="constant_10",
                        help="Constant full cooling (u=1.0)")
    parser.add_argument("--constant-05", dest="controllers", action="append_const", const="constant_05",
                        help="Constant half cooling (u=0.5)")
    parser.add_argument(
        "--controller", dest="extra_controllers", action="append", default=[],
        metavar="KEY",
        help=(
            "Any controller key directly: no_cooling, constant_05, constant_10, "
            "bang_bang, proportional, pi_tuned, ppo_best, ppo_final, sac_best, sac_final"
        ),
    )
    parser.add_argument(
        "--profile", default="NonuniformStep",
        choices=["UniformConstant", "NonuniformStep", "PulsedHotspot", "RandomNonuniform"],
        help="Heat profile to simulate (default: NonuniformStep)",
    )
    parser.add_argument("--fps", type=int, default=10, help="Animation frames per second (default: 10)")
    parser.add_argument("--stride", type=int, default=10,
                        help="Simulation steps between animation frames (default: 10)")

    args = parser.parse_args()

    # Merge flag-based and --controller keys; fall back to pi_tuned if nothing specified
    keys = list(args.controllers or [])
    for k in args.extra_controllers:
        resolved = _ALIASES.get(k.lower().replace("-", "_"), k)
        keys.append(resolved)
    if not keys:
        keys = ["pi_tuned"]

    return keys, args.profile, args.fps, args.stride


def _run_one(
    controller_key: str,
    profile_name: str,
    fps: int,
    stride: int,
    output_dir: Path,
) -> None:
    cell_config = CellConfig()
    pack_config = PackConfig(shape=(4, 3, 2))

    controller = make_controller(controller_key, pack_config)

    print(f"\nRunning 3D pack: {controller.name} on {profile_name} ...")
    (time, T_max, T_avg, T_gradient, u, q_total,
     T3d_frames, frame_times) = run_episode_with_frames(
        controller=controller,
        profile_name=profile_name,
        cell_config=cell_config,
        pack_config=pack_config,
        seed=7,
        stride=stride,
    )

    print(f"  {len(T3d_frames)} animation frames from {len(time)} simulation steps")
    print(f"  T_max peak: {T_max.max():.2f} °C  |  T_gradient max: {T_gradient.max():.2f} °C")

    slug_c = sanitize(controller.name)
    slug_p = sanitize(profile_name)
    gif_path = output_dir / f"3d_animate_{slug_c}_{slug_p}.gif"
    png_path = output_dir / f"3d_animate_{slug_c}_{slug_p}_final_frame.png"
    csv_path = output_dir / f"3d_animate_{slug_c}_{slug_p}.csv"

    print(f"Generating animation ({len(T3d_frames)} frames) ...")
    animate_3d_pack(
        controller_name=controller.name,
        profile_name=profile_name,
        time=time,
        T_max=T_max,
        T_avg=T_avg,
        T_gradient=T_gradient,
        u=u,
        T3d_frames=T3d_frames,
        frame_times=frame_times,
        pack_config=pack_config,
        output_gif=gif_path,
        output_png=png_path,
        fps=fps,
    )

    save_csv(time, T_max, T_avg, T_gradient, u, controller.name, profile_name, csv_path)

    print(f"Saved:")
    print(f"  {gif_path}")
    print(f"  {png_path}")
    print(f"  {csv_path}")


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    controller_keys, profile_name, fps, stride = _parse_args()

    print(f"Profile: {profile_name}")
    print(f"Controllers: {controller_keys}")

    for key in controller_keys:
        try:
            _run_one(key, profile_name, fps, stride, output_dir)
        except (KeyError, FileNotFoundError) as exc:
            print(f"Skipping '{key}': {exc}")


if __name__ == "__main__":
    main()
