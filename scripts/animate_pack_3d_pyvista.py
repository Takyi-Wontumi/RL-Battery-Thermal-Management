"""
scripts/animate_pack_3d_pyvista.py

PyVista GIF animation of the 3D battery pack temperature field.

Unlike animate_pack_temperature_field_3d.py (which produces a matplotlib-based
animation with signal traces), this script renders the actual 3D cylinder pack
geometry using PyVista.  Each frame is a screenshot of the pack with cells
coloured by temperature on the RdYlBu_r scale.

The simulation, episode loop, and frame collection are delegated entirely to
the existing run_episode_with_frames() function in animate_pack_temperature_field_3d.py
so that no simulation logic is duplicated here.

Usage
-----
    # Run from project root:
    python -m scripts.animate_pack_3d_pyvista --PI
    python -m scripts.animate_pack_3d_pyvista --SAC --profile PulsedHotspot
    python -m scripts.animate_pack_3d_pyvista --PI --SAC --PPO --bang-bang
    python -m scripts.animate_pack_3d_pyvista --PI --stride 5 --fps 15

Outputs (one GIF per controller)
---------------------------------
    outputs/3d_pyvista_<controller>_<profile>.gif

Requirements
------------
    pip install pyvista
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.pack_config import CellConfig, PackConfig

from scripts.animate_pack_temperature_field_3d import run_episode_with_frames
from scripts.render_pack_3d_pyvista import save_temperature_gif
from scripts.visualize_pack_controller_cooling_3d import make_controller, sanitize


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Render a PyVista 3D GIF of the battery pack temperature field.",
    )
    parser.add_argument("--PI",           dest="controllers", action="append_const", const="pi_tuned",
                        help="PI tuned controller")
    parser.add_argument("--SAC",          dest="controllers", action="append_const", const="sac_best",
                        help="SAC best checkpoint")
    parser.add_argument("--SAC-final",    dest="controllers", action="append_const", const="sac_final",
                        help="SAC final model")
    parser.add_argument("--PPO",          dest="controllers", action="append_const", const="ppo_best",
                        help="PPO best checkpoint")
    parser.add_argument("--PPO-final",    dest="controllers", action="append_const", const="ppo_final",
                        help="PPO final model")
    parser.add_argument("--bang-bang",    dest="controllers", action="append_const", const="bang_bang",
                        help="Bang-bang controller")
    parser.add_argument("--proportional", dest="controllers", action="append_const", const="proportional",
                        help="Proportional controller")
    parser.add_argument("--no-cooling",   dest="controllers", action="append_const", const="no_cooling",
                        help="No cooling baseline")
    parser.add_argument("--constant-05",  dest="controllers", action="append_const", const="constant_05",
                        help="Constant u=0.5 baseline")
    parser.add_argument("--constant-1",   dest="controllers", action="append_const", const="constant_10",
                        help="Constant u=1.0 baseline")
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
        help="Heat load profile (default: NonuniformStep)",
    )
    parser.add_argument("--stride", type=int, default=10,
                        help="Simulation steps between frames (default: 10)")
    parser.add_argument("--fps",    type=int, default=10,
                        help="GIF frames per second (default: 10)")
    parser.add_argument("--seed",   type=int, default=7,
                        help="RNG seed (default: 7)")
    parser.add_argument("--vmin",   type=float, default=None,
                        help="Color scale minimum °C (default: pack ambient_temp_c)")
    parser.add_argument("--vmax",   type=float, default=None,
                        help="Color scale maximum °C (default: pack safe_temp_c)")

    args = parser.parse_args()

    _ALIASES = {
        "pi":           "pi_tuned",
        "sac":          "sac_best",
        "sac_best":     "sac_best",
        "sac_final":    "sac_final",
        "ppo":          "ppo_best",
        "ppo_best":     "ppo_best",
        "ppo_final":    "ppo_final",
        "bang_bang":    "bang_bang",
        "bang-bang":    "bang_bang",
        "proportional": "proportional",
        "no_cooling":   "no_cooling",
        "constant_05":  "constant_05",
        "constant_10":  "constant_10",
    }

    keys: List[str] = list(args.controllers or [])
    for k in args.extra_controllers:
        resolved = _ALIASES.get(k.lower().replace("-", "_"), k)
        keys.append(resolved)
    if not keys:
        keys = ["pi_tuned"]

    return keys, args.profile, args.stride, args.fps, args.seed, args.vmin, args.vmax


# ---------------------------------------------------------------------------
# Per-controller run
# ---------------------------------------------------------------------------

def _run_one(
    controller_key: str,
    profile_name: str,
    stride: int,
    fps: int,
    seed: int,
    vmin: float | None,
    vmax: float | None,
    output_dir: Path,
) -> None:
    cell_config = CellConfig()
    pack_config = PackConfig(shape=(4, 3, 2))

    controller = make_controller(controller_key, pack_config)

    print(f"\nRunning: {controller.name} | profile={profile_name} | stride={stride}")
    (_, T_max, T_avg, T_gradient, _, _,
     T3d_frames, frame_times) = run_episode_with_frames(
        controller=controller,
        profile_name=profile_name,
        cell_config=cell_config,
        pack_config=pack_config,
        seed=seed,
        stride=stride,
    )

    print(
        f"  {len(T3d_frames)} frames | "
        f"T_max peak={T_max.max():.2f} °C | "
        f"T_gradient max={T_gradient.max():.2f} °C"
    )

    slug_c = sanitize(controller.name)
    slug_p = sanitize(profile_name)
    gif_path = output_dir / f"3d_pyvista_{slug_c}_{slug_p}.gif"

    print(f"Rendering PyVista GIF ({len(T3d_frames)} frames) → {gif_path}")
    save_temperature_gif(
        T_frames=T3d_frames,
        frame_times=frame_times,
        cell=cell_config,
        pack=pack_config,
        output_path=gif_path,
        fps=fps,
        vmin=vmin,
        vmax=vmax,
    )
    print(f"Saved: {gif_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        import pyvista  # noqa: F401
    except ImportError:
        print("ERROR: PyVista not installed.  Run: pip install pyvista")
        sys.exit(1)

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    controller_keys, profile_name, stride, fps, seed, vmin, vmax = _parse_args()

    print(f"Profile:     {profile_name}")
    print(f"Controllers: {controller_keys}")
    print(f"Stride:      {stride}  |  FPS: {fps}  |  Seed: {seed}")

    for key in controller_keys:
        try:
            _run_one(key, profile_name, stride, fps, seed, vmin, vmax, output_dir)
        except (KeyError, FileNotFoundError) as exc:
            print(f"Skipping '{key}': {exc}")


if __name__ == "__main__":
    main()
