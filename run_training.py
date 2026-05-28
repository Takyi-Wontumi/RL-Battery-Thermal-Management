#!/usr/bin/env python3
"""
run_training.py

Python equivalent of run_training.sh — optimised for Google Colab but runs
anywhere.  Uses sys.executable so it always picks up the active venv or Colab
Python without any path gymnastics.

Stages
------
  1  Physics demo        scripts/run_3d_simulation.py
  2  Baseline benchmark  scripts/compare_pack_baselines_3d.py
  3  PPO training        training/train_pack_ppo_3d.py
  4  SAC training        training/train_pack_sac_3d.py
  5  Full evaluation     evaluation/compare_controllers.py
  6  3D screenshots      scripts/visualize_3d_controllers.py
  7  GIF animations      scripts/animate_pack_3d_pyvista.py

─────────────────────────────────────────────────────────────────────────────
Local usage
─────────────────────────────────────────────────────────────────────────────
    python run_training.py                        # CPU defaults (100K steps)
    python run_training.py --ppo-steps 200000
    python run_training.py --baselines-only
    python run_training.py --skip-demo --skip-gif

─────────────────────────────────────────────────────────────────────────────
Google Colab
─────────────────────────────────────────────────────────────────────────────
    # Minimal — runs everything on Colab GPU with 1M PPO / 750K SAC:
    !python run_training.py --colab

    # Save models to Google Drive so they survive session resets:
    from google.colab import drive
    drive.mount('/content/drive')
    !python run_training.py --colab \\
        --save-dir /content/drive/MyDrive/rl_battery/models \\
        --log-dir  /content/drive/MyDrive/rl_battery/logs

    # Baselines only (no GPU needed, fast sanity check):
    !python run_training.py --baselines-only

    # Resume an interrupted RL run:
    !python run_training.py --colab \\
        --ppo-resume models/ppo_pack_3d/ppo_pack_final.zip \\
        --sac-resume models/sac_pack_3d/sac_pack_final.zip
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable   # always the interpreter running this script


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RL Battery Thermal Management — full training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Presets ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--colab", action="store_true",
        help="Colab GPU preset: 1M PPO steps, 750K SAC steps, 8 parallel envs",
    )

    # ── Training scale ───────────────────────────────────────────────────────
    p.add_argument("--ppo-steps",  type=int,   default=100_000,
                   help="PPO total timesteps")
    p.add_argument("--sac-steps",  type=int,   default=100_000,
                   help="SAC total timesteps")
    p.add_argument("--n-envs",     type=int,   default=4,
                   help="Parallel envs for PPO")
    p.add_argument("--device",     type=str,   default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="Training device")
    p.add_argument("--seed",       type=int,   default=7)

    # ── Output paths ─────────────────────────────────────────────────────────
    p.add_argument("--save-dir",   type=str,   default=None,
                   help="Root dir for model weights (sub-dirs ppo_pack_3d/ sac_pack_3d/ created automatically)")
    p.add_argument("--log-dir",    type=str,   default=None,
                   help="Root dir for TensorBoard logs")

    # ── Resume ───────────────────────────────────────────────────────────────
    p.add_argument("--ppo-resume", type=str,   default=None,
                   help="Path to PPO .zip to resume from")
    p.add_argument("--sac-resume", type=str,   default=None,
                   help="Path to SAC .zip to resume from")

    # ── Evaluation ───────────────────────────────────────────────────────────
    p.add_argument("--episodes",   type=int,   default=20,
                   help="Evaluation episodes across all profiles")

    # ── Stage control ────────────────────────────────────────────────────────
    p.add_argument("--baselines-only",  action="store_true",
                   help="Run stages 1–2 only (no RL training)")
    p.add_argument("--skip-baselines",  action="store_true",
                   help="Skip stage 2 baseline benchmark")
    p.add_argument("--skip-demo",       action="store_true",
                   help="Skip stage 1 physics demo")
    p.add_argument("--skip-gif",        action="store_true",
                   help="Skip stage 7 GIF animations (saves ~5–15 min)")

    return p.parse_args()


def apply_colab_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.colab:
        args.ppo_steps = 1_000_000
        args.sac_steps = 750_000
        args.n_envs    = 8
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Stage runner
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(label: str, cmd: List[str], failed: List[str]) -> bool:
    """Run one pipeline stage. On failure, records it and continues."""
    border = "─" * 62
    print(f"\n{border}")
    print(f"  STAGE: {label}")
    print(f"  CMD:   {' '.join(str(c) for c in cmd)}")
    print(border)
    sys.stdout.flush()

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  ✓ {label}  ({elapsed:.0f}s)")
        return True
    else:
        print(f"  ✗ {label} FAILED  ({elapsed:.0f}s) — continuing...")
        failed.append(label)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# GPU / device info
# ─────────────────────────────────────────────────────────────────────────────

def describe_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda  ({torch.cuda.get_device_name(0)})"
        return "cpu"
    except ImportError:
        return "cpu (torch not found)"


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_root(args: argparse.Namespace) -> Path:
    return Path(args.save_dir) if args.save_dir else PROJECT_ROOT / "models"


def _log_root(args: argparse.Namespace) -> Path:
    return Path(args.log_dir) if args.log_dir else PROJECT_ROOT / "logs"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args = apply_colab_preset(args)

    model_root = _model_root(args)
    log_root   = _log_root(args)

    ppo_model_dir = model_root / "ppo_pack_3d_multizone_sensor"
    sac_model_dir = model_root / "sac_pack_3d_multizone_sensor"
    ppo_model     = ppo_model_dir / "best_model.zip"
    sac_model     = sac_model_dir / "best_model.zip"
    eval_dir      = PROJECT_ROOT / "outputs" / "comparison"

    print()
    print("=" * 62)
    print("  RL Battery Thermal Management — Full Training Pipeline")
    print("=" * 62)
    print(f"  Device:         {describe_device()}")
    print(f"  PPO timesteps:  {args.ppo_steps:,}")
    print(f"  SAC timesteps:  {args.sac_steps:,}")
    print(f"  Parallel envs:  {args.n_envs}  (PPO)")
    print(f"  Model root:     {model_root}")
    print(f"  Log root:       {log_root}")
    print(f"  Baselines only: {args.baselines_only}")
    print(f"  Skip GIF:       {args.skip_gif}")
    print("=" * 62)

    failed: List[str] = []
    t_total = time.time()

    # ── Stage 1 — Physics demo ────────────────────────────────────────────
    if not args.skip_demo:
        run_stage(
            "Physics demo (HPPC + multi-zone)",
            [PYTHON, str(PROJECT_ROOT / "scripts" / "run_3d_simulation.py")],
            failed,
        )

    # ── Stage 2 — Baseline benchmark ─────────────────────────────────────
    if not args.skip_baselines:
        run_stage(
            "Baseline benchmark (6 classical controllers)",
            [PYTHON, "-m", "scripts.compare_pack_baselines_3d"],
            failed,
        )

    if args.baselines_only:
        print("\n  Baselines-only run complete.")
        _print_summary(failed, t_total, ppo_model, sac_model, eval_dir)
        sys.exit(1 if failed else 0)

    # ── Stage 3 — PPO training ────────────────────────────────────────────
    ppo_cmd = [
        PYTHON, str(PROJECT_ROOT / "training" / "train_pack_ppo_3d.py"),
        "--timesteps", str(args.ppo_steps),
        "--n-envs",    str(args.n_envs),
        "--device",    args.device,
        "--save-dir",  str(ppo_model_dir),
        "--log-dir",   str(log_root / "ppo_pack_3d_multizone_sensor"),
        "--seed",      str(args.seed),
    ]
    if args.ppo_resume:
        ppo_cmd += ["--resume-from", args.ppo_resume]
    run_stage(f"PPO training ({args.ppo_steps:,} steps)", ppo_cmd, failed)

    # ── Stage 4 — SAC training ────────────────────────────────────────────
    sac_cmd = [
        PYTHON, str(PROJECT_ROOT / "training" / "train_pack_sac_3d.py"),
        "--timesteps", str(args.sac_steps),
        "--device",    args.device,
        "--save-dir",  str(sac_model_dir),
        "--log-dir",   str(log_root / "sac_pack_3d_multizone_sensor"),
        "--seed",      str(args.seed),
    ]
    if args.sac_resume:
        sac_cmd += ["--resume-from", args.sac_resume]
    run_stage(f"SAC training ({args.sac_steps:,} steps)", sac_cmd, failed)

    # ── Stage 5 — Full evaluation ─────────────────────────────────────────
    eval_cmd = [
        PYTHON, str(PROJECT_ROOT / "evaluation" / "compare_controllers.py"),
        "--results-dir", str(eval_dir),
        "--episodes",    str(args.episodes),
        "--seed",        str(args.seed),
    ]
    if ppo_model.exists():
        eval_cmd += ["--ppo-model", str(ppo_model)]
    if sac_model.exists():
        eval_cmd += ["--sac-model", str(sac_model)]
    run_stage("Full evaluation (baselines + RL)", eval_cmd, failed)

    # ── Stage 6 — 3D PyVista screenshots ─────────────────────────────────
    run_stage(
        "3D screenshots (PI / SAC / PPO)",
        [
            PYTHON, str(PROJECT_ROOT / "scripts" / "visualize_3d_controllers.py"),
            "--output-dir", str(PROJECT_ROOT / "outputs" / "3d_viz"),
            "--profile",    "NonuniformStep",
            "--seed",       str(args.seed),
        ],
        failed,
    )

    # ── Stage 7 — GIF animations ──────────────────────────────────────────
    if not args.skip_gif:
        gif_cmd = [
            PYTHON, "-m", "scripts.animate_pack_3d_pyvista",
            "--no-cooling",
            "--constant-05",
            "--constant-1",
            "--bang-bang",
            "--proportional",
            "--PI",
            "--profile", "NonuniformStep",
            "--stride",  "10",
            "--fps",     "12",
            "--seed",    str(args.seed),
        ]
        if sac_model.exists():
            gif_cmd.append("--SAC")
        if ppo_model.exists():
            gif_cmd.append("--PPO")
        run_stage("GIF animations (baselines + RL)", gif_cmd, failed)

    _print_summary(failed, t_total, ppo_model, sac_model, eval_dir)
    sys.exit(1 if failed else 0)


def _print_summary(
    failed: List[str],
    t_start: float,
    ppo_model: Path,
    sac_model: Path,
    eval_dir: Path,
) -> None:
    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)

    print()
    print("=" * 62)
    print(f"  Run complete  ({mins}m {secs}s total)")
    print("=" * 62)
    print()
    print("  Key outputs:")
    print("    simulation_results.png          — HPPC + multi-zone demo")
    print("    outputs/                        — baseline benchmark plots")
    print(f"    {eval_dir}/  — full evaluation CSV + plots")
    print("    outputs/3d_viz/                 — 3D PyVista screenshots")
    print("    outputs/3d_pyvista_*_*.gif      — GIF animations")
    print(f"    {ppo_model.parent}/        — PPO weights")
    print(f"    {sac_model.parent}/        — SAC weights")
    print()
    print("  TensorBoard:")
    print(f"    {PYTHON} -m tensorboard.main --logdir logs/")
    print()

    if failed:
        print("  Failed stages:")
        for s in failed:
            print(f"    ✗ {s}")
        print()
    else:
        print("  All stages passed.")
    print()


if __name__ == "__main__":
    main()
