"""
training/train_pack_sac_3d.py

Train SAC on the 3D cell-resolved battery pack thermal environment.

Run from project root — minimal smoke test:
    python training/train_pack_sac_3d.py --timesteps 10000

Full training (Google Colab / GPU):
    python training/train_pack_sac_3d.py \\
        --timesteps 3000000 \\
        --save-dir /path/to/models/sac_3d_pack \\
        --log-dir  /path/to/logs/sac_3d_pack

Outputs (in --save-dir):
    sac_pack_final.zip
    best_model.zip
    checkpoints/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from configs.pack_config import CellConfig, PackConfig
from envs.battery_pack_thermal_env_3d import BatteryPackThermalEnv3D
from training.train_pack_ppo_3d import randomized_3d_heat_profile, make_pack_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SAC on 3D battery pack thermal environment"
    )
    p.add_argument("--timesteps",       type=int,   default=750_000,
                   help="Total training timesteps (default: 750K)")
    p.add_argument("--save-dir",        type=str,   default=None,
                   help="Model save directory (default: models/sac_pack_3d/)")
    p.add_argument("--log-dir",         type=str,   default=None,
                   help="TensorBoard log directory (default: logs/sac_pack_3d/)")
    p.add_argument("--learning-rate",   type=float, default=3e-4,
                   help="SAC learning rate (default: 3e-4)")
    p.add_argument("--buffer-size",     type=int,   default=500_000,
                   help="Replay buffer size (default: 500K)")
    p.add_argument("--batch-size",      type=int,   default=256,
                   help="SAC minibatch size (default: 256)")
    p.add_argument("--learning-starts", type=int,   default=10_000,
                   help="Steps before learning begins (default: 10K)")
    p.add_argument("--device",          type=str,   default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="Training device (default: auto)")
    p.add_argument("--seed",            type=int,   default=7,
                   help="Random seed (default: 7)")
    p.add_argument("--resume-from",     type=str,   default=None,
                   help="Path to model.zip to resume training from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def make_train_env(seed: int) -> Monitor:
    env = BatteryPackThermalEnv3D(
        cell_config=CellConfig(),
        pack_config=make_pack_config(),
        heat_profile=randomized_3d_heat_profile(),
        seed=seed,
    )
    return Monitor(env)


def make_eval_env(seed: int) -> Monitor:
    env = BatteryPackThermalEnv3D(
        cell_config=CellConfig(),
        pack_config=make_pack_config(),
        heat_profile=randomized_3d_heat_profile(),
        seed=seed,
    )
    return Monitor(env)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class ProgressCallback(BaseCallback):
    def __init__(self, print_freq: int = 25_000, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"  [{self.num_timesteps:,} steps]")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Paths
    model_dir = Path(args.save_dir) if args.save_dir else PROJECT_ROOT / "models" / "sac_pack_3d"
    log_dir   = Path(args.log_dir)  if args.log_dir  else PROJECT_ROOT / "logs"   / "sac_pack_3d"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "checkpoints").mkdir(exist_ok=True)

    # Verify continuous action space (SAC requirement)
    _check_env = BatteryPackThermalEnv3D(
        cell_config=CellConfig(),
        pack_config=make_pack_config(),
        heat_profile=randomized_3d_heat_profile(),
        seed=123,
    )
    if not isinstance(_check_env.action_space, spaces.Box):
        raise RuntimeError(
            "SAC requires a continuous Box action space. "
            f"Got: {type(_check_env.action_space)}. "
            "Convert to spaces.Box before training SAC."
        )
    check_env(_check_env, warn=True)
    print("3D pack env passed SB3 check. Action space is continuous Box — SAC compatible.")

    # Environments
    train_env = make_train_env(seed=42)
    eval_env  = make_eval_env(seed=900)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir / "eval"),
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="sac_pack_checkpoint",
        verbose=1,
    )

    # Model — new or resumed
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        model = SAC.load(args.resume_from, env=train_env, device=device)
    else:
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.995,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_entropy="auto",
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=args.seed,
            device=device,
            policy_kwargs={"net_arch": [256, 256]},
        )

    pack_cfg = make_pack_config()
    print(f"\nSAC training — {args.timesteps:,} timesteps")
    print(f"Pack shape: {pack_cfg.shape}  ({int(np.prod(pack_cfg.shape))} cells)")
    print(f"Replay buffer: {model.buffer_size:,}  "
          f"Learning starts: {model.learning_starts:,}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback, ProgressCallback()],
        tb_log_name="SAC_pack_3d",
        progress_bar=True,
        reset_num_timesteps=not bool(args.resume_from),
    )

    final_path = model_dir / "sac_pack_final.zip"
    model.save(str(final_path))

    print("\nTraining complete.")
    print(f"  best_model:  {model_dir / 'best_model.zip'}")
    print(f"  final_model: {final_path}")
    print(f"  checkpoints: {model_dir / 'checkpoints'}/")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
