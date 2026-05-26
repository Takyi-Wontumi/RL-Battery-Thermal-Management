"""
training/train_pack_sac_1d.py

Train SAC on the 1D battery pack with 50 mm inter-cell spacing.

Run from project root:
    python -m training.train_pack_sac_1d

Outputs:
    models/sac_pack_1d/best_model.zip
    models/sac_pack_1d/final_model.zip
    logs/sac_pack_1d/
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from envs.battery_pack_thermal_env import BatteryPackThermalEnv
from scripts.compare_pack_baselines_1d_50mm import make_50mm_config
from training.train_pack_ppo import randomized_pack_heat_profile


def make_train_env(seed: int) -> Monitor:
    config = make_50mm_config()
    config.seed = seed
    env = BatteryPackThermalEnv(
        config=config,
        heat_profile=randomized_pack_heat_profile(),
    )
    return Monitor(env)


def make_eval_env(seed: int) -> Monitor:
    config = make_50mm_config()
    config.seed = seed
    env = BatteryPackThermalEnv(
        config=config,
        heat_profile=randomized_pack_heat_profile(),
    )
    return Monitor(env)


class ProgressCallback(BaseCallback):
    def __init__(self, print_freq: int = 25_000):
        super().__init__(verbose=1)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"  Training: {self.num_timesteps:,} steps")
        return True


def main() -> None:
    model_dir = PROJECT_ROOT / "models" / "sac_pack_1d"
    log_dir = PROJECT_ROOT / "logs" / "sac_pack_1d"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    check_config = make_50mm_config()
    check_env(BatteryPackThermalEnv(config=check_config,
                                    heat_profile=randomized_pack_heat_profile()), warn=True)
    print("1D pack env (50 mm spacing) passed env check.")

    train_env = make_train_env(seed=42)
    eval_env = make_eval_env(seed=900)

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

    config = make_50mm_config()
    obs_dim = 2 * config.n_cells + 3

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.995,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=7,
        device="auto",
        policy_kwargs={"net_arch": [256, 256]},
    )

    # 50_000 for a smoke test; 500_000 for a full run.
    total_timesteps = 500_000

    print(f"\nStarting 1D pack SAC training — {total_timesteps:,} timesteps")
    print(f"n_cells={config.n_cells}  obs_dim={obs_dim}  "
          f"spacing={config.cell_spacing_m*1000:.0f} mm  "
          f"conduction={config.conduction_coupling:.3f} W/K")
    print(f"Replay buffer: {model.buffer_size:,}  learning starts: {model.learning_starts:,}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, ProgressCallback()],
        tb_log_name="SAC_pack_1d",
        progress_bar=True,
    )

    final_path = model_dir / "final_model.zip"
    model.save(str(final_path))

    print("\nTraining complete.")
    print(f"  {model_dir / 'best_model.zip'}")
    print(f"  {final_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
