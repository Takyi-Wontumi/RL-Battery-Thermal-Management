"""
training/train_ppo.py

Phase 3 PPO training script for RL battery thermal management.

Run from project root:
    python -m training.train_ppo

Outputs:
    models/ppo_battery_thermal/best_model.zip
    models/ppo_battery_thermal/final_model.zip
    models/ppo_battery_thermal/vec_normalize.pkl
    logs/ppo_battery_thermal/

Important:
    PPO is trained on randomized heat profiles, not one fixed curve.
    If you train on one profile only, the agent will memorize a curve and your result is trash.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env

from envs.battery_thermal_env import BatteryThermalConfig, BatteryThermalEnv


HeatProfile = Callable[[float, np.random.Generator], float]


# -----------------------------------------------------------------------------
# Randomized training heat profiles
# -----------------------------------------------------------------------------

def constant_heat_profile(q_gen: float) -> HeatProfile:
    def profile(t: float, rng: np.random.Generator) -> float:
        return float(q_gen)

    return profile


def step_heat_profile(q_low: float, q_high: float, step_time: float) -> HeatProfile:
    def profile(t: float, rng: np.random.Generator) -> float:
        return float(q_low if t < step_time else q_high)

    return profile


def pulsed_heat_profile(q_low: float, q_high: float, period: float, duty_cycle: float) -> HeatProfile:
    def profile(t: float, rng: np.random.Generator) -> float:
        phase = (t % period) / period
        return float(q_high if phase < duty_cycle else q_low)

    return profile


def random_heat_profile(q_mean: float, q_std: float, smoothing: float, q_min: float, q_max: float) -> HeatProfile:
    state = {"q": q_mean}

    def profile(t: float, rng: np.random.Generator) -> float:
        disturbance = rng.normal(0.0, q_std)
        state["q"] = smoothing * state["q"] + (1.0 - smoothing) * q_mean + disturbance
        return float(np.clip(state["q"], q_min, q_max))

    return profile


def make_random_training_profile(rng: np.random.Generator) -> HeatProfile:
    """
    Randomly sample a heat profile family and parameters.

    This is the heart of generalization. The PPO agent must learn thermal control,
    not memorize the exact benchmark profiles.
    """
    profile_type = rng.choice(["constant", "step", "pulsed", "random"])

    if profile_type == "constant":
        q_gen = float(rng.uniform(450.0, 850.0))
        return constant_heat_profile(q_gen=q_gen)

    if profile_type == "step":
        q_low = float(rng.uniform(300.0, 550.0))
        q_high = float(rng.uniform(800.0, 1_250.0))
        step_time = float(rng.uniform(350.0, 1_100.0))
        return step_heat_profile(q_low=q_low, q_high=q_high, step_time=step_time)

    if profile_type == "pulsed":
        q_low = float(rng.uniform(250.0, 450.0))
        q_high = float(rng.uniform(850.0, 1_300.0))
        period = float(rng.uniform(90.0, 220.0))
        duty_cycle = float(rng.uniform(0.25, 0.55))
        return pulsed_heat_profile(q_low=q_low, q_high=q_high, period=period, duty_cycle=duty_cycle)

    q_mean = float(rng.uniform(500.0, 750.0))
    q_std = float(rng.uniform(8.0, 25.0))
    smoothing = float(rng.uniform(0.84, 0.94))
    return random_heat_profile(
        q_mean=q_mean,
        q_std=q_std,
        smoothing=smoothing,
        q_min=350.0,
        q_max=1_000.0,
    )


# -----------------------------------------------------------------------------
# Environment factories
# -----------------------------------------------------------------------------

def make_training_config(seed: Optional[int] = None) -> BatteryThermalConfig:
    """Training config aligned with Phase 2 benchmark difficulty."""
    return BatteryThermalConfig(
        total_time=1800.0,
        dt=1.0,
        initial_temp=25.0,
        ambient_temp=25.0,
        thermal_capacitance=40_000.0,
        surface_area=1.0,
        h_min=5.0,
        h_max=95.0,
        direct_cooling_max=0.0,
        target_temp=35.0,
        soft_max_temp=45.0,
        hard_max_temp=60.0,
        temp_error_weight=1.0,
        over_temp_weight=8.0,
        action_weight=0.05,
        action_smoothness_weight=0.1,
        hard_violation_penalty=100.0,
        seed=seed,
    )


class RandomizedBatteryThermalEnv(BatteryThermalEnv):
    """
    BatteryThermalEnv wrapper that samples a new heat profile every episode.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.profile_rng = np.random.default_rng(seed)
        config = make_training_config(seed=seed)
        heat_profile = make_random_training_profile(self.profile_rng)
        super().__init__(config=config, heat_profile=heat_profile, render_mode=None)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.profile_rng = np.random.default_rng(seed)

        # New profile every episode.
        self.heat_profile = make_random_training_profile(self.profile_rng)

        options = options or {}
        options.setdefault("randomize", True)
        return super().reset(seed=seed, options=options)


def make_env(rank: int, base_seed: int = 42) -> Callable[[], Monitor]:
    """Factory for vectorized training environments."""

    def _init() -> Monitor:
        env = RandomizedBatteryThermalEnv(seed=base_seed + rank)
        env = Monitor(env)
        return env

    return _init


# -----------------------------------------------------------------------------
# Callback
# -----------------------------------------------------------------------------

class SaveVecNormalizeCallback(BaseCallback):
    """Save VecNormalize statistics during training."""

    def __init__(self, save_path: Path, save_freq: int = 25_000, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vec_norm = self.model.get_vec_normalize_env()
            if vec_norm is not None:
                vec_norm.save(str(self.save_path))
        return True


# -----------------------------------------------------------------------------
# Main training
# -----------------------------------------------------------------------------

def main() -> None:
    model_dir = PROJECT_ROOT / "models" / "ppo_battery_thermal"
    log_dir = PROJECT_ROOT / "logs" / "ppo_battery_thermal"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_seed = 42
    n_envs = 8
    total_timesteps = 500_000

    # Quick Gymnasium API sanity check before expensive training.
    check_env(RandomizedBatteryThermalEnv(seed=base_seed), warn=True)

    train_env = SubprocVecEnv([make_env(rank=i, base_seed=base_seed) for i in range(n_envs)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    eval_env = DummyVecEnv([make_env(rank=10_000, base_seed=base_seed)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    # Share observation normalization stats from training env during evaluation.
    eval_env.obs_rms = train_env.obs_rms

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_battery_thermal",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=25_000 // n_envs,
        n_eval_episodes=8,
        deterministic=True,
        render=False,
    )

    save_vecnorm_callback = SaveVecNormalizeCallback(
        save_path=model_dir / "vec_normalize.pkl",
        save_freq=25_000,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=base_seed,
        device="auto",
    )

    print("\nStarting PPO training...")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Parallel envs:    {n_envs}")
    print(f"Model dir:        {model_dir}")
    print(f"TensorBoard:      {log_dir}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, save_vecnorm_callback],
        tb_log_name="PPO_battery_thermal",
        progress_bar=True,
    )

    final_model_path = model_dir / "final_model"
    vecnorm_path = model_dir / "vec_normalize.pkl"

    model.save(str(final_model_path))
    train_env.save(str(vecnorm_path))

    train_env.close()
    eval_env.close()

    print("\nTraining complete.")
    print(f"Saved final model: {final_model_path}.zip")
    print(f"Saved VecNormalize stats: {vecnorm_path}")
    print(f"Best model, if evaluation improved, is in: {model_dir / 'best_model.zip'}")


if __name__ == "__main__":
    main()
