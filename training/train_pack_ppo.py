"""
training/train_pack_ppo.py

Train PPO on the Phase 4 multi-node battery pack thermal environment.

Run from project root:
    python -m training.train_pack_ppo

Outputs:
    models/ppo_battery_pack_thermal/best_model.zip
    models/ppo_battery_pack_thermal/final_model.zip
    models/ppo_battery_pack_thermal/vec_normalize.pkl
    logs/ppo_battery_pack_thermal/

This is the pack-level PPO controller. It trains on randomized pack heat profiles,
cell-to-cell heat imbalance, and random initial/ambient conditions.

The PPO benchmark to beat after training:
    Pack max-temp PI tuned mean reward ≈ -395.19
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.battery_pack_thermal_env import (
    BatteryPackThermalConfig,
    BatteryPackThermalEnv,
    PackHeatProfile,
    random_nonuniform_pack_heat,
)


# -----------------------------------------------------------------------------
# Randomized training heat profile
# -----------------------------------------------------------------------------

def randomized_pack_heat_profile() -> PackHeatProfile:
    """
    Create a randomized pack heat profile for PPO training.

    This deliberately varies:
        - total heat level
        - step/pulse/random behavior
        - hotspot location
        - hotspot strength
        - cell-to-cell heat distribution

    This matters. Training only on the four fixed benchmark profiles would let PPO
    memorize instead of learning robust thermal control.
    """
    state = {
        "mode": None,
        "q_total": None,
        "weights": None,
        "step_time": None,
        "period": None,
        "duty_cycle": None,
        "hotspot_idx": None,
    }

    def reset_state(rng: np.random.Generator, n_cells: int) -> None:
        mode = rng.choice(["constant", "step", "pulsed", "random"])
        q_base = float(rng.uniform(450.0, 850.0))
        hotspot_idx = int(rng.integers(0, n_cells))
        hotspot_factor = float(rng.uniform(1.2, 2.4))

        raw_weights = rng.uniform(0.80, 1.20, size=n_cells).astype(np.float32)
        raw_weights[hotspot_idx] *= hotspot_factor

        # Sometimes create a neighboring hotspot too.
        if rng.random() < 0.35:
            neighbor = int(np.clip(hotspot_idx + rng.choice([-1, 1]), 0, n_cells - 1))
            raw_weights[neighbor] *= float(rng.uniform(1.1, 1.6))

        raw_weights = np.clip(raw_weights, 0.01, None)
        raw_weights /= np.sum(raw_weights)

        state["mode"] = mode
        state["q_total"] = q_base
        state["weights"] = raw_weights
        state["step_time"] = float(rng.uniform(300.0, 1_000.0))
        state["period"] = float(rng.uniform(100.0, 260.0))
        state["duty_cycle"] = float(rng.uniform(0.25, 0.60))
        state["hotspot_idx"] = hotspot_idx

    def profile(t: float, rng: np.random.Generator, n_cells: int) -> np.ndarray:
        # Re-randomize at the beginning of each episode.
        if state["mode"] is None or t <= 0.0 or state["weights"] is None:
            reset_state(rng, n_cells)

        mode = str(state["mode"])
        q_total_state = float(state["q_total"])
        weights = np.asarray(state["weights"], dtype=np.float32)

        if mode == "constant":
            q_total = q_total_state

        elif mode == "step":
            step_time = float(state["step_time"])
            low = 0.70 * q_total_state
            high = 1.55 * q_total_state
            q_total = low if t < step_time else high

        elif mode == "pulsed":
            period = float(state["period"])
            duty_cycle = float(state["duty_cycle"])
            phase = (t % period) / period
            low = 0.55 * q_total_state
            high = 1.75 * q_total_state
            q_total = high if phase < duty_cycle else low

        else:  # random
            disturbance = rng.normal(0.0, 25.0)
            q_total = 0.90 * q_total_state + 0.10 * 650.0 + disturbance
            q_total = float(np.clip(q_total, 350.0, 1_150.0))
            state["q_total"] = q_total

            # Slowly drift the heat distribution.
            weight_noise = rng.normal(0.0, 0.0025, size=n_cells).astype(np.float32)
            weights = np.clip(weights + weight_noise, 0.01, None)
            weights /= np.sum(weights)
            state["weights"] = weights

        q_total = float(np.clip(q_total, 250.0, 1_300.0))
        return (q_total * weights).astype(np.float32)

    return profile


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------

def make_training_config(seed: Optional[int] = 7) -> BatteryPackThermalConfig:
    return BatteryPackThermalConfig(
        n_cells=8,
        total_time=1800.0,
        dt=1.0,
        initial_temp=25.0,
        ambient_temp=25.0,
        thermal_capacitance=8_000.0,
        surface_area_per_cell=0.12,
        h_min=5.0,
        h_max=95.0,
        cooling_nonlinearity=1.0,
        direct_cooling_max_per_cell=0.0,
        conduction_coupling=8.0,
        target_temp=35.0,
        soft_max_temp=45.0,
        hard_max_temp=60.0,
        max_temp_weight=1.0,
        mean_temp_weight=0.15,
        imbalance_weight=0.40,
        over_temp_weight=10.0,
        action_weight=0.04,
        action_smoothness_weight=0.08,
        hard_violation_penalty=150.0,
        initial_temp_randomization=1.0,
        ambient_randomization=1.5,
        cell_heat_variation=0.12,
        seed=seed,
    )


def make_eval_config(seed: Optional[int] = 17) -> BatteryPackThermalConfig:
    # Same physics as training, different seed.
    return make_training_config(seed=seed)


# -----------------------------------------------------------------------------
# Env factories
# -----------------------------------------------------------------------------

def make_train_env(seed: int):
    def _init():
        config = make_training_config(seed=seed)
        env = BatteryPackThermalEnv(
            config=config,
            heat_profile=randomized_pack_heat_profile(),
            render_mode=None,
        )
        env = Monitor(env)
        return env

    return _init


def make_eval_env(seed: int):
    def _init():
        config = make_eval_config(seed=seed)
        env = BatteryPackThermalEnv(
            config=config,
            heat_profile=randomized_pack_heat_profile(),
            render_mode=None,
        )
        env = Monitor(env)
        return env

    return _init


# -----------------------------------------------------------------------------
# Callback
# -----------------------------------------------------------------------------

class TrainingProgressCallback(BaseCallback):
    """Lightweight training progress logger."""

    def __init__(self, print_freq: int = 25_000, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"Training progress: {self.num_timesteps:,} timesteps")
        return True


# -----------------------------------------------------------------------------
# Main training
# -----------------------------------------------------------------------------

def main() -> None:
    model_dir = PROJECT_ROOT / "models" / "ppo_battery_pack_thermal"
    log_dir = PROJECT_ROOT / "logs" / "ppo_battery_pack_thermal"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Quick single-env check before vectorized training.
    check_config = make_training_config(seed=123)
    check_env_instance = BatteryPackThermalEnv(
        config=check_config,
        heat_profile=randomized_pack_heat_profile(),
        render_mode=None,
    )
    check_env(check_env_instance, warn=True)
    print("Pack environment passed Stable-Baselines3 env check.")

    n_envs = 4
    train_env = DummyVecEnv([make_train_env(seed=100 + i) for i in range(n_envs)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.995,
    )

    eval_env = DummyVecEnv([make_eval_env(seed=900)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.995,
    )

    # Evaluation env must share normalization statistics with training.
    eval_env.obs_rms = train_env.obs_rms
    eval_env.training = False

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

    progress_callback = TrainingProgressCallback(print_freq=25_000)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.002,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=7,
        device="auto",
        policy_kwargs={
            "net_arch": {
                "pi": [128, 128],
                "vf": [128, 128],
            }
        },
    )

    # Use 50_000 for a smoke test. Use 750_000 to 1_500_000 for a real pack run.
    total_timesteps = 1_000_000

    print("Starting pack PPO training...")
    print(f"Total timesteps: {total_timesteps:,}")
    print("Baseline to beat later: Pack max-temp PI tuned ≈ -395.19 mean reward")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, progress_callback],
        tb_log_name="PPO_pack_thermal",
        progress_bar=True,
    )

    final_model_path = model_dir / "final_model.zip"
    vecnorm_path = model_dir / "vec_normalize.pkl"

    model.save(str(final_model_path))
    train_env.save(str(vecnorm_path))

    print("\nTraining complete.")
    print("Saved outputs:")
    print(f"  {model_dir / 'best_model.zip'}")
    print(f"  {final_model_path}")
    print(f"  {vecnorm_path}")
    print(f"  {log_dir}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
