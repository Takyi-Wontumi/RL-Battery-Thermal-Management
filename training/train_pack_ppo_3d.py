"""
training/train_pack_ppo_3d.py

Train PPO on the 3D cell-resolved battery pack thermal environment.

Run from project root — minimal smoke test:
    python training/train_pack_ppo_3d.py --timesteps 10000 --n-envs 2

Full training (Google Colab / GPU):
    python training/train_pack_ppo_3d.py \\
        --timesteps 3000000 \\
        --n-envs 8 \\
        --save-dir /path/to/models/ppo_3d_pack \\
        --log-dir  /path/to/logs/ppo_3d_pack

Outputs (in --save-dir):
    ppo_pack_final.zip
    best_model.zip
    vec_normalize.pkl
    checkpoints/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from configs.pack_config import CellConfig, PackConfig
from configs.sensor_simulation import SensorConfig, ActuatorConfig
from envs.battery_pack_thermal_env_3d import (
    BatteryPackThermalEnv3D,
    Pack3DHeatProfile,
    nonuniform_step_3d_heat,
    uniform_constant_3d_heat,
)


# ---------------------------------------------------------------------------
# Curriculum stage configs
# ---------------------------------------------------------------------------

def make_perfect_sensor_config(pack_config: PackConfig) -> SensorConfig:
    """Stage 1 — perfect sensors, no noise or delay."""
    return SensorConfig(
        num_zones=pack_config.num_cooling_zones,
        enabled=True,
        use_sparse_thermistors=False,
        temp_noise_std_c=0.0,
        current_noise_std_a=0.0,
        pack_voltage_noise_std_v=0.0,
        group_voltage_noise_std_v=0.0,
        coolant_temp_noise_std_c=0.0,
        actuator_feedback_noise_std=0.0,
        temp_bias_range_c=0.0,
        current_bias_range_a=0.0,
        voltage_bias_range_v=0.0,
        sensor_delay_s=0.0,
        enable_sensor_dropout=False,
        enable_lowpass_filter=False,
    )


def make_perfect_actuator_config(pack_config: PackConfig) -> ActuatorConfig:
    """Stage 1 — no actuator delay or rate limiting."""
    return ActuatorConfig(
        num_zones=pack_config.num_cooling_zones,
        cooling_delay_s=0.0,
        enable_rate_limit=False,
        effectiveness=1.0,
        enable_actuator_fault=False,
    )


def make_light_sensor_config(pack_config: PackConfig) -> SensorConfig:
    """Stage 2 — light noise + bias, no delay, full thermistors."""
    return SensorConfig(
        num_zones=pack_config.num_cooling_zones,
        enabled=True,
        use_sparse_thermistors=False,
        temp_noise_std_c=0.10,
        current_noise_std_a=0.10,
        pack_voltage_noise_std_v=0.01,
        group_voltage_noise_std_v=0.002,
        coolant_temp_noise_std_c=0.07,
        actuator_feedback_noise_std=0.005,
        temp_bias_range_c=0.25,
        current_bias_range_a=0.05,
        voltage_bias_range_v=0.005,
        sensor_delay_s=0.0,
        enable_sensor_dropout=False,
        enable_lowpass_filter=True,
        lowpass_alpha=0.50,
    )


def make_light_actuator_config(pack_config: PackConfig) -> ActuatorConfig:
    """Stage 2 — mild actuator delay, no rate limiting."""
    return ActuatorConfig(
        num_zones=pack_config.num_cooling_zones,
        cooling_delay_s=2.0,
        enable_rate_limit=False,
        effectiveness=1.0,
        enable_actuator_fault=False,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PPO on 3D battery pack thermal environment"
    )
    p.add_argument("--timesteps",     type=int,   default=5_000_000,
                   help="Total training timesteps (default: 5M)")
    p.add_argument("--n-envs",        type=int,   default=4,
                   help="Parallel training environments (default: 4)")
    p.add_argument("--save-dir",      type=str,   default=None,
                   help="Model save directory (default: models/ppo_pack_3d_multizone_sensor/)")
    p.add_argument("--log-dir",       type=str,   default=None,
                   help="TensorBoard log directory (default: logs/ppo_pack_3d_multizone_sensor/)")
    p.add_argument("--learning-rate", type=float, default=2e-4,
                   help="PPO learning rate (default: 2e-4)")
    p.add_argument("--batch-size",    type=int,   default=256,
                   help="PPO minibatch size (default: 256)")
    p.add_argument("--device",        type=str,   default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="Training device (default: auto)")
    p.add_argument("--seed",          type=int,   default=7,
                   help="Random seed (default: 7)")
    p.add_argument("--resume-from",   type=str,   default=None,
                   help="Path to model.zip to resume training from")
    p.add_argument("--curriculum",    action="store_true",
                   help="Use curriculum training: perfect → light → full-realism sensors")
    p.add_argument("--stage2-at",     type=int,   default=1_000_000,
                   help="Timestep to advance to Stage 2 (light noise) [curriculum only]")
    p.add_argument("--stage3-at",     type=int,   default=2_500_000,
                   help="Timestep to advance to Stage 3 (full realism) [curriculum only]")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Randomized training heat profile
# ---------------------------------------------------------------------------

def randomized_3d_heat_profile() -> Pack3DHeatProfile:
    """
    Time-varying heat profile that re-randomizes at the start of each episode.
    Training on diverse disturbances prevents PPO from memorising a single pattern.
    """
    _state: dict = {
        "mode": None, "q_base": None, "weights": None,
        "step_time": None, "period": None, "duty_cycle": None,
    }

    def _reset(rng: np.random.Generator, shape: Tuple[int, int, int]) -> None:
        mode = rng.choice(["constant", "step", "pulsed", "random"])
        q_base = float(rng.uniform(10.0, 35.0))
        weights = rng.uniform(0.80, 1.20, size=shape).astype(np.float64)
        hx = int(rng.integers(0, shape[0]))
        hy = int(rng.integers(0, shape[1]))
        hz = int(rng.integers(0, shape[2]))
        weights[hx, hy, hz] *= float(rng.uniform(1.2, 2.5))
        if rng.random() < 0.4:
            nx = int(np.clip(hx + rng.choice([-1, 1]), 0, shape[0] - 1))
            weights[nx, hy, hz] *= float(rng.uniform(1.1, 1.6))
        weights = np.clip(weights, 0.01, None)
        weights /= weights.sum()
        _state.update({
            "mode": mode, "q_base": q_base, "weights": weights,
            "step_time": float(rng.uniform(300.0, 1_000.0)),
            "period": float(rng.uniform(100.0, 260.0)),
            "duty_cycle": float(rng.uniform(0.25, 0.60)),
        })

    def profile(t: float, rng: np.random.Generator, shape: Tuple[int, int, int]) -> np.ndarray:
        if _state["mode"] is None or t <= 0.0 or _state["weights"] is None:
            _reset(rng, shape)
        q_base   = float(_state["q_base"])
        weights  = np.asarray(_state["weights"], dtype=np.float64)
        mode     = str(_state["mode"])
        if mode == "constant":
            q_total = q_base
        elif mode == "step":
            q_total = 0.70 * q_base if t < float(_state["step_time"]) else 1.55 * q_base
        elif mode == "pulsed":
            phase   = (t % float(_state["period"])) / float(_state["period"])
            q_total = 1.75 * q_base if phase < float(_state["duty_cycle"]) else 0.55 * q_base
        else:
            noise = rng.normal(0.0, 1.5)
            _state["q_base"] = 0.90 * q_base + 0.10 * 20.0 + noise
            q_total = float(np.clip(float(_state["q_base"]), 8.0, 50.0))
            w_noise = rng.normal(0.0, 0.002, size=shape)
            weights = np.clip(weights + w_noise, 0.01, None)
            weights /= weights.sum()
            _state["weights"] = weights
        q_total = float(np.clip(q_total, 8.0, 50.0))
        return (q_total * weights).astype(np.float64)

    return profile


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_pack_config(seed: int = 7) -> PackConfig:
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


def make_sensor_config(pack_config: PackConfig) -> SensorConfig:
    """Combined-realism sensor config — matches Scenario I in the validation matrix."""
    return SensorConfig(
        num_zones=pack_config.num_cooling_zones,
        enabled=True,
        use_sparse_thermistors=True,
        temp_noise_std_c=0.20,
        current_noise_std_a=0.20,
        pack_voltage_noise_std_v=0.02,
        group_voltage_noise_std_v=0.005,
        coolant_temp_noise_std_c=0.15,
        actuator_feedback_noise_std=0.01,
        temp_bias_range_c=0.50,
        current_bias_range_a=0.10,
        voltage_bias_range_v=0.01,
        sensor_delay_s=2.0,
        enable_sensor_dropout=False,
        enable_lowpass_filter=True,
        lowpass_alpha=0.35,
    )


def make_actuator_config(pack_config: PackConfig) -> ActuatorConfig:
    """Combined-realism actuator config — matches Scenario I in the validation matrix."""
    return ActuatorConfig(
        num_zones=pack_config.num_cooling_zones,
        cooling_delay_s=5.0,
        enable_rate_limit=True,
        max_cooling_rate_per_s=0.05,
        effectiveness=1.0,
        enable_actuator_fault=False,
    )


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """
    Progressively increases sensor/actuator realism at fixed timestep milestones.

    Stage 1 (0  → stage2_at):  Perfect sensors, no delay.
    Stage 2 (stage2_at → stage3_at): Light noise, mild delay.
    Stage 3 (stage3_at → end): Full combined realism (Scenario I).

    Uses env.update_sensor_config() so VecNormalize statistics are preserved
    across stage transitions.
    """

    def __init__(
        self,
        pack_config: PackConfig,
        stage2_at: int,
        stage3_at: int,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._pack_config = pack_config
        self._stage2_at   = stage2_at
        self._stage3_at   = stage3_at
        self._current_stage = 1

    def _on_step(self) -> bool:
        new_stage = self._current_stage

        if self.num_timesteps >= self._stage3_at and self._current_stage < 3:
            new_stage = 3
        elif self.num_timesteps >= self._stage2_at and self._current_stage < 2:
            new_stage = 2

        if new_stage != self._current_stage:
            self._current_stage = new_stage
            if new_stage == 2:
                s_cfg = make_light_sensor_config(self._pack_config)
                a_cfg = make_light_actuator_config(self._pack_config)
            else:  # stage 3
                s_cfg = make_sensor_config(self._pack_config)
                a_cfg = make_actuator_config(self._pack_config)

            self.training_env.env_method("update_sensor_config", s_cfg, a_cfg)
            if self.verbose:
                print(f"\n  [Curriculum] *** Stage {new_stage} at {self.num_timesteps:,} steps ***")
        return True


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def make_train_env(seed: int, use_curriculum_stage1: bool = False):
    """
    Build one training env.

    use_curriculum_stage1=True starts with perfect sensors for curriculum training.
    Default (False) starts with full combined-realism sensors.
    """
    def _init():
        cfg = make_pack_config(seed=seed)
        s_cfg = make_perfect_sensor_config(cfg) if use_curriculum_stage1 else make_sensor_config(cfg)
        a_cfg = make_perfect_actuator_config(cfg) if use_curriculum_stage1 else make_actuator_config(cfg)
        env = BatteryPackThermalEnv3D(
            cell_config=CellConfig(),
            pack_config=cfg,
            heat_profile=randomized_3d_heat_profile(),
            seed=seed,
            enable_sensor_simulation=True,
            sensor_config=s_cfg,
            actuator_config=a_cfg,
        )
        return Monitor(env)
    return _init


def make_eval_env(seed: int):
    def _init():
        cfg = make_pack_config(seed=seed)
        env = BatteryPackThermalEnv3D(
            cell_config=CellConfig(),
            pack_config=cfg,
            heat_profile=randomized_3d_heat_profile(),
            seed=seed,
            enable_sensor_simulation=True,
            sensor_config=make_sensor_config(cfg),
            actuator_config=make_actuator_config(cfg),
        )
        return Monitor(env)
    return _init


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
    model_dir = Path(args.save_dir) if args.save_dir else PROJECT_ROOT / "models" / "ppo_pack_3d_multizone_sensor"
    log_dir   = Path(args.log_dir)  if args.log_dir  else PROJECT_ROOT / "logs"   / "ppo_pack_3d_multizone_sensor"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "checkpoints").mkdir(exist_ok=True)

    # Env check
    _chk_cfg = make_pack_config(seed=123)
    check_env(BatteryPackThermalEnv3D(
        cell_config=CellConfig(),
        pack_config=_chk_cfg,
        heat_profile=randomized_3d_heat_profile(),
        seed=123,
        enable_sensor_simulation=True,
        sensor_config=make_sensor_config(_chk_cfg),
        actuator_config=make_actuator_config(_chk_cfg),
    ), warn=True)
    print("3D pack env (sensor sim, 29D obs) passed SB3 check.")

    use_curriculum = bool(getattr(args, "curriculum", False))
    pack_cfg = make_pack_config()

    # Environments
    n_envs    = args.n_envs
    train_env = DummyVecEnv([make_train_env(seed=100 + i, use_curriculum_stage1=use_curriculum) for i in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0, gamma=0.995)

    eval_env = DummyVecEnv([make_eval_env(seed=900)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, gamma=0.995)
    eval_env.obs_rms  = train_env.obs_rms
    eval_env.training = False

    # Callbacks
    callbacks = []
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir / "eval"),
        eval_freq=max(25_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks += [eval_callback, ProgressCallback()]
    callbacks.append(CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_pack_checkpoint",
        verbose=1,
    ))
    if use_curriculum:
        callbacks.append(CurriculumCallback(
            pack_config=pack_cfg,
            stage2_at=args.stage2_at,
            stage3_at=args.stage3_at,
            verbose=1,
        ))
        print(f"Curriculum: Stage 1→2 at {args.stage2_at:,}, Stage 2→3 at {args.stage3_at:,}")

    # Model — new or resumed
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        model = PPO.load(args.resume_from, env=train_env, device=device)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=1024,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.002,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=args.seed,
            device=device,
            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        )

    print(f"\nPPO training — {args.timesteps:,} timesteps  curriculum={use_curriculum}")
    print(f"Pack shape: {pack_cfg.shape}  ({int(np.prod(pack_cfg.shape))} cells)  "
          f"n_envs: {n_envs}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name="PPO_pack_3d",
        progress_bar=True,
        reset_num_timesteps=not bool(args.resume_from),
    )

    final_path   = model_dir / "ppo_pack_final.zip"
    vecnorm_path = model_dir / "vec_normalize.pkl"
    model.save(str(final_path))
    train_env.save(str(vecnorm_path))

    print("\nTraining complete.")
    print(f"  best_model:    {model_dir / 'best_model.zip'}")
    print(f"  final_model:   {final_path}")
    print(f"  vec_normalize: {vecnorm_path}")
    print(f"  checkpoints:   {model_dir / 'checkpoints'}/")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
