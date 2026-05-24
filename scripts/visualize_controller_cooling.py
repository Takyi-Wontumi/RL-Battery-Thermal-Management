"""
General cooling visualization for both PPO and classical controllers.

It visualizes:
    - heat generation Q_gen(t)
    - cooling command u(t)
    - heat transfer coefficient h(u)
    - cooling power Q_cool(t)
    - battery temperature T(t)
    - cumulative reward

Run from project root:
    python -m scripts.visualize_controller_cooling

Outputs:
    outputs/visualize_<controller>_<profile>.png
    outputs/visualize_<controller>_<profile>.csv

This is the script to use when explaining what the controller actually did.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from controllers.baseline_controllers import (
    BangBangController,
    ConstantCoolingController,
    NoCoolingController,
    PIController,
    ProportionalController,
)
from envs.battery_thermal_env import BatteryThermalConfig, BatteryThermalEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


HeatProfile = Callable[[float, np.random.Generator], float]


class ControllerLike(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        ...


# -----------------------------------------------------------------------------
# Heat profiles: match Phase 2 / Phase 3 benchmark
# -----------------------------------------------------------------------------

def constant_heat_profile(q_gen: float = 700.0) -> HeatProfile:
    def profile(t: float, rng: np.random.Generator) -> float:
        return float(q_gen)

    return profile


def step_heat_profile(
    q_low: float = 400.0,
    q_high: float = 1_100.0,
    step_time: float = 500.0,
) -> HeatProfile:
    def profile(t: float, rng: np.random.Generator) -> float:
        return float(q_low if t < step_time else q_high)

    return profile


def pulsed_heat_profile(
    q_low: float = 300.0,
    q_high: float = 1_200.0,
    period: float = 160.0,
    duty_cycle: float = 0.40,
) -> HeatProfile:
    def profile(t: float, rng: np.random.Generator) -> float:
        phase = (t % period) / period
        return float(q_high if phase < duty_cycle else q_low)

    return profile


def random_heat_profile(
    q_mean: float = 650.0,
    q_std: float = 18.0,
    smoothing: float = 0.88,
) -> HeatProfile:
    state = {"q": q_mean}

    def profile(t: float, rng: np.random.Generator) -> float:
        disturbance = rng.normal(0.0, q_std)
        state["q"] = smoothing * state["q"] + (1.0 - smoothing) * q_mean + disturbance
        return float(np.clip(state["q"], 450.0, 900.0))

    return profile


def make_profile(profile_name: str) -> HeatProfile:
    profiles: Dict[str, HeatProfile] = {
        "Constant": constant_heat_profile(),
        "Step": step_heat_profile(),
        "Pulsed": pulsed_heat_profile(),
        "Random": random_heat_profile(),
    }

    if profile_name not in profiles:
        raise KeyError(
            f"Unknown profile '{profile_name}'. Choose from: {list(profiles.keys())}"
        )

    return profiles[profile_name]


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

def make_eval_config(seed: Optional[int] = 7) -> BatteryThermalConfig:
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


# -----------------------------------------------------------------------------
# PPO adapter
# -----------------------------------------------------------------------------

class PPOController:
    """Adapter that makes a Stable-Baselines3 PPO model act like a controller."""

    def __init__(self, model, vec_normalize, name: str = "PPO final model") -> None:
        self.model = model
        self.vec_normalize = vec_normalize
        self.name = name

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        norm_obs = self.vec_normalize.normalize_obs(obs_batch)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        u = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        return np.array([u], dtype=np.float32)


def make_dummy_vec_env(config: BatteryThermalConfig):
    def _init():
        env = BatteryThermalEnv(
            config=config,
            heat_profile=constant_heat_profile(),
            render_mode=None,
        )
        return Monitor(env)

    return DummyVecEnv([_init])


def load_ppo_controller(config: BatteryThermalConfig, model_name: str = "final_model") -> PPOController:
    if not SB3_AVAILABLE:
        raise ImportError(
            "Stable-Baselines3 is not installed. Install it with: "
            "python -m pip install 'stable-baselines3[extra]>=2.3.0' tensorboard"
        )

    model_dir = PROJECT_ROOT / "models" / "ppo_battery_thermal"
    model_path = model_dir / f"{model_name}.zip"
    vecnorm_path = model_dir / "vec_normalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing PPO model: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"Missing VecNormalize stats: {vecnorm_path}")

    dummy_env = make_dummy_vec_env(config)
    vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False

    model = PPO.load(str(model_path), env=None, device="auto")

    pretty_name = "PPO final model" if model_name == "final_model" else "PPO best model"
    return PPOController(model=model, vec_normalize=vec_normalize, name=pretty_name)


# -----------------------------------------------------------------------------
# Controller factory
# -----------------------------------------------------------------------------

def make_controller(controller_name: str, config: BatteryThermalConfig) -> ControllerLike:
    """
    Create any supported controller by name.

    Supported names:
        no_cooling
        constant_05
        constant_10
        bang_bang
        proportional
        pi_tuned
        ppo_final
        ppo_best
    """
    controller_name = controller_name.lower().strip()

    if controller_name == "no_cooling":
        return NoCoolingController()

    if controller_name == "constant_05":
        return ConstantCoolingController(cooling_level=0.5)

    if controller_name == "constant_10":
        return ConstantCoolingController(cooling_level=1.0)

    if controller_name == "bang_bang":
        return BangBangController(target_temp=config.target_temp, deadband=1.0)

    if controller_name == "proportional":
        return ProportionalController(target_temp=config.target_temp, kp=0.08, bias=0.15)

    if controller_name == "pi_tuned":
        return PIController(
            target_temp=config.target_temp,
            kp=0.20,
            ki=0.006,
            bias=0.25,
            dt=config.dt,
            integral_limit=75.0,
            name="PI controller tuned",
        )

    if controller_name == "ppo_final":
        return load_ppo_controller(config=config, model_name="final_model")

    if controller_name == "ppo_best":
        return load_ppo_controller(config=config, model_name="best_model")

    raise KeyError(
        f"Unknown controller '{controller_name}'. Choose from: "
        "no_cooling, constant_05, constant_10, bang_bang, proportional, "
        "pi_tuned, ppo_final, ppo_best"
    )


# -----------------------------------------------------------------------------
# Physics helpers
# -----------------------------------------------------------------------------

def cooling_coefficient(config: BatteryThermalConfig, u: float) -> float:
    shaped_u = u ** config.cooling_nonlinearity
    return float(config.h_min + (config.h_max - config.h_min) * shaped_u)


def sanitize_filename(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("=", "")
        .replace(".", "")
        .replace("-", "_")
        .replace("/", "_")
    )


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

def run_controller_on_profile(
    controller: ControllerLike,
    profile_name: str,
    config: BatteryThermalConfig,
    seed: int = 7,
) -> pd.DataFrame:
    env = BatteryThermalEnv(
        config=config,
        heat_profile=make_profile(profile_name),
        render_mode=None,
    )

    controller.reset()
    obs, info = env.reset(seed=seed, options={"randomize": False})

    terminated = False
    truncated = False
    total_reward = 0.0
    rows = []

    while not (terminated or truncated):
        u = float(np.clip(np.asarray(controller.act(obs)).reshape(-1)[0], 0.0, 1.0))

        # Compute physical cooling quantities using current state before stepping.
        current_temp = float(obs[0])
        current_ambient = float(obs[1])
        current_heat = float(obs[2])

        h_coeff = cooling_coefficient(config, u)
        convective_cooling_W = h_coeff * config.surface_area * (current_temp - current_ambient)
        direct_cooling_W = config.direct_cooling_max * u
        total_cooling_W = convective_cooling_W + direct_cooling_W

        next_obs, reward, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))
        total_reward += reward

        rows.append(
            {
                "time_s": info["time"],
                "controller": controller.name,
                "profile": profile_name,
                "temperature_C": info["temperature"],
                "ambient_temperature_C": info["ambient_temperature"],
                "heat_generation_W": info["heat_generation"],
                "cooling_command_u": u,
                "h_coeff_W_m2K": h_coeff,
                "convective_cooling_W": convective_cooling_W,
                "direct_cooling_W": direct_cooling_W,
                "total_cooling_W": total_cooling_W,
                "reward": reward,
                "cumulative_reward": total_reward,
                "pre_step_temperature_C": current_temp,
                "pre_step_ambient_C": current_ambient,
                "pre_step_heat_generation_W": current_heat,
            }
        )

        obs = next_obs

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_single_controller(df: pd.DataFrame, config: BatteryThermalConfig, output_path: Path) -> None:
    controller_name = str(df["controller"].iloc[0])
    profile_name = str(df["profile"].iloc[0])

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 12), sharex=True)

    fig.suptitle(
        f"Cooling behavior — {controller_name} on {profile_name} heat profile",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].plot(df["time_s"], df["heat_generation_W"], linewidth=1.5)
    axes[0].set_ylabel("Q_gen (W)")
    axes[0].set_title("Heat generation demand")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(df["time_s"], df["cooling_command_u"], linewidth=1.5)
    axes[1].set_ylabel("u")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Cooling command")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(df["time_s"], df["h_coeff_W_m2K"], linewidth=1.5)
    axes[2].set_ylabel("h(u) W/m²K")
    axes[2].set_title("Effective heat-transfer coefficient")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(df["time_s"], df["total_cooling_W"], linewidth=1.5)
    axes[3].set_ylabel("Q_cool (W)")
    axes[3].set_title("Cooling power removed")
    axes[3].grid(True, alpha=0.25)

    axes[4].plot(df["time_s"], df["temperature_C"], linewidth=1.5, label="Battery temperature")
    axes[4].axhline(config.target_temp, linestyle=":", linewidth=1.0, label=f"Target {config.target_temp:.0f}°C")
    axes[4].axhline(config.soft_max_temp, linestyle="--", linewidth=1.0, label=f"Safe limit {config.soft_max_temp:.0f}°C")
    axes[4].set_ylabel("T_batt (°C)")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_title("Battery temperature response")
    axes[4].legend(fontsize=8)
    axes[4].grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(df: pd.DataFrame, config: BatteryThermalConfig) -> None:
    controller_name = str(df["controller"].iloc[0])
    profile_name = str(df["profile"].iloc[0])

    max_temp = float(df["temperature_C"].max())
    mean_temp = float(df["temperature_C"].mean())
    total_effort = float(df["cooling_command_u"].sum() * config.dt)
    mean_u = float(df["cooling_command_u"].mean())
    mean_q_cool = float(df["total_cooling_W"].mean())
    peak_q_cool = float(df["total_cooling_W"].max())
    total_reward = float(df["reward"].sum())
    time_above_safe = float((df["temperature_C"] > config.soft_max_temp).sum() * config.dt)

    print("\nVisualization summary")
    print(f"  Controller:           {controller_name}")
    print(f"  Profile:              {profile_name}")
    print(f"  Max temperature:       {max_temp:.2f} C")
    print(f"  Mean temperature:      {mean_temp:.2f} C")
    print(f"  Time above safe:       {time_above_safe:.1f} s")
    print(f"  Mean cooling command:  {mean_u:.3f}")
    print(f"  Total cooling effort:  {total_effort:.1f}")
    print(f"  Mean cooling power:    {mean_q_cool:.2f} W")
    print(f"  Peak cooling power:    {peak_q_cool:.2f} W")
    print(f"  Total reward:          {total_reward:.2f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Change these two values to visualize different policies and heat profiles.
    controller_name = "ppo_final"
    profile_name = "Step"

    # Supported controllers:
    #   no_cooling, constant_05, constant_10, bang_bang,
    #   proportional, pi_tuned, ppo_final, ppo_best
    # Supported profiles:
    #   Constant, Step, Pulsed, Random

    config = make_eval_config(seed=7)
    controller = make_controller(controller_name=controller_name, config=config)

    df = run_controller_on_profile(
        controller=controller,
        profile_name=profile_name,
        config=config,
        seed=7,
    )

    controller_slug = sanitize_filename(controller.name)
    profile_slug = sanitize_filename(profile_name)

    csv_path = output_dir / f"visualize_{controller_slug}_{profile_slug}.csv"
    png_path = output_dir / f"visualize_{controller_slug}_{profile_slug}.png"

    df.to_csv(csv_path, index=False)
    plot_single_controller(df=df, config=config, output_path=png_path)
    print_summary(df=df, config=config)

    print("\nSaved visualization:")
    print(f"  {png_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
