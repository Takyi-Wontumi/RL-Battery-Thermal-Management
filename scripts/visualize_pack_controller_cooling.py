"""
scripts/visualize_pack_controller_cooling.py

General cooling visualization for the multi-node battery pack environment.
Works for both classical pack controllers and trained pack PPO models.

Run from project root:
    python -m scripts.visualize_pack_controller_cooling

Outputs:
    outputs/pack_visualize_<controller>_<profile>.png
    outputs/pack_visualize_<controller>_<profile>.csv

Supported controllers:
    no_cooling
    constant_05
    constant_10
    bang_bang
    proportional
    pi_tuned
    ppo_best
    ppo_final

Supported profiles:
    UniformConstant
    NonuniformStep
    PulsedHotspot
    RandomNonuniform
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envs.battery_pack_thermal_env import (
    BatteryPackThermalConfig,
    BatteryPackThermalEnv,
    make_pack_profile,
    uniform_constant_pack_heat,
)
from scripts.compare_pack_baselines import (
    PackBangBangController,
    PackConstantCoolingController,
    PackNoCoolingController,
    PackPIController,
    PackProportionalController,
)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class PackControllerLike(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        ...


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

def make_eval_config(seed: int = 7) -> BatteryPackThermalConfig:
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
        seed=seed,
    )


# -----------------------------------------------------------------------------
# PPO adapter
# -----------------------------------------------------------------------------

class PackPPOController:
    def __init__(self, model: PPO, vec_normalize: VecNormalize, name: str) -> None:
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


def make_dummy_vec_env(config: BatteryPackThermalConfig) -> DummyVecEnv:
    def _init():
        env = BatteryPackThermalEnv(
            config=config,
            heat_profile=uniform_constant_pack_heat(),
            render_mode=None,
        )
        return Monitor(env)

    return DummyVecEnv([_init])


def load_pack_ppo_controller(config: BatteryPackThermalConfig, model_name: str) -> PackPPOController:
    if not SB3_AVAILABLE:
        raise ImportError(
            "Stable-Baselines3 is required for PPO visualization. Install with: "
            "python -m pip install 'stable-baselines3[extra]>=2.3.0' tensorboard"
        )

    model_dir = PROJECT_ROOT / "models" / "ppo_battery_pack_thermal"
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
    pretty_name = "Pack PPO final model" if model_name == "final_model" else "Pack PPO best model"

    return PackPPOController(model=model, vec_normalize=vec_normalize, name=pretty_name)


# -----------------------------------------------------------------------------
# Controller factory
# -----------------------------------------------------------------------------

def make_controller(controller_key: str, config: BatteryPackThermalConfig) -> PackControllerLike:
    controller_key = controller_key.lower().strip()

    if controller_key == "no_cooling":
        return PackNoCoolingController()

    if controller_key == "constant_05":
        return PackConstantCoolingController(cooling_level=0.5)

    if controller_key == "constant_10":
        return PackConstantCoolingController(cooling_level=1.0)

    if controller_key == "bang_bang":
        return PackBangBangController(
            config=config,
            target_temp=config.target_temp,
            deadband=1.0,
            name="Pack max-temp bang-bang",
        )

    if controller_key == "proportional":
        return PackProportionalController(
            config=config,
            target_temp=config.target_temp,
            kp=0.10,
            bias=0.15,
            imbalance_gain=0.04,
            name="Pack max-temp proportional",
        )

    if controller_key == "pi_tuned":
        return PackPIController(
            config=config,
            target_temp=config.target_temp,
            kp=0.30,
            ki=0.001,
            bias=0.30,
            imbalance_gain=0.08,
            integral_limit=50.0,
            name="Pack max-temp PI tuned",
        )

    if controller_key == "ppo_best":
        return load_pack_ppo_controller(config=config, model_name="best_model")

    if controller_key == "ppo_final":
        return load_pack_ppo_controller(config=config, model_name="final_model")

    raise KeyError(
        f"Unknown controller '{controller_key}'. Choose from: no_cooling, constant_05, "
        "constant_10, bang_bang, proportional, pi_tuned, ppo_best, ppo_final"
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def sanitize_filename(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("=", "")
        .replace(".", "")
        .replace("-", "_")
        .replace("/", "_")
    )


def cooling_coefficient(config: BatteryPackThermalConfig, u: float) -> float:
    shaped_u = u ** config.cooling_nonlinearity
    return float(config.h_min + (config.h_max - config.h_min) * shaped_u)


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

def run_controller(
    controller: PackControllerLike,
    profile_name: str,
    config: BatteryPackThermalConfig,
    seed: int = 7,
) -> pd.DataFrame:
    env = BatteryPackThermalEnv(
        config=config,
        heat_profile=make_pack_profile(profile_name),
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

        pre_temperatures = np.asarray(info["temperatures"], dtype=np.float32)
        pre_heat = np.asarray(info["heat_generation"], dtype=np.float32)
        ambient = float(info["ambient_temperature"])

        h_coeff = cooling_coefficient(config, u)
        cooling_per_cell = h_coeff * config.surface_area_per_cell * (pre_temperatures - ambient)
        cooling_per_cell += config.direct_cooling_max_per_cell * u
        total_cooling_power = float(np.sum(cooling_per_cell))

        obs, reward, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))
        total_reward += float(reward)

        temps = np.asarray(info["temperatures"], dtype=np.float32)
        heat = np.asarray(info["heat_generation"], dtype=np.float32)

        row: Dict[str, float | str] = {
            "time_s": float(info["time"]),
            "controller": controller.name,
            "profile": profile_name,
            "cooling_command_u": u,
            "h_coeff_W_m2K": h_coeff,
            "total_cooling_power_W": total_cooling_power,
            "ambient_temperature_C": float(info["ambient_temperature"]),
            "total_heat_generation_W": float(info["total_heat_generation"]),
            "max_cell_temperature_C": float(np.max(temps)),
            "mean_pack_temperature_C": float(np.mean(temps)),
            "min_cell_temperature_C": float(np.min(temps)),
            "temperature_spread_C": float(np.max(temps) - np.min(temps)),
            "temperature_std_C": float(np.std(temps)),
            "reward": float(reward),
            "cumulative_reward": total_reward,
        }

        for i, temp in enumerate(temps):
            row[f"cell_{i + 1}_temperature_C"] = float(temp)
        for i, q in enumerate(heat):
            row[f"cell_{i + 1}_heat_generation_W"] = float(q)
        for i, qcool in enumerate(cooling_per_cell):
            row[f"cell_{i + 1}_cooling_power_W"] = float(qcool)

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_pack_controller(df: pd.DataFrame, config: BatteryPackThermalConfig, output_path: Path) -> None:
    controller_name = str(df["controller"].iloc[0])
    profile_name = str(df["profile"].iloc[0])
    time = df["time_s"]

    cell_temp_cols = [col for col in df.columns if col.startswith("cell_") and col.endswith("_temperature_C")]
    cell_heat_cols = [col for col in df.columns if col.startswith("cell_") and col.endswith("_heat_generation_W")]
    cell_cooling_cols = [col for col in df.columns if col.startswith("cell_") and col.endswith("_cooling_power_W")]

    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(13, 15), sharex=True)
    fig.suptitle(
        f"Pack cooling behavior — {controller_name} on {profile_name}",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Heat generation.
    axes[0].plot(time, df["total_heat_generation_W"], linewidth=1.8, label="Total pack heat")
    if cell_heat_cols:
        axes[0].plot(time, df[cell_heat_cols].max(axis=1), linestyle="--", linewidth=1.0, label="Max cell heat")
        axes[0].plot(time, df[cell_heat_cols].mean(axis=1), linestyle=":", linewidth=1.0, label="Mean cell heat")
    axes[0].set_ylabel("Heat (W)")
    axes[0].set_title("Pack heat generation")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    # 2. Cooling command.
    axes[1].plot(time, df["cooling_command_u"], linewidth=1.6)
    axes[1].set_ylabel("u")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Shared cooling command")
    axes[1].grid(True, alpha=0.25)

    # 3. Cooling power.
    axes[2].plot(time, df["total_cooling_power_W"], linewidth=1.8, label="Total cooling power")
    if cell_cooling_cols:
        axes[2].plot(time, df[cell_cooling_cols].max(axis=1), linestyle="--", linewidth=1.0, label="Max cell cooling")
        axes[2].plot(time, df[cell_cooling_cols].mean(axis=1), linestyle=":", linewidth=1.0, label="Mean cell cooling")
    axes[2].set_ylabel("Cooling (W)")
    axes[2].set_title("Cooling power removed")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.25)

    # 4. Max/mean/min pack temperature.
    axes[3].plot(time, df["max_cell_temperature_C"], linewidth=1.8, label="Max cell")
    axes[3].plot(time, df["mean_pack_temperature_C"], linewidth=1.4, label="Mean pack")
    axes[3].plot(time, df["min_cell_temperature_C"], linewidth=1.2, label="Min cell")
    axes[3].axhline(config.target_temp, linestyle=":", linewidth=1.0, label=f"Target {config.target_temp:.0f}°C")
    axes[3].axhline(config.soft_max_temp, linestyle="--", linewidth=1.0, label=f"Safe limit {config.soft_max_temp:.0f}°C")
    axes[3].set_ylabel("Temp (°C)")
    axes[3].set_title("Pack temperature envelope")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.25)

    # 5. Individual cell temperatures.
    for col in cell_temp_cols:
        axes[4].plot(time, df[col], linewidth=0.9, alpha=0.85)
    axes[4].axhline(config.soft_max_temp, linestyle="--", linewidth=1.0)
    axes[4].set_ylabel("Cell temp (°C)")
    axes[4].set_title("Individual cell/module temperatures")
    axes[4].grid(True, alpha=0.25)

    # 6. Temperature imbalance.
    axes[5].plot(time, df["temperature_spread_C"], linewidth=1.7, label="Max-min spread")
    axes[5].plot(time, df["temperature_std_C"], linewidth=1.2, label="Temperature std")
    axes[5].set_ylabel("Imbalance (°C)")
    axes[5].set_xlabel("Time (s)")
    axes[5].set_title("Pack temperature imbalance")
    axes[5].legend(fontsize=8)
    axes[5].grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(df: pd.DataFrame, config: BatteryPackThermalConfig) -> None:
    controller_name = str(df["controller"].iloc[0])
    profile_name = str(df["profile"].iloc[0])

    max_temp = float(df["max_cell_temperature_C"].max())
    mean_temp = float(df["mean_pack_temperature_C"].mean())
    max_spread = float(df["temperature_spread_C"].max())
    mean_spread = float(df["temperature_spread_C"].mean())
    effort = float(df["cooling_command_u"].sum() * config.dt)
    mean_u = float(df["cooling_command_u"].mean())
    mean_q_cool = float(df["total_cooling_power_W"].mean())
    peak_q_cool = float(df["total_cooling_power_W"].max())
    total_reward = float(df["reward"].sum())
    time_above_safe = float((df["max_cell_temperature_C"] > config.soft_max_temp).sum() * config.dt)

    print("\nPack visualization summary")
    print(f"  Controller:             {controller_name}")
    print(f"  Profile:                {profile_name}")
    print(f"  Max cell temperature:    {max_temp:.2f} C")
    print(f"  Mean pack temperature:   {mean_temp:.2f} C")
    print(f"  Max temperature spread:  {max_spread:.2f} C")
    print(f"  Mean temperature spread: {mean_spread:.2f} C")
    print(f"  Time above safe:         {time_above_safe:.1f} s")
    print(f"  Mean cooling command:    {mean_u:.3f}")
    print(f"  Cooling effort:          {effort:.1f}")
    print(f"  Mean cooling power:      {mean_q_cool:.2f} W")
    print(f"  Peak cooling power:      {peak_q_cool:.2f} W")
    print(f"  Total reward:            {total_reward:.2f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Change these two lines for different cases.
    controller_key = "ppo_final"
    profile_name = "NonuniformStep"

    # Good comparison cases:
    #   controller_key = "no_cooling"     profile_name = "NonuniformStep"
    #   controller_key = "pi_tuned"       profile_name = "NonuniformStep"
    #   controller_key = "ppo_final"      profile_name = "NonuniformStep"
    #   controller_key = "bang_bang"      profile_name = "PulsedHotspot"
    #   controller_key = "ppo_best"       profile_name = "RandomNonuniform"

    config = make_eval_config(seed=7)
    controller = make_controller(controller_key=controller_key, config=config)

    df = run_controller(
        controller=controller,
        profile_name=profile_name,
        config=config,
        seed=7,
    )

    controller_slug = sanitize_filename(controller.name)
    profile_slug = sanitize_filename(profile_name)

    csv_path = output_dir / f"pack_visualize_{controller_slug}_{profile_slug}.csv"
    png_path = output_dir / f"pack_visualize_{controller_slug}_{profile_slug}.png"

    df.to_csv(csv_path, index=False)
    plot_pack_controller(df=df, config=config, output_path=png_path)
    print_summary(df=df, config=config)

    print("\nSaved pack visualization:")
    print(f"  {png_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
