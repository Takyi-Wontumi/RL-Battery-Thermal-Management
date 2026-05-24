"""
scripts/animate_battery_temperature_field.py

Create a 2D contour-style animation of battery/cell temperature changing over time.

Important limitation:
    The current BatteryThermalEnv is a lumped thermal model, not a real 2D/3D heat
    conduction model. This script creates an illustrative pseudo-spatial field using
    the simulated average battery temperature, cooling command, and heat generation.

The visualization shows:
    - circular battery/cell cross-section
    - warmer center under heat generation
    - cooler outer boundary under cooling
    - time-varying contour animation

Run from project root:
    python -m scripts.animate_battery_temperature_field

Outputs:
    outputs/battery_temperature_field_<controller>_<profile>.gif
    outputs/battery_temperature_field_<controller>_<profile>.mp4  optional if ffmpeg exists
    outputs/battery_temperature_field_<controller>_<profile>.png  final frame

Supported controllers:
    no_cooling, constant_05, constant_10, bang_bang, proportional, pi_tuned,
    ppo_final, ppo_best

Supported profiles:
    Constant, Step, Pulsed, Random
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

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
# Benchmark heat profiles
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
        raise KeyError(f"Unknown profile '{profile_name}'. Choose from {list(profiles)}")
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
        action_weight=0.03,
        action_smoothness_weight=0.08,
        hard_violation_penalty=100.0,
        seed=seed,
    )


# -----------------------------------------------------------------------------
# PPO adapter
# -----------------------------------------------------------------------------

class PPOController:
    def __init__(self, model, vec_normalize, name: str) -> None:
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
        env = BatteryThermalEnv(config=config, heat_profile=constant_heat_profile(), render_mode=None)
        return Monitor(env)

    return DummyVecEnv([_init])


def load_ppo_controller(config: BatteryThermalConfig, model_name: str) -> PPOController:
    if not SB3_AVAILABLE:
        raise ImportError(
            "Stable-Baselines3 is required for PPO visualization. Install with: "
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
    name = "PPO final model" if model_name == "final_model" else "PPO best model"
    return PPOController(model=model, vec_normalize=vec_normalize, name=name)


# -----------------------------------------------------------------------------
# Controller factory
# -----------------------------------------------------------------------------

def make_controller(controller_name: str, config: BatteryThermalConfig) -> ControllerLike:
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
        f"Unknown controller '{controller_name}'. Choose: no_cooling, constant_05, "
        "constant_10, bang_bang, proportional, pi_tuned, ppo_final, ppo_best"
    )


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

def run_controller(
    controller: ControllerLike,
    profile_name: str,
    config: BatteryThermalConfig,
    seed: int = 7,
) -> Dict[str, np.ndarray]:
    env = BatteryThermalEnv(config=config, heat_profile=make_profile(profile_name), render_mode=None)

    controller.reset()
    obs, info = env.reset(seed=seed, options={"randomize": False})

    terminated = False
    truncated = False
    total_reward = 0.0

    rows = {
        "time_s": [],
        "temperature_C": [],
        "ambient_temperature_C": [],
        "heat_generation_W": [],
        "cooling_command_u": [],
        "cooling_power_W": [],
        "reward": [],
        "cumulative_reward": [],
    }

    while not (terminated or truncated):
        u = float(np.clip(np.asarray(controller.act(obs)).reshape(-1)[0], 0.0, 1.0))

        temp = float(obs[0])
        ambient = float(obs[1])
        h_coeff = config.h_min + (config.h_max - config.h_min) * (u ** config.cooling_nonlinearity)
        cooling_power = h_coeff * config.surface_area * (temp - ambient) + config.direct_cooling_max * u

        next_obs, reward, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))
        total_reward += reward

        rows["time_s"].append(float(info["time"]))
        rows["temperature_C"].append(float(info["temperature"]))
        rows["ambient_temperature_C"].append(float(info["ambient_temperature"]))
        rows["heat_generation_W"].append(float(info["heat_generation"]))
        rows["cooling_command_u"].append(u)
        rows["cooling_power_W"].append(float(cooling_power))
        rows["reward"].append(float(reward))
        rows["cumulative_reward"].append(float(total_reward))

        obs = next_obs

    return {key: np.asarray(value) for key, value in rows.items()}


# -----------------------------------------------------------------------------
# Pseudo-spatial field
# -----------------------------------------------------------------------------

def create_battery_grid(n: int = 160):
    """Create a circular battery/cell cross-section grid."""
    x = np.linspace(-1.15, 1.15, n)
    y = np.linspace(-1.15, 1.15, n)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    mask = R <= 1.0
    return X, Y, R, mask


def pseudo_temperature_field(
    mean_temp_C: float,
    ambient_C: float,
    heat_W: float,
    cooling_u: float,
    R: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Convert one lumped temperature into an illustrative 2D radial field.

    This is not CFD. It creates a plausible-looking radial gradient:
        - center gets hotter with high heat generation
        - edge gets cooler with high cooling command
        - field average tracks the lumped environment temperature
    """
    heat_intensity = np.clip((heat_W - 300.0) / 900.0, 0.0, 1.0)
    cooling_intensity = np.clip(cooling_u, 0.0, 1.0)

    # Center hot spot term. Stronger when heat generation is high.
    center_hotspot = (1.5 + 2.8 * heat_intensity) * (1.0 - R**2)

    # Edge cooling term. Stronger near outer radius when cooling command is high.
    edge_cooling = (0.8 + 2.6 * cooling_intensity) * (R**1.8)

    # Keep the field anchored around mean_temp_C.
    T_field = mean_temp_C + center_hotspot - edge_cooling

    # Do not let cooled edge unrealistically drop too far below ambient.
    T_field = np.maximum(T_field, ambient_C - 0.5)

    T_field = np.where(mask, T_field, np.nan)
    return T_field


# -----------------------------------------------------------------------------
# Animation
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


def animate_temperature_field(
    data: Dict[str, np.ndarray],
    controller_name: str,
    profile_name: str,
    output_dir: Path,
    stride: int = 8,
) -> None:
    X, Y, R, mask = create_battery_grid(n=170)

    frame_indices = np.arange(0, len(data["time_s"]), stride)
    if frame_indices[-1] != len(data["time_s"]) - 1:
        frame_indices = np.append(frame_indices, len(data["time_s"]) - 1)

    all_temps = []
    for idx in frame_indices:
        all_temps.append(
            pseudo_temperature_field(
                mean_temp_C=float(data["temperature_C"][idx]),
                ambient_C=float(data["ambient_temperature_C"][idx]),
                heat_W=float(data["heat_generation_W"][idx]),
                cooling_u=float(data["cooling_command_u"][idx]),
                R=R,
                mask=mask,
            )
        )

    global_min = min(np.nanmin(T) for T in all_temps)
    global_max = max(np.nanmax(T) for T in all_temps)
    levels = np.linspace(global_min, global_max, 32)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax_field, ax_signals = axes

    fig.suptitle(
        f"Battery temperature field animation — {controller_name} on {profile_name}",
        fontsize=13,
        fontweight="bold",
    )

    # Static signal plot on the right.
    time = data["time_s"]
    ax_temp = ax_signals
    ax_u = ax_signals.twinx()

    ax_temp.plot(time, data["temperature_C"], linewidth=1.5, label="T_batt")
    ax_temp.axhline(35.0, linestyle=":", linewidth=1.0, label="Target")
    ax_temp.axhline(45.0, linestyle="--", linewidth=1.0, label="Safe limit")
    ax_u.plot(time, data["cooling_command_u"], linewidth=1.0, alpha=0.75, label="u")

    moving_line = ax_temp.axvline(time[0], linewidth=1.2)

    ax_temp.set_xlabel("Time (s)")
    ax_temp.set_ylabel("Temperature (°C)")
    ax_u.set_ylabel("Cooling command u")
    ax_u.set_ylim(-0.05, 1.05)
    ax_temp.grid(True, alpha=0.25)

    lines_1, labels_1 = ax_temp.get_legend_handles_labels()
    lines_2, labels_2 = ax_u.get_legend_handles_labels()
    ax_temp.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=8, loc="upper right")

    contour_holder = {"contour": None, "colorbar": None}

    def update(frame_number: int):
        idx = int(frame_indices[frame_number])

        T_field = pseudo_temperature_field(
            mean_temp_C=float(data["temperature_C"][idx]),
            ambient_C=float(data["ambient_temperature_C"][idx]),
            heat_W=float(data["heat_generation_W"][idx]),
            cooling_u=float(data["cooling_command_u"][idx]),
            R=R,
            mask=mask,
        )

        ax_field.clear()
        contour = ax_field.contourf(X, Y, T_field, levels=levels, cmap="inferno")
        ax_field.contour(X, Y, R, levels=[1.0], colors="black", linewidths=1.5)
        ax_field.set_aspect("equal")
        ax_field.set_xticks([])
        ax_field.set_yticks([])
        ax_field.set_title(
            f"t = {data['time_s'][idx]:.0f} s | "
            f"Tavg = {data['temperature_C'][idx]:.2f}°C | "
            f"u = {data['cooling_command_u'][idx]:.2f}"
        )

        if contour_holder["colorbar"] is None:
            contour_holder["colorbar"] = fig.colorbar(contour, ax=ax_field, fraction=0.046, pad=0.04)
            contour_holder["colorbar"].set_label("Temperature (°C)")

        moving_line.set_xdata([data["time_s"][idx], data["time_s"][idx]])
        return []

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=80, blit=False)

    controller_slug = sanitize_filename(controller_name)
    profile_slug = sanitize_filename(profile_name)

    gif_path = output_dir / f"battery_temperature_field_{controller_slug}_{profile_slug}.gif"
    png_path = output_dir / f"battery_temperature_field_{controller_slug}_{profile_slug}_final_frame.png"

    anim.save(gif_path, writer=PillowWriter(fps=12))

    # Save final frame as PNG.
    update(len(frame_indices) - 1)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved contour-style visualization:")
    print(f"  {gif_path}")
    print(f"  {png_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Change these to visualize different cases.
    controller_key = "ppo_final"
    profile_name = "Step"

    # Good examples:
    #   controller_key = "no_cooling"   profile_name = "Step"
    #   controller_key = "pi_tuned"     profile_name = "Step"
    #   controller_key = "ppo_final"    profile_name = "Step"
    #   controller_key = "bang_bang"    profile_name = "Pulsed"

    config = make_eval_config(seed=7)
    controller = make_controller(controller_key, config)
    data = run_controller(controller, profile_name, config, seed=7)

    animate_temperature_field(
        data=data,
        controller_name=controller.name,
        profile_name=profile_name,
        output_dir=output_dir,
        stride=8,
    )

    print("\nSummary:")
    print(f"  Controller:          {controller.name}")
    print(f"  Profile:             {profile_name}")
    print(f"  Max average temp:     {np.max(data['temperature_C']):.2f} °C")
    print(f"  Mean cooling command: {np.mean(data['cooling_command_u']):.3f}")
    print(f"  Cooling effort:       {np.sum(data['cooling_command_u']) * config.dt:.1f}")
    print(f"  Total reward:         {np.sum(data['reward']):.2f}")


if __name__ == "__main__":
    main()
