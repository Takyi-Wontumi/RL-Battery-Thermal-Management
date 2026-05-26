"""
scripts/animate_pack_temperature_field.py

Create an animated battery-pack temperature heatmap/GIF.

Unlike the single-cell pseudo-contour animation, this is based on the actual
multi-node pack temperatures from BatteryPackThermalEnv. Each block represents
one cell/module node in the pack model.

Run from project root:

    python -m scripts.animate_pack_temperature_field --PI
    python -m scripts.animate_pack_temperature_field --PI --SAC --PPO --bang-bang
    python -m scripts.animate_pack_temperature_field --PI --SAC --50mm
    python -m scripts.animate_pack_temperature_field --no-cooling --constant-1

    Add --50mm to use the 50 mm inter-cell spacing config and load models
    from models/sac_pack_1d/ and models/ppo_pack_1d/.

Outputs (one set per controller):
    outputs/pack_temperature_animation_<controller>_<profile>.gif
    outputs/pack_temperature_animation_<controller>_<profile>_final_frame.png
    outputs/pack_temperature_animation_<controller>_<profile>.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Protocol, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

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
    from stable_baselines3 import PPO, SAC
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


def load_pack_ppo_controller(
    config: BatteryPackThermalConfig,
    model_name: str,
    use_50mm: bool = False,
) -> PackPPOController:
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 required for PPO animation.")

    model_dir = PROJECT_ROOT / "models" / ("ppo_pack_1d" if use_50mm else "ppo_battery_pack_thermal")
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
    suffix = " (50mm)" if use_50mm else ""
    pretty_name = f"Pack PPO {'final' if model_name == 'final_model' else 'best'} model{suffix}"
    return PackPPOController(model=model, vec_normalize=vec_normalize, name=pretty_name)


class PackSACController:
    def __init__(self, model: SAC, name: str) -> None:
        self.model = model
        self.name = name

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_input = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action, _ = self.model.predict(obs_input, deterministic=True)
        u = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        return np.array([u], dtype=np.float32)


def load_pack_sac_controller(model_name: str, use_50mm: bool = False) -> PackSACController:
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 required for SAC animation.")

    model_dir = PROJECT_ROOT / "models" / ("sac_pack_1d" if use_50mm else "sac_pack_1d")
    model_path = model_dir / f"{model_name}.zip"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing SAC model: {model_path}")

    model = SAC.load(str(model_path), env=None, device="auto")
    suffix = " (50mm)" if use_50mm else ""
    pretty_name = f"Pack SAC {'final' if model_name == 'final_model' else 'best'} model{suffix}"
    return PackSACController(model=model, name=pretty_name)


# -----------------------------------------------------------------------------
# Controller factory
# -----------------------------------------------------------------------------

def make_controller(
    controller_key: str,
    config: BatteryPackThermalConfig,
    use_50mm: bool = False,
) -> PackControllerLike:
    controller_key = controller_key.lower().strip()

    if controller_key == "no_cooling":
        return PackNoCoolingController()
    if controller_key == "constant_05":
        return PackConstantCoolingController(cooling_level=0.5)
    if controller_key == "constant_10":
        return PackConstantCoolingController(cooling_level=1.0)
    if controller_key == "bang_bang":
        return PackBangBangController(config=config, target_temp=config.target_temp,
                                      deadband=1.0, name="Pack max-temp bang-bang")
    if controller_key == "proportional":
        return PackProportionalController(config=config, target_temp=config.target_temp,
                                          kp=0.10, bias=0.15, imbalance_gain=0.04,
                                          name="Pack max-temp proportional")
    if controller_key == "pi_tuned":
        return PackPIController(config=config, target_temp=config.target_temp,
                                kp=0.30, ki=0.001, bias=0.30, imbalance_gain=0.08,
                                integral_limit=50.0, name="Pack max-temp PI tuned")
    if controller_key == "ppo_best":
        return load_pack_ppo_controller(config=config, model_name="best_model", use_50mm=use_50mm)
    if controller_key == "ppo_final":
        return load_pack_ppo_controller(config=config, model_name="final_model", use_50mm=use_50mm)
    if controller_key == "sac_best":
        return load_pack_sac_controller(model_name="best_model", use_50mm=use_50mm)
    if controller_key == "sac_final":
        return load_pack_sac_controller(model_name="final_model", use_50mm=use_50mm)

    raise KeyError(
        f"Unknown controller '{controller_key}'. Choose from: no_cooling, constant_05, "
        "constant_10, bang_bang, proportional, pi_tuned, "
        "ppo_best, ppo_final, sac_best, sac_final"
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


def pack_grid_shape(n_cells: int) -> Tuple[int, int]:
    """Return a readable grid shape for the pack animation."""
    if n_cells == 8:
        return 2, 4
    if n_cells == 12:
        return 3, 4
    if n_cells == 16:
        return 4, 4

    n_cols = int(np.ceil(np.sqrt(n_cells)))
    n_rows = int(np.ceil(n_cells / n_cols))
    return n_rows, n_cols


def temperatures_to_grid(temperatures: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    flat = np.asarray(temperatures, dtype=np.float32).reshape(-1)
    for idx, temp in enumerate(flat):
        row = idx // n_cols
        col = idx % n_cols
        if row < n_rows:
            grid[row, col] = temp
    return grid


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
        ambient = float(info["ambient_temperature"])
        h_coeff = cooling_coefficient(config, u)
        cooling_per_cell = h_coeff * config.surface_area_per_cell * (pre_temperatures - ambient)
        cooling_per_cell += config.direct_cooling_max_per_cell * u

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
            "total_cooling_power_W": float(np.sum(cooling_per_cell)),
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
# Animation
# -----------------------------------------------------------------------------

def animate_pack_temperature(
    df: pd.DataFrame,
    config: BatteryPackThermalConfig,
    output_gif_path: Path,
    output_png_path: Path,
    stride: int = 8,
    fps: int = 12,
) -> None:
    controller_name = str(df["controller"].iloc[0])
    profile_name = str(df["profile"].iloc[0])

    cell_temp_cols = [
        f"cell_{i + 1}_temperature_C" for i in range(config.n_cells)
        if f"cell_{i + 1}_temperature_C" in df.columns
    ]

    if len(cell_temp_cols) == 0:
        raise RuntimeError("No per-cell temperature columns found in dataframe.")

    temperatures = df[cell_temp_cols].to_numpy(dtype=np.float32)
    time = df["time_s"].to_numpy(dtype=np.float32)
    max_temp = df["max_cell_temperature_C"].to_numpy(dtype=np.float32)
    mean_temp = df["mean_pack_temperature_C"].to_numpy(dtype=np.float32)
    spread = df["temperature_spread_C"].to_numpy(dtype=np.float32)
    action = df["cooling_command_u"].to_numpy(dtype=np.float32)
    heat_total = df["total_heat_generation_W"].to_numpy(dtype=np.float32)

    frame_indices = np.arange(0, len(df), stride)
    if frame_indices[-1] != len(df) - 1:
        frame_indices = np.append(frame_indices, len(df) - 1)

    n_rows, n_cols = pack_grid_shape(config.n_cells)

    # Fixed color scale across animation. Keep it presentation-friendly.
    vmin = min(float(np.nanmin(temperatures)), config.ambient_temp)
    vmax = max(float(np.nanmax(temperatures)), config.soft_max_temp)
    if vmax - vmin < 5.0:
        vmax = vmin + 5.0

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5.5))
    ax_pack, ax_signals = axes

    fig.suptitle(
        f"Battery pack temperature animation — {controller_name} on {profile_name}",
        fontsize=13,
        fontweight="bold",
    )

    # Signal plot on the right.
    ax_temp = ax_signals
    ax_u = ax_signals.twinx()

    ax_temp.plot(time, max_temp, linewidth=1.8, label="Max cell temp")
    ax_temp.plot(time, mean_temp, linewidth=1.3, label="Mean pack temp")
    ax_temp.axhline(config.target_temp, linestyle=":", linewidth=1.0, label="Target")
    ax_temp.axhline(config.soft_max_temp, linestyle="--", linewidth=1.0, label="Safe limit")
    ax_u.plot(time, action, linewidth=1.0, alpha=0.75, label="Cooling u")

    moving_line = ax_temp.axvline(time[0], linewidth=1.3)

    ax_temp.set_xlabel("Time (s)")
    ax_temp.set_ylabel("Temperature (°C)")
    ax_u.set_ylabel("Cooling command u")
    ax_u.set_ylim(-0.05, 1.05)
    ax_temp.grid(True, alpha=0.25)

    lines_1, labels_1 = ax_temp.get_legend_handles_labels()
    lines_2, labels_2 = ax_u.get_legend_handles_labels()
    ax_temp.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=8, loc="upper right")

    image_holder = {"image": None, "colorbar": None}

    def update(frame_number: int):
        idx = int(frame_indices[frame_number])
        temp_grid = temperatures_to_grid(temperatures[idx], n_rows, n_cols)

        ax_pack.clear()
        im = ax_pack.imshow(temp_grid, vmin=vmin, vmax=vmax, cmap="RdYlBu_r", aspect="equal")
        image_holder["image"] = im

        # Draw cell labels and temperatures.
        for cell_idx in range(config.n_cells):
            r = cell_idx // n_cols
            c = cell_idx % n_cols
            temp = temperatures[idx, cell_idx]
            ax_pack.text(
                c,
                r,
                f"C{cell_idx + 1}\n{temp:.1f}°C",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if temp > (vmin + vmax) * 0.58 else "black",
                fontweight="bold",
            )

        ax_pack.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax_pack.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax_pack.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax_pack.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        ax_pack.set_title(
            f"t={time[idx]:.0f}s | max={max_temp[idx]:.2f}°C | "
            f"spread={spread[idx]:.2f}°C | u={action[idx]:.2f} | Q={heat_total[idx]:.0f}W"
        )

        if image_holder["colorbar"] is None:
            image_holder["colorbar"] = fig.colorbar(im, ax=ax_pack, fraction=0.046, pad=0.04)
            image_holder["colorbar"].set_label("Cell temperature (°C)")

        moving_line.set_xdata([time[idx], time[idx]])
        return []

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 / fps, blit=False)

    anim.save(output_gif_path, writer=PillowWriter(fps=fps))

    # Save final frame.
    update(len(frame_indices) - 1)
    plt.savefig(output_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(df: pd.DataFrame, config: BatteryPackThermalConfig) -> None:
    controller_name = str(df["controller"].iloc[0])
    profile_name = str(df["profile"].iloc[0])

    print("\nPack animation summary")
    print(f"  Controller:             {controller_name}")
    print(f"  Profile:                {profile_name}")
    print(f"  Max cell temperature:    {df['max_cell_temperature_C'].max():.2f} C")
    print(f"  Mean pack temperature:   {df['mean_pack_temperature_C'].mean():.2f} C")
    print(f"  Max temperature spread:  {df['temperature_spread_C'].max():.2f} C")
    print(f"  Mean temperature spread: {df['temperature_spread_C'].mean():.2f} C")
    print(f"  Time above safe:         {(df['max_cell_temperature_C'] > config.soft_max_temp).sum() * config.dt:.1f} s")
    print(f"  Mean cooling command:    {df['cooling_command_u'].mean():.3f}")
    print(f"  Cooling effort:          {df['cooling_command_u'].sum() * config.dt:.1f}")
    print(f"  Total reward:            {df['reward'].sum():.2f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Animate 1D battery pack temperature field for one or more controllers."
    )
    parser.add_argument("--PI",          dest="controllers", action="append_const", const="pi_tuned",
                        help="PI tuned controller")
    parser.add_argument("--SAC",         dest="controllers", action="append_const", const="sac_best",
                        help="SAC best model")
    parser.add_argument("--PPO",         dest="controllers", action="append_const", const="ppo_best",
                        help="PPO best model")
    parser.add_argument("--bang-bang",   dest="controllers", action="append_const", const="bang_bang",
                        help="Bang-bang controller")
    parser.add_argument("--proportional",dest="controllers", action="append_const", const="proportional",
                        help="Proportional controller")
    parser.add_argument("--no-cooling",  dest="controllers", action="append_const", const="no_cooling",
                        help="No cooling baseline")
    parser.add_argument("--constant-1",  dest="controllers", action="append_const", const="constant_10",
                        help="Constant full cooling (u=1.0)")
    parser.add_argument("--constant-05", dest="controllers", action="append_const", const="constant_05",
                        help="Constant half cooling (u=0.5)")
    parser.add_argument(
        "--controller", dest="extra_controllers", action="append", default=[], metavar="KEY",
        help="Any controller key directly: no_cooling, bang_bang, pi_tuned, "
             "ppo_best, ppo_final, sac_best, sac_final, constant_05, constant_10",
    )
    parser.add_argument(
        "--profile", default="NonuniformStep",
        choices=["UniformConstant", "NonuniformStep", "PulsedHotspot", "RandomNonuniform"],
        help="Heat profile (default: NonuniformStep)",
    )
    parser.add_argument("--50mm", dest="use_50mm", action="store_true",
                        help="Use 50 mm cell spacing config and load from models/sac_pack_1d, ppo_pack_1d")
    parser.add_argument("--fps",    type=int, default=12, help="Animation FPS (default: 12)")
    parser.add_argument("--stride", type=int, default=8,
                        help="Simulation steps between animation frames (default: 8)")

    args = parser.parse_args()

    keys = list(args.controllers or []) + args.extra_controllers
    if not keys:
        keys = ["pi_tuned"]

    return keys, args.profile, args.fps, args.stride, args.use_50mm


def _run_one(
    controller_key: str,
    profile_name: str,
    fps: int,
    stride: int,
    use_50mm: bool,
    output_dir: Path,
) -> None:
    from scripts.compare_pack_baselines_1d_50mm import make_50mm_config

    config = make_50mm_config() if use_50mm else make_eval_config(seed=7)
    controller = make_controller(controller_key, config, use_50mm=use_50mm)

    print(f"\nRunning: {controller.name} on {profile_name} "
          f"({'50 mm spacing' if use_50mm else 'default spacing'}) ...")

    df = run_controller(controller=controller, profile_name=profile_name, config=config, seed=7)

    controller_slug = sanitize_filename(controller.name)
    profile_slug = sanitize_filename(profile_name)
    gif_path = output_dir / f"pack_temperature_animation_{controller_slug}_{profile_slug}.gif"
    png_path = output_dir / f"pack_temperature_animation_{controller_slug}_{profile_slug}_final_frame.png"
    csv_path = output_dir / f"pack_temperature_animation_{controller_slug}_{profile_slug}.csv"

    df.to_csv(csv_path, index=False)
    animate_pack_temperature(df=df, config=config,
                             output_gif_path=gif_path, output_png_path=png_path,
                             stride=stride, fps=fps)
    print_summary(df=df, config=config)
    print(f"Saved:\n  {gif_path}\n  {png_path}\n  {csv_path}")


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    controller_keys, profile_name, fps, stride, use_50mm = _parse_args()

    spacing_label = "50 mm spacing" if use_50mm else "default spacing"
    print(f"Profile: {profile_name}  |  {spacing_label}")
    print(f"Controllers: {controller_keys}")

    for key in controller_keys:
        try:
            _run_one(key, profile_name, fps, stride, use_50mm, output_dir)
        except (KeyError, FileNotFoundError) as exc:
            print(f"Skipping '{key}': {exc}")


if __name__ == "__main__":
    main()
