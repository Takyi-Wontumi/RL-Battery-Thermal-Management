"""
scripts/plot_phase1_heat_profiles.py

Phase 1 validation plot:
Compares heat-generation profiles against battery temperature response
for multiple fixed cooling commands.

Run from project root:
    python scripts/plot_phase1_heat_profiles.py

Expected output:
    outputs/phase1_heat_profiles_vs_temperature.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from envs.battery_thermal_env import BatteryThermalConfig, BatteryThermalEnv


HeatProfile = Callable[[float, np.random.Generator], float]


def constant_heat_profile(q_gen: float = 500.0) -> HeatProfile:
    """Constant heat generation profile."""

    def profile(t: float, rng: np.random.Generator) -> float:
        return float(q_gen)

    return profile


def step_heat_profile(
    q_low: float = 300.0,
    q_high: float = 700.0,
    step_time: float = 300.0,
) -> HeatProfile:
    """Step increase in heat generation."""

    def profile(t: float, rng: np.random.Generator) -> float:
        return float(q_low if t < step_time else q_high)

    return profile


def pulsed_heat_profile(
    q_low: float = 200.0,
    q_high: float = 700.0,
    period: float = 100.0,
    duty_cycle: float = 0.35,
) -> HeatProfile:
    """Repeated high-load pulses."""

    def profile(t: float, rng: np.random.Generator) -> float:
        phase = (t % period) / period
        return float(q_high if phase < duty_cycle else q_low)

    return profile


def random_heat_profile(
    q_mean: float = 400.0,
    q_std: float = 22.0,
    smoothing: float = 0.92,
) -> HeatProfile:
    """
    Smooth random heat profile using a closure-based stochastic process.

    This avoids garbage white-noise heat that jumps unrealistically every timestep.
    """
    state = {"q": q_mean}

    def profile(t: float, rng: np.random.Generator) -> float:
        disturbance = rng.normal(0.0, q_std)
        state["q"] = smoothing * state["q"] + (1.0 - smoothing) * q_mean + disturbance
        return float(np.clip(state["q"], 330.0, 460.0))

    return profile


def run_fixed_cooling_case(
    heat_profile: HeatProfile,
    cooling_command: float,
    config: BatteryThermalConfig,
    seed: int = 7,
) -> Dict[str, np.ndarray]:
    """Run one simulation using a fixed normalized cooling command."""
    env = BatteryThermalEnv(
        config=config,
        heat_profile=heat_profile,
        render_mode=None,
    )

    obs, info = env.reset(seed=seed, options={"randomize": False})

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = np.array([cooling_command], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

    return env.get_episode_log()


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_time = 1000.0
    dt = 1.0

    config = BatteryThermalConfig(
        total_time=total_time,
        dt=dt,
        initial_temp=25.0,
        ambient_temp=25.0,
        thermal_capacitance=50_000.0,
        surface_area=1.0,
        h_min=5.0,
        h_max=80.0,
        direct_cooling_max=0.0,
        target_temp=30.0,
        soft_max_temp=40.0,
        hard_max_temp=60.0,
        seed=7,
    )

    profiles: Dict[str, Tuple[str, HeatProfile]] = {
        "Constant": ("Constant", constant_heat_profile(q_gen=500.0)),
        "Step": ("Step", step_heat_profile(q_low=300.0, q_high=700.0, step_time=300.0)),
        "Pulsed": ("Pulsed", pulsed_heat_profile(q_low=200.0, q_high=700.0, period=100.0, duty_cycle=0.35)),
        "Random": ("Random", random_heat_profile(q_mean=400.0, q_std=6.5, smoothing=0.85)),
    }

    cooling_cases = {
        "u=0.0 — no cooling": 0.0,
        "u=0.5 — medium cooling": 0.5,
        "u=1.0 — full cooling": 1.0,
    }

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(12, 9),
        sharex=True,
    )

    fig.suptitle(
        "Phase 1 — Heat profiles vs Battery temperature (all profiles, all cooling levels)",
        fontsize=13,
        fontweight="bold",
    )

    profile_colors = {
        "Constant": "tab:green",
        "Step": "tab:blue",
        "Pulsed": "tab:orange",
        "Random": "tab:pink",
    }

    cooling_styles = {
        "u=0.0 — no cooling": {"linestyle": "-", "linewidth": 1.8},
        "u=0.5 — medium cooling": {"linestyle": "--", "linewidth": 1.5},
        "u=1.0 — full cooling": {"linestyle": ":", "linewidth": 1.7},
    }

    for row_idx, (profile_name, (plot_label, heat_profile)) in enumerate(profiles.items()):
        # Important: each profile must be recreated for each profile row if it has internal state.
        if profile_name == "Constant":
            row_heat_profile = constant_heat_profile(q_gen=500.0)
        elif profile_name == "Step":
            row_heat_profile = step_heat_profile(q_low=300.0, q_high=700.0, step_time=300.0)
        elif profile_name == "Pulsed":
            row_heat_profile = pulsed_heat_profile(q_low=200.0, q_high=700.0, period=100.0, duty_cycle=0.35)
        elif profile_name == "Random":
            row_heat_profile = random_heat_profile(q_mean=400.0, q_std=6.5, smoothing=0.85)
        else:
            raise ValueError(f"Unknown profile: {profile_name}")

        # Run one case first to plot heat load on the left.
        reference_log = run_fixed_cooling_case(
            heat_profile=row_heat_profile,
            cooling_command=0.0,
            config=config,
            seed=7,
        )

        ax_heat = axes[row_idx, 0]
        ax_temp = axes[row_idx, 1]

        ax_heat.plot(
            reference_log["time"],
            reference_log["heat_generation"],
            color=profile_colors[profile_name],
            linewidth=1.6,
        )
        ax_heat.set_title(f"{plot_label} — heat load", fontsize=10)
        ax_heat.set_ylabel("Q_gen (W)")
        ax_heat.grid(True, alpha=0.25)

        # Plot temperature response for all cooling levels.
        for case_label, u in cooling_cases.items():
            # Recreate stochastic/stateful heat profile each run so comparisons are fair.
            if profile_name == "Constant":
                sim_profile = constant_heat_profile(q_gen=500.0)
            elif profile_name == "Step":
                sim_profile = step_heat_profile(q_low=300.0, q_high=700.0, step_time=300.0)
            elif profile_name == "Pulsed":
                sim_profile = pulsed_heat_profile(q_low=200.0, q_high=700.0, period=100.0, duty_cycle=0.35)
            else:
                sim_profile = random_heat_profile(q_mean=400.0, q_std=6.5, smoothing=0.85)

            log = run_fixed_cooling_case(
                heat_profile=sim_profile,
                cooling_command=u,
                config=config,
                seed=7,
            )

            ax_temp.plot(
                log["time"],
                log["temperature"],
                label=case_label,
                **cooling_styles[case_label],
            )

        ax_temp.axhline(
            config.soft_max_temp,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label="T_safe = 40°C" if row_idx == 0 else None,
        )
        ax_temp.axhline(
            config.target_temp,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            label="T_target = 30°C" if row_idx == 0 else None,
        )

        ax_temp.set_title(f"{plot_label} — battery temperature", fontsize=10)
        ax_temp.set_ylabel("Temperature (°C)")
        ax_temp.grid(True, alpha=0.25)
        ax_temp.set_ylim(24.0, 41.0)

        if row_idx == 0:
            ax_temp.legend(fontsize=8, loc="center right")

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = output_dir / "phase1_heat_profiles_vs_temperature.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
