"""
scripts/run_3d_simulation.py

End-to-end 3D simulation demo.

Demonstrates:
    1. HPPC electrochemical heat generation (SOC-tracked, per-cell resistance variation)
    2. Multi-zone cooling (independent command per x-column)
    3. Zone-proportional controller (each zone responds to its own max temperature)
    4. Side-by-side comparison: multi-zone vs uniform-zone PI

Run from project root:
    python scripts/run_3d_simulation.py

Outputs:
    - Printed summary table
    - simulation_results.png (5-panel figure)

== Why multi-zone matters ==

In the 4×4×1 Samsung INR18650-25R pack, cells at the center (x=1,2, y=1,2)
can only lose heat through conduction to boundary cells — they have no direct
exposure to the coolant.  With a single cooling command the agent has two bad
choices:

    a) Low u  →  boundary cells stay cool, center cells overheat.
    b) High u →  boundary cells are over-cooled (energy waste), center may
                 still be above target.

With 4 zones the agent drives zone 0 and zone 3 (boundary columns) at moderate
effort and concentrates cooling on zones 1 and 2 (inner columns) where cells
have less direct exposure.  Result: lower peak temperature AND lower total
cooling energy.

== Why HPPC heat matters ==

A prescribed sinusoidal profile produces the same heat regardless of SOC.
Real Joule heating is Q = I²·R(SOC).  R(SOC) for NMC-graphite rises from
~0.022 Ω at 60% SOC to ~0.055 Ω at 0% SOC — a 2.5× increase.  At high
C-rate the pack heats much faster late in discharge.  A controller that does
not observe SOC (or adapt to it) will be surprised by this acceleration.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from configs.pack_config import CellConfig, PackConfig
from envs.battery_pack_thermal_env_3d import BatteryPackThermalEnv3D, nonuniform_step_3d_heat
from models.electrochemical_model import (
    ElectrochemicalHeatProfile,
    sinusoidal_current_profile,
    stepped_current_profile,
)


# ---------------------------------------------------------------------------
# Controller definitions
# ---------------------------------------------------------------------------

class ZoneProportionalController:
    """
    Independent proportional controller per zone.

    Each zone i reads the maximum temperature of cells in column i and
    applies proportional cooling:
        u_i = clip( Kp * (T_zone_i_max - target), 0, 1 )

    WHY per-zone max (not pack max):
        Using pack-level T_max would give all zones the same command —
        that degenerates back to single-zone control.  Using the zone's
        own T_max lets each zone respond independently.
    """

    def __init__(self, target_temp: float = 35.0, kp: float = 0.08, bias: float = 0.05):
        self.target_temp = target_temp
        self.kp = kp
        self.bias = bias

    def act(self, zone_max_temps: np.ndarray) -> np.ndarray:
        errors = zone_max_temps - self.target_temp
        u = self.bias + self.kp * errors
        return np.clip(u, 0.0, 1.0).astype(np.float32)


class UniformPIController:
    """
    Single PI controller — all zones get the same command.
    This is equivalent to the old single-u design broadcast to Nx zones.
    Used as the comparison baseline.
    """

    def __init__(self, target_temp: float = 35.0, kp: float = 0.08, ki: float = 0.002,
                 bias: float = 0.05, dt: float = 1.0, n_zones: int = 4):
        self.target_temp = target_temp
        self.kp = kp
        self.ki = ki
        self.bias = bias
        self.dt = dt
        self.n_zones = n_zones
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def act(self, T_max: float) -> np.ndarray:
        error = T_max - self.target_temp
        self._integral += error * self.dt
        self._integral = np.clip(self._integral, -100.0, 100.0)
        u = self.bias + self.kp * error + self.ki * self._integral
        u = float(np.clip(u, 0.0, 1.0))
        return np.full(self.n_zones, u, dtype=np.float32)


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(env: BatteryPackThermalEnv3D, controller, mode: str) -> dict:
    """Run one full episode and return the episode log."""
    obs, info = env.reset(seed=42)
    controller.reset() if hasattr(controller, "reset") else None

    terminated = truncated = False

    while not (terminated or truncated):
        zone_max = np.array(info["zone_max_temps"])
        T_max = info["T_max"]

        if mode == "zone":
            action = controller.act(zone_max)
        else:
            action = controller.act(T_max)

        obs, reward, terminated, truncated, info = env.step(action)

    log = env.get_episode_log()
    log["mode"] = mode
    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cell = CellConfig()
    pack = PackConfig(shape=(4, 4, 1))

    # Drive cycle: low discharge → medium → high (simulates city-highway-city)
    current_profile = stepped_current_profile([
        (0,    10.0),   #  0–600 s:  city  (10 A pack = 2.5 A/cell at 4P)
        (600,  20.0),   #  600–1200 s: highway (20 A)
        (1200, 12.0),   # 1200–1800 s: return city
    ])

    echem_profile = ElectrochemicalHeatProfile(
        parallel_count=pack.parallel_count,
        capacity_ah=cell.capacity_ah,
        dt_s=pack.dt_s,
        pack_shape=pack.shape,
        current_profile=current_profile,
        soc_initial=0.85,
        soc_variation_std=0.015,
        resistance_variation_std=0.008,
        include_entropic=True,
        rng=np.random.default_rng(0),
    )

    env = BatteryPackThermalEnv3D(
        cell_config=cell,
        pack_config=pack,
        heat_profile=echem_profile,
        total_time_s=pack.episode_time_s,
        dt_s=pack.dt_s,
        seed=42,
    )

    print("=" * 60)
    print("3D Battery Pack Simulation — Samsung INR18650-25R  4S4P")
    print(f"Pack shape:  {pack.shape}  ({4*4*1} cells)")
    print(f"Obs dim:     {env.observation_space.shape[0]}  "
          f"(6 thermal + 2 SOC + {env.n_zones} zone actions)")
    print(f"Action dim:  {env.action_space.shape[0]}  ({env.n_zones} zones)")
    print("=" * 60)

    zone_ctrl = ZoneProportionalController(
        target_temp=pack.target_temp_c, kp=0.10, bias=0.05
    )
    uniform_ctrl = UniformPIController(
        target_temp=pack.target_temp_c, kp=0.10, ki=0.003,
        bias=0.05, dt=pack.dt_s, n_zones=env.n_zones
    )

    print("\nRunning multi-zone controller...")
    log_zone = run_episode(env, zone_ctrl, mode="zone")

    print("Running uniform PI controller...")
    log_uniform = run_episode(env, uniform_ctrl, mode="uniform")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    def summarise(log, label):
        T_max_arr = np.array(log["T_max"])
        q_cool = np.array(log["q_cool_total"])
        time_over = float(np.sum(T_max_arr > pack.safe_temp_c))
        print(f"\n  [{label}]")
        print(f"    Peak T_max:          {T_max_arr.max():.2f} °C")
        print(f"    Final T_max:         {T_max_arr[-1]:.2f} °C")
        print(f"    Mean T_max:          {T_max_arr.mean():.2f} °C")
        print(f"    Steps above safe:    {int(time_over)} s  "
              f"(safe limit = {pack.safe_temp_c} °C)")
        print(f"    Total cooling energy: {q_cool.sum():.0f} W·s")
        if "soc_mean" in log:
            soc = np.array(log["soc_mean"])
            print(f"    Final SOC (mean):    {soc[-1]:.3f}")

    print("\n" + "=" * 60)
    print("RESULTS")
    summarise(log_zone,    "Multi-zone proportional")
    summarise(log_uniform, "Uniform PI (all zones same)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    time_z = np.array(log_zone["time"])
    time_u = np.array(log_uniform["time"])

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "3D Battery Pack — Multi-zone vs Uniform Cooling\n"
        "Samsung INR18650-25R  4S4P  |  HPPC Electrochemical Heat Model",
        fontsize=12,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Pack max temperature
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_z / 60, np.array(log_zone["T_max"]),    label="Multi-zone", color="steelblue")
    ax1.plot(time_u / 60, np.array(log_uniform["T_max"]), label="Uniform PI",  color="tomato",
             linestyle="--")
    ax1.axhline(pack.safe_temp_c,     color="orange", linestyle=":", linewidth=1.2, label="Safe limit")
    ax1.axhline(pack.critical_temp_c, color="red",    linestyle=":", linewidth=1.2, label="Critical")
    ax1.axhline(pack.target_temp_c,   color="green",  linestyle=":", linewidth=1.0, alpha=0.6)
    ax1.set_xlabel("Time [min]")
    ax1.set_ylabel("T_max [°C]")
    ax1.set_title("Pack Max Temperature")
    ax1.legend(fontsize=7)

    # Panel 2: Temperature gradient (non-uniformity)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_z / 60, np.array(log_zone["T_gradient"]),    color="steelblue", label="Multi-zone")
    ax2.plot(time_u / 60, np.array(log_uniform["T_gradient"]), color="tomato", linestyle="--",
             label="Uniform PI")
    ax2.set_xlabel("Time [min]")
    ax2.set_ylabel("T_max − T_min [°C]")
    ax2.set_title("Temperature Non-uniformity")
    ax2.legend(fontsize=7)

    # Panel 3: Zone-wise cooling commands (multi-zone only)
    ax3 = fig.add_subplot(gs[1, 0])
    actions_z = np.array(log_zone["actions"])  # shape (T, n_zones)
    colors_z = ["steelblue", "darkorange", "green", "purple"]
    for i in range(env.n_zones):
        ax3.plot(time_z / 60, actions_z[:, i],
                 label=f"Zone {i} (col x={i})", color=colors_z[i], alpha=0.85)
    ax3.set_xlabel("Time [min]")
    ax3.set_ylabel("Cooling command u")
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title("Multi-zone Commands (each zone independent)")
    ax3.legend(fontsize=7)

    # Panel 4: Per-zone max temperatures (multi-zone)
    ax4 = fig.add_subplot(gs[1, 1])
    zone_temps_z = np.array(log_zone["zone_max_temps"])  # shape (T, n_zones)
    for i in range(env.n_zones):
        ax4.plot(time_z / 60, zone_temps_z[:, i],
                 label=f"Zone {i}", color=colors_z[i], alpha=0.85)
    ax4.axhline(pack.safe_temp_c, color="orange", linestyle=":", linewidth=1.2)
    ax4.axhline(pack.target_temp_c, color="green", linestyle=":", linewidth=1.0, alpha=0.6)
    ax4.set_xlabel("Time [min]")
    ax4.set_ylabel("Zone T_max [°C]")
    ax4.set_title("Per-zone Max Temperature")
    ax4.legend(fontsize=7)

    # Panel 5: SOC depletion
    ax5 = fig.add_subplot(gs[2, 0])
    if "soc_mean" in log_zone:
        ax5.plot(time_z / 60, np.array(log_zone["soc_mean"]),
                 label="SOC mean", color="steelblue")
        ax5.plot(time_z / 60, np.array(log_zone["soc_min"]),
                 label="SOC min", color="steelblue", linestyle="--", alpha=0.6)
    ax5.set_xlabel("Time [min]")
    ax5.set_ylabel("State of Charge")
    ax5.set_ylim(0, 1)
    ax5.set_title("SOC Depletion (HPPC model)")
    ax5.legend(fontsize=7)

    # Panel 6: Total cooling energy rate
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(time_z / 60, np.array(log_zone["q_cool_total"]),    color="steelblue",
             label="Multi-zone")
    ax6.plot(time_u / 60, np.array(log_uniform["q_cool_total"]), color="tomato", linestyle="--",
             label="Uniform PI")
    ax6.set_xlabel("Time [min]")
    ax6.set_ylabel("Total cooling power [W]")
    ax6.set_title("Total Cooling Power")
    ax6.legend(fontsize=7)

    out_path = PROJECT_ROOT / "simulation_results.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
