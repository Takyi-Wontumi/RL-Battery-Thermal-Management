"""
scripts/run_stress_test.py

Three-scenario stress test for the 3D battery pack thermal environment.

Scenarios
---------
A  Mild HPPC      — nominal q_gen; no-cooling stays below 45°C.
                    Purpose: validate model behaviour.
B  Stress HPPC    — q_gen × 2.5; no-cooling crosses 45°C at ~13 min.
                    Purpose: prove active cooling is necessary.
C  Hotspot stress  — spatial hotspot 2×–4× on one zone; q_gen × 1.5.
                    Purpose: show zone-proportional outperforms global PI.

Controllers tested in every scenario
-------------------------------------
  No cooling         — u = 0 (danger reference)
  Global PI          — single command broadcast to all zones
  Zone-proportional  — separate command per quadrant zone

Usage
-----
    python -m scripts.run_stress_test
    python -m scripts.run_stress_test --output-dir outputs/stress_test --seed 42
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.pack_config import CellConfig, PackConfig
from envs.battery_pack_thermal_env_3d import (
    BatteryPackThermalEnv3D,
    Pack3DHeatProfile,
    nonuniform_step_3d_heat,
    uniform_constant_3d_heat,
)


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

@dataclass
class NoCooling:
    name: str = "No cooling"
    n_zones: int = 4

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        return np.zeros(self.n_zones, dtype=np.float32)


@dataclass
class GlobalPI:
    """PI controller — returns one command broadcast to all zones."""
    name: str = "Global PI"
    n_zones: int = 4
    target_temp_c: float = 35.0
    kp: float = 0.30
    ki: float = 0.001
    bias: float = 0.30
    integral_limit: float = 50.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        self.integral_error: float = 0.0

    def reset(self) -> None:
        self.integral_error = 0.0

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        # obs[0] = (T_max - target) / (safe - target), so T_max = obs[0]*scale + target
        scale = 45.0 - 35.0   # safe - target (approximation; good enough for control)
        T_max = float(obs[0]) * scale + self.target_temp_c
        error = T_max - self.target_temp_c

        self.integral_error = float(
            np.clip(self.integral_error + error * self.dt, -self.integral_limit, self.integral_limit)
        )
        u = float(np.clip(self.bias + self.kp * error + self.ki * self.integral_error, 0.0, 1.0))
        return np.full(self.n_zones, u, dtype=np.float32)


@dataclass
class ZonePI:
    """
    Per-zone PI controller — same gains as GlobalPI but runs a separate
    integrator per cooling zone.  Uses zone_max_temps from info so each
    zone responds to its local hottest cell rather than the pack-wide T_max.
    This is the fair multi-zone comparison against GlobalPI.
    """
    name: str = "Zone PI"
    n_zones: int = 4
    target_temp_c: float = 35.0
    kp: float = 0.30
    ki: float = 0.001
    bias: float = 0.30
    integral_limit: float = 50.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        self.integral_errors: np.ndarray = np.zeros(self.n_zones)

    def reset(self) -> None:
        self.integral_errors = np.zeros(self.n_zones)

    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        if info is not None and "zone_max_temps" in info:
            zone_max = np.array(info["zone_max_temps"], dtype=np.float64)
        else:
            scale = 45.0 - 35.0
            T_max = float(obs[0]) * scale + self.target_temp_c
            zone_max = np.full(self.n_zones, T_max)

        errors = zone_max - self.target_temp_c
        self.integral_errors = np.clip(
            self.integral_errors + errors * self.dt,
            -self.integral_limit, self.integral_limit,
        )
        u = self.bias + self.kp * errors + self.ki * self.integral_errors
        return np.clip(u, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-scenario runner
# ---------------------------------------------------------------------------

def run_scenario(
    pack_config: PackConfig,
    heat_profile: Pack3DHeatProfile,
    controllers: List,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Run every controller on the same env config.

    Returns dict: controller_name → episode_log dict.
    """
    results: Dict[str, Dict] = {}

    for ctrl in controllers:
        ctrl.reset()
        env = BatteryPackThermalEnv3D(
            cell_config=CellConfig(),
            pack_config=pack_config,
            heat_profile=heat_profile,
            seed=seed,
        )
        obs, info = env.reset(seed=seed)

        log: Dict[str, list] = {
            "time_min": [], "T_max": [], "T_avg": [],
            "zone_max_temps": [], "u_applied": [],
        }

        done = False
        while not done:
            action = ctrl.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            log["time_min"].append(info["time_s"] / 60.0)
            log["T_max"].append(info["T_max"])
            log["T_avg"].append(info["T_avg"])
            log["zone_max_temps"].append(info["zone_max_temps"])
            log["u_applied"].append(info["u_applied_zones"])

        results[ctrl.name] = log
        env.close()

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_CTRL_STYLE = {
    "No cooling":       dict(color="#e74c3c", ls="--",  lw=1.8),
    "Global PI":        dict(color="#2980b9", ls="-",   lw=1.8),
    "Zone PI":           dict(color="#27ae60", ls="-",   lw=2.0),
}

_LIMIT_KW = dict(lw=1.2, alpha=0.75)


def _add_limits(ax: plt.Axes, pack_cfg: PackConfig) -> None:
    ax.axhline(pack_cfg.target_temp_c,   color="#2ecc71", ls=":",  **_LIMIT_KW, label=f"Target {pack_cfg.target_temp_c:.0f}°C")
    ax.axhline(pack_cfg.safe_temp_c,     color="darkorange", ls="--", **_LIMIT_KW, label=f"Safe {pack_cfg.safe_temp_c:.0f}°C")
    ax.axhline(pack_cfg.critical_temp_c, color="#c0392b", ls="-",  **_LIMIT_KW, label=f"Critical {pack_cfg.critical_temp_c:.0f}°C")


def plot_stress_test(
    scenario_results: Dict[str, Dict],   # scenario_label → {ctrl_name → log}
    scenario_titles: Dict[str, str],
    pack_cfg: PackConfig,
    output_path: Path,
    seed: int,
) -> None:
    n_scenarios = len(scenario_results)
    # Two rows: top = T_max safety, bottom = mean cooling command (energy)
    fig, axes = plt.subplots(2, n_scenarios, figsize=(5 * n_scenarios, 8), sharex=True)

    for col, (label, results) in enumerate(scenario_results.items()):
        ax_t = axes[0, col]
        ax_u = axes[1, col]

        _add_limits(ax_t, pack_cfg)

        for ctrl_name, log in results.items():
            style = _CTRL_STYLE.get(ctrl_name, dict(color="gray", ls="-", lw=1.5))
            t = log["time_min"]
            ax_t.plot(t, log["T_max"], label=ctrl_name, **style)

            if ctrl_name != "No cooling":
                u_arr = np.array(log["u_applied"])   # (T, n_zones)
                u_mean = u_arr.mean(axis=1)
                ax_u.plot(t, u_mean, label=ctrl_name, **style)

        ax_t.set_title(scenario_titles[label], fontsize=9, pad=6)
        ax_t.set_ylim(22, 58)
        ax_t.set_xlim(0, 30)
        ax_t.grid(True, alpha=0.3)
        ax_t.legend(fontsize=8, loc="upper left")
        if col == 0:
            ax_t.set_ylabel("Pack T_max (°C)")

        ax_u.set_xlabel("Time (min)")
        ax_u.set_ylim(-0.05, 1.05)
        ax_u.grid(True, alpha=0.3)
        ax_u.legend(fontsize=8, loc="upper left")
        if col == 0:
            ax_u.set_ylabel("Mean cooling command\n(applied, 0–1)")

    fig.suptitle(
        "Stress Test — Samsung INR18650-25R 4S4P Pack\n"
        "Multi-zone vs Global Cooling under Escalating Heat Load\n"
        "(Top: safety | Bottom: cooling effort — Zone PI uses less energy on cold zones)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_zone_detail(
    hotspot_results: Dict[str, Dict],
    pack_cfg: PackConfig,
    output_path: Path,
) -> None:
    """
    Extra panel for scenario C: per-zone temperatures and zone cooling commands.
    Shows that Zone-proportional applies more cooling to the hot zone.
    """
    n_ctrl = len(hotspot_results)
    fig, axes = plt.subplots(2, n_ctrl, figsize=(4.5 * n_ctrl, 7), sharex=True)
    zone_colors = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12"]

    for col, (ctrl_name, log) in enumerate(hotspot_results.items()):
        t = np.array(log["time_min"])
        zone_max = np.array(log["zone_max_temps"])   # (T, n_zones)
        u_applied = np.array(log["u_applied"])        # (T, n_zones)

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        for z in range(zone_max.shape[1]):
            ax_top.plot(t, zone_max[:, z], color=zone_colors[z], lw=1.5, label=f"Zone {z}")
        ax_top.axhline(pack_cfg.safe_temp_c, color="darkorange", ls="--", lw=1.2, alpha=0.75)
        ax_top.axhline(pack_cfg.target_temp_c, color="#2ecc71", ls=":", lw=1.2, alpha=0.75)
        ax_top.set_title(ctrl_name, fontsize=9)
        ax_top.set_ylim(22, 58)
        ax_top.grid(True, alpha=0.3)
        if col == 0:
            ax_top.set_ylabel("Zone max temp (°C)")
            ax_top.legend(fontsize=7, loc="upper left")

        for z in range(u_applied.shape[1]):
            ax_bot.plot(t, u_applied[:, z], color=zone_colors[z], lw=1.5)
        ax_bot.set_ylim(-0.05, 1.05)
        ax_bot.set_xlabel("Time (min)")
        ax_bot.grid(True, alpha=0.3)
        if col == 0:
            ax_bot.set_ylabel("Cooling command u (applied)")

    fig.suptitle(
        "Scenario C — Per-Zone Temperature & Cooling Commands\n"
        "(Hotspot stress: 2×–4× heat multiplier in one random zone)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-scenario stress test")
    p.add_argument("--output-dir", default="outputs/stress_test")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _make_pack(
    q_gen_multiplier: float = 1.0,
    enable_random_hotspot: bool = False,
) -> PackConfig:
    return PackConfig(
        shape=(4, 3, 2),
        cell_spacing_m=0.002,
        ambient_temp_c=25.0,
        initial_temp_c=25.0,
        h_min_w_per_m2_k=5.0,
        h_max_w_per_m2_k=80.0,
        target_temp_c=35.0,
        target_low_temp_c=30.0,
        target_high_temp_c=38.0,
        safe_temp_c=45.0,
        critical_temp_c=55.0,
        g_cond_w_per_k=0.25,
        enable_heat_variation=True,
        heat_variation_std=0.05,
        # Actuator
        enable_cooling_delay=True,
        cooling_delay_s=10.0,
        enable_cooling_rate_limit=True,
        max_cooling_rate_per_s=0.05,
        # Stress knobs
        q_gen_multiplier=q_gen_multiplier,
        enable_random_hotspot=enable_random_hotspot,
        num_hotspot_cells=3,
        hotspot_multiplier_min=2.0,
        hotspot_multiplier_max=4.0,
        hotspot_persistent=True,
        hotspot_seed=None,
    )


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    seed = args.seed

    n_zones = 4
    controllers = [
        NoCooling(n_zones=n_zones),
        GlobalPI(n_zones=n_zones),
        ZonePI(n_zones=n_zones),
    ]

    # ── Scenario A: Mild ───────────────────────────────────────────────────
    # UniformConstant 12 W — no cooling stays safely below 45°C.
    # Purpose: verify model runs correctly before applying stress.
    print("\nScenario A — Mild uniform heat (q_gen × 1.0, no hotspot) ...")
    cfg_a = _make_pack(q_gen_multiplier=1.0, enable_random_hotspot=False)
    res_a = run_scenario(cfg_a, uniform_constant_3d_heat(q_total_w=12.0), controllers, seed=seed)
    for name, log in res_a.items():
        print(f"  {name:28s}  T_max_peak={max(log['T_max']):.1f}°C")

    # ── Scenario B: Stress HPPC ────────────────────────────────────────────
    print("\nScenario B — Stress HPPC (q_gen × 2.5, no hotspot) ...")
    cfg_b = _make_pack(q_gen_multiplier=2.5, enable_random_hotspot=False)
    res_b = run_scenario(cfg_b, nonuniform_step_3d_heat(), controllers, seed=seed)
    for name, log in res_b.items():
        print(f"  {name:28s}  T_max_peak={max(log['T_max']):.1f}°C")

    # ── Scenario C: Hotspot stress ──────────────────────────────────────────
    print("\nScenario C — Hotspot stress (q_gen × 1.5, hotspot 2×–4×) ...")
    cfg_c = _make_pack(q_gen_multiplier=1.5, enable_random_hotspot=True)
    res_c = run_scenario(cfg_c, nonuniform_step_3d_heat(), controllers, seed=seed)
    for name, log in res_c.items():
        print(f"  {name:28s}  T_max_peak={max(log['T_max']):.1f}°C")

    # ── Main stress-test plot ──────────────────────────────────────────────
    scenario_results = {
        "A": res_a,
        "B": res_b,
        "C": res_c,
    }
    scenario_titles = {
        "A": "A: Mild uniform (q × 1.0)\nNo hotspot — model validation",
        "B": "B: Stress HPPC (q × 2.5)\nNo hotspot",
        "C": "C: Hotspot stress (q × 1.5)\nRandom zone 2×–4×",
    }
    plot_stress_test(
        scenario_results, scenario_titles, cfg_a,   # pack_cfg for limits
        out / "stress_test_Tmax.png", seed=seed,
    )

    # ── Zone detail for scenario C ─────────────────────────────────────────
    plot_zone_detail(
        res_c, cfg_c,
        out / "stress_test_zone_detail.png",
    )

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Summary — peak T_max per scenario and controller")
    print("=" * 65)
    rows = []
    for sc_label, results in scenario_results.items():
        for ctrl_name, log in results.items():
            t_peak = max(log["T_max"])
            u_arr = np.array(log["u_applied"])
            u_mean = round(float(u_arr.mean()), 3) if ctrl_name != "No cooling" else "—"
            rows.append({
                "Scenario": sc_label,
                "Controller": ctrl_name,
                "T_max_peak_C": round(t_peak, 1),
                "Safe": "PASS" if t_peak < cfg_a.safe_temp_c else "FAIL",
                "u_mean": u_mean,
            })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    csv_path = out / "stress_test_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved summary: {csv_path}")

    # Print interpretation
    no_cool_b = max(res_b["No cooling"]["T_max"])
    pi_b = max(res_b["Global PI"]["T_max"])
    zp_b = max(res_b["Zone PI"]["T_max"])
    no_cool_c = max(res_c["No cooling"]["T_max"])
    zp_c = max(res_c["Zone PI"]["T_max"])
    pi_c = max(res_c["Global PI"]["T_max"])

    print()
    print("  Interpretation:")
    print(f"    B — No cooling peak:  {no_cool_b:.1f}°C ({'ABOVE' if no_cool_b>45 else 'below'} 45°C safe limit)")
    print(f"    B — Global PI peak:   {pi_b:.1f}°C  ({'PASS' if pi_b<45 else 'FAIL'})")
    print(f"    B — Zone PI peak:     {zp_b:.1f}°C  ({'PASS' if zp_b<45 else 'FAIL'})")
    print(f"    C — No cooling peak:  {no_cool_c:.1f}°C ({'ABOVE' if no_cool_c>45 else 'below'} 45°C safe limit)")
    print(f"    C — Global PI peak:   {pi_c:.1f}°C  ({'PASS' if pi_c<45 else 'FAIL'})")
    print(f"    C — Zone PI peak:     {zp_c:.1f}°C  ({'PASS' if zp_c<45 else 'FAIL'})")
    if zp_c < pi_c:
        print(f"    Zone PI reduced T_max by {pi_c - zp_c:.1f}°C vs global PI in hotspot scenario.")
    elif zp_c > pi_c:
        print(f"    Note: Global PI kept T_max {zp_c - pi_c:.1f}°C lower than Zone PI.")


if __name__ == "__main__":
    main()
