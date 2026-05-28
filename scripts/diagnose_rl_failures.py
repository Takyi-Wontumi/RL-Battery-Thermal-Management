"""
scripts/diagnose_rl_failures.py

Deep-dive diagnostic for PPO/SAC safety failures under sensor-realistic evaluation.

Produces:
  outputs/diagnosis/
    failure_<ctrl>_<profile>_seed<seed>.png  — per-failed-episode 4-panel timeseries
    rl_diagnosis_summary.csv                 — per-episode stats (unshielded + shielded)
    rl_diagnosis_report.txt                  — text summary

Run:
    python -m scripts.diagnose_rl_failures

Root causes captured:
  - Undercooling: u_mean << Zone PI baseline
  - Late cooling: T_max exceeds safe limit before action ramps up
  - Wrong zone: max_cool_zone != hot_zone (targeting_correct = 0)
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from configs.pack_config import CellConfig, PackConfig
from configs.sensor_simulation import SensorConfig, ActuatorConfig
from envs.battery_pack_thermal_env_3d import BatteryPackThermalEnv3D, make_3d_profile
from scripts.compare_pack_baselines_3d import (
    ZonePI3D,
    PROFILE_NAMES,
)
from training.train_pack_ppo_3d import (
    make_pack_config,
    make_sensor_config,
    make_actuator_config,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnosis"

SEEDS = [7, 107, 207, 307, 407]
SAFE_TEMP_C    = 45.0
SHIELD_TEMP_C  = 44.0

ZONE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

# ---------------------------------------------------------------------------
# Controller adapters (same pattern as compare_controllers.py)
# ---------------------------------------------------------------------------

class _SACAdapter:
    controller_type = "Multi-zone RL"

    def __init__(self, model: SAC, name: str = "SAC") -> None:
        self.model = model
        self.name  = name

    def reset(self) -> None: pass

    def act(self, obs: np.ndarray, info=None) -> np.ndarray:
        obs_in = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action, _ = self.model.predict(obs_in, deterministic=True)
        return np.clip(np.asarray(action).reshape(-1), 0.0, 1.0).astype(np.float32)


class _PPOAdapter:
    controller_type = "Multi-zone RL"

    def __init__(self, model: PPO, vec_normalize: VecNormalize, name: str = "PPO") -> None:
        self.model         = model
        self.vec_normalize = vec_normalize
        self.name          = name

    def reset(self) -> None: pass

    def act(self, obs: np.ndarray, info=None) -> np.ndarray:
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        norm_obs  = self.vec_normalize.normalize_obs(obs_batch)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        return np.clip(np.asarray(action).reshape(-1), 0.0, 1.0).astype(np.float32)


class SafetyShieldedRL:
    """
    Wraps any RL controller with a Zone PI override when temperature nears the safe limit.

    When T_max_meas >= shield_temp_c, the per-zone action is element-wise max(u_rl, u_pi).
    This prevents undercooling emergencies while letting RL act freely in nominal conditions.
    When T_max_meas >= full_cooling_temp_c, full cooling (u=1) is applied unconditionally.
    """
    controller_type = "Multi-zone RL"

    def __init__(
        self,
        rl_controller,
        pack_config: PackConfig,
        shield_temp_c: float = SHIELD_TEMP_C,
        full_cooling_temp_c: float = 44.5,
    ) -> None:
        self.rl             = rl_controller
        self.name           = f"{rl_controller.name} + Shield"
        self.shield_temp_c  = shield_temp_c
        self.full_temp_c    = full_cooling_temp_c
        self._zone_pi = ZonePI3D(
            pack_config=pack_config,
            num_zones=pack_config.num_cooling_zones,
        )

    def reset(self) -> None:
        self.rl.reset()
        self._zone_pi.reset()

    def act(self, obs: np.ndarray, info=None) -> np.ndarray:
        u_rl = self.rl.act(obs, info)

        if info is not None:
            T_max = float(info.get("T_max_meas", info.get("T_max", 0.0)))
        else:
            # Fallback: decode pack_max_norm from obs position 4 (29D layout)
            T_max = float(obs[4]) * 10.0 + 35.0

        if T_max >= self.full_temp_c:
            return np.ones(len(u_rl), dtype=np.float32)

        if T_max >= self.shield_temp_c:
            u_pi = self._zone_pi.act(obs, info)
            return np.maximum(u_rl, u_pi).astype(np.float32)

        return u_rl


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_sac(path: Path, pack_config: PackConfig) -> Optional[_SACAdapter]:
    if not path.exists():
        print(f"  [skip] SAC not found: {path}")
        return None
    try:
        model = SAC.load(str(path), env=None, device="cpu")
        obs_dim = model.observation_space.shape[0]
        expected = _expected_obs_dim(pack_config)
        if obs_dim != expected:
            print(f"  [skip] SAC obs_dim={obs_dim} != expected {expected}")
            return None
        print(f"  Loaded SAC ({obs_dim}D obs, {model.action_space.shape[0]}D act)")
        return _SACAdapter(model, name="SAC")
    except Exception as exc:
        print(f"  [skip] SAC load error: {exc}")
        return None


def load_ppo(model_path: Path, pack_config: PackConfig) -> Optional[_PPOAdapter]:
    vecnorm_path = model_path.parent / "vec_normalize.pkl"
    if not model_path.exists():
        print(f"  [skip] PPO not found: {model_path}")
        return None
    if not vecnorm_path.exists():
        print(f"  [skip] PPO vec_normalize not found: {vecnorm_path}")
        return None
    try:
        model = PPO.load(str(model_path), env=None, device="cpu")
        obs_dim = model.observation_space.shape[0]
        expected = _expected_obs_dim(pack_config)
        if obs_dim != expected:
            print(f"  [skip] PPO obs_dim={obs_dim} != expected {expected}")
            return None

        cfg = pack_config
        sensor_cfg   = make_sensor_config(cfg)
        actuator_cfg = make_actuator_config(cfg)
        def _dummy_init():
            from envs.battery_pack_thermal_env_3d import uniform_constant_3d_heat
            env = BatteryPackThermalEnv3D(
                cell_config=CellConfig(),
                pack_config=cfg,
                heat_profile=uniform_constant_3d_heat(),
                seed=0,
                enable_sensor_simulation=True,
                sensor_config=sensor_cfg,
                actuator_config=actuator_cfg,
            )
            return Monitor(env)
        dummy = DummyVecEnv([_dummy_init])
        vn = VecNormalize.load(str(vecnorm_path), dummy)
        vn.training    = False
        vn.norm_reward = False
        print(f"  Loaded PPO ({obs_dim}D obs, {model.action_space.shape[0]}D act)")
        return _PPOAdapter(model, vn, name="PPO")
    except Exception as exc:
        print(f"  [skip] PPO load error: {exc}")
        return None


def _expected_obs_dim(pack_config: PackConfig) -> int:
    n = pack_config.num_cooling_zones
    return n + 8 + pack_config.series_count + n + n + n + 1  # = 29


# ---------------------------------------------------------------------------
# Episode runner — returns full step-by-step log
# ---------------------------------------------------------------------------

def run_episode(
    controller,
    profile_name: str,
    pack_config: PackConfig,
    cell_config: CellConfig,
    seed: int,
    sensor_cfg: SensorConfig,
    actuator_cfg: ActuatorConfig,
) -> Dict:
    env = BatteryPackThermalEnv3D(
        cell_config=cell_config,
        pack_config=pack_config,
        heat_profile=make_3d_profile(profile_name),
        seed=seed,
        enable_sensor_simulation=True,
        sensor_config=sensor_cfg,
        actuator_config=actuator_cfg,
    )
    controller.reset()
    obs, info = env.reset(seed=seed, options={"randomize": False})
    done = False
    total_reward = 0.0
    while not done:
        action = controller.act(obs, info)
        obs, rew, terminated, truncated, info = env.step(action)
        total_reward += rew
        done = terminated or truncated

    log = env.get_episode_log()
    return log, total_reward


# ---------------------------------------------------------------------------
# Metrics from episode log
# ---------------------------------------------------------------------------

def episode_metrics(log: Dict, pack_config: PackConfig, dt_s: float = 1.0) -> Dict:
    T_max = np.asarray(log["T_max"])
    time_above_safe = float(np.sum(T_max > pack_config.safe_temp_c) * dt_s)
    is_safe = (time_above_safe == 0.0) and (T_max.max() < pack_config.critical_temp_c)

    targeting = np.asarray(log.get("targeting_correct", []))
    zone_target_acc = float(np.mean(targeting)) if len(targeting) > 0 else float("nan")

    actions_2d = np.asarray(log.get("actions", [[0.5] * pack_config.num_cooling_zones]))
    if actions_2d.ndim == 1:
        actions_2d = actions_2d.reshape(-1, 1)
    mean_u = float(actions_2d.mean())

    return {
        "T_max_peak_C":          float(T_max.max()),
        "time_above_safe_s":     time_above_safe,
        "is_safe":               bool(is_safe),
        "zone_targeting_acc":    zone_target_acc,
        "mean_cooling_action":   mean_u,
    }


# ---------------------------------------------------------------------------
# Failure plot — 4-panel timeseries for one failed episode
# ---------------------------------------------------------------------------

def plot_failure(
    log: Dict,
    controller_name: str,
    profile: str,
    seed: int,
    pack_config: PackConfig,
    out_path: Path,
) -> None:
    T_max = np.asarray(log["T_max"])
    time  = np.asarray(log.get("time", np.arange(len(T_max))))
    n_zones = pack_config.num_cooling_zones

    actions_2d = np.asarray(log.get("actions", np.zeros((len(T_max), n_zones))))
    if actions_2d.ndim == 1:
        actions_2d = actions_2d.reshape(-1, 1)

    zone_max_true = np.asarray(log.get("zone_max_temps_true", np.zeros((len(T_max), n_zones))))
    zone_max_meas = np.asarray(log.get("zone_max_temps",      np.zeros((len(T_max), n_zones))))
    hot_zone_meas = np.asarray(log.get("hot_zone_meas",     np.argmax(zone_max_meas, axis=1) if zone_max_meas.ndim == 2 else np.zeros(len(T_max))))
    max_cool_zone = np.asarray(log.get("max_cool_zone",     np.argmax(actions_2d, axis=1)))
    targeting     = np.asarray(log.get("targeting_correct", (hot_zone_meas == max_cool_zone).astype(int)))

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"{controller_name} — {profile} — seed {seed}\n"
        f"T_max peak = {T_max.max():.1f}°C  "
        f"time above safe = {float(np.sum(T_max > SAFE_TEMP_C)):.0f}s  "
        f"u_mean = {actions_2d.mean():.3f}  "
        f"targeting = {float(np.mean(targeting))*100:.1f}%",
        fontsize=11, fontweight="bold",
    )
    gs = gridspec.GridSpec(4, 1, hspace=0.45)

    # Panel 1 — T_max over time
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, T_max, color="black", lw=1.5, label="T_max (true)")
    ax1.axhline(SAFE_TEMP_C,   color="red",    lw=1.0, ls="--", label=f"Safe {SAFE_TEMP_C}°C")
    ax1.axhline(SHIELD_TEMP_C, color="orange", lw=1.0, ls=":",  label=f"Shield {SHIELD_TEMP_C}°C")
    ax1.axhline(pack_config.target_temp_c, color="green", lw=0.8, ls=":", label="Target")
    # Shade violation region
    violation = T_max > SAFE_TEMP_C
    if violation.any():
        ax1.fill_between(time, SAFE_TEMP_C, T_max, where=violation, alpha=0.25, color="red")
    ax1.set_ylabel("T_max (°C)")
    ax1.set_title("Pack max temperature", fontsize=9)
    ax1.legend(fontsize=7, loc="upper left", ncol=4)
    ax1.grid(True, alpha=0.25)

    # Panel 2 — Per-zone cooling actions
    ax2 = fig.add_subplot(gs[1])
    for z in range(min(n_zones, actions_2d.shape[1])):
        ax2.plot(time, actions_2d[:, z], color=ZONE_COLORS[z % len(ZONE_COLORS)],
                 lw=1.2, label=f"Zone {z}")
    ax2.set_ylabel("u (cooling)")
    ax2.set_title("Per-zone cooling commands", fontsize=9)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7, loc="upper right", ncol=n_zones)
    ax2.grid(True, alpha=0.25)

    # Panel 3 — Hotspot zone (meas) vs max-cooled zone
    ax3 = fig.add_subplot(gs[2])
    ax3.step(time, hot_zone_meas, color="red",  lw=1.5, where="post", label="Hottest zone (meas)")
    ax3.step(time, max_cool_zone, color="blue", lw=1.2, where="post", label="Max-cooled zone", ls="--")
    ax3.set_ylabel("Zone index")
    ax3.set_yticks(range(n_zones))
    ax3.set_title("Hotspot zone vs highest-cooled zone", fontsize=9)
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(True, alpha=0.25)

    # Panel 4 — Rolling targeting accuracy (window=50)
    ax4 = fig.add_subplot(gs[3])
    window = min(50, len(targeting))
    rolling_acc = pd.Series(targeting.astype(float)).rolling(window, min_periods=1).mean().values * 100
    ax4.plot(time, rolling_acc, color="purple", lw=1.3, label=f"Targeting acc (roll-{window})")
    ax4.axhline(float(np.mean(targeting)) * 100, color="purple", lw=0.8, ls="--",
                label=f"Episode mean = {float(np.mean(targeting))*100:.1f}%")
    ax4.set_ylabel("Targeting (%)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylim(-5, 105)
    ax4.set_title("Zone targeting accuracy (rolling)", fontsize=9)
    ax4.legend(fontsize=7, loc="lower right")
    ax4.grid(True, alpha=0.25)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Root-cause summary for one controller across all profiles/seeds
# ---------------------------------------------------------------------------

def print_root_cause(rows: List[Dict], ctrl_name: str) -> None:
    df = pd.DataFrame(rows)
    df_ctrl = df[df["controller"] == ctrl_name]
    if df_ctrl.empty:
        return

    failed = df_ctrl[~df_ctrl["is_safe"]]
    print(f"\n{'='*70}")
    print(f"  {ctrl_name} — root cause analysis")
    print(f"{'='*70}")
    print(f"  Safety pass rate:  {df_ctrl['is_safe'].mean()*100:.0f}%  "
          f"({int(df_ctrl['is_safe'].sum())}/{len(df_ctrl)} episodes)")
    print(f"  Mean u (all eps):  {df_ctrl['mean_cooling_action'].mean():.3f}")
    print(f"  Mean targeting:    {df_ctrl['zone_targeting_acc'].mean()*100:.1f}%")
    print(f"  T_max peak (mean): {df_ctrl['T_max_peak_C'].mean():.1f}°C")

    if not failed.empty:
        print(f"\n  Failed episodes ({len(failed)}):")
        print(f"  {'Profile':<18} {'Seed':>6} {'T_max':>7} {'AboveSafe':>10} {'u_mean':>7} {'Target%':>8}")
        print("  " + "-" * 65)
        for _, r in failed.iterrows():
            print(f"  {r['profile']:<18} {r['seed']:>6} {r['T_max_peak_C']:>6.1f}°C "
                  f"{r['time_above_safe_s']:>9.0f}s {r['mean_cooling_action']:>7.3f} "
                  f"{r['zone_targeting_acc']*100:>7.1f}%")

        print(f"\n  Failure breakdown:")
        undercool = (failed["mean_cooling_action"] < 0.25).sum()
        no_target = (failed["zone_targeting_acc"] < 0.20).sum()
        print(f"    Undercooling (u_mean < 0.25): {undercool}/{len(failed)} failures")
        print(f"    Poor targeting (acc < 20%):   {no_target}/{len(failed)} failures")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pack_config  = make_pack_config()
    cell_config  = CellConfig()
    sensor_cfg   = make_sensor_config(pack_config)
    actuator_cfg = make_actuator_config(pack_config)

    ppo_path = PROJECT_ROOT / "models" / "ppo_pack_3d_multizone_sensor" / "best_model.zip"
    sac_path = PROJECT_ROOT / "models" / "sac_pack_3d_multizone_sensor" / "best_model.zip"

    print("\n=== Loading RL models ===")
    ppo_ctrl = load_ppo(ppo_path, pack_config)
    sac_ctrl = load_sac(sac_path, pack_config)

    base_controllers = [c for c in [ppo_ctrl, sac_ctrl] if c is not None]
    if not base_controllers:
        print("No RL models loaded — cannot diagnose. Check model paths.")
        return

    # Build shielded variants
    all_controllers = list(base_controllers)
    for ctrl in base_controllers:
        shielded = SafetyShieldedRL(ctrl, pack_config, shield_temp_c=SHIELD_TEMP_C)
        all_controllers.append(shielded)

    print(f"\n=== Running diagnosis: {len(all_controllers)} controllers × "
          f"{len(PROFILE_NAMES)} profiles × {len(SEEDS)} seeds ===\n")

    rows: List[Dict] = []
    failure_count = 0

    for ctrl in all_controllers:
        print(f"  [{ctrl.name}]")
        for profile in PROFILE_NAMES:
            for seed in SEEDS:
                log, total_reward = run_episode(
                    ctrl, profile, pack_config, cell_config, seed,
                    sensor_cfg, actuator_cfg,
                )
                m = episode_metrics(log, pack_config)
                m.update({
                    "controller": ctrl.name,
                    "profile":    profile,
                    "seed":       seed,
                    "total_reward": total_reward,
                    "shielded":   isinstance(ctrl, SafetyShieldedRL),
                })
                rows.append(m)

                safe_str = "PASS" if m["is_safe"] else f"FAIL T_max={m['T_max_peak_C']:.1f}°C"
                print(f"    {profile:<18} seed={seed:>3}  {safe_str:<25}  "
                      f"u={m['mean_cooling_action']:.3f}  "
                      f"target={m['zone_targeting_acc']*100:.0f}%")

                # Plot failed unshielded episodes
                if not m["is_safe"] and not isinstance(ctrl, SafetyShieldedRL):
                    failure_count += 1
                    fname = f"failure_{ctrl.name.lower()}_{profile.lower()}_seed{seed:03d}.png"
                    plot_failure(
                        log, ctrl.name, profile, seed, pack_config,
                        OUTPUT_DIR / fname,
                    )
                    print(f"    --> plot: {fname}")

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "rl_diagnosis_summary.csv"
    df.to_csv(csv_path, index=False)

    # Root cause analysis
    for ctrl in base_controllers:
        print_root_cause(rows, ctrl.name)

    # Shield improvement table
    print(f"\n{'='*70}")
    print("  Safety shield improvement")
    print(f"{'='*70}")
    print(f"  {'Controller':<32} {'Base safe%':>10} {'Shield safe%':>12} {'Base u':>7} {'Shield u':>8}")
    print("  " + "-" * 72)
    for ctrl in base_controllers:
        base_rows    = df[(df["controller"] == ctrl.name) & ~df["shielded"]]
        shield_rows  = df[(df["controller"] == ctrl.name + " + Shield") & df["shielded"]]
        if base_rows.empty or shield_rows.empty:
            continue
        base_safe    = base_rows["is_safe"].mean() * 100
        shield_safe  = shield_rows["is_safe"].mean() * 100
        base_u       = base_rows["mean_cooling_action"].mean()
        shield_u     = shield_rows["mean_cooling_action"].mean()
        print(f"  {ctrl.name:<32} {base_safe:>9.0f}%  {shield_safe:>11.0f}%  "
              f"{base_u:>6.3f}  {shield_u:>7.3f}")

    # Text report
    report_lines = [
        "RL Failure Diagnosis Report",
        "=" * 70,
        f"Models: {ppo_path.name if ppo_ctrl else 'N/A'} (PPO), "
        f"{sac_path.name if sac_ctrl else 'N/A'} (SAC)",
        f"Profiles: {PROFILE_NAMES}",
        f"Seeds: {SEEDS}",
        f"Shield threshold: {SHIELD_TEMP_C}°C",
        "",
        "Summary:",
    ]
    for ctrl in all_controllers:
        ctrl_df = df[df["controller"] == ctrl.name]
        if ctrl_df.empty:
            continue
        report_lines.append(
            f"  {ctrl.name:<40} safe={ctrl_df['is_safe'].mean()*100:.0f}%  "
            f"u_mean={ctrl_df['mean_cooling_action'].mean():.3f}  "
            f"target={ctrl_df['zone_targeting_acc'].mean()*100:.1f}%  "
            f"T_max={ctrl_df['T_max_peak_C'].mean():.1f}°C"
        )
    report_lines += ["", f"Failure plots written to: {OUTPUT_DIR}/", f"CSV: {csv_path}"]
    report_path = OUTPUT_DIR / "rl_diagnosis_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")

    print(f"\nSaved:")
    print(f"  {csv_path}")
    print(f"  {report_path}")
    print(f"  {failure_count} failure plot(s) in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
