"""
scripts/compare_pack_baselines_3d.py

Phase 2 baseline benchmark for the 3D cell-resolved battery pack thermal model.

Run from project root:
    python -m scripts.compare_pack_baselines_3d

Expected outputs:
    outputs/phase2_3d_baseline_summary.csv
    outputs/phase2_3d_baseline_temperatures.png
    outputs/phase2_3d_baseline_actions.png
    outputs/phase2_3d_baseline_gradient.png
    outputs/phase2_3d_baseline_heat_profiles.png
    outputs/phase2_3d_layer_heatmaps_<profile>.png  (one per profile, final step)

All controllers receive the 5-element normalized observation:
    obs = [T_max_norm, T_avg_norm, T_min_norm, T_gradient_norm, u_prev]

Controllers respond primarily to T_max (obs[0]) — not T_avg — because average
temperature hides dangerous local overheating.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Protocol, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.pack_config import PackConfig, CellConfig
from envs.battery_pack_thermal_env_3d import (
    BatteryPackThermalEnv3D,
    make_3d_profile,
    Pack3DHeatProfile,
)


# ---------------------------------------------------------------------------
# Observation decoder
# ---------------------------------------------------------------------------

def extract_3d_pack_state(obs: np.ndarray, pack_config: PackConfig) -> Dict:
    """Convert normalized 5-element obs back to physical temperature values."""
    normalizer = max(1e-6, pack_config.safe_temp_c - pack_config.target_temp_c)
    return {
        "T_max": float(obs[0]) * normalizer + pack_config.target_temp_c,
        "T_avg": float(obs[1]) * normalizer + pack_config.target_temp_c,
        "T_min": float(obs[2]) * normalizer + pack_config.target_temp_c,
        "T_gradient": float(obs[3]) * normalizer,
        "u_prev": float(obs[4]),
    }


# ---------------------------------------------------------------------------
# Controller protocol
# ---------------------------------------------------------------------------

class Pack3DController(Protocol):
    name: str

    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Baseline controllers (all use T_max, not T_avg)
# ---------------------------------------------------------------------------

@dataclass
class NoCooling3D:
    name: str = "No cooling"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


@dataclass
class ConstantCooling3D:
    cooling_level: float = 0.5
    name: str = "Constant cooling"

    def __post_init__(self) -> None:
        self.cooling_level = float(np.clip(self.cooling_level, 0.0, 1.0))
        self.name = f"Constant cooling u={self.cooling_level:.2f}"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.array([self.cooling_level], dtype=np.float32)


@dataclass
class BangBang3D:
    pack_config: PackConfig
    target_temp_c: float = 35.0
    deadband_c: float = 1.0
    name: str = "Bang-bang (T_max)"

    def __post_init__(self) -> None:
        self._is_high: bool = False

    def reset(self) -> None:
        self._is_high = False

    def act(self, obs: np.ndarray) -> np.ndarray:
        state = extract_3d_pack_state(obs, self.pack_config)
        T_max = state["T_max"]

        if T_max >= self.target_temp_c + self.deadband_c:
            self._is_high = True
        elif T_max <= self.target_temp_c - self.deadband_c:
            self._is_high = False

        u = 1.0 if self._is_high else 0.0
        return np.array([u], dtype=np.float32)


@dataclass
class Proportional3D:
    pack_config: PackConfig
    target_temp_c: float = 35.0
    kp: float = 0.10
    bias: float = 0.15
    imbalance_gain: float = 0.04
    name: str = "Proportional (T_max)"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        state = extract_3d_pack_state(obs, self.pack_config)
        error = state["T_max"] - self.target_temp_c
        u = self.bias + self.kp * error + self.imbalance_gain * state["T_gradient"]
        u = float(np.clip(u, 0.0, 1.0))
        return np.array([u], dtype=np.float32)


@dataclass
class PI3D:
    pack_config: PackConfig
    target_temp_c: float = 35.0
    kp: float = 0.30
    ki: float = 0.001
    bias: float = 0.30
    imbalance_gain: float = 0.08
    integral_limit: float = 50.0
    dt: float = 1.0
    name: str = "PI (T_max)"

    def __post_init__(self) -> None:
        self.integral_error: float = 0.0

    def reset(self) -> None:
        self.integral_error = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        state = extract_3d_pack_state(obs, self.pack_config)
        error = state["T_max"] - self.target_temp_c

        self.integral_error += error * self.dt
        self.integral_error = float(
            np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        )

        u = (
            self.bias
            + self.kp * error
            + self.ki * self.integral_error
            + self.imbalance_gain * state["T_gradient"]
        )
        u = float(np.clip(u, 0.0, 1.0))
        return np.array([u], dtype=np.float32)


def build_3d_baseline_controllers(pack_config: PackConfig) -> List[Pack3DController]:
    return [
        NoCooling3D(),
        ConstantCooling3D(cooling_level=0.5),
        ConstantCooling3D(cooling_level=1.0),
        BangBang3D(pack_config=pack_config, target_temp_c=pack_config.target_temp_c),
        Proportional3D(
            pack_config=pack_config,
            target_temp_c=pack_config.target_temp_c,
            kp=0.10,
            bias=0.15,
            imbalance_gain=0.04,
        ),
        PI3D(
            pack_config=pack_config,
            target_temp_c=pack_config.target_temp_c,
            kp=0.30,
            ki=0.001,
            bias=0.30,
            imbalance_gain=0.08,
            integral_limit=50.0,
            name="PI tuned (T_max)",
        ),
    ]


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_controller_case(
    controller: Pack3DController,
    profile_name: str,
    cell_config: CellConfig,
    pack_config: PackConfig,
    seed: int = 7,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    env = BatteryPackThermalEnv3D(
        cell_config=cell_config,
        pack_config=pack_config,
        heat_profile=make_3d_profile(profile_name),
        seed=seed,
    )

    controller.reset()
    obs, info = env.reset(seed=seed, options={"randomize": False})

    terminated = truncated = False
    total_reward = 0.0
    final_T3d = None

    while not (terminated or truncated):
        action = controller.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        final_T3d = info["temperatures_3d"]

    log = env.get_episode_log()

    # Derive a scalar action series from multi-zone actions for backward-compatible
    # metrics and plotting (mean across zones each step).
    if "actions" in log and "action" not in log:
        actions_2d = np.asarray(log["actions"])          # shape (T, n_zones)
        log["action"] = np.mean(actions_2d, axis=1)      # shape (T,) mean per step

    failed = bool(terminated and log["T_max"].max() >= pack_config.critical_temp_c)
    time_above_safe = float(np.sum(log["T_max"] > pack_config.safe_temp_c) * env.dt_s)

    # Safety status for two-stage ranking (R2)
    is_safe = (time_above_safe == 0.0) and (float(log["T_max"].max()) < pack_config.critical_temp_c)

    metrics: Dict = {
        "profile": profile_name,
        "controller": controller.name,
        "T_max_peak_C": float(log["T_max"].max()),
        "T_max_final_C": float(log["T_max"][-1]),
        "T_avg_peak_C": float(log["T_avg"].max()),
        "T_avg_final_C": float(log["T_avg"][-1]),
        "T_gradient_max_C": float(log["T_gradient"].max()),
        "T_gradient_mean_C": float(log["T_gradient"].mean()),
        "n_cells_above_safe_peak": int(log["n_cells_above_safe"].max()),
        "time_above_safe_s": time_above_safe,
        "mean_cooling_action": float(log["action"].mean()),
        "total_cooling_effort": float(log["action"].sum() * env.dt_s),
        "action_variation": float(np.abs(np.diff(log["action"])).sum()),
        "total_reward": float(total_reward),
        "is_safe": bool(is_safe),
        "failed": failed,
    }

    # Mean reward component values for per-controller diagnosis (R6)
    from envs.battery_pack_thermal_env_3d import _REWARD_COMPONENT_KEYS
    for k in _REWARD_COMPONENT_KEYS:
        arr = log.get(k)
        if arr is not None and len(arr) > 0:
            metrics[f"mean_{k}"] = float(np.mean(arr))

    log["final_T3d"] = final_T3d
    return log, metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PROFILE_NAMES = ["UniformConstant", "NonuniformStep", "PulsedHotspot", "RandomNonuniform"]


def _plot_metric_grid(
    all_logs: Dict[Tuple[str, str], Dict],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    hlines: Optional[List[Tuple[float, str, str]]] = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, PROFILE_NAMES):
        for (pname, cname), log in all_logs.items():
            if pname != profile_name:
                continue
            lw = 2.3 if "PPO" in cname else 1.3
            ax.plot(log["time"], log[metric_key], linewidth=lw, label=cname)

        if hlines:
            for val, ls, label in hlines:
                ax.axhline(val, linestyle=ls, linewidth=1.0, color="gray", label=label if profile_name == PROFILE_NAMES[0] else None)

        ax.set_title(profile_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.82, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_layer_heatmaps(
    log: Dict,
    profile_name: str,
    pack_config: PackConfig,
    output_path: Path,
) -> None:
    """Per-layer temperature heatmap at the final simulation step."""
    T3d = log.get("final_T3d")
    if T3d is None:
        return

    Nz = pack_config.shape[2]
    fig, axes = plt.subplots(1, Nz, figsize=(5 * Nz, 4))
    if Nz == 1:
        axes = [axes]

    vmin = pack_config.ambient_temp_c
    vmax = pack_config.safe_temp_c

    for k, ax in enumerate(axes):
        im = ax.imshow(
            T3d[:, :, k],
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="RdYlBu_r",
        )
        ax.set_title(f"Layer z={k}")
        ax.set_xlabel("y index")
        ax.set_ylabel("x index")
        fig.colorbar(im, ax=ax, label="Temperature (°C)")

    fig.suptitle(
        f"3D pack temperature field — {profile_name}\n(final step, best PI controller)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heat_profiles(all_logs: Dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.flatten()
    fig.suptitle("Phase 2 — 3D pack heat generation profiles", fontsize=14, fontweight="bold")

    for ax, profile_name in zip(axes, PROFILE_NAMES):
        log = next(
            (l for (pn, _), l in all_logs.items() if pn == profile_name), None
        )
        if log is None:
            continue
        ax.plot(log["time"], log["q_gen_total"], linewidth=1.6, label="Total heat (W)")
        ax.plot(log["time"], log["q_gen_max_cell"], linewidth=1.0, linestyle="--", label="Max cell heat (W)")
        ax.set_title(profile_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heat generation (W)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.84, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_rankings(df: pd.DataFrame, pack_config: Optional["PackConfig"] = None) -> None:
    """
    Two-stage ranking (R2):
      Stage 1 — Safety (pass/fail): time_above_safe == 0 AND T_max < critical
      Stage 2 — Among safe controllers: ranked by mean total reward (descending)
    """
    agg = df.groupby("controller").agg(
        is_safe=("is_safe", "all"),
        total_reward=("total_reward", "mean"),
        T_max_peak=("T_max_peak_C", "mean"),
        time_above_safe=("time_above_safe_s", "mean"),
        T_gradient_mean=("T_gradient_mean_C", "mean"),
        mean_cooling=("mean_cooling_action", "mean"),
    )

    safe_ctrl   = agg[agg["is_safe"]].sort_values("total_reward", ascending=False)
    unsafe_ctrl = agg[~agg["is_safe"]].sort_values("T_max_peak", ascending=True)

    W = 72
    print("\n" + "=" * W)
    print("  STAGE 1 — Safety check  (pass = 0s above safe AND T_max < critical)")
    print("=" * W)
    for name, row in agg.iterrows():
        status = "PASS" if row["is_safe"] else f"FAIL  ({row['time_above_safe']:.0f}s above safe)"
        print(f"  {'PASS' if row['is_safe'] else 'FAIL':4s}  {name}")

    print("\n" + "=" * W)
    print("  STAGE 2 — Ranking among SAFE controllers  (higher reward = better)")
    print(f"  {'Rank':<5} {'Controller':<32} {'Reward':>8} {'T_max':>8} {'Grad':>7} {'u_mean':>7}")
    print("  " + "-" * (W - 2))
    for rank, (name, row) in enumerate(safe_ctrl.iterrows(), 1):
        tag = " ← RL" if ("SAC" in name or "PPO" in name) else ""
        print(f"  {rank:<5} {name:<32} {row['total_reward']:>8.2f} "
              f"{row['T_max_peak']:>7.2f}° {row['T_gradient_mean']:>6.2f}° "
              f"{row['mean_cooling']:>6.3f}{tag}")

    if not unsafe_ctrl.empty:
        print("\n  UNSAFE controllers (excluded from ranking):")
        for name, row in unsafe_ctrl.iterrows():
            print(f"  ✗  {name:<32}  T_max={row['T_max_peak']:.2f}°C  "
                  f"above_safe={row['time_above_safe']:.0f}s")

    # Reward component breakdown (R6)
    comp_cols = [c for c in df.columns if c.startswith("mean_reward_")]
    if comp_cols:
        print("\n" + "=" * W)
        print("  Reward component breakdown (mean per step, higher = better)")
        print(f"  {'Controller':<32} " + "  ".join(f"{c.replace('mean_reward_','')[:8]:>8}" for c in comp_cols))
        print("  " + "-" * (W - 2))
        for name in list(safe_ctrl.index) + list(unsafe_ctrl.index):
            row = df[df["controller"] == name][comp_cols].mean()
            vals = "  ".join(f"{row[c]:>8.3f}" for c in comp_cols)
            print(f"  {name:<32} {vals}")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

from typing import Optional  # noqa: E402 — after type annotations used above


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_config = CellConfig()
    pack_config = PackConfig(shape=(4, 3, 2))

    controllers = build_3d_baseline_controllers(pack_config)

    all_logs: Dict[Tuple[str, str], Dict] = {}
    summary_rows: List[Dict] = []

    for profile_name in PROFILE_NAMES:
        for controller in controllers:
            log, metrics = run_controller_case(
                controller=controller,
                profile_name=profile_name,
                cell_config=cell_config,
                pack_config=pack_config,
                seed=7,
            )
            all_logs[(profile_name, controller.name)] = log
            summary_rows.append(metrics)

            print(
                f"{profile_name:16s} | {controller.name:28s} | "
                f"T_max={metrics['T_max_peak_C']:.1f}°C  "
                f"grad={metrics['T_gradient_max_C']:.2f}°C  "
                f"above_safe={metrics['time_above_safe_s']:.0f}s  "
                f"reward={metrics['total_reward']:.1f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "phase2_3d_baseline_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    # Max temperature
    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="T_max",
        title="Phase 2 (3D) — Max cell temperature",
        ylabel="T_max (°C)",
        output_path=output_dir / "phase2_3d_baseline_temperatures.png",
        hlines=[
            (pack_config.target_temp_c, ":", "Target"),
            (pack_config.safe_temp_c, "--", "Safe limit"),
        ],
    )

    # Cooling actions
    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="action",
        title="Phase 2 (3D) — Cooling commands",
        ylabel="Cooling command u",
        output_path=output_dir / "phase2_3d_baseline_actions.png",
    )

    # Temperature gradient
    _plot_metric_grid(
        all_logs=all_logs,
        metric_key="T_gradient",
        title="Phase 2 (3D) — Temperature gradient (T_max - T_min)",
        ylabel="T_gradient (°C)",
        output_path=output_dir / "phase2_3d_baseline_gradient.png",
    )

    # Heat profiles
    plot_heat_profiles(all_logs=all_logs, output_path=output_dir / "phase2_3d_baseline_heat_profiles.png")

    # Layer heatmaps — use the best PI controller for each profile
    pi_name = "PI tuned (T_max)"
    for profile_name in PROFILE_NAMES:
        key = (profile_name, pi_name)
        if key in all_logs:
            plot_layer_heatmaps(
                log=all_logs[key],
                profile_name=profile_name,
                pack_config=pack_config,
                output_path=output_dir / f"phase2_3d_layer_heatmaps_{profile_name.lower()}.png",
            )

    print_rankings(summary_df)

    print("\nSaved outputs:")
    print(f"  {csv_path}")
    print(f"  {output_dir / 'phase2_3d_baseline_temperatures.png'}")
    print(f"  {output_dir / 'phase2_3d_baseline_actions.png'}")
    print(f"  {output_dir / 'phase2_3d_baseline_gradient.png'}")
    print(f"  {output_dir / 'phase2_3d_baseline_heat_profiles.png'}")
    for p in PROFILE_NAMES:
        print(f"  {output_dir / f'phase2_3d_layer_heatmaps_{p.lower()}.png'}")


if __name__ == "__main__":
    main()
