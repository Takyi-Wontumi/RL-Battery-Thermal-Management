"""
models/electrochemical_model.py

HPPC-based electrochemical heat generation model for Samsung INR18650-25R.

WHY this matters over a prescribed heat profile:
    A prescribed profile (sinusoidal, step, etc.) is a pattern, not physics.
    Real heat generation depends on current AND state of charge — resistance rises
    sharply at low SOC, so the same current produces more heat late in discharge.
    This model couples heat to the electrical state so the controller must manage
    both thermal and electrochemical limits simultaneously.

Physics (per cell):
    Q_gen = Q_joule + Q_entropic
    Q_joule    = I_cell² · R_dc(SOC)          [Ohmic / irreversible]
    Q_entropic = I_cell · (T_cell + 273.15) · dU/dT  [reversible entropic term]

SOC evolution:
    SOC(t+dt) = SOC(t) - I_cell · dt / (3600 · C_Ah)

Reference:
    Samsung INR18650-25R (NMC cathode / graphite anode)
    R_dc(SOC) table: representative HPPC values at ~25 °C.
    Entropic coefficient: conservative NMC-graphite estimate.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# HPPC resistance lookup table — Samsung INR18650-25R
# ---------------------------------------------------------------------------

# SOC breakpoints (fraction, 0 = empty, 1 = full)
HPPC_SOC = np.array([
    0.00, 0.05, 0.10, 0.20, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
], dtype=np.float64)

# DC internal resistance [Ohm] measured via HPPC 10-s discharge pulse
# Pattern: high at low SOC (diffusion-limited lithiation), minimum ~0.022 Ω at 60-70% SOC,
# slight upturn near full charge (intercalation strain).
HPPC_R_DC = np.array([
    0.055, 0.042, 0.036, 0.030, 0.026,
    0.024, 0.022, 0.022, 0.022, 0.023, 0.025, 0.030,
], dtype=np.float64)

# Entropic heat coefficient dU/dT [V/K] for NMC-graphite.
# Negative means discharge is exothermic (adds to Joule heat).
# Magnitude is small (~0.1 mV/K) — included for physical completeness.
ENTROPIC_DUDT_V_PER_K: float = -0.0001


def hppc_resistance(soc: np.ndarray) -> np.ndarray:
    """
    Interpolate DC internal resistance [Ohm] from the HPPC table.

    Args:
        soc: SOC values in [0, 1], any shape.
    Returns:
        R_dc: Resistance array, same shape as soc, in Ohm.
    """
    return np.interp(np.asarray(soc, dtype=np.float64), HPPC_SOC, HPPC_R_DC)


# ---------------------------------------------------------------------------
# Current profile callables
# ---------------------------------------------------------------------------

CurrentProfileFn = Callable[[float], float]  # t_s → I_pack [A]


def constant_current_profile(i_pack_a: float) -> CurrentProfileFn:
    """Constant pack discharge current."""
    def profile(t: float) -> float:
        return float(i_pack_a)
    return profile


def sinusoidal_current_profile(
    i_mean_a: float = 15.0,
    i_amplitude_a: float = 8.0,
    period_s: float = 300.0,
) -> CurrentProfileFn:
    """Sinusoidal current simulating a charge-sustaining drive cycle."""
    def profile(t: float) -> float:
        return float(max(0.0, i_mean_a + i_amplitude_a * np.sin(2.0 * np.pi * t / period_s)))
    return profile


def stepped_current_profile(
    steps: list[tuple[float, float]],
) -> CurrentProfileFn:
    """
    Piecewise constant current profile.

    Args:
        steps: List of (time_start_s, current_a) pairs, sorted by time.
               e.g. [(0, 10), (600, 20), (1200, 5)]
    """
    times = np.array([s[0] for s in steps], dtype=np.float64)
    currents = np.array([s[1] for s in steps], dtype=np.float64)

    def profile(t: float) -> float:
        idx = int(np.searchsorted(times, t, side='right')) - 1
        idx = max(0, min(idx, len(currents) - 1))
        return float(currents[idx])
    return profile


# ---------------------------------------------------------------------------
# Stateful electrochemical heat profile
# ---------------------------------------------------------------------------

class ElectrochemicalHeatProfile:
    """
    Cell-resolved HPPC heat generation profile with SOC tracking.

    WHY stateful:
        SOC must be integrated over time — each step's heat output depends on
        the accumulated current drawn so far.  The environment resets this
        object at the start of each episode.

    WHY per-cell SOC variation:
        Manufacturing spread means cells in the same parallel group drift apart
        in SOC and resistance over time.  This creates realistic non-uniform heat
        maps that a single-zone controller cannot handle well, but a multi-zone
        controller can adapt to.

    Usage:
        profile = ElectrochemicalHeatProfile(...)
        # In env.reset():
        profile.reset()
        # In env.step():
        q_gen = profile(t, rng, shape, T=thermal_model.T)
    """

    def __init__(
        self,
        parallel_count: int,
        capacity_ah: float,
        dt_s: float,
        pack_shape: Tuple[int, int, int],
        current_profile: CurrentProfileFn,
        soc_initial: float = 0.80,
        soc_variation_std: float = 0.015,
        resistance_variation_std: float = 0.008,
        include_entropic: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Args:
            parallel_count:           Cells in parallel — current divides evenly.
            capacity_ah:              Capacity per physical cell [Ah].
            dt_s:                     Simulation timestep [s] — must match env dt.
            pack_shape:               (Nx, Ny, Nz) grid.
            current_profile:          Callable: t_s → pack current [A].
            soc_initial:              Starting SOC (0=empty, 1=full).
            soc_variation_std:        Cell-to-cell SOC spread at reset.
            resistance_variation_std: Multiplicative R spread (manufacturing).
            include_entropic:         Add reversible entropic heat term.
            rng:                      Optional RNG for reproducibility.
        """
        self.parallel_count = parallel_count
        self.capacity_ah = capacity_ah
        self.dt_s = dt_s
        self.pack_shape = pack_shape
        self.current_profile = current_profile
        self.soc_initial = soc_initial
        self.soc_variation_std = soc_variation_std
        self.resistance_variation_std = resistance_variation_std
        self.include_entropic = include_entropic
        self.rng = rng if rng is not None else np.random.default_rng(0)

        self.SOC = np.full(pack_shape, soc_initial, dtype=np.float64)
        self._r_multiplier = np.ones(pack_shape, dtype=np.float64)
        self.reset(soc_initial)

    def reset(self, soc_initial: Optional[float] = None) -> None:
        """Reset SOC and resample per-cell variation. Call at episode start."""
        if soc_initial is None:
            soc_initial = self.soc_initial

        self.SOC = np.clip(
            self.rng.normal(soc_initial, self.soc_variation_std, size=self.pack_shape),
            0.01, 0.99,
        ).astype(np.float64)

        # Fixed per-episode resistance spread — captures manufacturing variation
        self._r_multiplier = np.clip(
            self.rng.normal(1.0, self.resistance_variation_std, size=self.pack_shape),
            0.80, 1.20,
        ).astype(np.float64)

    def __call__(
        self,
        t: float,
        rng: np.random.Generator,
        shape: Tuple[int, int, int],
        T: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute Q_gen [W] for all cells and advance SOC by one timestep.

        The entropic term requires cell temperature T [°C].  If T is None,
        the entropic contribution is omitted (safe fallback for warm-up steps).
        """
        I_pack = self.current_profile(t)
        I_cell = I_pack / self.parallel_count  # A per cell in the parallel group

        R_cell = hppc_resistance(self.SOC) * self._r_multiplier  # Ohm, (Nx,Ny,Nz)

        Q_joule = I_cell**2 * R_cell  # W

        Q_entropic = np.zeros_like(Q_joule)
        if self.include_entropic and T is not None:
            T_K = np.asarray(T, dtype=np.float64) + 273.15  # K
            Q_entropic = I_cell * T_K * ENTROPIC_DUDT_V_PER_K  # W

        # SOC decreases as charge is consumed
        delta_soc = I_cell * self.dt_s / (3600.0 * self.capacity_ah)
        self.SOC = np.clip(self.SOC - delta_soc, 0.0, 1.0)

        return np.maximum(Q_joule + Q_entropic, 0.0).astype(np.float64)

    @property
    def soc_mean(self) -> float:
        return float(np.mean(self.SOC))

    @property
    def soc_min(self) -> float:
        return float(np.min(self.SOC))

    @property
    def soc_max(self) -> float:
        return float(np.max(self.SOC))
