"""
models/thermal_model_3d.py

3D lumped-parameter battery pack thermal network.

Each cell occupies one grid node T[i, j, k]. The governing ODE per cell is:

    C * dT[i,j,k]/dt = Q_gen[i,j,k]
                      + sum_neighbors( g * (T[n] - T[i,j,k]) )
                      - h * A_exposed[i,j,k] * (T[i,j,k] - T_amb)

Where:
    C           = cell heat capacity (J/K)
    Q_gen       = internal heat generation (W)
    g           = neighbor conduction conductance (W/K)
    A_exposed   = exposed surface area (geometry-based, m²)
    h           = convective heat transfer coefficient (W/m²·K)

This is NOT CFD. It is a control-oriented thermal network for developing and
comparing battery cooling strategies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.pack_config import CellConfig, PackConfig, compute_cell_surface_area, compute_cell_heat_capacity


class BatteryPackThermal3D:
    """
    3D cell-resolved lumped-parameter battery pack thermal model.

    Temperature is stored as a 3D NumPy array T[i, j, k] where i/j/k index
    the cell position in x/y/z.  Conduction and cooling are vectorized —
    no Python-level loops over cells during time integration.
    """

    def __init__(
        self,
        cell_config: CellConfig,
        pack_config: PackConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.cell = cell_config
        self.pack = pack_config
        self.Nx, self.Ny, self.Nz = pack_config.shape

        self.rng = rng if rng is not None else np.random.default_rng(0)

        self.cell_area = compute_cell_surface_area(cell_config)
        self.cell_heat_capacity = compute_cell_heat_capacity(cell_config)

        # Precompute exposed-area factor for every cell (vectorized)
        self._exposed_area = self._build_exposed_area_array()

        # Initial heat generation array — updated by environment each step
        self.q_gen = np.ones(pack_config.shape, dtype=np.float64) * cell_config.q_gen_nominal_w
        self._apply_initial_heat_variation()

        self.T = np.full(pack_config.shape, pack_config.initial_temp_c, dtype=np.float64)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_exposed_area_array(self) -> np.ndarray:
        """
        Precompute how many of each cell's 6 faces are exposed to the coolant.
        Interior cells: 0 exposed faces (surrounded on all sides).
        Corner cells: up to 3 exposed faces.

        Returns array of shape (Nx, Ny, Nz) with values in [0, 1] representing
        the fraction of the total cell surface area exposed to cooling.
        """
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        exposed_faces = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        exposed_faces[0, :, :] += 1
        exposed_faces[-1, :, :] += 1
        exposed_faces[:, 0, :] += 1
        exposed_faces[:, -1, :] += 1
        exposed_faces[:, :, 0] += 1
        exposed_faces[:, :, -1] += 1
        return exposed_faces / 6.0

    def _apply_initial_heat_variation(self) -> None:
        if not self.pack.enable_heat_variation:
            return
        variation = self.rng.normal(1.0, self.pack.heat_variation_std, size=self.pack.shape)
        variation = np.clip(variation, 0.5, 2.0)
        self.q_gen = self.q_gen * variation

        if self.pack.enable_center_hotspot:
            cx, cy, cz = self.Nx // 2, self.Ny // 2, self.Nz // 2
            self.q_gen[cx, cy, cz] *= self.pack.hotspot_multiplier

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, initial_temp_c: Optional[float] = None) -> np.ndarray:
        if initial_temp_c is None:
            initial_temp_c = self.pack.initial_temp_c
        self.T = np.full(self.pack.shape, initial_temp_c, dtype=np.float64)
        return self.T.copy()

    # ------------------------------------------------------------------
    # Physics kernels (fully vectorized — no Python loops over cells)
    # ------------------------------------------------------------------

    def compute_conduction(self) -> np.ndarray:
        """
        Vectorized neighbor conduction using numpy slice arithmetic.

        For each pair of adjacent cells (A, B):
            Q_cond_B += g * (T_A - T_B)
            Q_cond_A += g * (T_B - T_A)

        Equivalent to triple for-loop over cells but ~50–100× faster.
        """
        g = self.pack.g_cond_w_per_k
        T = self.T
        Q = np.zeros_like(T)

        # x-direction (i axis)
        # dT_x[i] = T[i] - T[i+1]; if > 0, heat flows from i to i+1
        dT_x = T[:-1, :, :] - T[1:, :, :]
        Q[:-1, :, :] -= g * dT_x   # cell i loses heat to i+1
        Q[1:, :, :] += g * dT_x    # cell i+1 gains heat from i

        # y-direction (j axis)
        dT_y = T[:, :-1, :] - T[:, 1:, :]
        Q[:, :-1, :] -= g * dT_y
        Q[:, 1:, :] += g * dT_y

        # z-direction (k axis)
        dT_z = T[:, :, :-1] - T[:, :, 1:]
        Q[:, :, :-1] -= g * dT_z
        Q[:, :, 1:] += g * dT_z

        return Q

    def compute_cooling(self, u_zones: np.ndarray) -> np.ndarray:
        """
        Convective cooling — accepts per-cell, per-x-column, or scalar commands.

        Routing logic (checked in order):
          1. shape == (Nx, Ny, Nz)  → per-cell commands (quadrant-zone path).
          2. size == 1              → scalar broadcast to all cells.
          3. size == Nx             → per-x-column broadcast (legacy path).

        Args:
            u_zones: Cooling commands in [0, 1].
                     Shape (Nx, Ny, Nz), (Nx,), or scalar.

        Returns:
            Q_cool: Heat removed [W], shape (Nx, Ny, Nz).
        """
        u = np.asarray(u_zones, dtype=np.float64)

        if u.shape == (self.Nx, self.Ny, self.Nz):
            # Per-cell commands from quadrant-zone environment
            u_cell = np.clip(u, 0.0, 1.0)
        elif u.size == 1:
            u_cell = np.full((self.Nx, self.Ny, self.Nz), float(u.flat[0]))
            u_cell = np.clip(u_cell, 0.0, 1.0)
        else:
            # Per-x-column broadcast (legacy: zone = x-column)
            u_col = np.clip(u.reshape(-1), 0.0, 1.0)
            u_cell = u_col[:, np.newaxis, np.newaxis] * np.ones((self.Nx, self.Ny, self.Nz))

        h_cell = (
            self.pack.h_min_w_per_m2_k
            + u_cell * (self.pack.h_max_w_per_m2_k - self.pack.h_min_w_per_m2_k)
        )
        exposed_area = self.cell_area * self._exposed_area  # (Nx, Ny, Nz)
        dT = np.maximum(self.T - self.pack.ambient_temp_c, 0.0)
        return h_cell * exposed_area * dT

    # ------------------------------------------------------------------
    # Time step
    # ------------------------------------------------------------------

    def step(self, u_zones, dt: float, q_gen: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Advance one time step.

        Args:
            u_zones: Cooling command(s).  Scalar or shape (Nx,) array.
                     Scalar is broadcast to all zones (backward compatible).
            dt:      Time step size [s].
            q_gen:   Optional 3D heat generation array [W], shape (Nx, Ny, Nz).
                     If provided, overrides stored q_gen for this step.

        Returns:
            T_copy:  Copy of the updated temperature array.
            metrics: Dict with T_max, T_avg, T_min, T_gradient, safe, critical,
                     q_cool_total, q_gen_total, q_cool_per_zone.
        """
        if q_gen is not None:
            self.q_gen = q_gen

        Q_cond = self.compute_conduction()
        Q_cool = self.compute_cooling(u_zones)

        dTdt = (self.q_gen + Q_cond - Q_cool) / self.cell_heat_capacity
        self.T = self.T + dt * dTdt

        metrics = self.get_metrics()
        metrics["q_cool_total"] = float(np.sum(Q_cool))
        metrics["q_gen_total"] = float(np.sum(self.q_gen))
        # Per-zone cooling power — useful for energy accounting and reward shaping
        metrics["q_cool_per_zone"] = np.sum(Q_cool, axis=(1, 2)).tolist()
        return self.T.copy(), metrics

    # ------------------------------------------------------------------
    # Metrics and observation
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict:
        T_max = float(np.max(self.T))
        T_avg = float(np.mean(self.T))
        T_min = float(np.min(self.T))
        return {
            "T_max": T_max,
            "T_avg": T_avg,
            "T_min": T_min,
            "T_gradient": T_max - T_min,
            "safe": T_max <= self.pack.safe_temp_c,
            "critical": T_max >= self.pack.critical_temp_c,
        }

    def get_thermal_obs(self) -> np.ndarray:
        """
        Six normalized pack-level thermal features (no action history).

        obs = [T_max_norm, T_avg_norm, T_min_norm,
               T_gradient_norm, T_variance_norm, T_center_norm]

        All temperatures normalized by (safe_temp - target_temp):
            0  → at target temperature
            1  → at the safe limit
        """
        metrics = self.get_metrics()
        scale = max(1e-6, self.pack.safe_temp_c - self.pack.target_temp_c)
        target = self.pack.target_temp_c

        T_center = float(self.T[self.Nx // 2, self.Ny // 2, self.Nz // 2])
        T_variance = float(np.std(self.T))

        return np.array([
            (metrics["T_max"] - target) / scale,
            (metrics["T_avg"] - target) / scale,
            (metrics["T_min"] - target) / scale,
            metrics["T_gradient"] / scale,
            T_variance / scale,
            (T_center - target) / scale,
        ], dtype=np.float32)

    def get_observation(self, u_prev: float = 0.0) -> np.ndarray:
        """
        Backward-compatible 7-element observation (single-zone legacy API).
        Appends the scalar u_prev to the 6 thermal features.
        """
        thermal = self.get_thermal_obs()
        return np.append(thermal, float(u_prev)).astype(np.float32)

    def get_zone_max_temps(self) -> np.ndarray:
        """Return the maximum temperature per x-zone, shape (Nx,)."""
        return np.max(self.T, axis=(1, 2))  # max over Ny, Nz for each zone


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from configs.pack_config import CellConfig, PackConfig

    cell = CellConfig()
    pack = PackConfig(shape=(4, 3, 2), enable_heat_variation=True)

    model = BatteryPackThermal3D(cell, pack)
    print(f"Pack shape:         {pack.shape}  ({4*3*2} cells)")
    print(f"Cell heat capacity: {model.cell_heat_capacity:.1f} J/K")
    print(f"Cell surface area:  {model.cell_area*1e4:.2f} cm²")
    print(f"Initial T_max:      {model.T.max():.2f} °C")

    for _ in range(100):
        T, metrics = model.step(u_zones=0.5, dt=1.0)

    print(f"After 100 s | T_max={metrics['T_max']:.2f}  T_avg={metrics['T_avg']:.2f}  "
          f"T_gradient={metrics['T_gradient']:.3f}  safe={metrics['safe']}")
