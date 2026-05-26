"""
configs/pack_config.py

Geometry, thermal, and cooling configuration for the 3D cell-resolved battery
pack thermal network.

This is not CFD. It is a lumped-parameter thermal network with cell-resolved
states and geometry-based cooling exposure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class CellConfig:
    # Cylindrical cell geometry — 18650-class dimensions used for pack sizing
    # and exposed-area calculations.  Thermal mass is scaled to represent a
    # small cell group (not a single cylindrical cell), giving physically
    # realistic thermal dynamics over an 1800 s control episode.
    diameter_m: float = 0.018     # m
    length_m: float = 0.065       # m

    # Thermal properties (scaled to ~8–10 18650 cells acting as one node)
    mass_kg: float = 0.40         # kg  (effective module mass per node)
    cp_j_per_kg_k: float = 1000.0 # J/(kg·K)
    k_cell_w_per_m_k: float = 1.0 # W/(m·K)

    # Nominal heat generation per node at moderate discharge
    q_gen_nominal_w: float = 8.0  # W


@dataclass
class PackConfig:
    # 3D arrangement: (Nx, Ny, Nz) — x, y in-plane; z vertical/stacking
    # Default: 4×3×2 = 24 cells
    shape: Tuple[int, int, int] = (4, 3, 2)

    # Physical gap between cell surfaces (gap material + air + holder)
    cell_spacing_m: float = 0.002  # m

    # Thermal environment
    ambient_temp_c: float = 25.0
    initial_temp_c: float = 25.0

    # Convective cooling: shared command u ∈ [0, 1]
    # 5–500 W/(m²·K) spans natural convection to aggressive liquid cooling.
    # High h_max reflects a liquid-cooled pack where coolant channels contact
    # the cell/module surfaces directly.
    h_min_w_per_m2_k: float = 5.0
    h_max_w_per_m2_k: float = 500.0

    # Controller targets
    target_temp_c: float = 35.0
    safe_temp_c: float = 45.0     # soft limit — reward penalty starts here
    critical_temp_c: float = 55.0 # hard limit — episode terminates

    # Effective thermal conductance between neighboring cells (W/K).
    # Captures cell casing, contact resistance, gap material, and holder
    # without requiring CFD or detailed geometry.
    # 1.5 W/K is representative of nodes in contact with a thermal interface
    # material (TIM) or thin aluminium heat spreader.
    g_cond_w_per_k: float = 1.5

    # Cell-to-cell heat generation variation
    enable_heat_variation: bool = True
    heat_variation_std: float = 0.05

    # Optional center hotspot for studying thermal non-uniformity
    enable_center_hotspot: bool = False
    hotspot_multiplier: float = 1.25


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def compute_cell_surface_area(cell: CellConfig) -> float:
    """Total outer surface area of one cylindrical cell (m²)."""
    r = cell.diameter_m / 2.0
    side = 2.0 * np.pi * r * cell.length_m
    ends = 2.0 * np.pi * r ** 2
    return float(side + ends)


def compute_cell_heat_capacity(cell: CellConfig) -> float:
    """Lumped thermal capacitance of one cell (J/K)."""
    return float(cell.mass_kg * cell.cp_j_per_kg_k)


def compute_pack_dimensions(cell: CellConfig, pack: PackConfig) -> Tuple[float, float, float]:
    """Outer bounding-box dimensions of the pack (m): (length_x, width_y, height_z)."""
    Nx, Ny, Nz = pack.shape
    s = pack.cell_spacing_m
    d = cell.diameter_m
    L = cell.length_m
    length_x = Nx * d + (Nx - 1) * s
    width_y  = Ny * d + (Ny - 1) * s
    height_z = Nz * L + (Nz - 1) * s
    return length_x, width_y, height_z


def compute_num_cells(pack: PackConfig) -> int:
    Nx, Ny, Nz = pack.shape
    return Nx * Ny * Nz


def build_default_configs():
    """Return (CellConfig, PackConfig, derived_dict) for default 4×3×2 pack."""
    cell = CellConfig()
    pack = PackConfig()

    dims = compute_pack_dimensions(cell, pack)
    derived = {
        "num_cells": compute_num_cells(pack),
        "cell_surface_area_m2": compute_cell_surface_area(cell),
        "cell_heat_capacity_j_per_k": compute_cell_heat_capacity(cell),
        "pack_dimensions_m": dims,
        "pack_volume_m3": float(np.prod(dims)),
    }

    return cell, pack, derived
