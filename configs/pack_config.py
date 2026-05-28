"""
configs/pack_config.py

Geometry, electrical, thermal, and cooling configuration for a cell-resolved
3D battery pack thermal network.

Reference cell:
    Samsung INR18650-25R cylindrical lithium-ion cell

Reference pack:
    16-cell 4S4P pack arranged as a 4 x 4 x 1 cylindrical-cell array.

Important:
    This is not CFD.
    This is a lumped-parameter thermal network with cell-resolved temperature
    states, geometry-based exposed cooling area, neighbor conduction, and
    pack-level electrical metadata.

    Each thermal state can represent either:
        1. one physical 18650 cell, or
        2. an effective group of cells using node_scale_factor.

    For a defensible validation model, start with node_scale_factor = 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Single-cell specification
# ---------------------------------------------------------------------------

@dataclass
class CellConfig:
    """
    Samsung INR18650-25R single-cell properties.

    Datasheet-level values are used for geometry and electrical sizing.
    Some thermal values are engineering assumptions because full anisotropic
    thermal properties are usually not provided in commercial datasheets.
    """

    # -----------------------------------------------------------------------
    # Geometry: Samsung INR18650-25R
    # -----------------------------------------------------------------------
    diameter_m: float = 0.01833       # m, 18.33 mm
    length_m: float = 0.06485         # m, 64.85 mm
    mass_kg: float = 0.0438           # kg, typical ~43.8 g; max about 45 g

    # -----------------------------------------------------------------------
    # Electrical specs
    # -----------------------------------------------------------------------
    nominal_voltage_v: float = 3.6    # V
    max_voltage_v: float = 4.2        # V
    cutoff_voltage_v: float = 2.5     # V
    capacity_ah: float = 2.5          # Ah
    nominal_energy_wh: float = 9.0    # Wh, approx 3.6 V * 2.5 Ah

    # Internal resistance
    # AC and DC resistance are both kept because heat generation can be
    # estimated differently depending on model fidelity.
    ac_ir_ohm: float = 0.0132         # Ohm, typical AC IR at 1 kHz
    dc_ir_ohm: float = 0.02215        # Ohm, typical DC IR

    # Current limits
    max_continuous_discharge_a: float = 20.0
    standard_charge_current_a: float = 1.25
    max_charge_current_a: float = 4.0

    # -----------------------------------------------------------------------
    # Thermal properties
    # -----------------------------------------------------------------------
    # cp for Li-ion cells is commonly approximated around 900–1100 J/kg-K.
    # Use 1000 as a clean baseline unless you have calorimetry data.
    cp_j_per_kg_k: float = 1000.0

    # Effective radial/through-node conductivity.
    # This is not exact anisotropic jelly-roll conductivity.
    # It is a lumped network approximation.
    k_cell_w_per_m_k: float = 1.0

    # Scaling factor for one thermal node.
    # node_scale_factor = 1 means each thermal state is one physical cell.
    # Increase only if you intentionally want each node to represent a group.
    node_scale_factor: int = 1

    # Nominal heat generation per physical cell.
    # For I^2R heating:
    # q = I_cell^2 * R_internal
    # At 5 A and R = 0.02215 Ohm, q ≈ 0.55 W per cell.
    q_gen_nominal_w: float = 0.55


# ---------------------------------------------------------------------------
# Pack-level configuration
# ---------------------------------------------------------------------------

@dataclass
class PackConfig:
    """
    16-cell 4S4P pack.

    Electrical:
        4 cells in series
        4 cells in parallel per series group
        total cells = 4 * 4 = 16

    Geometry:
        4 x 4 x 1 cylindrical cells
        Cylinders are assumed vertical along z.
        x-y plane contains the pack footprint.
    """

    # -----------------------------------------------------------------------
    # Electrical configuration
    # -----------------------------------------------------------------------
    series_count: int = 4
    parallel_count: int = 4

    # -----------------------------------------------------------------------
    # 3D geometry arrangement: (Nx, Ny, Nz)
    # -----------------------------------------------------------------------
    # 4 x 4 x 1 = 16 physical cells.
    shape: Tuple[int, int, int] = (4, 4, 1)

    # Surface-to-surface spacing between neighboring cylindrical cells.
    # 3 mm is a clean baseline for thermal studies because it is not unrealistically
    # tight, but still allows strong cell-to-cell interaction.
    cell_spacing_m: float = 0.003

    # -----------------------------------------------------------------------
    # Thermal environment
    # -----------------------------------------------------------------------
    ambient_temp_c: float = 25.0
    initial_temp_c: float = 25.0

    # -----------------------------------------------------------------------
    # Convective cooling model
    # -----------------------------------------------------------------------
    # u in [0, 1]
    # h(u) = h_min + u * (h_max - h_min)
    #
    # For air cooling, do NOT use 500 W/m^2-K unless you are pretending to have
    # liquid cooling. For forced air, 10–100 W/m^2-K is more defensible.
    h_min_w_per_m2_k: float = 5.0
    h_max_w_per_m2_k: float = 80.0

    # Optional liquid-cooling equivalent upper bound.
    # Use this only for a liquid-cooled study.
    h_liquid_max_w_per_m2_k: float = 500.0

    # -----------------------------------------------------------------------
    # Controller targets and safe operating band
    # -----------------------------------------------------------------------
    # target_temp_c is the obs-normalisation reference and PI set-point.
    # The reward uses target_low / target_high as a band — no penalty inside
    # [target_low, target_high], asymmetric penalty outside it.
    target_temp_c: float = 35.0      # PI reference / obs normalisation anchor
    target_low_temp_c: float = 30.0  # lower bound of ideal operating band
    target_high_temp_c: float = 38.0 # upper bound of ideal operating band
    safe_temp_c: float = 45.0        # hard safety limit
    critical_temp_c: float = 55.0    # BMS shutdown threshold

    # -----------------------------------------------------------------------
    # Neighbor conduction
    # -----------------------------------------------------------------------
    # Effective thermal conductance between neighboring cells.
    # This captures cell casing, holder, air gap, thermal pad/contact, and
    # simplified geometry. It is not a first-principles CFD value.
    #
    # For air/holder spacing only: 0.05–0.3 W/K may be more realistic.
    # For thermal pads/spreaders: 0.5–2.0 W/K can be used as a stronger coupling.
    g_cond_w_per_k: float = 0.25

    # -----------------------------------------------------------------------
    # Heat generation variation
    # -----------------------------------------------------------------------
    enable_heat_variation: bool = True
    heat_variation_std: float = 0.05

    # Optional center hotspot for nonuniform thermal loading.
    # In a 4x4 pack, the four center-adjacent cells are the harshest thermal region.
    enable_center_hotspot: bool = False
    hotspot_multiplier: float = 1.25

    # -----------------------------------------------------------------------
    # Simulation limits
    # -----------------------------------------------------------------------
    episode_time_s: float = 1800.0
    dt_s: float = 1.0

    # -----------------------------------------------------------------------
    # Multi-zone cooling
    # -----------------------------------------------------------------------
    # num_cooling_zones=4 gives a 2×2 spatial split (x-left/right × y-front/rear).
    # Use 1 to revert to global single-zone control.
    enable_multizone_cooling: bool = True
    num_cooling_zones: int = 4

    # -----------------------------------------------------------------------
    # Cooling actuator delay
    # -----------------------------------------------------------------------
    # Models the lag between a controller command and the actual coolant response
    # (valve travel, pump ramp-up, thermal inertia of the manifold).
    # 10 s is a realistic first estimate for forced-air cooling.
    enable_cooling_delay: bool = True
    cooling_delay_s: float = 10.0

    # -----------------------------------------------------------------------
    # Actuator rate limit
    # -----------------------------------------------------------------------
    # Maximum change in normalised cooling command per second.
    # 0.05 / s → full range (0→1) traversed in 20 s, preventing instantaneous
    # bang-bang switching that would damage actuators in hardware.
    enable_cooling_rate_limit: bool = True
    max_cooling_rate_per_s: float = 0.05

    # -----------------------------------------------------------------------
    # Randomized hotspot
    # -----------------------------------------------------------------------
    # Hotspot cells are selected from the full cell count at episode start.
    # Optionally constrained to a randomly selected cooling zone so the
    # multi-zone controller has a clear spatial target.
    #
    # hotspot_seed: if set, a dedicated RNG is used so hotspot location is
    # reproducible regardless of training episode order. None → shares the
    # environment's main RNG (fully random during training).
    enable_random_hotspot: bool = True
    num_hotspot_cells: int = 3
    hotspot_multiplier_min: float = 2.0   # was 1.5 — 2× min creates clear spatial gradient
    hotspot_multiplier_max: float = 4.0   # was 2.5 — 4× max stresses zone controller
    hotspot_persistent: bool = True          # keep hotspot fixed for whole episode
    hotspot_change_interval_s: float = 600.0  # only used when persistent=False
    hotspot_seed: Optional[int] = None        # None → use env RNG (random training)

    # -----------------------------------------------------------------------
    # Global heat-generation scale factor
    # -----------------------------------------------------------------------
    # Multiplies every cell's q_gen before the hotspot overlay.
    # 1.0 = nominal physics.  2.5 pushes no-cooling above 45°C in ~13 min.
    # Applied in env.step() after the heat profile, before hotspot.
    q_gen_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def compute_num_cells(pack: PackConfig) -> int:
    """Total number of physical cells in the pack."""
    nx, ny, nz = pack.shape
    return int(nx * ny * nz)


def validate_pack_config(cell: CellConfig, pack: PackConfig) -> None:
    """
    Check that geometry and electrical configuration agree.

    For the baseline:
        shape = (4, 4, 1) gives 16 cells.
        series_count * parallel_count = 4 * 4 = 16 cells.
    """
    geometric_cell_count = compute_num_cells(pack)
    electrical_cell_count = pack.series_count * pack.parallel_count

    if geometric_cell_count != electrical_cell_count:
        raise ValueError(
            f"Geometry/electrical mismatch: shape gives {geometric_cell_count} cells, "
            f"but {pack.series_count}S{pack.parallel_count}P gives "
            f"{electrical_cell_count} cells."
        )

    if cell.node_scale_factor < 1:
        raise ValueError("node_scale_factor must be >= 1.")


def compute_cell_surface_area(cell: CellConfig) -> float:
    """
    Total outer surface area of one cylindrical cell.

    Includes:
        cylindrical side area
        top circular face
        bottom circular face
    """
    r = cell.diameter_m / 2.0
    side_area = 2.0 * np.pi * r * cell.length_m
    end_area = 2.0 * np.pi * r**2
    return float(side_area + end_area)


def compute_cell_side_area(cell: CellConfig) -> float:
    """Cylindrical side area only."""
    r = cell.diameter_m / 2.0
    return float(2.0 * np.pi * r * cell.length_m)


def compute_cell_end_area(cell: CellConfig) -> float:
    """Combined top and bottom circular end area."""
    r = cell.diameter_m / 2.0
    return float(2.0 * np.pi * r**2)


def compute_cell_heat_capacity(cell: CellConfig) -> float:
    """
    Lumped thermal capacitance of one thermal node.

    If node_scale_factor = 1:
        C = one physical cell mass * cp

    If node_scale_factor > 1:
        C = grouped effective thermal mass
    """
    return float(cell.node_scale_factor * cell.mass_kg * cell.cp_j_per_kg_k)


def compute_pack_dimensions(
    cell: CellConfig,
    pack: PackConfig,
) -> Tuple[float, float, float]:
    """
    Outer bounding-box dimensions of the pack.

    Cylinders are assumed vertical:
        x dimension uses cell diameter
        y dimension uses cell diameter
        z dimension uses cell length
    """
    nx, ny, nz = pack.shape
    s = pack.cell_spacing_m
    d = cell.diameter_m
    l = cell.length_m

    length_x = nx * d + (nx - 1) * s
    width_y = ny * d + (ny - 1) * s
    height_z = nz * l + (nz - 1) * s

    return float(length_x), float(width_y), float(height_z)


def compute_pack_electrical_specs(
    cell: CellConfig,
    pack: PackConfig,
) -> Dict[str, float]:
    """
    Compute pack-level electrical quantities for S-P configuration.

    For 4S4P using Samsung INR18650-25R:
        nominal voltage = 4 * 3.6 = 14.4 V
        capacity = 4 * 2.5 = 10 Ah
        energy = 14.4 * 10 = 144 Wh
    """
    nominal_voltage = pack.series_count * cell.nominal_voltage_v
    max_voltage = pack.series_count * cell.max_voltage_v
    cutoff_voltage = pack.series_count * cell.cutoff_voltage_v

    capacity_ah = pack.parallel_count * cell.capacity_ah
    nominal_energy_wh = nominal_voltage * capacity_ah

    max_continuous_current_a = (
        pack.parallel_count * cell.max_continuous_discharge_a
    )

    return {
        "series_count": pack.series_count,
        "parallel_count": pack.parallel_count,
        "nominal_voltage_v": float(nominal_voltage),
        "max_voltage_v": float(max_voltage),
        "cutoff_voltage_v": float(cutoff_voltage),
        "capacity_ah": float(capacity_ah),
        "nominal_energy_wh": float(nominal_energy_wh),
        "max_continuous_current_a": float(max_continuous_current_a),
    }


def compute_heat_generation_from_current(
    cell: CellConfig,
    pack_current_a: float,
    use_dc_ir: bool = True,
) -> float:
    """
    Estimate heat generation per physical cell using I^2R.

    In a parallel group:
        current per cell = pack_current / parallel_count

    This function returns heat per physical cell, not total pack heat.

    Note:
        This function needs parallel_count, so for pack-level usage prefer
        compute_pack_heat_generation().
    """
    resistance = cell.dc_ir_ohm if use_dc_ir else cell.ac_ir_ohm
    return float((pack_current_a**2) * resistance)


def compute_pack_heat_generation(
    cell: CellConfig,
    pack: PackConfig,
    pack_current_a: float,
    use_dc_ir: bool = True,
) -> Dict[str, float]:
    """
    Estimate I^2R heat generation.

    pack_current_a is total current delivered by the full pack.
    Current per cell is divided by the number of parallel cells.

    Example:
        20 A pack current in 4P group:
            I_cell = 20 / 4 = 5 A
            q_cell = 5^2 * 0.02215 ≈ 0.554 W
            q_pack = 16 * 0.554 ≈ 8.86 W
    """
    resistance = cell.dc_ir_ohm if use_dc_ir else cell.ac_ir_ohm

    i_cell = pack_current_a / pack.parallel_count
    q_cell = i_cell**2 * resistance
    q_pack = q_cell * compute_num_cells(pack)

    return {
        "pack_current_a": float(pack_current_a),
        "cell_current_a": float(i_cell),
        "q_gen_per_cell_w": float(q_cell),
        "q_gen_total_pack_w": float(q_pack),
    }


def compute_cell_positions(
    cell: CellConfig,
    pack: PackConfig,
) -> np.ndarray:
    """
    Return cell center positions as an array of shape (num_cells, 3).

    Cylindrical cells are vertical along z.
    Cell centers are placed on a regular x-y-z grid.
    """
    nx, ny, nz = pack.shape
    pitch_xy = cell.diameter_m + pack.cell_spacing_m
    pitch_z = cell.length_m + pack.cell_spacing_m

    positions = []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = i * pitch_xy
                y = j * pitch_xy
                z = k * pitch_z
                positions.append([x, y, z])

    return np.array(positions, dtype=float)


def build_default_configs():
    """
    Return:
        cell: CellConfig
        pack: PackConfig
        derived: dictionary of useful derived values
    """
    cell = CellConfig()
    pack = PackConfig()

    validate_pack_config(cell, pack)

    dims = compute_pack_dimensions(cell, pack)
    electrical = compute_pack_electrical_specs(cell, pack)
    positions = compute_cell_positions(cell, pack)

    derived = {
        "num_cells": compute_num_cells(pack),

        "cell_surface_area_m2": compute_cell_surface_area(cell),
        "cell_side_area_m2": compute_cell_side_area(cell),
        "cell_end_area_m2": compute_cell_end_area(cell),
        "cell_heat_capacity_j_per_k": compute_cell_heat_capacity(cell),

        "pack_dimensions_m": dims,
        "pack_dimensions_mm": tuple(1000.0 * x for x in dims),
        "pack_volume_m3": float(np.prod(dims)),

        "electrical": electrical,

        "cell_positions_m": positions,

        # Useful baseline heat case:
        # 20 A total pack current, meaning 5 A per cell in a 4P group.
        "baseline_heat_20a_pack": compute_pack_heat_generation(
            cell=cell,
            pack=pack,
            pack_current_a=20.0,
            use_dc_ir=True,
        ),
    }

    return cell, pack, derived


if __name__ == "__main__":
    cell_cfg, pack_cfg, derived_cfg = build_default_configs()

    print("Cell config:")
    print(cell_cfg)

    print("\nPack config:")
    print(pack_cfg)

    print("\nDerived values:")
    for key, value in derived_cfg.items():
        if key == "cell_positions_m":
            print(f"{key}: array shape {value.shape}")
        else:
            print(f"{key}: {value}")