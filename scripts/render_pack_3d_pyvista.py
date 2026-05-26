"""
scripts/render_pack_3d_pyvista.py

Procedural 3D visualization of the battery pack using PyVista.

Each cell is rendered as a cylinder whose geometry comes from the same
CellConfig / PackConfig used in the thermal model — not a separate CAD file.

Three visual modes
------------------
render_temperature_field   cells coloured by temperature (hotspot detection)
render_cooling_exposure    cells coloured by direct cooling access fraction
render_both_side_by_side   both modes in a two-panel view

Animation helper
----------------
save_temperature_gif       writes a GIF from a list of T3d snapshots

Command-line use
----------------
# Interactive (requires a display)
python -m scripts.render_pack_3d_pyvista --mode temperature
python -m scripts.render_pack_3d_pyvista --mode cooling
python -m scripts.render_pack_3d_pyvista --mode both

# Headless / off-screen (Colab, SSH, CI)
python -m scripts.render_pack_3d_pyvista --mode temperature --off-screen \
    --screenshot outputs/pack_3d_temperature.png

Requirements
------------
pip install pyvista
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from configs.pack_config import CellConfig, PackConfig, build_default_configs


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _cell_pitch(cell: CellConfig, pack: PackConfig) -> Tuple[float, float, float]:
    """Center-to-center pitch in x/y (diameter-based) and z (length-based)."""
    return (
        cell.diameter_m + pack.cell_spacing_m,
        cell.diameter_m + pack.cell_spacing_m,
        cell.length_m   + pack.cell_spacing_m,
    )


def _compute_exposed_fraction(pack: PackConfig) -> np.ndarray:
    """
    Fraction of each cell's 6 faces exposed directly to the coolant.
    Interior cells → 0.0   (no direct cooling, only via neighbour conduction)
    Corner cells   → 0.5   (3 of 6 faces exposed)
    """
    Nx, Ny, Nz = pack.shape
    faces = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    faces[0, :, :] += 1;  faces[-1, :, :] += 1
    faces[:, 0, :] += 1;  faces[:, -1, :] += 1
    faces[:, :, 0] += 1;  faces[:, :, -1] += 1
    return faces / 6.0


# ---------------------------------------------------------------------------
# Internal mesh builder
# ---------------------------------------------------------------------------

def _build_cylinder_meshes(
    scalars_3d: np.ndarray,
    cell: CellConfig,
    pack: PackConfig,
    scalar_name: str,
) -> list:
    """
    Create one pv.Cylinder per cell, pre-loaded with scalar data.
    Returns [(mesh, i, j, k), ...] in grid order.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista is required for 3D visualization.\n"
            "Install with:  pip install pyvista"
        ) from exc

    Nx, Ny, Nz = pack.shape
    px, py, pz = _cell_pitch(cell, pack)
    r = cell.diameter_m / 2.0
    L = cell.length_m

    result = []
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                cx = i * px
                cy = j * py
                cz = k * pz

                mesh = pv.Cylinder(
                    center=(cx, cy, cz),
                    direction=(0, 0, 1),
                    radius=r,
                    height=L,
                    resolution=32,
                )
                val = float(scalars_3d[i, j, k])
                mesh[scalar_name] = np.full(mesh.n_points, val, dtype=np.float32)
                result.append((mesh, i, j, k))

    return result


def _add_meshes_to_plotter(
    plotter,
    cell_meshes: list,
    scalar_name: str,
    cmap: str,
    vmin: float,
    vmax: float,
    show_edges: bool,
) -> None:
    """Add all cylinder meshes to an existing plotter instance."""
    for idx, (mesh, i, j, k) in enumerate(cell_meshes):
        plotter.add_mesh(
            mesh,
            scalars=scalar_name,
            clim=[vmin, vmax],
            cmap=cmap,
            show_scalar_bar=False,
            show_edges=show_edges,
            edge_color="gray",
            opacity=1.0,
        )


# ---------------------------------------------------------------------------
# Mode 1 — Temperature field
# ---------------------------------------------------------------------------

def render_temperature_field(
    T: np.ndarray,
    cell: CellConfig,
    pack: PackConfig,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdYlBu_r",
    show_edges: bool = False,
    off_screen: bool = False,
    title: str = "Battery Pack — Temperature Field",
    screenshot_path: Optional[Path] = None,
) -> None:
    """
    3D pack with each cell coloured by temperature.

    Blue  = cool (at ambient temperature)
    Red   = hot  (at or above safe limit)

    vmin defaults to pack.ambient_temp_c
    vmax defaults to pack.safe_temp_c
    """
    import pyvista as pv

    vmin = pack.ambient_temp_c if vmin is None else vmin
    vmax = pack.safe_temp_c    if vmax is None else vmax

    cell_meshes = _build_cylinder_meshes(T, cell, pack, "Temperature (°C)")

    plotter = pv.Plotter(off_screen=off_screen, window_size=[1200, 800])
    plotter.set_background("white")
    plotter.enable_anti_aliasing()

    _add_meshes_to_plotter(plotter, cell_meshes, "Temperature (°C)",
                           cmap=cmap, vmin=vmin, vmax=vmax, show_edges=show_edges)

    plotter.add_scalar_bar(
        "Temperature (°C)",
        vertical=True,
        fmt="%.1f",
        n_labels=6,
        color="black",
    )
    plotter.add_title(title, font_size=11, color="black")
    plotter.add_axes(color="black")
    plotter.camera_position = "iso"
    plotter.add_text(
        f"T_max {T.max():.1f}°C    T_avg {T.mean():.1f}°C    "
        f"ΔT {T.max()-T.min():.2f}°C",
        position="lower_left",
        font_size=9,
        color="black",
    )

    if screenshot_path is not None:
        plotter.screenshot(str(screenshot_path))
        print(f"Saved: {screenshot_path}")

    if not off_screen:
        plotter.show()

    plotter.close()


# ---------------------------------------------------------------------------
# Mode 2 — Cooling exposure
# ---------------------------------------------------------------------------

def render_cooling_exposure(
    cell: CellConfig,
    pack: PackConfig,
    cmap: str = "YlOrRd",
    show_edges: bool = True,
    off_screen: bool = False,
    title: str = "Battery Pack — Direct Cooling Exposure",
    screenshot_path: Optional[Path] = None,
) -> None:
    """
    3D pack coloured by exposed cooling area fraction.

    Light/yellow = interior cell (no direct convection — heats up first)
    Dark/red     = boundary/corner cell (directly cooled by convection)

    This is why interior cells become hotspots: they are insulated from direct
    cooling and depend entirely on conduction through their neighbours.
    """
    import pyvista as pv

    exposed = _compute_exposed_fraction(pack)
    cell_meshes = _build_cylinder_meshes(
        exposed, cell, pack, "Cooling exposure fraction"
    )

    plotter = pv.Plotter(off_screen=off_screen, window_size=[1200, 800])
    plotter.set_background("white")
    plotter.enable_anti_aliasing()

    _add_meshes_to_plotter(
        plotter, cell_meshes, "Cooling exposure fraction",
        cmap=cmap, vmin=0.0, vmax=float(exposed.max()),
        show_edges=show_edges,
    )

    plotter.add_scalar_bar(
        "Cooling exposure  (0 = interior, 0.5 = corner)",
        vertical=True,
        fmt="%.2f",
        n_labels=5,
        color="black",
    )
    plotter.add_title(title, font_size=11, color="black")
    plotter.add_axes(color="black")
    plotter.camera_position = "iso"

    n_interior = int((exposed == 0.0).sum())
    n_total    = int(np.prod(pack.shape))
    plotter.add_text(
        f"{n_interior}/{n_total} interior cells — zero direct cooling",
        position="lower_left",
        font_size=9,
        color="black",
    )

    if screenshot_path is not None:
        plotter.screenshot(str(screenshot_path))
        print(f"Saved: {screenshot_path}")

    if not off_screen:
        plotter.show()

    plotter.close()


# ---------------------------------------------------------------------------
# Mode 3 — Side-by-side comparison
# ---------------------------------------------------------------------------

def render_both_side_by_side(
    T: np.ndarray,
    cell: CellConfig,
    pack: PackConfig,
    off_screen: bool = False,
    screenshot_path: Optional[Path] = None,
) -> None:
    """
    Left panel:  temperature field
    Right panel: cooling exposure
    """
    import pyvista as pv

    exposed = _compute_exposed_fraction(pack)

    plotter = pv.Plotter(shape=(1, 2), off_screen=off_screen, window_size=[1920, 800])
    plotter.set_background("white")

    # --- left: temperature ---
    plotter.subplot(0, 0)
    temp_meshes = _build_cylinder_meshes(T, cell, pack, "Temperature (°C)")
    _add_meshes_to_plotter(
        plotter, temp_meshes, "Temperature (°C)",
        cmap="RdYlBu_r",
        vmin=pack.ambient_temp_c,
        vmax=pack.safe_temp_c,
        show_edges=False,
    )
    plotter.add_scalar_bar("Temperature (°C)", fmt="%.1f", n_labels=5, color="black")
    plotter.add_title("Temperature Field", font_size=11, color="black")
    plotter.add_axes(color="black")
    plotter.camera_position = "iso"
    plotter.add_text(
        f"T_max {T.max():.1f}°C  T_avg {T.mean():.1f}°C",
        position="lower_left", font_size=9, color="black",
    )

    # --- right: cooling exposure ---
    plotter.subplot(0, 1)
    exp_meshes = _build_cylinder_meshes(exposed, cell, pack, "Cooling exposure")
    _add_meshes_to_plotter(
        plotter, exp_meshes, "Cooling exposure",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=float(exposed.max()),
        show_edges=True,
    )
    plotter.add_scalar_bar(
        "Cooling exposure (0=interior, 0.5=corner)",
        fmt="%.2f", n_labels=5, color="black",
    )
    plotter.add_title("Cooling Exposure", font_size=11, color="black")
    plotter.add_axes(color="black")
    plotter.camera_position = "iso"
    n_int = int((exposed == 0.0).sum())
    plotter.add_text(
        f"{n_int}/{int(np.prod(pack.shape))} interior cells",
        position="lower_left", font_size=9, color="black",
    )

    if screenshot_path is not None:
        plotter.screenshot(str(screenshot_path))
        print(f"Saved: {screenshot_path}")

    if not off_screen:
        plotter.show()

    plotter.close()


# ---------------------------------------------------------------------------
# Animation helper
# ---------------------------------------------------------------------------

def save_temperature_gif(
    T_frames: List[np.ndarray],
    frame_times: List[float],
    cell: CellConfig,
    pack: PackConfig,
    output_path: Path,
    fps: int = 10,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdYlBu_r",
    window_size: Tuple[int, int] = (1200, 800),
) -> None:
    """
    Write a GIF animation of the temperature field using PyVista.

    Builds the cylinder meshes once, updates their scalars in-place for each
    frame, then writes via PyVista's open_gif / write_frame API — efficient
    for large frame counts.

    Parameters
    ----------
    T_frames    : list of (Nx, Ny, Nz) temperature arrays
    frame_times : list of simulation times (seconds) — same length as T_frames
    output_path : GIF file path
    fps         : frames per second
    """
    import pyvista as pv

    vmin = pack.ambient_temp_c if vmin is None else vmin
    vmax = pack.safe_temp_c    if vmax is None else vmax

    Nx, Ny, Nz = pack.shape
    px, py, pz = _cell_pitch(cell, pack)
    r = cell.diameter_m / 2.0
    L = cell.length_m

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background("white")
    plotter.enable_anti_aliasing()

    # Build all meshes once from the first frame
    mesh_grid: dict = {}
    scalar_name = "Temperature (°C)"

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                cx, cy, cz = i * px, j * py, k * pz
                mesh = pv.Cylinder(
                    center=(cx, cy, cz),
                    direction=(0, 0, 1),
                    radius=r,
                    height=L,
                    resolution=24,
                )
                temp = float(T_frames[0][i, j, k])
                mesh[scalar_name] = np.full(mesh.n_points, temp, dtype=np.float32)

                is_first = (i == 0 and j == 0 and k == 0)
                plotter.add_mesh(
                    mesh,
                    scalars=scalar_name,
                    clim=[vmin, vmax],
                    cmap=cmap,
                    show_scalar_bar=is_first,
                    show_edges=False,
                    opacity=1.0,
                )
                mesh_grid[(i, j, k)] = mesh

    plotter.add_axes(color="black")
    plotter.camera_position = "iso"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.open_gif(str(output_path))

    for T3d, t in zip(T_frames, frame_times):
        for (i, j, k), mesh in mesh_grid.items():
            temp = float(T3d[i, j, k])
            mesh[scalar_name] = np.full(mesh.n_points, temp, dtype=np.float32)
        plotter.write_frame()

    plotter.close()
    print(f"Saved animation ({len(T_frames)} frames): {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3D PyVista battery pack visualization"
    )
    p.add_argument(
        "--mode", default="temperature",
        choices=["temperature", "cooling", "both"],
        help="Visualization mode (default: temperature)",
    )
    p.add_argument(
        "--off-screen", action="store_true",
        help="Render off-screen (no window) — required on headless systems",
    )
    p.add_argument(
        "--screenshot", type=str, default=None,
        help="Save a PNG screenshot to this path",
    )
    p.add_argument(
        "--vmin", type=float, default=None,
        help="Colormap lower bound (default: ambient_temp_c from PackConfig)",
    )
    p.add_argument(
        "--vmax", type=float, default=None,
        help="Colormap upper bound (default: safe_temp_c from PackConfig)",
    )
    p.add_argument(
        "--hotspot", action="store_true",
        help="Enable center hotspot in the default temperature field",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cell, pack, derived = build_default_configs()
    pack.enable_center_hotspot = args.hotspot
    pack.hotspot_multiplier = 1.5

    # Default temperature field — simulate a few hundred steps to show variation
    from models.thermal_model_3d import BatteryPackThermal3D
    model = BatteryPackThermal3D(cell, pack)
    for _ in range(300):
        model.step(u=0.3, dt=1.0)
    T = model.T.copy()

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    screenshot = Path(args.screenshot) if args.screenshot else None

    if args.mode == "temperature":
        render_temperature_field(
            T=T, cell=cell, pack=pack,
            vmin=args.vmin, vmax=args.vmax,
            off_screen=args.off_screen,
            screenshot_path=screenshot or (out_dir / "pack_3d_temperature.png" if args.off_screen else None),
        )

    elif args.mode == "cooling":
        render_cooling_exposure(
            cell=cell, pack=pack,
            off_screen=args.off_screen,
            screenshot_path=screenshot or (out_dir / "pack_3d_cooling_exposure.png" if args.off_screen else None),
        )

    elif args.mode == "both":
        render_both_side_by_side(
            T=T, cell=cell, pack=pack,
            off_screen=args.off_screen,
            screenshot_path=screenshot or (out_dir / "pack_3d_both.png" if args.off_screen else None),
        )


if __name__ == "__main__":
    main()
