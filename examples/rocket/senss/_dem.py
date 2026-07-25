"""SENNS lunar DEM terrain patch for the rocket viser scenes.

The rocket examples in this directory land on a heightfield sampled from
``senns_dem.png`` (3938x3938, 16-bit) rather than a flat ground plane.  This
module owns everything about that patch: loading and normalizing the PNG,
placing it in the world, shading it, and exposing the two GUI folders that let
you slide the patch under the trajectory until the landing site lines up.

Two frozen records carry the knobs.  :class:`DemPlacement` says *where the
patch is* (center, stretch, yaw, mirror, sampling resolution) and
:class:`DemShading` says *how it is lit*; together they keep
:func:`add_dem_terrain` down to three arguments instead of fifteen.

Everything here is lazy.  The DEM array and the triangle index array are built
on first use and cached, so importing an example that never opens a viser
window costs nothing — a 3938-grid patch is 62 MB of heights and 372 MB of
int32 triangle indices.

Nothing in this module affects the optimization; it is display geometry only.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import trimesh
import viser

DEM_PATH = os.path.join(os.path.dirname(__file__), "senns_dem.png")
NATIVE_GRID = 3938  # native resolution of senns_dem.png; larger grids oversample

#: Base albedo of the terrain before shading (matte lunar grey).
GREY_BASE = np.array([148, 150, 152], dtype=np.float32) / 255.0


@dataclass(frozen=True)
class DemPlacement:
    """Where the DEM patch sits in the world, in metres.

    ``origin_m`` is the patch center: (x, y) horizontal position and the
    elevation the DEM's *center pixel* is pinned to, so moving the patch never
    changes which feature is under the landing site.  ``scale_xyz`` stretches
    the patch about that center, ``yaw_deg`` spins it, and ``mirror_xy`` flips
    the sampling direction before the yaw (handy for matching a photo).
    """

    origin_m: tuple[float, float, float]
    scale_xyz: tuple[float, float, float] = (0.5, 0.5, 0.65)
    yaw_deg: float = 180.0
    mirror_xy: tuple[bool, bool] = (True, False)
    half_extent_m: float = 400.0
    base_relief_m: float = 150.0
    grid: int = 2048


@dataclass(frozen=True)
class DemShading:
    """Directional light baked into the terrain's vertex colours.

    Baking beats scene lighting here: the patch is one enormous mesh and the
    rest of the scene (cones, trails, plumes) wants flat ambient light, so the
    terrain carries its own shading and everything else is left alone.
    """

    azimuth_deg: float = 128.0
    elevation_deg: float = 10.5
    strength: float = 2.25
    ambient: float = 0.0
    enabled: bool = True


@lru_cache(maxsize=4)
def load_senns_dem(grid: int = 2048) -> np.ndarray:
    """Normalized ``(grid, grid)`` DEM heights in ``[0, 1]``.

    Normalization uses the *full-resolution* min/max so the relief is
    independent of ``grid``; only the sampling changes.  Cached per grid —
    callers may ask repeatedly without paying for the decode again.
    """
    from PIL import Image

    img = Image.open(DEM_PATH)
    raw = np.array(img, dtype=np.uint16)
    lo, hi = float(raw.min()), float(raw.max())
    arr = np.array(img.resize((grid, grid), Image.BILINEAR), dtype=np.float32)
    return (arr - lo) / max(hi - lo, 1.0)


def dem_center_norm(grid: int = 2048) -> float:
    """Normalized height of the DEM's center pixel — the patch's pin point."""
    c = (grid - 1) // 2
    return float(load_senns_dem(grid)[c, c])


@lru_cache(maxsize=4)
def terrain_faces(grid: int) -> np.ndarray:
    """Triangle indices for a ``grid x grid`` vertex lattice, two per cell.

    This is the expensive part of the patch (``2*(grid-1)**2`` int32 triples),
    so it is cached and never built at import time.
    """
    r = np.arange(grid - 1, dtype=np.int32)
    i = (r[:, None] * grid + r[None, :]).ravel()
    return np.concatenate(
        [
            np.stack([i, i + 1, i + grid], axis=-1),
            np.stack([i + 1, i + grid + 1, i + grid], axis=-1),
        ],
        axis=0,
    ).astype(np.int32)


def terrain_vertices(placement: DemPlacement, *, scene_scale: float) -> np.ndarray:
    """``(grid**2, 3)`` vertex positions in viser units (``scene_scale`` m per unit)."""
    ox_m, oy_m, oz_m = placement.origin_m
    sx, sy, sz = placement.scale_xyz
    mx = -1.0 if placement.mirror_xy[0] else 1.0
    my = -1.0 if placement.mirror_xy[1] else 1.0
    grid = placement.grid

    half_x_m = placement.half_extent_m * float(sx)
    half_y_m = placement.half_extent_m * float(sy)
    x_loc = mx * np.linspace(-half_x_m, half_x_m, grid, dtype=np.float32)
    y_loc = my * np.linspace(-half_y_m, half_y_m, grid, dtype=np.float32)
    XX_loc, YY_loc = np.meshgrid(x_loc, y_loc, indexing="xy")

    psi = np.radians(float(placement.yaw_deg))
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    XX_m = cos_p * XX_loc - sin_p * YY_loc + float(ox_m)
    YY_m = sin_p * XX_loc + cos_p * YY_loc + float(oy_m)

    dem = load_senns_dem(grid)
    relief_m = (dem - dem_center_norm(grid)) * placement.base_relief_m * float(sz)
    ZZ_m = float(oz_m) + relief_m

    return np.stack(
        [
            (XX_m / scene_scale).astype(np.float32).ravel(),
            (YY_m / scene_scale).astype(np.float32).ravel(),
            (ZZ_m / scene_scale).astype(np.float32).ravel(),
        ],
        axis=-1,
    )


def terrain_vertex_normals(placement: DemPlacement) -> np.ndarray:
    """``(grid**2, 3)`` unit normals from the DEM gradient.

    Independent of ``origin_m`` and of ``scene_scale`` — a uniform rescale of
    the patch leaves its surface normals unchanged.
    """
    sx, sy, sz = placement.scale_xyz
    mx = -1.0 if placement.mirror_xy[0] else 1.0
    my = -1.0 if placement.mirror_xy[1] else 1.0
    grid = placement.grid

    cell_x = 2.0 * placement.half_extent_m * float(sx) / (grid - 1)
    cell_y = 2.0 * placement.half_extent_m * float(sy) / (grid - 1)
    ZZ = load_senns_dem(grid) * placement.base_relief_m * float(sz)
    nx = -mx * np.gradient(ZZ, cell_x, axis=1).astype(np.float32).ravel()
    ny = -my * np.gradient(ZZ, cell_y, axis=0).astype(np.float32).ravel()
    nz = np.ones(grid * grid, dtype=np.float32)

    psi = np.radians(float(placement.yaw_deg))
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    normals = np.stack([cos_p * nx - sin_p * ny, sin_p * nx + cos_p * ny, nz], axis=-1)
    return normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)


def bake_shading_colors(normals: np.ndarray, shading: DemShading) -> np.ndarray:
    """``(n, 4)`` uint8 RGBA vertex colours for the given normals and light."""
    if shading.enabled:
        az, el = np.radians(shading.azimuth_deg), np.radians(shading.elevation_deg)
        L = np.array(
            [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)], dtype=np.float32
        )
        diffuse = np.maximum(0.0, normals @ L)
    else:
        diffuse = 0.0
    intensity = np.clip(shading.ambient + shading.strength * diffuse, 0.0, 1.0)
    rgb = (GREY_BASE[None, :] * intensity[:, None]).clip(0.0, 1.0)
    alpha = np.ones((len(rgb), 1), dtype=np.float32)
    return (np.hstack([rgb, alpha]) * 255).astype(np.uint8)


def dem_trimesh(
    placement: DemPlacement,
    shading: DemShading,
    *,
    scene_scale: float,
    normals: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Shaded terrain mesh ready for ``server.scene.add_mesh_trimesh``.

    Pass ``normals`` to reuse a previously computed set when only the light
    moved — recomputing the DEM gradient dominates the rebuild cost.
    """
    if normals is None:
        normals = terrain_vertex_normals(placement)
    # A single mirror (odd number of flipped axes) reverses triangle winding,
    # which flips the surface orientation and makes the renderer light the
    # terrain from below. Reverse the winding back so faces point up.
    faces = terrain_faces(placement.grid)
    if placement.mirror_xy[0] ^ placement.mirror_xy[1]:
        faces = faces[:, ::-1]
    return trimesh.Trimesh(
        vertices=terrain_vertices(placement, scene_scale=scene_scale),
        faces=faces,
        vertex_colors=bake_shading_colors(normals, shading),
        process=False,
    )


def add_dem_terrain(
    server: viser.ViserServer,
    *,
    placement: DemPlacement,
    shading: DemShading = DemShading(),
    scene_scale: float,
) -> None:
    """Draw the DEM patch at ``/terrain`` with live "DEM Terrain"/"DEM Lighting" GUIs.

    Scene lighting is switched to flat ambient so the baked vertex colours
    render exactly as computed.  The sliders rebuild the mesh in place, which
    is how the landing site gets aligned with a terrain feature by eye; the
    numbers you settle on become the ``DemPlacement`` literal in the example.
    """
    server.scene.configure_default_lights(enabled=False)
    server.scene.add_light_ambient("/lights/ambient", color=(255, 255, 255), intensity=1.0)

    state = {
        "placement": placement,
        "shading": shading,
        "normals": terrain_vertex_normals(placement),
        "lock": threading.Lock(),
    }

    def _refresh() -> None:
        with state["lock"]:
            mesh = dem_trimesh(
                state["placement"],
                state["shading"],
                scene_scale=scene_scale,
                normals=state["normals"],
            )
        server.scene.add_mesh_trimesh("/terrain", mesh)

    _refresh()

    with server.gui.add_folder("DEM Terrain"):
        pos_x = server.gui.add_slider(
            "Position X (m)", min=-1000.0, max=1000.0, step=1.0, initial_value=placement.origin_m[0]
        )
        pos_y = server.gui.add_slider(
            "Position Y (m)", min=-1000.0, max=1000.0, step=1.0, initial_value=placement.origin_m[1]
        )
        pos_z = server.gui.add_slider(
            "Position Z (m)", min=-1000.0, max=1000.0, step=1.0, initial_value=placement.origin_m[2]
        )
        scale_x = server.gui.add_slider(
            "Scale X", min=0.1, max=10.0, step=0.05, initial_value=placement.scale_xyz[0]
        )
        scale_y = server.gui.add_slider(
            "Scale Y", min=0.1, max=10.0, step=0.05, initial_value=placement.scale_xyz[1]
        )
        scale_z = server.gui.add_slider(
            "Scale Z", min=0.0, max=10.0, step=0.05, initial_value=placement.scale_xyz[2]
        )
        yaw = server.gui.add_slider(
            "Yaw Z (°)", min=0.0, max=360.0, step=1.0, initial_value=placement.yaw_deg
        )
        mirror_x = server.gui.add_checkbox("Mirror X", initial_value=placement.mirror_xy[0])
        mirror_y = server.gui.add_checkbox("Mirror Y", initial_value=placement.mirror_xy[1])

        def _sync_placement(_e=None) -> None:
            state["placement"] = replace(
                state["placement"],
                origin_m=(float(pos_x.value), float(pos_y.value), float(pos_z.value)),
                scale_xyz=(float(scale_x.value), float(scale_y.value), float(scale_z.value)),
                yaw_deg=float(yaw.value),
                mirror_xy=(bool(mirror_x.value), bool(mirror_y.value)),
            )
            state["normals"] = terrain_vertex_normals(state["placement"])
            _refresh()

        for ctrl in (pos_x, pos_y, pos_z, scale_x, scale_y, scale_z, yaw, mirror_x, mirror_y):
            ctrl.on_update(_sync_placement)

    with server.gui.add_folder("DEM Lighting"):
        server.gui.add_markdown("_Baked into DEM vertex colours; other scene objects unaffected._")
        enabled = server.gui.add_checkbox("Enabled", initial_value=shading.enabled)
        azimuth = server.gui.add_slider(
            "Azimuth (°)", min=0.0, max=360.0, step=1.0, initial_value=shading.azimuth_deg
        )
        elevation = server.gui.add_slider(
            "Elevation (°)", min=0.5, max=89.0, step=0.5, initial_value=shading.elevation_deg
        )
        strength = server.gui.add_slider(
            "Strength", min=0.0, max=5.0, step=0.05, initial_value=shading.strength
        )
        ambient = server.gui.add_slider(
            "Ambient", min=0.0, max=0.5, step=0.005, initial_value=shading.ambient
        )

        def _sync_shading(_e=None) -> None:
            state["shading"] = DemShading(
                azimuth_deg=float(azimuth.value),
                elevation_deg=float(elevation.value),
                strength=float(strength.value),
                ambient=float(ambient.value),
                enabled=bool(enabled.value),
            )
            _refresh()

        for ctrl in (enabled, azimuth, elevation, strength, ambient):
            ctrl.on_update(_sync_shading)


def dem_info_markdown(placement: DemPlacement) -> str:
    """One-line "what am I looking at" summary for an example's Info folder."""
    n_tris = 2 * (placement.grid - 1) ** 2
    return (
        f"**DEM patch center** — use Position sliders above  \n"
        f"DEM: {placement.grid}×{placement.grid} · {n_tris:,} tris"
    )
