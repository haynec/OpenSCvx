"""Orbit visualization helpers for viser.

These utilities are intentionally lightweight and composable:

- Static primitives to draw circular orbit rings in the XY plane
- A convenience server builder for Hohmann-transfer-like trajectories, which
  overlays two circular orbits and animates the transfer trajectory.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import viser

from openscvx.algorithms import OptimizationResults

from .animated import add_animated_trail, add_animation_controls, add_position_marker
from .primitives import add_ghost_trajectory
from .server import compute_velocity_colors, create_server


def _as_3d(points: np.ndarray) -> np.ndarray:
    """Ensure points are shape (..., 3) by appending z=0 when needed."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        if points.shape[0] == 2:
            return np.array([points[0], points[1], 0.0], dtype=np.float64)
        if points.shape[0] == 3:
            return points
        raise ValueError(f"Expected 2D or 3D point, got shape {points.shape}")
    if points.ndim == 2:
        if points.shape[1] == 2:
            z = np.zeros((points.shape[0], 1), dtype=np.float64)
            return np.concatenate([points, z], axis=1)
        if points.shape[1] == 3:
            return points
        raise ValueError(f"Expected points with 2 or 3 columns, got shape {points.shape}")
    raise ValueError(f"Expected points with ndim 1 or 2, got ndim {points.ndim}")


def _circle_points_xy(
    radius: float,
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    n_points: int = 256,
) -> np.ndarray:
    """Generate points on a circle in the XY plane."""
    if n_points < 3:
        raise ValueError("n_points must be >= 3")
    center_3d = _as_3d(np.asarray(center))
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    pts = np.stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=1)
    return pts + center_3d.reshape(1, 3)


def _line_segments_closed(points: np.ndarray) -> np.ndarray:
    """Convert polyline points (N,3) into closed line segments (N,2,3)."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N,3), got {points.shape}")
    nxt = np.roll(points, shift=-1, axis=0)
    return np.stack([points, nxt], axis=1)


def add_circular_orbit(
    server: viser.ViserServer,
    radius: float,
    *,
    name: str,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    plane: Literal["xy"] = "xy",
    n_points: int = 256,
    color: tuple[int, int, int] = (200, 200, 200),
    line_width: float = 2.0,
) -> viser.LineSegmentsHandle:
    """Add a static circular orbit ring.

    Currently supports only the XY plane (common for planar problems).
    """
    if plane != "xy":
        raise NotImplementedError("Only plane='xy' is currently supported")
    pts = _circle_points_xy(radius, center=center, n_points=n_points)
    segs = _line_segments_closed(pts)
    return server.scene.add_line_segments(
        f"/orbits/{name}",
        points=segs,
        colors=color,
        line_width=line_width,
    )


def create_hohmann_transfer_server(
    results: OptimizationResults,
    *,
    r1: float,
    r2: float,
    position_key: str = "position",
    velocity_key: str = "velocity",
    loop_animation: bool = True,
    show_grid: bool = True,
    scene_scale: float = 1.0,
    orbit_n_points: int = 512,
    orbit_color_inner: tuple[int, int, int] = (120, 180, 255),
    orbit_color_outer: tuple[int, int, int] = (255, 180, 120),
    transfer_point_size: float = 0.10,
    marker_radius: float = 0.6,
) -> viser.ViserServer:
    """Create an animated viser server for a planar Hohmann transfer.

    Draws two static circular orbit rings (r1, r2) and animates the transfer
    trajectory stored in ``results.trajectory``.
    """
    pos = results.trajectory.get(position_key)
    if pos is None:
        raise KeyError(f"results.trajectory is missing '{position_key}'")
    pos_3d = _as_3d(pos) / scene_scale

    vel = results.trajectory.get(velocity_key)
    vel_3d = None if vel is None else _as_3d(vel)
    colors = compute_velocity_colors(vel_3d, fallback_length=pos_3d.shape[0])

    traj_time = np.asarray(results.trajectory["time"], dtype=np.float64).flatten()

    server = create_server(pos_3d, show_grid=show_grid)

    # Static orbit rings
    add_circular_orbit(
        server,
        r1 / scene_scale,
        name="inner_orbit",
        n_points=orbit_n_points,
        color=orbit_color_inner,
        line_width=2.5,
    )
    add_circular_orbit(
        server,
        r2 / scene_scale,
        name="outer_orbit",
        n_points=orbit_n_points,
        color=orbit_color_outer,
        line_width=2.5,
    )

    # Static ghost + animated trail + marker
    add_ghost_trajectory(server, pos_3d, colors, opacity=0.25, point_size=transfer_point_size)
    _, update_trail = add_animated_trail(server, pos_3d, colors, point_size=transfer_point_size)
    _, update_marker = add_position_marker(server, pos_3d, radius=marker_radius)

    add_animation_controls(server, traj_time, [update_trail, update_marker], loop=loop_animation)
    return server

