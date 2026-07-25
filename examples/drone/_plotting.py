"""Plotting helpers for drone examples."""

from __future__ import annotations

import numpy as np
import trimesh
import viser
from scipy.interpolate import CubicSpline, PchipInterpolator

from examples.drone._terrain import heightfield_mesh
from examples.drone.logo_utils.quadrotor_mesh import make_quadrotor_mesh
from openscvx.algorithms import OptimizationResults
from openscvx.plotting.viser import compute_velocity_colors, create_server
from openscvx.plotting.viser.animated import add_animated_trail, add_animation_controls

_COLUMN_PALETTE = [
    (44, 160, 44),
    (31, 119, 180),
    (255, 127, 14),
    (148, 103, 189),
    (214, 39, 40),
    (140, 86, 75),
]


def _column_center_at(
    t: float,
    t_knots: np.ndarray,
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    *,
    method: str = "pchip",
) -> np.ndarray:
    if method == "pchip":
        cx = float(PchipInterpolator(t_knots, x_knots)(t))
        cy = float(PchipInterpolator(t_knots, y_knots)(t))
    else:
        cx = float(CubicSpline(t_knots, x_knots)(t))
        cy = float(CubicSpline(t_knots, y_knots)(t))
    return np.array([cx, cy], dtype=np.float64)


def _circle_segments_xy(
    center_xy: np.ndarray,
    radius: float,
    *,
    z: float = 0.0,
    n_sides: int = 64,
) -> np.ndarray:
    """Closed circle on the plane z=const as line-segment endpoints, shape (n_sides, 2, 3)."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False, dtype=np.float64)
    ring = np.column_stack(
        [
            center_xy[0] + radius * np.cos(angles),
            center_xy[1] + radius * np.sin(angles),
            np.full(n_sides, z),
        ]
    ).astype(np.float32)
    return np.stack([ring, np.roll(ring, -1, axis=0)], axis=1)


def create_moving_safe_columns_server(
    results: OptimizationResults,
    *,
    loop_animation: bool = True,
    dark_mode: bool = True,
) -> viser.ViserServer:
    """Viser animation: quadrotor mesh with moving safe-zone circles on z=0."""
    traj = results.trajectory
    pos = np.asarray(traj["position"], dtype=np.float64)
    vel = np.asarray(traj["velocity"], dtype=np.float64)
    att = np.asarray(traj["attitude"], dtype=np.float64)
    traj_time = np.asarray(traj["time"], dtype=np.float64).reshape(-1)
    if traj_time.size != len(pos):
        traj_time = np.linspace(0.0, float(traj_time[-1]) if traj_time.size else 1.0, len(pos))

    radius = float(results["column_radius"])
    t_knots = np.asarray(results["t_knots"], dtype=float)
    column_paths = results["column_paths"]
    interp_method = results.get("column_interp_method", "pchip")
    start = np.asarray(results["start"], dtype=float)
    goal = np.asarray(results["goal"], dtype=float)

    server = create_server(pos, dark_mode=dark_mode, show_grid=True)
    server.scene.set_up_direction("+z")
    print(f"  Moving safe columns → http://localhost:{server.get_port()}")
    print("  Press Play in the Animation folder.  Ctrl-C to exit.")

    if len(pos) >= 2:
        ghost_segs = np.stack([pos[:-1], pos[1:]], axis=1).astype(np.float32)
        server.scene.add_line_segments(
            "/trajectory/full",
            ghost_segs,
            np.tile(np.array([180, 180, 190], dtype=np.uint8), (len(ghost_segs), 2, 1)),
            line_width=2.0,
        )

    trail_colors = compute_velocity_colors(vel, fallback_length=len(pos))
    _, update_trail = add_animated_trail(server, pos.astype(np.float32), trail_colors)

    server.scene.add_icosphere(
        "/start",
        radius=0.12,
        color=(30, 200, 80),
        position=tuple(float(x) for x in start),
    )
    server.scene.add_icosphere(
        "/goal",
        radius=0.12,
        color=(220, 50, 50),
        position=tuple(float(x) for x in goal),
    )

    mesh_verts, mesh_faces = make_quadrotor_mesh(scale=1.0)
    mesh_handle = server.scene.add_mesh_simple(
        "/vehicle_mesh",
        vertices=np.asarray(mesh_verts, dtype=np.float32),
        faces=np.asarray(mesh_faces, dtype=np.uint32),
        color=(200, 200, 210),
        position=tuple(float(x) for x in pos[0]),
        wxyz=tuple(float(x) for x in att[0]),
    )

    def update_vehicle(frame_idx: int) -> None:
        mesh_handle.position = tuple(float(x) for x in pos[frame_idx])
        mesh_handle.wxyz = tuple(float(x) for x in att[frame_idx])

    n_cols = len(column_paths)
    circle_handles = []
    centers_over_time: list[np.ndarray] = []

    for i, (xk, yk) in enumerate(column_paths):
        xk = np.asarray(xk, dtype=float)
        yk = np.asarray(yk, dtype=float)
        centers = np.array(
            [_column_center_at(float(t), t_knots, xk, yk, method=interp_method) for t in traj_time],
            dtype=np.float64,
        )
        centers_over_time.append(centers)
        color = _COLUMN_PALETTE[i % len(_COLUMN_PALETTE)]
        segs0 = _circle_segments_xy(centers[0], radius, z=0.0)
        handle = server.scene.add_line_segments(
            f"/safe_circle/{i}",
            segs0,
            np.tile(np.asarray(color, dtype=np.uint8), (len(segs0), 2, 1)),
            line_width=3.0,
        )
        circle_handles.append(handle)

    def update_circles(frame_idx: int) -> None:
        for handle, centers in zip(circle_handles, centers_over_time):
            handle.points = _circle_segments_xy(centers[frame_idx], radius, z=0.0)

    add_animation_controls(
        server,
        traj_time,
        [update_trail, update_vehicle, update_circles],
        loop=loop_animation,
    )
    update_trail(0)
    update_vehicle(0)
    update_circles(0)

    with server.gui.add_folder("Legend"):
        server.gui.add_markdown(
            f"**{n_cols} colored circles on z=0** — moving safe-zone footprints  \n"
            f"**Mesh + trail** — 6DoF quadrotor (press Play)  \n"
            f"**Green / red spheres** — start / goal  \n"
            f"Spec: Always( Or(in_column_i) ) — stay inside ≥1 column (infinite height)"
        )

    mid = pos[len(pos) // 2]
    default_fov_deg = 75.0
    server.initial_camera.position = (float(mid[0]), float(mid[1] - 12.0), float(mid[2] + 6.0))
    server.initial_camera.look_at = (float(mid[0]), float(mid[1]), float(mid[2]))
    server.initial_camera.up = (0.0, 0.0, 1.0)
    server.initial_camera.fov = float(np.radians(default_fov_deg))

    with server.gui.add_folder("Camera"):
        fov_slider = server.gui.add_slider(
            "FoV [deg]",
            min=10.0,
            max=120.0,
            step=1.0,
            initial_value=default_fov_deg,
        )

    def _apply_fov(fov_deg: float) -> None:
        fov_rad = float(np.radians(fov_deg))
        server.initial_camera.fov = fov_rad
        for client in server.get_clients().values():
            client.camera.fov = fov_rad

    @fov_slider.on_update
    def _(_e) -> None:
        _apply_fov(float(fov_slider.value))

    @server.on_client_connect
    def _on_connect(client) -> None:
        client.camera.fov = float(np.radians(float(fov_slider.value)))

    return server


def create_lunar_terrain_agl_server(
    results: OptimizationResults,
    *,
    mesh_grid: int = 256,
    vehicle_scale: float = 20.0,
    loop_animation: bool = True,
    dark_mode: bool = True,
) -> viser.ViserServer:
    """Viser animation: quadrotor flying an AGL band over the SENNS lunar DEM.

    Reads the terrain grid (``x_grid``, ``y_grid``, ``H``), the AGL band
    (``agl_target``, ``agl_tol``) and the endpoints (``start``, ``goal``) that
    ``lunar_terrain_agl.py`` stashes on ``results``, and renders the DEM as a
    lit heightfield beneath the flown trajectory.

    Args:
        results: Post-processed results carrying the trajectory and the terrain
            entries listed above.
        mesh_grid: Cap on the terrain mesh resolution per axis. The DEM is
            decimated by an integer stride to reach it, so a fine solver grid
            stays cheap to render.
        vehicle_scale: Uniform scale on the quadrotor mesh. The default makes the
            vehicle readable against a few-hundred-metre terrain patch.
        loop_animation: Restart playback at the end of the trajectory.
        dark_mode: Use the dark viser theme.
    """
    traj = results.trajectory
    pos = np.asarray(traj["position"], dtype=np.float64)
    vel = np.asarray(traj["velocity"], dtype=np.float64)
    att = np.asarray(traj["attitude"], dtype=np.float64)
    traj_time = np.asarray(traj["time"], dtype=np.float64).reshape(-1)
    if traj_time.size != len(pos):
        traj_time = np.linspace(0.0, float(traj_time[-1]) if traj_time.size else 1.0, len(pos))

    x_grid = np.asarray(results["x_grid"], dtype=np.float64)
    y_grid = np.asarray(results["y_grid"], dtype=np.float64)
    H = np.asarray(results["H"], dtype=np.float64)
    agl_target = float(results["agl_target"])
    agl_tol = float(results["agl_tol"])
    start = np.asarray(results["start"], dtype=np.float64)
    goal = np.asarray(results["goal"], dtype=np.float64)

    server = create_server(pos, dark_mode=dark_mode, show_grid=False)
    server.scene.set_up_direction("+z")
    print(f"  Lunar terrain AGL → http://localhost:{server.get_port()}")
    print("  Press Play in the Animation folder.  Ctrl-C to exit.")

    stride = max(1, -(-max(H.shape) // int(mesh_grid)))
    verts, faces, colors = heightfield_mesh(
        x_grid[::stride], y_grid[::stride], H[::stride, ::stride]
    )
    server.scene.add_mesh_trimesh(
        "/terrain",
        trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, process=False),
    )

    if len(pos) >= 2:
        ghost_segs = np.stack([pos[:-1], pos[1:]], axis=1).astype(np.float32)
        server.scene.add_line_segments(
            "/trajectory/full",
            ghost_segs,
            np.tile(np.array([180, 180, 190], dtype=np.uint8), (len(ghost_segs), 2, 1)),
            line_width=2.0,
        )

    trail_colors = compute_velocity_colors(vel, fallback_length=len(pos))
    _, update_trail = add_animated_trail(server, pos.astype(np.float32), trail_colors)

    server.scene.add_icosphere(
        "/start",
        radius=4.0,
        color=(30, 200, 80),
        position=tuple(float(x) for x in start),
    )
    server.scene.add_icosphere(
        "/goal",
        radius=4.0,
        color=(220, 50, 50),
        position=tuple(float(x) for x in goal),
    )

    mesh_verts, mesh_faces = make_quadrotor_mesh(scale=vehicle_scale)
    mesh_handle = server.scene.add_mesh_simple(
        "/vehicle_mesh",
        vertices=np.asarray(mesh_verts, dtype=np.float32),
        faces=np.asarray(mesh_faces, dtype=np.uint32),
        color=(200, 200, 210),
        position=tuple(float(x) for x in pos[0]),
        wxyz=tuple(float(x) for x in att[0]),
    )

    def update_vehicle(frame_idx: int) -> None:
        mesh_handle.position = tuple(float(x) for x in pos[frame_idx])
        mesh_handle.wxyz = tuple(float(x) for x in att[frame_idx])

    add_animation_controls(
        server,
        traj_time,
        [update_trail, update_vehicle],
        loop=loop_animation,
    )
    update_trail(0)
    update_vehicle(0)

    dx = float(x_grid[1] - x_grid[0]) * stride
    with server.gui.add_folder("Legend"):
        server.gui.add_markdown(
            f"**Terrain** — SENNS lunar DEM, {H.shape[0]}² samples "
            f"rendered at ~{dx:.1f} m spacing  \n"
            f"**Mesh + trail** — 6DoF quadrotor, trail colored by speed (press Play)  \n"
            f"**Green / red spheres** — start / goal  \n"
            f"Spec: AGL ∈ [{agl_target - agl_tol:.0f}, {agl_target + agl_tol:.0f}] m "
            f"enforced continuously (CTCS)"
        )

    mid = pos[len(pos) // 2]
    span = float(x_grid[-1] - x_grid[0])
    server.initial_camera.position = (
        float(mid[0]),
        float(mid[1] - 0.6 * span),
        float(mid[2] + 0.35 * span),
    )
    server.initial_camera.look_at = (float(mid[0]), float(mid[1]), float(mid[2]))
    server.initial_camera.up = (0.0, 0.0, 1.0)

    return server
