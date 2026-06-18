"""Viser-based trajectory visualization templates.

This module provides high-level, domain-specific visualization servers built on
top of the composable primitives in ``openscvx.plotting.viser``.  It is the
right place for opinionated compositions (snapshot grids, full-arm snapshots,
pick-and-place waypoints) that depend on problem-specific trajectory keys.

Boundary
--------
- **Primitives** (reusable, stable API): ``openscvx.plotting.viser``
- **Templates** (opinionated, example-layer): this module

For real-time examples, see ``examples/realtime/*.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import viser

from examples.plotting import _results_has_moving_subject, _subject_world_trajectories
from openscvx.algorithms import OptimizationResults
from openscvx.plotting import plot_controls
from openscvx.plotting.viser import (
    add_animated_plotly_vline,
    add_animated_trail,
    add_animated_vector_norm_plot,
    add_animation_controls,
    add_attitude_frame,
    add_circular_orbit,
    add_ellipsoid_obstacles,
    add_gates,
    add_ghost_trajectory,
    add_glideslope_cone,
    add_position_marker,
    add_scp_animation_controls,
    add_scp_ghost_iterations,
    add_scp_iteration_attitudes,
    add_scp_iteration_nodes,
    add_scp_propagation_lines,
    add_target_markers,
    add_thrust_plume,
    add_thrust_vector,
    add_viewcone,
    compute_velocity_colors,
    create_server,
    extract_propagation_positions,
)
from openscvx.plotting.viser.animated import _normalize_wxyz, place_body_frame, place_viewcone

# =============================================================================
# Manual-stepping handle (for offline rendering)
# =============================================================================


@dataclass
class AnimatedServerHandle:
    """Handle for manually stepping an animated viser server, one frame at a time.

    Returned by ``create_animated_plotting_server(..., controls="manual")``
    instead of starting the GUI playback loop. Primitives registered on the
    server all take ``frame_idx: int``, so calling ``handle.step(i)`` fans out
    to every trail, marker, attitude frame, thrust vector, viewcone, etc. —
    driving the scene without any wall-clock timer. Used by
    ``examples/animations/_render.py`` to render frames one at a time and pipe
    them into ffmpeg.
    """

    server: viser.ViserServer
    traj_time: np.ndarray
    update_callbacks: list[Callable[[int], None]]

    @property
    def n_frames(self) -> int:
        return len(self.traj_time)

    def step(self, frame_idx: int) -> None:
        """Drive every registered primitive to show frame ``frame_idx``."""
        idx = int(np.clip(frame_idx, 0, self.n_frames - 1))
        for cb in self.update_callbacks:
            cb(idx)


# =============================================================================
# Template Visualization Servers
# =============================================================================


def create_animated_plotting_server(
    results: OptimizationResults,
    loop_animation: bool = True,
    position_key: str = "position",
    velocity_key: str = "velocity",
    thrust_key: str = "force",
    thrust_scale: float = 0.3,
    thrust_style: str = "line",
    thrust_plume_half_angle_deg: float = 12.0,
    thrust_plume_color: tuple[int, int, int] = (255, 120, 40),
    thrust_plume_opacity: float = 0.45,
    thrust_remap_world_to_viser: bool = False,
    attitude_key: str = "attitude",
    attitude_axes_length: float = 2.0,
    vehicle_mesh: tuple[np.ndarray, np.ndarray] | None = None,
    vehicle_mesh_color: tuple[int, int, int] = (200, 200, 210),
    show_viewcone: bool = True,
    viewcone_scale: float = 10.0,
    viewcone_ring_only: bool = False,
    target_radius: float = 1.0,
    show_control_plot: str | None = None,
    show_control_norm_plot: str | None = None,
    trail_point_size: float = 0.15,
    show_grid: bool = True,
    scene_scale: float = 1.0,
    dark_mode: bool = True,
    background_color: tuple[int, int, int] | None = None,
    controls: str = "gui",
) -> viser.ViserServer | AnimatedServerHandle:
    """Create an animated trajectory visualization server.

    This is a convenience function that composes the modular components.
    For custom visualizations, use the individual add_* functions directly.

    Features:
    - Play/pause button for animation
    - Time slider to scrub through trajectory (realtime playback)
    - Speed control slider
    - Velocity-colored trail that grows as animation progresses
    - Current position marker
    - Thrust vector visualization (if thrust data available)
    - Body frame attitude visualization (if attitude data available, for 6DOF), or a
      posed vehicle mesh when ``vehicle_mesh`` is passed instead of axes
    - Viewcone mesh (if R_sb in results and show_viewcone=True)
    - Target markers for viewplanning (if init_poses in results)
    - Optional ghost trajectory showing full path
    - Static obstacles/gates if present in results
    - Ellipsoidal obstacles (if obstacles_centers, obstacles_radii, obstacles_axes in results)

    Args:
        results: Optimization result dictionary containing trajectory data.
            Expected keys in results (beyond trajectory data):
            - vertices: Gate/obstacle vertices (optional)
            - R_sb: Body-to-sensor rotation matrix for viewcone (optional)
            - alpha_x, alpha_y: Sensor cone half-angle parameters (optional)
            - norm_type: Norm type for viewcone constraint (optional, default 2)
            - init_poses: List of viewplanning target positions (optional)
            - obstacles_centers, obstacles_radii, obstacles_axes: Ellipsoid obstacles (optional)
        loop_animation: If True, loop animation when it reaches the end
        position_key: Key for position data in trajectory dict (default: "position")
        velocity_key: Key for velocity data in trajectory dict (default: "velocity")
        thrust_key: Key for thrust/force data in trajectory dict (default: "force")
        thrust_scale: Scale factor for thrust / plume length
        thrust_style: ``"line"`` for a thrust arrow, ``"plume"`` for an exhaust cone
            (points opposite to the thrust vector).
        thrust_plume_half_angle_deg: Half-angle of the exhaust cone when ``thrust_style="plume"``.
        thrust_plume_color: RGB color for the plume mesh.
        thrust_plume_opacity: Opacity for the plume mesh (0–1).
        thrust_remap_world_to_viser: Apply PDG (z, y, x) → Viser (x, y, z) to world-frame thrust.
        attitude_key: Key for attitude quaternion data (default: "attitude")
        attitude_axes_length: Length of body frame axes (ignored when ``vehicle_mesh`` is set)
        vehicle_mesh: Optional ``(vertices, faces)`` for a body-fixed mesh (e.g. drone geometry).
            When provided, the mesh is posed at ``position`` with quaternion ``attitude`` each
            frame instead of drawing the default attitude axes / frame.
        vehicle_mesh_color: RGB color for ``vehicle_mesh`` when provided.
        show_viewcone: If True and R_sb is in results, show camera viewcone
        viewcone_scale: Size/depth of viewcone mesh
        viewcone_ring_only: If True, render viewcone as a base-ring outline only
        target_radius: Radius of target marker spheres
        show_control_plot: If provided with a control name, displays component plot
            showing each control component vs time with animated markers
        show_control_norm_plot: If provided with a control name, displays norm plot
            showing ‖control‖₂ vs time with animated marker
        show_grid: Whether to show the grid (default True)
        scene_scale: Divide all positions (and lengths) by this factor. Use >1 for
            large-scale trajectories (e.g., 100.0 for km-scale problems).
        dark_mode: Whether to use the viser dark GUI theme (default True).
        background_color: RGB canvas background. Defaults to black when
            ``dark_mode`` is True and white when False.
        controls: ``"gui"`` (default) wires up play/pause/slider GUI and starts
            the wall-clock playback thread, returning the raw ``ViserServer``.
            ``"manual"`` skips the GUI/playback loop and instead returns an
            :class:`AnimatedServerHandle` whose ``step(frame_idx)`` method drives
            every primitive by hand — used for offline rendering (see
            ``examples/animations/_render.py``).

    Returns:
        ``ViserServer`` when ``controls="gui"``, otherwise
        :class:`AnimatedServerHandle`.
    """
    # Extract data and convert to numpy (handles JAX arrays)
    pos = results.trajectory.get(position_key)
    if pos is not None:
        pos = np.asarray(pos, dtype=np.float64) / scene_scale
    vel = results.trajectory.get(velocity_key)
    thrust = results.trajectory.get(thrust_key)
    attitude = results.trajectory.get(attitude_key)
    traj_time = results.trajectory["time"]

    # Viewcone parameters from results
    R_sb = results.get("R_sb")
    alpha_x = results.get("alpha_x")
    alpha_y = results.get("alpha_y")
    norm_type = results.get("norm_type", 2)

    # Compute half-angles in radians from alpha parameters
    # alpha_x defines the cone half-angle as pi/alpha_x radians
    if alpha_x is not None:
        half_angle_x = np.pi / alpha_x
        half_angle_y = np.pi / alpha_y if alpha_y is not None else half_angle_x
    else:
        # Default: 60 degree full FOV
        half_angle_x = np.radians(30.0)
        half_angle_y = half_angle_x

    # Viewplanning target positions
    init_poses = results.get("init_poses")

    # Logo/moving subject parameters
    extend_boresight = results.get("extend_boresight", False)
    moving_subject = results.get("moving_subject", False)
    get_kp_pose = results.get("get_kp_pose")
    total_time = results.get("total_time")
    relative_vector = results.get("relative_vector", False)
    logo_trace_color = tuple(results.get("logo_trace_color", (0, 255, 255)))

    # Precompute colors
    colors = compute_velocity_colors(vel)

    # Create server
    server = create_server(pos, dark_mode=dark_mode, show_grid=show_grid)
    if background_color is None:
        background_color = (0, 0, 0) if dark_mode else (255, 255, 255)
    _set_scene_background(server, background_color)

    def _add_relative_vector(
        name: str,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
        color: tuple[int, int, int] = (50, 255, 50),
        line_width: float = 2.0,
        opacity: float = 0.8,
    ):
        """Add an animated line segment from drone -> target."""
        handle = server.scene.add_line_segments(
            name,
            points=np.array([[drone_pos[0], target_pos[0]]], dtype=np.float32),
            colors=(color[0], color[1], color[2]),
            line_width=line_width,
        )

        def update(frame_idx: int) -> None:
            handle.points = np.array(
                [[drone_pos[frame_idx], target_pos[frame_idx]]], dtype=np.float32
            )

        return handle, update

    def _add_attitude_axes_lines(
        name: str,
        pos_: np.ndarray,
        attitude_: np.ndarray,
        axes_length: float,
        boresight_multiplier: float = 1.0,
        line_width: float = 3.0,
    ):
        """Custom attitude axes (supports extending x-axis/"boresight")."""

        def q_to_R_wxyz(q: np.ndarray) -> np.ndarray:
            w, x, y, z = q
            return np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ],
                dtype=np.float32,
            )

        # Axis unit vectors scaled by desired lengths (x can be extended)
        lengths = np.array(
            [axes_length * boresight_multiplier, axes_length, axes_length], dtype=np.float32
        )
        rgb_per_axis = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        colors = np.stack(
            [np.stack([rgb_per_axis[i], rgb_per_axis[i]], axis=0) for i in range(3)], axis=0
        )

        R0 = q_to_R_wxyz(attitude_[0])
        axes_world_0 = (
            R0 @ (np.eye(3, dtype=np.float32) * lengths)
        ).T  # (3, 3): each row is axis vec
        points0 = np.stack([[pos_[0], pos_[0] + axes_world_0[i]] for i in range(3)], axis=0)

        handle = server.scene.add_line_segments(
            name,
            points=points0,
            colors=colors,
            line_width=line_width,
        )

        def update(frame_idx: int) -> None:
            Rk = q_to_R_wxyz(attitude_[frame_idx])
            axes_world = (Rk @ (np.eye(3, dtype=np.float32) * lengths)).T
            pts = np.stack(
                [[pos_[frame_idx], pos_[frame_idx] + axes_world[i]] for i in range(3)],
                axis=0,
            )
            handle.points = pts

        return handle, update

    # Add static elements (scale positions/lengths by scene_scale)
    if "vertices" in results:
        add_gates(
            server,
            [np.asarray(v) / scene_scale for v in results["vertices"]],
        )

    # Add ellipsoidal obstacles if present
    if "obstacles_centers" in results:
        add_ellipsoid_obstacles(
            server,
            centers=[np.asarray(c) / scene_scale for c in results["obstacles_centers"]],
            radii=[
                np.asarray(r) / scene_scale
                for r in results.get(
                    "obstacles_radii",
                    [np.ones(3)] * len(results["obstacles_centers"]),
                )
            ],
            axes=results.get("obstacles_axes"),
        )

    # Add animated elements (collect update callbacks)
    update_callbacks = []

    # Trajectory primitives and position-dependent elements only when pos is available
    if pos is not None:
        add_ghost_trajectory(server, pos, colors)

        _, update_trail = add_animated_trail(server, pos, colors, point_size=trail_point_size)
        update_callbacks.append(update_trail)

        def _append_vehicle_mesh_or_body_axes(
            *,
            axes_path: str = "/body_axes",
            use_boresight_axes: bool = False,
            boresight_multiplier: float = 1.0,
        ) -> None:
            """Draw a posed vehicle mesh, or fall back to body axes / frame."""
            if vehicle_mesh is not None:
                mesh_verts = np.asarray(vehicle_mesh[0], dtype=np.float32)
                mesh_faces = np.asarray(vehicle_mesh[1], dtype=np.uint32)
                mesh_handle = server.scene.add_mesh_simple(
                    "/vehicle_mesh",
                    vertices=mesh_verts,
                    faces=mesh_faces,
                    color=vehicle_mesh_color,
                    position=tuple(float(x) for x in pos[0]),
                    wxyz=tuple(float(x) for x in attitude[0]),
                )

                def update_vehicle_mesh(frame_idx: int) -> None:
                    mesh_handle.position = tuple(float(x) for x in pos[frame_idx])
                    mesh_handle.wxyz = tuple(float(x) for x in attitude[frame_idx])

                update_callbacks.append(update_vehicle_mesh)
            elif use_boresight_axes:
                _, update_axes = _add_attitude_axes_lines(
                    axes_path,
                    pos,
                    attitude,
                    axes_length=attitude_axes_length,
                    boresight_multiplier=boresight_multiplier,
                )
                update_callbacks.append(update_axes)
            else:
                _, update_attitude = add_attitude_frame(
                    server, pos, attitude, axes_length=attitude_axes_length
                )
                update_callbacks.append(update_attitude)

        # Use position marker for point-mass, attitude frame for 6DOF
        if attitude is not None:
            if extend_boresight:
                # Plane: use logo plane (point + normal) if provided, else horizontal plane from
                # path_offset
                path_offset = results.get("path_offset")
                plane_z = None
                logo_plane_point = results.get("logo_plane_point")
                logo_plane_normal = results.get("logo_plane_normal")
                if path_offset is not None:
                    path_offset = np.asarray(path_offset)
                    plane_z = float(path_offset[2] / scene_scale) if len(path_offset) > 2 else None
                use_logo_plane = (
                    logo_plane_point is not None
                    and logo_plane_normal is not None
                    and len(np.asarray(logo_plane_point).flatten()) >= 3
                    and len(np.asarray(logo_plane_normal).flatten()) >= 3
                )
                if use_logo_plane:
                    plane_n = np.asarray(logo_plane_normal, dtype=np.float32).reshape(3)
                    plane_n = plane_n / (np.linalg.norm(plane_n) + 1e-10)
                elif plane_z is not None:
                    plane_n = None
                else:
                    plane_n = None

                if plane_z is not None or use_logo_plane:
                    boresight_intersection_points = compute_boresight_intersection_trail(
                        results, pos, attitude, scene_scale=scene_scale
                    )
                    if boresight_intersection_points is not None:
                        boresight_intersection_0 = boresight_intersection_points[0]

                        # Draw extended boresight to plane intersection
                        _ = server.scene.add_line_segments(
                            "/boresight_extended",
                            points=np.array([[pos[0], boresight_intersection_0]], dtype=np.float32),
                            colors=(255, 0, 0),  # Red for boresight
                            line_width=3.0,
                        )

                        _logo_trail_rgb = np.array([list(logo_trace_color)], dtype=np.uint8)

                        # Growing point-cloud trail for the boresight intersection.
                        boresight_trail_cloud = server.scene.add_point_cloud(
                            "/boresight_intersection_trail",
                            points=boresight_intersection_points[:1],
                            colors=_logo_trail_rgb.copy(),
                            point_size=0.06,
                        )

                        def update_boresight(frame_idx: int) -> None:
                            idx = min(frame_idx, len(boresight_intersection_points) - 1)
                            # Re-add boresight line (LineSegmentsHandle has no
                            # mutable points); same scene path replaces the old.
                            server.scene.add_line_segments(
                                "/boresight_extended",
                                points=np.array(
                                    [[pos[idx], boresight_intersection_points[idx]]],
                                    dtype=np.float32,
                                ),
                                colors=(255, 0, 0),
                                line_width=3.0,
                            )

                            # Grow trail up to current frame
                            n_trail = idx + 1
                            boresight_trail_cloud.points = boresight_intersection_points[:n_trail]
                            boresight_trail_cloud.colors = np.broadcast_to(
                                _logo_trail_rgb,
                                (n_trail, 3),
                            ).copy()

                        update_callbacks.append(update_boresight)

                        _append_vehicle_mesh_or_body_axes()
                    else:
                        _append_vehicle_mesh_or_body_axes(
                            use_boresight_axes=True,
                            boresight_multiplier=3.0,
                        )
                else:
                    _append_vehicle_mesh_or_body_axes(
                        use_boresight_axes=True,
                        boresight_multiplier=3.0,
                    )
            else:
                _append_vehicle_mesh_or_body_axes()
        else:
            _, update_marker = add_position_marker(server, pos)
            update_callbacks.append(update_marker)

        if thrust_style == "plume":
            _, update_thrust = add_thrust_plume(
                server,
                pos,
                thrust,
                attitude=attitude,
                scale=thrust_scale,
                half_angle_deg=thrust_plume_half_angle_deg,
                color=thrust_plume_color,
                opacity=thrust_plume_opacity,
                remap_world_to_viser=thrust_remap_world_to_viser,
            )
        else:
            _, update_thrust = add_thrust_vector(
                server,
                pos,
                thrust,
                attitude=attitude,
                scale=thrust_scale,
                remap_world_to_viser=thrust_remap_world_to_viser,
            )
        update_callbacks.append(update_thrust)  # Will be filtered out if None

        # Add viewcone mesh if R_sb is available and enabled
        if show_viewcone and R_sb is not None and attitude is not None:
            if viewcone_ring_only:
                # Match thrust vector styling so ring + thrust read as one element.
                viewcone_color = (255, 100, 100)
            else:
                # Compute viewcone color from viridis colormap (fallback if matplotlib missing)
                global plt
                if plt is None:
                    try:  # pragma: no cover
                        import matplotlib.pyplot as _plt

                        plt = _plt
                    except Exception:
                        plt = None

                if plt is not None:
                    cmap = plt.get_cmap("viridis")
                    rgb = cmap(0.4)[:3]
                    viewcone_color = tuple(int(c * 255) for c in rgb)
                else:
                    viewcone_color = (80, 180, 200)

            _, update_viewcone = add_viewcone(
                server,
                pos,
                attitude,
                half_angle_x=half_angle_x,
                half_angle_y=half_angle_y,
                scale=viewcone_scale,
                norm_type=norm_type,
                R_sb=R_sb,
                color=viewcone_color,
                wireframe=False,
                ring_only=viewcone_ring_only,
                opacity=0.4,
            )
            update_callbacks.append(update_viewcone)

    # Add target markers for viewplanning problems
    if init_poses is not None:
        scaled_init_poses = [np.asarray(p) / scene_scale for p in init_poses]
        target_results = add_target_markers(
            server, scaled_init_poses, radius=target_radius / scene_scale
        )
        for _, update in target_results:
            if update is not None:
                update_callbacks.append(update)

    # Add "logo" moving subject (single moving target + optional drone->target vector)
    if moving_subject and get_kp_pose is not None and total_time is not None:
        # Build target trajectory aligned with results.trajectory["time"]
        tt = np.asarray(traj_time).reshape(-1)
        total_time_f = float(np.asarray(total_time).reshape(-1)[0])
        target_traj = np.stack(
            [np.asarray(get_kp_pose(float(t) / total_time_f), dtype=np.float32) for t in tt],
            axis=0,
        )
        target_traj = target_traj / scene_scale

        logo_target_results = add_target_markers(
            server,
            [target_traj],
            colors=[(255, 50, 50)],
            radius=target_radius / scene_scale,
            show_trails=True,
        )
        for _, update in logo_target_results:
            if update is not None:
                update_callbacks.append(update)

        if relative_vector and pos is not None:
            _, update_rel = _add_relative_vector(
                "/relative_vector/target_0",
                drone_pos=np.asarray(pos, dtype=np.float32),
                target_pos=target_traj,
            )
            update_callbacks.append(update_rel)

            # Plot precomputed traced path (relative vector ∩ plane) when provided
            traced_path_on_plane = results.get("traced_path_on_plane")
            if traced_path_on_plane is None:
                intersection_points = None
                n_frames = 0
            else:
                intersection_points = (
                    np.asarray(traced_path_on_plane, dtype=np.float32) / scene_scale
                )
                n_frames = len(intersection_points)

            path_offset = results.get("path_offset")
            plane_z = None
            if path_offset is not None:
                path_offset = np.asarray(path_offset)
                plane_z = float(path_offset[2] / scene_scale) if len(path_offset) > 2 else None

            if intersection_points is not None and n_frames > 0:
                # Static line: full "traced" path (relative vector ∩ plane over entire trajectory)
                if len(intersection_points) > 1:
                    traced_path_segments = np.array(
                        [
                            [intersection_points[i], intersection_points[i + 1]]
                            for i in range(len(intersection_points) - 1)
                        ],
                        dtype=np.float32,
                    )
                    server.scene.add_line_segments(
                        "/traced_path_on_plane",
                        points=traced_path_segments,
                        colors=logo_trace_color,
                        line_width=2.5,
                    )

                # Add intersection point marker (slightly above plane so visible when it coincides
                # with target)
                rel_int_pos_0 = intersection_points[0].copy()
                rel_int_pos_0[2] += (
                    0.08  # Small offset above plane so it's not hidden under target sphere
                )
                rel_intersection_handle = server.scene.add_icosphere(
                    "/relative_vector_intersection",
                    radius=0.08,
                    color=(50, 255, 50),  # Green for relative vector intersection
                    position=rel_int_pos_0,
                )

                # Growing point-cloud trail for the relative-vector intersection.
                intersection_trail_cloud = server.scene.add_point_cloud(
                    "/relative_vector_intersection_trail",
                    points=intersection_points[:1],
                    colors=np.array([[50, 200, 50]], dtype=np.uint8),
                    point_size=0.06,
                )

                # Line on plane from boresight intersection to relative-vector intersection
                # (if both exist)
                plane_segment_handle = None
                if (
                    extend_boresight
                    and pos is not None
                    and attitude is not None
                    and plane_z is not None
                ):
                    path_offset_boresight = results.get("path_offset")
                    plane_z_boresight = None
                    if path_offset_boresight is not None:
                        path_offset_boresight = np.asarray(path_offset_boresight)
                        plane_z_boresight = (
                            float(path_offset_boresight[2] / scene_scale)
                            if len(path_offset_boresight) > 2
                            else None
                        )
                    if plane_z_boresight is not None and abs(plane_z_boresight - plane_z) < 1e-9:
                        # Precompute boresight intersections for plane segment
                        boresight_body_arr = results.get(
                            "boresight_body", np.array([1.0, 0.0, 0.0])
                        )
                        boresight_body_arr = np.asarray(boresight_body_arr, dtype=np.float32)
                        if boresight_body_arr.shape[0] >= 3:
                            boresight_body_arr = boresight_body_arr[:3]
                        boresight_body_arr = boresight_body_arr / (
                            np.linalg.norm(boresight_body_arr) + 1e-10
                        )
                        boresight_pts = []
                        for i in range(n_frames):
                            w, x, y, z = attitude[i]
                            R = np.array(
                                [
                                    [
                                        1 - 2 * (y * y + z * z),
                                        2 * (x * y - z * w),
                                        2 * (x * z + y * w),
                                    ],
                                    [
                                        2 * (x * y + z * w),
                                        1 - 2 * (x * x + z * z),
                                        2 * (y * z - x * w),
                                    ],
                                    [
                                        2 * (x * z - y * w),
                                        2 * (y * z + x * w),
                                        1 - 2 * (x * x + y * y),
                                    ],
                                ],
                                dtype=np.float32,
                            )
                            bw = R @ boresight_body_arr
                            bi = _ray_plane_intersection_horizontal(pos[i], bw, plane_z)
                            boresight_pts.append(bi if bi is not None else intersection_points[i])
                        boresight_pts = np.array(boresight_pts, dtype=np.float32)
                        plane_segment_handle = server.scene.add_line_segments(
                            "/plane_segment_boresight_to_rel",
                            points=np.array(
                                [[boresight_pts[0], intersection_points[0]]], dtype=np.float32
                            ),
                            colors=(200, 200, 0),  # Yellow: segment on plane
                            line_width=2.5,
                        )

                def update_intersection(frame_idx: int) -> None:
                    idx = min(frame_idx, len(intersection_points) - 1)
                    p = intersection_points[idx].copy()
                    p[2] += 0.08
                    rel_intersection_handle.position = p

                    # Grow trail up to current frame
                    n_trail = idx + 1
                    intersection_trail_cloud.points = intersection_points[:n_trail]
                    intersection_trail_cloud.colors = np.broadcast_to(
                        np.array([[50, 200, 50]], dtype=np.uint8),
                        (n_trail, 3),
                    ).copy()

                    if plane_segment_handle is not None and idx < len(boresight_pts):
                        # Re-add (LineSegmentsHandle has no mutable points).
                        server.scene.add_line_segments(
                            "/plane_segment_boresight_to_rel",
                            points=np.array(
                                [[boresight_pts[idx], intersection_points[idx]]],
                                dtype=np.float32,
                            ),
                            colors=(200, 200, 0),
                            line_width=2.5,
                        )

                update_callbacks.append(update_intersection)

    # Add control norm plot if requested
    if show_control_norm_plot is not None:
        _, update_norm = add_animated_vector_norm_plot(
            server,
            results,
            show_control_norm_plot,
            title=f"‖{show_control_norm_plot}‖₂",
            folder_name=f"{show_control_norm_plot} Norm",
        )
        if update_norm is not None:
            update_callbacks.append(update_norm)

    # Add control component plot if requested
    if show_control_plot is not None:
        has_in_trajectory = bool(results.trajectory) and show_control_plot in results.trajectory
        has_in_nodes = show_control_plot in results.nodes

        if has_in_trajectory or has_in_nodes:
            # Create figure using plot_controls (with list of one control)
            fig = plot_controls(results, [show_control_plot])

            # Determine data source for vertical line position
            if has_in_trajectory:
                time_data = results.trajectory["time"].flatten()
                use_trajectory_indexing = True
            else:
                time_data = results.nodes["time"].flatten()
                use_trajectory_indexing = False

            # Add animated vertical line using generic utility
            _, update_vline = add_animated_plotly_vline(
                server,
                fig,
                time_array=time_data,
                use_trajectory_indexing=use_trajectory_indexing,
                folder_name=f"{show_control_plot} Components",
            )
            update_callbacks.append(update_vline)

    # Wire up playback — either the wall-clock GUI loop, or a manual-step handle.
    callbacks = [cb for cb in update_callbacks if cb is not None]
    if controls == "gui":
        add_animation_controls(server, traj_time, callbacks, loop=loop_animation)
        return server
    elif controls == "manual":
        return AnimatedServerHandle(
            server=server,
            traj_time=np.asarray(traj_time, dtype=np.float64).flatten(),
            update_callbacks=callbacks,
        )
    else:
        raise ValueError(f"controls must be 'gui' or 'manual', got {controls!r}")


def _snapshot_frame_indices(n_snapshots: int, n_frames: int) -> np.ndarray:
    """Evenly spaced trajectory indices for ``n_snapshots`` poses (inclusive endpoints)."""
    n_snapshots = int(np.clip(n_snapshots, 1, n_frames))
    if n_snapshots == 1:
        return np.array([0], dtype=int)
    return np.linspace(0, n_frames - 1, n_snapshots, dtype=int)


def _ray_plane_intersection_horizontal(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    plane_z: float,
) -> np.ndarray | None:
    """Intersection of a ray with a horizontal plane z = plane_z."""
    if abs(ray_direction[2]) < 1e-10:
        return None
    t = (plane_z - ray_origin[2]) / ray_direction[2]
    if t < 0:
        return None
    return ray_origin + t * ray_direction


def _ray_plane_intersection_general(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    """Intersection of a ray with a plane (point on plane + unit normal)."""
    denom = float(np.dot(ray_direction, plane_normal))
    if abs(denom) < 1e-10:
        return None
    t = float(np.dot(plane_point - ray_origin, plane_normal)) / denom
    if t < 0:
        return None
    return ray_origin + t * ray_direction


def _quat_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def compute_boresight_intersection_trail(
    results: OptimizationResults,
    pos: np.ndarray,
    attitude: np.ndarray,
    scene_scale: float = 1.0,
) -> np.ndarray | None:
    """Boresight ∩ logo-plane samples (same as ``/boresight_intersection_trail`` in animation).

    Args:
        results: Post-processed optimization results (plane / boresight metadata).
        pos: Position trajectory, already divided by ``scene_scale``.
        attitude: Attitude quaternions (wxyz), same length as ``pos``.
        scene_scale: Scene scale used for plane geometry in ``results``.

    Returns:
        ``(N, 3)`` intersection points in scene coordinates, or ``None`` if undefined.
    """
    path_offset = results.get("path_offset")
    plane_z = None
    logo_plane_point = results.get("logo_plane_point")
    logo_plane_normal = results.get("logo_plane_normal")
    if path_offset is not None:
        path_offset = np.asarray(path_offset)
        plane_z = float(path_offset[2] / scene_scale) if len(path_offset) > 2 else None
    use_logo_plane = (
        logo_plane_point is not None
        and logo_plane_normal is not None
        and len(np.asarray(logo_plane_point).flatten()) >= 3
        and len(np.asarray(logo_plane_normal).flatten()) >= 3
    )
    if plane_z is None and not use_logo_plane:
        return None

    if use_logo_plane:
        plane_pt = np.asarray(logo_plane_point, dtype=np.float32).reshape(3) / scene_scale
        plane_n = np.asarray(logo_plane_normal, dtype=np.float32).reshape(3)
        plane_n = plane_n / (np.linalg.norm(plane_n) + 1e-10)
    else:
        plane_pt = None
        plane_n = None

    boresight_body = results.get("boresight_body", np.array([1.0, 0.0, 0.0]))
    boresight_body = np.asarray(boresight_body, dtype=np.float32)
    if boresight_body.shape[0] >= 3:
        boresight_body = boresight_body[:3]
    boresight_body = boresight_body / (np.linalg.norm(boresight_body) + 1e-10)

    n_frames = len(pos)
    points: list[np.ndarray] = []
    fallback: np.ndarray | None = None
    for i in range(n_frames):
        boresight_world = _quat_wxyz_to_rotation_matrix(attitude[i]) @ boresight_body
        if use_logo_plane:
            intersection = _ray_plane_intersection_general(
                pos[i], boresight_world, plane_pt, plane_n
            )
        else:
            intersection = _ray_plane_intersection_horizontal(
                pos[i], boresight_world, float(plane_z)
            )
        if intersection is not None:
            fallback = intersection
            points.append(intersection)
        elif fallback is not None:
            points.append(fallback)
        else:
            return None

    if not points:
        return None
    return np.asarray(points, dtype=np.float32)


def _set_scene_background(
    server: viser.ViserServer,
    background_color: tuple[int, int, int],
) -> None:
    """Set the 3D canvas clear color (independent of GUI light/dark theme)."""
    bg_rgb = np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)
    bg_image = np.broadcast_to(bg_rgb, (2, 2, 3)).copy()
    server.scene.set_background_image(bg_image, format="png")


def compute_poe_joint_keypoints(
    results: OptimizationResults,
    joint_zero_pos: np.ndarray,
    n_joints: int,
    *,
    t_home: np.ndarray | None = None,
    transform_prefix: str = "T_j",
) -> np.ndarray:
    """World-frame joint + EE positions from PoE transforms stored in ``results.trajectory``.

    Returns:
        Array of shape ``(n_frames, n_joints + 1, 3)``.
    """
    joint_zero_pos = np.asarray(joint_zero_pos, dtype=np.float64)
    n_frames = len(results.trajectory["time"])
    keypoints = np.zeros((n_frames, n_joints + 1, 3), dtype=np.float64)
    t_home = np.eye(4) if t_home is None else np.asarray(t_home, dtype=np.float64)

    for t_idx in range(n_frames):
        for k in range(n_joints):
            t_key = f"{transform_prefix}{k + 1}"
            if t_key not in results.trajectory:
                raise KeyError(
                    f"results.trajectory is missing '{t_key}' "
                    "(required for manipulator snapshot keypoints)."
                )
            t_k = np.asarray(results.trajectory[t_key][t_idx], dtype=np.float64)
            q0 = np.append(joint_zero_pos[k], 1.0)
            keypoints[t_idx, k] = (t_k @ q0)[:3]
        t_n = np.asarray(
            results.trajectory[f"{transform_prefix}{n_joints}"][t_idx], dtype=np.float64
        )
        keypoints[t_idx, n_joints] = (t_n @ t_home)[:3, 3]

    return keypoints


def build_arm_line_snapshot_builder(
    keypoints: np.ndarray,
    *,
    line_color: tuple[int, int, int] = (200, 200, 200),
    line_width: float = 5.0,
    origin_at_world_zero: bool = True,
) -> Callable[[viser.ViserServer, int, int], list]:
    """Build line-segment snapshots for a serial manipulator (origin → J1 → … → EE)."""
    keypoints = np.asarray(keypoints, dtype=np.float64)
    n_joints = keypoints.shape[1] - 1
    n_segs = n_joints + (1 if origin_at_world_zero else 0)
    seg_col = np.full((n_segs, 2, 3), line_color, dtype=np.uint8)

    def _segment_points(frame_idx: int) -> np.ndarray:
        pts = np.zeros((n_segs, 2, 3), dtype=np.float32)
        seg = 0
        if origin_at_world_zero:
            pts[0] = [np.zeros(3, dtype=np.float32), keypoints[frame_idx, 0]]
            seg = 1
        for k in range(n_joints - 1):
            pts[seg + k] = [keypoints[frame_idx, k], keypoints[frame_idx, k + 1]]
        pts[-1] = [keypoints[frame_idx, n_joints - 1], keypoints[frame_idx, n_joints]]
        return pts

    def builder(server: viser.ViserServer, snapshot_i: int, frame_idx: int) -> list:
        handle = server.scene.add_line_segments(
            f"/snapshots/arm_{snapshot_i}",
            points=_segment_points(frame_idx),
            colors=seg_col,
            line_width=line_width,
        )
        return [handle]

    return builder


def build_cad_link_snapshot_builder(
    link_meshes: dict[str, tuple[np.ndarray, np.ndarray]],
    link_world_T: dict[str, np.ndarray],
    *,
    link_colors: dict[str, tuple[int, int, int]] | None = None,
    default_color: tuple[int, int, int] = (210, 210, 215),
) -> Callable[[viser.ViserServer, int, int], list]:
    """Place posed CAD link meshes at each snapshot frame (MuJoCo FK transforms)."""
    from scipy.spatial.transform import Rotation

    link_colors = link_colors or {}

    def _pose_from_T(
        T: np.ndarray,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        R = np.asarray(T, dtype=np.float64)[:3, :3]
        t = T[:3, 3]
        q_xyzw = Rotation.from_matrix(R).as_quat()
        wxyz = (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))
        pos = (float(t[0]), float(t[1]), float(t[2]))
        return pos, wxyz

    def builder(server: viser.ViserServer, snapshot_i: int, frame_idx: int) -> list:
        handles = []
        for link_name, (verts_local, faces) in link_meshes.items():
            pos, wxyz = _pose_from_T(link_world_T[link_name][frame_idx])
            handle = server.scene.add_mesh_simple(
                f"/snapshots/robot_{snapshot_i}/{link_name}",
                vertices=np.asarray(verts_local, dtype=np.float32, order="C"),
                faces=faces,
                color=link_colors.get(link_name, default_color),
                opacity=1.0,
                position=pos,
                wxyz=wxyz,
            )
            handles.append(handle)
        return handles

    return builder


def create_snapshot_plotting_server(
    results: OptimizationResults,
    position_key: str = "position",
    velocity_key: str | None = "velocity",
    attitude_key: str | None = "attitude",
    attitudes: np.ndarray | None = None,
    attitude_axes_length: float = 2.0,
    show_body_frame: bool | None = None,
    show_viewcone: bool | None = None,
    viewcone_scale: float = 10.0,
    target_radius: float = 1.0,
    target_positions: np.ndarray | list[np.ndarray] | None = None,
    waypoint_positions: list[np.ndarray] | None = None,
    waypoint_colors: list[tuple[int, int, int]] | None = None,
    obstacle_center: np.ndarray | None = None,
    obstacle_radius: float | None = None,
    arm_keypoints: np.ndarray | None = None,
    scene_scale: float = 1.0,
    initial_n_snapshots: int = 5,
    max_n_snapshots: int | None = None,
    background_color: tuple[int, int, int] = (255, 255, 255),
    show_grid: bool = False,
    ghost_point_size: float = 0.08,
    ghost_opacity: float = 0.35,
    folder_name: str = "Snapshots",
    show_targets: bool = True,
    logo_trace_point_size: float | None = None,
    snapshot_builder: Callable[[viser.ViserServer, int, int], list] | None = None,
    vehicle_mesh: tuple[np.ndarray, np.ndarray] | None = None,
    vehicle_mesh_color: tuple[int, int, int] = (200, 200, 210),
) -> viser.ViserServer:
    """Create a static multi-pose visualization with GUI-controlled snapshot count.

    Shows the full trajectory as a faint ghost path, plus optional body frames,
    viewcones, manipulator geometry, and targets at evenly spaced poses. The number
    of snapshots is controlled from the viser GUI (no time animation).

    Supports aerial viewplanning (``position`` + ``attitude``), manipulators
    (``ee_position`` + ``arm_keypoints`` / ``snapshot_builder``), waypoints, and
    spherical obstacles via ``obstacle_center`` / ``obstacle_radius``.

    Args:
        results: Post-processed optimization results.
        position_key: Trajectory key for the ghost path (e.g. ``"position"`` or
            ``"ee_position"``).
        velocity_key: Trajectory key for path coloring; ``None`` uses a flat color.
        attitude_key: Trajectory key for attitude quaternions; ``None`` if absent.
        attitudes: Optional ``(N, 4)`` wxyz quaternions; overrides ``attitude_key``.
        attitude_axes_length: Body-frame axis length at each snapshot.
        show_body_frame: Draw a coordinate frame at each snapshot when attitude data
            exists. Defaults to ``True`` only when attitude data is available.
        show_viewcone: Draw viewcones when ``R_sb`` and attitude exist. Defaults
            accordingly.
        target_positions: Viewplanning targets (falls back to ``init_poses``).
        waypoint_positions: Static task waypoints (e.g. pick-and-place poses).
        waypoint_colors: Per-waypoint RGB colors.
        obstacle_center: Single spherical obstacle center (metres).
        obstacle_radius: Radius for ``obstacle_center`` (metres).
        arm_keypoints: ``(n_frames, n_joints+1, 3)`` link positions; used for line-segment
            snapshots when ``snapshot_builder`` is not provided.
        snapshot_builder: ``(server, snapshot_i, frame_idx) -> [handles]`` for extra
            geometry (e.g. CAD link meshes via :func:`build_cad_link_snapshot_builder`).
        vehicle_mesh: Optional ``(vertices, faces)`` posed at each snapshot instead of
            body-frame axes when attitude data is available.
        vehicle_mesh_color: RGB color for ``vehicle_mesh``.
        viewcone_scale: Depth of each viewcone mesh.
        target_radius: Radius of target / waypoint marker spheres.
        scene_scale: Scale divisor for positions and lengths.
        initial_n_snapshots: Initial snapshot count (overridden by
            ``results["initial_n_snapshots"]`` when set).
        max_n_snapshots: Upper bound for the GUI slider (defaults to frame count).
        background_color: RGB canvas background (default white).
        show_grid: Whether to draw the ground grid.
        ghost_point_size: Point size for the ghost trajectory.
        ghost_opacity: Opacity multiplier on velocity-colored ghost points (1.0 matches the
            animated trail brightness).
        folder_name: viser GUI folder name for the snapshot slider.
        show_targets: Draw moving/static target markers and per-snapshot target spheres.
        logo_trace_point_size: When set, render the logo trace as a point cloud at this size.
            Uses ``compute_boresight_intersection_trail`` when available (same as the animated
            ``/boresight_intersection_trail``); otherwise falls back to ``traced_path_on_plane``.
            A GUI slider adjusts point size live.

    Returns:
        ViserServer instance.
    """
    pos = results.trajectory.get(position_key)
    if pos is None:
        raise KeyError(f"results.trajectory is missing '{position_key}'")
    pos = np.asarray(pos, dtype=np.float64) / scene_scale

    vel = None
    if velocity_key is not None:
        vel = results.trajectory.get(velocity_key)

    attitude = None
    if attitudes is not None:
        attitude = np.asarray(attitudes, dtype=np.float64)
    elif attitude_key is not None:
        raw_att = results.trajectory.get(attitude_key)
        if raw_att is not None:
            attitude = np.asarray(raw_att, dtype=np.float64)

    has_attitude = attitude is not None
    if show_body_frame is None:
        show_body_frame = has_attitude and vehicle_mesh is None
    if show_viewcone is None:
        show_viewcone = has_attitude and results.get("R_sb") is not None
    if show_body_frame and not has_attitude:
        show_body_frame = False
    if show_viewcone and not has_attitude:
        show_viewcone = False

    mesh_verts = mesh_faces = None
    if vehicle_mesh is not None:
        mesh_verts, mesh_faces = vehicle_mesh
        mesh_verts = np.asarray(mesh_verts, dtype=np.float32)
        mesh_faces = np.asarray(mesh_faces, dtype=np.uint32)

    n_frames = pos.shape[0]
    max_snapshots = n_frames if max_n_snapshots is None else min(max_n_snapshots, n_frames)
    stored_n = results.get("initial_n_snapshots")
    initial_n = int(
        np.clip(stored_n if stored_n is not None else initial_n_snapshots, 1, max_snapshots)
    )

    R_sb = results.get("R_sb")
    alpha_x = results.get("alpha_x")
    alpha_y = results.get("alpha_y")
    norm_type = results.get("norm_type", 2)
    if alpha_x is not None:
        half_angle_x = np.pi / alpha_x
        half_angle_y = np.pi / alpha_y if alpha_y is not None else half_angle_x
    else:
        half_angle_x = np.radians(30.0)
        half_angle_y = half_angle_x

    init_poses = results.get("init_poses")
    if target_positions is not None:
        init_poses = target_positions

    waypoints = (
        waypoint_positions if waypoint_positions is not None else results.get("waypoint_positions")
    )
    waypoint_colors = (
        waypoint_colors if waypoint_colors is not None else results.get("waypoint_colors")
    )

    obs_center = obstacle_center if obstacle_center is not None else results.get("obstacle_center")
    obs_radius = obstacle_radius if obstacle_radius is not None else results.get("obstacle_radius")

    arm_kp = arm_keypoints if arm_keypoints is not None else results.get("arm_keypoints")
    if snapshot_builder is None and arm_kp is not None:
        snapshot_builder = build_arm_line_snapshot_builder(
            np.asarray(arm_kp, dtype=np.float64) / scene_scale
        )

    colors = compute_velocity_colors(vel, fallback_length=n_frames)

    server = create_server(pos, dark_mode=False, show_grid=show_grid)
    _set_scene_background(server, background_color)

    if "vertices" in results:
        add_gates(
            server,
            [np.asarray(v) / scene_scale for v in results["vertices"]],
        )

    if "obstacles_centers" in results:
        add_ellipsoid_obstacles(
            server,
            centers=[np.asarray(c) / scene_scale for c in results["obstacles_centers"]],
            radii=[
                np.asarray(r) / scene_scale
                for r in results.get(
                    "obstacles_radii",
                    [np.ones(3)] * len(results["obstacles_centers"]),
                )
            ],
            axes=results.get("obstacles_axes"),
        )
    elif obs_center is not None and obs_radius is not None:
        add_ellipsoid_obstacles(
            server,
            centers=[np.asarray(obs_center, dtype=np.float64) / scene_scale],
            radii=[np.full(3, 1.0 / float(obs_radius) / scene_scale)],
        )

    add_ghost_trajectory(server, pos, colors, opacity=ghost_opacity, point_size=ghost_point_size)

    traj_time = np.asarray(results.trajectory["time"]).flatten()
    relative_vector = results.get("relative_vector", False)
    get_kp_pose = results.get("get_kp_pose")
    total_time = results.get("total_time")

    target_traj_scaled: np.ndarray | None = None
    if get_kp_pose is not None and total_time is not None:
        total_time_f = float(np.asarray(total_time).reshape(-1)[0])
        target_traj_scaled = (
            np.stack(
                [
                    np.asarray(get_kp_pose(float(t) / total_time_f), dtype=np.float32)
                    for t in traj_time
                ],
                axis=0,
            )
            / scene_scale
        )

    logo_trace_points: np.ndarray | None = None
    if has_attitude and results.get("extend_boresight", False):
        logo_trace_points = compute_boresight_intersection_trail(
            results, pos, attitude, scene_scale=scene_scale
        )
    if logo_trace_points is None:
        traced_path_on_plane = results.get("traced_path_on_plane")
        if traced_path_on_plane is not None:
            logo_trace_points = np.asarray(traced_path_on_plane, dtype=np.float32) / scene_scale

    logo_trace_color = tuple(results.get("logo_trace_color", (0, 0, 0)))

    if (
        logo_trace_points is not None
        and len(logo_trace_points) > 1
        and logo_trace_point_size is None
    ):
        traced_path_segments = np.array(
            [
                [logo_trace_points[i], logo_trace_points[i + 1]]
                for i in range(len(logo_trace_points) - 1)
            ],
            dtype=np.float32,
        )
        server.scene.add_line_segments(
            "/snapshots/boresight_intersection_trail",
            points=traced_path_segments,
            colors=logo_trace_color,
            line_width=2.5,
        )

    logo_trace_state: dict[str, object] = {"handle": None}

    def rebuild_logo_trace(point_size: float) -> None:
        if logo_trace_points is None or len(logo_trace_points) == 0:
            return
        if logo_trace_point_size is None:
            return
        handle = logo_trace_state["handle"]
        if handle is not None:
            handle.remove()
        logo_trace_state["handle"] = server.scene.add_point_cloud(
            "/snapshots/boresight_intersection_trail",
            points=logo_trace_points,
            colors=np.broadcast_to(
                np.array([logo_trace_color], dtype=np.uint8),
                (len(logo_trace_points), 3),
            ).copy(),
            point_size=float(point_size),
        )

    if logo_trace_point_size is not None:
        rebuild_logo_trace(logo_trace_point_size)

    if waypoints is not None:
        scaled_waypoints = [np.asarray(p, dtype=np.float64) / scene_scale for p in waypoints]
        wp_colors = list(waypoint_colors) if waypoint_colors is not None else None
        add_target_markers(
            server,
            scaled_waypoints,
            radius=target_radius / scene_scale,
            colors=wp_colors,
            show_trails=False,
        )

    dynamic_subjects = _results_has_moving_subject(results)
    subject_trajs_scaled: list[np.ndarray] | None = None
    if show_targets and dynamic_subjects:
        subject_trajs_scaled = [
            np.asarray(traj, dtype=np.float64) / scene_scale
            for traj in _subject_world_trajectories(results, traj_time)
        ]
        add_target_markers(
            server,
            subject_trajs_scaled,
            radius=target_radius / scene_scale,
            show_trails=True,
        )
    elif show_targets and init_poses is not None:
        scaled_init_poses = [np.asarray(p) / scene_scale for p in init_poses]
        add_target_markers(server, scaled_init_poses, radius=target_radius / scene_scale)

    snapshot_state: dict[str, list] = {"handles": []}

    _target_colors = [
        (255, 50, 50),
        (50, 255, 50),
        (50, 50, 255),
        (255, 255, 50),
        (255, 50, 255),
        (50, 255, 255),
    ]

    def _snapshot_color(i: int, n: int) -> tuple[int, int, int]:
        cmap = plt.get_cmap("tab10")
        rgb = cmap((i % 10) / 10.0)[:3]
        return tuple(int(c * 255) for c in rgb)

    def rebuild_snapshots(n_snapshots: int) -> None:
        for handle in snapshot_state["handles"]:
            handle.remove()
        snapshot_state["handles"] = []

        indices = _snapshot_frame_indices(n_snapshots, n_frames)
        for i, frame_idx in enumerate(indices):
            color = _snapshot_color(i, len(indices))
            if show_body_frame:
                frame_handle = place_body_frame(
                    server,
                    f"/snapshots/frame_{i}",
                    pos[frame_idx],
                    attitude[frame_idx],
                    axes_length=attitude_axes_length,
                )
                snapshot_state["handles"].append(frame_handle)
            elif mesh_verts is not None and has_attitude:
                mesh_handle = server.scene.add_mesh_simple(
                    f"/snapshots/vehicle_{i}",
                    vertices=mesh_verts,
                    faces=mesh_faces,
                    color=vehicle_mesh_color,
                    position=tuple(float(x) for x in pos[frame_idx]),
                    wxyz=tuple(float(x) for x in _normalize_wxyz(attitude[frame_idx])),
                )
                snapshot_state["handles"].append(mesh_handle)

            if show_viewcone and R_sb is not None:
                cone_handle = place_viewcone(
                    server,
                    f"/snapshots/viewcone_{i}",
                    pos[frame_idx],
                    attitude[frame_idx],
                    half_angle_x=half_angle_x,
                    half_angle_y=half_angle_y,
                    scale=viewcone_scale,
                    norm_type=norm_type,
                    R_sb=R_sb,
                    color=color,
                    opacity=0.45,
                )
                snapshot_state["handles"].append(cone_handle)

            if snapshot_builder is not None:
                snapshot_state["handles"].extend(snapshot_builder(server, i, int(frame_idx)))

            if show_targets and subject_trajs_scaled is not None:
                for sub_idx, traj in enumerate(subject_trajs_scaled):
                    kp_handle = server.scene.add_icosphere(
                        f"/snapshots/target_{i}/sub_{sub_idx}",
                        radius=target_radius / scene_scale,
                        color=_target_colors[sub_idx % len(_target_colors)],
                        position=np.asarray(traj[frame_idx], dtype=np.float32),
                    )
                    snapshot_state["handles"].append(kp_handle)

            if relative_vector and target_traj_scaled is not None:
                rel_handle = server.scene.add_line_segments(
                    f"/snapshots/relative_vector_{i}",
                    points=np.array(
                        [[pos[frame_idx], target_traj_scaled[frame_idx]]],
                        dtype=np.float32,
                    ),
                    colors=(50, 255, 50),
                    line_width=2.0,
                )
                snapshot_state["handles"].append(rel_handle)

    rebuild_snapshots(initial_n)

    with server.gui.add_folder(folder_name):
        count_slider = server.gui.add_slider(
            "Number of snapshots",
            min=1,
            max=max_snapshots,
            step=1,
            initial_value=float(initial_n),
        )
        trace_size_slider = None
        if logo_trace_point_size is not None and logo_trace_points is not None:
            trace_size_slider = server.gui.add_slider(
                "Logo trace point size",
                min=0.001,
                max=0.2,
                step=0.001,
                initial_value=float(logo_trace_point_size),
            )

    @count_slider.on_update
    def _(_) -> None:
        rebuild_snapshots(int(round(count_slider.value)))

    if trace_size_slider is not None:

        @trace_size_slider.on_update
        def _(_) -> None:
            rebuild_logo_trace(float(trace_size_slider.value))

    return server


def create_scp_animated_plotting_server(
    results: OptimizationResults,
    position_slice: slice | None = None,
    attitude_slice: slice | None = None,
    propagation_line_width: float = 2.0,
    show_attitudes: bool = True,
    attitude_stride: int = 3,
    attitude_axes_length: float = 1.5,
    node_point_size: float = 0.3,
    frame_duration_ms: int = 500,
    scene_scale: float = 1.0,
    cmap_name: str = "viridis",
    show_grid: bool = True,
) -> viser.ViserServer:
    """Create an animated visualization of SCP iteration convergence.

    This shows how the optimization nodes evolve across SCP iterations,
    allowing you to visualize the convergence process.

    Features:
    - Play/pause button for iteration animation
    - Previous/Next buttons to step through iterations
    - Iteration slider to scrub through convergence history
    - Speed control for playback
    - Node positions colored by iteration
    - Nonlinear propagation lines showing actual integrated trajectories
    - Ghost trails showing all previous iterations
    - Optional attitude frames at each node (for 6DOF problems)
    - Static obstacles/gates if present in results

    Args:
        results: Optimization results containing SCP iteration history (results.X).
        position_slice: Slice for extracting position from state vector.
            If None, auto-detected from results._states looking for "position".
        attitude_slice: Slice for extracting attitude quaternion from state vector.
            If None, auto-detected from results._states looking for "attitude".
        propagation_line_width: Width of propagation lines
        show_attitudes: If True and attitude data available, show body frames
        attitude_stride: Show attitude frame every N nodes (reduces clutter)
        attitude_axes_length: Length of attitude coordinate frame axes
        node_point_size: Size of node markers
        frame_duration_ms: Default milliseconds per iteration frame
        scene_scale: Divide all positions by this factor. Use >1 for large-scale
            trajectories (e.g., 100.0 for km-scale problems).
        cmap_name: Matplotlib colormap name for iteration coloring (default: "viridis")
        show_grid: Whether to show the grid (default True)

    Returns:
        ViserServer instance (animation runs in background thread)
    """
    # Get iteration history and convert to numpy (handles JAX arrays)
    X_history = [np.asarray(X) for X in results.X]
    n_iterations = len(X_history)

    if n_iterations == 0:
        raise ValueError("No SCP iteration history available in results.X")

    # Auto-detect slices from state metadata if not provided
    if position_slice is None or attitude_slice is None:
        states = getattr(results, "_states", [])
        for state in states:
            if position_slice is None and state.name.lower() == "position":
                position_slice = state._slice
            if attitude_slice is None and state.name.lower() == "attitude":
                attitude_slice = state._slice

    # Default position slice if still not found (assume first 3 states)
    if position_slice is None:
        position_slice = slice(0, 3)

    # Extract position history and apply scene scale
    positions = [X[:, position_slice] / scene_scale for X in X_history]

    # Extract attitude history if available
    attitudes = None
    if attitude_slice is not None:
        attitudes = [X[:, attitude_slice] for X in X_history]

    # Create server using final iteration's positions for grid sizing
    server = create_server(positions[-1], show_grid=show_grid)

    # Add static elements (gates, obstacles) if present
    if "vertices" in results:
        add_gates(server, results["vertices"])

    if "obstacles_centers" in results:
        add_ellipsoid_obstacles(
            server,
            centers=results["obstacles_centers"],
            radii=results.get("obstacles_radii", [np.ones(3)] * len(results["obstacles_centers"])),
            axes=results.get("obstacles_axes"),
        )

    # Collect update callbacks
    update_callbacks = []

    # Add ghost iterations (previous iterations)
    _, update_ghosts = add_scp_ghost_iterations(server, positions, cmap_name=cmap_name)
    update_callbacks.append(update_ghosts)

    # Add nonlinear propagation lines if discretization history is available
    if results.discretization_history:
        n_x = results.X[0].shape[1]
        n_u = results.U[0].shape[1]

        propagations = extract_propagation_positions(
            results.discretization_history,
            n_x=n_x,
            n_u=n_u,
            position_slice=position_slice,
            scene_scale=scene_scale,
        )

        _, update_propagation = add_scp_propagation_lines(
            server,
            propagations,
            line_width=propagation_line_width,
            cmap_name=cmap_name,
        )
        update_callbacks.append(update_propagation)

    # Add main iteration nodes
    _, update_nodes = add_scp_iteration_nodes(
        server,
        positions,
        point_size=node_point_size,
        cmap_name=cmap_name,
    )
    update_callbacks.append(update_nodes)

    # Add attitude frames if available and enabled
    if show_attitudes and attitudes is not None:
        _, update_attitudes = add_scp_iteration_attitudes(
            server,
            positions,
            attitudes,
            axes_length=attitude_axes_length,
            stride=attitude_stride,
        )
        update_callbacks.append(update_attitudes)

    # Add SCP animation controls
    add_scp_animation_controls(
        server,
        n_iterations,
        update_callbacks,
        frame_duration_ms=frame_duration_ms,
    )

    return server


def create_pdg_animated_plotting_server(
    results: OptimizationResults,
    show_ghost_trajectory: bool = True,
    loop_animation: bool = True,
    thrust_key: str = "thrust",
    thrust_scale: float = 0.0001,
    thrust_vector_scale: float = 1.0,
    show_glideslope: bool = True,
    glideslope_angle_deg: float | None = None,
    glideslope_height: float | None = None,
    marker_radius: float = 0.3,
    trail_point_size: float = 0.15,
    ghost_point_size: float = 0.05,
    scene_scale: float = 100.0,
) -> viser.ViserServer:
    """Create an animated visualization for Powered Descent Guidance problems.

    This is specialized for rocket landing trajectories with:
    - 3D position and velocity
    - Thrust vector visualization
    - Glideslope constraint cone

    All positions are divided by scene_scale to bring large-scale trajectories
    (e.g., 2000m) into a range that viser handles well (~20m).

    Args:
        results: Optimization result dictionary containing trajectory data.
            Expected keys:
            - trajectory["position"]: 3D position (N, 3)
            - trajectory["velocity"]: 3D velocity (N, 3)
            - trajectory[thrust_key]: Thrust vector (N, 3)
            - glideslope_angle_deg: Glideslope angle in degrees (optional, for cone)
        show_ghost_trajectory: If True, show faint full trajectory
        loop_animation: If True, loop animation when it reaches the end
        thrust_key: Key for thrust data in trajectory dict
        thrust_scale: Converts thrust magnitude (Newtons) to scene units.
            E.g., 0.0001 means 10000N becomes 1 scene unit.
        thrust_vector_scale: Additional multiplier for thrust vector length.
        show_glideslope: If True, show glideslope constraint cone
        glideslope_angle_deg: Glideslope angle in degrees. If None, uses value from
            results["glideslope_angle_deg"] or defaults to 86.0.
        glideslope_height: Height of glideslope cone visualization (in original units).
            If None, uses 10% of the initial altitude.
        marker_radius: Radius of position marker (in scaled scene units).
        trail_point_size: Size of trail points.
        ghost_point_size: Size of ghost trajectory points.
        scene_scale: Divide all positions by this factor. Default 100.0 brings
            km-scale trajectories into a ~10-20m range for viser.

    Returns:
        ViserServer instance (animation runs in background thread)
    """
    # Extract and scale position data
    pos = results.trajectory["position"] / scene_scale
    vel = results.trajectory["velocity"]
    thrust = results.trajectory.get(thrust_key)
    traj_time = results.trajectory["time"]

    # Combined thrust scale factor
    combined_thrust_scale = thrust_scale * thrust_vector_scale

    # Get glideslope parameters
    if glideslope_angle_deg is None:
        glideslope_angle_deg = results.get("glideslope_angle_deg", 86.0)

    if glideslope_height is None:
        # Default to 20% of initial altitude - just show near landing point
        glideslope_height = float(results.trajectory["position"][0, 2]) * 0.1
    glideslope_height_scaled = glideslope_height / scene_scale

    # Precompute colors (fallback when velocity key is missing)
    colors = compute_velocity_colors(vel, fallback_length=len(pos))

    # Create server
    server = create_server(pos)

    # Add static elements
    if show_glideslope:
        add_glideslope_cone(
            server,
            apex=(0, 0, 0),
            height=glideslope_height_scaled,
            glideslope_angle_deg=glideslope_angle_deg,
        )

    if show_ghost_trajectory:
        add_ghost_trajectory(server, pos, colors, point_size=ghost_point_size)

    # Add animated elements
    update_callbacks = []

    _, update_trail = add_animated_trail(server, pos, colors, point_size=trail_point_size)
    update_callbacks.append(update_trail)

    _, update_marker = add_position_marker(server, pos, radius=marker_radius)
    update_callbacks.append(update_marker)

    # Thrust vector (no attitude for 3DoF, thrust is in world frame)
    _, update_thrust = add_thrust_vector(
        server, pos, thrust, attitude=None, scale=combined_thrust_scale
    )
    update_callbacks.append(update_thrust)

    # Add animation controls
    add_animation_controls(server, traj_time, update_callbacks, loop=loop_animation)

    return server


# =============================================================================
# Real-time Visualization Utilities
# =============================================================================
# These utilities are used by the real-time examples in examples/realtime/.
# They extract common patterns for metrics display, trajectory parsing, etc.


def format_metrics_markdown(results: dict) -> str:
    """Format optimization metrics as a markdown string for viser GUI display.

    This provides a consistent format for displaying SCP iteration metrics
    in real-time visualization GUIs.

    Args:
        results: Dictionary containing optimization metrics with keys:
            - iter: Iteration number
            - J_tr: Trust region penalty
            - J_vb: Virtual buffer penalty
            - J_vc: Virtual control penalty
            - cost: Objective value
            - dis_time: Discretization time in ms
            - solve_time: Solve time in ms
            - prob_stat: Problem status string

    Returns:
        Markdown-formatted string for display in viser GUI.

    Example:
        >>> results = {"iter": 5, "J_tr": 1.2e-3, "cost": 42.5, ...}
        >>> metrics_text.content = format_metrics_markdown(results)
    """
    iter_num = results.get("iter", 0)
    j_tr = results.get("J_tr", 0.0)
    j_vb = results.get("J_vb", 0.0)
    j_vc = results.get("J_vc", 0.0)
    cost = results.get("cost", 0.0)
    dis_time = results.get("dis_time", 0.0)
    solve_time = results.get("solve_time", 0.0)
    status = results.get("prob_stat", "--")

    return f"""**Iteration:** {iter_num}
**J_tr:** {j_tr:.2E}
**J_vb:** {j_vb:.2E}
**J_vc:** {j_vc:.2E}
**Objective:** {cost:.2E}
**Dis Time:** {dis_time:.1f}ms
**Solve Time:** {solve_time:.1f}ms
**Status:** {status}"""


def extract_multishoot_trajectory(
    V_multi_shoot: np.ndarray,
    n_x: int,
    n_u: int,
    position_slice: slice = slice(0, 3),
    velocity_slice: slice | None = slice(3, 6),
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract position and velocity trajectories from multi-shoot data.

    Uses time-ordered chronological stitching with deduplicated segment boundaries.

    Args:
        V_multi_shoot: Multi-shoot data array of shape (n_segments * segment_size, n_substeps)
        n_x: Number of states
        n_u: Number of controls
        position_slice: Slice for extracting position from state (default: first 3)
        velocity_slice: Slice for extracting velocity from state (default: states 3-6).
            Set to None to skip velocity extraction.

    Returns:
        Tuple of (positions, velocities) as float32 arrays.
        positions: Shape (n_total_points, 3)
        velocities: Shape (n_total_points, 3) or None if velocity_slice is None
    """
    from openscvx.algorithms.multishot import unpack_multishot_V

    n_segments = V_multi_shoot.shape[0] // (n_x + n_x * n_x + 2 * n_x * n_u)
    placeholder_t = np.linspace(0.0, 1.0, n_segments + 1)
    prop = unpack_multishot_V(
        V_multi_shoot, n_x=n_x, n_u=n_u, t_nodes=placeholder_t, states=()
    )
    positions, _ = prop.slice_states(position_slice)
    positions = positions.astype(np.float32)
    if velocity_slice is not None:
        velocities, _ = prop.slice_states(velocity_slice)
        velocities = velocities.astype(np.float32)
    else:
        velocities = None
    return positions, velocities


def get_print_queue_data(optimization_problem) -> dict:
    """Safely extract data from optimization problem's print queue.

    The print queue contains timing and status information emitted during
    optimization. This function safely extracts that data with sensible
    defaults if the queue is empty or unavailable.

    Args:
        optimization_problem: OpenSCvx Problem instance with optional print_queue attribute

    Returns:
        Dictionary with keys: dis_time, prob_stat, cost
        Returns default values if queue is empty or unavailable.
    """
    defaults = {"dis_time": 0.0, "prob_stat": "--", "cost": 0.0}

    try:
        if (
            hasattr(optimization_problem, "print_queue")
            and not optimization_problem.print_queue.empty()
        ):
            emitted_data = optimization_problem.print_queue.get_nowait()
            return {
                "dis_time": emitted_data.get("dis_time", 0.0),
                "prob_stat": emitted_data.get("prob_stat", "--"),
                "cost": emitted_data.get("cost", 0.0),
            }
    except Exception:
        pass

    return defaults


def build_scp_step_results(step_result: dict, solve_time_ms: float) -> dict:
    """Build a results dictionary from an SCP step result.

    Extracts the standard metrics from an optimization step result and
    combines them with timing information.

    Args:
        step_result: Dictionary returned by optimization_problem.step()
        solve_time_ms: Total solve time in milliseconds

    Returns:
        Dictionary with keys: iter, J_tr, J_vb, J_vc, converged, solve_time
    """
    return {
        "iter": step_result["scp_k"] - 1,
        "J_tr": step_result["scp_J_tr"],
        "J_vb": step_result["scp_J_vb"],
        "J_vc": step_result["scp_J_vc"],
        "converged": step_result["converged"],
        "solve_time": solve_time_ms,
    }


def compute_velocity_colors_realtime(vel: np.ndarray, cmap) -> np.ndarray:
    """Compute RGB colors based on velocity magnitude (pyplot-free version).

    This version accepts a pre-loaded colormap to avoid importing matplotlib.pyplot,
    which can cause issues with viser's web visualization in real-time examples.

    Args:
        vel: Velocity array of shape (N, 3)
        cmap: Pre-loaded matplotlib colormap (e.g., matplotlib.colormaps["viridis"])

    Returns:
        Array of RGB colors with shape (N, 3), dtype uint8, values in [0, 255]

    Example:
        >>> import matplotlib
        >>> _viridis = matplotlib.colormaps["viridis"]  # Load at module level
        >>> colors = compute_velocity_colors_realtime(velocities, _viridis)
    """
    vel_norms = np.linalg.norm(vel, axis=1)
    vel_range = vel_norms.max() - vel_norms.min()
    if vel_range < 1e-8:
        vel_normalized = np.zeros_like(vel_norms)
    else:
        vel_normalized = (vel_norms - vel_norms.min()) / vel_range

    colors = np.array(
        [[int(c * 255) for c in cmap(v)[:3]] for v in vel_normalized],
        dtype=np.uint8,
    )
    return colors


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


def create_hohmann_transfer_server(
    results: OptimizationResults,
    *,
    r1: float,
    r2: float,
    position_key: str = "position",
    velocity_key: str = "velocity",
    loop_animation: bool = True,
    show_grid: bool = False,
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
