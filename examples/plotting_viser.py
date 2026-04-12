"""Viser-based trajectory visualization templates.

This module provides convenience functions for common visualization patterns.
These are templates meant to be copied and customized for specific problems.

For the composable primitives, see openscvx.plotting.animation.
For real-time examples, see examples/realtime/*.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import viser

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
    add_thrust_vector,
    add_viewcone,
    compute_velocity_colors,
    create_server,
    extract_propagation_positions,
)

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
    attitude_key: str = "attitude",
    attitude_axes_length: float = 2.0,
    show_viewcone: bool = True,
    viewcone_scale: float = 10.0,
    target_radius: float = 1.0,
    show_control_plot: str | None = None,
    show_control_norm_plot: str | None = None,
    trail_point_size: float = 0.15,
    show_grid: bool = True,
    scene_scale: float = 1.0,
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
    - Body frame attitude visualization (if attitude data available, for 6DOF)
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
        thrust_scale: Scale factor for thrust vector visualization
        attitude_key: Key for attitude quaternion data (default: "attitude")
        attitude_axes_length: Length of body frame axes
        show_viewcone: If True and R_sb is in results, show camera viewcone
        viewcone_scale: Size/depth of viewcone mesh
        target_radius: Radius of target marker spheres
        show_control_plot: If provided with a control name, displays component plot
            showing each control component vs time with animated markers
        show_control_norm_plot: If provided with a control name, displays norm plot
            showing ‖control‖₂ vs time with animated marker
        show_grid: Whether to show the grid (default True)
        scene_scale: Divide all positions (and lengths) by this factor. Use >1 for
            large-scale trajectories (e.g., 100.0 for km-scale problems).
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

    # Precompute colors
    colors = compute_velocity_colors(vel)

    # Create server
    server = create_server(pos, show_grid=show_grid)

    def _ray_plane_intersection(
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        plane_z: float,
    ) -> np.ndarray | None:
        """Intersection of a ray with a horizontal plane z=plane_z."""
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
        # viser expects colors shape (N, 2, 3) for N segments, 2 endpoints each, 3 RGB per point
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
                    plane_pt = (
                        np.asarray(logo_plane_point, dtype=np.float32).reshape(3) / scene_scale
                    )
                    plane_n = np.asarray(logo_plane_normal, dtype=np.float32).reshape(3)
                    plane_n = plane_n / (np.linalg.norm(plane_n) + 1e-10)
                elif plane_z is not None:
                    plane_pt = None
                    plane_n = None
                else:
                    plane_pt = None
                    plane_n = None

                if plane_z is not None or use_logo_plane:

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

                    boresight_body = results.get("boresight_body", np.array([1.0, 0.0, 0.0]))
                    boresight_body = np.asarray(boresight_body, dtype=np.float32)
                    if boresight_body.shape[0] >= 3:
                        boresight_body = boresight_body[:3]
                    boresight_body = boresight_body / np.linalg.norm(boresight_body)

                    R0 = q_to_R_wxyz(attitude[0])
                    boresight_world_0 = R0 @ boresight_body
                    if use_logo_plane:
                        boresight_intersection_0 = _ray_plane_intersection_general(
                            pos[0], boresight_world_0, plane_pt, plane_n
                        )
                    else:
                        boresight_intersection_0 = _ray_plane_intersection(
                            pos[0], boresight_world_0, plane_z
                        )

                    if boresight_intersection_0 is not None:
                        n_frames = len(pos)
                        boresight_intersection_points = []
                        for i in range(n_frames):
                            Rk = q_to_R_wxyz(attitude[i])
                            boresight_world = Rk @ boresight_body
                            if use_logo_plane:
                                intersection = _ray_plane_intersection_general(
                                    pos[i], boresight_world, plane_pt, plane_n
                                )
                            else:
                                intersection = _ray_plane_intersection(
                                    pos[i], boresight_world, plane_z
                                )
                            if intersection is not None:
                                boresight_intersection_points.append(intersection)
                            else:
                                boresight_intersection_points.append(
                                    boresight_intersection_0
                                )  # Fallback

                        boresight_intersection_points = np.array(
                            boresight_intersection_points, dtype=np.float32
                        )

                        # Draw extended boresight to plane intersection
                        boresight_handle = server.scene.add_line_segments(
                            "/boresight_extended",
                            points=np.array([[pos[0], boresight_intersection_0]], dtype=np.float32),
                            colors=(255, 0, 0),  # Red for boresight
                            line_width=3.0,
                        )

                        # Add intersection point marker
                        intersection_handle = server.scene.add_icosphere(
                            "/boresight_intersection",
                            radius=0.06,
                            color=(255, 100, 100),
                            position=boresight_intersection_0,
                        )

                        # Add trail for boresight intersection point
                        boresight_trail_handle = server.scene.add_line_segments(
                            "/boresight_intersection_trail",
                            points=np.array([], dtype=np.float32).reshape(0, 2, 3),
                            colors=(200, 50, 50),
                            line_width=2.0,
                        )

                        def update_boresight(frame_idx: int) -> None:
                            idx = min(frame_idx, len(boresight_intersection_points) - 1)
                            boresight_handle.points = np.array(
                                [[pos[idx], boresight_intersection_points[idx]]], dtype=np.float32
                            )
                            intersection_handle.position = boresight_intersection_points[idx]

                            # Update trail to show up to current frame
                            if idx > 0:
                                trail_segments = np.array(
                                    [
                                        [
                                            boresight_intersection_points[i],
                                            boresight_intersection_points[i + 1],
                                        ]
                                        for i in range(idx)
                                    ],
                                    dtype=np.float32,
                                )
                                boresight_trail_handle.points = trail_segments
                            else:
                                boresight_trail_handle.points = np.array(
                                    [], dtype=np.float32
                                ).reshape(0, 2, 3)

                        update_callbacks.append(update_boresight)
                    else:
                        # Fallback to fixed multiplier if intersection fails
                        boresight_multiplier = 3.0
                        _, update_axes = _add_attitude_axes_lines(
                            "/body_axes",
                            pos,
                            attitude,
                            axes_length=attitude_axes_length,
                            boresight_multiplier=boresight_multiplier,
                        )
                        update_callbacks.append(update_axes)
                else:
                    # Fallback to fixed multiplier if plane_z not available
                    boresight_multiplier = 3.0
                    _, update_axes = _add_attitude_axes_lines(
                        "/body_axes",
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
        else:
            _, update_marker = add_position_marker(server, pos)
            update_callbacks.append(update_marker)

        _, update_thrust = add_thrust_vector(
            server, pos, thrust, attitude=attitude, scale=thrust_scale
        )
        update_callbacks.append(update_thrust)  # Will be filtered out if None

        # Add viewcone mesh if R_sb is available and enabled
        if show_viewcone and R_sb is not None and attitude is not None:
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
                        colors=(0, 255, 255),  # Cyan: what the drone traced on the plane
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

                # Add trail for intersection point (grows with animation)
                intersection_trail_handle = server.scene.add_line_segments(
                    "/relative_vector_intersection_trail",
                    points=np.array([], dtype=np.float32).reshape(0, 2, 3),
                    colors=(50, 200, 50),
                    line_width=2.0,
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
                            bi = _ray_plane_intersection(pos[i], bw, plane_z)
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
                    if idx > 0:
                        trail_segments = np.array(
                            [
                                [intersection_points[i], intersection_points[i + 1]]
                                for i in range(idx)
                            ],
                            dtype=np.float32,
                        )
                        intersection_trail_handle.points = trail_segments
                    else:
                        intersection_trail_handle.points = np.array([], dtype=np.float32).reshape(
                            0, 2, 3
                        )
                    if plane_segment_handle is not None and idx < len(boresight_pts):
                        plane_segment_handle.points = np.array(
                            [[boresight_pts[idx], intersection_points[idx]]], dtype=np.float32
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
        raise ValueError(
            f"controls must be 'gui' or 'manual', got {controls!r}"
        )


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

    The multi-shoot format stores propagation segments with state, STM, and
    sensitivity data packed together. This function extracts the state
    components from each segment.

    Args:
        V_multi_shoot: Multi-shoot data array of shape (n_segments * segment_size, n_nodes)
        n_x: Number of states
        n_u: Number of controls
        position_slice: Slice for extracting position from state (default: first 3)
        velocity_slice: Slice for extracting velocity from state (default: states 3-6).
            Set to None to skip velocity extraction.

    Returns:
        Tuple of (positions, velocities) as float32 arrays.
        positions: Shape (n_total_points, 3)
        velocities: Shape (n_total_points, 3) or None if velocity_slice is None

    Note:
        The segment size is computed as: n_x + n_x² + 2*n_x*n_u
        This accounts for: state (n_x) + STM (n_x²) + sensitivities (2*n_x*n_u)
    """
    # Segment size: state + STM + control sensitivities
    segment_size = n_x + n_x * n_x + 2 * n_x * n_u

    all_pos_segments = []
    all_vel_segments = [] if velocity_slice is not None else None

    for i_node in range(V_multi_shoot.shape[1]):
        node_data = V_multi_shoot[:, i_node]
        segments_for_node = node_data.reshape(-1, segment_size)
        pos_segments = segments_for_node[:, position_slice]
        all_pos_segments.append(pos_segments)

        if velocity_slice is not None:
            vel_segments = segments_for_node[:, velocity_slice]
            all_vel_segments.append(vel_segments)

    positions = np.vstack(all_pos_segments).astype(np.float32)
    velocities = np.vstack(all_vel_segments).astype(np.float32) if all_vel_segments else None

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
