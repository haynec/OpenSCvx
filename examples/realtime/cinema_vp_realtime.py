"""Interactive real-time visualization for cinematic viewpoint planning using Viser.

This module provides a web-based GUI for interactively solving and visualizing
the cinematic viewpoint planning trajectory optimization problem in real-time.

Run this script and open the displayed URL in your browser.
"""

import os
import sys
import threading
import time

import matplotlib
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R

# Get viridis colormap without pyplot (avoids potential backend issues)
_viridis_cmap = matplotlib.colormaps["viridis"]

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.plotting_viser import (
    build_scp_step_results,
    compute_velocity_colors_realtime,
    extract_multishoot_trajectory,
    format_metrics_markdown,
    get_print_queue_data,
)
from examples.realtime.base_problems.cinema_vp_realtime_base import (
    kp_pose,
    plotting_dict,
    problem,
)

# Initialize the problem
problem.initialize()


def _generate_viewcone_vertices(
    half_angle_x: float,
    half_angle_y: float | None,
    scale: float,
    norm_type: float | str,
    n_segments: int = 32,
) -> np.ndarray:
    """Generate viewcone vertices in sensor frame (apex at origin, looking along +Z)."""
    if half_angle_y is None:
        half_angle_y = half_angle_x

    # Compute extent at z=scale
    tan_x = np.tan(half_angle_x)
    tan_y = np.tan(half_angle_y)

    # Generate points around the cone base depending on norm type
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)

    # Convert norm_type to numeric value
    if norm_type == "inf" or norm_type == np.inf:
        p = np.inf
    else:
        p = float(norm_type)

    # Generate unit vectors on the p-norm ball
    if p == np.inf:
        # Square cross-section
        # Use parametric form for a square
        t = theta / (2 * np.pi) * 4  # 0 to 4
        unit_x = np.zeros_like(t)
        unit_y = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti < 1:
                unit_x[i] = 1.0
                unit_y[i] = ti * 2 - 1
            elif ti < 2:
                unit_x[i] = 1 - (ti - 1) * 2
                unit_y[i] = 1.0
            elif ti < 3:
                unit_x[i] = -1.0
                unit_y[i] = 1 - (ti - 2) * 2
            else:
                unit_x[i] = -1 + (ti - 3) * 2
                unit_y[i] = -1.0
    else:
        # p-norm ball: |x|^p + |y|^p = 1
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        # Superellipse parametric form
        r = (np.abs(cos_t) ** p + np.abs(sin_t) ** p) ** (-1.0 / p)
        unit_x = r * cos_t
        unit_y = r * sin_t

    # Scale by tan values and cone depth
    x = unit_x * tan_x * scale
    y = unit_y * tan_y * scale
    z = np.full_like(x, scale)

    # Apex at origin
    apex = np.array([[0.0, 0.0, 0.0]])
    base_points = np.stack([x, y, z], axis=1)

    return np.vstack([apex, base_points])


def _generate_viewcone_faces(n_base_verts: int) -> np.ndarray:
    """Generate triangle faces for viewcone (fan from apex)."""
    faces = []
    for i in range(n_base_verts):
        next_i = (i + 1) % n_base_verts
        # Face: apex (0), current base vertex (i+1), next base vertex (next_i+1)
        faces.append([0, i + 1, next_i + 1])
    return np.array(faces, dtype=np.uint32)


def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    r = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x, y, z, w]
    return r.as_matrix()


def create_realtime_server(
    optimization_problem,
    keypoint_param,
    plot_dict: dict,
) -> viser.ViserServer:
    """Create a viser server for real-time cinematic viewpoint trajectory optimization.

    Args:
        optimization_problem: The OpenSCvx Problem instance
        keypoint_param: The keypoint position parameter object
        plot_dict: Dictionary containing visualization parameters (R_sb, alpha_x, etc.)

    Returns:
        ViserServer instance
    """
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    # Extract plotting parameters
    alpha_x = plot_dict.get("alpha_x", 6.0)
    alpha_y = plot_dict.get("alpha_y", 8.0)
    R_sb = np.array(plot_dict.get("R_sb", np.eye(3)))
    norm_type = plot_dict.get("norm_type", "inf")

    # Compute half-angles in radians
    half_angle_x = np.pi / alpha_x
    half_angle_y = np.pi / alpha_y

    # =========================================================================
    # Scene Setup
    # =========================================================================

    # Grid
    server.scene.add_grid(
        "/grid",
        width=50,
        height=50,
        position=(0.0, 0.0, 0.0),
    )

    # Origin frame
    server.scene.add_frame(
        "/origin",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=1.0,
    )

    # Trajectory point cloud (initially empty)
    trajectory_handle = server.scene.add_point_cloud(
        "/trajectory",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=(255, 255, 0),
        point_size=0.2,
    )

    # Keypoint marker (red sphere)
    initial_kp = keypoint_param.value
    keypoint_handle = server.scene.add_icosphere(
        "/keypoint",
        radius=0.3,
        color=(255, 0, 0),
        position=tuple(initial_kp),
    )

    # Line-of-sight visualization (line from drone to keypoint)
    los_handle = server.scene.add_line_segments(
        "/line_of_sight",
        points=np.zeros((1, 2, 3), dtype=np.float32),
        colors=(255, 255, 0),  # Yellow
        line_width=2.0,
    )

    # Drone body frame axes (will be updated with attitude)
    axis_handles = []
    axis_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB for XYZ
    for i, color in enumerate(axis_colors):
        handle = server.scene.add_line_segments(
            f"/drone_axes/axis_{i}",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=color,
            line_width=3.0,
        )
        axis_handles.append(handle)

    # View cone mesh (initially invisible, will be created on first update)
    viewcone_handle = {"mesh": None, "base_vertices": None}

    # Pre-compute viewcone base geometry in sensor frame
    viewcone_scale = 10.0
    viewcone_base_vertices = _generate_viewcone_vertices(
        half_angle_x, half_angle_y, viewcone_scale, norm_type, n_segments=32
    )
    viewcone_faces = _generate_viewcone_faces(len(viewcone_base_vertices) - 1)
    viewcone_handle["base_vertices"] = viewcone_base_vertices

    # Create initial viewcone mesh
    cmap = matplotlib.colormaps["viridis"]
    rgb = cmap(0.4)[:3]
    viewcone_color = tuple(int(c * 255) for c in rgb)

    viewcone_handle["mesh"] = server.scene.add_mesh_simple(
        "/viewcone",
        vertices=viewcone_base_vertices.astype(np.float32),
        faces=viewcone_faces,
        color=viewcone_color,
        wireframe=False,
        opacity=0.4,
    )

    # Keypoint draggable transform control
    keypoint_drag_handle = server.scene.add_transform_controls(
        "/keypoint_drag",
        position=tuple(initial_kp),
        scale=2.0,
        disable_rotations=True,
        visible=True,
    )

    # =========================================================================
    # Shared State
    # =========================================================================

    state = {
        "running": True,
        "reset_requested": False,
    }

    # =========================================================================
    # GUI Controls
    # =========================================================================

    # --- Optimization Metrics ---
    with server.gui.add_folder("Optimization Metrics"):
        metrics_text = server.gui.add_markdown(
            """**Iteration:** 0
**J_tr:** 0.00E+00
**J_vb:** 0.00E+00
**J_vc:** 0.00E+00
**Objective:** 0.00E+00
**Dis Time:** 0.0ms
**Solve Time:** 0.0ms
**Status:** --"""
        )

    # --- Optimization Weights ---
    with server.gui.add_folder("Optimization Weights"):
        lam_cost_input = server.gui.add_number(
            "lambda_cost",
            initial_value=optimization_problem.settings.scp.lam_cost,
            min=1e-6,
            max=1e6,
            step=0.01,
        )
        lam_tr_input = server.gui.add_number(
            "lambda_tr (lam_prox)",
            initial_value=optimization_problem.settings.scp.lam_prox,
            min=1e-6,
            max=1e6,
            step=0.1,
        )

        @lam_cost_input.on_update
        def _(_) -> None:
            optimization_problem.settings.scp.lam_cost = lam_cost_input.value

        @lam_tr_input.on_update
        def _(_) -> None:
            optimization_problem.settings.scp.lam_prox = lam_tr_input.value

    # --- Problem Control ---
    with server.gui.add_folder("Problem Control"):
        reset_button = server.gui.add_button("Reset Problem")

        @reset_button.on_click
        def _(_) -> None:
            state["reset_requested"] = True
            print("Problem reset requested")

    # --- Keypoint Position Controls ---
    with server.gui.add_folder("Keypoint Position (Line-of-Sight Target)"):
        server.gui.add_markdown("*Drag the control in 3D view or use inputs below*")

        kp_vector_input = server.gui.add_vector3(
            "Position",
            initial_value=tuple(initial_kp),
            step=0.5,
        )

        reset_kp_button = server.gui.add_button("Reset Keypoint")

        @reset_kp_button.on_click
        def _(_) -> None:
            original = plot_dict.get("init_poses", np.array([13.0, 0.0, 2.0]))
            kp_vector_input.value = tuple(original)
            keypoint_param.value = np.array(original)
            optimization_problem.parameters["kp_pose"] = np.array(original)
            keypoint_drag_handle.position = tuple(original)
            keypoint_handle.position = tuple(original)
            print("Keypoint reset to initial position")

        # Callback for GUI vector3 input -> update params and scene objects
        @kp_vector_input.on_update
        def _(_) -> None:
            new_pos = np.array(kp_vector_input.value)
            keypoint_param.value = new_pos
            optimization_problem.parameters["kp_pose"] = new_pos
            keypoint_drag_handle.position = tuple(new_pos)
            keypoint_handle.position = tuple(new_pos)

    # Wire up drag handle callback
    @keypoint_drag_handle.on_update
    def _(_) -> None:
        new_pos = np.array(keypoint_drag_handle.position)
        keypoint_param.value = new_pos
        optimization_problem.parameters["kp_pose"] = new_pos
        kp_vector_input.value = tuple(new_pos)
        keypoint_handle.position = tuple(new_pos)

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def update_metrics(results: dict) -> None:
        """Update the metrics markdown display."""
        metrics_text.content = format_metrics_markdown(results)

    def update_trajectory_and_drone(V_multi_shoot: np.ndarray, x_traj: np.ndarray) -> None:
        """Update trajectory visualization and drone pose."""
        try:
            n_x = optimization_problem.settings.sim.n_states
            n_u = optimization_problem.settings.sim.n_controls

            positions, velocities = extract_multishoot_trajectory(V_multi_shoot, n_x, n_u)

            if len(positions) > 0:
                colors = compute_velocity_colors_realtime(velocities, _viridis_cmap)
                trajectory_handle.points = positions
                trajectory_handle.colors = colors

                # Update line-of-sight (from last position to keypoint)
                current_kp = optimization_problem.parameters["kp_pose"]
                last_pos = positions[-1]
                los_points = np.array([[last_pos, current_kp]], dtype=np.float32)
                los_handle.points = los_points

            # Update drone axes and viewcone using x_traj (state trajectory)
            if x_traj is not None and len(x_traj) > 0 and x_traj.shape[1] >= 10:
                # Get last position and attitude
                last_pos = x_traj[-1, :3]
                att = x_traj[-1, 6:10]  # Quaternion [qw, qx, qy, qz]

                # Convert quaternion to rotation matrix
                rotmat = _quaternion_to_rotation_matrix(att)

                # Draw body frame axes (x=red, y=green, z=blue)
                axes_length = 2.0
                axes = axes_length * np.eye(3)
                axes_rot = rotmat @ axes

                for k in range(3):
                    axis_pts = np.array([[last_pos, last_pos + axes_rot[:, k]]], dtype=np.float32)
                    axis_handles[k].points = axis_pts

                # Update viewcone mesh
                if (
                    viewcone_handle["mesh"] is not None
                    and viewcone_handle["base_vertices"] is not None
                ):
                    # Sensor-to-body rotation
                    R_sensor_to_body = R_sb.T

                    # Full transform: sensor -> body -> world
                    R_sensor_to_world = rotmat @ R_sensor_to_body

                    # Transform vertices and translate to position
                    base_verts = viewcone_handle["base_vertices"]
                    world_vertices = (R_sensor_to_world @ base_verts.T).T + last_pos
                    viewcone_handle["mesh"].vertices = world_vertices.astype(np.float32)

        except Exception as e:
            print(f"Trajectory update error: {e}")

    # =========================================================================
    # Optimization Worker
    # =========================================================================

    def optimization_loop() -> None:
        """Background thread running continuous optimization."""
        iteration = 0

        while state["running"]:
            try:
                # Check for reset request
                if state["reset_requested"]:
                    optimization_problem.reset()
                    state["reset_requested"] = False
                    iteration = 0
                    print("Problem reset to initial conditions")

                # Run one SCP step
                start_time = time.time()
                step_result = optimization_problem.step()
                solve_time_ms = (time.time() - start_time) * 1000

                # Build results dict
                results = build_scp_step_results(step_result, solve_time_ms)
                results.update(get_print_queue_data(optimization_problem))

                # Update visualizations (viser is thread-safe)
                update_metrics(results)

                # Update trajectory from V_history
                if optimization_problem.state.V_history:
                    V_multi_shoot = np.array(optimization_problem.state.V_history[-1])
                    x_traj = np.array(optimization_problem.state.x)
                    update_trajectory_and_drone(V_multi_shoot, x_traj)

                iteration += 1
                time.sleep(0.05)  # Small delay to avoid overwhelming

            except Exception as e:
                print(f"Optimization error: {e}")
                time.sleep(1.0)

    # Start optimization in background thread
    opt_thread = threading.Thread(target=optimization_loop, daemon=True)
    opt_thread.start()

    return server


if __name__ == "__main__":
    print("Creating viser server for Cinema Viewpoint Planning Real-time Optimization...")
    print("Open the URL shown below in your browser\n")

    server = create_realtime_server(problem, kp_pose, plotting_dict)
    server.sleep_forever()
