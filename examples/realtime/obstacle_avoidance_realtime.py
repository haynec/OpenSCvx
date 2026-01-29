"""Interactive real-time visualization for drone obstacle avoidance using Viser.

This module provides a web-based GUI for interactively solving and visualizing
the drone obstacle avoidance trajectory optimization problem in real-time.

Run this script and open the displayed URL in your browser.
"""

import os
import sys
import threading
import time

import matplotlib
import numpy as np
import viser

# Get viridis colormap without pyplot (avoids potential backend issues)
_viridis_cmap = matplotlib.colormaps["viridis"]

# Add grandparent directory to path to import examples
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
from examples.realtime.base_problems.obstacle_avoidance_realtime_base import (
    obstacle_centers,
    plotting_dict,
    problem,
)

# Initialize the problem
problem.initialize()


def create_realtime_server(
    optimization_problem,
    obstacle_params: list,
    initial_centers: list,
    initial_radii: list[float],
    n_obstacles: int = 3,
) -> viser.ViserServer:
    """Create a viser server for real-time obstacle avoidance visualization.

    Args:
        optimization_problem: The OpenSCvx Problem instance
        obstacle_params: List of obstacle center parameter objects
        initial_centers: List of initial obstacle center positions
        initial_radii: List of initial radii for each obstacle (scalar)
        n_obstacles: Number of obstacles

    Returns:
        ViserServer instance
    """
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    # =========================================================================
    # Scene Setup
    # =========================================================================

    # Grid
    server.scene.add_grid(
        "/grid",
        width=30,
        height=30,
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
        point_size=0.3,
    )

    # Obstacle icospheres
    obstacle_handles = []
    for i in range(n_obstacles):
        center = obstacle_params[i].value
        handle = server.scene.add_icosphere(
            f"/obstacles/sphere_{i}",
            radius=initial_radii[i],
            color=(100, 255, 100),  # Green
            position=tuple(center),
        )
        obstacle_handles.append(handle)

    # Obstacle transform controls (draggable gizmos)
    obstacle_drag_handles = []
    for i in range(n_obstacles):
        initial_pos = obstacle_params[i].value
        drag_handle = server.scene.add_transform_controls(
            f"/obstacles/drag_{i}",
            position=tuple(initial_pos),
            scale=1.5,
            disable_rotations=True,  # Obstacles only need translation
            visible=False,  # Hidden by default
        )
        obstacle_drag_handles.append(drag_handle)

    # Track currently selected obstacle
    selected_obstacle = {"index": None}

    def select_obstacle(obs_idx: int | None) -> None:
        """Select an obstacle and show its transform control, hiding others."""
        # Hide previously selected
        if selected_obstacle["index"] is not None:
            obstacle_drag_handles[selected_obstacle["index"]].visible = False
            obstacle_handles[selected_obstacle["index"]].color = (100, 255, 100)

        # Show newly selected
        if obs_idx is not None:
            obstacle_drag_handles[obs_idx].visible = True
            obstacle_handles[obs_idx].color = (150, 255, 150)  # Highlight
            selected_obstacle["index"] = obs_idx
        else:
            selected_obstacle["index"] = None

    # Add click handlers to obstacle icospheres
    def make_obstacle_click_handler(obs_idx: int):
        @obstacle_handles[obs_idx].on_click
        def _(_) -> None:
            # Toggle: click selected obstacle again to deselect
            if selected_obstacle["index"] == obs_idx:
                select_obstacle(None)
            else:
                select_obstacle(obs_idx)

        return _

    for i in range(n_obstacles):
        make_obstacle_click_handler(i)

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
            "λ_cost",
            initial_value=optimization_problem.settings.scp.lam_cost,
            min=1e-6,
            max=1e6,
            step=0.01,
        )
        lam_tr_input = server.gui.add_number(
            "λ_tr (lam_prox)",
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

    # --- Obstacle Controls ---
    obstacle_vector_inputs = []
    with server.gui.add_folder("Obstacle Positions", expand_by_default=False):
        server.gui.add_markdown("*Click an obstacle in 3D view to select and drag it*")

        reset_obstacles_button = server.gui.add_button("Reset All Obstacles")

        @reset_obstacles_button.on_click
        def _(_) -> None:
            # Deselect any selected obstacle
            select_obstacle(None)
            for i, vec_input in enumerate(obstacle_vector_inputs):
                original = initial_centers[i]
                vec_input.value = tuple(original)
                obstacle_params[i].value = np.array(original)
                param_name = f"obstacle_center_{i + 1}"
                optimization_problem.parameters[param_name] = np.array(original)
                # Update drag handle and obstacle positions
                obstacle_drag_handles[i].position = tuple(original)
                obstacle_handles[i].position = tuple(original)
            print("Obstacles reset to initial positions")

        for i in range(n_obstacles):
            initial_pos = obstacle_params[i].value
            vec_input = server.gui.add_vector3(
                f"Obstacle {i + 1}",
                initial_value=tuple(initial_pos),
                step=0.5,
            )
            obstacle_vector_inputs.append(vec_input)

            # Callback for GUI vector3 input -> update params and scene objects
            def make_obstacle_gui_callback(obs_idx: int, input_handle):
                @input_handle.on_update
                def _(_) -> None:
                    new_center = np.array(input_handle.value)
                    obstacle_params[obs_idx].value = new_center
                    param_name = f"obstacle_center_{obs_idx + 1}"
                    optimization_problem.parameters[param_name] = new_center
                    # Sync drag handle and obstacle positions
                    obstacle_drag_handles[obs_idx].position = tuple(new_center)
                    obstacle_handles[obs_idx].position = tuple(new_center)

                return _

            make_obstacle_gui_callback(i, vec_input)

    # Wire up drag handle callbacks (must be done after obstacle_vector_inputs is populated)
    def make_drag_callback(obs_idx: int, drag_handle):
        @drag_handle.on_update
        def _(_) -> None:
            new_center = np.array(drag_handle.position)
            obstacle_params[obs_idx].value = new_center
            param_name = f"obstacle_center_{obs_idx + 1}"
            optimization_problem.parameters[param_name] = new_center
            # Sync GUI vector3 input and obstacle position
            obstacle_vector_inputs[obs_idx].value = tuple(new_center)
            obstacle_handles[obs_idx].position = tuple(new_center)

        return _

    for i in range(n_obstacles):
        make_drag_callback(i, obstacle_drag_handles[i])

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def update_metrics(results: dict) -> None:
        """Update the metrics markdown display."""
        metrics_text.content = format_metrics_markdown(results)

    def update_trajectory(V_multi_shoot: np.ndarray) -> None:
        """Update the trajectory point cloud from multi-shoot data."""
        try:
            n_x = optimization_problem.settings.sim.n_states
            n_u = optimization_problem.settings.sim.n_controls

            positions, velocities = extract_multishoot_trajectory(V_multi_shoot, n_x, n_u)

            if len(positions) > 0:
                colors = compute_velocity_colors_realtime(velocities, _viridis_cmap)
                trajectory_handle.points = positions
                trajectory_handle.colors = colors

        except Exception as e:
            print(f"Trajectory update error: {e}")

    def update_obstacles() -> None:
        """Update obstacle visualizations based on current obstacle parameters."""
        for i, handle in enumerate(obstacle_handles):
            center = obstacle_params[i].value
            if center is not None:
                handle.position = tuple(center)

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
                update_obstacles()

                # Update trajectory from V_history
                if optimization_problem.state.V_history:
                    V_multi_shoot = np.array(optimization_problem.state.V_history[-1])
                    update_trajectory(V_multi_shoot)

                iteration += 1
                time.sleep(0.05)  # Small delay to avoid overwhelming

            except Exception as e:
                print(f"Optimization error: {e}")
                time.sleep(1.0)

    # Start optimization thread
    opt_thread = threading.Thread(target=optimization_loop, daemon=True)
    opt_thread.start()

    return server


if __name__ == "__main__":
    # Get initial obstacle centers from plotting_dict
    initial_obstacle_centers = [center.copy() for center in plotting_dict["obstacles_centers"]]

    # Compute radii from ellipsoid parameters
    initial_radii = []
    for rad in plotting_dict["obstacles_radii"]:
        effective_radii = 1.0 / np.array(rad)
        sphere_radius = float(np.min(effective_radii))  # Use smallest extent
        initial_radii.append(sphere_radius)

    # Create the viser server
    server = create_realtime_server(
        optimization_problem=problem,
        obstacle_params=obstacle_centers,
        initial_centers=initial_obstacle_centers,
        initial_radii=initial_radii,
        n_obstacles=3,
    )

    print("Viser server started. Open the URL in your browser.")
    server.sleep_forever()
