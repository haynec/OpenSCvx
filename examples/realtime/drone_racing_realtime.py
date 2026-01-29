"""Interactive real-time visualization for drone racing using Viser.

This module provides a web-based GUI for interactively solving and visualizing
the drone racing trajectory optimization problem in real-time.

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
from examples.realtime.base_problems.drone_racing_realtime_base import (
    gate_center_params,
    initial_gate_centers,
    problem,
)
from openscvx.utils import gen_vertices

# Initialize the problem
problem.initialize()


def create_realtime_server(
    optimization_problem,
    gate_params: list,
    initial_centers: list,
    n_gates: int = 10,
) -> viser.ViserServer:
    """Create a viser server for real-time trajectory optimization visualization.

    Args:
        optimization_problem: The OpenSCvx Problem instance
        gate_params: List of gate center parameter objects
        initial_centers: List of initial gate center positions
        n_gates: Number of gates

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
        width=200,
        height=100,
        position=(100.0, -50.0, 0.0),
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

    # Gate line segments (visual only)
    gate_handles = []
    for i in range(n_gates):
        handle = server.scene.add_line_segments(
            f"/gates/gate_{i}",
            points=np.zeros((4, 2, 3), dtype=np.float32),
            colors=(255, 165, 0),  # Orange, matching non-realtime plots
            line_width=3.0,
        )
        gate_handles.append(handle)

    # Clickable spheres at gate centers (for selection)
    gate_click_targets = []
    for i in range(n_gates):
        initial_pos = gate_params[i].value
        click_target = server.scene.add_icosphere(
            f"/gates/click_target_{i}",
            radius=0.5,
            color=(255, 165, 0),  # Orange, matching gate color
            position=tuple(initial_pos),
        )
        gate_click_targets.append(click_target)

    # Gate transform controls (draggable gizmos)
    gate_drag_handles = []
    for i in range(n_gates):
        initial_pos = gate_params[i].value
        drag_handle = server.scene.add_transform_controls(
            f"/gates/drag_{i}",
            position=tuple(initial_pos),
            scale=3.0,
            disable_rotations=True,  # Gates only need translation
            visible=False,  # Hidden by default
        )
        gate_drag_handles.append(drag_handle)

    # Track currently selected gate
    selected_gate = {"index": None}

    def select_gate(gate_idx: int | None) -> None:
        """Select a gate and show its transform control, hiding others."""
        # Hide previously selected
        if selected_gate["index"] is not None:
            gate_drag_handles[selected_gate["index"]].visible = False
            gate_handles[selected_gate["index"]].colors = (255, 165, 0)  # Orange
            gate_click_targets[selected_gate["index"]].color = (255, 165, 0)

        # Show newly selected
        if gate_idx is not None:
            gate_drag_handles[gate_idx].visible = True
            gate_handles[gate_idx].colors = (255, 200, 0)  # Yellow/orange highlight
            gate_click_targets[gate_idx].color = (255, 200, 0)  # Highlight click target
            selected_gate["index"] = gate_idx
        else:
            selected_gate["index"] = None

    # Add click handlers to clickable spheres
    def make_gate_click_handler(gate_idx: int):
        @gate_click_targets[gate_idx].on_click
        def _(_) -> None:
            # Toggle: click selected gate again to deselect
            if selected_gate["index"] == gate_idx:
                select_gate(None)
            else:
                select_gate(gate_idx)

        return _

    for i in range(n_gates):
        make_gate_click_handler(i)

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
            step=0.1,
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

    # --- Gate Controls ---
    gate_vector_inputs = []
    with server.gui.add_folder("Gate Positions", expand_by_default=False):
        server.gui.add_markdown("*Click a gate in 3D view to select and drag it*")

        reset_gates_button = server.gui.add_button("Reset All Gates")

        @reset_gates_button.on_click
        def _(_) -> None:
            # Deselect any selected gate
            select_gate(None)
            for i, vec_input in enumerate(gate_vector_inputs):
                original = initial_centers[i]
                vec_input.value = tuple(original)
                gate_params[i].value = np.array(original)
                optimization_problem.parameters[gate_params[i].name] = np.array(original)
                # Also update drag handle and click target positions
                gate_drag_handles[i].position = tuple(original)
                gate_click_targets[i].position = tuple(original)
            print("Gates reset to initial positions")

        for i in range(n_gates):
            initial_pos = gate_params[i].value
            vec_input = server.gui.add_vector3(
                f"Gate {i + 1}",
                initial_value=tuple(initial_pos),
                step=1.0,
            )
            gate_vector_inputs.append(vec_input)

            # Callback for GUI vector3 input -> update params and scene objects
            def make_gate_gui_callback(gate_idx: int, input_handle):
                @input_handle.on_update
                def _(_) -> None:
                    new_center = np.array(input_handle.value)
                    gate_params[gate_idx].value = new_center
                    optimization_problem.parameters[gate_params[gate_idx].name] = new_center
                    # Sync drag handle and click target positions
                    gate_drag_handles[gate_idx].position = tuple(new_center)
                    gate_click_targets[gate_idx].position = tuple(new_center)

                return _

            make_gate_gui_callback(i, vec_input)

    # Wire up drag handle callbacks (must be done after gate_vector_inputs is populated)
    def make_drag_callback(gate_idx: int, drag_handle):
        @drag_handle.on_update
        def _(_) -> None:
            new_center = np.array(drag_handle.position)
            gate_params[gate_idx].value = new_center
            optimization_problem.parameters[gate_params[gate_idx].name] = new_center
            # Sync GUI vector3 input and click target
            gate_vector_inputs[gate_idx].value = tuple(new_center)
            gate_click_targets[gate_idx].position = tuple(new_center)

        return _

    for i in range(n_gates):
        make_drag_callback(i, gate_drag_handles[i])

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

    def update_gates() -> None:
        """Update gate visualizations based on current gate parameters."""
        radii = np.array([2.5, 1e-4, 2.5])
        for i, handle in enumerate(gate_handles):
            center = gate_params[i].value
            if center is not None:
                vertices = gen_vertices(center, radii)
                # Create line segments for closed polygon (4 edges)
                edges = np.array(
                    [
                        [vertices[0], vertices[1]],
                        [vertices[1], vertices[2]],
                        [vertices[2], vertices[3]],
                        [vertices[3], vertices[0]],
                    ],
                    dtype=np.float32,
                )
                handle.points = edges

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
                update_gates()

                # Update trajectory from V_history
                if optimization_problem.state.V_history:
                    V_multi_shoot = np.array(optimization_problem.state.V_history[-1])
                    update_trajectory(V_multi_shoot)

                iteration += 1
                time.sleep(0.05)  # Small delay to avoid overwhelming

            except Exception as e:
                print(f"Optimization error: {e}")
                time.sleep(1.0)

    # Start the optimization thread
    opt_thread = threading.Thread(target=optimization_loop, daemon=True)
    opt_thread.start()

    return server


if __name__ == "__main__":
    print("Starting Drone Racing Real-time Optimization (Viser)")
    print("Open the URL shown below in your browser\n")

    server = create_realtime_server(
        optimization_problem=problem,
        gate_params=gate_center_params,
        initial_centers=initial_gate_centers,
        n_gates=10,
    )

    server.sleep_forever()
