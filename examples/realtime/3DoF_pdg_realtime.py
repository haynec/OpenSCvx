"""Interactive real-time visualization for 3-DOF powered descent guidance.

This module provides a Viser GUI that allows live updates of PDG problem
parameters (dynamics, constraints, boundary conditions, and SCP weights).
"""

import os
import sys
import threading
import time

import matplotlib
import numpy as np
import viser

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import importlib.util

_base_path = os.path.join(current_dir, "base_problems", "3DoF_pdg_realtime_base.py")
_spec = importlib.util.spec_from_file_location("pdg3dof_realtime_base", _base_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load PDG realtime base module: {_base_path}")
pdg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdg)

from examples.plotting_viser import (
    build_scp_step_results,
    compute_velocity_colors_realtime,
    extract_multishoot_trajectory,
    format_metrics_markdown,
    get_print_queue_data,
)

_viridis_cmap = matplotlib.colormaps["viridis"]
VISER_SCENE_SCALE = 0.01

# Initialize once; updates are handled via parameter/state mutation + reset.
pdg.initialize_problem_quiet(pdg.problem)
pdg.print_scp_table_header_once(pdg.problem)


def _to_deg(rad: float) -> float:
    return float(rad) * 180.0 / np.pi


def _to_rad(deg: float) -> float:
    return float(deg) * np.pi / 180.0


def _generate_cone_mesh(
    apex: np.ndarray,
    height: float,
    half_angle_deg: float,
    n_segments: int = 32,
    axis: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a cone mesh for realtime glideslope visualization."""
    half_angle_rad = np.radians(half_angle_deg)
    base_radius = height * np.tan(half_angle_rad)

    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = ref - np.dot(ref, axis) * axis
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    vertices = [apex.copy()]
    base_center = apex + height * axis
    for i in range(n_segments):
        angle = 2.0 * np.pi * i / n_segments
        offset = base_radius * (np.cos(angle) * u + np.sin(angle) * v)
        vertices.append(base_center + offset)
    vertices.append(base_center.copy())
    vertices = np.array(vertices, dtype=np.float32)

    faces = []
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append([0, i + 1, next_i + 1])
    base_center_idx = n_segments + 1
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append([base_center_idx, next_i + 1, i + 1])
    faces = np.array(faces, dtype=np.int32)
    return vertices, faces


def _sync_problem_parameters_from_base() -> None:
    """Push current base parameter values into the active problem."""
    pdg.problem.parameters["I_sp"] = float(pdg.I_sp.value)
    pdg.problem.parameters["g"] = float(pdg.g.value)
    pdg.problem.parameters["theta"] = float(pdg.theta.value)
    pdg.problem.parameters["glideslope_angle"] = float(pdg.glideslope_angle.value)
    pdg.problem.parameters["thrust_pointing_angle"] = float(pdg.thrust_pointing_angle.value)
    pdg.problem.parameters["T1"] = float(pdg.T1.value)
    pdg.problem.parameters["T2"] = float(pdg.T2.value)
    pdg.problem.parameters["initial_position"] = np.array(pdg.initial_position.value, dtype=np.float64)
    pdg.problem.parameters["final_position"] = np.array(pdg.final_position.value, dtype=np.float64)


def create_realtime_server(optimization_problem) -> viser.ViserServer:
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    server.scene.add_grid(
        "/grid",
        width=7000 * VISER_SCENE_SCALE,
        height=7000 * VISER_SCENE_SCALE,
        position=(0.0, 0.0, 0.0),
    )
    server.scene.add_frame(
        "/origin",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=200.0 * VISER_SCENE_SCALE,
    )

    trajectory_handle = server.scene.add_point_cloud(
        "/trajectory",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=(255, 255, 0),
        point_size=30.0 * VISER_SCENE_SCALE,
    )
    start_handle = server.scene.add_icosphere(
        "/start",
        radius=40.0 * VISER_SCENE_SCALE,
        color=(100, 255, 100),
        position=tuple(np.asarray(pdg.position.initial, dtype=np.float64) * VISER_SCENE_SCALE),
    )
    target_handle = server.scene.add_icosphere(
        "/target",
        radius=40.0 * VISER_SCENE_SCALE,
        color=(255, 100, 100),
        position=(
            float(pdg.final_position.value[0]) * VISER_SCENE_SCALE,
            float(pdg.final_position.value[1]) * VISER_SCENE_SCALE,
            0.0,
        ),
    )
    start_drag = server.scene.add_transform_controls(
        "/start_drag",
        position=tuple(np.asarray(pdg.position.initial, dtype=np.float64) * VISER_SCENE_SCALE),
        scale=40.0 * VISER_SCENE_SCALE,
        disable_rotations=True,
        visible=True,
    )
    target_drag = server.scene.add_transform_controls(
        "/target_drag",
        position=(
            float(pdg.final_position.value[0]) * VISER_SCENE_SCALE,
            float(pdg.final_position.value[1]) * VISER_SCENE_SCALE,
            0.0,
        ),
        scale=40.0 * VISER_SCENE_SCALE,
        disable_rotations=True,
        visible=True,
    )

    def _cone_height_scaled() -> float:
        init_z = float(np.asarray(pdg.initial_position.value, dtype=np.float64)[2])
        return max(abs(init_z), 100.0) * VISER_SCENE_SCALE

    def _update_glideslope_cone() -> None:
        apex = np.array(
            [
                float(pdg.final_position.value[0]) * VISER_SCENE_SCALE,
                float(pdg.final_position.value[1]) * VISER_SCENE_SCALE,
                0.0,
            ],
            dtype=np.float32,
        )
        cone_vertices, cone_faces = _generate_cone_mesh(
            apex=apex,
            height=_cone_height_scaled(),
            half_angle_deg=_to_deg(float(pdg.glideslope_angle.value)),
            n_segments=48,
        )
        glideslope_cone_handle.vertices = cone_vertices
        glideslope_cone_handle.faces = cone_faces

    cone_vertices, cone_faces = _generate_cone_mesh(
        apex=np.array(
            [
                float(pdg.final_position.value[0]) * VISER_SCENE_SCALE,
                float(pdg.final_position.value[1]) * VISER_SCENE_SCALE,
                0.0,
            ],
            dtype=np.float32,
        ),
        height=_cone_height_scaled(),
        half_angle_deg=_to_deg(float(pdg.glideslope_angle.value)),
        n_segments=48,
    )
    glideslope_cone_handle = server.scene.add_mesh_simple(
        "/constraints/glideslope_cone",
        vertices=cone_vertices,
        faces=cone_faces,
        color=(80, 200, 120),
        wireframe=False,
        opacity=0.2,
    )

    state = {"running": True, "reset_requested": False}

    with server.gui.add_folder("Optimization Metrics"):
        metrics_text = server.gui.add_markdown(
            format_metrics_markdown(
                {
                    "iter": 0,
                    "J_tr": 0.0,
                    "J_vb": 0.0,
                    "J_vc": 0.0,
                    "cost": 0.0,
                    "dis_time": 0.0,
                    "solve_time": 0.0,
                    "prob_stat": "--",
                }
            )
        )

    with server.gui.add_folder("Problem Control", expand_by_default=True):
        reset_button = server.gui.add_button("Apply Changes + Reset Problem")

        @reset_button.on_click
        def _(_) -> None:
            state["reset_requested"] = True

    with server.gui.add_folder("Algorithm Weights"):
        lam_cost = server.gui.add_number("lam_cost", initial_value=optimization_problem.algorithm.lam_cost, min=1e-8, max=1e5, step=0.01)
        lam_vc = server.gui.add_number("lam_vc", initial_value=optimization_problem.algorithm.lam_vc, min=1e-8, max=1e5, step=0.01)
        lam_prox = server.gui.add_number("lam_prox", initial_value=optimization_problem.algorithm.lam_prox, min=1e-8, max=1e5, step=0.01)

        @lam_cost.on_update
        def _(_) -> None:
            optimization_problem.algorithm.lam_cost = float(lam_cost.value)

        @lam_vc.on_update
        def _(_) -> None:
            optimization_problem.algorithm.lam_vc = float(lam_vc.value)

        @lam_prox.on_update
        def _(_) -> None:
            optimization_problem.algorithm.lam_prox = float(lam_prox.value)

    with server.gui.add_folder("Dynamics / Constraint Parameters"):
        isp_input = server.gui.add_number("I_sp", initial_value=float(pdg.I_sp.value), min=50.0, max=450.0, step=1.0)
        g_input = server.gui.add_number("g (m/s^2)", initial_value=float(pdg.g.value), min=0.0, max=20.0, step=0.01)
        theta_deg_input = server.gui.add_number("theta (deg)", initial_value=_to_deg(pdg.theta.value), min=0.0, max=80.0, step=0.1)
        glideslope_deg_input = server.gui.add_number("glideslope (deg)", initial_value=_to_deg(pdg.glideslope_angle.value), min=1.0, max=89.0, step=0.1)
        thrust_pointing_deg_input = server.gui.add_number("thrust_pointing (deg)", initial_value=_to_deg(pdg.thrust_pointing_angle.value), min=0.0, max=89.0, step=0.1)
        t1_input = server.gui.add_number("T1 per-engine (N)", initial_value=float(pdg.T1.value), min=0.0, max=10000.0, step=10.0)
        t2_input = server.gui.add_number("T2 per-engine (N)", initial_value=float(pdg.T2.value), min=0.0, max=10000.0, step=10.0)
        initial_pos_input = server.gui.add_vector3(
            "initial_position (m)",
            initial_value=tuple(np.asarray(pdg.initial_position.value, dtype=np.float64)),
            step=10.0,
        )
        final_xy_input = server.gui.add_vector2("final_position [x,y] (m)", initial_value=tuple(np.asarray(pdg.final_position.value, dtype=np.float64)), step=10.0)

        @isp_input.on_update
        def _(_) -> None:
            pdg.I_sp.value = float(isp_input.value)
            optimization_problem.parameters["I_sp"] = float(isp_input.value)

        @g_input.on_update
        def _(_) -> None:
            pdg.g.value = float(g_input.value)
            optimization_problem.parameters["g"] = float(g_input.value)

        @theta_deg_input.on_update
        def _(_) -> None:
            val = _to_rad(theta_deg_input.value)
            pdg.theta.value = val
            optimization_problem.parameters["theta"] = val

        @glideslope_deg_input.on_update
        def _(_) -> None:
            val = _to_rad(glideslope_deg_input.value)
            pdg.glideslope_angle.value = val
            optimization_problem.parameters["glideslope_angle"] = val
            _update_glideslope_cone()

        @thrust_pointing_deg_input.on_update
        def _(_) -> None:
            val = _to_rad(thrust_pointing_deg_input.value)
            pdg.thrust_pointing_angle.value = val
            optimization_problem.parameters["thrust_pointing_angle"] = val

        @t1_input.on_update
        def _(_) -> None:
            val = float(t1_input.value)
            pdg.T1.value = val
            optimization_problem.parameters["T1"] = val

        @t2_input.on_update
        def _(_) -> None:
            val = float(t2_input.value)
            pdg.T2.value = val
            optimization_problem.parameters["T2"] = val

        @initial_pos_input.on_update
        def _(_) -> None:
            new_initial = np.array(initial_pos_input.value, dtype=np.float64)
            pdg.initial_position.value = new_initial
            optimization_problem.parameters["initial_position"] = new_initial
            pdg.position.initial = new_initial
            start_handle.position = tuple(new_initial * VISER_SCENE_SCALE)
            start_drag.position = tuple(new_initial * VISER_SCENE_SCALE)
            _update_glideslope_cone()

        @final_xy_input.on_update
        def _(_) -> None:
            vec = np.array(final_xy_input.value, dtype=np.float64)
            pdg.final_position.value = vec
            optimization_problem.parameters["final_position"] = vec
            target_handle.position = (
                float(vec[0]) * VISER_SCENE_SCALE,
                float(vec[1]) * VISER_SCENE_SCALE,
                0.0,
            )
            _update_glideslope_cone()
            target_drag.position = (
                float(vec[0]) * VISER_SCENE_SCALE,
                float(vec[1]) * VISER_SCENE_SCALE,
                0.0,
            )

    @start_drag.on_update
    def _(_) -> None:
        new_initial = np.array(start_drag.position, dtype=np.float64) / VISER_SCENE_SCALE
        pdg.initial_position.value = new_initial
        optimization_problem.parameters["initial_position"] = new_initial
        pdg.position.initial = new_initial
        start_handle.position = tuple(start_drag.position)
        initial_pos_input.value = tuple(new_initial)
        _update_glideslope_cone()

    @target_drag.on_update
    def _(_) -> None:
        x, y, _ = target_drag.position
        target_drag.position = (x, y, 0.0)
        new_final_xy = np.array([x, y], dtype=np.float64) / VISER_SCENE_SCALE
        pdg.final_position.value = new_final_xy
        optimization_problem.parameters["final_position"] = new_final_xy
        target_handle.position = (x, y, 0.0)
        final_xy_input.value = tuple(new_final_xy)
        _update_glideslope_cone()

    def apply_all_state_edits() -> None:
        # Drive state initial condition from the explicit initial_position parameter.
        pdg.position.initial = np.array(pdg.initial_position.value, dtype=np.float64)
        pdg.position.guess = np.linspace(pdg.position.initial, np.array([0.0, 0.0, 0.0]), pdg.n)
        pdg.velocity.guess = np.linspace(pdg.velocity.initial, np.array([0.0, 0.0, 0.0]), pdg.n)
        pdg.mass.guess = np.linspace(pdg.mass.initial, np.array([1690.0]), pdg.n).reshape(-1, 1)
        _sync_problem_parameters_from_base()

        rho_max = pdg.n_eng * float(pdg.T2.value) * np.cos(float(pdg.theta.value))
        pdg.thrust.min = pdg.n_eng * np.array([-pdg.T_bar, -pdg.T_bar, -pdg.T_bar], dtype=np.float64)
        pdg.thrust.max = pdg.n_eng * np.array([pdg.T_bar, pdg.T_bar, pdg.T_bar], dtype=np.float64)
        pdg.plotting_dict["rho_min"] = pdg.n_eng * float(pdg.T1.value) * np.cos(float(pdg.theta.value))
        pdg.plotting_dict["rho_max"] = rho_max
        pdg.plotting_dict["glideslope_angle_deg"] = _to_deg(float(pdg.glideslope_angle.value))

    def update_trajectory(V_multi_shoot: np.ndarray) -> None:
        try:
            n_x = optimization_problem.settings.sim.n_states
            n_u = optimization_problem.settings.sim.n_controls
            positions, velocities = extract_multishoot_trajectory(V_multi_shoot, n_x, n_u)
            if len(positions) > 0:
                # Reuse helper for speed-colored trajectory.
                colors = compute_velocity_colors_realtime(velocities, _viridis_cmap)
                trajectory_handle.points = (positions * VISER_SCENE_SCALE).astype(np.float32)
                trajectory_handle.colors = colors
        except Exception:
            x_traj = np.asarray(optimization_problem.state.x)
            if x_traj.size and x_traj.shape[1] >= 3:
                points = (x_traj[:, :3] * VISER_SCENE_SCALE).astype(np.float32)
                trajectory_handle.points = points
                trajectory_handle.colors = np.tile(np.array([[255, 255, 0]], dtype=np.uint8), (points.shape[0], 1))

    def optimization_loop() -> None:
        while state["running"]:
            try:
                if state["reset_requested"]:
                    apply_all_state_edits()
                    optimization_problem.reset()
                    state["reset_requested"] = False

                t0 = time.time()
                step_result = optimization_problem.step()
                solve_time_ms = (time.time() - t0) * 1000.0

                results = build_scp_step_results(step_result, solve_time_ms)
                results.update(get_print_queue_data(optimization_problem))
                metrics_text.content = format_metrics_markdown(results)

                if optimization_problem.state.V_history:
                    V_multi_shoot = np.asarray(optimization_problem.state.V_history[-1])
                    update_trajectory(V_multi_shoot)

                time.sleep(0.05)
            except Exception as e:
                print(f"Optimization error: {e}")
                time.sleep(0.5)

    threading.Thread(target=optimization_loop, daemon=True).start()
    return server


if __name__ == "__main__":
    print("Starting 3DoF PDG realtime optimization.")
    print("Open the URL shown below in your browser.\n")
    server = create_realtime_server(pdg.problem)
    server.sleep_forever()
