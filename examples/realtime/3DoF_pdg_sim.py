"""Realtime PDG simulation with receding-horizon replanning.

This script is similar to 3DoF_pdg_realtime.py, but it simulates a vehicle that
progresses along the optimized trajectory. When parameters are changed, a new
trajectory is solved from the vehicle's current state.
"""

import importlib.util
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

from examples.plotting_viser import (
    build_scp_step_results,
    compute_velocity_colors_realtime,
    extract_multishoot_trajectory,
    format_metrics_markdown,
    get_print_queue_data,
)

_viridis_cmap = matplotlib.colormaps["viridis"]
VISER_SCENE_SCALE = 0.01

_base_path = os.path.join(current_dir, "base_problems", "3DoF_pdg_realtime_base.py")
_spec = importlib.util.spec_from_file_location("pdg3dof_realtime_base_sim", _base_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load PDG realtime base module: {_base_path}")
pdg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdg)

pdg.problem.initialize()


def _to_deg(rad: float) -> float:
    return float(rad) * 180.0 / np.pi


def _to_rad(deg: float) -> float:
    return float(deg) * np.pi / 180.0


def _generate_cone_mesh(
    apex: np.ndarray,
    height: float,
    half_angle_deg: float,
    n_segments: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    half_angle_rad = np.radians(half_angle_deg)
    base_radius = height * np.tan(half_angle_rad)
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
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
    return vertices, np.array(faces, dtype=np.int32)


def create_realtime_server(optimization_problem) -> viser.ViserServer:
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    server.scene.add_grid(
        "/grid",
        width=7000 * VISER_SCENE_SCALE,
        height=7000 * VISER_SCENE_SCALE,
        position=(0.0, 0.0, 0.0),
    )

    trajectory_handle = server.scene.add_point_cloud(
        "/trajectory",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=(255, 255, 0),
        point_size=30.0 * VISER_SCENE_SCALE,
    )
    executed_traj_handle = server.scene.add_point_cloud(
        "/executed_trajectory",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=(100, 180, 255),
        point_size=35.0 * VISER_SCENE_SCALE,
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
    vehicle_handle = server.scene.add_icosphere(
        "/vehicle",
        radius=45.0 * VISER_SCENE_SCALE,
        color=(80, 160, 255),
        position=tuple(np.asarray(pdg.initial_position.value, dtype=np.float64) * VISER_SCENE_SCALE),
    )

    target_drag = server.scene.add_transform_controls(
        "/target_drag",
        position=target_handle.position,
        scale=40.0 * VISER_SCENE_SCALE,
        disable_rotations=True,
        visible=True,
    )

    state = {
        "running": True,
        "replan_requested": True,
        "reset_sim_requested": False,
        "reset_all_requested": False,
        "plan_elapsed_s": 0.0,
        "active_time_guess_s": float(pdg.total_time),
        "last_sim_wall_s": time.time(),
        "x_plan": None,
        "executed_points": [np.asarray(pdg.initial_position.value, dtype=np.float64)],
        "vehicle_pos": np.asarray(pdg.initial_position.value, dtype=np.float64),
        "vehicle_vel": np.asarray(pdg.velocity.initial, dtype=np.float64),
        "vehicle_mass": float(pdg.mass.initial[0]),
    }
    defaults = {
        "I_sp": float(pdg.I_sp.value),
        "g": float(pdg.g.value),
        "theta_deg": _to_deg(float(pdg.theta.value)),
        "glideslope_deg": _to_deg(float(pdg.glideslope_angle.value)),
        "thrust_pointing_deg": _to_deg(float(pdg.thrust_pointing_angle.value)),
        "T1": float(pdg.T1.value),
        "T2": float(pdg.T2.value),
        "final_xy": np.array(pdg.final_position.value, dtype=np.float64),
        "initial_pos": np.array(pdg.initial_position.value, dtype=np.float64),
        "initial_vel": np.array(pdg.velocity.initial, dtype=np.float64),
        "initial_mass": float(pdg.mass.initial[0]),
        "time_guess": float(pdg.total_time),
        "lam_cost": float(optimization_problem.algorithm.lam_cost),
        "lam_vc": float(optimization_problem.algorithm.lam_vc),
        "lam_prox": float(optimization_problem.algorithm.lam_prox),
    }

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
            + f"\n**Plan time guess:** {state['active_time_guess_s']:.2f}s\n**Plan elapsed:** 0.00s"
        )

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

    with server.gui.add_folder("Reset Controls", expand_by_default=True):
        reset_sim_button = server.gui.add_button("Reset Simulation Progress")
        reset_all_button = server.gui.add_button("Reset Everything")

        @reset_sim_button.on_click
        def _(_) -> None:
            state["reset_sim_requested"] = True

        @reset_all_button.on_click
        def _(_) -> None:
            state["reset_all_requested"] = True

    with server.gui.add_folder("Dynamics / Constraint Parameters"):
        isp_input = server.gui.add_number("I_sp", initial_value=float(pdg.I_sp.value), min=50.0, max=450.0, step=1.0)
        g_input = server.gui.add_number("g (m/s^2)", initial_value=float(pdg.g.value), min=0.0, max=20.0, step=0.01)
        theta_deg_input = server.gui.add_number("theta (deg)", initial_value=_to_deg(pdg.theta.value), min=0.0, max=80.0, step=0.1)
        glideslope_deg_input = server.gui.add_number("glideslope (deg)", initial_value=_to_deg(pdg.glideslope_angle.value), min=1.0, max=89.0, step=0.1)
        thrust_pointing_deg_input = server.gui.add_number("thrust_pointing (deg)", initial_value=_to_deg(pdg.thrust_pointing_angle.value), min=0.0, max=89.0, step=0.1)
        t1_input = server.gui.add_number("T1 per-engine (N)", initial_value=float(pdg.T1.value), min=0.0, max=10000.0, step=10.0)
        t2_input = server.gui.add_number("T2 per-engine (N)", initial_value=float(pdg.T2.value), min=0.0, max=10000.0, step=10.0)
        final_xy_input = server.gui.add_vector2("final_position [x,y] (m)", initial_value=tuple(np.asarray(pdg.final_position.value, dtype=np.float64)), step=10.0)

        def _mark_replan() -> None:
            state["replan_requested"] = True

        @isp_input.on_update
        def _(_) -> None:
            pdg.I_sp.value = float(isp_input.value)
            optimization_problem.parameters["I_sp"] = float(isp_input.value)
            _mark_replan()

        @g_input.on_update
        def _(_) -> None:
            pdg.g.value = float(g_input.value)
            optimization_problem.parameters["g"] = float(g_input.value)
            _mark_replan()

        @theta_deg_input.on_update
        def _(_) -> None:
            val = _to_rad(theta_deg_input.value)
            pdg.theta.value = val
            optimization_problem.parameters["theta"] = val
            _mark_replan()

        @glideslope_deg_input.on_update
        def _(_) -> None:
            val = _to_rad(glideslope_deg_input.value)
            pdg.glideslope_angle.value = val
            optimization_problem.parameters["glideslope_angle"] = val
            _update_glideslope_cone()
            _mark_replan()

        @thrust_pointing_deg_input.on_update
        def _(_) -> None:
            val = _to_rad(thrust_pointing_deg_input.value)
            pdg.thrust_pointing_angle.value = val
            optimization_problem.parameters["thrust_pointing_angle"] = val
            _mark_replan()

        @t1_input.on_update
        def _(_) -> None:
            val = float(t1_input.value)
            pdg.T1.value = val
            optimization_problem.parameters["T1"] = val
            _mark_replan()

        @t2_input.on_update
        def _(_) -> None:
            val = float(t2_input.value)
            pdg.T2.value = val
            optimization_problem.parameters["T2"] = val
            _mark_replan()

        @final_xy_input.on_update
        def _(_) -> None:
            vec = np.array(final_xy_input.value, dtype=np.float64)
            pdg.final_position.value = vec
            optimization_problem.parameters["final_position"] = vec
            target_handle.position = (float(vec[0]) * VISER_SCENE_SCALE, float(vec[1]) * VISER_SCENE_SCALE, 0.0)
            target_drag.position = target_handle.position
            _update_glideslope_cone()
            _mark_replan()

    def _cone_height_scaled() -> float:
        return max(abs(float(state["vehicle_pos"][2])), 100.0) * VISER_SCENE_SCALE

    cone_vertices, cone_faces = _generate_cone_mesh(
        apex=np.array(target_handle.position, dtype=np.float32),
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

    def _update_glideslope_cone() -> None:
        cone_vertices, cone_faces = _generate_cone_mesh(
            apex=np.array(target_handle.position, dtype=np.float32),
            height=_cone_height_scaled(),
            half_angle_deg=_to_deg(float(pdg.glideslope_angle.value)),
            n_segments=48,
        )
        glideslope_cone_handle.vertices = cone_vertices
        glideslope_cone_handle.faces = cone_faces

    @target_drag.on_update
    def _(_) -> None:
        x, y, _ = target_drag.position
        target_drag.position = (x, y, 0.0)
        target_handle.position = (x, y, 0.0)
        vec = np.array([x, y], dtype=np.float64) / VISER_SCENE_SCALE
        pdg.final_position.value = vec
        optimization_problem.parameters["final_position"] = vec
        final_xy_input.value = tuple(vec)
        _update_glideslope_cone()
        state["replan_requested"] = True

    def _replan_from_vehicle_state() -> None:
        remaining_guess = max(1.0, state["active_time_guess_s"] - state["plan_elapsed_s"])
        state["active_time_guess_s"] = remaining_guess
        state["plan_elapsed_s"] = 0.0

        current_pos = np.asarray(state["vehicle_pos"], dtype=np.float64)
        current_vel = np.asarray(state["vehicle_vel"], dtype=np.float64)
        current_mass = float(state["vehicle_mass"])

        pdg.initial_position.value = current_pos
        pdg.position.initial = current_pos
        pdg.velocity.initial = current_vel
        pdg.mass.initial = np.array([current_mass], dtype=np.float64)

        optimization_problem.parameters["initial_position"] = current_pos
        optimization_problem.parameters["I_sp"] = float(pdg.I_sp.value)
        optimization_problem.parameters["g"] = float(pdg.g.value)
        optimization_problem.parameters["theta"] = float(pdg.theta.value)
        optimization_problem.parameters["glideslope_angle"] = float(pdg.glideslope_angle.value)
        optimization_problem.parameters["thrust_pointing_angle"] = float(pdg.thrust_pointing_angle.value)
        optimization_problem.parameters["T1"] = float(pdg.T1.value)
        optimization_problem.parameters["T2"] = float(pdg.T2.value)
        optimization_problem.parameters["final_position"] = np.array(pdg.final_position.value, dtype=np.float64)

        pdg.position.guess = np.linspace(
            current_pos,
            np.array([pdg.final_position.value[0], pdg.final_position.value[1], 0.0], dtype=np.float64),
            pdg.n,
        )
        pdg.velocity.guess = np.linspace(current_vel, np.zeros(3, dtype=np.float64), pdg.n)
        terminal_mass_guess = max(1505.0, current_mass - 150.0)
        pdg.mass.guess = np.linspace(np.array([current_mass]), np.array([terminal_mass_guess]), pdg.n).reshape(-1, 1)

        pdg.time.final = ("free", remaining_guess)
        optimization_problem.reset()

    def _reset_simulation_progress() -> None:
        state["vehicle_pos"] = defaults["initial_pos"].copy()
        state["vehicle_vel"] = defaults["initial_vel"].copy()
        state["vehicle_mass"] = float(defaults["initial_mass"])
        state["executed_points"] = [defaults["initial_pos"].copy()]
        state["x_plan"] = None
        state["active_time_guess_s"] = float(defaults["time_guess"])
        state["plan_elapsed_s"] = 0.0
        state["last_sim_wall_s"] = time.time()
        vehicle_handle.position = tuple(state["vehicle_pos"] * VISER_SCENE_SCALE)
        executed_traj_handle.points = (
            np.asarray(state["executed_points"], dtype=np.float32) * VISER_SCENE_SCALE
        )
        _update_glideslope_cone()
        state["replan_requested"] = True

    def _reset_everything() -> None:
        lam_cost.value = defaults["lam_cost"]
        lam_vc.value = defaults["lam_vc"]
        lam_prox.value = defaults["lam_prox"]

        isp_input.value = defaults["I_sp"]
        g_input.value = defaults["g"]
        theta_deg_input.value = defaults["theta_deg"]
        glideslope_deg_input.value = defaults["glideslope_deg"]
        thrust_pointing_deg_input.value = defaults["thrust_pointing_deg"]
        t1_input.value = defaults["T1"]
        t2_input.value = defaults["T2"]
        final_xy_input.value = tuple(defaults["final_xy"])

        pdg.initial_position.value = defaults["initial_pos"].copy()
        pdg.position.initial = defaults["initial_pos"].copy()
        pdg.velocity.initial = defaults["initial_vel"].copy()
        pdg.mass.initial = np.array([defaults["initial_mass"]], dtype=np.float64)
        optimization_problem.parameters["initial_position"] = defaults["initial_pos"].copy()

        target_handle.position = (
            float(defaults["final_xy"][0]) * VISER_SCENE_SCALE,
            float(defaults["final_xy"][1]) * VISER_SCENE_SCALE,
            0.0,
        )
        target_drag.position = target_handle.position
        trajectory_handle.points = np.zeros((1, 3), dtype=np.float32)
        trajectory_handle.colors = (255, 255, 0)

        _reset_simulation_progress()

    def _interpolate_plan_state(tau: float) -> tuple[np.ndarray, np.ndarray, float]:
        x_plan = state["x_plan"]
        if x_plan is None or len(x_plan) == 0:
            return state["vehicle_pos"], state["vehicle_vel"], state["vehicle_mass"]
        grid = np.linspace(0.0, 1.0, x_plan.shape[0])
        pos = np.array([np.interp(tau, grid, x_plan[:, i]) for i in range(3)], dtype=np.float64)
        vel = np.array([np.interp(tau, grid, x_plan[:, i]) for i in range(3, 6)], dtype=np.float64)
        mass = float(np.interp(tau, grid, x_plan[:, 6]))
        return pos, vel, mass

    def optimization_loop() -> None:
        while state["running"]:
            try:
                now = time.time()
                dt = max(0.0, now - state["last_sim_wall_s"])
                state["last_sim_wall_s"] = now
                state["plan_elapsed_s"] += dt

                if state["reset_all_requested"]:
                    _reset_everything()
                    state["reset_all_requested"] = False
                    state["reset_sim_requested"] = False

                if state["reset_sim_requested"]:
                    _reset_simulation_progress()
                    state["reset_sim_requested"] = False

                if state["replan_requested"]:
                    _replan_from_vehicle_state()
                    state["replan_requested"] = False

                t0 = time.time()
                step_result = optimization_problem.step()
                solve_time_ms = (time.time() - t0) * 1000.0

                results = build_scp_step_results(step_result, solve_time_ms)
                results.update(get_print_queue_data(optimization_problem))
                metrics_text.content = (
                    format_metrics_markdown(results)
                    + f"\n**Plan time guess:** {state['active_time_guess_s']:.2f}s"
                    + f"\n**Plan elapsed:** {state['plan_elapsed_s']:.2f}s"
                )

                if optimization_problem.state.V_history:
                    V_multi_shoot = np.asarray(optimization_problem.state.V_history[-1])
                    n_x = optimization_problem.settings.sim.n_states
                    n_u = optimization_problem.settings.sim.n_controls
                    positions, velocities = extract_multishoot_trajectory(V_multi_shoot, n_x, n_u)
                    if len(positions) > 0:
                        trajectory_handle.points = (positions * VISER_SCENE_SCALE).astype(np.float32)
                        trajectory_handle.colors = compute_velocity_colors_realtime(velocities, _viridis_cmap)

                x_traj = np.asarray(optimization_problem.state.x)
                if x_traj.size and x_traj.shape[1] >= 7:
                    state["x_plan"] = x_traj.copy()

                if state["active_time_guess_s"] > 1e-6:
                    tau = np.clip(state["plan_elapsed_s"] / state["active_time_guess_s"], 0.0, 1.0)
                    pos, vel, mass = _interpolate_plan_state(tau)
                    state["vehicle_pos"] = pos
                    state["vehicle_vel"] = vel
                    state["vehicle_mass"] = mass
                    vehicle_handle.position = tuple(pos * VISER_SCENE_SCALE)
                    state["executed_points"].append(pos.copy())
                    executed = np.asarray(state["executed_points"], dtype=np.float32) * VISER_SCENE_SCALE
                    executed_traj_handle.points = executed
                    _update_glideslope_cone()

                time.sleep(0.05)
            except Exception as e:
                print(f"Optimization error: {e}")
                time.sleep(0.5)

    threading.Thread(target=optimization_loop, daemon=True).start()
    return server


if __name__ == "__main__":
    print("Starting 3DoF PDG realtime simulation.")
    print("Open the URL shown below in your browser.\n")
    server = create_realtime_server(pdg.problem)
    server.sleep_forever()
