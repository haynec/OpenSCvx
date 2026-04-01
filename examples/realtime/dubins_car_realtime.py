"""Interactive real-time visualization for Dubins car path planning using Viser.

Web-based GUI mirroring ``dubins_car_realtime.py`` (PyQt): continuous SCP steps,
draggable circular obstacle, metrics, and λ weights. Run the script and open the
printed URL in your browser.
"""

import os
import sys
import threading
import time

import numpy as np
import viser

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_this_dir, ".."))
sys.path.append(os.path.dirname(os.path.dirname(_this_dir)))

from examples.plotting_viser import (
    build_scp_step_results,
    extract_multishoot_trajectory,
    format_metrics_markdown,
    get_print_queue_data,
)
from realtime.base_problems.dubins_car_realtime_base import problem

problem.initialize()


def create_realtime_server(optimization_problem) -> viser.ViserServer:
    """Create a viser server for real-time Dubins car visualization."""
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    obs_center_init = np.asarray(optimization_problem.parameters["obs_center"], dtype=np.float64)
    obs_radius = float(optimization_problem.parameters["obs_radius"])

    # --- Scene (XY plane at z=0) ---
    server.scene.add_grid(
        "/grid",
        width=8,
        height=8,
        position=(0.0, 0.0, 0.0),
    )
    server.scene.add_frame(
        "/origin",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=0.5,
    )

    trajectory_handle = server.scene.add_point_cloud(
        "/trajectory",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=(0, 100, 255),
        point_size=0.06,
    )

    obstacle_handle = server.scene.add_icosphere(
        "/obstacle",
        radius=obs_radius,
        color=(100, 255, 100),
        position=(float(obs_center_init[0]), float(obs_center_init[1]), 0.0),
    )

    obstacle_drag = server.scene.add_transform_controls(
        "/obstacle_drag",
        position=(float(obs_center_init[0]), float(obs_center_init[1]), 0.0),
        scale=1.2,
        disable_rotations=True,
        visible=False,
    )

    with server.gui.add_folder("Obstacle", expand_by_default=True):
        server.gui.add_markdown("*Click the green obstacle to show the drag gizmo*")
        obs_vector = server.gui.add_vector3(
            "Center (x, y, z)",
            initial_value=(float(obs_center_init[0]), float(obs_center_init[1]), 0.0),
            step=0.1,
        )

    selected = {"on": False}

    def select_obstacle(on: bool) -> None:
        obstacle_drag.visible = on
        obstacle_handle.color = (150, 255, 150) if on else (100, 255, 100)
        selected["on"] = on

    @obstacle_handle.on_click
    def _(_) -> None:
        select_obstacle(not selected["on"])

    @obstacle_drag.on_update
    def _(_) -> None:
        cx, cy, _ = obstacle_drag.position
        c = np.array([cx, cy], dtype=np.float64)
        optimization_problem.parameters["obs_center"] = c
        obs_vector.value = (float(cx), float(cy), 0.0)
        obstacle_handle.position = (cx, cy, 0.0)

    @obs_vector.on_update
    def _(_) -> None:
        cx, cy, _ = obs_vector.value
        c = np.array([cx, cy], dtype=np.float64)
        optimization_problem.parameters["obs_center"] = c
        obstacle_drag.position = (cx, cy, 0.0)
        obstacle_handle.position = (cx, cy, 0.0)

    state = {
        "running": True,
        "reset_requested": False,
    }

    # --- GUI ---
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
            + f"\n**λ_cost:** {optimization_problem.algorithm.lam_cost:.2E}\n**λ_tr:** {optimization_problem.algorithm.lam_prox:.2E}"
        )

    with server.gui.add_folder("Optimization Weights"):
        lam_cost_input = server.gui.add_number(
            "λ_cost",
            initial_value=optimization_problem.algorithm.lam_cost,
            min=1e-6,
            max=1e6,
            step=0.01,
        )
        lam_tr_input = server.gui.add_number(
            "λ_tr (lam_prox)",
            initial_value=optimization_problem.algorithm.lam_prox,
            min=1e-6,
            max=1e6,
            step=0.1,
        )

        @lam_cost_input.on_update
        def _(_) -> None:
            optimization_problem.algorithm.lam_cost = lam_cost_input.value

        @lam_tr_input.on_update
        def _(_) -> None:
            optimization_problem.algorithm.lam_prox = lam_tr_input.value

    with server.gui.add_folder("Problem Control"):
        reset_button = server.gui.add_button("Reset Problem")

        @reset_button.on_click
        def _(_) -> None:
            state["reset_requested"] = True
            print("Problem reset requested")

    def sync_obstacle_from_params() -> None:
        c = np.asarray(optimization_problem.parameters["obs_center"], dtype=np.float64)
        obstacle_handle.position = (float(c[0]), float(c[1]), 0.0)
        obstacle_drag.position = (float(c[0]), float(c[1]), 0.0)
        obs_vector.value = (float(c[0]), float(c[1]), 0.0)

    def update_metrics(results: dict) -> None:
        base = format_metrics_markdown(results)
        lam = (
            f"\n**λ_cost:** {optimization_problem.algorithm.lam_cost:.2E}\n"
            f"**λ_tr:** {optimization_problem.algorithm.lam_prox:.2E}"
        )
        metrics_text.content = base + lam

    def update_trajectory(V_multi_shoot: np.ndarray) -> None:
        try:
            n_x = optimization_problem.settings.sim.n_states
            n_u = optimization_problem.settings.sim.n_controls
            pos_xy, _ = extract_multishoot_trajectory(
                V_multi_shoot,
                n_x,
                n_u,
                position_slice=slice(0, 2),
                velocity_slice=None,
            )
            if len(pos_xy) > 0:
                z = np.zeros((pos_xy.shape[0], 1), dtype=np.float32)
                positions = np.hstack([pos_xy.astype(np.float32), z])
                n = positions.shape[0]
                colors = np.tile(np.array([[0, 100, 255]], dtype=np.uint8), (n, 1))
                trajectory_handle.points = positions
                trajectory_handle.colors = colors
        except Exception as e:
            print(f"Trajectory update error: {e}")
            x_traj = np.asarray(optimization_problem.state.x)
            if x_traj.size and x_traj.shape[1] >= 2:
                xy = x_traj[:, :2].astype(np.float32)
                z = np.zeros((xy.shape[0], 1), dtype=np.float32)
                positions = np.hstack([xy, z])
                n = positions.shape[0]
                colors = np.tile(np.array([[0, 100, 255]], dtype=np.uint8), (n, 1))
                trajectory_handle.points = positions
                trajectory_handle.colors = colors

    def optimization_loop() -> None:
        while state["running"]:
            try:
                if state["reset_requested"]:
                    optimization_problem.reset()
                    state["reset_requested"] = False
                    sync_obstacle_from_params()
                    print("Problem reset to initial conditions")

                t0 = time.time()
                step_result = optimization_problem.step()
                solve_time_ms = (time.time() - t0) * 1000.0

                results = build_scp_step_results(step_result, solve_time_ms)
                results.update(get_print_queue_data(optimization_problem))

                update_metrics(results)

                if optimization_problem.state.V_history:
                    V_multi_shoot = np.asarray(optimization_problem.state.V_history[-1])
                    update_trajectory(V_multi_shoot)

                time.sleep(0.05)
            except Exception as e:
                print(f"Optimization error: {e}")
                time.sleep(1.0)

    threading.Thread(target=optimization_loop, daemon=True).start()
    return server


if __name__ == "__main__":
    server = create_realtime_server(problem)
    print("Viser server started. Open the URL in your browser.")
    server.sleep_forever()
