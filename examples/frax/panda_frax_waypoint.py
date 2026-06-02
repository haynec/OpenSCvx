"""Franka Panda point-to-point motion with an intermediate EE waypoint (frax dynamics).

Extends ``examples/frax/panda_frax.py``: same ``FraxDynamics`` setup, joint
start/goal boundary conditions, CTCS box limits, and Viser stick-model animation,
but adds a **nodal** end-effector position constraint at the midpoint of the
horizon (node ``N // 2``). Task kinematics use ``frax`` forward kinematics
(``ee_transform``) via a BYOF nodal constraint so the waypoint matches the
bundled Panda URDF exactly.

Nodal (non-CTCS) constraints are handled by the default ``CVXPyPTRSolver``; use
``lam_vb=1e3`` so the virtual-buffer penalty can drive the waypoint constraint
to feasibility alongside full frax rigid-body dynamics.

Requires:
    pip install openscvx[frax]
"""

import os
import sys

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

try:
    import frax
except ImportError:
    print(
        "frax is not installed. Install with: pip install openscvx[frax]",
        file=sys.stderr,
    )
    sys.exit(1)

import openscvx as ox

robot = frax.load_panda()
dyn = ox.FraxDynamics(robot)
q, qd = dyn.states
(tau,) = dyn.controls

n_j = robot.num_joints
n = 20
total_time = 5.0
waypoint_node = n // 2
ee_tol = 0.02  # m

q_start = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
q_goal = np.array([0.6, -0.3, 0.2, -1.8, 0.3, 1.2, 0.5])

ee_start = np.asarray(robot.ee_transform(q_start))[:3, 3]
ee_goal = np.asarray(robot.ee_transform(q_goal))[:3, 3]
# Intermediate target: modest offset from the start pose (reachable, visibly off the start→goal chord).
ee_waypoint = ee_start + np.array([0.08, 0.05, 0.04])

q_min = np.asarray(robot.joint_lower_limits, dtype=float)
q_max = np.asarray(robot.joint_upper_limits, dtype=float)


def _ik_position(target: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
    """Position-only IK for initial-guess seeding (not part of the optimization model)."""

    def objective(qv: np.ndarray) -> float:
        p = np.asarray(robot.ee_transform(qv))[:3, 3]
        return float(np.sum((p - target) ** 2))

    result = minimize(
        objective,
        q_seed,
        method="L-BFGS-B",
        bounds=list(zip(q_min, q_max)),
    )
    return result.x


def _build_initial_guess(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    ee_waypoint: np.ndarray,
    *,
    n_nodes: int,
    waypoint_node: int,
    total_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint / velocity guesses respecting BCs and the EE waypoint.

    * ``q_start`` / ``q_goal`` are pinned at nodes 0 and N-1 (match boundaries).
    * ``q_mid`` comes from position-only IK to ``ee_waypoint``, seeded from
      ``q_start`` so the leg does not jump toward the chord midpoint in joint space.
    * ``ox.init.linspace`` connects the three joint keyframes; interior nodes are
      filled consistently on both trajectory segments.
    * ``qd`` uses a nominal-time finite difference with zero velocity at endpoints.
    """
    keyframe_nodes = [0, waypoint_node, n_nodes - 1]
    q_mid = _ik_position(ee_waypoint, q_start)
    q_guess = ox.init.linspace([q_start, q_mid, q_goal], keyframe_nodes)
    q_guess[0] = q_start
    q_guess[waypoint_node] = q_mid
    q_guess[-1] = q_goal

    qd_guess = np.zeros((n_nodes, n_j))
    if n_nodes > 2:
        dt_nom = total_time / (n_nodes - 1)
        qd_guess[1:-1] = np.gradient(q_guess, dt_nom, axis=0)[1:-1]
    return q_guess, qd_guess


q.initial = q_start
q.final = q_goal
qd.initial = np.zeros(n_j)
qd.final = np.zeros(n_j)

q.guess, qd.guess = _build_initial_guess(
    q_start,
    q_goal,
    ee_waypoint,
    n_nodes=n,
    waypoint_node=waypoint_node,
    total_time=total_time,
)
# Gravity-compensating torque guess (not zero). frax integrates the full
# rigid-body dynamics, so a zero-torque guess implies free-fall — joint
# accelerations of tens of rad/s² — which produces large dynamics defects that
# can drive the convex subproblem infeasible at low node counts. Seeding tau with
# g(q) makes forward_dynamics(q, 0, g(q)) ≈ 0, so the qd channel of the guess is
# self-consistent and SCvx stays robust even on a coarse time grid.
grav = np.array([np.asarray(robot.gravity_vector(qi)) for qi in q.guess])
tau.guess = grav[:, n_j - robot.num_actuated_joints :]

ee_waypoint_param = ox.Parameter("ee_waypoint", shape=(3,), value=ee_waypoint)


def _ee_waypoint_residual(x, u, node, params):
    del u, node
    q_val = x[q.slice]
    p = jnp.asarray(robot.ee_transform(q_val))[:3, 3]
    return jnp.linalg.norm(p - params["ee_waypoint"]) - ee_tol


byof = {
    "parameters": [ee_waypoint_param],
    "nodal_constraints": [
        {
            "constraint_fn": _ee_waypoint_residual,
            "nodes": [waypoint_node],
        }
    ],
}

constraints = []
for state in dyn.states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

time = ox.Time(
    initial=0.0,
    final=ox.Minimize(total_time),
    min=0.0,
    max=2.0 * total_time,
    time_dilation_min=0.05 * total_time,
    time_dilation_max=2.0 * total_time,
)

problem = ox.Problem(
    dynamics=dyn,
    states=dyn.states,
    controls=dyn.controls,
    time=time,
    constraints=constraints,
    byof=byof,
    N=n,
    algorithm={
        "lam_vb": 1e1,
        "lam_vc": 1e1,
        "lam_cost": 1e-1,
    },
    discretizer={"diffrax_kwargs": {"atol": 1e-12, "rtol": 1e-12}},
    float_dtype="float64",
    solver={"cvx_solver": "QOCO",
    "solver_args": {"abstol": 1e-8, "reltol": 1e-11, "enforce_dpp": True}},
    licq_max=1e-8,
)


def _compute_panda_keypoints(q_traj: np.ndarray, robot) -> tuple[np.ndarray, np.ndarray]:
    q_traj = np.asarray(q_traj, dtype=float)
    n_frames = q_traj.shape[0]
    n_links = robot.num_joints
    keypoints = np.zeros((n_frames, n_links + 2, 3))
    ee_pos = np.zeros((n_frames, 3))

    for t in range(n_frames):
        qi = q_traj[t]
        links = np.asarray(robot.link_to_world_transforms(qi))
        ee = np.asarray(robot.ee_transform(qi))
        keypoints[t, 0] = (0.0, 0.0, 0.0)
        keypoints[t, 1 : 1 + n_links] = links[:, :3, 3]
        keypoints[t, -1] = ee[:3, 3]
        ee_pos[t] = ee[:3, 3]

    return keypoints, ee_pos


def visualize(
    results,
    robot,
    ee_targets: list[np.ndarray],
    target_colors: list[tuple[int, int, int]],
) -> None:
    import plotly.graph_objects as go

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        add_ghost_trajectory,
        add_position_marker,
        add_target_markers,
        compute_velocity_colors,
        create_server,
    )
    from openscvx.plotting.viser.plotly_integration import add_animated_plotly_vline

    t_vec = np.asarray(results.trajectory["time"]).flatten()
    q_traj = np.asarray(results.trajectory["q"])
    tau_traj = np.asarray(results.trajectory["tau"])

    keypoints, ee_pos = _compute_panda_keypoints(q_traj, robot)
    n_segs = robot.num_joints + 1

    ee_vel = np.gradient(ee_pos, t_vec, axis=0, edge_order=2)
    ee_colors = compute_velocity_colors(ee_vel)

    server = create_server(ee_pos, show_grid=False)
    server.scene.add_grid("/grid", width=1.5, height=1.5, cell_size=0.25)
    server.scene.add_frame("/origin", axes_length=0.08, axes_radius=0.003)

    add_target_markers(server, ee_targets, radius=0.012, colors=target_colors)

    add_ghost_trajectory(server, ee_pos, ee_colors, point_size=0.005)
    _, update_trail = add_animated_trail(server, ee_pos, ee_colors, point_size=0.008)
    _, update_marker = add_position_marker(server, ee_pos, radius=0.012)

    link_rgb = np.linspace([80, 100, 180], [255, 120, 80], n_segs).astype(np.uint8)
    link_colors = np.stack([link_rgb, link_rgb], axis=1)

    init_points = np.stack(
        [np.stack([keypoints[0, k], keypoints[0, k + 1]]) for k in range(n_segs)]
    ).astype(np.float32)
    arm_handle = server.scene.add_line_segments(
        "/panda_links",
        points=init_points,
        colors=link_colors,
        line_width=5.0,
    )

    server.scene.add_box(
        "/panda_base",
        dimensions=(0.12, 0.12, 0.08),
        position=(0.0, 0.0, 0.04),
        color=(60, 60, 60),
    )

    def update_arm(frame_idx: int) -> None:
        pts = np.stack(
            [np.stack([keypoints[frame_idx, k], keypoints[frame_idx, k + 1]]) for k in range(n_segs)]
        ).astype(np.float32)
        arm_handle.points = pts

    fig_joints = go.Figure()
    for j in range(robot.num_joints):
        fig_joints.add_trace(
            go.Scatter(
                x=t_vec.tolist(),
                y=np.rad2deg(q_traj[:, j]).tolist(),
                mode="lines",
                name=f"q{j + 1}",
            )
        )
    fig_joints.update_layout(
        title="Joint angles (deg)",
        xaxis_title="Time (s)",
        yaxis_title="θ (deg)",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    fig_tau = go.Figure()
    for j in range(robot.num_actuated_joints):
        fig_tau.add_trace(
            go.Scatter(
                x=t_vec.tolist(),
                y=tau_traj[:, j].tolist(),
                mode="lines",
                name=f"τ{j + 1}",
            )
        )
    fig_tau.update_layout(
        title="Joint torques (Nm)",
        xaxis_title="Time (s)",
        yaxis_title="τ (Nm)",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    with server.gui.add_folder("Plots"):
        _, update_joints = add_animated_plotly_vline(server, fig_joints, t_vec, folder_name=None)
        _, update_tau = add_animated_plotly_vline(server, fig_tau, t_vec, folder_name=None)

    add_animation_controls(
        server,
        t_vec,
        [update_arm, update_trail, update_marker, update_joints, update_tau],
    )
    server.sleep_forever()


if __name__ == "__main__":
    print("Franka Panda via frax dynamics — intermediate EE waypoint")
    print("=" * 60)
    print(f"num_joints = {n_j}, N = {n}, waypoint node = {waypoint_node}")
    print(f"EE start:     {np.round(ee_start, 3)}")
    print(f"EE waypoint:  {np.round(ee_waypoint, 3)}  (tol = {ee_tol} m)")
    print(f"EE goal:      {np.round(ee_goal, 3)}")
    ee_guess_mid = np.asarray(robot.ee_transform(q.guess[waypoint_node]))[:3, 3]
    print(
        f"Init guess @ node {waypoint_node}: "
        f"||EE - waypoint|| = {np.linalg.norm(ee_guess_mid - ee_waypoint):.4e} m"
    )
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    q_traj = np.asarray(results.trajectory["q"])
    ee_traj = np.array([np.asarray(robot.ee_transform(qi))[:3, 3] for qi in q_traj])

    err_joints = np.linalg.norm(q_traj[-1] - q_goal)
    err_waypoint = np.linalg.norm(ee_traj[waypoint_node] - ee_waypoint)

    print()
    print("Results:")
    print(f"  ||q_final - q_goal|| = {err_joints:.4e}")
    print(f"  ||EE(node {waypoint_node}) - waypoint|| = {err_waypoint:.4e} m")
    print(f"  CTCS violation = {results.ctcs_violation}")
    print()
    print("Launching Viser visualization (Ctrl+C to exit)...")
    visualize(
        results,
        robot,
        [ee_start, ee_waypoint, ee_goal],
        [(100, 150, 255), (255, 180, 50), (255, 80, 80)],
    )
