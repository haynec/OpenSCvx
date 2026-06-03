"""3D monoped contact-implicit locomotion on flat ground (Frax + CITO BYOF).

Minimum-time hop with fixed floating-base pose at start and end, free leg joints,
and contact forces / complementarity via ``CitoFraxDynamics``.

Requires:
    pip install openscvx[frax]
"""

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

try:
    import frax  # noqa: F401 — verify optional extra is installed
except ImportError:
    print(
        "frax is not installed. Install with: pip install openscvx[frax]",
        file=sys.stderr,
    )
    sys.exit(1)

import openscvx as ox
from openscvx.integrations.frax_cito import configure_impulsive_nodes, monoped_standing_pose

# ── CITO dynamics adapter ─────────────────────────────────────────────────────
# First successful solves are easier with impulses/cross off; enable for full CITO.
config = ox.ContactModelConfig(
    delta=0.15,
    mu=1.0,
    z_ground=0.0,
    enable_impulses=False,
    enable_cross_complementarity=False,
)
dyn = ox.CitoFraxDynamics(config=config)
robot = dyn.robot

q, qd, *aux_states = dyn.states
(tau, *contact_controls) = dyn.controls

nj = robot.num_joints
na = tau.shape[0]
n = 16
total_time = 1.2

BASE_POS = slice(0, 3)  # x, y, z — only positional BCs on the floating base
LEG = slice(6, 8)

# Upright stand, leg fully extended (hip=0, knee at URDF lower limit), foot on ground.
q_start = monoped_standing_pose(robot, foot_xy=(0.0, 0.0), z_ground=config.z_ground)
q_goal = monoped_standing_pose(robot, foot_xy=(0.35, 0.0), z_ground=config.z_ground)


def _q_boundary(values: np.ndarray, *, fix_base_position: bool, fix_leg: bool) -> list:
    bc = []
    for i in range(nj):
        if BASE_POS.start <= i < BASE_POS.stop and fix_base_position:
            bc.append(float(values[i]))
        elif LEG.start <= i < LEG.stop and fix_leg:
            bc.append(float(values[i]))
        else:
            bc.append(ox.Free(float(values[i])))
    return bc


# Fix base position at start/end; roll/pitch/yaw and leg joints are free.
q.initial = _q_boundary(q_start, fix_base_position=True, fix_leg=False)
q.final = _q_boundary(q_goal, fix_base_position=True, fix_leg=False)

qd.initial = np.zeros(nj)
qd.final = np.zeros(nj)

for aux in aux_states:
    aux.initial = np.zeros(1)
    aux.final = np.zeros(1)
    aux.guess = np.zeros((n, 1))

q_guess = np.linspace(q_start, q_goal, n)
q.guess = q_guess
qd.guess = np.zeros((n, nj))

grav = np.array([np.asarray(robot.gravity_vector(qi)) for qi in q_guess])
# Actuated joints are the last ``na`` entries (hip, knee); base DOFs are unactuated.
tau.guess = grav[:, -na:]

for c in contact_controls:
    c.guess = np.zeros((n, int(np.prod(c.shape))))

configure_impulsive_nodes(dyn.controls, n)

constraints = []
for state in dyn.states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
for control in dyn.controls:
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

time = ox.Time(
    initial=0.0,
    final=ox.Minimize(total_time),
    min=0.0,
    max=3.0 * total_time,
    uniform_time_grid=True,
)

problem = ox.Problem(
    dynamics=dyn,
    states=dyn.states,
    controls=dyn.controls,
    time=time,
    constraints=constraints,
    N=n,
    algorithm={
        "lam_vc": 1e2,
        "lam_cost": 4e-1,
        "k_max": 40,
    },
    float_dtype="float64",
    solver={
        "cvx_solver": "CLARABEL",
        "solver_args": {},
    },
)


def _monoped_keypoints(q_traj: np.ndarray, robot) -> tuple[np.ndarray, np.ndarray]:
    """Skeleton keypoints and foot positions along a joint trajectory."""
    q_traj = np.asarray(q_traj, dtype=float)
    n_frames = q_traj.shape[0]
    n_links = robot.num_joints
    keypoints = np.zeros((n_frames, n_links + 1, 3))
    foot_pos = np.zeros((n_frames, 3))

    for t in range(n_frames):
        q = q_traj[t]
        links = np.asarray(robot.link_to_world_transforms(q))
        keypoints[t, 0] = links[0, :3, 3]
        keypoints[t, 1:] = links[:, :3, 3]
        foot = np.asarray(robot.foot_transform(q))
        foot_pos[t] = foot[:3, 3]

    return keypoints, foot_pos


def visualize(results, robot, q_start: np.ndarray, q_goal: np.ndarray) -> None:
    """Animate the optimized monoped trajectory (Viser stick model + foot trail)."""
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

    keypoints, foot_pos = _monoped_keypoints(q_traj, robot)
    n_segs = robot.num_joints

    foot_start = np.asarray(robot.foot_transform(q_start))[:3, 3]
    foot_goal = np.asarray(robot.foot_transform(q_goal))[:3, 3]

    foot_vel = np.gradient(foot_pos, t_vec, axis=0, edge_order=2)
    foot_colors = compute_velocity_colors(foot_vel)

    server = create_server(foot_pos, show_grid=False)
    server.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.25, position=(0.0, 0.0, 0.0))
    server.scene.add_frame("/origin", axes_length=0.15, axes_radius=0.004)

    add_target_markers(
        server,
        [foot_start, foot_goal],
        radius=0.02,
        colors=[(100, 150, 255), (255, 80, 80)],
    )

    add_ghost_trajectory(server, foot_pos, foot_colors, point_size=0.008)
    _, update_trail = add_animated_trail(server, foot_pos, foot_colors, point_size=0.012)
    _, update_marker = add_position_marker(server, foot_pos, radius=0.02)

    link_rgb = np.linspace([60, 90, 200], [255, 140, 60], n_segs).astype(np.uint8)
    link_colors = np.stack([link_rgb, link_rgb], axis=1)

    init_points = np.stack(
        [np.stack([keypoints[0, k], keypoints[0, k + 1]]) for k in range(n_segs)]
    ).astype(np.float32)
    arm_handle = server.scene.add_line_segments(
        "/monoped_links",
        points=init_points,
        colors=link_colors,
        line_width=4.0,
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
                y=q_traj[:, j].tolist(),
                mode="lines",
                name=f"q{j}",
            )
        )
    fig_joints.update_layout(
        title="Joint coordinates",
        xaxis_title="Time (s)",
        yaxis_title="q",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    fig_tau = go.Figure()
    for j in range(robot.num_actuated_joints):
        fig_tau.add_trace(
            go.Scatter(
                x=t_vec.tolist(),
                y=tau_traj[:, j].tolist(),
                mode="lines",
                name=f"τ{j}",
            )
        )
    fig_tau.update_layout(
        title="Actuator torques (Nm)",
        xaxis_title="Time (s)",
        yaxis_title="τ",
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
    print("3D monoped CITO flat-ground hop (frax)")
    print("=" * 60)
    print(f"num_joints = {nj}, num_actuated = {na}, N = {n}")
    print(f"joint names: {list(robot.joint_names)}")
    print(f"q limits (min): {np.array2string(q.min, precision=2)}")
    print(f"q limits (max): {np.array2string(q.max, precision=2)}")
    print(f"q_start pos={q_start[BASE_POS]}  leg=[{q_start[6]:.3f}, {q_start[7]:.3f}]")
    print(f"q_goal  pos={q_goal[BASE_POS]}  leg=[{q_goal[6]:.3f}, {q_goal[7]:.3f}]")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    q_final = np.asarray(results.nodes["q"][-1])
    base_err = np.linalg.norm(q_final[BASE_POS] - q_goal[BASE_POS])
    print()
    print(f"Final base position:  {np.array2string(q_final[BASE_POS], precision=3)}")
    print(f"Target base position: {np.array2string(q_goal[BASE_POS], precision=3)}")
    print(f"||q_base_pos_final - q_base_pos_goal|| = {base_err:.4e}")
    print(f"Standing guess: hip={q_start[6]:.3f}, knee={q_start[7]:.3f} (extended)")
    print()
    print("Launching Viser visualization (Ctrl+C to exit)...")
    visualize(results, robot, q_start, q_goal)
