"""Franka Panda view planning with frax dynamics and wrist camera viewcone.

Mirrors ``examples/arm/franka_fr3v2_viewplanning.py``: move the end-effector to a
goal position in minimum time while keeping workspace targets inside a
wrist-mounted camera field of view (continuous viewcone inequality).

Dynamics use ``FraxDynamics`` (full rigid-body ``frax`` forward dynamics). Task
constraints (viewcone, terminal EE pose, floor clearance) use symbolic Product
of Exponentials FK with FR3-class screw axes — the same PoE convention as the
FR3 view-planning example and a close match to the bundled Panda URDF. Viser
animation uses ``frax`` FK (``ee_transform`` / ``link_to_world_transforms``).

Requires:
    pip install openscvx[frax,lie]
"""

import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

try:
    import frax
except ImportError:
    print(
        "frax is not installed. Install with: pip install openscvx[frax,lie]",
        file=sys.stderr,
    )
    sys.exit(1)

import openscvx as ox
from openscvx.plotting import plot_controls, plot_scp_iterations

# =============================================================================
# Robot + dynamics adapter
# =============================================================================

robot = frax.load_panda()
dyn = ox.FraxDynamics(robot)
q, qd = dyn.states
(tau,) = dyn.controls
n_j = robot.num_joints

# =============================================================================
# PoE kinematics (same FR3-class parameters as franka_fr3v2_viewplanning.py)
# =============================================================================

d1 = 0.333
d3 = 0.316
a4 = 0.0825
d5 = 0.384
d7 = 0.088
d_ee = 0.107

z1 = d1
z35 = d1 + d3
z567 = d1 + d3 + d5

screw_axes = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [-z1, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [z35, 0.0, -a4, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [z567, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, d7, 0.0, 0.0, 0.0, -1.0],
    ]
)

T_home = np.array(
    [
        [1.0, 0.0, 0.0, d7],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, z567 - d_ee],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

T_chain = ox.Constant(np.eye(4))
joint_transforms = {}
for j in range(n_j):
    xi = ox.Constant(screw_axes[j])
    T_chain = T_chain @ ox.lie.SE3Exp(xi * q[j])
    joint_transforms[f"T_j{j + 1}"] = T_chain

T_ee = T_chain @ ox.Constant(T_home)
p_ee = ox.Concat(T_ee[0, 3], T_ee[1, 3], T_ee[2, 3])
R_ee = T_ee[:3, :3]

# =============================================================================
# Wrist camera (viewcone)
# =============================================================================

alpha_x = 6.0
alpha_y = 6.0
A_cone = np.diag(
    [
        1 / np.tan(np.pi / alpha_x),
        1 / np.tan(np.pi / alpha_y),
        0,
    ]
)
c = jnp.array([0, 0, 1])
norm_type = 2
R_sb = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

vp_targets = np.array(
    [
        [0.25, 0.00, 0.10],
        [0.35, 0.00, 0.10],
        [0.30, 0.05, 0.05],
    ]
)

# =============================================================================
# Discretisation + task
# =============================================================================

n = 9
total_time = 5.0
ee_tolerance = 0.01

home_ee_pos = np.array([0.3, 0.25, 0.25])
goal_ee_pos = np.array([0.3, -0.25, 0.25])
target = ox.Parameter("target", shape=(3,), value=goal_ee_pos)

q.initial = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
q.final = [("free", 0.0)] * n_j
qd.initial = np.zeros(n_j)
qd.final = np.zeros(n_j)

qd.max = np.minimum(np.asarray(qd.max), 2.5)
qd.min = -qd.max

# =============================================================================
# Constraints
# =============================================================================

constraints = []
for state in dyn.states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
for control in dyn.controls:
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

constraints.append(ox.ctcs(p_ee[2] >= 0.0))
constraints.append((ox.linalg.Norm(p_ee - target, ord=2) <= ee_tolerance).at([n - 1]))


def g_vp(p_target):
    p_s_s = R_sb @ R_ee.T @ (p_target - p_ee)
    return ox.linalg.Norm(A_cone @ p_s_s, ord=norm_type) - (c.T @ p_s_s)


constraints.append(
    ox.ctcs(
        ox.Vmap(
            lambda pose: g_vp(pose),
            batch=vp_targets,
        )
        <= 0.0
    )
)

# =============================================================================
# Initial guess (PoE IK — same seeding strategy as franka_fr3v2_viewplanning.py)
# =============================================================================

mean_vp = np.mean(vp_targets, axis=0)
boresight_body = np.array(R_sb).T @ np.array(c)


def look_at_quat(ee_pos, target_pos):
    """Quaternion (wxyz) aligning body boresight with (target_pos - ee_pos)."""
    d = target_pos - ee_pos
    d = d / np.linalg.norm(d)
    q_xyz = np.cross(boresight_body, d)
    q_w = np.sqrt(np.dot(boresight_body, boresight_body) * np.dot(d, d)) + np.dot(boresight_body, d)
    quat = np.hstack(([q_w], q_xyz))
    return quat / np.linalg.norm(quat)


q_home_orient = look_at_quat(home_ee_pos, mean_vp)
q_goal_orient = look_at_quat(goal_ee_pos, mean_vp)

q.guess = ox.init.ik_interpolation(
    keyframes=[
        (home_ee_pos, q_home_orient),
        (goal_ee_pos, q_goal_orient),
    ],
    nodes=[0, n - 1],
    screw_axes=screw_axes,
    T_home=T_home,
    angles_init=q.initial,
    angles_min=np.asarray(q.min),
    angles_max=np.asarray(q.max),
)
q.initial = np.clip(q.guess[0], np.asarray(q.min), np.asarray(q.max))
qd.guess = np.zeros((n, n_j))
tau.guess = np.zeros((n, robot.num_actuated_joints))

# =============================================================================
# Problem
# =============================================================================

time = ox.Time(
    initial=0.0,
    final=ox.Minimize(total_time),
    min=0.0,
    max=total_time,
)

problem = ox.Problem(
    dynamics=dyn,
    states=dyn.states,
    controls=dyn.controls,
    time=time,
    constraints=constraints,
    N=n,
    algorithm={
        "lam_vb": 1e1,
        "lam_vc": 1e2,
        "autotuner": ox.AugmentedLagrangian(eta_lambda=1e0),
    },
    algebraic_prop={
        "ee_position": p_ee,
        **{name: T for name, T in joint_transforms.items()},
    },
    float_dtype="float64",
)

problem.settings.prp.dt = 0.01


# =============================================================================
# Visualization (frax FK)
# =============================================================================


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


def visualize(results, robot, goal_ee_pos: np.ndarray) -> None:
    """Viser animation: stick-model arm, viewcone, EE trail, viewpoint markers."""
    import jaxlie

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        add_ghost_trajectory,
        add_position_marker,
        add_target_markers,
        add_viewcone,
        compute_velocity_colors,
        create_server,
    )

    q_traj = np.asarray(results.trajectory["q"])
    ee_pos = np.asarray(results.trajectory["ee_position"])
    n_frames = len(q_traj)
    keypoints, _ = _compute_panda_keypoints(q_traj, robot)
    n_segs = robot.num_joints + 1

    ee_quats = np.zeros((n_frames, 4))
    for t in range(n_frames):
        R = np.asarray(robot.ee_transform(q_traj[t]))[:3, :3]
        ee_quats[t] = jaxlie.SO3.from_matrix(R).wxyz

    ee_vel = np.gradient(ee_pos, np.asarray(results.trajectory["time"]).flatten(), axis=0, edge_order=2)
    ee_colors = compute_velocity_colors(ee_vel)

    server = create_server(ee_pos, show_grid=False)
    server.scene.add_grid("/grid", width=1.5, height=1.5, cell_size=0.2)
    server.scene.add_frame("/origin", axes_length=0.08, axes_radius=0.003)

    add_target_markers(server, np.asarray(goal_ee_pos).reshape(1, 3), radius=0.015, colors=[(255, 50, 50)])
    add_target_markers(
        server,
        vp_targets,
        radius=0.01,
        colors=[(50, 255, 50)] * len(vp_targets),
    )

    add_ghost_trajectory(server, ee_pos, ee_colors, point_size=0.005)
    _, update_trail = add_animated_trail(server, ee_pos, ee_colors, point_size=0.008)
    _, update_marker = add_position_marker(server, ee_pos, radius=0.012)

    _, update_viewcone = add_viewcone(
        server,
        ee_pos,
        ee_quats,
        half_angle_x=np.pi / alpha_x,
        half_angle_y=np.pi / alpha_y,
        scale=0.15,
        norm_type=norm_type,
        R_sb=np.array(R_sb),
        color=(80, 180, 200),
        opacity=0.3,
    )

    seg_col = np.full((n_segs, 2, 3), [200, 200, 200], dtype=np.uint8)

    def _build_segments(frame_idx: int) -> np.ndarray:
        return np.stack(
            [np.stack([keypoints[frame_idx, k], keypoints[frame_idx, k + 1]]) for k in range(n_segs)]
        ).astype(np.float32)

    arm_handle = server.scene.add_line_segments(
        "/panda_links",
        points=_build_segments(0),
        colors=seg_col,
        line_width=5.0,
    )

    def update_arm(frame_idx: int) -> None:
        arm_handle.points = _build_segments(frame_idx)

    traj_time = np.asarray(results.trajectory["time"]).flatten()
    add_animation_controls(
        server,
        traj_time,
        [update_trail, update_marker, update_viewcone, update_arm],
        loop=True,
    )
    server.sleep_forever()


if __name__ == "__main__":
    print("Franka Panda view planning — frax dynamics + wrist camera viewcone")
    print("=" * 60)
    print(f"Nodes: {n}  |  View targets: {len(vp_targets)}")
    print(f"Start EE (nominal): {home_ee_pos}")
    print(f"Goal EE target:      {target.value}")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    final_q = np.asarray(results.trajectory["q"][-1])
    final_ee = np.asarray(results.trajectory["ee_position"][-1])
    tgt = np.asarray(target.value)
    err = np.linalg.norm(final_ee - tgt)

    print()
    print("Results:")
    print(f"  Final joint angles [deg]: {np.round(np.rad2deg(final_q), 1)}")
    print(f"  Final EE: [{final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f}]")
    print(f"  Goal:     [{tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}]")
    print(f"  Position error: {err:.4f} m")

    plot_scp_iterations(results).show()
    plot_controls(results).show()

    print()
    print("Launching Viser visualization (Ctrl+C to exit)...")
    visualize(results, robot, goal_ee_pos)
