"""7-DOF redundant arm pick-and-place with Product of Exponentials FK.

This example demonstrates trajectory optimization for a 7-DOF spatial arm
performing a full pick-and-place motion: home -> pre-grasp -> grasp ->
pre-place -> place. Uses Lie algebra operations for forward kinematics
and IK-generated initial guesses.

- 7 revolute joints with alternating z-y rotation axes
- Product of Exponentials (PoE) forward kinematics using SE3Exp
- Multi-waypoint pick-and-place trajectory
- EE position constraints at each waypoint
- Downward orientation constraints at grasp/place waypoints
- IK-generated initial guess via damped least-squares
- Joint torque control inputs

The PoE formula computes forward kinematics as:
    T_ee(q) = exp(ξ₁q₁) @ ... @ exp(ξ₇q₇) @ T_home

Requires jaxlie: pip install openscvx[lie]
"""

import os
import sys

import numpy as np

# Add grandparent directory to path to import examples.plotting
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem
from openscvx.plotting import plot_controls, plot_scp_iterations

# =============================================================================
# Robot Parameters
# =============================================================================

N_JOINTS = 7

# Link lengths (meters)
d1 = 0.340  # Base height
a2 = 0.300  # Shoulder to elbow
a3 = 0.250  # Elbow to wrist
a4 = 0.150  # Wrist to end-effector

# Joint inertias (simplified, kg*m^2) — decreasing from base to tip
inertia = np.array([0.08, 0.06, 0.05, 0.04, 0.02, 0.01, 0.005])

# Number of discretization nodes (parameterized by segments between waypoints)
nodes_per_segment = 2
total_time = 4.0

# =============================================================================
# Pick-and-Place Waypoints
# =============================================================================
# Orientation: Ry(90°) points body +x toward world -z (EE facing down)
q_down = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
q_identity = np.array([1.0, 0.0, 0.0, 0.0])

pre_height = 0.15  # height above grasp/place positions

grasp_pos = np.array([0.35, -0.25, 0.05])
place_pos = np.array([0.35, 0.25, 0.05])
home_pos = np.array([a2 + a3 + a4 - 0.01, 0.0, d1])
pre_grasp_pos = grasp_pos + np.array([0.0, 0.0, pre_height])
pre_place_pos = place_pos + np.array([0.0, 0.0, pre_height])

waypoint_names = ["home", "pre_grasp", "grasp", "pre_grasp", "pre_place", "place"]
waypoint_positions = [home_pos, pre_grasp_pos, grasp_pos, pre_grasp_pos, pre_place_pos, place_pos]
waypoint_orientations = [q_identity, q_down, q_down, q_down, q_down, q_down]

n_segments = len(waypoint_positions) - 1
n = nodes_per_segment * n_segments + 1
waypoint_nodes = [i * nodes_per_segment for i in range(len(waypoint_positions))]

# =============================================================================
# Screw Axes for Product of Exponentials
# =============================================================================
# Alternating z-y rotation axes (iiwa/Panda-like layout).
# Home configuration: arm extended along +x at height d1.
#
# Each screw axis ξ = [v; ω] where ω is the rotation axis and v = -ω × q
# for a point q on the joint axis.

screw_axes = np.array(
    [
        # Joint 1: z-rotation at origin (base yaw)
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        # Joint 2: y-rotation at [0, 0, d1] (shoulder pitch)
        [-d1, 0.0, 0.0, 0.0, 1.0, 0.0],
        # Joint 3: z-rotation at [a2, 0, d1] (upper arm roll)
        [0.0, -a2, 0.0, 0.0, 0.0, 1.0],
        # Joint 4: y-rotation at [a2, 0, d1] (elbow pitch)
        [-d1, 0.0, a2, 0.0, 1.0, 0.0],
        # Joint 5: z-rotation at [a2+a3, 0, d1] (forearm roll)
        [0.0, -(a2 + a3), 0.0, 0.0, 0.0, 1.0],
        # Joint 6: y-rotation at [a2+a3, 0, d1] (wrist pitch)
        [-d1, 0.0, a2 + a3, 0.0, 1.0, 0.0],
        # Joint 7: z-rotation at [a2+a3+a4, 0, d1] (tool roll)
        [0.0, -(a2 + a3 + a4), 0.0, 0.0, 0.0, 1.0],
    ]
)

# Home configuration: EE at [a2+a3+a4, 0, d1] with identity rotation
T_home = np.array(
    [
        [1.0, 0.0, 0.0, a2 + a3 + a4],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, d1],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# =============================================================================
# States
# =============================================================================

# Joint angles (7,)
angle = ox.State("angle", shape=(N_JOINTS,))
angle.max = np.deg2rad([170, 120, 170, 120, 170, 120, 175])
angle.min = -angle.max
angle.initial = np.zeros(N_JOINTS)
angle.final = [("free", 0.0)] * N_JOINTS

# Joint velocities (7,)
velocity = ox.State("velocity", shape=(N_JOINTS,))
velocity.max = np.full(N_JOINTS, 3.0)
velocity.min = -velocity.max
velocity.initial = np.zeros(N_JOINTS)
velocity.final = np.zeros(N_JOINTS)

states = [angle, velocity]

# =============================================================================
# Controls
# =============================================================================

# Joint torques (7,) — decreasing limits from base to tip
torque = ox.Control("torque", shape=(N_JOINTS,))
torque.max = np.array([8.0, 8.0, 4.0, 4.0, 2.0, 1.0, 0.5])
torque.min = -torque.max

controls = [torque]

# =============================================================================
# Forward Kinematics using Product of Exponentials
# =============================================================================
# T_ee(q) = exp(ξ₁q₁) @ ... @ exp(ξ₇q₇) @ T_home
#
# We build the FK chain incrementally so we can capture the intermediate
# transforms at each keypoint (base, shoulder, elbow, wrist, ee) for
# visualization via algebraic_prop.

joint_names = ["base", "shoulder", "elbow", "wrist", "ee"]
keypoint_after_joint = [0, 1, 2, 4, 7]  # how many joints precede each keypoint

T_chain = ox.Constant(np.eye(4))
joint_idx = 0
joint_transforms = {}
for name, n_joints in zip(joint_names, keypoint_after_joint):
    while joint_idx < n_joints:
        xi = ox.Constant(screw_axes[joint_idx])
        T_chain = T_chain @ ox.lie.SE3Exp(xi * angle[joint_idx])
        joint_idx += 1
    joint_transforms[name] = T_chain

T_ee = T_chain @ ox.Constant(T_home)

# Extract end-effector position from homogeneous transform
p_ee = ox.Concat(T_ee[0, 3], T_ee[1, 3], T_ee[2, 3])

# =============================================================================
# Dynamics (simplified second-order)
# =============================================================================
# Using simplified dynamics: I * qdd = tau
#
# Note: Full manipulator dynamics M(q)q̈ + C(q,q̇)q̇ + G(q) = τ are not needed
# here. This example demonstrates the Lie algebra functionality (SE3Exp for
# Product of Exponentials FK), which is independent of the dynamics model.

I_inv = ox.Constant(1.0 / inertia)

dynamics = {
    "angle": velocity,
    "velocity": I_inv * torque,
}

# =============================================================================
# Constraints
# =============================================================================

# Box constraints
constraints = []
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max),
            ox.ctcs(state.min <= state),
        ]
    )

# EE stays above the table
constraints.append(ox.ctcs(p_ee[2] >= 0.0))

# EE position constraint at each waypoint
ee_tolerance = 0.01  # 1cm tolerance
for i, (wp, node) in enumerate(zip(waypoint_positions, waypoint_nodes)):
    wp_param = ox.Parameter(f"waypoint_{waypoint_names[i]}", shape=(3,), value=wp)
    constraints.append((ox.linalg.Norm(p_ee - wp_param, ord=2) <= ee_tolerance).at([node]))

# Downward orientation constraint at task waypoints (pre_grasp, grasp, pre_place, place)
w, x, y, z = q_down
R_des = np.array(
    [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
)
ori_error = ox.lie.SO3Log(R_des.T @ T_ee[:3, :3])  # (3,) rotation error vector
ori_norm = ox.linalg.Norm(ori_error, ord=2)

# Loose tolerance during transit, tight during grasp/place approach
ori_tolerance = np.deg2rad(5.0)
ori_tolerance_tight = np.deg2rad(0.1)

# Loose: full task range (pre_grasp through place)
constraints.append((ori_norm <= ori_tolerance).over((waypoint_nodes[1], waypoint_nodes[-1])))
# Tight: pre_grasp -> grasp -> pre_grasp
constraints.append((ori_norm <= ori_tolerance_tight).over((waypoint_nodes[1], waypoint_nodes[3])))
# Tight: pre_place -> place
constraints.append((ori_norm <= ori_tolerance_tight).over((waypoint_nodes[4], waypoint_nodes[5])))

# =============================================================================
# Initial Guesses (via IK)
# =============================================================================

angle.guess = ox.init.ik_interpolation(
    keyframes=list(zip(waypoint_positions, waypoint_orientations)),
    nodes=waypoint_nodes,
    screw_axes=screw_axes,
    T_home=T_home,
    angles_init=angle.initial,
    angles_min=angle.min,
    angles_max=angle.max,
    sequential=True,
)
angle.initial = np.clip(angle.guess[0], angle.min, angle.max)
velocity.guess = np.zeros((n, N_JOINTS))
torque.guess = np.zeros((n, N_JOINTS))

# =============================================================================
# Problem Setup
# =============================================================================

time = ox.Time(
    initial=0.0,
    final=ox.Minimize(total_time),
    min=0.0,
    max=total_time,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    algorithm={"lam_vb": 1e1, "lam_vc": 1e2, "autotuner": ox.AugmentedLagrangian(eta_lambda=1e0)},
    algebraic_prop={
        "ee_position": p_ee,
        **{f"T_{name}": T for name, T in joint_transforms.items()},
    },
)

problem.settings.prp.dt = 0.01

if __name__ == "__main__":
    print("7-DOF Arm Pick-and-Place Trajectory Optimization with PoE FK")
    print("=" * 60)
    print(f"Link lengths: d1={d1}m, a2={a2}m, a3={a3}m, a4={a4}m")
    print(f"Nodes: {n} ({nodes_per_segment} per segment, {n_segments} segments)")
    for name, pos, node in zip(waypoint_names, waypoint_positions, waypoint_nodes):
        print(f"  {name:>12s} (node {node:2d}): {list(pos)}")
    print()

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    # Extract results
    final_q = results.trajectory["angle"][-1]
    ee_pos = results.trajectory["ee_position"]  # (T, 3) from algebraic_prop

    print()
    print("Results:")
    print(f"Final joint angles [deg]: {np.round(np.rad2deg(final_q), 1)}")
    for name, pos, node in zip(waypoint_names, waypoint_positions, waypoint_nodes):
        # Find the closest propagated frame to each waypoint node
        t_node = node / (n - 1)
        frame_idx = int(round(t_node * (len(ee_pos) - 1)))
        ee_at_wp = ee_pos[frame_idx]
        err = np.linalg.norm(ee_at_wp - pos)
        print(
            f"  {name:>12s}: pos=[{ee_at_wp[0]:.3f}, {ee_at_wp[1]:.3f}, {ee_at_wp[2]:.3f}]  "
            f"err={err:.4f}m"
        )

    plot_scp_iterations(results).show()
    plot_controls(results).show()

    # =========================================================================
    # Viser 3D Arm Animation
    # =========================================================================

    import jaxlie

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        add_ghost_trajectory,
        add_position_marker,
        add_target_markers,
        compute_velocity_colors,
        create_server,
    )

    # -- Extract joint keypoint positions from propagated transforms -------------
    # Each T_{name} is (T, 4, 4) from algebraic_prop. Apply the keypoint's
    # home position to get the world-frame position at each timestep.

    keypoint_home_positions = {
        "base": np.array([0.0, 0.0, 0.0]),
        "shoulder": np.array([0.0, 0.0, d1]),
        "elbow": np.array([a2, 0.0, d1]),
        "wrist": np.array([a2 + a3, 0.0, d1]),
        "ee": np.array([a2 + a3 + a4, 0.0, d1]),
    }

    n_frames = len(ee_pos)
    keypoints = np.zeros((n_frames, 5, 3))
    joint_quats = np.zeros((n_frames, 5, 4))
    for k, name in enumerate(joint_names):
        T_k = np.asarray(results.trajectory[f"T_{name}"])  # (T, 4, 4)
        p_home = np.append(keypoint_home_positions[name], 1.0)
        keypoints[:, k] = (T_k @ p_home)[:, :3]
        joint_quats[:, k] = np.array(
            [jaxlie.SO3.from_matrix(T_k[t, :3, :3]).wxyz for t in range(n_frames)]
        )

    # -- Create viser server ----------------------------------------------------

    server = create_server(ee_pos, show_grid=False)
    server.scene.add_grid("/grid", width=1.5, height=1.5, cell_size=0.25)
    server.scene.add_frame("/origin", axes_length=0.1, axes_radius=0.003)

    # Waypoint markers: grasp/place in red, pre-grasp/pre-place in orange, home in blue
    marker_colors = {
        "home": (100, 150, 255),
        "pre_grasp": (255, 180, 50),
        "grasp": (255, 50, 50),
        "pre_place": (255, 180, 50),
        "place": (255, 50, 50),
    }
    add_target_markers(
        server,
        waypoint_positions,
        radius=0.015,
        colors=[marker_colors[name] for name in waypoint_names],
    )

    # Ghost EE trajectory (faint full path)
    ee_colors = compute_velocity_colors(np.asarray(results.trajectory.get("velocity")))
    add_ghost_trajectory(server, ee_pos, ee_colors, point_size=0.005)

    # Animated EE trail
    _, update_trail = add_animated_trail(server, ee_pos, ee_colors, point_size=0.008)

    # Animated EE position marker
    _, update_marker = add_position_marker(server, ee_pos, radius=0.015)

    # Animated arm links (line segments between consecutive keypoints)
    # Per-segment colors: (N_segments, 2, 3) — same color at both endpoints
    link_rgb = np.array(
        [
            [180, 180, 180],  # base -> shoulder
            [100, 180, 255],  # shoulder -> elbow
            [100, 255, 150],  # elbow -> wrist
            [255, 200, 100],  # wrist -> ee
        ],
        dtype=np.uint8,
    )
    link_colors = np.stack([link_rgb, link_rgb], axis=1)  # (4, 2, 3)

    # Initial arm segments: (4, 2, 3)
    init_points = np.stack(
        [np.stack([keypoints[0, k], keypoints[0, k + 1]]) for k in range(4)]
    ).astype(np.float32)

    arm_handle = server.scene.add_line_segments(
        "/arm_links",
        points=init_points,
        colors=link_colors,
        line_width=5.0,
    )

    # Joint coordinate frames
    joint_frame_handles = []
    for k in range(5):
        h = server.scene.add_frame(
            f"/joint_{joint_names[k]}",
            wxyz=joint_quats[0, k].astype(np.float32),
            position=keypoints[0, k].astype(np.float32),
            axes_length=0.06,
            axes_radius=0.002,
        )
        joint_frame_handles.append(h)

    def update_arm(frame_idx: int) -> None:
        pts = np.stack(
            [np.stack([keypoints[frame_idx, k], keypoints[frame_idx, k + 1]]) for k in range(4)]
        ).astype(np.float32)
        arm_handle.points = pts
        for k, h in enumerate(joint_frame_handles):
            h.position = keypoints[frame_idx, k].astype(np.float32)
            h.wxyz = joint_quats[frame_idx, k].astype(np.float32)

    # Animation controls
    traj_time = np.asarray(results.trajectory["time"])
    add_animation_controls(
        server,
        traj_time,
        [update_trail, update_marker, update_arm],
        loop=True,
    )

    # Keep server running
    server.sleep_forever()
