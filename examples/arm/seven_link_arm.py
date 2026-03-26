"""7-DOF redundant arm with Product of Exponentials forward kinematics.

This example demonstrates trajectory optimization for a 7-DOF spatial arm
(similar to a Kuka iiwa / Franka Panda layout) using Lie algebra operations
for forward kinematics. The redundant kinematic structure means IK is needed
to generate the SCP initial guess.

- 7 revolute joints with alternating z-y rotation axes
- Product of Exponentials (PoE) forward kinematics using SE3Exp
- IK-generated initial guess via damped least-squares
- End-effector position tracking objective
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
from openscvx.plotting import plot_scp_convergence_histories, plot_scp_iterations

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

# Number of discretization nodes
n = 5
total_time = 3.0

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
torque.max = np.array([80.0, 80.0, 40.0, 40.0, 20.0, 10.0, 5.0])
torque.min = -torque.max

controls = [torque]

# =============================================================================
# Forward Kinematics using Product of Exponentials
# =============================================================================
# T_ee(q) = exp(ξ₁q₁) @ ... @ exp(ξ₇q₇) @ T_home

xi = ox.Constant(screw_axes[0])
T_ee = ox.lie.SE3Exp(xi * angle[0])
for i in range(1, N_JOINTS):
    xi = ox.Constant(screw_axes[i])
    T_ee = T_ee @ ox.lie.SE3Exp(xi * angle[i])
T_ee = T_ee @ ox.Constant(T_home)

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

# Target end-effector position
target = ox.Parameter("target", shape=(3,), value=np.array([0.3, 0.3, 0.5]))

# Box constraints
constraints = []
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max),
            ox.ctcs(state.min <= state),
        ]
    )

# End-effector target constraint at final node
ee_tolerance = 0.01  # 1cm tolerance
ee_target_constraint = (ox.linalg.Norm(p_ee - target, ord=2) <= ee_tolerance).at([n - 1])
constraints.append(ee_target_constraint)

# =============================================================================
# Initial Guesses (via IK)
# =============================================================================

from ik import ik_solve

# Solve IK for terminal joint angles that reach the target
q_terminal = ik_solve(
    screw_axes,
    T_home,
    target.value,
    q_min=angle.min,
    q_max=angle.max,
)

angle.guess = np.linspace(angle.initial, q_terminal, n)
velocity.guess = np.zeros((n, N_JOINTS))
torque.guess = np.zeros((n, N_JOINTS))

# =============================================================================
# Problem Setup
# =============================================================================

time = ox.Time(
    initial=0.0,
    final=total_time,
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
    algorithm={"lam_vb": 1e1},
    algebraic_prop={"ee_position": p_ee},
)

problem.settings.prp.dt = 0.01

if __name__ == "__main__":
    print("7-DOF Redundant Arm Trajectory Optimization with PoE FK")
    print("=" * 60)
    print(f"Link lengths: d1={d1}m, a2={a2}m, a3={a3}m, a4={a4}m")
    print(f"Home EE position: [{a2 + a3 + a4:.2f}, 0.00, {d1:.2f}]")
    print(f"Target position: {target.value}")
    print(f"IK solution [deg]: {np.round(np.rad2deg(q_terminal), 1)}")
    print()

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    # Extract results
    final_q = results.trajectory["angle"][-1]
    ee_pos = results.trajectory["ee_position"]  # (T, 3) from algebraic_prop

    tgt = target.value
    final_ee = ee_pos[-1]
    error = np.linalg.norm(final_ee - tgt)

    print()
    print("Results:")
    print(f"Final joint angles [deg]: {np.round(np.rad2deg(final_q), 1)}")
    print(f"Final EE position: [{final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f}]")
    print(f"Target position:   [{tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}]")
    print(f"Position error:    {error:.4f} m")

    plot_scp_iterations(results).show()
    plot_scp_convergence_histories(results).show()

    # =========================================================================
    # Viser 3D Arm Animation
    # =========================================================================

    import jaxlie
    import viser

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        add_ghost_trajectory,
        add_position_marker,
        add_target_markers,
        compute_velocity_colors,
        create_server,
    )

    # -- Compute joint keypoint positions at each propagation timestep ----------
    # Keypoint home positions and how many joints affect each:
    #   base [0,0,0]: 0 joints
    #   shoulder [0,0,d1]: 1 joint  (on z-axis, stays put under j1)
    #   elbow [a2,0,d1]: 2 joints   (on j3/j4 axes, unaffected by them)
    #   wrist [a2+a3,0,d1]: 4 joints (on j5/j6 axes)
    #   ee [a2+a3+a4,0,d1]: 7 joints

    keypoint_home = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, d1],
            [a2, 0.0, d1],
            [a2 + a3, 0.0, d1],
            [a2 + a3 + a4, 0.0, d1],
        ]
    )
    keypoint_n_joints = [0, 1, 2, 4, 7]

    angle_traj = np.asarray(results.trajectory["angle"])  # (T, 7)
    n_frames = len(angle_traj)

    # keypoints shape: (T, 5, 3)
    keypoints = np.zeros((n_frames, 5, 3))
    for t in range(n_frames):
        q_t = angle_traj[t]
        T_partial = np.eye(4)
        joint_idx = 0
        for k in range(5):
            # Advance the partial chain up to keypoint_n_joints[k]
            while joint_idx < keypoint_n_joints[k]:
                T_partial = T_partial @ np.asarray(
                    jaxlie.SE3.exp(screw_axes[joint_idx] * q_t[joint_idx]).as_matrix()
                )
                joint_idx += 1
            p_home = np.append(keypoint_home[k], 1.0)
            keypoints[t, k] = (T_partial @ p_home)[:3]

    # -- Create viser server ----------------------------------------------------

    server = create_server(ee_pos, show_grid=False)
    server.scene.add_grid("/grid", width=1.5, height=1.5)
    server.scene.add_frame("/origin", axes_length=0.1, axes_radius=0.003)

    # Target marker
    add_target_markers(server, [tgt], radius=0.015, colors=[(255, 50, 50)])

    # Ghost EE trajectory (faint full path)
    ee_colors = compute_velocity_colors(
        np.asarray(results.trajectory.get("velocity"))
    )
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
        [
            np.stack([keypoints[0, k], keypoints[0, k + 1]])
            for k in range(4)
        ]
    ).astype(np.float32)

    arm_handle = server.scene.add_line_segments(
        "/arm_links",
        points=init_points,
        colors=link_colors,
        line_width=5.0,
    )

    # Joint spheres
    joint_handles = []
    for k in range(5):
        h = server.scene.add_icosphere(
            f"/joint_{k}",
            radius=0.012,
            color=(220, 220, 220),
            position=keypoints[0, k].astype(np.float32),
        )
        joint_handles.append(h)

    def update_arm(frame_idx: int) -> None:
        pts = np.stack(
            [
                np.stack([keypoints[frame_idx, k], keypoints[frame_idx, k + 1]])
                for k in range(4)
            ]
        ).astype(np.float32)
        arm_handle.points = pts
        for k, h in enumerate(joint_handles):
            h.position = keypoints[frame_idx, k].astype(np.float32)

    # Animation controls
    traj_time = np.asarray(results.trajectory["time"])
    add_animation_controls(
        server,
        traj_time,
        [update_trail, update_marker, update_arm],
        loop=True,
    )

    print()
    print("Viser server running — open the URL above in your browser.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass
