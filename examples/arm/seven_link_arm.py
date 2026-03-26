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

    # Extract final joint angles
    final_q = results.trajectory["angle"][-1]

    print()
    print("Results:")
    print(f"Final joint angles [deg]: {np.round(np.rad2deg(final_q), 1)}")

    # Verify EE position using jaxlie
    from ik import poe_fk_position

    tgt = target.value
    final_ee = poe_fk_position(screw_axes, T_home, final_q)
    error = np.linalg.norm(final_ee - tgt)

    print(f"Final EE position: [{final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f}]")
    print(f"Target position:   [{tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}]")
    print(f"Position error:    {error:.4f} m")

    plot_scp_iterations(results).show()
    plot_scp_convergence_histories(results).show()
