"""3-link robotic arm with Product of Exponentials forward kinematics.

This example demonstrates trajectory optimization for a 3-link spatial arm
using Lie algebra operations for forward kinematics. The problem includes:

- 3 revolute joints with angle and velocity states
- Product of Exponentials (PoE) forward kinematics using SE3Exp
- End-effector position tracking objective
- Joint torque control inputs

The PoE formula computes forward kinematics as:
    T_ee(q) = exp(ξ₁q₁) @ exp(ξ₂q₂) @ exp(ξ₃q₃) @ T_home

where ξᵢ are the screw axes and T_home is the end-effector pose at q=0.

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

# Link lengths (meters)
L1 = 0.4  # Base to shoulder
L2 = 0.3  # Shoulder to elbow
L3 = 0.2  # Elbow to end-effector

# Joint inertias (simplified, kg*m^2)
inertia = np.array([0.05, 0.03, 0.01])

# Number of discretization nodes
n = 3
total_time = 2.0

# =============================================================================
# Screw Axes for Product of Exponentials
# =============================================================================
# Each screw axis ξ = [v; ω] where:
#   ω = unit rotation axis
#   v = -ω × q (q = point on the axis)
#
# Robot configuration at home (q=0):
#   - Joint 1: z-rotation at origin
#   - Joint 2: y-rotation at height L1
#   - Joint 3: y-rotation at height L1, reach L2 in x

# Screw axes as rows of a 3x6 matrix
screw_axes = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Joint 1: z-rotation at origin
        [-L1, 0.0, 0.0, 0.0, 1.0, 0.0],  # Joint 2: y-rotation at [0, 0, L1]
        [-L1, 0.0, L2, 0.0, 1.0, 0.0],  # Joint 3: y-rotation at [L2, 0, L1]
    ]
)

# Home configuration: end-effector at [L2+L3, 0, L1] with identity rotation
T_home = np.array(
    [
        [1.0, 0.0, 0.0, L2 + L3],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, L1],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# =============================================================================
# States
# =============================================================================

# Joint angles (3,)
angle = ox.State("angle", shape=(3,))
angle.max = np.array([np.pi, np.pi / 2, np.pi])
angle.min = np.array([-np.pi, -np.pi / 2, -np.pi])
angle.initial = np.array([0.0, 0.0, 0.0])
angle.final = [("free", 0.0), ("free", 0.0), ("free", 0.0)]

# Joint velocities (3,)
velocity = ox.State("velocity", shape=(3,))
velocity.max = np.array([5.0, 5.0, 5.0])
velocity.min = np.array([-5.0, -5.0, -5.0])
velocity.initial = np.array([0.0, 0.0, 0.0])
velocity.final = np.array([0.0, 0.0, 0.0])

states = [angle, velocity]

# =============================================================================
# Controls
# =============================================================================

# Joint torques (3,)
torque = ox.Control("torque", shape=(3,))
torque.max = np.array([10.0, 5.0, 2.0])
torque.min = np.array([-10.0, -5.0, -2.0])

controls = [torque]

# =============================================================================
# Forward Kinematics using Product of Exponentials
# =============================================================================
# T_ee(q) = exp(ξ₁q₁) @ exp(ξ₂q₂) @ exp(ξ₃q₃) @ T_home

# Symbolic screw axis expressions scaled by joint angles
xi_1 = ox.Constant(screw_axes[0])
xi_2 = ox.Constant(screw_axes[1])
xi_3 = ox.Constant(screw_axes[2])

twist_1 = xi_1 * angle[0]
twist_2 = xi_2 * angle[1]
twist_3 = xi_3 * angle[2]

# Exponential maps for each joint
T_01 = ox.lie.SE3Exp(twist_1)  # 4x4 transform
T_12 = ox.lie.SE3Exp(twist_2)
T_23 = ox.lie.SE3Exp(twist_3)

# Chain the transforms: T_ee = T_01 @ T_12 @ T_23 @ T_home
T_0_home = ox.Constant(T_home)
T_ee = T_01 @ T_12 @ T_23 @ T_0_home

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
    "velocity": I_inv * torque,  # Element-wise: qdd_i = tau_i / I_i
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

# End-effector target constraint at final node (commented out for debugging)
ee_tolerance = 0.01  # 1cm tolerance
ee_target_constraint = (ox.linalg.Norm(p_ee - target, ord=2) <= ee_tolerance).at([n - 1])
constraints.append(ee_target_constraint)

# =============================================================================
# Initial Guesses
# =============================================================================

np.random.seed(42)  # For reproducibility
# Terminal angles that reach the target (from workspace analysis)
q_terminal = np.deg2rad([47.8, -38.6, 62.4])
angle.guess = np.linspace(angle.initial, q_terminal, n)
velocity.guess = np.zeros((n, 3))
torque.guess = np.zeros((n, 3))

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
)

# Solver settings
problem.settings.prp.dt = 0.01
problem.settings.scp.lam_vb = 1e1

if __name__ == "__main__":
    print("3-Link Arm Trajectory Optimization with Product of Exponentials FK")
    print("=" * 60)
    print(f"Link lengths: L1={L1}m, L2={L2}m, L3={L3}m")
    print(f"Home EE position: [{L2 + L3:.2f}, 0.00, {L1:.2f}]")
    print(f"Target position: {target.value}")
    print()

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    # Extract final joint angles
    final_q = results.trajectory["angle"][-1]

    print()
    print("Results:")
    print(
        f"Final joint angles [deg]: q1={np.rad2deg(final_q[0]):.1f}, "
        f"q2={np.rad2deg(final_q[1]):.1f}, q3={np.rad2deg(final_q[2]):.1f}"
    )

    # Compute final EE position (using jaxlie for verification)
    import jaxlie

    def compute_ee_position(q_vals):
        T1 = jaxlie.SE3.exp(screw_axes[0] * q_vals[0]).as_matrix()
        T2 = jaxlie.SE3.exp(screw_axes[1] * q_vals[1]).as_matrix()
        T3 = jaxlie.SE3.exp(screw_axes[2] * q_vals[2]).as_matrix()
        T_final = T1 @ T2 @ T3 @ T_home
        return T_final[:3, 3]

    plot_scp_iterations(results).show()
    plot_scp_convergence_histories(results).show()
    tgt = target.value
    final_ee = compute_ee_position(final_q)
    error = np.linalg.norm(final_ee - tgt)

    print(f"Final EE position: [{final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f}]")
    print(f"Target position:   [{tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}]")
    print(f"Position error:    {error:.4f} m")
