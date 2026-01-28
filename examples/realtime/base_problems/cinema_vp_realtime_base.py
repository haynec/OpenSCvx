"""Base problem setup for real-time cinematic viewpoint planning.

This module defines the base optimization problem for real-time aerial
cinematography, designed to be imported by interactive visualization examples.
The problem includes:

- 6-DOF dynamics with parametric target positions
- Field-of-view constraints for visual tracking
- Sensor cone constraints for camera pointing
- Configured for real-time re-optimization with moving targets
"""

import os
import sys

import jax.numpy as jnp
import numpy as np
import numpy.linalg as la

# Add grandparent directory to path to import examples.plotting
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem

n = 12  # Number of Nodes
total_time = 40.0  # Total time for the simulation

# Define state components
position = ox.State("position", shape=(3,))  # 3D position [x, y, z]
position.max = np.array([200.0, 100, 50])
position.min = np.array([-100.0, -100, -10])
position.initial = np.array([8.0, -0.2, 2.2])
position.final = [("free", -10.0), ("free", 0), ("free", 2)]

velocity = ox.State("velocity", shape=(3,))  # 3D velocity [vx, vy, vz]
velocity.max = np.array([100, 100, 100])
velocity.min = np.array([-100, -100, -100])
velocity.initial = np.array([0, 0, 0])
velocity.final = [("free", 0), ("free", 0), ("free", 0)]

attitude = ox.State("attitude", shape=(4,))  # Quaternion [qw, qx, qy, qz]
attitude.max = np.array([1, 1, 1, 1])
attitude.min = np.array([-1, -1, -1, -1])
attitude.initial = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]
attitude.final = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]

angular_velocity = ox.State("angular_velocity", shape=(3,))  # Angular velocity [wx, wy, wz]
angular_velocity.max = np.array([10, 10, 10])
angular_velocity.min = np.array([-10, -10, -10])
angular_velocity.initial = [("free", 0), ("free", 0), ("free", 0)]
angular_velocity.final = [("free", 0), ("free", 0), ("free", 0)]

fuel = ox.State("fuel", shape=(1,))  # Fuel consumption
fuel.max = np.array([2000])
fuel.min = np.array([0])
fuel.initial = np.array([0])
fuel.final = [("minimize", 0)]

# Define control components
thrust_force = ox.Control("thrust_force", shape=(3,))  # Thrust forces [fx, fy, fz]
thrust_force.max = np.array([0, 0, 4.179446268 * 9.81])
thrust_force.min = np.array([0, 0, 0])
thrust_force.guess = np.repeat(np.array([[0.0, 0, 10]]), n, axis=0)

torque = ox.Control("torque", shape=(3,))  # Control torques [tau_x, tau_y, tau_z]
torque.max = np.array([18.665, 18.665, 0.55562])
torque.min = np.array([-18.665, -18.665, -0.55562])
torque.guess = np.zeros((n, 3))

# Initial keypoint position (will be controlled by user in realtime)
initial_kp_pose = np.array([13.0, 0.0, 2.0])
min_range = 4.0
max_range = 16.0

### View Planning Params ###
n_subs = 1  # Number of Subjects
alpha_x = 6.0  # Angle for the x-axis of Sensor Cone
alpha_y = 8.0  # Angle for the y-axis of Sensor Cone
A_cone = np.diag(
    [
        1 / np.tan(np.pi / alpha_x),
        1 / np.tan(np.pi / alpha_y),
        0,
    ]
)  # Conic Matrix in Sensor Frame
c = jnp.array([0, 0, 1])  # Boresight Vector in Sensor Frame
norm_type = "inf"
R_sb = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

# Define keypoint position as a parameter (can be updated in realtime)
kp_pose = ox.Parameter("kp_pose", shape=(3,), value=initial_kp_pose)

# Define list of all states (needed for Problem and constraints)
states = [position, velocity, attitude, angular_velocity, fuel]
controls = [thrust_force, torque]

# Create symbolic dynamics
m = 1.0  # Mass of the drone
g_const = -9.18
J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone

# Normalize quaternion for dynamics
q_norm = ox.linalg.Norm(attitude)
attitude_normalized = attitude / q_norm

# Define dynamics as dictionary mapping state names to their derivatives
J_b_inv = 1.0 / J_b
J_b_diag = ox.linalg.Diag(J_b)

# Concatenate all controls for fuel calculation
all_controls = ox.Concat(thrust_force, torque)

dynamics = {
    "position": velocity,
    "velocity": (1.0 / m) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
    + np.array([0, 0, g_const], dtype=np.float64),
    "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
    "angular_velocity": ox.linalg.Diag(J_b_inv)
    @ (torque - ox.spatial.SSM(angular_velocity) @ J_b_diag @ angular_velocity),
    "fuel": ox.linalg.Norm(all_controls),
}

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# View planning constraint using parameter keypoint pose
p_s_s = R_sb @ ox.spatial.QDCM(attitude).T @ (kp_pose - position)
vp_constraint = np.sqrt(2e1) * (ox.linalg.Norm(A_cone @ p_s_s, ord=norm_type) - (c.T @ p_s_s))

# Range constraints using parameter keypoint pose
min_range_constraint = min_range - ox.linalg.Norm(kp_pose - position)
max_range_constraint = ox.linalg.Norm(kp_pose - position) - max_range

constraints.extend(
    [
        ox.ctcs(vp_constraint <= 0.0),
        ox.ctcs(min_range_constraint <= 0.0),
        ox.ctcs(max_range_constraint <= 0.0),
    ]
)

# Initialize initial guess
position_final_values = np.array(
    [
        position.final[0][1] if isinstance(position.final[0], tuple) else position.final[0],
        position.final[1][1] if isinstance(position.final[1], tuple) else position.final[1],
        position.final[2][1] if isinstance(position.final[2], tuple) else position.final[2],
    ]
)
position_bar = np.linspace(position.initial, position_final_values, n)
velocity_bar = np.zeros((n, 3))
attitude_bar = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
angular_velocity_bar = np.zeros((n, 3))
fuel_bar = np.zeros((n, 1))

# Modify position to follow offset from keypoint
# Create array of positions, all offset from keypoint
offset = np.array([-5, 0.2, 0.2])
position_bar = np.tile(initial_kp_pose + offset, (n, 1))

# Modify attitude to point sensor at keypoint
b = R_sb @ np.array([0, 1, 0])
for k in range(n):
    a = initial_kp_pose - position_bar[k]
    # Determine the direction cosine matrix that aligns the z-axis of the sensor frame with the
    # relative position vector
    q_xyz = np.cross(b, a)
    q_w = np.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2) + np.dot(a, b)
    q_no_norm = np.hstack((q_w, q_xyz))
    q = q_no_norm / la.norm(q_no_norm)
    attitude_bar[k] = q

# Set all guesses
position.guess = position_bar
velocity.guess = velocity_bar
attitude.guess = attitude_bar
angular_velocity.guess = angular_velocity_bar
fuel.guess = fuel_bar

time_config = ox.Time(
    initial=0.0,
    final=total_time,
    min=0.0,
    max=total_time,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time_config,
    constraints=constraints,
    N=n,
    licq_max=1e-8,
)

problem.settings.scp.lam_prox = 4e0  # Weight on the Trust Region
problem.settings.scp.lam_cost = 1e-2  # Weight on the Minimal Fuel Objective
problem.settings.scp.lam_vc = 1e1  # Weight on the Virtual Control Objective

problem.settings.scp.ep_tr = 1e-6  # Trust Region Tolerance
problem.settings.scp.ep_vb = 1e-4  # Virtual Control Tolerance
problem.settings.scp.ep_vc = 1e-8  # Virtual Control Tolerance for CTCS
problem.settings.scp.lam_prox_adapt = 1.3  # Trust Region Adaptation Factor
problem.settings.scp.lam_prox_max_scaling_factor = 1e3  # Maximum Trust Region Weight

plotting_dict = {
    "n_subs": n_subs,
    "alpha_x": alpha_x,
    "alpha_y": alpha_y,
    "R_sb": R_sb,
    "init_poses": initial_kp_pose,
    "norm_type": norm_type,
    "min_range": min_range,
    "max_range": max_range,
    "moving_subject": False,  # Not moving in realtime version
    "kp_pose": kp_pose,
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    results.update(plotting_dict)
