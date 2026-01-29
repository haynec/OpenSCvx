"""Cinematic viewpoint planning for aerial filming.

This example demonstrates optimal trajectory planning for a quadrotor performing
aerial cinematography with viewpoint constraints. The problem includes:

- 6-DOF dynamics with fuel consumption tracking
- _Continuous_ field-of-view (FOV) constraints to keep moving target in view
- Sensor pointing constraints using camera cone geometry
- Fuel-optimal trajectory generation
- Attitude planning to maintain visual coverage
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
from examples.plotting_viser import (
    create_animated_plotting_server,
    create_scp_animated_plotting_server,
)
from openscvx import Problem
from openscvx.utils import get_kp_pose

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

# Define time (needed for time-dependent constraints)
# Time is a State subclass, so it can be used directly in expressions
time = ox.Time(initial=0.0, final=total_time, min=0.0, max=total_time)

# Define control components
thrust_force = ox.Control("thrust_force", shape=(3,))  # Thrust forces [fx, fy, fz]
thrust_force.max = np.array([0, 0, 4.179446268 * 9.81])
thrust_force.min = np.array([0, 0, 0])
thrust_force.guess = np.repeat(np.array([[0.0, 0, 10]]), n, axis=0)

torque = ox.Control("torque", shape=(3,))  # Control torques [tau_x, tau_y, tau_z]
torque.max = np.array([18.665, 18.665, 0.55562])
torque.min = np.array([-18.665, -18.665, -0.55562])
torque.guess = np.zeros((n, 3))

init_pose = np.array([13.0, 0.0, 2.0])
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


# Define list of all states (needed for Problem and constraints)
states = [position, velocity, attitude, angular_velocity, fuel, time]
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
    "time": 1.0,  # Real time derivative
}


# Symbolic implementation of get_kp_pose function
def get_kp_pose_symbolic(t_expr, init_pose):
    loop_time = 40.0
    loop_radius = 20.0

    # Convert the trajectory parameters to symbolic constants
    loop_time_const = loop_time
    loop_radius_const = loop_radius
    two_pi_const = 2 * np.pi
    init_pose_const = init_pose
    half_const = 0.5

    # Compute symbolic trajectory: t_angle = t / loop_time * (2 * pi)
    t_angle = t_expr / loop_time_const * two_pi_const

    # x = loop_radius * sin(t_angle)
    x_pos = loop_radius_const * ox.Sin(t_angle)

    # y = x * cos(t_angle)
    y_pos = x_pos * ox.Cos(t_angle)

    # z = 0.5 * x * sin(t_angle)
    z_pos = half_const * x_pos * ox.Sin(t_angle)

    # Stack into position vector and add initial pose
    kp_trajectory = ox.Concat(x_pos, y_pos, z_pos) + init_pose_const
    return kp_trajectory


# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Get the symbolic keypoint pose based on time
kp_pose_symbolic = get_kp_pose_symbolic(time[0], init_pose)

# View planning constraint using symbolic keypoint pose
p_s_s = R_sb @ ox.spatial.QDCM(attitude).T @ (kp_pose_symbolic - position)
vp_constraint = np.sqrt(2e1) * (ox.linalg.Norm(A_cone @ p_s_s, ord=norm_type) - (c.T @ p_s_s))

# Range constraints using symbolic keypoint pose
min_range_constraint = min_range - ox.linalg.Norm(kp_pose_symbolic - position)
max_range_constraint = ox.linalg.Norm(kp_pose_symbolic - position) - max_range

constraints.extend(
    [
        ox.ctcs(vp_constraint <= 0.0),
        ox.ctcs(min_range_constraint <= 0.0),
        ox.ctcs(max_range_constraint <= 0.0),
    ]
)


# Initialize initial guess (will be modified by symbolic trajectory)
# Extract final values from tuples (position.final has free values, use their default guesses)
position_final_values = np.array(
    [
        position.final[0][1] if isinstance(position.final[0], tuple) else position.final[0],
        position.final[1][1] if isinstance(position.final[1], tuple) else position.final[1],
        position.final[2][1] if isinstance(position.final[2], tuple) else position.final[2],
    ]
)
position_bar = np.linspace(position.initial, position_final_values, n)
velocity_bar = np.zeros((n, 3))  # Velocity is free at final, start with zeros
attitude_bar = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
angular_velocity_bar = np.zeros((n, 3))
fuel_bar = np.zeros((n, 1))  # Fuel starts at 0 and is minimized

# Time guess for trajectory computation
time_bar = np.linspace(0, total_time, n)

# Modify position to follow offset from keypoint trajectory
position_bar = get_kp_pose(time_bar, init_pose) + np.array([-5, 0.2, 0.2])

# Modify attitude to point sensor at targets
b = R_sb @ np.array([0, 1, 0])
for k in range(n):
    kp = get_kp_pose(time_bar[k], init_pose)
    a = kp - position_bar[k]
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

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,  # Time is already defined above as ox.Time
    constraints=constraints,
    N=n,
    licq_max=1e-8,
)


problem.settings.scp.lam_prox = 4e0  # Weight on the Trust Reigon
problem.settings.scp.ep_tr = 1e-6  # Trust Region Tolerance

plotting_dict = {
    "n_subs": n_subs,
    "alpha_x": alpha_x,
    "alpha_y": alpha_y,
    "R_sb": R_sb,
    "norm_type": norm_type,
    "min_range": min_range,
    "max_range": max_range,
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    # Compute moving target trajectory using the post-processed time array
    traj_time = results.trajectory["time"].flatten()
    target_trajectory = np.asarray(get_kp_pose(traj_time, init_pose))
    plotting_dict["init_poses"] = [target_trajectory]  # List of target trajectories

    results.update(plotting_dict)

    # Create both visualization servers (viser auto-assigns ports)
    traj_server = create_animated_plotting_server(
        results,
        thrust_key="thrust_force",
        viewcone_scale=10.0,
    )
    scp_server = create_scp_animated_plotting_server(
        results,
        attitude_stride=3,
        frame_duration_ms=200,
    )

    # Keep both servers running
    traj_server.sleep_forever()
