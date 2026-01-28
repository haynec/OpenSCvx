"""Drone racing with continuous viewpoint constraints using polytope target arrangement.

This example demonstrates drone racing through polytope (polyhedron-shaped)
gates with sensor visibility constraints. The problem includes:

- 6-DOF rigid body dynamics (position, velocity, attitude quaternion, angular velocity)
- Sequential gate passage constraints
- Attitude planning for simultaneous gate navigation and visual tracking
- _Continuous_ sensor visibility constraints to keep targets in FOV
- Viewplanning targets are arranged in a polytope
- Minimal time objective
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
from openscvx.utils import gen_vertices, rot

n = 33  # Number of Nodes
total_time = 30.0  # Total time for the simulation

# Define state components
position = ox.State("position", shape=(3,))  # 3D position [x, y, z]
position.max = np.array([200.0, 100, 50])
position.min = np.array([-200.0, -100, 15])
position.initial = np.array([10.0, 0, 20])
position.final = [10.0, 0, 20]

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

# Define control components
thrust_force = ox.Control("thrust_force", shape=(3,))  # Thrust forces [fx, fy, fz]
thrust_force.max = np.array([0, 0, 4.179446268 * 9.81])
thrust_force.min = np.array([0, 0, 0])
thrust_force.guess = np.repeat(np.array([[0.0, 0, 10]]), n, axis=0)

torque = ox.Control("torque", shape=(3,))  # Control torques [tau_x, tau_y, tau_z]
torque.max = np.array([18.665, 18.665, 0.55562])
torque.min = np.array([-18.665, -18.665, -0.55562])
torque.guess = np.zeros((n, 3))


### Sensor Params ###
alpha_x = 6.0  # Angle for the x-axis of Sensor Cone
alpha_y = 6.0  # Angle for the y-axis of Sensor Cone
A_cone = np.diag(
    [
        1 / np.tan(np.pi / alpha_x),
        1 / np.tan(np.pi / alpha_y),
        0,
    ]
)  # Conic Matrix in Sensor Frame
c = jnp.array([0, 0, 1])  # Boresight Vector in Sensor Frame
norm_type = 2  # Norm Type
R_sb = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
### End Sensor Params ###

n_subs = 10
polytope_point = np.array(
    [
        [95.38, -54.62, 15.38],
        [95.38, -54.62, 24.62],
        [95.38, -45.38, 15.38],
        [95.38, -45.38, 24.62],
        [104.62, -54.62, 15.38],
        [104.62, -54.62, 24.62],
        [104.62, -45.38, 15.38],
        [104.62, -45.38, 24.62],
        [100.00, -52.85, 12.53],
        [100.00, -52.85, 27.47],
        [100.00, -47.15, 12.53],
        [100.00, -47.15, 27.47],
        [97.15, -57.47, 20.00],
        [97.15, -42.53, 20.00],
        [102.85, -57.47, 20.00],
        [102.85, -42.53, 20.00],
        [92.53, -50.00, 17.15],
        [92.53, -50.00, 22.85],
        [107.47, -50.00, 17.15],
        [107.47, -50.00, 22.85],
    ]
)
init_poses = polytope_point  # Shape: (20, 3)
### Gate Parameters ###
n_gates = 10
gate_centers = [
    np.array([59.436, 0.0000, 20.0000]),
    np.array([92.964, -23.750, 25.5240]),
    np.array([92.964, -29.274, 20.0000]),
    np.array([92.964, -23.750, 20.0000]),
    np.array([130.150, -23.750, 20.0000]),
    np.array([152.400, -73.152, 20.0000]),
    np.array([92.964, -75.080, 20.0000]),
    np.array([92.964, -68.556, 20.0000]),
    np.array([59.436, -81.358, 20.0000]),
    np.array([22.250, -42.672, 20.0000]),
]
radii = np.array([2.5, 1e-4, 2.5])
A_gate = rot @ np.diag(1 / radii) @ rot.T
A_gate_cen = []
for center in gate_centers:
    center[0] = center[0] + 2.5
    center[2] = center[2] + 2.5
    A_gate_cen.append(A_gate @ center)
nodes_per_gate = 3
gate_nodes = np.arange(nodes_per_gate, n, nodes_per_gate)
vertices = []
for center in gate_centers:
    vertices.append(gen_vertices(center, radii))


### End Gate Parameters ###


# Define list of all states (needed for Problem and constraints)
states = [position, velocity, attitude, angular_velocity]
controls = [thrust_force, torque]


# Symbolic sensor visibility constraint function
def g_vp(p_s_I, x_pos, x_quat):
    p_s_s = R_sb @ ox.spatial.QDCM(x_quat).T @ (p_s_I - x_pos)
    return ox.linalg.Norm(A_cone @ p_s_s, ord=norm_type) - (c.T @ p_s_s)


# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Add visibility constraints using Vmap for parallel evaluation
# Single CTCS constraint with vectorized evaluation over all polytope points
visibility_constraint = ox.ctcs(
    ox.Vmap(
        lambda pose: g_vp(pose, position, attitude),
        batch=init_poses,
    )
    <= 0.0
)
constraints.append(visibility_constraint)

# Add gate constraints using symbolic expressions
for node, cen in zip(gate_nodes, A_gate_cen):
    A_gate_const = A_gate
    cen_const = cen

    # Gate constraint: ||A @ pos - c||_inf <= 1
    gate_constraint = (
        (ox.linalg.Norm(A_gate_const @ position - cen_const, ord="inf") <= 1.0).convex().at([node])
    )
    constraints.append(gate_constraint)


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

dynamics = {
    "position": velocity,
    "velocity": (1.0 / m) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
    + np.array([0, 0, g_const], dtype=np.float64),
    "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
    "angular_velocity": ox.linalg.Diag(J_b_inv)
    @ (torque - ox.spatial.SSM(angular_velocity) @ J_b_diag @ angular_velocity),
}


# Generate initial guess for position trajectory through gates
position_bar = ox.init.linspace(
    keyframes=[position.initial] + gate_centers + [position.final],
    nodes=[0] + list(gate_nodes) + [n - 1],
)

# Generate attitude guess to point sensor at mean target position
b = R_sb @ np.array([0, 1, 0])
mean_target = np.mean(init_poses, axis=0)
attitude_bar = np.zeros((n, 4))
for k in range(n):
    a = mean_target - position_bar[k]
    # Determine the direction cosine matrix that aligns the z-axis of the sensor frame with the
    # relative position vector
    q_xyz = np.cross(b, a)
    q_w = np.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2) + np.dot(a, b)
    q_no_norm = np.hstack((q_w, q_xyz))
    attitude_bar[k] = q_no_norm / la.norm(q_no_norm)

# Set guesses
position.guess = position_bar
attitude.guess = attitude_bar
time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
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

problem.settings.scp.lam_prox = 2e0  # Weight on the Trust Reigon
problem.settings.scp.lam_vc = 4e1  # Weight on the Virtual Control Objective

problem.settings.scp.ep_tr = 1e-5  # Trust Region Tolerance

plotting_dict = {
    "vertices": vertices,
    "n_subs": n_subs,
    "alpha_x": alpha_x,
    "alpha_y": alpha_y,
    "R_sb": R_sb,
    "init_poses": init_poses,
    "norm_type": norm_type,
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    # Create both visualization servers (viser auto-assigns ports)
    traj_server = create_animated_plotting_server(
        results,
        thrust_key="thrust_force",
        viewcone_scale=10.0,
        show_control_plot="thrust_force",
        show_control_norm_plot="thrust_force",
    )
    scp_server = create_scp_animated_plotting_server(
        results,
        attitude_stride=3,
        frame_duration_ms=200,
    )

    # Keep both servers running
    traj_server.sleep_forever()
