"""Simplified drone racing using double integrator dynamics.

This example demonstrates time-optimal racing through gates using simplified
double integrator (point mass) dynamics instead of full 6-DOF dynamics. The problem includes:

- 3-DOF point mass dynamics (position and velocity only)
- Direct force control inputs (no attitude dynamics)
- Sequential gate passage constraints
- Minimal time objective
- Loop closure constraint
"""

import os
import sys

import jax.numpy as jnp
import numpy as np

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

n = 22  # Number of Nodes
total_time = 24.0  # Total time for the simulation

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

# Define control
force = ox.Control("force", shape=(3,))  # Control forces [fx, fy, fz]
f_max = 4.179446268 * 9.81
force.max = np.array([f_max, f_max, f_max])
force.min = np.array([-f_max, -f_max, -f_max])
initial_control = np.array([0.0, 0, 10])
force.guess = np.repeat(initial_control[np.newaxis, :], n, axis=0)

m = 1.0  # Mass of the drone
g_const = -9.18
J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone


### Gate Parameters ###
n_gates = 10
gate_centers = [
    np.array([59.436, 0.000, 20.0000]),
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
nodes_per_gate = 2
gate_nodes = np.arange(nodes_per_gate, n, nodes_per_gate)
vertices = []
for center in gate_centers:
    vertices.append(gen_vertices(center, radii))
### End Gate Parameters ###

# Define list of all states (needed for Problem and constraints)
states = [position, velocity]
controls = [force]

# Generate box constraints for all states
constraint_exprs = []
for state in states:
    constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Add gate constraints
for node, cen in zip(gate_nodes, A_gate_cen):
    A_gate_const = A_gate
    c_const = cen
    gate_constraint = (
        (ox.linalg.Norm(A_gate_const @ position - c_const, ord="inf") <= np.array([1.0]))
        .convex()
        .at([node])
    )
    constraint_exprs.append(gate_constraint)


# Define dynamics as dictionary mapping state names to their derivatives
dynamics = {
    "position": velocity,
    "velocity": (1 / m) * force + np.array([0, 0, g_const], dtype=np.float64),
}


# Generate initial guess for position trajectory through gates
position.guess = ox.init.linspace(
    keyframes=[position.initial] + gate_centers + [position.final],
    nodes=[0] + list(gate_nodes) + [n - 1],
)

t = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=t,
    constraints=constraint_exprs,
    N=n,
)

problem.settings.scp.ep_tr = 1e-3  # Trust Region Tolerance

plotting_dict = {"vertices": vertices}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    results.update(plotting_dict)

    # Create both visualization servers (viser auto-assigns ports)
    traj_server = create_animated_plotting_server(
        results,
        thrust_key="force",
        viewcone_scale=10.0,
    )
    scp_server = create_scp_animated_plotting_server(
        results,
        attitude_stride=3,
        frame_duration_ms=200,
    )

    # Keep both servers running
    traj_server.sleep_forever()
