"""Proximity Operations (ProxOps) using Clohessy-Wiltshire dynamics.

This example demonstrates optimal trajectory generation for spacecraft
proximity operations and docking using the Clohessy-Wiltshire (CW) equations
for relative motion in a circular orbit.
See [Clohessy-Wiltshire equations](https://en.wikipedia.org/wiki/Clohessy%E2%80%93Wiltshire_equations)
for further details.
The problem includes:

- 3D relative position and velocity dynamics (CW equations)
- Fuel-optimal control (minimize delta-v)
- Thrust magnitude constraints
- Approach cone constraint for safe docking corridor
- Final docking at target (origin)

!!! tip "The CW frame convention:"
    - x: radial direction (outward from Earth)
    - y: along-track direction (velocity direction)
    - z: cross-track direction (normal to orbit plane)
"""

import os
import sys

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
from openscvx.plotting import plot_scp_iterations
from openscvx.plotting.viser import add_glideslope_cone

# Problem parameters
n_nodes = 5  # Number of discretization nodes
total_time = 180.0  # Total maneuver time in seconds

# Orbital parameters (ISS-like orbit at ~400 km altitude)
mu = 3.986004418e14  # Earth gravitational parameter [m^3/s^2]
a = 6.778e6  # Semi-major axis [m] (Earth radius + 400 km)
n = np.sqrt(mu / a**3)  # Mean motion [rad/s] (~0.00113 rad/s)

# Define state components
# Position in CW frame [x, y, z] in meters
position = ox.State("position", shape=(3,))
position.max = np.array([100.0, 100.0, 100.0])
position.min = np.array([-100.0, -100.0, -100.0])
position.initial = np.array([0.0, -100.0, 0.0])  # Start 100m behind target (V-bar position)
position.final = np.array([0.0, 0.0, 0.0])  # Dock at origin

# Velocity in CW frame [vx, vy, vz] in m/s
velocity = ox.State("velocity", shape=(3,))
v_max = 2.0  # Maximum relative velocity [m/s]
velocity.max = np.array([v_max, v_max, v_max])
velocity.min = np.array([-v_max, -v_max, -v_max])
velocity.initial = [0.0, 2.0, 0.0]
velocity.final = np.array([0.0, 0.0, 0.0])  # Zero relative velocity at docking
velocity.guess = np.zeros((n_nodes, 3))

# Define control: acceleration from thrusters [ax, ay, az] in m/s^2
accel = ox.Control("accel", shape=(3,))
a_max = 0.1  # Maximum acceleration [m/s^2] (typical for reaction control thrusters)
accel.max = np.array([a_max, a_max, a_max])
accel.min = np.array([-a_max, -a_max, -a_max])
accel.guess = np.zeros((n_nodes, 3))

# Define list of all states and controls
states = [position, velocity]
controls = [accel]

# Generate constraints
constraints = []

# Box constraints for states
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max),
            ox.ctcs(state.min <= state),
        ]
    )

# R-bar approach cone constraint (from below, -x direction)
# Enforces sqrt(y^2 + z^2) <= tan(theta) * (-x)
# This requires the spacecraft to approach from negative x (below target)
# and stay within a cone centered on the -x axis
cone_half_angle = 20 * np.pi / 180  # 20 degree half-angle
constraints.append(
    ox.ctcs(ox.linalg.Norm(position[1:]) <= np.tan(cone_half_angle) * (-position[0])).over(
        (n_nodes - 3, n_nodes - 1)
    )
)
# Enforce entrance to the cone at safe distance
constraints.append((-position[0] >= 20.0).at([n_nodes - 3]))

# Clohessy-Wiltshire dynamics
dynamics = {
    "position": velocity,
    "velocity": ox.Concat(
        3 * n**2 * position[0] + 2 * n * velocity[1] + accel[0],
        -2 * n * velocity[0] + accel[1],
        -(n**2) * position[2] + accel[2],
    ),
}

# Time configuration (free final time to minimize fuel)
time = ox.Time(
    initial=0.0,
    final=("free", total_time),
    min=0.0,
    max=total_time,
)

# Build the problem
problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n_nodes,
)

# Solver settings (FOH is default, no need to set explicitly)
problem.settings.dis.dis_type = "ZOH"
problem.settings.scp.lam_vb = 1e0

# Plotting metadata
plotting_dict = {
    "mean_motion": n,
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    # Plot results
    plot_scp_iterations(results).show()

    # Create animation
    traj_server = create_animated_plotting_server(results, thrust_key="accel", show_grid=False)

    # Add R-bar approach cone (opens in -x direction)
    add_glideslope_cone(
        traj_server,
        apex=(0, 0, 0),
        height=20.0,  # Cone extends 100m in -x direction
        glideslope_angle_deg=cone_half_angle * 180 / np.pi,
        axis=(-1, 0, 0),  # R-bar: negative radial direction
        color=(200, 80, 80),
        opacity=0.5,
    )

    # Create SCP iteration visualization
    scp_server = create_scp_animated_plotting_server(
        results,
        frame_duration_ms=200,
        scene_scale=1.0,
        show_grid=False,
    )

    # Add R-bar approach cone to SCP visualization
    add_glideslope_cone(
        scp_server,
        apex=(0, 0, 0),
        height=20.0,
        glideslope_angle_deg=cone_half_angle * 180 / np.pi,
        axis=(-1, 0, 0),
        color=(200, 80, 80),
        opacity=0.5,
    )

    scp_server.sleep_forever()
