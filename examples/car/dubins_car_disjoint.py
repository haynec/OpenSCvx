"""Dubins car with disjoint waypoint visiting constraints.

This example demonstrates a Dubins car that must visit one of two waypoints
using a smooth max approximation for the disjoint constraint. The problem includes:

- 2D position and heading dynamics
- Disjoint waypoint visiting requirement (visit wp1 OR wp2)
- Smooth max approximation for non-convex OR constraint
- Loop closure constraint requiring similar start/end positions
- Minimal time objective
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
from examples.plotting import plot_dubins_car_disjoint
from openscvx import Problem

n = 8
total_time = 6.0  # Total simulation time

# Define state components
position = ox.State("position", shape=(2,))  # 2D position [x, y]
position.min = np.array([-5.0, -5.0])
position.max = np.array([5.0, 5.0])
position.initial = np.array([0, -2])
position.final = [ox.Free(0), ox.Free(-1.5)]
position.guess = np.linspace(position.initial, [0, 2], n)

theta = ox.State("theta", shape=(1,))  # Heading angle
theta.min = np.array([-2 * jnp.pi])
theta.max = np.array([2 * jnp.pi])
theta.initial = np.array([0])
theta.final = [("free", 0)]

# Define control components
speed = ox.Control("speed", shape=(1,))  # Forward speed
speed.min = np.array([0])
speed.max = np.array([10])
speed.guess = np.zeros((n, 1))

angular_rate = ox.Control("angular_rate", shape=(1,))  # Angular velocity
angular_rate.min = np.array([-5])
angular_rate.max = np.array([5])
angular_rate.guess = np.zeros((n, 1))

# Define list of all states and controls
states = [position, theta]
controls = [speed, angular_rate]
# Define Parameters for wp radius and center
wp1_center = ox.Parameter("wp1_center", shape=(2,), value=np.array([-2.1, 0.0]))
wp1_radius = ox.Parameter("wp1_radius", shape=(), value=0.5)
wp2_center = ox.Parameter("wp2_center", shape=(2,), value=np.array([1.9, 0.0]))
wp2_radius = ox.Parameter("wp2_radius", shape=(), value=0.5)

# Define dynamics as dictionary mapping state names to their derivatives
dynamics = {
    "position": ox.Concat(
        speed[0] * ox.Sin(theta[0]),  # x_dot
        speed[0] * ox.Cos(theta[0]),  # y_dot
    ),
    "theta": angular_rate[0],
}


# Create symbolic visit waypoint OR constraint
def create_visit_wp_OR_expr():
    # Visit wp1 or wp2 using smooth max
    d1 = ox.linalg.Norm(position - wp1_center)
    d2 = ox.linalg.Norm(position - wp2_center)
    v1 = wp1_radius - d1
    v2 = wp2_radius - d2
    alpha = 10.0  # smoothing parameter; higher = closer to max
    smooth_max = (1.0 / alpha) * ox.Log(ox.Exp(alpha * v1) + ox.Exp(alpha * v2))
    return -smooth_max


visit_wp_expr = create_visit_wp_OR_expr()

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Visit waypoint constraints using smooth max
constraints.append(ox.ctcs(visit_wp_expr <= 0.0).over((3, 5)))

constraints.append((ox.linalg.Norm(position.at(0) - position.at(-1)) <= 1.0).convex())

# Build the problem
time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=20,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
)
# Set solver parameters
problem.settings.scp.lam_vc = 6e2
problem.settings.scp.uniform_time_grid = True
plotting_dict = {
    "wp1_radius": problem.parameters["wp1_radius"],
    "wp1_center": problem.parameters["wp1_center"],
    "wp2_radius": problem.parameters["wp2_radius"],
    "wp2_center": problem.parameters["wp2_center"],
}
if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)
    plot_dubins_car_disjoint(results, problem.settings).show()
