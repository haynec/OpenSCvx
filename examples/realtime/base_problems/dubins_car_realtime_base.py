"""Dubins car path planning with obstacle avoidance.

This example demonstrates minimum-time path planning for a Dubins car
(car-like vehicle with minimum turning radius) navigating around a
circular obstacle. The problem includes:

- 2D position and heading dynamics
- Speed and angular rate control inputs
- Circular obstacle avoidance constraint
- Minimal time objective with free final heading
- Parameter updates for multiple scenarios
"""

import os
import sys

import jax.numpy as jnp
import numpy as np

# Add repo root to path (this file lives in examples/realtime/base_problems/)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(_this_dir)))
sys.path.append(_repo_root)

import openscvx as ox
from openscvx import Problem

n = 8
total_time = 1.2  # Total simulation time

# Define state components
position = ox.State("position", shape=(2,))  # 2D position [x, y]
position.min = np.array([-5.0, -5.0])
position.max = np.array([5.0, 5.0])
position.initial = np.array([0, -2])
position.final = np.array([0, 2])

theta = ox.State("theta", shape=(1,))  # Heading angle
theta.min = np.array([-2 * jnp.pi])
theta.max = np.array([2 * jnp.pi])
theta.initial = np.array([0])
theta.final = [ox.Free(0)]

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

# Define Parameters with initial values for obstacle radius and center
obs_center = ox.Parameter("obs_center", shape=(2,), value=np.array([-2.01, 0.0]))
obs_radius = ox.Parameter("obs_radius", shape=(), value=1.0)

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Add obstacle avoidance constraint
constraints.append(ox.ctcs(obs_radius <= ox.linalg.Norm(position - obs_center)))


# Define dynamics as dictionary mapping state names to their derivatives
dynamics = {
    "position": ox.Concat(
        speed[0] * ox.Sin(theta[0]),  # x_dot
        speed[0] * ox.Cos(theta[0]),  # y_dot
    ),
    "theta": angular_rate[0],
}


# Build the problem (parameters auto-collected from Parameter objects)
time = ox.Time(
    initial=0.0,
    final=ox.Minimize(total_time),
    min=0.0,
    max=20,
    time_dilation_min=0.02 * total_time,
    uniform_time_grid=True,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    licq_max=1e-6,
    algorithm={
        "lam_prox": 1e-1,
        "lam_cost": 4e-2,
        "lam_vc": 2e0,
        "autotuner": ox.ConstantProximalWeight(),
    },
)

plotting_dict = {
    "obs_radius": problem.parameters["obs_radius"],
    "obs_center": problem.parameters["obs_center"],
}

if __name__ == "__main__":
    problem.initialize()
    problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

#     plot_dubins_car(results, problem.settings).show()

#     # Second run with different parameters
#     problem.reset()
#     problem.parameters["obs_center"] = np.array([0.5, 0.0])
#     total_time = 0.7  # Adjust total time for second run
#     problem.algorithm.lam_prox = 1e-2
#     problem.algorithm.lam_vc = 1e0
#     problem.algorithm.lam_cost = 1e-3
#     position.guess = np.linspace([0, -2], [0, 2], n)
#     theta.guess = np.zeros((n, 1))
#     speed.guess = np.zeros((n, 1))
#     angular_rate.guess = np.zeros((n, 1))

#     plotting_dict["obs_center"] = np.array([0.5, 0.0])

#     results = problem.solve()
#     results = problem.post_process()
#     results.update(plotting_dict)
#     plot_dubins_car(results, problem.settings).show()
