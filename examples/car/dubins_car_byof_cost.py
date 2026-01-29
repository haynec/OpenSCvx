"""Dubins car path planning with custom convex cost using byof interface.

This example demonstrates how to add arbitrary convex costs to the convex
subproblem using the byof (bring-your-own-functions) interface. It's based
on the dubins_car example but adds a control effort cost.

The example includes:
- 2D position and heading dynamics
- Speed and angular rate control inputs
- Circular obstacle avoidance constraint
- Custom convex cost on control effort (via byof)
- Minimal time objective with free final heading
"""

import os
import sys

import jax.numpy as jnp
import numpy as np
import cvxpy as cp

# Add grandparent directory to path to import examples.plotting
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting import plot_dubins_car
from openscvx import Problem, ByofSpec

n = 8
total_time = 2.0  # Total simulation time

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
obs_center = ox.Parameter("obs_center", shape=(2,), value=np.array([-0.5, 0.0]))
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

# Define byof specification with convex costs
# The convex cost function receives:
# - ocp_vars: CVXPyVariables dataclass with all variables and parameters
# - settings: Config object with problem settings
# - params: Dict of user parameters (same as problem.parameters)
# Note: Control order is [speed, angular_rate, time_dilation]
#       speed is at index 0, angular_rate is at index 1
byof: ByofSpec = {
    "convex_costs": [
        # Add a quadratic cost on control effort (speed and angular rate)
        # This penalizes large control inputs to encourage smoother trajectories
        lambda ocp_vars, settings, params: sum(
            cp.sum_squares(settings.sim.inv_S_u @ (ocp_vars.u_nonscaled[i] - ocp_vars.u_bar[i])) + cp.sum_squares(settings.sim.inv_S_x @ (ocp_vars.x_nonscaled[i] - ocp_vars.x_bar[i])) for i in range(settings.scp.n)
        ),
    ]
}

# Build the problem (parameters auto-collected from Parameter objects)
time = ox.Time(
    initial=0.0,
    final=ox.Minimize(total_time),
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
    licq_max=1e-8,
    time_dilation_factor_min=0.02,
    byof=byof,  # Pass byof specification
    autotuner=ox.ConstantProximalWeight()
)

# Set solver parameters
problem.settings.prp.dt = 0.01
problem.settings.scp.lam_prox = 0e0
problem.settings.scp.lam_cost = 1e-1
problem.settings.scp.lam_vc = 1e3
problem.settings.scp.uniform_time_grid = True

# Enable CLI printing for optimization iterations
problem.settings.dev.printing = True
problem.settings.cvx.solver_args = {}

# Set parameter for angular rate weight in convex cost
problem.parameters["angular_rate_weight"] = 0.1

plotting_dict = {
    "obs_radius": problem.parameters["obs_radius"],
    "obs_center": problem.parameters["obs_center"],
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    plot_dubins_car(results, problem.settings).show()
