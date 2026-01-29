"""Brachistochrone problem: finding the fastest descent path.

This classic calculus of variations problem finds the curve of fastest descent
between two points under gravity. The solution demonstrates time-optimal
trajectory generation with:

- 2D position dynamics
- Speed dynamics under gravitational acceleration
- Angle control subject to bounds
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
from examples.plotting import (
    plot_brachistochrone_position,
    plot_brachistochrone_velocity,
)
from openscvx import Problem

n = 2
total_time = 2.0
g = 9.81

# Define state components
position = ox.State("position", shape=(2,))  # 2D position [x, y]
position.max = np.array([10.0, 10.0])
position.min = np.array([0.0, 0.0])
position.initial = np.array([0.0, 10.0])
position.final = [10.0, 5.0]

velocity = ox.State("velocity", shape=(1,))  # Scalar speed
velocity.max = np.array([10.0])
velocity.min = np.array([0.0])
velocity.initial = np.array([0.0])
velocity.final = [("free", 10.0)]

# Define control
theta = ox.Control("theta", shape=(1,))  # Angle from vertical
theta.max = np.array([100.5 * jnp.pi / 180])
theta.min = np.array([0.0])
theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

# Define list of all states (needed for Problem and constraints)
states = [position, velocity]
controls = [theta]

# Define dynamics as dictionary mapping state names to their derivatives
dynamics = {
    "position": ox.Concat(
        velocity[0] * ox.Sin(theta[0]),  # x_dot
        -velocity[0] * ox.Cos(theta[0]),  # y_dot
    ),
    "velocity": g * ox.Cos(theta[0]),
}

# Generate box constraints for all states
constraint_exprs = []
for state in states:
    constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

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
    constraints=constraint_exprs,
    N=n,
    licq_max=1e-8,
    autotuner=ox.ConstantProximalWeight(),
)

problem.settings.prp.dt = 0.01

problem.settings.scp.lam_prox = 1e1  # Weight on the Trust Reigon
problem.settings.scp.lam_cost = 1e0  # Weight on the Minimal Time Objective
problem.settings.scp.uniform_time_grid = True


plotting_dict = {}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    results.update(plotting_dict)

    plot_brachistochrone_position(results).show()
    plot_brachistochrone_velocity(results).show()
