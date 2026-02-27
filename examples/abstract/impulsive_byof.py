"""Impulsive control example using BYOF for dynamics_discrete.

This is a variant of impulsive.py that expresses the discrete (impulsive)
velocity update via bring-your-own-functions (byof) instead of symbolic
expressions. Same physics as the original:

- Continuous: position dot = velocity, velocity dot = acceleration
- Discrete (at impulsive nodes): position unchanged, velocity += (|dv|^1.5) * dv
  with a small clamp to avoid non-differentiability at dv=0.
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
from openscvx import ByofSpec, Problem
from openscvx.plotting import plot_controls, plot_states

n = 10

p = ox.State("position", shape=(1,))
p.max = np.array([1.0])
p.min = np.array([0.0])
p.initial = np.array([0.0])
p.final = np.array([1.0])
p.guess = np.linspace(p.initial, p.final, n).reshape(-1, 1)

v = ox.State("velocity", shape=(1,))
v.max = np.array([1.0])
v.min = np.array([-1.0])
v.initial = np.array([0.0])
v.final = np.array([0.0])
v.guess = np.linspace(v.initial, v.final, n).reshape(-1, 1)

dv = ox.Control(
    "delta_v",
    shape=(1,),
    impulsive=True,
    nodes=[0, n - 1],
)
dv.max = np.array([0.2])
dv.min = np.array([-0.2])
dv.guess = np.linspace(np.array([0]), np.array([0]), n)
dv.scaling_min = np.array([-0.2])
dv.scaling_max = np.array([0.2])

a = ox.Control("acceleration", shape=(1,))
a.max = np.array([0.01])
a.min = np.array([-0.01])
a.guess = np.linspace(np.array([0]), np.array([0]), n)
a.scaling_min = np.array([-1])
a.scaling_max = np.array([1])

# Continuous dynamics unchanged
dynamics = {
    "position": v,
    "velocity": a,
}

byof: ByofSpec = {
    "dynamics_discrete": {
        "position": lambda x, u, node, params: x[p.slice],
        "velocity": lambda x, u, node, params: x[v.slice]
        + jnp.power(jnp.maximum(jnp.abs(u[dv.slice]), 1e-6), 1.5) * u[dv.slice],
    },
}

states = [p, v]
controls = [dv, a]

constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

time = ox.Time(initial=0.0, final=ox.Minimize(10.0), min=0.0, max=20.0)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    byof=byof,
)

problem.algorithm.lam_prox = 1e-2  # Weight on the Trust Region
problem.algorithm.lam_vc = 5e1  # Weight on the Trust Region
problem.algorithm.lam_cost = 1e0  # Weight on the Minimal Time Objective
problem.algorithm.k_max = 100

plotting_dict = {}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    plot_states(results).show()
    plot_controls(results).show()
