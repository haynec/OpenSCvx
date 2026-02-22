"""Impulsive control example with mixed continuous/impulsive inputs.

This example demonstrates a simple 1D position/velocity system with:

- Continuous acceleration control
- Impulsive delta-v control applied only at selected nodes
- Time-dilation for minimum-time objective
- CTCS bounds on states
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
from openscvx import Problem
from openscvx.plotting import plot_controls, plot_states

n = 10

p = ox.State("position", shape=(1,))
p.max = np.array([1.0])
p.min = np.array([0.0])
p.initial = np.array([0.0])
p.final = np.array([1.0])
p.guess = np.linspace(p.initial, p.final, n).reshape(-1, 1)

v = ox.State("velocity", shape=(1,))  # Scalar speed
v.max = np.array([1.0])
v.min = np.array([-1.0])
v.initial = np.array([0.0])
v.final = np.array([0.0])
v.guess = np.linspace(v.initial, v.final, n).reshape(-1, 1)

dv = ox.Control(
    "delta_v",
    shape=(1,),
    impulsive=True,
    allocation_matrix=np.array([[0], [1]]),
    nodes=[0, n - 1],
)
dv.max = np.array([1.0])
dv.min = np.array([-1.0])
dv.guess = np.linspace(np.array([0]), np.array([0]), n)
dv.scaling_min = np.array([-1])
dv.scaling_max = np.array([1])

a = ox.Control("acceleration", shape=(1,))
a.max = np.array([0.01])
a.min = np.array([-0.01])
a.guess = np.linspace(np.array([0]), np.array([0]), n)
a.scaling_min = np.array([-1])
a.scaling_max = np.array([1])

dynamics = {
    "position": v,
    "velocity": a,
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
)

problem.algorithm.lam_prox = 1e-2  # Weight on the Trust Reigon
problem.algorithm.lam_vc = 5e1  # Weight on the Trust Reigon
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
