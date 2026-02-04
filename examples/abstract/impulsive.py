
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
from openscvx.plotting import plot_states, plot_controls

from openscvx import Problem

n = 10

p           = ox.State("position", shape=(1,)) 
p.max       = np.array([10.0])
p.min       = np.array([0.0])
p.initial   = np.array([0.0])
p.final     = np.array([10.0])
p.guess     = np.linspace(p.initial, p.final, n).reshape(-1, 1)

v           = ox.State("velocity", shape=(1,))  # Scalar speed
v.max       = np.array([10.0])
v.min       = np.array([-10.0])
v.initial   = np.array([0.0])
v.final     = np.array([0.0])
v.guess     = np.linspace(v.initial, v.final, n).reshape(-1, 1)

a           = ox.Control("acceleration", shape=(1,))
a.max       = np.array([10.0])
a.min       = np.array([-10.0])
a.guess     = np.linspace(0, 0, n).reshape(-1, 1)

dynamics    = {
    "position": v,
    "velocity": a
}

states      = [p, v]
controls    = [a]

constraints = []
for state in states:
    constraints.extend([ ox.ctcs( state<=state.max ), ox.ctcs(state.min <= state) ]) 

time = ox.Time( 
    initial = 0.0,
    final=ox.Minimize(10.0),
    min = 0.0, 
    max = 100.0
 )

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
)

problem.settings.scp.lam_prox = 1e0  # Weight on the Trust Reigon
problem.settings.scp.lam_cost = 1e-1  # Weight on the Minimal Time Objective
problem.settings.scp.uniform_time_grid = False

plotting_dict = {}

if __name__ == "__main__":

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    plot_states(results).show()
    plot_controls(results).show()