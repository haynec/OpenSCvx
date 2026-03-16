"""Simple example using `ox.stl.IntegerVariable` to enforce discrete-valued states.

This example sets up a 1D integrator where a second "gear" state must always
take one of three discrete values {-1, 0, 1}. The control drives the position
state `x`, while the gear state `g` is itself a continuous variable that is
penalized whenever it deviates from the allowed discrete set.

- State `x`: position, driven by control `u`
- State `g`: gear-like variable constrained to {-1, 0, 1} at every node
- `u` also acts as the rate of change for `g` (both share the same control)
- The IntegerVariable STL constraint is enforced over the full horizon

Run it directly to solve and print the resulting trajectory.
"""

import os
import sys

import numpy as np

import openscvx as ox
from openscvx import Problem

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)


# Discretization parameters
N = 10
total_time = 10.0

# Scalar position state
x = ox.State("x", shape=(1,))
x.min = np.array([0.0])
x.max = np.array([5.0])
x.initial = np.array([0.0])
x.final = [5.0]


# Control: drives position; gear integrates toward chosen discrete level
u = ox.Control("u", shape=(1,))
u.min = np.array([0.0])
u.max = np.array([10.0])
u.guess = np.ones((N, 1)) * 8.0

states = [x]
controls = [u]

# Dynamics: x_dot = u, g_dot = v
dynamics = {
    "x": u,
}

# Box constraints on states and controls
constraints = []
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max),
            ox.ctcs(state.min <= state),
        ]
    )

# IntegerVariable: g must equal one of the allowed levels at every node
allowed_levels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
levels = ox.stl.IntegerVariable(u, allowed_levels, c=1e-8)
constraints.append(levels.over((0, N - 1)))


# Time configuration
time = ox.Time(
    initial=0.0,
    final=ox.Minimize(total_time),
    min=0.0,
    max=total_time,
    uniform_time_grid=True,
    time_dilation_min=0.2,
    time_dilation_max=2.0,
)

problem = Problem(
    dynamics=dynamics,
    constraints=constraints,
    states=states,
    controls=controls,
    N=N,
    time=time,
    algorithm={"autotuner": ox.ConstantProximalWeight(), "ep_vc": 1e-3, "lam_cost": 1e1},
    discretizer={"dis_type": "ZOH"},
    float_dtype="float64",
)

problem.settings.prp.dt = 0.001


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
