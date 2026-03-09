"""Very simple example using the GMSR-based STL `Or` operator (`ox.stl.Or`).

This example sets up a 1D integrator with a single STL reachability specification:

- State `x` must reach either goal position `x_a` OR `x_b` by the final node.
- The STL formula is built with `ox.stl.Or` and enforced via `.at(N - 1)`.

The example is intentionally minimal and does not use any plotting utilities.
Run it directly to solve and print the resulting trajectory statistics.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import openscvx as ox
from openscvx import Problem
from openscvx.plotting import plot_states, plot_controls, plot_virtual_control_heatmap

# Ensure examples can be run directly by adding project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)


# Discretization parameters
N = 20
total_time = 2.0

# Scalar state: position x
x = ox.State("x", shape=(1,))
x.min = np.array([-2.0])
x.max = np.array([2.0])
x.initial = np.array([0.0])
x.final = [ox.Free(0.1)]

# Scalar control: acceleration u
u = ox.Control("u", shape=(1,))
u.min = np.array([-1.0])
u.max = np.array([1.0])
u.guess = np.zeros((N, 1))

states = [x]
controls = [u]

# Simple integrator dynamics: x_dot = u
dynamics = {
    "x": u,
}

# Box constraints for state bounds
constraints = []
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max),
            ox.ctcs(state.min <= state),
        ]
    )

# STL reachability specification: at the final node, be near either
# goal position x_a OR goal position x_b.
x_a = np.array([-1.0])
x_b = np.array([1.0])
radius = np.array([0.1])

reach_a = ox.linalg.Norm(x - x_a) <= radius
reach_b = ox.linalg.Norm(x - x_b) <= radius

reach_either = ox.stl.Or(reach_a, reach_b)
# Enforce the STL Or condition over the whole horizon
constraints.append(reach_either.over((N - 2, N - 1)))

# Time configuration (auto-created "time" trajectory)
time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
    uniform_time_grid=True,
)

problem = Problem(
    dynamics=dynamics,
    constraints=constraints,
    states=states,
    controls=controls,
    N=N,
    time=time,
    algorithm={"autotuner": ox.ConstantProximalWeight()},
    float_dtype="float64",
)

problem.algorithm.lam_vb = 1e2

problem.algorithm.lam_vc = 1e1

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    plot_states(results).show()
    plot_controls(results).show()
    plot_virtual_control_heatmap(results).show()