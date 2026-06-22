"""Very simple example using the GMSR-based STL `Or` operator (`ox.stl.Or`).

This example sets up a 1D integrator with a single STL reachability specification:

- State `x` must reach either goal position `x_a` OR `x_b` by the final node.
- The STL formula is built with `ox.stl.Or` and enforced via `.at(N - 1)`.

Two solvers are compared side-by-side:
- PTR with an STL Or constraint (`ox.stl.Or`).
- ProxConvex with the same OR condition encoded as an `SRComposite`.

Run it directly to solve both and print the resulting trajectory statistics.
"""

import os
import sys

import numpy as np

import openscvx as ox
from openscvx import Problem
from openscvx.algorithms.scvx.prox_convex import ProxConvex, SRComposite
from openscvx.plotting import plot_controls, plot_states, plot_virtual_control_heatmap
from openscvx.solvers.cvxpy_ptr_solver import CVXPyProxConvexSolver
from openscvx.symbolic.lowerers.jax.stl import OR

# Ensure examples can be run directly by adding project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)


# Discretization parameters
N = 20
total_time = 2.0

# Goal positions and reach radius
x_a = np.array([-1.0])
x_b = np.array([1.0])
radius = np.array([0.1])

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
dynamics = {"x": u}

# Time configuration
time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
    uniform_time_grid=True,
)

# --- PTR problem: OR encoded as an STL constraint ---------------------------

ptr_constraints = []
for state in states:
    ptr_constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

reach_a = ox.linalg.Norm(x - x_a) <= radius
reach_b = ox.linalg.Norm(x - x_b) <= radius
reach_either = ox.stl.Or(reach_a, reach_b)
ptr_constraints.append(reach_either.over((N - 2, N - 1), penalty="squared_relu"))

problem = Problem(
    dynamics=dynamics,
    constraints=ptr_constraints,
    states=states,
    controls=controls,
    N=N,
    time=time,
    float_dtype="float64",
)

# --- ProxConvex problem: OR encoded as an SRComposite -----------------------
#
# r_i = distance from the final state to goal i minus the reach radius.
# s(R) = OR(R): approximately zero iff at least one r_i ≤ 0.
# Minimising s drives the trajectory to satisfy at least one reach condition.

composite = SRComposite(
    s=lambda R, p: OR(R),
    r=[
        ox.linalg.Norm(x - x_a) - float(radius[0]),
        ox.linalg.Norm(x - x_b) - float(radius[0]),
    ],
    nodes=N - 1,
)

prox_constraints = []
for state in states:
    prox_constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

prox_problem = Problem(
    dynamics=dynamics,
    constraints=prox_constraints,
    states=states,
    controls=controls,
    N=N,
    time=time,
    algorithm=ProxConvex(
        composite=composite, autotuner=ox.AugmentedLagrangian()
    ),
    solver=CVXPyProxConvexSolver(),
    float_dtype="float64",
)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # PTR run
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    plot_states(results).show()
    plot_controls(results).show()
    plot_virtual_control_heatmap(results).show()

    print("\n--- PTR (STL Or) result ---")
    print(f"final state x: {results.x[-1]}")
    dist_a_ptr = float(((results.x[-1, :1] - x_a) ** 2).sum() ** 0.5)
    dist_b_ptr = float(((results.x[-1, :1] - x_b) ** 2).sum() ** 0.5)
    print(f"distance to goal a: {dist_a_ptr:.4f}  (radius {float(radius[0])})")
    print(f"distance to goal b: {dist_b_ptr:.4f}  (radius {float(radius[0])})")
    print(f"OR reached: {min(dist_a_ptr, dist_b_ptr) <= float(radius[0])}")

    # ProxConvex run
    prox_problem.initialize()
    prox_results = prox_problem.solve()
    prox_results = prox_problem.post_process()

    print("\n--- ProxConvex result ---")
    print(f"final state x: {prox_results.x[-1]}")
    dist_a = float(((prox_results.x[-1, :1] - x_a) ** 2).sum() ** 0.5)
    dist_b = float(((prox_results.x[-1, :1] - x_b) ** 2).sum() ** 0.5)
    print(f"distance to goal a: {dist_a:.4f}  (radius {float(radius[0])})")
    print(f"distance to goal b: {dist_b:.4f}  (radius {float(radius[0])})")
    print(f"OR reached: {min(dist_a, dist_b) <= float(radius[0])}")

    plot_states(prox_results).show()
    plot_controls(prox_results).show()
