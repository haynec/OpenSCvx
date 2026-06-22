"""Dubins car with disjoint waypoint visiting via ProxConvex SR composite.

This example demonstrates a Dubins car that must visit one of two waypoints
using the ProxConvex algorithm and an :class:`SRComposite` to encode the OR
disjunction (Uzun et al., arXiv:2512.20602v1). The problem includes:

- 2D position and heading dynamics with time state
- Disjoint waypoint visiting (wp1 OR wp2) at nodes 3–5 via GMSR ``OR``
- A cross-node time constraint between nodes 3 and 5
- Minimal-time objective
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
from openscvx.algorithms.scvx.prox_convex import ProxConvex, SRComposite
from openscvx.solvers.cvxpy_ptr_solver import CVXPyProxConvexSolver
from openscvx.symbolic.lowerers.jax.stl import AND, OR

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
speed.guess = np.ones((n, 1)) * 4.0

angular_rate = ox.Control("angular_rate", shape=(1,))  # Angular velocity
angular_rate.min = np.array([-5])
angular_rate.max = np.array([5])
angular_rate.guess = np.zeros((n, 1))

# Define time (needed for time-dependent constraints)
time = ox.Time(
    initial=0.0, final=ox.Minimize(total_time), min=0.0, max=10.0,
)

# Define list of all states and controls
states = [position, theta, time]
controls = [speed, angular_rate]

# Define Parameters for wp radius and center
wp1_center = ox.Parameter("wp1_center", shape=(2,), value=np.array([-2.1, 0.0]))
wp1_radius = ox.Parameter("wp1_radius", shape=(), value=0.5)
wp2_center = ox.Parameter("wp2_center", shape=(2,), value=np.array([2.09999, 0.0]))
wp2_radius = ox.Parameter("wp2_radius", shape=(), value=0.5)

# Define dynamics as dictionary mapping state names to their derivatives
dynamics = {
    "position": ox.Concat(
        speed[0] * ox.Sin(theta[0]),  # x_dot
        speed[0] * ox.Cos(theta[0]),  # y_dot
    ),
    "theta": angular_rate[0],
    "time": 1.0,
}

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Cross-node time constraint (unchanged from the STL version)
constraints.append((time.at(5) - time.at(3) == 1.23).convex())

# --- ProxConvex: OR encoded as an SRComposite --------------------------------
#
# At each visit node k, r_{2k} and r_{2k+1} are the signed distances to wp1
# and wp2 (positive outside the ball). OR(r_{2k}, r_{2k+1}) ≈ 0 iff the car
# is inside at least one waypoint at node k. AND over nodes matches
# Or(wp1, wp2).at([3, 4, 5]) semantics (conjunction over the node set).

visit_nodes = [3, 4, 5]

composite = SRComposite(
    s=lambda R, p: OR(R),
    r=[
        ox.linalg.Norm(position - wp1_center.value) - float(wp1_radius.value),
        ox.linalg.Norm(position - wp2_center.value) - float(wp2_radius.value),
    ],
    nodes=visit_nodes,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    algorithm=ProxConvex(composite=composite, autotuner=ox.AugmentedLagrangian()),
    solver=CVXPyProxConvexSolver(),
    float_dtype="float64",
)

problem.algorithm.lam_vb = 1e0 
# problem.algorithm.lam_cost = 1e-1
# problem.algorithm.lam_prox = 1e1

# Waypoint geometry for plotting (Parameters are not auto-collected when only
# referenced via .value inside the SR composite).
plotting_dict = {
    "wp1_center": wp1_center.value,
    "wp1_radius": wp1_radius.value,
    "wp2_center": wp2_center.value,
    "wp2_radius": wp2_radius.value,
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    plot_dubins_car_disjoint(results, problem.settings).show()
