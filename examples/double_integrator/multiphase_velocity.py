"""1D double integrator with a multiphase velocity constraint (OpenSCvx port).

Port of the acados multiphase OCP example ``export_double_integrator_model`` /
``create_mocp`` (acados_template). The horizon is split into two phases:

- Phase 0 (nodes 0–10): velocity is unconstrained.
- Phase 1 (nodes 10–20): velocity must be nonpositive (``v <= 0``).

The vehicle starts at ``q = 1``, ``v = 0.25`` with fixed duration ``T_f = 1`` s,
minimizes a quadratic stage cost ``q^2 + 10 v^2 + 10 u^2``, and has
``|u| <= 1``.

Set ``SOFTEN_V = True`` to relax the phase-1 velocity cap with a CTCS soft
penalty (analogous to acados ``soften_h=True``).
"""

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from openscvx import Problem
from openscvx.plotting import plot_controls, plot_states

# ── Problem parameters (match acados create_mocp defaults) ───────────────────
N = 21  # nodes (acados uses N_horizon = 20 intervals)
TF = 1.0

Q_Q = 1.0
Q_V = 10.0
R_U = 10.0
F_MAX = 1.0

# Phase split: N_list = [10, 10] → phase 1 starts at node 10
PHASE1_START = 10

# Optional soft velocity cap in phase 1 (acados soften_h)
SOFTEN_V = False

# ── States ───────────────────────────────────────────────────────────────────
position = ox.State("position", shape=(1,))
position.min = np.array([-10.0])
position.max = np.array([10.0])
position.initial = np.array([1.0])
position.final = [ox.Free(0.0)]

velocity = ox.State("velocity", shape=(1,))
velocity.min = np.array([-10.0])
velocity.max = np.array([10.0])
velocity.initial = np.array([0.25])
velocity.final = [ox.Free(0.0)]

stage_cost = ox.State("stage_cost", shape=(1,))
stage_cost.min = np.array([0.0])
stage_cost.max = np.array([1e3])
stage_cost.initial = np.array([0.0])
stage_cost.final = [ox.Minimize(0.0)]

# ── Control ──────────────────────────────────────────────────────────────────
force = ox.Control("force", shape=(1,), parameterization="ZOH")
force.min = np.array([-F_MAX])
force.max = np.array([F_MAX])
force.guess = np.linspace(-0.5, -0.5, N).reshape(-1, 1)

states = [position, velocity, stage_cost]
controls = [force]

# ── Dynamics: continuous double integrator ───────────────────────────────────
dynamics = {
    "position": velocity[0],
    "velocity": force[0],
    "stage_cost": Q_Q * position[0] ** 2 + Q_V * velocity[0] ** 2 + R_U * force[0] ** 2,
}

# ── Constraints ──────────────────────────────────────────────────────────────
constraints: list = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
constraints.extend([ox.ctcs(force <= force.max), ox.ctcs(force.min <= force)])

# Phase 1 only: velocity must be nonpositive (acados con_h_expr = x[nq:], ub = 0).
# Nodal convex constraint matches acados pathwise enforcement at discrete nodes.
if SOFTEN_V:
    constraints.append(
        ox.ctcs(velocity[0] <= 0.0, penalty="smooth_relu").over((PHASE1_START, N - 1))
    )
else:
    constraints.append(
        (velocity[0] <= 0.0).convex().at(list(range(PHASE1_START, N)))
    )

# ── Initial guess ────────────────────────────────────────────────────────────
t_guess = np.linspace(0.0, TF, N)
# velocity.guess = np.linspace(0.25, -0.1, N).reshape(-1, 1)
# position.guess = np.column_stack(
#     [
#         position.initial[0]
#         + np.cumsum(velocity.guess[:, 0] * np.gradient(t_guess))
#     ]
# )
# stage_cost.guess = np.cumsum(
#     (
#         Q_Q * position.guess[:, 0] ** 2
#         + Q_V * velocity.guess[:, 0] ** 2
#         + R_U * force.guess[:, 0] ** 2
#     )
#     * np.gradient(t_guess)
# ).reshape(-1, 1)

time = ox.Time(
    initial=0.0,
    final=TF,
    min=0.0,
    max=TF,
    uniform_time_grid=True,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=N,
    float_dtype="float64",
    algorithm={
        "lam_cost": 6e-1,
        # "autotuner": ox.ConstantProximalWeight(),
    },
    solver = {
        "cvx_solver": "qocogen",
        "cvxpygen": True,
        "solver_args": {
        }
    }
)

plotting_dict = {
    "phase1_start": PHASE1_START,
    "tf": TF,
}


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    nodes = results.nodes
    print(f"Final position: {nodes['position'][-1, 0]:.4f}")
    print(f"Final velocity: {nodes['velocity'][-1, 0]:.4f}")
    print(f"Integrated stage cost: {nodes['stage_cost'][-1, 0]:.4f}")

    phase1_v = nodes["velocity"][PHASE1_START:, 0]
    print(
        f"Phase-1 velocity max: {phase1_v.max():.4e} "
        f"(should be <= 0, soften={SOFTEN_V})"
    )

    plot_states(results).show()
    plot_controls(results).show()
