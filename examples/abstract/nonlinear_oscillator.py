"""Nominal nonlinear oscillator problem with impulsive velocity updates.

- Nonlinear dynamics: p_dot = v, v_dot = -p - p^3 - k_v v
- Two-node horizon (initial and final node)
- Impulsive control on velocity at both nodes
- Exponentially decaying position envelope: |p(t)| <= p_max * exp(-alpha t)
- Objective: minimize accumulated impulse magnitude
"""

import os
import sys

import jax
import numpy as np
import plotly.graph_objects as go

# Add grandparent directory to path for local package imports.
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem
from openscvx.plotting import plot_controls, plot_scp_iterations, plot_states

# Use float64 in JAX for higher-accuracy propagation/linearization.
jax.config.update("jax_enable_x64", True)

# Problem parameters.
n_nodes = 2
t_final = 15.0
x0_guess = np.array([0.4, 0.6], dtype=float)

p_max_0 = 0.5
exp_damp = 0.02
k_v = 0.1
integration_tol = 1e-12
ctcs_penalty = "smooth_relu"

# Disturbance parameterization (prepared for first-order buffered formulation).
# State ordering for this disturbance map is [p, v, xi], where xi is the CTCS
# augmented state used in robustified formulations.
disturbances_per_velocity_component = 4
n_velocity_components = 1
n_disturbance = disturbances_per_velocity_component * n_velocity_components
disturbance_signs = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
B_disturbance = np.zeros((3, n_disturbance), dtype=float)
B_disturbance[1, :] = disturbance_signs
eps_impulse_disturbance = 0.25
reference_impulse = np.zeros(2, dtype=float)

# States
position = ox.State("position", shape=(1,))
velocity = ox.State("velocity", shape=(1,))
impulse_cost = ox.State("impulse_cost", shape=(1,))

position.initial = np.array([x0_guess[0]])
velocity.initial = np.array([x0_guess[1]])
impulse_cost.initial = np.array([0.0])

position.final = [ox.Free(0.0)]
velocity.final = [ox.Free(0.0)]
impulse_cost.final = [("minimize", 10.0)]

# Broad box bounds; the time-varying envelope is enforced via CTCS constraints below.
position.min = np.array([-2.0])
position.max = np.array([2.0])
velocity.min = np.array([-4.0])
velocity.max = np.array([4.0])
impulse_cost.min = np.array([0.0])
impulse_cost.max = np.array([10.0])

position.guess = np.linspace(position.initial, np.array([0.0]), n_nodes)
velocity.guess = np.linspace(velocity.initial, np.array([0.0]), n_nodes)
impulse_cost.guess = np.zeros((n_nodes, 1))

# Impulsive velocity update at initial and final nodes.
delta_v = ox.Control(
    "delta_v",
    shape=(1,),
    parameterization="impulsive",
    nodes=[0, n_nodes - 1],
)
delta_v.min = np.array([-4.0])
delta_v.max = np.array([4.0])
delta_v.guess = np.zeros((n_nodes, 1))

states = [position, velocity, impulse_cost]
controls = [delta_v]

# Continuous-time nonlinear oscillator dynamics.
dynamics = {
    "position": velocity,
    "velocity": -position - position**3 - k_v * velocity,
    "impulse_cost": 0.0,
}

# Discrete jump map: velocity receives impulse, cost accumulates impulse magnitude.
eps_impulse = 1e-8
dynamics_discrete = {
    "position": position,
    "velocity": velocity + delta_v,
    "impulse_cost": impulse_cost + ox.Sqrt(delta_v[0] ** 2 + eps_impulse),
}

time = ox.Time(
    initial=0.0,
    final=t_final,
    min=0.0,
    max=t_final,
    uniform_time_grid=True,
)

# Time-varying envelope |p(t)| <= p_max_0 * exp(-exp_damp * t).
t_now = time[0]
p_bound = p_max_0 * ox.Exp(-exp_damp * t_now)
# p_bound = p_max_0


constraints = []
constraints.extend(
    [
        ox.ctcs(position[0] <= p_bound, penalty=ctcs_penalty),
        ox.ctcs(-position[0] <= p_bound, penalty=ctcs_penalty),
    ]
)

problem = Problem(
    dynamics=dynamics,
    dynamics_discrete=dynamics_discrete,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    discretizer={
        "ode_solver": "Dopri8",
        "diffrax_kwargs": {"atol": integration_tol, "rtol": integration_tol},
    },
    licq_min=0,
    licq_max=1e-10,
    N=n_nodes,
    algorithm={
        "k_max": 100,
        "lam_prox": 1e-5,
        "lam_vc": 1e2,
        "autotuner": ox.AugmentedLagrangian(eta_lambda=1e0),
    },
    solver={"cvx_solver": "CLARABEL", "solver_args": {}},
    float_dtype="float64",
)

problem.settings.prp.solver = "Dopri8"
problem.settings.prp.atol = integration_tol
problem.settings.prp.rtol = integration_tol


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    delta_v_nodes = np.asarray(results.nodes["delta_v"], dtype=float).reshape(-1)
    print(f"Converged: {bool(results.converged)}")
    print(f"Initial impulse dv0: {delta_v_nodes[0]: .9f}")
    print(f"Final impulse dvf:   {delta_v_nodes[-1]: .9f}")
    print(f"Total |dv| sum:      {np.sum(np.abs(delta_v_nodes)):.9f}")

    fig_states = plot_states(results, ["position", "velocity", "impulse_cost"], cols=3)
    if results.trajectory and "time" in results.trajectory:
        t_plot = np.asarray(results.trajectory["time"], dtype=float).reshape(-1)
    else:
        t_plot = np.asarray(results.nodes["time"], dtype=float).reshape(-1)
    p_envelope = p_max_0 * np.exp(-exp_damp * t_plot)

    fig_states.add_trace(
        go.Scatter(
            x=t_plot,
            y=p_envelope,
            mode="lines",
            name="p upper envelope",
            line={"color": "red", "dash": "dash", "width": 1.5},
        ),
        row=1,
        col=1,
    )
    fig_states.add_trace(
        go.Scatter(
            x=t_plot,
            y=-p_envelope,
            mode="lines",
            name="p lower envelope",
            line={"color": "red", "dash": "dash", "width": 1.5},
        ),
        row=1,
        col=1,
    )
    fig_states.show()
    plot_controls(results).show()
    plot_scp_iterations(results).show()
