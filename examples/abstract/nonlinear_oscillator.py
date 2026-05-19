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

# Disturbance parameterization. Disturbance enters the velocity channel only;
# sign-flipped columns span ±ε on that channel so a single |·| of the propagated
# position deviation covers both directions. Physical-block ordering is [p, v].
n_phys_stm = 2
eps_impulse_disturbance = 0.25

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

# Running STM Φ(τ) = ∂x_phys(τ)/∂x_phys(0) over the physical block [p, v].
# Flat row-major layout in the unified state: phi[0]=∂p/∂p0, phi[1]=∂p/∂v0,
# phi[2]=∂v/∂p0, phi[3]=∂v/∂v0. ``mode="exact"`` turns on Ψ integration so
# the SCP Jacobian of robustified CTCS rows sees second-order sensitivity.
phi = ox.STMPhysical("phi", n_phys=n_phys_stm, mode="approx")

B_disturbance = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, -1.0]])

# First-order buffered envelope: |p + Φ · B_v · ε| <= p_bound. B_v = [0, 1],
# so the position deviation from a ±ε velocity disturbance is phi[1] · ε.
constraints = []

phi_matrix = ox.Stack(
    [
        ox.Concat(phi[0], phi[1]),
        ox.Concat(phi[2], phi[3]),
    ]
)
state_sensitivity = phi_matrix @ B_disturbance

positive_impulse =  ox.Max(0.0, eps_impulse_disturbance * delta_v[0])
negative_impulse = -ox.Min(0.0, eps_impulse_disturbance * delta_v[0])
beta = ox.Stack(
    [
        positive_impulse**1.001,
        positive_impulse**1.001,
        negative_impulse**1.001,
        negative_impulse**1.001,
    ]
)

jac_UB = np.array([1.0, 0.0])
dual_UB = jac_UB @ state_sensitivity
optimal_dual_UB = ox.Max(np.zeros(4), dual_UB) ** 1.001
worst_deviation_UB = optimal_dual_UB @ beta

jac_LB = np.array([-1.0, 0.0])
dual_LB = jac_LB @ state_sensitivity
optimal_dual_LB = ox.Max(np.zeros(4), dual_LB) ** 1.001
worst_deviation_LB = optimal_dual_LB @ beta

constraints.extend(
    [
        ox.ctcs(
            position[0] + worst_deviation_UB <= p_bound,
            penalty=ctcs_penalty,
        ),
        ox.ctcs(
            -position[0] + worst_deviation_LB <= p_bound,
            penalty=ctcs_penalty,
        ),
    ]
)

problem = Problem(
    dynamics=dynamics,
    dynamics_discrete=dynamics_discrete,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    discretizer=ox.LinearizeDiscretize(
        ode_solver="Dopri8",
        diffrax_kwargs={"atol": integration_tol, "rtol": integration_tol},
    ),
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
    # Monte Carlo over velocity disturbances δv ~ Uniform[-ε, +ε].
    # The robust envelope was sized to cover exactly this family of perturbations.
    from scipy.integrate import solve_ivp

    N_mc = 1000
    rng = np.random.default_rng(42)
    t_mc = np.linspace(0.0, t_final, 500)

    p0_nom = float(position.initial[0])
    v0_nom = float(velocity.initial[0])
    dv0_opt = float(delta_v_nodes[0])

    def _osc_rhs(t, y):
        p, v = y
        return [v, -p - p**3 - k_v * v]

    p_mc_all = []
    for _ in range(N_mc):
        dv_dist = rng.uniform(-eps_impulse_disturbance, eps_impulse_disturbance)
        v_init = v0_nom + dv0_opt * (1 + dv_dist)
        sol_mc = solve_ivp(
            _osc_rhs,
            [0.0, t_final],
            [p0_nom, v_init],
            t_eval=t_mc,
            rtol=1e-10,
            atol=1e-10,
            method="DOP853",
        )
        p_mc_all.append(sol_mc.y[0])

    p_mc_all = np.array(p_mc_all)  # (N_mc, n_t)

    # Plot envelope as shaded band (max/min across samples).
    p_mc_max = p_mc_all.max(axis=0)
    p_mc_min = p_mc_all.min(axis=0)

    fig_states.add_trace(
        go.Scatter(
            x=np.concatenate([t_mc, t_mc[::-1]]),
            y=np.concatenate([p_mc_max, p_mc_min[::-1]]),
            fill="toself",
            fillcolor="rgba(100, 149, 237, 0.20)",
            line={"width": 0},
            name=f"MC band (N={N_mc}, ε={eps_impulse_disturbance})",
        ),
        row=1,
        col=1,
    )
    # Overlay individual traces (light, to show spread).
    for p_trace in p_mc_all[::10]:
        fig_states.add_trace(
            go.Scatter(
                x=t_mc,
                y=p_trace,
                mode="lines",
                line={"color": "rgba(100, 149, 237, 0.15)", "width": 0.8},
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig_states.show()
    plot_controls(results).show()
    plot_scp_iterations(results).show()
