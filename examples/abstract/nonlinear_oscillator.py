"""Multi-impulse nonlinear oscillator problem with anchored Φ robustification.

- Nonlinear dynamics: p_dot = v, v_dot = -p - p^3 + k_v v  (anti-damped excitation)
- Three-node horizon (impulses at nodes 0 and 1)
- Each impulse has its own velocity disturbance δv_q ~ Uniform[-ε|dv_q|, +ε|dv_q|]
- Exponentially decaying position envelope: |p(t)| <= p_max * exp(-alpha t)
- Objective: minimize accumulated impulse magnitude

The two impulses are modeled as two separate impulsive controls (one per node),
which lets the framework auto-enforce u = 0 at the other's node. For each
impulse we declare an **anchored** STMPhysical state Φ_q that is identity-
injected only at the impulse's node and propagates continuously through later
segments without per-segment reset — so Φ_q(τ) ≈ ∂x(τ)/∂x(t_{j_q}^+) directly,
without needing to chain past segments' terminal Φ values. Persistent recorder
states dv_q_persist capture each impulse's magnitude at injection so the
worst-case buffer can reference it from any later segment.
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
n_nodes = 3
impulse_nodes = [0, 1]  # one impulsive control per node
t_final = 15.0
x0_guess = np.array([0.4, 0.6], dtype=float)

p_max_0 = 0.5
exp_damp = 0.02
k_v = 0.015
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

# One impulsive control per impulse node. Splitting the impulses lets us read
# each magnitude individually inside CTCS at later segments (the framework
# auto-constrains the inactive control to 0 at the other's node).
delta_v_controls = []
for q, node_q in enumerate(impulse_nodes):
    dv_q = ox.Control(
        f"dv_{q}",
        shape=(1,),
        parameterization="impulsive",
        nodes=[node_q],
    )
    dv_q.min = np.array([-4.0])
    dv_q.max = np.array([4.0])
    dv_q.guess = np.zeros((n_nodes, 1))
    delta_v_controls.append(dv_q)

# Persistent state recorders: dv_q_persist accumulates dv_q via the discrete
# map. Because dv_q is forced to 0 at every node except its impulse node, the
# accumulator captures δv_q exactly once (at node j_q) and carries it forward
# unchanged. SCP gradient flow through dv_q_persist → CTCS buffer is preserved.
dv_persist_states = []
for q in range(len(impulse_nodes)):
    dv_persist = ox.State(f"dv_{q}_persist", shape=(1,))
    dv_persist.initial = np.array([0.0])
    dv_persist.final = [ox.Free(0.0)]
    dv_persist.min = np.array([-4.0])
    dv_persist.max = np.array([4.0])
    dv_persist.guess = np.zeros((n_nodes, 1))
    dv_persist_states.append(dv_persist)

states = [position, velocity, impulse_cost, *dv_persist_states]
controls = list(delta_v_controls)

# Continuous-time nonlinear oscillator dynamics. Recorder states are constant
# during continuous time; they only change in the discrete jump map below.
dynamics = {
    "position": velocity,
    "velocity": -position - position**3 + k_v * velocity,
    "impulse_cost": 0.0,
}
for dv_persist in dv_persist_states:
    dynamics[dv_persist.name] = 0.0

# Discrete jump map: velocity receives the sum of all impulses (each is 0
# except at its own node); cost accumulates impulse magnitude; each persistent
# recorder accumulates its associated impulse.
eps_impulse = 1e-8
velocity_jump = velocity
impulse_cost_jump = impulse_cost
for dv_q in delta_v_controls:
    velocity_jump = velocity_jump + dv_q
    impulse_cost_jump = impulse_cost_jump + ox.Sqrt(dv_q[0] ** 2 + eps_impulse)
dynamics_discrete = {
    "position": position,
    "velocity": velocity_jump,
    "impulse_cost": impulse_cost_jump,
}
for q, (dv_q, dv_persist) in enumerate(zip(delta_v_controls, dv_persist_states)):
    dynamics_discrete[dv_persist.name] = dv_persist + dv_q

time = ox.Time(
    initial=0.0,
    final=ox.Maximize(t_final),
    min=0.0,
    max=t_final,
    uniform_time_grid=True,
)

# Time-varying envelope |p(t)| <= p_max_0 * exp(-exp_damp * t).
t_now = time[0]
p_bound = p_max_0 * ox.Exp(-exp_damp * t_now)

# One anchored Φ per impulse. ``anchor_node=j`` makes Φ_q identity-injected
# only at node j and propagates continuously through every subsequent segment
# without per-segment reset — within segment k ≥ j, Φ_q(τ) = ∂x(τ)/∂x(t_j^+).
# Before node j, Φ_q is zero, so its contribution to the buffer vanishes
# automatically in earlier segments.
phi_states = [
    ox.STMPhysical(f"phi_{q}", n_phys=n_phys_stm, mode="approx", anchor_node=node_q)
    for q, node_q in enumerate(impulse_nodes)
]

B_disturbance = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, -1.0]])
jac_UB = np.array([1.0, 0.0])
jac_LB = np.array([-1.0, 0.0])


def _impulse_buffer(phi_q, dv_q_persist):
    """Worst-case position deviation (UB, LB) from disturbance on one impulse.

    ``phi_q`` is the anchored STM matrix for the impulse; ``dv_q_persist`` is
    the persistent recorder state holding that impulse's magnitude. Same
    structure as the original single-impulse robustification — see lines 124-151
    of the prior single-impulse version of this example.
    """
    phi_q_matrix = ox.Stack(
        [
            ox.Concat(phi_q[0], phi_q[1]),
            ox.Concat(phi_q[2], phi_q[3]),
        ]
    )
    state_sensitivity_q = phi_q_matrix @ B_disturbance

    positive_impulse = ox.Max(0.0, eps_impulse_disturbance * dv_q_persist[0])
    negative_impulse = -ox.Min(0.0, eps_impulse_disturbance * dv_q_persist[0])
    beta_q = ox.Stack(
        [
            positive_impulse**1.001,
            positive_impulse**1.001,
            negative_impulse**1.001,
            negative_impulse**1.001,
        ]
    )

    dual_UB_q = jac_UB @ state_sensitivity_q
    optimal_dual_UB_q = ox.Max(np.zeros(4), dual_UB_q) ** 1.001
    worst_dev_UB_q = optimal_dual_UB_q @ beta_q

    dual_LB_q = jac_LB @ state_sensitivity_q
    optimal_dual_LB_q = ox.Max(np.zeros(4), dual_LB_q) ** 1.001
    worst_dev_LB_q = optimal_dual_LB_q @ beta_q

    return worst_dev_UB_q, worst_dev_LB_q


# Sum the per-impulse worst-case deviations: with each disturbance independent
# and ε|dv_q|-bounded, the worst-case combined position offset is the sum.
worst_deviation_UB = None
worst_deviation_LB = None
for phi_q, dv_persist in zip(phi_states, dv_persist_states):
    buf_UB, buf_LB = _impulse_buffer(phi_q, dv_persist)
    worst_deviation_UB = buf_UB if worst_deviation_UB is None else worst_deviation_UB + buf_UB
    worst_deviation_LB = buf_LB if worst_deviation_LB is None else worst_deviation_LB + buf_LB

constraints = [
    ox.ctcs(
        position[0] + worst_deviation_UB <= p_bound,
        penalty=ctcs_penalty,
    ),
    ox.ctcs(
        -position[0] + worst_deviation_LB <= p_bound,
        penalty=ctcs_penalty,
    ),
]

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

    # Per-impulse optimal magnitudes (each impulsive control is non-zero only
    # at its own node, so picking that row gives the impulse value).
    dv_opt = np.array(
        [
            float(np.asarray(results.nodes[dv.name], dtype=float).reshape(-1)[node_q])
            for dv, node_q in zip(delta_v_controls, impulse_nodes)
        ]
    )
    print(f"Converged: {bool(results.converged)}")
    for q, node_q in enumerate(impulse_nodes):
        print(f"Impulse {q} at node {node_q}: {dv_opt[q]: .9f}")
    print(f"Total |dv| sum:      {np.sum(np.abs(dv_opt)):.9f}")

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
    # Monte Carlo over independent velocity disturbances at each impulse:
    # δv_q^perturbed = δv_q · (1 + δ_q), δ_q ~ Uniform[-ε, +ε]. Propagate
    # segment-by-segment, applying each perturbed impulse at its node.
    from scipy.integrate import solve_ivp

    N_mc = 1000
    rng = np.random.default_rng(42)
    # Nodal times (uniform grid: nodes equispaced over [0, t_final]).
    t_nodes_mc = np.linspace(0.0, t_final, n_nodes)
    # Per-segment time grid for the MC integration, concatenated to one array.
    n_per_seg = 250
    segment_grids = [
        np.linspace(t_nodes_mc[k], t_nodes_mc[k + 1], n_per_seg) for k in range(n_nodes - 1)
    ]
    t_mc = np.concatenate(segment_grids)

    p0_nom = float(position.initial[0])
    v0_nom = float(velocity.initial[0])

    # Map node index → impulse index (so we know whether to apply an impulse
    # at the start of each segment).
    node_to_impulse = {node_q: q for q, node_q in enumerate(impulse_nodes)}

    def _osc_rhs(t, y):
        p, v = y
        return [v, -p - p**3 + k_v * v]

    p_mc_all = []
    for _ in range(N_mc):
        deltas = rng.uniform(
            -eps_impulse_disturbance, eps_impulse_disturbance, size=len(impulse_nodes)
        )
        p_state, v_state = p0_nom, v0_nom
        traces = []
        for k in range(n_nodes - 1):
            # Apply impulse at start of segment k if one is configured here.
            if k in node_to_impulse:
                q = node_to_impulse[k]
                v_state = v_state + dv_opt[q] * (1.0 + deltas[q])
            sol_mc = solve_ivp(
                _osc_rhs,
                [t_nodes_mc[k], t_nodes_mc[k + 1]],
                [p_state, v_state],
                t_eval=segment_grids[k],
                rtol=1e-10,
                atol=1e-10,
                method="DOP853",
            )
            traces.append(sol_mc.y[0])
            p_state = float(sol_mc.y[0, -1])
            v_state = float(sol_mc.y[1, -1])
        p_mc_all.append(np.concatenate(traces))

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
