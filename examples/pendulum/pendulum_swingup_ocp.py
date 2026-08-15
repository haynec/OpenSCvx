"""Cart-pendulum single-shot swing-up OCP (OpenSCvx port).

Port of the acados ``run_single_shot_ocp`` example
(``export_pendulum_ode_model`` + ``setup_ocp`` with ``for_mpc=False``).

The cart-pole starts hanging downward (``theta = pi``) and is driven toward the
upright equilibrium at the origin over a fixed horizon ``T_f = 1`` s with
``N = 20`` intervals. The objective is a quadratic tracking cost on
``[x, theta, v, dtheta, F]`` toward zero, matching acados ``NONLINEAR_LS`` with
``Q = 2 diag(1e3, 1e3, 1e-2, 1e-2)`` and ``R = 2 diag(1e-2)`` (effective
weights ``1e3, 1e3, 1e-2, 1e-2, 1e-2`` after the acados ``1/2`` factor).
Control is bounded by ``|F| <= 80`` N.

Physical parameters match acados ``pendulum_model.py``:
``M = 1.0`` kg, ``m = 0.1`` kg, ``l = 0.8`` m, ``g = 9.81`` m/s².

This is the *single-shot* case only (one open-loop solve, fixed ``T_f``), not the
closed-loop MPC branch in the acados script.
"""

import os
import sys

import numpy as np
import plotly.graph_objects as go

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from openscvx import Problem
from openscvx.plotting import plot_scp_iterations

# isort: split
# Must be imported after openscvx: openscvx sets EQX_ON_ERROR=nan at import time,
# and equinox (pulled in by diffrax) reads that variable once, when it is first
# imported. Import diffrax first and equinox's error path stays a host callback,
# which jax.export cannot serialize when the propagation solver is compiled.
import diffrax as dfx

# ── Problem parameters (match acados run_single_shot_ocp defaults) ───────────
N = 21  # nodes (acados N_horizon = 20 intervals)
TF = 1.0
F_MAX = 80.0

# Effective LS weights after acados 0.5 * ||·||_W with W = 2 * diag(·)
Q_X1 = 1e3
Q_THETA = 1e3
Q_V1 = 1e-2
Q_DTHETA = 1e-2
R_F = 1e-2


# Physical constants (acados pendulum_model.py)
M = 1.0
M_POLE = 0.1
G = 9.81
L = 0.8

# ── States ───────────────────────────────────────────────────────────────────
cart_pos = ox.State("cart_pos", shape=(1,))
cart_pos.min = np.array([-5.0])
cart_pos.max = np.array([5.0])
cart_pos.initial = np.array([0.0])
cart_pos.final = [0.0]

theta = ox.State("theta", shape=(1,))
theta.min = np.array([-2.0 * np.pi])
theta.max = np.array([2.0 * np.pi])
theta.initial = [np.pi]
theta.final = [0.0]

cart_vel = ox.State("cart_vel", shape=(1,))
cart_vel.min = np.array([-20.0])
cart_vel.max = np.array([20.0])
cart_vel.initial = np.array([0.0])
cart_vel.final = [ox.Free(0.0)]

theta_dot = ox.State("theta_dot", shape=(1,))
theta_dot.min = np.array([-20.0])
theta_dot.max = np.array([20.0])
theta_dot.initial = np.array([0.0])
theta_dot.final = [ox.Free(0.0)]

stage_cost = ox.State("stage_cost", shape=(1,))
stage_cost.min = np.array([0.0])
stage_cost.max = np.array([4e4])
stage_cost.scaling_max = [1e4]
stage_cost.initial = np.array([0.0])
stage_cost.final = [ox.Minimize(0.0)]

# ── Control ──────────────────────────────────────────────────────────────────
force = ox.Control("force", shape=(1,), parameterization="ZOH")
force.min = np.array([-F_MAX])
force.max = np.array([F_MAX])

states = [cart_pos, theta, cart_vel, theta_dot, stage_cost]
controls = [force]

# ── Dynamics (acados export_pendulum_ode_model) ──────────────────────────────
cos_theta = ox.Cos(theta[0])
sin_theta = ox.Sin(theta[0])
denom = ox.Constant(M + M_POLE) - ox.Constant(M_POLE) * cos_theta * cos_theta

cart_acc_num = (
    -ox.Constant(M_POLE * L) * sin_theta * theta_dot[0] * theta_dot[0]
    + ox.Constant(M_POLE * G) * cos_theta * sin_theta
    + force[0]
)
theta_acc_num = (
    -ox.Constant(M_POLE * L) * cos_theta * sin_theta * theta_dot[0] * theta_dot[0]
    + force[0] * cos_theta
    + ox.Constant((M + M_POLE) * G) * sin_theta
)

dynamics = {
    "cart_pos": cart_vel[0],
    "theta": theta_dot[0],
    "cart_vel": cart_acc_num / denom,
    "theta_dot": theta_acc_num / (ox.Constant(L) * denom),
    "stage_cost": (
        Q_X1 * cart_pos[0] ** 2
        + Q_THETA * theta[0] ** 2
        + Q_V1 * cart_vel[0] ** 2
        + Q_DTHETA * theta_dot[0] ** 2
        + R_F * force[0] ** 2
    ),
}

# ── Constraints ──────────────────────────────────────────────────────────────
constraints: list = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
constraints.extend([ox.ctcs(force <= force.max), ox.ctcs(force.min <= force)])

# ── Initial guess (swing-up pump heuristic) ────────────────────────────────────
t_guess = np.linspace(0.0, TF, N)
theta.guess = np.where(t_guess < 0.35 * TF, np.pi, np.linspace(np.pi, 0.0, N)).reshape(-1, 1)
force.guess = np.zeros((N, 1))

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
    algorithm={
        "lam_prox": 1e-1,
        "lam_vc": 1e1,
        "lam_cost": 2e0,
        "autotuner": ox.ConstantProximalWeight(),
    },
    discretizer=ox.DiscretizeLinearizeVectorize(
        dis_type="ZOH",
        ode_solver="Euler",
        diffrax_kwargs={"stepsize_controller": dfx.StepTo(np.linspace(0.0, 1 / (N - 1), 2))},
    ),
    solver={
        "cvx_solver": "PIQP",
        "solver_args": {"canon_backend": "COO", "enforce_dpp": True},
    },
)

problem.settings.dev.printing = False


def plot_multishot_phase_portrait(results) -> go.Figure:
    """Phase portrait of the swing-up: pole angle against pole rate.

    Deliberately not a library plot. ``openscvx.plotting`` draws every variable
    against time; the swing-up's spiral into the upright equilibrium is only
    legible in the theta-theta_dot plane, and that framing is specific to this
    problem. Samples come from the SCP multishot propagation, so the curve is the
    integrated path between nodes rather than the linear interpolation.
    """
    prop = results.multishot_propagation()
    if prop is None:
        raise ValueError("No discretization history on results; run solve() first.")

    theta_prop, _ = prop.state("theta")
    theta_dot_prop, _ = prop.state("theta_dot")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.rad2deg(theta_prop[:, 0]),
            y=theta_dot_prop[:, 0],
            mode="lines",
            name="Multishot propagation",
            line={"color": "darkorange", "width": 2, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.rad2deg(np.asarray(results.nodes["theta"], dtype=np.float64).reshape(-1)),
            y=np.asarray(results.nodes["theta_dot"], dtype=np.float64).reshape(-1),
            mode="markers",
            name="SCP nodes",
            marker={"color": "cyan", "size": 7},
        )
    )
    fig.update_layout(title_text="Phase portrait (multishot)", template="plotly_dark")
    fig.update_xaxes(title_text="theta (deg)")
    fig.update_yaxes(title_text="theta_dot (rad/s)")
    return fig


def _show_plot(fig):
    try:
        fig.show()
    except PermissionError as exc:
        print(f"Skipping plot display: {exc}")


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    # No post_process: it would integrate one continuous open-loop trajectory, which
    # diverges from the upright equilibrium. This example inspects the SCP multishot
    # propagation held in discretization_history instead.

    nodes = results.nodes
    print(f"Final cart position: {nodes['cart_pos'][-1, 0]:.4f} m")
    print(f"Final pole angle:    {np.rad2deg(nodes['theta'][-1, 0]):.2f} deg")
    print(f"Final cart velocity: {nodes['cart_vel'][-1, 0]:.4f} m/s")
    print(f"Final pole rate:     {nodes['theta_dot'][-1, 0]:.4f} rad/s")
    print(f"Integrated cost:     {nodes['stage_cost'][-1, 0]:.4f}")

    prop = results.multishot_propagation()
    print(f"Multishot segments:  {prop.n_segments} (from final SCP discretization)")

    if os.environ.get("OPENSCVX_NO_PLOT") is None:
        # States, controls, bounds and the propagated segments, colored by SCP
        # iteration — the swing-up's convergence is the story worth telling here.
        _show_plot(plot_scp_iterations(results, show_propagation=True))
        _show_plot(plot_multishot_phase_portrait(results))
