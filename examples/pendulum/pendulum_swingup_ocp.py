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
from plotly.subplots import make_subplots

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from openscvx import Problem
from openscvx.plotting.plotting import _get_var

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
theta.guess = np.where(t_guess < 0.35 * TF, np.pi, np.linspace(np.pi, 0.0, N)).reshape(
    -1, 1
)
# cart_pos.guess = np.zeros((N, 1))
# cart_vel.guess = np.linspace(0.0, 0.5, N).reshape(-1, 1)
# theta_dot.guess = np.linspace(0.0, 2.0, N).reshape(-1, 1)
# force.guess = np.where(
#     t_guess < 0.5 * TF,
#     np.full(N, 0.75 * F_MAX),
#     np.full(N, -0.75 * F_MAX),
# ).reshape(-1, 1)
force.guess = np.zeros((N, 1))

# stage_cost.guess = np.cumsum(
#     (
#         Q_X1 * cart_pos.guess[:, 0] ** 2
#         + Q_THETA * theta.guess[:, 0] ** 2
#         + Q_V1 * cart_vel.guess[:, 0] ** 2
#         + Q_DTHETA * theta_dot.guess[:, 0] ** 2
#         + R_F * force.guess[:, 0] ** 2
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

import diffrax as dfx 

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
    discretizer = ox.DiscretizeLinearizeVectorize(dis_type="ZOH", ode_solver="Euler", diffrax_kwargs = {"stepsize_controller": dfx.StepTo(np.linspace(0.0, 1 / (N - 1), 2))},),
    solver = {
        "cvx_solver": "PIQP",
        "solver_args": {"canon_backend": "COO", "enforce_dpp": True},
    }
    # solver = {
    #     "cvx_solver": "qocogen",
    #     "solver_args": {},
    #     "cvxpygen": True,
    # }
)


problem.settings.dev.printing = False
plotting_dict = {"tf": TF}

# States to visualize (exclude integrated cost from phase-style plots if desired)
PLOT_STATE_NAMES = ["cart_pos", "theta", "cart_vel", "theta_dot", "stage_cost"]
PLOT_CONTROL_NAMES = ["force"]


def _extract_multishot_segments(results) -> list[np.ndarray]:
    """Per-interval propagated states from the final SCP discretization history."""
    v_history = getattr(results, "discretization_history", None) or []
    if not v_history:
        return []

    n_x = results.x.shape[1]
    n_u = results.u.shape[1]
    i4 = n_x + n_x * n_x + 2 * n_x * n_u
    v = np.asarray(v_history[-1], dtype=np.float64)
    n_segments = v.shape[0] // i4
    segments: list[np.ndarray] = []
    for seg_idx in range(n_segments):
        rows = []
        for t_idx in range(v.shape[1]):
            block = v[seg_idx * i4 : (seg_idx + 1) * i4, t_idx]
            rows.append(block[:n_x])
        segments.append(np.asarray(rows))
    return segments


def _multishot_segment_times(results, seg_idx: int, n_samples: int) -> np.ndarray:
    """Map multishot sample indices on one interval to physical time."""
    t_nodes = np.asarray(results.nodes["time"], dtype=np.float64).flatten()
    t0 = float(t_nodes[seg_idx])
    t1 = float(t_nodes[seg_idx + 1])
    j0 = 0 if seg_idx == 0 else 1
    times = []
    for j in range(j0, n_samples):
        alpha = j / (n_samples - 1) if n_samples > 1 else 0.0
        times.append((1.0 - alpha) * t0 + alpha * t1)
    return np.asarray(times, dtype=np.float64)


def _state_component(results, state_name: str, component: int = 0) -> int:
    var = _get_var(results, state_name, results._states)
    s = var._slice
    start = s.start if isinstance(s, slice) else s
    return start + component


def plot_multishot_propagation(results) -> go.Figure:
    """Plot SCP multishot propagation and nodes only (no post-process trajectory)."""
    segments = _extract_multishot_segments(results)
    if not segments:
        raise ValueError(
            "No discretization_history on results; multishot propagation unavailable."
        )

    state_names = [n for n in PLOT_STATE_NAMES if n in results.nodes]
    n_cols = min(4, len(state_names))
    n_rows = (len(state_names) + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=state_names,
    )
    fig.update_layout(title_text="Multishot propagation (SCP discretization)", template="plotly_dark")

    t_nodes = np.asarray(results.nodes["time"], dtype=np.float64).flatten()

    for idx, state_name in enumerate(state_names):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        comp_idx = _state_component(results, state_name, 0)
        var = _get_var(results, state_name, results._states)

        for seg_idx, seg in enumerate(segments):
            t_seg = _multishot_segment_times(results, seg_idx, len(seg))
            j0 = 0 if seg_idx == 0 else 1
            y_seg = seg[j0:, comp_idx]
            fig.add_trace(
                go.Scatter(
                    x=t_seg,
                    y=y_seg,
                    mode="lines",
                    name="Multishot propagation",
                    legendgroup="multishot",
                    showlegend=(idx == 0 and seg_idx == 0),
                    line={"color": "darkorange", "width": 2, "dash": "dot"},
                ),
                row=row,
                col=col,
            )

        y_nodes = np.asarray(results.nodes[state_name], dtype=np.float64).reshape(-1)
        fig.add_trace(
            go.Scatter(
                x=t_nodes,
                y=y_nodes,
                mode="markers",
                name="SCP nodes",
                legendgroup="nodes",
                showlegend=(idx == 0),
                marker={"color": "cyan", "size": 6},
            ),
            row=row,
            col=col,
        )

        if var.min is not None and np.isfinite(var.min[0]):
            fig.add_hline(
                y=float(var.min[0]),
                line={"color": "red", "width": 1.5, "dash": "dash"},
                row=row,
                col=col,
            )
        if var.max is not None and np.isfinite(var.max[0]):
            fig.add_hline(
                y=float(var.max[0]),
                line={"color": "red", "width": 1.5, "dash": "dash"},
                row=row,
                col=col,
            )

    for col_idx in range(1, n_cols + 1):
        fig.update_xaxes(title_text="Time (s)", row=n_rows, col=col_idx)

    return fig


def plot_multishot_controls(results) -> go.Figure:
    """Plot ZOH controls at SCP nodes (no dense post-process control trajectory)."""
    t_nodes = np.asarray(results.nodes["time"], dtype=np.float64).flatten()
    fig = go.Figure()
    fig.update_layout(title_text="Control (SCP nodes, ZOH)", template="plotly_dark")

    for ctrl_name in PLOT_CONTROL_NAMES:
        if ctrl_name not in results.nodes:
            continue
        u_nodes = np.asarray(results.nodes[ctrl_name], dtype=np.float64).reshape(-1)
        # ZOH stair steps between nodes
        t_step = []
        u_step = []
        for k in range(len(t_nodes) - 1):
            t_step.extend([t_nodes[k], t_nodes[k + 1], None])
            u_step.extend([u_nodes[k], u_nodes[k], None])
        fig.add_trace(
            go.Scatter(
                x=t_step,
                y=u_step,
                mode="lines",
                name=f"{ctrl_name} (ZOH)",
                line={"color": "darkorange", "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_nodes,
                y=u_nodes,
                mode="markers",
                name="SCP nodes",
                marker={"color": "cyan", "size": 6},
            )
        )

    ctrl = _get_var(results, "force", results._controls)
    if ctrl.min is not None:
        fig.add_hline(y=float(ctrl.min[0]), line={"color": "red", "width": 1.5, "dash": "dash"})
    if ctrl.max is not None:
        fig.add_hline(y=float(ctrl.max[0]), line={"color": "red", "width": 1.5, "dash": "dash"})

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="force")
    return fig


def plot_multishot_phase_portrait(results) -> go.Figure:
    """Phase portrait from multishot samples (theta vs theta_dot)."""
    segments = _extract_multishot_segments(results)
    if not segments:
        raise ValueError("No discretization_history on results.")

    theta_idx = _state_component(results, "theta", 0)
    theta_dot_idx = _state_component(results, "theta_dot", 0)
    fig = go.Figure()
    fig.update_layout(title_text="Phase portrait (multishot)", template="plotly_dark")

    for seg_idx, seg in enumerate(segments):
        j0 = 0 if seg_idx == 0 else 1
        fig.add_trace(
            go.Scatter(
                x=np.rad2deg(seg[j0:, theta_idx]),
                y=seg[j0:, theta_dot_idx],
                mode="lines",
                name="Multishot propagation",
                legendgroup="multishot",
                showlegend=(seg_idx == 0),
                line={"color": "darkorange", "width": 2, "dash": "dot"},
            )
        )

    theta_nodes = np.asarray(results.nodes["theta"], dtype=np.float64).reshape(-1)
    theta_dot_nodes = np.asarray(results.nodes["theta_dot"], dtype=np.float64).reshape(-1)
    fig.add_trace(
        go.Scatter(
            x=np.rad2deg(theta_nodes),
            y=theta_dot_nodes,
            mode="markers",
            name="SCP nodes",
            marker={"color": "cyan", "size": 7},
        )
    )
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
    problem.post_process()
    # Do not post_process: that integrates a single continuous open-loop trajectory.
    # For this example we inspect SCP multishot propagation from discretization_history.
    results.update(plotting_dict)

    nodes = results.nodes
    print(f"Final cart position: {nodes['cart_pos'][-1, 0]:.4f} m")
    print(f"Final pole angle:    {np.rad2deg(nodes['theta'][-1, 0]):.2f} deg")
    print(f"Final cart velocity: {nodes['cart_vel'][-1, 0]:.4f} m/s")
    print(f"Final pole rate:     {nodes['theta_dot'][-1, 0]:.4f} rad/s")
    print(f"Integrated cost:     {nodes['stage_cost'][-1, 0]:.4f}")

    n_ms = len(_extract_multishot_segments(results))
    print(f"Multishot segments:  {n_ms} (from final SCP discretization)")

    if os.environ.get("OPENSCVX_NO_PLOT") is None:
        _show_plot(plot_multishot_propagation(results))
        _show_plot(plot_multishot_controls(results))
        _show_plot(plot_multishot_phase_portrait(results))
