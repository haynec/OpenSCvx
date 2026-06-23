"""2D rocket landing OCP (OpenSCvx port of lqrax ``ilqr_example.ipynb``).

A planar rocket with thrust ``T`` and body torque ``tau`` is driven from a high
offset initial state toward the origin over a fixed horizon. The objective is a
quadratic tracking cost on ``[x, y, vx, vy, theta, omega]`` toward zero with
state weights ``[1, 1, 1, 1, 10, 1]`` (so ``theta`` is weighted by 10 in the
error, i.e. ``100 * theta^2`` in the cost) plus ``0.001 * (T^2 + tau^2)``.

Dynamics (``m = 1`` kg, ``I = 5`` kg·m², ``g = 9.81`` m/s²):

    x_dot     = vx
    y_dot     = vy
    vx_dot    = -(T/m) sin(theta)
    vy_dot    = (T/m) cos(theta) - g
    theta_dot = omega
    omega_dot = tau / I

Default horizon matches the notebook: ``T_f = 6`` s with ``dt = 0.02`` s
(``N = 301`` nodes). Reduce ``N`` for faster prototyping.
"""

from __future__ import annotations

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from openscvx import Problem
from openscvx.algorithms.optimization_results import OptimizationResults
from openscvx.plotting import plot_controls, plot_states

# ── Problem parameters (match lqrax ilqr_example.ipynb) ──────────────────────
# DT = 0.02
TF = 6.0
N = 10 #int(TF / DT) + 1  # 301 nodes (300 intervals); notebook uses dt=0.02

M = 1.0
I_BODY = 5.0
G = 9.81

# runtime_loss weights: error = (x - target) * [1, 1, 1, 1, 10, 1]
W_X = 1.0
W_Y = 1.0
W_VX = 1.0
W_VY = 1.0
W_THETA = 10.0
W_OMEGA = 1.0
R_T = 0.001
R_TAU = 0.001

X0 = np.array([5.0, 5.0, -2.0, 0.0, -0.5, 0.0])
TARGET = np.zeros(6)

U_INIT = 0.2

# ── States ───────────────────────────────────────────────────────────────────
x_pos = ox.State("x", shape=(1,))
x_pos.min = np.array([-20.0])
x_pos.max = np.array([20.0])
x_pos.initial = np.array([X0[0]])
x_pos.final = [0.0]

y_pos = ox.State("y", shape=(1,))
y_pos.min = np.array([0.0])
y_pos.max = np.array([20.0])
y_pos.initial = np.array([X0[1]])
y_pos.final = [0.0]

vx = ox.State("vx", shape=(1,))
vx.min = np.array([-20.0])
vx.max = np.array([20.0])
vx.initial = np.array([X0[2]])
vx.final = [0.0]

vy = ox.State("vy", shape=(1,))
vy.min = np.array([-20.0])
vy.max = np.array([20.0])
vy.initial = np.array([X0[3]])
vy.final = [0.0]

theta = ox.State("theta", shape=(1,))
theta.min = np.array([- np.pi])
theta.max = np.array([np.pi])
theta.initial = np.array([X0[4]])
theta.final = [0.0]

omega = ox.State("omega", shape=(1,))
omega.min = np.array([-20.0])
omega.max = np.array([20.0])
omega.initial = np.array([X0[5]])
omega.final = [0.0]

stage_cost = ox.State("stage_cost", shape=(1,))
stage_cost.min = np.array([0.0])
stage_cost.max = np.array([1e4])
stage_cost.initial = np.array([0.0])
stage_cost.final = [ox.Minimize(0.0)]

# ── Controls ─────────────────────────────────────────────────────────────────
thrust = ox.Control("T", shape=(1,))
thrust.min = np.array([-50.0])
thrust.max = np.array([50.0])

torque = ox.Control("tau", shape=(1,))
torque.min = np.array([-50.0])
torque.max = np.array([50.0])

states = [x_pos, y_pos, vx, vy, theta, omega, stage_cost]
controls = [thrust, torque]

# ── Dynamics ─────────────────────────────────────────────────────────────────
sin_theta = ox.Sin(theta[0])
cos_theta = ox.Cos(theta[0])

dynamics = {
    "x": vx[0],
    "y": vy[0],
    "vx": -(thrust[0] / ox.Constant(M)) * sin_theta,
    "vy": (thrust[0] / ox.Constant(M)) * cos_theta - ox.Constant(G),
    "theta": omega[0],
    "omega": torque[0] / ox.Constant(I_BODY),
    "stage_cost": (
        W_X * (x_pos[0] - ox.Constant(TARGET[0])) ** 2
        + W_Y * (y_pos[0] - ox.Constant(TARGET[1])) ** 2
        + W_VX * (vx[0] - ox.Constant(TARGET[2])) ** 2
        + W_VY * (vy[0] - ox.Constant(TARGET[3])) ** 2
        + W_THETA * (theta[0] - ox.Constant(TARGET[4])) ** 2
        + W_OMEGA * (omega[0] - ox.Constant(TARGET[5])) ** 2
        + R_T * thrust[0] ** 2
        + R_TAU * torque[0] ** 2
    ),
}

# ── Constraints ──────────────────────────────────────────────────────────────
constraints: list = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# ── Initial guess (constant control, linear state interpolation) ─────────────
tau_interp = np.linspace(0.0, 1.0, N)
x_pos.guess = ((1.0 - tau_interp) * X0[0]).reshape(-1, 1)
y_pos.guess = ((1.0 - tau_interp) * X0[1]).reshape(-1, 1)
vx.guess = ((1.0 - tau_interp) * X0[2]).reshape(-1, 1)
vy.guess = ((1.0 - tau_interp) * X0[3]).reshape(-1, 1)
theta.guess = ((1.0 - tau_interp) * X0[4]).reshape(-1, 1)
omega.guess = ((1.0 - tau_interp) * X0[5]).reshape(-1, 1)
thrust.guess = np.full((N, 1), U_INIT)
torque.guess = np.full((N, 1), U_INIT)
t_guess = np.linspace(0.0, TF, N)
stage_cost.guess = np.cumsum(
    (
        W_X * x_pos.guess[:, 0] ** 2
        + W_Y * y_pos.guess[:, 0] ** 2
        + W_VX * vx.guess[:, 0] ** 2
        + W_VY * vy.guess[:, 0] ** 2
        + W_THETA * theta.guess[:, 0] ** 2
        + W_OMEGA * omega.guess[:, 0] ** 2
        + R_T * thrust.guess[:, 0] ** 2
        + R_TAU * torque.guess[:, 0] ** 2
    )
    * np.gradient(t_guess)
).reshape(-1, 1)

time = ox.Time(
    initial=0.0,
    final=TF,
    min=0.0,
    max=TF,
    # uniform_time_grid=True,
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
        "lam_prox": 1e0,
        "lam_vc": 2e1,
        "lam_cost": 1e1,
        "ep_vc": 2.4e-8,
        "autotuner": ox.ConstantProximalWeight(),
    },
)
problem.settings.dev.printing = True
problem.settings.prp.dt = 0.02

def _trajectory_arrays(results: OptimizationResults) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``[x, y, vx, vy, theta, omega]`` and ``[T, tau]`` for animation."""
    traj = results.trajectory
    if traj:
        source = traj
    else:
        source = results.nodes

    states = np.column_stack(
        [
            np.asarray(source["x"], dtype=np.float64).flatten(),
            np.asarray(source["y"], dtype=np.float64).flatten(),
            np.asarray(source["vx"], dtype=np.float64).flatten(),
            np.asarray(source["vy"], dtype=np.float64).flatten(),
            np.asarray(source["theta"], dtype=np.float64).flatten(),
            np.asarray(source["omega"], dtype=np.float64).flatten(),
        ]
    )
    controls = np.column_stack(
        [
            np.asarray(source["T"], dtype=np.float64).flatten(),
            np.asarray(source["tau"], dtype=np.float64).flatten(),
        ]
    )
    n = min(len(states), len(controls))
    return states[:n], controls[:n]


def animate_rocket_trajectory(
    states: np.ndarray,
    controls: np.ndarray,
    *,
    save_path: str | None = None,
    show: bool = True,
    interval_ms: int = 20,
) -> None:
    """Matplotlib animation matching lqrax ``ilqr_example.ipynb``."""
    import matplotlib.animation as mpl_animation
    import matplotlib.pyplot as plt

    rocket_length = 1.0
    prop_offset = 0.5
    fire_scale = 0.10
    max_arrow_len = 1.0
    margin = 2.0

    xmin, xmax = states[:, 0].min() - margin, states[:, 0].max() + margin
    ymin, ymax = states[:, 1].min() - margin * 0.5, states[:, 1].max() + margin * 1.5

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=120, tight_layout=True)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot(0.0, 0.0, marker="x", markersize=15, color="b")

    rocket_body_line, = ax.plot([], [], "k-", lw=2)
    orientation_line, = ax.plot([], [], "k-", lw=5)
    left_thrust_line, = ax.plot([], [], color="orange", lw=4)
    right_thrust_line, = ax.plot([], [], color="orange", lw=4)

    def init():
        rocket_body_line.set_data([], [])
        orientation_line.set_data([], [])
        left_thrust_line.set_data([], [])
        right_thrust_line.set_data([], [])
        return (
            rocket_body_line,
            orientation_line,
            left_thrust_line,
            right_thrust_line,
        )

    def update(i: int):
        state = states[i]
        control = controls[i]
        x, y, _vx, _vy, theta_i, _omega = state
        thrust_val, tau = control
        center = np.array([x, y])

        body_dir = np.array([-np.sin(theta_i), np.cos(theta_i)])
        perp = np.array([-np.cos(theta_i), -np.sin(theta_i)])

        left_propeller = center + prop_offset * perp
        right_propeller = center - prop_offset * perp

        rocket_body_line.set_data(
            [left_propeller[0], right_propeller[0]],
            [left_propeller[1], right_propeller[1]],
        )

        orientation_end = center + (rocket_length * 2.0) * body_dir
        orientation_line.set_data(
            [center[0], orientation_end[0]],
            [center[1], orientation_end[1]],
        )

        thrust_left = 0.6 * (thrust_val + tau / prop_offset)
        thrust_right = 0.6 * (thrust_val - tau / prop_offset)

        left_dir = body_dir if thrust_left >= 0 else 0.0
        left_thrust_length = np.minimum(fire_scale * abs(thrust_left), max_arrow_len)
        left_thrust_end = left_propeller - left_thrust_length * left_dir
        left_thrust_line.set_data(
            [left_propeller[0], left_thrust_end[0]],
            [left_propeller[1], left_thrust_end[1]],
        )

        right_dir = body_dir if thrust_right >= 0 else 0.0
        right_thrust_length = np.minimum(fire_scale * abs(thrust_right), max_arrow_len)
        right_thrust_end = right_propeller - right_thrust_length * right_dir
        right_thrust_line.set_data(
            [right_propeller[0], right_thrust_end[0]],
            [right_propeller[1], right_thrust_end[1]],
        )

        return (
            rocket_body_line,
            orientation_line,
            left_thrust_line,
            right_thrust_line,
        )

    ani = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=len(states),
        init_func=init,
        interval=interval_ms,
        blit=True,
    )

    if save_path is not None:
        if save_path.endswith(".gif"):
            writer = mpl_animation.PillowWriter()
            ani.save(save_path, writer=writer)
        else:
            ani.save(save_path, writer="ffmpeg")
        print(f"Saved animation to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _show_plot(fig):
    try:
        fig.show()
    except PermissionError as exc:
        print(f"Skipping plot display: {exc}")


if __name__ == "__main__":
    plot_solution = os.environ.get("OPENSCVX_NO_PLOT") is None
    animation_path = os.environ.get(
        "OPENSCVX_ROCKET_ANIM",
        os.path.join(current_dir, "rocket_ilqr.mp4"),
    )

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    nodes = results.nodes
    print(f"x0 = {X0}")
    print(
        f"  x(T) = {nodes['x'][-1, 0]:.4f}, y(T) = {nodes['y'][-1, 0]:.4f}, "
        f"vx(T) = {nodes['vx'][-1, 0]:.4f}, vy(T) = {nodes['vy'][-1, 0]:.4f}"
    )
    print(
        f"  theta(T) = {nodes['theta'][-1, 0]:.4f}, "
        f"omega(T) = {nodes['omega'][-1, 0]:.4f}"
    )
    print(f"  stage_cost(T) = {nodes['stage_cost'][-1, 0]:.4f}")
    print(f"  converged: {results.converged}")

    if plot_solution:
        _show_plot(plot_states(results))
        _show_plot(plot_controls(results))
        state_traj, control_traj = _trajectory_arrays(results)
        try:
            animate_rocket_trajectory(
                state_traj,
                control_traj,
                save_path=animation_path,
                show=False,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            gif_path = os.path.splitext(animation_path)[0] + ".gif"
            print(f"Could not save MP4 ({exc}); trying GIF at {gif_path}")
            animate_rocket_trajectory(
                state_traj,
                control_traj,
                save_path=gif_path,
                show=False,
            )
