"""MPCC example: 3D double integrator tracking a circular reference trajectory (discrete).

Demonstrates model-predictive contouring control (MPCC) with:
- 3D double integrator (point mass) dynamics (position + velocity, force control)
- Discrete reference path via Cinterp (sampled from a tilted circle, generalizes to arbitrary 3D paths)
- Lag/contour error decomposition following Romero 2022
- Receding horizon closed-loop simulation
"""

import os
import sys

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add grandparent directory to path to import openscvx
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem
from openscvx.plotting import plot_scp_iterations, plot_states

###############################################################################
# Reference circle parameters
###############################################################################
R_circle = 3.0  # Radius of the reference circle
center = np.array([0.0, 0.0, 0.0])  # Center of the reference circle (3D)
tilt_angle = np.radians(45)  # Tilt the circle out of the xy-plane (rotation about x-axis)
total_arc_length = 2 * np.pi * R_circle  # One full lap

# Rotation matrix: tilt about x-axis by tilt_angle
R_tilt = np.array([
    [1, 0, 0],
    [0, np.cos(tilt_angle), -np.sin(tilt_angle)],
    [0, np.sin(tilt_angle), np.cos(tilt_angle)],
])

###############################################################################
# Discrete reference path (sampled from the tilted circle)
###############################################################################
M = 3  # Number of samples per lap

# Sample one lap in the local xy-plane, then rotate into 3D
s_lap = np.linspace(0, total_arc_length, M, endpoint=False)
angle_lap = s_lap / R_circle
circle_local = np.column_stack([
    R_circle * np.cos(angle_lap),
    R_circle * np.sin(angle_lap),
    np.zeros(M),
])
circle_rotated = (R_tilt @ circle_local.T).T + center
px_lap = circle_rotated[:, 0]
py_lap = circle_rotated[:, 1]
pz_lap = circle_rotated[:, 2]

# Tile to cover [progress.min, progress.max] = [-0.5L, 1.5L]
s_min, s_max = -0.5 * total_arc_length, 1.5 * total_arc_length
n_before = int(np.ceil(-s_min / total_arc_length))
n_after = int(np.ceil(s_max / total_arc_length))
laps = range(-n_before, n_after + 1)

s_data = np.concatenate([s_lap + k * total_arc_length for k in laps])
px_data = np.tile(px_lap, len(laps))
py_data = np.tile(py_lap, len(laps))
pz_data = np.tile(pz_lap, len(laps))


###############################################################################
# MPCC parameters
###############################################################################
n_mpc = 10  # Horizon nodes
horizon_duration = 1.0  # Horizon length [s]

Q_LAG = 1e0  # Lag error weight (high -> accurate progress tracking)
Q_CONTOUR = 1e-1  # Contour error weight
Q_PROGRESS = 1e-1

###############################################################################
# MPCC problem definition
###############################################################################

# --- States ---
position = ox.State("position", shape=(3,))
position.min = np.array([-10.0, -10.0, -10.0])
position.max = np.array([10.0, 10.0, 10.0])
position.initial = R_tilt @ np.array([R_circle, 0.0, 0.0]) + center  # Start on circle at theta=0
position.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

velocity = ox.State("velocity", shape=(3,))
velocity.min = np.array([-20.0, -20.0, -20.0])
velocity.max = np.array([20.0, 20.0, 20.0])
velocity.initial = np.array([0.0, 0.0, 0.0])
velocity.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

progress = ox.State("progress", shape=(1,))  # Arc-length progress theta_hat
progress.min = np.array([-0.5 * total_arc_length])
progress.max = np.array([1.5 * total_arc_length])
progress.initial = np.array([0.0])
progress.final = [ox.Maximize(0.0)]

lag_sum = ox.State("lag_sum", shape=(1,))  # Integrated lag cost
lag_sum.min = np.array([0.0])
lag_sum.max = np.array([1e-3])
lag_sum.initial = np.array([0.0])
lag_sum.final = [ox.Minimize(0.0)]

contour_sum = ox.State("contour_sum", shape=(1,))  # Integrated contour cost
contour_sum.min = np.array([0.0])
contour_sum.max = np.array([5e-3])
contour_sum.initial = np.array([0.0])
contour_sum.final = [ox.Minimize(0.0)]

# --- Controls ---
force = ox.Control("force", shape=(3,))
f_max = 20.0
force.min = np.array([-f_max, -f_max, -f_max])
force.max = np.array([f_max, f_max, f_max])
force.guess = np.zeros((n_mpc, 3))

progress_rate = ox.Control("progress_rate", shape=(1,))  # d(theta_hat)/dt
progress_rate.min = np.array([0.0])  # Forward only
progress_rate.max = np.array([10.0])
progress_rate.guess = np.full((n_mpc, 1), 5.0)

m = 1.0  # Mass

# --- Reference trajectory (discrete, via Cinterp) ---
p_ref = ox.Concat(
    ox.Cinterp(progress[0], s_data, px_data),
    ox.Cinterp(progress[0], s_data, py_data),
    ox.Cinterp(progress[0], s_data, pz_data),
)

# Tangent: derivative of the position spline, sampled at breakpoints and
# re-interpolated with a second Cinterp for a smooth tangent field.
from scipy.interpolate import CubicSpline as _CS

_dpx = _CS(s_data, px_data)(s_data, 1)
_dpy = _CS(s_data, py_data)(s_data, 1)
_dpz = _CS(s_data, pz_data)(s_data, 1)
_tnorm = np.sqrt(_dpx**2 + _dpy**2 + _dpz**2)
tx_data = _dpx / _tnorm
ty_data = _dpy / _tnorm
tz_data = _dpz / _tnorm

tangent = ox.Concat(
    ox.Cinterp(progress[0], s_data, tx_data),
    ox.Cinterp(progress[0], s_data, ty_data),
    ox.Cinterp(progress[0], s_data, tz_data),
)

# --- Error decomposition (position-only, per Romero 2022 Fig. 2) ---
e = position - p_ref  # Position error vector (3,)

# Lag: projection of error onto tangent direction
lag_scalar = ox.Sum(e * tangent)  # Dot product (scalar)
lag_cost = lag_scalar**2

# Contour: Pythagorean decomposition  |e_c|^2 = |e|^2 - |e_l|^2
# Use Sum(e*e) instead of Norm(e)**2 to avoid d(Norm)/de = e/||e|| singularity at e=0
contour_cost = ox.Max(ox.Sum(e * e) - lag_scalar**2, 0.0)

# --- Dynamics ---
dynamics = {
    "position": velocity,
    "velocity": (1 / m) * force,
    "progress": progress_rate,
    "lag_sum": lag_cost,
    "contour_sum": contour_cost,
}

# --- Constraints ---
states = [position, velocity, progress, lag_sum, contour_sum]
controls = [force, progress_rate]

constraints = []
for state in [position, velocity]:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# --- Time ---
t = ox.Time(
    initial=0.0, final=horizon_duration, min=0.0, max=horizon_duration, uniform_time_grid=True
)

# --- Problem ---
problem_mpc = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=t,
    constraints=constraints,
    N=n_mpc,
    algorithm={
        "autotuner": ox.ConstantProximalWeight(),
        "lam_cost": {"lag_sum": Q_LAG, "contour_sum": Q_CONTOUR, "progress": Q_PROGRESS},
    },
)

problem_mpc.settings.dev.printing = False


###############################################################################
# Initial guesses
###############################################################################
def set_initial_guess(
    theta_start: float = 0.0,
    ref_speed: float = 5.0,
):
    """Set guesses by interpolating the discrete reference path.

    Only requires (s_data, px_data, py_data, pz_data) and a nominal speed — no
    analytical path knowledge or precomputed tangent data.
    """
    arc_guess = np.linspace(theta_start, theta_start + ref_speed * horizon_duration, n_mpc)

    # Position: interpolate from reference sample nodes
    pos_guess = np.column_stack([
        np.interp(arc_guess, s_data, px_data),
        np.interp(arc_guess, s_data, py_data),
        np.interp(arc_guess, s_data, pz_data),
    ])
    position.guess = pos_guess

    # Velocity: finite-difference of position guess
    dt = horizon_duration / (n_mpc - 1)
    vel_guess = np.gradient(pos_guess, dt, axis=0)
    velocity.guess = vel_guess
    velocity.initial = vel_guess[0]

    # Force: finite-difference of velocity (mass * acceleration)
    acc_guess = np.gradient(vel_guess, dt, axis=0)
    force.guess = m * acc_guess

    progress.guess = arc_guess.reshape(-1, 1)
    lag_sum.guess = np.zeros((n_mpc, 1))
    contour_sum.guess = np.zeros((n_mpc, 1))

    progress_rate.guess = np.full((n_mpc, 1), ref_speed)


###############################################################################
# Closed-loop simulation
###############################################################################
def shift_guess(nodes: dict):
    """Shift previous solution by one node for warm-starting."""
    dt = horizon_duration / (n_mpc - 1)

    # Extrapolate a new final node
    pos_last = nodes["position"][-1]
    vel_last = nodes["velocity"][-1]
    force_last = nodes["force"][-1]
    pr_last = nodes["progress_rate"][-1, 0]

    ext_pos = pos_last + dt * vel_last
    ext_vel = vel_last + dt * (1 / m) * force_last
    ext_prog = nodes["progress"][-1, 0] + dt * pr_last

    shifted_progress = np.vstack([nodes["progress"][1:], [[ext_prog]]])
    wrap_offset = (nodes["progress"][1, 0] // total_arc_length) * total_arc_length
    shifted_progress -= wrap_offset

    position.guess = np.vstack([nodes["position"][1:], [ext_pos]])
    velocity.guess = np.vstack([nodes["velocity"][1:], [ext_vel]])
    progress.guess = shifted_progress
    lag_sum.guess = np.zeros((n_mpc, 1))
    contour_sum.guess = np.zeros((n_mpc, 1))

    force.guess = np.vstack([nodes["force"][1:], nodes["force"][-1:]])
    progress_rate.guess = np.vstack([nodes["progress_rate"][1:], nodes["progress_rate"][-1:]])


def update_initial_conditions(nodes: dict):
    """Set initial conditions from node 1 of previous solution (simulate one step)."""
    position.initial = nodes["position"][1]
    velocity.initial = nodes["velocity"][1]

    wrap_offset = (nodes["progress"][1, 0] // total_arc_length) * total_arc_length
    progress.initial = np.array([nodes["progress"][1, 0] - wrap_offset])

    # Cost integrators always restart from zero each horizon
    lag_sum.initial = np.array([0.0])
    contour_sum.initial = np.array([0.0])


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    set_initial_guess(theta_start=0.0)

    problem_mpc.initialize()

    max_steps = 100

    fig = go.Figure()

    # Reference circle (tilted)
    circle_theta = np.linspace(0, 2 * np.pi, 200)
    circle_pts = R_tilt @ np.array([
        R_circle * np.cos(circle_theta),
        R_circle * np.sin(circle_theta),
        np.zeros_like(circle_theta),
    ]) + center[:, None]
    fig.add_trace(
        go.Scatter3d(
            x=circle_pts[0], y=circle_pts[1], z=circle_pts[2],
            mode="lines",
            line={"color": "red", "width": 4, "dash": "dash"},
            name="Reference",
        )
    )

    colors = px.colors.sample_colorscale("Viridis", np.linspace(0, 1, max_steps))

    for step in range(max_steps):
        problem_mpc.reset()
        results = problem_mpc.solve()
        results = problem_mpc.post_process()
        nodes = results.nodes

        traj = results.trajectory["position"]
        fig.add_trace(
            go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode="lines",
                line={"color": colors[step], "width": 3},
                name=f"Step {step}",
            )
        )

        cur_pos = nodes["position"][0]
        cur_progress = nodes["progress"][0, 0]
        cur_lag = nodes["lag_sum"][-1, 0]
        cur_contour = nodes["contour_sum"][-1, 0]

        laps = cur_progress / total_arc_length
        print(
            f"step {step:3d}: progress={cur_progress:7.2f} "
            f"({laps:.2f} laps), "
            f"lag_cost={cur_lag:.4f}, contour_cost={cur_contour:.4f}, "
            f"pos=[{cur_pos[0]:+6.2f}, {cur_pos[1]:+6.2f}, {cur_pos[2]:+6.2f}]"
        )

        update_initial_conditions(nodes)
        shift_guess(nodes)

    fig.update_layout(
        title="Double Integrator MPCC", title_x=0.5, template="plotly_dark",
        scene={"aspectmode": "data"},
    )
    fig.show()

    plot_states(results).show()
    plot_scp_iterations(results).show()
