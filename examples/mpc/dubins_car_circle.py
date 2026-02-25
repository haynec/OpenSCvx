"""MPCC example: Dubins car tracking a circular reference trajectory.

Demonstrates model-predictive contouring control (MPCC) with:
- 2D Dubins car dynamics (position + heading, speed + angular rate controls)
- Analytical circular reference path (no Linterp, no pre-solved trajectory)
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
center = np.array([0.0, 0.0])  # Center of the reference circle
total_arc_length = 2 * np.pi * R_circle  # One full lap

###############################################################################
# MPCC parameters
###############################################################################
n_mpc = 8  # Horizon nodes
horizon_duration = 1.5  # Horizon length [s]

Q_LAG = 1e2  # Lag error weight (high -> accurate progress tracking)
Q_CONTOUR = 1e1  # Contour error weight

###############################################################################
# MPCC problem definition
###############################################################################

# --- States ---
position = ox.State("position", shape=(2,))
position.min = np.array([-10.0, -10.0])
position.max = np.array([10.0, 10.0])
position.initial = np.array([R_circle, 0.0])  # Start on circle at theta=0
position.final = [ox.Free(0.0), ox.Free(0.0)]

heading = ox.State("heading", shape=(1,))
heading.min = np.array([-4 * np.pi])
heading.max = np.array([4 * np.pi])
heading.initial = [0.0]  # Facing tangent (CCW)
heading.final = [ox.Free(0.0)]

progress = ox.State("progress", shape=(1,))  # Arc-length progress theta_hat
progress.min = np.array([-0.5 * total_arc_length])
progress.max = np.array([1.5 * total_arc_length])
progress.initial = np.array([0.0])
progress.final = [ox.Maximize(0.0)]

lag_sum = ox.State("lag_sum", shape=(1,))  # Integrated lag cost
lag_sum.min = np.array([0.0])
lag_sum.max = np.array([1e1])
lag_sum.initial = np.array([0.0])
lag_sum.final = [ox.Minimize(0.0)]

contour_sum = ox.State("contour_sum", shape=(1,))  # Integrated contour cost
contour_sum.min = np.array([0.0])
contour_sum.max = np.array([1e1])
contour_sum.initial = np.array([0.0])
contour_sum.final = [ox.Minimize(0.0)]

# --- Controls ---
speed = ox.Control("speed", shape=(1,))
speed.min = np.array([0.0])
speed.max = np.array([10.0])
speed.guess = np.full((n_mpc, 1), 5.0)

angular_rate = ox.Control("angular_rate", shape=(1,))
angular_rate.min = np.array([-5.0])
angular_rate.max = np.array([5.0])
angular_rate.guess = np.full((n_mpc, 1), 5.0 / R_circle)

progress_rate = ox.Control("progress_rate", shape=(1,))  # d(theta_hat)/dt
progress_rate.min = np.array([0.0])  # Forward only
progress_rate.max = np.array([10.0])
progress_rate.guess = np.full((n_mpc, 1), 5.0)

# --- Reference trajectory (analytical circle) ---
# Circle parameterized by arc-length s: p(s) = center + R * [cos(s/R), sin(s/R)]
# Tangent: t(s) = [-sin(s/R), cos(s/R)]  (unit tangent, already normalized)
angle = progress[0] / R_circle  # s / R
p_ref = ox.Concat(
    center[0] + R_circle * ox.Cos(angle),
    center[1] + R_circle * ox.Sin(angle),
)
tangent = ox.Concat(
    -ox.Sin(angle),
    ox.Cos(angle),
)

# --- Error decomposition (position-only, per Romero 2022 Fig. 2) ---
e = position - p_ref  # Position error vector (2,)

# Lag: projection of error onto tangent direction
lag_scalar = ox.Sum(e * tangent)  # Dot product (scalar)
lag_cost = lag_scalar**2

# Contour: Pythagorean decomposition  |e_c|^2 = |e|^2 - |e_l|^2
# Use Sum(e*e) instead of Norm(e)**2 to avoid d(Norm)/de = e/||e|| singularity at e=0
contour_cost = ox.Max(ox.Sum(e * e) - lag_scalar**2, 0.0)

# --- Dynamics ---
dynamics = {
    "position": ox.Concat(
        speed[0] * ox.Sin(heading[0]),
        speed[0] * ox.Cos(heading[0]),
    ),
    "heading": angular_rate[0],
    "progress": progress_rate,
    "lag_sum": Q_LAG * lag_cost,
    "contour_sum": Q_CONTOUR * contour_cost,
}

# --- Constraints ---
states = [position, heading, progress, lag_sum, contour_sum]
controls = [speed, angular_rate, progress_rate]

constraints = []
for state in [position, heading]:
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
    algorithm={"autotuner": ox.ConstantProximalWeight()},
)


###############################################################################
# Initial guesses
###############################################################################
def set_initial_guess(
    theta_start: float = 0.0,
):
    """Set guesses assuming the car drives along the circle from theta_start."""
    ref_speed = 5.0
    theta_end = theta_start + ref_speed * horizon_duration
    arc_guess = np.linspace(theta_start, theta_end, n_mpc)
    angle_guess = arc_guess / R_circle

    position.guess = np.column_stack(
        [
            center[0] + R_circle * np.cos(angle_guess),
            center[1] + R_circle * np.sin(angle_guess),
        ]
    )
    heading.guess = (-angle_guess + np.pi / 2).reshape(-1, 1)
    progress.guess = arc_guess.reshape(-1, 1)
    lag_sum.guess = np.zeros((n_mpc, 1))
    contour_sum.guess = np.zeros((n_mpc, 1))

    speed.guess = np.full((n_mpc, 1), ref_speed)
    angular_rate.guess = np.full((n_mpc, 1), -ref_speed / R_circle)
    progress_rate.guess = np.full((n_mpc, 1), ref_speed)


###############################################################################
# Closed-loop simulation
###############################################################################
def shift_guess(nodes: dict):
    """Shift previous solution by one node for warm-starting."""
    dt = horizon_duration / (n_mpc - 1)

    # Shift states: drop node 0, extrapolate a new final node
    pos_last = nodes["position"][-1]
    hdg_last = nodes["heading"][-1, 0]
    spd_last = nodes["speed"][-1, 0]
    ar_last = nodes["angular_rate"][-1, 0]
    pr_last = nodes["progress_rate"][-1, 0]

    ext_pos = pos_last + dt * spd_last * np.array([np.sin(hdg_last), np.cos(hdg_last)])
    ext_hdg = hdg_last + dt * ar_last
    ext_prog = nodes["progress"][-1, 0] + dt * pr_last

    shifted_progress = np.vstack([nodes["progress"][1:], [[ext_prog]]])
    wrap_offset = (nodes["progress"][1, 0] // total_arc_length) * total_arc_length
    shifted_progress -= wrap_offset

    shifted_heading = np.vstack([nodes["heading"][1:], [[ext_hdg]]])
    hdg_wrap_offset = np.round(nodes["heading"][1, 0] / (2 * np.pi)) * (2 * np.pi)
    shifted_heading -= hdg_wrap_offset

    position.guess = np.vstack([nodes["position"][1:], ext_pos])
    heading.guess = shifted_heading
    progress.guess = shifted_progress
    lag_sum.guess = np.zeros((n_mpc, 1))
    contour_sum.guess = np.zeros((n_mpc, 1))

    speed.guess = np.vstack([nodes["speed"][1:], nodes["speed"][-1:]])
    angular_rate.guess = np.vstack([nodes["angular_rate"][1:], nodes["angular_rate"][-1:]])
    progress_rate.guess = np.vstack([nodes["progress_rate"][1:], nodes["progress_rate"][-1:]])


def update_initial_conditions(nodes: dict):
    """Set initial conditions from node 1 of previous solution (simulate one step)."""
    position.initial = nodes["position"][1]

    hdg_wrap_offset = np.round(nodes["heading"][1, 0] / (2 * np.pi)) * (2 * np.pi)
    heading.initial = nodes["heading"][1] - hdg_wrap_offset

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

    max_steps = 30

    fig = go.Figure()

    # Reference circle
    circle_theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(
        go.Scatter(
            x=center[0] + R_circle * np.cos(circle_theta),
            y=center[1] + R_circle * np.sin(circle_theta),
            mode="lines",
            line={"color": "red", "width": 2, "dash": "dash"},
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
            go.Scatter(
                x=traj[:, 0],
                y=traj[:, 1],
                mode="lines",
                line={"color": colors[step], "width": 2},
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
            f"pos=[{cur_pos[0]:+6.2f}, {cur_pos[1]:+6.2f}]"
        )

        update_initial_conditions(nodes)
        shift_guess(nodes)

    fig.update_layout(title="Dubins Car MPCC", title_x=0.5, template="plotly_dark")
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.show()

    plot_states(results).show()
    plot_scp_iterations(results).show()
