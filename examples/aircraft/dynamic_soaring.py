"""Minimum wind-gradient dynamic soaring of a glider (Zhao 2004).

The dynamic soaring benchmark from Zhao, *"Optimal Pattern of Glider Dynamic
Soaring,"* Optimal Control Applications and Methods, Vol. 25, 2004, pp. 67-89,
reproduced in the GPOPS-II user's guide. A glider flies one closed loop through
a linear wind shear, extracting enough energy from the gradient to return to
its initial airspeed, flight-path angle, and (heading - 2 pi). The objective is
the *smallest* wind-gradient slope ``beta`` for which such an energy-neutral
loop exists.

Point-mass dynamics over a flat Earth with horizontal wind ``W_x = beta h``
(state ``[x, y, h, v, gamma, psi]`` = position, airspeed, flight-path angle,
heading; controls ``C_L`` = lift coefficient and ``phi`` = bank angle):

    x_dot     = v cos(gamma) sin(psi) + beta h
    y_dot     = v cos(gamma) cos(psi)
    h_dot     = v sin(gamma)
    v_dot     = -(rho S / 2 m)(CD0 + K C_L^2) v^2 - g sin(gamma) - W sin(psi) cos(gamma)
    gamma_dot = (rho S / 2 m) C_L v cos(phi) - g cos(gamma) / v + W sin(psi) sin(gamma) / v
    psi_dot   = ((rho S / 2 m) C_L v sin(phi) - W cos(psi) / v) / cos(gamma)

with wind-shear rate ``W = dW_x/dt = beta v sin(gamma)``. The load factor
``n = (rho S / 2 m g) C_L v^2`` is limited to ``[-2, 5]`` along the whole loop.

Three GPOPS features map onto OpenSCvx as follows:

- The free static parameter ``beta`` becomes a state with zero dynamics, so a
  single optimized value is shared by every node. Its final value carries a
  ``Minimize`` boundary condition — the Mayer objective.
- The periodicity events ``v_f = v_0``, ``gamma_f = gamma_0``, and
  ``psi_f = psi_0 - 2 pi`` become cross-node equality constraints tying node
  ``N-1`` to node 0. They are affine, so they are marked ``.convex()`` and
  enforced exactly by the subproblem solver.
- The load-factor path constraint becomes a CTCS constraint, satisfied in
  continuous time between nodes — which is why far fewer nodes are needed
  here than GPOPS' collocation mesh.

The problem is nonconvex with several locally optimal loops; from the guide's
initial guess the SCP converges to ``beta ~ 0.069 1/s`` with a ~20 s period, a
slightly different basin than the IPOPT solution pictured in the guide. The
solution rides the ``n = 5`` load-factor ceiling through the high-speed bottom
turn, and forward propagation confirms the loop is periodic in continuous time.

Units are English (ft, s, slug), matching the reference.
"""

import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx.plotting import plot_controls, plot_scp_iterations, plot_states

# ---------------------------------------------------------------------------
# Physical constants (English units, Zhao 2004)
# ---------------------------------------------------------------------------
rho = 0.002378  # air density (slug/ft^3)
CD0 = 0.00873  # zero-lift drag coefficient
K = 0.045  # induced drag factor
g = 32.2  # gravitational acceleration (ft/s^2)
m = 5.6  # glider mass (slug)
S = 45.09703  # wing reference area (ft^2)

load_min, load_max = -2.0, 5.0  # load factor limits
beta_min, beta_max = 0.005, 0.15  # wind-gradient slope bounds (1/s)

n = 20  # discretization nodes — CTCS keeps the path constraint tight between them
tf_guess = 24.0  # initial guess for the (free) loop period

# ---------------------------------------------------------------------------
# States — the guess is one circular loop through the shear layer: climbing
# into the wind, diving out of it.
# ---------------------------------------------------------------------------
_t = np.linspace(0.0, tf_guess, n)
_th = 2 * np.pi * _t / tf_guess

position = ox.State("position", shape=(3,))  # [x, y, h] (ft)
position.min = [-1000.0, -1000.0, 0.0]
position.max = [1000.0, 1000.0, 1000.0]
position.initial = [0.0, 0.0, 0.0]
position.final = [0.0, 0.0, 0.0]
position.guess = np.stack(
    [500 * np.cos(_th) - 500, 300 * np.sin(_th), 400 - 400 * np.cos(_th)], axis=1
)

v = ox.State("speed", shape=(1,))  # airspeed (ft/s)
v.min, v.max = [10.0], [350.0]
v.initial, v.final = [ox.Free(200.0)], [ox.Free(200.0)]
v.guess = (80 * (1.5 + np.cos(_th))).reshape(-1, 1)

gamma = ox.State("flight_path_angle", shape=(1,))
gamma.min, gamma.max = [np.deg2rad(-75.0)], [np.deg2rad(75.0)]
gamma.initial, gamma.final = [ox.Free(0.0)], [ox.Free(0.0)]
gamma.guess = (np.pi / 6 * np.sin(_th)).reshape(-1, 1)

psi = ox.State("heading", shape=(1,))
psi.min, psi.max = [-3 * np.pi], [np.pi / 2]
psi.initial, psi.final = [ox.Free(-1.0)], [ox.Free(-1.0 - 2 * np.pi)]
psi.guess = (-1.0 - _t / 4).reshape(-1, 1)

# The wind-gradient slope is a single optimized constant: a state with zero
# dynamics whose final value is the Mayer objective.
beta = ox.State("wind_gradient", shape=(1,))
beta.min, beta.max = [beta_min], [beta_max]
beta.initial, beta.final = [ox.Free(0.08)], [ox.Minimize(0.08)]
beta.guess = np.full((n, 1), 0.08)

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
CL = ox.Control("lift_coefficient", shape=(1,))
CL.min, CL.max = [-0.5], [1.5]
CL.guess = np.full((n, 1), 0.5)

phi = ox.Control("bank_angle", shape=(1,))
phi.min, phi.max = [np.deg2rad(-75.0)], [np.deg2rad(75.0)]
phi.guess = np.full((n, 1), -1.0)

states = [position, v, gamma, psi, beta]
controls = [CL, phi]

# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
hs = position[2]
vs, fpa, head, b = v[0], gamma[0], psi[0], beta[0]
cl, bank = CL[0], phi[0]

aero = rho * S / (2 * m)  # (rho S / 2 m), the common aerodynamic factor
shear = b * vs * ox.Sin(fpa)  # W = dW_x/dt, the wind-shear rate

dynamics = {
    "position": ox.Concat(
        vs * ox.Cos(fpa) * ox.Sin(head) + b * hs,
        vs * ox.Cos(fpa) * ox.Cos(head),
        vs * ox.Sin(fpa),
    ),
    "speed": -aero * (CD0 + K * cl**2) * vs**2
    - g * ox.Sin(fpa)
    - shear * ox.Sin(head) * ox.Cos(fpa),
    "flight_path_angle": aero * cl * vs * ox.Cos(bank)
    - g * ox.Cos(fpa) / vs
    + shear * ox.Sin(head) * ox.Sin(fpa) / vs,
    "heading": (aero * cl * vs * ox.Sin(bank) - shear * ox.Cos(head) / vs) / ox.Cos(fpa),
    "wind_gradient": 0.0 * b,
}

# ---------------------------------------------------------------------------
# Constraints: state boxes and the load-factor limit hold in continuous time;
# the periodicity events tie the last node back to the first.
# ---------------------------------------------------------------------------
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

load_factor = (aero / g) * cl * vs**2
constraints.extend([ox.ctcs(load_factor <= load_max), ox.ctcs(load_min <= load_factor)])

constraints.extend(
    [
        (v.at(n - 1) - v.at(0) == 0.0).convex(),
        (gamma.at(n - 1) - gamma.at(0) == 0.0).convex(),
        (psi.at(n - 1) - psi.at(0) == -2 * np.pi).convex(),
    ]
)

# ---------------------------------------------------------------------------
# Free final time — the loop period is optimized but carries no cost.
# ---------------------------------------------------------------------------
time = ox.Time(
    initial=0.0,
    final=ox.Free(tf_guess),
    min=0.0,
    max=30.0,
)

# A fixed proximal weight converges in ~25 iterations here, where the adaptive
# default stalls just short of the dynamics-feasibility tolerance.
problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    licq_max=1e-8,
    algorithm={"lam_prox": 2e-2, "autotuner": ox.ConstantProximalWeight()},
)


def plot_gpops_comparison(results):
    """Reproduce the six GPOPS-II diagnostic plots for this benchmark.

    Lays out the 3-D soaring loop (x, y, h) followed by airspeed, flight-path
    angle, heading, lift coefficient, and bank angle versus time. The
    high-resolution propagated trajectory is drawn as a line with the
    discretization nodes overlaid.
    """
    nodes, traj = results.nodes, results.trajectory

    def col(source, key, idx=0):
        arr = np.asarray(source[key])
        return arr[:, idx] if arr.ndim > 1 else arr

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "scene"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "Soaring Loop",
            "Airspeed vs. Time",
            "Flight-Path Angle vs. Time",
            "Heading vs. Time",
            "Lift Coefficient vs. Time",
            "Bank Angle vs. Time",
        ),
    )

    line_style = {"color": "#19d3f3", "width": 3}
    marker_style = {"color": "#ffa600", "size": 5}

    fig.add_trace(
        go.Scatter3d(
            x=col(traj, "position", 0),
            y=col(traj, "position", 1),
            z=col(traj, "position", 2),
            mode="lines",
            name="Propagated",
            line=line_style,
            legendgroup="traj",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=col(nodes, "position", 0),
            y=col(nodes, "position", 1),
            z=col(nodes, "position", 2),
            mode="markers",
            name="Nodes",
            marker=marker_style,
            legendgroup="nodes",
        ),
        row=1,
        col=1,
    )

    # (row, col, y-trajectory, y-nodes) for the five time-series panels.
    t_traj, t_nodes = col(traj, "time"), col(nodes, "time")
    panels = [
        (1, 2, col(traj, "speed"), col(nodes, "speed")),
        (
            1,
            3,
            np.rad2deg(col(traj, "flight_path_angle")),
            np.rad2deg(col(nodes, "flight_path_angle")),
        ),
        (2, 1, np.rad2deg(col(traj, "heading")), np.rad2deg(col(nodes, "heading"))),
        (2, 2, col(traj, "lift_coefficient"), col(nodes, "lift_coefficient")),
        (2, 3, np.rad2deg(col(traj, "bank_angle")), np.rad2deg(col(nodes, "bank_angle"))),
    ]
    for r, c, yt, yn in panels:
        fig.add_trace(
            go.Scatter(
                x=t_traj,
                y=yt,
                mode="lines",
                name="Propagated",
                line={"color": line_style["color"], "width": 2},
                legendgroup="traj",
                showlegend=False,
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=t_nodes,
                y=yn,
                mode="markers",
                name="Nodes",
                marker=marker_style,
                legendgroup="nodes",
                showlegend=False,
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="Time [s]", row=r, col=c)

    fig.update_yaxes(title_text="Airspeed [ft/s]", row=1, col=2)
    fig.update_yaxes(title_text="Flight-path angle [deg]", row=1, col=3)
    fig.update_yaxes(title_text="Heading [deg]", row=2, col=1)
    fig.update_yaxes(title_text="Lift coefficient", row=2, col=2)
    fig.update_yaxes(title_text="Bank angle [deg]", row=2, col=3)
    fig.update_layout(
        title="Dynamic Soaring",
        template="plotly_dark",
        scene={
            "xaxis_title": "x [ft]",
            "yaxis_title": "y [ft]",
            "zaxis_title": "h [ft]",
            "aspectmode": "data",
        },
    )
    return fig


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    beta_opt = float(np.asarray(results.nodes["wind_gradient"]).flatten()[-1])
    final_time = float(np.asarray(results.nodes["time"]).flatten()[-1])
    print(f"Minimum wind-gradient slope: {beta_opt:.4f} 1/s (loop period {final_time:.2f} s)")

    plot_gpops_comparison(results).show()
    plot_states(results).show()
    plot_controls(results).show()
    plot_scp_iterations(results).show()
