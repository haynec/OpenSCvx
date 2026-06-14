"""Minimum time-to-climb of a supersonic aircraft (Bryson's interceptor).

This is the classic minimum time-to-climb problem for a supersonic aircraft,
taken from

    Bryson, A. E., Desai, M. N. and Hoffman, W. C., "Energy-State Approximation
    in Performance Optimization of Supersonic Aircraft," Journal of Aircraft,
    Vol. 6, No. 6, 1969, pp. 481-488.

and reproduced as a benchmark in the GPOPS-II user's guide. The aircraft starts
at sea level in level flight and must reach 20 km altitude at a prescribed speed
and flight-path angle in minimum time. The optimal trajectory famously dives
through the transonic region to trade altitude for kinetic energy before
zoom-climbing — energy management, not a monotonic climb.

Point-mass dynamics over a spherical, non-rotating Earth (state ``[h, v, gamma,
m]`` = altitude, speed, flight-path angle, mass; control ``alpha`` = angle of
attack)::

    h_dot     = v sin(gamma)
    v_dot     = (T cos(alpha) - D) / m - mu sin(gamma) / r^2
    gamma_dot = (T sin(alpha) + L) / (m v) + cos(gamma) (v / r - mu / (v r^2))
    m_dot     = -T / (g0 Isp)

with ``r = Re + h``, dynamic pressure ``q = 0.5 rho v^2``, lift ``L = q S CL``
and drag ``D = q S CD``, ``CL = Clalpha alpha`` and ``CD = CD0 + eta Clalpha
alpha^2``. The objective is the free final time.

Tabulated data and interpolation
---------------------------------
The aircraft model is defined by lookup tables, and the choice of interpolant
matters for SCP: the dynamics are linearized every iteration, so a C0 interpolant
(``Linterp``) injects a piecewise-constant Jacobian with jumps at every
breakpoint, which tends to chatter. We use ``Cinterp`` (cubic spline) for the
smooth, monotone atmosphere tables — density ``rho(h)`` and speed of sound
``a(h)`` from the U.S. 1976 Standard Atmosphere, truncated to the climb envelope.

For the aerodynamic coefficients ``CD0(M)``, ``Clalpha(M)``, ``eta(M)`` we use
PCHIP (``Cinterp(..., method="pchip")``): those have a sharp transonic peak that a
plain cubic overshoots non-physically (e.g. ``eta`` dipping ~30% below its
tabulated floor), right in the dash's Mach band. PCHIP is shape-preserving — no
overshoot — and matches the piecewise (flat below M=0.8, spline above) shape GPOPS
uses for the same data.

Engine thrust ``T(h, M)`` is a 2-D table, and the only 2-D primitive available is
bilinear (``Bilerp``, C0). GPOPS used a 2-D spline here; bilinear is a faithful
enough stand-in and converges with default settings. If the kinks across grid
lines ever stall convergence, pre-fit a smooth surface offline and resample onto
a finer grid — ``Bilerp`` stays the evaluator.

States are kept in SI units; OpenSCvx auto-scales them from their min/max bounds,
so the tables stay in natural units.
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
# Physical constants (SI)
# ---------------------------------------------------------------------------
Re = 6378145.0  # Earth radius (m)
mu = 3.986e14  # Earth gravitational parameter (m^3 / s^2)
S = 49.2386  # aerodynamic reference area (m^2)
g0 = 9.80665  # standard gravity (m / s^2)
Isp = 1600.0  # specific impulse (s)

# ---------------------------------------------------------------------------
# Tabulated aircraft data
# ---------------------------------------------------------------------------
# U.S. 1976 Standard Atmosphere, truncated to the climb envelope
# (altitude m, density kg/m^3, speed of sound m/s).
atmosphere = np.array(
    [
        [-2000.0, 1.478e00, 3.479e02],
        [0.0, 1.225e00, 3.403e02],
        [2000.0, 1.007e00, 3.325e02],
        [4000.0, 8.193e-01, 3.246e02],
        [6000.0, 6.601e-01, 3.165e02],
        [8000.0, 5.258e-01, 3.081e02],
        [10000.0, 4.135e-01, 2.995e02],
        [12000.0, 3.119e-01, 2.951e02],
        [14000.0, 2.279e-01, 2.951e02],
        [16000.0, 1.665e-01, 2.951e02],
        [18000.0, 1.216e-01, 2.951e02],
        [20000.0, 8.891e-02, 2.951e02],
        [22000.0, 6.451e-02, 2.964e02],
        [24000.0, 4.694e-02, 2.977e02],
        [26000.0, 3.426e-02, 2.991e02],
        [28000.0, 2.508e-02, 3.004e02],
        [30000.0, 1.841e-02, 3.017e02],
        [32000.0, 1.355e-02, 3.030e02],
    ]
)
alt_table = atmosphere[:, 0]
rho_table = atmosphere[:, 1]
sos_table = atmosphere[:, 2]

# Aerodynamic coefficients vs. Mach number (Bryson 1969).
mach_table = np.array([0.0, 0.4, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8])
Clalpha_table = np.array([3.44, 3.44, 3.44, 3.58, 4.44, 3.44, 3.01, 2.86, 2.44])
CD0_table = np.array([0.013, 0.013, 0.013, 0.014, 0.031, 0.041, 0.039, 0.036, 0.035])
eta_table = np.array([0.54, 0.54, 0.54, 0.75, 0.79, 0.78, 0.89, 0.93, 0.93])

# Engine thrust (N) on an (altitude, Mach) grid. The raw table is indexed
# [Mach, altitude] and given in 1000-lbf; convert to Newtons and transpose so
# that thrust_grid[i, j] is the thrust at (thrust_alt[i], thrust_mach[j]).
thrust_mach = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
thrust_alt = 304.8 * np.array([0.0, 5, 10, 15, 20, 25, 30, 40, 50, 70])
thrust_lbf = np.array(
    [
        [24.2, 24.0, 20.3, 17.3, 14.5, 12.2, 10.2, 5.7, 3.4, 0.1],
        [28.0, 24.6, 21.1, 18.1, 15.2, 12.8, 10.7, 6.5, 3.9, 0.2],
        [28.3, 25.2, 21.9, 18.7, 15.9, 13.4, 11.2, 7.3, 4.4, 0.4],
        [30.8, 27.2, 23.8, 20.5, 17.3, 14.7, 12.3, 8.1, 4.9, 0.8],
        [34.5, 30.3, 26.6, 23.2, 19.8, 16.8, 14.1, 9.4, 5.6, 1.1],
        [37.9, 34.3, 30.4, 26.8, 23.3, 19.8, 16.8, 11.2, 6.8, 1.4],
        [36.1, 38.0, 34.9, 31.3, 27.3, 23.6, 20.1, 13.4, 8.3, 1.7],
        [36.1, 36.6, 38.5, 36.1, 31.6, 28.1, 24.2, 16.2, 10.0, 2.2],
        [36.1, 35.2, 42.1, 38.7, 35.7, 32.0, 28.1, 19.3, 11.9, 2.9],
        [36.1, 33.8, 45.7, 41.3, 39.8, 34.6, 31.1, 21.7, 13.3, 3.1],
    ]
)
thrust_grid = (4448.222 * thrust_lbf).T  # -> N, shape (alt, Mach)

# ---------------------------------------------------------------------------
# Boundary conditions and limits
# ---------------------------------------------------------------------------
alt0, altf = 0.0, 19994.88  # altitude (m)
speed0, speedf = 129.314, 295.092  # speed (m/s)
fpa0, fpaf = 0.0, 0.0  # flight-path angle (rad)
mass0 = 19050.864  # initial mass (kg)

# Free per-node time dilation (Time.uniform_time_grid=False) concentrates nodes
# where the dynamics move fastest, but the node-count floor is set by dynamics
# resolution, not the CTCS constraints: below ~50 nodes the solve closes only via
# virtual control and the propagated trajectory undershoots the terminal altitude.
# N=80 is propagation-consistent (t_f ~ 325 s) — verify with results.trajectory.
n = 80  # number of nodes
tf_guess = 300.0  # initial guess for the (free) final time (s)

# ---------------------------------------------------------------------------
# Initial guess — delay the climb
# ---------------------------------------------------------------------------
# Minimum time-to-climb is nonconvex: a naive linear-climb guess lands in a
# suboptimal "climb, then accelerate at altitude" basin (t_f ~ 345 s), while the
# global optimum dashes at low altitude to build kinetic energy, then zoom-climbs.
# Since SCvx is local, the guess only has to pick that basin, not supply the
# answer: a short hold near the floor (dash_fraction) before climbing is enough —
# the solver discovers the dash / zoom / plateau / zoom structure itself.
dash_fraction = 0.15
_climb = np.clip((np.linspace(0.0, 1.0, n) - dash_fraction) / (1.0 - dash_fraction), 0.0, 1.0)

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
h = ox.State("altitude", shape=(1,))
h.min = np.array([0.0])
h.max = np.array([21031.2])
h.initial = np.array([alt0])
h.final = np.array([altf])
h.guess = (altf * _climb).reshape(-1, 1)  # flat at the floor, then a linear climb

v = ox.State("speed", shape=(1,))
v.min = np.array([5.0])
v.max = np.array([1000.0])
v.initial = np.array([speed0])
v.final = np.array([speedf])
v.guess = np.linspace(speed0, speedf, n).reshape(-1, 1)

gamma = ox.State("flight_path_angle", shape=(1,))
gamma.min = np.array([np.deg2rad(-40.0)])
gamma.max = np.array([np.deg2rad(40.0)])
gamma.initial = np.array([fpa0])
gamma.final = np.array([fpaf])
gamma.guess = np.zeros((n, 1))

mass = ox.State("mass", shape=(1,))
mass.min = np.array([22.0])
mass.max = np.array([20410.0])
mass.initial = np.array([mass0])
mass.final = [ox.Free(mass0)]  # final mass is free (fuel burned to climb)
mass.guess = np.linspace(mass0, 0.85 * mass0, n).reshape(-1, 1)

# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------
alpha = ox.Control("angle_of_attack", shape=(1,))
alpha.min = np.array([np.deg2rad(-45.0)])
alpha.max = np.array([np.deg2rad(45.0)])
alpha.guess = np.zeros((n, 1))

states = [h, v, gamma, mass]
controls = [alpha]

# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
hs, vs, fpas, ms = h[0], v[0], gamma[0], mass[0]
a = alpha[0]

r = hs + Re
rho = ox.Cinterp(hs, alt_table, rho_table)
sos = ox.Cinterp(hs, alt_table, sos_table)
mach = vs / sos

# PCHIP (shape-preserving) for the aero coefficients: their sharp transonic peak
# makes a plain cubic overshoot non-physically (e.g. eta dipping ~30% below its
# tabulated floor), right in the dash's Mach band.
CD0 = ox.Cinterp(mach, mach_table, CD0_table, method="pchip")
Clalpha = ox.Cinterp(mach, mach_table, Clalpha_table, method="pchip")
eta = ox.Cinterp(mach, mach_table, eta_table, method="pchip")
thrust = ox.Bilerp(hs, mach, thrust_alt, thrust_mach, thrust_grid)

CL = Clalpha * a
CD = CD0 + eta * Clalpha * a**2
q = 0.5 * rho * vs**2
lift = q * S * CL
drag = q * S * CD

dynamics = {
    "altitude": vs * ox.Sin(fpas),
    "speed": (thrust * ox.Cos(a) - drag) / ms - mu * ox.Sin(fpas) / r**2,
    "flight_path_angle": (thrust * ox.Sin(a) + lift) / (ms * vs)
    + ox.Cos(fpas) * (vs / r - mu / (vs * r**2)),
    "mass": -thrust / (g0 * Isp),
}

# ---------------------------------------------------------------------------
# Constraints: keep every state inside its box along the trajectory.
# ---------------------------------------------------------------------------
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# ---------------------------------------------------------------------------
# Free final time, minimized.
# ---------------------------------------------------------------------------
time = ox.Time(
    initial=0.0,
    final=("minimize", tf_guess),
    min=0.0,
    max=800.0,
)

# A fixed proximal weight (ConstantProximalWeight) outperforms the adaptive
# Augmented-Lagrangian default here: with no reject-driven trust-region shrinkage
# it takes steady moderate steps into a cleaner, faster minimum (t_f ~ 327 s with
# the floor sag essentially gone). lam_prox = 2e-2 is the sweet spot — looser
# reintroduces the floor sag, tighter slows the climb.
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
    """Reproduce the four GPOPS-II diagnostic plots for this benchmark.

    Lays out altitude vs. time, the altitude-vs-speed energy path, flight-path
    angle vs. time, and angle of attack vs. time. The high-resolution propagated
    trajectory is drawn as a line with the discretization nodes overlaid.
    """
    nodes, traj = results.nodes, results.trajectory

    def col(source, key):
        return np.asarray(source[key]).flatten()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Altitude vs. Time",
            "Altitude vs. Speed (energy path)",
            "Flight-Path Angle vs. Time",
            "Angle of Attack vs. Time",
        ),
    )

    # (row, col, x, y) for each panel, in (trajectory, node) units.
    panels = [
        (1, 1, col(traj, "time"), col(traj, "altitude") / 1e3,
         col(nodes, "time"), col(nodes, "altitude") / 1e3),
        (1, 2, col(traj, "speed"), col(traj, "altitude") / 1e3,
         col(nodes, "speed"), col(nodes, "altitude") / 1e3),
        (2, 1, col(traj, "time"), np.rad2deg(col(traj, "flight_path_angle")),
         col(nodes, "time"), np.rad2deg(col(nodes, "flight_path_angle"))),
        (2, 2, col(traj, "time"), np.rad2deg(col(traj, "angle_of_attack")),
         col(nodes, "time"), np.rad2deg(col(nodes, "angle_of_attack"))),
    ]
    for i, (r, c, xt, yt, xn, yn) in enumerate(panels):
        first = i == 0
        fig.add_trace(
            go.Scatter(x=xt, y=yt, mode="lines", name="Propagated",
                       line={"color": "#19d3f3", "width": 2},
                       legendgroup="traj", showlegend=first),
            row=r, col=c,
        )
        fig.add_trace(
            go.Scatter(x=xn, y=yn, mode="markers", name="Nodes",
                       marker={"color": "#ffa600", "size": 5},
                       legendgroup="nodes", showlegend=first),
            row=r, col=c,
        )

    fig.update_xaxes(title_text="Time [s]", row=1, col=1)
    fig.update_yaxes(title_text="Altitude [km]", row=1, col=1)
    fig.update_xaxes(title_text="Speed [m/s]", row=1, col=2)
    fig.update_yaxes(title_text="Altitude [km]", row=1, col=2)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_yaxes(title_text="Flight-path angle [deg]", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=2)
    fig.update_yaxes(title_text="Angle of attack [deg]", row=2, col=2)
    fig.update_layout(title="Supersonic Minimum Time-to-Climb", template="plotly_dark")
    return fig


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    final_time = float(np.asarray(results.nodes["time"]).flatten()[-1])
    print(f"Minimum time-to-climb: {final_time:.2f} s")

    plot_gpops_comparison(results).show()
    # plot_states(results).show()
    # plot_controls(results).show()
    # plot_scp_iterations(results).show()
