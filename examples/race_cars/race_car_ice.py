"""Race car minimum-lap-time optimization, pure-combustion ablation.

The same car, track, and lap structure as ``race_car_hybrid.py`` with the
hybrid power unit removed: no battery state, no recovery accounting, no
deploy/harvest controls. The full Kloeser drive envelope (Cm1 - Cm2·v)·D
goes through the throttle alone, so D = -1 is full-strength friction braking
— physically the "unrestricted ICE" car of the hybrid example's three-car
comparison, which strictly dominates the hybrid because its braking and
drive are unmetered.

!!! note "Twin example"
    ``race_car_hybrid.py`` is the full hybrid version; the files differ
    only in the power unit. Everything that shapes the *driving* is
    retained:

    * LMS kart track uniformly scaled 4x (power-limited straights,
      genuine braking zones),
    * friction-ellipse tyre model  a_lat² + a_long² ≤ A_MAX²,
    * flying-lap periodicity on the driving states (n, α, v, D, δ),
    * the trail-braking initial guess.

The point of the ablation is numerical, not physical: the hybrid variant
carries two extra states (E, R) and two extra controls (deploy, regen) that
exist purely for energy strategy. Removing them shrinks the subproblem and
the Jacobians, so this file is the reference for how much of the hybrid
example's solve time and convergence behaviour is paid for the energy
management rather than the racing. Lap time should land near the
unrestricted-ICE car of ``race_car_hybrid.py`` (same envelope, same grip).

State vector  x = [s, n, α, v, D, δ]
Control vector  u = [Ḋ, δ̇]

Objective: free final time T, minimise T subject to s(T) = pathlength.
"""

from __future__ import annotations

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Track loader lives alongside this file
sys.path.insert(0, current_dir)

from tracks.readDataFcn import getTrack

import openscvx as ox
from openscvx.plotting import plot_controls, plot_states

# ── Track data ─────────────────────────────────────────────────────────────────
# LMS kart track scaled 4x: long enough that the car becomes power-limited on
# the straights and must genuinely brake for the corners.
TRACK_FILE = "LMS_Track_x4.txt"
sref_data, _, _, _, kapparef_data = getTrack(TRACK_FILE)
pathlength = float(sref_data[-1])  # ≈ 34.84 m

# Pad slightly beyond [0, pathlength] so Cinterp never extrapolates at the
# boundary nodes when s is at exactly 0 or pathlength.
_pad_lo, _pad_hi = 2.5, 0.5
s_interp = np.concatenate([[sref_data[0] - _pad_lo], sref_data, [pathlength + _pad_hi]])
kappa_interp = np.concatenate([[kapparef_data[0]], kapparef_data, [kapparef_data[-1]]])

# ── Vehicle parameters (Kloeser et al. 2020, Table I) ─────────────────────────
m = 0.043  # vehicle mass [kg]
C1 = 0.5  # front-axle normalised position
C2 = 15.5  # lateral slip-force parameter [1/m]
Cm1 = 0.28  # drive-force coefficient 1 [N]
Cm2 = 0.05  # drive-force coefficient 2 [N·s/m]
Cr0 = 0.011  # rolling-resistance constant [N]
Cr2 = 0.006  # rolling-resistance quadratic [N·s²/m²]
A_MAX = 4.0  # tyre grip limit — friction-ellipse radius [m/s²]

# ── Discretisation ─────────────────────────────────────────────────────────────
N = 80  # shooting nodes
T_guess = 20.0  # initial guess for lap time [s]

# Tie the start-line state to the finish-line state (n, α, v, D, δ). The start
# line *is* the finish line on a flying lap, so without this the free boundary
# is a loophole: an unconstrained v(0) lets the optimizer cross the line at a
# speed the car could never regain.
PERIODIC_LAP = True

# ── Trail-braking throttle guess ───────────────────────────────────────────────
# SCP anchors near its guess, so rather than a flat throttle the guess encodes
# how a lap is driven: brake hardest at turn-in, then feed the throttle back in
# through the corner ("trail braking"). Corners are the contiguous stretches of
# the guess arc-length grid where the track curvature exceeds KAPPA_CORNER;
# through each one the throttle ramps linearly from full braking at entry to
# full throttle at exit, and the straights run flat out.

KAPPA_CORNER = 0.5  # curvature magnitude above which a node counts as cornering [1/m]


def trail_brake_throttle(s_nodes: np.ndarray) -> np.ndarray:
    """Trail-braking throttle profile D(s) ∈ [-1, 1] on the guess grid ``s_nodes``."""
    kappa_nodes = np.interp(s_nodes, s_interp, kappa_interp)
    corner_nodes = np.flatnonzero(np.abs(kappa_nodes) > KAPPA_CORNER)
    D = np.ones_like(s_nodes)
    for corner in np.split(corner_nodes, np.flatnonzero(np.diff(corner_nodes) > 1) + 1):
        D[corner] = np.linspace(-1.0, 1.0, corner.size)
    return D


# ── States ─────────────────────────────────────────────────────────────────────
S_INIT = 0.0

s = ox.State("s", shape=(1,))
s.min = [S_INIT - 0.1]
s.max = [pathlength + 0.1]
s.initial = [S_INIT]
s.final = [pathlength]  # must complete exactly one lap
s.guess = np.linspace(S_INIT, pathlength, N).reshape(-1, 1)

n = ox.State("n", shape=(1,))
n.min = [-0.12]
n.max = [0.12]
n.initial = [ox.Free(0.0)]
n.final = [ox.Free(0.0)]
n.guess = np.zeros((N, 1))

alpha = ox.State("alpha", shape=(1,))
alpha.min = [-np.pi / 2]
alpha.max = [np.pi / 2]
alpha.initial = [ox.Free(0.0)]  # flying lap: heading unconstrained at the line
alpha.final = [ox.Free(0.0)]
alpha.guess = np.zeros((N, 1))

v = ox.State("v", shape=(1,))
v.min = [0.0]
v.max = [6.0]
v.initial = [ox.Free(0.0)]
v.final = [ox.Free(0.0)]
# Flat guess at racing speed — this is a flying lap, not a standing start.
v.guess = 2.0 * np.ones((N, 1))

# Throttle / friction brake: the whole envelope, no combustion/electric split.
D_trail = trail_brake_throttle(s.guess[:, 0])

D_throt = ox.State("D", shape=(1,))
D_throt.min = [-1.0]
D_throt.max = [1.0]
D_throt.initial = [ox.Free(1.0)]  # flying lap: cross the line on the throttle
D_throt.final = [ox.Free(0.0)]
D_throt.guess = D_trail.reshape(-1, 1)

delta = ox.State("delta", shape=(1,))
delta.min = [-0.40]
delta.max = [0.40]
delta.initial = [ox.Free(0.0)] if PERIODIC_LAP else [0.0]
delta.final = [ox.Free(0.0)]
delta.guess = np.zeros((N, 1))

# ── Controls ───────────────────────────────────────────────────────────────────
derD = ox.Control("derD", shape=(1,), parameterization="ZOH")
derD.min = [-10.0]
derD.max = [10.0]
# Throttle rate consistent with the trail-braking profile (Ḋ = derD).
derD.guess = np.clip(
    np.diff(D_trail, append=D_trail[-1]) * (N - 1) / T_guess, derD.min, derD.max
).reshape(-1, 1)

derDelta = ox.Control("derDelta", shape=(1,), parameterization="ZOH")
derDelta.min = [-2.0]
derDelta.max = [2.0]
derDelta.guess = np.zeros((N, 1))

# ── Time: free final time, minimise T ─────────────────────────────────────────
time = ox.Time(
    initial=0.0,
    final=ox.Minimize(T_guess),
    min=0.0,
    max=60.0,
    guess=np.linspace(0.0, T_guess, N).reshape(-1, 1),
)

# ── Dynamics ───────────────────────────────────────────────────────────────────
# Curvature κ(s) via PCHIP spline (smooth, monotone-preserving between knots)
kappa = ox.Cinterp(s[0], s_interp, kappa_interp, method="pchip")

# Longitudinal tyre force [N]: the full drive envelope through one throttle.
Fxd = (
    (ox.Constant(Cm1) - ox.Constant(Cm2) * v[0]) * D_throt[0]
    - ox.Constant(Cr2) * v[0] ** 2
    - ox.Constant(Cr0) * ox.Tanh(ox.Constant(5.0) * v[0])
)

# Effective slip angle at front tyre
slip_angle = alpha[0] + ox.Constant(C1) * delta[0]

# Arc-length rate ṡ = v·cos(α + C1·δ) / (1 − κ·n)
sdot = (v[0] * ox.Cos(slip_angle)) / (ox.Constant(1.0) - kappa * n[0])

dynamics = {
    "s": sdot,
    "n": v[0] * ox.Sin(slip_angle),
    "alpha": v[0] * ox.Constant(C2) * delta[0] - kappa * sdot,
    "v": (Fxd / ox.Constant(m)) * ox.Cos(ox.Constant(C1) * delta[0]),
    "D": derD[0],
    "delta": derDelta[0],
}

# ── Constraints ────────────────────────────────────────────────────────────────
states = [s, n, alpha, v, D_throt, delta]
controls = [derD, derDelta]

constraints: list = []

# Path constraints on all states except the free lateral deviation n.
for state in [s, alpha, v, D_throt, delta]:
    constraints.extend(
        [
            ox.ctcs(state <= state.max, penalty="huber"),
            ox.ctcs(state.min <= state, penalty="huber"),
        ]
    )

# Flying-lap periodicity: convex cross-node equalities pin each driving state
# at the start line to its value at the flag.
if PERIODIC_LAP:
    constraints.extend((x.at(0) == x.at(N - 1)).convex() for x in [n, alpha, v, D_throt, delta])

# Friction ellipse: lateral and longitudinal grip share one tyre.
a_lat = ox.Constant(C2) * v[0] ** 2 * delta[0] + Fxd * ox.Sin(
    ox.Constant(C1) * delta[0]
) / ox.Constant(m)
a_long = Fxd / ox.Constant(m)

constraints.append(ox.ctcs(a_lat**2 + a_long**2 <= A_MAX**2, penalty="huber"))

# ── Problem ────────────────────────────────────────────────────────────────────
problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=N,
    float_dtype="float64",
    licq_max=1e-12,
    algorithm={
        # lam_prox from a cold-solve sweep on this problem: at 1e-1 a single
        # solve reaches the anchored optimum (no warm-start continuation
        # needed, unlike the hybrid twin at 3e-1). Beware loosening it
        # further: the lane is a soft (huber) CTCS penalty, so looser anchors
        # (~1e-2) still report convergence while the propagated trajectory
        # cuts corners — validate a retune against the lane violation, not
        # just the lap time.
        "lam_prox": 1e-1,
        "lam_cost": 3e0,
        "lam_vc": 1e2,
        "autotuner": ox.AugmentedLagrangian(eta_lambda=1e0),
    },
    discretizer={
        "diffrax_kwargs": {"atol": 1e-8, "rtol": 1e-8},
    },
)

problem.settings.prp.atol = 1e-10
problem.settings.prp.rtol = 1e-10


def plot_race_results(results) -> None:
    """Track projection coloured by throttle, plus the g-g diagram.

    All signals come from ``results.trajectory`` (dense single-shot propagation
    from post_process), not from the sparse optimisation nodes.
    """
    import plotly.graph_objects as go
    from time2spatial import transformProj2Orig

    traj = results.trajectory
    t = results.t_full  # (n_times,)  dense time vector

    s_sol = traj["s"][:, 0]
    n_sol = traj["n"][:, 0]
    alpha_sol = traj["alpha"][:, 0]
    v_sol = traj["v"][:, 0]
    D_sol = traj["D"][:, 0]
    delta_sol = traj["delta"][:, 0]

    # Trim warm-up region (s < 0) to match the acados plotting convention.
    lap_start = np.searchsorted(s_sol, 0.0)
    sl = slice(lap_start, None)
    s_sol, n_sol, alpha_sol, v_sol = s_sol[sl], n_sol[sl], alpha_sol[sl], v_sol[sl]
    D_sol, delta_sol, t = D_sol[sl], delta_sol[sl], t[sl]

    # ── Plot 1: track projection coloured by throttle/braking ─────────────────
    cart_x, cart_y, _, _ = transformProj2Orig(s_sol, n_sol, alpha_sol, v_sol, TRACK_FILE)

    sref_d, xref_d, yref_d, psiref_d, _ = getTrack(TRACK_FILE)
    dist = 0.12
    xbl = xref_d - dist * np.sin(psiref_d)
    ybl = yref_d + dist * np.cos(psiref_d)
    xbr = xref_d + dist * np.sin(psiref_d)
    ybr = yref_d - dist * np.cos(psiref_d)

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=xref_d,
            y=yref_d,
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            name="centreline",
        )
    )
    for xb, yb in [(xbl, ybl), (xbr, ybr)]:
        fig1.add_trace(
            go.Scatter(
                x=xb,
                y=yb,
                mode="lines",
                line=dict(color="black", width=1.5),
                showlegend=False,
            )
        )
    fig1.add_trace(
        go.Scatter(
            x=cart_x,
            y=cart_y,
            mode="markers",
            marker=dict(
                color=D_sol,
                colorscale="RdBu_r",
                cmid=0.0,
                size=4,
                colorbar=dict(title="throttle D<br>drive > 0 > brake"),
                showscale=True,
            ),
            name="single-shot (post_process)",
        )
    )
    for i in range(0, int(sref_d[-1]) + 1, 4):
        k = int(np.argmin(np.abs(sref_d - i)))
        fig1.add_annotation(
            x=xref_d[k], y=yref_d[k], text=f"{i}m", showarrow=False, font=dict(size=10)
        )
    fig1.update_layout(
        title=f"OpenSCvx — pure-ICE minimum-time lap  (T = {t[-1]:.2f} s)",
        xaxis=dict(title="x [m]", scaleanchor="y"),
        yaxis=dict(title="y [m]"),
        height=600,
    )
    fig1.show()

    # ── Plot 2: g-g diagram against the friction ellipse ──────────────────────
    Fxd_sol = (Cm1 - Cm2 * v_sol) * D_sol - Cr2 * v_sol**2 - Cr0 * np.tanh(5.0 * v_sol)
    a_lat_sol = C2 * v_sol**2 * delta_sol + Fxd_sol * np.sin(C1 * delta_sol) / m
    a_long_sol = Fxd_sol / m

    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=A_MAX * np.cos(theta),
            y=A_MAX * np.sin(theta),
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            name="friction ellipse",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=a_lat_sol,
            y=a_long_sol,
            mode="markers",
            marker=dict(
                color=v_sol,
                colorscale="Rainbow",
                size=4,
                colorbar=dict(title="v [m/s]"),
                showscale=True,
            ),
            name="trajectory",
        )
    )
    fig2.update_layout(
        title="OpenSCvx — g-g diagram",
        xaxis=dict(title="a_lat [m/s²]", scaleanchor="y"),
        yaxis=dict(title="a_long [m/s²]"),
        height=600,
    )
    fig2.show()


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    nodes = results.nodes
    print("\n=== Pure-ICE Race Car Results ===")
    print(f"  Lap time  : {nodes['time'][-1, 0]:.3f} s")
    print(f"  Final s   : {nodes['s'][-1, 0]:.4f} m  (target {pathlength:.4f} m)")
    print(f"  Max speed : {nodes['v'].max():.3f} m/s")
    print(f"  Converged : {results.converged}")

    plot_states(results).show()
    plot_controls(results).show()
    plot_race_results(results)

    from race_car_viser import (
        create_race_car_chase_viser_server,
        create_race_car_comparison_viser_server,
    )

    overview_server = create_race_car_comparison_viser_server(
        [results],
        labels=["pure ICE"],
        colors=[(220, 35, 45)],
        track_file=TRACK_FILE,
        lane_width=n.max[0],
        distance_marker_step=None,  # clean look — set "auto" to bring markers back
    )
    chase_server = create_race_car_chase_viser_server(
        results,
        track_file=TRACK_FILE,
        lane_width=n.max[0],
        distance_marker_step=None,
        title="Pure ICE",
    )
    print("Track overview and chase camera are on separate Viser ports.")
    chase_server.sleep_forever()
