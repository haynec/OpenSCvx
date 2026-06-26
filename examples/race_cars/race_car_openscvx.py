"""Race car minimum-lap-time trajectory optimization (OpenSCvx formulation).

Ports the acados NMPC benchmark from:
  Kloeser et al., "NMPC for Racing Using a Singularity-Free Path-Parametric
  Model with Obstacle Avoidance", IFAC World Congress, Berlin, 2020.
  https://www.youtube.com/watch?v=1JDBQXVrZbo

The spatial (path-parametric) bicycle model uses arc length s as the
independent variable of the *state*, while physical time t is the time axis
of the ODE.  The OCP seeks the minimum time required to complete one lap of
the LMS track (s: 0 → pathlength ≈ 8.71 m).

State vector  x = [s, n, α, v, D, δ]
  s     arc-length progress along track centreline [m]
  n     lateral deviation from centreline [m]
  α     heading error w.r.t. track tangent [rad]
  v     longitudinal speed [m/s]
  D     normalised throttle input [-1, 1]
  δ     steering angle [rad]

Control vector  u = [Ḋ, δ̇]
  Ḋ     throttle rate [1/s]
  δ̇     steering rate [rad/s]

Curvature κ(s) is read from the track file and interpolated with a PCHIP
cubic spline via ox.Cinterp, which is JAX-differentiable.

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

import openscvx as ox
from openscvx.plotting import plot_controls, plot_states
from tracks.readDataFcn import getTrack

# ── Track data ─────────────────────────────────────────────────────────────────
sref_data, _, _, _, kapparef_data = getTrack("LMS_Track.txt")
pathlength = float(sref_data[-1])  # ≈ 8.71 m

# Pad slightly beyond [0, pathlength] so Cinterp never extrapolates at the
# boundary nodes when s is at exactly 0 or pathlength.
# Extend by 2.5 m on the low side to cover the s ∈ [-2, 0] warm-up region
# (the track is straight there so kappa ≈ 0 — we clamp with the boundary value).
_pad_lo, _pad_hi = 2.5, 0.5
s_interp = np.concatenate([[sref_data[0] - _pad_lo], sref_data, [pathlength + _pad_hi]])
kappa_interp = np.concatenate([[kapparef_data[0]], kapparef_data, [kapparef_data[-1]]])

# ── Vehicle parameters (Kloeser et al. 2020, Table I) ─────────────────────────
m = 0.043    # vehicle mass [kg]
C1 = 0.5     # front-axle normalised position
C2 = 15.5    # lateral slip-force parameter [1/m]
Cm1 = 0.28   # drive-force coefficient 1 [N]
Cm2 = 0.05   # drive-force coefficient 2 [N·s/m]
Cr0 = 0.011  # rolling-resistance constant [N]
Cr2 = 0.006  # rolling-resistance quadratic [N·s²/m²]

# ── Discretisation ─────────────────────────────────────────────────────────────
N = 80         # shooting nodes
T_guess = 6.0  # initial guess for lap time [s]

# ── States ─────────────────────────────────────────────────────────────────────
S_INIT = 0.0   # matches acados model.x0[0]: 2 m warm-up before the start line

s = ox.State("s", shape=(1,))
s.min = [S_INIT - 0.1]
s.max = [pathlength + 0.1]
s.initial = [S_INIT]
s.final = [pathlength]   # must complete exactly one lap
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
alpha.initial = [0.0]
alpha.final = [ox.Free(0.0)]
alpha.guess = np.zeros((N, 1))

v = ox.State("v", shape=(1,))
v.min = [0.0]
v.max = [6.0]   # generous upper bound; typical racing speed ~ 1–3 m/s
v.initial = [ox.Free(0.0)]
v.final = [ox.Free(0.0)]
# Trapezoidal speed guess: ramp up then hold
_tau = np.linspace(0.0, 1.0, N)
_v_profile = np.where(_tau < 0.2, _tau / 0.2 * 1.5, 1.5)
v.guess = _v_profile.reshape(-1, 1)

D_throt = ox.State("D", shape=(1,))
D_throt.min = [-1.0]
D_throt.max = [1.0]
D_throt.initial = [0.0]
D_throt.final = [ox.Free(0.0)]
D_throt.guess = 0.5 * np.ones((N, 1))

delta = ox.State("delta", shape=(1,))
delta.min = [-0.40]
delta.max = [0.40]
delta.initial = [0.0]
delta.final = [ox.Free(0.0)]
delta.guess = np.zeros((N, 1))

# ── Controls ───────────────────────────────────────────────────────────────────
derD = ox.Control("derD", shape=(1,), parameterization="ZOH")
derD.min = [-10.0]
derD.max = [10.0]
derD.guess = np.zeros((N, 1))

derDelta = ox.Control("derDelta", shape=(1,), parameterization="ZOH")
derDelta.min = [-2.0]
derDelta.max = [2.0]
derDelta.guess = np.zeros((N, 1))

# ── Time: free final time, minimise T ─────────────────────────────────────────
time = ox.Time(
    initial=0.0,
    final=ox.Minimize(T_guess),
    min=0.0,
    max=10.0,
    guess=np.linspace(0.0, T_guess, N).reshape(-1, 1),
    # uniform_time_grid=True,
)

# ── Dynamics ───────────────────────────────────────────────────────────────────
# Curvature κ(s) via PCHIP spline (smooth, monotone-preserving between knots)
kappa = ox.Cinterp(s[0], s_interp, kappa_interp, method="pchip")

# Longitudinal tyre force [N]
#   Fxd = (Cm1 - Cm2·v)·D - Cr2·v² - Cr0·tanh(5v)
Fxd = (ox.Constant(Cm1) - ox.Constant(Cm2) * v[0]) * D_throt[0] \
      - ox.Constant(Cr2) * v[0] ** 2 \
      - ox.Constant(Cr0) * ox.Tanh(ox.Constant(5.0) * v[0])

# Effective slip angle at front tyre
slip_angle = alpha[0] + ox.Constant(C1) * delta[0]

# Arc-length rate ṡ = v·cos(α + C1·δ) / (1 − κ·n)
sdot = (v[0] * ox.Cos(slip_angle)) / (ox.Constant(1.0) - kappa * n[0])

dynamics = {
    "s":     sdot,
    "n":     v[0] * ox.Sin(slip_angle),
    "alpha": v[0] * ox.Constant(C2) * delta[0] - kappa * sdot,
    "v":     (Fxd / ox.Constant(m)) * ox.Cos(ox.Constant(C1) * delta[0]),
    "D":     derD[0],
    "delta": derDelta[0],
}

# ── Constraints ────────────────────────────────────────────────────────────────
states = [s, n, alpha, v, D_throt, delta]
controls = [derD, derDelta]

constraints: list = []

# Lane-keeping: enforce n ∈ [-0.12, 0.12] continuously along the trajectory.
# These are the primary safety constraints — the car must stay within the track
# width at every instant, not just at node points.
# LANE_WIDTH = 0.12   # half-width of the track [m]
# constraints.extend([
#     ox.ctcs(n[0] <=  LANE_WIDTH, penalty="huber"),
#     ox.ctcs(-LANE_WIDTH <= n[0], penalty="huber"),
# ])

# Path constraints on all other states
for state in [s, alpha, v, D_throt, delta]:
    constraints.extend([
        ox.ctcs(state <= state.max, penalty="huber"),
        ox.ctcs(state.min <= state, penalty="huber"),
    ])

# Nonlinear acceleration constraints (prevent tyre saturation)
#   a_lat  = C2·v²·δ + Fxd·sin(C1·δ) / m  ∈ [-4, 4] m/s²
#   a_long = Fxd / m                        ∈ [-4, 4] m/s²
a_lat  = ox.Constant(C2) * v[0] ** 2 * delta[0] \
         + Fxd * ox.Sin(ox.Constant(C1) * delta[0]) / ox.Constant(m)
a_long = Fxd / ox.Constant(m)

A_MAX = 4.0   # [m/s²]

constraints.extend([
    ox.ctcs(a_lat  <=  A_MAX, penalty="huber"),
    ox.ctcs(-A_MAX <= a_lat,  penalty="huber"),
    ox.ctcs(a_long <=  A_MAX, penalty="huber"),
    ox.ctcs(-A_MAX <= a_long, penalty="huber"),
])

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
        # "lam_prox": 1e0,
        "lam_cost": 1e-1,
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
    """Three Plotly figures mirroring the acados benchmark plots.

    All signals come from ``results.trajectory`` (dense single-shot propagation
    from post_process), not from the sparse optimisation nodes.
    """
    import plotly.graph_objects as go
    from time2spatial import transformProj2Orig

    traj = results.trajectory
    t = results.t_full                  # (n_times,)  dense time vector

    s_sol     = traj["s"][:, 0]
    n_sol     = traj["n"][:, 0]
    alpha_sol = traj["alpha"][:, 0]
    v_sol     = traj["v"][:, 0]
    D_sol     = traj["D"][:, 0]
    delta_sol = traj["delta"][:, 0]
    # ── Plot 1: track projection coloured by speed ────────────────────────────
    # Trim warm-up region (s < 0) to match the acados plotting convention.
    lap_start = np.searchsorted(s_sol, 0.0)
    s_sol     = s_sol[lap_start:]
    n_sol     = n_sol[lap_start:]
    alpha_sol = alpha_sol[lap_start:]
    v_sol     = v_sol[lap_start:]
    D_sol     = D_sol[lap_start:]
    delta_sol = delta_sol[lap_start:]
    t         = t[lap_start:]

    # Convert path-parametric (s, n) → Cartesian (x, y) via track geometry
    cart_x, cart_y, _, _ = transformProj2Orig(s_sol, n_sol, alpha_sol, v_sol, "LMS_Track.txt")

    # Track boundaries (±0.12 m from centreline)
    sref_d, xref_d, yref_d, psiref_d, _ = getTrack("LMS_Track.txt")
    dist = 0.12
    xbl = xref_d - dist * np.sin(psiref_d)
    ybl = yref_d + dist * np.cos(psiref_d)
    xbr = xref_d + dist * np.sin(psiref_d)
    ybr = yref_d - dist * np.cos(psiref_d)

    fig2 = go.Figure()

    # Centreline
    fig2.add_trace(go.Scatter(
        x=xref_d, y=yref_d, mode="lines",
        line=dict(color="black", dash="dash", width=1), name="centreline",
    ))
    # Left / right boundaries
    fig2.add_trace(go.Scatter(
        x=xbl, y=ybl, mode="lines",
        line=dict(color="black", width=1.5), name="boundary", showlegend=False,
    ))
    fig2.add_trace(go.Scatter(
        x=xbr, y=ybr, mode="lines",
        line=dict(color="black", width=1.5), showlegend=False,
    ))

    # ── Multishot segments ────────────────────────────────────────────────────
    # Each SCP shooting interval is integrated independently; plotting them as
    # individual thin lines reveals inter-segment defects (gaps = linearisation
    # error still present at this iteration).
    ms = results.multishot_propagation()
    if ms is not None:
        first_ms_seg = True
        for seg_idx in range(ms.n_segments):
            seg_states = ms.segment_states(seg_idx)   # (n_substeps, n_x)
            # columns follow Problem state order: [s, n, alpha, v, D, delta, ...]
            seg_s     = seg_states[:, 0]
            seg_n     = seg_states[:, 1]
            seg_alpha = seg_states[:, 2]
            seg_v     = seg_states[:, 3]
            mx, my, _, _ = transformProj2Orig(seg_s, seg_n, seg_alpha, seg_v, "LMS_Track.txt")
            fig2.add_trace(go.Scatter(
                x=mx, y=my, mode="lines",
                line=dict(color="rgba(100,100,100,0.35)", width=1),
                name="multishot segments" if first_ms_seg else None,
                showlegend=first_ms_seg,
                legendgroup="multishot",
            ))
            first_ms_seg = False

    # Single-shot propagation coloured by speed (on top)
    fig2.add_trace(go.Scatter(
        x=cart_x, y=cart_y, mode="markers",
        marker=dict(color=v_sol, colorscale="Rainbow", size=4,
                    colorbar=dict(title="v [m/s]"), showscale=True),
        name="single-shot (post_process)",
    ))

    # Arc-length distance markers
    for i in range(int(sref_d[-1]) + 1):
        k = int(np.argmin(np.abs(sref_d - i)))
        fig2.add_annotation(x=xref_d[k], y=yref_d[k], text=f"{i}m",
                            showarrow=False, font=dict(size=10))

    fig2.update_layout(
        title=f"OpenSCvx — track projection  (T = {t[-1]:.2f} s)",
        xaxis=dict(title="x [m]", scaleanchor="y"),
        yaxis=dict(title="y [m]"),
        height=600,
    )
    fig2.show()

    # ── Plot 3: lateral & longitudinal acceleration vs bounds ──────────────────
    Fxd_sol = (Cm1 - Cm2 * v_sol) * D_sol - Cr2 * v_sol**2 - Cr0 * np.tanh(5.0 * v_sol)
    a_lat_sol  = C2 * v_sol**2 * delta_sol + Fxd_sol * np.sin(C1 * delta_sol) / m
    a_long_sol = Fxd_sol / m

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=t, y=a_lat_sol,  name="a_lat",  line=dict(color="blue")))
    fig3.add_trace(go.Scatter(x=t, y=a_long_sol, name="a_long", line=dict(color="orange")))
    for sign, show in [(1, True), (-1, False)]:
        fig3.add_hline(
            y=sign * A_MAX,
            line=dict(color="black", dash="dash", width=1),
            annotation_text="±bound" if show else None,
        )
    fig3.update_layout(
        title="OpenSCvx — lateral & longitudinal acceleration",
        xaxis_title="t [s]",
        yaxis_title="acceleration [m/s²]",
        height=400,
    )
    fig3.show()


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    nodes = results.nodes
    print(f"\n=== Race Car Results ===")
    print(f"  Lap time     : {nodes['time'][-1, 0]:.3f} s")
    print(f"  Final s      : {nodes['s'][-1, 0]:.4f} m  (target {pathlength:.4f} m)")
    print(f"  Max speed    : {nodes['v'].max():.3f} m/s")
    print(f"  Converged    : {results.converged}")

    plot_states(results).show()
    plot_controls(results).show()
    plot_race_results(results)

    from race_car_viser import create_race_car_chase_viser_server, create_race_car_viser_server

    overview_server = create_race_car_viser_server(
        results,
        track_file="LMS_Track.txt",
        lane_width=n.max[0],
    )
    chase_server = create_race_car_chase_viser_server(
        results,
        track_file="LMS_Track.txt",
        lane_width=n.max[0],
    )
    print("Overview camera and chase camera are on separate Viser ports (two browser tabs).")
    chase_server.sleep_forever()
