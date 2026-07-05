"""Race car minimum-lap-time optimization with F1 2026-style energy deployment.

Extends ``race_car_openscvx.py`` (Kloeser et al. 2020 spatial bicycle model)
with a hybrid power unit patterned on the 2026 Formula 1 regulations:

  * Peak drive power is split ~55/45 between combustion and electric
    (2026 caps the MGU-K at 350 kW against ~400 kW of combustion power).
  * The battery stores 4 MJ and recovery is limited to ~8 MJ per lap,
    harvested exclusively from the driven axle under braking.
  * The lap is charge-sustaining — E(T) = E(0) — so every joule deployed
    must first be harvested and the lap is infinitely repeatable.

No gearbox is needed: the Kloeser drive force (Cm1 - Cm2·v)·D is already a
driveline *envelope* — wheel force after gearing, falling off with speed like
a power limit — so the hybrid split divides that envelope rather than
modelling discrete gears (which would introduce integer modes SCvx cannot
handle, without changing the energy strategy).

The car races the LMS kart track uniformly scaled 4x (``LMS_Track_x4.txt``).
On the original track the corner-speed profile varies too gently to demand
real braking, so energy strategy degenerates; the longer straights of the
scaled track push the car into its power-limited regime (above ~1.7 m/s the
full envelope cannot reach A_MAX) and end in genuine braking zones, which is
what makes deployment worth optimizing. The power unit stays sized from the
car's ~6 s lap of the *unscaled* track — the car does not grow with the
track — so here the per-lap recovery cap actively binds.

Tyre grip is a friction ellipse, a_lat² + a_long² ≤ A_MAX², rather than the
independent box bounds of the baseline example: corner-exit deployment and
trail-brake harvesting now compete with lateral grip for the same tyre, as
they do in a real car.

Scaled to the RC car by matching the dimensionless ratios that shape the
strategy rather than the raw SI figures:

  * electric share of peak drive power:      45 %
  * battery capacity / MGU-K peak power:     4 MJ / 350 kW ≈ 11.4 s of full
                                             deployment ≈ 13 % of a 90 s lap
  * lap recovery cap / battery capacity:     8 MJ / 4 MJ = 2

Additional states  x += [E, R]
  E     battery energy [J], 0 ≤ E ≤ E_BATT_MAX, charge-sustaining boundary
  R     cumulative energy recovered this lap [J], R(T) ≤ R_LAP_MAX

Additional controls  u += [deploy, regen]
  deploy   normalised MGU-K deployment  ∈ [0, 1]
  regen    normalised MGU-K harvesting  ∈ [0, 1]

Friction brakes carry the combustion share (55 %) of the force envelope and
regen the electric share (45 %), so braking at full strength harvests energy
— the coupling that makes corner entries the harvesting opportunities they
are in F1. Deploying and harvesting simultaneously only burns energy through
the round-trip efficiency, so the optimizer never does both.

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

# ── Hybrid power unit (F1 2026 ratios, RC-car scale) ──────────────────────────
ICE_SHARE = 0.55  # combustion share of the drive-force envelope
ELEC_SHARE = 0.45  # electric (MGU-K) share of the drive-force envelope
ETA_BATT = 0.90  # battery round-trip efficiency, charged on the harvest side

# The power unit belongs to the car, not the track: it is sized from the
# car's ~6 s lap of the unscaled kart track and stays fixed on the 4x track.
T_LAP_KART = 6.0  # power-unit sizing reference: lap of the unscaled track [s]
P_PEAK = Cm1**2 / (4.0 * Cm2)  # peak envelope wheel power ≈ 0.39 W
P_ELEC_PEAK = ELEC_SHARE * P_PEAK  # peak MGU-K wheel power ≈ 0.18 W

E_BATT_MAX = 0.13 * T_LAP_KART * P_ELEC_PEAK  # ≈ 0.14 J — the scaled 4 MJ store
R_LAP_MAX = 2.0 * E_BATT_MAX  # ≈ 0.27 J — the scaled 8 MJ/lap recovery cap
E_INIT = 0.5 * E_BATT_MAX  # start (and finish) half-charged

# ── Discretisation ─────────────────────────────────────────────────────────────
N = 80  # shooting nodes
T_guess = 20.0  # initial guess for lap time [s]

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
alpha.initial = [0.0]
alpha.final = [ox.Free(0.0)]
alpha.guess = np.zeros((N, 1))

v = ox.State("v", shape=(1,))
v.min = [0.0]
v.max = [6.0]
v.initial = [ox.Free(0.0)]
v.final = [ox.Free(0.0)]
# Trapezoidal speed guess: ramp up then hold
_tau = np.linspace(0.0, 1.0, N)
_v_profile = np.where(_tau < 0.2, _tau / 0.2 * 2.0, 2.0)
v.guess = _v_profile.reshape(-1, 1)

# Combustion throttle / friction brake. Negative D is friction braking, which
# carries only the combustion share of the envelope — full-strength braking
# requires harvesting the electric share through regen.
D_throt = ox.State("D", shape=(1,))
D_throt.min = [-1.0]
D_throt.max = [1.0]
D_throt.initial = [0.0]
D_throt.final = [ox.Free(0.0)]
# Guess near full throttle. The SCP trust region anchors solutions to the
# guess, and a mid-range throttle guess quietly caps the converged throttle
# (and lap time) well below optimal.
D_throt.guess = 0.9 * np.ones((N, 1))

delta = ox.State("delta", shape=(1,))
delta.min = [-0.40]
delta.max = [0.40]
delta.initial = [0.0]
delta.final = [ox.Free(0.0)]
delta.guess = np.zeros((N, 1))

E_batt = ox.State("E", shape=(1,))
E_batt.min = [0.0]
E_batt.max = [E_BATT_MAX]
E_batt.initial = [E_INIT]
E_batt.final = [E_INIT]  # charge-sustaining: the lap must be repeatable
E_batt.guess = E_INIT * np.ones((N, 1))

E_rec = ox.State("R", shape=(1,))
E_rec.min = [0.0]
E_rec.max = [R_LAP_MAX]
E_rec.initial = [0.0]
E_rec.final = [ox.Free(0.0)]
E_rec.guess = np.linspace(0.0, 0.5 * R_LAP_MAX, N).reshape(-1, 1)

# ── Controls ───────────────────────────────────────────────────────────────────
derD = ox.Control("derD", shape=(1,), parameterization="ZOH")
derD.min = [-10.0]
derD.max = [10.0]
derD.guess = np.zeros((N, 1))

derDelta = ox.Control("derDelta", shape=(1,), parameterization="ZOH")
derDelta.min = [-2.0]
derDelta.max = [2.0]
derDelta.guess = np.zeros((N, 1))

# MGU-K deployment and harvesting. Direct (unrated) controls: the electric
# machine responds far faster than the SCP time grid resolves.
deploy = ox.Control("deploy", shape=(1,), parameterization="ZOH")
deploy.min = [0.0]
deploy.max = [1.0]
deploy.guess = 0.3 * np.ones((N, 1))

regen = ox.Control("regen", shape=(1,), parameterization="ZOH")
regen.min = [0.0]
regen.max = [1.0]
regen.guess = 0.1 * np.ones((N, 1))

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

# Drive-force envelope, split between the combustion engine and the MGU-K
F_env = ox.Constant(Cm1) - ox.Constant(Cm2) * v[0]
F_ice = ox.Constant(ICE_SHARE) * F_env * D_throt[0]
F_elec = ox.Constant(ELEC_SHARE) * F_env * (deploy[0] - regen[0])

# Longitudinal tyre force [N]
Fxd = (
    F_ice
    + F_elec
    - ox.Constant(Cr2) * v[0] ** 2
    - ox.Constant(Cr0) * ox.Tanh(ox.Constant(5.0) * v[0])
)

# Battery power flows: deployment drains at wheel power, harvesting charges
# through the round-trip efficiency.
P_deploy = ox.Constant(ELEC_SHARE) * F_env * deploy[0] * v[0]
P_harvest = ox.Constant(ETA_BATT) * ox.Constant(ELEC_SHARE) * F_env * regen[0] * v[0]

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
    "E": P_harvest - P_deploy,
    "R": P_harvest,
}

# ── Constraints ────────────────────────────────────────────────────────────────
states = [s, n, alpha, v, D_throt, delta, E_batt, E_rec]
controls = [derD, derDelta, deploy, regen]

constraints: list = []

# Path constraints on all states except the free lateral deviation n.
# The bounds on E enforce the battery capacity; the upper bound on the
# monotone R enforces the per-lap recovery cap.
for state in [s, alpha, v, D_throt, delta, E_batt, E_rec]:
    constraints.extend(
        [
            ox.ctcs(state <= state.max, penalty="huber"),
            ox.ctcs(state.min <= state, penalty="huber"),
        ]
    )

# Friction ellipse: lateral and longitudinal grip share one tyre, so
# corner-exit deployment and trail-brake harvesting compete with cornering.
a_lat = ox.Constant(C2) * v[0] ** 2 * delta[0] + Fxd * ox.Sin(
    ox.Constant(C1) * delta[0]
) / ox.Constant(m)
a_long = Fxd / ox.Constant(m)

A_MAX = 4.0  # [m/s²]

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
    """Track projection coloured by MGU-K power, plus energy and acceleration.

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
    E_sol = traj["E"][:, 0]
    R_sol = traj["R"][:, 0]
    deploy_sol = traj["deploy"][:, 0]
    regen_sol = traj["regen"][:, 0]

    # Trim warm-up region (s < 0) to match the acados plotting convention.
    lap_start = np.searchsorted(s_sol, 0.0)
    sl = slice(lap_start, None)
    s_sol, n_sol, alpha_sol, v_sol = s_sol[sl], n_sol[sl], alpha_sol[sl], v_sol[sl]
    D_sol, delta_sol, E_sol, R_sol = D_sol[sl], delta_sol[sl], E_sol[sl], R_sol[sl]
    deploy_sol, regen_sol, t = deploy_sol[sl], regen_sol[sl], t[sl]

    # Net MGU-K wheel power (deploy > 0, harvest < 0)
    F_env_sol = Cm1 - Cm2 * v_sol
    P_elec_sol = ELEC_SHARE * F_env_sol * (deploy_sol - regen_sol) * v_sol

    # ── Plot 1: track projection coloured by MGU-K power ──────────────────────
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
                color=P_elec_sol,
                colorscale="RdBu_r",
                cmid=0.0,
                size=4,
                colorbar=dict(title="MGU-K power [W]<br>deploy > 0 > harvest"),
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
        title=f"OpenSCvx — F1 2026 energy deployment  (T = {t[-1]:.2f} s)",
        xaxis=dict(title="x [m]", scaleanchor="y"),
        yaxis=dict(title="y [m]"),
        height=600,
    )
    fig1.show()

    # ── Plot 2: battery state of charge and lap recovery vs caps ──────────────
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=E_sol, name="E (battery)", line=dict(color="blue")))
    fig2.add_trace(go.Scatter(x=t, y=R_sol, name="R (recovered)", line=dict(color="green")))
    fig2.add_hline(
        y=E_BATT_MAX,
        line=dict(color="blue", dash="dash", width=1),
        annotation_text="battery capacity",
    )
    fig2.add_hline(
        y=R_LAP_MAX,
        line=dict(color="green", dash="dash", width=1),
        annotation_text="lap recovery cap",
    )
    fig2.update_layout(
        title="OpenSCvx — battery energy and lap recovery",
        xaxis_title="t [s]",
        yaxis_title="energy [J]",
        height=400,
    )
    fig2.show()

    # ── Plot 3: g-g diagram against the friction ellipse ──────────────────────
    Fxd_sol = (
        ICE_SHARE * F_env_sol * D_sol
        + ELEC_SHARE * F_env_sol * (deploy_sol - regen_sol)
        - Cr2 * v_sol**2
        - Cr0 * np.tanh(5.0 * v_sol)
    )
    a_lat_sol = C2 * v_sol**2 * delta_sol + Fxd_sol * np.sin(C1 * delta_sol) / m
    a_long_sol = Fxd_sol / m

    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=A_MAX * np.cos(theta),
            y=A_MAX * np.sin(theta),
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            name="friction ellipse",
        )
    )
    fig3.add_trace(
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
    fig3.update_layout(
        title="OpenSCvx — g-g diagram",
        xaxis=dict(title="a_lat [m/s²]", scaleanchor="y"),
        yaxis=dict(title="a_long [m/s²]"),
        height=600,
    )
    fig3.show()


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    nodes = results.nodes
    print("\n=== F1 2026 Energy Race Car Results ===")
    print(f"  Lap time     : {nodes['time'][-1, 0]:.3f} s")
    print(f"  Final s      : {nodes['s'][-1, 0]:.4f} m  (target {pathlength:.4f} m)")
    print(f"  Max speed    : {nodes['v'].max():.3f} m/s")
    print(f"  Recovered    : {nodes['R'][-1, 0]:.4f} J  (cap {R_LAP_MAX:.4f} J)")
    print(
        f"  Battery swing: {nodes['E'].min():.4f} – {nodes['E'].max():.4f} J"
        f"  (capacity {E_BATT_MAX:.4f} J)"
    )
    print(f"  Converged    : {results.converged}")

    plot_states(results).show()
    plot_controls(results).show()
    plot_race_results(results)

    from race_car_viser import create_race_car_chase_viser_server, create_race_car_viser_server

    overview_server = create_race_car_viser_server(
        results,
        track_file=TRACK_FILE,
        lane_width=n.max[0],
    )
    chase_server = create_race_car_chase_viser_server(
        results,
        track_file=TRACK_FILE,
        lane_width=n.max[0],
    )
    print("Overview camera and chase camera are on separate Viser ports (two browser tabs).")
    chase_server.sleep_forever()
