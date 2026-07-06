"""Race car minimum-lap-time optimization with F1 2026-style energy deployment.

Extends ``race_car_openscvx.py`` (Kloeser et al. 2020 spatial bicycle model)
with a hybrid power unit patterned on the 2026 Formula 1 regulations:

  * Peak drive power is split ~55/45 between combustion and electric
    (2026 caps the MGU-K at 350 kW against ~400 kW of combustion power).
  * The battery stores 4 MJ and recovery is limited to ~8 MJ per lap,
    harvested exclusively from the driven axle under braking.
  * The lap is a qualifying lap: state of charge is free at both ends, so
    the car may start full and cross the flag empty. The recovery cap still
    limits how much braking energy can top the battery up along the way.

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
  E     battery energy [J], 0 ≤ E ≤ E_BATT_MAX, free at both ends
  R     cumulative energy recovered this lap [J], R(T) ≤ R_LAP_MAX

Additional controls  u += [deploy, regen]
  deploy   normalised MGU-K deployment  ∈ [0, 1]
  regen    normalised MGU-K harvesting  ∈ [0, 1]

Friction brakes carry the combustion share (55 %) of the force envelope and
regen the electric share (45 %), so braking at full strength harvests energy
— the coupling that makes corner entries the harvesting opportunities they
are in F1. Deploying and harvesting simultaneously only burns energy through
the round-trip efficiency, so the optimizer never does both.

Running the example solves all three power-unit variants in one batched
solve (``solve_batched`` over the ``mgu_k`` and ``ice_share`` parameters) —
the hybrid, an MGU-K failure lap (electric off), and an unrestricted ICE lap
with the full envelope — then polishes with a few warm-start continuation
rounds and races the cars on one Viser track. The failure lap shows what the
electric system is worth to this car; the unrestricted lap shows what the
energy regulations cost, since with free fuel an equal-peak-power pure ICE
strictly dominates the hybrid.

Objective: free final time T, minimise T subject to s(T) = pathlength.
"""

from __future__ import annotations

import copy
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
E_START_GUESS = 0.9 * E_BATT_MAX  # quali lap: guess a near-full start, drained by the flag

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

# Combustion throttle / friction brake. Negative D is friction braking, which
# carries only the combustion share of the envelope — full-strength braking
# requires harvesting the electric share through regen.
D_throt = ox.State("D", shape=(1,))
D_throt.min = [-1.0]
D_throt.max = [1.0]
D_throt.initial = [ox.Free(0.9)]  # flying lap: cross the line on the throttle
D_throt.final = [ox.Free(0.0)]
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
# Qualifying lap: state of charge is free at both ends — the optimizer picks
# its starting charge and may cross the flag empty. The guess drains from
# near-full (just off the capacity bound) to empty to match.
E_batt.initial = [ox.Free(E_START_GUESS)]
E_batt.final = [ox.Free(0.0)]
E_batt.guess = np.linspace(E_START_GUESS, 0.0, N).reshape(-1, 1)

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

# MGU-K deployment and harvesting, held FOH so power ramps rather than
# steps. They stay separate controls because the battery sees them
# asymmetrically — harvest pays the round-trip efficiency and counts against
# the recovery cap — whereas a single signed control would put max(·, 0)
# kinks in the dynamics. Overlap is self-penalizing through η.
deploy = ox.Control("deploy", shape=(1,), parameterization="FOH")
deploy.min = [0.0]
deploy.max = [1.0]
deploy.guess = 0.3 * np.ones((N, 1))

regen = ox.Control("regen", shape=(1,), parameterization="FOH")
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

# Power-unit switches. One batched solve over these parameters yields the
# three cars raced in the Viser comparison:
#   (ice_share=0.55, mgu_k=1)  hybrid
#   (ice_share=0.55, mgu_k=0)  MGU-K failure — loses electric drive *and* the
#                              electric share of its braking
#   (ice_share=1.0,  mgu_k=0)  unrestricted ICE with the full envelope — what
#                              the energy regulations cost in lap time
mgu_k = ox.Parameter("mgu_k", shape=(), value=1.0)
ice_share = ox.Parameter("ice_share", shape=(), value=ICE_SHARE)

# Drive-force envelope, split between the combustion engine and the MGU-K
F_env = ox.Constant(Cm1) - ox.Constant(Cm2) * v[0]
F_ice = ice_share * F_env * D_throt[0]
F_elec = mgu_k * ox.Constant(ELEC_SHARE) * F_env * (deploy[0] - regen[0])

# Longitudinal tyre force [N]
Fxd = (
    F_ice
    + F_elec
    - ox.Constant(Cr2) * v[0] ** 2
    - ox.Constant(Cr0) * ox.Tanh(ox.Constant(5.0) * v[0])
)

# Battery power flows: deployment drains at wheel power, harvesting charges
# through the round-trip efficiency.
P_deploy = mgu_k * ox.Constant(ELEC_SHARE) * F_env * deploy[0] * v[0]
P_harvest = mgu_k * ox.Constant(ETA_BATT) * ox.Constant(ELEC_SHARE) * F_env * regen[0] * v[0]

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

constraints.append(ox.ctcs(a_lat**2 + a_long**2 <= A_MAX**2, penalty="huber"))

# ── Problem ────────────────────────────────────────────────────────────────────
# Any convex backend handles the batched three-car solve below: the default
# (CVXPy) runs the subproblems sequentially under the hood, while a JAX-native
# backend — e.g. solver={"backend": "qpax"} — vectorizes all three cars'
# subproblems into a single XLA program.
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
        # lam_prox/lam_cost from a 4x4 log-grid swept in one solve_batched call
        # over algorithm hyperparameters: this pair is the fastest converged
        # config; pushing lam_cost/lam_prox much past ~10 stops converging.
        "lam_prox": 3e-1,
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


def batch_element(results, b: int):
    """Unbatched view of car ``b`` from post-processed batched results.

    ``solve_batched`` stacks every array with a leading batch axis; the
    plotting helpers and Viser servers consume one car at a time.
    """
    car = copy.copy(results)
    car.converged = bool(np.asarray(results.converged).reshape(-1)[b])
    car.t_final = float(np.asarray(results.t_final).reshape(-1)[b])
    car.nodes = {k: np.asarray(v)[b] for k, v in results.nodes.items()}
    car.trajectory = {k: np.asarray(v)[b] for k, v in results.trajectory.items()}
    car.t_full = np.asarray(results.t_full)[b]
    car.x_full = np.asarray(results.x_full)[b]
    car.u_full = np.asarray(results.u_full)[b]
    return car


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

    # All three cars in one batched solve: element b of each parameter array
    # is one car — hybrid, MGU-K failure, unrestricted ICE.
    cars = {
        "mgu_k": np.array([1.0, 0.0, 0.0]),
        "ice_share": np.array([ICE_SHARE, ICE_SHARE, 1.0]),
    }
    results = problem.solve_batched(parameters=cars)

    # Warm-start continuation: SCP anchors each solve to its guess, so a single
    # solve stops well short of the optimum on this problem. Re-anchoring at
    # the previous solution resumes the descent — a few rounds recover over a
    # second of lap time per car and keep the physical ordering of the three
    # cars intact; returns diminish quickly beyond this.
    for _ in range(4):
        results = problem.solve_batched(
            parameters=cars,
            x_guess=np.asarray(results.x),
            u_guess=np.asarray(results.u),
        )

    results = problem.post_process_batched(results)

    hybrid, ice, full_ice = (batch_element(results, b) for b in range(3))

    nodes = hybrid.nodes
    print("\n=== F1 2026 Energy Race Car Results ===")
    print(f"  Lap time     : {nodes['time'][-1, 0]:.3f} s")
    print(f"  Final s      : {nodes['s'][-1, 0]:.4f} m  (target {pathlength:.4f} m)")
    print(f"  Max speed    : {nodes['v'].max():.3f} m/s")
    print(f"  Recovered    : {nodes['R'][-1, 0]:.4f} J  (cap {R_LAP_MAX:.4f} J)")
    print(
        f"  Battery swing: {nodes['E'].min():.4f} – {nodes['E'].max():.4f} J"
        f"  (capacity {E_BATT_MAX:.4f} J)"
    )
    print(f"  Converged    : {hybrid.converged}")

    lap_hybrid = nodes["time"][-1, 0]
    lap_ice = ice.nodes["time"][-1, 0]
    lap_full_ice = full_ice.nodes["time"][-1, 0]
    print(f"  MGU-K failure lap: {lap_ice:.3f} s  ({lap_ice - lap_hybrid:+.3f} s vs hybrid)")
    print(
        f"  Full-power ICE   : {lap_full_ice:.3f} s  ({lap_full_ice - lap_hybrid:+.3f} s vs hybrid)"
    )

    plot_states(hybrid).show()
    plot_controls(hybrid).show()
    plot_race_results(hybrid)

    from race_car_viser import (
        create_race_car_chase_viser_server,
        create_race_car_comparison_viser_server,
    )

    comparison_server = create_race_car_comparison_viser_server(
        [hybrid, ice, full_ice],
        labels=["hybrid", "MGU-K failure", "full-power ICE"],
        colors=[(150, 70, 200), (90, 140, 235), (220, 35, 45)],
        track_file=TRACK_FILE,
        lane_width=n.max[0],
        distance_marker_step=None,  # clean look — set "auto" to bring markers back
    )
    chase_server = create_race_car_chase_viser_server(
        hybrid,
        track_file=TRACK_FILE,
        lane_width=n.max[0],
        distance_marker_step=None,
        title="Hybrid",
    )
    print("Comparison overview and hybrid chase camera are on separate Viser ports.")
    chase_server.sleep_forever()
