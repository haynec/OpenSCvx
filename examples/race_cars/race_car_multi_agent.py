"""Multi-agent race: K hybrid cars racing wheel-to-wheel via batched MPC.

Puts a whole field of the F1 2026-style hybrid cars from
``race_car_hybrid.py`` on the 4x LMS track at once. Each car runs the same
receding-horizon controller as ``race_car_mpc.py`` — maximise arc-length
progress over a short fixed horizon — and the cars interact only through
collision-avoidance constraints, so overtakes, defence, and energy strategy
all emerge from the optimization rather than being scripted.

One symbolic ``Problem`` describes a single car. Everything that differs
between cars enters through runtime inputs, so advancing the whole field one
MPC step is a single ``solve_batched`` call with a leading agent axis:

  * ``x_initial`` — each car's current state, as batched boundary pins;
  * ``power_scale`` / ``mass_scale`` / ``battery_scale`` — the car's spec;
  * ``opp_s`` / ``opp_n`` — the opponents' latest predicted trajectories.

Cars negotiate by the *communicated plans* scheme standard in decentralized
MPC: at every step each car re-plans against the plans its opponents
published on the previous step (shifted one node so the horizons stay
aligned), keeping an elliptical clearance in track coordinates

    ((s - s_j(t)) / SEP_LONG)² + ((n - n_j(t)) / SEP_LAT)² ≥ 1

from every opponent j, enforced continuously in time (CTCS). All cars share
one fixed, uniform horizon clock, so an opponent's forecast — known at the
horizon nodes — interpolates to a trajectory s_j(t), n_j(t) the constraint
can evaluate between nodes too. The Frenet model makes the constraint this
simple — "gap along the track, gap across it" is exactly the (s, n) state,
with none of the reference-path bookkeeping an MPCC formulation needs
(compare ``examples/mpc/double_integrator_drone_racing.py``).

The race runs ``M_LAPS`` laps from a standing start on an F1-style grid:
cars staggered by ``GRID_ROW_GAP`` down the track, alternating left and
right of the centreline. As in the MPCC example, the ``s`` state lives on a
single lap: the race loop wraps it at each line crossing (pin, warm start,
and published plan together), counts laps, and resets the per-lap recovery
budget ``R`` — so solver scaling and weights never depend on race length,
and the separation gap is lap-periodic so a car being lapped is still an
obstacle. The ``AGENTS`` roster is the single scaling knob — add an entry
and the grid, the batch, the avoidance constraints, and the plots all grow
with it. Spec differences are runtime parameters, so tweaking the field
(a down-on-power engine, an oversize battery, ballast) never recompiles.
The default roster puts a car that is 10% down on power on pole, a healthy
reference car behind it, an overweight car in row two, and a second
reference car charging from the back: every pass has to happen on track.

Run headless (no Plotly/Viser) with ``OPENSCVX_NO_PLOT=1``.
"""

from __future__ import annotations

import os
import sys
import time as _time
from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.insert(0, current_dir)

from tracks.readDataFcn import getTrack

import openscvx as ox

# ── Roster ─────────────────────────────────────────────────────────────────────
# One entry per car, in grid order (index 0 starts on pole). ``power_scale``
# scales the whole drive-force envelope, ``mass_scale`` the chassis mass, and
# ``battery_scale`` the energy store (the per-lap recovery cap is a regulation,
# so it stays fixed). All three are runtime parameters of one compiled problem:
# edit or extend this list and everything downstream follows.
AGENTS = [
    dict(
        name="down on power",
        color=(220, 35, 45),
        power_scale=0.9,
        mass_scale=1.0,
        battery_scale=1.0,
    ),
    dict(
        name="reference spec",
        color=(90, 140, 235),
        power_scale=1.0,
        mass_scale=1.0,
        battery_scale=1.0,
    ),
    dict(
        name="overweight",
        color=(240, 190, 50),
        power_scale=1.0,
        mass_scale=1.15,
        battery_scale=1.0,
    ),
    dict(
        name="reference P4",
        color=(120, 200, 120),
        power_scale=1.0,
        mass_scale=1.0,
        battery_scale=1.0,
    ),
]
K = len(AGENTS)

# Race length. As in the MPCC example, the s state lives on a single lap and
# the race loop wraps it at each line crossing — pin, warm start, and
# published plan together — while counting laps and resetting the per-lap
# recovery budget. Solver scaling, weights, and the curvature spline are
# therefore all independent of race length.
M_LAPS = 1

# ── Track data ─────────────────────────────────────────────────────────────────
# 4x LMS kart track, as in race_car_hybrid.py: long enough that the cars are
# power-limited on the straights and the energy strategy matters.
TRACK_FILE = "LMS_Track_x4.txt"
sref_data, _, _, _, kapparef_data = getTrack(TRACK_FILE)
pathlength = float(sref_data[-1])  # ≈ 34.84 m
RACE_DISTANCE = M_LAPS * pathlength

# Two cars racing side by side need room: open the lane to two car-widths per
# side instead of the single-file 0.12 m of the one-car examples.
LANE_HALF_WIDTH = 0.24

# Overrun past the flag: finished cars keep driving until the last car crosses,
# and every horizon looks this far beyond its own position.
S_OVERRUN = 8.0

# ── Grid (F1 standing start) ───────────────────────────────────────────────────
GRID_ROW_GAP = 0.5  # longitudinal stagger between consecutive grid slots [m]


def grid_slot(i: int) -> tuple[float, float]:
    """(s, n) of grid slot ``i``: one row gap per car, alternating sides."""
    return -(i + 1) * GRID_ROW_GAP, 0.5 * LANE_HALF_WIDTH * (1.0 if i % 2 == 0 else -1.0)


# κ is lap-periodic, so two tiled copies cover every horizon: s is wrapped
# back to the first copy at each line crossing, long before it could reach
# the second copy's end. The low pad covers the deepest grid slot (the track
# is straight there, so the boundary value is right).
_pad_lo = (K + 1) * GRID_ROW_GAP + 1.0
_s_tiled = np.concatenate([sref_data[:-1], sref_data + pathlength])
_kappa_tiled = np.concatenate([kapparef_data[:-1], kapparef_data])
s_interp = np.concatenate([[_s_tiled[0] - _pad_lo], _s_tiled])
kappa_interp = np.concatenate([[_kappa_tiled[0]], _kappa_tiled])

# ── Vehicle parameters (Kloeser et al. 2020, Table I) ─────────────────────────
m = 0.043  # vehicle mass [kg]
C1 = 0.5  # front-axle normalised position
C2 = 15.5  # lateral slip-force parameter [1/m]
Cm1 = 0.28  # drive-force coefficient 1 [N]
Cm2 = 0.05  # drive-force coefficient 2 [N·s/m]
Cr0 = 0.011  # rolling-resistance constant [N]
Cr2 = 0.006  # rolling-resistance quadratic [N·s²/m²]
A_MAX = 4.0  # tyre grip limit — friction-ellipse radius [m/s²]

# ── Hybrid power unit (F1 2026 ratios, RC-car scale — see race_car_hybrid.py) ──
ICE_SHARE = 0.55  # combustion share of the drive-force envelope
ELEC_SHARE = 0.45  # electric (MGU-K) share of the drive-force envelope
ETA_BATT = 0.90  # battery round-trip efficiency, charged on the harvest side

T_LAP_KART = 6.0  # power-unit sizing reference: lap of the unscaled track [s]
P_PEAK = Cm1**2 / (4.0 * Cm2)  # peak envelope wheel power ≈ 0.39 W
P_ELEC_PEAK = ELEC_SHARE * P_PEAK  # peak MGU-K wheel power ≈ 0.18 W

E_BATT_MAX = 0.13 * T_LAP_KART * P_ELEC_PEAK  # reference battery capacity [J]
R_LAP_MAX = 2.0 * E_BATT_MAX  # per-lap recovery cap [J] (a regulation: same for all)
E_CAP_TOP = E_BATT_MAX * max(spec["battery_scale"] for spec in AGENTS)

# ── Separation ellipse (track coordinates) ─────────────────────────────────────
# Longer than it is wide, like the safe zone around a real car, and about as
# tight as the bodywork allows: the 1:43 body is ~0.07 x 0.03 m, so these
# centre-to-centre semi-axes leave roughly half a body of daylight in each
# direction. Cars race nose-to-gearbox and wheel-to-wheel. The daylight is
# also the robustness margin: each car avoids the plan its opponent published
# one step earlier, so in a hard scrap the realized gap can sag a few percent
# into the bubble — half a body absorbs that without bodywork contact.
SEP_LONG = 0.10  # semi-axis along the track [m]
SEP_LAT = 0.05  # semi-axis across the track [m]

# ── MPC horizon ────────────────────────────────────────────────────────────────
# Two seconds reaches through an entire braking zone and out the other side,
# which is what lets a car weigh harvesting now against deploying later.
N_MPC = 21  # horizon nodes
HORIZON_TF = 2.0  # [s] prediction horizon
DT_MPC = HORIZON_TF / (N_MPC - 1)  # time between consecutive nodes = one race step [s]
RACE_TIME_MAX = 15.0 + 20.0 * M_LAPS  # [s] give up if the field has not finished by then
MAX_STEPS = int(np.ceil(RACE_TIME_MAX / DT_MPC))

# Real-time iteration: a fixed SCP budget per step instead of solving each
# horizon to convergence — the shifted warm start carries optimality from one
# step into the next, as in any SQP-style MPC.
SCP_ITERS_PER_STEP = 10

# ── States ─────────────────────────────────────────────────────────────────────
# Boundary values and guesses below describe the pole slot only: every solve
# overrides them per car through the batched ``x_initial`` pins and guesses.
S_POLE, N_POLE = grid_slot(0)
S_MIN = grid_slot(K - 1)[0] - 0.1

s = ox.State("s", shape=(1,))
s.min = [S_MIN]
s.max = [pathlength + S_OVERRUN]
s.initial = [S_POLE]
s.final = [ox.Maximize(0.0)]  # maximise arc-length progress each horizon
s.guess = np.full((N_MPC, 1), S_POLE)

n = ox.State("n", shape=(1,))
n.min = [-LANE_HALF_WIDTH]
n.max = [LANE_HALF_WIDTH]
n.initial = [N_POLE]
n.final = [ox.Free(0.0)]
n.guess = np.full((N_MPC, 1), N_POLE)

alpha = ox.State("alpha", shape=(1,))
alpha.min = [-np.pi / 2]
alpha.max = [np.pi / 2]
alpha.initial = [0.0]
alpha.final = [ox.Free(0.0)]
alpha.guess = np.zeros((N_MPC, 1))

v = ox.State("v", shape=(1,))
v.min = [0.0]
v.max = [6.0]
v.initial = [0.0]  # standing start
v.final = [ox.Free(0.0)]
v.guess = np.zeros((N_MPC, 1))

D_throt = ox.State("D", shape=(1,))
D_throt.min = [-1.0]
D_throt.max = [1.0]
D_throt.initial = [0.0]
D_throt.final = [ox.Free(0.0)]
D_throt.guess = np.zeros((N_MPC, 1))

delta = ox.State("delta", shape=(1,))
delta.min = [-0.40]
delta.max = [0.40]
delta.initial = [0.0]
delta.final = [ox.Free(0.0)]
delta.guess = np.zeros((N_MPC, 1))

E_batt = ox.State("E", shape=(1,))
E_batt.min = [0.0]
E_batt.max = [E_CAP_TOP]  # scaling bound; the binding capacity is per-car below
E_batt.initial = [E_BATT_MAX]  # lights out on a full charge
E_batt.final = [ox.Free(0.0)]
E_batt.guess = np.full((N_MPC, 1), E_BATT_MAX)

E_rec = ox.State("R", shape=(1,))
E_rec.min = [0.0]
E_rec.max = [R_LAP_MAX]
E_rec.initial = [0.0]
E_rec.final = [ox.Free(0.0)]
E_rec.guess = np.zeros((N_MPC, 1))

# Unified state layout: the driver states above in declared order, then the
# time state, then the CTCS integrators appended by constraint augmentation.
DRIVER_STATES = ("s", "n", "alpha", "v", "D", "delta", "E", "R")
COL = {name: i for i, name in enumerate(DRIVER_STATES)}
TIME_COL = len(DRIVER_STATES)

# ── Controls ───────────────────────────────────────────────────────────────────
derD = ox.Control("derD", shape=(1,), parameterization="ZOH")
derD.min = [-10.0]
derD.max = [10.0]
derD.guess = np.zeros((N_MPC, 1))

derDelta = ox.Control("derDelta", shape=(1,), parameterization="ZOH")
derDelta.min = [-2.0]
derDelta.max = [2.0]
derDelta.guess = np.zeros((N_MPC, 1))

deploy = ox.Control("deploy", shape=(1,), parameterization="FOH")
deploy.min = [0.0]
deploy.max = [1.0]
deploy.guess = 0.3 * np.ones((N_MPC, 1))

regen = ox.Control("regen", shape=(1,), parameterization="FOH")
regen.min = [0.0]
regen.max = [1.0]
regen.guess = 0.1 * np.ones((N_MPC, 1))

# ── Time: fixed horizon, uniform grid ─────────────────────────────────────────
# Every car's horizon runs on this same clock, which is what lets node k of one
# car's plan be checked against node k of an opponent's.
time = ox.Time(
    initial=0.0,
    final=HORIZON_TF,
    min=0.0,
    max=HORIZON_TF,
    uniform_time_grid=True,
)

# ── Parameters: car spec and opponent plans ────────────────────────────────────
power_scale = ox.Parameter("power_scale", shape=(), value=1.0)
mass_scale = ox.Parameter("mass_scale", shape=(), value=1.0)
battery_scale = ox.Parameter("battery_scale", shape=(), value=1.0)

# Opponent (s, n) forecasts, one row per horizon node, one column per opponent.
# Refreshed every race step from the other cars' previous plans; parameters are
# hashed by shape, so the updates never recompile. Initialised far behind the
# grid so the constraints start inactive.
if K > 1:
    opp_s = ox.Parameter("opp_s", shape=(N_MPC, K - 1), value=np.full((N_MPC, K - 1), S_MIN - 10.0))
    opp_n = ox.Parameter("opp_n", shape=(N_MPC, K - 1), value=np.zeros((N_MPC, K - 1)))

# ── Dynamics (hybrid spatial bicycle model, per-car spec via parameters) ───────
kappa = ox.Cinterp(s[0], s_interp, kappa_interp, method="pchip")
m_car = ox.Constant(m) * mass_scale

# Drive-force envelope, scaled by engine health and split ICE / MGU-K
F_env = power_scale * (ox.Constant(Cm1) - ox.Constant(Cm2) * v[0])
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
# through the round-trip efficiency and counts against the recovery cap.
P_deploy = ox.Constant(ELEC_SHARE) * F_env * deploy[0] * v[0]
P_harvest = ox.Constant(ETA_BATT) * ox.Constant(ELEC_SHARE) * F_env * regen[0] * v[0]

slip_angle = alpha[0] + ox.Constant(C1) * delta[0]
sdot = (v[0] * ox.Cos(slip_angle)) / (ox.Constant(1.0) - kappa * n[0])

dynamics = {
    "s": sdot,
    "n": v[0] * ox.Sin(slip_angle),
    "alpha": v[0] * ox.Constant(C2) * delta[0] - kappa * sdot,
    "v": (Fxd / m_car) * ox.Cos(ox.Constant(C1) * delta[0]),
    "D": derD[0],
    "delta": derDelta[0],
    "E": P_harvest - P_deploy,
    "R": P_harvest,
}

# ── Constraints ────────────────────────────────────────────────────────────────
states = [s, n, alpha, v, D_throt, delta, E_batt, E_rec]
controls = [derD, derDelta, deploy, regen]

constraints: list = []

# Track limits and path constraints, continuous between nodes.
for state in [s, n, alpha, v, D_throt, delta, E_rec]:
    constraints.extend(
        [
            ox.ctcs(state <= state.max, penalty="huber"),
            ox.ctcs(state.min <= state, penalty="huber"),
        ]
    )

# Battery: the box bound above only scales; the capacity that binds is the car's.
constraints.extend(
    [
        ox.ctcs(E_batt[0] <= battery_scale * ox.Constant(E_BATT_MAX), penalty="huber"),
        ox.ctcs(0.0 <= E_batt[0], penalty="huber"),
    ]
)

# Friction ellipse: lateral and longitudinal grip share one tyre.
a_lat = ox.Constant(C2) * v[0] ** 2 * delta[0] + Fxd * ox.Sin(ox.Constant(C1) * delta[0]) / m_car
a_long = Fxd / m_car

constraints.append(ox.ctcs(a_lat**2 + a_long**2 <= A_MAX**2, penalty="huber"))

# Opponent separation, continuous in time. The forecasts are only known at
# the horizon nodes, but every car shares the same uniform clock, so hat
# weights in the time state turn each opponent's forecast into a
# piecewise-linear trajectory the CTCS penalty can evaluate *between* nodes
# too — no gap for a car to slip through at a node boundary. The ``W_SEP``
# scaling makes contact far dearer than any progress the reward could buy;
# without it, driving through an opponent costs less than lifting, and the
# solver ghost-passes. It also sets how crisply the soft huber penalty holds
# the boundary — the bubble is barely half a body wide, so the few-percent
# sag a lighter weight allows is already wheel-banging.
W_SEP = 4e3

if K > 1:
    hat = ox.Max(1.0 - ox.Abs(time[0] / DT_MPC - np.arange(N_MPC, dtype=float)), 0.0)
    for j in range(K - 1):
        opp_s_t = ox.Sum(hat * opp_s[:, j])
        opp_n_t = ox.Sum(hat * opp_n[:, j])
        # Plans are published in each car's own lap frame, so Δs can be off
        # by whole laps (a freshly wrapped car vs. one still approaching the
        # line, or a car being lapped). The gap is therefore lap-periodic:
        # (L/π)·sin(π·Δs/L) matches Δs exactly near whole-lap multiples — the
        # only place the bubble can bind — and stays harmlessly large in
        # between.
        ds = s[0] - opp_s_t
        ds_wrap = ox.Constant(pathlength / np.pi) * ox.Sin(ox.Constant(np.pi / pathlength) * ds)
        gap = (ds_wrap / SEP_LONG) ** 2 + ((n[0] - opp_n_t) / SEP_LAT) ** 2
        constraints.append(ox.ctcs(W_SEP * (1.0 - gap) <= 0.0, penalty="huber"))

# ── Problem ────────────────────────────────────────────────────────────────────
# One car's horizon problem; the race batches it over the roster. The default
# CVXPy backend solves the K subproblems sequentially; a JAX-native backend
# (e.g. solver={"backend": "qpax"}) vectorizes the whole field into one XLA
# program.
problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=N_MPC,
    float_dtype="float64",
    algorithm={
        # The progress reward must dominate the proximal anchor (states are
        # scaled by their ranges, and s spans the whole 4x track, so a metre
        # of progress is a small scaled step) — too small and each solve
        # "converges" glued to its warm start and the field crawls. lam_vc
        # must in turn dominate the reward, or virtual control buys progress
        # at the horizon's last node instead of driving there.
        "lam_vc": 1e3,
        "lam_prox": 4e0,
        "lam_cost": {"s": 4e1},
        "autotuner": ox.ConstantProximalWeight(),
    },
    # solver=ox.MoreauPTRSolver(),
)
problem.settings.dev.printing = False


# ── Race-loop helpers ──────────────────────────────────────────────────────────


def initial_pins() -> np.ndarray:
    """(K, n_x) boundary pins: every car parked in its grid slot, battery full."""
    pin = np.asarray(problem.state.x_init_pin)
    x0 = np.broadcast_to(pin, (K, pin.size)).copy()
    for i, spec in enumerate(AGENTS):
        s0, n0 = grid_slot(i)
        x0[i, COL["s"]] = s0
        x0[i, COL["n"]] = n0
        x0[i, COL["E"]] = spec["battery_scale"] * E_BATT_MAX
    return x0


def cold_start_guesses() -> tuple[np.ndarray, np.ndarray]:
    """Per-car copies of the default guess, relocated to the grid slots."""
    x = np.repeat(np.asarray(problem.state.x)[None], K, axis=0)
    u = np.repeat(np.asarray(problem.state.u)[None], K, axis=0)
    for i, spec in enumerate(AGENTS):
        s0, n0 = grid_slot(i)
        x[i, :, COL["s"]] = s0
        x[i, :, COL["n"]] = n0
        x[i, :, COL["E"]] = spec["battery_scale"] * E_BATT_MAX
    return x, u


def shift_horizon(plan: np.ndarray) -> np.ndarray:
    """Advance a horizon one node: drop node 0, hold the last node."""
    return np.concatenate([plan[:, 1:], plan[:, -1:]], axis=1)


def shifted_guesses(results) -> tuple[np.ndarray, np.ndarray]:
    """Warm starts for the next step: previous plans shifted one node.

    The horizon clock and the CTCS violation integrators restart from zero
    each solve, so those columns are reset rather than shifted.
    """
    x = shift_horizon(np.asarray(results.x))
    u = shift_horizon(np.asarray(results.u))
    x[:, :, TIME_COL] = np.linspace(0.0, HORIZON_TF, N_MPC)
    x[:, :, TIME_COL + 1 :] = 0.0
    return x, u


def opponent_view(pred: np.ndarray) -> np.ndarray:
    """Each car's view of the others' plans: (K, N) → (K, N, K-1)."""
    return np.stack([np.delete(pred, i, axis=0).T for i in range(K)])


def accelerations(x: np.ndarray, u: np.ndarray, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """(a_lat, a_long) [m/s²] along one car's dense log slice.

    Mirrors the symbolic tyre-force model with the car's spec parameters, so
    the returned points live on the same friction ellipse the solver saw.
    """
    v, D, delta = x[:, COL["v"]], x[:, COL["D"]], x[:, COL["delta"]]
    deploy, regen = u[:, 2], u[:, 3]
    F_env = spec["power_scale"] * (Cm1 - Cm2 * v)
    Fxd = (
        ICE_SHARE * F_env * D
        + ELEC_SHARE * F_env * (deploy - regen)
        - Cr2 * v**2
        - Cr0 * np.tanh(5.0 * v)
    )
    m_car = m * spec["mass_scale"]
    a_lat = C2 * v**2 * delta + Fxd * np.sin(C1 * delta) / m_car
    a_long = Fxd / m_car
    return a_lat, a_long


# ── Plotly visualisation ───────────────────────────────────────────────────────


def plot_race(log: RaceLog) -> None:
    """Track projection, separation, speed/battery striplines, and g-g diagrams.

    Everything is drawn from the dense propagated log; the striplines run
    against race distance in laps rather than time, so the same corner lines
    up vertically across cars and laps.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from time2spatial import transformProj2Orig

    css = [f"rgb{spec['color']}" for spec in AGENTS]
    laps_x = log.dense_x[:, :, COL["s"]] / pathlength  # (K, Td) race distance in laps

    # ── Track projection ───────────────────────────────────────────────────────
    sref_d, xref_d, yref_d, psiref_d, _ = getTrack(TRACK_FILE)
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
    for sign in (-1.0, 1.0):
        fig1.add_trace(
            go.Scatter(
                x=xref_d + sign * LANE_HALF_WIDTH * np.sin(psiref_d),
                y=yref_d - sign * LANE_HALF_WIDTH * np.cos(psiref_d),
                mode="lines",
                line=dict(color="black", width=1.5),
                showlegend=False,
            )
        )
    for i, spec in enumerate(AGENTS):
        x_i = log.dense_x[i]
        cart_x, cart_y, _, _ = transformProj2Orig(
            x_i[:, COL["s"]], x_i[:, COL["n"]], x_i[:, COL["alpha"]], x_i[:, COL["v"]], TRACK_FILE
        )
        fig1.add_trace(
            go.Scatter(
                x=cart_x,
                y=cart_y,
                mode="lines",
                line=dict(color=css[i], width=2),
                name=spec["name"],
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=cart_x[:1],
                y=cart_y[:1],
                mode="markers",
                marker=dict(color=css[i], size=10, symbol="square"),
                name=f"{spec['name']} grid slot",
                showlegend=False,
            )
        )
    fig1.update_layout(
        title=f"Multi-agent race — propagated closed-loop trajectories  ({len(log.sim)} steps)",
        xaxis=dict(title="x [m]", scaleanchor="y"),
        yaxis=dict(title="y [m]"),
        height=600,
    )
    fig1.show()

    # ── Pairwise separation vs the ellipse limit, against race distance ───────
    fig2 = go.Figure()
    for i in range(K):
        for j in range(i + 1, K):
            ds = log.dense_x[i, :, COL["s"]] - log.dense_x[j, :, COL["s"]]
            ds = (pathlength / np.pi) * np.sin(np.pi * ds / pathlength)  # lap-periodic gap
            dn = log.dense_x[i, :, COL["n"]] - log.dense_x[j, :, COL["n"]]
            gap = np.sqrt((ds / SEP_LONG) ** 2 + (dn / SEP_LAT) ** 2)
            fig2.add_trace(
                go.Scatter(
                    x=0.5 * (laps_x[i] + laps_x[j]),
                    y=gap,
                    name=f"{AGENTS[i]['name']} ↔ {AGENTS[j]['name']}",
                )
            )
    fig2.add_hline(y=1.0, line=dict(color="black", dash="dash", width=1), annotation_text="contact")
    fig2.update_layout(
        title="Separation (ellipse metric — below 1 is contact)",
        xaxis_title="race distance [laps]",
        yaxis_title="normalised gap",
        yaxis_type="log",
        height=400,
    )
    fig2.show()

    # ── Speed and battery striplines against race distance ────────────────────
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["v [m/s]", "E [J]"])
    for i, spec in enumerate(AGENTS):
        fig3.add_trace(
            go.Scatter(
                x=laps_x[i],
                y=log.dense_x[i, :, COL["v"]],
                line=dict(color=css[i]),
                name=spec["name"],
            ),
            row=1,
            col=1,
        )
        fig3.add_trace(
            go.Scatter(
                x=laps_x[i],
                y=log.dense_x[i, :, COL["E"]],
                line=dict(color=css[i]),
                name=spec["name"],
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig3.add_hline(
            y=spec["battery_scale"] * E_BATT_MAX,
            line=dict(color=css[i], dash="dash", width=1),
            row=2,
            col=1,
        )
    for lap in range(1, M_LAPS):
        fig3.add_vline(x=float(lap), line=dict(color="black", dash="dot", width=1))
    fig3.update_xaxes(title_text="race distance [laps]", row=2, col=1)
    fig3.update_layout(title="Speed and battery state of charge", height=600)
    fig3.show()

    # ── g-g diagrams: one traction circle per car ──────────────────────────────
    n_cols = 2
    n_rows = (K + n_cols - 1) // n_cols
    fig4 = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[spec["name"] for spec in AGENTS])
    theta = np.linspace(0.0, 2.0 * np.pi, 120)
    for i, spec in enumerate(AGENTS):
        row, col = divmod(i, n_cols)
        a_lat, a_long = accelerations(log.dense_x[i], log.dense_u[i], spec)
        fig4.add_trace(
            go.Scatter(
                x=A_MAX * np.cos(theta),
                y=A_MAX * np.sin(theta),
                mode="lines",
                line=dict(color="black", dash="dash", width=1),
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
        fig4.add_trace(
            go.Scatter(
                x=a_lat[::3],
                y=a_long[::3],
                mode="markers",
                marker=dict(color=css[i], size=3, opacity=0.4),
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
        fig4.update_xaxes(title_text="a_lat [m/s²]", row=row + 1, col=col + 1)
        fig4.update_yaxes(
            title_text="a_long [m/s²]", scaleanchor=f"x{i + 1}", row=row + 1, col=col + 1
        )
    fig4.update_layout(title="g-g diagrams vs the friction ellipse", height=450 * n_rows)
    fig4.show()


def build_viser_panels(log: RaceLog) -> list[dict]:
    """Live plot panels for the race replay: speed, battery, and g-g.

    Each panel is a compact Plotly figure whose last ``K`` traces are
    one-point markers, plus an ``update(t)`` closure that moves those markers
    to the cars' state at race time ``t``. The striplines run against race
    distance in laps; every car's series is trimmed at its own flag crossing
    so its marker parks with the car.
    """
    import plotly.graph_objects as go

    css = [f"rgb{spec['color']}" for spec in AGENTS]
    end = [crossing_index(log.dense_x[i, :, COL["s"]]) for i in range(K)]
    t_car = [log.dense_t[: end[i]] for i in range(K)]
    lap_car = [log.dense_x[i, : end[i], COL["s"]] / pathlength for i in range(K)]

    def compact(fig: go.Figure, title: str, xaxis: str, yaxis: str) -> go.Figure:
        # Dark styling matched to viser's control-panel grey so the panels
        # read as part of the sidebar rather than white cutouts.
        panel_bg = "#1a1b1e"
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=panel_bg,
            plot_bgcolor=panel_bg,
            title=dict(text=title, font=dict(size=13, color="#c1c2c5")),
            xaxis_title=xaxis,
            yaxis_title=yaxis,
            margin=dict(l=45, r=10, t=30, b=35),
            font=dict(size=10, color="#909296"),
            showlegend=False,
        )
        fig.update_xaxes(gridcolor="#2c2e33", zerolinecolor="#2c2e33")
        fig.update_yaxes(gridcolor="#2c2e33", zerolinecolor="#2c2e33")
        return fig

    def stripline_panel(signal: list[np.ndarray], title: str, yaxis: str) -> dict:
        fig = go.Figure()
        for i in range(K):
            stride = max(1, len(t_car[i]) // 500)
            fig.add_trace(
                go.Scatter(
                    x=lap_car[i][::stride],
                    y=signal[i][::stride],
                    mode="lines",
                    line=dict(color=css[i], width=1.5),
                )
            )
        for i in range(K):
            fig.add_trace(
                go.Scatter(
                    x=lap_car[i][:1],
                    y=signal[i][:1],
                    mode="markers",
                    marker=dict(color=css[i], size=10, line=dict(color="white", width=1)),
                )
            )
        compact(fig, title, "race distance [laps]", yaxis)

        def update(t: float) -> None:
            for i in range(K):
                fig.data[K + i].x = (float(np.interp(t, t_car[i], lap_car[i])),)
                fig.data[K + i].y = (float(np.interp(t, t_car[i], signal[i])),)

        return {"figure": fig, "update": update, "aspect": 1.9}

    v_car = [log.dense_x[i, : end[i], COL["v"]] for i in range(K)]
    e_car = [log.dense_x[i, : end[i], COL["E"]] for i in range(K)]
    gg_car = [
        accelerations(log.dense_x[i, : end[i]], log.dense_u[i, : end[i]], spec)
        for i, spec in enumerate(AGENTS)
    ]

    # g-g panel: friction circle, a faint cloud per car, and one live dot each.
    theta = np.linspace(0.0, 2.0 * np.pi, 90)
    gg_fig = go.Figure()
    gg_fig.add_trace(
        go.Scatter(
            x=A_MAX * np.cos(theta),
            y=A_MAX * np.sin(theta),
            mode="lines",
            line=dict(color="gray", dash="dash", width=1),
        )
    )
    for i in range(K):
        a_lat, a_long = gg_car[i]
        stride = max(1, len(a_lat) // 200)
        gg_fig.add_trace(
            go.Scatter(
                x=a_lat[::stride],
                y=a_long[::stride],
                mode="markers",
                marker=dict(color=css[i], size=3, opacity=0.2),
            )
        )
    for i in range(K):
        gg_fig.add_trace(
            go.Scatter(
                x=gg_car[i][0][:1],
                y=gg_car[i][1][:1],
                mode="markers",
                marker=dict(color=css[i], size=10, line=dict(color="white", width=1)),
            )
        )
    compact(gg_fig, "g-g vs friction ellipse", "a_lat [m/s²]", "a_long [m/s²]")
    gg_fig.update_yaxes(scaleanchor="x")

    def update_gg(t: float) -> None:
        for i in range(K):
            gg_fig.data[1 + K + i].x = (float(np.interp(t, t_car[i], gg_car[i][0])),)
            gg_fig.data[1 + K + i].y = (float(np.interp(t, t_car[i], gg_car[i][1])),)

    return [
        stripline_panel(v_car, "speed", "v [m/s]"),
        stripline_panel(e_car, "battery", "E [J]"),
        {"figure": gg_fig, "update": update_gg, "aspect": 1.0},
    ]


# ── Race loop ──────────────────────────────────────────────────────────────────


@dataclass
class RaceLog:
    """Closed-loop record of one race.

    ``sim`` samples the driver states at the MPC rate (``T`` steps of
    ``DT_MPC``) and is what the race logic consumes — finish detection,
    classification, audits. ``dense_*`` stitch each step's *propagated*
    executed interval together (``post_process_batched`` under the hood), so
    playback and plots see the continuous trajectories the cars actually
    drove, not the node samples. ``s`` entries in both logs are cumulative
    race distance.
    """

    sim: np.ndarray  # (T, K, len(DRIVER_STATES)) at the MPC rate
    t_sim: np.ndarray  # (T,)
    finish_time: list  # per car; None if the time cap expired first
    dense_t: np.ndarray  # (Td,)
    dense_x: np.ndarray  # (K, Td, len(DRIVER_STATES))
    dense_u: np.ndarray  # (K, Td, 4)  [derD, derDelta, deploy, regen]


def run_race(max_steps: int = MAX_STEPS) -> RaceLog:
    """Race the roster to the flag and return the :class:`RaceLog`.

    One ``solve_batched`` call advances the whole field by one node
    (``DT_MPC``) per iteration; each step's executed interval is propagated
    densely for the log.
    """
    x0 = initial_pins()
    x_guess, u_guess = cold_start_guesses()
    # Published plans; before lights out every car assumes the field holds station.
    pred_s = x_guess[:, :, COL["s"]].copy()
    pred_n = x_guess[:, :, COL["n"]].copy()

    spec_params = {
        key: np.array([spec[key] for spec in AGENTS])
        for key in ("power_scale", "mass_scale", "battery_scale")
    }

    sim_rows: list[np.ndarray] = []
    dense_t: list[np.ndarray] = []
    dense_x: list[np.ndarray] = []
    dense_u: list[np.ndarray] = []
    finish_time: list = [None] * K
    laps = np.zeros(K)  # completed line crossings per car
    t_now = 0.0
    solve_ms: list[float] = []

    for step in range(max_steps):
        params = dict(spec_params)
        if K > 1:
            params["opp_s"] = opponent_view(shift_horizon(pred_s))
            params["opp_n"] = opponent_view(shift_horizon(pred_n))

        tic = _time.perf_counter()
        results = problem.solve_batched(
            x_initial=jnp.asarray(x0),
            parameters=params,
            x_guess=jnp.asarray(x_guess),
            u_guess=jnp.asarray(u_guess),
            max_iters=SCP_ITERS_PER_STEP,
        )
        solve_ms.append((_time.perf_counter() - tic) * 1e3)

        # Propagate the horizon and keep the executed interval [0, DT_MPC):
        # stitched across steps this is the continuous closed-loop trajectory.
        post = problem.post_process_batched(results)
        t_prop = np.asarray(post.t_full)[0]  # horizon clock, shared by all cars
        keep = t_prop < DT_MPC - 1e-9
        seg_x = np.asarray(post.x_full)[:, keep, : len(DRIVER_STATES)].copy()
        seg_x[:, :, COL["s"]] += laps[:, None] * pathlength
        dense_t.append(t_now + t_prop[keep])
        dense_x.append(seg_x)
        dense_u.append(np.asarray(post.u_full)[:, keep, :4])

        nodes = {name: np.asarray(results.nodes[name]) for name in DRIVER_STATES}
        row = np.stack([nodes[name][:, 0, 0] for name in DRIVER_STATES], axis=1)  # (K, n)
        row[:, COL["s"]] += laps * pathlength  # the log carries cumulative race distance

        # Finish detection: interpolate the flag crossing inside the last step.
        for i in range(K):
            if finish_time[i] is None and row[i, COL["s"]] >= RACE_DISTANCE:
                s_prev = sim_rows[-1][i, COL["s"]] if sim_rows else grid_slot(i)[0]
                frac = (RACE_DISTANCE - s_prev) / max(row[i, COL["s"]] - s_prev, 1e-9)
                finish_time[i] = t_now - DT_MPC * (1.0 - frac)

        sim_rows.append(row)
        status = "  |  ".join(
            f"{AGENTS[i]['name']}: L{lap_of(row[i, COL['s']])}"
            f" s={row[i, COL['s']]:6.2f} v={row[i, COL['v']]:.2f}"
            f" E={row[i, COL['E']]:.3f}"
            for i in range(K)
        )
        print(f"step {step:4d}  t={t_now:6.2f} s  {status}  ({solve_ms[-1]:.0f} ms)")

        if all(t is not None for t in finish_time):
            break

        # Advance the field one node and publish this step's plans.
        for name in DRIVER_STATES:
            x0[:, COL[name]] = nodes[name][:, 1, 0]
        x_guess, u_guess = shifted_guesses(results)
        pred_s = nodes["s"][:, :, 0].copy()
        pred_n = nodes["n"][:, :, 0]

        # Lap handling, as in the MPCC example: when a car takes the line its
        # s frame wraps back one lap — pin, warm start, and published plan
        # together — and its recovery budget R resets. Opponents only ever see
        # the frame through the lap-periodic gap, which is invariant to the
        # shift. A horizon straddling the line charges its first metres of
        # new-lap harvesting to the old lap's budget: conservative by at most
        # one horizon, exact again at the reset.
        for i in range(K):
            if x0[i, COL["s"]] >= pathlength:
                laps[i] += 1
                x0[i, COL["s"]] -= pathlength
                x_guess[i, :, COL["s"]] -= pathlength
                pred_s[i] -= pathlength
                x_guess[i, :, COL["R"]] = np.maximum(x_guess[i, :, COL["R"]] - x0[i, COL["R"]], 0.0)
                x0[i, COL["R"]] = 0.0

        t_now += DT_MPC

    print(f"mean solve {np.mean(solve_ms):.0f} ms, max {np.max(solve_ms):.0f} ms")
    return RaceLog(
        sim=np.stack(sim_rows),
        t_sim=np.arange(len(sim_rows)) * DT_MPC,
        finish_time=finish_time,
        dense_t=np.concatenate(dense_t),
        dense_x=np.concatenate(dense_x, axis=1),
        dense_u=np.concatenate(dense_u, axis=1),
    )


def lap_of(s_cum: float) -> int:
    """Current lap number (1-based) at cumulative race distance ``s_cum``."""
    return int(min(max(s_cum, 0.0) // pathlength + 1, M_LAPS))


def crossing_index(s_cum: np.ndarray) -> int:
    """Number of samples of a cumulative-distance log up to the flag crossing."""
    return min(int(np.searchsorted(s_cum, RACE_DISTANCE)) + 1, len(s_cum))


def print_classification(log: RaceLog) -> list[int]:
    """Print the final order and per-car energy summary; returns the order."""
    sim, finish_time = log.sim, log.finish_time
    order = sorted(range(K), key=lambda i: np.inf if finish_time[i] is None else finish_time[i])
    print("\n=== Race classification ===")
    for place, i in enumerate(order, start=1):
        spec = AGENTS[i]
        if finish_time[i] is None:
            print(f"  P{place}  {spec['name']:<20s}  DNF (time cap)")
            continue
        cross = crossing_index(sim[:, i, COL["s"]])
        at_flag = sim[cross - 1, i]
        # R saws per lap (reset at each line crossing); the race total is the
        # final value plus everything the resets discarded.
        rec = sim[:cross, i, COL["R"]]
        recovered = rec[-1] - np.diff(rec)[np.diff(rec) < 0.0].sum()
        gap = "" if place == 1 else f"  +{finish_time[i] - finish_time[order[0]]:.3f} s"
        print(f"  P{place}  {spec['name']:<20s}  {finish_time[i]:7.3f} s{gap}")
        print(
            f"       battery {sim[0, i, COL['E']]:.3f} → {at_flag[COL['E']]:.3f} J,"
            f" recovered {recovered:.3f} J (cap {R_LAP_MAX:.3f} J/lap)"
        )
    return order


# ── Main: run the race ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    problem.initialize()
    log = run_race()
    order = print_classification(log)

    if os.environ.get("OPENSCVX_NO_PLOT") is None:
        plot_race(log)

        from race_car_viser import (
            create_race_car_chase_viser_server,
            create_race_car_comparison_viser_server,
        )

        # Trim each car's dense log at its own flag crossing so the replay
        # parks it at the line and the finishing gaps stay visible.
        cross = [crossing_index(log.dense_x[i, :, COL["s"]]) for i in range(K)]
        comparison_server = create_race_car_comparison_viser_server(
            simX_list=[log.dense_x[i, : cross[i], :6] for i in range(K)],
            t_sim_list=[log.dense_t[: cross[i]] for i in range(K)],
            labels=[spec["name"] for spec in AGENTS],
            colors=[spec["color"] for spec in AGENTS],
            track_file=TRACK_FILE,
            lane_width=LANE_HALF_WIDTH,
            trim_warmup=False,
            distance_marker_step=None,
            title="Multi-agent race",
            plot_panels=build_viser_panels(log),
        )
        winner = order[0]
        chase_server = create_race_car_chase_viser_server(
            simX=log.dense_x[winner, : cross[winner], :6],
            t_sim=log.dense_t[: cross[winner]],
            track_file=TRACK_FILE,
            lane_width=LANE_HALF_WIDTH,
            trim_warmup=False,
            title=f"Winner — {AGENTS[winner]['name']}",
        )
        print("Race replay and winner chase camera are on separate Viser ports.")
        chase_server.sleep_forever()
