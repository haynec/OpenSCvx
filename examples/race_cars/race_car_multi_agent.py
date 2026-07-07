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

    ((s - s_j) / SEP_LONG)² + ((n - n_j) / SEP_LAT)² ≥ 1

from every opponent j at every horizon node. All cars share one fixed,
uniform horizon time grid, so node k of every plan refers to the same wall
clock and node-wise separation is meaningful. The Frenet model makes the
constraint this simple — "gap along the track, gap across it" is exactly
the (s, n) state, with none of the reference-path bookkeeping an MPCC
formulation needs (compare ``examples/mpc/double_integrator_drone_racing.py``).

The race is a standing start from an F1-style grid: cars staggered by
``GRID_ROW_GAP`` down the track, alternating left and right of the
centreline. The ``AGENTS`` roster is the single scaling knob — add an entry
and the grid, the batch, the avoidance constraints, and the plots all grow
with it. Spec differences are runtime parameters, so tweaking the field
(a down-on-power engine, an oversize battery, ballast) never recompiles.
The default roster puts a car that is 10% down on power on pole and a
healthy one behind it: the chaser must find a way past on track.

Run headless (no Plotly/Viser) with ``OPENSCVX_NO_PLOT=1``.
"""

from __future__ import annotations

import os
import sys
import time as _time

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
]
K = len(AGENTS)

# ── Track data ─────────────────────────────────────────────────────────────────
# 4x LMS kart track, as in race_car_hybrid.py: long enough that the cars are
# power-limited on the straights and the energy strategy matters.
TRACK_FILE = "LMS_Track_x4.txt"
sref_data, _, _, _, kapparef_data = getTrack(TRACK_FILE)
pathlength = float(sref_data[-1])  # ≈ 34.84 m

# Two cars racing side by side need room: open the lane to two car-widths per
# side instead of the single-file 0.12 m of the one-car examples.
LANE_HALF_WIDTH = 0.24

# Overrun past the flag: finished cars keep driving until the last car crosses,
# and every horizon looks this far beyond its own position.
S_OVERRUN = 6.0

# ── Grid (F1 standing start) ───────────────────────────────────────────────────
GRID_ROW_GAP = 0.5  # longitudinal stagger between consecutive grid slots [m]


def grid_slot(i: int) -> tuple[float, float]:
    """(s, n) of grid slot ``i``: one row gap per car, alternating sides."""
    return -(i + 1) * GRID_ROW_GAP, 0.5 * LANE_HALF_WIDTH * (1.0 if i % 2 == 0 else -1.0)


# Pad the curvature spline so it never extrapolates: below 0 it must cover the
# deepest grid slot, above pathlength the post-finish overrun.
_pad_lo = (K + 1) * GRID_ROW_GAP + 1.0
_pad_hi = S_OVERRUN + 1.0
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
# Longer than it is wide, like the safe zone around a real car: SEP_LONG covers
# a car length plus a braking margin, SEP_LAT one car width of daylight. Two
# cars can run side by side (|Δn| = 2·|grid n| = LANE_HALF_WIDTH ≥ SEP_LAT) but
# never nose-to-tail closer than SEP_LONG.
SEP_LONG = 0.35  # semi-axis along the track [m]
SEP_LAT = 0.12  # semi-axis across the track [m]

# ── MPC horizon ────────────────────────────────────────────────────────────────
N_MPC = 15  # horizon nodes
HORIZON_TF = 1.0  # [s] prediction horizon — covers braking from top speed
DT_MPC = HORIZON_TF / (N_MPC - 1)  # time between consecutive nodes = one race step [s]
RACE_TIME_MAX = 40.0  # [s] give up if the field has not finished by then
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

# Opponent separation: at horizon node k, stay outside every opponent's ellipse,
# evaluated against their forecast position at that same node. Node 0 is the
# pinned current state, so the constraint starts at node 1. The heavy
# ``.weight`` makes the linearization's virtual buffer far dearer than any
# progress the reward could buy — without it, driving through an opponent
# costs less than lifting, and the solver ghost-passes.
W_SEP = 1e3

if K > 1:
    for k in range(1, N_MPC):
        gap = ((s[0] - opp_s[k]) / SEP_LONG) ** 2 + ((n[0] - opp_n[k]) / SEP_LAT) ** 2
        constraints.append((1.0 <= gap).at([k]).weight(W_SEP))

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


# ── Plotly visualisation ───────────────────────────────────────────────────────


def plot_race(sim: np.ndarray, t_sim: np.ndarray) -> None:
    """Track projection per car, pairwise separation vs the limit, speed and
    battery histories — all from the closed-loop log ``sim`` (T, K, 8)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from time2spatial import transformProj2Orig

    css = [f"rgb{spec['color']}" for spec in AGENTS]

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
        cart_x, cart_y, _, _ = transformProj2Orig(
            sim[:, i, COL["s"]],
            sim[:, i, COL["n"]],
            sim[:, i, COL["alpha"]],
            sim[:, i, COL["v"]],
            TRACK_FILE,
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
        title=f"Multi-agent race — closed-loop trajectories  ({len(sim)} steps)",
        xaxis=dict(title="x [m]", scaleanchor="y"),
        yaxis=dict(title="y [m]"),
        height=600,
    )
    fig1.show()

    # ── Pairwise separation vs the ellipse limit ───────────────────────────────
    fig2 = go.Figure()
    for i in range(K):
        for j in range(i + 1, K):
            gap = np.sqrt(
                ((sim[:, i, COL["s"]] - sim[:, j, COL["s"]]) / SEP_LONG) ** 2
                + ((sim[:, i, COL["n"]] - sim[:, j, COL["n"]]) / SEP_LAT) ** 2
            )
            fig2.add_trace(
                go.Scatter(x=t_sim, y=gap, name=f"{AGENTS[i]['name']} ↔ {AGENTS[j]['name']}")
            )
    fig2.add_hline(y=1.0, line=dict(color="black", dash="dash", width=1), annotation_text="contact")
    fig2.update_layout(
        title="Separation (ellipse metric — below 1 is contact)",
        xaxis_title="t [s]",
        yaxis_title="normalised gap",
        height=400,
    )
    fig2.show()

    # ── Speed and battery ──────────────────────────────────────────────────────
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["v [m/s]", "E [J]"])
    for i, spec in enumerate(AGENTS):
        fig3.add_trace(
            go.Scatter(x=t_sim, y=sim[:, i, COL["v"]], line=dict(color=css[i]), name=spec["name"]),
            row=1,
            col=1,
        )
        fig3.add_trace(
            go.Scatter(
                x=t_sim,
                y=sim[:, i, COL["E"]],
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
    fig3.update_xaxes(title_text="t [s]", row=2, col=1)
    fig3.update_layout(title="Speed and battery state of charge", height=600)
    fig3.show()


# ── Race loop ──────────────────────────────────────────────────────────────────


def run_race(max_steps: int = MAX_STEPS) -> tuple[np.ndarray, np.ndarray, list]:
    """Race the roster to the flag; returns ``(sim, t_sim, finish_time)``.

    ``sim`` is the closed-loop log, shape ``(T, K, len(DRIVER_STATES))``;
    ``finish_time`` holds each car's flag-crossing time (``None`` if the time
    cap expired first). One ``solve_batched`` call advances the whole field by
    one node (``DT_MPC``) per iteration.
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
    finish_time: list = [None] * K
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

        nodes = {name: np.asarray(results.nodes[name]) for name in DRIVER_STATES}
        row = np.stack([nodes[name][:, 0, 0] for name in DRIVER_STATES], axis=1)  # (K, n)

        # Finish detection: interpolate the flag crossing inside the last step.
        for i in range(K):
            if finish_time[i] is None and row[i, COL["s"]] >= pathlength:
                s_prev = sim_rows[-1][i, COL["s"]] if sim_rows else grid_slot(i)[0]
                frac = (pathlength - s_prev) / max(row[i, COL["s"]] - s_prev, 1e-9)
                finish_time[i] = t_now - DT_MPC * (1.0 - frac)

        sim_rows.append(row)
        status = "  |  ".join(
            f"{AGENTS[i]['name']}: s={row[i, COL['s']]:6.2f} v={row[i, COL['v']]:.2f}"
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
        pred_s = nodes["s"][:, :, 0]
        pred_n = nodes["n"][:, :, 0]
        t_now += DT_MPC

    print(f"mean solve {np.mean(solve_ms):.0f} ms, max {np.max(solve_ms):.0f} ms")
    return np.stack(sim_rows), np.arange(len(sim_rows)) * DT_MPC, finish_time


def crossing_index(sim: np.ndarray, i: int) -> int:
    """Number of log rows up to and including car ``i``'s flag crossing."""
    return min(int(np.searchsorted(sim[:, i, COL["s"]], pathlength)) + 1, len(sim))


def print_classification(sim: np.ndarray, finish_time: list) -> list[int]:
    """Print the final order and per-car energy summary; returns the order."""
    order = sorted(range(K), key=lambda i: np.inf if finish_time[i] is None else finish_time[i])
    print("\n=== Race classification ===")
    for place, i in enumerate(order, start=1):
        spec = AGENTS[i]
        if finish_time[i] is None:
            print(f"  P{place}  {spec['name']:<20s}  DNF (time cap)")
            continue
        at_flag = sim[crossing_index(sim, i) - 1, i]
        gap = "" if place == 1 else f"  +{finish_time[i] - finish_time[order[0]]:.3f} s"
        print(f"  P{place}  {spec['name']:<20s}  {finish_time[i]:7.3f} s{gap}")
        print(
            f"       battery {sim[0, i, COL['E']]:.3f} → {at_flag[COL['E']]:.3f} J,"
            f" recovered {at_flag[COL['R']]:.3f} J (cap {R_LAP_MAX:.3f} J)"
        )
    return order


# ── Main: run the race ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    problem.initialize()
    sim, t_sim, finish_time = run_race()
    order = print_classification(sim, finish_time)

    if os.environ.get("OPENSCVX_NO_PLOT") is None:
        plot_race(sim, t_sim)

        from race_car_viser import (
            create_race_car_chase_viser_server,
            create_race_car_comparison_viser_server,
        )

        # Trim each car's log at its own flag crossing so the replay parks it
        # at the line and the finishing gaps stay visible.
        cross = [crossing_index(sim, i) for i in range(K)]
        comparison_server = create_race_car_comparison_viser_server(
            simX_list=[sim[: cross[i], i, :6] for i in range(K)],
            t_sim_list=[t_sim[: cross[i]] for i in range(K)],
            labels=[spec["name"] for spec in AGENTS],
            colors=[spec["color"] for spec in AGENTS],
            track_file=TRACK_FILE,
            lane_width=LANE_HALF_WIDTH,
            trim_warmup=False,
            distance_marker_step=None,
            title="Multi-agent race",
        )
        winner = order[0]
        chase_server = create_race_car_chase_viser_server(
            simX=sim[: cross[winner], winner, :6],
            t_sim=t_sim[: cross[winner]],
            track_file=TRACK_FILE,
            lane_width=LANE_HALF_WIDTH,
            trim_warmup=False,
            title=f"Winner — {AGENTS[winner]['name']}",
        )
        print("Race replay and winner chase camera are on separate Viser ports.")
        chase_server.sleep_forever()
