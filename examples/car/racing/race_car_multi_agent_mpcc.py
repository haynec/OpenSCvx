"""Multi-agent race where the field tracks a precomputed optimal lap.

Same field, track, and collision-avoidance scheme as
``race_car_multi_agent.py``, but the horizon objective changes from "maximise
arc-length progress" to "follow the reference lap", with the two-phase
structure of the MPCC drone-racing example
(``examples/mpc/double_integrator_drone_racing.py``): phase 1 solves the
charge-sustaining minimum-time flying lap for each car's spec (power, mass,
battery) in one batched solve, and phase 2 races the field while every car
regulates around the nominal lap, paced to its own spec.

Because the model is already in Frenet coordinates, no virtual progress state
is needed — ``s`` *is* progress, and the MPCC error decomposition collapses to
plain per-state tracking against reference profiles keyed on ``s``:

  * contour error   (n − n_ref(s))²  — distance off the racing line
  * pace error      (v − v_ref(s))²  — speed deficit/surplus at this point
  * energy error    (E − E_ref(s))²  — deviation from the lap energy plan

Each error integrates into its own state (``track_n``/``track_v``/
``track_E``) — physical quantities in m²·s, (m/s)²·s, and J²·s whose box
bounds are sized to real racing deviations — and the final values are
minimised with per-state weights, alongside a small residual progress reward
that gives cars a reason to overtake rather than follow. All cars regulate
around one *shared* reference: the nominal unity-spec lap, baked into pchip
splines over the tiled arc-length grid exactly like ``kappa`` — a smooth,
kink-free lookup the SCP linearises cleanly. Spec differences enter through
one scalar ``pace_scale`` parameter that rescales the speed profile by the
ratio of phase-1 lap times: the pace hierarchy that lets wheel-to-wheel
battles resolve instead of two cars chasing the same point indefinitely.

The payoff over the progress-max controller: the horizon no longer has to
rediscover the racing line, corner speeds, and — critically — the energy
strategy, which a 3 s horizon fundamentally cannot plan; the whole-lap-aware
deploy/harvest schedule lives in the reference and the MPC only regulates
around it while racing the field.

!!! note "Twin example"
    ``race_car_multi_agent.py`` and ``race_car_multi_agent_mpcc.py`` share
    the field, track, model, and collision scheme; they differ only in the
    horizon objective — maximise progress vs. regulate around a precomputed
    reference lap.

!!! warning "Not real time"
    This is an offline closed-loop simulation: each 0.1 s MPC step takes on
    the order of 0.1–1 s to solve on a laptop CPU — growing with the size of
    the field — and phase 1 solves the reference laps once up front (several
    minutes).

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
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from examples.car.racing._tracks.readDataFcn import getTrack

# ── Roster ─────────────────────────────────────────────────────────────────────
AGENTS = [
    dict(
        name="red",  # pole sitter, slightly down on power — must defend
        color=(220, 35, 45),
        power_scale=0.9,
        mass_scale=1.0,
        battery_scale=1.0,
    ),
    dict(
        name="blue",  # the reference car
        color=(90, 140, 235),
        power_scale=1.0,
        mass_scale=1.0,
        battery_scale=1.0,
    ),
    dict(
        name="yellow",  # carrying ballast
        color=(240, 190, 50),
        power_scale=1.0,
        mass_scale=1.15,
        battery_scale=1.0,
    ),
    dict(
        name="green",  # hot engine, starting from the back
        color=(120, 200, 120),
        power_scale=1.1,
        mass_scale=1.0,
        battery_scale=1.0,
    ),
]
K = len(AGENTS)

M_LAPS = 2

# ── Track data ─────────────────────────────────────────────────────────────────
TRACK_FILE = "LMS_Track_x4.txt"
sref_data, _, _, _, kapparef_data = getTrack(TRACK_FILE)
pathlength = float(sref_data[-1])  # ≈ 34.84 m
RACE_DISTANCE = M_LAPS * pathlength

LANE_HALF_WIDTH = 0.24

# ── Grid (F1 standing start) ───────────────────────────────────────────────────
GRID_ROW_GAP = 0.5


def grid_slot(i: int) -> tuple[float, float]:
    """(s, n) of grid slot ``i``: one row gap per car, alternating sides."""
    return -(i + 1) * GRID_ROW_GAP, 0.5 * LANE_HALF_WIDTH * (1.0 if i % 2 == 0 else -1.0)


# κ is lap-periodic: two tiled copies cover every horizon, the low pad covers
# the grid slots.
_pad_lo = (K + 1) * GRID_ROW_GAP + 1.0
_s_tiled = np.concatenate([sref_data[:-1], sref_data + pathlength])
_kappa_tiled = np.concatenate([kapparef_data[:-1], kapparef_data])
s_interp = np.concatenate([[_s_tiled[0] - _pad_lo], _s_tiled])
kappa_interp = np.concatenate([[_kappa_tiled[0]], _kappa_tiled])

# ── Vehicle parameters (Kloeser et al. 2020, Table I) ─────────────────────────
m = 0.043
C1 = 0.5
C2 = 15.5
Cm1 = 0.28
Cm2 = 0.05
Cr0 = 0.011
Cr2 = 0.006
A_MAX = 4.0

# ── Hybrid power unit (F1 2026 ratios, RC-car scale) ──────────────────────────
ICE_SHARE = 0.55
ELEC_SHARE = 0.45
ETA_BATT = 0.90

T_LAP_KART = 6.0
P_PEAK = Cm1**2 / (4.0 * Cm2)
P_ELEC_PEAK = ELEC_SHARE * P_PEAK

E_BATT_MAX = 0.13 * T_LAP_KART * P_ELEC_PEAK
R_LAP_MAX = 2.0 * E_BATT_MAX
E_CAP_TOP = E_BATT_MAX * max(spec["battery_scale"] for spec in AGENTS)

# ── Separation ellipse (track coordinates) ─────────────────────────────────────
SEP_LONG = 0.12
SEP_LAT = 0.06

# ── MPC horizon ────────────────────────────────────────────────────────────────
N_MPC = 31
HORIZON_TF = 3.0
DT_MPC = HORIZON_TF / (N_MPC - 1)
RACE_TIME_MAX = 15.0 + 20.0 * M_LAPS
MAX_STEPS = int(np.ceil(RACE_TIME_MAX / DT_MPC))

S_OVERRUN = 4.0 * HORIZON_TF

SCP_ITERS_PER_STEP = 10

# ── Trail-braking throttle guess (for the reference lap solve) ─────────────────
# Corners are the intervals where the tiled curvature exceeds KAPPA_CORNER and
# D(s) ramps linearly from full braking at entry to full throttle at exit —
# the guess shape validated on race_car_hybrid.py's open-loop lap.
KAPPA_CORNER = 0.5

_corner_mask = np.abs(kappa_interp) > KAPPA_CORNER
_corner_edges = np.diff(_corner_mask.astype(int))
_CORNERS = list(
    zip(
        s_interp[np.flatnonzero(_corner_edges == 1) + 1],
        s_interp[np.flatnonzero(_corner_edges == -1)],
    )
)


def trail_brake_throttle(s_query: np.ndarray) -> np.ndarray:
    """Trail-braking throttle profile D(s) ∈ [-1, 1]: -1 at corner entry, +1 at exit."""
    s_query = np.asarray(s_query, dtype=float)
    D = np.ones_like(s_query)
    for s0, s1 in _CORNERS:
        in_corner = (s_query >= s0) & (s_query <= s1)
        D = np.where(in_corner, -1.0 + 2.0 * (s_query - s0) / (s1 - s0), D)
    return D


# ── Phase 1: reference laps ────────────────────────────────────────────────────
# Each car's charge-sustaining minimum-time flying lap for its own spec, under
# the race bounds: periodic driving states AND periodic battery — the lap's
# deploy spend equals its capped recovery, the right plan for the middle of a
# race, unlike the drain-to-zero qualifying lap of race_car_hybrid.py. The
# result is sampled on a uniform s grid (coarse is fine — the profiles are
# smooth, and every extra knot costs Jacobian work in the tracking dynamics).
M_REF_LAP = 64


def _solve_reference_laps() -> dict:
    """Solve the batched per-spec reference laps and return the lap tables."""
    N = 80
    T_guess = 20.0

    rs = ox.State("s", shape=(1,))
    rs.min, rs.max = [-0.1], [pathlength + 0.1]
    rs.initial, rs.final = [0.0], [pathlength]
    rs.guess = np.linspace(0.0, pathlength, N).reshape(-1, 1)

    rn = ox.State("n", shape=(1,))
    rn.min, rn.max = [-LANE_HALF_WIDTH], [LANE_HALF_WIDTH]
    rn.initial, rn.final = [ox.Free(0.0)], [ox.Free(0.0)]
    rn.guess = np.zeros((N, 1))

    ralpha = ox.State("alpha", shape=(1,))
    ralpha.min, ralpha.max = [-np.pi / 2], [np.pi / 2]
    ralpha.initial, ralpha.final = [ox.Free(0.0)], [ox.Free(0.0)]
    ralpha.guess = np.zeros((N, 1))

    rv = ox.State("v", shape=(1,))
    rv.min, rv.max = [0.0], [6.0]
    rv.initial, rv.final = [ox.Free(2.0)], [ox.Free(2.0)]
    rv.guess = 2.0 * np.ones((N, 1))

    rD = ox.State("D", shape=(1,))
    rD.min, rD.max = [-1.0], [1.0]
    rD.initial, rD.final = [ox.Free(1.0)], [ox.Free(1.0)]
    D_trail = trail_brake_throttle(rs.guess[:, 0])
    rD.guess = D_trail.reshape(-1, 1)

    rdelta = ox.State("delta", shape=(1,))
    rdelta.min, rdelta.max = [-0.40], [0.40]
    rdelta.initial, rdelta.final = [ox.Free(0.0)], [ox.Free(0.0)]
    rdelta.guess = np.zeros((N, 1))

    rE = ox.State("E", shape=(1,))
    rE.min, rE.max = [0.0], [E_CAP_TOP]
    rE.initial, rE.final = [ox.Free(0.5 * E_BATT_MAX)], [ox.Free(0.5 * E_BATT_MAX)]
    rE.guess = 0.5 * E_BATT_MAX * np.ones((N, 1))

    rR = ox.State("R", shape=(1,))
    rR.min, rR.max = [0.0], [R_LAP_MAX]
    rR.initial, rR.final = [0.0], [ox.Free(0.0)]
    rR.guess = np.linspace(0.0, 0.8 * R_LAP_MAX, N).reshape(-1, 1)

    rderD = ox.Control("derD", shape=(1,), parameterization="ZOH")
    rderD.min, rderD.max = [-10.0], [10.0]
    rderD.guess = np.clip(
        np.diff(D_trail, append=D_trail[-1]) * (N - 1) / T_guess, rderD.min, rderD.max
    ).reshape(-1, 1)

    rderDelta = ox.Control("derDelta", shape=(1,), parameterization="ZOH")
    rderDelta.min, rderDelta.max = [-2.0], [2.0]
    rderDelta.guess = np.zeros((N, 1))

    rdeploy = ox.Control("deploy", shape=(1,), parameterization="FOH")
    rdeploy.min, rdeploy.max = [0.0], [1.0]
    rdeploy.guess = np.clip(D_trail, 0.0, 1.0).reshape(-1, 1)

    rregen = ox.Control("regen", shape=(1,), parameterization="FOH")
    rregen.min, rregen.max = [0.0], [1.0]
    rregen.guess = np.clip(-D_trail, 0.0, 1.0).reshape(-1, 1)

    rtime = ox.Time(
        initial=0.0,
        final=ox.Minimize(T_guess),
        min=0.0,
        max=60.0,
        guess=np.linspace(0.0, T_guess, N).reshape(-1, 1),
    )

    rpower = ox.Parameter("power_scale", shape=(), value=1.0)
    rmass = ox.Parameter("mass_scale", shape=(), value=1.0)
    rbattery = ox.Parameter("battery_scale", shape=(), value=1.0)

    rkappa = ox.Cinterp(rs[0], s_interp, kappa_interp, method="pchip")
    rm_car = ox.Constant(m) * rmass
    rF_env = rpower * (ox.Constant(Cm1) - ox.Constant(Cm2) * rv[0])
    rFxd = (
        ox.Constant(ICE_SHARE) * rF_env * rD[0]
        + ox.Constant(ELEC_SHARE) * rF_env * (rdeploy[0] - rregen[0])
        - ox.Constant(Cr2) * rv[0] ** 2
        - ox.Constant(Cr0) * ox.Tanh(ox.Constant(5.0) * rv[0])
    )
    rP_deploy = ox.Constant(ELEC_SHARE) * rF_env * rdeploy[0] * rv[0]
    rP_harvest = ox.Constant(ETA_BATT) * ox.Constant(ELEC_SHARE) * rF_env * rregen[0] * rv[0]
    rslip = ralpha[0] + ox.Constant(C1) * rdelta[0]
    rsdot = (rv[0] * ox.Cos(rslip)) / (ox.Constant(1.0) - rkappa * rn[0])

    rdynamics = {
        "s": rsdot,
        "n": rv[0] * ox.Sin(rslip),
        "alpha": rv[0] * ox.Constant(C2) * rdelta[0] - rkappa * rsdot,
        "v": (rFxd / rm_car) * ox.Cos(ox.Constant(C1) * rdelta[0]),
        "D": rderD[0],
        "delta": rderDelta[0],
        "E": rP_harvest - rP_deploy,
        "R": rP_harvest,
    }

    rstates = [rs, rn, ralpha, rv, rD, rdelta, rE, rR]
    rcontrols = [rderD, rderDelta, rdeploy, rregen]

    rconstraints: list = []
    for state in [rs, rn, ralpha, rv, rD, rdelta, rR]:
        rconstraints.extend(
            [
                ox.ctcs(state <= state.max, penalty="huber"),
                ox.ctcs(state.min <= state, penalty="huber"),
            ]
        )
    rconstraints.extend(
        [
            ox.ctcs(rE[0] <= rbattery * ox.Constant(E_BATT_MAX), penalty="huber"),
            ox.ctcs(0.0 <= rE[0], penalty="huber"),
        ]
    )
    # Flying-lap periodicity, including the battery (charge sustain).
    rconstraints.extend((x.at(0) == x.at(N - 1)).convex() for x in [rn, ralpha, rv, rD, rdelta, rE])
    ra_lat = (
        ox.Constant(C2) * rv[0] ** 2 * rdelta[0]
        + rFxd * ox.Sin(ox.Constant(C1) * rdelta[0]) / rm_car
    )
    rconstraints.append(ox.ctcs(ra_lat**2 + (rFxd / rm_car) ** 2 <= A_MAX**2, penalty="huber"))

    ref_problem = ox.Problem(
        dynamics=rdynamics,
        states=rstates,
        controls=rcontrols,
        time=rtime,
        constraints=rconstraints,
        N=N,
        float_dtype="float64",
        licq_max=1e-12,
        algorithm={
            # Anchor loose enough to escape the initial guess but stiff enough
            # that the warm-start continuation rounds below stay finite.
            "lam_prox": 1e-1,
            "lam_cost": 3e0,
            "lam_vc": 1e2,
            "autotuner": ox.AugmentedLagrangian(eta_lambda=1e0),
        },
        discretizer={"diffrax_kwargs": {"atol": 1e-8, "rtol": 1e-8}},
    )
    ref_problem.settings.dev.printing = False
    ref_problem.settings.prp.atol = 1e-10
    ref_problem.settings.prp.rtol = 1e-10
    ref_problem.initialize()

    specs = {
        key: np.array([spec[key] for spec in AGENTS])
        for key in ("power_scale", "mass_scale", "battery_scale")
    }
    results = ref_problem.solve_batched(parameters=specs)
    # Warm-start continuation escapes the guess anchoring; a diverged round
    # keeps the previous (finite) solution.
    for _ in range(3):
        try:
            nxt = ref_problem.solve_batched(
                parameters=specs,
                x_guess=np.asarray(results.x),
                u_guess=np.asarray(results.u),
            )
        except Exception:
            break
        if np.isnan(np.asarray(nxt.x)).any():
            break
        results = nxt
    results = ref_problem.post_process_batched(results)

    # Sample every signal onto the uniform lap grid, clipped to the feasible
    # box: the huber CTCS penalties let the solve leak slightly past a bound,
    # and the race MPC must never be asked to track an infeasible point.
    s_grid = np.linspace(0.0, pathlength, M_REF_LAP, endpoint=False)
    x_full = np.asarray(results.x_full)
    u_full = np.asarray(results.u_full)
    cols = {name: i for i, name in enumerate(DRIVER_STATES)}
    n_margin = LANE_HALF_WIDTH - 0.005
    signals = {
        "ref_n": (x_full, cols["n"], (-n_margin, n_margin)),
        "ref_v": (x_full, cols["v"], (0.0, 6.0)),
        "ref_E": (x_full, cols["E"], (0.0, E_BATT_MAX)),
        "ref_D": (x_full, cols["D"], (-1.0, 1.0)),
        "ref_deploy": (u_full, 2, (0.0, 1.0)),
        "ref_regen": (u_full, 3, (0.0, 1.0)),
    }
    tables = {sig: np.zeros((K, M_REF_LAP)) for sig in signals}
    for i in range(K):
        order = np.argsort(x_full[i, :, cols["s"]])
        s_o = x_full[i, order, cols["s"]]
        for sig, (src, col, (lo, hi)) in signals.items():
            tables[sig][i] = np.clip(np.interp(s_grid, s_o, src[i, order, col]), lo, hi)

    return dict(
        s_grid=s_grid,
        lap_times=np.asarray(results.t_final).reshape(-1),
        **tables,
    )


DRIVER_STATES = ("s", "n", "alpha", "v", "D", "delta", "E", "R")

print("Phase 1: solving each car's reference lap (a few minutes)...")
_ref = _solve_reference_laps()
REF_S_GRID = _ref["s_grid"]  # (M,) uniform on [0, pathlength)
REF_LAP_TIMES = _ref["lap_times"]  # (K,)
_M_LAP = len(REF_S_GRID)
_DS_REF = pathlength / _M_LAP

# Tile the lap-periodic tables onto a uniform grid spanning every s a horizon
# can visit: grid slots (negative) through the post-flag overrun.
S_REF_LO = -np.ceil((_pad_lo + 1.0) / _DS_REF) * _DS_REF
_k_lo = int(round(S_REF_LO / _DS_REF))
_k_hi = int(np.ceil((pathlength + S_OVERRUN + 1.0) / _DS_REF))
_k_ext = np.arange(_k_lo, _k_hi + 1)
M_REF = len(_k_ext)


def _tile(table: np.ndarray) -> np.ndarray:
    """(K, M) lap table -> (K, M_REF) lap-periodic extended table."""
    return table[:, _k_ext % _M_LAP]


REF_N = _tile(_ref["ref_n"])
REF_V = _tile(_ref["ref_v"])
REF_E = _tile(_ref["ref_E"])
REF_D = _tile(_ref["ref_D"])
REF_DEPLOY = _tile(_ref["ref_deploy"])
REF_REGEN = _tile(_ref["ref_regen"])

# Uniform arc-length grid of the tiled tables — the breakpoints the shared
# reference splines and the warm-start lookups both key on.
REF_S_EXT = S_REF_LO + np.arange(M_REF) * _DS_REF

# All cars regulate around the nominal unity-spec lap; if no unity-spec car is
# in the roster (e.g. a truncated field) the first car's lap stands in.
REF_IDX = next(
    (
        i
        for i, spec in enumerate(AGENTS)
        if spec["power_scale"] == spec["mass_scale"] == spec["battery_scale"] == 1.0
    ),
    0,
)

# Pace factor: the nominal speed profile rescaled by the ratio of phase-1 lap
# times. Without it two neighbouring cars chase the same point at the same
# speed and a duel never resolves; the pace hierarchy is what lets battles
# end and the field string out.
PACE = REF_LAP_TIMES[REF_IDX] / REF_LAP_TIMES

# ── States ─────────────────────────────────────────────────────────────────────
S_POLE, N_POLE = grid_slot(0)
S_MIN = grid_slot(K - 1)[0] - 0.1

s = ox.State("s", shape=(1,))
s.min = [S_MIN]
s.max = [pathlength + S_OVERRUN]
s.initial = [S_POLE]
s.final = [ox.Maximize(0.0)]  # residual progress reward: the incentive to overtake
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
v.initial = [0.0]
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
E_batt.max = [E_CAP_TOP]
E_batt.initial = [E_BATT_MAX]
E_batt.final = [ox.Free(0.0)]
E_batt.guess = np.full((N_MPC, 1), E_BATT_MAX)

E_rec = ox.State("R", shape=(1,))
E_rec.min = [0.0]
E_rec.max = [R_LAP_MAX]
E_rec.initial = [0.0]
E_rec.final = [ox.Free(0.0)]
E_rec.guess = np.zeros((N_MPC, 1))

# Tracking cost integrators (the lag_sum/contour_sum idiom of the drone
# example): each accumulates one squared tracking error along the horizon, in
# its own physical units, and is minimised at the final node with its own
# weight. The box caps are hard nodal bounds as well as solver scaling, so
# each is sized to a genuine racing deviation held for a whole horizon (a
# car-width off line, ~1 m/s off pace, a half-battery split): going
# off-reference to pass must sit comfortably inside the cap, because an
# integrator that rails its box fights the solver on every horizon it does.
track_n = ox.State("track_n", shape=(1,))  # ∫(n − n_ref)² dt  [m²·s]
track_n.min, track_n.max = [0.0], [1.0]
track_n.initial, track_n.final = [0.0], [ox.Minimize(0.0)]
track_n.guess = np.zeros((N_MPC, 1))

track_v = ox.State("track_v", shape=(1,))  # ∫(v − v_ref)² dt  [(m/s)²·s]
track_v.min, track_v.max = [0.0], [30.0]
track_v.initial, track_v.final = [0.0], [ox.Minimize(0.0)]
track_v.guess = np.zeros((N_MPC, 1))

track_E = ox.State("track_E", shape=(1,))  # ∫(E − E_ref)² dt  [J²·s]
track_E.min, track_E.max = [0.0], [0.01]
track_E.initial, track_E.final = [0.0], [ox.Minimize(0.0)]
track_E.guess = np.zeros((N_MPC, 1))

TRACK_STATES = ("track_n", "track_v", "track_E")
COL = {name: i for i, name in enumerate(DRIVER_STATES + TRACK_STATES)}
TIME_COL = len(DRIVER_STATES) + len(TRACK_STATES)

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
time = ox.Time(
    initial=0.0,
    final=HORIZON_TF,
    min=0.0,
    max=HORIZON_TF,
    uniform_time_grid=True,
)

# ── Parameters: car spec, pace factor, and opponent plans ────────────────────
power_scale = ox.Parameter("power_scale", shape=(), value=1.0)
mass_scale = ox.Parameter("mass_scale", shape=(), value=1.0)
battery_scale = ox.Parameter("battery_scale", shape=(), value=1.0)
pace_scale = ox.Parameter("pace_scale", shape=(), value=1.0)

if K > 1:
    opp_s = ox.Parameter("opp_s", shape=(N_MPC, K - 1), value=np.full((N_MPC, K - 1), S_MIN - 10.0))
    opp_n = ox.Parameter("opp_n", shape=(N_MPC, K - 1), value=np.zeros((N_MPC, K - 1)))

# ── Dynamics (hybrid spatial bicycle model, per-car spec via parameters) ───────
kappa = ox.Cinterp(s[0], s_interp, kappa_interp, method="pchip")
m_car = ox.Constant(m) * mass_scale

F_env = power_scale * (ox.Constant(Cm1) - ox.Constant(Cm2) * v[0])
F_ice = ox.Constant(ICE_SHARE) * F_env * D_throt[0]
F_elec = ox.Constant(ELEC_SHARE) * F_env * (deploy[0] - regen[0])

Fxd = (
    F_ice
    + F_elec
    - ox.Constant(Cr2) * v[0] ** 2
    - ox.Constant(Cr0) * ox.Tanh(ox.Constant(5.0) * v[0])
)

P_deploy = ox.Constant(ELEC_SHARE) * F_env * deploy[0] * v[0]
P_harvest = ox.Constant(ETA_BATT) * ox.Constant(ELEC_SHARE) * F_env * regen[0] * v[0]

slip_angle = alpha[0] + ox.Constant(C1) * delta[0]
sdot = (v[0] * ox.Cos(slip_angle)) / (ox.Constant(1.0) - kappa * n[0])

# Reference lookup: the shared nominal lap as pchip splines of arc length, the
# same construction ``kappa`` uses — smooth and kink-free, so the tracking
# error linearises cleanly. Only the speed profile is per-car, through the
# scalar pace factor.
n_ref_s = ox.Cinterp(s[0], REF_S_EXT, REF_N[REF_IDX], method="pchip")
v_ref_s = pace_scale * ox.Cinterp(s[0], REF_S_EXT, REF_V[REF_IDX], method="pchip")
E_ref_s = ox.Cinterp(s[0], REF_S_EXT, REF_E[REF_IDX], method="pchip")

dynamics = {
    "s": sdot,
    "n": v[0] * ox.Sin(slip_angle),
    "alpha": v[0] * ox.Constant(C2) * delta[0] - kappa * sdot,
    "v": (Fxd / m_car) * ox.Cos(ox.Constant(C1) * delta[0]),
    "D": derD[0],
    "delta": derDelta[0],
    "E": P_harvest - P_deploy,
    "R": P_harvest,
    "track_n": (n[0] - n_ref_s) ** 2,
    "track_v": (v[0] - v_ref_s) ** 2,
    "track_E": (E_batt[0] - E_ref_s) ** 2,
}

# ── Constraints ────────────────────────────────────────────────────────────────
states = [s, n, alpha, v, D_throt, delta, E_batt, E_rec, track_n, track_v, track_E]
controls = [derD, derDelta, deploy, regen]

constraints: list = []

for state in [s, n, alpha, v, D_throt, delta, E_rec]:
    constraints.extend(
        [
            ox.ctcs(state <= state.max, penalty="huber"),
            ox.ctcs(state.min <= state, penalty="huber"),
        ]
    )

constraints.extend(
    [
        ox.ctcs(E_batt[0] <= battery_scale * ox.Constant(E_BATT_MAX), penalty="huber"),
        ox.ctcs(0.0 <= E_batt[0], penalty="huber"),
    ]
)

a_lat = ox.Constant(C2) * v[0] ** 2 * delta[0] + Fxd * ox.Sin(ox.Constant(C1) * delta[0]) / m_car
a_long = Fxd / m_car

constraints.append(ox.ctcs(a_lat**2 + a_long**2 <= A_MAX**2, penalty="huber"))

W_SEP = 4e3

if K > 1:
    hat = ox.Max(1.0 - ox.Abs(time[0] / DT_MPC - np.arange(N_MPC, dtype=float)), 0.0)
    for j in range(K - 1):
        opp_s_t = ox.Sum(hat * opp_s[:, j])
        opp_n_t = ox.Sum(hat * opp_n[:, j])
        ds = s[0] - opp_s_t
        ds_wrap = ox.Constant(pathlength / np.pi) * ox.Sin(ox.Constant(np.pi / pathlength) * ds)
        gap = (ds_wrap / SEP_LONG) ** 2 + ((n[0] - opp_n_t) / SEP_LAT) ** 2
        constraints.append(ox.ctcs(W_SEP * (1.0 - gap) <= 0.0, penalty="huber"))

# ── Problem ────────────────────────────────────────────────────────────────────
# The reference carries the racing line, pace, and energy plan, so the cost is
# regulation around it plus a residual progress reward that gives cars a
# reason to overtake rather than follow. ``lam_cost`` weighs the *scaled*
# integrators — state scaling divides by max(1, half-range of the box), so a
# weight's physical strength is its lam over that factor, and the narrow
# track_n / track_E boxes clamp to 1 (their lams ARE their strengths).
# Contour is priced cheapest: lateral room is the overtaking degree of
# freedom, and racing thrashes (convergence and solve time both suffer) when
# the line or pace is too dear to leave — but on a shared line it cannot be
# too cheap either, or side-by-side duels stop resolving. Energy is priced so
# the charge plan actually binds: any weaker and the progress reward runs the
# field net-negative every lap until the battery pins at empty and the
# track_E integrator rails its box all through the following laps.
W_PROGRESS = 3e1
W_TRACK_N = 5e0
W_TRACK_V = 9e1
W_TRACK_E = 1e2

problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=N_MPC,
    float_dtype="float64",
    algorithm={
        "lam_vc": 3e3,
        "lam_prox": 2e0,
        "lam_cost": {
            "s": W_PROGRESS,
            "track_n": W_TRACK_N,
            "track_v": W_TRACK_V,
            "track_E": W_TRACK_E,
        },
        "autotuner": ox.ConstantProximalWeight(),
        "ep_tr": 3e-2,
    },
    discretizer={
        "diffrax_kwargs": {"atol": 1e-6, "rtol": 1e-6},
    },
    solver=ox.MoreauPTRSolver(),
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


def shift_forecast(plan: np.ndarray) -> np.ndarray:
    """Advance published plans one node so they align with the next horizon."""
    tail = 2.0 * plan[:, -1:] - plan[:, -2:-1]
    return np.concatenate([plan[:, 1:], tail], axis=1)


_X_LO = np.array([st.min[0] for st in states])
_X_HI = np.array([st.max[0] for st in states])


def _ref_lookup(table: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    """Per-car linear interpolation of a (K, M_REF) table at s_query (K,)."""
    pos = np.clip((s_query - S_REF_LO) / _DS_REF, 0.0, M_REF - 1.001)
    k0 = pos.astype(int)
    w = pos - k0
    rows = np.arange(len(s_query))
    return (1.0 - w) * table[rows, k0] + w * table[rows, k0 + 1]


def shifted_guesses(results) -> tuple[np.ndarray, np.ndarray]:
    """Warm starts for the next step: previous plans shifted one node.

    The freed tip node is seeded from the car's own phase-1 lap at its
    predicted arc length — feasible for the car's spec, so unlike a raw
    heuristic profile this costs no virtual control, even though the horizon
    tracks the shared nominal lap — and the tracking integrators restart with
    their accumulated offset removed. Other controls hold their last value;
    the horizon clock and CTCS integrators restart from zero.
    """
    x = np.asarray(results.x)
    u = np.asarray(results.u)
    n_d = len(DRIVER_STATES)
    tail = x[:, -1:].copy()
    tail[:, :, :n_d] = np.clip(2.0 * x[:, -1:, :n_d] - x[:, -2:-1, :n_d], _X_LO[:n_d], _X_HI[:n_d])
    s_tip = tail[:, 0, COL["s"]]
    for name, table in [("n", REF_N), ("v", REF_V), ("D", REF_D), ("E", REF_E)]:
        tail[:, 0, COL[name]] = _ref_lookup(table, s_tip)
    x = np.concatenate([x[:, 1:], tail], axis=1)
    u = np.concatenate([u[:, 1:], u[:, -1:]], axis=1)
    u[:, -1, 2] = _ref_lookup(REF_DEPLOY, s_tip)
    u[:, -1, 3] = _ref_lookup(REF_REGEN, s_tip)
    # Tracking integrators: drop the first node's accumulation so each horizon
    # restarts at zero, as its pinned initial condition requires.
    for name in TRACK_STATES:
        col = x[:, :, COL[name]]
        x[:, :, COL[name]] = np.maximum(col - col[:, :1], 0.0)
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
    """Track projection and speed/battery striplines.

    As in ``race_car_multi_agent.plot_race``, everything is drawn from the
    dense propagated log against race distance in laps. The one addition is
    the dotted shared reference on each stripline: the nominal lap every car
    tracks, tiled over the race, so the tracking quality — and any deviation
    bought to race the field — is visible directly.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from examples.car.racing._plotting import stack_height, track_figure
    from examples.car.racing._time2spatial import transformProj2Orig

    css = [f"rgb{spec['color']}" for spec in AGENTS]
    laps_x = log.dense_x[:, :, COL["s"]] / pathlength  # (K, Td) race distance in laps

    # ── Track projection ───────────────────────────────────────────────────────
    fig1 = track_figure(
        TRACK_FILE,
        LANE_HALF_WIDTH,
        title=f"Tracking race — propagated closed-loop trajectories  ({len(log.sim)} steps)",
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
    fig1.show()

    # ── Speed and battery striplines against race distance ────────────────────
    ref_lap_x = np.concatenate([lap + REF_S_GRID / pathlength for lap in range(M_LAPS)])
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["v [m/s]", "E [J]"])
    for row, ref in [(1, _ref["ref_v"][REF_IDX]), (2, _ref["ref_E"][REF_IDX])]:
        fig2.add_trace(
            go.Scatter(
                x=ref_lap_x,
                y=np.tile(ref, M_LAPS),
                line=dict(color="black", dash="dot", width=1),
                name="shared reference",
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
    for i, spec in enumerate(AGENTS):
        for row, signal in [(1, log.dense_x[i, :, COL["v"]]), (2, log.dense_x[i, :, COL["E"]])]:
            fig2.add_trace(
                go.Scatter(
                    x=laps_x[i],
                    y=signal,
                    line=dict(color=css[i]),
                    name=spec["name"],
                    showlegend=row == 1,
                ),
                row=row,
                col=1,
            )
        fig2.add_hline(
            y=spec["battery_scale"] * E_BATT_MAX,
            line=dict(color=css[i], dash="dash", width=1),
            row=2,
            col=1,
        )
    for lap in range(1, M_LAPS):
        fig2.add_vline(x=float(lap), line=dict(color="black", dash="dot", width=1))
    fig2.update_xaxes(title_text="race distance [laps]", row=2, col=1)
    fig2.update_layout(
        title="Speed and battery state of charge vs the shared reference lap",
        height=stack_height(2),
    )
    fig2.show()


def build_viser_panels(log: RaceLog) -> list[dict]:
    """Live plot panels for the race replay: speed, battery, and g-g.

    Every car's series is trimmed at its own flag crossing, so its live marker
    parks with the car once it has finished. The panels themselves are built by
    ``_plotting``; what this example contributes is *which* telemetry the
    sidebar shows.
    """
    from examples.car.racing._plotting import gg_panel, stripline_panel

    css = [f"rgb{spec['color']}" for spec in AGENTS]
    end = [crossing_index(log.dense_x[i, :, COL["s"]]) for i in range(K)]
    t_car = [log.dense_t[: end[i]] for i in range(K)]
    lap_car = [log.dense_x[i, : end[i], COL["s"]] / pathlength for i in range(K)]

    v_car = [log.dense_x[i, : end[i], COL["v"]] for i in range(K)]
    e_car = [log.dense_x[i, : end[i], COL["E"]] for i in range(K)]
    gg_car = [
        accelerations(log.dense_x[i, : end[i]], log.dense_u[i, : end[i]], spec)
        for i, spec in enumerate(AGENTS)
    ]

    return [
        stripline_panel(lap_car, v_car, t_car, css, title="speed", yaxis="v [m/s]"),
        stripline_panel(lap_car, e_car, t_car, css, title="battery", yaxis="E [J]"),
        gg_panel(gg_car, t_car, css, a_max=A_MAX),
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

    sim: np.ndarray
    t_sim: np.ndarray
    finish_time: list
    dense_t: np.ndarray
    dense_x: np.ndarray
    dense_u: np.ndarray
    converged: np.ndarray


def run_race(max_steps: int = MAX_STEPS) -> RaceLog:
    """Race the roster to the flag and return the :class:`RaceLog`.

    One ``solve_batched`` call plans the whole field; the plant then advances
    one node (``DT_MPC``) along the *propagated* executed interval, so a defect
    in an unconverged plan stays a plan instead of becoming real motion. That
    same executed interval is logged densely for playback and plots.
    """
    x0 = initial_pins()
    x_guess, u_guess = cold_start_guesses()
    pred_s = x_guess[:, :, COL["s"]].copy()
    pred_n = x_guess[:, :, COL["n"]].copy()

    fixed_params = {
        key: np.array([spec[key] for spec in AGENTS])
        for key in ("power_scale", "mass_scale", "battery_scale")
    }
    fixed_params["pace_scale"] = PACE

    sim_rows: list[np.ndarray] = []
    dense_t: list[np.ndarray] = []
    dense_x: list[np.ndarray] = []
    dense_u: list[np.ndarray] = []
    finish_time: list = [None] * K
    laps = np.zeros(K)
    t_now = 0.0
    solve_ms: list[float] = []
    conv_flags: list[np.ndarray] = []

    for step in range(max_steps):
        params = dict(fixed_params)
        if K > 1:
            params["opp_s"] = opponent_view(shift_forecast(pred_s))
            params["opp_n"] = opponent_view(shift_forecast(pred_n))

        tic = _time.perf_counter()
        results = problem.solve_batched(
            x_initial=jnp.asarray(x0),
            parameters=params,
            x_guess=jnp.asarray(x_guess),
            u_guess=jnp.asarray(u_guess),
            max_iters=SCP_ITERS_PER_STEP,
        )
        solve_ms.append((_time.perf_counter() - tic) * 1e3)
        conv_flags.append(np.asarray(results.converged).reshape(-1))

        nodes = {name: np.asarray(results.nodes[name]) for name in DRIVER_STATES}
        row = np.stack([nodes[name][:, 0, 0] for name in DRIVER_STATES], axis=1)
        row[:, COL["s"]] += laps * pathlength

        # Propagate every step's executed interval through the nonlinear
        # dynamics: the segment [0, DT_MPC) is both the dense log and, at its
        # endpoint, the honest plant update below.
        post = problem.post_process_batched(results)
        t_prop = np.asarray(post.t_full)  # (K, n_times)
        x_prop = np.asarray(post.x_full)  # (K, n_times, n_prop_states)
        keep = t_prop[0] < DT_MPC - 1e-9
        seg_x = x_prop[:, keep, : len(DRIVER_STATES)].copy()
        seg_x[:, :, COL["s"]] += laps[:, None] * pathlength
        dense_t.append(t_now + t_prop[0][keep])
        dense_x.append(seg_x)
        dense_u.append(np.asarray(post.u_full)[:, keep, :4])

        for i in range(K):
            if finish_time[i] is None and row[i, COL["s"]] >= RACE_DISTANCE:
                s_prev = sim_rows[-1][i, COL["s"]] if sim_rows else grid_slot(i)[0]
                frac = (RACE_DISTANCE - s_prev) / max(row[i, COL["s"]] - s_prev, 1e-9)
                finish_time[i] = t_now - DT_MPC * (1.0 - frac)

        sim_rows.append(row)
        status = "  |  ".join(
            f"{AGENTS[i]['name']}: s={row[i, COL['s']]:6.2f} v={row[i, COL['v']]:.2f}"
            f" E={row[i, COL['E']]:.3f}"
            for i in range(K)
        )
        n_conv = int(conv_flags[-1].sum())
        conv_note = "" if n_conv == K else f", conv {n_conv}/{K}"
        print(f"step {step:4d}  t={t_now:6.2f} s  {status}  ({solve_ms[-1]:.0f} ms{conv_note})")

        if all(t is not None for t in finish_time):
            break

        # Advance the plant to the executed segment's endpoint at t = DT_MPC
        # rather than trusting solved node 1: node 1 is the convex plan's value,
        # so on an unconverged step its first interval carries a virtual-control
        # defect. Propagating the nonlinear dynamics keeps that defect a plan,
        # not car motion — the field drives the segment it was handed and
        # recovers on the next horizon. The propagation grid is a linspace over
        # the whole horizon, so interpolate each driver state at t = DT_MPC.
        # Clip to the state box: propagation is free to drift past a soft bound
        # (e.g. draining the battery a hair below empty, or running wide of the
        # lane on an unconverged plan), but an x_initial pinned outside the box
        # cannot satisfy the subproblem's node bounds and poisons that car's
        # solve, so the plant enforces the box at the handoff.
        for name in DRIVER_STATES:
            col = COL[name]
            at_dt = [np.interp(DT_MPC, t_prop[i], x_prop[i, :, col]) for i in range(K)]
            x0[:, col] = np.clip(at_dt, _X_LO[col], _X_HI[col])
        x_guess, u_guess = shifted_guesses(results)
        pred_s = nodes["s"][:, :, 0].copy()
        pred_n = nodes["n"][:, :, 0]

        for i in range(K):
            if x0[i, COL["s"]] >= pathlength:
                laps[i] += 1
                x0[i, COL["s"]] -= pathlength
                x_guess[i, :, COL["s"]] -= pathlength
                pred_s[i] -= pathlength
                x_guess[i, :, COL["R"]] = np.maximum(x_guess[i, :, COL["R"]] - x0[i, COL["R"]], 0.0)
                x0[i, COL["R"]] = 0.0

        t_now += DT_MPC

    conv_rate = float(np.mean(np.concatenate(conv_flags)))
    print(
        f"mean solve {np.mean(solve_ms):.0f} ms, max {np.max(solve_ms):.0f} ms, "
        f"converged within {SCP_ITERS_PER_STEP} iters: {100 * conv_rate:.0f}% of car-steps"
    )
    return RaceLog(
        sim=np.stack(sim_rows),
        t_sim=np.arange(len(sim_rows)) * DT_MPC,
        finish_time=finish_time,
        dense_t=np.concatenate(dense_t),
        dense_x=np.concatenate(dense_x, axis=1),
        dense_u=np.concatenate(dense_u, axis=1),
        converged=np.stack(conv_flags),
    )


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

        from examples.car.racing._viser import create_race_car_comparison_viser_server

        # Trim each car's dense log at its own flag crossing so the replay
        # parks it at the line and the finishing gaps stay visible.
        cross = [crossing_index(log.dense_x[i, :, COL["s"]]) for i in range(K)]
        server = create_race_car_comparison_viser_server(
            simX_list=[log.dense_x[i, : cross[i], :6] for i in range(K)],
            t_sim_list=[log.dense_t[: cross[i]] for i in range(K)],
            labels=[spec["name"] for spec in AGENTS],
            colors=[spec["color"] for spec in AGENTS],
            track_file=TRACK_FILE,
            lane_width=LANE_HALF_WIDTH,
            trim_warmup=False,
            distance_marker_step=None,
            title="Tracking race",
            plot_panels=build_viser_panels(log),
        )
        server.sleep_forever()
