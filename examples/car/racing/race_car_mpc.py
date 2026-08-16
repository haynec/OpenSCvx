"""Race car receding-horizon MPC (OpenSCvx formulation).

Mirrors the closed-loop simulation from the acados ``race_cars/main.py`` demo, which
uses acados SQP in a receding-horizon loop.

The spatial bicycle model is the same as ``race_car_openscvx.py`` (Kloeser et
al., IFAC 2020).  Instead of solving one global minimum-time OCP, we roll a
short prediction horizon forward one node at a time:

  • Horizon: ``Tf = 1.0 s``,  ``N = 50`` nodes  (matches acados defaults)
  • Objective: **maximise arc-length progress** ``s`` at end of horizon +
    small running penalties on lane deviation ``n²`` and heading error ``α²``
    (analogous to the acados least-squares cost with the s-tracking term
    dominating all others)
  • Closed-loop: node 1 of each solution is applied; the horizon shifts
    forward and warm-starts from the shifted previous solution
  • Simulation terminates when ``s`` passes the finish line (``s = pathlength``)

State vector  x = [s, n, α, v, D, δ, stage_cost]
  s           arc-length progress [m]
  n           lateral offset from centreline [m]
  α           heading error w.r.t. track tangent [rad]
  v           longitudinal speed [m/s]
  D           normalised throttle [-1, 1]
  δ           steering angle [rad]
  stage_cost  integrated running cost (minimised at horizon end)

Control vector  u = [Ḋ, δ̇]
"""

from __future__ import annotations

import os
import sys
import time as _time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from examples.car.racing._tracks.readDataFcn import getTrack

# ── Track data ─────────────────────────────────────────────────────────────────
sref_data, _, _, _, kapparef_data = getTrack("LMS_Track.txt")
pathlength = float(sref_data[-1])  # ≈ 8.71 m

# Curvature spline: extend well beyond [−2, pathlength+3] to cover the full
# MPC horizon at any point on the track (including the warm-up region).
_pad_lo, _pad_hi = 4.0, 4.0
s_interp = np.concatenate([[sref_data[0] - _pad_lo], sref_data, [pathlength + _pad_hi]])
kappa_interp = np.concatenate([[kapparef_data[0]], kapparef_data, [kapparef_data[-1]]])

# ── Vehicle parameters (Kloeser et al. 2020, Table I) ─────────────────────────
m = 0.043
C1 = 0.5
C2 = 15.5
Cm1 = 0.28
Cm2 = 0.05
Cr0 = 0.011
Cr2 = 0.006

# ── MPC horizon parameters (matching acados main.py defaults) ──────────────────
N_MPC = 20  # horizon nodes
HORIZON_TF = 1.0  # [s] prediction horizon length
T_SIM = 10.0  # [s] total closed-loop simulation time
SREF_N = 3.0  # [m] reference progress ahead of current s (acados sref_N)

dt_mpc = HORIZON_TF / (N_MPC - 1)  # time between consecutive MPC nodes [s]
Nsim_max = int(T_SIM * N_MPC / HORIZON_TF)  # maximum simulation steps

# Acados matching initial condition
S_INIT = -2.0

# ── Cost weights (analogous to acados Q/R diagonal) ───────────────────────────
# acados Q_s = 2*1e-1 (s tracking), all others ≈ 0; we translate to:
#   • Maximize s at terminal node (lam_cost["s"])
#   • Minimise integrated stage cost (lam_cost["stage_cost"])
Q_N = 1e-1  # lane-deviation  n²
Q_ALPHA = 1e-2  # heading error   α²
Q_DERD = 1e-3  # throttle-rate   Ḋ²
Q_DERD2 = 5e-3  # steering-rate   δ̇²

# ── States ─────────────────────────────────────────────────────────────────────
s = ox.State("s", shape=(1,))
s.min = [S_INIT - 0.1]
s.max = [pathlength + HORIZON_TF * 3.0]  # generous: horizon can look well ahead
s.initial = [S_INIT]
s.final = [ox.Maximize(0.0)]  # maximise arc-length progress each horizon
s.guess = np.linspace(S_INIT, S_INIT + 1.0, N_MPC).reshape(-1, 1)

n = ox.State("n", shape=(1,))
n.min = [-0.12]
n.max = [0.12]
n.initial = [0.0]
n.final = [ox.Free(0.0)]
n.guess = np.zeros((N_MPC, 1))

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

# Integrated running cost (restarted to 0 at each horizon)
stage_cost = ox.State("stage_cost", shape=(1,))
stage_cost.min = [0.0]
stage_cost.max = [1e4]
stage_cost.initial = [0.0]
stage_cost.final = [ox.Minimize(0.0)]
stage_cost.guess = np.zeros((N_MPC, 1))

# ── Controls ───────────────────────────────────────────────────────────────────
derD = ox.Control("derD", shape=(1,))
derD.min = [-10.0]
derD.max = [10.0]
derD.guess = np.zeros((N_MPC, 1))

derDelta = ox.Control("derDelta", shape=(1,))
derDelta.min = [-2.0]
derDelta.max = [2.0]
derDelta.guess = np.zeros((N_MPC, 1))

# ── Time: fixed horizon, uniform grid ─────────────────────────────────────────
time = ox.Time(
    initial=0.0,
    final=HORIZON_TF,
    min=0.0,
    max=HORIZON_TF,
    uniform_time_grid=True,
)

# ── Symbolic dynamics ──────────────────────────────────────────────────────────
kappa = ox.Cinterp(s[0], s_interp, kappa_interp, method="pchip")

Fxd = (
    (ox.Constant(Cm1) - ox.Constant(Cm2) * v[0]) * D_throt[0]
    - ox.Constant(Cr2) * v[0] ** 2
    - ox.Constant(Cr0) * ox.Tanh(ox.Constant(5.0) * v[0])
)

slip_angle = alpha[0] + ox.Constant(C1) * delta[0]
sdot = (v[0] * ox.Cos(slip_angle)) / (ox.Constant(1.0) - kappa * n[0])

running_cost = (
    ox.Constant(Q_N) * n[0] ** 2
    + ox.Constant(Q_ALPHA) * alpha[0] ** 2
    + ox.Constant(Q_DERD) * derD[0] ** 2
    + ox.Constant(Q_DERD2) * derDelta[0] ** 2
)

dynamics = {
    "s": sdot,
    "n": v[0] * ox.Sin(slip_angle),
    "alpha": v[0] * ox.Constant(C2) * delta[0] - kappa * sdot,
    "v": (Fxd / ox.Constant(m)) * ox.Cos(ox.Constant(C1) * delta[0]),
    "D": derD[0],
    "delta": derDelta[0],
    "stage_cost": running_cost,
}

# ── Constraints ────────────────────────────────────────────────────────────────
LANE_WIDTH = 0.12
A_MAX = 4.0  # [m/s²] lateral / longitudinal acceleration bound

states = [s, n, alpha, v, D_throt, delta, stage_cost]
controls = [derD, derDelta]

constraints: list = []

# Lane-keeping (CTCS — continuous between nodes)
constraints.extend(
    [
        ox.ctcs(n[0] <= LANE_WIDTH, penalty="huber"),
        ox.ctcs(-LANE_WIDTH <= n[0], penalty="huber"),
    ]
)

# Box constraints on remaining states
for state in [s, alpha, v, D_throt, delta, stage_cost]:
    constraints.extend(
        [
            ox.ctcs(state <= state.max, penalty="huber"),
            ox.ctcs(state.min <= state, penalty="huber"),
        ]
    )

# Nonlinear acceleration limits
a_lat = ox.Constant(C2) * v[0] ** 2 * delta[0] + Fxd * ox.Sin(
    ox.Constant(C1) * delta[0]
) / ox.Constant(m)
a_long = Fxd / ox.Constant(m)

constraints.extend(
    [
        ox.ctcs(a_lat <= A_MAX, penalty="huber"),
        ox.ctcs(-A_MAX <= a_lat, penalty="huber"),
        ox.ctcs(a_long <= A_MAX, penalty="huber"),
        ox.ctcs(-A_MAX <= a_long, penalty="huber"),
    ]
)

# ── Problem ────────────────────────────────────────────────────────────────────
problem_mpc = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=N_MPC,
    float_dtype="float64",
    algorithm={
        "lam_vc": 1e1,
        "lam_prox": 4e0,
        "autotuner": ox.ConstantProximalWeight(),
        "lam_cost": {
            "s": 4e0,  # reward progress strongly
            "stage_cost": 1e-1,  # penalise deviation + control effort
        },
    },
)
problem_mpc.settings.dev.printing = False


# ── MPC helper functions ───────────────────────────────────────────────────────


def set_initial_guess(s0: float, v0: float = 0.0) -> None:
    """Straight-line guess starting from ``s0`` at speed ``v0``."""
    s.guess = np.linspace(s0, s0 + v0 * HORIZON_TF, N_MPC).reshape(-1, 1)
    n.guess = np.zeros((N_MPC, 1))
    alpha.guess = np.zeros((N_MPC, 1))
    v.guess = np.full((N_MPC, 1), v0)
    D_throt.guess = np.zeros((N_MPC, 1))
    delta.guess = np.zeros((N_MPC, 1))
    stage_cost.guess = np.zeros((N_MPC, 1))
    derD.guess = np.zeros((N_MPC, 1))
    derDelta.guess = np.zeros((N_MPC, 1))


def update_initial_conditions(nodes: dict) -> None:
    """Advance initial conditions to node 1 of the previous solution."""
    s.initial = nodes["s"][1]
    n.initial = nodes["n"][1]
    alpha.initial = nodes["alpha"][1]
    v.initial = nodes["v"][1]
    D_throt.initial = nodes["D"][1]
    delta.initial = nodes["delta"][1]
    stage_cost.initial = np.array([0.0])  # restart integrator each horizon


def shift_guess(nodes: dict) -> None:
    """Warm-start next solve by shifting previous solution one node forward."""
    for state, key in [
        (s, "s"),
        (n, "n"),
        (alpha, "alpha"),
        (v, "v"),
        (D_throt, "D"),
        (delta, "delta"),
    ]:
        state.guess = np.vstack([nodes[key][1:], nodes[key][-1:]])

    for ctrl, key in [(derD, "derD"), (derDelta, "derDelta")]:
        ctrl.guess = np.vstack([nodes[key][1:], nodes[key][-1:]])

    stage_cost.guess = np.zeros((N_MPC, 1))


# ── Plotly visualisation ───────────────────────────────────────────────────────


def plot_mpc_results(
    simX: np.ndarray,
    simU: np.ndarray,
    t_vec: np.ndarray,
    horizon_snapshots: list[np.ndarray],
) -> None:
    """Three Plotly figures:
    1. Track projection of the closed-loop trajectory (coloured by speed)
       with all MPC horizon roll-outs in the background.
    2. State trajectories vs time.
    3. Lateral & longitudinal acceleration vs bounds.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from examples.car.racing._plotting import acceleration_figure, stack_height, track_figure
    from examples.car.racing._time2spatial import transformProj2Orig

    # Convert closed-loop trajectory to Cartesian
    cart_x, cart_y, _, _ = transformProj2Orig(
        simX[:, 0], simX[:, 1], simX[:, 2], simX[:, 3], "LMS_Track.txt"
    )

    # ── Figure 1: track projection ────────────────────────────────────────────
    fig1 = track_figure(
        "LMS_Track.txt",
        LANE_WIDTH,
        title=f"OpenSCvx MPC — track projection  ({len(simX)} steps, T={t_vec[-1]:.2f} s)",
        distance_marker_step=1.0,
    )

    # MPC horizon roll-outs (faint background)
    first = True
    for snap in horizon_snapshots:
        hx, hy, _, _ = transformProj2Orig(
            snap[:, 0], snap[:, 1], snap[:, 2], snap[:, 3], "LMS_Track.txt"
        )
        fig1.add_trace(
            go.Scatter(
                x=hx,
                y=hy,
                mode="lines",
                line=dict(color="rgba(180,180,220,0.25)", width=1),
                name="MPC horizons" if first else None,
                showlegend=first,
                legendgroup="horizon",
            )
        )
        first = False

    # Closed-loop trajectory coloured by speed
    fig1.add_trace(
        go.Scatter(
            x=cart_x,
            y=cart_y,
            mode="markers",
            marker=dict(
                color=simX[:, 3],
                colorscale="Rainbow",
                size=4,
                colorbar=dict(title="v [m/s]"),
                showscale=True,
            ),
            name="closed-loop",
        )
    )
    fig1.show()

    # ── Figure 2: states & controls vs time ───────────────────────────────────
    # ``plot_states``/``plot_controls`` cannot draw this one: they take an
    # ``OptimizationResults``, and a closed-loop run has raw (N, 6) arrays
    # instead. The layout mirrors theirs so the two read the same.
    labels_x = ["s [m]", "n [m]", "α [rad]", "v [m/s]", "D [-]", "δ [rad]"]
    labels_u = ["Ḋ [1/s]", "δ̇ [rad/s]"]
    rows = len(labels_x) + len(labels_u)

    fig2 = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=labels_x + labels_u,
        vertical_spacing=0.03,
    )
    for i, name in enumerate(labels_x):
        fig2.add_trace(
            go.Scatter(x=t_vec, y=simX[:, i], mode="lines", name=name, showlegend=False),
            row=i + 1,
            col=1,
        )
    for i, name in enumerate(labels_u):
        fig2.add_trace(
            go.Scatter(x=t_vec, y=simU[:, i], mode="lines", name=name, showlegend=False),
            row=len(labels_x) + i + 1,
            col=1,
        )
    fig2.update_xaxes(title_text="t [s]", row=rows, col=1)
    fig2.update_layout(
        title="OpenSCvx MPC — states & controls",
        template="plotly_dark",
        height=stack_height(rows),
    )
    fig2.show()

    # ── Figure 3: acceleration vs bounds ─────────────────────────────────────
    vs = simX[:, 3]
    Ds = simX[:, 4]
    dels = simX[:, 5]
    Fxd_np = (Cm1 - Cm2 * vs) * Ds - Cr2 * vs**2 - Cr0 * np.tanh(5.0 * vs)
    a_lat_np = C2 * vs**2 * dels + Fxd_np * np.sin(C1 * dels) / m
    a_long_np = Fxd_np / m

    acceleration_figure(
        t_vec,
        a_lat_np,
        a_long_np,
        a_max=A_MAX,
        title="OpenSCvx MPC — lateral & longitudinal acceleration",
    ).show()


# ── Main: closed-loop MPC simulation ──────────────────────────────────────────
if __name__ == "__main__":
    set_initial_guess(s0=S_INIT, v0=0.0)
    problem_mpc.initialize()

    simX = np.zeros((Nsim_max, 6))
    simU = np.zeros((Nsim_max, 2))
    t_sim = np.zeros(Nsim_max)
    horizon_snapshots: list[np.ndarray] = []

    t_total = 0.0
    tcomp_sum = 0.0
    tcomp_max = 0.0
    lap_complete = False

    for step in range(Nsim_max):
        t_start = _time.perf_counter()

        problem_mpc.reset()
        results = problem_mpc.solve()
        results = problem_mpc.post_process()

        elapsed = _time.perf_counter() - t_start
        tcomp_sum += elapsed
        tcomp_max = max(tcomp_max, elapsed)

        nodes = results.nodes
        s0_cur = float(nodes["s"][0, 0])
        v0_cur = float(nodes["v"][0, 0])

        # Record current state (node 0) and applied control (node 0)
        simX[step] = [
            nodes["s"][0, 0],
            nodes["n"][0, 0],
            nodes["alpha"][0, 0],
            nodes["v"][0, 0],
            nodes["D"][0, 0],
            nodes["delta"][0, 0],
        ]
        simU[step] = [nodes["derD"][0, 0], nodes["derDelta"][0, 0]]
        t_sim[step] = t_total

        # Save a downsample of horizon state arrays for plotting
        if step % 5 == 0:
            traj = results.trajectory
            horizon_snapshots.append(
                np.column_stack(
                    [
                        traj["s"][:, 0],
                        traj["n"][:, 0],
                        traj["alpha"][:, 0],
                        traj["v"][:, 0],
                    ]
                )
            )

        print(
            f"step {step:4d}: s={s0_cur:6.3f} m  v={v0_cur:.3f} m/s  "
            f"n={nodes['n'][0, 0]:+.4f} m  t_cpu={elapsed * 1e3:.1f} ms"
        )

        update_initial_conditions(nodes)
        shift_guess(nodes)
        t_total += dt_mpc

        # Check lap completion (s crossed pathlength)
        if s0_cur > pathlength + 0.1:
            # Trim to one timed lap: find first crossing of s=0
            N0 = int(np.where(np.diff(np.sign(simX[: step + 1, 0])))[0][0])
            Nsim_actual = step - N0
            simX = simX[N0:step]
            simU = simU[N0:step]
            t_sim = t_sim[N0:step] - t_sim[N0]
            lap_complete = True
            break

    if not lap_complete:
        Nsim_actual = step + 1
        simX = simX[:Nsim_actual]
        simU = simU[:Nsim_actual]
        t_sim = t_sim[:Nsim_actual]

    print("\n=== Race Car MPC Results ===")
    print(f"  Steps simulated : {Nsim_actual}")
    print(f"  Lap time        : {t_sim[-1]:.3f} s")
    print(f"  Avg solve time  : {tcomp_sum / (step + 1) * 1e3:.1f} ms")
    print(f"  Max solve time  : {tcomp_max * 1e3:.1f} ms")
    print(f"  Avg speed       : {simX[:, 3].mean():.3f} m/s")
    print(f"  Lap complete    : {lap_complete}")

    if os.environ.get("OPENSCVX_NO_PLOT") is None:
        plot_mpc_results(simX, simU, t_sim, horizon_snapshots)

        from examples.car.racing._viser import (
            create_race_car_chase_viser_server,
            create_race_car_viser_server,
        )

        create_race_car_viser_server(
            simX=simX,
            t_sim=t_sim,
            track_file="LMS_Track.txt",
            lane_width=LANE_WIDTH,
            trim_warmup=False,
            title="Race Car MPC",
        )
        chase_server = create_race_car_chase_viser_server(
            simX=simX,
            t_sim=t_sim,
            track_file="LMS_Track.txt",
            lane_width=LANE_WIDTH,
            trim_warmup=False,
            title="Race Car MPC",
        )
        print("Overview camera and chase camera are on separate Viser ports (two browser tabs).")
        chase_server.sleep_forever()
