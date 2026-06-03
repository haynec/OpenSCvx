"""Batched 6-DoF powered descent guidance — ``solve_batched`` over initial conditions.

Runs ``N_BATCH = 50`` SCP solves in parallel, each starting from a different random
initial position drawn uniformly within the position bounds.  Demonstrates the
param-fix-pin mechanism introduced to support
``(state == parameter).convex().at([0])`` in the JAX-native backends:

* At lowering time :meth:`~openscvx.solvers.moreau_ptr_solver.MoreauPTRSolver` detects
  ``(position == initial_position).convex().at([0])`` as a param-fix pin and absorbs
  it into the ``"Fix"`` boundary-condition infrastructure.
* At call time :meth:`~openscvx.problem.Problem.solve_batched` receives a
  ``(B, 3)`` batched ``initial_position`` parameter and auto-builds the
  ``x0_stack`` from it — no manual stack construction required.

The same physics as ``base_problems/6DoF_pdg_realtime_base.py`` but rebuilt
with a Moreau backend so the whole SCP loop is a single XLA kernel.

After solving, all 50 trajectories are propagated through the full nonlinear
dynamics via :meth:`~openscvx.problem.Problem.post_process_batched`, and the
smooth high-fidelity paths are shown in the Viser visualisation.

Run::

    python examples/realtime/6DoF_pdg_batched_ic.py

Requires ``pip install openscvx[moreau]``.
"""

from __future__ import annotations

import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import viser

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(repo_root_dir)

import openscvx as ox
from openscvx import Problem
from openscvx.plotting.viser import (
    add_animated_trail,
    add_animation_controls,
    add_attitude_frame,
    add_glideslope_cone,
    add_position_marker,
    compute_velocity_colors,
    create_server,
)

# ── Problem dimensions ──────────────────────────────────────────────────────
N = 5          # discretization nodes
N_BATCH = 50   # number of simultaneous solves

# ── Random initial-condition generator ─────────────────────────────────────
# Position bounds: [-10, 10]^3.  For physical realism we keep altitude
# (position[0]) in [3, 9] m and lateral coordinates in [-7, 7] m.
# All samples trivially satisfy the glide-slope constraint
# (||pos[1:]|| <= tan(75°) * pos[0]) because tan(75°) ≈ 3.73 and the
# maximum lateral norm (~9.9) is well below 3.73 * 3 ≈ 11.2.
def sample_initial_positions(n: int, *, seed: int = 42) -> np.ndarray:
    """Return shape ``(n, 3)`` array of random initial positions."""
    rng = np.random.default_rng(seed)
    altitude = rng.uniform(3.0, 9.0, size=(n,))
    lateral_y = rng.uniform(-7.0, 7.0, size=(n,))
    lateral_z = rng.uniform(-7.0, 7.0, size=(n,))
    return np.stack([altitude, lateral_y, lateral_z], axis=-1)

# ── Runtime parameters ──────────────────────────────────────────────────────
gI         = ox.Parameter("gI",    value=1.0)
l_arm      = ox.Parameter("l",     value=0.25)
J_diag     = ox.Parameter("J_diag", shape=(3,), value=np.array([0.168 * 2e-2, 0.168, 0.168]))
J_mat      = ox.Diag(J_diag)
J_inv_mat  = ox.Inv(ox.Diag(J_diag))
g0         = ox.Parameter("g0",    value=1.0)
Isp        = ox.Parameter("Isp",   value=30.0)
m_dry      = ox.Parameter("m_dry", value=1.0)
v_max      = ox.Parameter("v_max", value=3.0)
w_max      = ox.Parameter("w_max", value=0.3752)
del_max    = ox.Parameter("del_max", value=20.0)
theta_max  = ox.Parameter("theta_max", value=75.0)
T_min      = ox.Parameter("T_min", value=1.5)
T_max      = ox.Parameter("T_max", value=6.5)
gamma      = ox.Parameter("gamma", value=75.0)
beta       = ox.Parameter("beta",  value=0.01)
c_ax       = ox.Parameter("c_ax",  value=0.5)
c_ayz      = ox.Parameter("c_ayz", value=1.0)
S_a        = ox.Parameter("S_a",   value=0.5)
rho        = ox.Parameter("rho",   value=1.0)
l_p        = ox.Parameter("l_p",   value=0.05)

# Boundary-condition parameters (the ones we will batch over)
initial_position = ox.Parameter("initial_position", shape=(3,), value=np.array([7.5, 4.5, 2.5]))
final_position   = ox.Parameter("final_position",   shape=(2,), value=np.array([0.0, 0.0]))

CA    = ox.Diag(ox.Concat(c_ax, c_ayz, c_ayz))
r_arm = ox.Concat(-l_arm, 0.0, 0.0)
r_cp  = ox.Concat(l_p,    0.0, 0.0)

# ── States ──────────────────────────────────────────────────────────────────
mass = ox.State("mass", shape=(1,))
mass.max = [2.0]
mass.min = [1.0]
mass.initial = [2.0]
mass.final   = [ox.Maximize(1.5)]

position = ox.State("position", shape=(3,))
position.max = [10.0, 10.0, 10.0]
position.min = [-10.0, -10.0, -10.0]
position.initial = [
    ox.Free(float(initial_position.value[0])),
    ox.Free(float(initial_position.value[1])),
    ox.Free(float(initial_position.value[2])),
]
position.final = [0.0, ox.Free(0.0), ox.Free(0.0)]

velocity = ox.State("velocity", shape=(3,))
velocity.max = [ v_max.value,  v_max.value,  v_max.value]
velocity.min = [-v_max.value, -v_max.value, -v_max.value]
velocity.initial = [-0.5, -2.8, 0.0]
velocity.final   = [-0.1,  0.0, 0.0]

attitude = ox.State("attitude", shape=(4,))
attitude.max = [1.0, 1.0, 1.0, 1.0]
attitude.min = [-1.0, -1.0, -1.0, -1.0]
attitude.initial = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0), ox.Free(1.0)]
attitude.final   = [0.0, 0.0, 0.0, 1.0]

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = [ w_max.value,  w_max.value,  w_max.value]
angular_velocity.min = [-w_max.value, -w_max.value, -w_max.value]
angular_velocity.initial = [1e-8, 0.0, 0.0]
angular_velocity.final   = [1e-8, 0.0, 0.0]

# ── Controls ─────────────────────────────────────────────────────────────────
thrust = ox.Control("thrust", shape=(3,))
thrust.max = [ T_max.value,  T_max.value,  T_max.value]
thrust.min = [-T_max.value, -T_max.value, -T_max.value]
thrust.guess = np.linspace(
    np.array([gI.value * mass.initial[0], 0, 0]),
    np.array([gI.value * m_dry.value,     0, 0]),
    N,
).reshape(-1, 3)

# ── Quaternion kinematics ────────────────────────────────────────────────────
q1, q2, q3, q4 = attitude[0], attitude[1], attitude[2], attitude[3]

CBI = ox.Block(
    [
        [q4**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q4*q3),              2*(q4*q2 + q1*q3)             ],
        [2*(q4*q3 + q1*q2),              q4**2 - q1**2 + q2**2 - q3**2,  2*(q2*q3 - q4*q1)             ],
        [2*(q1*q3 - q4*q2),              2*(q4*q1 + q2*q3),              q4**2 - q1**2 - q2**2 + q3**2],
    ]
).T


def cross(a, b):
    return ox.Concat(
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


w1, w2, w3 = angular_velocity[0], angular_velocity[1], angular_velocity[2]
attitude_dot = ox.Concat(
    0.5*(w1*q4 - w2*q3 + w3*q2),
    0.5*(w1*q3 - w3*q1 + w2*q4),
    0.5*(w2*q1 - w1*q2 + w3*q4),
    -0.5*(w1*q1 + w2*q2 + w3*q3),
)

A_aero = -0.5 * rho * ox.linalg.Norm(velocity) * S_a * CA @ CBI @ velocity

dynamics = {
    "mass":             -(1 / (Isp * g0)) * ox.linalg.Norm(thrust) - beta,
    "position":         velocity,
    "velocity":         CBI.T @ (thrust + A_aero) / mass[0] + ox.Concat(-gI, 0.0, 0.0),
    "attitude":         attitude_dot,
    "angular_velocity": J_inv_mat @ (
        cross(r_arm, thrust)
        + cross(r_cp, A_aero)
        - cross(angular_velocity, J_mat @ angular_velocity)
    ),
}

# ── Constraints ──────────────────────────────────────────────────────────────
states   = [mass, position, velocity, attitude, angular_velocity]
controls = [thrust]

constraint_exprs = []
for st in states:
    constraint_exprs.extend([ox.ctcs(st <= st.max), ox.ctcs(st.min <= st)])

# Param-fix boundary constraints — these are the pins we batch over.
constraint_exprs.append((position       == initial_position).convex().at([0]))
constraint_exprs.append((position[1:3]  == final_position).convex().at([N - 1]))

constraint_exprs.append(ox.ctcs(1.0 * (mass - m_dry) >= 0))
constraint_exprs.append(ox.ctcs(
    0.1 * ox.linalg.Norm(position[1:]) - ox.Tan(gamma * np.pi / 180.0) * position[0] <= 0
))
constraint_exprs.append(ox.ctcs(0.1 * ox.linalg.Norm(velocity)**2 - v_max**2 <= 0))
constraint_exprs.append(ox.ctcs(
    1.0 * ox.Cos(theta_max * np.pi / 180.0) - 1.0 + 2.0 * (q2**2 + q3**2) <= 0
))
constraint_exprs.append(ox.ctcs(1.0 * ox.linalg.Norm(angular_velocity)**2 - w_max**2 <= 0))
constraint_exprs.append(ox.ctcs(
    0.1 * ox.linalg.Norm(thrust) - thrust[0] / ox.Cos(del_max * np.pi / 180.0) <= 0
))
constraint_exprs.append(ox.ctcs(0.1 * ox.linalg.Norm(thrust)**2 - T_max**2 <= 0))
constraint_exprs.append(ox.ctcs(0.1 * T_min**2 - ox.linalg.Norm(thrust)**2 <= 0))

# ── Time ─────────────────────────────────────────────────────────────────────
t_final_guess = 10.0
time_config = ox.Time(
    initial=0.0,
    final=ox.Free(t_final_guess),
    min=0.0,
    max=10.0,
    time_dilation_min=0.2 * t_final_guess,
    time_dilation_max=2.0 * t_final_guess,
)

# ── Problem — Moreau backend required for solve_batched ──────────────────────
problem = Problem(
    N=N,
    states=states,
    controls=controls,
    dynamics=dynamics,
    constraints=constraint_exprs,
    time=time_config,
    float_dtype="float64",
    solver={"backend": "moreau", "tol_gap_abs": 1e-7, "tol_feas": 1e-7},
    algorithm={
        "autotuner": ox.ConstantProximalWeight(),
        "lam_cost":  1e-2,
        "lam_vc":    1e1,
        "lam_prox":  1e0,
        "ep_tr":     5e-3,
        "ep_vc":     1e-6,
    },
)


# ── Viser coordinate helpers ─────────────────────────────────────────────────
# The model uses position = [altitude, lat_y, lat_z].
# Viser convention follows the realtime PDG example: model (a, b, c) → Viser (c, b, a).
# Altitude (position[0]) therefore maps to the Viser z-axis.

def model_to_viser(pts: np.ndarray) -> np.ndarray:
    """Remap ``(*, 3)`` model-frame positions to Viser XYZ.

    Model [altitude, lat_y, lat_z] → Viser [lat_z, lat_y, altitude].
    """
    pts = np.asarray(pts, dtype=np.float64)
    return np.stack([pts[..., 2], pts[..., 1], pts[..., 0]], axis=-1)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert ``(*, 4)`` quaternion from [q1,q2,q3,q4=w] to Viser [w,x,y,z]."""
    q = np.asarray(q, dtype=np.float64)
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)


def create_pdg_batched_viser_server(
    results,
    ic_batch: np.ndarray,
    *,
    gamma_deg: float = 75.0,
) -> viser.ViserServer:
    """Build a Viser scene showing all batched PDG propagated trajectories.

    Trajectories are drawn from ``results.x_full`` (high-fidelity nonlinear
    propagation) rather than the coarse SCP nodes, giving smooth continuous
    paths.  All ``B`` trajectories are shown simultaneously (green = converged,
    red = diverged).  A slider lets the user pick one run to highlight with an
    animated trail, position marker, and attitude frame.

    Args:
        results: :class:`~openscvx.algorithms.OptimizationResults` returned by
            :meth:`~openscvx.problem.Problem.post_process_batched`.  Must have
            ``x_full`` (shape ``(B, T, n_x)``) and ``t_full`` (shape
            ``(B, T)``) populated.
        ic_batch: ``(B, 3)`` array of initial positions (model frame).
        gamma_deg: Glide-slope half-angle used in the constraint
            ``0.1 * ||pos[1:]|| <= tan(gamma) * pos[0]``.  Converted to
            the visualised cone half-angle internally.
    """
    x_full    = np.asarray(results.x_full)          # (B, T, n_x)
    t_full    = np.asarray(results.t_full)           # (B, T)
    converged = np.asarray(results.converged, dtype=bool).reshape(-1)
    B, T = x_full.shape[0], x_full.shape[1]

    # State layout: mass[0], position[1:4], velocity[4:7], attitude[7:11],
    # angular_velocity[11:14].  Same indices in x_full as in the SCP x array.
    pos_model = x_full[:, :, 1:4]                   # (B, T, 3)
    att_xyzw  = x_full[:, :, 7:11]                  # (B, T, 4) — [q1,q2,q3,q4=w]
    vel_model = x_full[:, :, 4:7]                   # (B, T, 3)

    pos_viser = model_to_viser(pos_model)            # (B, T, 3)  Viser frame
    att_wxyz  = quat_xyzw_to_wxyz(att_xyzw)         # (B, T, 4)  [w,x,y,z]

    # Create server — frame camera around all trajectory points.
    server = create_server(pos_viser.reshape(-1, 3))
    server.gui.configure_theme(dark_mode=True)

    # ── Static scene ─────────────────────────────────────────────────────────
    server.scene.add_grid(
        "/grid",
        width=30.0,
        height=30.0,
        position=(0.0, 0.0, 0.0),
    )
    server.scene.add_icosphere(
        "/markers/landing",
        radius=0.25,
        position=(0.0, 0.0, 0.0),
        color=(255, 80, 80),
    )

    # Glideslope cone.  The constraint is 0.1*||pos[1:]|| <= tan(γ)*pos[0],
    # i.e. ||pos[1:]|| <= 10·tan(γ)·pos[0], so the effective half-angle is
    # atan(10·tan(γ)).  In Viser, altitude is the +z axis.
    effective_half_angle = float(
        np.degrees(np.arctan(10.0 * np.tan(np.radians(gamma_deg))))
    )
    add_glideslope_cone(
        server,
        apex=(0.0, 0.0, 0.0),
        height=12.0,
        glideslope_angle_deg=effective_half_angle,
        axis=(0.0, 0.0, 1.0),
        color=(80, 200, 120),
        opacity=0.12,
    )

    # ── All propagated trajectories ───────────────────────────────────────────
    for b in range(B):
        color = (80, 210, 100) if converged[b] else (220, 70, 70)
        pts = pos_viser[b].astype(np.float32)        # (T, 3)
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        server.scene.add_line_segments(
            f"/runs/{b}/path",
            points=segments,
            colors=color,
            line_width=2.0,
        )
        server.scene.add_icosphere(
            f"/runs/{b}/start",
            radius=0.12,
            position=tuple(pts[0]),
            color=(120, 220, 255) if converged[b] else (255, 160, 80),
        )

    # ── Highlighted run (animated over propagated time) ───────────────────────
    ctx: dict = {"run": 0}
    _pos    = pos_viser[0].copy()      # (T, 3)
    _att    = att_wxyz[0].copy()       # (T, 4)
    _vel    = vel_model[0].copy()      # (T, 3)
    _colors = compute_velocity_colors(_vel)

    _, update_trail    = add_animated_trail(server, _pos, _colors, point_size=0.18)
    _, update_marker   = add_position_marker(server, _pos, radius=0.25, color=(100, 200, 255))
    _, update_attitude = add_attitude_frame(server, _pos, _att, axes_length=0.6)
    highlight_cbs = [cb for cb in (update_trail, update_marker, update_attitude) if cb is not None]

    # traj_time is the real-time axis for run 0 (updated in _load_run).
    traj_time = t_full[0].copy()       # (T,)

    def _load_run(run_idx: int) -> None:
        ctx["run"] = run_idx
        _pos[...]    = pos_viser[run_idx]
        _att[...]    = att_wxyz[run_idx]
        _vel[...]    = vel_model[run_idx]
        _colors[...] = compute_velocity_colors(_vel)
        traj_time[:] = t_full[run_idx]
        for cb in highlight_cbs:
            cb(0)

    # ── GUI ──────────────────────────────────────────────────────────────────
    n_ok = int(converged.sum())
    with server.gui.add_folder("Batch overview"):
        server.gui.add_markdown(
            f"**{B} initial conditions** — "
            f"green = converged ({n_ok}), red = diverged ({B - n_ok})."
        )
        run_slider = server.gui.add_slider(
            "Highlight run",
            min=0,
            max=B - 1,
            step=1,
            initial_value=0,
        )

        @run_slider.on_update
        def _(_) -> None:
            _load_run(int(run_slider.value))

    add_animation_controls(server, traj_time, highlight_cbs, folder_name="Playback")

    return server


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initializing 6-DoF PDG problem (Moreau backend) …")
    t0 = time.perf_counter()
    problem.initialize()
    t_init = time.perf_counter() - t0
    print(f"  initialize()  {t_init:.2f} s")

    # Sample N_BATCH random initial positions
    ic_batch = sample_initial_positions(N_BATCH)
    print(f"\nSampled {N_BATCH} random initial positions (altitude ∈ [3, 9] m, lateral ∈ [-7, 7] m)")
    print(f"  ic_batch.shape = {ic_batch.shape}")

    # ── First call: compile + solve ───────────────────────────────────────────
    print(f"\nRunning solve_batched (first call — includes XLA compile) …")
    t0 = time.perf_counter()
    results = problem.solve_batched(
        parameters={"initial_position": ic_batch},
    )
    t_first = time.perf_counter() - t0
    print(f"  solve_batched  {t_first:.2f} s  (incl. compile)")

    # ── Post-process: propagate all B solutions through nonlinear dynamics ────
    print(f"\nRunning post_process_batched ({N_BATCH} elements, sequential) …")
    t0 = time.perf_counter()
    results = problem.post_process_batched(results)
    t_prop = time.perf_counter() - t0
    print(f"  post_process_batched  {t_prop:.2f} s")
    print(f"  propagated shapes: x_full={results.x_full.shape}, t_full={results.t_full.shape}")

    # ── Second call: cached artifact ──────────────────────────────────────────
    ic_batch2 = sample_initial_positions(N_BATCH, seed=99)
    print(f"\nRunning solve_batched (second call — compiled artifact cached) …")
    t0 = time.perf_counter()
    results2 = problem.solve_batched(
        parameters={"initial_position": ic_batch2},
    )
    t_second = time.perf_counter() - t0
    print(f"  solve_batched  {t_second:.2f} s")

    # ── Convergence statistics ────────────────────────────────────────────────
    # results.x has shape (B, N, n_x); results.converged has shape (B,)
    converged = np.asarray(results.converged)
    converged2 = np.asarray(results2.converged)

    print(f"\n── First batch results ──────────────────────────────────────────")
    print(f"  Converged:   {int(converged.sum())} / {N_BATCH}")
    print(f"  Diverged:    {int((~converged).sum())} / {N_BATCH}")

    if converged.any():
        x_traj = np.asarray(results.x)          # (B, N, n_x)
        # Position occupies state indices 1:4 (mass is index 0)
        pos_traj = x_traj[converged, :, 1:4]    # (n_conv, N, 3)
        x0_pos   = pos_traj[:, 0, :]            # initial position at node 0
        xf_pos   = pos_traj[:, -1, :]           # terminal position at node N-1

        print(f"\n  Requested initial altitudes (first 5 converged):")
        conv_idx = np.where(converged)[0][:5]
        for i in conv_idx:
            req  = ic_batch[i]
            got  = x_traj[i, 0, 1:4]
            print(f"    sample {i:2d}  requested=[{req[0]:.2f}, {req[1]:.2f}, {req[2]:.2f}]  "
                  f"got=[{got[0]:.2f}, {got[1]:.2f}, {got[2]:.2f}]")

        print(f"\n  Terminal position errors (||pos_f - [0,0,0]||, first 5 converged):")
        for i in conv_idx:
            err = float(np.linalg.norm(xf_pos[i - conv_idx[0]]))
            print(f"    sample {i:2d}  terminal error = {err:.4f} m")

    print(f"\n── Second batch results ─────────────────────────────────────────")
    print(f"  Converged:   {int(converged2.sum())} / {N_BATCH}")

    # ── Viser visualisation ───────────────────────────────────────────────────
    print("\nLaunching Viser — all 50 propagated PDG trajectories + highlight playback …")
    viser_server = create_pdg_batched_viser_server(results, ic_batch)
    viser_server.sleep_forever()
