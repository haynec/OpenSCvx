"""6DoF PDG – single static solve with DEM terrain landing.

Same dynamics and solve pattern as ``6DoF_pdg.py``, but the model's
position/velocity states follow the standard ``(x, y, z)`` convention where
``z`` is altitude (up) and ``x``, ``y`` are horizontal — matching Viser's world
frame directly, so no coordinate remapping is needed when plotting.

The user specifies only the horizontal landing coordinates (X, Y in model
space); the terminal altitude (Z) is automatically looked up from the DEM
surface via bilinear interpolation.

The glideslope cone is centred on the DEM landing point rather than the world
origin, so it stays physically meaningful for non-zero landing altitudes.

Three Viser servers are started (same as ``6DoF_pdg.py``):
  • http://localhost:8080  – animated trajectory with DEM terrain
  • http://localhost:8081  – SCP iteration animation
  • http://localhost:8082  – static snapshots

Configure your landing target by editing the constants below:

    FINAL_X     – model-X  (horizontal, across-track)
    FINAL_Y     – model-Y  (horizontal, cross-track)
    ELEV_SCALE  – DEM vertical exaggeration (model units, e.g. 6.0)
    Z_OFFSET    – additional DEM vertical shift (model units, fixed at -3.0)

Usage::

    python examples/rocket/6DoF_pdg_dem_static.py
    # Open http://localhost:8080 in your browser for the animated view.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import viser

# ── Path setup ────────────────────────────────────────────────────────────────
# File lives in examples/rocket/senss/ — three parents up is the repo root.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import openscvx as ox
from examples.plotting_viser import (
    create_animated_plotting_server,
    create_scp_animated_plotting_server,
    create_snapshot_plotting_server,
)
from examples.rocket.senss._dem import (
    DemPlacement,
    DemShading,
    add_dem_terrain,
    dem_center_norm,
    dem_info_markdown,
    load_senns_dem,
)
from openscvx import Problem

# ── USER CONFIG ───────────────────────────────────────────────────────────────
FINAL_X: float = 0.0    # model-X horizontal coordinate of landing target
FINAL_Y: float = 0.0    # model-Y horizontal coordinate of landing target
ELEV_SCALE: float = 6.0 # DEM vertical exaggeration (model units)
Z_OFFSET: float = -3.0  # DEM vertical shift in model units (fixed)

# ── Scene / DEM constants ─────────────────────────────────────────────────────
# The animated server draws in model units, so one viser unit is one model unit.
SCENE_SCALE: float = 1.0
DEM_GRID: int = 2048
TERRAIN_HALF_EXTENT: float = 15.0

# The shared patch pins the DEM's *center pixel* to ``origin_m[2]``, so the
# offset that reproduces this example's "surface = norm * ELEV_SCALE + Z_OFFSET"
# convention is Z_OFFSET plus the center pixel's own relief.
DEM_PLACEMENT = DemPlacement(
    origin_m=(0.0, 0.0, Z_OFFSET + dem_center_norm(DEM_GRID) * ELEV_SCALE),
    scale_xyz=(1.0, 1.0, 1.0),
    yaw_deg=0.0,
    mirror_xy=(False, False),
    half_extent_m=TERRAIN_HALF_EXTENT,
    base_relief_m=ELEV_SCALE,
    grid=DEM_GRID,
)
DEM_SHADING = DemShading(azimuth_deg=0.0, elevation_deg=5.0, strength=2.5, ambient=0.03)


def _dem_altitude_at(model_x: float, model_y: float, elev_scale: float, z_offset: float) -> float:
    """Model altitude (world-z) at horizontal position (model_x, model_y) on the DEM surface."""
    dem = load_senns_dem(DEM_GRID)
    half = TERRAIN_HALF_EXTENT
    fi = (model_y + half) / (2.0 * half) * (DEM_GRID - 1)
    fj = (model_x + half) / (2.0 * half) * (DEM_GRID - 1)
    i0 = int(np.clip(fi, 0, DEM_GRID - 2))
    j0 = int(np.clip(fj, 0, DEM_GRID - 2))
    di = float(np.clip(fi - i0, 0.0, 1.0))
    dj = float(np.clip(fj - j0, 0.0, 1.0))
    h = (
        (1 - di) * (1 - dj) * dem[i0, j0]
        + di * (1 - dj) * dem[i0 + 1, j0]
        + (1 - di) * dj * dem[i0, j0 + 1]
        + di * dj * dem[i0 + 1, j0 + 1]
    )
    return float(h) * elev_scale + z_offset


# Compute terminal altitude from DEM at the user-specified landing point
FINAL_ALTITUDE: float = _dem_altitude_at(FINAL_X, FINAL_Y, ELEV_SCALE, Z_OFFSET)

# ── Problem definition (mirrors 6DoF_pdg.py with DEM-aware terminal) ──────────
n = 5
gI = 1.0
l_arm = 0.25
J_diag_vals = np.array([0.168 * 2e-2, 0.168, 0.168])
J_mat = ox.Diag(J_diag_vals)
J_inv_mat = ox.Inv(ox.Diag(J_diag_vals))
g0 = 1.0
Isp = 30.0
m_dry = 1.0
v_max = 3.0
w_max = 0.3752
del_max = 20.0
theta_max = 75.0
T_min = 1.5
T_max = 6.5
gamma = 75.0
beta = 0.01
c_ax = 0.5
c_ayz = 1.0
S_a = 0.5
rho_air = 1.0
l_p = 0.05
SQRT1_2 = float(np.sqrt(0.5))
# (x, y, z=altitude) — was (altitude=7.5, y=4.5, z=2.5) under the old
# (altitude, y, z) model convention.
initial_position = np.array([2.5, 4.5, 7.5])

CA = ox.Diag(ox.Concat(c_ax, c_ayz, c_ayz))
r_arm = ox.Concat(-l_arm, 0.0, 0.0)
r_cp = ox.Concat(l_p, 0.0, 0.0)

mass = ox.State("mass", shape=(1,))
mass.max, mass.min, mass.initial = [2.0], [1.0], [2.0]
mass.final = [ox.Maximize(1.5)]

position = ox.State("position", shape=(3,))
position.max = [15.0, 15.0, 20.0]
position.min = [-15.0, -15.0, -2.0]
position.initial = [ox.Free(float(v)) for v in initial_position]
position.final = [ox.Free(FINAL_X), ox.Free(FINAL_Y), ox.Free(FINAL_ALTITUDE)]

velocity = ox.State("velocity", shape=(3,))
velocity.max = [v_max, v_max, v_max]
velocity.min = [-v_max, -v_max, -v_max]
velocity.initial = [0.0, -2.8, -0.5]
velocity.final = [0.0, 0.0, 0.0]

# Attitude uses the body-frame (x, y, z, w) quaternion convention, independent
# of the world (x, y, z=altitude) convention. A quaternion of (0, -√½, 0, √½)
# rotates the body's axial (thrust) axis, body-x, onto world +z, i.e. "upright"
# for a hover/landing where thrust must counteract gravity along world -z.
attitude = ox.State("attitude", shape=(4,))
attitude.max, attitude.min = [1.0] * 4, [-1.0] * 4
attitude.initial = [ox.Free(0.0), ox.Free(-SQRT1_2), ox.Free(0.0), ox.Free(SQRT1_2)]
attitude.final = [0.0, -SQRT1_2, 0.0, SQRT1_2]

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = [w_max, w_max, w_max]
angular_velocity.min = [-w_max, -w_max, -w_max]
angular_velocity.initial = [1e-8, 0.0, 0.0]
angular_velocity.final = [1e-8, 0.0, 0.0]

thrust = ox.Control("thrust", shape=(3,))
thrust.max = [T_max, T_max, T_max]
thrust.min = [-T_max, -T_max, -T_max]
thrust.guess = np.linspace(
    np.array([gI * mass.initial[0], 0, 0]),
    np.array([gI * m_dry, 0, 0]),
    n,
).reshape(-1, 3)

q1, q2, q3, q4 = attitude[0], attitude[1], attitude[2], attitude[3]
CBI = ox.Block(
    [
        [q4**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q4*q3), 2*(q4*q2 + q1*q3)],
        [2*(q4*q3 + q1*q2), q4**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q4*q1)],
        [2*(q1*q3 - q4*q2), 2*(q4*q1 + q2*q3), q4**2 - q1**2 - q2**2 + q3**2],
    ]
).T

w1, w2, w3 = angular_velocity[0], angular_velocity[1], angular_velocity[2]
attitude_dot = ox.Concat(
    0.5 * (w1*q4 - w2*q3 + w3*q2),
    0.5 * (w1*q3 - w3*q1 + w2*q4),
    0.5 * (w2*q1 - w1*q2 + w3*q4),
    -0.5 * (w1*q1 + w2*q2 + w3*q3),
)


def _cross(a, b):
    return ox.Concat(
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


A_aero = -0.5 * rho_air * ox.linalg.Norm(velocity) * S_a * CA @ CBI @ velocity

dynamics = {
    "mass": -(1 / (Isp * g0)) * ox.linalg.Norm(thrust) - beta,
    "position": velocity,
    "velocity": CBI.T @ (thrust + A_aero) / mass[0] + ox.Concat(0.0, 0.0, -gI),
    "attitude": attitude_dot,
    "angular_velocity": J_inv_mat @ (
        _cross(r_arm, thrust) + _cross(r_cp, A_aero)
        - _cross(angular_velocity, J_mat @ angular_velocity)
    ),
}

states = [mass, position, velocity, attitude, angular_velocity]
controls = [thrust]

constraint_exprs = []
for state in states:
    constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

constraint_exprs.append((position == initial_position).convex().at([0]))

# Terminal: all 3 position components fixed (altitude = DEM surface at (X, Y))
_final_pos_3d = np.array([FINAL_X, FINAL_Y, FINAL_ALTITUDE])
constraint_exprs.append((position == _final_pos_3d).convex().at([n - 1]))

constraint_exprs.append(ox.ctcs(1.0 * (mass - m_dry) >= 0))

# Glideslope cone centred on landing point (not world origin)
_horiz_rel = ox.Concat(position[0] - FINAL_X, position[1] - FINAL_Y)
_alt_above = position[2] - FINAL_ALTITUDE
constraint_exprs.append(
    ox.ctcs(
        0.1 * ox.linalg.Norm(_horiz_rel)
        - ox.Tan(ox.Constant(np.array(gamma * np.pi / 180.0))) * _alt_above
        <= 0
    )
)

constraint_exprs.append(ox.ctcs(0.1 * ox.linalg.Norm(velocity) ** 2 - v_max**2 <= 0))
# Tilt constraint: keeps the body's axial (thrust) axis within theta_max of
# world +z (altitude/up). This is the world-z analogue of the world-x tilt
# term (cos(theta_max) - 1 + 2*(q2^2+q3^2) <= 0) used when altitude was the
# model's x-axis: cos(theta_max) - R[2,0] <= 0, where R[2,0] = 2*(q1*q3 - q4*q2)
# is the world-z component of the rotated body-x axis.
constraint_exprs.append(
    ox.ctcs(
        1.0 * ox.Cos(ox.Constant(np.array(theta_max * np.pi / 180.0)))
        + 2.0 * (q4*q2 - q1*q3) <= 0
    )
)
constraint_exprs.append(ox.ctcs(1.0 * ox.linalg.Norm(angular_velocity) ** 2 - w_max**2 <= 0))
constraint_exprs.append(
    ox.ctcs(
        0.1 * ox.linalg.Norm(thrust)
        - thrust[0] / ox.Cos(ox.Constant(np.array(del_max * np.pi / 180.0)))
        <= 0
    )
)
constraint_exprs.append(ox.ctcs(0.1 * ox.linalg.Norm(thrust) ** 2 - T_max**2 <= 0))
constraint_exprs.append(ox.ctcs(0.1 * T_min**2 - ox.linalg.Norm(thrust) ** 2 <= 0))

t_final_guess = 10.0
time_cfg = ox.Time(
    initial=0.0,
    final=ox.Free(t_final_guess),
    min=0.0,
    max=10.0,
    time_dilation_min=0.2 * t_final_guess,
    time_dilation_max=2.0 * t_final_guess,
)

problem = Problem(
    N=n,
    states=states,
    controls=controls,
    dynamics=dynamics,
    constraints=constraint_exprs,
    time=time_cfg,
    float_dtype="float64",
    algorithm={
        "autotuner": ox.ConstantProximalWeight(),
        "lam_cost": 1e-2,
        "lam_vc": 1e1,
        "lam_prox": 1e0,
        "ep_tr": 5e-3,
        "ep_vc": 1e-6,
    },
)
problem.settings.dev.printing = False

# ── Viser result remapping ────────────────────────────────────────────────────
# The model's (x, y, z=altitude) position/velocity convention now matches
# Viser's world frame directly, so no coordinate axis remapping is needed
# (unlike 6DoF_pdg.py, which uses the older (altitude, y, z) model convention).
# Only the attitude quaternion's scalar-component order needs remapping.


def _model_attitude_xyzw_to_viser_wxyz(attitude_arr: np.ndarray) -> np.ndarray:
    q = np.asarray(attitude_arr, dtype=np.float64)
    if q.ndim == 1:
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)


def _remap_attitude_for_viser(result) -> None:
    attitude_data = result.trajectory.get("attitude")
    if attitude_data is not None:
        result.trajectory["attitude"] = _model_attitude_xyzw_to_viser_wxyz(
            np.asarray(attitude_data)
        )


def prepare_rocket_results_for_viser(results) -> None:
    """Remap PDG attitude quaternions (xyzw → wxyz) for Viser visualization."""
    _remap_attitude_for_viser(results)


# ── DEM overlay for an existing Viser server ─────────────────────────────────


def add_terrain(server: viser.ViserServer) -> None:
    """Overlay the DEM patch and a landing-target summary onto ``server``."""
    add_dem_terrain(
        server, placement=DEM_PLACEMENT, shading=DEM_SHADING, scene_scale=SCENE_SCALE
    )
    with server.gui.add_folder("Info"):
        server.gui.add_markdown(
            f"**Landing target**  \n"
            f"Model X = {FINAL_X:.2f}  Y = {FINAL_Y:.2f}  \n"
            f"DEM altitude (Z) = **{FINAL_ALTITUDE:.3f} m**  \n\n"
            + dem_info_markdown(DEM_PLACEMENT)
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Landing target: X={FINAL_X}, Y={FINAL_Y} → altitude (Z)={FINAL_ALTITUDE:.3f} m")
    print("Initializing and solving 6DoF PDG …")

    problem.initialize()
    result = problem.solve()
    result = problem.post_process()
    prepare_rocket_results_for_viser(result)

    converged = getattr(result, "converged", "?")
    print(f"  converged: {converged}")

    # Animated trajectory server with body-frame attitude and thrust plume
    traj_server = create_animated_plotting_server(
        result,
        thrust_key="thrust",
        thrust_style="plume",
        thrust_scale=0.4,
        thrust_plume_half_angle_deg=14.0,
        thrust_plume_color=(255, 130, 50),
        thrust_plume_opacity=0.5,
        thrust_remap_world_to_viser=False,
        show_grid=False,  # DEM terrain replaces the ground grid
    )

    # SCP iteration animation and static snapshot servers
    scp_server = create_scp_animated_plotting_server(result, frame_duration_ms=50.0)
    snapshot_server = create_snapshot_plotting_server(
        result, initial_n_snapshots=5, show_grid=True
    )

    # Overlay DEM terrain onto the trajectory server.
    # create_animated_plotting_server returns a ViserServer when controls="gui" (the default).
    _srv: viser.ViserServer = getattr(traj_server, "server", traj_server)  # type: ignore[assignment]
    add_terrain(_srv)

    print("  Open http://localhost:8080 (trajectory), :8081 (SCP), :8082 (snapshots)")
    traj_server.sleep_forever()
