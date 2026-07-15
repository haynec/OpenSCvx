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
import threading

import numpy as np
import trimesh
import viser
from PIL import Image

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
from openscvx import Problem

# ── USER CONFIG ───────────────────────────────────────────────────────────────
FINAL_X: float = 0.0    # model-X horizontal coordinate of landing target
FINAL_Y: float = 0.0    # model-Y horizontal coordinate of landing target
ELEV_SCALE: float = 6.0 # DEM vertical exaggeration (model units)
Z_OFFSET: float = -3.0  # DEM vertical shift in model units (fixed)

# ── Scene / DEM constants ─────────────────────────────────────────────────────
SCENE_SCALE: float = 1.0  # Animated server uses model-unit viser coords; no extra scale needed
DEM_GRID: int = 2048
TERRAIN_HALF_EXTENT: float = 15.0
GREY_BASE = np.array([148, 150, 152], dtype=np.float32) / 255.0
_K_AMBIENT: float = 0.03
_K_PRIMARY: float = 2.5

# ── DEM loading ───────────────────────────────────────────────────────────────
_DEM_PATH = os.path.join(os.path.dirname(__file__), "senns_dem.png")


def _load_dem_normalized() -> np.ndarray:
    img = Image.open(_DEM_PATH)
    raw = np.array(img, dtype=np.uint16)
    lo, hi = float(raw.min()), float(raw.max())
    arr = np.array(img.resize((DEM_GRID, DEM_GRID), Image.BILINEAR), dtype=np.float32)
    return (arr - lo) / max(hi - lo, 1.0)


_dem_norm: np.ndarray = _load_dem_normalized()


def _dem_altitude_at(model_x: float, model_y: float, elev_scale: float, z_offset: float) -> float:
    """Model altitude (world-z) at horizontal position (model_x, model_y) on the DEM surface."""
    half = TERRAIN_HALF_EXTENT
    fi = (model_y + half) / (2.0 * half) * (DEM_GRID - 1)
    fj = (model_x + half) / (2.0 * half) * (DEM_GRID - 1)
    i0 = int(np.clip(fi, 0, DEM_GRID - 2))
    j0 = int(np.clip(fj, 0, DEM_GRID - 2))
    di = float(np.clip(fi - i0, 0.0, 1.0))
    dj = float(np.clip(fj - j0, 0.0, 1.0))
    h = (
        (1 - di) * (1 - dj) * _dem_norm[i0, j0]
        + di * (1 - dj) * _dem_norm[i0 + 1, j0]
        + (1 - di) * dj * _dem_norm[i0, j0 + 1]
        + di * dj * _dem_norm[i0 + 1, j0 + 1]
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


# ── Terrain helpers ────────────────────────────────────────────────────────────

def _make_terrain_faces() -> np.ndarray:
    N = DEM_GRID
    r, c = np.arange(N - 1, dtype=np.int32), np.arange(N - 1, dtype=np.int32)
    i = (r[:, None] * N + c[None, :]).ravel()
    return np.concatenate(
        [np.stack([i, i + 1, i + N], axis=-1),
         np.stack([i + 1, i + N + 1, i + N], axis=-1)],
        axis=0,
    ).astype(np.int32)


_terrain_faces = _make_terrain_faces()


def _make_terrain_vertices(elev_scale: float, z_offset: float = 0.0) -> np.ndarray:
    N = DEM_GRID
    ext = TERRAIN_HALF_EXTENT * SCENE_SCALE
    xs = np.linspace(-ext, ext, N, dtype=np.float32)
    ys = np.linspace(-ext, ext, N, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ZZ = ((_dem_norm * float(elev_scale) + float(z_offset)) * SCENE_SCALE).astype(np.float32)
    return np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)


def _compute_vertex_normals(elev_scale: float) -> np.ndarray:
    N = DEM_GRID
    ext = TERRAIN_HALF_EXTENT * SCENE_SCALE
    cell = 2.0 * ext / (N - 1)
    ZZ = _dem_norm * float(elev_scale) * SCENE_SCALE
    dz_dx = np.gradient(ZZ, cell, axis=1).astype(np.float32)
    dz_dy = np.gradient(ZZ, cell, axis=0).astype(np.float32)
    normals = np.stack([-dz_dx.ravel(), -dz_dy.ravel(), np.ones(N * N, dtype=np.float32)], axis=-1)
    return normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)


def _bake_colors(normals: np.ndarray, k_amb: float, k_pri: float,
                 az_deg: float, el_deg: float, enabled: bool) -> np.ndarray:
    if enabled:
        az, el = np.radians(az_deg), np.radians(el_deg)
        L = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)], dtype=np.float32)
        d = np.maximum(0.0, normals @ L)
    else:
        d = 0.0
    intensity = np.clip(k_amb + k_pri * d, 0.0, 1.0)
    rgb = (GREY_BASE[None, :] * intensity[:, None]).clip(0.0, 1.0)
    return (np.hstack([rgb, np.ones((len(rgb), 1), dtype=np.float32)]) * 255).astype(np.uint8)


def _build_trimesh(elev_scale: float, z_offset: float, normals: np.ndarray,
                   k_amb: float, k_pri: float, az: float, el: float, on: bool) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=_make_terrain_vertices(elev_scale, z_offset),
        faces=_terrain_faces,
        vertex_colors=_bake_colors(normals, k_amb, k_pri, az, el, on),
        process=False,
    )


# ── DEM overlay for an existing Viser server ─────────────────────────────────

def _add_dem_to_server(server: viser.ViserServer) -> None:
    """Overlay DEM terrain and lighting/elevation GUI onto an existing server."""
    # Disable scene lighting so baked vertex colours render exactly as computed
    server.scene.configure_default_lights(enabled=False)
    server.scene.add_light_ambient("/lights/ambient", color=(255, 255, 255), intensity=1.0)

    _st: dict = {
        "elev_scale": ELEV_SCALE,
        "k_amb": _K_AMBIENT,
        "k_pri": _K_PRIMARY,
        "az": 0.0,
        "el": 5.0,
        "on": True,
        "normals": _compute_vertex_normals(ELEV_SCALE),
        "_lock": threading.Lock(),
    }

    def _refresh() -> None:
        with _st["_lock"]:
            mesh = _build_trimesh(
                _st["elev_scale"], Z_OFFSET, _st["normals"],
                _st["k_amb"], _st["k_pri"], _st["az"], _st["el"], _st["on"],
            )
        server.scene.add_mesh_trimesh("/terrain", mesh)

    _refresh()

    with server.gui.add_folder("DEM Terrain"):
        elev_sl = server.gui.add_slider(
            "Elevation Scale (m)", min=0.1, max=20.0, step=0.1, initial_value=ELEV_SCALE
        )

        @elev_sl.on_update
        def _(_e=None) -> None:
            _st["elev_scale"] = float(elev_sl.value)
            _st["normals"] = _compute_vertex_normals(_st["elev_scale"])
            _refresh()

    with server.gui.add_folder("DEM Lighting"):
        server.gui.add_markdown("_Baked into DEM vertex colours; other scene objects unaffected._")
        p_on  = server.gui.add_checkbox("Enabled", initial_value=True)
        p_az  = server.gui.add_slider("Azimuth (°)",   min=0.0,  max=360.0, step=1.0,   initial_value=0.0)
        p_el  = server.gui.add_slider("Elevation (°)", min=0.5,  max=89.0,  step=0.5,   initial_value=5.0)
        p_str = server.gui.add_slider("Strength",      min=0.0,  max=5.0,   step=0.05,  initial_value=_K_PRIMARY)
        amb_sl = server.gui.add_slider("Ambient",      min=0.0,  max=0.5,   step=0.005, initial_value=_K_AMBIENT)

        def _sync_light(_e=None) -> None:
            _st.update(
                on=bool(p_on.value), az=float(p_az.value), el=float(p_el.value),
                k_pri=float(p_str.value), k_amb=float(amb_sl.value),
            )
            _refresh()

        for _ctrl in (p_on, p_az, p_el, p_str, amb_sl):
            _ctrl.on_update(_sync_light)

    with server.gui.add_folder("Info"):
        server.gui.add_markdown(
            f"**Landing target**  \n"
            f"Model X = {FINAL_X:.2f}  Y = {FINAL_Y:.2f}  \n"
            f"DEM altitude (Z) = **{FINAL_ALTITUDE:.3f} m**  \n\n"
            f"DEM: {DEM_GRID}×{DEM_GRID} · {_terrain_faces.shape[0]:,} tris"
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
    _add_dem_to_server(_srv)

    print("  Open http://localhost:8080 (trajectory), :8081 (SCP), :8082 (snapshots)")
    traj_server.sleep_forever()
