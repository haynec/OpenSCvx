"""6DoF PDG on DEM terrain – realtime with viser.

Same 6-DoF powered-descent dynamics; the flat ground plane is replaced by a
realistic terrain surface from ``senss_dem.png``.

Key differences vs the static ``6DoF_pdg.py`` + ``6DoF_pdg_realtime.py``:
  * Problem defined inline with two extra parameters: ``final_altitude`` (scalar)
    and ``final_horiz`` (2-D) so the terminal position is always the DEM surface.
  * Glideslope cone centred on the DEM landing point, not the world origin.
  * DEM Z-offset slider shifts the whole terrain and auto-updates the terminal
    altitude constraint.
  * Realtime optimizer (``problem.step()``) runs in a background thread.
  * No secondary fill light (primary grazing + ambient only).

Usage::

    python examples/rocket/6DoF_pdg_dem.py
    # Open http://localhost:8080 in your browser.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib
import numpy as np
import viser

# ── Path setup ────────────────────────────────────────────────────────────────
# File lives in examples/rocket/senss/ — three parents up is the repo root.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import openscvx as ox
from examples.plotting_viser import (
    build_scp_step_results,
    extract_multishoot_trajectory,
    format_metrics_markdown,
    get_print_queue_data,
)
from examples.rocket.senss._dem import (
    DemPlacement,
    DemShading,
    dem_center_norm,
    dem_info_markdown,
    dem_trimesh,
    load_senss_dem,
    terrain_vertex_normals,
)
from openscvx import Problem
from openscvx.plotting.viser import compute_velocity_colors, model_vec_to_viser_xyz
from openscvx.utils import printing as _printing

# ── Silence console noise ─────────────────────────────────────────────────────
_printing.intro = lambda: None
_printing.print_problem_summary = lambda *a, **k: None
_printing.print_results_summary = lambda *a, **k: None

# ── Scene / DEM constants ─────────────────────────────────────────────────────
SCENE_SCALE: float = 2.0  # viser display units per model unit
DEM_GRID: int = 2048  # downsampled DEM resolution
TERRAIN_HALF_EXTENT: float = 15.0  # model-space half-width of terrain patch
DEFAULT_ELEV_SCALE: float = 6.0  # default DEM vertical exaggeration (model units)
DEFAULT_Z_OFFSET: float = -3.0  # DEM Z-offset in model units
DEFAULT_SHADING = DemShading(azimuth_deg=0.0, elevation_deg=5.0, strength=2.5, ambient=0.03)
_viridis = matplotlib.colormaps["viridis"]

# The shared DEM patch measures everything in metres; this example measures the
# scene in model units, so its "metres per viser unit" is 1 / SCENE_SCALE.
_TERRAIN_SCENE_SCALE: float = 1.0 / SCENE_SCALE


def _terrain_placement(elev_scale: float, z_offset: float) -> DemPlacement:
    """DEM patch for this example's ``surface = norm * elev + z_offset`` convention.

    The shared patch pins the DEM's *center pixel* to ``origin_m[2]``, so the
    equivalent offset is ``z_offset`` plus that pixel's own relief.
    """
    return DemPlacement(
        origin_m=(0.0, 0.0, z_offset + dem_center_norm(DEM_GRID) * elev_scale),
        scale_xyz=(1.0, 1.0, 1.0),
        yaw_deg=0.0,
        mirror_xy=(False, False),
        half_extent_m=TERRAIN_HALF_EXTENT,
        base_relief_m=elev_scale,
        grid=DEM_GRID,
    )


# DEM height at the centre (default landing point) with default elevation scale
_DEM_CENTER_H0 = dem_center_norm(DEM_GRID) * DEFAULT_ELEV_SCALE


def _dem_altitude_at(model_y: float, model_z: float, elev_scale: float, z_offset: float) -> float:
    """Model altitude (position[0]) of the DEM surface at horizontal (model_y, model_z).

    Row i  ↔  viser-Y  ↔  model-Y.
    Col j  ↔  viser-X  ↔  model-Z.
    """
    dem = load_senss_dem(DEM_GRID)
    half = TERRAIN_HALF_EXTENT
    frac_i = (model_y + half) / (2.0 * half) * (DEM_GRID - 1)
    frac_j = (model_z + half) / (2.0 * half) * (DEM_GRID - 1)
    i0 = int(np.clip(frac_i, 0, DEM_GRID - 2))
    j0 = int(np.clip(frac_j, 0, DEM_GRID - 2))
    i1, j1 = i0 + 1, j0 + 1
    fi = float(np.clip(frac_i - i0, 0.0, 1.0))
    fj = float(np.clip(frac_j - j0, 0.0, 1.0))
    h = (
        (1 - fi) * (1 - fj) * dem[i0, j0]
        + fi * (1 - fj) * dem[i1, j0]
        + (1 - fi) * fj * dem[i0, j1]
        + fi * fj * dem[i1, j1]
    )
    return float(h) * elev_scale + z_offset


# ── Problem parameters ────────────────────────────────────────────────────────
n = 5
gI = ox.Parameter("gI", value=1.0)
l_arm = ox.Parameter("l", value=0.25)
J_diag = ox.Parameter("J_diag", shape=(3,), value=np.array([0.168 * 2e-2, 0.168, 0.168]))
J_mat = ox.Diag(J_diag)
J_inv_mat = ox.Inv(ox.Diag(J_diag))
g0 = ox.Parameter("g0", value=1.0)
Isp = ox.Parameter("Isp", value=30.0)
m_dry = ox.Parameter("m_dry", value=1.0)
v_max = ox.Parameter("v_max", value=3.0)
w_max = ox.Parameter("w_max", value=0.3752)
del_max = ox.Parameter("del_max", value=20.0)
theta_max = ox.Parameter("theta_max", value=75.0)
T_min = ox.Parameter("T_min", value=1.5)
T_max = ox.Parameter("T_max", value=6.5)
gamma = ox.Parameter("gamma", value=75.0)
beta = ox.Parameter("beta", value=0.01)
c_ax = ox.Parameter("c_ax", value=0.5)
c_ayz = ox.Parameter("c_ayz", value=1.0)
S_a = ox.Parameter("S_a", value=0.5)
rho = ox.Parameter("rho", value=1.0)
l_p = ox.Parameter("l_p", value=0.05)
initial_position = ox.Parameter("initial_position", shape=(3,), value=np.array([7.5, 4.5, 2.5]))

# DEM-linked terminal position parameters
final_altitude = ox.Parameter("final_altitude", value=_DEM_CENTER_H0)  # auto-updated
final_horiz = ox.Parameter("final_horiz", shape=(2,), value=np.array([0.0, 0.0]))

CA = ox.Diag(ox.Concat(c_ax, c_ayz, c_ayz))
r_arm = ox.Concat(-l_arm, 0.0, 0.0)
r_cp = ox.Concat(l_p, 0.0, 0.0)

# ── States & controls ─────────────────────────────────────────────────────────
mass = ox.State("mass", shape=(1,))
mass.max, mass.min, mass.initial = [2.0], [1.0], [2.0]
mass.final = [ox.Maximize(1.5)]

position = ox.State("position", shape=(3,))
position.max = [20.0, 15.0, 15.0]
position.min = [-2.0, -15.0, -15.0]
position.initial = list(float(v) for v in initial_position.value)
position.final = [ox.Free(_DEM_CENTER_H0), ox.Free(0.0), ox.Free(0.0)]  # constrained below

velocity = ox.State("velocity", shape=(3,))
velocity.max = [v_max.value] * 3
velocity.min = [-v_max.value] * 3
velocity.initial = [-0.5, -2.8, 0.0]
velocity.final = [-0.1, 0.0, 0.0]

attitude = ox.State("attitude", shape=(4,))
attitude.max, attitude.min = [1.0] * 4, [-1.0] * 4
attitude.initial = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0), ox.Free(1.0)]
attitude.final = [0.0, 0.0, 0.0, 1.0]

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = [w_max.value] * 3
angular_velocity.min = [-w_max.value] * 3
angular_velocity.initial = [1e-8, 0.0, 0.0]
angular_velocity.final = [1e-8, 0.0, 0.0]

thrust = ox.Control("thrust", shape=(3,))
thrust.max = [T_max.value] * 3
thrust.min = [-T_max.value] * 3
thrust.guess = np.linspace(
    np.array([gI.value * mass.initial[0], 0, 0]),
    np.array([gI.value * m_dry.value, 0, 0]),
    n,
).reshape(-1, 3)

# ── Dynamics ──────────────────────────────────────────────────────────────────
q1, q2, q3, q4 = attitude[0], attitude[1], attitude[2], attitude[3]
CBI = ox.Block(
    [
        [q4**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q4 * q3), 2 * (q4 * q2 + q1 * q3)],
        [2 * (q4 * q3 + q1 * q2), q4**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q4 * q1)],
        [2 * (q1 * q3 - q4 * q2), 2 * (q4 * q1 + q2 * q3), q4**2 - q1**2 - q2**2 + q3**2],
    ]
).T

w1, w2, w3 = angular_velocity[0], angular_velocity[1], angular_velocity[2]
attitude_dot = ox.Concat(
    0.5 * (w1 * q4 - w2 * q3 + w3 * q2),
    0.5 * (w1 * q3 - w3 * q1 + w2 * q4),
    0.5 * (w2 * q1 - w1 * q2 + w3 * q4),
    -0.5 * (w1 * q1 + w2 * q2 + w3 * q3),
)


def _cross(a, b):
    return ox.Concat(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


A_aero = -0.5 * rho * ox.linalg.Norm(velocity) * S_a * CA @ CBI @ velocity

dynamics = {
    "mass": -(1 / (Isp * g0)) * ox.linalg.Norm(thrust) - beta,
    "position": velocity,
    "velocity": CBI.T @ (thrust + A_aero) / mass[0] + ox.Concat(-gI, 0.0, 0.0),
    "attitude": attitude_dot,
    "angular_velocity": J_inv_mat
    @ (
        _cross(r_arm, thrust)
        + _cross(r_cp, A_aero)
        - _cross(angular_velocity, J_mat @ angular_velocity)
    ),
}

# ── Constraints ───────────────────────────────────────────────────────────────
states = [mass, position, velocity, attitude, angular_velocity]
controls = [thrust]

constraint_exprs = []
for state in states:
    constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Initial position
constraint_exprs.append((position == initial_position).convex().at([0]))

# Terminal position: altitude on DEM surface, horizontal at final_horiz
constraint_exprs.append((position[0] == final_altitude).convex().at([n - 1]))
constraint_exprs.append((position[1:3] == final_horiz).convex().at([n - 1]))

# Remaining constraints
constraint_exprs.append(ox.ctcs(1.0 * (mass - m_dry) >= 0))

# Glideslope cone centred on the landing point (not the world origin)
_alt_above = position[0] - final_altitude
_horiz_off = position[1:3] - final_horiz
constraint_exprs.append(
    ox.ctcs(0.1 * ox.linalg.Norm(_horiz_off) - ox.Tan(gamma * np.pi / 180.0) * _alt_above <= 0)
)

constraint_exprs.append(ox.ctcs(0.1 * ox.linalg.Norm(velocity) ** 2 - v_max**2 <= 0))
constraint_exprs.append(
    ox.ctcs(1.0 * ox.Cos(theta_max * np.pi / 180.0) - 1.0 + 2.0 * (q2**2 + q3**2) <= 0)
)
constraint_exprs.append(ox.ctcs(1.0 * ox.linalg.Norm(angular_velocity) ** 2 - w_max**2 <= 0))
constraint_exprs.append(
    ox.ctcs(0.1 * ox.linalg.Norm(thrust) - thrust[0] / ox.Cos(del_max * np.pi / 180.0) <= 0)
)
constraint_exprs.append(ox.ctcs(0.1 * ox.linalg.Norm(thrust) ** 2 - T_max**2 <= 0))
constraint_exprs.append(ox.ctcs(0.1 * T_min**2 - ox.linalg.Norm(thrust) ** 2 <= 0))

# ── Time ──────────────────────────────────────────────────────────────────────
t_final_guess = 10.0
time_config = ox.Time(
    initial=0.0,
    final=ox.Free(t_final_guess),
    min=0.0,
    max=10.0,
    time_dilation_min=0.2 * t_final_guess,
    time_dilation_max=2.0 * t_final_guess,
)

# ── Problem ───────────────────────────────────────────────────────────────────
problem = Problem(
    N=n,
    states=states,
    controls=controls,
    dynamics=dynamics,
    constraints=constraint_exprs,
    time=time_config,
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
problem.solver.solver_args = {"abstol": 1e-7, "reltol": 1e-7}
problem.settings.dev.printing = True

# ── Problem parameter sync ────────────────────────────────────────────────────


def _sync_params() -> None:
    """Push current ox.Parameter values into problem.parameters."""
    p = problem
    p.parameters["gI"] = float(gI.value)
    p.parameters["l"] = float(l_arm.value)
    p.parameters["J_diag"] = np.asarray(J_diag.value, dtype=np.float64).reshape(3)
    p.parameters["g0"] = float(g0.value)
    p.parameters["Isp"] = float(Isp.value)
    p.parameters["m_dry"] = float(m_dry.value)
    p.parameters["v_max"] = float(v_max.value)
    p.parameters["w_max"] = float(w_max.value)
    p.parameters["del_max"] = float(del_max.value)
    p.parameters["theta_max"] = float(theta_max.value)
    p.parameters["T_min"] = float(T_min.value)
    p.parameters["T_max"] = float(T_max.value)
    p.parameters["gamma"] = float(gamma.value)
    p.parameters["beta"] = float(beta.value)
    p.parameters["c_ax"] = float(c_ax.value)
    p.parameters["c_ayz"] = float(c_ayz.value)
    p.parameters["S_a"] = float(S_a.value)
    p.parameters["rho"] = float(rho.value)
    p.parameters["l_p"] = float(l_p.value)
    p.parameters["initial_position"] = np.asarray(initial_position.value, dtype=np.float64)
    p.parameters["final_altitude"] = float(final_altitude.value)
    p.parameters["final_horiz"] = np.asarray(final_horiz.value, dtype=np.float64)


# ── Viser scene ───────────────────────────────────────────────────────────────


def create_dem_realtime_server() -> viser.ViserServer:
    server = viser.ViserServer(port=8080)
    server.gui.configure_theme(dark_mode=True, brand_color=(220, 80, 40))

    # Ambient-only: baked vertex colours render as-is on the DEM
    server.scene.configure_default_lights(enabled=False)
    server.scene.add_light_ambient("/lights/ambient", color=(255, 255, 255), intensity=1.0)

    # ── Shared mutable state ──────────────────────────────────────────────────
    _st: dict = {
        "elev_scale": DEFAULT_ELEV_SCALE,
        "z_offset": DEFAULT_Z_OFFSET,
        "shading": DEFAULT_SHADING,
        "normals": terrain_vertex_normals(_terrain_placement(DEFAULT_ELEV_SCALE, DEFAULT_Z_OFFSET)),
        "_lock": threading.Lock(),
        "running": True,
        "reset_requested": False,
        # Scene handles stored after creation so callbacks can reach them safely
        "target_handle": None,
        "target_drag": None,
        "start_handle": None,
    }

    def _viser_from_model(p: np.ndarray) -> tuple[float, float, float]:
        v = model_vec_to_viser_xyz(np.asarray(p, dtype=np.float64)) * SCENE_SCALE
        return (float(v[0]), float(v[1]), float(v[2]))

    def _update_final_altitude() -> None:
        """Recompute terminal altitude from current DEM params and push to problem.

        Also moves the target sphere and drag gizmo to stay on the (possibly
        shifted) DEM surface at the current horizontal landing position.
        """
        h_target = float(final_horiz.value[0])
        z_target = float(final_horiz.value[1])
        alt = _dem_altitude_at(h_target, z_target, _st["elev_scale"], _st["z_offset"])
        final_altitude.value = alt
        problem.parameters["final_altitude"] = alt
        if _st["target_handle"] is not None:
            new_pos = _viser_from_model(np.array([alt, h_target, z_target], dtype=np.float64))
            _st["target_handle"].position = new_pos
            if _st["target_drag"] is not None:
                _st["target_drag"].position = new_pos

    def _refresh_terrain() -> None:
        placement = _terrain_placement(_st["elev_scale"], _st["z_offset"])
        with _st["_lock"]:
            mesh = dem_trimesh(
                placement,
                _st["shading"],
                scene_scale=_TERRAIN_SCENE_SCALE,
                normals=_st["normals"],
            )
        server.scene.add_mesh_trimesh("/terrain", mesh)
        _update_final_altitude()

    # ── Initial scene objects ─────────────────────────────────────────────────
    # Upload terrain first (target_handle not yet set, so _update_final_altitude
    # just updates the parameter without touching any scene handle)
    _refresh_terrain()

    trajectory_handle = server.scene.add_point_cloud(
        "/trajectory",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=(255, 200, 80),
        point_size=0.08 * SCENE_SCALE,
    )

    init_pos = np.asarray(initial_position.value, dtype=np.float64)
    _start_handle = server.scene.add_icosphere(
        "/markers/start",
        radius=0.18 * SCENE_SCALE,
        color=(80, 220, 100),
        position=_viser_from_model(init_pos),
    )
    _st["start_handle"] = _start_handle

    _landing = np.array([float(final_altitude.value), 0.0, 0.0], dtype=np.float64)
    _landing_vis = _viser_from_model(_landing)
    _target_handle = server.scene.add_icosphere(
        "/markers/target",
        radius=0.25 * SCENE_SCALE,
        color=(240, 80, 60),
        position=_landing_vis,
    )
    _st["target_handle"] = _target_handle  # now safe for callbacks to update

    target_drag = server.scene.add_transform_controls(
        "/target_drag",
        position=_landing_vis,
        scale=0.4 * SCENE_SCALE,
        disable_rotations=True,
    )
    _st["target_drag"] = target_drag

    start_drag = server.scene.add_transform_controls(
        "/start_drag",
        position=_viser_from_model(init_pos),
        scale=0.4 * SCENE_SCALE,
        disable_rotations=True,
    )

    @target_drag.on_update
    def _(_) -> None:
        """Drag target; snap Z to DEM surface and update problem parameters."""
        vx, vy, _ = target_drag.position
        # viser_x = model_z * SCENE_SCALE,  viser_y = model_y * SCENE_SCALE
        model_y = vy / SCENE_SCALE
        model_z = vx / SCENE_SCALE
        alt = _dem_altitude_at(model_y, model_z, _st["elev_scale"], _st["z_offset"])
        vz = alt * SCENE_SCALE  # snap to DEM surface
        # Lock the gizmo and sphere to the surface
        target_drag.position = (vx, vy, vz)
        _st["target_handle"].position = (vx, vy, vz)
        # Update parameters
        fh = np.array([model_y, model_z], dtype=np.float64)
        final_horiz.value = fh
        final_altitude.value = alt
        problem.parameters["final_horiz"] = fh
        problem.parameters["final_altitude"] = alt
        _st["reset_requested"] = True

    server.scene.add_frame(
        "/origin",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=1.5 * SCENE_SCALE,
        axes_radius=0.03 * SCENE_SCALE,
    )

    # ── GUI ───────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Optimization Metrics"):
        metrics_md = server.gui.add_markdown(
            format_metrics_markdown(
                {
                    "iter": 0,
                    "J_tr": 0.0,
                    "J_vb": 0.0,
                    "J_vc": 0.0,
                    "cost": 0.0,
                    "dis_time": 0.0,
                    "solve_time": 0.0,
                    "prob_stat": "--",
                }
            )
        )

    with server.gui.add_folder("Problem Control", expand_by_default=True):
        reset_btn = server.gui.add_button("Apply Changes + Reset Problem")

        @reset_btn.on_click
        def _(_) -> None:
            _st["reset_requested"] = True

    with server.gui.add_folder("Algorithm Weights"):
        lam_cost_in = server.gui.add_number(
            "lam_cost", initial_value=problem.algorithm.lam_cost, min=1e-8, max=1e5, step=0.01
        )
        lam_vc_in = server.gui.add_number(
            "lam_vc", initial_value=problem.algorithm.lam_vc, min=1e-8, max=1e5, step=0.01
        )
        lam_prox_in = server.gui.add_number(
            "lam_prox", initial_value=problem.algorithm.lam_prox, min=1e-8, max=1e5, step=0.01
        )

        @lam_cost_in.on_update
        def _(_) -> None:
            problem.algorithm.lam_cost = float(lam_cost_in.value)

        @lam_vc_in.on_update
        def _(_) -> None:
            problem.algorithm.lam_vc = float(lam_vc_in.value)

        @lam_prox_in.on_update
        def _(_) -> None:
            problem.algorithm.lam_prox = float(lam_prox_in.value)

    with server.gui.add_folder("DEM Terrain"):
        elev_sl = server.gui.add_slider(
            "Elevation Scale (m)", min=0.1, max=20.0, step=0.1, initial_value=DEFAULT_ELEV_SCALE
        )

        @elev_sl.on_update
        def _(_e=None) -> None:
            _st["elev_scale"] = float(elev_sl.value)
            _st["normals"] = terrain_vertex_normals(
                _terrain_placement(_st["elev_scale"], _st["z_offset"])
            )
            _refresh_terrain()
            _st["reset_requested"] = True

    with server.gui.add_folder("Lighting"):
        server.gui.add_markdown("_Lights are baked into DEM vertex colours only._")
        p_on = server.gui.add_checkbox("Primary Light Enabled", initial_value=True)
        p_az = server.gui.add_slider("Azimuth (°)", min=0.0, max=360.0, step=1.0, initial_value=0.0)
        p_el = server.gui.add_slider(
            "Elevation (°)",
            min=0.5,
            max=89.0,
            step=0.5,
            initial_value=5.0,
        )
        p_str = server.gui.add_slider(
            "Strength", min=0.0, max=5.0, step=0.05, initial_value=DEFAULT_SHADING.strength
        )
        amb_sl = server.gui.add_slider(
            "Ambient Level", min=0.0, max=0.5, step=0.005, initial_value=DEFAULT_SHADING.ambient
        )

        def _sync_light(_e=None) -> None:
            _st["shading"] = DemShading(
                azimuth_deg=float(p_az.value),
                elevation_deg=float(p_el.value),
                strength=float(p_str.value),
                ambient=float(amb_sl.value),
                enabled=bool(p_on.value),
            )
            _refresh_terrain()

        for _ctrl in (p_on, p_az, p_el, p_str, amb_sl):
            _ctrl.on_update(_sync_light)

    with server.gui.add_folder("Dynamics / Constraint Parameters"):
        gI_in = server.gui.add_number(
            "gI",
            initial_value=float(gI.value),
            min=0.01,
            max=20.0,
            step=0.01,
        )
        g0_in = server.gui.add_number(
            "g0",
            initial_value=float(g0.value),
            min=0.01,
            max=20.0,
            step=0.01,
        )
        isp_in = server.gui.add_number(
            "Isp",
            initial_value=float(Isp.value),
            min=1.0,
            max=500.0,
            step=1.0,
        )
        m_dry_in = server.gui.add_number(
            "m_dry",
            initial_value=float(m_dry.value),
            min=0.5,
            max=2.0,
            step=0.01,
        )
        v_max_in = server.gui.add_number(
            "v_max",
            initial_value=float(v_max.value),
            min=0.1,
            max=20.0,
            step=0.05,
        )
        w_max_in = server.gui.add_number(
            "w_max",
            initial_value=float(w_max.value),
            min=0.01,
            max=5.0,
            step=0.01,
        )
        del_max_in = server.gui.add_number(
            "del_max (deg)",
            initial_value=float(del_max.value),
            min=1.0,
            max=89.0,
            step=0.5,
        )
        theta_max_in = server.gui.add_number(
            "theta_max (deg)",
            initial_value=float(theta_max.value),
            min=1.0,
            max=89.0,
            step=0.5,
        )
        t_min_in = server.gui.add_number(
            "T_min",
            initial_value=float(T_min.value),
            min=0.1,
            max=20.0,
            step=0.1,
        )
        t_max_in = server.gui.add_number(
            "T_max",
            initial_value=float(T_max.value),
            min=0.1,
            max=20.0,
            step=0.1,
        )
        gamma_in = server.gui.add_number(
            "gamma (deg)",
            initial_value=float(gamma.value),
            min=1.0,
            max=89.0,
            step=0.5,
        )
        beta_in = server.gui.add_number(
            "beta",
            initial_value=float(beta.value),
            min=0.0,
            max=1.0,
            step=0.001,
        )
        init_pos_in = server.gui.add_vector3(
            "initial_position",
            initial_value=tuple(float(v) for v in initial_position.value),
            step=0.1,
        )

        def _set(name: str, val) -> None:
            problem.parameters[name] = val

        @gI_in.on_update
        def _(_) -> None:
            gI.value = float(gI_in.value)
            _set("gI", gI.value)

        @g0_in.on_update
        def _(_) -> None:
            g0.value = float(g0_in.value)
            _set("g0", g0.value)

        @isp_in.on_update
        def _(_) -> None:
            Isp.value = float(isp_in.value)
            _set("Isp", Isp.value)

        @m_dry_in.on_update
        def _(_) -> None:
            m_dry.value = float(m_dry_in.value)
            _set("m_dry", m_dry.value)

        @v_max_in.on_update
        def _(_) -> None:
            v_max.value = float(v_max_in.value)
            _set("v_max", v_max.value)

        @w_max_in.on_update
        def _(_) -> None:
            w_max.value = float(w_max_in.value)
            _set("w_max", w_max.value)

        @del_max_in.on_update
        def _(_) -> None:
            del_max.value = float(del_max_in.value)
            _set("del_max", del_max.value)

        @theta_max_in.on_update
        def _(_) -> None:
            theta_max.value = float(theta_max_in.value)
            _set("theta_max", theta_max.value)

        @t_min_in.on_update
        def _(_) -> None:
            T_min.value = float(t_min_in.value)
            _set("T_min", T_min.value)

        @t_max_in.on_update
        def _(_) -> None:
            T_max.value = float(t_max_in.value)
            _set("T_max", T_max.value)

        @gamma_in.on_update
        def _(_) -> None:
            gamma.value = float(gamma_in.value)
            _set("gamma", gamma.value)

        @beta_in.on_update
        def _(_) -> None:
            beta.value = float(beta_in.value)
            _set("beta", beta.value)

        @init_pos_in.on_update
        def _(_) -> None:
            new_init = np.array(init_pos_in.value, dtype=np.float64)
            initial_position.value = new_init
            _set("initial_position", new_init)
            position.initial = list(float(v) for v in new_init)
            _st["start_handle"].position = _viser_from_model(new_init)
            start_drag.position = _viser_from_model(new_init)

    @start_drag.on_update
    def _(_) -> None:
        raw = np.asarray(start_drag.position, dtype=np.float64) / SCENE_SCALE
        # The remap is its own inverse, so the same call takes viser back to model.
        new_init = model_vec_to_viser_xyz(raw)
        initial_position.value = new_init
        problem.parameters["initial_position"] = new_init
        position.initial = list(float(v) for v in new_init)
        _st["start_handle"].position = tuple(start_drag.position)
        init_pos_in.value = tuple(new_init)

    with server.gui.add_folder("Info"):
        server.gui.add_markdown(
            "**6DoF PDG on DEM**  \n"
            + dem_info_markdown(_terrain_placement(DEFAULT_ELEV_SCALE, DEFAULT_Z_OFFSET))
            + "  \nTerminal altitude auto-locked to DEM surface."
        )

    # ── Apply edits + reset ───────────────────────────────────────────────────
    def _apply_all_edits() -> None:
        new_init = np.asarray(initial_position.value, dtype=np.float64)
        position.initial = list(float(v) for v in new_init)

        vm = float(v_max.value)
        velocity.max = [vm] * 3
        velocity.min = [-vm] * 3
        wm = float(w_max.value)
        angular_velocity.max = [wm] * 3
        angular_velocity.min = [-wm] * 3
        tm = float(T_max.value)
        thrust.max = [tm] * 3
        thrust.min = [-tm] * 3
        md = float(m_dry.value)
        mass.min = [md]

        # Recompute and set terminal altitude
        fh = np.asarray(final_horiz.value, dtype=np.float64)
        alt = _dem_altitude_at(float(fh[0]), float(fh[1]), _st["elev_scale"], _st["z_offset"])
        final_altitude.value = alt

        final_xyz = np.array([alt, float(fh[0]), float(fh[1])], dtype=np.float64)
        position.guess = np.linspace(new_init, final_xyz, n)
        velocity.guess = np.linspace(
            np.asarray(velocity.initial, dtype=np.float64),
            np.asarray(velocity.final, dtype=np.float64),
            n,
        )
        _upright = np.array([0.0, 0.0, 0.0, 1.0])
        attitude.guess = np.linspace(_upright, _upright, n)
        angular_velocity.guess = np.linspace(
            np.asarray(angular_velocity.initial, dtype=np.float64),
            np.asarray(angular_velocity.final, dtype=np.float64),
            n,
        )
        m0 = float(np.asarray(mass.initial, dtype=np.float64).flatten()[0])
        mass.guess = np.linspace(np.array([m0]), np.array([max(md, m0 - 0.2)]), n).reshape(-1, 1)
        thrust.guess = np.linspace(
            np.array([float(gI.value) * m0, 0.0, 0.0]),
            np.array([float(gI.value) * md, 0.0, 0.0]),
            n,
        ).reshape(-1, 3)
        _sync_params()

    # ── Trajectory update ─────────────────────────────────────────────────────
    def _show_node_positions() -> None:
        """Draw the current SCP iterate's node positions, unpropagated.

        The fallback for both ways the propagated trajectory can be missing:
        before ``V_history`` is first populated, and when extracting it fails.
        """
        try:
            x_traj = np.asarray(problem.state.x)
            if x_traj.size and x_traj.shape[1] >= 4:
                pts = (
                    model_vec_to_viser_xyz(np.asarray(x_traj[:, 1:4], dtype=np.float64))
                    * SCENE_SCALE
                ).astype(np.float32)
                trajectory_handle.points = pts
                trajectory_handle.colors = np.tile(
                    np.array([[255, 200, 80]], dtype=np.uint8), (pts.shape[0], 1)
                )
        except Exception:
            pass

    def _update_trajectory(V_ms: np.ndarray) -> None:
        """Extract the propagated multishot trajectory and push it to the point cloud."""
        try:
            nx = problem.settings.sim.n_states
            nu = problem.settings.sim.n_controls
            positions, velocities = extract_multishoot_trajectory(
                V_ms, nx, nu, position_slice=slice(1, 4), velocity_slice=slice(4, 7)
            )
            if len(positions) > 0:
                pts = (
                    model_vec_to_viser_xyz(np.asarray(positions, dtype=np.float64)) * SCENE_SCALE
                ).astype(np.float32)
                trajectory_handle.points = pts
                trajectory_handle.colors = compute_velocity_colors(velocities, cmap=_viridis)
        except Exception as exc:
            print(f"[trajectory] multishot extraction failed: {exc}")
            _show_node_positions()

    # ── Optimization loop ─────────────────────────────────────────────────────
    def _opt_loop() -> None:
        while _st["running"]:
            try:
                if _st["reset_requested"]:
                    _apply_all_edits()
                    problem.reset()
                    _st["reset_requested"] = False

                t0 = time.time()
                step = problem.step()
                elapsed_ms = (time.time() - t0) * 1000.0

                results = build_scp_step_results(step, elapsed_ms)
                results.update(get_print_queue_data(problem))
                metrics_md.content = format_metrics_markdown(results)

                # Prefer fully propagated multishot trajectory; fall back to node states
                if problem.history.V_history:
                    _update_trajectory(np.asarray(problem.history.V_history[-1]))
                else:
                    _show_node_positions()

                time.sleep(0.05)
            except Exception as exc:
                print(f"[opt_loop] {exc}")
                time.sleep(0.5)

    threading.Thread(target=_opt_loop, daemon=True).start()
    return server


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initializing 6DoF PDG on DEM terrain …")

    with contextlib.redirect_stdout(io.StringIO()):
        problem.initialize()

    _sync_params()
    print(f"  DEM center landing altitude: {_DEM_CENTER_H0:.2f} m")
    print("  Open http://localhost:8080 in your browser.\n")

    server = create_dem_realtime_server()
    server.sleep_forever()
