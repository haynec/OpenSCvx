"""
6-DoF Powered Descent Guidance with Compound State-Triggered Constraints (cSTC)

Adapts the problem from CT-cSTC/CT-cSTC.ipynb using OpenSCvx's symbolic framework.
The cSTC structure — tighter operational limits that activate once state thresholds
are crossed — is encoded via continuous-time constraint satisfaction (CTCS) enforced
over trajectory-phase sub-intervals:

    (tight_constraint).over((k_start, N-1))
x
where k_start is computed from the trigger threshold applied to the linear
initialization trajectory (N=15 nodes, IC → TC straight line).

Trigger thresholds and resulting node intervals (from notebook cell 16):
    alt < 100 m  → nodes (12, 14): gimbal, tilt, ω, speed, tight glideslope
    alt < 200 m  → nodes  (9, 14): LOS boresight cone
    v < 35 m/s AND tilt < 60°  → nodes (5, 14): single-engine thrust limits

Reference: CT-cSTC/CT-cSTC.ipynb

When run as a script, launches three viser windows after solving:
  1. Animated trajectory – thrust plume, attitude frame, velocity-colored trail
  2. SCP convergence – node positions across iterations
  3. Snapshot grid – evenly-spaced body poses along the final path

Viser scene uses the same ENU frame as the model (x, y horizontal; z = altitude up).
"""

import os
import sys

import diffrax as dfx
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting_viser import (
    AnimatedServerHandle,
    create_animated_plotting_server,
    create_scp_animated_plotting_server,
    create_snapshot_plotting_server,
)
from openscvx import Problem
from openscvx.plotting.viser import add_glideslope_cone, add_animation_controls
from openscvx.plotting.viser.animated import (
    _generate_viewcone_faces,
    _generate_viewcone_vertices,
    _sensor_pose_in_world,
)

# ── Physical parameters (notebook cell 16) ───────────────────────────────────
G0  = 9.806    # m/s²
ISP = 330.0    # s
M_WET = 100_000.0   # kg
M_DRY =  85_000.0   # kg

# Initial conditions
R_I_INIT = np.array([200.0, 200.0, 500.0])   # m
V_I_INIT = np.array([  0.0,   0.0, -50.0])   # m/s
# 90° tilt about x-axis: euler_to_quat([90,0,0]) → [w, x, y, z] = [√2/2, √2/2, 0, 0]
# OpenSCvx quaternion convention [x, y, z, w]: Q_INIT = [√2/2, 0, 0, √2/2]
Q_INIT  = np.array([np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)])
W_INIT  = np.zeros(3)   # rad/s

# Terminal conditions
R_I_FINAL = np.zeros(3)                      # m
V_I_FINAL = np.array([0.0, 0.0, -5.0])       # m/s (gentle touchdown)
Q_FINAL   = np.array([0.0, 0.0, 0.0, 1.0])   # upright
W_FINAL   = np.zeros(3)

# Thrust limits (N)
T_MAX     = 2_200_000.0 * 3.0        # 3-engine
T_MIN     = 2_200_000.0 * 0.4 * 3.0
T_MAX_AFT = 2_200_000.0              # 1-engine aft phase
T_MIN_AFT = 2_200_000.0 * 0.4

# Hard control angle limits (deg)
DELTA_ENGINE_MAX_DEG    = 10.0
DELTA_BORESIGHT_MAX_DEG = 20.0

# State constraint limits
THETA_MAX_DEG   = 90.0              # max tilt from vertical
W_B_MAX_RAD_S   = np.pi/2  # max angular rate
GS_MAX_DEG      = 90.0 - 35.0      # glideslope cone half-angle from horizontal = 55°

# cSTC constraint alpha-scaling factors (notebook cell 16)
_ALPHA_SPD   = 0.95
_ALPHA_OMEGA = 0.99
_ALPHA_THETA = 0.92
_ALPHA_GS    = 0.92
_ALPHA_LOS   = 0.98

# cSTC tight constraint values (alpha applied)
DELTA_STC_DEG   = 1.0
THETA_STC_DEG   = 5.0 * _ALPHA_THETA        # ≈ 4.6°
OMEGA_STC_RAD_S = 2.5 * np.pi/180 * _ALPHA_OMEGA
V_STC_CONS      = 20.0 * _ALPHA_SPD         # ≈ 19 m/s
GS_STC_DEG      = (90.0 - 5.0) * _ALPHA_GS  # ≈ 78.2°
LOS_STC_DEG     = 5.0 * _ALPHA_LOS          # ≈ 4.9°

# Inertia (diagonal, kg·m²)
J_B_DIAG     = np.array([60.0,   60.0,   1.5])
J_B_INV_DIAG = np.array([1/60.0, 1/60.0, 1/1.5])

# Gravity (m/s²)
G_I = np.array([0.0, 0.0, -G0])

# Fuel consumption  ṁ = −α_m · T
ALPHA_M = 1.0 / (ISP * G0)

# Aerodynamics
RHO_AIR     = 1.225   # kg/m³
S_AREA      = 545.0   # m²
C_AERO_DIAG = np.array([0.4068, 0.4068, 0.0522])

# Lever arms (m) — engine to CM, aero-centre to engine
R_CM = np.array([ 0.0,  0.0, -14.0])
R_CP = np.array([ 0.0,  0.0,   3.0])

# ── Scaling (notebook cell 18) ────────────────────────────────────────────────
R_SCALE = np.linalg.norm(R_I_INIT)   # ≈ 574.46 m
M_SCALE = M_WET

_alpha_m_s = ALPHA_M * R_SCALE
_g_I_s     = G_I / R_SCALE
_r_cm_s    = R_CM / R_SCALE
_r_cp_s    = R_CP / R_SCALE
_J_B_s     = J_B_DIAG     / R_SCALE**2
_J_B_inv_s = J_B_INV_DIAG * R_SCALE**2
_rho_s     = RHO_AIR / (M_SCALE * R_SCALE) * R_SCALE**2   # = RHO_AIR * R_SCALE / M_SCALE

_T_max_s     = T_MAX     / (M_SCALE * R_SCALE)
_T_min_s     = T_MIN     / (M_SCALE * R_SCALE)  # noqa: F841
_T_max_aft_s = T_MAX_AFT / (M_SCALE * R_SCALE)
_T_min_aft_s = T_MIN_AFT / (M_SCALE * R_SCALE)

_r_init_s  = R_I_INIT / R_SCALE
_v_init_s  = V_I_INIT / R_SCALE
_v_final_s = V_I_FINAL / R_SCALE
_m_wet_s   = M_WET / M_SCALE   # 1.0
_m_dry_s   = M_DRY / M_SCALE   # 0.85

_v_stc_s   = V_STC_CONS / R_SCALE

# ── Precomputed constraint thresholds ─────────────────────────────────────────
def _tilt_sq_bound(theta_deg: float) -> float:
    """Squared quaternion tilt bound: ||[q_x, q_y]||² ≤ (1-cos θ)/2."""
    return (1.0 - np.cos(np.pi/180 * (theta_deg))) / 2.0


_tilt_sq_max    = _tilt_sq_bound(THETA_MAX_DEG)   # 0.5
_tilt_sq_stc    = _tilt_sq_bound(THETA_STC_DEG)
_tan_gs_max     = np.tan(np.pi/180 * (GS_MAX_DEG))
_tan_gs_stc     = np.tan(np.pi/180 * (GS_STC_DEG))
_cos_psi_stc    = np.cos(np.pi/180 * (LOS_STC_DEG))
_delta_stc_rad  = np.pi/180 * (DELTA_STC_DEG)
_omega_sq_stc   = OMEGA_STC_RAD_S**2

# ── cSTC trigger node intervals (from linear initialization) ──────────────────
# h_k = 500·(1 − k/14) m,  v_k = 50 − 45k/14 m/s,  θ_k = 90·(1 − k/14)°
# k_h1 = 12 : h_12 ≈ 71 m < 100 m  (h_11 ≈ 107 m)
# k_h2 =  9 : h_9 ≈ 179 m < 200 m  (h_8  ≈ 214 m)
# k_aft =  5 : v_5 ≈ 33.9 m/s < 35  AND  θ_5 ≈ 57.9° < 60°
N     = 15
K_H1  = 12
K_H2  =  9
K_AFT =  5
ALT_TRIGGER_H1_M = 100.0   # h < 100 m → tight terminal phase (nodes K_H1…N-1)
ALT_TRIGGER_H2_M = 200.0   # h < 200 m → LOS boresight phase (nodes K_H2…N-1)

# ── States ────────────────────────────────────────────────────────────────────
mass = ox.State("mass", shape=(1,))
mass.max = [_m_wet_s]
mass.min = [_m_dry_s]
mass.initial = [_m_wet_s]
mass.final   = [ox.Maximize(_m_dry_s)]   # fuel-optimal objective

position = ox.State("position", shape=(3,))
position.max = [ 1.5,  1.5,  1.5]
position.min = [-1.5, -1.5,  0.0]   # z ≥ 0 (above ground)
position.initial = [ox.Free(float(_r_init_s[0])),
                    ox.Free(float(_r_init_s[1])),
                    ox.Free(float(_r_init_s[2]))]
position.final   = [0.0, 0.0, 0.0]

velocity = ox.State("velocity", shape=(3,))
_v_box = 150.0 / R_SCALE
velocity.max = [ _v_box,  _v_box,  _v_box]
velocity.min = [-_v_box, -_v_box, -_v_box]
velocity.initial = [float(_v_init_s[0]), float(_v_init_s[1]), float(_v_init_s[2])]
velocity.final   = [float(_v_final_s[0]), float(_v_final_s[1]), float(_v_final_s[2])]

attitude = ox.State("attitude", shape=(4,))
attitude.max = [ 1.0,  1.0,  1.0,  1.0]
attitude.min = [-1.0, -1.0, -1.0, -1.0]
attitude.initial = [ox.Free(float(Q_INIT[0])), ox.Free(float(Q_INIT[1])),
                    ox.Free(float(Q_INIT[2])), ox.Free(float(Q_INIT[3]))]
attitude.final   = list(Q_FINAL)

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = [ W_B_MAX_RAD_S,  W_B_MAX_RAD_S,  W_B_MAX_RAD_S]
angular_velocity.min = [-W_B_MAX_RAD_S, -W_B_MAX_RAD_S, -W_B_MAX_RAD_S]
angular_velocity.initial = [1e-8, 0.0, 0.0]
angular_velocity.final   = [0.0, 0.0, 0.0]

# ── Controls: [T, δ_e, φ_e, δ_b, φ_b] ────────────────────────────────────────
thrust_mag = ox.Control("thrust_mag", shape=(1,))
thrust_mag.max   = [_T_max_s]
thrust_mag.min   = [_T_min_aft_s]   # always at least single-engine minimum
thrust_mag.guess = np.full((N, 1), float((_T_max_s + _T_min_aft_s) / 2))

gimbal_elev = ox.Control("gimbal_elev", shape=(1,))
gimbal_elev.max   = [ np.radians(DELTA_ENGINE_MAX_DEG)]
gimbal_elev.min   = [-np.radians(DELTA_ENGINE_MAX_DEG)]
gimbal_elev.guess = np.zeros((N, 1))

gimbal_az = ox.Control("gimbal_az", shape=(1,))
gimbal_az.max   = [ np.pi]
gimbal_az.min   = [-np.pi]
gimbal_az.guess = np.zeros((N, 1))

los_elev = ox.Control("los_elev", shape=(1,))
los_elev.max   = [ np.radians(DELTA_BORESIGHT_MAX_DEG)]
los_elev.min   = [-np.radians(DELTA_BORESIGHT_MAX_DEG)]
los_elev.guess = np.full((N, 1), np.radians(DELTA_BORESIGHT_MAX_DEG) * 0.5)

los_az = ox.Control("los_az", shape=(1,))
los_az.max   = [ np.pi]
los_az.min   = [-np.pi]
los_az.guess = np.zeros((N, 1))

# ── State & Control ───────────────────────────────────────────────────────────────────
states   = [mass, position, velocity, attitude, angular_velocity]
controls = [thrust_mag, gimbal_elev, gimbal_az, los_elev, los_az]

# ── Quaternion kinematics ─────────────────────────────────────────────────────
# OpenSCvx convention: attitude = [q_x, q_y, q_z, q_w]
q1, q2, q3, q4 = attitude[0], attitude[1], attitude[2], attitude[3]

# CBI: inertial→body DCM  (CBI.T is body→inertial, same convention as 6DoF_pdg.py)
CBI = ox.Block(
    [
        [q4**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q4*q3), 2*(q4*q2 + q1*q3)],
        [2*(q4*q3 + q1*q2), q4**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q4*q1)],
        [2*(q1*q3 - q4*q2), 2*(q4*q1 + q2*q3), q4**2 - q1**2 - q2**2 + q3**2],
    ]
).T

w1, w2, w3 = angular_velocity[0], angular_velocity[1], angular_velocity[2]
q1_dot = 0.5 * (w1*q4 - w2*q3 + w3*q2)
q2_dot = 0.5 * (w1*q3 - w3*q1 + w2*q4)
q3_dot = 0.5 * (w2*q1 - w1*q2 + w3*q4)
q4_dot = -0.5 * (w1*q1 + w2*q2 + w3*q3)
attitude_dot = ox.Concat(q1_dot, q2_dot, q3_dot, q4_dot)


def cross(a, b):
    """Symbolic 3-D cross product."""
    return ox.Concat(
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


# ── Thrust vector in body frame (polar parameterisation) ──────────────────────
T  = thrust_mag[0]
de = gimbal_elev[0]
pe = gimbal_az[0]
T_B = ox.Concat(
    T * ox.Sin(de) * ox.Cos(pe),
    T * ox.Sin(de) * ox.Sin(pe),
    T * ox.Cos(de),
)

# ── Aerodynamic drag in body frame ────────────────────────────────────────────
C_aero_ox = ox.Diag(ox.Concat(
    float(C_AERO_DIAG[0]), float(C_AERO_DIAG[1]), float(C_AERO_DIAG[2])
))
A_B = -0.5 * _rho_s * ox.linalg.Norm(velocity) * S_AREA * C_aero_ox @ CBI @ velocity

# ── Inertia tensors (diagonal, scaled) ────────────────────────────────────────
# Mass-varying model: J_eff = J_s × m_s  →  ω̇ = J_inv_s @ (τ/m_s − ω × J_s @ ω)
J_B_ox     = ox.Diag(ox.Concat(float(_J_B_s[0]),     float(_J_B_s[1]),     float(_J_B_s[2])))
J_B_inv_ox = ox.Diag(ox.Concat(float(_J_B_inv_s[0]), float(_J_B_inv_s[1]), float(_J_B_inv_s[2])))
r_cm_ox    = ox.Concat(float(_r_cm_s[0]), float(_r_cm_s[1]), float(_r_cm_s[2]))
r_cp_ox    = ox.Concat(float(_r_cp_s[0]), float(_r_cp_s[1]), float(_r_cp_s[2]))
g_I_ox     = ox.Concat(float(_g_I_s[0]),  float(_g_I_s[1]),  float(_g_I_s[2]))

torque = cross(r_cm_ox, T_B) + cross(r_cp_ox, A_B)
gyro   = cross(angular_velocity, J_B_ox @ angular_velocity)

# ── Dynamics (all in scaled units) ────────────────────────────────────────────
dynamics = {
    "mass":             -_alpha_m_s * T,
    "position":          velocity,
    "velocity":          CBI.T @ (T_B + A_B) / mass[0] + g_I_ox,
    "attitude":          attitude_dot,
    "angular_velocity":  J_B_inv_ox @ (torque / mass[0] - gyro),
}

# ── Shared sub-expressions for constraints ────────────────────────────────────
# Tilt: ||[q_x, q_y]||²  (matches notebook's ||[x[8], x[9]]||² in [w,x,y,z] ordering)
tilt_sq   = attitude[0]**2 + attitude[1]**2
r_xy_norm = ox.linalg.Norm(position[0:2])
v_sq      = ox.linalg.Norm(velocity)**2
omega_sq  = ox.linalg.Norm(angular_velocity)**2

# LOS boresight in inertial frame via body→inertial rotation
db = los_elev[0]
pb = los_az[0]
los_B = ox.Concat(
    ox.Sin(db) * ox.Cos(pb),
    ox.Sin(db) * ox.Sin(pb),
    ox.Cos(db),
)
los_I = CBI.T @ los_B   # unit-norm boresight in inertial frame

# ── Constraints ───────────────────────────────────────────────────────────────
constraints = []

# Boundary conditions (convex equality constraints)
constraints.append((position        == _r_init_s).convex().at([0]))
constraints.append((attitude        == Q_INIT).convex().at([0]))
constraints.append((velocity        == _v_init_s).convex().at([0]))
constraints.append((angular_velocity == W_INIT).convex().at([0]))
constraints.append((position        == R_I_FINAL).convex().at([N - 1]))
constraints.append((velocity        == _v_final_s).convex().at([N - 1]))
constraints.append((attitude        == Q_FINAL).convex().at([N - 1]))
constraints.append((angular_velocity == W_FINAL).convex().at([N - 1]))

# ── Always-on CTCS (entire trajectory) ────────────────────────────────────────
# Tilt angle: ||[q_x, q_y]||² ≤ (1−cos θ_max)/2
constraints.append(ox.ctcs(tilt_sq - _tilt_sq_max <= 0))

# Angular rate
constraints.append(ox.ctcs(omega_sq - W_B_MAX_RAD_S**2 <= 0))

# Glideslope cone: ||r_xy|| · tan(γ_max) ≤ z
constraints.append(ox.ctcs(r_xy_norm * _tan_gs_max - position[2] <= 0))

# Minimum dry mass
constraints.append(ox.ctcs(_m_dry_s - mass[0] <= 0))

# ── Phase 1: h < 100 m — tight constraints (nodes 12–14) ─────────────────────
# Scaled altitude trigger constraints
constraints.append((position[2] == ALT_TRIGGER_H1_M / R_SCALE).convex().at(K_H1))

constraints.append(ox.ctcs(gimbal_elev[0] - _delta_stc_rad <= 0, penalty="squared_relu").over((K_H1, N - 1)))
constraints.append(ox.ctcs(-gimbal_elev[0] - _delta_stc_rad <= 0, penalty="squared_relu").over((K_H1, N - 1)))
constraints.append(ox.ctcs(tilt_sq - _tilt_sq_stc <= 0, penalty="squared_relu").over((K_H1, N - 1)))
constraints.append(ox.ctcs(omega_sq - _omega_sq_stc <= 0, penalty="squared_relu").over((K_H1, N - 1)))
constraints.append(ox.ctcs(v_sq - _v_stc_s**2 <= 0, penalty="squared_relu").over((K_H1, N - 1)))
constraints.append(ox.ctcs(r_xy_norm * _tan_gs_stc - position[2] <= 0, penalty="squared_relu").over((K_H1, N - 1)))

# ── Phase 2: h < 200 m — LOS boresight (nodes 9–14) ─────────────────────────
# Scaled altitude trigger constraints
constraints.append((position[2] == ALT_TRIGGER_H2_M / R_SCALE).convex().at(K_H2))

# Angle between position vector and LOS ≤ ψ_stc:
#   r_I · los_I ≥ ||r_I|| · cos(ψ_stc)  →  ||r_I|| · cos_psi − r·los ≤ 0
r_dot_los = (position[0] * los_I[0]
             + position[1] * los_I[1]
             + position[2] * los_I[2])
constraints.append(
    ox.ctcs(ox.linalg.Norm(position) * _cos_psi_stc - r_dot_los <= 0, penalty="squared_relu").over((K_H2, N - 1))
)

# ── Aft phase: slow + upright — single-engine limit (nodes 5–14) ─────────────
constraints.append(ox.ctcs(thrust_mag[0] - _T_max_aft_s <= 0, penalty="squared_relu").over((K_AFT, N - 1)))

# ── Time (free final time with per-segment dilation) ─────────────────────────
_t_f_guess = 21.0
_t_scp     = _t_f_guess / (N - 1)   # ≈ 1.5 s per segment

time = ox.Time(
    initial=0.0,
    final=ox.Free(_t_f_guess),
    min=0.0,
    max=_t_f_guess * 2.0,
    # time_dilation_min=_t_scp * 0.5,
    # time_dilation_max=_t_scp * 10.0,
)

# ── Problem Assembly ───────────────────────────────────────────────────────────────────

problem = Problem(
    N=N,
    states=states,
    controls=controls,
    dynamics=dynamics,
    constraints=constraints,
    time=time,
    float_dtype="float64",
    algorithm={
        "autotuner": ox.AugmentedLagrangian(eta_lambda=1E3),
        "k_max": 1000,
    },
)

problem.settings.dev.debug = True

# ── Viser display parameters ──────────────────────────────────────────────────
# cSTC uses ENU inertial frame (x, y horizontal; z = altitude). Viser is Z-up,
# so positions map directly without the (z, y, x) swap used by the toy 6DoF_pdg
# example where altitude lives on model x.
SCENE_SCALE = 10.0          # 1 viser unit = 100 m
PLUME_SCALE = 8.0
ATTITUDE_AXES_LENGTH = 2.0
VIEWCONE_SCALE = 4.0          # viser units (~40 m at SCENE_SCALE=10)
_VISER_UP_AXIS = (0.0, 0.0, 1.0)
GS_HALFANGLE_DEG = 90.0 - GS_MAX_DEG
GS_STC_HALFANGLE_DEG = 90.0 - GS_STC_DEG
_POSITION_STATE_SLICE = slice(1, 4)


def cstc_model_to_viser_xyz(v: np.ndarray) -> np.ndarray:
    """Map cSTC ENU vectors to Viser XYZ (both Z-up; no axis permutation)."""
    return np.asarray(v, dtype=np.float64)


def _horizontal_disc_mesh(
    center: np.ndarray,
    radius: float,
    *,
    n_segments: int = 48,
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Flat disc in the plane perpendicular to ``normal`` (for altitude triggers)."""
    axis = np.asarray(normal, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(axis[0]) < 0.9 else np.array(
        [0.0, 1.0, 0.0], dtype=np.float32
    )
    u = ref - np.dot(ref, axis) * axis
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    center = np.asarray(center, dtype=np.float32)
    vertices = [center.copy()]
    for i in range(n_segments):
        angle = 2.0 * np.pi * i / n_segments
        vertices.append(center + radius * (np.cos(angle) * u + np.sin(angle) * v))
    vertices_arr = np.asarray(vertices, dtype=np.float32)

    faces = [
        [0, i + 1, ((i + 1) % n_segments) + 1]
        for i in range(n_segments)
    ]
    return vertices_arr, np.asarray(faces, dtype=np.int32)


def add_cstc_altitude_triggers(
    server,
    pos: np.ndarray,
    *,
    scene_scale_m: float = SCENE_SCALE,
) -> None:
    """Draw horizontal surfaces where altitude-based cSTC phases activate."""
    xy_extent = float(np.max(np.linalg.norm(pos[:, :2], axis=1)))
    radius = max(xy_extent * 1.25, 2.0)

    triggers = [
        (
            ALT_TRIGGER_H2_M,
            (80, 160, 255),
            K_H2,
            "LOS boresight cone",
        ),
        (
            ALT_TRIGGER_H1_M,
            (255, 80, 80),
            K_H1,
            "tight gimbal / tilt / ω / speed / GS",
        ),
    ]
    for alt_m, color, k_start, description in triggers:
        z = alt_m / scene_scale_m
        center = np.array([0.0, 0.0, z], dtype=np.float32)
        verts, faces = _horizontal_disc_mesh(center, radius)
        server.scene.add_mesh_simple(
            f"/cstc_triggers/alt_{int(alt_m)}",
            vertices=verts,
            faces=faces,
            color=color,
            opacity=0.16,
        )

        ring = verts[1:]
        ring_segments = np.stack(
            [[ring[i], ring[(i + 1) % len(ring)]] for i in range(len(ring))],
            axis=0,
        )
        server.scene.add_line_segments(
            f"/cstc_triggers/alt_{int(alt_m)}_ring",
            points=ring_segments.astype(np.float32),
            colors=color,
            line_width=2.5,
        )

        server.scene.add_label(
            f"/cstc_triggers/alt_{int(alt_m)}_label",
            text=f"h < {int(alt_m)} m → k≥{k_start}: {description}",
            position=(radius * 0.85, 0.0, z + 0.05),
        )

        # Vertical guide from landing pad to the trigger plane (shows altitude level)
        server.scene.add_line_segments(
            f"/cstc_triggers/alt_{int(alt_m)}_stem",
            points=np.array([[[-radius * 0.95, 0.0, 0.0], [-radius * 0.95, 0.0, z]]], dtype=np.float32),
            colors=tuple(int(c * 0.7) for c in color),
            line_width=1.5,
        )


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion array from xyzw (model) to wxyz (viser)."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)


def prepare_for_viser(result) -> None:
    """Remap cSTC trajectory in-place for viser display."""
    traj = result.trajectory

    pos_m = np.asarray(traj["position"], dtype=np.float64) * R_SCALE
    traj["position"] = cstc_model_to_viser_xyz(pos_m) / SCENE_SCALE

    vel_ms = np.asarray(traj["velocity"], dtype=np.float64) * R_SCALE
    traj["velocity"] = cstc_model_to_viser_xyz(vel_ms)

    traj["attitude"] = _xyzw_to_wxyz(np.asarray(traj["attitude"], dtype=np.float64))

    T_s = np.asarray(traj["thrust_mag"], dtype=np.float64).flatten()
    d_e = np.asarray(traj["gimbal_elev"], dtype=np.float64).flatten()
    phi_e = np.asarray(traj["gimbal_az"], dtype=np.float64).flatten()
    traj["thrust_body"] = np.stack(
        [
            T_s * np.sin(d_e) * np.cos(phi_e),
            T_s * np.sin(d_e) * np.sin(phi_e),
            T_s * np.cos(d_e),
        ],
        axis=-1,
    )

    traj["los_elev"] = np.asarray(traj["los_elev"], dtype=np.float64).flatten()
    traj["los_az"] = np.asarray(traj["los_az"], dtype=np.float64).flatten()

    for i in range(len(result.X)):
        X = np.asarray(result.X[i], dtype=np.float64, copy=True)
        pos_cols = X[:, _POSITION_STATE_SLICE] * R_SCALE
        X[:, _POSITION_STATE_SLICE] = cstc_model_to_viser_xyz(pos_cols) / SCENE_SCALE
        result.X[i] = X


def _los_body_to_sensor(de: float, pe: float) -> np.ndarray:
    """Body-to-sensor rotation aligning sensor +Z with the LOS boresight in body frame."""
    los_b = np.array(
        [np.sin(de) * np.cos(pe), np.sin(de) * np.sin(pe), np.cos(de)],
        dtype=np.float64,
    )
    los_b = los_b / (np.linalg.norm(los_b) + 1e-12)
    # Sensor +Z axis in body coordinates is the third row of R_sb (see _sensor_pose_in_world).
    ref = np.array([1.0, 0.0, 0.0]) if abs(los_b[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(ref, los_b)
    x = x / np.linalg.norm(x)
    y = np.cross(los_b, x)
    return np.stack([x, y, los_b], axis=0)


def add_cstc_los_viewcone(server, result):
    """Animated LOS boresight viewcone (active below 200 m altitude trigger)."""
    traj = result.trajectory
    pos = np.asarray(traj["position"], dtype=np.float64)
    attitude = np.asarray(traj["attitude"], dtype=np.float64)
    los_elev = np.asarray(traj["los_elev"], dtype=np.float64).flatten()
    los_az = np.asarray(traj["los_az"], dtype=np.float64).flatten()

    half_angle = np.radians(LOS_STC_DEG)
    base_vertices = _generate_viewcone_vertices(
        half_angle, half_angle, VIEWCONE_SCALE, norm_type=2
    )
    n_base = len(base_vertices) - 1
    faces = _generate_viewcone_faces(n_base)
    color = (80, 160, 255)

    R_sb_series = [_los_body_to_sensor(de, pe) for de, pe in zip(los_elev, los_az)]
    init_pos, init_wxyz = _sensor_pose_in_world(pos[0], attitude[0], R_sb_series[0])
    frame = server.scene.add_frame(
        "/los_viewcone",
        wxyz=init_wxyz,
        position=init_pos,
        axes_length=0.0,
        axes_radius=0.0,
    )
    server.scene.add_mesh_simple(
        "/los_viewcone/mesh",
        vertices=base_vertices,
        faces=faces,
        color=color,
        wireframe=False,
        opacity=0.4,
    )

    def update(frame_idx: int) -> None:
        p, w = _sensor_pose_in_world(
            pos[frame_idx], attitude[frame_idx], R_sb_series[frame_idx]
        )
        frame.position = p
        frame.wxyz = w

    return frame, update


def launch_viser_servers(result) -> None:
    """Create trajectory, SCP convergence, and snapshot viser servers."""
    pos = np.asarray(result.trajectory["position"])
    initial_alt_vis = float(np.max(pos[:, 2])) * 1.15

    handle = create_animated_plotting_server(
        result,
        thrust_key="thrust_body",
        thrust_style="plume",
        thrust_scale=PLUME_SCALE,
        thrust_plume_half_angle_deg=13.0,
        thrust_plume_color=(255, 140, 40),
        thrust_plume_opacity=0.55,
        thrust_remap_world_to_viser=False,
        attitude_key="attitude",
        attitude_axes_length=ATTITUDE_AXES_LENGTH,
        show_viewcone=False,
        trail_point_size=0.08,
        show_grid=True,
        dark_mode=True,
        scene_scale=1.0,
        controls="manual",
    )
    assert isinstance(handle, AnimatedServerHandle)
    _, update_viewcone = add_cstc_los_viewcone(handle.server, result)
    callbacks = handle.update_callbacks + [update_viewcone]
    add_animation_controls(handle.server, handle.traj_time, callbacks, loop=True)
    traj_server = handle.server

    add_glideslope_cone(
        traj_server,
        apex=(0.0, 0.0, 0.0),
        height=initial_alt_vis,
        glideslope_angle_deg=GS_HALFANGLE_DEG,
        axis=_VISER_UP_AXIS,
        color=(80, 200, 80),
        opacity=0.12,
    )
    add_glideslope_cone(
        traj_server,
        apex=(0.0, 0.0, 0.0),
        height=ALT_TRIGGER_H1_M / SCENE_SCALE,
        glideslope_angle_deg=GS_STC_HALFANGLE_DEG,
        axis=_VISER_UP_AXIS,
        color=(255, 180, 40),
        opacity=0.15,
    )

    add_cstc_altitude_triggers(traj_server, pos, scene_scale_m=SCENE_SCALE)

    traj_server.scene.add_icosphere(
        "/landing_pad",
        radius=0.12,
        color=(50, 255, 80),
        position=(0.0, 0.0, 0.0),
    )

    for k, color in [
        (K_AFT, (255, 210, 50)),
        (K_H2, (80, 160, 255)),
        (K_H1, (255, 80, 80)),
    ]:
        traj_server.scene.add_icosphere(
            f"/phase_markers/k{k}",
            radius=0.14,
            color=color,
            position=tuple(float(v) for v in pos[k]),
        )

    with traj_server.gui.add_folder("cSTC Phase Boundaries"):
        traj_server.gui.add_markdown(
            f"**Phase structure (N={N} nodes)**\n\n"
            f"**Altitude triggers** (horizontal discs):\n"
            f"- 🔵 h < {int(ALT_TRIGGER_H2_M)} m → k≥{K_H2}: LOS boresight viewcone  \n"
            f"- 🔴 h < {int(ALT_TRIGGER_H1_M)} m → k≥{K_H1}: tight terminal  \n\n"
            f"**Other trigger** (node marker only):\n"
            f"- 🟡 k={K_AFT}: single-engine thrust (v & tilt, not altitude)  \n"
        )

    scp_server = create_scp_animated_plotting_server(
        result,
        position_slice=_POSITION_STATE_SLICE,
        attitude_slice=slice(7, 11),
        show_attitudes=True,
        attitude_stride=3,
        attitude_axes_length=ATTITUDE_AXES_LENGTH,
        frame_duration_ms=80,
        scene_scale=1.0,
    )
    add_cstc_altitude_triggers(scp_server, pos, scene_scale_m=SCENE_SCALE)

    snap_server = create_snapshot_plotting_server(
        result,
        attitude_axes_length=ATTITUDE_AXES_LENGTH,
        show_body_frame=True,
        initial_n_snapshots=6,
        show_grid=True,
        background_color=(240, 240, 245),
    )
    add_cstc_altitude_triggers(snap_server, pos, scene_scale_m=SCENE_SCALE)

    traj_server.sleep_forever()


if __name__ == "__main__":
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    traj = result.trajectory
    pos = np.asarray(traj["position"]) * R_SCALE
    vel = np.asarray(traj["velocity"]) * R_SCALE
    m = np.asarray(traj["mass"]) * M_SCALE

    print("\n── Solution summary ─────────────────────────────────────────────")
    print(f"  Final position (m):   {pos[-1]}")
    print(f"  Final velocity (m/s): {vel[-1]}")
    print(f"  Final mass (kg):      {m[-1, 0]:.1f}  (dry mass: {M_DRY:.0f} kg)")
    print(f"  Fuel used (kg):       {M_WET - m[-1, 0]:.1f}")

    prepare_for_viser(result)
    launch_viser_servers(result)
