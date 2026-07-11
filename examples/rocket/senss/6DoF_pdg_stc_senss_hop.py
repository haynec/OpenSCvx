"""
6-DoF cSTC hop: ascent from launch pad → apex waypoint → nearby land pad.

Sibling of ``6DoF_pdg_stc_senss_gimble.py``. Same compound state-triggered
constraint (cSTC) encoding and vehicle model; boundary geometry is a hop:

    node 0       — launch at former touchdown pad (upright, small upward V)
    node K_APEX  — pass through former descent IC (position only; vel/att free)
    node N-1     — soft touchdown at a nearby land pad

Glideslope and LoS are enforced only relative to the **landing** pad, and only
on the descent node interval ``[K_APEX, N-1]``. Non-geometric altitude STCs
(gimbal, tilt, ω, speed) apply on both legs.

Compound state-triggered constraints
------------------------------------
    h < 100 m                       → tight gimbal, tilt
    h < 100 m & descent                → tight glideslope (land pad)
    h < 200 m & h>2 & descent          → LOS boresight toward land pad
    ||v|| < 35 m/s  AND  tilt < 60° → single-engine thrust limits
    ||v|| > 35 m/s  OR   tilt > 60° → three-engine thrust limits

When run as a script, launches four viser windows after solving:
  1. Animated trajectory – thrust plume, attitude frame, velocity-colored
     trail, DEM terrain patch (position adjustable in Viser GUI)
  2. Onboard sensor FPV – camera locked to the LOS gimbal sensor with
     matched DEM placement and an adjustable sensor FOV slider
  3. SCP convergence – node positions across iterations
  4. Snapshot grid – evenly-spaced body poses along the final path

Viser scene uses the same ENU frame as the model (x, y horizontal; z = altitude up).
"""

import os
import sys
import threading

import numpy as np
import trimesh
import viser
import viser.transforms as vtf
from PIL import Image

# File lives in examples/rocket/senss/ — three parents up is the repo root.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import openscvx as ox
from examples.plotting_viser import (
    AnimatedServerHandle,
    create_animated_plotting_server,
    create_scp_animated_plotting_server,
    create_snapshot_plotting_server,
)
from openscvx import Problem
from openscvx.plotting.viser import add_animation_controls, create_server
from openscvx.plotting.viser.animated import (
    _generate_viewcone_faces,
    _generate_viewcone_vertices,
    _sensor_pose_in_world,
)
from openscvx.plotting.viser.primitives import _generate_cone_mesh

# ── Physical parameters (notebook cell 16 / cell 33) ──────────────────────────
G0  = 9.806    # m/s²
ISP = 330.0    # s
M_WET = 100_000.0   # kg
M_DRY =  85_000.0   # kg

# Hop geometry: launch (former touchdown) → apex (former descent IC) → nearby land
R_I_LAUNCH = np.array([-10.0, -125.0, -260.0])   # m — start on pad
R_I_APEX   = np.array([200.0, -200.0, 250.0])    # m — mid-hop waypoint
R_I_LAND   = np.array([50.0, 200.0, -260.0])   # m — nearby land pad (same elevation)

# Launch IC: upright, small upward liftoff
Q_LAUNCH = np.array([0.0, 0.0, 0.0, 1.0])
V_LAUNCH = np.array([0.0, 0.0, 0.5])   # m/s
W_LAUNCH = np.zeros(3)

# Mid-hop attitude reference (optional guess only; not a hard BC)
Q_APEX = np.array([0.0, 0.0, 0.0, 1.0])

# Terminal conditions (soft touchdown at land pad)
V_I_FINAL = np.array([0.0, 0.0, -0.5])       # m/s
Q_FINAL   = np.array([0.0, 0.0, 0.0, 1.0])   # upright
W_FINAL   = np.zeros(3)

LOS_DEPTH_BELOW_TOUCHDOWN_M = 50.0  # m — LoS target sits below land pad (z only)
R_I_LOS_LAND = R_I_LAND - np.array([0.0, 0.0, LOS_DEPTH_BELOW_TOUCHDOWN_M])


def _height_above_pad(pos_m: np.ndarray) -> np.ndarray:
    """Height above the shared pad z-level (m); launch and land share elevation."""
    return np.asarray(pos_m)[:, 2] - R_I_LAUNCH[2]

# Thrust limits (N)
T_MAX     = 1_900_000.0 * 3.0        # 3-engine
T_MIN     = 1_900_000.0 * 0.4 * 3.0
T_MAX_AFT = 1_900_000.0              # 1-engine aft phase
T_MIN_AFT = 1_900_000.0 * 0.4

# Hard control angle limits (deg)
DELTA_ENGINE_MAX_DEG    = 60.0
DELTA_BORESIGHT_MAX_DEG = 0.0

# State constraint limits
THETA_MAX_DEG   = 10.0              # max tilt from vertical
W_B_MAX_RAD_S   = np.pi/2          # max angular rate
GS_MAX_DEG      = 90.0 - 35.0      # glideslope cone half-angle from horizontal = 55°

# cSTC trigger thresholds (notebook cell 16)
ALT_TRIGGER_H1_M = 100.0   # h < 100 m  → tight gimbal / tilt / glideslope
ALT_TRIGGER_H2_M = 200.0   # h < 200 m  → LOS boresight cone
ALT_DONE_M       = 2.0     # h > 2 m    → finalizer (disable cone near touchdown)
SPD_STC_TRIG     = 35.0    # m/s        → speed trigger for single/multi-engine
THETA_STC_TRIG   = 60.0    # deg        → tilt trigger for single/multi-engine

# Trigger alpha-tightening factors (notebook cell 16/33)
_ALPHA_ALT       = 1.1     # altitude trigger tightening
_ALPHA_ALT_LOS   = 1.1
_ALPHA_T_MIN     = 1.02

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

# ── STC penalty weights ───────────────────────────────────────────────────────
# Applied *inside* the CTCS argument; because CTCS squares the argument the
# effective penalty weight is W**2.  The relative magnitudes mirror the
# notebook's hand-tuned emphasis (notebook cell 33 ``w_stc_*``), with the
# geometric line-of-sight / glideslope and the engine-count selection getting
# the strongest pull (these are the hardest to enforce).  The outer
# constraint-violation weight (``lam_vc``, ramped by the AugmentedLagrangian
# autotuner) plays the role of the notebook's ``w_con_dyn``.
W_GIMBAL = 60.0
W_TILT   = 120.0
W_OMEGA  = 60.0
W_SPD    = 120.0
W_GS     = 250.0
W_LOS    = 400.0
W_THR    = 400.0

# ── Scaling ───────────────────────────────────────────────────────────────────
R_SCALE = max(
    np.linalg.norm(R_I_APEX),
    np.linalg.norm(R_I_LAUNCH),
    np.linalg.norm(R_I_LAND),
) / 2.5
M_SCALE = M_WET

_alpha_m_s = ALPHA_M * R_SCALE
_g_I_s     = G_I / R_SCALE
_r_cm_s    = R_CM / R_SCALE
_r_cp_s    = R_CP / R_SCALE
_J_B_s     = J_B_DIAG     / R_SCALE**2
_J_B_inv_s = J_B_INV_DIAG * R_SCALE**2
_rho_s     = RHO_AIR / (M_SCALE * R_SCALE) * R_SCALE**2   # = RHO_AIR * R_SCALE / M_SCALE

_T_max_s     = T_MAX     / (M_SCALE * R_SCALE)
_T_min_s     = T_MIN     / (M_SCALE * R_SCALE)
_T_max_aft_s = T_MAX_AFT / (M_SCALE * R_SCALE)
_T_min_aft_s = T_MIN_AFT / (M_SCALE * R_SCALE)

_r_launch_s = R_I_LAUNCH / R_SCALE
_r_apex_s   = R_I_APEX / R_SCALE
_r_land_s   = R_I_LAND / R_SCALE
_r_gs_land_s    = R_I_LAND / R_SCALE
_r_los_land_s   = R_I_LOS_LAND / R_SCALE
_v_launch_s = V_LAUNCH / R_SCALE
_v_final_s  = V_I_FINAL / R_SCALE
_m_wet_s   = M_WET / M_SCALE   # 1.0
_m_dry_s   = M_DRY / M_SCALE   # 0.85

_v_stc_s   = V_STC_CONS / R_SCALE

# ── Precomputed constraint thresholds ─────────────────────────────────────────
def _tilt_sq_bound(theta_deg: float) -> float:
    """Squared quaternion tilt bound: ||[q_x, q_y]||² ≤ (1-cos θ)/2."""
    return (1.0 - np.cos(np.pi/180 * theta_deg)) / 2.0


_tilt_sq_max    = _tilt_sq_bound(THETA_MAX_DEG)    # 0.5
_tilt_sq_stc    = _tilt_sq_bound(THETA_STC_DEG)    # tight tilt (≈4.6°)
_tilt_sq_trig   = _tilt_sq_bound(THETA_STC_TRIG)   # tilt trigger (60° → 0.25)
_tan_gs_max     = np.tan(np.pi/180 * GS_MAX_DEG)
_tan_gs_stc     = np.tan(np.pi/180 * GS_STC_DEG)
_cos_psi_stc    = np.cos(np.pi/180 * LOS_STC_DEG)
_delta_stc_rad  = np.pi/180 * DELTA_STC_DEG
_omega_sq_stc   = OMEGA_STC_RAD_S**2

# Scaled altitude / speed trigger thresholds
_alt_h1_s     = ALT_TRIGGER_H1_M / R_SCALE
_alt_h1a_s    = _ALPHA_ALT * ALT_TRIGGER_H1_M / R_SCALE
_alt_h2a_s    = _ALPHA_ALT_LOS * ALT_TRIGGER_H2_M / R_SCALE
_alt_done_s   = ALT_DONE_M / R_SCALE
_spd_trig_s   = SPD_STC_TRIG / R_SCALE

# ── Discretization ────────────────────────────────────────────────────────────
N = 60
K_APEX = N // 2   # 30 — ascent [0, K_APEX], descent [K_APEX, N-1]

# ── States ────────────────────────────────────────────────────────────────────
mass = ox.State("mass", shape=(1,))
mass.max = [_m_wet_s]
mass.min = [_m_dry_s]
mass.initial = [_m_wet_s]
mass.final   = [ox.Maximize(_m_dry_s)]   # fuel-optimal objective

position = ox.State("position", shape=(3,))
position.max = [ 2.5,  2.0,  2.5]
position.min = [-2.5, -2.0, -2.0]   # z ≥ pad level (scaled)
position.initial = [float(_r_launch_s[0]), float(_r_launch_s[1]), float(_r_launch_s[2])]
position.final   = [float(_r_land_s[0]), float(_r_land_s[1]), float(_r_land_s[2])]

velocity = ox.State("velocity", shape=(3,))
# _v_box = 150.0 / R_SCALE
velocity.max = [ 2.0,  2.0,  2.0]
velocity.min = [-2.0, -2.0, -2.0]
velocity.initial = [float(_v_launch_s[0]), float(_v_launch_s[1]), float(_v_launch_s[2])]
velocity.final   = [float(_v_final_s[0]), float(_v_final_s[1]), float(_v_final_s[2])]

attitude = ox.State("attitude", shape=(4,))
attitude.max = [ 1.0,  1.0,  1.0,  1.0]
attitude.min = [-1.0, -1.0, -1.0, -1.0]
attitude.initial = list(Q_LAUNCH)
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
los_az.max   = [ np.radians(DELTA_BORESIGHT_MAX_DEG)]
los_az.min   = [-np.radians(DELTA_BORESIGHT_MAX_DEG)]
los_az.guess = np.zeros((N, 1))

# Piecewise-linear position guess: launch → apex → land
_tau_asc = np.linspace(0.0, 1.0, K_APEX + 1)
_tau_des = np.linspace(0.0, 1.0, N - K_APEX)
_pos_guess = np.zeros((N, 3))
_pos_guess[: K_APEX + 1] = (
    (1.0 - _tau_asc)[:, None] * _r_launch_s + _tau_asc[:, None] * _r_apex_s
)
_pos_guess[K_APEX:] = (
    (1.0 - _tau_des)[:, None] * _r_apex_s + _tau_des[:, None] * _r_land_s
)
position.guess = _pos_guess

# ── State & Control ───────────────────────────────────────────────────────────
states   = [mass, position, velocity, attitude, angular_velocity]
controls = [thrust_mag, gimbal_elev, gimbal_az, los_elev, los_az]

# ── Quaternion kinematics ─────────────────────────────────────────────────────
# OpenSCvx convention: attitude = [q_x, q_y, q_z, q_w]
q1, q2, q3, q4 = attitude[0], attitude[1], attitude[2], attitude[3]

# CBI: inertial→body DCM  (CBI.T is body→inertial)
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

torque = cross(r_cm_ox, T_B) #+ cross(r_cp_ox, A_B)
gyro   = cross(angular_velocity, J_B_ox @ angular_velocity)

# ── Dynamics (all in scaled units) ────────────────────────────────────────────
dynamics = {
    "mass":             -_alpha_m_s * T,
    "position":          velocity,
    # "velocity":          CBI.T @ (T_B + A_B) / mass[0] + g_I_ox,
    "velocity":          CBI.T @ (T_B) / mass[0] + g_I_ox,
    "attitude":          attitude_dot,
    "angular_velocity":  J_B_inv_ox @ (torque / mass[0] - gyro),
}

# ── Shared sub-expressions for constraints ────────────────────────────────────
# Tilt: ||[q_x, q_y]||²  (matches notebook's ||[x[8], x[9]]||² in [w,x,y,z] ordering)
tilt_sq   = attitude[0]**2 + attitude[1]**2
speed     = ox.linalg.Norm(velocity)
omega_sq  = ox.linalg.Norm(angular_velocity)**2
z_alt     = position[2] - float(_r_land_s[2])   # height above shared pad z

# Position relative to land glideslope apex (landing pad only)
pos_rel_x = position[0] - float(_r_gs_land_s[0])
pos_rel_y = position[1] - float(_r_gs_land_s[1])
pos_rel_z = position[2] - float(_r_gs_land_s[2])
r_xy_land = ox.linalg.Norm(ox.Concat(pos_rel_x, pos_rel_y))

# Position relative to land LoS target (below land pad)
pos_los_x = position[0] - float(_r_los_land_s[0])
pos_los_y = position[1] - float(_r_los_land_s[1])
pos_los_z = position[2] - float(_r_los_land_s[2])
r_los_land = ox.linalg.Norm(ox.Concat(pos_los_x, pos_los_y, pos_los_z))

# LOS boresight in inertial frame via body→inertial rotation
db = los_elev[0]
pb = los_az[0]
los_B = ox.Concat(
    ox.Sin(db) * ox.Cos(pb),
    ox.Sin(db) * ox.Sin(pb),
    ox.Cos(db),
)
los_I = CBI.T @ los_B   # unit-norm boresight in inertial frame
r_dot_los = (pos_los_x * los_I[0]
             + pos_los_y * los_I[1]
             + pos_los_z * los_I[2])


# ── State-triggered-constraint helper ─────────────────────────────────────────
def relu(expr):
    """Smooth-free positive part (max(expr, 0)); same primitive CTCS uses."""
    return ox.Max(expr, 0)


def stc(*factors, weight: float = 1.0, penalty="huber", over=None):
    """Compound state-triggered constraint as a single CTCS term.

    ``factors`` are the relu(trigger)/relu(constraint) expressions whose product
    forms the penalty.  Optional ``over=(k0, k1)`` restricts land-pad geometry
    to the descent node interval so ascent is not pulled into the land cone.
    """
    prod = factors[0]
    for f in factors[1:]:
        prod = prod * f
    c = ox.ctcs((weight * prod) <= 0, penalty=penalty)
    if over is not None:
        c = c.over(over)
    return c


# ── Trigger indicators (relu, > 0 when active) ────────────────────────────────
T_alt100 = relu(_alt_h1_s  - z_alt)            # h < 100 m
T_alt110 = relu(_alt_h1a_s - z_alt)            # h < 110 m (alpha-tightened)
T_alt220 = relu(_alt_h2a_s - z_alt)            # h < 220 m (LOS, alpha)
T_altgt2 = relu(z_alt - _alt_done_s)           # h >   2 m (finalizer)
T_vlt35  = relu(_spd_trig_s - speed)           # ||v|| < 35 m/s
T_vgt35  = relu(speed - _spd_trig_s)           # ||v|| > 35 m/s
T_tlt60  = relu(_tilt_sq_trig - tilt_sq)       # tilt < 60°
T_tgt60  = relu(tilt_sq - _tilt_sq_trig)       # tilt > 60°

# ── Constraint violations (relu, > 0 when violated) ───────────────────────────
C_gimbal_p = relu( de - _delta_stc_rad)                 # δ_e ≤ +1°
C_gimbal_n = relu(-de - _delta_stc_rad)                 # δ_e ≥ −1°
C_gs_land  = relu(r_xy_land * _tan_gs_stc - pos_rel_z)  # tight GS (land pad)
C_omega    = relu(omega_sq - _omega_sq_stc)             # tight angular rate
C_tilt     = relu(tilt_sq - _tilt_sq_stc)               # tight tilt
C_spd      = relu(speed - _v_stc_s)                     # tight speed
C_los_land = relu(r_los_land * _cos_psi_stc - r_dot_los)  # LOS toward land
C_Tmin_f   = relu(_T_min_aft_s - T)                     # single-engine min
C_Tmax_f   = relu(T - _T_max_aft_s)                     # single-engine max
C_Tmin_i   = relu(_ALPHA_T_MIN * _T_min_s - T)          # three-engine min
C_Tmax_i   = relu(T - _T_max_s)                         # three-engine max

# ── Constraints ───────────────────────────────────────────────────────────────
constraints = []

# Boundary conditions (convex equality constraints)
constraints.append((position         == _r_launch_s).convex().at([0]))
constraints.append((attitude         == Q_LAUNCH).convex().at([0]))
constraints.append((velocity         == _v_launch_s).convex().at([0]))
constraints.append((angular_velocity == W_LAUNCH).convex().at([0]))
constraints.append((position[2]         == _r_apex_s[2]).convex().at([K_APEX]))  # position only
constraints.append((position         == _r_land_s).convex().at([N - 1]))
constraints.append((velocity         == _v_final_s).convex().at([N - 1]))
constraints.append((attitude         == Q_FINAL).convex().at([N - 1]))
constraints.append((angular_velocity == W_FINAL).convex().at([N - 1]))

# ── Always-on CTCS ────────────────────────────────────────────────────────────
constraints.append(ox.ctcs(tilt_sq - _tilt_sq_max <= 0, idx=1))          # tilt ≤ 90°
constraints.append(ox.ctcs(omega_sq - W_B_MAX_RAD_S**2 <= 0, penalty="huber", idx=0))
constraints.append(ox.ctcs(_m_dry_s - mass[0] <= 0, penalty="huber", idx=0))
# Wide land glideslope on descent nodes only (ascent is unconstrained by GS/LoS)
constraints.append(
    ox.ctcs(r_xy_land * _tan_gs_max - pos_rel_z <= 0, penalty="huber")
    .over((K_APEX, N - 1))
)

# ── Non-geometric altitude STCs (both legs) ───────────────────────────────────
constraints.append(stc(T_alt100, C_gimbal_p, weight=W_GIMBAL))
constraints.append(stc(T_alt100, C_gimbal_n, weight=W_GIMBAL))
constraints.append(stc(T_alt100, C_tilt, weight=W_TILT))
constraints.append(stc(T_alt100, C_omega, weight=W_OMEGA))
constraints.append(stc(T_alt100, C_spd,   weight=W_SPD))

# ── Land-pad geometric STCs (descent phase only) ──────────────────────────────
constraints.append(stc(T_alt100, T_altgt2, C_gs_land, weight=W_GS, over=(K_APEX, N - 1)))
constraints.append(stc(T_alt220, T_altgt2, C_los_land, weight=W_LOS, over=(K_APEX, N - 1)))

# ||v|| < 35 m/s AND tilt < 60° → single-engine thrust limits
constraints.append(stc(T_vlt35, T_tlt60, C_Tmax_f, weight=W_THR))
constraints.append(stc(T_vlt35, T_tlt60, C_Tmin_f, weight=W_THR))
# ||v|| > 35 m/s OR tilt > 60° → three-engine thrust limits (OR = sum of products)
constraints.append(stc(T_vgt35, C_Tmax_i, weight=W_THR))
constraints.append(stc(T_vgt35, C_Tmin_i, weight=W_THR))
constraints.append(stc(T_tgt60, C_Tmax_i, weight=W_THR))
constraints.append(stc(T_tgt60, C_Tmin_i, weight=W_THR))

# ── Time (free final time with per-segment dilation) ─────────────────────────
_t_f_guess = 42.0   # 2 × descent-only guess

# Velocity from finite differences of the position guess (avoids ||v||=0 nodes,
# which NaN the CTCS Jacobian through ox.linalg.Norm(velocity)).
_dt_guess = _t_f_guess / (N - 1)
_vel_guess = np.gradient(_pos_guess, _dt_guess, axis=0)
_speed = np.linalg.norm(_vel_guess, axis=1)
_vel_guess[_speed < 1e-6] = np.array([0.0, 0.0, 1e-3])
velocity.guess = _vel_guess

time = ox.Time(
    initial=0.0,
    final=ox.Free(_t_f_guess),
    min=0.0,
    max=1.5 * _t_f_guess,   # 63.0 — 2 × descent-only max
    # uniform_time_grid=True,
)

# ── Problem Assembly ──────────────────────────────────────────────────────────
problem = Problem(
    N=N,
    states=states,
    controls=controls,
    dynamics=dynamics,
    constraints=constraints,
    time=time,
    float_dtype="float64",
    licq_max = 1e-6,
    algorithm={
        # "lam_vc": 4e0,
        "lam_prox": 4e-1,
        "lam_vc": 4e0,
        "lam_cost": 1e-6,

        "k_max": 800,
        "autotuner": ox.ConstantProximalWeight(),
        # "autotuner": ox.AdaptiveProximalWeight(),
    },
    discretizer={
        "diffrax_kwargs": {"atol": 1e-6, "rtol": 1e-6},
        # "ode_solver": "Dopri8",
    },
    # solver={
    #     "solver_args": {"abs_tol": 1e-10, "rel_tol": 1e-10},
    # }
)

problem.settings.prp.dt = 0.001

# Prop tolerances
problem.settings.prp.atol = 1e-12
problem.settings.prp.rtol = 1e-12

# ── Viser display parameters ──────────────────────────────────────────────────
# cSTC uses ENU inertial frame (x, y horizontal; z = altitude). Viser is Z-up,
# so positions map directly without any axis permutation.
SCENE_SCALE = 10.0          # 1 viser unit = 10 m
PLUME_SCALE = 8.0
ATTITUDE_AXES_LENGTH = 2.0
VIEWCONE_SCALE = 4.0
CAMERA_FOV_DEG = 76.0       # vertical field of view (adjust in Viser GUI)
SENSOR_FPV_ROLL_DEG = 0.0   # fixed roll about boresight in sensor FPV window
_VISER_UP_AXIS = (0.0, 0.0, 1.0)
GS_HALFANGLE_DEG = 90.0 - GS_MAX_DEG
GS_STC_HALFANGLE_DEG = 90.0 - GS_STC_DEG
_POSITION_STATE_SLICE = slice(1, 4)

# ── DEM visualization (viser only; does not affect optimization) ──────────────
TERRAIN_HALF_EXTENT_M: float = 400.0   # m — base patch half-width (× Scale X/Y)
DEM_BASE_RELIEF_M: float = 150.0       # m — base relief peak-to-peak at Scale Z = 1
DEM_POS_X_M: float = 39.0              # m — patch center (adjust in Viser GUI)
DEM_POS_Y_M: float = -175.0
DEM_POS_Z_M: float = -244.0            # m — elevation at DEM center pixel
DEM_SCALE_X: float = 0.5
DEM_SCALE_Y: float = 0.5
DEM_SCALE_Z: float = 0.65
DEM_YAW_DEG: float = 180.0            # ° — rotation about +z through patch center
DEM_MIRROR_X: bool = True            # flip patch across local y-axis
DEM_MIRROR_Y: bool = False           # flip patch across local x-axis
DEM_GRID: int = 3938                 # downsampled DEM resolution
_DEM_PATH = os.path.join(os.path.dirname(__file__), "senns_dem.png")


def _load_dem_normalized() -> np.ndarray:
    img = Image.open(_DEM_PATH)
    raw = np.array(img, dtype=np.uint16)
    lo, hi = float(raw.min()), float(raw.max())
    arr = np.array(img.resize((DEM_GRID, DEM_GRID), Image.BILINEAR), dtype=np.float32)
    return (arr - lo) / max(hi - lo, 1.0)


_dem_norm: np.ndarray = _load_dem_normalized()
_dem_center_i = _dem_center_j = (DEM_GRID - 1) // 2
_dem_center_norm: float = float(_dem_norm[_dem_center_i, _dem_center_j])


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
    """Draw horizontal surfaces where altitude-based cSTC phases activate.

    Triggers are height above the shared pad z-level (launch and land match).
    Discs are centered on the midpoint between the two pads.
    """
    mid = 0.5 * (R_I_LAUNCH + R_I_LAND)
    cx, cy = mid[0] / scene_scale_m, mid[1] / scene_scale_m
    pad_z = R_I_LAUNCH[2] / scene_scale_m
    xy_extent = float(np.max(np.linalg.norm(pos[:, :2] - np.array([cx, cy]), axis=1)))
    radius = max(xy_extent * 1.25, 2.0)

    triggers = [
        (ALT_TRIGGER_H2_M, (80, 160, 255), "h < 200 m → LOS boresight cone"),
        (ALT_TRIGGER_H1_M, (255, 80, 80),  "h < 100 m → tight gimbal/tilt/ω/speed/GS"),
    ]
    for alt_m, color, description in triggers:
        z = (R_I_LAUNCH[2] + alt_m) / scene_scale_m
        center = np.array([cx, cy, z], dtype=np.float32)
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
            text=description,
            position=(cx + radius * 0.85, cy, z + 0.05),
        )

        server.scene.add_line_segments(
            f"/cstc_triggers/alt_{int(alt_m)}_stem",
            points=np.array(
                [[[cx - radius * 0.95, cy, pad_z], [cx - radius * 0.95, cy, z]]],
                dtype=np.float32,
            ),
            colors=tuple(int(c * 0.7) for c in color),
            line_width=1.5,
        )


# ── DEM terrain mesh (viser display units: real meters / SCENE_SCALE) ────────
GREY_BASE = np.array([148, 150, 152], dtype=np.float32) / 255.0
_K_AMBIENT: float = 0.00
_K_PRIMARY: float = 2.25
_LIGHT_AZ_DEG: float = 128.0
_LIGHT_EL_DEG: float = 10.5


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


def _make_terrain_vertices(
    origin_m: tuple[float, float, float],
    scale_xyz: tuple[float, float, float],
    yaw_deg: float = 0.0,
    mirror_xy: tuple[bool, bool] = (False, False),
) -> np.ndarray:
    """Terrain vertices in viser display units.

    ``origin_m`` is the patch center (x, y) and the elevation at the DEM center
    pixel (z), in meters.  ``scale_xyz`` stretches the patch in each axis.
    ``yaw_deg`` rotates the patch about +z through the patch center.
    ``mirror_xy`` flips the local sampling direction in x and/or y (applied
    before the yaw rotation).
    """
    ox, oy, oz = origin_m
    sx, sy, sz = scale_xyz
    mx = -1.0 if mirror_xy[0] else 1.0
    my = -1.0 if mirror_xy[1] else 1.0
    N = DEM_GRID
    half_x_m = TERRAIN_HALF_EXTENT_M * float(sx)
    half_y_m = TERRAIN_HALF_EXTENT_M * float(sy)
    x_loc = mx * np.linspace(-half_x_m, half_x_m, N, dtype=np.float32)
    y_loc = my * np.linspace(-half_y_m, half_y_m, N, dtype=np.float32)
    XX_loc, YY_loc = np.meshgrid(x_loc, y_loc, indexing="xy")

    psi = np.radians(float(yaw_deg))
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    XX_m = cos_p * XX_loc - sin_p * YY_loc + float(ox)
    YY_m = sin_p * XX_loc + cos_p * YY_loc + float(oy)

    relief_m = (_dem_norm - _dem_center_norm) * DEM_BASE_RELIEF_M * float(sz)
    ZZ_m = float(oz) + relief_m

    XX = (XX_m / SCENE_SCALE).astype(np.float32)
    YY = (YY_m / SCENE_SCALE).astype(np.float32)
    ZZ = (ZZ_m / SCENE_SCALE).astype(np.float32)
    return np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)


def _compute_vertex_normals(
    scale_xyz: tuple[float, float, float],
    yaw_deg: float = 0.0,
    mirror_xy: tuple[bool, bool] = (False, False),
) -> np.ndarray:
    sx, sy, sz = scale_xyz
    mx = -1.0 if mirror_xy[0] else 1.0
    my = -1.0 if mirror_xy[1] else 1.0
    N = DEM_GRID
    half_x_m = TERRAIN_HALF_EXTENT_M * float(sx)
    half_y_m = TERRAIN_HALF_EXTENT_M * float(sy)
    cell_x = 2.0 * half_x_m / (N - 1)
    cell_y = 2.0 * half_y_m / (N - 1)
    ZZ = (_dem_norm - _dem_center_norm) * DEM_BASE_RELIEF_M * float(sz)
    dz_dxi = np.gradient(ZZ, cell_x, axis=1).astype(np.float32)
    dz_dyj = np.gradient(ZZ, cell_y, axis=0).astype(np.float32)
    nx = -mx * dz_dxi.ravel()
    ny = -my * dz_dyj.ravel()
    nz = np.ones(N * N, dtype=np.float32)

    psi = np.radians(float(yaw_deg))
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    nx_w = cos_p * nx - sin_p * ny
    ny_w = sin_p * nx + cos_p * ny
    normals = np.stack([nx_w, ny_w, nz], axis=-1)
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


def _build_trimesh(
    origin_m: tuple[float, float, float],
    scale_xyz: tuple[float, float, float],
    yaw_deg: float,
    mirror_xy: tuple[bool, bool],
    normals: np.ndarray,
    k_amb: float,
    k_pri: float,
    az: float,
    el: float,
    on: bool,
) -> trimesh.Trimesh:
    # A single mirror (odd number of flipped axes) reverses triangle winding,
    # which flips the geometric surface orientation and makes the renderer light
    # the terrain from below.  Reverse winding to keep faces pointing up.
    faces = _terrain_faces[:, ::-1] if (mirror_xy[0] ^ mirror_xy[1]) else _terrain_faces
    return trimesh.Trimesh(
        vertices=_make_terrain_vertices(origin_m, scale_xyz, yaw_deg, mirror_xy),
        faces=faces,
        vertex_colors=_bake_colors(normals, k_amb, k_pri, az, el, on),
        process=False,
    )


def _add_dem_to_server(
    server: viser.ViserServer,
    *,
    fov_slider: bool = True,
    fov_folder_name: str = "Camera",
    fov_initial_deg: float = CAMERA_FOV_DEG,
    default_k_ambient: float = _K_AMBIENT,
) -> viser.GuiInputHandle | None:
    """Overlay DEM terrain and lighting/elevation GUI onto an existing server.

    Returns the FOV slider handle when ``fov_slider`` is True, else None.
    """
    server.scene.configure_default_lights(enabled=False)
    server.scene.add_light_ambient("/lights/ambient", color=(255, 255, 255), intensity=1.0)

    _st: dict = {
        "origin": (DEM_POS_X_M, DEM_POS_Y_M, DEM_POS_Z_M),
        "scale": (DEM_SCALE_X, DEM_SCALE_Y, DEM_SCALE_Z),
        "yaw_deg": DEM_YAW_DEG,
        "mirror": (DEM_MIRROR_X, DEM_MIRROR_Y),
        "k_amb": default_k_ambient,
        "k_pri": _K_PRIMARY,
        "az": _LIGHT_AZ_DEG,
        "el": _LIGHT_EL_DEG,
        "on": True,
        "normals": _compute_vertex_normals(
            (DEM_SCALE_X, DEM_SCALE_Y, DEM_SCALE_Z), DEM_YAW_DEG,
            (DEM_MIRROR_X, DEM_MIRROR_Y),
        ),
        "_lock": threading.Lock(),
    }

    def _refresh() -> None:
        with _st["_lock"]:
            mesh = _build_trimesh(
                _st["origin"], _st["scale"], _st["yaw_deg"], _st["mirror"], _st["normals"],
                _st["k_amb"], _st["k_pri"], _st["az"], _st["el"], _st["on"],
            )
        server.scene.add_mesh_trimesh("/terrain", mesh)

    _refresh()

    with server.gui.add_folder("DEM Terrain"):
        pos_x_sl = server.gui.add_slider(
            "Position X (m)", min=-1000.0, max=1000.0, step=1.0, initial_value=DEM_POS_X_M
        )
        pos_y_sl = server.gui.add_slider(
            "Position Y (m)", min=-1000.0, max=1000.0, step=1.0, initial_value=DEM_POS_Y_M
        )
        pos_z_sl = server.gui.add_slider(
            "Position Z (m)", min=-1000.0, max=1000.0, step=1.0, initial_value=DEM_POS_Z_M
        )
        scale_x_sl = server.gui.add_slider(
            "Scale X", min=0.1, max=10.0, step=0.05, initial_value=DEM_SCALE_X
        )
        scale_y_sl = server.gui.add_slider(
            "Scale Y", min=0.1, max=10.0, step=0.05, initial_value=DEM_SCALE_Y
        )
        scale_z_sl = server.gui.add_slider(
            "Scale Z", min=0.0, max=10.0, step=0.05, initial_value=DEM_SCALE_Z
        )
        yaw_sl = server.gui.add_slider(
            "Yaw Z (°)", min=0.0, max=360.0, step=1.0, initial_value=DEM_YAW_DEG
        )
        mirror_x_cb = server.gui.add_checkbox("Mirror X", initial_value=DEM_MIRROR_X)
        mirror_y_cb = server.gui.add_checkbox("Mirror Y", initial_value=DEM_MIRROR_Y)

        def _sync_terrain(_e=None) -> None:
            _st["origin"] = (float(pos_x_sl.value), float(pos_y_sl.value), float(pos_z_sl.value))
            _st["scale"] = (float(scale_x_sl.value), float(scale_y_sl.value), float(scale_z_sl.value))
            _st["yaw_deg"] = float(yaw_sl.value)
            _st["mirror"] = (bool(mirror_x_cb.value), bool(mirror_y_cb.value))
            _st["normals"] = _compute_vertex_normals(_st["scale"], _st["yaw_deg"], _st["mirror"])
            _refresh()

        for _ctrl in (
            pos_x_sl, pos_y_sl, pos_z_sl, scale_x_sl, scale_y_sl, scale_z_sl,
            yaw_sl, mirror_x_cb, mirror_y_cb,
        ):
            _ctrl.on_update(_sync_terrain)

    with server.gui.add_folder("DEM Lighting"):
        server.gui.add_markdown("_Baked into DEM vertex colours; other scene objects unaffected._")
        p_on  = server.gui.add_checkbox("Enabled", initial_value=True)
        p_az  = server.gui.add_slider("Azimuth (°)",   min=0.0,  max=360.0, step=1.0,   initial_value=_LIGHT_AZ_DEG)
        p_el  = server.gui.add_slider("Elevation (°)", min=0.5,  max=89.0,  step=0.5,   initial_value=_LIGHT_EL_DEG)
        p_str = server.gui.add_slider("Strength",      min=0.0,  max=5.0,   step=0.05,  initial_value=_K_PRIMARY)
        amb_sl = server.gui.add_slider("Ambient",      min=0.0,  max=0.5,   step=0.005, initial_value=default_k_ambient)

        def _sync_light(_e=None) -> None:
            _st.update(
                on=bool(p_on.value), az=float(p_az.value), el=float(p_el.value),
                k_pri=float(p_str.value), k_amb=float(amb_sl.value),
            )
            _refresh()

        for _ctrl in (p_on, p_az, p_el, p_str, amb_sl):
            _ctrl.on_update(_sync_light)

    fov_sl: viser.GuiInputHandle | None = None
    if fov_slider:
        with server.gui.add_folder(fov_folder_name):
            fov_sl = server.gui.add_slider(
                "FOV (°)", min=5.0, max=120.0, step=1.0, initial_value=fov_initial_deg
            )

            def _apply_fov(client: viser.ClientHandle) -> None:
                client.camera.fov = float(np.radians(fov_sl.value))

            @fov_sl.on_update
            def _(_e=None) -> None:
                for client in server.get_clients().values():
                    _apply_fov(client)

            @server.on_client_connect
            def _(client: viser.ClientHandle) -> None:
                _apply_fov(client)

    with server.gui.add_folder("Info"):
        server.gui.add_markdown(
            f"**Hop pads**  \n"
            f"Launch: ({R_I_LAUNCH[0]:.1f}, {R_I_LAUNCH[1]:.1f}, {R_I_LAUNCH[2]:.1f}) m  \n"
            f"Apex: ({R_I_APEX[0]:.1f}, {R_I_APEX[1]:.1f}, {R_I_APEX[2]:.1f}) m  \n"
            f"Land: ({R_I_LAND[0]:.1f}, {R_I_LAND[1]:.1f}, {R_I_LAND[2]:.1f}) m  \n\n"
            f"**DEM patch center** — use Position sliders above  \n"
            f"DEM: {DEM_GRID}×{DEM_GRID} · {_terrain_faces.shape[0]:,} tris"
        )

    return fov_sl


def _los_sensor_fpv_pose(
    position: np.ndarray,
    attitude_wxyz: np.ndarray,
    R_sb: np.ndarray,
    *,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """FPV pose matching the flipped LOS viewcone (forward = sensor -Z).

    Roll is locked to the sensor frame (no twist from a world-up look-at). An
    optional ``roll_deg`` rotates the image plane about the boresight.
    """
    w, x, y, z = attitude_wxyz
    R_bw = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    R_sensor_to_world = R_bw @ R_sb.T
    forward = -R_sensor_to_world[:, 2]
    forward /= np.linalg.norm(forward) + 1e-12

    # OpenCV camera (+X right, +Y down, +Z forward), aligned with sensor axes.
    right = -R_sensor_to_world[:, 0]
    right -= forward * np.dot(right, forward)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)
    down /= np.linalg.norm(down) + 1e-12

    phi = np.radians(float(roll_deg))
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    right_r = cos_p * right + sin_p * down
    down_r = -sin_p * right + cos_p * down

    R_world_cam = np.stack([right_r, down_r, forward], axis=1)
    wxyz = vtf.SO3.from_matrix(R_world_cam).wxyz
    cam_pos = np.asarray(position, dtype=np.float64)
    return cam_pos, wxyz


def create_cstc_sensor_fpv_server(
    result,
    traj_time: np.ndarray,
) -> viser.ViserServer:
    """Second viser window: onboard LOS sensor first-person view with matched DEM."""
    traj = result.trajectory
    pos = np.asarray(traj["position"], dtype=np.float64)
    attitude = np.asarray(traj["attitude"], dtype=np.float64)
    los_elev = np.asarray(traj["los_elev"], dtype=np.float64).flatten()
    los_az = np.asarray(traj["los_az"], dtype=np.float64).flatten()
    R_sb_series = [_los_body_to_sensor(de, pe) for de, pe in zip(los_elev, los_az)]

    server = create_server(pos, dark_mode=True, show_grid=False)
    fov_sl = _add_dem_to_server(
        server,
        fov_folder_name="Sensor",
        fov_initial_deg=CAMERA_FOV_DEG,
        default_k_ambient=0.45,
    )

    los_land_vis = tuple((R_I_LOS_LAND / SCENE_SCALE).tolist())
    launch_vis = tuple((R_I_LAUNCH / SCENE_SCALE).tolist())
    land_vis = tuple((R_I_LAND / SCENE_SCALE).tolist())
    server.scene.add_icosphere(
        "/los_target_land",
        radius=0.08,
        color=(255, 220, 80),
        position=los_land_vis,
    )
    server.scene.add_icosphere(
        "/launch_pad",
        radius=0.12,
        color=(80, 180, 255),
        position=launch_vis,
    )
    server.scene.add_icosphere(
        "/landing_pad",
        radius=0.12,
        color=(50, 255, 80),
        position=land_vis,
    )

    frame_state = {"idx": 0, "roll_deg": float(SENSOR_FPV_ROLL_DEG)}

    def _apply_sensor_camera(client: viser.ClientHandle, frame_idx: int) -> None:
        cam_pos, cam_wxyz = _los_sensor_fpv_pose(
            pos[frame_idx],
            attitude[frame_idx],
            R_sb_series[frame_idx],
            roll_deg=frame_state["roll_deg"],
        )
        client.camera.position = tuple(float(x) for x in cam_pos)
        client.camera.wxyz = tuple(float(x) for x in cam_wxyz)
        if fov_sl is not None:
            client.camera.fov = float(np.radians(fov_sl.value))

    with server.gui.add_folder("Sensor FPV"):
        roll_sl = server.gui.add_slider(
            "Roll about boresight (°)",
            min=-180.0,
            max=180.0,
            step=1.0,
            initial_value=SENSOR_FPV_ROLL_DEG,
        )
        server.gui.add_markdown(
            "_Camera forward matches the blue LOS viewcone (sensor -Z). "
            "Roll is fixed to the sensor frame; use the slider above to rotate "
            "the image about the boresight. **Sensor → FOV** sets field of view._"
        )

        @roll_sl.on_update
        def _(_e=None) -> None:
            frame_state["roll_deg"] = float(roll_sl.value)
            for client in server.get_clients().values():
                _apply_sensor_camera(client, frame_state["idx"])

    def update_sensor_camera(frame_idx: int) -> None:
        frame_state["idx"] = int(frame_idx)
        for client in server.get_clients().values():
            _apply_sensor_camera(client, frame_state["idx"])

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        _apply_sensor_camera(client, frame_state["idx"])

    add_animation_controls(server, traj_time, [update_sensor_camera], loop=True)

    return server


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
    """Body-to-sensor DCM for los_elev/los_az spherical gimbal (neutral = identity)."""
    los_b = np.array(
        [np.sin(de) * np.cos(pe), np.sin(de) * np.sin(pe), np.cos(de)],
        dtype=np.float64,
    )
    los_b = los_b / (np.linalg.norm(los_b) + 1e-12)
    az_axis = np.array([0.0, 0.0, 1.0])
    x_raw = np.cross(az_axis, los_b)
    x_norm = np.linalg.norm(x_raw)
    if x_norm < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x_raw / x_norm
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
    base_vertices = base_vertices.copy()
    base_vertices[1:, 2] *= -1.0
    n_base = len(base_vertices) - 1
    faces = _generate_viewcone_faces(n_base)[:, ::-1]
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


def _first_node_index(mask: np.ndarray) -> int | None:
    """Index of the first True entry, or None if the trigger never activates."""
    if not mask.any():
        return None
    return int(np.argmax(mask))


def _node_trigger_indices(nodes) -> tuple[int | None, int | None, int | None]:
    """Node indices where altitude / thrust STC triggers first activate."""
    pos_m = np.asarray(nodes["position"]) * R_SCALE
    vel_ms = np.asarray(nodes["velocity"]) * R_SCALE
    q = np.asarray(nodes["attitude"])
    h_pad = _height_above_pad(pos_m)
    spd = np.linalg.norm(vel_ms, axis=1)
    tilt_deg = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 0] ** 2 + q[:, 1] ** 2), -1.0, 1.0)))

    k_h1 = _first_node_index(h_pad < ALT_TRIGGER_H1_M)
    k_h2 = _first_node_index(h_pad < ALT_TRIGGER_H2_M)
    k_aft = _first_node_index((spd < SPD_STC_TRIG) & (tilt_deg < THETA_STC_TRIG))
    return k_h1, k_h2, k_aft


def launch_viser_servers(result) -> None:
    """Create trajectory, sensor FPV, SCP convergence, and snapshot viser servers."""
    pos = np.asarray(result.trajectory["position"])
    initial_alt_vis = float(np.max(pos[:, 2])) * 1.15

    k_h1, k_h2, k_aft = _node_trigger_indices(result.nodes)

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
        show_grid=False,   # DEM terrain replaces the flat ground grid
        dark_mode=True,
        scene_scale=1.0,
        controls="manual",
    )
    assert isinstance(handle, AnimatedServerHandle)
    _, update_viewcone = add_cstc_los_viewcone(handle.server, result)
    callbacks = handle.update_callbacks + [update_viewcone]
    add_animation_controls(handle.server, handle.traj_time, callbacks, loop=True)
    traj_server = handle.server

    _srv: viser.ViserServer = getattr(traj_server, "server", traj_server)  # type: ignore[assignment]
    _add_dem_to_server(_srv)

    def _add_gs_cone(name: str, apex, height: float, angle_deg: float, color, opacity: float) -> None:
        verts, faces = _generate_cone_mesh(
            np.asarray(apex, dtype=np.float32),
            height,
            angle_deg,
            32,
            axis=_VISER_UP_AXIS,
        )
        traj_server.scene.add_mesh_simple(
            name,
            vertices=verts,
            faces=faces,
            color=color,
            opacity=opacity,
        )

    gs_land_vis = tuple((R_I_LAND / SCENE_SCALE).tolist())
    los_land_vis = tuple((R_I_LOS_LAND / SCENE_SCALE).tolist())
    launch_vis = tuple((R_I_LAUNCH / SCENE_SCALE).tolist())
    apex_vis = tuple((R_I_APEX / SCENE_SCALE).tolist())
    _add_gs_cone(
        "/constraints/glideslope_wide_land",
        gs_land_vis,
        initial_alt_vis,
        GS_HALFANGLE_DEG,
        (80, 200, 80),
        0.10,
    )
    _add_gs_cone(
        "/constraints/glideslope_tight_land",
        gs_land_vis,
        ALT_TRIGGER_H1_M / SCENE_SCALE,
        GS_STC_HALFANGLE_DEG,
        (255, 180, 40),
        0.14,
    )

    add_cstc_altitude_triggers(traj_server, pos, scene_scale_m=SCENE_SCALE)

    traj_server.scene.add_icosphere(
        "/los_target_land",
        radius=0.08,
        color=(255, 220, 80),
        position=los_land_vis,
    )
    traj_server.scene.add_icosphere(
        "/launch_pad",
        radius=0.12,
        color=(80, 180, 255),
        position=launch_vis,
    )
    traj_server.scene.add_icosphere(
        "/apex_waypoint",
        radius=0.14,
        color=(255, 140, 40),
        position=apex_vis,
    )
    traj_server.scene.add_icosphere(
        "/landing_pad",
        radius=0.12,
        color=(50, 255, 80),
        position=gs_land_vis,
    )

    for k, color in [
        (k_aft, (255, 210, 50)),
        (k_h2, (80, 160, 255)),
        (k_h1, (255, 80, 80)),
    ]:
        if k is None:
            continue
        k = int(np.clip(k, 0, len(pos) - 1))
        traj_server.scene.add_icosphere(
            f"/phase_markers/k{k}",
            radius=0.14,
            color=color,
            position=tuple(float(v) for v in pos[k]),
        )

    def _node_label(k: int | None) -> str:
        return f"k={k}" if k is not None else "not crossed at nodes"

    with traj_server.gui.add_folder("cSTC Phase Boundaries"):
        traj_server.gui.add_markdown(
            f"**Phase structure (N={N} nodes)**\n\n"
            f"**Altitude triggers** (horizontal discs):\n"
            f"- 🔵 h < {int(ALT_TRIGGER_H2_M)} m → {_node_label(k_h2)}: LOS boresight viewcone  \n"
            f"- 🔴 h < {int(ALT_TRIGGER_H1_M)} m → {_node_label(k_h1)}: tight terminal  \n\n"
            f"**Speed ∧ tilt trigger** (node marker only):\n"
            f"- 🟡 {_node_label(k_aft)}: single-engine thrust (||v||<35 ∧ θ<60°)  \n"
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

    sensor_server = create_cstc_sensor_fpv_server(result, handle.traj_time)
    print("  Sensor FPV view — open the second viser URL printed above")

    traj_server.sleep_forever()


def _cbi_transpose(q_xyzw: np.ndarray) -> np.ndarray:
    """Body-to-inertial DCM (CBI^T) from a quaternion in [qx, qy, qz, qw] order."""
    qx, qy, qz, qw = q_xyzw
    return np.array([
        [qw**2 + qx**2 - qy**2 - qz**2, 2*(qx*qy - qw*qz),               2*(qw*qy + qx*qz)],
        [2*(qw*qz + qx*qy),              qw**2 - qx**2 + qy**2 - qz**2,  2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),              2*(qw*qx + qy*qz),              qw**2 - qx**2 - qy**2 + qz**2],
    ])


def _compute_los_angle(pos_arr: np.ndarray, q_arr: np.ndarray,
                       d_b_arr: np.ndarray, phi_b_arr: np.ndarray,
                       apex: np.ndarray | None = None) -> np.ndarray:
    """Angle (deg) between the apex-relative position and the LOS boresight."""
    nk  = pos_arr.shape[0]
    ang = np.zeros(nk)
    apex = np.zeros(3) if apex is None else np.asarray(apex, dtype=np.float64)
    for k in range(nk):
        los_b = np.array([
            np.sin(d_b_arr[k]) * np.cos(phi_b_arr[k]),
            np.sin(d_b_arr[k]) * np.sin(phi_b_arr[k]),
            np.cos(d_b_arr[k]),
        ])
        los_i = _cbi_transpose(q_arr[k]) @ los_b
        r_k   = pos_arr[k] - apex
        r_n   = np.linalg.norm(r_k)
        if r_n > 1e-8:
            cos_val = np.dot(r_k, los_i) / (r_n * (np.linalg.norm(los_i) + 1e-12))
            ang[k] = np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))
    return ang


def _save_plotly_figure(fig, basename: str) -> None:
    """Save a Plotly figure as HTML and, when kaleido is available, PNG/PDF."""
    fig.write_html(f"{basename}.html")
    print(f"  Saved {basename}.html")
    try:
        fig.write_image(f"{basename}.png", scale=2)
        fig.write_image(f"{basename}.pdf")
        print(f"  Saved {basename}.{{png,pdf}}")
    except Exception as exc:
        print(f"  Skipped PNG/PDF for {basename} ({exc}); install kaleido for static export.")


def _quat_xyzw_to_rpy(quat_xyzw: np.ndarray, *, degrees: bool = True) -> np.ndarray:
    """Convert quaternion(s) in OpenSCvx order [qx, qy, qz, qw] to roll, pitch, yaw.

    Uses extrinsic XYZ Euler angles (roll about x, pitch about y, yaw about z),
    matching CT-cSTC/CT-cSTC.ipynb ``rotation_matrix`` / ``euler_to_quat``.
    """
    from scipy.spatial.transform import Rotation as R

    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    if quat_xyzw.ndim == 1:
        quat_xyzw = quat_xyzw.reshape(1, 4)
    rpy = R.from_quat(quat_xyzw).as_euler("XYZ")
    if degrees:
        rpy = np.degrees(rpy)
    return rpy


def _round_sigfigs(x: np.ndarray, sig: int = 5) -> np.ndarray:
    """Round array elements to a fixed number of significant figures."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    nonzero = x != 0.0
    if np.any(nonzero):
        power = np.floor(np.log10(np.abs(x[nonzero])))
        scale = 10.0 ** (sig - 1 - power)
        out[nonzero] = np.round(x[nonzero] * scale) / scale
    return out


def export_trajectory_rpy_csv(
    result,
    path: str = "cstc_hop_trajectory_rpy.csv",
    *,
    degrees: bool = True,
) -> str:
    """Write trajectory CSV with columns time, pos_x, pos_y, pos_z, roll, pitch, yaw.

    Positions are the solver's scaled coordinates (not multiplied by ``R_SCALE``).
    """
    traj = result.trajectory
    t = np.asarray(traj["time"], dtype=np.float64).reshape(-1)
    pos_raw = np.asarray(traj["position"], dtype=np.float64)
    q = np.asarray(traj["attitude"], dtype=np.float64)
    rpy = _quat_xyzw_to_rpy(q, degrees=degrees)

    # Build a fresh array with x/y swapped — do NOT modify the trajectory in-place;
    # result.trajectory["position"] must remain untouched for prepare_for_viser().
    pos = np.empty_like(pos_raw)
    pos[:, 0] = pos_raw[:, 1]   # senss x ← model y
    pos[:, 1] = -pos_raw[:, 0]   # senss y ← model x
    pos[:, 2] = pos_raw[:, 2]

    # pitch = rpy[:, 1].copy()
    # roll = rpy[:, 0].copy()

    # senss_gimble_pitch = roll 
    # senss_gimble_roll = pitch

    # rpy[:, 0] = senss_gimble_roll
    # rpy[:, 1] = senss_gimble_pitch

    data = _round_sigfigs(np.column_stack([t, pos, rpy]))
    header = "t,rx,ry,rz,phi,theta,psi"
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    print(f"  Saved {path}")
    return path


def plot_cstc_results(result, *, show: bool = True, save_prefix: str = "cstc_hop") -> tuple:
    """Generate state/control panel and 3-D trajectory plots matching CT-cSTC notebook.

    Trigger times are computed from where the actual trajectory first crosses each
    trigger threshold, so the tightened bounds (drawn after the trigger) reflect
    the *real* state-triggered behaviour rather than fixed node intervals.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from openscvx.plotting.publication import show_plotly_with_latin_modern

    # ── Dense propagated trajectory (scaled) ──────────────────────────────────
    traj   = result.trajectory
    pos_s  = np.asarray(traj["position"])
    vel_s  = np.asarray(traj["velocity"])
    m_s    = np.asarray(traj["mass"]).flatten()
    q      = np.asarray(traj["attitude"])
    w      = np.asarray(traj["angular_velocity"])
    T_s    = np.asarray(traj["thrust_mag"]).flatten()
    d_e    = np.asarray(traj["gimbal_elev"]).flatten()
    d_b    = np.asarray(traj["los_elev"]).flatten()
    phi_b  = np.asarray(traj["los_az"]).flatten()
    t_full = np.asarray(traj["time"]).flatten()

    # ── Node-level data (for scatter) ─────────────────────────────────────────
    nodes   = result.nodes
    pos_n_s = np.asarray(nodes["position"])
    vel_n_s = np.asarray(nodes["velocity"])
    m_n_s   = np.asarray(nodes["mass"]).flatten()
    q_n     = np.asarray(nodes["attitude"])
    w_n     = np.asarray(nodes["angular_velocity"])
    T_n_s   = np.asarray(nodes["thrust_mag"]).flatten()
    d_e_n   = np.asarray(nodes["gimbal_elev"]).flatten()
    d_b_n   = np.asarray(nodes["los_elev"]).flatten()
    phi_b_n = np.asarray(nodes["los_az"]).flatten()

    t_nodes = np.asarray(nodes["time"]).flatten()

    # ── Unscale to physical units ─────────────────────────────────────────────
    pos   = pos_s   * R_SCALE
    vel   = vel_s   * R_SCALE
    m     = m_s     * M_SCALE
    T_N   = T_s     * M_SCALE * R_SCALE

    pos_n = pos_n_s * R_SCALE
    vel_n = vel_n_s * R_SCALE
    m_n   = m_n_s   * M_SCALE
    T_n_N = T_n_s   * M_SCALE * R_SCALE

    # ── Derived quantities ────────────────────────────────────────────────────
    speed_d   = np.linalg.norm(vel,   axis=1)
    speed_n = np.linalg.norm(vel_n, axis=1)

    tilt_deg   = np.degrees(np.arccos(np.clip(1 - 2*(q[:,0]**2   + q[:,1]**2),   -1.0, 1.0)))
    tilt_deg_n = np.degrees(np.arccos(np.clip(1 - 2*(q_n[:,0]**2 + q_n[:,1]**2), -1.0, 1.0)))

    omega_dps   = np.degrees(np.linalg.norm(w,   axis=1))
    omega_dps_n = np.degrees(np.linalg.norm(w_n, axis=1))

    # GS / LoS relative to land pad only
    pos_rel   = pos   - R_I_LAND
    pos_rel_n = pos_n - R_I_LAND
    r_xy   = np.linalg.norm(pos_rel[:,   :2], axis=1)
    r_xy_n = np.linalg.norm(pos_rel_n[:, :2], axis=1)
    gs_deg   = 90.0 - np.degrees(np.arctan2(pos_rel[:,   2], r_xy   + 1e-8))
    gs_deg_n = 90.0 - np.degrees(np.arctan2(pos_rel_n[:, 2], r_xy_n + 1e-8))

    los_ang   = _compute_los_angle(pos,   q,   d_b,   phi_b,   apex=R_I_LOS_LAND)
    los_ang_n = _compute_los_angle(pos_n, q_n, d_b_n, phi_b_n, apex=R_I_LOS_LAND)

    # ── Trigger times from actual trajectory crossings ────────────────────────
    h_pad = _height_above_pad(pos)

    def _first_time(mask):
        if mask.any():
            return float(t_full[int(np.argmax(mask))])
        return float(t_full[-1])

    t_h1  = _first_time(h_pad < ALT_TRIGGER_H1_M)
    t_h2  = _first_time(h_pad < ALT_TRIGGER_H2_M)
    t_aft = _first_time((speed_d < SPD_STC_TRIG) & (tilt_deg < THETA_STC_TRIG))
    t_end = float(t_nodes[-1])

    print(f"  Trigger times:  h<100m @ {t_h1:.2f}s   h<200m @ {t_h2:.2f}s   "
          f"(v<35 & θ<60) @ {t_aft:.2f}s")

    # ── Colour palette (matches notebook) ────────────────────────────────────
    c_node = "black"
    c_plt  = "blue"
    c_h1   = "red"
    c_h2   = "orange"
    c_aft  = "lightseagreen"
    c_spd  = "burlywood"
    c_up   = "green"
    c_low  = "purple"

    legend_seen: set[str] = set()

    def _show(name: str) -> bool:
        if name in legend_seen:
            return False
        legend_seen.add(name)
        return True

    def _seg(fig, x0, x1, y, *, row, col, color, dash="dash", name=None):
        fig.add_trace(
            go.Scatter(
                x=[x0, x1], y=[y, y], mode="lines",
                line={"color": color, "dash": dash, "width": 1.5},
                name=name, showlegend=_show(name) if name else False, legendgroup=name,
            ),
            row=row, col=col,
        )

    def _line(fig, x, y, *, row, col, color, name=None, width=2):
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines",
                line={"color": color, "width": width},
                name=name, showlegend=_show(name) if name else False, legendgroup=name,
            ),
            row=row, col=col,
        )

    def _nodes(fig, x, y, *, row, col, name="Node point"):
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers",
                marker={"color": c_node, "size": 7},
                name=name, showlegend=_show(name), legendgroup=name,
            ),
            row=row, col=col,
        )

    def _vline(fig, x, *, row, col, color):
        fig.add_vline(x=x, line={"color": color, "dash": "dash", "width": 1.5}, row=row, col=col)

    # ── Figure 1: 9-panel state/control plot ─────────────────────────────────
    fig_panel = make_subplots(rows=3, cols=3, vertical_spacing=0.10, horizontal_spacing=0.08)

    # ── Thrust (row 1, col 1) ───────────────────────────────────────────────
    T_kN   = T_N   * 1e-3
    T_n_kN = T_n_N * 1e-3
    _seg(fig_panel, 0, t_aft, T_MAX * 1e-3, row=1, col=1, color=c_up, name="Upper bound")
    _seg(fig_panel, 0, t_aft, T_MIN * 1e-3, row=1, col=1, color=c_low, name="Lower bound")
    _seg(fig_panel, t_aft, t_end, T_MAX_AFT * 1e-3, row=1, col=1, color=c_up)
    _seg(fig_panel, t_aft, t_end, T_MIN_AFT * 1e-3, row=1, col=1, color=c_low)
    _vline(fig_panel, t_aft, row=1, col=1, color=c_aft)
    _line(fig_panel, t_full, T_kN, row=1, col=1, color="red", name="Control input")
    _nodes(fig_panel, t_nodes, T_n_kN, row=1, col=1)
    fig_panel.update_yaxes(title_text="Thrust, T [kN]", row=1, col=1)

    # ── Speed (row 1, col 2) ────────────────────────────────────────────────
    _line(fig_panel, t_full, speed_d, row=1, col=2, color=c_plt, name="State")
    _nodes(fig_panel, t_nodes, speed_n, row=1, col=2)
    _seg(fig_panel, 0, t_end, SPD_STC_TRIG, row=1, col=2, color=c_spd, name="$v^{\\mathrm{trig}}$")
    _vline(fig_panel, t_h1, row=1, col=2, color=c_h1)
    _vline(fig_panel, t_aft, row=1, col=2, color=c_aft)
    _seg(fig_panel, t_h1, t_end, V_STC_CONS, row=1, col=2, color=c_up, name="STC bound")
    fig_panel.update_yaxes(title_text="Speed, ||v||₂ [m s⁻¹]", range=[0, speed_d.max() + 5], row=1, col=2)

    # ── Tilt (row 1, col 3) ─────────────────────────────────────────────────
    _line(fig_panel, t_full, tilt_deg, row=1, col=3, color=c_plt)
    _nodes(fig_panel, t_nodes, tilt_deg_n, row=1, col=3)
    _seg(fig_panel, 0, t_h1, THETA_MAX_DEG, row=1, col=3, color=c_up)
    _seg(fig_panel, t_h1, t_end, THETA_STC_DEG, row=1, col=3, color=c_up)
    _seg(fig_panel, 0, t_end, THETA_STC_TRIG, row=1, col=3, color=c_spd, name="$\\theta^{\\mathrm{trig}}$")
    _vline(fig_panel, t_h1, row=1, col=3, color=c_h1)
    fig_panel.update_yaxes(title_text="Tilt angle, θ [deg]", row=1, col=3)

    # ── Engine gimbal deflection (row 2, col 1) ─────────────────────────────
    d_e_deg   = np.degrees(d_e)
    d_e_deg_n = np.degrees(d_e_n)
    _line(fig_panel, t_full, d_e_deg, row=2, col=1, color="red")
    _nodes(fig_panel, t_nodes, d_e_deg_n, row=2, col=1)
    _seg(fig_panel, 0, t_h1,  DELTA_ENGINE_MAX_DEG, row=2, col=1, color=c_up)
    _seg(fig_panel, 0, t_h1, -DELTA_ENGINE_MAX_DEG, row=2, col=1, color=c_low)
    _seg(fig_panel, t_h1, t_end,  DELTA_STC_DEG, row=2, col=1, color=c_up)
    _seg(fig_panel, t_h1, t_end, -DELTA_STC_DEG, row=2, col=1, color=c_low)
    _vline(fig_panel, t_h1, row=2, col=1, color=c_h1)
    fig_panel.update_yaxes(title_text="Engine gimbal, δᵉ [deg]", row=2, col=1)

    # ── Angular velocity (row 2, col 2) ─────────────────────────────────────
    _line(fig_panel, t_full, omega_dps, row=2, col=2, color=c_plt)
    _nodes(fig_panel, t_nodes, omega_dps_n, row=2, col=2)
    _seg(fig_panel, 0, t_h1, np.degrees(W_B_MAX_RAD_S), row=2, col=2, color=c_up)
    _seg(fig_panel, t_h1, t_end, np.degrees(OMEGA_STC_RAD_S), row=2, col=2, color=c_up)
    _vline(fig_panel, t_h1, row=2, col=2, color=c_h1)
    fig_panel.update_yaxes(title_text="Angular velocity, ω_B [deg s⁻¹]", row=2, col=2)

    # ── Glideslope (row 2, col 3) ─────────────────────────────────────────────
    gs_bound_pre = 90.0 - GS_MAX_DEG
    gs_bound_stc = 90.0 - GS_STC_DEG
    _line(fig_panel, t_full[:-1], gs_deg[:-1], row=2, col=3, color=c_plt)
    _nodes(fig_panel, t_nodes[:-1], gs_deg_n[:-1], row=2, col=3)
    _seg(fig_panel, 0, t_h1, gs_bound_pre, row=2, col=3, color=c_up)
    _seg(fig_panel, t_h1, t_end, gs_bound_stc, row=2, col=3, color=c_up)
    _vline(fig_panel, t_h1, row=2, col=3, color=c_h1)
    fig_panel.update_yaxes(
        title_text="Glideslope, γ [deg]",
        range=[0, max(gs_bound_pre, gs_deg[:-1].max()) + 5], row=2, col=3,
    )

    # ── LOS boresight elevation angle (row 3, col 1) ────────────────────────
    d_b_deg   = np.degrees(d_b)
    d_b_deg_n = np.degrees(d_b_n)
    _line(fig_panel, t_full, d_b_deg, row=3, col=1, color="red")
    _nodes(fig_panel, t_nodes, d_b_deg_n, row=3, col=1)
    _seg(fig_panel, t_h2, t_end, DELTA_BORESIGHT_MAX_DEG, row=3, col=1, color=c_up)
    _vline(fig_panel, t_h2, row=3, col=1, color=c_h2)
    fig_panel.update_xaxes(title_text="Time [s]", row=3, col=1)
    fig_panel.update_yaxes(title_text="Boresight deflection, δᵇ [deg]", row=3, col=1)

    # ── Mass (row 3, col 2) ─────────────────────────────────────────────────
    _line(fig_panel, t_full, m / 1e3, row=3, col=2, color=c_plt)
    _nodes(fig_panel, t_nodes, m_n / 1e3, row=3, col=2)
    fig_panel.add_hline(y=M_DRY / 1e3, line={"color": c_low, "dash": "dash", "width": 1.5}, row=3, col=2)
    fig_panel.update_xaxes(title_text="Time [s]", row=3, col=2)
    fig_panel.update_yaxes(title_text="Mass, m [10³ kg]", row=3, col=2)

    # ── LOS view angle (row 3, col 3) ───────────────────────────────────────
    _line(fig_panel, t_full, los_ang, row=3, col=3, color=c_plt)
    _nodes(fig_panel, t_nodes, los_ang_n, row=3, col=3)
    _seg(fig_panel, t_h2, t_end, LOS_STC_DEG, row=3, col=3, color=c_up)
    _vline(fig_panel, t_h2, row=3, col=3, color=c_h2)
    fig_panel.update_xaxes(title_text="Time [s]", row=3, col=3)
    fig_panel.update_yaxes(title_text="LOS angle, ψ [deg]", row=3, col=3)

    for row in range(1, 4):
        for col in range(1, 4):
            fig_panel.update_xaxes(range=[0, t_end], row=row, col=col)

    fig_panel.update_layout(
        template="plotly_white", width=1100, height=650,
        margin={"t": 80, "b": 40, "l": 50, "r": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "left", "x": 0, "font": {"size": 10}},
    )

    if save_prefix:
        _save_plotly_figure(fig_panel, f"{save_prefix}_states_controls")

    # ── Figure 2: 3-D trajectory coloured by speed ────────────────────────────
    fig_3d = go.Figure()
    fig_3d.add_trace(
        go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="lines",
            customdata=speed_d,
            line={"color": speed_d, "colorscale": "Rainbow",
                  "cmin": float(speed_d.min()), "cmax": float(speed_d.max()),
                  "width": 4, "colorbar": {"title": "Speed [m/s]"}},
            showlegend=False,
            hovertemplate=("Crossrange: %{x:.1f} m<br>Downrange: %{y:.1f} m<br>"
                           "Altitude: %{z:.1f} m<br>Speed: %{customdata:.1f} m/s<extra></extra>"),
        )
    )
    fig_3d.add_trace(
        go.Scatter3d(x=pos_n[:, 0], y=pos_n[:, 1], z=pos_n[:, 2], mode="markers",
                     marker={"color": "black", "size": 4}, name="Node")
    )
    fig_3d.add_trace(
        go.Scatter3d(
            x=[R_I_LAUNCH[0]], y=[R_I_LAUNCH[1]], z=[R_I_LAUNCH[2]], mode="markers",
            marker={"color": "cyan", "size": 10, "symbol": "diamond"}, name="Launch pad")
    )
    fig_3d.add_trace(
        go.Scatter3d(
            x=[R_I_APEX[0]], y=[R_I_APEX[1]], z=[R_I_APEX[2]], mode="markers",
            marker={"color": "orange", "size": 10, "symbol": "diamond"}, name="Apex")
    )
    fig_3d.add_trace(
        go.Scatter3d(
            x=[R_I_LAND[0]], y=[R_I_LAND[1]], z=[R_I_LAND[2]], mode="markers",
            marker={"color": "lime", "size": 10, "symbol": "diamond"}, name="Landing pad")
    )

    all_xyz = np.vstack([pos, pos_n])
    pad = 30.0
    fig_3d.update_layout(
        template="plotly_white", title={"text": "6-DoF cSTC Hop Trajectory", "x": 0.5},
        width=900, height=800,
        scene={
            "xaxis_title": "Crossrange [m]", "yaxis_title": "Downrange [m]", "zaxis_title": "Altitude [m]",
            "xaxis": {"range": [all_xyz[:, 0].min() - pad, all_xyz[:, 0].max() + pad]},
            "yaxis": {"range": [all_xyz[:, 1].min() - pad, all_xyz[:, 1].max() + pad]},
            "zaxis": {"range": [0, all_xyz[:, 2].max() + 30]},
            "aspectmode": "data",
        },
        legend={"x": 0.02, "y": 0.98},
    )

    if save_prefix:
        _save_plotly_figure(fig_3d, f"{save_prefix}_trajectory_3d")

    if show:
        show_plotly_with_latin_modern(fig_panel)
        show_plotly_with_latin_modern(fig_3d)

    return fig_panel, fig_3d


if __name__ == "__main__":
    problem.settings.dev.debug = True   # bypass JAX export (see repo issue)
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    from openscvx.plotting import plot_states, plot_controls

    plot_states(result).show()
    plot_controls(result).show()

    traj = result.trajectory
    pos = np.asarray(traj["position"]) * R_SCALE
    vel = np.asarray(traj["velocity"]) * R_SCALE
    m = np.asarray(traj["mass"]) * M_SCALE

    print("\n── Solution summary ─────────────────────────────────────────────")
    print(f"  Launch → apex → land hop (N={N}, K_APEX={K_APEX})")
    print(f"  Apex node position (m): {np.asarray(result.nodes['position'])[K_APEX] * R_SCALE}")
    print(f"  Final position (m):   {pos[-1]}")
    print(f"  Final velocity (m/s): {vel[-1]}")
    print(f"  Final mass (kg):      {m[-1, 0]:.1f}  (dry mass: {M_DRY:.0f} kg)")
    print(f"  Fuel used (kg):       {M_WET - m[-1, 0]:.1f}")

    print("\n── Generating plots ─────────────────────────────────────────────")
    plot_cstc_results(result, show=True)
    export_trajectory_rpy_csv(result)

    prepare_for_viser(result)
    launch_viser_servers(result)
