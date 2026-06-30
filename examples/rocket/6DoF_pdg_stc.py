"""
6-DoF Powered Descent Guidance with *true* Compound State-Triggered Constraints (cSTC)

Port of CT-cSTC/CT-cSTC.ipynb into OpenSCvx's symbolic framework.

This example encodes state-triggered constraints exactly as the notebook does:
each compound STC is a product of smooth trigger indicators and constraint
penalties that is integrated over the trajectory and driven to zero.

How the STCs are encoded
------------------------
OpenSCvx's CTCS mechanism turns a constraint ``g(x, u) <= 0`` into the penalty
``Square(PositivePart(g))`` which is integrated into an augmented state that is
driven to zero (continuous-time constraint satisfaction).  We exploit this:

    g  :=  relu(trigger_1) * ... * relu(trigger_M) * relu(constraint)   ( >= 0 )

so that ``Square(PositivePart(g)) = relu(trigger_1)^2 * ... * relu(constraint)^2``.

This is exactly the notebook's compound penalty (notebook cell 39, state ``x[14]``):
the constraint only contributes a penalty when *every* trigger is active AND the
constraint is violated.  An ``OR`` over triggers is encoded as a *sum* of such
product terms (matching ``spd_stc_trig_i + theta_stc_trig_i`` in the notebook).

Compound state-triggered constraints (notebook cell 16 / cell 39)
-----------------------------------------------------------------
    h < 100 m                       → tight gimbal, tilt, glideslope
    h < 110 m  (alpha-tightened)    → tight angular-rate and speed
    h < 200 m  (220 m alpha) & h>2  → LOS boresight cone toward the pad
    ||v|| < 35 m/s  AND  tilt < 60° → single-engine thrust limits
    ||v|| > 35 m/s  OR   tilt > 60° → three-engine thrust limits

Reference: CT-cSTC/CT-cSTC.ipynb

When run as a script, launches three viser windows after solving:
  1. Animated trajectory – thrust plume, attitude frame, velocity-colored trail
  2. SCP convergence – node positions across iterations
  3. Snapshot grid – evenly-spaced body poses along the final path

Viser scene uses the same ENU frame as the model (x, y horizontal; z = altitude up).
"""

import os
import sys

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

# ── Physical parameters (notebook cell 16 / cell 33) ──────────────────────────
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
R_I_FINAL = np.array([0.0, 0.0, 0.001])      # m — touchdown target (slightly above pad)
GS_LOS_APEX_OFFSET_M = 0.001                 # m — GS / LoS cone apex sits below touchdown
R_I_APEX = R_I_FINAL - np.array([0.0, 0.0, GS_LOS_APEX_OFFSET_M])  # cone vertex (pad)
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

# ── Scaling (notebook cell 18 / cell 37) ──────────────────────────────────────
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
_T_min_s     = T_MIN     / (M_SCALE * R_SCALE)
_T_max_aft_s = T_MAX_AFT / (M_SCALE * R_SCALE)
_T_min_aft_s = T_MIN_AFT / (M_SCALE * R_SCALE)

_r_init_s  = R_I_INIT / R_SCALE
_r_final_s = R_I_FINAL / R_SCALE
_r_apex_s  = R_I_APEX / R_SCALE
_v_init_s  = V_I_INIT / R_SCALE
_v_final_s = V_I_FINAL / R_SCALE
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
N = 15

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
position.final   = [float(_r_final_s[0]), float(_r_final_s[1]), float(_r_final_s[2])]

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
los_az.max   = [ np.radians(DELTA_BORESIGHT_MAX_DEG)]
los_az.min   = [-np.radians(DELTA_BORESIGHT_MAX_DEG)]
los_az.guess = np.zeros((N, 1))

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
speed     = ox.linalg.Norm(velocity)
omega_sq  = ox.linalg.Norm(angular_velocity)**2
z_alt     = position[2]   # height above pad (z = 0); triggers use absolute altitude

# Position relative to GS / LoS cone apex (below touchdown, not at touchdown)
pos_rel_x = position[0] - float(_r_apex_s[0])
pos_rel_y = position[1] - float(_r_apex_s[1])
pos_rel_z = position[2] - float(_r_apex_s[2])
r_xy_norm = ox.linalg.Norm(ox.Concat(pos_rel_x, pos_rel_y))
r_los_norm = ox.linalg.Norm(ox.Concat(pos_rel_x, pos_rel_y, pos_rel_z))

# LOS boresight in inertial frame via body→inertial rotation
db = los_elev[0]
pb = los_az[0]
los_B = ox.Concat(
    ox.Sin(db) * ox.Cos(pb),
    ox.Sin(db) * ox.Sin(pb),
    ox.Cos(db),
)
los_I = CBI.T @ los_B   # unit-norm boresight in inertial frame
r_dot_los = (pos_rel_x * los_I[0]
             + pos_rel_y * los_I[1]
             + pos_rel_z * los_I[2])


# ── State-triggered-constraint helper ─────────────────────────────────────────
def relu(expr):
    """Smooth-free positive part (max(expr, 0)); same primitive CTCS uses."""
    return ox.Max(expr, 0)


def stc(*factors, weight: float = 1.0):
    """Compound state-triggered constraint as a single CTCS term.

    ``factors`` are the relu(trigger)/relu(constraint) expressions whose product
    forms the penalty.  Because the product is non-negative, the CTCS penalty
    ``Square(PositivePart(weight * prod))`` evaluates to
    ``weight^2 * prod_i relu(factor_i)^2`` — the notebook's compound STC penalty.
    All such terms share one augmented state (nodes=None), exactly mirroring the
    notebook's single CTCS accumulator (state x[14]).
    """
    prod = factors[0]
    for f in factors[1:]:
        prod = prod * f
    return ox.ctcs((weight * prod) <= 0, penalty="huber")


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
C_gs_stc   = relu(r_xy_norm * _tan_gs_stc - pos_rel_z)  # tight glideslope (apex at R_I_APEX)
C_omega    = relu(omega_sq - _omega_sq_stc)             # tight angular rate
C_tilt     = relu(tilt_sq - _tilt_sq_stc)               # tight tilt
C_spd      = relu(speed - _v_stc_s)                     # tight speed
C_los      = relu(r_los_norm * _cos_psi_stc - r_dot_los)                # LOS cone (toward R_I_APEX)
C_Tmin_f   = relu(_T_min_aft_s - T)                     # single-engine min
C_Tmax_f   = relu(T - _T_max_aft_s)                     # single-engine max
C_Tmin_i   = relu(_ALPHA_T_MIN * _T_min_s - T)          # three-engine min
C_Tmax_i   = relu(T - _T_max_s)                         # three-engine max

# ── Constraints ───────────────────────────────────────────────────────────────
constraints = []

# Boundary conditions (convex equality constraints)
constraints.append((position         == _r_init_s).convex().at([0]))
constraints.append((attitude         == Q_INIT).convex().at([0]))
constraints.append((velocity         == _v_init_s).convex().at([0]))
constraints.append((angular_velocity == W_INIT).convex().at([0]))
constraints.append((position         == _r_final_s).convex().at([N - 1]))
constraints.append((velocity         == _v_final_s).convex().at([N - 1]))
constraints.append((attitude         == Q_FINAL).convex().at([N - 1]))
constraints.append((angular_velocity == W_FINAL).convex().at([N - 1]))

# ── Always-on CTCS (entire trajectory) ────────────────────────────────────────
constraints.append(ox.ctcs(tilt_sq - _tilt_sq_max <= 0, penalty="huber"))                 # tilt ≤ 90°
constraints.append(ox.ctcs(omega_sq - W_B_MAX_RAD_S**2 <= 0, penalty="huber"))            # angular rate
constraints.append(ox.ctcs(r_xy_norm * _tan_gs_max - pos_rel_z <= 0, penalty="huber"))    # glideslope 55°
constraints.append(ox.ctcs(_m_dry_s - mass[0] <= 0, penalty="huber"))                     # dry-mass floor

# ── Compound state-triggered constraints (notebook cell 39) ───────────────────
# h < 100 m → tight gimbal deflection (|δ_e| ≤ 1°)
constraints.append(stc(T_alt100, C_gimbal_p, weight=W_GIMBAL))
constraints.append(stc(T_alt100, C_gimbal_n, weight=W_GIMBAL))
# h < 100 m AND h > 2 m → tight glideslope cone
constraints.append(stc(T_alt100, T_altgt2, C_gs_stc, weight=W_GS))
# h < 100 m → tight tilt
constraints.append(stc(T_alt100, C_tilt, weight=W_TILT))
# h < 110 m → tight angular rate and tight speed
constraints.append(stc(T_alt110, C_omega, weight=W_OMEGA))
constraints.append(stc(T_alt110, C_spd,   weight=W_SPD))
# h < 220 m AND h > 2 m → LOS boresight cone toward R_I_APEX (below touchdown)
constraints.append(stc(T_alt220, T_altgt2, C_los, weight=W_LOS))

# ||v|| < 35 m/s AND tilt < 60° → single-engine thrust limits
constraints.append(stc(T_vlt35, T_tlt60, C_Tmax_f, weight=W_THR))
constraints.append(stc(T_vlt35, T_tlt60, C_Tmin_f, weight=W_THR))
# ||v|| > 35 m/s OR tilt > 60° → three-engine thrust limits (OR = sum of products)
constraints.append(stc(T_vgt35, C_Tmax_i, weight=W_THR))
constraints.append(stc(T_vgt35, C_Tmin_i, weight=W_THR))
constraints.append(stc(T_tgt60, C_Tmax_i, weight=W_THR))
constraints.append(stc(T_tgt60, C_Tmin_i, weight=W_THR))

# ── Time (free final time with per-segment dilation) ─────────────────────────
_t_f_guess = 21.0

time = ox.Time(
    initial=0.0,
    final=ox.Free(_t_f_guess),
    min=0.0,
    max=_t_f_guess * 1.5,
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
    # licq_max = 1e-6,
    algorithm={
        # A high *constant* constraint-violation weight (lam_vc) plays the role
        # of the notebook's w_con_dyn, driving the integrated STC penalty to
        # zero.  PTR (the default acceptance-ratio loop) keeps the SCP stable;
        # the adaptive AugmentedLagrangian was found to overshoot and diverge.
        "lam_vc": 4E0,
        "lam_cost": 1e-2,
        "k_max": 800,
        "autotuner": ox.ConstantProximalWeight(),
        # "autotuner": ox.AugmentedLagrangian(ep=1E-1, eta_lambda=1e3),
    },
    discretizer={
        "diffrax_kwargs": {"atol": 1e-10, "rtol": 1e-10},
    },
)

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
        (ALT_TRIGGER_H2_M, (80, 160, 255), "h < 200 m → LOS boresight cone"),
        (ALT_TRIGGER_H1_M, (255, 80, 80),  "h < 100 m → tight gimbal/tilt/ω/speed/GS"),
    ]
    for alt_m, color, description in triggers:
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
            text=description,
            position=(radius * 0.85, 0.0, z + 0.05),
        )

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
    alt = pos_m[:, 2]
    spd = np.linalg.norm(vel_ms, axis=1)
    tilt_deg = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 0] ** 2 + q[:, 1] ** 2), -1.0, 1.0)))

    k_h1 = _first_node_index(alt < ALT_TRIGGER_H1_M)
    k_h2 = _first_node_index(alt < ALT_TRIGGER_H2_M)
    k_aft = _first_node_index((spd < SPD_STC_TRIG) & (tilt_deg < THETA_STC_TRIG))
    return k_h1, k_h2, k_aft


def launch_viser_servers(result) -> None:
    """Create trajectory, SCP convergence, and snapshot viser servers."""
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

    apex_vis = tuple((R_I_APEX / SCENE_SCALE).tolist())
    touchdown_vis = tuple((R_I_FINAL / SCENE_SCALE).tolist())
    add_glideslope_cone(
        traj_server,
        apex=apex_vis,
        height=initial_alt_vis,
        glideslope_angle_deg=GS_HALFANGLE_DEG,
        axis=_VISER_UP_AXIS,
        color=(80, 200, 80),
        opacity=0.12,
    )
    add_glideslope_cone(
        traj_server,
        apex=apex_vis,
        height=ALT_TRIGGER_H1_M / SCENE_SCALE,
        glideslope_angle_deg=GS_STC_HALFANGLE_DEG,
        axis=_VISER_UP_AXIS,
        color=(255, 180, 40),
        opacity=0.15,
    )

    add_cstc_altitude_triggers(traj_server, pos, scene_scale_m=SCENE_SCALE)

    traj_server.scene.add_icosphere(
        "/cone_apex",
        radius=0.08,
        color=(255, 220, 80),
        position=apex_vis,
    )
    traj_server.scene.add_icosphere(
        "/landing_pad",
        radius=0.12,
        color=(50, 255, 80),
        position=touchdown_vis,
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


def plot_cstc_results(result, *, show: bool = True, save_prefix: str = "cstc_stc") -> tuple:
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

    pos_rel   = pos   - R_I_APEX
    pos_rel_n = pos_n - R_I_APEX
    r_xy   = np.linalg.norm(pos_rel[:,   :2], axis=1)
    r_xy_n = np.linalg.norm(pos_rel_n[:, :2], axis=1)
    gs_deg   = 90.0 - np.degrees(np.arctan2(pos_rel[:,   2], r_xy   + 1e-8))
    gs_deg_n = 90.0 - np.degrees(np.arctan2(pos_rel_n[:, 2], r_xy_n + 1e-8))

    los_ang   = _compute_los_angle(pos,   q,   d_b,   phi_b,   apex=R_I_APEX)
    los_ang_n = _compute_los_angle(pos_n, q_n, d_b_n, phi_b_n, apex=R_I_APEX)

    # ── Trigger times from actual trajectory crossings ────────────────────────
    alt_d = pos[:, 2]

    def _first_time(mask):
        if mask.any():
            return float(t_full[int(np.argmax(mask))])
        return float(t_full[-1])

    t_h1  = _first_time(alt_d < ALT_TRIGGER_H1_M)
    t_h2  = _first_time(alt_d < ALT_TRIGGER_H2_M)
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
        go.Scatter3d(x=[0.0], y=[0.0], z=[0.0], mode="markers",
                     marker={"color": "lime", "size": 10, "symbol": "diamond"}, name="Landing pad")
    )

    all_xyz = np.vstack([pos, pos_n])
    pad = 30.0
    fig_3d.update_layout(
        template="plotly_white", title={"text": "6-DoF PDG Trajectory (true cSTC)", "x": 0.5},
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

    traj = result.trajectory
    pos = np.asarray(traj["position"]) * R_SCALE
    vel = np.asarray(traj["velocity"]) * R_SCALE
    m = np.asarray(traj["mass"]) * M_SCALE

    print("\n── Solution summary ─────────────────────────────────────────────")
    print(f"  Final position (m):   {pos[-1]}")
    print(f"  Final velocity (m/s): {vel[-1]}")
    print(f"  Final mass (kg):      {m[-1, 0]:.1f}  (dry mass: {M_DRY:.0f} kg)")
    print(f"  Fuel used (kg):       {M_WET - m[-1, 0]:.1f}")

    print("\n── Generating plots ─────────────────────────────────────────────")
    plot_cstc_results(result, show=True)

    prepare_for_viser(result)
    launch_viser_servers(result)
