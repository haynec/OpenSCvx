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
    h < 200 m  (220 m alpha) & h>2  → LOS boresight cone toward the pad
    ||v|| < 35 m/s  AND  tilt < 60° → single-engine thrust limits
    ||v|| > 35 m/s  OR   tilt > 60° → three-engine thrust limits

Identical to ``6DoF_pdg_stc_senss.py`` except: the initial state (offset
start, 90° initial tilt), ``DELTA_ENGINE_MAX_DEG`` 60° → 10°,
``THETA_MAX_DEG`` 10° → 90°, no aerodynamics, and the SENSS gimbal-rig CSV
convention (swapped horizontal axes, −90° pitch trim, 2× time stretch).

Reference: CT-cSTC/CT-cSTC.ipynb

When run as a script, launches four viser windows after solving:
  1. Animated trajectory – thrust plume, attitude frame, velocity-colored
     trail, DEM terrain patch (position adjustable in Viser GUI)
  2. Onboard sensor FPV – camera locked to the LOS gimbal sensor with
     matched DEM placement, adjustable sensor FOV, and XYZ position offset sliders
  3. SCP convergence – node positions across iterations
  4. Snapshot grid – evenly-spaced body poses along the final path

Viser scene uses the same ENU frame as the model (x, y horizontal; z = altitude up).
"""

import os
import sys

import numpy as np

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
from examples.rocket._cstc_plotting import (
    CstcLimits,
    CstcScales,
    add_altitude_trigger_discs,
    add_camera_fov_slider,
    add_cstc_phase_markers,
    add_los_viewcone,
    add_site_markers,
    create_sensor_fpv_server,
    node_trigger_indices,
    plot_cstc_panel,
    plot_cstc_trajectory_3d,
    prepare_for_viser,
    quat_xyzw_to_rpy,
    resample_uniform,
    save_plotly_figure,
    write_trajectory_rpy_csv,
)
from examples.rocket.senss._dem import (
    NATIVE_GRID,
    DemPlacement,
    DemShading,
    add_dem_terrain,
    dem_info_markdown,
)
from openscvx import Problem
from openscvx.plotting.viser import add_animation_controls, add_glideslope_cone

# ── Physical parameters (notebook cell 16 / cell 33) ──────────────────────────
G0  = 9.806    # m/s²
ISP = 330.0    # s
M_WET = 100_000.0   # kg
M_DRY =  85_000.0   # kg

# Initial conditions
R_I_INIT = np.array([200.0, -200.0, 250.0])   # m
V_I_INIT = np.array([-10.0,   100.0, -10.0])   # m/s
# 90° tilt about x-axis: euler_to_quat([90,0,0]) → [w, x, y, z] = [√2/2, √2/2, 0, 0]
# OpenSCvx quaternion convention [x, y, z, w]: Q_INIT = [√2/2, 0, 0, √2/2]
# Q_INIT  = np.array([np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)])
Q_INIT  = np.array([np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)])
W_INIT  = np.zeros(3)   # rad/s

# Terminal conditions
R_I_FINAL = np.array([-10.0, -125.0, -200.0])      # m — touchdown target (slightly above pad)


def _height_above_landing_pad(pos_m: np.ndarray) -> np.ndarray:
    """Height above the touchdown z-level (m), not the world origin."""
    return np.asarray(pos_m)[:, 2] - R_I_FINAL[2]


R_I_GS_APEX = R_I_FINAL  # glideslope cone vertex (touchdown)
LOS_DEPTH_BELOW_TOUCHDOWN_M = 50.0  # m — LoS target sits below touchdown (z only)
R_I_LOS = R_I_FINAL - np.array([0.0, 0.0, LOS_DEPTH_BELOW_TOUCHDOWN_M])
V_I_FINAL = np.array([0.0, 0.0, -0.5])       # m/s (gentle touchdown)
Q_FINAL   = np.array([0.0, 0.0, 0.0, 1.0])   # upright
W_FINAL   = np.zeros(3)

# Thrust limits (N)
T_MAX     = 1_900_000.0 * 3.0        # 3-engine
T_MIN     = 1_900_000.0 * 0.4 * 3.0
T_MAX_AFT = 1_900_000.0              # 1-engine aft phase
T_MIN_AFT = 1_900_000.0 * 0.4

# Hard control angle limits (deg)
DELTA_ENGINE_MAX_DEG    = 10.0
DELTA_BORESIGHT_MAX_DEG = 0.0

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

# Lever arm (m) — engine to CM
R_CM = np.array([ 0.0,  0.0, -14.0])

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
R_SCALE = np.linalg.norm(R_I_INIT) / 2.5   # ≈ 574.46 m
M_SCALE = M_WET

_alpha_m_s = ALPHA_M * R_SCALE
_g_I_s     = G_I / R_SCALE
_r_cm_s    = R_CM / R_SCALE
_J_B_s     = J_B_DIAG     / R_SCALE**2
_J_B_inv_s = J_B_INV_DIAG * R_SCALE**2

_T_max_s     = T_MAX     / (M_SCALE * R_SCALE)
_T_min_s     = T_MIN     / (M_SCALE * R_SCALE)
_T_max_aft_s = T_MAX_AFT / (M_SCALE * R_SCALE)
_T_min_aft_s = T_MIN_AFT / (M_SCALE * R_SCALE)

_r_init_s  = R_I_INIT / R_SCALE
_r_final_s = R_I_FINAL / R_SCALE
_r_gs_apex_s    = R_I_GS_APEX / R_SCALE
_r_los_target_s = R_I_LOS / R_SCALE
_v_init_s  = V_I_INIT / R_SCALE
_v_final_s = V_I_FINAL / R_SCALE
_m_wet_s   = M_WET / M_SCALE   # 1.0
_m_dry_s   = M_DRY / M_SCALE   # 0.85

_v_stc_s   = V_STC_CONS / R_SCALE

# ── Precomputed constraint thresholds ─────────────────────────────────────────
def _tilt_sq_bound(theta_deg: float) -> float:
    """Squared quaternion tilt bound: ||[q_x, q_y]||² ≤ (1-cos θ)/2."""
    return (1.0 - np.cos(np.pi/180 * theta_deg)) / 2.0


_tilt_sq_max    = _tilt_sq_bound(THETA_MAX_DEG)
_tilt_sq_stc    = _tilt_sq_bound(THETA_STC_DEG)    # tight tilt (≈4.6°)
_tilt_sq_trig   = _tilt_sq_bound(THETA_STC_TRIG)   # tilt trigger (60° → 0.25)
_tan_gs_max     = np.tan(np.pi/180 * GS_MAX_DEG)
_tan_gs_stc     = np.tan(np.pi/180 * GS_STC_DEG)
_cos_psi_stc    = np.cos(np.pi/180 * LOS_STC_DEG)
_delta_stc_rad  = np.pi/180 * DELTA_STC_DEG
_omega_sq_stc   = OMEGA_STC_RAD_S**2

# Scaled altitude / speed trigger thresholds
_alt_h1_s     = ALT_TRIGGER_H1_M / R_SCALE
_alt_h2a_s    = _ALPHA_ALT_LOS * ALT_TRIGGER_H2_M / R_SCALE
_alt_done_s   = ALT_DONE_M / R_SCALE
_spd_trig_s   = SPD_STC_TRIG / R_SCALE

# ── Discretization ────────────────────────────────────────────────────────────
N = 30

# ── States ────────────────────────────────────────────────────────────────────
mass = ox.State("mass", shape=(1,))
mass.max = [_m_wet_s]
mass.min = [_m_dry_s]
mass.initial = [_m_wet_s]
mass.final   = [ox.Maximize(_m_dry_s)]   # fuel-optimal objective

position = ox.State("position", shape=(3,))
position.max = [ 2.5,  2.0,  2.5]
position.min = [-2.5, -2.0, -2.0]   # z ≥ 0 (above ground)
position.initial = [ox.Free(float(_r_init_s[0])),
                    ox.Free(float(_r_init_s[1])),
                    ox.Free(float(_r_init_s[2]))]
position.final   = [float(_r_final_s[0]), float(_r_final_s[1]), float(_r_final_s[2])]

velocity = ox.State("velocity", shape=(3,))
# _v_box = 150.0 / R_SCALE
velocity.max = [ 2.0,  2.0,  2.0]
velocity.min = [-2.0, -2.0, -2.0]
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

# Aerodynamics are disabled in this variant: neither the drag force nor the
# aero moment about the centre of pressure enters the model. (The canonical
# ``6DoF_pdg_stc_senss.py`` keeps the moment; ``6DoF_pdg_stc.py`` keeps both.)

# ── Inertia tensors (diagonal, scaled) ────────────────────────────────────────
# Mass-varying model: J_eff = J_s × m_s  →  ω̇ = J_inv_s @ (τ/m_s − ω × J_s @ ω)
J_B_ox     = ox.Diag(ox.Concat(float(_J_B_s[0]),     float(_J_B_s[1]),     float(_J_B_s[2])))
J_B_inv_ox = ox.Diag(ox.Concat(float(_J_B_inv_s[0]), float(_J_B_inv_s[1]), float(_J_B_inv_s[2])))
r_cm_ox    = ox.Concat(float(_r_cm_s[0]), float(_r_cm_s[1]), float(_r_cm_s[2]))
g_I_ox     = ox.Concat(float(_g_I_s[0]),  float(_g_I_s[1]),  float(_g_I_s[2]))

torque = cross(r_cm_ox, T_B)
gyro   = cross(angular_velocity, J_B_ox @ angular_velocity)

# ── Dynamics (all in scaled units) ────────────────────────────────────────────
dynamics = {
    "mass":             -_alpha_m_s * T,
    "position":          velocity,
    "velocity":          CBI.T @ T_B / mass[0] + g_I_ox,
    "attitude":          attitude_dot,
    "angular_velocity":  J_B_inv_ox @ (torque / mass[0] - gyro),
}

# ── Shared sub-expressions for constraints ────────────────────────────────────
# Tilt: ||[q_x, q_y]||²  (matches notebook's ||[x[8], x[9]]||² in [w,x,y,z] ordering)
tilt_sq   = attitude[0]**2 + attitude[1]**2
speed     = ox.linalg.Norm(velocity)
omega_sq  = ox.linalg.Norm(angular_velocity)**2
z_alt     = position[2] - float(_r_final_s[2])   # height above touchdown (R_I_FINAL)

# Position relative to glideslope apex (touchdown)
pos_rel_x = position[0] - float(_r_gs_apex_s[0])
pos_rel_y = position[1] - float(_r_gs_apex_s[1])
pos_rel_z = position[2] - float(_r_gs_apex_s[2])
r_xy_norm = ox.linalg.Norm(ox.Concat(pos_rel_x, pos_rel_y))

# Position relative to LoS target (below touchdown)
pos_los_x = position[0] - float(_r_los_target_s[0])
pos_los_y = position[1] - float(_r_los_target_s[1])
pos_los_z = position[2] - float(_r_los_target_s[2])
r_los_norm = ox.linalg.Norm(ox.Concat(pos_los_x, pos_los_y, pos_los_z))

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


def stc(*factors, weight: float = 1.0, penalty="huber"):
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
    return ox.ctcs((weight * prod) <= 0, penalty=penalty)


# ── Trigger indicators (relu, > 0 when active) ────────────────────────────────
T_alt100 = relu(_alt_h1_s  - z_alt)            # h < 100 m
T_alt220 = relu(_alt_h2a_s - z_alt)            # h < 220 m (LOS, alpha)
T_altgt2 = relu(z_alt - _alt_done_s)           # h >   2 m (finalizer)
T_vlt35  = relu(_spd_trig_s - speed)           # ||v|| < 35 m/s
T_vgt35  = relu(speed - _spd_trig_s)           # ||v|| > 35 m/s
T_tlt60  = relu(_tilt_sq_trig - tilt_sq)       # tilt < 60°
T_tgt60  = relu(tilt_sq - _tilt_sq_trig)       # tilt > 60°

# ── Constraint violations (relu, > 0 when violated) ───────────────────────────
C_gimbal_p = relu( de - _delta_stc_rad)                 # δ_e ≤ +1°
C_gimbal_n = relu(-de - _delta_stc_rad)                 # δ_e ≥ −1°
C_gs_stc   = relu(r_xy_norm * _tan_gs_stc - pos_rel_z)  # tight glideslope (apex at touchdown)
C_omega    = relu(omega_sq - _omega_sq_stc)             # tight angular rate
C_tilt     = relu(tilt_sq - _tilt_sq_stc)               # tight tilt
C_spd      = relu(speed - _v_stc_s)                     # tight speed
C_los      = relu(r_los_norm * _cos_psi_stc - r_dot_los)  # LOS cone (toward R_I_LOS)
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
constraints.append(ox.ctcs(tilt_sq - _tilt_sq_max <= 0, idx=1))          # tilt ≤ THETA_MAX_DEG
# angular rate
constraints.append(ox.ctcs(omega_sq - W_B_MAX_RAD_S**2 <= 0, penalty="huber", idx=0))
# glideslope 55°
constraints.append(ox.ctcs(r_xy_norm * _tan_gs_max - pos_rel_z <= 0, penalty="huber", idx=0))
constraints.append(ox.ctcs(_m_dry_s - mass[0] <= 0, penalty="huber", idx=0))  # dry-mass floor

# ── Compound state-triggered constraints (notebook cell 39) ───────────────────
# h < 100 m → tight gimbal deflection (|δ_e| ≤ 1°)
constraints.append(stc(T_alt100, C_gimbal_p, weight=W_GIMBAL))
constraints.append(stc(T_alt100, C_gimbal_n, weight=W_GIMBAL))
# h < 100 m AND h > 2 m → tight glideslope cone
constraints.append(stc(T_alt100, T_altgt2, C_gs_stc, weight=W_GS))
# h < 100 m → tight tilt
constraints.append(stc(T_alt100, C_tilt, weight=W_TILT))
# h < 100 m → tight angular rate and tight speed
constraints.append(stc(T_alt100, C_omega, weight=W_OMEGA))
constraints.append(stc(T_alt100, C_spd,   weight=W_SPD))
# h < 220 m AND h > 2 m → LOS boresight cone toward R_I_LOS (below touchdown)
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
    max=1.5 * _t_f_guess,
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
        "lam_cost": 1e-4,

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
_VISER_UP_AXIS = (0.0, 0.0, 1.0)
GS_HALFANGLE_DEG = 90.0 - GS_MAX_DEG
GS_STC_HALFANGLE_DEG = 90.0 - GS_STC_DEG

# DEM terrain patch — slide it under the trajectory with the Viser GUI sliders,
# then paste the values you settle on back here.
DEM_PLACEMENT = DemPlacement(origin_m=(39.0, -175.0, -244.0), grid=NATIVE_GRID)

SCALES = CstcScales(r_scale=R_SCALE, m_scale=M_SCALE)
LIMITS = CstcLimits(
    gs_apex_m=R_I_GS_APEX,
    los_apex_m=R_I_LOS,
    pad_z_m=float(R_I_FINAL[2]),
    alt_trigger_h1_m=ALT_TRIGGER_H1_M,
    alt_trigger_h2_m=ALT_TRIGGER_H2_M,
    spd_stc_trig=SPD_STC_TRIG,
    theta_stc_trig_deg=THETA_STC_TRIG,
    t_max=T_MAX,
    t_min=T_MIN,
    t_max_aft=T_MAX_AFT,
    t_min_aft=T_MIN_AFT,
    v_stc_cons=V_STC_CONS,
    theta_max_deg=THETA_MAX_DEG,
    theta_stc_deg=THETA_STC_DEG,
    delta_engine_max_deg=DELTA_ENGINE_MAX_DEG,
    delta_stc_deg=DELTA_STC_DEG,
    w_b_max_rad_s=W_B_MAX_RAD_S,
    omega_stc_rad_s=OMEGA_STC_RAD_S,
    gs_max_deg=GS_MAX_DEG,
    gs_stc_deg=GS_STC_DEG,
    delta_boresight_max_deg=DELTA_BORESIGHT_MAX_DEG,
    los_stc_deg=LOS_STC_DEG,
    m_dry=M_DRY,
)

SITE_MARKERS = [
    ("/los_target", R_I_LOS, (255, 220, 80), 0.08),
    ("/landing_pad", R_I_FINAL, (50, 255, 80), 0.12),
]
ALT_TRIGGERS = [
    (ALT_TRIGGER_H2_M, (80, 160, 255), "h < 200 m → LOS boresight cone"),
    (ALT_TRIGGER_H1_M, (255, 80, 80), "h < 100 m → tight gimbal/tilt/ω/speed/GS"),
]


def add_terrain(server, *, ambient: float = 0.0) -> None:
    """Overlay the DEM patch and a landing-site summary onto ``server``.

    The first-person sensor window needs a brighter ambient floor than the
    third-person one, which is looking at the terrain edge-on under grazing
    light; everything else about the patch is identical between the two.
    """
    add_dem_terrain(
        server,
        placement=DEM_PLACEMENT,
        shading=DemShading(ambient=ambient),
        scene_scale=SCENE_SCALE,
    )
    with server.gui.add_folder("Info"):
        server.gui.add_markdown(
            f"**Landing target**  \n"
            f"X = {R_I_FINAL[0]:.2f} m   Y = {R_I_FINAL[1]:.2f} m  \n"
            f"Altitude (Z) = **{R_I_FINAL[2]:.2f} m**  \n\n" + dem_info_markdown(DEM_PLACEMENT)
        )


def launch_viser_servers(result) -> None:
    """Create trajectory, sensor FPV, SCP convergence, and snapshot viser servers."""
    pos = np.asarray(result.trajectory["position"])
    node_indices = node_trigger_indices(result.nodes, LIMITS, scales=SCALES)

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
    traj_server = handle.server
    _, update_viewcone = add_los_viewcone(
        traj_server, result, half_angle_deg=LOS_STC_DEG, scale=VIEWCONE_SCALE
    )
    add_animation_controls(
        traj_server, handle.traj_time, handle.update_callbacks + [update_viewcone], loop=True
    )

    add_terrain(traj_server)
    add_camera_fov_slider(traj_server, initial_deg=CAMERA_FOV_DEG)

    # Two cones share the scene, so each needs its own scene path: the wide
    # always-on glideslope and the tight one the h < 100 m cSTC switches to.
    add_glideslope_cone(
        traj_server,
        apex=tuple((R_I_GS_APEX / SCENE_SCALE).tolist()),
        height=float(np.max(pos[:, 2])) * 1.15,
        glideslope_angle_deg=GS_HALFANGLE_DEG,
        axis=_VISER_UP_AXIS,
        color=(80, 200, 80),
        opacity=0.12,
        name="/constraints/glideslope_wide",
    )
    add_glideslope_cone(
        traj_server,
        apex=tuple((R_I_GS_APEX / SCENE_SCALE).tolist()),
        height=ALT_TRIGGER_H1_M / SCENE_SCALE,
        glideslope_angle_deg=GS_STC_HALFANGLE_DEG,
        axis=_VISER_UP_AXIS,
        color=(255, 180, 40),
        opacity=0.15,
        name="/constraints/glideslope_tight",
    )

    add_altitude_trigger_discs(
        traj_server,
        pos,
        center_xy=(R_I_FINAL[0], R_I_FINAL[1]),
        base_z_m=float(R_I_FINAL[2]),
        scene_scale=SCENE_SCALE,
        triggers=ALT_TRIGGERS,
    )
    add_site_markers(traj_server, SITE_MARKERS, scene_scale=SCENE_SCALE)
    add_cstc_phase_markers(traj_server, pos, node_indices, n_nodes=N, limits=LIMITS)

    scp_server = create_scp_animated_plotting_server(
        result,
        position_slice=slice(1, 4),
        attitude_slice=slice(7, 11),
        show_attitudes=True,
        attitude_stride=3,
        attitude_axes_length=ATTITUDE_AXES_LENGTH,
        frame_duration_ms=80,
        scene_scale=1.0,
    )
    snap_server = create_snapshot_plotting_server(
        result,
        attitude_axes_length=ATTITUDE_AXES_LENGTH,
        show_body_frame=True,
        initial_n_snapshots=6,
        show_grid=True,
        background_color=(240, 240, 245),
    )
    for server in (scp_server, snap_server):
        add_altitude_trigger_discs(
            server,
            pos,
            center_xy=(R_I_FINAL[0], R_I_FINAL[1]),
            base_z_m=float(R_I_FINAL[2]),
            scene_scale=SCENE_SCALE,
            triggers=ALT_TRIGGERS,
        )

    create_sensor_fpv_server(
        result,
        handle.traj_time,
        markers=SITE_MARKERS,
        scene_scale=SCENE_SCALE,
        decorate_scene=lambda server: add_terrain(server, ambient=0.45),
        fov_deg=CAMERA_FOV_DEG,
    )
    print("  Sensor FPV view — open the second viser URL printed above")

    traj_server.sleep_forever()


def export_trajectory_rpy_csv(
    result,
    path: str = "cstc_trajectory_rpy.csv",
    *,
    degrees: bool = True,
    slowdown: int = 2,
) -> str:
    """Write the trajectory in the SENSS gimbal rig's frame.

    Positions are the solver's scaled coordinates (not multiplied by
    ``R_SCALE``), with the rig's x/y axes taken from the model's y and -x.  The
    rig's roll is the model's pitch, and its pitch is the model's roll less the
    90° that trims the gimbal's mechanical zero — see the report accompanying
    this example set before changing that offset.  ``slowdown`` stretches
    wall-clock time and interpolates back to the original sample rate so the
    rig can replay the descent slower than real time.
    """
    traj = result.trajectory
    pos_model = np.asarray(traj["position"], dtype=np.float64)
    rpy = quat_xyzw_to_rpy(np.asarray(traj["attitude"], dtype=np.float64), degrees=degrees)

    pos = np.stack([pos_model[:, 1], -pos_model[:, 0], pos_model[:, 2]], axis=-1)
    roll, pitch = rpy[:, 0].copy(), rpy[:, 1].copy()
    rpy[:, 0] = pitch
    rpy[:, 1] = roll - 90.0

    t, (pos, rpy) = resample_uniform(traj["time"], [pos, rpy], slowdown)
    return write_trajectory_rpy_csv(path, t, pos, rpy)


def plot_cstc_results(result, *, show: bool = True, save_prefix: str = "cstc_stc") -> tuple:
    """Nine-panel states/controls figure and the speed-coloured 3-D path."""
    from openscvx.plotting.publication import show_plotly_with_latin_modern

    fig_panel = plot_cstc_panel(result, LIMITS, scales=SCALES)
    fig_3d = plot_cstc_trajectory_3d(
        result,
        scales=SCALES,
        markers=[("Landing pad", R_I_FINAL, "lime")],
        title="6-DoF PDG Trajectory (true cSTC)",
    )

    if save_prefix:
        save_plotly_figure(fig_panel, f"{save_prefix}_states_controls")
        save_plotly_figure(fig_3d, f"{save_prefix}_trajectory_3d")

    if show:
        show_plotly_with_latin_modern(fig_panel)
        show_plotly_with_latin_modern(fig_3d)

    return fig_panel, fig_3d


if __name__ == "__main__":
    problem.settings.dev.debug = True   # bypass JAX export (see repo issue)
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    from openscvx.plotting import plot_controls, plot_states

    plot_states(result).show()
    plot_controls(result).show()

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
    export_trajectory_rpy_csv(result)

    prepare_for_viser(result, scales=SCALES, scene_scale=SCENE_SCALE)
    launch_viser_servers(result)
