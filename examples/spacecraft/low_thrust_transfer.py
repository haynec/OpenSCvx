"""Low-Thrust Orbit Transfer using Modified Equinoctial Elements.

Faithful reproduction of the **GPOPS-II User's Guide, Section 5.2** benchmark
(Patterson & Rao, GPOPS-II User's Guide v2.3, 2016), itself taken from
Betts, *Practical Methods for Optimal Control Using Nonlinear Programming*,
SIAM, 2009.

The spacecraft starts in a **circular low-Earth orbit** (i = 28.5 deg) and must
reach a **highly elliptic, inclined orbit** (e = 0.7355, i = 63.4 deg) while
maximizing final mass. State is in Modified Equinoctial Elements (MEE) and the
control is the thrust direction in radial-transverse-normal (RTN) coordinates
together with a throttle. The dynamics include the J2/J3/J4 Earth-oblateness
perturbations.

The GPOPS-II constants and boundary values are given in English units
(ft, lbf, lbm); here they are converted to **metric km-kg-s** with explicit
factors so the physics matches the reference verbatim. In SI the English
``g0·T/w`` thrust-acceleration factor becomes the standard ``F/m``, and the mass
flow becomes ``F/Ve`` with ``Ve = Isp·g0``.

States  x = (p, f, g, h, k, L, m):
  p [km]   — semi-latus rectum
  f, g     — MEE eccentricity vector components
  h, k     — MEE inclination vector components
  L [rad]  — true longitude (unwrapped, monotonically increasing)
  m [kg]   — spacecraft mass

Controls:
  u = (u_r, u_t, u_h) — thrust unit-direction (RTN);  path constraint ‖u‖ = 1
  tau                  — throttle parameter, tau ∈ [-50, 0]; thrust ∝ (1 + 0.01 tau)

Dynamics (Eqs. 13-30 of the reference):
  ẋ = A(x) Δ + b,     ṁ = -F (1 + 0.01 tau) / Ve
with Δ = Δ_g + Δ_T, where Δ_g is the J2-J4 oblateness acceleration projected
into the RTN frame and Δ_T = F (1 + 0.01 tau) / m · u is the thrust force.

Objective:  minimize J = -m(t_f)   (i.e. maximize final mass).

Terminal (event) constraints (Eq. 17):
  p(t_f) = p_f
  f² + g²              = 0.73550320568829²     (final eccentricity)
  h² + k²              = 0.61761258786099²     (final tan(i/2))
  f·h + g·k            = 0
  g·h - k·f            ≤ 0

The active transcription below keeps the benchmark geometry but replaces the
bilinear final-perigee orientation constraints with their affine equivalent,
using the fixed terminal magnitudes: f = (e_f/χ_f) k and g = -(e_f/χ_f) h.

Initial guess (Section 5.3): propagate the dynamics from the initial condition
with a fixed throttle tau = -25 and the control aligned with the inertial
velocity, u = Qᵣᵀ v / ‖v‖, then sample onto the discretization nodes.

References:
  - "A set of modified equinoctial orbit elements," Walker, Ireland & Owens, 1985.
  - "Optimal Low-Thrust Interplanetary Trajectories by Direct Method Techniques,"
    Kluever, 1997.
  - "Practical Methods for Optimal Control and Estimation Using Nonlinear
    Programming," Betts, 2010.
"""

import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem
from openscvx.plotting import plot_controls, plot_scp_iterations, plot_states

# ── Unit conversion factors (GPOPS-II English → metric) ──────────────────────
FT = 0.0003048  # km per ft
LBF = 4.4482216152605  # N per lbf
LBM = 0.45359237  # kg per lbm
G0 = 9.80665  # m/s², standard gravity

# ── Physical constants in km-kg-s, from GPOPS-II Eq. (30) ─────────────────────
mu_phys = 1.407645794e16 * FT**3  # km³/s², Earth gravitational parameter
Re_phys = 20925662.73 * FT  # km, Earth equatorial radius
F_thrust = 4.446618e-3 * LBF  # N, thrust magnitude
Isp = 450.0  # s, specific impulse
Ve_phys = Isp * G0 / 1000.0  # km/s, effective exhaust velocity
J2 = 1082.639e-6
J3 = -2.565e-6
J4 = -1.608e-6

# Physical boundary values (km-kg-s) before non-dimensionalization.
p0_phys = 21837080.052835 * FT  # km
pf_phys = 40007346.015232 * FT  # km
p_min_phys = 20000000.0 * FT  # km, lower state bound on p
p_max_phys = 60000000.0 * FT  # km, upper state bound on p  (= LU below)
m0_phys = 1.0 * LBM  # kg (w0 = 1 lbm)

# ── Canonical (non-dimensional) units ─────────────────────────────────────────
# The dynamics are integrated in canonical units so the integrator sees O(1)
# states. Length is scaled by p_max so normalized p ranges in [p_min/p_max, 1] =
# [1/3, 1]; time by TU = sqrt(LU³/μ) so the normalized gravitational parameter is
# exactly 1; mass by the initial mass so normalized mass ranges in [0.1, 1].
LU = p_max_phys  # km, length unit (=> p_n = p/LU ∈ [1/3, 1])
TU = np.sqrt(LU**3 / mu_phys)  # s, time unit  (=> μ_n = 1)
VU = LU / TU  # km/s, velocity unit
AU = LU / TU**2  # km/s², acceleration unit (= μ_phys / LU²)
MU = m0_phys  # kg, mass unit (initial mass)

# ── Non-dimensional constants used by the (integrated) dynamics ───────────────
mu = 1.0  # μ_phys / (LU³/TU²) ≡ 1 by construction
Re = Re_phys / LU
# thrust accel [AU] = c_thrust · (1 + 0.01 tau) / mass_n
#   (physical F/m in km/s², divided by the acceleration unit AU)
c_thrust = (F_thrust / 1000.0) / (MU * AU)
# mass flow [MU/TU] = -c_mdot · (1 + 0.01 tau)
#   (physical F/Ve in kg/s, expressed per mass unit and per time unit)
c_mdot = (F_thrust / (Isp * G0)) * TU / MU

# ── Boundary conditions (non-dimensional), GPOPS-II Eq. (17) ─────────────────
# Initial: circular LEO, i = 28.5 deg.
p0_val = p0_phys / LU
f0_val = 0.0
g0_val = 0.0
h0_val = -0.25396764647494
k0_val = 0.0
L0_val = np.pi  # rad (dimensionless)
m0_val = m0_phys / MU  # = 1.0

# Final: highly elliptic, inclined orbit.
pf_val = pf_phys / LU
ecc_f = 0.73550320568829  # sqrt(f_f² + g_f²)
chi_f = 0.61761258786099  # sqrt(h_f² + k_f²)

# ── Problem size and free-time guess (time in TU) ─────────────────────────────
N = 75  # discretization nodes
tf_guess = 90000.0 / TU  # Section 5.3
tf_min = 50000.0 / TU
tf_max = 100000.0 / TU

# Largest allowed true longitude (Lmax = 9·2π, GPOPS-II bounds).
L_max = 9.0 * 2.0 * np.pi


# ── Reference dynamics in numpy (used only to build the initial guess) ─────────
# This mirrors the symbolic dynamics below exactly; keeping the two in one file
# makes the propagated guess dynamically consistent with the optimized model.
def _grav_rtn_and_vel(p, f, g, h, k, L):
    """Oblateness acceleration in RTN and the inertial velocity (numpy)."""
    q = 1.0 + f * np.cos(L) + g * np.sin(L)
    r = p / q
    alpha2 = h * h - k * k
    s2 = 1.0 + h * h + k * k
    cL, sL = np.cos(L), np.sin(L)

    rX = (r / s2) * (cL + alpha2 * cL + 2 * h * k * sL)
    rY = (r / s2) * (sL - alpha2 * sL + 2 * h * k * cL)
    rZ = (2 * r / s2) * (h * sL - k * cL)
    rVec = np.array([rX, rY, rZ])
    rMag = np.sqrt(rX**2 + rY**2 + rZ**2)
    rXZMag = np.sqrt(rX**2 + rZ**2)

    smp = np.sqrt(mu / p)
    vX = -(1.0 / s2) * smp * (sL + alpha2 * sL - 2 * h * k * cL + g - 2 * f * h * k + alpha2 * g)
    vY = -(1.0 / s2) * smp * (-cL + alpha2 * cL + 2 * h * k * sL - f + 2 * g * h * k + alpha2 * f)
    vZ = (2.0 / s2) * smp * (h * cL + k * sL + f * h + g * k)
    vVec = np.array([vX, vY, vZ])

    rCrossv = np.cross(rVec, vVec)
    rCrossvMag = np.linalg.norm(rCrossv)
    rCrossvCrossr = np.cross(rCrossv, rVec)

    ir = rVec / rMag
    it = rCrossvCrossr / (rCrossvMag * rMag)
    ih = rCrossv / rCrossvMag

    enir = ir[2]
    enen = np.array([-enir * ir[0], -enir * ir[1], 1.0 - enir * ir[2]])
    inn = enen / np.linalg.norm(enen)

    sinphi = rZ / rXZMag
    cosphi = np.sqrt(1.0 - sinphi**2)

    P2 = (3 * sinphi**2 - 2) / 2
    P3 = (5 * sinphi**3 - 3 * sinphi) / 2
    P4 = (35 * sinphi**4 - 30 * sinphi**2 + 3) / 8
    dP2 = 3 * sinphi
    dP3 = (15 * sinphi - 3) / 2  # verbatim from GPOPS-II source
    dP4 = (140 * sinphi**3 - 60 * sinphi) / 8

    sumn = (Re / r) ** 2 * dP2 * J2 + (Re / r) ** 3 * dP3 * J3 + (Re / r) ** 4 * dP4 * J4
    sumr = 3 * (Re / r) ** 2 * P2 * J2 + 4 * (Re / r) ** 3 * P3 * J3 + 5 * (Re / r) ** 4 * P4 * J4
    deltagn = -(mu * cosphi / r**2) * sumn
    deltagr = -(mu / r**2) * sumr

    dgv = deltagn * inn - deltagr * ir
    return np.array([ir @ dgv, it @ dgv, ih @ dgv]), ir, it, ih, vVec


def _eom_guess(t, x):
    """MEE EOM with throttle tau = -25 and velocity-aligned control (numpy)."""
    p, f, g, h, k, L, m = x
    Dg, ir, it, ih, vVec = _grav_rtn_and_vel(p, f, g, h, k, L)
    vmag = np.linalg.norm(vVec)
    ur, ut, uh = ir @ vVec / vmag, it @ vVec / vmag, ih @ vVec / vmag
    thr = c_thrust * (1 + 0.01 * (-25.0)) / m
    D1, D2, D3 = Dg + np.array([thr * ur, thr * ut, thr * uh])

    q = 1.0 + f * np.cos(L) + g * np.sin(L)
    s2 = 1.0 + h * h + k * k
    cL, sL = np.cos(L), np.sin(L)
    smp = np.sqrt(p / mu)
    return [
        (2 * p / q) * smp * D2,
        smp * sL * D1 + smp * ((q + 1) * cL + f) / q * D2 - smp * g / q * (h * sL - k * cL) * D3,
        -smp * cL * D1 + smp * ((q + 1) * sL + g) / q * D2 + smp * f / q * (h * sL - k * cL) * D3,
        smp * s2 * cL / (2 * q) * D3,
        smp * s2 * sL / (2 * q) * D3,
        smp / q * (h * sL - k * cL) * D3 + np.sqrt(mu * p) * (q / p) ** 2,
        -c_mdot * (1 + 0.01 * (-25.0)),
    ]


def build_initial_guess():
    """Propagate the reference control law and sample onto N uniform-time nodes."""
    x0 = [p0_val, f0_val, g0_val, h0_val, k0_val, L0_val, m0_val]
    sol = solve_ivp(
        _eom_guess,
        [0.0, tf_guess],
        x0,
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
        max_step=tf_guess / 2000,
    )
    t_nodes = np.linspace(0.0, tf_guess, N)
    Xg = sol.sol(t_nodes)  # (7, N)

    # Recover the velocity-aligned control direction at each node.
    u_g = np.zeros((N, 3))
    for i in range(N):
        _, ir, it, ih, vVec = _grav_rtn_and_vel(*Xg[:6, i])
        vmag = np.linalg.norm(vVec)
        u_g[i] = [ir @ vVec / vmag, it @ vVec / vmag, ih @ vVec / vmag]
    tau_g = np.full((N, 1), -25.0)
    return t_nodes, Xg.T, u_g, tau_g


t_nodes, Xg, u_guess, tau_guess = build_initial_guess()

# State bounds (non-dimensional); also used to clip the open-loop guess.
state_bounds = {
    "p": (p_min_phys / LU, p_max_phys / LU),  # = (1/3, 1)
    "f": (-1.0, 1.0),
    "g": (-1.0, 1.0),
    "h": (-1.0, 1.0),
    "k": (-1.0, 1.0),
    "L": (L0_val, L_max),
    "m": (0.1, m0_val),  # normalized mass ∈ [0.1, 1]
}
_names = ["p", "f", "g", "h", "k", "L", "m"]
Xg = np.column_stack([np.clip(Xg[:, i], *state_bounds[nm]) for i, nm in enumerate(_names)])

# ── States ───────────────────────────────────────────────────────────────────
p_el = ox.State("p", (1,))
p_el.initial, p_el.final = np.array([p0_val]), np.array([pf_val])  # p(t_f) fixed
p_el.min, p_el.max = np.array([state_bounds["p"][0]]), np.array([state_bounds["p"][1]])
p_el.guess = Xg[:, 0:1]

f_el = ox.State("f", (1,))
f_el.initial, f_el.final = np.array([f0_val]), [ox.Free(float(Xg[-1, 1]))]
f_el.min, f_el.max = np.array([-1.0]), np.array([1.0])
f_el.guess = Xg[:, 1:2]

g_el = ox.State("g", (1,))
g_el.initial, g_el.final = np.array([g0_val]), [ox.Free(float(Xg[-1, 2]))]
g_el.min, g_el.max = np.array([-1.0]), np.array([1.0])
g_el.guess = Xg[:, 2:3]

h_el = ox.State("h", (1,))
h_el.initial, h_el.final = np.array([h0_val]), [ox.Free(float(Xg[-1, 3]))]
h_el.min, h_el.max = np.array([-1.0]), np.array([1.0])
h_el.guess = Xg[:, 3:4]

k_el = ox.State("k", (1,))
k_el.initial, k_el.final = np.array([k0_val]), [ox.Free(float(Xg[-1, 4]))]
k_el.min, k_el.max = np.array([-1.0]), np.array([1.0])
k_el.guess = Xg[:, 4:5]

L_el = ox.State("L", (1,))
L_el.initial, L_el.final = np.array([L0_val]), [ox.Free(float(Xg[-1, 5]))]
L_el.min, L_el.max = np.array([L0_val]), np.array([L_max])
L_el.guess = Xg[:, 5:6]

m_el = ox.State("mass", (1,))
m_el.initial = np.array([m0_val])
m_el.final = [("maximize", 0.4 * LBM)]  # objective: maximize final mass
m_el.min, m_el.max = np.array([state_bounds["m"][0]]), np.array([m0_val])
m_el.guess = Xg[:, 6:7]

states = [p_el, f_el, g_el, h_el, k_el, L_el, m_el]

# ── Controls ─────────────────────────────────────────────────────────────────
u = ox.Control("u", shape=(3,), parameterization="zoh")
u.min, u.max = np.full(3, -1.0), np.full(3, 1.0)
u.guess = u_guess

tau = ox.Control("tau", shape=(1,), parameterization="zoh")
tau.min, tau.max = np.array([-50.0]), np.array([0.0])
tau.guess = tau_guess

controls = [u, tau]

# ── Symbolic MEE dynamics with J2-J4 oblateness ──────────────────────────────
p, f, g, h, k, L = p_el[0], f_el[0], g_el[0], h_el[0], k_el[0], L_el[0]
m = m_el[0]
ur, ut, uh = u[0], u[1], u[2]
tau_s = tau[0]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm3(a):
    return ox.Sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


q = 1.0 + f * ox.Cos(L) + g * ox.Sin(L)
r = p / q
alpha2 = h * h - k * k
s2 = 1.0 + h * h + k * k
cL, sL = ox.Cos(L), ox.Sin(L)

# Inertial position (ECI), Eq. (20) helpers.
rX = (r / s2) * (cL + alpha2 * cL + 2 * h * k * sL)
rY = (r / s2) * (sL - alpha2 * sL + 2 * h * k * cL)
rZ = (2 * r / s2) * (h * sL - k * cL)
rVec = (rX, rY, rZ)
rMag = _norm3(rVec)
rXZMag = ox.Sqrt(rX * rX + rZ * rZ)

# Inertial velocity (ECI).
smp = ox.Sqrt(mu / p)
vX = -(1.0 / s2) * smp * (sL + alpha2 * sL - 2 * h * k * cL + g - 2 * f * h * k + alpha2 * g)
vY = -(1.0 / s2) * smp * (-cL + alpha2 * cL + 2 * h * k * sL - f + 2 * g * h * k + alpha2 * f)
vZ = (2.0 / s2) * smp * (h * cL + k * sL + f * h + g * k)
vVec = (vX, vY, vZ)

# RTN basis vectors i_r, i_θ, i_h, Eq. (24).
rCrossv = _cross(rVec, vVec)
rCrossvMag = _norm3(rCrossv)
rCrossvCrossr = _cross(rCrossv, rVec)
ir = tuple(c / rMag for c in rVec)
it = tuple(c / (rCrossvMag * rMag) for c in rCrossvCrossr)
ih = tuple(c / rCrossvMag for c in rCrossv)

# Local North direction i_n, Eq. (26), with e_n = (0, 0, 1).
enir = ir[2]
enen = (-enir * ir[0], -enir * ir[1], 1.0 - enir * ir[2])
enenMag = _norm3(enen)
inn = tuple(c / enenMag for c in enen)

# Geocentric latitude and Legendre polynomials, Eqs. (27)-(28).
sinphi = rZ / rXZMag
cosphi = ox.Sqrt(1.0 - sinphi * sinphi)
P2 = (3 * sinphi * sinphi - 2) / 2
P3 = (5 * sinphi**3 - 3 * sinphi) / 2
P4 = (35 * sinphi**4 - 30 * sinphi * sinphi + 3) / 8
dP2 = 3 * sinphi
dP3 = (15 * sinphi - 3) / 2  # verbatim from GPOPS-II source
dP4 = (140 * sinphi**3 - 60 * sinphi) / 8

sumn = (Re / r) ** 2 * dP2 * J2 + (Re / r) ** 3 * dP3 * J3 + (Re / r) ** 4 * dP4 * J4
sumr = 3 * (Re / r) ** 2 * P2 * J2 + 4 * (Re / r) ** 3 * P3 * J3 + 5 * (Re / r) ** 4 * P4 * J4
deltagn = -(mu * cosphi / (r * r)) * sumn
deltagr = -(mu / (r * r)) * sumr

# Oblateness acceleration in ECI then projected into RTN, Eqs. (22)-(25).
dgv = (
    deltagn * inn[0] - deltagr * ir[0],
    deltagn * inn[1] - deltagr * ir[1],
    deltagn * inn[2] - deltagr * ir[2],
)
Deltag1 = _dot(ir, dgv)
Deltag2 = _dot(it, dgv)
Deltag3 = _dot(ih, dgv)

# Thrust acceleration in RTN, Eq. (29) (SI form: F/m, no g0 factor).
thr = c_thrust * (1.0 + 0.01 * tau_s) / m
DeltaT1, DeltaT2, DeltaT3 = thr * ur, thr * ut, thr * uh

D1 = Deltag1 + DeltaT1
D2 = Deltag2 + DeltaT2
D3 = Deltag3 + DeltaT3

# Equations of motion, Eqs. (13)-(14)/(18)-(19).
p_dot = (2 * p / q) * smp * D2
f_dot = smp * sL * D1 + smp * ((q + 1) * cL + f) / q * D2 - smp * g / q * (h * sL - k * cL) * D3
g_dot = -smp * cL * D1 + smp * ((q + 1) * sL + g) / q * D2 + smp * f / q * (h * sL - k * cL) * D3
h_dot = smp * s2 * cL / (2 * q) * D3
k_dot = smp * s2 * sL / (2 * q) * D3
L_dot = smp / q * (h * sL - k * cL) * D3 + ox.Sqrt(mu * p) * (q / p) ** 2
m_dot = -c_mdot * (1.0 + 0.01 * tau_s)

dynamics = {
    "p": p_dot,
    "f": f_dot,
    "g": g_dot,
    "h": h_dot,
    "k": k_dot,
    "L": L_dot,
    "mass": m_dot,
}

# ── Constraints ───────────────────────────────────────────────────────────────
# OpenSCvx only accepts *affine* equality constraints. Nonlinear equalities are
# written as matched inequalities (≤ and ≥), while affine equalities are kept hard.
constraints = []


# Path constraint: thrust direction has unit norm, ‖u‖ = 1 (Eq. 15). ‖u‖ ≤ 1 is
# convex (enforced continuously); ‖u‖ ≥ 1 is non-convex (nodal, virtual buffer).
unorm = _norm3((ur, ut, uh))
constraints.append(ox.ctcs(unorm <= 1.0))

# Terminal (event) constraints at the final node, Eq. (17).
fN, gN, hN, kN = f_el[0], g_el[0], h_el[0], k_el[0]
nodal_eqs = [
    (fN**2 + gN**2, ecc_f**2),  # final eccentricity
    (hN**2 + kN**2, chi_f**2),  # final tan(i/2)
    (fN * hN + gN * kN, 0.0),
]
for expr, val in nodal_eqs:
    constraints.append(ox.NodalConstraint(expr == val, nodes=[N - 1]))
constraints.append(ox.NodalConstraint(gN * hN - kN * fN <= 0.0, nodes=[N - 1]))

# ── Time (free final time) ────────────────────────────────────────────────────
time = ox.Time(
    initial=0.0,
    final=ox.Free(tf_guess),
    min=0.0,
    max=3 * tf_max,
    guess=t_nodes.reshape(-1, 1),
    time_dilation_min=tf_min,
    time_dilation_max=tf_max,
    uniform_time_grid=True,
)
time.time_dilation_guess = np.full((N, 1), tf_guess)

# ── Problem ───────────────────────────────────────────────────────────────────
problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    constraints=constraints,
    time=time,
    N=N,
    float_dtype="float64",
    discretizer={"ode_solver": "Dopri8"},
    algorithm={
        "autotuner": ox.AugmentedLagrangian(),
        "k_max": 200,
        "lam_prox": 1e0,
        "lam_vc": 1e2,
        "lam_vb": 1e1,
    },
    solver={"cvx_solver": "QOCO", "solver_args": {}},
)


def mee_to_cartesian(p_a, f_a, g_a, h_a, k_a, L_a):
    """MEE → ECI position [LU] using the GPOPS-II Eq. (20) convention.

    Inputs are non-dimensional (``p`` in LU); multiply the result by ``LU`` for km.
    """
    q = 1.0 + f_a * np.cos(L_a) + g_a * np.sin(L_a)
    r = p_a / q
    alpha2 = h_a**2 - k_a**2
    s2 = 1.0 + h_a**2 + k_a**2
    cL, sL = np.cos(L_a), np.sin(L_a)
    x = (r / s2) * (cL + alpha2 * cL + 2 * h_a * k_a * sL)
    y = (r / s2) * (sL - alpha2 * sL + 2 * h_a * k_a * cL)
    z = (2 * r / s2) * (h_a * sL - k_a * cL)
    return x, y, z


def plot_final_orbit(results):
    """3-D ECI plot overlaying the initial guess and the converged transfer (km)."""
    import plotly.graph_objects as go

    tr = results.trajectory
    p, f, g, h, k, L = (np.asarray(tr[s]).flatten() for s in ("p", "f", "g", "h", "k", "L"))
    x, y, z = mee_to_cartesian(p, f, g, h, k, L)
    x, y, z = x * LU, y * LU, z * LU  # canonical (LU) -> km

    # Initial guess (module-level Xg, columns p,f,g,h,k,L,mass), also in km.
    xg, yg, zg = mee_to_cartesian(Xg[:, 0], Xg[:, 1], Xg[:, 2], Xg[:, 3], Xg[:, 4], Xg[:, 5])
    xg, yg, zg = xg * LU, yg * LU, zg * LU

    # Earth sphere
    u_s = np.linspace(0.0, 2.0 * np.pi, 60)
    v_s = np.linspace(0.0, np.pi, 30)
    xe = Re_phys * np.outer(np.cos(u_s), np.sin(v_s))
    ye = Re_phys * np.outer(np.sin(u_s), np.sin(v_s))
    ze = Re_phys * np.outer(np.ones_like(u_s), np.cos(v_s))

    fig = go.Figure()
    fig.add_trace(
        go.Surface(x=xe, y=ye, z=ze, colorscale="Blues", showscale=False, opacity=0.5, name="Earth")
    )
    fig.add_trace(
        go.Scatter3d(
            x=xg,
            y=yg,
            z=zg,
            mode="lines",
            line=dict(color="lightgray", width=2, dash="dash"),
            name="Initial guess",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color="orange", width=3),
            name="Transfer (solved)",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[x[0]],
            y=[y[0]],
            z=[z[0]],
            mode="markers",
            marker=dict(size=6, color="green"),
            name="Start (LEO)",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[x[-1]],
            y=[y[-1]],
            z=[z[-1]],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="Final orbit",
        )
    )
    fig.update_layout(
        title="Low-Thrust Transfer — ECI Position [km]",
        scene=dict(
            xaxis_title="x [km]", yaxis_title="y [km]", zaxis_title="z [km]", aspectmode="data"
        ),
    )
    return fig


if __name__ == "__main__":
    problem.initialize()
    problem.solve()
    results = problem.post_process()

    # Results are in canonical units; convert back to physical for reporting.
    tf_sol = float(np.asarray(results.t_final).reshape(-1)[0]) * TU  # s
    mf_n = float(results.trajectory["mass"][-1, 0])  # normalized mass
    mf_sol = mf_n * MU  # kg
    print(f"Transfer time : {tf_sol:.0f} s  ({tf_sol / 3600:.2f} hr)")
    print(f"Final mass    : {mf_sol:.6f} kg  (propellant {(m0_val - mf_n) * MU:.6f} kg)")

    plot_final_orbit(results).show()
    plot_states(results).show()
    plot_controls(results).show()
    plot_scp_iterations(results).show()
