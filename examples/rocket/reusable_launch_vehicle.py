import os
import sys

import jax

# use float64
jax.config.update("jax_enable_x64", True)

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx.plotting import plot_controls, plot_scp_convergence_histories, plot_states

# Number of discretization nodes
n = 200

# Physical constants from Betts reference
# Earth radius (m)
R_earth = 6371203.92  # m

# Vehicle parameters
S = 249.9091776  # Vehicle reference area (m^2)
mass = 92079.2525560557  # Vehicle mass (kg)

# Aerodynamic coefficients
cl = np.array([-0.2070, 1.6756])  # Lift coefficient parameters [cl0, cl1]
cd = np.array([0.0785, -0.3529, 2.0400])  # Drag coefficient parameters [cd0, cd1, cd2]

# Atmospheric model parameters
rho0 = 1.225570827014494  # Sea level atmospheric density (kg/m^3)
H = 7254.24  # Density scale height (m)

# Earth gravitational parameter
mu_earth = 3.986031954093051e14  # m^3/s^2

# Boundary conditions from problem statement (matching Betts reference exactly)
# Heights: h(0) = 79248 m, h(t_f) = 24384 m
h_initial = 79248.0  # m
h_final = 24384.0  # m

# Initial conditions from reference
lon0 = 0.0  # deg
lat0 = 0.0  # deg
speed0 = 7802.88  # m/s (reference uses 7802.88, problem statement says 7803)
speedf = 762.0  # m/s
fpa0 = -1.0 * np.pi / 180  # deg
fpaf = -5.0 * np.pi / 180  # deg
azi0 = 90.0 * np.pi / 180  # deg (reference shows +90, not -90)
azif = -90.0 * np.pi / 180  # deg (reference shows -90)

# Define state components
# h: height above Earth surface (m)
h = ox.State("h", shape=(1,))
h.max = np.array([h_initial])  # m (max is initial altitude, matching reference radMax = rad0)
h.min = np.array([0.0])  # m
h.initial = np.array([h_initial])
h.final = np.array([h_final])
h.guess = np.linspace(h_initial, h_final, n).reshape(-1, 1)

# θ: longitude (radians)
theta = ox.State("theta", shape=(1,))
theta.max = np.array([np.pi])  # rad
theta.min = np.array([-np.pi])  # rad
theta.initial = np.array([lon0])  # 0 deg
theta.final = [ox.Free(lon0 + 10.0 * np.pi / 180)]  # Free (reference guess: lon0+10 deg)
theta.guess = np.linspace(lon0, lon0 + 10.0 * np.pi / 180, n).reshape(-1, 1)

# φ: latitude (radians) - this is what we want to maximize (crossrange)
phi = ox.State("phi", shape=(1,))
phi.max = np.array([70.0 * np.pi / 180])  # rad (70 deg)
phi.min = np.array([-70.0 * np.pi / 180])  # rad (-70 deg)
phi.initial = np.array([lat0])  # 0 deg
phi.final = [
    ox.Maximize(lat0 + 10.0 * np.pi / 180)
]  # Maximize final latitude (reference guess: lat0+10 deg)
phi.guess = np.linspace(lat0, lat0 + 10.0 * np.pi / 180, n).reshape(-1, 1)
phi.scaling_max = np.array([15.0 * np.pi / 180])
phi.scaling_min = np.array([0.0 * np.pi / 180])

# v: velocity magnitude (m/s)
v = ox.State("v", shape=(1,))
v.max = np.array([45000.0])  # m/s
v.min = np.array([10.0])  # m/s
v.initial = np.array([speed0])  # m/s (reference uses 7802.88)
v.final = np.array([speedf])  # m/s
v.guess = np.linspace(speed0, speedf, n).reshape(-1, 1)
v.scaling_max = np.array([1e4])
v.scaling_min = np.array([0.0])

# γ: flight path angle (radians)
gamma = ox.State("gamma", shape=(1,))
gamma.max = np.array([80.0 * np.pi / 180])  # rad (80 deg)
gamma.min = np.array([-80.0 * np.pi / 180])  # rad (-80 deg)
gamma.initial = np.array([fpa0])  # -1 deg
gamma.final = np.array([fpaf])  # -5 deg
gamma.guess = np.linspace(fpa0, fpaf, n).reshape(-1, 1)

# ψ: heading angle (azimuth, radians)
psi = ox.State("psi", shape=(1,))
psi.max = np.array([np.pi])  # rad (180 deg)
psi.min = np.array([-np.pi])  # rad (-180 deg)
psi.initial = np.array([azi0])  # 90 deg (reference: +90 deg)
psi.final = [ox.Free(azif)]  # Free (reference guess: -90 deg)
psi.guess = np.linspace(azi0, azif, n).reshape(-1, 1)

# Define control components
# α: angle of attack (radians)
alpha = ox.Control("alpha", shape=(1,))
alpha.max = np.array([90.0 * np.pi / 180])  # rad (90 deg)
alpha.min = np.array([-90.0 * np.pi / 180])  # rad (-90 deg)
alpha.guess = np.repeat(np.array([[0.0]]), n, axis=0)

# σ: bank angle (radians)
sigma = ox.Control("sigma", shape=(1,))
sigma.max = np.array([1.0 * np.pi / 180])  # rad (1 deg)
sigma.min = np.array([-90.0 * np.pi / 180])  # rad (-90 deg)
sigma.guess = np.linspace(sigma.min, sigma.max, n)

# Define list of all states and controls
states = [h, theta, phi, v, gamma, psi]
controls = [alpha, sigma]

# Physical parameters
m = ox.Parameter("m", value=mass)  # kg

# Define dynamics as dictionary mapping state names to their derivatives
# Dynamics from Betts reference MATLAB code (using r = h + R_earth):
# \dot{r} = v sin γ
# θ̇ = (v cos γ sin ψ) / ((h + R_earth) cos φ)
# φ̇ = (v cos γ cos ψ) / (h + R_earth)
# v̇ = -D - g sin γ
# γ̇ = (L cos σ - cos γ (g - v²/r)) / v
# ψ̇ = (L sin σ / cos γ + v² cos γ sin ψ tan φ / r) / v

# Convert height to radial distance: r = h (h already represents r = altitude + R_earth)
r = h + R_earth

# Compute gravitational acceleration: g = μ / r^2
g = mu_earth / (r**2)  # m/s^2

# Aerodynamic model (matching Betts reference exactly)
# Lift and drag coefficients
CL = cl[0] + cl[1] * alpha[0]  # Linear lift coefficient: CL = cl0 + cl1*α
CD = (
    cd[0] + cd[1] * alpha[0] + cd[2] * alpha[0] ** 2
)  # Quadratic drag coefficient: CD = cd0 + cd1*α + cd2*α²

# Atmospheric density: ρ = ρ0 * exp(-altitude/H)
# Note: altitude = h - R_earth = r - R_earth, where r = h (since h is already r = altitude + R_earth)
altitude = h  # Height above Earth surface
rho = rho0 * ox.Exp(-altitude / H)  # kg/m^3

# Dynamic pressure: q = 0.5 * ρ * v²
q = 0.5 * rho * v**2  # Pa

# Lift and drag forces (normalized by mass): L = q*S*CL/m, D = q*S*CD/m
# These are accelerations (m/s^2), matching the MATLAB code
L = q * S * CL / m  # m/s^2 (acceleration)
D = q * S * CD / m  # m/s^2 (acceleration)

# Dynamics equations matching MATLAB code exactly
# Note: The reference uses division by v and cos(gamma), which are bounded away from zero
# by the state bounds (v >= 10 m/s, |gamma| <= 80 deg, |phi| <= 70 deg)
dynamics = {
    "h": v * ox.Sin(gamma),
    "theta": (v * ox.Cos(gamma) * ox.Sin(psi)) / (r * ox.Cos(phi)),
    "phi": (v * ox.Cos(gamma) * ox.Cos(psi)) / r,
    "v": -D - g * ox.Sin(gamma),
    "gamma": (L * ox.Cos(sigma) - ox.Cos(gamma) * (g - v**2 / r)) / v,
    "psi": (
        L * ox.Sin(sigma) / ox.Cos(gamma) + v**2 * ox.Cos(gamma) * ox.Sin(psi) * ox.Tan(phi) / r
    )
    / v,
}

# Generate box constraints for all states
# Use separate idx for each constraint and huber penalty to prevent huge penalty accumulation
constraints = []

# State constraints - use huber penalty and separate groups to prevent penalty blowup
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max),
            ox.ctcs(state.min <= state),
        ]
    )

# Build the problem
# Time is free (t_f is free)
time = ox.Time(
    initial=0.0,
    final=ox.Free(1000.0),  # Free final time with initial guess
    min=0.0,
    max=3000.0,  # Maximum time in seconds
)

problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    algorithm={
        "autotuner": ox.AugmentedLagrangian(eta_lambda=1E2),
        "lam_vc": 2e0,
        "lam_prox": 5e-1,
        "lam_cost": 9e-1,
    },
    float_dtype="float64",
)

problem.algorithm.k_max = 500

plotting_dict = {
    "R_earth": R_earth,
    "mu_earth": mu_earth,
    "S": S,
    "mass": mass,
    "rho0": rho0,
    "H": H,
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    plot_scp_convergence_histories(results).show()

    plot_states(results).show()
    plot_controls(results).show()
