"""Hohmann transfer with impulsive delta-v controls (LEO → GEO).

This example mirrors the Hohmann transfer calculation from
[_Orbital Mechanics & Astrodynamics_](https://orbital-mechanics.space/orbital-maneuvers/hohmann-transfer-example.html)
by Bryan Weber and embeds it in a trajectory optimization problem that uses impulsive
delta-v controls.

We consider a planar, two-body Earth-centered problem:

- Initial circular orbit: 250 km altitude LEO
- Final circular orbit: GEO with 1 sidereal-day period
- 2D position/velocity states in an inertial frame (km, km/s)
- Central gravity with mu = 3.986e5 km^3/s^2
- Two impulsive delta-v maneuvers applied at the initial and final nodes
- A scalar cost state that accumulates the L2 norm of each impulse and is
  minimized at the final time (approximate minimum-fuel objective)

Continuous dynamics between impulses:

    x_dot = vx
    y_dot = vy
    vx_dot = -mu * x / r^3
    vy_dot = -mu * y / r^3

Discrete dynamics at impulsive nodes:

- position unchanged
- velocity += delta_v
- cost += ||delta_v||

The transfer time is fixed to the Hohmann half-period of the transfer ellipse.
In the __main__ block we also compute and print the analytic Hohmann Δv and
propellant mass from Weber's example for comparison.
"""

import os
import sys

import numpy as np

# Add grandparent directory to path to import examples.plotting
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting_viser import create_hohmann_transfer_server
from openscvx import Problem
from openscvx.plotting import plot_controls, plot_scp_iterations, plot_states

# Problem configuration (match Weber's LEO → GEO example)
n = 15

mu = 3.986e5  # km^3 / s^2
R_E = 6378.0  # km
r_leo = 250.0 + R_E

sidereal_day = 86164.0905  # s
r_cubed = mu * sidereal_day**2 / (4 * np.pi**2)
r_geo = r_cubed ** (1.0 / 3.0)

r1 = r_leo
r2 = r_geo

# Hohmann transfer half-period for the transfer ellipse (fixed final time)
a_transfer = 0.5 * (r1 + r2)
T_transfer = np.pi * np.sqrt(a_transfer**3 / mu)

# States: planar position, planar velocity, and scalar accumulated cost
position = ox.State("position", shape=(2,))
position.initial = np.array([r1, 0.0])
position.final = np.array([-r2, 0.0])  # Opposite side for half-period transfer
position.min = np.array([-(1.5 * r2), -(1.5 * r2)])
position.max = np.array([1.5 * r2, 1.5 * r2])

# Initial guess: follow an approximate Hohmann-like arc that never passes
# through the origin (to avoid r ≈ 0 causing singular dynamics).
theta_guess = np.linspace(0.0, np.pi, n)
radius_guess = np.linspace(r1, r2, n)
position_guess = np.stack(
    [radius_guess * np.cos(theta_guess), radius_guess * np.sin(theta_guess)],
    axis=1,
)
position.guess = position_guess

velocity = ox.State("velocity", shape=(2,))
v_c1 = np.sqrt(mu / r1)
v_c2 = np.sqrt(mu / r2)
velocity.initial = np.array([0.0, v_c1])
velocity.final = np.array([0.0, -v_c2])
velocity_min_mag = -2.0 * max(v_c1, v_c2)
velocity_max_mag = 2.0 * max(v_c1, v_c2)
velocity.min = np.array([velocity_min_mag, velocity_min_mag])
velocity.max = np.array([velocity_max_mag, velocity_max_mag])
velocity.guess = np.tile(np.array([0.0, (v_c1 + v_c2) / 2.0]), (n, 1))

cost = ox.State("cost", shape=(1,))
cost.initial = np.array([0.0])
# Minimize final accumulated impulse norm with a loose upper bound
cost.final = [("minimize", 10.0)]
cost.min = np.array([0.0])
cost.max = np.array([10.0])
cost.guess = np.zeros((n, 1))

# Impulsive delta-v control applied only at the first and last nodes
dv = ox.Control(
    "delta_v",
    shape=(2,),
    parameterization="impulsive",
    nodes=[0, n - 1],
)
dv_bound = 5.0  # km/s
dv.min = np.array([-dv_bound, -dv_bound])
dv.max = np.array([dv_bound, dv_bound])
dv.guess = np.zeros((n, 2))

states = [position, velocity, cost]
controls = [dv]

# Continuous dynamics: planar two-body gravity with no continuous thrust.
# We clamp the radius away from zero for robustness in the linearization.
r = ox.linalg.Norm(position)

dynamics = {
    "position": velocity,
    "velocity": ox.Concat(
        -mu * position[0] / r**3,
        -mu * position[1] / r**3,
    ),
    "cost": 0.0,
}

# Discrete dynamics at impulsive nodes: update velocity and accumulate cost.
# Use a small epsilon inside the norm to avoid NaNs in derivatives at dv = 0.
eps_impulse = 1e-6
d_impulse = ox.linalg.Norm(dv + eps_impulse)

dynamics_discrete = {
    "position": position,
    "velocity": velocity + dv,
    "cost": cost + d_impulse,
}

# Box constraints on all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

time = ox.Time(
    initial=0.0,
    final=T_transfer,
    min=0.0,
    max=T_transfer,
)

problem = Problem(
    dynamics=dynamics,
    dynamics_discrete=dynamics_discrete,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
)

problem.discretizer.ode_solver = "Dopri8"
problem.settings.prp.dt = 10.0

plotting_dict = {}

if __name__ == "__main__":
    # Analytic Hohmann values from Weber's example for reference
    v_leo = np.sqrt(mu / r_leo)
    v_geo = np.sqrt(mu / r_geo)
    r_p = r_leo
    r_a = r_geo
    h_t = np.sqrt(2 * mu * r_a * r_p / (r_a + r_p))
    v_tp = h_t / r_p
    v_ta = h_t / r_a

    Delta_v_analytic = abs(v_geo - v_ta) + abs(v_tp - v_leo)
    I_sp = 450.5  # s
    goes_mass = 5192.0  # kg
    Delta_m_analytic = goes_mass * (1.0 - np.exp(-Delta_v_analytic / (I_sp * 9.81e-3)))

    print(f"Analytic Hohmann Δv ≈ {Delta_v_analytic:.3f} km/s")
    print(f"Analytic propellant mass ≈ {Delta_m_analytic:.3f} kg")

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    # Always show standard Plotly diagnostics.
    plot_states(results).show()
    plot_controls(results).show()
    plot_scp_iterations(results).show()

    server = create_hohmann_transfer_server(
        results,
        r1=r1,
        r2=r2,
        scene_scale=1000.0,  # km -> Mm-ish for nicer viewing
    )
    server.sleep_forever()
