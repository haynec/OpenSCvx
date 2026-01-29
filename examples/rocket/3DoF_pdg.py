"""3-DOF Powered Descent Guidance (PDG) for planetary landing.

This example demonstrates optimal trajectory generation for a rocket performing
powered descent guidance, similar to SpaceX Falcon 9 or Blue Origin landings.
The problem includes:

- 3D position and velocity dynamics
- Fuel-optimal mass minimization
- Thrust magnitude and pointing constraints
- Glideslope constraint for safe landing approach
"""

import os
import sys

import numpy as np

# Add grandparent directory to path to import examples.plotting
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting_viser import (
    create_pdg_animated_plotting_server,
    create_scp_animated_plotting_server,
)
from openscvx import Problem
from openscvx.plotting import plot_controls, plot_projections_2d, plot_states, plot_vector_norm

n = 10
total_time = 95.0  # Total simulation time

# Define state components
position = ox.State("position", shape=(3,))  # 3D position [x, y, z]
v_max = 500 * 1e3 / 3600  # Maximum velocity in m/s (800 km/h converted to m/s)
position.max = np.array([3000, 3000, 3000])
position.min = np.array([-3000, -3000, 0])
position.initial = np.array([2000, 0, 1500])
position.final = np.array([0, 0, 0])
position.guess = np.linspace(position.initial, position.final, n)

velocity = ox.State("velocity", shape=(3,))  # 3D velocity [vx, vy, vz]
velocity.max = np.array([v_max, v_max, v_max])
velocity.min = np.array([-v_max, -v_max, -v_max])
velocity.initial = np.array([80, 30, -75])
velocity.final = np.array([0, 0, 0])
velocity.guess = np.linspace(velocity.initial, velocity.final, n)

mass = ox.State("mass", shape=(1,))  # Vehicle mass
mass.max = np.array([1905])
mass.min = np.array([1505])
mass.initial = np.array([1905])
mass.final = [("maximize", 1700)]
mass.scaling_min = np.array([1700])
# mass.scaling_max = np.array([1700])
mass.guess = np.linspace(mass.initial, 1690, n).reshape(-1, 1)

# Define control
thrust = ox.Control("thrust", shape=(3,))  # Thrust force vector [Tx, Ty, Tz]

T_bar = 3.1 * 1e3
T1 = 0.3 * T_bar
T2 = 0.8 * T_bar
n_eng = 6

# Set bounds on control
thrust.min = n_eng * np.array([-T_bar, -T_bar, -T_bar])
thrust.max = n_eng * np.array([T_bar, T_bar, T_bar])

# Set initial control guess
thrust.guess = np.repeat(np.expand_dims(np.array([0, 0, n_eng * (T2) / 2]), axis=0), n, axis=0)

# Define list of all states and controls
states = [position, velocity, mass]
controls = [thrust]


# Define Parameters for physical constants
g_e = 9.807  # Gravitational acceleration on Earth in m/s^2

# Create parameters for the problem
I_sp = ox.Parameter("I_sp", value=225.0)
g = ox.Parameter("g", value=3.7114)
theta = ox.Parameter("theta", value=27 * np.pi / 180)

# These will be computed symbolically in constraints
rho_min = n_eng * T1 * np.cos(theta.value)  # Minimum thrust-to-weight ratio
rho_max = n_eng * T2 * np.cos(theta.value)  # Maximum thrust-to-weight ratio

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max, idx=0),
            ox.ctcs(state.min <= state, idx=0),
        ]
    )

# Thrust magnitude constraints
constraints.extend(
    [
        ox.ctcs(rho_min <= ox.linalg.Norm(thrust), idx=1),
        ox.ctcs(ox.linalg.Norm(thrust) <= rho_max, idx=1),
    ]
)

# Thrust pointing constraint (thrust cant angle)
constraints.append(
    ox.ctcs(np.cos((180 - 40 * np.pi / 180)) <= thrust[2] / ox.linalg.Norm(thrust), idx=2)
)

# Glideslope constraint
constraints.append(
    ox.ctcs(ox.linalg.Norm(position[:2]) <= np.tan(86 * np.pi / 180) * position[2], idx=3)
)


# Define dynamics as dictionary mapping state names to their derivatives
g_vec = np.array([0, 0, 1], dtype=np.float64) * g  # Gravitational acceleration vector

dynamics = {
    "position": velocity,
    "velocity": thrust / mass[0] - g_vec,
    "mass": -ox.linalg.Norm(thrust) / (I_sp * g_e * ox.Cos(theta)),
}

# Build the problem
time = ox.Time(
    initial=0.0,
    final=("free", total_time),
    min=0.0,
    max=1e2,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    autotuner=ox.RampProximalWeight(),
)

problem.settings.scp.autotuner.ramp_factor = 1.04
problem.settings.scp.autotuner.lam_prox_max = 1e2

# Set solver parameters
problem.settings.scp.lam_cost = 5e-1
problem.settings.scp.lam_vc = 1.5e0
problem.settings.scp.lam_prox = 2e-1

problem.settings.dis.dis_type = "ZOH"

problem.settings.dis.solver = "Dopri8"


plotting_dict = {
    "rho_min": rho_min,
    "rho_max": rho_max,
    "glideslope_angle_deg": 86.0,
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    plot_states(results, ["position", "velocity"]).show()
    plot_controls(results, ["thrust"]).show()
    plot_vector_norm(results, "thrust", bounds=(rho_min, rho_max)).show()
    plot_projections_2d(results, velocity_var_name="velocity").show()

    # Create PDG trajectory visualization
    scene_scale = 100
    traj_server = create_pdg_animated_plotting_server(
        results,
        thrust_key="thrust",
        glideslope_angle_deg=86.0,
        scene_scale=1.0,
    )

    # Create SCP iteration visualization
    scp_server = create_scp_animated_plotting_server(
        results,
        frame_duration_ms=200,
        scene_scale=100.0,
    )

    # Keep servers running
    traj_server.sleep_forever()
