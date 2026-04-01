"""Dubins car path planning with STL waypoint inside a 2-norm ball.

This example demonstrates the use of GMSR-based STL operators to encode a
waypoint specified as a 2-norm ball, together with a time-windowed speed
restriction:

- At nodes 10–14, the position must stay inside a 2-norm ball (the waypoint)
- While in this waypoint window, the speed is restricted to a lower value
  (here, from the global max 10.0 down to 1.0)

This showcases how to use `ox.stl` operators over a specific node interval to
impose waypoint and speed-profile constraints.
"""

import os
import sys

import jax.numpy as jnp
import numpy as np

# Add grandparent directory to path to import examples.plotting
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting import plot_dubins_car, plot_velocity_vs_waypoint
from openscvx import Problem
from openscvx.plotting import plot_controls, plot_states

n = 20
total_time = 3.0  # Total simulation time

# Define state components
position = ox.State("position", shape=(2,))  # 2D position [x, y]
position.min = np.array([-5.0, -5.0])
position.max = np.array([5.0, 5.0])
position.initial = np.array([0, -2])
position.final = np.array([0, 2])

theta = ox.State("theta", shape=(1,))  # Heading angle
theta.min = np.array([-2 * jnp.pi])
theta.max = np.array([2 * jnp.pi])
theta.initial = np.array([0])
theta.final = [ox.Free(0)]

speed = ox.State("speed", shape=(1,))  # Forward speed
speed.min = np.array([0])
speed.max = np.array([10])
speed.initial = [ox.Free(10)]
speed.final = [ox.Free(10)]

# Define control components
acceleration = ox.Control("acceleration", shape=(1,))  # Acceleration
acceleration.min = np.array([-10])
acceleration.max = np.array([10])
acceleration.guess = np.zeros((n, 1))


angular_rate = ox.Control("angular_rate", shape=(1,))  # Angular velocity
angular_rate.min = np.array([-5])
angular_rate.max = np.array([5])
angular_rate.guess = np.zeros((n, 1))

# Define list of all states and controls
states = [position, speed, theta]
controls = [acceleration, angular_rate]

# Define Parameters with initial values for obstacle radius and center
obs_center = ox.Parameter("obs_center", shape=(2,), value=np.array([-2.0, 0.0]))
obs_radius = ox.Parameter("obs_radius", shape=(), value=1.0)

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Define a waypoint as a 2-norm ball and enforce it only over nodes 10–14,
# together with a stronger speed restriction in that same window.
waypoint_region = ox.linalg.Norm(position - obs_center) <= obs_radius

safety_region = ox.linalg.Norm(position - obs_center) <= 1.6

# Use the scalar speed magnitude so the STL residuals are 0-D scalars.
# Globally the control bound is 10.0; inside the waypoint window we restrict to 1.0.
slow_speed_in_waypoint = ox.linalg.Norm(speed) <= 1.0

speed_constraint = ox.stl.IfThen(safety_region, slow_speed_in_waypoint)

constraints.append(
    speed_constraint.over(
        (0, n - 1),
    )
)

constraints.append(
    ox.ctcs(waypoint_region).over(
        (8, 12),
    )
)


# Define normal dynamics (no conditional logic here)
dynamics = {
    "position": ox.Concat(
        speed * ox.Sin(theta),  # x_dot
        speed * ox.Cos(theta),  # y_dot
    ),
    "speed": acceleration,
    "theta": angular_rate,
}


# Build the problem (parameters auto-collected from Parameter objects)
time = ox.Time(
    initial=0.0,
    final=ox.Minimize(total_time),
    min=0.0,
    max=5.0,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    algorithm={"autotuner": ox.RampProximalWeight()},
    float_dtype="float64",
    licq_max=1e-10,
)

# problem.algorithm.ep_vc = 1e-6
# problem.algorithm.ep_tr = 5e-4
problem.algorithm.lam_prox = 5e-4
problem.algorithm.lam_vc = 1e0
problem.algorithm.lam_cost = 5e-5

plotting_dict = {
    "obs_radius": problem.parameters["obs_radius"],
    "obs_center": problem.parameters["obs_center"],
    "safety_threshold": 1.6,
    "reduced_speed": 1.0,
}

if __name__ == "__main__":
    print("Dubins Car with STL Waypoint and Time-Windowed Speed Constraint")
    print("=" * 70)
    print("Max velocity is 10.0 globally, but at nodes 10–14 the car must")
    print("be inside the 2-norm waypoint ball and its speed is restricted to 1.0.")
    print("=" * 70)

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    # Plot trajectory
    plot_dubins_car(results, problem.settings).show()
    plot_states(results).show()
    plot_controls(results).show()

    # Plot velocity vs distance to waypoint, using safety radius and reduced speed
    plot_velocity_vs_waypoint(results, problem.settings).show()
