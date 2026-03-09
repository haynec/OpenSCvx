"""Dubins car path planning with conditional velocity constraint using Cond operator.

This example demonstrates the use of the Cond operator for conditional logic
in constraints. The car has different maximum velocity limits based on proximity
to the obstacle:

- When far from obstacle: higher maximum velocity (10.0)
- When near obstacle: lower maximum velocity (5.0) for safer navigation

This showcases how to use ox.Cond for JAX-traceable conditional branching
in constraint expressions.
"""

import os
from shutil import SpecialFileError
import sys

import jax.numpy as jnp
import numpy as np

# Add grandparent directory to path to import examples.plotting
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.car.dubins_car_waypoint_stl import acceleration
import openscvx as ox
from examples.plotting import plot_dubins_car, plot_velocity_vs_distance
from openscvx.plotting import plot_states, plot_controls
from openscvx import Problem

n = 8
total_time = 0.8  # Total simulation time

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
speed.guess = np.ones((n, 1)) * 10.0

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
obs_center = ox.Parameter("obs_center", shape=(2,), value=np.array([-0.25, 0.0]))
obs_radius = ox.Parameter("obs_radius", shape=(), value=1.0)

# Safety distance threshold for conditional velocity constraint
safety_threshold = 1.1  # Distance from obstacle where we reduce max velocity

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Add obstacle avoidance constraint
constraints.append(ox.ctcs(obs_radius <= ox.linalg.Norm(position - obs_center)))

# Add the conditional velocity constraint as an STL specification:
# - If the car is within safety_threshold of the obstacle, then speed <= 5.0
# - If the car is farther than safety_threshold, then speed <= 10.0
distance_to_obstacle = ox.linalg.Norm(position - obs_center)

near_obstacle = distance_to_obstacle <= safety_threshold

# Use the scalar speed magnitude so the IfThen residuals are 0-D scalars
slow_speed = ox.linalg.Norm(speed) <= 5.0

velocity_rule = ox.stl.IfThen(near_obstacle, slow_speed, c = 1e-12)

constraints.append(
    velocity_rule.over(
        (0, n - 1),
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
    float_dtype="float64",
)

plotting_dict = {
    "obs_radius": problem.parameters["obs_radius"],
    "obs_center": problem.parameters["obs_center"],
    "safety_threshold": safety_threshold,
}

if __name__ == "__main__":
    print("Dubins Car with Conditional Velocity Constraint (using Cond operator)")
    print("=" * 70)
    print("Max velocity is 5.0 when within 2.0 units of obstacle, else 10.0")
    print("=" * 70)

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    # Plot trajectory
    plot_dubins_car(results, problem.settings).show()
    plot_states(results).show()
    plot_controls(results).show()

    # Plot velocity vs distance to obstacle
    plot_velocity_vs_distance(results, problem.settings).show()
