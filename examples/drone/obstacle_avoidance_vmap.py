"""Double integrator obstacle avoidance with Vmap for parallel constraint evaluation.

This example demonstrates using ox.Vmap directly in constraints for data-parallel
obstacle avoidance with numerous spherical obstacles.

The approach:

1. Stack all obstacle centers into a single array
2. Use Vmap directly in the constraint to compute distances in parallel
3. The vector-valued constraint is enforced element-wise

Compare with:
  - obstacle_avoidance.py (manual loop over 3 obstacles)
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
    create_animated_plotting_server,
    create_scp_animated_plotting_server,
)
from openscvx import Problem

n = 6  # Number of nodes
total_time = 10.0  # Total time for the simulation

# =============================================================================
# State and Control Definitions
# =============================================================================

# 3D position
position = ox.State("position", shape=(3,))
position.max = np.array([15.0, 15.0, 15.0])
position.min = np.array([-15.0, -15.0, 0.0])
position.initial = np.array([-10.0, -10.0, 2.0])
position.final = np.array([10.0, 10.0, 2.0])

# 3D velocity
velocity = ox.State("velocity", shape=(3,))
velocity.max = np.array([10.0, 10.0, 10.0])
velocity.min = np.array([-10.0, -10.0, -10.0])
velocity.initial = np.array([0.0, 0.0, 0.0])
velocity.final = [("free", 0.0), ("free", 0.0), ("free", 0.0)]

# Control force
force = ox.Control("force", shape=(3,))
f_max = 20.0
force.max = np.array([f_max, f_max, f_max])
force.min = np.array([-f_max, -f_max, -f_max])

# Physical parameters
m = 1.0  # Mass
g = -9.81  # Gravity

# =============================================================================
# Obstacle Configuration (3D grid)
# =============================================================================

obstacle_radius_min, obstacle_radius_max = 1.0, 2.5

# Generate obstacle positions in a 3D grid pattern between start and goal
np.random.seed(42)
obstacle_centers = []

# Tweak number of obstacles by changing the grid sizes
n_rows = 4
n_cols = 4
n_lays = 4

# Create a 3D field of obstacles: rows (x) x columns (y) x layers (z)
for i in range(n_rows):
    for j in range(n_cols):
        for k in range(n_lays):
            # Base grid position
            x = -6.0 + i * 6.0
            y = -7.5 + j * 5.0
            z = 1.0 + k * 2.5  # Layers at z = 1.0, 3.5, 6.0
            # Add some randomness
            x += np.random.uniform(-1.0, 1.0)
            y += np.random.uniform(-1.0, 1.0)
            z += np.random.uniform(-0.5, 0.5)
            obstacle_centers.append([x, y, z])

n_obstacles = len(obstacle_centers)  # 36 obstacles

obstacle_centers = np.array(obstacle_centers)  # Shape: (n_obstacles, 3)
obstacle_radii = np.random.uniform(obstacle_radius_min, obstacle_radius_max, size=n_obstacles)

print(f"Created {n_obstacles} obstacles")
print(f"Obstacle centers shape: {obstacle_centers.shape}")

# =============================================================================
# Dynamics (simple double integrator)
# =============================================================================

dynamics = {
    "position": velocity,
    "velocity": (1.0 / m) * force + np.array([0.0, 0.0, g]),
}

# =============================================================================
# Constraints
# =============================================================================

states = [position, velocity]
controls = [force]

constraints = []

# Box constraints on states
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max),
            ox.ctcs(state.min <= state),
        ]
    )

# Box constraints on controls
constraints.extend(
    [
        force <= force.max,
        force.min <= force,
    ]
)

# =============================================================================
# Obstacle Avoidance
# =============================================================================

obstacle_avoidance = ox.ctcs(
    obstacle_radii
    <= ox.Vmap(
        lambda obs_center: ox.linalg.Norm(position - obs_center),
        batch=obstacle_centers,
    )
)
constraints.append(obstacle_avoidance)

# =============================================================================
# Initial Guesses
# =============================================================================

straight_line = np.linspace(position.initial, position.final, n)
position.guess = straight_line
velocity.guess = np.zeros((n, 3))
force.guess = np.tile([0.0, 0.0, -m * g], (n, 1))  # Hover thrust

# =============================================================================
# Problem Setup
# =============================================================================

time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
)

problem.settings.scp.ep_tr = 1e-3

# =============================================================================
# Solve and Visualize
# =============================================================================

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    # Store obstacle info for visualization
    results.update(
        {
            "obstacles_centers": [c for c in obstacle_centers],
            "obstacles_radii": [[1 / r, 1 / r, 1 / r] for r in obstacle_radii],
            "obstacles_axes": [np.eye(3) for _ in range(n_obstacles)],
        }
    )

    # Create viser visualization servers
    traj_server = create_animated_plotting_server(
        results,
        thrust_key="force",
        viewcone_scale=5.0,
    )
    scp_server = create_scp_animated_plotting_server(
        results,
        attitude_stride=3,
        frame_duration_ms=200,
    )

    # Keep servers running
    traj_server.sleep_forever()
