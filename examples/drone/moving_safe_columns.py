"""6DoF quadrotor with STL moving safe columns.

Lift of ``examples/double_integrator/moving_safe_zones.py``: the vehicle must
stay inside at least one of several moving vertical columns (extruded circles)
while flying from start to goal. Each column center follows a time-varying xy
path (PCHIP spline in physical time); the specification is

    Always( Or(in_column_0, ..., in_column_N) )

over the full horizon. Columns are infinite in z — only the horizontal offset
from the moving center is constrained.

The columns are a time-segmented relay: each covers a different time window
along a feasible hand-off path. Only column 0 covers the start and only
column 5 covers the goal, so the vehicle must switch columns to finish.

Run::

    python examples/drone/moving_safe_columns.py
"""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from examples.drone._plotting import create_moving_safe_columns_server
from openscvx import Problem

# Discretization — same scale as the 2D moving-safe-zones example
N = 30
TOTAL_TIME = 10.0

Z_CRUISE = 2.0
START = np.array([-4.0, 0.0, Z_CRUISE])
GOAL = np.array([4.0, 0.0, Z_CRUISE])

COLUMN_RADIUS = 0.6

T_KNOTS = np.array([0.0, 2.0, 3.5, 5.0, 7.5, 10.0])

RELAY_KEYFRAMES_XY = [
    START[:2],
    np.array([-2.0, 0.9]),
    np.array([-0.8, 0.2]),
    np.array([0.5, -1.0]),
    np.array([2.5, -0.7]),
    GOAL[:2],
]
RELAY_KEYFRAMES = [np.array([*xy, Z_CRUISE]) for xy in RELAY_KEYFRAMES_XY]
RELAY_NODES = [0, 6, 10, 15, 22, N - 1]

# Same xy knot paths as BALL_PATHS in moving_safe_zones.py
COLUMN_PATHS = [
    (
        np.array([-4.0, -2.0, -0.5, 1.0, 2.2, 2.8]),
        np.array([0.0, 0.9, 1.1, 1.3, 1.4, 1.5]),
    ),
    (
        np.array([-3.5, -2.2, -0.8, 0.2, 1.3, 2.2]),
        np.array([-0.4, 0.55, 0.2, -0.15, -0.35, -0.55]),
    ),
    (
        np.array([-2.8, -1.4, 0.3, 1.6, 2.8, 3.4]),
        np.array([1.25, 1.15, 1.0, 0.9, 0.8, 0.7]),
    ),
    (
        np.array([-2.5, -1.4, 0.0, 0.5, 2.5, 3.1]),
        np.array([-1.1, -1.0, -0.95, -1.0, -0.7, -0.65]),
    ),
    (
        np.array([-1.5, -0.2, 1.2, 2.3, 3.1, 3.5]),
        np.array([1.15, 1.05, 0.95, 0.75, 0.55, 0.5]),
    ),
    (
        np.array([-1.2, 0.3, 1.4, 2.4, 3.4, 4.0]),
        np.array([-0.55, -0.45, -0.35, -0.25, -0.1, 0.0]),
    ),
]

MASS = 1.0
G_CONST = -9.18
GRAVITY = np.array([0.0, 0.0, G_CONST], dtype=np.float64)
THRUST_MAX = 4.179446268 * 9.81
HOVER_THRUST = MASS * abs(G_CONST)

# States
position = ox.State("position", shape=(3,))
position.min = np.array([-6.0, -2.5, 0.5])
position.max = np.array([6.0, 2.5, 4.0])
position.initial = START
position.final = GOAL

velocity = ox.State("velocity", shape=(3,))
velocity.min = np.array([-8.0, -8.0, -4.0])
velocity.max = np.array([8.0, 8.0, 4.0])
velocity.initial = np.array([0.0, 0.0, 0.0])
velocity.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

attitude = ox.State("attitude", shape=(4,))
attitude.max = np.array([1.0, 1.0, 1.0, 1.0])
attitude.min = np.array([-1.0, -1.0, -1.0, -1.0])
attitude.initial = [ox.Free(1.0), ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]
attitude.final = [ox.Free(1.0), ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = np.array([10.0, 10.0, 10.0])
angular_velocity.min = np.array([-10.0, -10.0, -10.0])
angular_velocity.initial = np.array([0.0, 0.0, 0.0])
angular_velocity.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

# Fixed final time so column centers stay synchronized with physical time.
time = ox.Time(
    initial=0.0,
    final=TOTAL_TIME,
    min=0.0,
    max=TOTAL_TIME,
    uniform_time_grid=True,
)

thrust_force = ox.Control("thrust_force", shape=(3,))
thrust_force.max = np.array([0.0, 0.0, THRUST_MAX])
thrust_force.min = np.array([0.0, 0.0, 0.0])

torque = ox.Control("torque", shape=(3,))
torque.max = np.array([18.665, 18.665, 0.55562])
torque.min = np.array([-18.665, -18.665, -0.55562])
torque.guess = np.zeros((N, 3))

states = [position, velocity, attitude, angular_velocity, time]
controls = [thrust_force, torque]

column_radius = ox.Parameter("column_radius", shape=(), value=COLUMN_RADIUS)

in_column_predicates = []
for col_x, col_y in COLUMN_PATHS:
    center_xy = ox.Concat(
        ox.Cinterp(time[0], T_KNOTS, col_x, method="pchip"),
        ox.Cinterp(time[0], T_KNOTS, col_y, method="pchip"),
    )
    delta_xy = position[:2] - center_xy
    in_column_predicates.append(ox.Sum(delta_xy * delta_xy) <= column_radius * column_radius)

in_some_column = ox.stl.Or(*in_column_predicates)

constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

constraints.append(ox.stl.Always(in_some_column, (0, N - 1)).over())

J_b = jnp.array([1.0, 1.0, 1.0])
J_b_inv = 1.0 / J_b
J_b_diag = ox.linalg.Diag(J_b)
q_norm = ox.linalg.Norm(attitude)
attitude_normalized = attitude / q_norm

dynamics = {
    "position": velocity,
    "velocity": (1.0 / MASS) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
    + np.array([0.0, 0.0, G_CONST], dtype=np.float64),
    "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
    "angular_velocity": ox.linalg.Diag(J_b_inv)
    @ (torque - ox.spatial.SSM(angular_velocity) @ J_b_diag @ angular_velocity),
    "time": 1.0,
}

# Position / velocity seed along the relay; attitude / thrust from differential flatness
# (identity attitude cannot produce horizontal accel with body-z-only thrust).
dt = TOTAL_TIME / (N - 1)
position.guess = ox.init.linspace(keyframes=RELAY_KEYFRAMES, nodes=RELAY_NODES)
velocity.guess = np.gradient(position.guess, dt, axis=0)
accel_guess = np.gradient(velocity.guess, dt, axis=0)


def _orientation_from_accel(accel: np.ndarray) -> np.ndarray:
    """Unit quaternion aligning body +z with specific thrust (diff. flatness)."""
    thrust_dir = accel - GRAVITY
    norm = float(np.linalg.norm(thrust_dir))
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    z_des = thrust_dir / norm
    z_body = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_body, z_des)
    dot = float(np.dot(z_body, z_des))
    if dot < -0.999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    q = np.array([1.0 + dot, cross[0], cross[1], cross[2]])
    return q / np.linalg.norm(q)


att_guess = np.array([_orientation_from_accel(accel_guess[k]) for k in range(N)])
att_guess /= np.linalg.norm(att_guess, axis=1, keepdims=True)
attitude.guess = att_guess
angular_velocity.guess = np.zeros((N, 3))

thrust_mag = MASS * np.linalg.norm(accel_guess - GRAVITY[None, :], axis=1)
thrust_mag = np.clip(thrust_mag, 0.0, THRUST_MAX)
thrust_force.guess = np.column_stack([np.zeros(N), np.zeros(N), thrust_mag])

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=N,
    algorithm={
        "autotuner": ox.ConstantProximalWeight(),
        "lam_prox": 5e-3,
        "lam_vc": 2e1,
        "lam_cost": 1e-3,
    },
    float_dtype="float64",
)

plotting_data = {
    "column_radius": COLUMN_RADIUS,
    "t_knots": T_KNOTS,
    "column_paths": COLUMN_PATHS,
    "column_interp_method": "pchip",
    "start": START,
    "goal": GOAL,
}

if __name__ == "__main__":
    n_cols = len(COLUMN_PATHS)
    print("6DoF Quadrotor — Moving Safe Columns (STL Or + Always)")
    print("=" * 60)
    print(f"Start: {START}, Goal: {GOAL}")
    print(f"Columns: {n_cols}, radius: {COLUMN_RADIUS} (infinite height, xy constraint)")
    print("Design: time-segmented relay (mandatory column hand-offs)")
    print(f"Spec: Always( Or(in_column_0, ..., in_column_{n_cols - 1}) ) over the horizon")
    print("=" * 60)

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_data)

    converged = getattr(results, "converged", "?")
    print(f"Converged: {converged}")

    server = create_moving_safe_columns_server(results)
    server.sleep_forever()
