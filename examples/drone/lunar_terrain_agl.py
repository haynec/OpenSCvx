"""6DoF quadrotor A→B over SENNS lunar DEM at ~20 m AGL.

Fixed-horizon feasibility problem: fly from point A to point B in fixed time
while staying inside a thin above-ground-level (AGL) band relative to the real
SENNS lunar heightmap (``examples/rocket/senss/senns_dem.png``), sampled finely
onto the mission domain.

Terrain height enters the CTCS constraints via ``ox.Bilerp``. Gravity and
actuators match the other Earth-g quadrotor demos — the DEM supplies the
terrain geometry.

Run::

    python examples/drone/lunar_terrain_agl.py
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
from examples.drone._plotting import create_lunar_terrain_agl_server
from examples.drone._terrain import bilinear_height, load_senns_dem
from openscvx import Problem

# ── Mission ───────────────────────────────────────────────────────────────────
N = 50
TOTAL_TIME = 25.0  # s — fixed horizon
AGL_TARGET = 20.0  # m
AGL_TOL = 1.0  # m — advertised band: AGL ∈ [19, 21]

START_XY = np.array([-120.0, 0.0])
GOAL_XY = np.array([120.0, 40.0])

# Square world patch (SENNS PNG is square; keep 1:1 so features aren't stretched).
TERRAIN_HALF = 160.0
TERRAIN_X_MIN, TERRAIN_X_MAX = -TERRAIN_HALF, TERRAIN_HALF
TERRAIN_Y_MIN, TERRAIN_Y_MAX = -TERRAIN_HALF, TERRAIN_HALF

# Fine SENNS DEM sample (rocket dem_static uses 2048). Native PNG is 3938² —
# bump DEM_GRID toward that for even more detail (cost: compile / mesh size).
DEM_GRID = 2048
ELEV_SCALE = 50.0  # m — peak-to-peak relief after normalization

x_grid, y_grid, H = load_senns_dem(
    TERRAIN_X_MIN,
    TERRAIN_X_MAX,
    TERRAIN_Y_MIN,
    TERRAIN_Y_MAX,
    grid=DEM_GRID,
    elev_scale=ELEV_SCALE,
)

z_start = bilinear_height(START_XY[0], START_XY[1], x_grid, y_grid, H) + AGL_TARGET
z_goal = bilinear_height(GOAL_XY[0], GOAL_XY[1], x_grid, y_grid, H) + AGL_TARGET
START = np.array([START_XY[0], START_XY[1], z_start])
GOAL = np.array([GOAL_XY[0], GOAL_XY[1], z_goal])

# ── Vehicle (Earth g; same boxes as obstacle_avoidance.py) ────────────────────
MASS = 1.0
G_CONST = -9.18
GRAVITY = np.array([0.0, 0.0, G_CONST], dtype=np.float64)
THRUST_MAX = 4.179446268 * 9.81

_INSET = 5.0
_z_lo = float(H.min()) - 2.0
_z_hi = float(H.max()) + AGL_TARGET + AGL_TOL + 10.0

position = ox.State("position", shape=(3,))
position.min = np.array([TERRAIN_X_MIN + _INSET, TERRAIN_Y_MIN + _INSET, _z_lo])
position.max = np.array([TERRAIN_X_MAX - _INSET, TERRAIN_Y_MAX - _INSET, _z_hi])
position.initial = START
position.final = GOAL

velocity = ox.State("velocity", shape=(3,))
velocity.min = np.array([-30.0, -30.0, -15.0])
velocity.max = np.array([30.0, 30.0, 15.0])
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

thrust_force = ox.Control("thrust_force", shape=(3,))
thrust_force.max = np.array([0.0, 0.0, THRUST_MAX])
thrust_force.min = np.array([0.0, 0.0, 0.0])

torque = ox.Control("torque", shape=(3,))
torque.max = np.array([18.665, 18.665, 0.55562])
torque.min = np.array([-18.665, -18.665, -0.55562])
torque.guess = np.zeros((N, 3))

states = [position, velocity, attitude, angular_velocity]
controls = [thrust_force, torque]

time = ox.Time(
    initial=0.0,
    final=TOTAL_TIME,
    min=0.0,
    max=TOTAL_TIME,
    uniform_time_grid=True,
)

# ── AGL CTCS via Bilerp ───────────────────────────────────────────────────────
z_terrain = ox.Bilerp(position[0], position[1], x_grid, y_grid, H)
agl = position[2] - z_terrain

# Slightly inward of the advertised [19, 21] m band so post-propagation AGL
# residuals (~10 cm) still land inside the user-facing window.
_AGL_LO = AGL_TARGET - AGL_TOL + 0.25  # 19.25
_AGL_HI = AGL_TARGET + AGL_TOL - 0.25  # 20.75

constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
constraints.append(ox.ctcs(agl >= _AGL_LO, idx=1))
constraints.append(ox.ctcs(agl <= _AGL_HI, idx=1))

# ── Dynamics ──────────────────────────────────────────────────────────────────
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
}

# ── AGL-following guess + differential flatness ───────────────────────────────
dt = TOTAL_TIME / (N - 1)
alphas = np.linspace(0.0, 1.0, N)
xy_guess = START_XY[None, :] * (1.0 - alphas[:, None]) + GOAL_XY[None, :] * alphas[:, None]
z_guess = np.array(
    [bilinear_height(xy[0], xy[1], x_grid, y_grid, H) + AGL_TARGET for xy in xy_guess]
)
pos_guess = np.column_stack([xy_guess, z_guess])
position.guess = pos_guess
velocity.guess = np.gradient(pos_guess, dt, axis=0)
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
        "lam_prox": 1e-2,
        "lam_vc": 1e2,
        "lam_cost": 1e-3,
    },
    float_dtype="float64",
)

plotting_data = {
    "x_grid": x_grid,
    "y_grid": y_grid,
    "H": H,
    "agl_target": AGL_TARGET,
    "agl_tol": AGL_TOL,
    "start": START,
    "goal": GOAL,
}

if __name__ == "__main__":
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    print("6DoF Quadrotor — SENNS Lunar DEM AGL (~20 m band)")
    print("=" * 60)
    print(f"Start: {START}")
    print(f"Goal:  {GOAL}")
    print(f"AGL band: [{AGL_TARGET - AGL_TOL}, {AGL_TARGET + AGL_TOL}] m")
    print(
        f"SENNS DEM: {DEM_GRID}×{DEM_GRID}  "
        f"(~{dx:.2f} m × {dy:.2f} m spacing, elev_scale={ELEV_SCALE} m)"
    )
    print(f"Terrain relief: [{float(H.min()):.1f}, {float(H.max()):.1f}] m")
    print(f"Fixed time: {TOTAL_TIME} s, N={N}")
    print("=" * 60)

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_data)

    pos = np.asarray(results.trajectory["position"], dtype=np.float64)
    agl_traj = np.array(
        [
            float(pos[k, 2] - bilinear_height(pos[k, 0], pos[k, 1], x_grid, y_grid, H))
            for k in range(len(pos))
        ]
    )
    converged = getattr(results, "converged", "?")
    print(f"Converged: {converged}")
    print(
        f"AGL along traj: min={agl_traj.min():.2f}, "
        f"max={agl_traj.max():.2f}, mean={agl_traj.mean():.2f} m"
    )

    server = create_lunar_terrain_agl_server(results)
    server.sleep_forever()
