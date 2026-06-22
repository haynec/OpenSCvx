"""RobotZoo quadrotor zigzag waypoint tracking (OpenSCvx port).

Port of the ``RobotZoo.Quadrotor`` ``:zigzag`` scenario from TrajOptMethods /
RobotDynamics.jl. The vehicle flies from ``[0, -10, 1]`` through three
position waypoints in a fixed ``5`` s horizon:

- node 33  → ``[10, 0, 1]``
- node 66  → ``[-10, 0, 1]``
- node 101 → ``[0, 10, 1]``

Dynamics use the default ``RobotZoo.Quadrotor`` parameters (``0.5`` kg,
diagonal inertia, ``0.175`` m motor span). Controls are the body-frame
thrust vector and torque (the wrench produced by the four rotors with
``u ∈ [0, 12]^4`` and ``kf = 1``). The Julia reference applies per-node
LQR tracking costs; here waypoints are enforced with nodal position balls
and a quadratic control-effort integrator approximates the ``R`` penalty.
"""

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem
from openscvx.plotting import plot_controls, plot_states

# ── Horizon (Julia: N = 101, tf = 5.0) ─────────────────────────────────────
# n = 101
n = 10
total_time = 5.0

# ── RobotZoo.Quadrotor defaults ──────────────────────────────────────────────
MASS = 0.5  # kg
GRAVITY = np.array([0.0, 0.0, -9.81])
J_B = np.array([0.0023, 0.0023, 0.004])
J_B_INV = 1.0 / J_B
MOTOR_DIST = 0.175  # m
KM = 0.0245
MOTOR_MAX = 12.0  # N per rotor
HOVER_THRUST = -GRAVITY[2] * MASS  # total body-z thrust at hover

# Wrench limits implied by independent rotors in [0, MOTOR_MAX].
TAU_ROLL_PITCH_MAX = MOTOR_DIST * MOTOR_MAX
TAU_YAW_MAX = 2.0 * KM * MOTOR_MAX

# ── Waypoints (Julia 1-based nodes 33, 66, 101 → 0-based 32, 65, 100) ─────
waypoints = [
    np.array([10.0, 0.0, 1.0]),
    np.array([-10.0, 0.0, 1.0]),
    np.array([0.0, 10.0, 1.0]),
]
waypoint_nodes = [3, 6, 9]
WP_RADIUS = 2.0  # m; soft analogue of Julia LQR waypoint costs

x0_pos = np.array([0.0, -10.0, 1.0])
xf_pos = waypoints[-1]

# ── States ───────────────────────────────────────────────────────────────────
position = ox.State("position", shape=(3,))
position.min = np.array([-30.0, -30.0, 0.0])
position.max = np.array([30.0, 30.0, 5.0])
position.initial = x0_pos
position.final = [ox.Free(float(v)) for v in xf_pos]

velocity = ox.State("velocity", shape=(3,))
velocity.min = np.array([-10.0, -10.0, -10.0])
velocity.max = np.array([10.0, 10.0, 10.0])
velocity.initial = np.zeros(3)
velocity.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

attitude = ox.State("attitude", shape=(4,))  # [qw, qx, qy, qz]
attitude.min = np.array([-1.0, -1.0, -1.0, -1.0])
attitude.max = np.array([1.0, 1.0, 1.0, 1.0])
attitude.initial = [ox.Free(1.0), ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]
attitude.final = [ox.Free(1.0), ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.min = np.array([-10.0, -10.0, -10.0])
angular_velocity.max = np.array([10.0, 10.0, 10.0])
angular_velocity.initial = np.zeros(3)
angular_velocity.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

control_cost = ox.State("control_cost", shape=(1,))
control_cost.min = np.array([0.0])
control_cost.max = np.array([1e4])
control_cost.initial = np.array([0.0])
control_cost.final = [ox.Minimize(0.0)]

# ── Controls: body thrust + torque (aggregate 4-rotor wrench) ───────────────
thrust_force = ox.Control("thrust_force", shape=(3,))
thrust_force.min = np.array([0.0, 0.0, 0.0])
thrust_force.max = np.array([0.0, 0.0, 4.0 * MOTOR_MAX])
thrust_force.guess = np.repeat(np.array([[0.0, 0.0, HOVER_THRUST]]), n, axis=0)

torque = ox.Control("torque", shape=(3,))
torque.min = np.array([-TAU_ROLL_PITCH_MAX, -TAU_ROLL_PITCH_MAX, -TAU_YAW_MAX])
torque.max = np.array([TAU_ROLL_PITCH_MAX, TAU_ROLL_PITCH_MAX, TAU_YAW_MAX])
torque.guess = np.zeros((n, 3))

states = [position, velocity, attitude, angular_velocity, control_cost]
controls = [thrust_force, torque]

# ── Constraints ──────────────────────────────────────────────────────────────
constraints: list = []
for state in [position, velocity, attitude, angular_velocity, control_cost]:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
for control in controls:
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

for node, wp in zip(waypoint_nodes, waypoints):
    constraints.append(
        (ox.linalg.Norm(position - ox.Constant(wp)) <= WP_RADIUS).convex().at([node])
    )

# ── Dynamics ─────────────────────────────────────────────────────────────────
q_norm = ox.linalg.Norm(attitude)
attitude_normalized = attitude / q_norm
J_b_inv_ox = ox.linalg.Diag(J_B_INV)
J_b_diag_ox = ox.linalg.Diag(J_B)

dynamics = {
    "position": velocity,
    "velocity": (1.0 / MASS) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
    + ox.Constant(GRAVITY),
    "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
    "angular_velocity": J_b_inv_ox
    @ (torque - ox.spatial.SSM(angular_velocity) @ J_b_diag_ox @ angular_velocity),
    "control_cost": ox.Sum(thrust_force * thrust_force) + ox.Sum(torque * torque),
}


def _orientation_from_accel(accel: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] aligning body +z with specific thrust."""
    thrust_dir = accel - GRAVITY
    norm = np.linalg.norm(thrust_dir)
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


def _build_initial_guess() -> None:
    """Kinematic zigzag seed (Julia ``rollout!`` with hover controls)."""
    position.guess = ox.init.linspace(
        keyframes=[x0_pos] + waypoints,
        nodes=[0] + waypoint_nodes,
    )
    dt = total_time / (n - 1)
    velocity.guess = np.gradient(position.guess, dt, axis=0)
    velocity.guess[0] = np.zeros(3)
    velocity.guess[-1] = np.zeros(3)
    accel = np.gradient(velocity.guess, dt, axis=0)
    attitude.guess = np.array([_orientation_from_accel(accel[k]) for k in range(n)])
    attitude.guess /= np.linalg.norm(attitude.guess, axis=1, keepdims=True)
    angular_velocity.guess = np.zeros((n, 3))
    control_cost.guess = np.zeros((n, 1))


_build_initial_guess()

time = ox.Time(
    initial=0.0,
    final=total_time,
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
    float_dtype="float64",
    algorithm={
        # # Julia SolverOptions: penalty_scaling=100, penalty_initial=0.1
        # "lam_prox": 0.1,
        # "lam_vc": 1e2,
        # "lam_cost": 1e-4,
        # "autotuner": {"type": "RampProximalWeight", "ramp_factor": 1.04},
        # "k_max": 250,
    },
)

plotting_dict = {
    "waypoint_positions": waypoints,
    "waypoint_nodes": waypoint_nodes,
}


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    plot_states(results).show()
    plot_controls(results).show()

    try:
        from examples.plotting_viser import (
            create_animated_plotting_server,
            create_scp_animated_plotting_server,
            create_snapshot_plotting_server,
        )

        traj_server = create_animated_plotting_server(
            results,
            thrust_key="thrust_force",
            show_viewcone=False,
            show_control_plot="thrust_force",
            show_control_norm_plot="thrust_force",
            waypoint_positions=waypoints,
        )
        create_scp_animated_plotting_server(
            results,
            attitude_stride=5,
            frame_duration_ms=100,
        )
        create_snapshot_plotting_server(
            results,
            waypoint_positions=waypoints,
            initial_n_snapshots=5,
        )
        traj_server.sleep_forever()
    except ImportError:
        print("viser not installed; skipping interactive 3D plot.")
