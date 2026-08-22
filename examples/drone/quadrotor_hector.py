"""Simple quadrotor dynamics example (OpenSCvx port).

Inspired by the aligator ``quadrotor.py`` example and
`crocoddyl quadrotor <https://github.com/loco-3d/crocoddyl/blob/master/examples/quadrotor.py>`_.

The original uses Pinocchio multibody dynamics for the Hector quadrotor with
four rotor thrusts mapped to a body wrench. This port uses the standard
OpenSCvx 6-DOF rigid-body model with the same actuator map, horizon, targets,
and optional cylindrical obstacles / control bounds as the aligator script.

Modes (``argparse`` flags mirror aligator ``Args``):

* **default** — two-phase position tracking from ``x0`` through waypoint
  ``x_tar1``, switching at 70% of the horizon to ``x_tar2``.
* ``--obstacles`` — fly to ``x_tar3`` while avoiding two vertical columns.
* ``--bounds`` — enforce per-rotor thrust limits ``u ∈ [0, u_lim]``.
* ``--term-cstr`` — hard terminal position constraint instead of soft cost.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from openscvx import Problem

# ── Hector / aligator actuator parameters ─────────────────────────────────────
D_COG = 0.1525  # m, motor arm length
CF = 6.6e-5
CM = 1.0e-6
U_LIM = 5.0  # N per rotor

MASS = 1.477  # kg (Hector URDF total mass)
GRAVITY = np.array([0.0, 0.0, -9.81])
J_B = np.array([0.005, 0.005, 0.009])  # kg·m², diagonal body inertia
J_B_INV = 1.0 / J_B

# Wrench limits implied by independent rotors in [0, U_LIM].
TAU_ROLL_PITCH_MAX = D_COG * U_LIM
TAU_YAW_MAX = 2.0 * (CM / CF) * U_LIM
HOVER_PER_ROTOR = MASS * (-GRAVITY[2]) / 4.0

# ── Horizon (aligator: dt = 0.01, Tf = 1.8) ───────────────────────────────────
DT = 0.01
TF = 1.8
N = 5  # int(TF / DT) + 1  # 181 nodes

# ── Targets (aligator x_tar1, x_tar2, x_tar3) ────────────────────────────────
X_TAR1 = np.array([0.9, 0.8, 1.0])
X_TAR2 = np.array([1.4, -0.6, 1.0])
X_TAR3 = np.array([-0.1, 3.2, 1.0])

# ── Obstacles (aligator column centers / radius) ───────────────────────────────
CYL_RADIUS = 0.22
QUAD_RADIUS = 0.12  # approximate Hector collision AABB radius
CENTER_COLUMN1 = np.array([-0.45, 1.2])
CENTER_COLUMN2 = np.array([0.4, 2.4])

# ── Tracking weights (aligator task_schedule) ──────────────────────────────────
W_POS_PHASE1 = 4.0
W_POS_PHASE2 = 1.0
W_POS_OBST = 0.1
W_VEL = 1e-3
W_OMEGA = 1e-3
W_U = 0.1
TERMINAL_COST_SCALE = 12.0

IDX_SWITCH = int(0.7 * (N - 1))
WP_RADIUS = 0.35  # m; nodal balls approximate aligator quadratic waypoint costs


def build_problem(
    *,
    obstacles: bool = False,
    bounds: bool = False,
    term_cstr: bool = False,
) -> tuple[Problem, dict]:
    """Build the quadrotor trajectory optimization problem."""
    x0_pos = np.array([0.0, 0.0, 0.18])
    x_target = X_TAR3 if obstacles else X_TAR2

    # ── States ───────────────────────────────────────────────────────────────
    position = ox.State("position", shape=(3,))
    position.min = np.array([-2.0, -1.0, 0.0])
    position.max = np.array([3.0, 4.0, 2.0])
    position.initial = x0_pos
    if term_cstr:
        position.final = [float(v) for v in x_target]
    else:
        position.final = [ox.Free(float(v)) for v in x_target]

    velocity = ox.State("velocity", shape=(3,))
    velocity.min = np.array([-5.0, -5.0, -5.0])
    velocity.max = np.array([5.0, 5.0, 5.0])
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

    stage_cost = ox.State("stage_cost", shape=(1,))
    stage_cost.min = np.array([0.0])
    stage_cost.max = np.array([1e4])
    stage_cost.initial = np.array([0.0])
    stage_cost.final = [ox.Minimize(0.0)]

    # ── Controls: four rotors (aligator nu = 4) ───────────────────────────────
    rotors = ox.Control("rotors", shape=(4,), parameterization="ZOH")
    if bounds:
        rotors.min = np.zeros(4)
        rotors.max = U_LIM * np.ones(4)
    else:
        rotors.min = -U_LIM * np.ones(4)
        rotors.max = U_LIM * np.ones(4)
    rotors.guess = np.full((N, 4), HOVER_PER_ROTOR)

    states = [position, velocity, attitude, angular_velocity, stage_cost]
    controls = [rotors]

    # Map rotors → body wrench (aligator QUAD_ACT_MATRIX).
    cm_cf = CM / CF
    thrust_force = ox.Concat(
        ox.Constant(0.0),
        ox.Constant(0.0),
        rotors[0] + rotors[1] + rotors[2] + rotors[3],
    )
    torque = ox.Concat(
        D_COG * (rotors[1] - rotors[3]),
        D_COG * (rotors[2] - rotors[0]),
        cm_cf * (-rotors[0] + rotors[1] - rotors[2] + rotors[3]),
    )

    q_norm = ox.linalg.Norm(attitude)
    attitude_normalized = attitude / q_norm
    j_b_inv = ox.linalg.Diag(J_B_INV)
    j_b_diag = ox.linalg.Diag(J_B)

    # ── Running cost ─────────────────────────────────────────────────────────
    u_hover = ox.Constant(HOVER_PER_ROTOR)
    control_reg = W_U * ox.Sum((rotors - u_hover) ** 2)
    vel_reg = W_VEL * ox.Sum(velocity * velocity)
    omega_reg = W_OMEGA * ox.Sum(angular_velocity * angular_velocity)

    if obstacles:
        pos_reg = W_POS_OBST * ox.Sum((position - ox.Constant(X_TAR3)) ** 2)
    else:
        pos_reg_phase1 = W_POS_PHASE1 * ox.Sum((position - ox.Constant(X_TAR1)) ** 2)
        pos_reg_phase2 = W_POS_PHASE2 * ox.Sum((position - ox.Constant(X_TAR2)) ** 2)
        pos_reg = ox.Cond(
            None,
            pos_reg_phase1,
            pos_reg_phase2,
            node_ranges=[(0, IDX_SWITCH)],
        )

    dynamics = {
        "position": velocity,
        "velocity": (1.0 / MASS) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
        + ox.Constant(GRAVITY),
        "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
        "angular_velocity": j_b_inv
        @ (torque - ox.spatial.SSM(angular_velocity) @ j_b_diag @ angular_velocity),
        "stage_cost": pos_reg + vel_reg + omega_reg + control_reg,
    }

    # ── Constraints ──────────────────────────────────────────────────────────
    constraints: list = []
    for state in states:
        constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
    if bounds:
        constraints.extend([ox.ctcs(rotors <= rotors.max), ox.ctcs(rotors.min <= rotors)])

    # Floor: z >= 0 (aligator create_halfspace_z).
    constraints.append(ox.ctcs(ox.Constant(0.0) <= position[2]))

    if obstacles:
        clearance = CYL_RADIUS + QUAD_RADIUS
        for center in (CENTER_COLUMN1, CENTER_COLUMN2):
            xy = ox.Concat(position[0], position[1])
            center_xy = ox.Constant(np.array([center[0], center[1]]))
            constraints.append(ox.ctcs(clearance <= ox.linalg.Norm(xy - center_xy)))
        constraints.append(
            (ox.linalg.Norm(position - ox.Constant(X_TAR3)) <= WP_RADIUS).convex().at([N - 1])
        )
    else:
        constraints.append(
            (ox.linalg.Norm(position - ox.Constant(X_TAR1)) <= WP_RADIUS).convex().at([IDX_SWITCH])
        )
        if not term_cstr:
            constraints.append(
                (ox.linalg.Norm(position - ox.Constant(X_TAR2)) <= WP_RADIUS).convex().at([N - 1])
            )

    if not term_cstr:
        # Soft terminal cost boost (aligator wterm *= 12 when term_cstr is False).
        terminal_pos_cost = (
            TERMINAL_COST_SCALE * W_POS_PHASE2 * ox.Sum((position - ox.Constant(x_target)) ** 2)
        )
        dynamics["stage_cost"] = dynamics["stage_cost"] + ox.Cond(
            None,
            terminal_pos_cost,
            ox.Constant(0.0),
            node_ranges=[(N - 1, N)],
        )

    # ── Initial guess: straight-line position, hover rotors ──────────────────
    if obstacles:
        keyframes = [x0_pos, X_TAR3]
        nodes = [0, N - 1]
    else:
        keyframes = [x0_pos, X_TAR1, X_TAR2]
        nodes = [0, IDX_SWITCH, N - 1]

    position.guess = ox.init.linspace(keyframes=keyframes, nodes=nodes)
    velocity.guess = np.zeros((N, 3))
    attitude.guess = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1))
    angular_velocity.guess = np.zeros((N, 3))
    t_guess = np.linspace(0.0, TF, N)
    pos_guess = position.guess
    stage_cost.guess = np.cumsum(
        (
            W_POS_PHASE1 * np.sum((pos_guess - X_TAR1) ** 2, axis=1)
            + W_VEL * np.sum(velocity.guess**2, axis=1)
            + W_U * np.sum((rotors.guess - HOVER_PER_ROTOR) ** 2, axis=1)
        )
        * np.gradient(t_guess)
    ).reshape(-1, 1)

    time = ox.Time(
        initial=0.0,
        final=TF,
        min=0.0,
        max=TF,
    )

    problem = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraints,
        N=N,
        float_dtype="float64",
        algorithm={
            "lam_prox": 1e-3,
            "lam_vc": 1e2,
            "lam_cost": 1e-1,
            "k_max": 300,
        },
    )

    plotting_dict = {
        "x_tar1": X_TAR1,
        "x_tar2": X_TAR2,
        "x_tar3": X_TAR3,
        "idx_switch": IDX_SWITCH,
        "obstacle_centers": [CENTER_COLUMN1, CENTER_COLUMN2],
        "obstacle_radius": CYL_RADIUS + QUAD_RADIUS,
        "obstacles": obstacles,
    }
    return problem, plotting_dict


def body_thrust_from_rotors(rotors: np.ndarray) -> np.ndarray:
    """Body-frame thrust ``(N, 3)`` from the four rotor forces ``(N, 4)``.

    Every rotor pushes along body +z, so collective thrust is their sum. The
    viser templates draw a ``thrust_force`` vector; this example's control is
    ``rotors``, so the two are bridged explicitly rather than implicitly.
    """
    rotors = np.asarray(rotors, dtype=np.float64)
    zeros = np.zeros(len(rotors))
    return np.column_stack([zeros, zeros, rotors.sum(axis=1)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenSCvx Hector quadrotor example.")
    parser.add_argument("--obstacles", action="store_true", help="Add cylindrical columns.")
    parser.add_argument("--bounds", action="store_true", help="Use rotor thrust bounds [0, u_lim].")
    parser.add_argument(
        "--term-cstr",
        action="store_true",
        help="Hard terminal position constraint (default: soft terminal cost).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    problem, plotting_dict = build_problem(
        obstacles=args.obstacles,
        bounds=args.bounds,
        term_cstr=args.term_cstr,
    )

    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)
    results.trajectory["thrust_force"] = body_thrust_from_rotors(results.trajectory["rotors"])

    nodes = results.nodes
    target_key = "x_tar3" if args.obstacles else "x_tar2"
    print(f"nsteps = {N - 1}, dt = {DT:.3f} s, tf = {TF:.1f} s")
    print(f"Final position: {nodes['position'][-1]} (target {plotting_dict[target_key]})")
    print(f"Integrated cost: {nodes['stage_cost'][-1, 0]:.6f}")

    try:
        from examples.plotting_viser import (
            create_animated_plotting_server,
            create_scp_animated_plotting_server,
        )

        waypoint_positions = [plotting_dict["x_tar1"], plotting_dict["x_tar2"]]
        if args.obstacles:
            waypoint_positions = [plotting_dict["x_tar3"]]

        traj_server = create_animated_plotting_server(
            results,
            thrust_key="thrust_force",
            show_viewcone=False,
            waypoint_positions=waypoint_positions,
            scene_scale=0.1,
        )
        create_scp_animated_plotting_server(results, attitude_stride=5, frame_duration_ms=100)
        traj_server.sleep_forever()
    except ImportError:
        print("viser not installed; skipping interactive 3D plot.")
