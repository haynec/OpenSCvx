"""Receding horizon control example for double integrator drone-racing.

This example demonstrates time-optimal racing through gates using simplified
double integrator (point mass) dynamics instead of full 6-DOF dynamics. The problem includes:

- 3-DOF point mass dynamics (position and velocity only)
- Direct force control inputs (no attitude dynamics)
- Sequential gate passage constraints
- Minimal time objective
- Loop closure constraint
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
from examples.plotting_viser import (
    create_animated_plotting_server,
    create_scp_animated_plotting_server,
)
from openscvx import Problem
from openscvx.plotting import plot_scp_iterations
from openscvx.plotting.viser import add_ghost_trajectory, compute_velocity_colors
from openscvx.utils import gen_vertices, rot

n = 22  # Number of Nodes
total_time = 24.0  # Total time for the simulation

# Define state components
position = ox.State("position", shape=(3,))  # 3D position [x, y, z]
position.max = np.array([200.0, 100, 50])
position.min = np.array([-200.0, -100, 15])
position.initial = np.array([10.0, 0, 20])
position.final = [10.0, 0, 20]

velocity = ox.State("velocity", shape=(3,))  # 3D velocity [vx, vy, vz]
velocity.max = np.array([100, 100, 100])
velocity.min = np.array([-100, -100, -100])
velocity.initial = [("free", 0), ("free", 0), ("free", 0)]
velocity.final = [("free", 0), ("free", 0), ("free", 0)]

# Define control
force = ox.Control("force", shape=(3,))  # Control forces [fx, fy, fz]
f_max = 4.179446268 * 9.81
force.max = np.array([f_max, f_max, f_max])
force.min = np.array([-f_max, -f_max, -f_max])
initial_control = np.array([0.0, 0, 10])
force.guess = np.repeat(initial_control[np.newaxis, :], n, axis=0)

m = 1.0  # Mass of the drone
g_const = -9.18
J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone


### Gate Parameters ###
n_gates = 10
gate_centers = [
    np.array([59.436, 0.000, 20.0000]),
    np.array([92.964, -23.750, 25.5240]),
    np.array([92.964, -29.274, 20.0000]),
    np.array([92.964, -23.750, 20.0000]),
    np.array([130.150, -23.750, 20.0000]),
    np.array([152.400, -73.152, 20.0000]),
    np.array([92.964, -75.080, 20.0000]),
    np.array([92.964, -68.556, 20.0000]),
    np.array([59.436, -81.358, 20.0000]),
    np.array([22.250, -42.672, 20.0000]),
]

radii = np.array([2.5, 1e-4, 2.5])
A_gate = rot @ np.diag(1 / radii) @ rot.T
A_gate_cen = []
for center in gate_centers:
    center[0] = center[0] + 2.5
    center[2] = center[2] + 2.5
    A_gate_cen.append(A_gate @ center)
nodes_per_gate = 2
gate_nodes = np.arange(nodes_per_gate, n, nodes_per_gate)
vertices = []
for center in gate_centers:
    vertices.append(gen_vertices(center, radii))
### End Gate Parameters ###

# Define list of all states (needed for Problem and constraints)
states = [position, velocity]
controls = [force]

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Add gate constraints
for node, cen in zip(gate_nodes, A_gate_cen):
    A_gate_const = A_gate
    c_const = cen
    gate_constraint = (
        (ox.linalg.Norm(A_gate_const @ position - c_const, ord="inf") <= np.array([1.0]))
        .convex()
        .at([node])
    )
    constraints.append(gate_constraint)

constraints.extend(
    [(force.at(0) == force.at(n - 1)).convex(), (velocity.at(0) == velocity.at(n - 1)).convex()]
)


# Define dynamics as dictionary mapping state names to their derivatives
dynamics = {
    "position": velocity,
    "velocity": (1 / m) * force + np.array([0, 0, g_const], dtype=np.float64),
}


# Generate initial guess for position trajectory through gates
position.guess = ox.init.linspace(
    keyframes=[position.initial] + gate_centers + [position.final],
    nodes=[0] + list(gate_nodes) + [n - 1],
)

t = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=t,
    constraints=constraints,
    N=n,
)

problem.settings.scp.ep_tr = 1e-3  # Trust Region Tolerance

plotting_dict = {"vertices": vertices}

###############################################################################
# MPCC Problem Formulation
###############################################################################

n_mpc = 4  # Number of nodes for MPC horizon

# Cost weights
Q_LAG = 1e1  # Lag weight
Q_CONTOUR = 1e0  # Contour weight


def create_mpcc_problem(
    arc_length_grid: np.ndarray,
    p_ref_data: np.ndarray,
    v_ref_data: np.ndarray,
) -> tuple:
    """Create MPCC problem with reference trajectory baked in as constants.

    Args:
        arc_length_grid: Arc-length at each reference point, shape (N,)
        p_ref_data: Reference positions, shape (N, 3)
        v_ref_data: Reference velocities, shape (N, 3)

    Returns:
        Tuple of (problem, states_dict, controls_dict) for updating between solves
    """
    total_arc_length = arc_length_grid[-1]

    # MPCC states (default initial conditions from start of reference trajectory)
    position_mpc = ox.State("position", shape=(3,))
    position_mpc.max = np.array([200.0, 100, 50])
    position_mpc.min = np.array([-200.0, -100, 15])
    position_mpc.initial = p_ref_data[0]
    position_mpc.final = [("free", 0), ("free", 0), ("free", 0)]

    velocity_mpc = ox.State("velocity", shape=(3,))
    velocity_mpc.max = np.array([100, 100, 100])
    velocity_mpc.min = np.array([-100, -100, -100])
    velocity_mpc.initial = v_ref_data[0]
    velocity_mpc.final = [("free", 0), ("free", 0), ("free", 0)]

    theta_hat = ox.State("theta_hat", shape=(1,))  # Estimated progress along path
    theta_hat.min = np.array([0.0])
    theta_hat.max = np.array([total_arc_length])
    theta_hat.initial = np.array([0.0])

    lag_sum = ox.State("lag_sum", shape=(1,))  # Integral of lag cost
    lag_sum.initial = np.array([0.0])
    lag_sum.min = np.array([0.0])
    lag_sum.max = np.array([1e6])

    contour_sum = ox.State("contour_sum", shape=(1,))  # Integral of contour cost
    contour_sum.initial = np.array([0.0])
    contour_sum.min = np.array([0.0])
    contour_sum.max = np.array([1e6])

    # Set state guesses from reference trajectory
    # Estimate arc length covered in horizon based on average reference speed
    avg_speed = np.mean(np.linalg.norm(v_ref_data, axis=1))
    horizon_arc_length = min(avg_speed * 1.0, total_arc_length * 0.1)  # 1 second horizon

    # Sample reference trajectory for guess
    theta_guess = np.linspace(0, horizon_arc_length, n_mpc).reshape(-1, 1)
    position_mpc.guess = np.column_stack(
        [
            np.interp(theta_guess.flatten(), arc_length_grid, p_ref_data[:, 0]),
            np.interp(theta_guess.flatten(), arc_length_grid, p_ref_data[:, 1]),
            np.interp(theta_guess.flatten(), arc_length_grid, p_ref_data[:, 2]),
        ]
    )
    velocity_mpc.guess = np.column_stack(
        [
            np.interp(theta_guess.flatten(), arc_length_grid, v_ref_data[:, 0]),
            np.interp(theta_guess.flatten(), arc_length_grid, v_ref_data[:, 1]),
            np.interp(theta_guess.flatten(), arc_length_grid, v_ref_data[:, 2]),
        ]
    )
    theta_hat.guess = theta_guess
    lag_sum.guess = np.zeros((n_mpc, 1))
    contour_sum.guess = np.zeros((n_mpc, 1))

    # MPCC controls
    force_mpc = ox.Control("force", shape=(3,))
    force_mpc.max = np.array([f_max, f_max, f_max])
    force_mpc.min = np.array([-f_max, -f_max, -f_max])
    force_mpc.guess = np.tile([0.0, 0.0, m * abs(g_const)], (n_mpc, 1))  # Hover thrust

    v_theta = ox.Control("v_theta", shape=(1,))  # Progress rate control
    v_theta.min = np.array([0.0])  # Only move forward along path
    v_theta.max = np.array([50.0])  # Max progress rate
    v_theta.guess = np.ones((n_mpc, 1)) * 10.0  # Initial progress rate guess

    # Interpolate reference trajectory at current theta_hat (data baked in as constants)
    # Use theta_hat[0] (scalar) for Linterp, then Stack to get (3,) vector
    p_ref_interp = ox.Stack(
        [
            ox.Linterp(theta_hat[0], arc_length_grid, p_ref_data[:, 0]),
            ox.Linterp(theta_hat[0], arc_length_grid, p_ref_data[:, 1]),
            ox.Linterp(theta_hat[0], arc_length_grid, p_ref_data[:, 2]),
        ]
    )
    v_ref_interp = ox.Stack(
        [
            ox.Linterp(theta_hat[0], arc_length_grid, v_ref_data[:, 0]),
            ox.Linterp(theta_hat[0], arc_length_grid, v_ref_data[:, 1]),
            ox.Linterp(theta_hat[0], arc_length_grid, v_ref_data[:, 2]),
        ]
    )

    # Error vector
    e = position_mpc - p_ref_interp

    # Unit tangent direction
    tangent_norm = ox.linalg.Norm(v_ref_interp)
    tangent_unit = v_ref_interp / tangent_norm

    # Lag and contour costs (squared errors)
    # Compute lag as projection of error onto tangent direction
    lag_scalar = ox.Sum(e * tangent_unit)  # e · t_hat
    lag_cost = lag_scalar**2

    # Compute contour cost as norm of perpendicular component (more numerically stable)
    # e_parallel = (e · t_hat) * t_hat, e_perp = e - e_parallel
    e_perp = e - lag_scalar * tangent_unit
    contour_cost = ox.linalg.Norm(e_perp) ** 2

    # MPCC dynamics
    dynamics_mpc = {
        "position": velocity_mpc,
        "velocity": (1 / m) * force_mpc + np.array([0, 0, g_const], dtype=np.float64),
        "theta_hat": v_theta,
        "lag_sum": Q_LAG * lag_cost,
        "contour_sum": Q_CONTOUR * contour_cost,
    }

    # Terminal cost: minimize integrated costs, maximize progress
    lag_sum.final = [("minimize", 0.0)]
    contour_sum.final = [("minimize", 0.0)]
    theta_hat.final = [("maximize", 0.0)]

    # MPCC constraints (box constraints)
    states_mpc = [position_mpc, velocity_mpc, theta_hat, lag_sum, contour_sum]
    controls_mpc = [force_mpc, v_theta]

    constraints_mpc = []
    for state in [position_mpc, velocity_mpc]:
        constraints_mpc.extend(
            [
                ox.ctcs(state <= state.max),
                ox.ctcs(state.min <= state),
            ]
        )

    # Fixed time horizon for MPC
    t_mpc = ox.Time(
        initial=0.0,
        final=1.0,  # Fixed horizon length
        min=0.0,
        max=1.0,
    )

    problem_mpc = Problem(
        dynamics=dynamics_mpc,
        states=states_mpc,
        controls=controls_mpc,
        time=t_mpc,
        constraints=constraints_mpc,
        N=n_mpc,
    )

    problem_mpc.settings.scp.uniform_time_grid = True

    # Return states/controls for updating .initial and .guess between solves
    states_dict = {
        "position": position_mpc,
        "velocity": velocity_mpc,
        "theta_hat": theta_hat,
        "lag_sum": lag_sum,
        "contour_sum": contour_sum,
    }
    controls_dict = {
        "force": force_mpc,
        "v_theta": v_theta,
    }

    return problem_mpc, states_dict, controls_dict


def compute_arc_length_grid(velocity_trajectory: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Compute cumulative arc length from velocity trajectory.

    Args:
        velocity_trajectory: Velocity at each time point, shape (N, 3)
        time: Time values, shape (N,) or (N, 1)

    Returns:
        Cumulative arc length at each point, shape (N,)
    """
    from scipy.integrate import cumulative_trapezoid

    speeds = np.linalg.norm(velocity_trajectory, axis=1)
    time_1d = np.asarray(time).flatten()  # Ensure 1D
    arc_length = np.concatenate([[0.0], cumulative_trapezoid(speeds, time_1d)])
    return arc_length


def find_closest_arc_length(
    position: np.ndarray,
    p_ref_data: np.ndarray,
    arc_length_grid: np.ndarray,
) -> float:
    """Find arc length of closest point on reference trajectory.

    Args:
        position: Current position, shape (3,)
        p_ref_data: Reference trajectory positions, shape (N, 3)
        arc_length_grid: Arc length at each reference point, shape (N,)

    Returns:
        Arc length of closest point
    """
    distances = np.linalg.norm(p_ref_data - position, axis=1)
    closest_idx = np.argmin(distances)
    return arc_length_grid[closest_idx]


if __name__ == "__main__":
    # Solve time-optimal problem
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    results.update(plotting_dict)

    # Extract reference trajectory for MPCC
    p_ref_data = results.trajectory["position"]  # (N, 3)
    v_ref_data = results.trajectory["velocity"]  # (N, 3)
    time_data = results.trajectory["time"]  # (N,)

    trajectory_length = len(time_data)
    print(f"Reference trajectory length: {trajectory_length} points")

    # Compute arc-length parameterization
    arc_length_grid = compute_arc_length_grid(v_ref_data, time_data)
    total_arc_length = arc_length_grid[-1]
    print(f"Total arc length: {total_arc_length:.2f} m")

    # Create MPCC problem with reference trajectory baked in
    problem_mpc, states, controls = create_mpcc_problem(arc_length_grid, p_ref_data, v_ref_data)

    # Initialize MPCC from start of reference path
    initial_theta = find_closest_arc_length(p_ref_data[0], p_ref_data, arc_length_grid)
    states["position"].initial = p_ref_data[0]
    states["velocity"].initial = v_ref_data[0]
    states["theta_hat"].initial = np.array([initial_theta])

    # Initialize and first solve
    problem_mpc.initialize()
    results_mpc = problem_mpc.solve()
    print("Initial MPCC solve complete!")
    results_mpc = problem_mpc.post_process()
    results_mpc.update(plotting_dict)

    # Then need to solve in a loop!

    plot_scp_iterations(results_mpc).show()

    # Create both visualization servers (viser auto-assigns ports)
    traj_server = create_animated_plotting_server(
        results_mpc,
        thrust_key="force",
        viewcone_scale=10.0,
    )
    # Add time-optimal reference as a ghost trail
    add_ghost_trajectory(
        traj_server,
        p_ref_data,
        compute_velocity_colors(v_ref_data),
        # point_size=0.05,  # Smaller than the animated trail
        # name="reference_trajectory",
    )
    scp_server = create_scp_animated_plotting_server(
        results_mpc,
        attitude_stride=3,
        frame_duration_ms=200,
    )

    # Keep both servers running
    traj_server.sleep_forever()
