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
horizon_duration = 1.0  # MPC horizon length in seconds

# Cost weights
Q_LAG = 1e1  # Lag weight
Q_CONTOUR = 1e0  # Contour weight


def create_mpcc_problem(
    arc_length_grid: np.ndarray,
    p_ref: np.ndarray,
    v_ref: np.ndarray,
    a_ref: np.ndarray,
    f_ref: np.ndarray,
    f_ref_arc_length: np.ndarray,
    time_ref: np.ndarray,
    horizon_duration: float = horizon_duration,
) -> tuple:
    """Create MPCC problem with reference trajectory baked in as constants.

    Args:
        arc_length_grid: Arc-length at each reference point, shape (N,)
        p_ref: Reference positions, shape (N, 3)
        v_ref: Reference velocities, shape (N, 3)
        a_ref: Reference accelerations, shape (N, 3)
        f_ref: Reference forces, shape (M, 3) where M may differ from N
        f_ref_arc_length: Arc-length grid for force data, shape (M,)
        time_ref: Time at each reference point, shape (N,)
        horizon_duration: MPC horizon length in seconds

    Returns:
        Tuple of (problem, states_dict, controls_dict) for updating between solves
    """
    total_arc_length = arc_length_grid[-1]

    # MPCC states (default initial conditions from start of reference trajectory)
    position_mpc = ox.State("position", shape=(3,))
    position_mpc.max = np.array([200.0, 100, 50])
    position_mpc.min = np.array([-200.0, -100, 15])
    position_mpc.initial = p_ref[0]
    position_mpc.final = [("free", 0), ("free", 0), ("free", 0)]

    velocity_mpc = ox.State("velocity", shape=(3,))
    velocity_mpc.max = np.array([100, 100, 100])
    velocity_mpc.min = np.array([-100, -100, -100])
    velocity_mpc.initial = v_ref[0]
    velocity_mpc.final = [("free", 0), ("free", 0), ("free", 0)]

    progress = ox.State("progress", shape=(1,))  # Estimated progress along path
    progress.min = np.array([0.0])
    progress.max = np.array([total_arc_length])
    progress.initial = np.array([0.0])
    progress.final = [("maximize", 0.0)]

    lag_sum = ox.State("lag_sum", shape=(1,))  # Integral of lag cost
    lag_sum.min = np.array([0.0])
    lag_sum.max = np.array([1e6])
    lag_sum.initial = np.array([0.0])
    lag_sum.final = [("minimize", 0.0)]

    contour_sum = ox.State("contour_sum", shape=(1,))  # Integral of contour cost
    contour_sum.min = np.array([0.0])
    contour_sum.max = np.array([1e6])
    contour_sum.initial = np.array([0.0])
    contour_sum.final = [("minimize", 0.0)]

    # Set state guesses from reference trajectory
    # Use reference trajectory timing to estimate arc length covered in horizon
    time_1d = np.asarray(time_ref).flatten()
    horizon_arc_length = np.interp(horizon_duration, time_1d, arc_length_grid)

    # Sample reference trajectory for guess
    theta_guess = np.linspace(0, horizon_arc_length, n_mpc).reshape(-1, 1)
    position_mpc.guess = np.column_stack(
        [
            np.interp(theta_guess.flatten(), arc_length_grid, p_ref[:, 0]),
            np.interp(theta_guess.flatten(), arc_length_grid, p_ref[:, 1]),
            np.interp(theta_guess.flatten(), arc_length_grid, p_ref[:, 2]),
        ]
    )
    velocity_mpc.guess = np.column_stack(
        [
            np.interp(theta_guess.flatten(), arc_length_grid, v_ref[:, 0]),
            np.interp(theta_guess.flatten(), arc_length_grid, v_ref[:, 1]),
            np.interp(theta_guess.flatten(), arc_length_grid, v_ref[:, 2]),
        ]
    )
    progress.guess = theta_guess
    lag_sum.guess = np.zeros((n_mpc, 1))
    contour_sum.guess = np.zeros((n_mpc, 1))

    # MPCC controls
    force_mpc = ox.Control("force", shape=(3,))
    force_mpc.max = np.array([f_max, f_max, f_max])
    force_mpc.min = np.array([-f_max, -f_max, -f_max])
    # Initialize force guess from reference trajectory
    force_mpc.guess = np.column_stack(
        [
            np.interp(theta_guess.flatten(), f_ref_arc_length, f_ref[:, 0]),
            np.interp(theta_guess.flatten(), f_ref_arc_length, f_ref[:, 1]),
            np.interp(theta_guess.flatten(), f_ref_arc_length, f_ref[:, 2]),
        ]
    )

    progress_rate = ox.Control("progress_rate", shape=(1,))  # Progress rate control
    progress_rate.min = np.array([0.0])  # Only move forward along path
    progress_rate.max = np.array([50.0])  # Max progress rate
    # Initialize progress rate from reference speeds (dθ/dt = ||v||)
    ref_speeds = np.linalg.norm(v_ref, axis=1)
    progress_rate.guess = np.interp(theta_guess.flatten(), arc_length_grid, ref_speeds).reshape(
        -1, 1
    )

    # Interpolate reference trajectory at current progress (data baked in as constants)
    # Use progress[0] (scalar) for Linterp, then Stack to get (3,) vector
    p_ref_interp = ox.Stack(
        [
            ox.Linterp(progress[0], arc_length_grid, p_ref[:, 0]),
            ox.Linterp(progress[0], arc_length_grid, p_ref[:, 1]),
            ox.Linterp(progress[0], arc_length_grid, p_ref[:, 2]),
        ]
    )
    v_ref_interp = ox.Stack(
        [
            ox.Linterp(progress[0], arc_length_grid, v_ref[:, 0]),
            ox.Linterp(progress[0], arc_length_grid, v_ref[:, 1]),
            ox.Linterp(progress[0], arc_length_grid, v_ref[:, 2]),
        ]
    )
    a_ref_interp = ox.Stack(
        [
            ox.Linterp(progress[0], arc_length_grid, a_ref[:, 0]),
            ox.Linterp(progress[0], arc_length_grid, a_ref[:, 1]),
            ox.Linterp(progress[0], arc_length_grid, a_ref[:, 2]),
        ]
    )

    # 6D error vector: position and velocity errors
    e_pos = position_mpc - p_ref_interp
    e_vel = velocity_mpc - v_ref_interp

    # 6D tangent direction in state space
    # Since θ is position arc-length: t_6d = [v/||v||, a/||v||]
    speed = ox.linalg.Norm(v_ref_interp)
    t_pos = v_ref_interp / speed  # Unit tangent in position space
    t_vel = a_ref_interp / speed  # Velocity-space component of tangent

    # ||t_6d||² = 1 + ||a||²/||v||²
    tangent_norm_sq = 1.0 + ox.linalg.Norm(a_ref_interp) ** 2 / speed**2
    tangent_norm = ox.Sqrt(tangent_norm_sq)

    # Lag = (e_6d · t_6d) / ||t_6d|| = (e_pos · t_pos + e_vel · t_vel) / ||t_6d||
    lag_unnorm = ox.Sum(e_pos * t_pos) + ox.Sum(e_vel * t_vel)
    lag_scalar = lag_unnorm / tangent_norm
    lag_cost = lag_scalar**2

    # Contour² = ||e_6d||² - lag²
    e_norm_sq = ox.linalg.Norm(e_pos) ** 2 + ox.linalg.Norm(e_vel) ** 2
    contour_cost = e_norm_sq - lag_cost

    # MPCC dynamics
    dynamics_mpc = {
        "position": velocity_mpc,
        "velocity": (1 / m) * force_mpc + np.array([0, 0, g_const], dtype=np.float64),
        "progress": progress_rate,
        "lag_sum": Q_LAG * lag_cost,
        "contour_sum": Q_CONTOUR * contour_cost,
    }

    # MPCC constraints (box constraints)
    states_mpc = [position_mpc, velocity_mpc, progress, lag_sum, contour_sum]
    controls_mpc = [force_mpc, progress_rate]

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
        final=horizon_duration,
        min=0.0,
        max=horizon_duration,
        uniform_time_grid=True,
    )

    problem_mpc = Problem(
        dynamics=dynamics_mpc,
        states=states_mpc,
        controls=controls_mpc,
        time=t_mpc,
        constraints=constraints_mpc,
        N=n_mpc,
        autotuner=ox.ConstantProximalWeight(),
    )

    # Return states/controls for updating .initial and .guess between solves
    states_dict = {
        "position": position_mpc,
        "velocity": velocity_mpc,
        "progress": progress,
        "lag_sum": lag_sum,
        "contour_sum": contour_sum,
    }
    controls_dict = {
        "force": force_mpc,
        "progress_rate": progress_rate,
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


def update_initial_conditions(
    states_dict: dict,
    prev_nodes: dict,
) -> None:
    """Update initial conditions from previous solution's node 1.

    Args:
        states_dict: Dictionary of state objects to update
        prev_nodes: Previous solution nodes dict
    """
    states_dict["position"].initial = prev_nodes["position"][1]
    states_dict["velocity"].initial = prev_nodes["velocity"][1]
    states_dict["progress"].initial = prev_nodes["progress"][1]


def shift_guess(
    states_dict: dict,
    controls_dict: dict,
    prev_trajectory: dict,
    arc_length_grid: np.ndarray,
    horizon_duration: float,
) -> None:
    """Shift previous solution to create warm-start guess for next solve.

    Shifts trajectory by one node and extrapolates the last node using
    the MPC solution's terminal state and dynamics.

    Args:
        states_dict: Dictionary of state objects to update
        controls_dict: Dictionary of control objects to update
        prev_trajectory: Previous solution trajectory dict
        arc_length_grid: Arc-length grid (used for progress clamping)
        horizon_duration: MPC horizon duration in seconds
    """
    n_nodes = prev_trajectory["position"].shape[0]
    dt = horizon_duration / (n_nodes - 1)

    # Extrapolate position using MPC solution's terminal velocity
    extrapolated_pos = prev_trajectory["position"][-1] + prev_trajectory["velocity"][-1] * dt
    states_dict["position"].guess = np.vstack([prev_trajectory["position"][1:], extrapolated_pos])

    # Extrapolate velocity using MPC solution's terminal acceleration (from force)
    terminal_accel = prev_trajectory["force"][-1] / m + np.array([0, 0, g_const])
    extrapolated_vel = prev_trajectory["velocity"][-1] + terminal_accel * dt
    states_dict["velocity"].guess = np.vstack([prev_trajectory["velocity"][1:], extrapolated_vel])

    # Extrapolate progress using terminal progress rate
    extrapolated_progress = (
        prev_trajectory["progress"][-1] + prev_trajectory["progress_rate"][-1] * dt
    )
    extrapolated_progress = np.clip(extrapolated_progress, 0, arc_length_grid[-1])
    states_dict["progress"].guess = np.vstack(
        [prev_trajectory["progress"][1:], extrapolated_progress]
    )

    # Reset cost states to zero (they integrate from zero each horizon)
    states_dict["lag_sum"].guess = np.zeros((n_nodes, 1))
    states_dict["contour_sum"].guess = np.zeros((n_nodes, 1))

    # Extrapolate force (zero-order hold)
    extrapolated_force = prev_trajectory["force"][-1]
    controls_dict["force"].guess = np.vstack([prev_trajectory["force"][1:], extrapolated_force])

    # Shift progress rate guess (zero-order hold)
    controls_dict["progress_rate"].guess = np.vstack(
        [prev_trajectory["progress_rate"][1:], prev_trajectory["progress_rate"][-1:]]
    )


if __name__ == "__main__":
    # Solve time-optimal problem
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    results.update(plotting_dict)

    # Extract reference trajectory for MPCC
    p_ref = results.trajectory["position"]  # (N, 3)
    v_ref = results.trajectory["velocity"]  # (N, 3)
    f_ref = results.trajectory["force"]  # (M, 3), may be N-1 or N
    time_ref = results.trajectory["time"]  # (N,)

    # Compute reference acceleration from force (a = f/m + g)
    # Need to interpolate f_ref to state nodes if lengths differ
    if f_ref.shape[0] == len(time_ref):
        a_ref = f_ref / m + np.array([0, 0, g_const])
    else:
        # Force at midpoints - interpolate to state nodes
        time_1d = np.asarray(time_ref).flatten()
        t_mid = (time_1d[:-1] + time_1d[1:]) / 2
        a_ref = np.column_stack(
            [
                np.interp(time_1d, t_mid, f_ref[:, 0] / m),
                np.interp(time_1d, t_mid, f_ref[:, 1] / m),
                np.interp(time_1d, t_mid, f_ref[:, 2] / m + g_const),
            ]
        )

    trajectory_length = len(time_ref)
    print(f"Reference trajectory length: {trajectory_length} points")

    # Compute arc-length parameterization for states
    arc_length_grid = compute_arc_length_grid(v_ref, time_ref)
    total_arc_length = arc_length_grid[-1]
    print(f"Total arc length: {total_arc_length:.2f} m")

    # Compute arc-length grid for controls (may have different length than states)
    n_force = f_ref.shape[0]
    if n_force == len(time_ref):
        # Control at same nodes as state
        f_ref_arc_length = arc_length_grid
    else:
        # Control at midpoints (N-1 points) - use midpoint arc lengths
        f_ref_arc_length = (arc_length_grid[:-1] + arc_length_grid[1:]) / 2

    # Create MPCC problem with reference trajectory baked in
    problem_mpc, states, controls = create_mpcc_problem(
        arc_length_grid, p_ref, v_ref, a_ref, f_ref, f_ref_arc_length, time_ref
    )

    # Initialize MPCC problem
    problem_mpc.initialize()

    # Run closed-loop MPC
    max_iterations = 2

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration} ---")
        problem_mpc.reset()

        results_mpc = problem_mpc.solve()
        results_mpc = problem_mpc.post_process()
        nodes = results_mpc.nodes

        current_progress = nodes["progress"][0, 0]
        print(
            f"Iteration {iteration:3d}: progress = {current_progress:7.2f}/{total_arc_length:.2f} "
            f"({100 * current_progress / total_arc_length:5.1f}%), "
            f"pos = [{nodes['position'][0, 0]:6.1f}, {nodes['position'][0, 1]:6.1f}, "
            f"{nodes['position'][0, 2]:6.1f}]"
        )

        update_initial_conditions(states, nodes)
        shift_guess(states, controls, nodes, arc_length_grid, horizon_duration)

    results_mpc.update(plotting_dict)
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
        p_ref,
        compute_velocity_colors(v_ref),
    )
    scp_server = create_scp_animated_plotting_server(
        results_mpc,
        attitude_stride=3,
        frame_duration_ms=200,
    )

    # Keep both servers running
    traj_server.sleep_forever()
