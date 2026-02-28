"""MPCC drone racing: 3D double integrator with time-optimal reference trajectory.

Two-phase example:
1. Solve a time-optimal drone racing trajectory through gates (double integrator)
2. Use model-predictive contouring control (MPCC) to track it in closed loop

The reference path for MPCC is extracted from the solved time-optimal trajectory,
arc-length parametrized, and tiled for periodic (multi-lap) tracking.
"""

import os
import sys

import numpy as np

# Add grandparent directory to path to import openscvx
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem
from openscvx.plotting import plot_scp_iterations, plot_states
from openscvx.plotting.viser import (
    add_animated_trail,
    add_animation_controls,
    add_gates,
    add_ghost_trajectory,
    add_position_marker,
    compute_velocity_colors,
    create_server,
)
from openscvx.utils import gen_vertices, rot

###############################################################################
# Phase 1: Time-optimal trajectory through gates
###############################################################################

n_traj = 22  # Number of Nodes
total_time_traj = 24.0  # Total time for the simulation

# Define state components
position_traj = ox.State("position", shape=(3,))  # 3D position [x, y, z]
position_traj.max = np.array([200.0, 100, 50])
position_traj.min = np.array([-200.0, -100, 15])
position_traj.initial = np.array([10.0, 0, 20])
position_traj.final = [10.0, 0, 20]

velocity_traj = ox.State("velocity", shape=(3,))  # 3D velocity [vx, vy, vz]
velocity_traj.max = np.array([100, 100, 100])
velocity_traj.min = np.array([-100, -100, -100])
velocity_traj.initial = [ox.Free(0), ox.Free(0), ox.Free(0)]
velocity_traj.final = [ox.Free(0), ox.Free(0), ox.Free(0)]

# Define control
force_traj = ox.Control("force", shape=(3,))  # Control forces [fx, fy, fz]
f_max_traj = 4.179446268 * 9.81
force_traj.max = np.array([f_max_traj, f_max_traj, f_max_traj])
force_traj.min = np.array([-f_max_traj, -f_max_traj, -f_max_traj])
initial_control = np.array([0.0, 0, 10])
force_traj.guess = np.repeat(initial_control[np.newaxis, :], n_traj, axis=0)

m = 1.0  # Mass of the drone
g_const = -9.18

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
gate_nodes = np.arange(nodes_per_gate, n_traj, nodes_per_gate)
vertices = []
for center in gate_centers:
    vertices.append(gen_vertices(center, radii))
### End Gate Parameters ###

# Define list of all states (needed for Problem and constraints)
states_traj = [position_traj, velocity_traj]
controls_traj = [force_traj]

# Generate box constraints for all states
constraints_traj = []
for state in states_traj:
    constraints_traj.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Add gate constraints
for node, cen in zip(gate_nodes, A_gate_cen):
    gate_constraint = (
        (ox.linalg.Norm(A_gate @ position_traj - cen, ord="inf") <= np.array([1.0]))
        .convex()
        .at([node])
    )
    constraints_traj.append(gate_constraint)

# Loop closure: force and velocity must match at start and end
constraints_traj.extend(
    [
        (force_traj.at(0) == force_traj.at(n_traj - 1)).convex(),
        (velocity_traj.at(0) == velocity_traj.at(n_traj - 1)).convex(),
    ]
)

# Define dynamics as dictionary mapping state names to their derivatives
dynamics_traj = {
    "position": velocity_traj,
    "velocity": (1 / m) * force_traj + np.array([0, 0, g_const], dtype=np.float64),
}

# Generate initial guess for position trajectory through gates
position_traj.guess = ox.init.linspace(
    keyframes=[position_traj.initial] + gate_centers + [position_traj.final],
    nodes=[0] + list(gate_nodes) + [n_traj - 1],
)

t_traj = ox.Time(
    initial=0.0,
    final=("minimize", total_time_traj),
    min=0.0,
    max=total_time_traj,
)

problem_traj = Problem(
    dynamics=dynamics_traj,
    states=states_traj,
    controls=controls_traj,
    time=t_traj,
    constraints=constraints_traj,
    N=n_traj,
    algorithm={"ep_tr": 1e-3},
)

###############################################################################
# MPCC parameters
###############################################################################
n_mpc = 10  # Horizon nodes
horizon_duration = 1.0  # Horizon length [s]

Q_LAG = 1e0  # Lag error weight (high -> accurate progress tracking)
Q_CONTOUR = 1e-1  # Contour error weight
Q_PROGRESS = 1e-1

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    from scipy.interpolate import CubicSpline as _CS

    # =================================================================
    # Phase 1: Solve time-optimal trajectory
    # =================================================================
    problem_traj.initialize()
    results_traj = problem_traj.solve()
    results_traj = problem_traj.post_process()

    # =================================================================
    # Extract reference path from solved trajectory
    # =================================================================
    ref_pos = results_traj.trajectory["position"]  # (N_dense, 3)
    ref_vel = results_traj.trajectory["velocity"]  # (N_dense, 3)
    ref_time = results_traj.trajectory["time"].flatten()

    # Arc-length parametrization via cumulative speed integral
    ref_speeds = np.linalg.norm(ref_vel, axis=1)
    ds = ref_speeds[:-1] * np.diff(ref_time)
    s_lap = np.concatenate([[0.0], np.cumsum(ds)])
    total_arc_length = s_lap[-1]

    # Drop last point before tiling — it duplicates the first point of the
    # next lap (closed loop: initial == final position), which would create
    # non-strictly-increasing s values at tile boundaries.
    s_lap = s_lap[:-1]
    ref_pos_lap = ref_pos[:-1]

    px_lap = ref_pos_lap[:, 0]
    py_lap = ref_pos_lap[:, 1]
    pz_lap = ref_pos_lap[:, 2]

    print(
        f"Reference: {len(ref_pos)} points, arc length = {total_arc_length:.1f} m, "
        f"time = {ref_time[-1]:.2f} s"
    )

    # Tile for periodicity (closed-loop racing)
    s_min, s_max = -0.5 * total_arc_length, 1.5 * total_arc_length
    n_before = int(np.ceil(-s_min / total_arc_length))
    n_after = int(np.ceil(s_max / total_arc_length))
    tile_laps = range(-n_before, n_after + 1)

    s_data = np.concatenate([s_lap + k * total_arc_length for k in tile_laps])
    px_data = np.tile(px_lap, len(tile_laps))
    py_data = np.tile(py_lap, len(tile_laps))
    pz_data = np.tile(pz_lap, len(tile_laps))

    # Tangent field: derivative of cubic spline, sampled at breakpoints
    _dpx = _CS(s_data, px_data)(s_data, 1)
    _dpy = _CS(s_data, py_data)(s_data, 1)
    _dpz = _CS(s_data, pz_data)(s_data, 1)
    _tnorm = np.sqrt(_dpx**2 + _dpy**2 + _dpz**2)
    tx_data = _dpx / _tnorm
    ty_data = _dpy / _tnorm
    tz_data = _dpz / _tnorm

    # =================================================================
    # Phase 2: Build MPCC problem (needs reference data from Phase 1)
    # =================================================================

    # --- States ---
    position = ox.State("position", shape=(3,))
    position.min = position_traj.min
    position.max = position_traj.max
    position.initial = ref_pos[0]
    position.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

    velocity = ox.State("velocity", shape=(3,))
    velocity.min = velocity_traj.min
    velocity.max = velocity_traj.max
    velocity.initial = ref_vel[0]
    velocity.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

    progress = ox.State("progress", shape=(1,))  # Arc-length progress theta_hat
    progress.min = np.array([-0.5 * total_arc_length])
    progress.max = np.array([1.5 * total_arc_length])
    progress.initial = np.array([0.0])
    progress.final = [ox.Maximize(0.0)]

    lag_sum = ox.State("lag_sum", shape=(1,))  # Integrated lag cost
    lag_sum.min = np.array([0.0])
    lag_sum.max = np.array([1e-2])
    lag_sum.initial = np.array([0.0])
    lag_sum.final = [ox.Minimize(0.0)]

    contour_sum = ox.State("contour_sum", shape=(1,))  # Integrated contour cost
    contour_sum.min = np.array([0.0])
    contour_sum.max = np.array([1e-2])
    contour_sum.initial = np.array([0.0])
    contour_sum.final = [ox.Minimize(0.0)]

    # --- Controls ---
    force = ox.Control("force", shape=(3,))
    force.min = np.array([-f_max_traj, -f_max_traj, -f_max_traj])
    force.max = np.array([f_max_traj, f_max_traj, f_max_traj])
    force.guess = np.zeros((n_mpc, 3))

    progress_rate = ox.Control("progress_rate", shape=(1,))  # d(theta_hat)/dt
    progress_rate.min = np.array([0.0])  # Forward only
    progress_rate.max = np.array([10.0])  # High enough for racing speeds
    progress_rate.guess = np.full((n_mpc, 1), ref_speeds.mean())

    # --- Reference trajectory (discrete, via Cinterp) ---
    p_ref = ox.Concat(
        ox.Cinterp(progress[0], s_data, px_data),
        ox.Cinterp(progress[0], s_data, py_data),
        ox.Cinterp(progress[0], s_data, pz_data),
    )

    tangent = ox.Concat(
        ox.Cinterp(progress[0], s_data, tx_data),
        ox.Cinterp(progress[0], s_data, ty_data),
        ox.Cinterp(progress[0], s_data, tz_data),
    )

    # --- Error decomposition (position-only, per Romero 2022 Fig. 2) ---
    e = position - p_ref  # Position error vector (3,)

    # Lag: projection of error onto tangent direction
    lag_scalar = ox.Sum(e * tangent)  # Dot product (scalar)
    lag_cost = lag_scalar**2

    # Contour: Pythagorean decomposition  |e_c|^2 = |e|^2 - |e_l|^2
    # Use Sum(e*e) instead of Norm(e)**2 to avoid d(Norm)/de = e/||e|| singularity at e=0
    contour_cost = ox.Max(ox.Sum(e * e) - lag_scalar**2, 0.0)

    # --- Dynamics (with gravity, matching the trajectory problem) ---
    dynamics = {
        "position": velocity,
        "velocity": (1 / m) * force + np.array([0, 0, g_const], dtype=np.float64),
        "progress": progress_rate,
        "lag_sum": lag_cost,
        "contour_sum": contour_cost,
    }

    # --- Constraints ---
    states = [position, velocity, progress, lag_sum, contour_sum]
    controls = [force, progress_rate]

    constraints = []
    for state in [position, velocity]:
        constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    # --- Time ---
    t = ox.Time(
        initial=0.0,
        final=horizon_duration,
        min=0.0,
        max=horizon_duration,
        uniform_time_grid=True,
    )

    # --- Problem ---
    problem_mpc = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=t,
        constraints=constraints,
        N=n_mpc,
        algorithm={
            "autotuner": ox.ConstantProximalWeight(),
            "lam_cost": {"lag_sum": Q_LAG, "contour_sum": Q_CONTOUR, "progress": Q_PROGRESS},
        },
    )
    problem_mpc.settings.dev.printing = False

    # =================================================================
    # Initial guesses
    # =================================================================
    def set_initial_guess(
        theta_start: float = 0.0,
        ref_speed: float = ref_speeds.mean(),
    ):
        """Set guesses by interpolating the discrete reference path."""
        arc_guess = np.linspace(theta_start, theta_start + ref_speed * horizon_duration, n_mpc)

        # Position: interpolate from reference sample nodes
        pos_guess = np.column_stack(
            [
                np.interp(arc_guess, s_data, px_data),
                np.interp(arc_guess, s_data, py_data),
                np.interp(arc_guess, s_data, pz_data),
            ]
        )
        position.guess = pos_guess

        # Velocity: finite-difference of position guess
        dt = horizon_duration / (n_mpc - 1)
        vel_guess = np.gradient(pos_guess, dt, axis=0)
        velocity.guess = vel_guess
        velocity.initial = vel_guess[0]

        # Force: finite-difference of velocity (mass * acceleration)
        acc_guess = np.gradient(vel_guess, dt, axis=0)
        force.guess = m * acc_guess

        progress.guess = arc_guess.reshape(-1, 1)
        lag_sum.guess = np.zeros((n_mpc, 1))
        contour_sum.guess = np.zeros((n_mpc, 1))

        progress_rate.guess = np.full((n_mpc, 1), ref_speed)

    # =================================================================
    # Closed-loop simulation
    # =================================================================
    def shift_guess(nodes: dict):
        """Shift previous solution by one node for warm-starting."""
        dt = horizon_duration / (n_mpc - 1)

        # Extrapolate progress for the new final node from the reference path
        pr_last = nodes["progress_rate"][-1, 0]
        ext_prog = nodes["progress"][-1, 0] + dt * pr_last

        ext_pos = np.array([
            np.interp(ext_prog, s_data, px_data),
            np.interp(ext_prog, s_data, py_data),
            np.interp(ext_prog, s_data, pz_data),
        ])
        ext_tangent = np.array([
            np.interp(ext_prog, s_data, tx_data),
            np.interp(ext_prog, s_data, ty_data),
            np.interp(ext_prog, s_data, tz_data),
        ])
        ext_vel = ext_tangent * pr_last

        # Force: back out from velocity difference at the tail
        vel_prev = nodes["velocity"][-1]
        ext_force = m * (ext_vel - vel_prev) / dt - np.array([0, 0, g_const])

        shifted_progress = np.vstack([nodes["progress"][1:], [[ext_prog]]])
        wrap_offset = (nodes["progress"][1, 0] // total_arc_length) * total_arc_length
        shifted_progress -= wrap_offset

        position.guess = np.vstack([nodes["position"][1:], [ext_pos]])
        velocity.guess = np.vstack([nodes["velocity"][1:], [ext_vel]])
        progress.guess = shifted_progress
        lag_sum.guess = np.zeros((n_mpc, 1))
        contour_sum.guess = np.zeros((n_mpc, 1))

        force.guess = np.vstack([nodes["force"][1:], [ext_force]])
        progress_rate.guess = np.vstack([nodes["progress_rate"][1:], nodes["progress_rate"][-1:]])

    def update_initial_conditions(nodes: dict):
        """Set initial conditions from node 1 of previous solution (simulate one step)."""
        position.initial = nodes["position"][1]
        velocity.initial = nodes["velocity"][1]

        wrap_offset = (nodes["progress"][1, 0] // total_arc_length) * total_arc_length
        progress.initial = np.array([nodes["progress"][1, 0] - wrap_offset])

        # Cost integrators always restart from zero each horizon
        lag_sum.initial = np.array([0.0])
        contour_sum.initial = np.array([0.0])

    set_initial_guess(theta_start=0.0)

    problem_mpc.initialize()

    max_steps = 1000
    dt_mpc = horizon_duration / n_mpc  # Time between MPC steps
    node1_time = dt_mpc  # Time of node 1 in each horizon

    # --- Run MPC loop, collecting data ---
    actual_segments = []
    actual_vel_segments = []
    actual_time_segments = []
    horizon_trajectories = []
    horizon_velocities = []

    for step in range(max_steps):
        problem_mpc.reset()
        results = problem_mpc.solve()
        results = problem_mpc.post_process()
        nodes = results.nodes

        # Slice trajectory from node 0 to node 1 (the executed segment)
        traj_time = results.trajectory["time"].flatten()
        seg_end = np.searchsorted(traj_time, node1_time, side="right")
        actual_segments.append(results.trajectory["position"][:seg_end].copy())
        actual_vel_segments.append(results.trajectory["velocity"][:seg_end].copy())
        actual_time_segments.append(traj_time[:seg_end].copy())

        horizon_trajectories.append(results.trajectory["position"].copy())
        horizon_velocities.append(results.trajectory["velocity"].copy())

        cur_pos = nodes["position"][0]
        cur_progress = nodes["progress"][0, 0]
        cur_lag = nodes["lag_sum"][-1, 0]
        cur_contour = nodes["contour_sum"][-1, 0]

        laps_done = cur_progress / total_arc_length
        print(
            f"step {step:3d}: progress={cur_progress:7.2f} "
            f"({laps_done:.2f} laps), "
            f"lag_cost={cur_lag:.4f}, contour_cost={cur_contour:.4f}, "
            f"pos=[{cur_pos[0]:+7.2f}, {cur_pos[1]:+7.2f}, {cur_pos[2]:+7.2f}]"
        )

        update_initial_conditions(nodes)
        shift_guess(nodes)

    # =================================================================
    # Visualization
    # =================================================================
    actual_path = np.concatenate(actual_segments, axis=0)
    actual_vel_cat = np.concatenate(actual_vel_segments, axis=0)
    actual_colors = compute_velocity_colors(actual_vel_cat)

    frames_per_step = np.array([len(seg) for seg in actual_segments])
    step_boundaries = np.cumsum(frames_per_step)  # frame index where each step ends

    # Dense time array: offset each segment's local time by the MPC step
    actual_time = np.concatenate(
        [seg_t + i * dt_mpc for i, seg_t in enumerate(actual_time_segments)]
    )

    # Horizon rollout colors (single global viridis mapping across all steps)
    all_horizon_vel = np.concatenate(horizon_velocities, axis=0)
    all_horizon_colors = compute_velocity_colors(all_horizon_vel)
    cumulative_horizon_pts = np.cumsum([len(hv) for hv in horizon_velocities])
    horizon_colors = np.split(all_horizon_colors, cumulative_horizon_pts[:-1])

    # --- Viser visualization ---
    server = create_server(actual_path)

    # Reference trajectory from time-optimal solve (static, faint red)
    ref_colors = np.full((len(ref_pos), 3), fill_value=[255, 80, 80], dtype=np.uint8)
    server.scene.add_point_cloud(
        "/reference",
        points=ref_pos.astype(np.float32),
        colors=(ref_colors * 0.3).astype(np.uint8),
        point_size=0.05,
    )

    # Gate markers (wireframe, consistent with non-MPC examples)
    add_gates(server, vertices)

    # Ghost of all MPC horizons (faint background)
    all_horizon_points = np.concatenate(horizon_trajectories, axis=0)
    add_ghost_trajectory(server, all_horizon_points, all_horizon_colors)

    # Animated actual trail (grows as drone flies)
    _, update_trail = add_animated_trail(server, actual_path, actual_colors)

    # Position marker at current drone position
    _, update_marker = add_position_marker(server, actual_path)

    # Horizon rollout pop-in: shows the current planned horizon
    horizon_handle = server.scene.add_point_cloud(
        "/horizon_rollout",
        points=horizon_trajectories[0].astype(np.float32),
        colors=horizon_colors[0],
        point_size=0.3,
    )

    def update_horizon(frame_idx):
        """Swap in the horizon rollout for the current MPC step."""
        step_idx = int(np.searchsorted(step_boundaries, frame_idx, side="right"))
        step_idx = min(step_idx, max_steps - 1)
        horizon_handle.points = horizon_trajectories[step_idx].astype(np.float32)
        horizon_handle.colors = horizon_colors[step_idx]

    add_animation_controls(
        server,
        actual_time,
        [update_trail, update_marker, update_horizon],
    )

    plot_states(results).show()
    plot_scp_iterations(results).show()

    server.sleep_forever()
