"""Realtime MPCC drone racing: 3D double integrator with live visualization.

Two-phase example:
1. Solve a time-optimal drone racing trajectory through gates (double integrator)
2. Use model-predictive contouring control (MPCC) to track it in closed loop,
   with the viser visualization updating in realtime after each MPC solve
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
from openscvx.plotting.viser import (
    add_gates,
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
g_const = -9.81

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

### Gate Cone Parameters ###
gate_cone_indices = list(range(n_gates))
gate_normals = [
    np.array([-1, 0, 0]),  # gate 0: -x
    np.array([-1, 0, 0]),  # gate 1: -x
    np.array([+1, 0, 0]),  # gate 2: +x
    np.array([-1, 0, 0]),  # gate 3: -x
    np.array([-1, 0, 0]),  # gate 4: -x
    np.array([+1, 0, 0]),  # gate 5: +x
    np.array([+1, 0, 0]),  # gate 6: +x
    np.array([-1, 0, 0]),  # gate 7: -x
    np.array([+1, 0, 0]),  # gate 8: +x
    np.array([+1, 0, 0]),  # gate 9: +x
]
cone_half_angle = np.deg2rad(60)
tan_a = np.tan(cone_half_angle)
gate_half_width = 2.5
apex_offset = gate_half_width / tan_a

A_cone = np.diag([1.0, 1.0, 0.0])
c_cone = np.array([0.0, 0.0, 1.0])

gate_cone_rotations = []
gate_cone_apexes = []
for center, n in zip(gate_centers, gate_normals):
    n_hat = n / np.linalg.norm(n)
    up = np.array([0, 0, 1.0])
    if abs(np.dot(n_hat, up)) > 0.99:
        up = np.array([0, 1.0, 0])
    y = np.cross(n_hat, up)
    y /= np.linalg.norm(y)
    x = np.cross(y, n_hat)
    R = np.column_stack([x, y, n_hat])
    gate_cone_rotations.append(R)
    gate_cone_apexes.append(center - apex_offset * n_hat)


def g_gate_cone(apex, R_gate, x_pos):
    p_local = R_gate.T @ (x_pos - apex)
    return ox.linalg.Norm(A_cone @ p_local) - tan_a * (c_cone.T @ p_local)


### End Gate Cone Parameters ###

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

### Obstacle Parameters ###
obstacle_center_positions = [
    np.array([76.2, -8.5, 22.762]),
    np.array([151.8, -48.5, 22.5]),
    np.array([40.3, -62.0, 22.5]),
]
obstacle_centers = [
    ox.Parameter(f"obstacle_center_{i}", shape=(3,), value=pos)
    for i, pos in enumerate(obstacle_center_positions)
]
n_obstacles = len(obstacle_centers)
obstacle_radius = 5.0
### End Obstacle Parameters ###

###############################################################################
# MPCC parameters
###############################################################################
n_mpc = 11  # Horizon nodes
horizon_duration = 2.0  # Horizon length [s]

Q_LAG = 1e0  # Lag error weight (high -> accurate progress tracking)
Q_CONTOUR = 5e-2  # Contour error weight
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
    ref_force = results_traj.trajectory["force"]  # (N_dense, 3)
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

    vx_data = np.tile(ref_vel[:-1, 0], len(tile_laps))
    vy_data = np.tile(ref_vel[:-1, 1], len(tile_laps))
    vz_data = np.tile(ref_vel[:-1, 2], len(tile_laps))

    fx_data = np.tile(ref_force[:-1, 0], len(tile_laps))
    fy_data = np.tile(ref_force[:-1, 1], len(tile_laps))
    fz_data = np.tile(ref_force[:-1, 2], len(tile_laps))

    speed_data = np.tile(ref_speeds[:-1], len(tile_laps))

    lap_time = ref_time[-1]
    t_data = np.concatenate([ref_time[:-1] + k * lap_time for k in tile_laps])

    # Tangent field: derivative of cubic spline, sampled at breakpoints
    _dpx = _CS(s_data, px_data)(s_data, 1)
    _dpy = _CS(s_data, py_data)(s_data, 1)
    _dpz = _CS(s_data, pz_data)(s_data, 1)
    _tnorm = np.sqrt(_dpx**2 + _dpy**2 + _dpz**2)
    tx_data = _dpx / _tnorm
    ty_data = _dpy / _tnorm
    tz_data = _dpz / _tnorm

    # =================================================================
    # Gate crossing progress (arc-length where reference is closest to each gate)
    # =================================================================
    gate_crossing_progress = []
    for j, center in enumerate(gate_centers):
        dists = np.linalg.norm(ref_pos_lap - center, axis=1)
        closest_idx = np.argmin(dists)
        gate_crossing_progress.append(s_lap[closest_idx])
        print(f"  Gate {j}: crossing progress s = {s_lap[closest_idx]:.1f} m")

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
    lag_sum.max = np.array([1e0])
    lag_sum.initial = np.array([0.0])
    lag_sum.final = [ox.Minimize(0.0)]

    contour_sum = ox.State("contour_sum", shape=(1,))  # Integrated contour cost
    contour_sum.min = np.array([0.0])
    contour_sum.max = np.array([2e2])
    contour_sum.initial = np.array([0.0])
    contour_sum.final = [ox.Minimize(0.0)]

    # --- Controls ---
    force = ox.Control("force", shape=(3,))
    force.min = np.array([-f_max_traj, -f_max_traj, -f_max_traj])
    force.max = np.array([f_max_traj, f_max_traj, f_max_traj])
    force.guess = np.zeros((n_mpc, 3))

    progress_rate = ox.Control("progress_rate", shape=(1,))  # d(theta_hat)/dt
    progress_rate.min = np.array([0.0])  # Forward only
    progress_rate.max = np.array([100.0])  # High enough for racing speeds
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

    # Obstacle avoidance: ||position - center|| >= radius
    for obs_center in obstacle_centers:
        constraints.append(ox.ctcs(obstacle_radius <= ox.linalg.Norm(position - obs_center)))

    # Gate cone constraints — active only when progress is between prev and current gate
    for i, (apex, R_gate) in enumerate(zip(gate_cone_apexes, gate_cone_rotations)):
        gate_idx = gate_cone_indices[i]
        s_gate = gate_crossing_progress[gate_idx]
        s_prev = gate_crossing_progress[(gate_idx - 1) % n_gates]

        n_hat = gate_normals[gate_idx].astype(float)
        approaching = ox.Sum(velocity * n_hat) <= 0.0  # v opposes cone normal when approaching
        pred = ox.All([progress[0] >= s_prev, progress[0] <= s_gate, approaching])
        cone_expr = ox.Cond(pred, g_gate_cone(apex, R_gate, position), -1.0)
        constraints.append(ox.ctcs(cone_expr <= 0.0))

    # --- Time ---
    t = ox.Time(
        initial=0.0,
        final=horizon_duration,
        min=0.0,
        max=horizon_duration,
        # uniform_time_grid=True,
    )

    constraints.append((t == horizon_duration / (n_mpc - 1)).convex().at(1))

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
    def set_initial_guess(theta_start: float = 0.0):
        """Set guesses by interpolating the reference trajectory via time lookup."""
        t_start = np.interp(theta_start, s_data, t_data)
        t_guess = np.linspace(t_start, t_start + horizon_duration, n_mpc)
        arc_guess = np.interp(t_guess, t_data, s_data)

        # Position: interpolate from reference sample nodes
        pos_guess = np.column_stack(
            [
                np.interp(arc_guess, s_data, px_data),
                np.interp(arc_guess, s_data, py_data),
                np.interp(arc_guess, s_data, pz_data),
            ]
        )
        position.guess = pos_guess

        # Velocity: interpolate from reference trajectory
        vel_guess = np.column_stack(
            [
                np.interp(arc_guess, s_data, vx_data),
                np.interp(arc_guess, s_data, vy_data),
                np.interp(arc_guess, s_data, vz_data),
            ]
        )
        velocity.guess = vel_guess
        velocity.initial = vel_guess[0]

        # Force: interpolate from reference trajectory
        force.guess = np.column_stack(
            [
                np.interp(arc_guess, s_data, fx_data),
                np.interp(arc_guess, s_data, fy_data),
                np.interp(arc_guess, s_data, fz_data),
            ]
        )

        progress.guess = arc_guess.reshape(-1, 1)
        lag_sum.guess = np.zeros((n_mpc, 1))
        contour_sum.guess = np.zeros((n_mpc, 1))

        progress_rate.guess = np.interp(arc_guess, s_data, speed_data).reshape(-1, 1)

    # =================================================================
    # Closed-loop simulation
    # =================================================================
    def shift_guess(nodes: dict):
        """Shift previous solution by one node for warm-starting."""
        # Horizon time is local (zeroed each solve), so map progress -> ref time first
        t_last = np.interp(nodes["progress"][-1, 0], s_data, t_data)
        ext_prog = np.interp(t_last + dt_mpc, t_data, s_data)

        ext_pos = np.array(
            [
                np.interp(ext_prog, s_data, px_data),
                np.interp(ext_prog, s_data, py_data),
                np.interp(ext_prog, s_data, pz_data),
            ]
        )
        ext_vel = np.array(
            [
                np.interp(ext_prog, s_data, vx_data),
                np.interp(ext_prog, s_data, vy_data),
                np.interp(ext_prog, s_data, vz_data),
            ]
        )

        ext_force = np.array(
            [
                np.interp(ext_prog, s_data, fx_data),
                np.interp(ext_prog, s_data, fy_data),
                np.interp(ext_prog, s_data, fz_data),
            ]
        )

        shifted_progress = np.vstack([nodes["progress"][1:], [[ext_prog]]])
        wrap_offset = (nodes["progress"][1, 0] // total_arc_length) * total_arc_length
        shifted_progress -= wrap_offset

        position.guess = np.vstack([nodes["position"][1:], [ext_pos]])
        velocity.guess = np.vstack([nodes["velocity"][1:], [ext_vel]])
        progress.guess = shifted_progress
        lag_offset = nodes["lag_sum"][1]
        lag_sum.guess = np.maximum(
            np.vstack([nodes["lag_sum"][1:] - lag_offset, nodes["lag_sum"][-1:] - lag_offset]),
            0.0,
        )

        contour_offset = nodes["contour_sum"][1]
        contour_sum.guess = np.maximum(
            np.vstack(
                [
                    nodes["contour_sum"][1:] - contour_offset,
                    nodes["contour_sum"][-1:] - contour_offset,
                ]
            ),
            0.0,
        )

        force.guess = np.vstack([nodes["force"][1:], [ext_force]])
        ext_speed = np.interp(ext_prog, s_data, speed_data)
        progress_rate.guess = np.vstack([nodes["progress_rate"][1:], [[ext_speed]]])

        # Time: shift and renormalize so horizon starts at t=0
        dtau = 1.0 / (n_mpc - 1)
        ext_time = nodes["time"][-1, 0] + nodes["_time_dilation"][-1, 0] * dtau
        shifted_time = np.vstack([nodes["time"][1:], [[ext_time]]])
        shifted_time -= shifted_time[0]
        t.guess = shifted_time

        t._time_dilation_control.guess = np.vstack(
            [nodes["_time_dilation"][1:], nodes["_time_dilation"][-1:]]
        )

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

    dt_mpc = horizon_duration / (n_mpc - 1)  # Time between MPC steps
    node1_time = dt_mpc  # Time of node 1 in each horizon

    # =================================================================
    # Start viser server with static scene elements
    # =================================================================
    server = create_server(ref_pos, show_grid=False)

    # Reference trajectory (static, faint red)
    ref_colors = np.full((len(ref_pos), 3), fill_value=[255, 80, 80], dtype=np.uint8)
    server.scene.add_point_cloud(
        "/reference",
        points=ref_pos.astype(np.float32),
        colors=(ref_colors * 0.3).astype(np.uint8),
        point_size=0.05,
    )

    add_gates(server, vertices)

    # --- Interactive obstacles (click to select, drag to move) ---
    obstacle_handles = []
    for i in range(n_obstacles):
        handle = server.scene.add_icosphere(
            f"/obstacles/sphere_{i}",
            radius=obstacle_radius,
            color=(255, 100, 100),
            position=tuple(obstacle_centers[i].value),
        )
        obstacle_handles.append(handle)

    obstacle_drag_handles = []
    for i in range(n_obstacles):
        drag_handle = server.scene.add_transform_controls(
            f"/obstacles/drag_{i}",
            position=tuple(obstacle_centers[i].value),
            scale=12.0,
            disable_rotations=True,
            visible=False,
        )
        obstacle_drag_handles.append(drag_handle)

    selected_obstacle = {"index": None}

    def select_obstacle(obs_idx: int | None) -> None:
        """Select an obstacle and show its transform control, hiding others."""
        if selected_obstacle["index"] is not None:
            obstacle_drag_handles[selected_obstacle["index"]].visible = False
            obstacle_handles[selected_obstacle["index"]].color = (255, 100, 100)
        if obs_idx is not None:
            obstacle_drag_handles[obs_idx].visible = True
            obstacle_handles[obs_idx].color = (255, 150, 150)
            selected_obstacle["index"] = obs_idx
        else:
            selected_obstacle["index"] = None

    def make_obstacle_click_handler(obs_idx: int):
        @obstacle_handles[obs_idx].on_click
        def _(_) -> None:
            if selected_obstacle["index"] == obs_idx:
                select_obstacle(None)
            else:
                select_obstacle(obs_idx)
        return _

    for i in range(n_obstacles):
        make_obstacle_click_handler(i)

    # GUI folder for obstacle positions
    obstacle_vector_inputs = []
    with server.gui.add_folder("Obstacle Positions", expand_by_default=False):
        server.gui.add_markdown("*Click an obstacle in 3D view to select and drag it*")

        reset_obstacles_button = server.gui.add_button("Reset All Obstacles")

        @reset_obstacles_button.on_click
        def _(_) -> None:
            select_obstacle(None)
            for i, vec_input in enumerate(obstacle_vector_inputs):
                original = obstacle_center_positions[i]
                vec_input.value = tuple(original)
                obstacle_centers[i].value = np.array(original)
                problem_mpc.parameters[obstacle_centers[i].name] = np.array(original)
                obstacle_drag_handles[i].position = tuple(original)
                obstacle_handles[i].position = tuple(original)
            print("Obstacles reset to initial positions")

        for i in range(n_obstacles):
            initial_pos = obstacle_centers[i].value
            vec_input = server.gui.add_vector3(
                f"Obstacle {i + 1}",
                initial_value=tuple(initial_pos),
                step=0.5,
            )
            obstacle_vector_inputs.append(vec_input)

            def make_obstacle_gui_callback(obs_idx: int, input_handle):
                @input_handle.on_update
                def _(_) -> None:
                    new_center = np.array(input_handle.value)
                    obstacle_centers[obs_idx].value = new_center
                    problem_mpc.parameters[obstacle_centers[obs_idx].name] = new_center
                    obstacle_drag_handles[obs_idx].position = tuple(new_center)
                    obstacle_handles[obs_idx].position = tuple(new_center)
                return _

            make_obstacle_gui_callback(i, vec_input)

    def make_drag_callback(obs_idx: int, drag_handle):
        @drag_handle.on_update
        def _(_) -> None:
            new_center = np.array(drag_handle.position)
            obstacle_centers[obs_idx].value = new_center
            problem_mpc.parameters[obstacle_centers[obs_idx].name] = new_center
            obstacle_vector_inputs[obs_idx].value = tuple(new_center)
            obstacle_handles[obs_idx].position = tuple(new_center)
        return _

    for i in range(n_obstacles):
        make_drag_callback(i, obstacle_drag_handles[i])

    # --- Fixed-range velocity colormap (avoids recomputing min/max each frame) ---
    import matplotlib.pyplot as plt
    from collections import deque

    vel_max_norm = 0.25 * np.linalg.norm(velocity_traj.max)
    _cmap = plt.get_cmap("viridis")
    _cmap_lut = np.array([_cmap(v / 255.0)[:3] for v in range(256)], dtype=np.float32)
    _cmap_lut_u8 = (_cmap_lut * 255).astype(np.uint8)

    def velocity_colors(vel: np.ndarray) -> np.ndarray:
        """Map velocity array (N,3) to RGB (N,3) uint8 using fixed bounds [0, vel_max_norm]."""
        norms = np.linalg.norm(vel, axis=1)
        idx = np.clip((norms / vel_max_norm * 255).astype(int), 0, 255)
        return _cmap_lut_u8[idx]

    # Dynamic scene handles
    trail_handle = server.scene.add_point_cloud(
        "/trail",
        points=ref_pos[:1].astype(np.float32),
        colors=np.array([[100, 200, 255]], dtype=np.uint8),
        point_size=0.15,
    )

    marker_handle = server.scene.add_icosphere(
        "/current_pos",
        radius=0.5,
        color=(100, 200, 255),
        position=ref_pos[0],
    )

    horizon_handle = server.scene.add_point_cloud(
        "/horizon_rollout",
        points=ref_pos[:1].astype(np.float32),
        colors=np.array([[200, 200, 200]], dtype=np.uint8),
        point_size=0.3,
    )

    # Rolling buffer: keep last K segments for the visible trail
    trail_max_segments = 150
    trail_pos_buf: deque[np.ndarray] = deque(maxlen=trail_max_segments)
    trail_color_buf: deque[np.ndarray] = deque(maxlen=trail_max_segments)

    # Pre-built arrays for the committed trail (rebuilt only when a segment is committed)
    # Use a dict so nested rebuild_committed() can mutate without nonlocal.
    committed = {
        "pos": np.empty((0, 3), dtype=np.float32),
        "colors": np.empty((0, 3), dtype=np.uint8),
    }

    def rebuild_committed():
        """Rebuild committed trail arrays from the rolling buffer."""
        if trail_pos_buf:
            committed["pos"] = np.concatenate(list(trail_pos_buf), axis=0).astype(np.float32)
            committed["colors"] = np.concatenate(list(trail_color_buf), axis=0).astype(np.uint8)
        else:
            committed["pos"] = np.empty((0, 3), dtype=np.float32)
            committed["colors"] = np.empty((0, 3), dtype=np.uint8)

    # =================================================================
    # Realtime MPC loop
    # =================================================================
    import time

    step = 0
    t_prev_step = time.perf_counter()
    while True:
        t_step_start = time.perf_counter()
        dt_wall = t_step_start - t_prev_step
        t_prev_step = t_step_start

        problem_mpc.reset()
        results = problem_mpc.solve()
        results = problem_mpc.post_process()
        nodes = results.nodes

        t_solve = time.perf_counter() - t_step_start

        # --- Extract executed segment (node 0 -> node 1) ---
        traj_time = results.trajectory["time"].flatten()
        seg_end = np.searchsorted(traj_time, node1_time, side="right")
        seg_pos = results.trajectory["position"][:seg_end].copy().astype(np.float32)
        seg_vel = results.trajectory["velocity"][:seg_end].copy()
        seg_time = traj_time[:seg_end].copy()
        seg_colors = velocity_colors(seg_vel)

        # Update horizon rollout immediately (shows new plan while animating)
        horizon_pos = results.trajectory["position"]
        horizon_vel = results.trajectory["velocity"]
        horizon_handle.points = horizon_pos.astype(np.float32)
        horizon_handle.colors = (velocity_colors(horizon_vel) * 0.6).astype(np.uint8)

        # --- Prepare next step (before animation so overhead is included in budget) ---
        update_initial_conditions(nodes)
        shift_guess(nodes)

        # --- Animate through the executed segment in realtime ---
        t_anim_start = time.perf_counter()
        anim_budget = dt_mpc - (t_anim_start - t_prev_step)
        n_seg = len(seg_pos)
        n_committed = len(committed["pos"])

        for i in range(n_seg):
            # Append current-segment points up to i onto the committed trail
            n_new = i + 1
            frame_pos = np.empty((n_committed + n_new, 3), dtype=np.float32)
            frame_colors = np.empty((n_committed + n_new, 3), dtype=np.uint8)
            frame_pos[:n_committed] = committed["pos"]
            frame_pos[n_committed:] = seg_pos[:n_new]
            frame_colors[:n_committed] = committed["colors"]
            frame_colors[n_committed:] = seg_colors[:n_new]

            trail_handle.points = frame_pos
            trail_handle.colors = frame_colors
            marker_handle.position = seg_pos[i]

            # Sleep until this frame's wall-clock target
            if anim_budget > 0 and i < n_seg - 1:
                target_frac = seg_time[i + 1] / seg_time[-1] if seg_time[-1] > 0 else 1.0
                target_wall = t_anim_start + target_frac * anim_budget
                wait = target_wall - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)

        # Commit this segment to the rolling buffer
        trail_pos_buf.append(seg_pos)
        trail_color_buf.append(seg_colors)
        rebuild_committed()

        # --- Log ---
        cur_pos = nodes["position"][0]
        cur_progress = nodes["progress"][0, 0]
        cur_lag = nodes["lag_sum"][-1, 0]
        cur_contour = nodes["contour_sum"][-1, 0]
        laps_done = cur_progress / total_arc_length
        print(
            f"step {step:3d}: progress={cur_progress:7.2f} "
            f"({laps_done:.2f} laps), "
            f"lag={cur_lag:.4f}, contour={cur_contour:.4f}, "
            f"pos=[{cur_pos[0]:+7.2f}, {cur_pos[1]:+7.2f}, {cur_pos[2]:+7.2f}], "
            f"solve={t_solve:.3f}s, "
            f"dt={dt_wall:.3f}/{dt_mpc:.3f} ({dt_wall / dt_mpc * 100:.0f}%)"
        )

        step += 1

    server.sleep_forever()
