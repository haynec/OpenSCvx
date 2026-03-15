"""MPCC example: 3D double integrator tracking a circular reference trajectory (discrete).

Based on the MPCC formulation from [Romero _et al._ (2022)](https://arxiv.org/abs/2108.13205).

Demonstrates model-predictive contouring control (MPCC) with:

- 3D double integrator (point mass) dynamics (position + velocity, force control)
- Discrete reference path via Cinterp (sampled from a periodic function arbitrary 3D paths)
- Lag/contour error decomposition following Romero 2022
- Receding horizon closed-loop simulation
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
    add_ghost_trajectory,
    add_position_marker,
    compute_velocity_colors,
    create_server,
)

###############################################################################
# Reference path shape
###############################################################################


def wavy_circle(M: int, R: float = 3.0, z_amp: float = 0.5, z_freq: int = 3) -> np.ndarray:
    """Circle in the xy-plane with a sinusoidal z-component."""
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    return np.column_stack(
        [
            R * np.cos(theta),
            R * np.sin(theta),
            z_amp * np.sin(z_freq * theta),
        ]
    )


def torus_knot(M: int, R: float = 3.0, r: float = 1.0, p: int = 2, q: int = 3) -> np.ndarray:
    """(p, q) torus knot. p=2, q=3 gives a trefoil knot."""
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    return np.column_stack(
        [
            (R + r * np.cos(q * theta)) * np.cos(p * theta),
            (R + r * np.cos(q * theta)) * np.sin(p * theta),
            r * np.sin(q * theta),
        ]
    )


def lissajous_3d(
    M: int,
    R: float = 3.0,
    a: int = 1,
    b: int = 2,
    c: int = 3,
    phi: float = np.pi / 2,
    z_amp: float = 1.0,
) -> np.ndarray:
    """3D Lissajous curve. Periodic when a, b, c are integers."""
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    return np.column_stack(
        [
            R * np.cos(a * theta),
            R * np.sin(b * theta + phi),
            z_amp * np.sin(c * theta),
        ]
    )


def figure_eight(M: int, R: float = 3.0, z_amp: float = 1.0) -> np.ndarray:
    """Figure-eight knot curve."""
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    scale = R / 3.0  # Normalize so max radial extent ~ R
    return np.column_stack(
        [
            scale * (2 + np.cos(2 * theta)) * np.cos(3 * theta),
            scale * (2 + np.cos(2 * theta)) * np.sin(3 * theta),
            z_amp * np.sin(4 * theta),
        ]
    )


###############################################################################
# Reference path parameters
###############################################################################
R_circle = 3.0  # Radius of the reference circle
center = np.array([0.0, 0.0, 0.0])  # Center of the reference path (3D)
tilt_angle = np.radians(45)  # Tilt out of the xy-plane (rotation about x-axis)

# Rotation matrix: tilt about x-axis by tilt_angle
R_tilt = np.array(
    [
        [1, 0, 0],
        [0, np.cos(tilt_angle), -np.sin(tilt_angle)],
        [0, np.sin(tilt_angle), np.cos(tilt_angle)],
    ]
)

###############################################################################
# Discrete reference path
###############################################################################
M = 30  # Number of samples per lap
path_fn = wavy_circle  # Swap this for any (M) -> (M, 3) function

# Sample one lap in the local frame, then rotate into 3D
path_local = path_fn(M, R=R_circle)
path_rotated = (R_tilt @ path_local.T).T + center
px_lap = path_rotated[:, 0]
py_lap = path_rotated[:, 1]
pz_lap = path_rotated[:, 2]

# Arc-length parametrization via cumulative chord lengths
dp = np.diff(path_rotated, axis=0, append=path_rotated[:1])  # wrap to start
chord_lengths = np.linalg.norm(dp, axis=1)
total_arc_length = np.sum(chord_lengths)  # Recompute from actual path
s_lap = np.concatenate([[0.0], np.cumsum(chord_lengths[:-1])])

# Tile to cover [progress.min, progress.max] = [-0.5L, 1.5L]
s_min, s_max = -0.5 * total_arc_length, 1.5 * total_arc_length
n_before = int(np.ceil(-s_min / total_arc_length))
n_after = int(np.ceil(s_max / total_arc_length))
laps = range(-n_before, n_after + 1)

s_data = np.concatenate([s_lap + k * total_arc_length for k in laps])
px_data = np.tile(px_lap, len(laps))
py_data = np.tile(py_lap, len(laps))
pz_data = np.tile(pz_lap, len(laps))


###############################################################################
# MPCC parameters
###############################################################################
n_mpc = 10  # Horizon nodes
horizon_duration = 1.0  # Horizon length [s]

Q_LAG = 1e0  # Lag error weight (high -> accurate progress tracking)
Q_CONTOUR = 1e-1  # Contour error weight
Q_PROGRESS = 1e-1

###############################################################################
# MPCC problem definition
###############################################################################

# --- States ---
position = ox.State("position", shape=(3,))
position.min = np.array([-10.0, -10.0, -10.0])
position.max = np.array([10.0, 10.0, 10.0])
position.initial = path_rotated[0]  # Start at first sample point
position.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

velocity = ox.State("velocity", shape=(3,))
velocity.min = np.array([-20.0, -20.0, -20.0])
velocity.max = np.array([20.0, 20.0, 20.0])
velocity.initial = np.array([0.0, 0.0, 0.0])
velocity.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]

progress = ox.State("progress", shape=(1,))  # Arc-length progress theta_hat
progress.min = np.array([-0.5 * total_arc_length])
progress.max = np.array([1.5 * total_arc_length])
progress.initial = np.array([0.0])
progress.final = [ox.Maximize(0.0)]

lag_sum = ox.State("lag_sum", shape=(1,))  # Integrated lag cost
lag_sum.min = np.array([0.0])
lag_sum.max = np.array([1e-3])
lag_sum.initial = np.array([0.0])
lag_sum.final = [ox.Minimize(0.0)]

contour_sum = ox.State("contour_sum", shape=(1,))  # Integrated contour cost
contour_sum.min = np.array([0.0])
contour_sum.max = np.array([5e-3])
contour_sum.initial = np.array([0.0])
contour_sum.final = [ox.Minimize(0.0)]

# --- Controls ---
force = ox.Control("force", shape=(3,))
f_max = 20.0
force.min = np.array([-f_max, -f_max, -f_max])
force.max = np.array([f_max, f_max, f_max])
force.guess = np.zeros((n_mpc, 3))

progress_rate = ox.Control("progress_rate", shape=(1,))  # d(theta_hat)/dt
progress_rate.min = np.array([1.5])  # Forward only
progress_rate.max = np.array([10.0])
progress_rate.guess = np.full((n_mpc, 1), 5.0)

m = 1.0  # Mass

# --- Reference trajectory (discrete, via Cinterp) ---
p_ref = ox.Concat(
    ox.Cinterp(progress[0], s_data, px_data),
    ox.Cinterp(progress[0], s_data, py_data),
    ox.Cinterp(progress[0], s_data, pz_data),
)

# Tangent: derivative of the position spline, sampled at breakpoints and
# re-interpolated with a second Cinterp for a smooth tangent field.
from scipy.interpolate import CubicSpline as _CS

_dpx = _CS(s_data, px_data)(s_data, 1)
_dpy = _CS(s_data, py_data)(s_data, 1)
_dpz = _CS(s_data, pz_data)(s_data, 1)
_tnorm = np.sqrt(_dpx**2 + _dpy**2 + _dpz**2)
tx_data = _dpx / _tnorm
ty_data = _dpy / _tnorm
tz_data = _dpz / _tnorm

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

# --- Dynamics ---
dynamics = {
    "position": velocity,
    "velocity": (1 / m) * force,
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
    initial=0.0, final=horizon_duration, min=0.0, max=horizon_duration, uniform_time_grid=True
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


###############################################################################
# Initial guesses
###############################################################################
def set_initial_guess(
    theta_start: float = 0.0,
    ref_speed: float = 5.0,
):
    """Set guesses by interpolating the discrete reference path.

    Only requires (s_data, px_data, py_data, pz_data) and a nominal speed — no
    analytical path knowledge or precomputed tangent data.
    """
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


###############################################################################
# Closed-loop simulation
###############################################################################
def shift_guess(nodes: dict):
    """Shift previous solution by one node for warm-starting."""
    dt = horizon_duration / (n_mpc - 1)

    # Extrapolate a new final node
    pos_last = nodes["position"][-1]
    vel_last = nodes["velocity"][-1]
    force_last = nodes["force"][-1]
    pr_last = nodes["progress_rate"][-1, 0]

    ext_pos = pos_last + dt * vel_last
    ext_vel = vel_last + dt * (1 / m) * force_last
    ext_prog = nodes["progress"][-1, 0] + dt * pr_last

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

    force.guess = np.vstack([nodes["force"][1:], [0, 0, 0]])
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


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    set_initial_guess(theta_start=0.0)

    problem_mpc.initialize()

    max_steps = 100
    dt_mpc = horizon_duration / (n_mpc - 1)  # Time between MPC steps
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

        laps = cur_progress / total_arc_length
        print(
            f"step {step:3d}: progress={cur_progress:7.2f} "
            f"({laps:.2f} laps), "
            f"lag_cost={cur_lag:.4f}, contour_cost={cur_contour:.4f}, "
            f"pos=[{cur_pos[0]:+6.2f}, {cur_pos[1]:+6.2f}, {cur_pos[2]:+6.2f}]"
        )

        update_initial_conditions(nodes)
        shift_guess(nodes)

    # --- Build dense actual trajectory and horizon lookup ---
    actual_path = np.concatenate(actual_segments, axis=0)
    actual_vel = np.concatenate(actual_vel_segments, axis=0)
    actual_colors = compute_velocity_colors(actual_vel)

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

    # Reference path (static)
    ref_dense = path_fn(400, R=R_circle)
    ref_dense = (R_tilt @ ref_dense.T).T + center
    ref_dense = np.vstack([ref_dense, ref_dense[:1]])
    ref_colors = np.full((len(ref_dense), 3), fill_value=[255, 80, 80], dtype=np.uint8)
    server.scene.add_point_cloud(
        "/reference",
        points=ref_dense.astype(np.float32),
        colors=(ref_colors * 0.5).astype(np.uint8),
        point_size=0.02,
    )

    # Ghost of all MPC horizons (faint background)
    all_horizon_points = np.concatenate(horizon_trajectories, axis=0)
    add_ghost_trajectory(server, all_horizon_points, all_horizon_colors, point_size=0.005)

    # Animated actual trail (grows as drone flies, uses library primitive)
    _, update_trail = add_animated_trail(server, actual_path, actual_colors, point_size=0.03)

    # Position marker at current drone position
    _, update_marker = add_position_marker(server, actual_path, radius=0.05)

    # Horizon rollout pop-in: shows the current planned horizon
    horizon_handle = server.scene.add_point_cloud(
        "/horizon_rollout",
        points=horizon_trajectories[0].astype(np.float32),
        colors=horizon_colors[0],
        point_size=0.015,
    )

    def update_horizon(frame_idx):
        """Swap in the horizon rollout for the current MPC step."""
        step = int(np.searchsorted(step_boundaries, frame_idx, side="right"))
        step = min(step, max_steps - 1)
        horizon_handle.points = horizon_trajectories[step].astype(np.float32)
        horizon_handle.colors = horizon_colors[step]

    add_animation_controls(
        server,
        actual_time,
        [update_trail, update_marker, update_horizon],
    )

    plot_states(results).show()
    plot_scp_iterations(results).show()

    server.sleep_forever()
