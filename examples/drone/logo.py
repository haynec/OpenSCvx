"""6-DOF quadrotor tracing out the Autonomous Controls Laboratory Logo

This example demonstrates trajectory planning for a quadrotor tracing out the Autonomous Controls
Laboratory Logo. The problem includes:

- 6-DOF rigid body dynamics (position, velocity, attitude quaternion, angular velocity)
- Utilizes the BYOF framework to model the angle between the boresight vector and the vector to the
    target.
"""

import os
import sys

import jax.numpy as jnp
import numpy as np
import numpy.linalg as la

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting_viser import create_animated_plotting_server
from openscvx import ByofSpec, Free, Minimize, Problem

# Prompt the user to instal svgpathtools if not already installed via pip
try:
    import svgpathtools  # noqa: F401
except ImportError:
    print("svgpathtools not found. Please install it with: pip install svgpathtools")
    sys.exit(1)

from examples.drone.logo_utils.svg_path_utils import get_svg_path_function

# -----------------------------------------------------------------------------
# SVG path utilities
# -----------------------------------------------------------------------------

# Load the SVG path function, using only Path 0 (the logo, no border)
svg_path_function = get_svg_path_function(
    "examples/drone/logo_utils/acl_logo.svg", path_indices=[0]
)


# -----------------------------------------------------------------------------
# JAX helpers (numeric versions of spatial utilities)
# -----------------------------------------------------------------------------


def qdcm(q: jnp.ndarray) -> jnp.ndarray:
    """Quaternion to DCM using [w, x, y, z] convention."""
    q_norm = jnp.linalg.norm(q)
    q_norm = jnp.where(q_norm > 1e-12, q_norm, 1e-12)
    w, qx, qy, qz = q / q_norm
    return jnp.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * w), 2 * (qx * qz + qy * w)],
            [2 * (qx * qy + qz * w), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * w)],
            [2 * (qx * qz - qy * w), 2 * (qy * qz + qx * w), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def SSMP(w: jnp.ndarray) -> jnp.ndarray:
    """4x4 skew-symmetric matrix for quaternion dynamics."""
    wx, wy, wz = w[0], w[1], w[2]
    return jnp.array(
        [
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0],
        ]
    )


def SSM(w: jnp.ndarray) -> jnp.ndarray:
    """3x3 skew-symmetric matrix for cross products."""
    wx, wy, wz = w[0], w[1], w[2]
    return jnp.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])


# -----------------------------------------------------------------------------
# Problem setup
# -----------------------------------------------------------------------------

n = 500  # Number of Nodes
total_time = 20.0  # Total time for the simulation

# Fixed initial and final positions
initial_pos = np.array([-15.0, 0.0, 5.0])
final_pos = np.array([-14.0, -10.0, 5.0])

# State components (like other drone examples)
position = ox.State("position", shape=(3,))
position.max = np.array([20.0, 20.0, 20.0])
position.min = np.array([-20.0, -20.0, -5.0])
position.initial = [Free(initial_pos[0]), Free(initial_pos[1]), Free(initial_pos[2])]
position.final = [Free(final_pos[0]), Free(final_pos[1]), Free(final_pos[2])]

velocity = ox.State("velocity", shape=(3,))
velocity.max = np.array([20.0, 20.0, 20.0])
velocity.min = np.array([-20.0, -20.0, -20.0])
velocity.initial = [0.0, 0.0, 0.0]
velocity.final = [0.0, 0.0, 0.0]

attitude = ox.State("attitude", shape=(4,))  # Quaternion [w, x, y, z]
attitude.max = np.array([1.0, 1.0, 1.0, 1.0])
attitude.min = np.array([-1.0, -1.0, -1.0, -1.0])
attitude.initial = [Free(1.0), Free(0), Free(0), Free(0)]
attitude.final = [Free(1.0), Free(0), Free(0), Free(0)]

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = np.array([10.0, 10.0, 10.0])
angular_velocity.min = np.array([-10.0, -10.0, -10.0])
angular_velocity.initial = np.array([0.0, 0.0, 0.0])
angular_velocity.final = [Free(0), Free(0), Free(0)]

angle_metric = ox.State("angle_metric", shape=(1,))  # Viewing angle (minimize at final)
angle_metric.max = np.array([0.3])
angle_metric.min = np.array([0.0])
angle_metric.initial = np.array([0.0])
angle_metric.final = [Minimize(0.0)]

time_state = ox.Time(
    initial=0.0,
    final=total_time,
    min=0.0,
    max=total_time,
    time_dilation_min=0.001 * total_time,
    time_dilation_max=3.0 * total_time,
)

# Control components
thrust_force = ox.Control("thrust_force", shape=(3,))
thrust_force.max = np.array([0.0, 0.0, 4.179446268 * 9.81])
thrust_force.min = np.array([0.0, 0.0, 0.0])
thrust_force.guess = np.repeat(np.array([[0.0, 0.0, 10.0]]), n, axis=0)

torque = ox.Control("torque", shape=(3,))
torque.max = np.array([10.0, 10.0, 1.0])
torque.min = np.array([-10.0, -10.0, -1.0])
torque.guess = np.zeros((n, 3))

# Offset from the SVG path for the drone to follow
path_offset = np.array([0.0, 0.0, 3.0])  # 3 meters above the path

# Boresight vector in body frame (x-axis of the drone)
boresight_body = jnp.array([1.0, 0.0, 0.0])


def get_kp_pose(t_normalized):
    """
    Get the target position along the SVG path at normalized time t, applying rotations about y and
    x axes.

    Args:
        t_normalized: Normalized time in [0, 1] (scalar or array)
    Returns:
        Target position [x, y, z]
    """
    original_pos = jnp.array(svg_path_function(t_normalized))

    scale_factor = 0.5
    scaled_pos = original_pos * scale_factor

    theta_y = jnp.pi / 2
    cos_y = jnp.cos(theta_y)
    sin_y = jnp.sin(theta_y)
    R_y = jnp.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])

    theta_x = jnp.pi / 2
    cos_x = jnp.cos(theta_x)
    sin_x = jnp.sin(theta_x)
    R_x = jnp.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])

    rotated_pos = R_x @ (R_y @ scaled_pos)
    return rotated_pos + path_offset


# -----------------------------------------------------------------------------
# BYOF: angle_metric derivative (uses get_kp_pose, not easily symbolic)
# -----------------------------------------------------------------------------


def angle_to_target_components(pos: jnp.ndarray, att: jnp.ndarray, t_val: float) -> jnp.ndarray:
    """Compute angle between boresight and vector to target from position, attitude, and time."""
    t_normalized = t_val / total_time
    target_pos = get_kp_pose(t_normalized)

    to_target_inertial = target_pos - pos
    boresight_inertial = qdcm(att) @ boresight_body

    to_target_norm = jnp.linalg.norm(to_target_inertial)
    boresight_norm = jnp.linalg.norm(boresight_inertial)

    lhs = jnp.dot(to_target_inertial, boresight_inertial) / (to_target_norm * boresight_norm)

    angle = jnp.arccos(lhs)
    return angle


def dynamics_angle_metric(x, u, node, params):
    """BYOF dynamics for angle_metric: derivative = time-weighted angle to target."""
    pos = x[position.slice]
    att = x[attitude.slice]
    t_val = x[time_state.slice][0]
    angle = angle_to_target_components(pos, att, t_val)
    return jnp.array([angle])


# -----------------------------------------------------------------------------
# Symbolic dynamics (drone: same as drone_racing / cinema_vp)
# -----------------------------------------------------------------------------

m = 1.0
g_const = -9.18
J_b = 1e-5 * jnp.array([1.0, 1.0, 1.0])
J_b_inv = 1.0 / J_b
J_b_diag = ox.linalg.Diag(J_b)

# Normalize quaternion for dynamics
q_norm = ox.linalg.Norm(attitude)
attitude_normalized = attitude / q_norm

dynamics = {
    "position": velocity,
    "velocity": (1.0 / m) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
    + ox.Constant(np.array([0, 0, g_const], dtype=np.float64)),
    "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
    "angular_velocity": ox.linalg.Diag(J_b_inv)
    @ (torque - ox.spatial.SSM(angular_velocity) @ J_b_diag @ angular_velocity),
    "time": 1.0,
}

# -----------------------------------------------------------------------------
# Plotting patch: full_subject_traj_time for moving target
# -----------------------------------------------------------------------------

import examples.plotting


def patched_full_subject_traj_time(results, params):
    t_full = results.t_full
    t_nodes = (
        results.nodes["time"].flatten()
        if "time" in results.nodes
        else np.linspace(0, total_time, n)
    )

    subs_traj = []
    subs_traj_node = []
    subs_traj_sen = []
    subs_traj_sen_node = []

    if "moving_subject" in results.plotting_data and "get_kp_pose" in results.plotting_data:
        custom_get_kp_pose = results.plotting_data["get_kp_pose"]

        target_positions_full = []
        target_positions_nodes = []
        for i in range(len(t_full)):
            t_normalized = t_full[i] / total_time
            target_positions_full.append(np.asarray(custom_get_kp_pose(t_normalized)))
        for i in range(len(t_nodes)):
            t_normalized = t_nodes[i] / total_time
            target_positions_nodes.append(np.asarray(custom_get_kp_pose(t_normalized)))

        subs_traj.append(np.array(target_positions_full))
        subs_traj_node.append(np.array(target_positions_nodes))
    elif "init_poses" in results.plotting_data:
        init_poses = results.plotting_data["init_poses"]
        n_full = len(t_full)
        n_nodes = len(t_nodes)
        for pose in init_poses:
            pose_full = np.repeat(pose[:, np.newaxis], n_full, axis=1).T
            pose_node = np.repeat(pose[:, np.newaxis], n_nodes, axis=1).T
            subs_traj.append(pose_full)
            subs_traj_node.append(pose_node)
    else:
        raise ValueError("No valid method to get keypoint poses.")

    subs_traj_sen = []
    subs_traj_sen_node = []
    return subs_traj, subs_traj_sen, subs_traj_node, subs_traj_sen_node


examples.plotting.full_subject_traj_time = patched_full_subject_traj_time


def angle_to_target(x_):
    """Compute the angle between boresight vector and vector to target (for analysis/plotting)."""
    pos = x_[:3]
    att = x_[6:10]
    t_val = x_[14]
    return angle_to_target_components(pos, att, t_val)


# -----------------------------------------------------------------------------
# States, controls, constraints, initial guess
# -----------------------------------------------------------------------------

states = [position, velocity, attitude, angular_velocity, angle_metric, time_state]
controls = [thrust_force, torque]

constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])


byof: ByofSpec = {
    "dynamics": {
        "angle_metric": dynamics_angle_metric,
    }
}

# Initial guess: position drone to maintain LoS with moving target
position_bar = np.linspace(initial_pos, final_pos, n)
velocity_bar = np.zeros((n, 3))
angular_velocity_bar = np.zeros((n, 3))
angle_metric_bar = np.zeros((n, 1))
time_bar = np.linspace(0, total_time, n).reshape(-1, 1)

b = np.array([1, 0, 0])
attitude_bar = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
for k in range(n):
    t_normalized = k / (n - 1)
    target_pos = np.asarray(get_kp_pose(t_normalized))
    a = target_pos - position_bar[k]
    if np.linalg.norm(a) > 1e-6:
        q_xyz = np.cross(b, a)
        q_w = np.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2) + np.dot(a, b)
        q_no_norm = np.hstack((q_w, q_xyz))
        q = q_no_norm / la.norm(q_no_norm)
        attitude_bar[k] = q

# attitude.initial = attitude_bar[0]

position.guess = position_bar
velocity.guess = velocity_bar
attitude.guess = attitude_bar
angular_velocity.guess = angular_velocity_bar
angle_metric.guess = angle_metric_bar
# time_state guess is typically set by the framework; set explicitly if needed
time_state.guess = time_bar

# -----------------------------------------------------------------------------
# Problem
# -----------------------------------------------------------------------------

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time_state,
    constraints=constraints,
    N=n,
    byof=byof,
    algorithm={
        "autotuner": {"type": "RampProximalWeight", "ramp_factor": 1.06, "lam_prox_max": 1e3},
        "lam_prox": 1e-3,
        "lam_vc": 1e0,
        "lam_cost": 6e-3,
    },
    float_dtype="float64",
    discretizer=ox.DiscretizeLinearizeVectorize(diffrax_kwargs={"atol": 1e-4}),
    solver={"solver_args": {"canon_backend": "COO", "enforce_dpp": True}},
)


plotting_dict = {
    "svg_path_function": svg_path_function,
    "path_offset": path_offset,
    "total_time": total_time,
    "initial_pos": initial_pos,
    "final_pos": final_pos,
    "boresight_body": boresight_body,
    "moving_subject": True,
    "get_kp_pose": get_kp_pose,
    "angle_to_target": angle_to_target,
    "extend_boresight": True,
    "relative_vector": True,
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    results.update_plotting_data(**plotting_dict)

    # Define the plane the logo actually lies in (from three points on the logo)
    p0 = np.asarray(get_kp_pose(0.0))
    p1 = np.asarray(get_kp_pose(0.33))
    p2 = np.asarray(get_kp_pose(0.67))
    v1 = p1 - p0
    v2 = p2 - p0
    plane_normal = np.cross(v1, v2)
    nrm = np.linalg.norm(plane_normal)
    if nrm < 1e-10:
        plane_normal = np.array([1.0, 0.0, 0.0])  # fallback
    else:
        plane_normal = plane_normal / nrm
    plane_point = p0

    def ray_plane_intersection(ray_origin, ray_direction, p_plane, n_plane):
        denom = np.dot(ray_direction, n_plane)
        if abs(denom) < 1e-10:
            return None
        t = np.dot(p_plane - ray_origin, n_plane) / denom
        if t < 0:
            return None
        return ray_origin + t * ray_direction

    pos_traj = np.asarray(results.trajectory["position"])
    tt = np.asarray(results.trajectory["time"]).reshape(-1)
    total_time_f = float(total_time)
    target_traj = np.stack(
        [np.asarray(get_kp_pose(float(t) / total_time_f)) for t in tt],
        axis=0,
    )
    traced_path = []
    for i in range(len(pos_traj)):
        rel = target_traj[i] - pos_traj[i]
        nrm = np.linalg.norm(rel)
        rel_dir = rel / nrm
        pt = ray_plane_intersection(pos_traj[i], rel_dir, plane_point, plane_normal)
        traced_path.append(pt if pt is not None else target_traj[i])
    results["traced_path_on_plane"] = np.array(traced_path, dtype=np.float64)
    results["logo_plane_point"] = plane_point
    results["logo_plane_normal"] = plane_normal

    from openscvx.plotting import (
        plot_controls,
        plot_states,
        plot_trust_region_heatmap,
        plot_virtual_control_heatmap,
    )

    plot_states(results).show()
    plot_controls(results).show()
    plot_virtual_control_heatmap(results).show()
    plot_trust_region_heatmap(results).show()

    server = create_animated_plotting_server(
        results,
        thrust_key="thrust_force",
        show_grid=True,
    )
    server.sleep_forever()
