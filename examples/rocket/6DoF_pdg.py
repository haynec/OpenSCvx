"""

6DoF PDG Rocket Trajectory Optimization

This example was adapted from the SCvxGEN repository by Abhi Kamath, https://scvxgen.mintlify.app/introduction.
"""

import os
import sys

import numpy as np

# Add grandparent directory to path to import examples.plotting
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting_viser import (
    create_animated_plotting_server,
    create_scp_animated_plotting_server,
    create_snapshot_plotting_server
)
from openscvx import Problem
from openscvx.plotting.viser.coordinates import model_vec_to_viser_xyz

# Position components live at indices 1:4 in the state vector (mass is index 0).
_POSITION_STATE_SLICE = slice(1, 4)

_ROCKET_STEP_PATH = os.path.join(current_dir, "Rocket.STEP")
_ROCKET_MESH_CACHE_PATH = os.path.join(current_dir, "Rocket_viser_mesh_upright.npz")
# SolidWorks STEP: rocket length is +Y; align with body +Z so it lands upright in Viser (Z-up).
_CAD_TO_BODY = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_ROCKET_MESH_TARGET_LENGTH = 2.0
_ROCKET_VOXEL_PITCH = 0.05


def model_attitude_xyzw_to_viser_wxyz(attitude: np.ndarray) -> np.ndarray:
    """PDG stores ``[q1, q2, q3, q4]`` (xyzw); Viser expects ``[w, x, y, z]``."""
    q = np.asarray(attitude, dtype=np.float64)
    if q.ndim == 1:
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)


def load_rocket_vehicle_mesh() -> tuple[np.ndarray, np.ndarray] | None:
    """Load ``Rocket.STEP`` as a decimated mesh for Viser, or ``None`` to fall back to axes."""
    if os.path.isfile(_ROCKET_MESH_CACHE_PATH) and os.path.isfile(_ROCKET_STEP_PATH):
        if os.path.getmtime(_ROCKET_MESH_CACHE_PATH) >= os.path.getmtime(_ROCKET_STEP_PATH):
            cached = np.load(_ROCKET_MESH_CACHE_PATH)
            return cached["vertices"], cached["faces"]

    try:
        import trimesh
    except ImportError:
        print(
            "[viser] Install trimesh and cascadio to show Rocket.STEP "
            "(pip install trimesh cascadio); using attitude axes."
        )
        return None

    try:
        mesh = trimesh.load(_ROCKET_STEP_PATH, force="mesh")
    except Exception as exc:
        print(f"[viser] Failed to load Rocket.STEP: {exc}; using attitude axes.")
        return None

    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh.vertices -= mesh.centroid
    mesh.vertices = mesh.vertices @ _CAD_TO_BODY.T
    length = float(np.max(mesh.extents))
    if length > 1e-9:
        mesh.vertices *= _ROCKET_MESH_TARGET_LENGTH / length

    try:
        voxel_grid = mesh.voxelized(pitch=_ROCKET_VOXEL_PITCH)
        simplified = voxel_grid.marching_cubes
        simplified.apply_transform(voxel_grid.transform)
    except Exception:
        simplified = mesh

    vertices = np.asarray(simplified.vertices, dtype=np.float32)
    faces = np.asarray(simplified.faces, dtype=np.uint32)
    np.savez(_ROCKET_MESH_CACHE_PATH, vertices=vertices, faces=faces)
    return vertices, faces


def remap_attitude_for_viser(result) -> None:
    """Convert propagated attitude quaternions to Viser wxyz for mesh / thrust display."""
    attitude = result.trajectory.get("attitude")
    if attitude is not None:
        result.trajectory["attitude"] = model_attitude_xyzw_to_viser_wxyz(np.asarray(attitude))


def remap_optimization_results_for_viser_xyz(
    results,
    position_slice: slice = _POSITION_STATE_SLICE,
) -> None:
    """After ``post_process()``, remap stored trajectories and SCP history for Viser axes."""
    traj = results.trajectory
    if traj.get("position") is not None:
        traj["position"] = model_vec_to_viser_xyz(np.asarray(traj["position"]))
    if traj.get("velocity") is not None:
        traj["velocity"] = model_vec_to_viser_xyz(np.asarray(traj["velocity"]))

    idx_list = list(range(position_slice.start, position_slice.stop, position_slice.step or 1))
    if len(idx_list) != 3:
        idx_list = None

    def _remap_state_rows(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64, copy=True)
        sl = position_slice
        X[:, sl] = model_vec_to_viser_xyz(X[:, sl])
        return X

    for i in range(len(results.X)):
        results.X[i] = _remap_state_rows(results.X[i])

    if getattr(results, "x_full", None) is not None:
        results.x_full = _remap_state_rows(results.x_full)

    if idx_list is None or not results.discretization_history:
        return

    n_x = int(results.X[-1].shape[1]) if results.X else 0
    n_u = int(results.U[-1].shape[1]) if results.U else 0
    if n_x <= 0:
        return

    i4 = n_x + n_x * n_x + 2 * n_x * n_u
    i0, i1, i2 = idx_list[0], idx_list[1], idx_list[2]

    for k in range(len(results.discretization_history)):
        V = np.asarray(results.discretization_history[k], dtype=np.float64, copy=True)
        n_timesteps = V.shape[1]
        n_segments = V.shape[0] // i4
        for seg in range(n_segments):
            b = seg * i4
            for t in range(n_timesteps):
                old = np.array([V[b + i0, t], V[b + i1, t], V[b + i2, t]], dtype=np.float64)
                new = model_vec_to_viser_xyz(old)
                V[b + i0, t] = new[0]
                V[b + i1, t] = new[1]
                V[b + i2, t] = new[2]
        results.discretization_history[k] = V


def prepare_rocket_results_for_viser(results) -> None:
    """Remap PDG model-frame trajectories and attitudes for Viser visualization."""
    remap_optimization_results_for_viser_xyz(results)
    remap_attitude_for_viser(results)


n = 5

# Fixed physical and scenario constants
gI = 1.0
l_arm = 0.25
J_diag = np.array([0.168 * 2e-2, 0.168, 0.168])
J_mat = ox.Diag(J_diag)
J_inv_mat = ox.Inv(ox.Diag(J_diag))
g0 = 1.0
Isp = 30.0
m_dry = 1.0
v_max = 3.0
w_max = 0.3752
del_max = 20.0
theta_max = 75.0
T_min = 1.5
T_max = 6.5
gamma = 75.0
beta = 0.01
c_ax = 0.5
c_ayz = 1.0
S_a = 0.5
rho = 1.0
l_p = 0.05
initial_position = np.array([7.5, 4.5, 2.5])
final_position = np.array([0.0, 0.0])

# Concat (not Stack): JAX lowering stacks scalars as (3,1); Diag needs a length-3 vector (3,).
CA = ox.Diag(ox.Concat(c_ax, c_ayz, c_ayz))
r_arm = ox.Concat(-l_arm, 0.0, 0.0)
r_cp = ox.Concat(l_p, 0.0, 0.0)

mass = ox.State("mass", shape=(1,))
mass.max = [2.0]
mass.min = [1.0]
mass.initial = [2.0]
mass.final = [ox.Maximize(1.5)]

position = ox.State("position", shape=(3,))
position.max = [10.0, 10.0, 10.0]
position.min = [-10.0, -10.0, -10.0]
position.initial = [
    ox.Free(float(initial_position[0])),
    ox.Free(float(initial_position[1])),
    ox.Free(float(initial_position[2])),
]
position.final = [ox.Free(0.0), ox.Free(0.0), 0.0]

velocity = ox.State("velocity", shape=(3,))
velocity.max = [v_max, v_max, v_max]
velocity.min = [-v_max, -v_max, -v_max]
velocity.initial = [-0.5, -2.8, 0.0]
velocity.final = [-0.1, 0.0, 0.0]

attitude = ox.State("attitude", shape=(4,))
attitude.max = [1.0, 1.0, 1.0, 1.0]
attitude.min = [-1.0, -1.0, -1.0, -1.0]
attitude.initial = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0), ox.Free(1.0)]
attitude.final = [0.0, 0.0, 0.0, 1.0]

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = [w_max, w_max, w_max]
angular_velocity.min = [-w_max, -w_max, -w_max]
angular_velocity.initial = [1e-8, 0.0, 0.0]
angular_velocity.final = [1e-8, 0.0, 0.0]

thrust = ox.Control("thrust", shape=(3,))
thrust.max = [T_max, T_max, T_max]
thrust.min = [-T_max, -T_max, -T_max]
thrust.guess = np.linspace(
    np.array([gI * mass.initial[0], 0, 0]),
    np.array([gI * m_dry, 0, 0]),
    n,
).reshape(-1, 3)

# Extract quaternion components
q1 = attitude[0]
q2 = attitude[1]
q3 = attitude[2]
q4 = attitude[3]

# Direction cosine matrix (DCM) from quaternion
CBI = ox.Block(
    [
        [q4**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q4 * q3), 2 * (q4 * q2 + q1 * q3)],
        [2 * (q4 * q3 + q1 * q2), q4**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q4 * q1)],
        [2 * (q1 * q3 - q4 * q2), 2 * (q4 * q1 + q2 * q3), q4**2 - q1**2 - q2**2 + q3**2],
    ]
).T  # Transpose to get inertial to body frame


def cross(a, b):
    """Cross product of two vectors"""
    return ox.Concat(
        a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
    )


w1 = angular_velocity[0]
w2 = angular_velocity[1]
w3 = angular_velocity[2]

q1_dot = 0.5 * (w1 * q4 - w2 * q3 + w3 * q2)
q2_dot = 0.5 * (w1 * q3 - w3 * q1 + w2 * q4)
q3_dot = 0.5 * (w2 * q1 - w1 * q2 + w3 * q4)
q4_dot = -0.5 * (w1 * q1 + w2 * q2 + w3 * q3)

attitude_dot = ox.Concat(q1_dot, q2_dot, q3_dot, q4_dot)

# Aerodynamic force in body frame (computed symbolically)
A = -0.5 * rho * ox.linalg.Norm(velocity) * S_a * CA @ CBI @ velocity

dynamics = {
    "mass": -(1 / (Isp * g0)) * ox.linalg.Norm(thrust) - beta,
    "position": velocity,
    "velocity": CBI.T @ (thrust + A) / mass[0] + ox.Concat(-gI, 0.0, 0.0),
    "attitude": attitude_dot,
    "angular_velocity": J_inv_mat
    @ (cross(r_arm, thrust) + cross(r_cp, A) - cross(angular_velocity, (J_mat @ angular_velocity))),
}


states = [mass, position, velocity, attitude, angular_velocity]
controls = [thrust]

constraint_exprs = []
for state in states:
    constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Boundary Constraints
# Initial position constraint
constraint_exprs.append((position == initial_position).convex().at([0]))

# Terminal position constraint
constraint_exprs.append((position[0:2] == final_position).convex().at([n - 1]))


constraint_exprs.append(ox.ctcs(1.0 * (mass - m_dry) >= 0))
constraint_exprs.append(
    ox.ctcs(
        0.1 * ox.linalg.Norm(position[1:])
        - ox.Tan(ox.Constant(np.array(gamma * np.pi / 180.0))) * position[0]
        <= 0
    )
)
constraint_exprs.append(ox.ctcs(0.1 * ox.linalg.Norm(velocity) ** 2 - v_max**2 <= 0))
constraint_exprs.append(
    ox.ctcs(
        1.0 * ox.Cos(ox.Constant(np.array(theta_max * np.pi / 180.0))) - 1.0 + 2.0 * (q2**2 + q3**2)
        <= 0
    )
)
constraint_exprs.append(ox.ctcs(1.0 * ox.linalg.Norm(angular_velocity) ** 2 - w_max**2 <= 0))
constraint_exprs.append(
    ox.ctcs(
        0.1 * ox.linalg.Norm(thrust)
        - thrust[0] / ox.Cos(ox.Constant(np.array(del_max * np.pi / 180.0)))
        <= 0
    )
)
constraint_exprs.append(ox.ctcs(0.1 * ox.linalg.Norm(thrust) ** 2 - T_max**2 <= 0))
constraint_exprs.append(ox.ctcs(0.1 * T_min**2 - ox.linalg.Norm(thrust) ** 2 <= 0))

# Nominal final time (must match free-final guess). Old API used
# time_dilation_factor_min/max as factors times this value for absolute bounds.
t_final_guess = 10.0
time = ox.Time(
    initial=0.0,
    final=ox.Free(t_final_guess),
    min=0.0,
    max=10.0,
    time_dilation_min=0.2 * t_final_guess,
    time_dilation_max=2.0 * t_final_guess,
)

import diffrax as dfx  # noqa: F401

problem = Problem(
    N=n,
    states=states,
    controls=controls,
    dynamics=dynamics,
    constraints=constraint_exprs,
    time=time,
    float_dtype="float64",
    algorithm={
        "autotuner": ox.ConstantProximalWeight(),
        "lam_cost": 1e-2,
        "lam_vc": 1e1,
        "lam_prox": 1e0,
        "ep_tr": 5e-3,
        "ep_vc": 1e-6,
    },
    discretizer={
        "diffrax_kwargs": {"stepsize_controller": dfx.StepTo(np.linspace(0.0, 1 / (n - 1), 3))}
    },
)

problem.settings.dev.printing = False

if __name__ == "__main__":
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()
    prepare_rocket_results_for_viser(result)

    vehicle_mesh = load_rocket_vehicle_mesh()
    if vehicle_mesh is not None:
        print("[viser] vehicle_mesh: Rocket.STEP (decimated)")
    else:
        print("[viser] vehicle_mesh: None — using attitude axes")

    # Create PDG trajectory visualization
    traj_server = create_animated_plotting_server(
        result,
        thrust_key="thrust",
        thrust_style="plume",
        thrust_scale=0.4,
        thrust_plume_half_angle_deg=14.0,
        thrust_plume_color=(255, 130, 50),
        thrust_plume_opacity=0.5,
        thrust_remap_world_to_viser=True,
        vehicle_mesh=vehicle_mesh,
        vehicle_mesh_color=(185, 190, 200),
    )

    # Create SCP iteration visualization
    scp_server = create_scp_animated_plotting_server(
        result,
        frame_duration_ms=50.0,
    )

    snapshot_server = create_snapshot_plotting_server(
        result,
        initial_n_snapshots=5,
        show_grid=True,
        vehicle_mesh=vehicle_mesh,
        vehicle_mesh_color=(185, 190, 200),
    )

    # Keep both servers running
    traj_server.sleep_forever()
