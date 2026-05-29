"""Franka FR3 v2 view planning with nodal viewcone constraints — PoE FK.

Same kinematics and MuJoCo Menagerie visualisation as
``franka_fr3v2_pick_place.py``, but the task matches ``7_dof_arm_vp.py``:
move the end-effector while keeping several workspace points inside a
wrist-mounted camera field of view (**nodal** viewcone inequality at each node).

Kinematics
----------
Space-frame Product of Exponentials (PoE) with screw axes from
``franka_fr3_v2/fr3v2.xml`` (see pick-place example for derivation).

Task
----
- Terminal EE position at a goal (1 cm tolerance).
- Viewcone: each viewpoint target must satisfy
      ||A_cone @ p_sensor|| - c^T @ p_sensor <= 0
  with p_sensor = R_sb @ R_ee^T @ (p_target - p_ee).

Requires
--------
    pip install openscvx[lie]
    (optional) pip install mujoco trimesh
"""

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting import plot_camera_view
from examples.plotting_viser import (
    build_arm_line_snapshot_builder,
    build_cad_link_snapshot_builder,
    compute_poe_joint_keypoints,
    create_snapshot_plotting_server,
)
from openscvx.plotting import plot_controls, plot_scp_iterations

# =============================================================================
# FR3 v2 Kinematic Parameters (same as franka_fr3v2_pick_place.py)
# =============================================================================

N_JOINTS = 7

d1 = 0.333
d3 = 0.316
a4 = 0.0825
d5 = 0.384
d7 = 0.088
d_ee = 0.107

z1 = d1
z35 = d1 + d3
z567 = d1 + d3 + d5

angle_min = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
angle_max = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])

torque_max = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

inertia = np.array([0.10, 0.08, 0.07, 0.06, 0.03, 0.02, 0.01])

q_home = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4])

screw_axes = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [-z1, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [z35, 0.0, -a4, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [z567, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, d7, 0.0, 0.0, 0.0, -1.0],
    ]
)

T_home = np.array(
    [
        [1.0, 0.0, 0.0, d7],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, z567 - d_ee],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# =============================================================================
# Wrist camera (viewcone) — same convention as 7_dof_arm_vp.py
# =============================================================================

alpha_x = 6.0
alpha_y = 6.0
A_cone = np.diag(
    [
        1 / np.tan(np.pi / alpha_x),
        1 / np.tan(np.pi / alpha_y),
        0,
    ]
)
c = jnp.array([0, 0, 1])
norm_type = 2
# R_sb = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
R_sb = np.eye(3)

vp_targets = np.array(
    [
        [0.25, 0.00, 0.10],
        [0.35, 0.00, 0.10],
        [0.30, 0.05, 0.05],
    ]
)

# =============================================================================
# Discretisation
# =============================================================================

n = 2
total_time = 1.5

# =============================================================================
# States and controls
# =============================================================================

angle = ox.State("angle", shape=(N_JOINTS,))
angle.max = angle_max
angle.min = angle_min
angle.initial = np.clip(q_home, angle_min, angle_max)
angle.final = [("free", 0.0)] * N_JOINTS

velocity = ox.State("velocity", shape=(N_JOINTS,))
velocity.max = np.full(N_JOINTS, 2.5)
velocity.min = -velocity.max
velocity.initial = np.zeros(N_JOINTS)
velocity.final = np.zeros(N_JOINTS)

states = [angle, velocity]

torque = ox.Control("torque", shape=(N_JOINTS,))
torque.max = torque_max
torque.min = -torque_max

controls = [torque]

# =============================================================================
# Forward kinematics (PoE)
# =============================================================================

T_chain = ox.Constant(np.eye(4))
joint_transforms = {}
for j in range(N_JOINTS):
    xi = ox.Constant(screw_axes[j])
    T_chain = T_chain @ ox.lie.SE3Exp(xi * angle[j])
    joint_transforms[f"T_j{j + 1}"] = T_chain

T_ee = T_chain @ ox.Constant(T_home)
p_ee = ox.Concat(T_ee[0, 3], T_ee[1, 3], T_ee[2, 3])
R_ee = T_ee[:3, :3]

# =============================================================================
# Dynamics
# =============================================================================

I_inv = ox.Constant(1.0 / inertia)
dynamics = {
    "angle": velocity,
    "velocity": I_inv * torque,
}

# =============================================================================
# Task: start / goal EE positions (same style as 7_dof_arm_vp)
# =============================================================================

home_ee_pos = np.array([0.3, 0.25, 0.25])
goal_ee_pos = np.array([0.3, -0.25, 0.25])

# =============================================================================
# Constraints
# =============================================================================

target = ox.Parameter("target", shape=(3,), value=goal_ee_pos)

constraints = []

for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

constraints.append(ox.ctcs(p_ee[2] >= 0.0))

ee_tolerance = 0.01
constraints.append((ox.linalg.Norm(p_ee - target, ord=2) <= ee_tolerance).at([n - 1]))


def g_vp(p_target):
    p_s_s = R_sb @ R_ee.T @ (p_target - p_ee)
    return ox.linalg.Norm(A_cone @ p_s_s, ord=norm_type) - (c.T @ p_s_s)


constraints.append(
    ox.Vmap(
        lambda pose: g_vp(pose),
        batch=vp_targets,
    )
    <= 0.0
)

# =============================================================================
# Initial guess (IK) — point camera boresight toward viewpoint centroid
# =============================================================================

mean_vp = np.mean(vp_targets, axis=0)
boresight_body = np.array(R_sb).T @ np.array(c)


def look_at_quat(ee_pos, target_pos):
    """Quaternion (wxyz) aligning body boresight with (target_pos - ee_pos)."""
    d = target_pos - ee_pos
    d = d / np.linalg.norm(d)
    q_xyz = np.cross(boresight_body, d)
    q_w = np.sqrt(np.dot(boresight_body, boresight_body) * np.dot(d, d)) + np.dot(boresight_body, d)
    q = np.hstack(([q_w], q_xyz))
    return q / np.linalg.norm(q)


q_home_orient = look_at_quat(home_ee_pos, mean_vp)
q_goal_orient = look_at_quat(goal_ee_pos, mean_vp)

angle.guess = ox.init.ik_interpolation(
    keyframes=[
        (home_ee_pos, q_home_orient),
        (goal_ee_pos, q_goal_orient),
    ],
    nodes=[0, n - 1],
    screw_axes=screw_axes,
    T_home=T_home,
    angles_init=angle.initial,
    angles_min=angle_min,
    angles_max=angle_max,
)
angle.initial = np.clip(angle.guess[0], angle_min, angle_max)
velocity.guess = np.zeros((n, N_JOINTS))
torque.guess = np.zeros((n, N_JOINTS))

# =============================================================================
# Problem
# =============================================================================

time = ox.Time(
    initial=0.0,
    final=ox.Minimize(total_time),
    min=0.0,
    max=total_time,
)

problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    algorithm={
        "lam_vb": 1e0,
        "lam_vc": 1e1,
        "autotuner": ox.AugmentedLagrangian(eta_lambda=1e0),
    },
    float_dtype="float64",
    licq_max=1e-8,
    algebraic_prop={
        "ee_position": p_ee,
        **{name: T for name, T in joint_transforms.items()},
    },
)

problem.settings.prp.dt = 0.01

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Franka FR3 v2 View Planning (nodal) — PoE FK + wrist camera viewcone")
    print("=" * 60)
    print(f"Nodes: {n}  |  View targets: {len(vp_targets)}")
    print(f"Start EE (nominal): {home_ee_pos}")
    print(f"Goal EE target:      {target.value}")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    angle_traj = np.asarray(results.trajectory["angle"])
    n_frames = len(angle_traj)
    traj_time = np.asarray(results.trajectory["time"]).flatten()
    T_home_np = np.asarray(T_home)
    T_j7_traj = np.asarray(results.trajectory["T_j7"])
    ee_quats = np.zeros((n_frames, 4))
    try:
        import jaxlie

        for t in range(n_frames):
            t_ee = T_j7_traj[t] @ T_home_np
            ee_quats[t] = jaxlie.SO3.from_matrix(t_ee[:3, :3]).wxyz
    except ImportError:
        from scipy.spatial.transform import Rotation

        for t in range(n_frames):
            q_xyzw = Rotation.from_matrix((T_j7_traj[t] @ T_home_np)[:3, :3]).as_quat()
            ee_quats[t] = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    results.update_plotting_data(
        init_poses=[np.asarray(p) for p in vp_targets],
        R_sb=np.array(R_sb),
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        norm_type=norm_type,
        T_home=T_home_np,
        ee_attitude=ee_quats,
        initial_n_snapshots=5,
    )

    plot_camera_view(results, None).show()

    ee_pos_traj = results.trajectory["ee_position"]
    final_q = results.trajectory["angle"][-1]
    final_ee = ee_pos_traj[-1]
    tgt = target.value
    err = np.linalg.norm(final_ee - tgt)

    print()
    print("Results:")
    print(f"  Final joint angles [deg]: {np.round(np.rad2deg(final_q), 1)}")
    print(f"  Final EE: [{final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f}]")
    print(f"  Goal:     [{tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}]")
    print(f"  Position error: {err:.4f} m")

    plot_scp_iterations(results).show()
    plot_controls(results).show()

    # =========================================================================
    # Viser 3D
    # =========================================================================

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        add_ghost_trajectory,
        add_position_marker,
        add_target_markers,
        add_viewcone,
        compute_velocity_colors,
        create_server,
    )

    ee_pos = np.asarray(ee_pos_traj)

    _joint_zero_pos = np.array(
        [
            [0.0, 0.0, z1],
            [0.0, 0.0, z1],
            [0.0, 0.0, z35],
            [a4, 0.0, z35],
            [0.0, 0.0, z567],
            [0.0, 0.0, z567],
            [d7, 0.0, z567],
        ]
    )
    keypoints = compute_poe_joint_keypoints(
        results, _joint_zero_pos, N_JOINTS, t_home=T_home_np
    )

    _use_cad_mesh = False
    _link_meshes_local: dict = {}
    _link_body_ids: dict = {}
    _link_world_T: dict = {}

    try:
        import mujoco
        import trimesh  # type: ignore

        from openscvx.integrations.menagerie import get_xml_path

        xml_path = str(get_xml_path("franka_fr3_v2", prefer_mjx=False))
        mj_model_vis = mujoco.MjModel.from_xml_path(xml_path)
        mj_model_vis.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        mj_data_vis = mujoco.MjData(mj_model_vis)
        asset_dir = Path(xml_path).parent / "assets"

        link_visual_files = {
            "fr3v2_link0": [
                "link0_0.obj",
                "link0_1.obj",
                "link0_2.obj",
                "link0_3.obj",
                "link0_4.obj",
                "link0_5.obj",
            ],
            "fr3v2_link1": ["link1_3.obj"],
            "fr3v2_link2": ["link2_3.obj"],
            "fr3v2_link3": ["link3_3.obj"],
            "fr3v2_link4": ["link4_3.obj"],
            "fr3v2_link5": ["link5_3.obj", "link5_4.obj"],
            "fr3v2_link6": [
                "link6_0.obj",
                "link6_1.obj",
                "link6_2.obj",
                "link6_3.obj",
                "link6_4.obj",
                "link6_5.obj",
                "link6_6.obj",
                "link6_7.obj",
                "link6_8.obj",
                "link6_9.obj",
            ],
            "fr3v2_link7": [
                "link7_0.obj",
                "link7_1.obj",
                "link7_2.obj",
                "link7_3.obj",
                "link7_4.obj",
                "link7_5.obj",
                "link7_6.obj",
            ],
        }

        for link_name, files in link_visual_files.items():
            all_verts, all_faces, offset = [], [], 0
            for fname in files:
                obj_path = asset_dir / fname
                if not obj_path.exists():
                    continue
                tm = trimesh.load(str(obj_path), force="mesh", process=False)
                all_verts.append(np.asarray(tm.vertices, dtype=np.float32))
                all_faces.append(np.asarray(tm.faces, dtype=np.uint32) + offset)
                offset += len(tm.vertices)
            if not all_verts:
                continue
            _link_meshes_local[link_name] = (np.vstack(all_verts), np.vstack(all_faces))
            _link_body_ids[link_name] = mujoco.mj_name2id(
                mj_model_vis, mujoco.mjtObj.mjOBJ_BODY, link_name
            )

        for name in _link_meshes_local:
            _link_world_T[name] = np.zeros((n_frames, 4, 4))

        for t_idx in range(n_frames):
            mj_data_vis.qpos[:N_JOINTS] = angle_traj[t_idx]
            mujoco.mj_kinematics(mj_model_vis, mj_data_vis)
            for name, body_id in _link_body_ids.items():
                T = np.eye(4)
                T[:3, :3] = mj_data_vis.xmat[body_id].copy().reshape(3, 3)
                T[:3, 3] = mj_data_vis.xpos[body_id].copy()
                _link_world_T[name][t_idx] = T

        _use_cad_mesh = len(_link_meshes_local) > 0
        if _use_cad_mesh:
            print(
                f"[viser] Loaded {len(_link_meshes_local)} CAD link meshes from menagerie assets."
            )

    except Exception as exc:
        print(f"[viser] CAD mesh unavailable ({type(exc).__name__}: {exc}); using line segments.")

    server = create_server(ee_pos, show_grid=False)
    server.scene.add_grid("/grid", width=1.5, height=1.5, cell_size=0.2)
    server.scene.add_frame("/origin", axes_length=0.08, axes_radius=0.003)

    tgt_arr = np.asarray(target.value).reshape(1, 3)
    add_target_markers(server, tgt_arr, radius=0.015, colors=[(255, 50, 50)])
    add_target_markers(
        server,
        vp_targets,
        radius=0.01,
        colors=[(50, 255, 50)] * len(vp_targets),
    )

    ee_colors = compute_velocity_colors(np.asarray(results.trajectory.get("velocity")))
    add_ghost_trajectory(server, ee_pos, ee_colors, point_size=0.005)
    _, update_trail = add_animated_trail(server, ee_pos, ee_colors, point_size=0.008)
    _, update_marker = add_position_marker(server, ee_pos, radius=0.012)

    _, update_viewcone = add_viewcone(
        server,
        ee_pos,
        ee_quats,
        half_angle_x=np.pi / alpha_x,
        half_angle_y=np.pi / alpha_y,
        scale=0.15,
        norm_type=norm_type,
        R_sb=np.array(R_sb),
        color=(80, 180, 200),
        opacity=0.3,
    )

    update_robot = None

    if _use_cad_mesh:
        _link_color = {
            "fr3v2_link0": (120, 120, 120),
            "fr3v2_link1": (215, 215, 218),
            "fr3v2_link2": (215, 215, 218),
            "fr3v2_link3": (215, 215, 218),
            "fr3v2_link4": (215, 215, 218),
            "fr3v2_link5": (210, 210, 215),
            "fr3v2_link6": (200, 205, 215),
            "fr3v2_link7": (190, 195, 210),
        }
        _link_handles = {}
        for link_name, (verts_local, faces) in _link_meshes_local.items():
            T0 = _link_world_T[link_name][0]
            verts_world = (T0[:3, :3] @ verts_local.T).T + T0[:3, 3]
            handle = server.scene.add_mesh_simple(
                f"/robot/{link_name}",
                vertices=verts_world.astype(np.float32),
                faces=faces,
                color=_link_color.get(link_name, (210, 210, 215)),
                opacity=1.0,
            )
            _link_handles[link_name] = (handle, verts_local)

        def update_robot(frame_idx: int) -> None:
            for link_name, (handle, verts_local) in _link_handles.items():
                T = _link_world_T[link_name][frame_idx]
                verts_world = (T[:3, :3] @ verts_local.T).T + T[:3, 3]
                handle.vertices = verts_world.astype(np.float32)

    else:
        n_segs = N_JOINTS + 1
        seg_col = np.full((n_segs, 2, 3), [200, 200, 200], dtype=np.uint8)

        def _build_segments(frame_idx: int) -> np.ndarray:
            pts = np.zeros((n_segs, 2, 3), dtype=np.float32)
            pts[0] = [np.zeros(3), keypoints[frame_idx, 0]]
            for k in range(N_JOINTS - 1):
                pts[k + 1] = [keypoints[frame_idx, k], keypoints[frame_idx, k + 1]]
            pts[N_JOINTS] = [
                keypoints[frame_idx, N_JOINTS - 1],
                keypoints[frame_idx, N_JOINTS],
            ]
            return pts

        arm_handle = server.scene.add_line_segments(
            "/arm_links",
            points=_build_segments(0),
            colors=seg_col,
            line_width=5.0,
        )

        def update_robot(frame_idx: int) -> None:
            arm_handle.points = _build_segments(frame_idx)

    callbacks = [update_trail, update_marker, update_viewcone]
    if update_robot is not None:
        callbacks.append(update_robot)
    add_animation_controls(server, traj_time, callbacks, loop=True)

    _cad_snapshot_builder = None
    if _use_cad_mesh:
        _cad_snapshot_builder = build_cad_link_snapshot_builder(
            _link_meshes_local,
            _link_world_T,
            link_colors=_link_color,
        )
    create_snapshot_plotting_server(
        results,
        position_key="ee_position",
        velocity_key="velocity",
        attitudes=ee_quats,
        show_body_frame=True,
        show_viewcone=True,
        viewcone_scale=0.15,
        target_positions=[np.asarray(p) for p in vp_targets],
        target_radius=0.01,
        arm_keypoints=None if _use_cad_mesh else keypoints,
        snapshot_builder=_cad_snapshot_builder,
        initial_n_snapshots=5,
        ghost_point_size=0.005,
    )

    server.sleep_forever()
