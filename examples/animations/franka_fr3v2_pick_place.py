"""Cinematic offline render of the Franka FR3 v2 pick-and-place example.

The optimization problem itself lives in
``examples/arm/franka_fr3v2_pick_place.py``; this script imports the problem,
solves it, builds a viser scene, and renders an mp4 with a static overview
camera that frames the full arm/workspace.

Run with:

    python examples/animations/franka_fr3v2_pick_place.py
"""

import os
import sys
from pathlib import Path

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.animations._camera import overview_pose
from examples.animations._render import render_animation_to_video
from examples.arm.franka_fr3v2_pick_place import (
    N_JOINTS,
    _joint_zero_pos,
    obstacle_center,
    obstacle_radius,
    problem,
    waypoint_positions,
)
from examples.plotting_viser import AnimatedServerHandle
from openscvx.plotting.viser import (
    add_animated_trail,
    add_ellipsoid_obstacles,
    add_ghost_trajectory,
    add_position_marker,
    add_target_markers,
    compute_velocity_colors,
    create_server,
)

# --- Render settings ---------------------------------------------------------
OUTPUT_PATH = os.path.join(current_dir, "mp4", "franka_fr3v2_pick_place_overview.mp4")
WIDTH = 1080
HEIGHT = 1080
FPS = 60
CRF = 16

# Oversampling for smooth trails.
STRIDE = 4
PROPAGATION_HZ = FPS * STRIDE

# --- Camera settings ---------------------------------------------------------
OVERVIEW_AZIMUTH = np.radians(55.0)
OVERVIEW_ELEVATION = np.radians(17.0)
# Slightly tighter framing while keeping the full FR3 visible.
OVERVIEW_RADIUS_MARGIN = 0.5
OVERVIEW_FOV_DEG = 60.0


if __name__ == "__main__":
    problem.settings.prp.dt = 1.0 / PROPAGATION_HZ
    problem.initialize()
    problem.solve()
    results = problem.post_process()

    ee_pos = np.asarray(results.trajectory["ee_position"], dtype=np.float64)
    n_frames = len(ee_pos)

    # Joint keypoints from propagated transforms.
    keypoints = np.zeros((n_frames, N_JOINTS + 1, 3))
    for t_idx in range(n_frames):
        for k in range(N_JOINTS):
            T_k = np.asarray(results.trajectory[f"T_j{k + 1}"])[t_idx]
            q0 = np.append(_joint_zero_pos[k], 1.0)
            keypoints[t_idx, k] = (T_k @ q0)[:3]
        T_7 = np.asarray(results.trajectory["T_j7"])[t_idx]
        keypoints[t_idx, N_JOINTS] = T_7[:3, 3]

    # -- Build viser scene ----------------------------------------------------
    server = create_server(ee_pos, show_grid=False)
    server.scene.add_grid("/grid", width=1.8, height=1.8, cell_size=0.25)
    server.scene.add_frame("/origin", axes_length=0.1, axes_radius=0.003)

    add_target_markers(
        server,
        waypoint_positions,
        radius=0.012,
        colors=[
            (100, 150, 255),
            (255, 180, 50),
            (255, 50, 50),
            (255, 180, 50),
            (255, 180, 50),
            (255, 50, 50),
        ],
    )

    add_ellipsoid_obstacles(
        server,
        centers=[obstacle_center],
        radii=[np.full(3, 1.0 / obstacle_radius)],
    )

    ee_colors = compute_velocity_colors(np.asarray(results.trajectory.get("velocity")))
    add_ghost_trajectory(server, ee_pos, ee_colors, point_size=0.005)
    _, update_trail = add_animated_trail(server, ee_pos, ee_colors, point_size=0.008)
    _, update_marker = add_position_marker(server, ee_pos, radius=0.012)

    # Prefer full FR3 CAD meshes from menagerie assets.
    update_arm = None
    try:
        import mujoco
        import trimesh  # type: ignore

        from openscvx.integrations.menagerie import get_xml_path

        xml_path = Path(str(get_xml_path("franka_fr3_v2", prefer_mjx=False)))
        mj_model_vis = mujoco.MjModel.from_xml_path(str(xml_path))
        mj_data_vis = mujoco.MjData(mj_model_vis)
        asset_dir = xml_path.parent / "assets"

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

        link_meshes_local = {}
        link_body_ids = {}
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
            link_meshes_local[link_name] = (np.vstack(all_verts), np.vstack(all_faces))
            link_body_ids[link_name] = mujoco.mj_name2id(
                mj_model_vis, mujoco.mjtObj.mjOBJ_BODY, link_name
            )

        if not link_meshes_local:
            raise RuntimeError("No FR3 mesh assets were loaded.")

        angle_traj = np.asarray(results.trajectory["angle"], dtype=np.float64)
        link_world_T = {name: np.zeros((n_frames, 4, 4)) for name in link_meshes_local}
        for t_idx in range(n_frames):
            mj_data_vis.qpos[:N_JOINTS] = angle_traj[t_idx]
            mujoco.mj_kinematics(mj_model_vis, mj_data_vis)
            for name, body_id in link_body_ids.items():
                T = np.eye(4)
                T[:3, :3] = mj_data_vis.xmat[body_id].copy().reshape(3, 3)
                T[:3, 3] = mj_data_vis.xpos[body_id].copy()
                link_world_T[name][t_idx] = T

        link_colors = {
            "fr3v2_link0": (120, 120, 120),
            "fr3v2_link1": (215, 215, 218),
            "fr3v2_link2": (215, 215, 218),
            "fr3v2_link3": (215, 215, 218),
            "fr3v2_link4": (215, 215, 218),
            "fr3v2_link5": (210, 210, 215),
            "fr3v2_link6": (200, 205, 215),
            "fr3v2_link7": (190, 195, 210),
        }

        link_handles = {}
        for link_name, (verts_local, faces) in link_meshes_local.items():
            T0 = link_world_T[link_name][0]
            verts_world = (T0[:3, :3] @ verts_local.T).T + T0[:3, 3]
            link_handles[link_name] = (  # (mesh_handle, verts_local)
                server.scene.add_mesh_simple(
                    f"/robot/{link_name}",
                    vertices=verts_world.astype(np.float32),
                    faces=faces,
                    color=link_colors.get(link_name, (210, 210, 215)),
                    opacity=1.0,
                ),
                verts_local,
            )

        def update_arm(frame_idx: int) -> None:
            for link_name, (handle, verts_local) in link_handles.items():
                T = link_world_T[link_name][frame_idx]
                handle.vertices = ((T[:3, :3] @ verts_local.T).T + T[:3, 3]).astype(np.float32)

        print(f"[animation] Using FR3 menagerie meshes from {asset_dir}")
    except Exception as exc:
        print(f"[animation] Mesh visualisation unavailable ({type(exc).__name__}: {exc})")
        print("[animation] Falling back to line-segment arm rendering.")

        n_segs = N_JOINTS + 1
        seg_col = np.full((n_segs, 2, 3), [210, 210, 210], dtype=np.uint8)

        def _build_segments(frame_idx: int) -> np.ndarray:
            pts = np.zeros((n_segs, 2, 3), dtype=np.float32)
            pts[0] = [np.zeros(3), keypoints[frame_idx, 0]]
            for k in range(N_JOINTS - 1):
                pts[k + 1] = [keypoints[frame_idx, k], keypoints[frame_idx, k + 1]]
            pts[N_JOINTS] = [keypoints[frame_idx, N_JOINTS - 1], keypoints[frame_idx, N_JOINTS]]
            return pts

        arm_handle = server.scene.add_line_segments(
            "/arm_links",
            points=_build_segments(0),
            colors=seg_col,
            line_width=5.0,
        )

        def update_arm(frame_idx: int) -> None:
            arm_handle.points = _build_segments(frame_idx)

    traj_time = np.asarray(results.trajectory["time"], dtype=np.float64).flatten()
    handle = AnimatedServerHandle(
        server=server,
        traj_time=traj_time,
        update_callbacks=[update_trail, update_marker, update_arm],
    )

    # Camera framing: include EE trajectory, waypoints, and all joints to ensure
    # the full arm remains visible.
    all_points = np.vstack(
        [
            ee_pos,
            np.asarray(waypoint_positions, dtype=np.float64),
            keypoints.reshape(-1, 3),
            np.zeros((1, 3)),
        ]
    )
    static_pose = overview_pose(
        all_points,
        azimuth=OVERVIEW_AZIMUTH,
        elevation=OVERVIEW_ELEVATION,
        radius_margin=OVERVIEW_RADIUS_MARGIN,
        fov_deg=OVERVIEW_FOV_DEG,
    )

    def camera_pose_fn(frame_idx: int):
        return static_pose

    render_animation_to_video(
        handle,
        OUTPUT_PATH,
        camera_pose_fn,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        crf=CRF,
        stride=STRIDE,
    )
