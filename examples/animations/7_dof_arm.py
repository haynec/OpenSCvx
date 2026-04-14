"""Cinematic offline render of the 7-DOF arm pick-and-place example.

The trajectory optimization problem itself lives in
``examples/arm/7_dof_arm.py``; this file imports that ``problem`` and the
robot parameters, solves it, builds a viser scene with arm links / joint
frames / EE trail, and drives it frame-by-frame while piping raw RGB into
ffmpeg to produce an mp4.

Run it with::

    python examples/animations/7_dof_arm.py

The script prints a viser URL and waits. Open the URL in a browser — as soon
as the client connects, the render begins. Requires ``ffmpeg`` on ``PATH``;
``openscvx`` does not depend on it.

Only one camera mode is provided — ``"overview"`` — a static elevated camera
that frames the full pick-and-place workspace.

Tweak ``OUTPUT_PATH`` / ``WIDTH`` / ``HEIGHT`` / ``FPS`` below for different
output variants.
"""

import importlib
import os
import sys

import numpy as np

# Add the project root so `examples.*` imports resolve.
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.animations._camera import overview_pose
from examples.animations._render import render_animation_to_video
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

# Import the arm example by path (filename starts with a digit).
_arm_spec = importlib.util.spec_from_file_location(
    "arm_7dof", os.path.join(grandparent_dir, "examples", "arm", "7_dof_arm_collision.py")
)
_arm_mod = importlib.util.module_from_spec(_arm_spec)
_arm_spec.loader.exec_module(_arm_mod)

problem = _arm_mod.problem
d1 = _arm_mod.d1
a2 = _arm_mod.a2
a3 = _arm_mod.a3
a4 = _arm_mod.a4
joint_names = _arm_mod.joint_names
waypoint_names = _arm_mod.waypoint_names
waypoint_positions = _arm_mod.waypoint_positions
obstacle_center = _arm_mod.obstacle_center
obstacle_radius = _arm_mod.obstacle_radius

# Keypoint home positions (derived from link lengths — defined in __main__ of
# the arm example, so we replicate them here).
keypoint_home_positions = {
    "base": np.array([0.0, 0.0, 0.0]),
    "shoulder": np.array([0.0, 0.0, d1]),
    "elbow": np.array([a2, 0.0, d1]),
    "wrist": np.array([a2 + a3, 0.0, d1]),
    "ee": np.array([a2 + a3 + a4, 0.0, d1]),
}

# --- Render settings ---------------------------------------------------------
OUTPUT_PATH = os.path.join(current_dir, "7_dof_arm_overview.mp4")
WIDTH = 1080
HEIGHT = 1080
FPS = 60
CRF = 16

# Oversampling for smooth trails.
STRIDE = 4
PROPAGATION_HZ = FPS * STRIDE

# --- Camera settings ---------------------------------------------------------
OVERVIEW_AZIMUTH = np.radians(60.0)
OVERVIEW_ELEVATION = np.radians(15.0)
OVERVIEW_RADIUS_MARGIN = 1.0
OVERVIEW_FOV_DEG = 60.0


if __name__ == "__main__":
    import jaxlie

    problem.settings.prp.dt = 1.0 / PROPAGATION_HZ
    problem.initialize()
    problem.solve()
    results = problem.post_process()

    ee_pos = np.asarray(results.trajectory["ee_position"], dtype=np.float64)
    n_frames = len(ee_pos)

    # -- Extract joint keypoint positions from propagated transforms -----------
    keypoints = np.zeros((n_frames, 5, 3))
    joint_quats = np.zeros((n_frames, 5, 4))
    for k, name in enumerate(joint_names):
        T_k = np.asarray(results.trajectory[f"T_{name}"])  # (T, 4, 4)
        p_home = np.append(keypoint_home_positions[name], 1.0)
        keypoints[:, k] = (T_k @ p_home)[:, :3]
        joint_quats[:, k] = np.array(
            [jaxlie.SO3.from_matrix(T_k[t, :3, :3]).wxyz for t in range(n_frames)]
        )

    # -- Build viser scene (mirrors examples/arm/7_dof_arm.py) -----------------
    server = create_server(ee_pos, show_grid=False)
    server.scene.add_grid("/grid", width=1.5, height=1.5, cell_size=0.25)
    server.scene.add_frame("/origin", axes_length=0.1, axes_radius=0.003)

    # Waypoint markers
    marker_colors = {
        "home": (100, 150, 255),
        "pre_grasp": (255, 180, 50),
        "grasp": (255, 50, 50),
        "pre_place": (255, 180, 50),
        "place": (255, 50, 50),
    }
    add_target_markers(
        server,
        waypoint_positions,
        radius=0.015,
        colors=[marker_colors[name] for name in waypoint_names],
    )

    add_ellipsoid_obstacles(
        server,
        centers=[obstacle_center],
        radii=[np.full(3, 1.0 / obstacle_radius)],
    )

    # Ghost EE trajectory + animated trail
    ee_colors = compute_velocity_colors(np.asarray(results.trajectory.get("velocity")))
    add_ghost_trajectory(server, ee_pos, ee_colors, point_size=0.005)
    _, update_trail = add_animated_trail(server, ee_pos, ee_colors, point_size=0.008)

    # Animated EE position marker
    _, update_marker = add_position_marker(server, ee_pos, radius=0.015)

    # Animated arm links (line segments between consecutive keypoints)
    link_rgb = np.array(
        [
            [180, 180, 180],  # base -> shoulder
            [100, 180, 255],  # shoulder -> elbow
            [100, 255, 150],  # elbow -> wrist
            [255, 200, 100],  # wrist -> ee
        ],
        dtype=np.uint8,
    )
    link_colors = np.stack([link_rgb, link_rgb], axis=1)  # (4, 2, 3)

    init_points = np.stack(
        [np.stack([keypoints[0, k], keypoints[0, k + 1]]) for k in range(4)]
    ).astype(np.float32)

    arm_handle = server.scene.add_line_segments(
        "/arm_links",
        points=init_points,
        colors=link_colors,
        line_width=5.0,
    )

    # Joint coordinate frames
    joint_frame_handles = []
    for k in range(5):
        h = server.scene.add_frame(
            f"/joint_{joint_names[k]}",
            wxyz=joint_quats[0, k].astype(np.float32),
            position=keypoints[0, k].astype(np.float32),
            axes_length=0.06,
            axes_radius=0.002,
        )
        joint_frame_handles.append(h)

    def update_arm(frame_idx: int) -> None:
        pts = np.stack(
            [np.stack([keypoints[frame_idx, k], keypoints[frame_idx, k + 1]]) for k in range(4)]
        ).astype(np.float32)
        arm_handle.points = pts
        for k, h in enumerate(joint_frame_handles):
            h.position = keypoints[frame_idx, k].astype(np.float32)
            h.wxyz = joint_quats[frame_idx, k].astype(np.float32)

    # -- Manual-step handle for the renderer -----------------------------------
    traj_time = np.asarray(results.trajectory["time"], dtype=np.float64).flatten()
    handle = AnimatedServerHandle(
        server=server,
        traj_time=traj_time,
        update_callbacks=[update_trail, update_marker, update_arm],
    )

    # -- Camera: overview framing the full workspace ---------------------------
    # Frame on waypoint positions + EE path so the full motion is visible.
    all_points = np.vstack([ee_pos, np.array(waypoint_positions)])
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
