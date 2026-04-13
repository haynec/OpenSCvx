"""Cinematic offline render of the ACL-logo-tracing drone example.

The trajectory optimization problem itself lives in
``examples/drone/logo.py``; this file imports that ``problem`` (and
the associated ``plotting_dict``), solves it, and drives a viser scene
frame-by-frame while piping raw RGB into ffmpeg to produce an mp4.

Run it with::

    python examples/animations/logo.py

The script prints a viser URL and waits. Open the URL in a browser — as soon
as the client connects, the render begins. Requires ``ffmpeg`` on ``PATH``;
``openscvx`` does not depend on it.

Two camera modes are available — set ``CAMERA_MODE`` below:

- ``"overview"`` — static elevated camera framing the full trajectory so we can
  watch the drone draw out the logo.
- ``"chase"``   — over-the-shoulder behind the drone, focused on the current
  SVG path target so we can see the motion up close.

Tweak ``OUTPUT_PATH`` / ``WIDTH`` / ``HEIGHT`` / ``FPS`` below for different
output variants.
"""

import os
import sys

import numpy as np

# Add the project root so `examples.*` imports resolve.
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.animations._camera import chase_pose, overview_pose
from examples.animations._render import render_animation_to_video
from examples.drone.logo import get_kp_pose, plotting_dict, problem, total_time
from examples.plotting_viser import create_animated_plotting_server

# Camera mode: "chase" | "overview"
CAMERA_MODE = "overview"

# --- Render settings ---------------------------------------------------------
OUTPUT_PATH = os.path.join(current_dir, f"logo_{CAMERA_MODE}.mp4")
WIDTH = 1080
HEIGHT = 1080
FPS = 60
CRF = 16  # lower = crisper; 16 is visually near-lossless

# Oversampling: propagate at FPS*STRIDE Hz for smooth trails, then stride
# through every STRIDE-th sample so the video plays at realtime.
STRIDE = 4
PROPAGATION_HZ = FPS * STRIDE

# --- Camera settings ---------------------------------------------------------
# Chase mode
CHASE_DISTANCE = 8.0  # camera distance past the drone along target->drone ray
VERTICAL_OFFSET = 2.0  # lift above the drone

# Overview mode — spherical coordinates from the trajectory centroid.
OVERVIEW_AZIMUTH = np.radians(30.0)
OVERVIEW_ELEVATION = np.radians(30.0)
OVERVIEW_RADIUS_MARGIN = 0.85
OVERVIEW_FOV_DEG = 55.0


if __name__ == "__main__":
    problem.settings.prp.dt = 1.0 / PROPAGATION_HZ
    problem.initialize()
    problem.solve()
    results = problem.post_process()
    results.update_plotting_data(
        **{
            **plotting_dict,
            # Hide elements that clutter the cinematic render — keep only the
            # boresight intersection trail (the red "drawing" on the logo plane).
            "moving_subject": False,
            "relative_vector": False,
        }
    )

    positions = np.asarray(results.trajectory["position"], dtype=np.float64)
    traj_time = np.asarray(results.trajectory["time"], dtype=np.float64).flatten()
    n_frames = len(positions)

    # Centroid of the SVG path target trajectory — the chase camera focuses here
    # so the framing stays stable instead of jerking with the moving target.
    target_positions = np.array(
        [np.asarray(get_kp_pose(float(traj_time[i]) / total_time)) for i in range(n_frames)],
        dtype=np.float64,
    )
    target_centroid = target_positions.mean(axis=0)

    # Define the logo drawing plane so the boresight-plane intersection works.
    # We intentionally skip traced_path_on_plane (static cyan path + green dot)
    # and the other diagnostic overlays to keep the render clean.
    p0 = np.asarray(get_kp_pose(0.0))
    p1 = np.asarray(get_kp_pose(0.33))
    p2 = np.asarray(get_kp_pose(0.67))
    v1, v2 = p1 - p0, p2 - p0
    plane_normal = np.cross(v1, v2)
    nrm = np.linalg.norm(plane_normal)
    plane_normal = plane_normal / nrm if nrm > 1e-10 else np.array([1.0, 0.0, 0.0])
    results["logo_plane_point"] = p0
    results["logo_plane_normal"] = plane_normal

    # Build the scene in manual-step mode.
    handle = create_animated_plotting_server(
        results,
        thrust_key="thrust_force",
        controls="manual",
        show_grid=False,
        trail_point_size=0.1,
        target_radius=0.3,
    )

    # Select camera pose function based on mode.
    if CAMERA_MODE == "overview":
        static_pose = overview_pose(
            target_positions,
            azimuth=OVERVIEW_AZIMUTH,
            elevation=OVERVIEW_ELEVATION,
            radius_margin=OVERVIEW_RADIUS_MARGIN,
            fov_deg=OVERVIEW_FOV_DEG,
        )

        def camera_pose_fn(frame_idx: int):
            return static_pose

    elif CAMERA_MODE == "chase":

        def camera_pose_fn(frame_idx: int):
            return chase_pose(
                positions[frame_idx],
                target_centroid,
                chase_distance=CHASE_DISTANCE,
                vertical_offset=VERTICAL_OFFSET,
            )

    else:
        raise ValueError(f"Unknown CAMERA_MODE: {CAMERA_MODE!r}")

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
