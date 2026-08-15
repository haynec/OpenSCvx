"""Cinematic offline render of the vmap obstacle-avoidance example.

The trajectory optimization problem itself lives in
``examples/double_integrator/obstacle_avoidance_vmap.py``; this file imports that
``problem`` (plus obstacle metadata), solves it, and drives a viser scene
frame-by-frame while piping raw RGB into ffmpeg to produce an mp4.

Run it with::

    python examples/animations/obstacle_avoidance_vmap.py

The script prints a viser URL and waits. Open the URL in a browser — as soon
as the client connects, the render begins. Requires ``ffmpeg`` on ``PATH``;
``openscvx`` does not depend on it.

Two camera modes are available — set ``CAMERA_MODE`` below:

- ``"chase"``    — over-the-shoulder behind the double-integrator body, looking
  toward the goal so you watch it thread the obstacle field.
- ``"overview"`` — static elevated camera framing the full trajectory and every
  obstacle sphere.

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
from examples.double_integrator.obstacle_avoidance_vmap import (
    n_obstacles,
    obstacle_centers,
    obstacle_radii,
    position,
    problem,
)
from examples.plotting_viser import create_animated_plotting_server

# Camera mode: "chase" | "overview"
CAMERA_MODE = "overview"

# --- Render settings ---------------------------------------------------------
OUTPUT_PATH = os.path.join(current_dir, "mp4", f"obstacle_avoidance_vmap_{CAMERA_MODE}.mp4")
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
CHASE_DISTANCE = 8.0  # camera sits this far past the drone along goal->drone ray
VERTICAL_OFFSET = 2.0  # lift so the body isn't a 1-pixel occlusion of the goal

# Overview mode — spherical coordinates from the scene centroid.
# Azimuth: angle in the XY plane from +X, CCW (radians).
# Elevation: angle above the horizon (radians).
OVERVIEW_AZIMUTH = np.radians(-60.0)
OVERVIEW_ELEVATION = np.radians(25.0)
OVERVIEW_RADIUS_MARGIN = 0.9
OVERVIEW_FOV_DEG = 60.0


if __name__ == "__main__":
    problem.settings.prp.dt = 1.0 / PROPAGATION_HZ
    problem.initialize()
    problem.solve()
    results = problem.post_process()

    # Re-attach obstacle metadata — the base example sets this inside its own
    # ``__main__`` block, so we have to repeat it here.
    results.update(
        {
            "obstacles_centers": [c for c in obstacle_centers],
            "obstacles_radii": [[1 / r, 1 / r, 1 / r] for r in obstacle_radii],
            "obstacles_axes": [np.eye(3) for _ in range(n_obstacles)],
        }
    )

    positions = np.asarray(results.trajectory["position"], dtype=np.float64)
    goal = np.asarray(position.final, dtype=np.float64)

    # Build the scene in manual-step mode — no GUI playback loop, no wall-clock
    # thread; we drive every frame ourselves from the render loop. No viewcone /
    # attitude because this is a point-mass double integrator.
    handle = create_animated_plotting_server(
        results,
        thrust_key="force",
        controls="manual",
        show_grid=False,
        trail_point_size=0.12,
    )

    # Select camera pose function based on mode.
    if CAMERA_MODE == "chase":

        def camera_pose_fn(frame_idx: int):
            return chase_pose(
                positions[frame_idx],
                goal,
                chase_distance=CHASE_DISTANCE,
                vertical_offset=VERTICAL_OFFSET,
            )

    elif CAMERA_MODE == "overview":
        # Frame the trajectory AND the obstacle field. Push each obstacle
        # center out by its radius along each world axis so overview_pose's
        # centroid/extent math accounts for obstacle volumes, not just centers.
        axis_offsets = np.vstack([np.eye(3), -np.eye(3)])  # (6, 3)
        obstacle_shell = (
            obstacle_centers[:, None, :] + obstacle_radii[:, None, None] * axis_offsets[None, :, :]
        ).reshape(-1, 3)
        all_points = np.vstack([positions, obstacle_shell])

        static_pose = overview_pose(
            all_points,
            azimuth=OVERVIEW_AZIMUTH,
            elevation=OVERVIEW_ELEVATION,
            radius_margin=OVERVIEW_RADIUS_MARGIN,
            fov_deg=OVERVIEW_FOV_DEG,
        )

        def camera_pose_fn(frame_idx: int):
            return static_pose

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
