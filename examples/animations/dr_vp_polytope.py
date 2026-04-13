"""Cinematic offline render of the drone-racing-with-polytope-viewplanning example.

The trajectory optimization problem itself lives in
``examples/drone/dr_vp_polytope.py``; this file imports that ``problem`` (and
the associated ``plotting_dict``), solves it, and drives a viser scene
frame-by-frame while piping raw RGB into ffmpeg to produce an mp4 suitable for
landing-page / presentation captures.

Run it with::

    python examples/animations/dr_vp_polytope.py

The script prints a viser URL and waits. Open the URL in a browser — as soon
as the client connects, the render begins. Requires ``ffmpeg`` on ``PATH``;
``openscvx`` does not depend on it.

Three camera modes are available — set ``CAMERA_MODE`` below:

- ``"chase"``   — over-the-shoulder behind the drone, looking at the polytope.
- ``"onboard"`` — rigidly mounted to the drone's sensor frame (FPV view).
- ``"overview"`` — static elevated camera framing the full track.

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

from examples.animations._camera import chase_pose, onboard_pose, overview_pose
from examples.animations._render import render_animation_to_video
from examples.drone.dr_vp_polytope import plotting_dict, problem
from examples.plotting_viser import create_animated_plotting_server

# Camera mode: "chase" | "onboard" | "overview"
CAMERA_MODE = "chase"

# --- Render settings ---------------------------------------------------------
OUTPUT_PATH = os.path.join(current_dir, f"dr_vp_polytope_{CAMERA_MODE}.mp4")
WIDTH = 1080
HEIGHT = 1080
FPS = 30
CRF = 16  # lower = crisper; 16 is visually near-lossless

# Oversampling factor for the propagation: how many trajectory samples we
# propagate per rendered video frame. STRIDE=1 means one sample per frame (the
# trail polyline looks chunky when the drone is fast). STRIDE=4 means the
# propagation runs at 4x the video rate, so the trail is drawn from 4x denser
# samples — smoother curves at speed — while `render_animation_to_video` strides
# through every 4th sample to keep the video at realtime FPS.
STRIDE = 6
PROPAGATION_HZ = FPS * STRIDE

# --- Camera settings ---------------------------------------------------------
# Chase mode
CHASE_DISTANCE = 15.0  # camera sits this far past the drone along polytope->drone ray
VERTICAL_OFFSET = 2.0  # lift so the drone isn't a 1-pixel occlusion of the polytope

# Onboard mode
ONBOARD_FORWARD_OFFSET = 0.0  # shift camera forward along boresight to clear body-frame axes
ONBOARD_FOV_PADDING = 5.0  # degrees added to sensor FOV so the viewcone ring is visible

# Overview mode — spherical coordinates from the trajectory centroid.
# Azimuth: angle in the XY plane from +X, CCW (radians).
# Elevation: angle above the horizon (radians). pi/2 = straight down.
OVERVIEW_AZIMUTH = np.radians(135.0)
OVERVIEW_ELEVATION = np.radians(25.0)
OVERVIEW_RADIUS_MARGIN = 0.75  # multiplier on the auto-computed radius
OVERVIEW_FOV_DEG = 60.0


if __name__ == "__main__":
    problem.settings.prp.dt = 1.0 / PROPAGATION_HZ
    problem.initialize()
    problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    # Center of the viewplanning polytope (mean of its vertices).
    polytope_center = np.asarray(results["init_poses"]).mean(axis=0)
    positions = np.asarray(results.trajectory["position"], dtype=np.float64)
    attitude = np.asarray(results.trajectory["attitude"], dtype=np.float64)
    R_sb = np.asarray(results["R_sb"], dtype=np.float64)

    # Build the scene in manual-step mode — no GUI playback loop, no wall-clock
    # thread; we'll drive every frame ourselves from the render loop.
    handle = create_animated_plotting_server(
        results,
        thrust_key="thrust_force",
        viewcone_scale=10.0,
        show_control_plot="thrust_force",
        show_control_norm_plot="thrust_force",
        controls="manual",
        show_grid=False,
        trail_point_size=0.075,
        viewcone_ring_only=True,
    )

    # Sensor FOV derived from the viewplanning cone half-angle parameter.
    # alpha_x defines the half-angle as pi/alpha_x radians; full vertical FOV
    # is 2 * pi/alpha_x converted to degrees.
    alpha_x = results.get("alpha_x")
    sensor_fov_deg = float(np.degrees(2 * np.pi / alpha_x)) if alpha_x is not None else None

    # Select camera pose function based on mode.
    render_fov: float | None = None
    if CAMERA_MODE == "chase":

        def camera_pose_fn(frame_idx: int):
            return chase_pose(
                positions[frame_idx],
                polytope_center,
                chase_distance=CHASE_DISTANCE,
                vertical_offset=VERTICAL_OFFSET,
            )
    elif CAMERA_MODE == "onboard":
        render_fov = sensor_fov_deg + ONBOARD_FOV_PADDING

        def camera_pose_fn(frame_idx: int):
            return onboard_pose(
                positions[frame_idx],
                attitude[frame_idx],
                R_sb,
                forward_offset=ONBOARD_FORWARD_OFFSET,
            )
    elif CAMERA_MODE == "overview":
        static_pose = overview_pose(
            positions,
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
        fov_deg=render_fov,
    )
