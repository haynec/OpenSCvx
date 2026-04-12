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

Tweak ``OUTPUT_PATH`` / ``WIDTH`` / ``HEIGHT`` / ``FPS`` below for different
output variants.
"""

import os
import sys

import numpy as np
import viser.transforms as vtf

# Add the project root so `examples.*` imports resolve.
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.animations._render import render_animation_to_video
from examples.drone.dr_vp_polytope import plotting_dict, problem
from examples.plotting_viser import create_animated_plotting_server

# --- Render settings ---------------------------------------------------------
OUTPUT_PATH = os.path.join(current_dir, "dr_vp_polytope.mp4")
WIDTH = 1080
HEIGHT = 1080
FPS = 60
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
CHASE_DISTANCE = 15.0  # camera sits this far past the drone along polytope->drone ray
VERTICAL_OFFSET = 2.0  # lift so the drone isn't a 1-pixel occlusion of the polytope


def _look_at_wxyz(pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) for a camera at ``pos`` looking at ``target``.

    Uses the OpenCV camera convention that viser expects: +X right, +Y down,
    +Z forward. See ``examples/animations/camera_control_notes.md``.
    """
    forward = target - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Gimbal lock: forward is (nearly) parallel to up. Pick an arbitrary
        # world axis that isn't, so the camera stays defined. The chosen axis
        # determines the "roll" of the camera at the singularity — not great
        # cinematically, but prevents a NaN crash.
        fallback = np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, fallback)
        right_norm = np.linalg.norm(right)
    right /= right_norm
    cam_down = np.cross(forward, right)  # = -world_up projected perp to forward
    R_world_cam = np.stack([right, cam_down, forward], axis=1)
    return vtf.SO3.from_matrix(R_world_cam).wxyz


def polytope_follow_pose(
    drone: np.ndarray,
    polytope_center: np.ndarray,
    *,
    chase_distance: float = CHASE_DISTANCE,
    vertical_offset: float = VERTICAL_OFFSET,
    up=(0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cam_pos, cam_wxyz, look_at)`` for a chase camera behind the drone.

    The camera sits on the ray from ``polytope_center`` through ``drone``,
    extended ``chase_distance`` units past the drone, then lifted along world
    up by ``vertical_offset`` so the drone silhouettes (but doesn't pixel-occlude)
    the target cluster. It always looks at ``polytope_center``.

    This is exactly the geometry the viewplanning constraint is enforcing:
    the drone's sensor boresight points along ``drone -> polytope_center``,
    so the chase camera naturally frames "what the drone is looking at" from
    over-the-shoulder.
    """
    drone = np.asarray(drone, dtype=np.float64)
    polytope_center = np.asarray(polytope_center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    ray = drone - polytope_center
    ray_norm = np.linalg.norm(ray)
    if ray_norm < 1e-6:
        # Degenerate: drone sitting exactly at the polytope center. Shouldn't
        # happen on a viewplanning trajectory, but don't divide by zero.
        cam_pos = drone + vertical_offset * up
    else:
        cam_pos = drone + chase_distance * (ray / ray_norm) + vertical_offset * up

    wxyz = _look_at_wxyz(cam_pos, polytope_center, up)
    return cam_pos, wxyz, polytope_center


if __name__ == "__main__":
    problem.settings.prp.dt = 1.0 / PROPAGATION_HZ
    problem.initialize()
    problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    # Center of the viewplanning polytope (mean of its vertices).
    polytope_center = np.asarray(results["init_poses"]).mean(axis=0)
    positions = np.asarray(results.trajectory["position"], dtype=np.float64)

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

    def camera_pose_fn(frame_idx: int):
        return polytope_follow_pose(positions[frame_idx], polytope_center)

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
