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
import viser.transforms as vtf

# Add the project root so `examples.*` imports resolve.
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.animations._render import render_animation_to_video
from examples.drone.dr_vp_polytope import plotting_dict, problem
from examples.plotting_viser import create_animated_plotting_server

# Camera mode: "chase" | "onboard" | "overview"
CAMERA_MODE = "chase"

# --- Render settings ---------------------------------------------------------
OUTPUT_PATH = os.path.join(current_dir, f"dr_vp_polytope_{CAMERA_MODE}.mp4")
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
# Chase mode
CHASE_DISTANCE = 15.0  # camera sits this far past the drone along polytope->drone ray
VERTICAL_OFFSET = 2.0  # lift so the drone isn't a 1-pixel occlusion of the polytope

# Onboard mode
ONBOARD_FORWARD_OFFSET = 0.0  # shift camera forward along boresight to clear body-frame axes

# Overview mode — spherical coordinates from the trajectory centroid.
# Azimuth: angle in the XY plane from +X, CCW (radians).
# Elevation: angle above the horizon (radians). pi/2 = straight down.
OVERVIEW_AZIMUTH = np.radians(135.0)
OVERVIEW_ELEVATION = np.radians(25.0)
OVERVIEW_RADIUS_MARGIN = 0.75  # multiplier on the auto-computed radius
OVERVIEW_FOV_DEG = 60.0


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


def onboard_pose(
    drone: np.ndarray,
    attitude_wxyz: np.ndarray,
    R_sb: np.ndarray,
    *,
    forward_offset: float = ONBOARD_FORWARD_OFFSET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cam_pos, cam_wxyz, look_at)`` for a sensor-mounted FPV camera.

    Places the camera at the drone's position (shifted slightly forward along
    the sensor boresight to avoid being occluded by the body-frame coordinate
    axes) with orientation matching the sensor frame — i.e. looking along the
    sensor boresight (+Z in sensor frame, mapped through R_sb and the body
    attitude to world coordinates). This is exactly what the viewplanning
    constraint keeps pointed at the target polytope.
    """
    # R_body_to_world from the attitude quaternion (w, x, y, z)
    w, x, y, z = attitude_wxyz
    R_bw = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

    # Sensor-to-world: columns are the sensor axes in world coords.
    # R_sb is body-to-sensor, so R_sb.T is sensor-to-body.
    R_sensor_to_world = R_bw @ R_sb.T

    # Viser uses OpenCV convention (+X right, +Y DOWN, +Z forward).
    # The sensor frame has +Y UP, so we rotate 180° around the boresight (Z)
    # to flip both X and Y, converting to the convention viser expects.
    R_opencv_from_sensor = np.diag([-1.0, -1.0, 1.0])
    R_cam_to_world = R_sensor_to_world @ R_opencv_from_sensor

    wxyz = vtf.SO3.from_matrix(R_cam_to_world).wxyz
    # Boresight is sensor +Z expressed in world frame (unchanged by the flip).
    boresight_world = R_sensor_to_world[:, 2]
    cam_pos = drone + forward_offset * boresight_world
    look_at = cam_pos + 10.0 * boresight_world
    return cam_pos, wxyz, look_at


def overview_pose(
    positions: np.ndarray,
    *,
    azimuth: float = OVERVIEW_AZIMUTH,
    elevation: float = OVERVIEW_ELEVATION,
    radius_margin: float = OVERVIEW_RADIUS_MARGIN,
    fov_deg: float = OVERVIEW_FOV_DEG,
    up=(0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a static ``(cam_pos, cam_wxyz, look_at)`` that frames the full track.

    The camera is placed at a point on a sphere around the trajectory centroid,
    parameterized by ``azimuth`` (angle in XY from +X, CCW) and ``elevation``
    (angle above the horizon). The radius is auto-computed so the full
    trajectory fits within ``fov_deg``, then scaled by ``radius_margin``.
    """
    up = np.asarray(up, dtype=np.float64)
    center = positions.mean(axis=0)
    # Max distance from centroid to any trajectory point (3D).
    max_extent = np.max(np.linalg.norm(positions - center, axis=1))
    # Radius so the full extent subtends half the FOV.
    half_fov_rad = np.radians(fov_deg / 2.0)
    radius = max_extent / np.sin(half_fov_rad) * radius_margin

    # Spherical -> Cartesian offset from centroid.
    cos_el = np.cos(elevation)
    cam_pos = center + radius * np.array([
        cos_el * np.cos(azimuth),
        cos_el * np.sin(azimuth),
        np.sin(elevation),
    ])
    look_at = center.copy()
    wxyz = _look_at_wxyz(cam_pos, look_at, up)
    return cam_pos, wxyz, look_at


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
            return polytope_follow_pose(positions[frame_idx], polytope_center)
    elif CAMERA_MODE == "onboard":
        render_fov = sensor_fov_deg + 5
        def camera_pose_fn(frame_idx: int):
            return onboard_pose(positions[frame_idx], attitude[frame_idx], R_sb)
    elif CAMERA_MODE == "overview":
        static_pose = overview_pose(positions)
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
