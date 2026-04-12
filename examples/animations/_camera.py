"""Reusable camera pose helpers for viser animation rendering.

All functions return ``(cam_position, cam_wxyz, look_at)`` tuples suitable for
passing to ``render_animation_to_video``'s ``camera_pose_fn`` argument. They
depend only on numpy and ``viser.transforms`` — no openscvx imports.
"""

from __future__ import annotations

import numpy as np
import viser.transforms as vtf


def look_at_wxyz(
    pos: np.ndarray, target: np.ndarray, up: np.ndarray
) -> np.ndarray:
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


def chase_pose(
    subject: np.ndarray,
    focus: np.ndarray,
    *,
    chase_distance: float = 15.0,
    vertical_offset: float = 2.0,
    up=(0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cam_pos, cam_wxyz, look_at)`` for a chase camera behind ``subject``.

    The camera sits on the ray from ``focus`` through ``subject``, extended
    ``chase_distance`` units past the subject, then lifted along world up by
    ``vertical_offset``. It always looks at ``focus``.
    """
    subject = np.asarray(subject, dtype=np.float64)
    focus = np.asarray(focus, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    ray = subject - focus
    ray_norm = np.linalg.norm(ray)
    if ray_norm < 1e-6:
        cam_pos = subject + vertical_offset * up
    else:
        cam_pos = subject + chase_distance * (ray / ray_norm) + vertical_offset * up

    wxyz = look_at_wxyz(cam_pos, focus, up)
    return cam_pos, wxyz, focus


def onboard_pose(
    position: np.ndarray,
    attitude_wxyz: np.ndarray,
    R_sb: np.ndarray,
    *,
    forward_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cam_pos, cam_wxyz, look_at)`` for a sensor-mounted FPV camera.

    Places the camera at ``position`` (shifted slightly forward along the sensor
    boresight by ``forward_offset``) with orientation matching the sensor frame —
    i.e. looking along the sensor boresight (+Z in sensor frame, mapped through
    ``R_sb`` and the body attitude to world coordinates).
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
    # The sensor frame has +Y UP, so we rotate 180deg around the boresight (Z)
    # to flip both X and Y, converting to the convention viser expects.
    R_opencv_from_sensor = np.diag([-1.0, -1.0, 1.0])
    R_cam_to_world = R_sensor_to_world @ R_opencv_from_sensor

    wxyz = vtf.SO3.from_matrix(R_cam_to_world).wxyz
    # Boresight is sensor +Z expressed in world frame (unchanged by the flip).
    boresight_world = R_sensor_to_world[:, 2]
    cam_pos = position + forward_offset * boresight_world
    look_at = cam_pos + 10.0 * boresight_world
    return cam_pos, wxyz, look_at


def overview_pose(
    positions: np.ndarray,
    *,
    azimuth: float = np.radians(135.0),
    elevation: float = np.radians(25.0),
    radius_margin: float = 0.75,
    fov_deg: float = 60.0,
    up=(0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a static ``(cam_pos, cam_wxyz, look_at)`` that frames all ``positions``.

    The camera is placed on a sphere around the centroid of ``positions``,
    parameterized by ``azimuth`` (angle in XY from +X, CCW) and ``elevation``
    (angle above the horizon). The radius is auto-computed so the full extent
    fits within ``fov_deg``, then scaled by ``radius_margin``.
    """
    up = np.asarray(up, dtype=np.float64)
    center = positions.mean(axis=0)
    max_extent = np.max(np.linalg.norm(positions - center, axis=1))
    half_fov_rad = np.radians(fov_deg / 2.0)
    radius = max_extent / np.sin(half_fov_rad) * radius_margin

    cos_el = np.cos(elevation)
    cam_pos = center + radius * np.array([
        cos_el * np.cos(azimuth),
        cos_el * np.sin(azimuth),
        np.sin(elevation),
    ])
    look_at = center.copy()
    wxyz = look_at_wxyz(cam_pos, look_at, up)
    return cam_pos, wxyz, look_at
