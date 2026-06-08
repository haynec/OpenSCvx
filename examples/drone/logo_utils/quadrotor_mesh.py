"""Procedural low-poly quadrotor mesh for Viser visualization.

Body frame: +x boresight (forward), +z up. Motor discs lie in the x–y plane so +z is
perpendicular to every motor axis. Two diagonal tubes join opposite motor pairs.
"""

from __future__ import annotations

import numpy as np


def _merge_parts(
    parts: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    verts_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    offset = 0
    for verts, faces in parts:
        verts_list.append(verts)
        faces_list.append(faces + offset)
        offset += len(verts)
    return np.vstack(verts_list), np.vstack(faces_list)


def _rotation_align_z_to(direction: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping local +z to ``direction`` (unit vector)."""
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    d = d / norm
    if np.allclose(d, z):
        return np.eye(3, dtype=np.float64)
    if np.allclose(d, -z):
        return np.diag([1.0, -1.0, -1.0]).astype(np.float64)

    v = np.cross(z, d)
    s = np.linalg.norm(v)
    c = float(np.dot(z, d))
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _flat_motor_disk(
    center: tuple[float, float, float],
    radius: float,
    thickness: float,
    n_sides: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Short cylinder (flat disc) with axis parallel to +z."""
    cx, cy, cz = center
    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False, dtype=np.float64)
    ring_xy = np.stack([np.cos(angles) * radius, np.sin(angles) * radius], axis=1)
    z_bot = cz - 0.5 * thickness
    z_top = cz + 0.5 * thickness

    bottom = np.column_stack([cx + ring_xy[:, 0], cy + ring_xy[:, 1], np.full(n_sides, z_bot)])
    top = np.column_stack([cx + ring_xy[:, 0], cy + ring_xy[:, 1], np.full(n_sides, z_top)])
    center_bot = np.array([[cx, cy, z_bot]], dtype=np.float64)
    center_top = np.array([[cx, cy, z_top]], dtype=np.float64)
    vertices = np.vstack([bottom, top, center_bot, center_top]).astype(np.float32)

    faces: list[list[int]] = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        bi, bj = i, j
        ti, tj = i + n_sides, j + n_sides
        faces.append([bi, bj, tj])
        faces.append([bi, tj, ti])
        faces.append([2 * n_sides, bj, bi])
        faces.append([2 * n_sides + 1, ti, tj])

    return vertices, np.asarray(faces, dtype=np.uint32)


def _tube(
    p0: tuple[float, float, float] | np.ndarray,
    p1: tuple[float, float, float] | np.ndarray,
    radius: float,
    n_sides: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Circular tube (cylinder) between ``p0`` and ``p1``."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-9:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)

    R = _rotation_align_z_to(axis)
    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False, dtype=np.float64)
    ring = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros(n_sides)],
        axis=1,
    )
    ring_end = ring.copy()
    ring_end[:, 2] = length

    local = np.vstack([ring, ring_end]).astype(np.float64)
    vertices = (local @ R.T + p0).astype(np.float32)

    faces: list[list[int]] = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        a, b = i, j
        c, d = i + n_sides, j + n_sides
        faces.append([a, b, d])
        faces.append([a, d, c])

    return vertices, np.asarray(faces, dtype=np.uint32)


def make_quadrotor_mesh(
    arm_length: float = 0.30,
    motor_radius: float = 0.07,
    motor_thickness: float = 0.018,
    tube_radius: float = 0.014,
    hub_radius: float = 0.035,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, faces)`` for an X-quadrotor with disc motors and two tubes.

    Motors sit in the x–y plane (normal +z). Two tubes run along the diagonals between
    opposite motors. A small hub disc and +x nose stub mark the body frame.

    Args:
        arm_length: Hub-center to motor-center distance (meters).
        motor_radius: Radius of each flat motor disc.
        motor_thickness: Motor disc height along z.
        tube_radius: Radius of the diagonal connecting tubes.
        hub_radius: Central hub disc radius.
        scale: Uniform scale on all geometry.
    """
    s = float(scale)
    arm = float(arm_length) * s
    d = arm / np.sqrt(2.0)

    motor_positions = [
        (d, d, 0.0),
        (d, -d, 0.0),
        (-d, d, 0.0),
        (-d, -d, 0.0),
    ]

    parts: list[tuple[np.ndarray, np.ndarray]] = []

    # Central hub (flat disc, z axis up).
    parts.append(
        _flat_motor_disk(
            center=(0.0, 0.0, 0.0),
            radius=hub_radius * s,
            thickness=motor_thickness * s * 1.2,
        )
    )

    # Two diagonal tubes (X frame in the motor plane).
    parts.append(_tube(motor_positions[0], motor_positions[3], tube_radius * s))
    parts.append(_tube(motor_positions[1], motor_positions[2], tube_radius * s))

    # Four flat motor discs (axis || z).
    for mx, my, mz in motor_positions:
        parts.append(
            _flat_motor_disk(
                center=(mx, my, mz),
                radius=motor_radius * s,
                thickness=motor_thickness * s,
            )
        )

    # Short +x nose stub (boresight direction).
    parts.append(
        _tube(
            (0.04 * s, 0.0, 0.0),
            (0.14 * s, 0.0, 0.0),
            tube_radius * s * 0.85,
            n_sides=12,
        )
    )

    return _merge_parts(parts)
