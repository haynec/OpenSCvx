"""LiDAR scan simulation over a DEM (Tomita & Ho 2025, Sec. IV.B.1).

Generates a sparse, noisy point cloud (PCD) by casting a grid of rays from a
sensor at a given range and off-nadir angle onto the DEM height field. Each ray
is intersected with the terrain by a coarse march + bisection refinement (the
DEM is a single-valued height field, so the first downward crossing is the
visible surface, which also yields occlusion behind rocks for free).

Measurement noise is additive Gaussian along the ray (range noise), with a
standard deviation scaled linearly with range: ``3 sigma = noise_3sig_at_ref``
(default 5 cm) at ``ref_range_m`` (default 500 m), per the paper's model.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .dem import DEM


def _normalize(v: jnp.ndarray) -> jnp.ndarray:
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True)


def _ray_terrain_t(
    dem: DEM,
    origin: jnp.ndarray,
    direction: jnp.ndarray,
    t_min: float,
    t_max: float,
    n_march: int = 256,
    n_bisect: int = 24,
):
    """Return ``(t_hit, hit_mask)`` for one ray against the DEM height field.

    ``f(t) = (origin + t*dir).z - terrain((origin + t*dir).xy)`` is positive
    above the surface and negative below it. We march to find the first bracket
    where ``f`` changes ``+ -> -`` then bisect within it.
    """

    def f(t):
        p = origin + t * direction
        return p[2] - dem.sample(p[:2])

    ts = jnp.linspace(t_min, t_max, n_march)
    fs = jax.vmap(f)(ts)

    above = fs > 0.0
    # First index where we go from above (+) to below/at (-) the surface.
    crossing = above[:-1] & (~above[1:])
    idx = jnp.argmax(crossing)
    hit = jnp.any(crossing)

    lo = ts[idx]
    hi = ts[idx + 1]

    def body(_, bounds):
        a, b = bounds
        m = 0.5 * (a + b)
        fm = f(m)
        a = jnp.where(fm > 0.0, m, a)
        b = jnp.where(fm > 0.0, b, m)
        return (a, b)

    lo, hi = jax.lax.fori_loop(0, n_bisect, body, (lo, hi))
    t_hit = 0.5 * (lo + hi)
    return t_hit, hit


def simulate_scan(
    dem: DEM,
    range_m: float = 500.0,
    angle_deg: float = 0.0,
    n_det: int = 128,
    footprint_m: float = 100.0,
    ref_range_m: float = 500.0,
    noise_3sig_at_ref: float = 0.05,
    seed: int = 0,
    sensor_xy: tuple[float, float] = (0.0, 0.0),
    return_full: bool = False,
):
    """Simulate a grid LiDAR scan and return a point cloud.

    The sensor is placed at slant ``range_m`` from the aim point (the DEM
    surface directly below ``sensor_xy``), tilted ``angle_deg`` off nadir toward
    +x. The detector is an ``n_det x n_det`` grid whose field of view nominally
    covers ``footprint_m`` on the ground at ``ref_range_m`` when nadir.

    Args:
        dem: Ground-truth DEM to scan.
        range_m: Slant range from sensor to the aim point (m).
        angle_deg: Off-nadir tilt of the boresight toward +x (deg).
        n_det: Detector side length in pixels (n_det**2 rays).
        footprint_m: Nominal ground footprint side at ``ref_range_m`` nadir (m).
        ref_range_m: Reference range for FOV and noise scaling (m).
        noise_3sig_at_ref: 3-sigma range noise at ``ref_range_m`` (m).
        seed: RNG seed for the noise draw.
        sensor_xy: Horizontal aim point on the DEM (m).
        return_full: If True also return per-ray hit mask and clean ranges.

    Returns:
        ``pcd`` of shape ``(M, 3)`` (hits only) if ``return_full`` is False,
        else ``(pcd_grid (n_det, n_det, 3), hit_mask (n_det, n_det))``.
    """
    aim_z = float(dem.sample(jnp.asarray(sensor_xy)))
    aim = jnp.array([sensor_xy[0], sensor_xy[1], aim_z], dtype=jnp.float64)

    ang = np.deg2rad(angle_deg)
    # Boresight points from the sensor toward the aim point. Tilt toward +x.
    boresight = jnp.array([np.sin(ang), 0.0, -np.cos(ang)], dtype=jnp.float64)
    sensor_pos = aim - range_m * boresight

    # Angular half-extent so the footprint spans footprint_m at ref_range_m.
    half_fov = np.arctan2(0.5 * footprint_m, ref_range_m)
    offs = jnp.linspace(-half_fov, half_fov, n_det)

    # Build an orthonormal detector basis around the boresight. The boresight
    # always lies in the x-z plane (tilt is about the y-axis), so the
    # cross-track axis is world-y; this stays well-defined even at nadir where
    # ``cross(boresight, world_z)`` would be degenerate.
    right = jnp.array([0.0, 1.0, 0.0])  # cross-track
    down = _normalize(jnp.cross(boresight, right))  # down-track

    aa, bb = jnp.meshgrid(offs, offs, indexing="xy")

    def ray_dir(a, b):
        d = boresight + jnp.tan(a) * right + jnp.tan(b) * down
        return _normalize(d)

    dirs = jax.vmap(jax.vmap(ray_dir))(aa, bb)  # (n_det, n_det, 3)

    t_max = 3.0 * range_m

    def cast(d):
        return _ray_terrain_t(dem, sensor_pos, d, t_min=0.0, t_max=t_max)

    t_hit, hit = jax.vmap(jax.vmap(cast))(dirs)

    # Range-scaled Gaussian noise along each ray (3 sigma = noise at ref range).
    sigma = (noise_3sig_at_ref / 3.0) * (t_hit / ref_range_m)
    key = jax.random.PRNGKey(seed)
    noise = sigma * jax.random.normal(key, t_hit.shape)
    t_noisy = t_hit + noise

    pts = sensor_pos + t_noisy[..., None] * dirs  # (n_det, n_det, 3)

    # Reject rays that missed or landed outside the DEM footprint.
    xmin, xmax, ymin, ymax = dem.extent
    in_bounds = (
        (pts[..., 0] >= xmin)
        & (pts[..., 0] <= xmax)
        & (pts[..., 1] >= ymin)
        & (pts[..., 1] <= ymax)
    )
    valid = hit & in_bounds

    if return_full:
        return pts, valid

    pts = np.asarray(pts).reshape(-1, 3)
    valid = np.asarray(valid).reshape(-1)
    return pts[valid]


def _detector_basis_from_boresight(boresight: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Orthonormal right/down axes spanning the detector plane about ``boresight``."""
    # Prefer world-y as the cross-track axis (matches ``simulate_scan`` at nadir).
    y_axis = jnp.array([0.0, 1.0, 0.0], dtype=boresight.dtype)
    x_axis = jnp.array([1.0, 0.0, 0.0], dtype=boresight.dtype)
    right = y_axis - boresight * jnp.dot(y_axis, boresight)
    use_x = jnp.linalg.norm(right) < 1e-6
    right = jnp.where(use_x, x_axis - boresight * jnp.dot(x_axis, boresight), right)
    right = _normalize(right)
    down = _normalize(jnp.cross(boresight, right))
    return right, down


def simulate_scan_body(
    dem: DEM,
    sensor_pos,
    R_body_to_world,
    n_det: int = 128,
    footprint_m: float = 100.0,
    ref_range_m: float = 500.0,
    noise_3sig_at_ref: float = 0.05,
    seed: int = 0,
    boresight_body=(0.0, 0.0, -1.0),
    t_max: Optional[float] = None,
    return_full: bool = False,
):
    """Simulate a body-fixed LiDAR scan from an arbitrary vehicle pose.

    Same detector FOV / range-noise model as :func:`simulate_scan`, but the
    sensor sits at ``sensor_pos`` with orientation ``R_body_to_world`` and a
    boresight fixed in the body frame. The default ``boresight_body=(0,0,-1)``
    is nadir when the vehicle is upright (body +z along world +z / altitude).

    Args:
        dem: Ground-truth DEM to scan.
        sensor_pos: Sensor origin in world meters, shape ``(3,)``.
        R_body_to_world: Body→world DCM, shape ``(3, 3)``.
        n_det: Detector side length in pixels (n_det**2 rays).
        footprint_m: Nominal ground footprint side at ``ref_range_m`` nadir (m).
        ref_range_m: Reference range for FOV and noise scaling (m).
        noise_3sig_at_ref: 3-sigma range noise at ``ref_range_m`` (m).
        seed: RNG seed for the noise draw.
        boresight_body: Unit-ish look direction in the body frame.
        t_max: Max ray length (m); default ``3 * ref_range_m``.
        return_full: If True also return per-ray hit mask on the full grid.

    Returns:
        ``pcd`` of shape ``(M, 3)`` (hits only) if ``return_full`` is False,
        else ``(pcd_grid (n_det, n_det, 3), hit_mask (n_det, n_det))``.
    """
    sensor_pos = jnp.asarray(sensor_pos, dtype=jnp.float64).reshape(3)
    R_bw = jnp.asarray(R_body_to_world, dtype=jnp.float64).reshape(3, 3)
    b_body = _normalize(jnp.asarray(boresight_body, dtype=jnp.float64).reshape(3))
    boresight = _normalize(R_bw @ b_body)

    # Body-fixed detector axes: project world-preferred basis into the image
    # plane, then fall back so the frame stays well-defined at nadir.
    right, down = _detector_basis_from_boresight(boresight)

    # Keep the detector rigidly attached to the body when possible: if the
    # body provides a right axis nearly orthogonal to the boresight, use it.
    body_y_w = R_bw[:, 1]
    body_y_proj = body_y_w - boresight * jnp.dot(body_y_w, boresight)
    use_body = jnp.linalg.norm(body_y_proj) > 0.2
    right = jnp.where(use_body, _normalize(body_y_proj), right)
    down = jnp.where(use_body, _normalize(jnp.cross(boresight, right)), down)

    half_fov = np.arctan2(0.5 * footprint_m, ref_range_m)
    offs = jnp.linspace(-half_fov, half_fov, n_det)
    aa, bb = jnp.meshgrid(offs, offs, indexing="xy")

    def ray_dir(a, b):
        d = boresight + jnp.tan(a) * right + jnp.tan(b) * down
        return _normalize(d)

    dirs = jax.vmap(jax.vmap(ray_dir))(aa, bb)

    if t_max is None:
        t_max = 3.0 * ref_range_m

    def cast(d):
        return _ray_terrain_t(dem, sensor_pos, d, t_min=0.0, t_max=float(t_max))

    t_hit, hit = jax.vmap(jax.vmap(cast))(dirs)

    sigma = (noise_3sig_at_ref / 3.0) * (t_hit / ref_range_m)
    key = jax.random.PRNGKey(seed)
    noise = sigma * jax.random.normal(key, t_hit.shape)
    t_noisy = t_hit + noise

    pts = sensor_pos + t_noisy[..., None] * dirs

    xmin, xmax, ymin, ymax = dem.extent
    in_bounds = (
        (pts[..., 0] >= xmin)
        & (pts[..., 0] <= xmax)
        & (pts[..., 1] >= ymin)
        & (pts[..., 1] <= ymax)
    )
    # Also drop rays that look away from the terrain (no downward component).
    looking_down = dirs[..., 2] < -1e-3
    valid = hit & in_bounds & looking_down

    if return_full:
        return pts, valid

    pts = np.asarray(pts).reshape(-1, 3)
    valid = np.asarray(valid).reshape(-1)
    return pts[valid]
