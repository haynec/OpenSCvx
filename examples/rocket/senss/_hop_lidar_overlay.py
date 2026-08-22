"""Body-fixed nadir LiDAR overlay for SENSS hop viser demos (post-process only).

Builds a world-frame DEM matching the hop scripts' visual terrain patch, then
casts body-fixed nadir LiDAR scans (see ``halo.lidar.simulate_scan_body``) along
the propagated trajectory and draws the point cloud as the animation plays.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import viser

from examples.rocket.senss._halo.dem import DEM
from examples.rocket.senss._halo.lidar import simulate_scan_body

# Visualization-only LiDAR defaults (do not affect the SCP problem).
LIDAR_N_DET: int = 48
LIDAR_FOOTPRINT_M: float = 100.0
LIDAR_REF_RANGE_M: float = 500.0
LIDAR_NOISE_3SIG_M: float = 0.05
LIDAR_DEM_GRID: int = 512
LIDAR_MAX_KEYFRAMES: int = 160
LIDAR_POINT_SIZE: float = 0.12
LIDAR_CMAP: str = "turbo"


def _R_bw_from_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Body→world DCM from a viser-order quaternion ``[w, x, y, z]``."""
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def build_terrain_dem(
    dem_norm: np.ndarray,
    *,
    origin_m: tuple[float, float, float],
    scale_xyz: tuple[float, float, float],
    yaw_deg: float,
    mirror_xy: tuple[bool, bool],
    terrain_half_extent_m: float,
    base_relief_m: float,
    dem_center_norm: float,
    grid_n: int = LIDAR_DEM_GRID,
) -> DEM:
    """Resample the visual terrain heightfield onto a world-aligned ``DEM``.

    Matches the hop scripts' ``_make_terrain_vertices`` geometry (origin, scale,
    yaw about +z, optional x/y mirrors) so LiDAR hits land on the same map.
    """
    ox, oy, oz = (float(v) for v in origin_m)
    sx, sy, sz = (float(v) for v in scale_xyz)
    mx = -1.0 if mirror_xy[0] else 1.0
    my = -1.0 if mirror_xy[1] else 1.0
    half_x = terrain_half_extent_m * sx
    half_y = terrain_half_extent_m * sy
    psi = np.radians(float(yaw_deg))
    cos_p, sin_p = np.cos(psi), np.sin(psi)

    # World AABB of the (possibly rotated) patch.
    corners = np.array(
        [[-half_x, -half_y], [half_x, -half_y], [half_x, half_y], [-half_x, half_y]],
        dtype=np.float64,
    )
    corners[:, 0] *= mx
    corners[:, 1] *= my
    world_xy = np.stack(
        [
            cos_p * corners[:, 0] - sin_p * corners[:, 1] + ox,
            sin_p * corners[:, 0] + cos_p * corners[:, 1] + oy,
        ],
        axis=-1,
    )
    xmin, ymin = world_xy.min(axis=0)
    xmax, ymax = world_xy.max(axis=0)
    # Slight pad so border rays stay in-bounds after numerical noise.
    pad = 0.5 * max((xmax - xmin), (ymax - ymin)) / max(grid_n - 1, 1)
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    res = max(xmax - xmin, ymax - ymin) / max(grid_n - 1, 1)
    xs = xmin + np.arange(grid_n) * res
    ys = ymin + np.arange(grid_n) * res
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    dx = XX - ox
    dy = YY - oy
    x_loc = cos_p * dx + sin_p * dy
    y_loc = -sin_p * dx + cos_p * dy
    x_pre = x_loc / mx
    y_pre = y_loc / my

    # Map local meters → continuous pixel coords in the source heightmap.
    src = np.asarray(dem_norm, dtype=np.float64)
    ny, nx = src.shape
    u = (x_pre + half_x) / max(2.0 * half_x, 1e-9) * (nx - 1)
    v = (y_pre + half_y) / max(2.0 * half_y, 1e-9) * (ny - 1)
    inside = (u >= 0.0) & (u <= nx - 1) & (v >= 0.0) & (v <= ny - 1)

    u_c = np.clip(u, 0.0, nx - 1)
    v_c = np.clip(v, 0.0, ny - 1)
    u0 = np.floor(u_c).astype(np.int32)
    v0 = np.floor(v_c).astype(np.int32)
    u1 = np.minimum(u0 + 1, nx - 1)
    v1 = np.minimum(v0 + 1, ny - 1)
    du = u_c - u0
    dv = v_c - v0
    z00 = src[v0, u0]
    z10 = src[v0, u1]
    z01 = src[v1, u0]
    z11 = src[v1, u1]
    norm = z00 * (1 - du) * (1 - dv) + z10 * du * (1 - dv) + z01 * (1 - du) * dv + z11 * du * dv
    heights = oz + (norm - float(dem_center_norm)) * base_relief_m * sz
    # Outside the patch: push the surface far below so rays miss cleanly.
    heights = np.where(inside, heights, oz - 1.0e4)

    return DEM.from_array(heights, res=float(res), x0=float(xmin), y0=float(ymin))


def lidar_half_fov_rad(
    footprint_m: float = LIDAR_FOOTPRINT_M,
    ref_range_m: float = LIDAR_REF_RANGE_M,
) -> float:
    """Angular half-FOV matching ``simulate_scan_body`` (square detector)."""
    return float(np.arctan2(0.5 * float(footprint_m), float(ref_range_m)))


def dem_altitude_range_m(dem: DEM, origin_z_m: float) -> tuple[float, float]:
    """In-patch DEM height bounds (m), excluding far-below sentinel cells."""
    h = np.asarray(dem.heights, dtype=np.float64)
    valid = h > (float(origin_z_m) - 5.0e3)
    if not np.any(valid):
        return float(h.min()), float(h.max()) + 1.0
    z_lo = float(h[valid].min())
    z_hi = float(h[valid].max())
    if z_hi <= z_lo:
        z_hi = z_lo + 1.0
    return z_lo, z_hi


def render_nadir_camera_rgb(
    dem: DEM,
    sensor_pos_m: np.ndarray,
    R_body_to_world: np.ndarray,
    *,
    z_lo_m: float,
    z_hi_m: float,
    n_det: int = 256,
    footprint_m: float = LIDAR_FOOTPRINT_M,
    ref_range_m: float = LIDAR_REF_RANGE_M,
    cmap_name: str = LIDAR_CMAP,
    seed: int = 0,
    miss_rgb: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Render one body-fixed nadir camera frame through the LiDAR FOV.

    Uses the same detector angular footprint as ``simulate_scan_body``
    (``footprint_m`` at ``ref_range_m``) but with zero range noise so the
    image reads as a clean camera view. Hits are colored by world altitude.

    Returns:
        ``(n_det, n_det, 3)`` uint8 RGB image (row 0 = camera-up).
    """
    pos_m = np.asarray(sensor_pos_m, dtype=np.float64).reshape(3)
    R_bw = np.asarray(R_body_to_world, dtype=np.float64).reshape(3, 3)
    z_ground = float(dem.sample(np.asarray(pos_m[:2])))
    slant = max(float(pos_m[2] - z_ground), 5.0)
    pts, hit = simulate_scan_body(
        dem,
        sensor_pos=pos_m,
        R_body_to_world=R_bw,
        n_det=int(n_det),
        footprint_m=float(footprint_m),
        ref_range_m=float(ref_range_m),
        noise_3sig_at_ref=0.0,
        seed=int(seed),
        boresight_body=(0.0, 0.0, -1.0),
        t_max=max(3.0 * slant, 50.0),
        return_full=True,
    )
    pts = np.asarray(pts, dtype=np.float64)
    hit = np.asarray(hit, dtype=bool)
    z = pts[..., 2]
    span = max(float(z_hi_m) - float(z_lo_m), 1e-9)
    t = np.clip((z - float(z_lo_m)) / span, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgb = (np.asarray(cmap(t)[..., :3]) * 255.0).astype(np.uint8)
    miss = np.asarray(miss_rgb, dtype=np.uint8).reshape(1, 1, 3)
    return np.where(hit[..., None], rgb, miss)


def _empty_pcd() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


def add_body_fixed_lidar_overlay(
    server: viser.ViserServer,
    *,
    pos_vis: np.ndarray,
    attitude_wxyz: np.ndarray,
    scene_scale: float,
    dem_norm: np.ndarray,
    dem_center_norm: float,
    get_terrain_params: Callable[[], dict],
    terrain_half_extent_m: float,
    base_relief_m: float,
    n_det: int = LIDAR_N_DET,
    footprint_m: float = LIDAR_FOOTPRINT_M,
    ref_range_m: float = LIDAR_REF_RANGE_M,
    noise_3sig_at_ref: float = LIDAR_NOISE_3SIG_M,
    max_keyframes: int = LIDAR_MAX_KEYFRAMES,
    point_size: float = LIDAR_POINT_SIZE,
    cmap_name: str = LIDAR_CMAP,
    name: str = "/lidar_pcd",
    gui_folder: str = "LiDAR (body-fixed)",
) -> Callable[[int], None]:
    """Overlay an animated body-fixed nadir LiDAR point cloud on a DEM map.

    ``pos_vis`` / ``attitude_wxyz`` must already be in the hop viser frame
    (position in display units, attitude ``[w,x,y,z]``). Scans are computed
    lazily on a strided keyframe grid and **accumulated** as the animation
    advances (persistent map). Points are colored by altitude (``z``) with a
    fixed DEM-derived scale so the colormap stays stable as the map grows.
    Changing DEM pose via ``get_terrain_params`` invalidates the cache.

    Returns an animation callback ``update(frame_idx)``.
    """
    pos_vis = np.asarray(pos_vis, dtype=np.float64)
    attitude_wxyz = np.asarray(attitude_wxyz, dtype=np.float64)
    n_frames = int(pos_vis.shape[0])
    stride = max(1, int(np.ceil(n_frames / max(max_keyframes, 1))))
    keyframes = list(range(0, n_frames, stride))
    if keyframes[-1] != n_frames - 1:
        keyframes.append(n_frames - 1)
    cmap = plt.get_cmap(cmap_name)

    state = {
        "enabled": True,
        "cache": {},  # keyframe_idx -> per-scan points (viser units)
        "prefix": {},  # keyframe_idx -> accumulated points up to that keyframe
        "dem": None,
        "dem_key": None,
        "z_range": None,  # (z_lo, z_hi) in viser units for altitude coloring
        "handle": None,
        "lock": threading.Lock(),
        "status": "idle",
        "max_key_shown": -1,
    }

    def _terrain_key(params: dict) -> tuple:
        return (
            tuple(float(v) for v in params["origin"]),
            tuple(float(v) for v in params["scale"]),
            float(params["yaw_deg"]),
            tuple(bool(v) for v in params["mirror"]),
        )

    def _clear_caches() -> None:
        state["cache"] = {}
        state["prefix"] = {}
        state["max_key_shown"] = -1

    def _ensure_dem(params: dict) -> DEM:
        key = _terrain_key(params)
        if state["dem"] is not None and state["dem_key"] == key:
            return state["dem"]
        dem = build_terrain_dem(
            dem_norm,
            origin_m=params["origin"],
            scale_xyz=params["scale"],
            yaw_deg=params["yaw_deg"],
            mirror_xy=params["mirror"],
            terrain_half_extent_m=terrain_half_extent_m,
            base_relief_m=base_relief_m,
            dem_center_norm=dem_center_norm,
        )
        state["dem"] = dem
        state["dem_key"] = key
        # Stable altitude colormap from in-patch DEM heights (exclude sentinel lows).
        h = np.asarray(dem.heights, dtype=np.float64)
        valid = h > (float(params["origin"][2]) - 5.0e3)
        if np.any(valid):
            z_lo = float(h[valid].min()) / float(scene_scale)
            z_hi = float(h[valid].max()) / float(scene_scale)
        else:
            z_lo = float(np.min(pos_vis[:, 2]))
            z_hi = float(np.max(pos_vis[:, 2]))
        if z_hi <= z_lo:
            z_hi = z_lo + 1.0
        state["z_range"] = (z_lo, z_hi)
        _clear_caches()
        return dem

    def _scan_keyframe(k: int, dem: DEM) -> np.ndarray:
        if k in state["cache"]:
            return state["cache"][k]
        pos_m = pos_vis[k] * float(scene_scale)
        R_bw = _R_bw_from_wxyz(attitude_wxyz[k])
        # Altitude above local terrain → bound the ray march.
        z_ground = float(dem.sample(np.asarray(pos_m[:2])))
        slant = max(float(pos_m[2] - z_ground), 5.0)
        pcd = simulate_scan_body(
            dem,
            sensor_pos=pos_m,
            R_body_to_world=R_bw,
            n_det=n_det,
            footprint_m=footprint_m,
            ref_range_m=ref_range_m,
            noise_3sig_at_ref=noise_3sig_at_ref,
            seed=1000 + k,
            boresight_body=(0.0, 0.0, -1.0),
            t_max=max(3.0 * slant, 50.0),
        )
        pts = np.asarray(pcd, dtype=np.float32).reshape(-1, 3) / float(scene_scale)
        state["cache"][k] = pts
        return pts

    def _accumulated_up_to(key: int, dem: DEM) -> np.ndarray:
        """Persistent map: all scans from t=0 through the current keyframe."""
        if key in state["prefix"]:
            return state["prefix"][key]

        prev_keys = [k for k in keyframes if k <= key]
        if not prev_keys:
            return _empty_pcd()

        # Resume from the latest already-built prefix.
        resume_i = 0
        acc = _empty_pcd()
        for i, k in enumerate(prev_keys):
            if k in state["prefix"]:
                acc = state["prefix"][k]
                resume_i = i + 1

        for k in prev_keys[resume_i:]:
            scan = _scan_keyframe(k, dem)
            acc = scan if acc.shape[0] == 0 else np.concatenate([acc, scan], axis=0)
            state["prefix"][k] = acc

        return state["prefix"].get(key, acc)

    def _colors_for(pts: np.ndarray) -> np.ndarray:
        """RGB colors from point altitude (``z``), low→high along ``cmap_name``."""
        if pts.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.uint8)
        z = pts[:, 2].astype(np.float64)
        z_range = state.get("z_range")
        if z_range is None:
            z_lo, z_hi = float(z.min()), float(z.max())
        else:
            z_lo, z_hi = z_range
        span = z_hi - z_lo
        if span < 1e-9:
            t = np.full(z.shape, 0.5, dtype=np.float64)
        else:
            t = np.clip((z - z_lo) / span, 0.0, 1.0)
        rgba = cmap(t)
        return (np.asarray(rgba[:, :3]) * 255.0).astype(np.uint8)

    init_pts = _empty_pcd()
    handle = server.scene.add_point_cloud(
        name,
        points=init_pts,
        colors=_colors_for(init_pts),
        point_size=float(point_size),
        point_shape="circle",
    )
    handle.visible = True
    state["handle"] = handle

    def _set_points(pts: np.ndarray) -> None:
        h = state["handle"]
        if h is None:
            return
        h.points = pts.astype(np.float32)
        h.colors = _colors_for(pts)
        h.visible = bool(state["enabled"])

    with server.gui.add_folder(gui_folder):
        en_cb = server.gui.add_checkbox("Show point cloud", initial_value=True)
        clear_btn = server.gui.add_button("Clear accumulated map")
        status_md = server.gui.add_markdown(
            "_Body-fixed nadir LiDAR (post-process only), colored by altitude. "
            "Scans accumulate into a persistent map as the animation plays._"
        )

        @en_cb.on_update
        def _(_e=None) -> None:
            state["enabled"] = bool(en_cb.value)
            h = state["handle"]
            if h is not None:
                h.visible = state["enabled"]

        @clear_btn.on_click
        def _(_e=None) -> None:
            with state["lock"]:
                _clear_caches()
            _set_points(_empty_pcd())
            try:
                status_md.content = (
                    "_Body-fixed nadir LiDAR (post-process only), colored by altitude._  \n"
                    "Cleared — play again to rebuild the map."
                )
            except Exception:
                pass

    def update(frame_idx: int) -> None:
        if not state["enabled"]:
            return
        idx = int(np.clip(frame_idx, 0, n_frames - 1))
        # Nearest keyframe at or before the playhead.
        key = keyframes[int(np.searchsorted(keyframes, idx, side="right") - 1)]
        with state["lock"]:
            try:
                params = get_terrain_params()
                dem = _ensure_dem(params)
                pts = _accumulated_up_to(key, dem)
                n_scans = sum(1 for k in keyframes if k <= key)
                state["max_key_shown"] = max(state["max_key_shown"], key)
                state["status"] = f"{pts.shape[0]:,} pts from {n_scans} scans (through frame {key})"
            except Exception as exc:  # pragma: no cover - viz path
                pts = _empty_pcd()
                state["status"] = f"scan error: {exc}"
        _set_points(pts)
        try:
            status_md.content = (
                f"_Body-fixed nadir LiDAR (post-process only), colored by altitude "
                f"({cmap_name})._  \n"
                f"{state['status']}  ·  n_det={n_det}  ·  stride={stride}"
            )
        except Exception:
            pass

    # Warm the first keyframe so something is visible immediately.
    def _warmup() -> None:
        with state["lock"]:
            try:
                params = get_terrain_params()
                dem = _ensure_dem(params)
                pts = _accumulated_up_to(keyframes[0], dem)
            except Exception:
                return
        _set_points(pts)

    threading.Thread(target=_warmup, daemon=True).start()
    return update


def attach_terrain_param_getter(dem_state: dict) -> Callable[[], dict]:
    """Return a thread-safe getter over a DEM GUI state dict from the hop scripts."""

    def get_terrain_params() -> dict:
        lock: Optional[threading.Lock] = dem_state.get("_lock")
        if lock is not None:
            with lock:
                return {
                    "origin": dem_state["origin"],
                    "scale": dem_state["scale"],
                    "yaw_deg": dem_state["yaw_deg"],
                    "mirror": dem_state["mirror"],
                }
        return {
            "origin": dem_state["origin"],
            "scale": dem_state["scale"],
            "yaw_deg": dem_state["yaw_deg"],
            "mirror": dem_state["mirror"],
        }

    return get_terrain_params
