"""Viser-rendered nadir camera mp4 for the SENSS hop (LiDAR FOV / footprint).

Dedicated FPV scene: lit DEM terrain only — no trajectory trail, vehicle
markers, or LiDAR point cloud. Camera uses the same square angular FOV as the
body-fixed LiDAR (``footprint_m`` at ``ref_range_m``).

Output is 8"×8" at 300 dpi (2400×2400), 30 fps.

Run::

    python examples/animations/senss_hop_lidar_camera.py

Requires ``ffmpeg`` on ``PATH``. A headless Chromium client is started via
Playwright when available; otherwise open the printed viser URL in a browser.
"""

from __future__ import annotations

import importlib
import os
import pickle
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

hop = importlib.import_module("examples.rocket.senss.6DoF_pdg_stc_senss_hop")
from examples.animations._render import render_animation_to_video  # noqa: E402
from examples.plotting_viser import AnimatedServerHandle  # noqa: E402
from examples.rocket.senss.hop_lidar_overlay import (  # noqa: E402
    LIDAR_FOOTPRINT_M,
    LIDAR_REF_RANGE_M,
    lidar_half_fov_rad,
)
from openscvx.plotting.viser import create_server  # noqa: E402

OUTPUT_PATH = os.path.join(current_dir, "mp4", "senss_hop_lidar_camera.mp4")
CACHE_PATH = os.path.join(current_dir, "mp4", ".senss_hop_lidar_camera_result.pkl")

# 8" × 8" at 300 dpi
DPI = 300
WIDTH = 2400
HEIGHT = 2400
FPS = 30
PROPAGATION_HZ = FPS
CRF = 14
PRESET = "slow"

# Same native DEM as the hop example.
RENDER_DEM_GRID = 3938

# Identity body→sensor: hop FPV looks along sensor −Z = body −Z (nadir).
R_SB_NADIR = np.eye(3, dtype=np.float64)


def _prepare_render_dem(grid_n: int) -> None:
    """Reload hop DEM assets at ``grid_n`` (native size preferred)."""
    hop.DEM_GRID = int(grid_n)
    img = Image.open(hop._DEM_PATH)
    raw = np.array(img, dtype=np.uint16)
    lo, hi = float(raw.min()), float(raw.max())
    if raw.shape[0] == hop.DEM_GRID and raw.shape[1] == hop.DEM_GRID:
        arr = raw.astype(np.float32)
    else:
        arr = np.array(
            img.resize((hop.DEM_GRID, hop.DEM_GRID), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    hop._dem_norm = (arr - lo) / max(hi - lo, 1.0)
    hop._dem_center_i = hop._dem_center_j = (hop.DEM_GRID - 1) // 2
    hop._dem_center_norm = float(
        hop._dem_norm[hop._dem_center_i, hop._dem_center_j]
    )
    hop._terrain_faces = hop._make_terrain_faces()


def _open_viser_client(url: str) -> None:
    """Connect a browser client so ``get_render`` can capture frames."""

    def _playwright_connect() -> bool:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return False

        def _run() -> None:
            with sync_playwright() as p:
                try:
                    browser = p.chromium.launch(
                        channel="chrome",
                        headless=True,
                        args=["--ignore-gpu-blocklist", "--enable-webgl"],
                    )
                except Exception:
                    browser = p.chromium.launch(
                        headless=True,
                        args=["--ignore-gpu-blocklist", "--enable-webgl"],
                    )
                page = browser.new_page(
                    viewport={"width": WIDTH + 64, "height": HEIGHT + 64},
                    device_scale_factor=1,
                )
                page.goto(url, wait_until="networkidle")
                time.sleep(8.0)
                while True:
                    time.sleep(1.0)

        threading.Thread(target=_run, daemon=True).start()
        return True

    if _playwright_connect():
        print(f"[hop-cam] Chromium connecting to {url}")
        return
    print(f"[hop-cam] opening {url} in the system browser")
    try:
        webbrowser.open(url)
    except Exception as exc:  # pragma: no cover
        print(f"[hop-cam] could not auto-open browser ({exc}); open the URL manually")


def _create_nadir_camera_server(traj_time: np.ndarray, pos_vis: np.ndarray) -> AnimatedServerHandle:
    """DEM-only viser scene for the body-fixed nadir camera (no traj / LiDAR)."""
    # Cover launch→apex→land under the narrow nadir FOV. Illumination unchanged.
    hop.TERRAIN_HALF_EXTENT_M = 320.0
    hop.DEM_POS_X_M = 95.0
    hop.DEM_POS_Y_M = 0.0
    hop.DEM_SCALE_X = 1.0
    hop.DEM_SCALE_Y = 1.0

    server = create_server(pos_vis, dark_mode=True, show_grid=False)
    server.gui.configure_theme(
        dark_mode=True,
        control_layout="collapsible",
        control_width="small",
        show_logo=False,
        show_share_button=False,
        titlebar_content=None,
    )
    try:
        server.scene.remove_by_name("/origin")
    except Exception:
        pass

    hop._add_dem_to_server(server, fov_slider=False)
    return AnimatedServerHandle(
        server=server,
        traj_time=np.asarray(traj_time, dtype=np.float64).flatten(),
        update_callbacks=[],
    )


if __name__ == "__main__":
    reuse = os.environ.get("SENSS_HOP_CAM_REUSE", "1") != "0"
    result = None
    if reuse and os.path.isfile(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                result = pickle.load(f)
            print(f"[hop-cam] reusing cached solve: {CACHE_PATH}")
        except Exception as exc:
            print(f"[hop-cam] cache load failed ({exc}); re-solving")
            result = None

    if result is None:
        problem = hop.problem
        problem.settings.dev.debug = True
        problem.settings.prp.dt = 1.0 / PROPAGATION_HZ
        problem.initialize()
        problem.solve()
        result = problem.post_process()
        Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(result, f)
        print(f"[hop-cam] cached solve -> {CACHE_PATH}")

    hop.prepare_for_viser(result)

    pos = np.asarray(result.trajectory["position"], dtype=np.float64)
    attitude = np.asarray(result.trajectory["attitude"], dtype=np.float64)
    t = np.asarray(result.trajectory["time"], dtype=np.float64).flatten()

    fov_deg = float(np.degrees(2.0 * lidar_half_fov_rad(LIDAR_FOOTPRINT_M, LIDAR_REF_RANGE_M)))
    print(
        f"[hop-cam] {len(t)} samples, T={t[-1]:.2f}s  ·  "
        f"nadir camera FOV={fov_deg:.2f}° "
        f"(LiDAR footprint={LIDAR_FOOTPRINT_M:g} m @ {LIDAR_REF_RANGE_M:g} m)"
    )
    print(f"[hop-cam] capture {WIDTH}x{HEIGHT} @ {DPI} dpi")

    _prepare_render_dem(RENDER_DEM_GRID)
    handle = _create_nadir_camera_server(t, pos)

    def camera_pose_fn(frame_idx: int):
        cam_pos, cam_wxyz = hop._los_sensor_fpv_pose(
            pos[frame_idx],
            attitude[frame_idx],
            R_SB_NADIR,
            roll_deg=0.0,
        )
        return cam_pos, cam_wxyz, None

    port = handle.server.get_port()
    url = f"http://localhost:{port}"
    _open_viser_client(url)

    out = render_animation_to_video(
        handle,
        Path(OUTPUT_PATH),
        camera_pose_fn,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        crf=CRF,
        preset=PRESET,
        background_color=(0, 0, 0),
        fov_deg=fov_deg,
        settle_s=0.05,
        progress_every=30,
    )
    print(f"[hop-cam] wrote {out}  ({WIDTH}x{HEIGHT} @ {DPI} dpi, {FPS} fps)")
