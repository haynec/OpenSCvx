"""Cinematic animation of the drone-racing-with-polytope-viewplanning example.

The trajectory optimization problem itself lives in
`examples/drone/dr_vp_polytope.py`; this file imports that `problem` (and the
associated `plotting_dict`), solves it, and then drives a viser scene with a
cinematic camera orbit suitable for landing-page / presentation captures.
"""

import os
import sys
import threading
import time as _time

import numpy as np
import viser.transforms as vtf

# Add the project root so `examples.*` imports resolve.
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.drone.dr_vp_polytope import plotting_dict, problem
from examples.plotting_viser import (
    create_animated_plotting_server,
    create_scp_animated_plotting_server,
)


def _look_at_wxyz(pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) for a camera at `pos` looking at `target`.

    Uses the OpenCV camera convention that viser expects: +X right, +Y down,
    +Z forward. See `examples/animations/camera_control_notes.md`.
    """
    forward = target - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_down = np.cross(forward, right)  # = -world_up projected perp to forward
    R_world_cam = np.stack([right, cam_down, forward], axis=1)
    return vtf.SO3.from_matrix(R_world_cam).wxyz


def _set_camera(server, pos: np.ndarray, wxyz: np.ndarray, look_at: np.ndarray) -> None:
    """Atomically push a camera pose to every connected client."""
    for _, client in server.get_clients().items():
        with client.atomic():
            client.camera.position = pos
            client.camera.wxyz = wxyz
            client.camera.look_at = look_at


def orbit_camera(
    server,
    center=(100.0, -50.0, 20.0),
    radius=60.0,
    height=25.0,
    period_s=12.0,
    fps=60,
    up=(0.0, 0.0, 1.0),
):
    """Continuously orbit every connected client's camera around a fixed `center`.

    Useful for surveying a static scene (e.g. the polytope target cluster).
    For a camera that follows the drone, see `drone_tracking_camera`.
    """
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    def _loop():
        t0 = _time.time()
        while True:
            theta = 2 * np.pi * ((_time.time() - t0) % period_s) / period_s
            pos = center + np.array(
                [radius * np.cos(theta), radius * np.sin(theta), height]
            )
            _set_camera(server, pos, _look_at_wxyz(pos, center, up), center)
            _time.sleep(1.0 / fps)

    threading.Thread(target=_loop, daemon=True).start()


def drone_tracking_camera(
    server,
    positions: np.ndarray,
    traj_time: np.ndarray,
    radius: float = 8.0,
    height: float = 3.0,
    period_s: float = 8.0,
    fps: int = 60,
    up=(0.0, 0.0, 1.0),
):
    """Orbit the camera around the moving drone, keeping it centered in frame.

    The camera laps the drone every `period_s` seconds (constant angular speed
    in the drone's frame — cinematic regardless of how fast the drone moves).

    The function runs on its own wall-clock loop synchronized to the
    trajectory's realtime duration `(traj_time[-1] - traj_time[0])`. The
    trajectory animation server (`create_animated_plotting_server`) uses an
    independent realtime clock too — they stay in lock-step as long as the
    Animation panel's "Speed" slider is left at 1.0× and playback isn't
    paused or scrubbed. If they drift, restart playback from the beginning
    to re-sync.

    Args:
        server: The viser server returned by `create_animated_plotting_server`.
        positions: (N, 3) drone position trajectory in world coordinates.
        traj_time: (N,) timestamps matching `positions`, monotonically increasing.
        radius: Horizontal orbit radius around the drone (world units).
        height: Vertical offset of the camera above the drone (world units).
        period_s: Seconds per full camera revolution around the drone.
        fps: Camera update rate.
        up: World up direction (default z-up).
    """
    positions = np.asarray(positions, dtype=np.float64)
    traj_time = np.asarray(traj_time, dtype=np.float64).flatten()
    up = np.asarray(up, dtype=np.float64)

    t_start = float(traj_time[0])
    t_end = float(traj_time[-1])
    duration = t_end - t_start

    def drone_at(sim_t: float) -> np.ndarray:
        """Linearly interpolate the drone position at simulation time `sim_t`."""
        return np.array(
            [np.interp(sim_t, traj_time, positions[:, k]) for k in range(3)]
        )

    def _loop():
        t0 = _time.time()
        while True:
            wall = (_time.time() - t0) % duration
            sim_t = t_start + wall
            theta = 2 * np.pi * (wall / period_s)

            target = drone_at(sim_t)
            pos = target + np.array(
                [radius * np.cos(theta), radius * np.sin(theta), height]
            )
            _set_camera(server, pos, _look_at_wxyz(pos, target, up), target)
            _time.sleep(1.0 / fps)

    threading.Thread(target=_loop, daemon=True).start()


if __name__ == "__main__":
    problem.initialize()
    problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    # Create both visualization servers (viser auto-assigns ports)
    traj_server = create_animated_plotting_server(
        results,
        thrust_key="thrust_force",
        viewcone_scale=10.0,
        show_control_plot="thrust_force",
        show_control_norm_plot="thrust_force",
    )
    scp_server = create_scp_animated_plotting_server(
        results,
        attitude_stride=3,
        frame_duration_ms=200,
    )

    # Cinematic orbit around the polytope target cluster.
    # orbit_camera(traj_server)

    # Camera that orbits the drone, keeping it centered in frame.
    drone_tracking_camera(
        traj_server,
        results.trajectory["position"],
        results.trajectory["time"],
    )

    # Keep both servers running
    traj_server.sleep_forever()
