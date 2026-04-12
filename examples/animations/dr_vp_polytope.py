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


def orbit_camera(
    server,
    center=(100.0, -50.0, 20.0),
    radius=60.0,
    height=25.0,
    period_s=12.0,
    fps=60,
    up=(0.0, 0.0, 1.0),
):
    """Continuously orbit every connected client's camera around `center`.

    See `examples/animations/camera_control_notes.md` for the underlying
    viser API and the rationale for setting both `position` and `wxyz`
    atomically (assigning `look_at` alone does not reorient the camera).
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

            # viser uses the OpenCV camera convention: +X right, +Y down,
            # +Z forward (toward the look target).
            forward = center - pos
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            cam_down = np.cross(forward, right)  # = -world_up projected perp to forward
            R_world_cam = np.stack([right, cam_down, forward], axis=1)
            wxyz = vtf.SO3.from_matrix(R_world_cam).wxyz

            for _, client in server.get_clients().items():
                with client.atomic():
                    client.camera.position = pos
                    client.camera.wxyz = wxyz
                    client.camera.look_at = center
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
    orbit_camera(traj_server)

    # Keep both servers running
    traj_server.sleep_forever()
