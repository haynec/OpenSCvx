"""Offline video rendering for animated viser examples.

Pipes raw RGB frames from ``client.get_render()`` directly into ffmpeg over
stdin, so there are no intermediate PNG files and no Python image library
required. The only runtime dependency is ``ffmpeg`` on ``PATH``; ``openscvx``
itself gains nothing from this module.

Typical usage (from an animation example):

.. code-block:: python

    handle = create_animated_plotting_server(results, ..., controls="manual")

    def camera_pose_fn(frame_idx):
        return chase_pose(positions[frame_idx], target_center)

    render_animation_to_video(handle, "out.mp4", camera_pose_fn)

The render blocks on a viser client connection, so you run the script, wait
for the "[render] waiting for a viser client..." line, open the printed viser
URL in a browser, and the render starts automatically.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import viser

from examples.plotting_viser import AnimatedServerHandle

# frame_idx -> (camera_position, camera_wxyz, camera_look_at)
CameraPoseFn = Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]]


def wait_for_client(
    server: viser.ViserServer,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.25,
) -> viser.ClientHandle:
    """Block until at least one viser client connects; return the first one.

    Prints a reminder so the user knows they need to open the printed viser
    URL in a browser for the render to proceed.
    """
    print(
        f"[render] waiting for a viser client to connect "
        f"(open the viser URL above; timeout in {int(timeout_s)}s)..."
    )
    t0 = time.time()
    while True:
        clients = server.get_clients()
        if clients:
            client = next(iter(clients.values()))
            print(f"[render] client connected.")
            return client
        if time.time() - t0 > timeout_s:
            raise TimeoutError(
                f"No viser client connected within {timeout_s:.0f}s. "
                f"Open the viser URL in a browser to start the render."
            )
        time.sleep(poll_interval_s)


def render_animation_to_video(
    handle: AnimatedServerHandle,
    output_path: str | Path,
    camera_pose_fn: CameraPoseFn,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    crf: int = 16,
    preset: str = "slow",
    background_color: tuple[int, int, int] = (16, 17, 19),
    start_frame: int = 0,
    end_frame: int | None = None,
    stride: int = 1,
    settle_s: float = 0.0,
    client: viser.ClientHandle | None = None,
    progress_every: int = 30,
    fov_deg: float | None = None,
) -> Path:
    """Render frames of ``handle`` to an H.264 mp4 by piping raw RGB into ffmpeg.

    Each frame is fetched with ``transport_format="png"`` so we get a *lossless*
    RGBA array from the browser — no JPEG pre-compression stacked under the
    final h.264 pass. The alpha channel is composited onto ``background_color``
    in numpy before writing.

    Why compositing is necessary: viser's dark mode puts the canvas over a
    ``theme.colors.dark[9]`` (= ``#101113``) DOM element (see viser client
    ``App.tsx:426``), and the browser compositor blends the scene over that
    background at display time. But ``client.get_render`` returns only the
    WebGL canvas pixels, which are transparent where nothing is drawn — so
    we have to composite ourselves. The default ``background_color`` matches
    Mantine's ``dark[9]`` exactly so the rendered frames are indistinguishable
    from the live view.

    The video's playback rate (``fps``) is independent of how many frames the
    trajectory contains — ``stride`` controls the frame range from the trajectory
    that gets written. For realtime playback, pick ``fps`` so that
    ``len(range(start_frame, end_frame, stride)) / fps`` matches the trajectory
    duration. For slow-motion, raise ``fps`` relative to that; for time-lapse,
    lower it.

    Args:
        handle: Handle returned by
            ``create_animated_plotting_server(..., controls="manual")``.
        output_path: Destination mp4 file. Parent dirs are created.
        camera_pose_fn: ``frame_idx -> (position, wxyz, look_at)``. Called once
            per rendered frame to position the camera.
        width: Output video width in pixels.
        height: Output video height in pixels.
        fps: Output video frame rate.
        crf: H.264 constant-rate factor; lower is higher quality. 16 is
            visually near-lossless, 18 is high quality, 23 is ffmpeg's default.
        preset: ffmpeg x264 preset (``ultrafast``..``veryslow``). ``slow`` gives
            noticeably better quality/size than ``medium`` for static-heavy 3D
            scenes at modest encode-time cost.
        background_color: RGB tuple (0..255) for the scene background. Applied
            both to viser's scene (so the live view matches) and as the
            composite color for alpha channel in rendered frames.
        start_frame: First trajectory frame index to render (inclusive).
        end_frame: One past the last trajectory frame to render. ``None`` means
            up to ``handle.n_frames``.
        stride: Trajectory frame stride. ``stride=2`` renders every other
            frame, halving the output length at fixed ``fps``.
        settle_s: Optional sleep between pushing scene state and calling
            ``get_render``. Usually 0; bump if you see torn/partial frames.
        client: Pre-connected client. If ``None``, waits for one.
        progress_every: Print a progress line every N rendered frames.
        fov_deg: If given, override the client camera's vertical field of view
            (in degrees) before rendering. Useful for matching a sensor FOV.

    Returns:
        Absolute path to the written mp4.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it locally (e.g. "
            "`brew install ffmpeg` on macOS) — openscvx does not include it "
            "as a package dependency."
        )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Push the background color to viser so the live browser view matches the
    # rendered output. `configure_theme(dark_mode=True)` only styles GUI panels;
    # the 3D canvas clear color is controlled separately by set_background_image.
    bg_rgb = np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)
    bg_image = np.broadcast_to(bg_rgb, (2, 2, 3)).copy()
    handle.server.scene.set_background_image(bg_image, format="png")
    bg_float = np.asarray(background_color, dtype=np.float32).reshape(1, 1, 3)

    if client is None:
        client = wait_for_client(handle.server)

    if fov_deg is not None:
        client.camera.fov = np.radians(fov_deg)

    n = handle.n_frames
    if end_frame is None:
        end_frame = n
    frame_indices = list(range(start_frame, min(end_frame, n), max(stride, 1)))
    if not frame_indices:
        raise ValueError(
            f"No frames to render: start_frame={start_frame}, "
            f"end_frame={end_frame}, n_frames={n}, stride={stride}"
        )

    cmd = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", preset,
        "-movflags", "+faststart",
        str(output_path),
    ]

    print(
        f"[render] {len(frame_indices)} frame(s) @ {fps} fps, "
        f"{width}x{height} -> {output_path}"
    )

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    try:
        t_render_start = time.time()
        for out_i, frame_i in enumerate(frame_indices):
            handle.step(frame_i)
            pos, wxyz, look_at = camera_pose_fn(frame_i)
            with client.atomic():
                client.camera.position = pos
                client.camera.wxyz = wxyz
                client.camera.look_at = look_at
            if settle_s > 0:
                time.sleep(settle_s)
            # PNG transport is lossless RGBA. JPEG would introduce a second
            # compression pass on top of h.264 and visibly hurts quality.
            img = client.get_render(height=height, width=width, transport_format="png")
            if img.ndim != 3 or img.shape[0] != height or img.shape[1] != width:
                raise RuntimeError(
                    f"Unexpected get_render shape {img.shape}; "
                    f"expected ({height}, {width}, 3 or 4)."
                )
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            if img.shape[2] == 4:
                # Composite RGBA onto the chosen background color. Empty scene
                # pixels come back alpha=0 so they'd otherwise be undefined.
                alpha = img[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
                rgb = img[:, :, :3].astype(np.float32)
                composed = rgb * alpha + bg_float * (1.0 - alpha)
                img = composed.astype(np.uint8)
            proc.stdin.write(img.tobytes())
            if progress_every > 0 and (out_i + 1) % progress_every == 0:
                elapsed = time.time() - t_render_start
                done = out_i + 1
                rate = done / max(elapsed, 1e-6)
                eta = (len(frame_indices) - done) / max(rate, 1e-6)
                print(
                    f"[render]   frame {done}/{len(frame_indices)} "
                    f"({rate:.1f} fps, eta {eta:.0f}s)"
                )
        proc.stdin.close()
    except BrokenPipeError as e:
        proc.wait()
        raise RuntimeError(
            "ffmpeg closed its input pipe unexpectedly. See ffmpeg stderr above."
        ) from e

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg exited with code {ret}")

    print(f"[render] done: {output_path}")
    return output_path
