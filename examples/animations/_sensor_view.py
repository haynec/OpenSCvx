"""Offline mp4 render of the 2D camera animation panel.

Mirrors ``_render.py`` for the viser 3D scene: iterates the *same* frame-index
set (``range(start_frame, end_frame, stride)``) so the resulting mp4 is
frame-perfectly aligned with the viser recording and can be composited with
ffmpeg afterwards.

Uses matplotlib's Agg canvas rather than plotly + kaleido so there's no
headless-Chrome dependency — matplotlib is already a project dep. Raw RGB
frames are piped straight into ``ffmpeg -f rawvideo`` (same pattern as
``_render.py``), so there's no intermediate PNGs and no Python-side decode.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import matplotlib
import numpy as np

# Agg = headless raster backend; required for server/CI rendering, and avoids
# opening a GUI window during the frame loop.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from examples.plotting import full_subject_traj_time  # noqa: E402

# Default per-subject palette, cycled by index. Kept in sync with the viser
# target-marker palette at ``openscvx/plotting/viser/animated.py:182-189`` so
# that subject k in the camera panel is the same color as target k rendered
# in the viser scene. If that palette ever changes, update this one too.
VISER_SUBJECT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 50, 50),  # Red
    (50, 255, 50),  # Green
    (50, 50, 255),  # Blue
    (255, 255, 50),  # Yellow
    (255, 50, 255),  # Magenta
    (50, 255, 255),  # Cyan
)


def _cone_outline_xy(results, n_grid: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Closed (X, Y) polyline for the red camera-frame outline.

    Reproduces the math from ``plot_camera_animation`` exactly so this panel
    matches the GUI version pixel-for-pixel (modulo renderer).
    """
    A = np.diag(
        [
            1 / np.tan(np.pi / results["alpha_y"]),
            1 / np.tan(np.pi / results["alpha_x"]),
        ]
    )
    range_limit = 10 if "moving_subject" in results else 80
    norm = results["norm_type"]
    ord_ = np.inf if norm == "inf" else norm

    xs = np.linspace(-range_limit, range_limit, n_grid)
    ys = np.linspace(-range_limit, range_limit, n_grid)
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.flatten(), Y.flatten()
    # Keep the original (x_val outer, y_val inner) ordering so the sort-by-arctan
    # below produces the same outline as the existing GUI plot.
    Z = np.array(
        [np.linalg.norm(A @ np.array([x_val, y_val]), ord=ord_) for x_val in xs for y_val in ys]
    )
    X, Y = X / Z, Y / Z
    order = np.argsort(np.arctan2(Y, X))
    X, Y = X[order], Y[order]
    X = np.append(X, X[0])
    Y = np.append(Y, Y[0])
    return X, Y


def _project_subject(sub: np.ndarray) -> np.ndarray:
    """Divide sensor-frame (x, y, z) by z so trajectories live on the image plane."""
    out = np.asarray(sub, dtype=np.float64).copy()
    if out.size == 0:
        return out
    out[:, 0] = out[:, 0] / out[:, 2]
    out[:, 1] = out[:, 1] / out[:, 2]
    return out


def _rgb_from_0_255(color) -> tuple[float, float, float]:
    """Convert either a 0..255 (r, g, b) tuple or a plotly ``"rgb(r, g, b)"``
    string (what ``generate_subject_colors`` actually returns) to matplotlib 0..1.
    """
    if isinstance(color, str):
        inner = color[color.index("(") + 1 : color.rindex(")")]
        r, g, b = (int(v.strip()) for v in inner.split(","))
    else:
        r, g, b = color
    return (r / 255.0, g / 255.0, b / 255.0)


def render_camera_panel_to_video(
    results,
    output_path: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    stride: int = 1,
    fps: int = 30,
    width: int = 1080,
    height: int = 1080,
    crf: int = 16,
    preset: str = "slow",
    background_color: tuple[int, int, int] = (16, 17, 19),
    cone_color: tuple[int, int, int] = (255, 100, 100),
    subject_colors: tuple[tuple[int, int, int], ...] | None = None,
    progress_every: int = 30,
) -> Path:
    """Render the 2D camera-frame animation to an mp4, aligned with ``_render.py``.

    Iterates ``range(start_frame, min(end_frame, n), max(stride, 1))`` — pass the
    *exact same* ``start_frame`` / ``end_frame`` / ``stride`` you pass to
    :func:`render_animation_to_video` and the two mp4s will be the same length
    with matched frame indices, ready to composite side-by-side with ffmpeg
    (e.g. ``ffmpeg -i viser.mp4 -i camera.mp4 -filter_complex hstack=inputs=2 ...``).

    Args:
        results: Post-processed :class:`OptimizationResults` with ``init_poses``,
            ``R_sb``, ``alpha_x``, ``alpha_y``, ``norm_type`` populated (i.e.,
            after ``results.update(plotting_dict)``).
        output_path: Destination mp4. Parent dirs are created.
        start_frame, end_frame, stride: Frame-index range. Mirror whatever you
            pass to ``render_animation_to_video`` for frame-perfect alignment.
        fps: Output video frame rate. Match the viser-side ``fps``.
        width, height: Output pixel dimensions.
        crf, preset: x264 quality / speed knobs (see ``_render.py``).
        background_color: RGB tuple applied to both the figure and axes
            facecolors. Defaults to viser's Mantine ``dark[9]`` (#101113) so
            side-by-side composites are visually seamless.
        cone_color: RGB tuple for the camera-frame outline. Defaults to
            ``(255, 100, 100)`` to match the viser viewcone ring color used
            when ``viewcone_ring_only=True`` is passed to
            ``create_animated_plotting_server``.
        subject_colors: Palette cycled by subject index. Defaults to
            :data:`VISER_SUBJECT_PALETTE` so each subject's trail color matches
            the corresponding viser target-marker color.
        progress_every: Print a progress line every N rendered frames.

    Returns:
        Absolute path to the written mp4.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it locally (e.g. `brew install ffmpeg`)."
        )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # `full_subject_traj_time` currently ignores its `params` argument — passing
    # None keeps this helper decoupled from the example's Config object.
    _, subs_sen, _, subs_sen_node = full_subject_traj_time(results, None)
    palette = subject_colors if subject_colors is not None else VISER_SUBJECT_PALETTE
    colors = [palette[k % len(palette)] for k in range(len(subs_sen))]
    cone_X, cone_Y = _cone_outline_xy(results)

    # Pre-project every subject trajectory once — inside the loop we only slice.
    subs_sen_proj = [_project_subject(s) for s in subs_sen]
    subs_sen_node_proj = [_project_subject(s) for s in subs_sen_node]

    # Node markers must appear when the *time* of the current propagation sample
    # passes the node time, not when a sample-count ratio ticks over. The
    # original `plot_camera_animation` uses the ratio form, which silently
    # desyncs the markers from the continuous line whenever nodes aren't
    # uniformly spaced in propagation-time (common with free-final-time or
    # time-dilated problems).
    t_full = np.asarray(results.trajectory["time"], dtype=np.float64).flatten()
    t_nodes = np.asarray(results.nodes["time"], dtype=np.float64).flatten()

    n = len(subs_sen_proj[0])
    if end_frame is None:
        end_frame = n
    frame_indices = list(range(start_frame, min(end_frame, n), max(stride, 1)))
    if not frame_indices:
        raise ValueError(
            f"No frames to render: start_frame={start_frame}, "
            f"end_frame={end_frame}, n_samples={n}, stride={stride}"
        )

    bg_mpl = _rgb_from_0_255(background_color)

    # Build figure once; we'll reuse it and just update line data each frame.
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_facecolor(bg_mpl)
    ax = fig.add_axes([0, 0, 1, 1])  # full-figure axes, zero margins
    ax.set_facecolor(bg_mpl)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Static cone outline — default color matches the viser viewcone ring.
    ax.plot(cone_X, cone_Y, color=_rgb_from_0_255(cone_color), linewidth=5.0, zorder=2)

    # One Line2D + one scatter per subject — we mutate their data per frame
    # instead of recreating artists (much cheaper than cla+replot).
    line_artists = []
    marker_artists = []
    for sub_idx in range(len(subs_sen_proj)):
        c = _rgb_from_0_255(colors[sub_idx])
        (ln,) = ax.plot([], [], color=c, linewidth=3.0, zorder=3)
        # Use ax.plot with marker-only for a single Line2D (scatter is slower).
        (mk,) = ax.plot([], [], linestyle="none", marker="o", markersize=6.0, color=c, zorder=4)
        line_artists.append(ln)
        marker_artists.append(mk)

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    print(
        f"[camera-panel] {len(frame_indices)} frame(s) @ {fps} fps, "
        f"{width}x{height} -> {output_path}"
    )
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    try:
        t_start = time.time()
        for out_i, i in enumerate(frame_indices):
            current_t = t_full[i]
            # How many node times have we passed at this propagation sample?
            # `side="right"` includes nodes whose time equals the current time.
            n_nodes_visible = int(np.searchsorted(t_nodes, current_t, side="right"))
            for sub_idx, (sub, sub_nodal) in enumerate(zip(subs_sen_proj, subs_sen_node_proj)):
                line_artists[sub_idx].set_data(sub[: i + 1, 0], sub[: i + 1, 1])
                node_slice = sub_nodal[: min(n_nodes_visible, sub_nodal.shape[0])]
                if node_slice.size:
                    marker_artists[sub_idx].set_data(node_slice[:, 0], node_slice[:, 1])
                else:
                    marker_artists[sub_idx].set_data([], [])

            canvas.draw()
            # buffer_rgba is (H, W, 4) uint8; drop alpha (figure has opaque bg).
            rgba = np.asarray(canvas.buffer_rgba())
            if rgba.shape[0] != height or rgba.shape[1] != width:
                raise RuntimeError(
                    f"Canvas returned {rgba.shape[:2]}, expected {(height, width)}. "
                    f"Check dpi/figsize."
                )
            proc.stdin.write(rgba[..., :3].tobytes())

            if progress_every > 0 and (out_i + 1) % progress_every == 0:
                elapsed = time.time() - t_start
                done = out_i + 1
                rate = done / max(elapsed, 1e-6)
                eta = (len(frame_indices) - done) / max(rate, 1e-6)
                print(
                    f"[camera-panel]   frame {done}/{len(frame_indices)} "
                    f"({rate:.1f} fps, eta {eta:.0f}s)"
                )
        proc.stdin.close()
    except BrokenPipeError as e:
        proc.wait()
        raise RuntimeError(
            "ffmpeg closed its input pipe unexpectedly. See ffmpeg stderr above."
        ) from e
    finally:
        plt.close(fig)

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg exited with code {ret}")

    print(f"[camera-panel] done: {output_path}")
    return output_path
