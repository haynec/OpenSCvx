"""Export Viser scenes for static embedding (animated ``.viser`` recordings).

Follows the Viser embedded visualizations workflow:
https://viser.studio/main/embedded_visualizations/

Typical follow-up (once per machine / CI image):

1. Build the client bundle::

       viser-build-client --out-dir docs/assets/viser-client

2. Serve ``docs/assets/`` (or your site root) over HTTP and open::

       .../viser-client/?playbackPath=.../viser-recordings/<name>.viser

3. Embed with an ``<iframe>`` pointing at that URL (e.g. on GitHub Pages).

The ``.viser`` file is binary; keep recordings out of hot paths if they grow large.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np


def repo_root_containing_pyproject(start: Path | None = None) -> Path:
    """Walk upward from ``start`` (or cwd) until ``pyproject.toml`` is found."""
    p = (start or Path.cwd()).resolve()
    for _ in range(12):
        if (p / "pyproject.toml").exists():
            return p
        if p == p.parent:
            break
        p = p.parent
    return Path.cwd().resolve()


def parse_export_viser_path(
    argv: list[str] | None = None,
    *,
    default_output: Path | None = None,
) -> Path | None:
    """If ``--export-viser [PATH]`` is present in ``argv``, return output ``.viser`` path.

    If the flag is given without ``PATH``, ``default_output`` must be provided.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        i = argv.index("--export-viser")
    except ValueError:
        return None
    if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
        return Path(argv[i + 1]).expanduser().resolve()
    if default_output is None:
        raise SystemExit(
            "Usage: --export-viser [PATH]\n"
            "When PATH is omitted, this script must set default_output=... explicitly."
        )
    return default_output.expanduser().resolve()


def set_initial_camera_look_at_trajectory(
    server,
    positions: np.ndarray,
    *,
    eye_offset: tuple[float, float, float] | np.ndarray = (-0.65, -0.95, 0.5),
) -> None:
    """Configure ``server.initial_camera`` from a (N, 3) trajectory (before serialize)."""
    pts = np.asarray(positions, dtype=np.float64)
    center = np.mean(pts, axis=0)
    offset = np.asarray(eye_offset, dtype=np.float64).reshape(3)
    eye = center + offset
    server.initial_camera.position = tuple(float(x) for x in eye)
    server.initial_camera.look_at = tuple(float(x) for x in center)
    server.initial_camera.up = (0.0, 0.0, 1.0)


def export_animated_viser_recording(
    server,
    *,
    step_frame: Callable[[int], None],
    n_frames: int,
    output_path: str | Path,
    fps: float = 24.0,
    max_keyframes: int = 80,
) -> Path:
    """Serialize an animation: ``step_frame(i)`` then ``insert_sleep`` for each keyframe.

    Args:
        server: Active ``ViserServer`` with scene nodes already attached.
        step_frame: Callback that updates the scene for trajectory index ``i``
            (same convention as ``AnimatedServerHandle.step``).
        n_frames: Number of trajectory samples (last index ``n_frames - 1``).
        output_path: Destination ``.viser`` file.
        fps: Frames per second for sleep spacing between keyframes.
        max_keyframes: Uniformly subsample indices if ``n_frames`` exceeds this cap
            (keeps export size reasonable for the web; lower = smaller file, slightly fewer poses).
    """
    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    if n_frames <= max_keyframes:
        frame_indices = np.arange(n_frames, dtype=int)
    else:
        frame_indices = np.linspace(0, n_frames - 1, max_keyframes, dtype=int)
    serializer = server.get_scene_serializer()
    dt = 1.0 / float(fps)
    for fi in frame_indices:
        step_frame(int(fi))
        serializer.insert_sleep(dt)
    data = serializer.serialize()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def print_viser_embed_followup(saved: Path) -> None:
    """Print the usual next steps after writing a ``.viser`` file."""
    rel = saved.name
    print()
    print(f"Wrote Viser recording: {saved}")
    print("Next steps (see https://viser.studio/main/embedded_visualizations/):")
    print("  1) viser-build-client --out-dir docs/assets/viser-client")
    print(
        "  2) Serve the docs assets tree, then open e.g.\n"
        f"       .../viser-client/?playbackPath=.../viser-recordings/{rel}"
    )
    print("  3) Embed that URL in an <iframe> on your docs site.")
    print()
