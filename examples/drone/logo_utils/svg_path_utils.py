"""SVG path extraction utilities for the logo-tracing drone examples."""

import jax.numpy as jnp
import numpy as np
from svgpathtools import svg2paths2


def print_svg_path_attributes(svg_file_path):
    """
    Print the attributes of all paths in the SVG file for inspection.
    """
    paths, attributes, svg_attr = svg2paths2(svg_file_path)
    for i, attr in enumerate(attributes):
        print(f"Path {i}: {attr}")


def extract_svg_path(
    svg_file_path,
    n_points=2000,
    flip_y=True,
    path_indices=None,
    preserve_aspect_ratio=False,
):
    """
    Extract a continuous, high-resolution path from an SVG file using svgpathtools.
    Optionally, only use specific path indices.

    When ``preserve_aspect_ratio`` is True, both axes share a uniform scale so wide
    logos (e.g. wordmarks) are not squashed into a square bounding box.
    """
    paths, attributes, svg_attr = svg2paths2(svg_file_path)
    if path_indices is not None:
        paths = [paths[i] for i in path_indices]
    all_points = []
    for path in paths:
        for seg in path:
            seg_len = seg.length()
            n_seg_points = max(2, int(n_points * seg_len / path.length()))
            ts = np.linspace(0, 1, n_seg_points, endpoint=False)
            for t in ts:
                pt = seg.point(t)
                all_points.append([pt.real, pt.imag])
    all_points = np.array(all_points)
    min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    if flip_y:
        all_points[:, 1] = max_y - (all_points[:, 1] - min_y)
    if preserve_aspect_ratio:
        width = max_x - min_x
        height = max_y - min_y
        max_dim = max(width, height)
        scale = 20.0 / max_dim
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        all_points[:, 0] = (all_points[:, 0] - cx) * scale
        all_points[:, 1] = (all_points[:, 1] - cy) * scale
    else:
        all_points[:, 0] = 20 * (all_points[:, 0] - min_x) / (max_x - min_x) - 10
        all_points[:, 1] = 20 * (all_points[:, 1] - min_y) / (max_y - min_y) - 10
    all_points = np.column_stack([all_points, np.full(len(all_points), 2.0)])
    idxs = np.linspace(0, len(all_points) - 1, n_points).astype(int)
    sampled_points = all_points[idxs]
    # Convert to JAX array for JAX-compatible indexing
    sampled_points_jax = jnp.array(sampled_points)

    def path_function(t):
        t = jnp.clip(t, 0, 1)
        idx = jnp.clip(jnp.floor(t * (n_points - 1)), 0, n_points - 1).astype(int)
        return sampled_points_jax[idx]

    return path_function


def get_svg_path_function(svg_file_path, path_indices=None, preserve_aspect_ratio=False):
    return extract_svg_path(
        svg_file_path,
        n_points=2000,
        flip_y=True,
        path_indices=path_indices,
        preserve_aspect_ratio=preserve_aspect_ratio,
    )


def extract_svg_arc_length_path(svg_file_path, n_samples=1000, path_indices=None, width=20.0):
    """Sample an SVG path uniformly in arc length for progress-parametrized tracing.

    Unlike :func:`extract_svg_path`, which parametrizes the path by a normalized
    time in ``[0, 1]``, this returns the cumulative arc length ``s`` alongside
    the sampled points, so a contouring (MPCC-style) problem can use ``s``
    directly as its progress coordinate. Expects a single continuous path — a
    pen lift between subpaths would be traced as a straight-line jump.

    Args:
        svg_file_path: Path to the SVG file.
        n_samples: Number of arc-length-uniform samples to return.
        path_indices: Optional indices selecting which SVG paths to use.
        width: Uniform scale target — the larger bounding-box dimension of the
            returned points spans this many meters (aspect ratio preserved).

    Returns:
        Tuple ``(s, points)`` where ``s`` is a strictly increasing
        ``(n_samples,)`` array of arc length starting at 0, and ``points`` is
        an ``(n_samples, 2)`` array of planar coordinates centered at the
        origin with y up.
    """
    paths, _, _ = svg2paths2(svg_file_path)
    if path_indices is not None:
        paths = [paths[i] for i in path_indices]

    # Dense parameter-space sampling, allocated per segment by segment length
    dense = []
    total_length = sum(path.length() for path in paths)
    for path in paths:
        for seg in path:
            n_seg = max(2, int(4 * n_samples * seg.length() / total_length))
            for t in np.linspace(0, 1, n_seg, endpoint=False):
                pt = seg.point(t)
                dense.append([pt.real, pt.imag])
    end = paths[-1][-1].point(1.0)
    dense.append([end.real, end.imag])
    dense = np.array(dense)

    # Center at the origin with y up, then scale uniformly to the target width
    dense[:, 1] = -dense[:, 1]
    dense -= 0.5 * (dense.min(axis=0) + dense.max(axis=0))
    dense *= width / (dense.max(axis=0) - dense.min(axis=0)).max()

    # Cumulative arc length; drop stationary points so s is strictly increasing
    step = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    keep = np.concatenate([[True], step > 1e-9])
    dense = dense[keep]
    s_dense = np.concatenate([[0.0], np.cumsum(step[step > 1e-9])])

    # Resample uniformly in arc length
    s = np.linspace(0.0, s_dense[-1], n_samples)
    points = np.column_stack(
        [np.interp(s, s_dense, dense[:, 0]), np.interp(s, s_dense, dense[:, 1])]
    )
    return s, points
