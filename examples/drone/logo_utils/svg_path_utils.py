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
