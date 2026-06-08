"""Coordinate remaps for Viser scene axes."""

from __future__ import annotations

import numpy as np


def model_vec_to_viser_xyz(v: np.ndarray) -> np.ndarray:
    """Map model-frame 3-vectors to Viser (x, y, z): (z, y, x) component order → (x, y, z).

    Linear involution: same mapping converts Viser coordinates back to model.
    """
    a = np.asarray(v, dtype=np.float64)
    if a.size == 0:
        return a
    if a.ndim == 1 and a.shape[0] == 3:
        return np.array([a[2], a[1], a[0]], dtype=np.float64)
    if a.ndim >= 2 and a.shape[-1] == 3:
        return np.stack([a[..., 2], a[..., 1], a[..., 0]], axis=-1)
    return a
