"""Lunar heightfield helpers for drone examples (SENNS DEM + procedural)."""

from __future__ import annotations

import os

import numpy as np

# Default path to the SENNS lunar DEM shipped with the rocket examples.
_DEFAULT_SENNS_DEM = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "rocket",
        "senss",
        "senns_dem.png",
    )
)


def load_senns_dem(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    grid: int = 2048,
    elev_scale: float = 50.0,
    z_offset: float = 0.0,
    dem_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the SENNS lunar DEM onto a regular world-XY height grid.

    Matches the rocket SENNS loader (``6DoF_pdg_dem_static``): normalize the
    full-resolution 16-bit PNG, bilinear-resize to ``grid×grid``, then map onto
    ``[x_min, x_max] × [y_min, y_max]`` with vertical scale ``elev_scale``.

    Returns
    -------
    x_grid : (grid,) strictly increasing
    y_grid : (grid,) strictly increasing
    H : (grid, grid) with ``H[i, j]`` = height at ``(x_grid[i], y_grid[j])``
        (matches ``ox.Bilerp`` ``fp`` layout).
    """
    from PIL import Image

    path = dem_path or _DEFAULT_SENNS_DEM
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"SENNS DEM not found at {path}. Restore with:\n"
            f"  git checkout HEAD -- examples/rocket/senss/senns_dem.png"
        )

    img = Image.open(path)
    raw = np.array(img, dtype=np.uint16)
    lo, hi = float(raw.min()), float(raw.max())
    # Image.resize size is (width, height) = (nx, ny); array is (ny, nx) = (row=y, col=x).
    arr_yx = np.array(img.resize((grid, grid), Image.BILINEAR), dtype=np.float64)
    dem_yx = (arr_yx - lo) / max(hi - lo, 1.0)

    x_grid = np.linspace(x_min, x_max, grid, dtype=np.float64)
    y_grid = np.linspace(y_min, y_max, grid, dtype=np.float64)
    # Transpose so H[i, j] sits at (x_grid[i], y_grid[j]).
    H = dem_yx.T * float(elev_scale) + float(z_offset)
    H = H - float(H.min())  # floor at zero; AGL is relative
    return x_grid, y_grid, H


def synthesize_lunar_terrain(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
    *,
    seed: int = 7,
    n_large: int = 5,
    n_small: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a smooth multi-scale height grid (large hills + Fourier + small features).

    Returns
    -------
    x_grid : (nx,) strictly increasing
    y_grid : (ny,) strictly increasing
    H : (nx, ny) with ``H[i, j]`` = height at ``(x_grid[i], y_grid[j])``
        (matches ``ox.Bilerp`` ``fp`` layout).
    """
    rng = np.random.default_rng(seed)
    x_grid = np.linspace(x_min, x_max, nx, dtype=np.float64)
    y_grid = np.linspace(y_min, y_max, ny, dtype=np.float64)
    XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
    H = np.zeros((nx, ny), dtype=np.float64)

    lx = float(x_max - x_min)
    ly = float(y_max - y_min)

    for _ in range(n_large):
        amp = float(rng.uniform(8.0, 35.0))
        cx = float(rng.uniform(x_min, x_max))
        cy = float(rng.uniform(y_min, y_max))
        sx = float(rng.uniform(40.0, 120.0))
        sy = float(rng.uniform(40.0, 120.0))
        H += amp * np.exp(-(((XX - cx) / sx) ** 2) - (((YY - cy) / sy) ** 2))

    for kx in range(0, 4):
        for ky in range(0, 4):
            if kx == 0 and ky == 0:
                continue
            knorm2 = float(kx * kx + ky * ky)
            if knorm2 > 9.0:
                continue
            amp = float(rng.uniform(1.5, 6.0) / (1.0 + knorm2))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            H += amp * np.cos(
                2.0 * np.pi * (kx * (XX - x_min) / lx + ky * (YY - y_min) / ly) + phase
            )

    for _ in range(n_small):
        amp = float(rng.uniform(-4.0, 4.0))
        cx = float(rng.uniform(x_min, x_max))
        cy = float(rng.uniform(y_min, y_max))
        s = float(rng.uniform(3.0, 12.0))
        H += amp * np.exp(-(((XX - cx) / s) ** 2) - (((YY - cy) / s) ** 2))

    H -= float(H.min())
    return x_grid, y_grid, H


def bilinear_height(
    x: float,
    y: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    H: np.ndarray,
) -> float:
    """NumPy bilinear sample matching ``ox.Bilerp`` clamping at boundaries."""
    x = float(np.clip(x, x_grid[0], x_grid[-1]))
    y = float(np.clip(y, y_grid[0], y_grid[-1]))

    i = int(np.searchsorted(x_grid, x, side="right") - 1)
    j = int(np.searchsorted(y_grid, y, side="right") - 1)
    i = int(np.clip(i, 0, len(x_grid) - 2))
    j = int(np.clip(j, 0, len(y_grid) - 2))

    x0, x1 = float(x_grid[i]), float(x_grid[i + 1])
    y0, y1 = float(y_grid[j]), float(y_grid[j + 1])
    tx = 0.0 if x1 <= x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 <= y0 else (y - y0) / (y1 - y0)

    z00 = float(H[i, j])
    z10 = float(H[i + 1, j])
    z01 = float(H[i, j + 1])
    z11 = float(H[i + 1, j + 1])
    z0 = z00 * (1.0 - tx) + z10 * tx
    z1 = z01 * (1.0 - tx) + z11 * tx
    return z0 * (1.0 - ty) + z1 * ty


def heightfield_mesh(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate a height grid into vertices, faces, and height-based RGB colors.

    Returns
    -------
    vertices : (V, 3) float32
    faces : (F, 3) uint32
    colors : (V, 3) uint8 lunar gray/beige by height
    """
    nx, ny = H.shape
    XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
    vertices = np.stack(
        [XX.ravel(), YY.ravel(), H.ravel()],
        axis=1,
    ).astype(np.float32)

    ii, jj = np.mgrid[0 : nx - 1, 0 : ny - 1]
    v00 = (ii * ny + jj).ravel()
    v01 = (ii * ny + jj + 1).ravel()
    v10 = ((ii + 1) * ny + jj).ravel()
    v11 = ((ii + 1) * ny + jj + 1).ravel()
    faces = np.vstack(
        [
            np.stack([v00, v10, v11], axis=1),
            np.stack([v00, v11, v01], axis=1),
        ]
    ).astype(np.uint32)

    h = H.ravel()
    h_min = float(h.min())
    h_max = float(h.max())
    t = np.zeros_like(h) if h_max <= h_min else (h - h_min) / (h_max - h_min)
    low = np.array([92.0, 90.0, 88.0])
    high = np.array([186.0, 170.0, 142.0])
    colors = (low[None, :] * (1.0 - t[:, None]) + high[None, :] * t[:, None]).astype(np.uint8)
    return vertices, faces, colors
