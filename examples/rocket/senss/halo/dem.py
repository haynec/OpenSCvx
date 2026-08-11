"""Digital Elevation Model (DEM) abstraction and loaders.

A :class:`DEM` is a metric, raster elevation grid: a ``(H, W)`` array of heights
in meters anchored at a south-west corner ``(x0, y0)`` with a square cell size
``res`` (meters per pixel). Optionally it also carries a per-cell elevation
``variance`` array, which makes it a *Gaussian DEM* in the sense of Tomita & Ho
(2025) - each cell stores the mean and variance of the local elevation.

The class is registered as a JAX pytree so a ``DEM`` can be passed through
``jit``/``vmap`` and its :meth:`DEM.sample` method is differentiable and
batch-safe.

Loaders
-------
* :meth:`DEM.from_png`  - grayscale heightmap (default: bundled ``senns_dem.png``).
* :meth:`DEM.from_array` - any elevation array.
* :meth:`DEM.synthetic_rocky` - Tomita-Ho testbed (flat plane + random rocks),
  optionally superimposed on a base DEM via the complexity factor ``c``
  (paper Eq. 20, ``D_c = D_rock + c * D_terrain``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Path to the bundled default DEM (the "SENSS" grayscale heightmap).
DEFAULT_DEM_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "senns_dem.png")


@dataclass(frozen=True)
class GridSpec:
    """Metric raster grid definition shared by DEMs and safety maps.

    Attributes:
        nx: Number of columns (x / east cells).
        ny: Number of rows (y / north cells).
        res: Cell size in meters per pixel (square cells).
        x0: World x-coordinate (m) of the SW corner cell center.
        y0: World y-coordinate (m) of the SW corner cell center.
    """

    nx: int
    ny: int
    res: float
    x0: float = 0.0
    y0: float = 0.0

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) in meters, cell-center referenced."""
        return (
            self.x0,
            self.x0 + (self.nx - 1) * self.res,
            self.y0,
            self.y0 + (self.ny - 1) * self.res,
        )

    def cell_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(xs, ys)`` 1D arrays of column/row cell-center coordinates."""
        xs = self.x0 + np.arange(self.nx) * self.res
        ys = self.y0 + np.arange(self.ny) * self.res
        return xs, ys

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(XX, YY)`` (ny, nx) meshgrids of cell-center coordinates."""
        xs, ys = self.cell_centers()
        return np.meshgrid(xs, ys)  # XX, YY with shape (ny, nx)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DEM:
    """A metric elevation grid, optionally with per-cell variance.

    The array is indexed ``heights[row, col]`` where ``row`` increases with
    world ``y`` (north) and ``col`` increases with world ``x`` (east). Cell
    ``(row, col)`` is centered at world ``(x0 + col*res, y0 + row*res)``.
    """

    heights: jnp.ndarray  # (ny, nx) elevation in meters
    res: float
    x0: float = 0.0
    y0: float = 0.0
    variance: Optional[jnp.ndarray] = None  # (ny, nx) elevation variance (m^2)

    # -- pytree protocol -----------------------------------------------------
    def tree_flatten(self):
        children = (self.heights, self.variance)
        aux = (self.res, self.x0, self.y0)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        heights, variance = children
        res, x0, y0 = aux
        return cls(heights=heights, res=res, x0=x0, y0=y0, variance=variance)

    # -- geometry ------------------------------------------------------------
    @property
    def ny(self) -> int:
        return int(self.heights.shape[0])

    @property
    def nx(self) -> int:
        return int(self.heights.shape[1])

    @property
    def grid(self) -> GridSpec:
        return GridSpec(nx=self.nx, ny=self.ny, res=self.res, x0=self.x0, y0=self.y0)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        return self.grid.extent

    def with_variance(self, variance: jnp.ndarray) -> "DEM":
        return replace(self, variance=variance)

    # -- sampling ------------------------------------------------------------
    def sample(self, xy: jnp.ndarray) -> jnp.ndarray:
        """Bilinearly sample elevation at world coordinates ``xy``.

        Args:
            xy: ``(..., 2)`` array of ``(x, y)`` world coordinates in meters.

        Returns:
            Elevation(s) with shape ``xy.shape[:-1]``. Coordinates outside the
            grid are clamped to the border (edge-extend).
        """
        return _bilinear_sample(self.heights, self.res, self.x0, self.y0, xy)

    def sample_variance(self, xy: jnp.ndarray) -> jnp.ndarray:
        """Bilinearly sample the variance grid (zeros if this DEM has none)."""
        var = self.variance
        if var is None:
            var = jnp.zeros_like(self.heights)
        return _bilinear_sample(var, self.res, self.x0, self.y0, xy)

    # -- loaders -------------------------------------------------------------
    @classmethod
    def from_array(
        cls,
        heights,
        res: float,
        x0: float = 0.0,
        y0: float = 0.0,
        variance=None,
    ) -> "DEM":
        """Build a DEM directly from a ``(ny, nx)`` elevation array (meters)."""
        heights = jnp.asarray(heights, dtype=jnp.float64)
        if variance is not None:
            variance = jnp.asarray(variance, dtype=jnp.float64)
        return cls(heights=heights, res=float(res), x0=float(x0), y0=float(y0), variance=variance)

    @classmethod
    def from_png(
        cls,
        path: str = DEFAULT_DEM_PATH,
        size_m: float = 100.0,
        height_m: float = 3.0,
        res: float = 0.1,
        x0: Optional[float] = None,
        y0: Optional[float] = None,
        lo_pct: float = 1.0,
        hi_pct: float = 99.0,
    ) -> "DEM":
        """Load a grayscale heightmap PNG as a metric DEM.

        The image is normalized robustly (``lo_pct``/``hi_pct`` percentiles ->
        ``[0, 1]``), resized to ``round(size_m / res)`` pixels per side, and
        scaled so its full relief spans ``height_m`` meters over a
        ``size_m x size_m`` footprint. The DEM is centered on the origin unless
        ``x0``/``y0`` are given.

        Args:
            path: Path to the grayscale PNG (default: bundled ``senns_dem.png``).
            size_m: Physical side length of the (square) footprint in meters.
            height_m: Peak-to-trough relief in meters after normalization.
            res: Target resolution in meters per pixel.
            x0, y0: SW-corner world coordinates; default centers the footprint.
            lo_pct, hi_pct: Percentiles for robust min/max normalization.
        """
        from PIL import Image  # local import: keep Pillow off the hot path

        n = int(round(size_m / res))
        img = Image.open(path).convert("I")  # 32-bit integer grayscale
        raw = np.asarray(img, dtype=np.float64)
        lo = np.percentile(raw, lo_pct)
        hi = np.percentile(raw, hi_pct)
        img_r = img.resize((n, n), Image.BILINEAR)
        arr = np.asarray(img_r, dtype=np.float64)
        norm = np.clip((arr - lo) / max(hi - lo, 1.0), 0.0, 1.0)
        heights = norm * float(height_m)
        # PIL row 0 is the top of the image; flip so row index increases with +y.
        heights = np.flipud(heights)
        if x0 is None:
            x0 = -0.5 * (n - 1) * res
        if y0 is None:
            y0 = -0.5 * (n - 1) * res
        return cls.from_array(heights, res=res, x0=x0, y0=y0)

    @classmethod
    def synthetic_rocky(
        cls,
        size_m: float = 100.0,
        res: float = 0.1,
        n_rocks: int = 500,
        rock_diam: float = 1.0,
        seed: int = 0,
        base: Optional["DEM"] = None,
        c: float = 0.0,
        x0: Optional[float] = None,
        y0: Optional[float] = None,
    ) -> "DEM":
        """Generate a Tomita-Ho rocky testbed DEM (paper Sec. IV.B.1).

        Places ``n_rocks`` non-overlapping hemi-ellipsoid rocks of diameter
        ``rock_diam`` (height = half the semi-major axis) at random locations on
        a flat plane. When ``base`` is given, the terrains are superimposed with
        complexity factor ``c`` following Eq. (20): ``D_c = D_rock + c * D_base``.

        Args:
            size_m: Side length of the square footprint in meters.
            res: Resolution in meters per pixel.
            n_rocks: Number of rocks to place.
            rock_diam: Rock diameter in meters.
            seed: RNG seed for reproducibility.
            base: Optional base DEM to superimpose (resampled to this grid).
            c: Terrain complexity factor for the superposition.
            x0, y0: SW-corner world coordinates; default centers the footprint.
        """
        rng = np.random.default_rng(seed)
        n = int(round(size_m / res))
        if x0 is None:
            x0 = -0.5 * (n - 1) * res
        if y0 is None:
            y0 = -0.5 * (n - 1) * res

        heights = np.zeros((n, n), dtype=np.float64)
        xs = x0 + np.arange(n) * res
        ys = y0 + np.arange(n) * res
        XX, YY = np.meshgrid(xs, ys)

        a = 0.5 * rock_diam  # semi-major axis (horizontal radius)
        half = 0.5 * (n - 1) * res
        margin = a
        centers = []
        attempts = 0
        max_attempts = 50 * n_rocks
        while len(centers) < n_rocks and attempts < max_attempts:
            attempts += 1
            cx = rng.uniform(x0 + margin, x0 + (n - 1) * res - margin)
            cy = rng.uniform(y0 + margin, y0 + (n - 1) * res - margin)
            if all((cx - ox) ** 2 + (cy - oy) ** 2 >= (2 * a) ** 2 for ox, oy in centers):
                centers.append((cx, cy))

        rock_height = 0.5 * a  # height = half the semi-major axis (paper)
        for cx, cy in centers:
            d2 = (XX - cx) ** 2 + (YY - cy) ** 2
            inside = d2 <= a**2
            z = np.zeros_like(heights)
            z[inside] = rock_height * np.sqrt(np.clip(1.0 - d2[inside] / a**2, 0.0, 1.0))
            heights = np.maximum(heights, z)

        if base is not None and c != 0.0:
            # Resample the base DEM onto this grid and superimpose (Eq. 20).
            base_h = np.asarray(base.sample(jnp.stack([XX, YY], axis=-1)))
            base_h = np.nan_to_num(base_h, nan=0.0)
            heights = heights + c * base_h

        return cls.from_array(heights, res=res, x0=float(x0), y0=float(y0))


def _bilinear_sample(
    grid: jnp.ndarray, res: float, x0: float, y0: float, xy: jnp.ndarray
) -> jnp.ndarray:
    """Bilinearly sample ``grid[row, col]`` at world coordinates ``xy``.

    Row index increases with world ``y``; column index increases with world
    ``x``. Out-of-bounds coordinates are clamped to the border.
    """
    xy = jnp.asarray(xy)
    ny, nx = grid.shape
    fx = (xy[..., 0] - x0) / res  # fractional column
    fy = (xy[..., 1] - y0) / res  # fractional row

    fx = jnp.clip(fx, 0.0, nx - 1.0)
    fy = jnp.clip(fy, 0.0, ny - 1.0)

    x0i = jnp.floor(fx).astype(jnp.int32)
    y0i = jnp.floor(fy).astype(jnp.int32)
    x1i = jnp.minimum(x0i + 1, nx - 1)
    y1i = jnp.minimum(y0i + 1, ny - 1)

    dx = fx - x0i
    dy = fy - y0i

    g00 = grid[y0i, x0i]
    g01 = grid[y0i, x1i]
    g10 = grid[y1i, x0i]
    g11 = grid[y1i, x1i]

    return (
        g00 * (1 - dx) * (1 - dy)
        + g01 * dx * (1 - dy)
        + g10 * (1 - dx) * dy
        + g11 * dx * dy
    )
