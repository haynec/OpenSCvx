"""Minimal HALO exports used by the SENSS hop LiDAR visualization.

The hop examples only need the DEM + body-fixed LiDAR scan helpers for
post-process viser overlays. Broader HALO planning / perception modules are
intentionally not exported here.
"""

from .dem import DEM, GridSpec
from .lidar import simulate_scan, simulate_scan_body

__all__ = [
    "DEM",
    "GridSpec",
    "simulate_scan",
    "simulate_scan_body",
]
