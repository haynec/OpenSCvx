"""Analytical Hohmann transfer solution (LEO → GEO).

This mirrors the worked example from:

- Bryan Weber, *Orbital Mechanics & Astrodynamics*,
  "Example: Hohmann Transfer"
  https://orbital-mechanics.space/orbital-maneuvers/hohmann-transfer-example.html

We use:

- mu = 3.986e5 km^3 / s^2 (Earth)
- R_E = 6378 km
- LEO altitude = 250 km
- GEO radius computed from sidereal-day period
"""

from __future__ import annotations

import math as m


def hohmann_leo_to_geo_parameters():
    """Return radii and standard gravitational parameter for LEO → GEO case."""
    mu = 3.986e5  # km^3/s^2
    R_E = 6378.0  # km

    r_leo = 250.0 + R_E
    sidereal_day = 86164.0905  # s
    r_cubed = mu * sidereal_day**2 / (4 * m.pi**2)
    r_geo = r_cubed ** (1.0 / 3.0)

    return mu, r_leo, r_geo


def compute_hohmann_delta_v_and_mass(
) -> dict:
    """Compute analytical Hohmann Δv and propellant mass for LEO → GEO."""
    mu, r_leo, r_geo = hohmann_leo_to_geo_parameters()

    # Circular velocities
    v_leo = m.sqrt(mu / r_leo)
    v_geo = m.sqrt(mu / r_geo)

    # Transfer ellipse parameters
    r_p = r_leo
    r_a = r_geo
    h_t = m.sqrt(2 * mu * r_a * r_p / (r_a + r_p))
    v_tp = h_t / r_p
    v_ta = h_t / r_a

    # Total Hohmann Δv
    delta_v = abs(v_geo - v_ta) + abs(v_tp - v_leo)

    return {
        "delta_v": delta_v,
    }