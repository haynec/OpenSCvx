"""Trajectory initialization utilities.

This module provides functions for generating initial trajectory guesses
using various interpolation methods. Keyframe-based APIs allow specifying
values at specific nodes with automatic interpolation for intermediate nodes.

Example:
    Generate initial guess for a quadrotor passing through gates::

        import openscvx as ox

        n = 50
        position.guess = ox.init.linspace(
            keyframes=[start_pos, gate1, gate2, end_pos],
            nodes=[0, 15, 35, n-1],
        )

        attitude.guess = ox.init.nlerp(
            keyframes=[q_start, q_gate1, q_gate2, q_end],
            nodes=[0, 15, 35, n-1],
        )
"""

from openscvx.init.interpolation import linspace, nlerp, slerp
from openscvx.init.inverse_kinematics import ik_interpolation

__all__ = [
    "ik_interpolation",
    "linspace",
    "nlerp",
    "slerp",
]
