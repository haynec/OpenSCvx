"""Discretization methods for trajectory optimization.

This module provides implementations of discretization schemes that convert
continuous-time optimal control problems into discrete-time approximations
suitable for numerical optimization.

Discretization and linearization are combined into a single interface
(:class:`Discretizer`) because different schemes may linearize then discretize,
discretize then linearize, or use other approaches. The ordering changes the
intermediate types, but the input (continuous nonlinear dynamics + reference
trajectory) and output (discrete-time linear matrices A_d, B_d, C_d) are
always consistent.

The default implementation is :class:`MultiShootDiscretizer`, which computes
continuous-time Jacobians via JAX autodiff and integrates them alongside the
nonlinear dynamics through an augmented state vector.
"""

from .base import Discretizer
from .discretization import (
    MultiShootDiscretizer,
    calculate_discretization,
    dVdt,
    get_discretization_solver,
)

__all__ = [
    "Discretizer",
    "MultiShootDiscretizer",
    "calculate_discretization",
    "get_discretization_solver",
    "dVdt",
]
