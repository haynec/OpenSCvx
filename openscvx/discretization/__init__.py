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

:class:`Problem` uses :class:`LinearizeDiscretizeSparse` by default (sparse
Jacobians and compact variational integration when sparsity patterns exist).
:class:`LinearizeDiscretize` is the dense linearize-then-discretize scheme.
"""

import inspect
from typing import Any

from openscvx.sparse import color_columns, make_sparse_jacobian_fns

from .base import Discretizer
from .linearize_discretize import (
    LinearizeDiscretize,
    calculate_impulsive_discretization,
    get_impulsive_discretization_solver,
)
from .linearize_discretize_sparse import LinearizeDiscretizeSparse

# ---------------------------------------------------------------------------
# Spec resolver — turn a dict into a Discretizer instance
# ---------------------------------------------------------------------------

_DISCRETIZER_MAP = {
    "LinearizeDiscretize": LinearizeDiscretize,
    "LinearizeDiscretizeSparse": LinearizeDiscretizeSparse,
}


def _resolve_discretizer(val: Any) -> Discretizer:
    """Resolve a discretizer specification into an instance.

    Accepted forms:

    * **instance** — already-constructed :class:`Discretizer` (pass-through).
    * **dict** — keyword arguments passed to the selected discretizer class.
      An optional ``"type"`` key selects the class (defaults to
      :class:`LinearizeDiscretizeSparse`).

    Examples::

        # Dict with keyword overrides (default class: LinearizeDiscretizeSparse)
        _resolve_discretizer({"dis_type": "ZOH", "ode_solver": "Dopri8"})

        # Dict with explicit dense discretizer
        _resolve_discretizer({"type": "LinearizeDiscretize", "dis_type": "ZOH"})

        # Instance pass-through
        _resolve_discretizer(LinearizeDiscretize(dis_type="ZOH"))
    """
    if isinstance(val, Discretizer):
        return val

    if not isinstance(val, dict):
        raise TypeError(f"Expected a Discretizer instance or dict, got {type(val).__name__}")

    kwargs = dict(val)  # copy to avoid mutating caller's dict
    name = kwargs.pop("type", "LinearizeDiscretizeSparse")

    cls = _DISCRETIZER_MAP.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown discretizer {name!r}; expected one of {sorted(_DISCRETIZER_MAP)}"
        )

    try:
        return cls(**kwargs)
    except TypeError as e:
        valid = list(inspect.signature(cls.__init__).parameters.keys())
        valid.remove("self")
        raise TypeError(f"Invalid discretizer keyword argument: {e}. Valid keys: {valid}") from None


__all__ = [
    "Discretizer",
    "LinearizeDiscretize",
    "LinearizeDiscretizeSparse",
    "_resolve_discretizer",
    "calculate_impulsive_discretization",
    "color_columns",
    "get_impulsive_discretization_solver",
    "make_sparse_jacobian_fns",
]
