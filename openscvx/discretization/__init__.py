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

from .base import Discretizer, DisType, _make_foh_mask, _resolve_foh_mask
from .discretize_linearize import DiscretizeLinearizeVectorize, VectorizeDiscretizeLinearize
from .linearize_discretize import (
    LinearizeDiscretize,
    calculate_impulsive_discretization,
    get_impulsive_discretization_solver,
)
from .linearize_discretize_sparse import LinearizeDiscretizeSparse
from .sparse_utils import color_columns, make_sparse_jacobian_fns

# ---------------------------------------------------------------------------
# Spec resolver — turn a dict into a Discretizer instance
# ---------------------------------------------------------------------------

_DISCRETIZER_MAP = {
    "DiscretizeLinearizeVectorize": DiscretizeLinearizeVectorize,
    "LinearizeDiscretize": LinearizeDiscretize,
    "LinearizeDiscretizeSparse": LinearizeDiscretizeSparse,
    "VectorizeDiscretizeLinearize": VectorizeDiscretizeLinearize,
}


def _resolve_discretizer(val: Any) -> Discretizer:
    """Resolve a discretizer specification into an instance.

    Accepted forms:

    * **instance** — already-constructed :class:`Discretizer` (pass-through).
    * **dict** — keyword arguments passed to the selected discretizer class.
      An optional ``"type"`` key selects the class (defaults to
      :class:`VectorizeDiscretizeLinearize`).

    Examples::

        # Dict with keyword overrides (default class: VectorizeDiscretizeLinearize)
        _resolve_discretizer({"ode_solver": "Dopri8"})

        # Global hold on the discretizer (or ``Control(..., parameterization="FOH"|"ZOH")``)
        _resolve_discretizer({"dis_type": "ZOH", "ode_solver": "Dopri8"})

        # Configure integrator behavior (forwarded to Diffrax / diffeqsolve)
        _resolve_discretizer(
            {"diffrax_kwargs": {"num_substeps": 100, "max_steps": 20_000}}
        )

        # Dict with explicit dense discretizer
        _resolve_discretizer({"type": "LinearizeDiscretize", "ode_solver": "Dopri8"})

        # Instance pass-through
        _resolve_discretizer(LinearizeDiscretize(ode_solver="Dopri8"))
    """
    if isinstance(val, Discretizer):
        return val

    if not isinstance(val, dict):
        raise TypeError(f"Expected a Discretizer instance or dict, got {type(val).__name__}")

    kwargs = dict(val)  # copy to avoid mutating caller's dict
    name = kwargs.pop("type", "VectorizeDiscretizeLinearize")

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
    "DisType",
    "Discretizer",
    "DiscretizeLinearizeVectorize",
    "LinearizeDiscretize",
    "LinearizeDiscretizeSparse",
    "VectorizeDiscretizeLinearize",
    "_make_foh_mask",
    "_resolve_discretizer",
    "_resolve_foh_mask",
    "calculate_impulsive_discretization",
    "color_columns",
    "get_impulsive_discretization_solver",
    "make_sparse_jacobian_fns",
]
