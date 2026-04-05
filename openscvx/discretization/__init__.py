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

from typing import Annotated, Any, Union

from pydantic import Field, TypeAdapter

from .base import Discretizer, DisType
from .discretize_linearize import DiscretizeLinearizeVectorize, VectorizeDiscretizeLinearize
from .linearize_discretize import (
    LinearizeDiscretize,
    calculate_impulsive_discretization,
    get_impulsive_discretization_solver,
)
from .linearize_discretize_sparse import LinearizeDiscretizeSparse
from .sparse_utils import color_columns, make_sparse_jacobian_fns

# ---------------------------------------------------------------------------
# Discretizer config — discriminated union of each discretizer's inner Spec
# ---------------------------------------------------------------------------

DEFAULT_DISCRETIZER_TYPE = "VectorizeDiscretizeLinearize"

DiscretizerConfig = Annotated[
    Union[
        VectorizeDiscretizeLinearize.Spec,
        DiscretizeLinearizeVectorize.Spec,
        LinearizeDiscretize.Spec,
        LinearizeDiscretizeSparse.Spec,
    ],
    Field(discriminator="type"),
]

discretizer_config_adapter = TypeAdapter(DiscretizerConfig)


def resolve_discretizer_config(val: Any) -> Discretizer.Spec:
    """Validate a dict/Spec into a :class:`Discretizer.Spec` instance.

    Injects the default ``type`` (``VectorizeDiscretizeLinearize``) when the
    input dict omits it, preserving backwards compatibility.
    """
    if isinstance(val, Discretizer.Spec):
        return val
    if isinstance(val, dict) and "type" not in val:
        val = {**val, "type": DEFAULT_DISCRETIZER_TYPE}
    return discretizer_config_adapter.validate_python(val)


__all__ = [
    "DisType",
    "Discretizer",
    "DiscretizerConfig",
    "DiscretizeLinearizeVectorize",
    "LinearizeDiscretize",
    "LinearizeDiscretizeSparse",
    "VectorizeDiscretizeLinearize",
    "resolve_discretizer_config",
    "calculate_impulsive_discretization",
    "color_columns",
    "get_impulsive_discretization_solver",
    "make_sparse_jacobian_fns",
]
