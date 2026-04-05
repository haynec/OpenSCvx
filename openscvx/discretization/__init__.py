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

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

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
# Spec resolver — turn a dict into a Discretizer instance
# ---------------------------------------------------------------------------

_DISCRETIZER_MAP = {
    "DiscretizeLinearizeVectorize": DiscretizeLinearizeVectorize,
    "LinearizeDiscretize": LinearizeDiscretize,
    "LinearizeDiscretizeSparse": LinearizeDiscretizeSparse,
    "VectorizeDiscretizeLinearize": VectorizeDiscretizeLinearize,
}


class DiscretizerConfig(BaseModel):
    """Validates discretizer configuration from dict input.

    An optional ``type`` key selects the discretizer class (defaults to
    ``VectorizeDiscretizeLinearize``).  Remaining fields are forwarded as
    keyword arguments to the selected class.
    """

    type: str = "VectorizeDiscretizeLinearize"
    dis_type: Union[str, List[str]] = "FOH"
    ode_solver: str = "Tsit5"
    custom_integrator: bool = False
    diffrax_kwargs: Optional[Dict[str, Any]] = None
    args: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")

    def to_discretizer(self) -> Discretizer:
        cls = _DISCRETIZER_MAP.get(self.type)
        if cls is None:
            raise ValueError(
                f"Unknown discretizer {self.type!r}; "
                f"expected one of {sorted(_DISCRETIZER_MAP)}"
            )
        # Only forward explicitly-set fields so constructors with fewer
        # parameters don't receive unexpected keyword arguments.
        kwargs = self.model_dump(exclude={"type"}, exclude_defaults=True)
        return cls(**kwargs)


__all__ = [
    "DisType",
    "Discretizer",
    "DiscretizerConfig",
    "DiscretizeLinearizeVectorize",
    "LinearizeDiscretize",
    "LinearizeDiscretizeSparse",
    "VectorizeDiscretizeLinearize",
    "calculate_impulsive_discretization",
    "color_columns",
    "get_impulsive_discretization_solver",
    "make_sparse_jacobian_fns",
]
