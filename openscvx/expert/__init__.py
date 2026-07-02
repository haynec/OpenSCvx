"""Expert-mode features for advanced users.

This module contains features for expert users who need fine-grained control
and are willing to bypass higher-level abstractions.
"""

from openscvx.expert.byof import (
    ByofSpec,
    ConvexCostSpec,
    CtcsConstraintSpec,
    NodalConstraintSpec,
    PenaltyFunction,
)
from openscvx.expert.lowering import apply_byof
from openscvx.expert.validation import validate_byof

__all__ = [
    "ByofSpec",
    "ConvexCostSpec",
    "CtcsConstraintSpec",
    "NodalConstraintSpec",
    "PenaltyFunction",
    "apply_byof",
    "validate_byof",
]
