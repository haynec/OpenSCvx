"""Logical and control flow operations for symbolic expressions.

This module provides logical and control flow operations used in optimization problems,
enabling conditional logic in dynamics and constraints. These operations are
JAX-only and not supported in CVXPy lowering.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from .constraint import Inequality
from .expr import Expr, to_expr


class Cond(Expr):
    """Conditional expression for JAX-traceable branching.

    Implements a conditional expression that selects between two branches based
    on one or more Inequality predicates. This wraps `jax.lax.cond` to enable
    conditional logic in symbolic expressions for dynamics and constraints.

    The predicate can be either:
    - A single Inequality constraint (created with `<=` or `>=`)
    - A list of Inequality constraints (AND semantics: all must be satisfied)

    After canonicalization, each constraint is in the form `lhs <= 0`, so the
    predicate evaluates to True when the constraint is satisfied (lhs <= 0) and
    False when violated (lhs > 0). For multiple predicates, all must be satisfied.

    The true and false branches must have broadcastable shapes (following
    JAX/NumPy broadcasting rules).

    Optionally, the conditional can be restricted to specific node ranges using
    the `node_ranges` parameter. Outside these ranges, the false branch is
    always evaluated.

    Attributes:
        predicates: List of Inequality constraints used as predicates (AND semantics).
        true_branch: Expression to evaluate when all predicates are True
        false_branch: Expression to evaluate when any predicate is False
        node_ranges: Optional list of (start, end) tuples specifying node ranges
            where the conditional is active. None means active at all nodes.

    Example:
        Conditional velocity limit based on distance::

            distance = ox.Norm(position - obstacle)
            expr = ox.Cond(
                distance <= safety_threshold,  # predicate: True when close
                5.0,                           # true branch: slow speed
                10.0                           # false branch: fast speed
            )

        Multiple predicates with AND semantics::

            expr = ox.Cond(
                [x >= 0.0, x <= 10.0],  # True when x in [0, 10]
                1.0,                     # in range
                0.0                      # out of range
            )

        Conditional active only during specific trajectory phases::

            expr = ox.Cond(
                distance <= safety_threshold,
                5.0,
                10.0,
                node_ranges=[(0, 2), (5, 7)]  # active at nodes 0-1 and 5-6
            )

    Note:
        This operation is only supported for JAX lowering. CVXPy lowering will
        raise NotImplementedError since conditional logic is not DCP-compliant.
    """

    def __init__(
        self,
        pred: Union[Inequality, List[Inequality]],
        true_branch,
        false_branch,
        node_ranges: Optional[List[Tuple[int, int]]] = None,
    ):
        """Initialize a conditional expression.

        Args:
            pred: Inequality constraint or list of Inequality constraints used as
                the predicate(s). For a single constraint (e.g., x <= 5, y >= 0),
                the predicate is True when the constraint is satisfied. For a list
                of constraints, all must be satisfied (AND semantics).
                After canonicalization, each constraint is in the form (lhs <= 0).
            true_branch: Expression to evaluate when all predicates are True
            false_branch: Expression to evaluate when any predicate is False
            node_ranges: Optional list of (start, end) tuples specifying node ranges
                where the conditional is active. Each tuple defines a half-open
                interval [start, end) of node indices. Outside these ranges, the
                false branch is always evaluated. None means active at all nodes.

        Raises:
            TypeError: If pred is not an Inequality or list of Inequalities
            ValueError: If node_ranges contains invalid ranges
        """
        # Normalize pred to a list
        if isinstance(pred, Inequality):
            predicates = [pred]
        elif isinstance(pred, list):
            if len(pred) == 0:
                raise ValueError("Cond predicate list cannot be empty")
            for i, p in enumerate(pred):
                if not isinstance(p, Inequality):
                    raise TypeError(
                        f"Cond predicate[{i}] must be an Inequality constraint "
                        f"(e.g., x <= 5, y >= 0), got {type(p).__name__}."
                    )
            predicates = pred
        else:
            raise TypeError(
                f"Cond predicate must be an Inequality constraint or list of Inequalities "
                f"(e.g., x <= 5, [x >= 0, x <= 10]), got {type(pred).__name__}."
            )

        # Validate node_ranges
        if node_ranges is not None:
            if not isinstance(node_ranges, list):
                raise TypeError("node_ranges must be a list of (start, end) tuples")
            for i, r in enumerate(node_ranges):
                if not isinstance(r, tuple) or len(r) != 2:
                    raise ValueError(f"node_ranges[{i}] must be a (start, end) tuple, got {r!r}")
                start, end = r
                if not isinstance(start, int) or not isinstance(end, int):
                    start_type = type(start).__name__
                    end_type = type(end).__name__
                    raise ValueError(
                        f"node_ranges[{i}] must contain integers, got ({start_type}, {end_type})"
                    )
                if start >= end:
                    raise ValueError(
                        f"node_ranges[{i}] must have start < end, got ({start}, {end})"
                    )

        self.predicates = predicates
        self.true_branch = to_expr(true_branch)
        self.false_branch = to_expr(false_branch)
        self.node_ranges = node_ranges

    def children(self):
        """Return the child expressions: predicates, true branch, and false branch."""
        return [*self.predicates, self.true_branch, self.false_branch]

    def canonicalize(self) -> "Expr":
        """Canonicalize by canonicalizing all children, preserving node_ranges."""
        predicates = [p.canonicalize() for p in self.predicates]
        true_branch = self.true_branch.canonicalize()
        false_branch = self.false_branch.canonicalize()
        return Cond(predicates, true_branch, false_branch, node_ranges=self.node_ranges)

    def check_shape(self) -> Tuple[int, ...]:
        """Check and return the output shape of the conditional.

        All predicates must be scalar, and the true and false branches must have
        broadcastable shapes. The output shape is the broadcasted shape of the
        two branches.

        Returns:
            tuple: The broadcasted shape of true_branch and false_branch

        Raises:
            ValueError: If any predicate is not scalar or branches have incompatible shapes
        """
        # All predicates must be scalar
        for i, pred in enumerate(self.predicates):
            pred_shape = pred.check_shape()
            if pred_shape != ():
                if len(self.predicates) == 1:
                    raise ValueError(f"Cond predicate must be scalar, got shape {pred_shape}")
                else:
                    raise ValueError(
                        f"Cond predicate[{i}] must be scalar, got shape {pred_shape}"
                    )

        true_shape = self.true_branch.check_shape()
        false_shape = self.false_branch.check_shape()

        # True and false branches must be broadcastable
        try:
            return np.broadcast_shapes(true_shape, false_shape)
        except ValueError as e:
            raise ValueError(
                f"Cond branches have incompatible shapes: {true_shape} and {false_shape}"
            ) from e

    def __repr__(self):
        """Return string representation of the conditional."""
        if len(self.predicates) == 1:
            pred_repr = repr(self.predicates[0])
        else:
            pred_repr = repr(self.predicates)
        base = f"cond({pred_repr}, {self.true_branch!r}, {self.false_branch!r}"
        if self.node_ranges is not None:
            return f"{base}, node_ranges={self.node_ranges!r})"
        return f"{base})"
