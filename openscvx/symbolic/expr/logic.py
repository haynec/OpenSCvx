"""Logical and control flow operations for symbolic expressions.

This module provides logical and control flow operations used in optimization problems,
enabling conditional logic in dynamics and constraints. These operations are
JAX-only and not supported in CVXPy lowering.

Operations:
    - **Conditional:** `Cond` - Conditional expression using jax.lax.cond for
        JAX-traceable branching. Predicate must be an Inequality constraint.
        Optionally restricted to specific node ranges.

Example:
    Using conditional logic in constraints::

        import openscvx as ox

        position = ox.State("position", shape=(3,))
        obstacle = ox.Parameter("obstacle", shape=(3,))

        # Conditional max speed: slow down when close to obstacle
        distance = ox.Norm(position - obstacle)
        max_speed = ox.Cond(
            distance <= 2.0,  # predicate: True when close to obstacle
            5.0,              # true branch: reduced speed limit
            10.0              # false branch: normal speed limit
        )

    Using node_ranges to restrict when the conditional is active::

        # Only apply conditional logic during nodes 1-3 and 5-7
        # Outside these ranges, the false branch is always used
        max_speed = ox.Cond(
            distance <= 2.0,
            5.0,
            10.0,
            node_ranges=[(1, 3), (5, 7)]
        )
"""

from typing import List, Optional, Tuple

import numpy as np

from .constraint import Inequality
from .expr import Expr, to_expr


class Cond(Expr):
    """Conditional expression for JAX-traceable branching.

    Implements a conditional expression that selects between two branches based
    on an Inequality predicate. This wraps `jax.lax.cond` to enable conditional
    logic in symbolic expressions for dynamics and constraints.

    The predicate must be an Inequality constraint (created with `<=` or `>=`).
    After canonicalization, the constraint is in the form `lhs <= 0`, so the
    predicate evaluates to True when the constraint is satisfied (lhs <= 0) and
    False when violated (lhs > 0).

    The true and false branches must have broadcastable shapes (following
    JAX/NumPy broadcasting rules).

    Optionally, the conditional can be restricted to specific node ranges using
    the `node_ranges` parameter. Outside these ranges, the false branch is
    always evaluated.

    Attributes:
        pred: Inequality constraint used as predicate. True when satisfied.
        true_branch: Expression to evaluate when predicate is True
        false_branch: Expression to evaluate when predicate is False
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
        pred,
        true_branch,
        false_branch,
        node_ranges: Optional[List[Tuple[int, int]]] = None,
    ):
        """Initialize a conditional expression.

        Args:
            pred: Inequality constraint used as the predicate (e.g., x <= 5, y >= 0).
                After canonicalization, the constraint is in the form (lhs <= 0),
                so the predicate is True when the constraint is satisfied.
            true_branch: Expression to evaluate when predicate is True
            false_branch: Expression to evaluate when predicate is False
            node_ranges: Optional list of (start, end) tuples specifying node ranges
                where the conditional is active. Each tuple defines a half-open
                interval [start, end) of node indices. Outside these ranges, the
                false branch is always evaluated. None means active at all nodes.

        Raises:
            TypeError: If pred is not an Inequality constraint
            ValueError: If node_ranges contains invalid ranges
        """
        if not isinstance(pred, Inequality):
            raise TypeError(
                f"Cond predicate must be an Inequality constraint (e.g., x <= 5, y >= 0), "
                f"got {type(pred).__name__}. Use comparison operators like '<=' or '>=' "
                f"to create a valid predicate."
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

        self.pred = pred
        self.true_branch = to_expr(true_branch)
        self.false_branch = to_expr(false_branch)
        self.node_ranges = node_ranges

    def children(self):
        """Return the child expressions: predicate, true branch, and false branch."""
        return [self.pred, self.true_branch, self.false_branch]

    def canonicalize(self) -> "Expr":
        """Canonicalize by canonicalizing all three children, preserving node_ranges."""
        pred = self.pred.canonicalize()
        true_branch = self.true_branch.canonicalize()
        false_branch = self.false_branch.canonicalize()
        return Cond(pred, true_branch, false_branch, node_ranges=self.node_ranges)

    def check_shape(self) -> Tuple[int, ...]:
        """Check and return the output shape of the conditional.

        The predicate must be scalar, and the true and false branches must have
        broadcastable shapes. The output shape is the broadcasted shape of the
        two branches.

        Returns:
            tuple: The broadcasted shape of true_branch and false_branch

        Raises:
            ValueError: If predicate is not scalar or branches have incompatible shapes
        """
        pred_shape = self.pred.check_shape()
        true_shape = self.true_branch.check_shape()
        false_shape = self.false_branch.check_shape()

        # Predicate must be scalar
        if pred_shape != ():
            raise ValueError(f"Cond predicate must be scalar, got shape {pred_shape}")

        # True and false branches must be broadcastable
        try:
            return np.broadcast_shapes(true_shape, false_shape)
        except ValueError as e:
            raise ValueError(
                f"Cond branches have incompatible shapes: {true_shape} and {false_shape}"
            ) from e

    def __repr__(self):
        """Return string representation of the conditional."""
        base = f"cond({self.pred!r}, {self.true_branch!r}, {self.false_branch!r}"
        if self.node_ranges is not None:
            return f"{base}, node_ranges={self.node_ranges!r})"
        return f"{base})"
