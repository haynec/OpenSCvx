"""Logical and control flow operations for symbolic expressions.

This module provides logical and control flow operations used in optimization problems,
enabling conditional logic in dynamics and constraints. These operations are
JAX-only and not supported in CVXPy lowering.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from .constraint import Inequality
from .expr import Expr, to_expr


class All(Expr):
    """Logical AND reduction over predicates. Wraps jnp.all.

    Reduces one or more Inequality predicates to a single scalar boolean using
    AND semantics. This is useful for:

    1. Combining multiple scalar predicates: ``All([x >= 0, x <= 10])``
    2. Reducing a vector predicate: ``All(position >= lower_bound)``

    After evaluation, returns True only if ALL predicates are satisfied.

    Attributes:
        predicates: List of Inequality constraints to combine with AND.

    Example:
        Combining scalar predicates::

            in_range = ox.All([x >= 0.0, x <= 10.0])
            ox.Cond(in_range, 1.0, 0.0)

        Reducing a vector predicate::

            all_positive = ox.All(position >= 0.0)  # position is shape (3,)
            ox.Cond(all_positive, safe_value, unsafe_value)

    Note:
        This operation is only supported for JAX lowering. CVXPy lowering will
        raise NotImplementedError since logical reductions are not DCP-compliant.
    """

    def __init__(self, predicates: Union[Inequality, List[Inequality]]):
        """Initialize an All expression.

        Args:
            predicates: Single Inequality or list of Inequalities to combine.
                For a single vector Inequality, reduces across all elements.
                For a list, combines all predicates with AND.

        Raises:
            TypeError: If predicates is not an Inequality or list of Inequalities
            ValueError: If predicates list is empty
        """
        if isinstance(predicates, Inequality):
            self.predicates = [predicates]
        elif isinstance(predicates, list):
            if len(predicates) == 0:
                raise ValueError("All predicate list cannot be empty")
            for i, p in enumerate(predicates):
                if not isinstance(p, Inequality):
                    raise TypeError(
                        f"All predicate[{i}] must be an Inequality constraint "
                        f"(e.g., x <= 5, y >= 0), got {type(p).__name__}."
                    )
            self.predicates = predicates
        else:
            raise TypeError(
                f"All predicates must be an Inequality or list of Inequalities "
                f"(e.g., x <= 5, [x >= 0, x <= 10]), got {type(predicates).__name__}."
            )

    def children(self):
        """Return the child expressions (all predicates)."""
        return list(self.predicates)

    def canonicalize(self) -> "Expr":
        """Canonicalize by canonicalizing all predicates."""
        return All([p.canonicalize() for p in self.predicates])

    def check_shape(self) -> Tuple[int, ...]:
        """Check shape and return scalar output shape.

        All always reduces to a scalar boolean.

        Returns:
            tuple: Empty tuple () representing scalar output
        """
        # Just validate that predicates have valid shapes
        for pred in self.predicates:
            pred.check_shape()
        return ()

    def __repr__(self) -> str:
        """Return string representation."""
        if len(self.predicates) == 1:
            return f"All({self.predicates[0]!r})"
        return f"All({self.predicates!r})"


class Any(Expr):
    """Logical OR reduction over predicates. Wraps jnp.any.

    Reduces one or more Inequality predicates to a single scalar boolean using
    OR semantics. This is useful for:

    1. Combining multiple scalar predicates: ``Any([in_region_a, in_region_b])``
    2. Reducing a vector predicate: ``Any(position >= threshold)``

    After evaluation, returns True if ANY predicate is satisfied.

    Attributes:
        predicates: List of Inequality constraints to combine with OR.

    Example:
        Combining scalar predicates (OR logic)::

            in_any_region = ox.Any([in_region_a, in_region_b])
            ox.Cond(in_any_region, region_value, default_value)

        Reducing a vector predicate::

            any_above = ox.Any(position >= threshold)  # position is shape (3,)
            ox.Cond(any_above, triggered_value, normal_value)

    Note:
        This operation is only supported for JAX lowering. CVXPy lowering will
        raise NotImplementedError since logical reductions are not DCP-compliant.
    """

    def __init__(self, predicates: Union[Inequality, List[Inequality]]):
        """Initialize an Any expression.

        Args:
            predicates: Single Inequality or list of Inequalities to combine.
                For a single vector Inequality, reduces across all elements.
                For a list, combines all predicates with OR.

        Raises:
            TypeError: If predicates is not an Inequality or list of Inequalities
            ValueError: If predicates list is empty
        """
        if isinstance(predicates, Inequality):
            self.predicates = [predicates]
        elif isinstance(predicates, list):
            if len(predicates) == 0:
                raise ValueError("Any predicate list cannot be empty")
            for i, p in enumerate(predicates):
                if not isinstance(p, Inequality):
                    raise TypeError(
                        f"Any predicate[{i}] must be an Inequality constraint "
                        f"(e.g., x <= 5, y >= 0), got {type(p).__name__}."
                    )
            self.predicates = predicates
        else:
            raise TypeError(
                f"Any predicates must be an Inequality or list of Inequalities "
                f"(e.g., x <= 5, [x >= 0, x <= 10]), got {type(predicates).__name__}."
            )

    def children(self):
        """Return the child expressions (all predicates)."""
        return list(self.predicates)

    def canonicalize(self) -> "Expr":
        """Canonicalize by canonicalizing all predicates."""
        return Any([p.canonicalize() for p in self.predicates])

    def check_shape(self) -> Tuple[int, ...]:
        """Check shape and return scalar output shape.

        Any always reduces to a scalar boolean.

        Returns:
            tuple: Empty tuple () representing scalar output
        """
        # Just validate that predicates have valid shapes
        for pred in self.predicates:
            pred.check_shape()
        return ()

    def __repr__(self) -> str:
        """Return string representation."""
        if len(self.predicates) == 1:
            return f"Any({self.predicates[0]!r})"
        return f"Any({self.predicates!r})"


class Cond(Expr):
    """Conditional expression for JAX-traceable branching.

    Implements a conditional expression that selects between two branches based
    on a predicate. This wraps `jax.lax.cond` to enable conditional logic in
    symbolic expressions for dynamics and constraints.

    The predicate can be:
    - A single Inequality constraint (created with `<=` or `>=`)
    - A list of Inequality constraints (AND semantics, shorthand for ``All([...])``)
    - An ``All`` expression for explicit AND semantics
    - An ``Any`` expression for OR semantics
    - ``None`` for purely node-based switching (requires ``node_ranges``)

    After canonicalization, each constraint is in the form `lhs <= 0`, so the
    predicate evaluates to True when the constraint is satisfied (lhs <= 0) and
    False when violated (lhs > 0).

    The true and false branches must have broadcastable shapes (following
    JAX/NumPy broadcasting rules).

    Optionally, the conditional can be restricted to specific node ranges using
    the `node_ranges` parameter. Outside these ranges, the false branch is
    always evaluated.

    Attributes:
        predicate: The predicate expression (All, Any, or single Inequality).
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

        Multiple predicates with AND semantics (explicit)::

            expr = ox.Cond(
                ox.All([x >= 0.0, x <= 10.0]),  # True when x in [0, 10]
                1.0,                             # in range
                0.0                              # out of range
            )

        Multiple predicates with OR semantics::

            expr = ox.Cond(
                ox.Any([in_region_a, in_region_b]),  # True if in either region
                region_value,
                default_value
            )

        Reduce vector predicate::

            expr = ox.Cond(
                ox.All(position >= lower_bound),  # True if all elements satisfy
                safe_value,
                unsafe_value
            )

        Conditional active only during specific trajectory phases::

            expr = ox.Cond(
                distance <= safety_threshold,
                5.0,
                10.0,
                node_ranges=[(0, 2), (5, 7)]  # active at nodes 0-1 and 5-6
            )

        Purely node-based switching (no predicate)::

            expr = ox.Cond(
                None,                          # no predicate
                boost_thrust,                  # true branch at specified nodes
                coast_thrust,                  # false branch elsewhere
                node_ranges=[(0, 10), (20, 30)]  # boost at nodes 0-9 and 20-29
            )

    Note:
        This operation is only supported for JAX lowering. CVXPy lowering will
        raise NotImplementedError since conditional logic is not DCP-compliant.
    """

    def __init__(
        self,
        pred: Union[Inequality, List[Inequality], "All", "Any", None],
        true_branch: Union[Expr, float, int, np.ndarray],
        false_branch: Union[Expr, float, int, np.ndarray],
        node_ranges: Optional[List[Tuple[int, int]]] = None,
    ):
        """Initialize a conditional expression.

        Args:
            pred: Predicate for the conditional. Can be:
                - Single Inequality (e.g., x <= 5)
                - List of Inequalities (AND semantics, shorthand for All([...]))
                - All expression for explicit AND
                - Any expression for OR semantics
                - None for purely node-based switching (requires node_ranges)
            true_branch: Expression to evaluate when predicate is True
            false_branch: Expression to evaluate when predicate is False
            node_ranges: Optional list of (start, end) tuples specifying node ranges
                where the conditional is active. Each tuple defines a half-open
                interval [start, end) of node indices. Outside these ranges, the
                false branch is always evaluated. None means active at all nodes.
                Required when pred is None.

        Raises:
            TypeError: If pred is not a valid predicate type
            ValueError: If node_ranges contains invalid ranges or pred=None without node_ranges
        """
        # Normalize pred to All/Any/Inequality/None
        if pred is None:
            if node_ranges is None:
                raise ValueError(
                    "Cond with pred=None requires node_ranges to be specified. "
                    "Use node_ranges to define which trajectory nodes take the true branch."
                )
            predicate = None
        elif isinstance(pred, (All, Any)):
            predicate = pred
        elif isinstance(pred, Inequality):
            predicate = pred
        elif isinstance(pred, list):
            if len(pred) == 0:
                raise ValueError("Cond predicate list cannot be empty")
            for i, p in enumerate(pred):
                if not isinstance(p, Inequality):
                    raise TypeError(
                        f"Cond predicate[{i}] must be an Inequality constraint "
                        f"(e.g., x <= 5, y >= 0), got {type(p).__name__}."
                    )
            # Wrap list in All for AND semantics (backwards compatibility)
            predicate = All(pred)
        else:
            raise TypeError(
                f"Cond predicate must be an Inequality, All, Any, None, or list of Inequalities "
                f"(e.g., x <= 5, ox.All([...]), ox.Any([...]), None), got {type(pred).__name__}."
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

        self.predicate = predicate
        self.true_branch = to_expr(true_branch)
        self.false_branch = to_expr(false_branch)
        self.node_ranges = node_ranges

    def children(self):
        """Return the child expressions: predicate (if any), true branch, and false branch."""
        if self.predicate is None:
            return [self.true_branch, self.false_branch]
        return [self.predicate, self.true_branch, self.false_branch]

    def canonicalize(self) -> "Expr":
        """Canonicalize by canonicalizing all children, preserving node_ranges."""
        predicate = self.predicate.canonicalize() if self.predicate is not None else None
        true_branch = self.true_branch.canonicalize()
        false_branch = self.false_branch.canonicalize()
        return Cond(predicate, true_branch, false_branch, node_ranges=self.node_ranges)

    def check_shape(self) -> Tuple[int, ...]:
        """Check and return the output shape of the conditional.

        The predicate must be scalar (or reduce to scalar via All/Any), and the
        true and false branches must have broadcastable shapes. The output shape
        is the broadcasted shape of the two branches.

        Returns:
            tuple: The broadcasted shape of true_branch and false_branch

        Raises:
            ValueError: If predicate is not scalar or branches have incompatible shapes
        """
        if self.predicate is not None:
            pred_shape = self.predicate.check_shape()
            if pred_shape != ():
                raise ValueError(f"Cond predicate must be scalar, got shape {pred_shape}")

        true_shape = self.true_branch.check_shape()
        false_shape = self.false_branch.check_shape()

        # True and false branches must be broadcastable
        try:
            return np.broadcast_shapes(true_shape, false_shape)
        except ValueError as e:
            raise ValueError(
                f"Cond branches have incompatible shapes: {true_shape} and {false_shape}"
            ) from e

    def __repr__(self) -> str:
        """Return string representation of the conditional."""
        pred_repr = "None" if self.predicate is None else repr(self.predicate)
        base = f"Cond({pred_repr}, {self.true_branch!r}, {self.false_branch!r}"
        if self.node_ranges is not None:
            return f"{base}, node_ranges={self.node_ranges!r})"
        return f"{base})"
