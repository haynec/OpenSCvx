"""Signal Temporal Logic (STL) operations for trajectory optimization.

This module provides symbolic expression nodes for Signal Temporal Logic (STL)
operations, enabling the specification of complex temporal and logical constraints
in optimization problems. STL is particularly useful for robotics and autonomous
systems where tasks involve temporal reasoning.

STL operators accept Constraint objects (predicates) and extract robustness
expressions which are lowered to STLJax during compilation.
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

from .constraint import Constraint
from .expr import Constant, Expr

if TYPE_CHECKING:
    from .constraint import CTCS, NodalConstraint


class STLExpr(Expr):
    """Base class for Signal Temporal Logic operators.

    STL operators combine predicates (constraints) using temporal and logical
    operations. This base class provides:
    - Common functionality for all STL operators
    - Helper methods like `.over()` and `.at()` to convert STL expressions to constraints
    - Future: utility methods for STL formula manipulation, pretty-printing, etc.

    STL operators are Expr nodes that store robustness expressions. During lowering,
    they are handled by STLJax which computes the appropriate smooth approximations.

    STL Robustness Convention:
        STL uses "robustness" values that are positive when constraints are satisfied.
        For an Inequality constraint `lhs <= rhs`:
        - Constraint residual: `lhs - rhs` (should be <= 0 when satisfied)
        - STL robustness: `rhs - lhs` (should be >= 0 when satisfied)

    Example:
        STL operators can be converted to constraints using helper methods:

            wp1 = Norm(pos - c_1) <= r_1
            wp2 = Norm(pos - c_2) <= r_2
            visit_either = ox.stl.Or(wp1, wp2)  # STL operator

            # Convert to constraint with .over()
            constraints = [visit_either.over((3, 5))]

    Note:
        This is a base class. Use concrete subclasses like Or, And,
        Eventually, Always, or Until for actual STL specifications.
    """

    def over(
        self,
        interval: tuple[int, int],
        penalty: str = "squared_relu",
        idx: Optional[int] = None,
        check_nodally: bool = False,
    ) -> "CTCS":
        """Apply this STL expression over a continuous interval using CTCS.

        Converts the STL expression to a constraint and wraps it in CTCS
        for continuous-time enforcement.

        Args:
            interval: Tuple of (start, end) node indices for enforcement interval
            penalty: Penalty function type for CTCS
            idx: Optional grouping index for multiple augmented states
            check_nodally: Whether to also enforce at discrete nodes

        Returns:
            Continuous-time constraint satisfaction wrapper

        Example:
            Enforce STL expression over an interval:

                visit_either = ox.stl.Or(wp1, wp2)
                constraint = visit_either.over((3, 5))
        """
        from .arithmetic import Neg
        from .constraint import CTCS, Inequality

        # Create constraint: -STL_expr(...) <= 0
        # STL expressions evaluate to robustness (positive when satisfied)
        # We negate to get constraint residual (negative when satisfied)
        constraint = Inequality(Neg(self), Constant(np.array(0.0)))

        return CTCS(
            constraint, penalty=penalty, nodes=interval, idx=idx, check_nodally=check_nodally
        )

    def at(self, nodes: Union[list, tuple]) -> "NodalConstraint":
        """Apply this STL expression only at specific nodes.

        Converts the STL expression to a constraint and wraps it in NodalConstraint.

        Args:
            nodes: List of node indices where the constraint should be enforced

        Returns:
            Nodal constraint wrapper

        Example:
            Enforce STL expression at specific nodes:

                visit_either = ox.stl.Or(wp1, wp2)
                constraint = visit_either.at([0, 5, 10])
        """
        from .arithmetic import Neg
        from .constraint import Inequality, NodalConstraint

        # Create constraint: -STL_expr(...) <= 0
        constraint = Inequality(Neg(self), Constant(np.array(0.0)))

        if isinstance(nodes, int):
            nodes = [nodes]
        return NodalConstraint(constraint, list(nodes))


class Or(STLExpr):
    """Logical OR operation for STL predicates.

    Combines constraint predicates with disjunction. The Or is satisfied if
    at least one of its operands is satisfied.

    During lowering, this is handled by STLJax which computes the smooth maximum
    (LogSumExp) of the robustness values automatically.

    The Or operation allows expressing constraints like:
    - "Reach either goal A OR goal B"
    - "Avoid obstacle 1 OR obstacle 2" (at least one must be satisfied)
    - "Use path 1 OR path 2 OR path 3"

    Attributes:
        predicates: List of predicates (Constraint or STLExpr objects)

    Example:
        Visit either waypoint 1 OR waypoint 2:

            import openscvx as ox
            position = ox.State("pos", shape=(2,))
            goal_a = ox.Parameter("goal_a", shape=(2,), value=[1.0, 1.0])
            goal_b = ox.Parameter("goal_b", shape=(2,), value=[-1.0, -1.0])

            # Define predicates as constraints
            reach_a = ox.Norm(position - goal_a) <= 0.5
            reach_b = ox.Norm(position - goal_b) <= 0.5

            # Combine with OR operator
            reach_either = ox.stl.Or(reach_a, reach_b)

            # Enforce continuously over time interval
            constraints = [reach_either.over((3, 5))]

        Nested STL operators are also supported:

            # Or of And expressions
            expr = ox.stl.Or(
                ox.stl.And(c1, c2),
                ox.stl.And(c3, c4),
            )

    Note:
        Or evaluates to a scalar robustness value (positive when satisfied).
        Use `.over()` or `.at()` to convert to a constraint, or manually create
        a constraint with: `-Or(...) <= 0`

    See Also:
        stljax.formula.Or: Underlying STLJax implementation used during lowering
    """

    def __init__(self, *predicates: Union[Constraint, "STLExpr"]):
        """Initialize a logical OR operation.

        Args:
            *predicates: Two or more Constraint or STLExpr objects to combine
                        with logical OR. Each represents a predicate to be satisfied.

        Raises:
            ValueError: If fewer than two predicates are provided
            TypeError: If predicates are not Constraint or STLExpr instances
        """
        if len(predicates) < 2:
            raise ValueError("Or requires at least two predicates")

        # Validate that all predicates are constraints or STL expressions
        for pred in predicates:
            if not isinstance(pred, (Constraint, STLExpr)):
                raise TypeError(
                    f"Or requires Constraint or STLExpr predicates, got "
                    f"{type(pred).__name__}. "
                    f"Did you mean to write a constraint like 'expr <= value'?"
                )

        # Store predicates directly - robustness extraction happens during lowering
        self.predicates = list(predicates)

    def children(self):
        """Return predicates as children."""
        return self.predicates

    def canonicalize(self) -> "Expr":
        """Canonicalize by flattening nested Or and canonicalizing predicates.

        Flattens nested Or operations into a single flat Or with all predicates
        at the same level. For example: Or(a, Or(b, c)) → Or(a, b, c).

        Returns:
            Expr: Canonical form. If only one predicate remains, returns it directly.
        """
        predicates = []

        for pred in self.predicates:
            canonicalized = pred.canonicalize()
            if isinstance(canonicalized, Or):
                # Flatten nested Or: Or(a, Or(b, c)) -> Or(a, b, c)
                predicates.extend(canonicalized.predicates)
            else:
                predicates.append(canonicalized)

        if len(predicates) == 1:
            return predicates[0]

        # Reconstruct Or with canonicalized predicates
        result = Or.__new__(Or)
        result.predicates = predicates
        return result

    def check_shape(self) -> Tuple[int, ...]:
        """Validate predicate shapes and return scalar shape.

        Returns:
            tuple: Empty tuple () indicating a scalar result (STL robustness)

        Raises:
            ValueError: If fewer than two predicates exist
        """
        if len(self.predicates) < 2:
            raise ValueError("Or requires at least two predicates")

        # Validate all predicates
        for pred in self.predicates:
            pred.check_shape()

        # Or produces a scalar (STL robustness value)
        return ()

    def __repr__(self) -> str:
        predicates_repr = " | ".join(repr(p) for p in self.predicates)
        return f"Or({predicates_repr})"
