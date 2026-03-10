"""Signal Temporal Logic (STL) operations using GMSR smooth robustness.

This module provides symbolic expression nodes for STL operations backed by
Generalized Mean-based Smooth Robustness (GMSR). These operators offer smooth,
differentiable approximations of Boolean logic for use in trajectory optimization.

Unlike the stljax module (which uses the stljax library), this module uses GMSR
parameterizations that work directly with constraint residuals.

GMSR Robustness Convention:
    GMSR functions operate on constraint residuals (negative when satisfied).
    The expression nodes in this module follow the standard STL convention where
    robustness is positive when satisfied. The conversion happens during lowering.

Reference:
    [https://doi.org/10.48550/arxiv.2405.10996]
    [https://doi.org/10.2514/6.2025-1895]
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

from .constraint import Constraint
from .expr import Constant, Expr

if TYPE_CHECKING:
    from .constraint import CTCS, NodalConstraint


class STLExpr(Expr):
    """Base class for GMSR-based Signal Temporal Logic operators.

    Provides common functionality for all GMSR STL operators including
    helper methods `.over()` and `.at()` to convert STL expressions
    to constraints for trajectory optimization.

    All subclasses evaluate to a scalar robustness value (positive when
    satisfied), consistent with the stljax module's convention.

    Example:
        STL operators can be converted to constraints using helper methods::

            wp1 = Norm(pos - c_1) <= r_1
            wp2 = Norm(pos - c_2) <= r_2
            visit_either = ox.stl.Or(wp1, wp2)

            constraints = [visit_either.over((3, 5))]
    """

    def over(
        self,
        interval: tuple[int, int],
        penalty: str = "smooth_relu",
        idx: Optional[int] = None,
        check_nodally: bool = False,
    ) -> "CTCS":
        """Apply this STL expression over a continuous interval using CTCS.

        Args:
            interval: Tuple of (start, end) node indices for enforcement interval
            penalty: Penalty function type for CTCS
            idx: Optional grouping index for multiple augmented states
            check_nodally: Whether to also enforce at discrete nodes

        Returns:
            Continuous-time constraint satisfaction wrapper

        Example::

            visit_either = ox.stl.Or(wp1, wp2)
            constraint = visit_either.over((3, 5))
        """
        from .arithmetic import Neg
        from .constraint import CTCS, Inequality

        constraint = Inequality(Neg(self), Constant(np.array(0.0)))

        return CTCS(
            constraint, penalty=penalty, nodes=interval, idx=idx, check_nodally=check_nodally
        )

    def at(self, nodes: Union[list, tuple]) -> "NodalConstraint":
        """Apply this STL expression only at specific nodes.

        Args:
            nodes: List of node indices where the constraint should be enforced

        Returns:
            Nodal constraint wrapper

        Example::

            visit_either = ox.stl.Or(wp1, wp2)
            constraint = visit_either.at([0, 5, 10])
        """
        from .arithmetic import Neg
        from .constraint import Inequality, NodalConstraint

        constraint = Inequality(Neg(self), Constant(np.array(0.0)))

        if isinstance(nodes, int):
            nodes = [nodes]
        return NodalConstraint(constraint, list(nodes))


def _validate_predicates(predicates, min_count, cls_name):
    """Validate that predicates are Constraint or STLExpr instances."""
    if len(predicates) < min_count:
        raise ValueError(f"{cls_name} requires at least {min_count} predicates")
    for pred in predicates:
        if not isinstance(pred, (Constraint, STLExpr)):
            raise TypeError(
                f"{cls_name} requires Constraint or STLExpr predicates, got "
                f"{type(pred).__name__}. "
                f"Did you mean to write a constraint like 'expr <= value'?"
            )


class Or(STLExpr):
    """GMSR smooth disjunction.

    Satisfied when at least one predicate is satisfied. Uses the GMSR smooth
    OR parameterization for differentiable optimization.

    Args:
        *predicates: Two or more Constraint or STLExpr objects
        c: Smoothing parameter (default 1e-4). Smaller values give tighter
            approximation to the true Boolean OR.
        lite: If True, use the lite variant that only considers the positive
            part of the OR function.

    Example::

        import openscvx as ox
        reach_a = ox.Norm(position - goal_a) <= 0.5
        reach_b = ox.Norm(position - goal_b) <= 0.5

        reach_either = ox.stl.Or(reach_a, reach_b)
        constraints = [reach_either.over((3, 5))]
    """

    def __init__(
        self, *predicates: Union[Constraint, "STLExpr"], c: float = 1e-4, lite: bool = False
    ):
        _validate_predicates(predicates, 2, "Or")
        self.predicates = list(predicates)
        self.c = c
        self.lite = lite

    def children(self):
        return self.predicates

    def canonicalize(self) -> "Expr":
        predicates = []
        for pred in self.predicates:
            canonicalized = pred.canonicalize()
            if (
                isinstance(canonicalized, Or)
                and canonicalized.c == self.c
                and canonicalized.lite == self.lite
            ):
                predicates.extend(canonicalized.predicates)
            else:
                predicates.append(canonicalized)

        if len(predicates) == 1:
            return predicates[0]

        result = Or.__new__(Or)
        result.predicates = predicates
        result.c = self.c
        result.lite = self.lite
        return result

    def check_shape(self) -> Tuple[int, ...]:
        if len(self.predicates) < 2:
            raise ValueError("Or requires at least two predicates")
        for pred in self.predicates:
            pred.check_shape()
        return ()

    def __repr__(self) -> str:
        predicates_repr = " | ".join(repr(p) for p in self.predicates)
        suffix = ", lite=True" if self.lite else ""
        return f"Or({predicates_repr}{suffix})"


class And(STLExpr):
    """GMSR smooth conjunction.

    Satisfied when all predicates are satisfied. Uses the GMSR smooth
    AND parameterization for differentiable optimization.

    Args:
        *predicates: Two or more Constraint or STLExpr objects
        c: Smoothing parameter (default 1e-4). Smaller values give tighter
            approximation to the true Boolean AND.
        lite: If True, use the lite variant that only considers the positive
            part of the AND function.

    Example::

        import openscvx as ox
        avoid_a = ox.Norm(position - obs_a) >= 1.0
        avoid_b = ox.Norm(position - obs_b) >= 1.0

        avoid_both = ox.stl.And(avoid_a, avoid_b)
        constraints = [avoid_both.over((0, 10))]
    """

    def __init__(
        self, *predicates: Union[Constraint, "STLExpr"], c: float = 1e-4, lite: bool = False
    ):
        _validate_predicates(predicates, 2, "And")
        self.predicates = list(predicates)
        self.c = c
        self.lite = lite

    def children(self):
        return self.predicates

    def canonicalize(self) -> "Expr":
        predicates = []
        for pred in self.predicates:
            canonicalized = pred.canonicalize()
            if (
                isinstance(canonicalized, And)
                and canonicalized.c == self.c
                and canonicalized.lite == self.lite
            ):
                predicates.extend(canonicalized.predicates)
            else:
                predicates.append(canonicalized)

        if len(predicates) == 1:
            return predicates[0]

        result = And.__new__(And)
        result.predicates = predicates
        result.c = self.c
        result.lite = self.lite
        return result

    def check_shape(self) -> Tuple[int, ...]:
        if len(self.predicates) < 2:
            raise ValueError("And requires at least two predicates")
        for pred in self.predicates:
            pred.check_shape()
        return ()

    def __repr__(self) -> str:
        predicates_repr = " & ".join(repr(p) for p in self.predicates)
        suffix = ", lite=True" if self.lite else ""
        return f"And({predicates_repr}{suffix})"


class IfThen(STLExpr):
    """GMSR smooth implication.

    Satisfied when: if the condition is satisfied, then the consequent
    is also satisfied. Uses the GMSR smooth implication parameterization.

    Formally: IfThen(cond, conseq) holds iff (cond => conseq),
    i.e., either the condition is NOT satisfied, or the consequent IS satisfied.

    Args:
        condition: Constraint or STLExpr representing the antecedent
        consequent: Constraint or STLExpr representing the consequent
        c: Smoothing parameter (default 1e-4)
        lite: If True, use the lite variant. The lite implication can also
            enforce continuous-time satisfaction via periodic auxiliary state.

    Example::

        import openscvx as ox
        in_zone = ox.Norm(position - zone_center) <= zone_radius
        speed_limit = speed <= max_speed

        # If in the zone, then obey speed limit
        rule = ox.stl.IfThen(in_zone, speed_limit)
        constraints = [rule.over((0, 10))]
    """

    def __init__(
        self,
        condition: Union[Constraint, "STLExpr"],
        consequent: Union[Constraint, "STLExpr"],
        c: float = 1e-4,
        lite: bool = False,
    ):
        _validate_predicates([condition, consequent], 2, "IfThen")
        self.condition = condition
        self.consequent = consequent
        self.c = c
        self.lite = lite

    def children(self):
        return [self.condition, self.consequent]

    def canonicalize(self) -> "Expr":
        result = IfThen.__new__(IfThen)
        result.condition = self.condition.canonicalize()
        result.consequent = self.consequent.canonicalize()
        result.c = self.c
        result.lite = self.lite
        return result

    def check_shape(self) -> Tuple[int, ...]:
        self.condition.check_shape()
        self.consequent.check_shape()
        return ()

    def __repr__(self) -> str:
        suffix = ", lite=True" if self.lite else ""
        return f"IfThen({self.condition!r} => {self.consequent!r}{suffix})"


class IntegerVariable(STLExpr):
    """GMSR smooth discrete/integer variable constraint.

    Constrains an expression to take one of a set of allowed discrete values.
    Uses smooth equality penalties combined with GMSR OR to enforce that the
    expression matches at least one of the given values.

    The expression evaluates to a penalty that is zero when the variable equals
    one of the allowed values, and positive otherwise. Use `.over()` or `.at()`
    to enforce this as a constraint.

    Args:
        expr: Symbolic expression to constrain (e.g. a State or Variable)
        values: Array-like of allowed discrete values
        c: Smoothing parameter (default 1e-4)

    Example::

        import openscvx as ox
        gear = ox.State("gear", shape=())

        # Constrain gear to discrete values {1, 2, 3, 4}
        discrete_gear = ox.stl.IntegerVariable(gear, [1, 2, 3, 4])
        constraints = [discrete_gear.over((0, 10))]
    """

    def __init__(self, expr: Expr, values, c: float = 1e-4):
        if not isinstance(expr, Expr):
            raise TypeError(f"IntegerVariable requires an Expr, got {type(expr).__name__}")
        self.expr = expr
        self.values = np.asarray(values)
        self.c = c

    def children(self):
        return [self.expr]

    def canonicalize(self) -> "Expr":
        result = IntegerVariable.__new__(IntegerVariable)
        result.expr = self.expr.canonicalize()
        result.values = self.values
        result.c = self.c
        return result

    def check_shape(self) -> Tuple[int, ...]:
        self.expr.check_shape()
        return ()

    def __repr__(self) -> str:
        return f"IntegerVariable({self.expr!r}, values={self.values.tolist()})"
