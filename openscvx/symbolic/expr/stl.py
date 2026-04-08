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

Author: Samet Uzun and Chris Hayner

Reference:
    https://doi.org/10.48550/arxiv.2405.10996
    https://doi.org/10.2514/6.2025-1895

See also:
    ``openscvx.symbolic.expr.logic`` provides ``All``/``Any``/``Cond`` for
    *hard-boolean* branching inside expressions (e.g. switching dynamics
    based on a predicate). Those are JAX-only and not differentiable
    across the branch. The operators in *this* module are smooth and
    differentiable everywhere, and are the right tool for composing
    constraints into a task specification.
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

    STL Robustness Convention:
        STL uses "robustness" values that are positive when constraints are satisfied.
        For an Inequality constraint `lhs <= rhs`:
        - Constraint residual: `lhs - rhs` (should be <= 0 when satisfied)
        - STL robustness: `rhs - lhs` (should be >= 0 when satisfied)

    Example:
        STL operators can be converted to constraints using helper methods:

            wp1 = Norm(pos - c_1) <= r_1
            wp2 = Norm(pos - c_2) <= r_2
            visit_either = ox.stl.Or(wp1, wp2) # STL Operator

            # Convert to constraint with .over()
            constraints = [visit_either.over((3, 5))]

    Note:
        This is a base class. Use concrete subclasses like Or, And,
        Eventually, Always, or Until for actual STL specifications.
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

        Example:
            Enforce STL expression over an interval:

                visit_either = ox.stl.Or(wp1, wp2)
                constraint = visit_either.at([0, 5, 10])

        Note:
            This is a base class. Use concrete subclasses like Or, And,
            Eventually, Always, or Until for actual STL specifications.
        """
        from .arithmetic import Neg
        from .constraint import Inequality, NodalConstraint

        constraint = Inequality(Neg(self), Constant(np.array(0.0)))

        if isinstance(nodes, int):
            nodes = [nodes]
        return NodalConstraint(constraint, list(nodes))

    # ------------------------------------------------------------------
    # Operator sugar for natural STL composition.
    #
    # These overloads only fire when at least one operand is already an
    # STLExpr — bare ``Constraint`` objects intentionally do *not* get
    # ``&``/``|``/``~`` so that ``Cond``/``All``/``Any`` users (see
    # ``logic.py``) are not surprised by smooth-GMSR semantics sneaking
    # into their hard-boolean branching. Lift into STL explicitly with
    # any STL constructor (e.g. ``ox.stl.Always``) and the rest composes:
    #
    #     spec = (Always(c1) & Always(c2)) | ~stuck
    #
    # Note that smoothing parameters (``c``, ``lite``) cannot be passed
    # through operator syntax — fall back to ``And(...)``/``Or(...)``/
    # ``Not(...)`` if you need to tune them.
    # ------------------------------------------------------------------

    def __and__(self, other):
        return And(self, other)

    def __rand__(self, other):
        return And(other, self)

    def __or__(self, other):
        return Or(self, other)

    def __ror__(self, other):
        return Or(other, self)

    def __invert__(self):
        return Not(self)


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

    def __init__(self, expr: Expr, values: Union[list, np.ndarray], c: float = 1e-4):
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


class Not(STLExpr):
    """GMSR smooth negation.

    Satisfied when the inner predicate is *not* satisfied. Under the GMSR
    convention this is just a sign flip on the residual: if the inner
    predicate has robustness ``r``, then ``Not`` has robustness ``-r``.

    Args:
        predicate: A Constraint or STLExpr to negate.

    Example::

        import openscvx as ox
        in_zone = ox.linalg.Norm(position - center) <= radius
        outside = ox.stl.Not(in_zone)
        # Equivalently, with operator syntax once lifted into STL:
        outside = ~ox.stl.Always(in_zone)  # not always in zone

    Note:
        ``Not`` is a thin sign flip — there is no smoothing parameter.
        Composing ``Not`` with ``Or``/``And`` recovers De Morgan duals
        through the GMSR machinery rather than as an algebraic rewrite.
    """

    def __init__(self, predicate: Union[Constraint, "STLExpr"]):
        _validate_predicates([predicate], 1, "Not")
        self.predicate = predicate

    def children(self):
        return [self.predicate]

    def canonicalize(self) -> "Expr":
        canon = self.predicate.canonicalize()
        # Double-negation elimination: ~~p -> p
        if isinstance(canon, Not):
            return canon.predicate
        result = Not.__new__(Not)
        result.predicate = canon
        return result

    def check_shape(self) -> Tuple[int, ...]:
        self.predicate.check_shape()
        return ()

    def __repr__(self) -> str:
        return f"Not({self.predicate!r})"


def Always(
    predicate: Union[Constraint, "STLExpr"],
    interval: tuple[int, int],
    penalty: str = "smooth_relu",
    idx: Optional[int] = None,
    check_nodally: bool = False,
) -> "CTCS":
    """Enforce ``predicate`` at every point in ``interval`` (STL ``Always``).

    This is syntactic sugar around CTCS: enforcing the integral of the
    constraint violation to be zero across the interval is equivalent to
    requiring the predicate to hold pointwise. ``Always`` is provided so
    that STL specifications can be written in notation that mirrors the
    math, without the user having to remember the CTCS-of-violation
    encoding.

    Args:
        predicate: A Constraint or STLExpr that should hold throughout
            the interval.
        interval: ``(start, end)`` node indices defining the enforcement
            window.
        penalty: CTCS penalty function name.
        idx: Optional grouping index for multiple augmented states.
        check_nodally: Whether to additionally enforce at discrete nodes.

    Returns:
        A ``CTCS`` constraint enforcing ``predicate`` over ``interval``.

    Example::

        import openscvx as ox
        avoid = ox.linalg.Norm(position - obs) >= safety_radius
        constraints.append(ox.stl.Always(avoid, (0, N - 1)))
    """
    if isinstance(predicate, STLExpr):
        return predicate.over(
            interval, penalty=penalty, idx=idx, check_nodally=check_nodally
        )
    if not isinstance(predicate, Constraint):
        raise TypeError(
            f"Always requires a Constraint or STLExpr predicate, got "
            f"{type(predicate).__name__}."
        )
    from .constraint import CTCS

    return CTCS(
        predicate, penalty=penalty, nodes=interval, idx=idx, check_nodally=check_nodally
    )


class _UnimplementedTemporal(STLExpr):
    """Base class for temporal STL nodes that are not yet implemented.

    Stores the predicate + interval so that future formulations can be
    swapped in without changing the surface API. Construction succeeds
    (so that imports and AST inspection work), but any attempt to lower
    or enforce the node raises NotImplementedError with a clear message.
    """

    _operator_name: str = "TemporalOperator"

    def __init__(
        self,
        predicate: Union[Constraint, "STLExpr"],
        interval: tuple[int, int],
    ):
        _validate_predicates([predicate], 1, self._operator_name)
        self.predicate = predicate
        self.interval = interval

    def children(self):
        return [self.predicate]

    def canonicalize(self) -> "Expr":
        raise NotImplementedError(
            f"{self._operator_name} is not yet implemented. A novel formulation "
            f"is in progress; for now, use ox.stl.Always for pointwise "
            f"enforcement, or ox.stl.Or/And for explicit disjunctions."
        )

    def check_shape(self) -> Tuple[int, ...]:
        self.predicate.check_shape()
        return ()

    def over(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self._operator_name} is not yet implemented. A novel formulation "
            f"is in progress; for now, use ox.stl.Always for pointwise "
            f"enforcement, or ox.stl.Or/And for explicit disjunctions."
        )

    def at(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self._operator_name} is not yet implemented. A novel formulation "
            f"is in progress; for now, use ox.stl.Always for pointwise "
            f"enforcement, or ox.stl.Or/And for explicit disjunctions."
        )

    def __repr__(self) -> str:
        return f"{self._operator_name}({self.predicate!r}, {self.interval!r})"


class Eventually(_UnimplementedTemporal):
    """STL ``Eventually`` operator (placeholder).

    Semantically: ``Eventually(p, (a, b))`` is satisfied iff ``p`` holds at
    *some* time in ``[a, b]``. Implementation is in progress (a novel
    formulation by a collaborator); constructing the node is allowed so
    that downstream code can be sketched, but any use will raise
    NotImplementedError.
    """

    _operator_name = "Eventually"


class Until(_UnimplementedTemporal):
    """STL ``Until`` operator (placeholder).

    Not yet implemented. Constructing the node is allowed so downstream
    code can be sketched; any use raises NotImplementedError.
    """

    _operator_name = "Until"

    def __init__(
        self,
        left: Union[Constraint, "STLExpr"],
        right: Union[Constraint, "STLExpr"],
        interval: tuple[int, int],
    ):
        _validate_predicates([left, right], 2, "Until")
        self.left = left
        self.right = right
        self.predicate = left  # for _UnimplementedTemporal.children/check_shape
        self.interval = interval

    def children(self):
        return [self.left, self.right]

    def check_shape(self) -> Tuple[int, ...]:
        self.left.check_shape()
        self.right.check_shape()
        return ()

    def __repr__(self) -> str:
        return f"Until({self.left!r}, {self.right!r}, {self.interval!r})"
