"""Signal Temporal Logic (STL) operations using GMSR smooth robustness.

This module provides symbolic expression nodes for STL operations backed by
Generalized Mean-based Smooth Robustness (GMSR). These operators offer smooth,
differentiable approximations of Boolean logic for use in trajectory optimization.

Unlike the stljax module (which uses the stljax library), this module uses GMSR
parameterizations that work directly with constraint residuals.

!!! warning "Interval and time semantics are a work in progress"
    The *logical* operators in this module (``And``, ``Or``, ``Not``,
    ``IfThen``) are stable and compose freely. The *temporal* side of
    STL — anything involving an enforcement interval or time-bounded
    quantification — is still being built out:

    - Intervals are typed: :class:`NodeInterval` (node indices) is
      implemented end to end. :class:`TimeInterval` (seconds, resolved
      against the model's explicit ``time`` state) is accepted at
      construction so downstream code can be sketched against the
      eventual API, but raises ``NotImplementedError`` at lowering.
      The kind of interval is selected explicitly by the
      ``interval_type`` kwarg on temporal constructors and on
      ``.over()`` — either ``"nodes"`` (the default) or ``"seconds"``.
      There is no implicit int-vs-float dispatch.
    - Mixed-interval nesting (e.g. ``□[0,5] p ∧ ◇[3,8] q``) has no
      defined lowering yet and is **rejected at construction time**.
      Temporal operators used as children of other STL operators must
      be interval-free and inherit the ambient window from the
      enclosing ``.over()``. Compose distinct windows as separate
      top-level constraints instead.
    - ``Eventually`` and ``Until`` are placeholder nodes only —
      construction is allowed so downstream code can be sketched, but
      any attempt to lower or enforce them raises ``NotImplementedError``.
      A novel formulation is in progress.

    Treat this surface as load-bearing for *logical* task composition
    and as preview for *temporal* composition until those gaps close.

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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import numpy as np

from .constraint import Constraint
from .expr import Constant, Expr

if TYPE_CHECKING:
    from .constraint import CTCS, NodalConstraint


# ---------------------------------------------------------------------------
# Interval value objects
#
# STL temporal operators take an enforcement interval. There are two
# fundamentally different *kinds* of interval, and they lower very
# differently:
#
#   - NodeInterval (start_node, end_node): resolved statically at graph
#     build time. The participating nodes are known before any
#     optimization runs and the lowering becomes a CTCS window over a
#     known slice of the trajectory.
#
#   - TimeInterval (t_start, t_end) in seconds: resolved at *evaluation*
#     time against the model's explicit `time` state. The set of nodes
#     that fall inside the window is iterate-dependent; the interval is
#     realised by gating the predicate residual on
#     `(time >= t_start) & (time <= t_end)`. There is no static
#     "convert to a node slice" step.
#
# These are not unit conversions of each other; they are different
# lowering strategies that share a surface API. The Interval ABC +
# Interval.coerce give a single, typed construction site so that the
# rest of the STL stack — constructors, parser, lowerer — never has to
# hand-roll the dispatch. The user picks the kind explicitly via the
# ``interval_type`` kwarg ("nodes" or "seconds"); there is no implicit
# int-vs-float inference.
#
# Status: NodeInterval is implemented end to end. TimeInterval is
# accepted at construction so that downstream code can be written
# against the eventual API, but raises NotImplementedError at lowering
# time.
# ---------------------------------------------------------------------------


IntervalKind = str  # "nodes" or "seconds"
_VALID_KINDS = ("nodes", "seconds")


class Interval:
    """Base class for STL enforcement intervals.

    Use :meth:`coerce` to build an ``Interval`` from a user-supplied
    value (a tuple, an existing ``Interval``, or a parser ``Constant``).
    Direct instantiation of the subclasses is also supported.
    """

    @staticmethod
    def coerce(value: object, kind: IntervalKind = "nodes") -> "Interval":
        """Coerce a user-supplied interval-like value into an ``Interval``.

        Args:
            value: One of
                - an existing ``Interval`` instance (returned as-is, with
                  a sanity check that ``kind`` matches its concrete type
                  unless the caller left ``kind`` at the default);
                - a length-2 tuple/list of numbers;
                - a parser :class:`Constant` wrapping a length-2 array.
            kind: ``"nodes"`` (default) selects :class:`NodeInterval`;
                ``"seconds"`` selects :class:`TimeInterval`. The user
                names the kind explicitly — there is no inference from
                int-vs-float.

        For ``kind="nodes"`` the bounds must be integral (Python ``int``
        or a ``float``/array element exactly equal to an integer); a
        non-integral bound is rejected loudly. For ``kind="seconds"``
        any numeric bound is accepted and cast to ``float``.
        """
        if kind not in _VALID_KINDS:
            raise ValueError(f"interval_type must be one of {_VALID_KINDS!r}, got {kind!r}")

        # Pre-typed Intervals pass through. If the caller named a kind,
        # require it to match — silent reinterpretation is exactly the
        # footgun this API is trying to remove.
        if isinstance(value, Interval):
            target_cls = NodeInterval if kind == "nodes" else TimeInterval
            if not isinstance(value, target_cls):
                raise TypeError(f"interval_type={kind!r} but received {type(value).__name__}")
            return value

        if isinstance(value, Constant):
            arr = value.value
            if arr.shape != (2,):
                raise TypeError(f"Interval Constant must have shape (2,), got {arr.shape}")
            a, b = arr[0].item(), arr[1].item()
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            a, b = value
            if isinstance(a, bool) or isinstance(b, bool):
                raise TypeError("Interval bounds cannot be bool")
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise TypeError(
                    f"Interval bounds must be numeric, got ({type(a).__name__}, {type(b).__name__})"
                )
        else:
            raise TypeError(
                f"Cannot coerce {value!r} to an Interval. Expected an Interval "
                f"instance, a length-2 tuple/list of numbers, or a length-2 Constant."
            )

        if kind == "nodes":
            for name, v in (("start", a), ("end", b)):
                if isinstance(v, float) and not v.is_integer():
                    raise TypeError(
                        f"interval_type='nodes' requires integral bounds, got "
                        f"{name}={v!r}. Pass interval_type='seconds' if you "
                        f"meant a wall-time interval."
                    )
            return NodeInterval(int(a), int(b))
        return TimeInterval(float(a), float(b))


@dataclass(frozen=True)
class NodeInterval(Interval):
    """An interval expressed as ``[start_node, end_node]`` discretization indices.

    Resolved statically at graph build time: the participating nodes are
    known before any optimization runs.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError(
                f"NodeInterval bounds must be int, got "
                f"({type(self.start).__name__}, {type(self.end).__name__})"
            )
        if self.end < self.start:
            raise ValueError(f"NodeInterval end ({self.end}) must be >= start ({self.start})")

    def __repr__(self) -> str:
        return f"NodeInterval({self.start}, {self.end})"


@dataclass(frozen=True)
class TimeInterval(Interval):
    """An interval expressed as ``[t_start, t_end]`` in seconds.

    Resolved at *evaluation* time against the model's explicit ``time``
    state — the set of nodes that fall inside the window is iterate-
    dependent and is realised by gating the predicate residual on
    ``(time >= t_start) & (time <= t_end)``.

    !!! warning "Not yet implemented"
        Construction is allowed so downstream code can be sketched
        against the eventual API, but lowering raises
        ``NotImplementedError``. Use :class:`NodeInterval` for now.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"TimeInterval end ({self.end}) must be >= start ({self.start})")

    def __repr__(self) -> str:
        return f"TimeInterval({self.start}, {self.end})"


IntervalLike = Union[Interval, Tuple[float, float]]


class STLExpr(Expr):
    """Base class for GMSR-based Signal Temporal Logic operators.

    Provides common functionality for all GMSR STL operators including
    helper methods `.over()` and `.at()` to convert STL expressions
    to constraints for trajectory optimization.

    Sealing a spec — `.over()` vs `.at()`:
        STL specs live in a symbolic world; to enforce one you must
        seal it into a concrete constraint via one of two entry points.
        Both bind the spec to a *where on the trajectory*, but they
        differ in the **shape** of that where, and that shape is the
        only thing that determines the lowering:

        - ``.over(interval)`` binds the spec to a contiguous interval
          (an :class:`Interval`, or a length-2 tuple coerced to one).
          Lowers to a ``CTCS`` that integrates the violation across
          the window — this is the natural realization of an STL
          temporal quantifier ``□[a,b] p``.
        - ``.at(nodes)`` binds the spec to a discrete list of node
          indices. Lowers to a ``NodalConstraint`` that enforces the
          predicate independently at each point — semantically a
          finite conjunction ``p(n_1) ∧ p(n_2) ∧ ...``.

        For a temporal operator like :class:`Always`, the interval is
        part of the formula itself (``Always(p, I)``) and ``.over()``
        seals it without taking a separate window. Specifying the
        interval *both* on the operator and at the seal call is a
        contradiction (not an override) and is rejected; the same
        rule applies to ``Always(p, I).at(...)``. To pick a discrete
        point set instead, leave the constructor's interval unset:
        ``Always(p).at([2, 3, 5])`` is the universal quantifier over
        a finite set, equivalent to a finite conjunction.

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

    # TODO: (griffin-norris, haynec, sametusun781) STL seal API status:
    #   [done]   Intervals are typed (NodeInterval / TimeInterval) and
    #            funnel through Interval.coerce; mixed-interval nesting
    #            is rejected at construction.
    #   [done]   Override semantics on Always.over() removed: passing
    #            an interval to both the constructor and .over() now
    #            raises. Same rule on Always.at() when the operator
    #            already carries an interval. The interval is part of
    #            the formula and can only live in one place.
    #   [done]   .over()/.at() unification: not needed. They are sugar
    #            for two different region shapes (contiguous interval
    #            vs. discrete point set) that map to two different
    #            lowerings (CTCS vs. NodalConstraint). The "shape
    #            determines lowering" story is the unification.
    #   [open]   TimeInterval lowering itself (gating on the model's
    #            explicit `time` state). Currently raises
    #            NotImplementedError at .over() time.

    def citation(self) -> List[str]:
        """Return the BibTeX citation for the GMSR robustness formulation."""
        return [
            r"""@misc{uzun2024gmsr,
  title={Optimization with Temporal and Logical Specifications via Generalized
    Mean-based Smooth Robustness Measures},
  author={Uzun, Samet and Elango, Purnanand and Garoche, Pierre-Lo{\"i}c and
    A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
  year={2024},
  eprint={2405.10996},
  archivePrefix={arXiv},
  primaryClass={math.OC}
}"""
        ]

    def over(
        self,
        interval: "IntervalLike",
        penalty: Union[str, Callable[[Expr], Expr]] = "smooth_relu",
        idx: Optional[int] = None,
        check_nodally: bool = False,
        licq_max: Optional[float] = None,
        interval_type: IntervalKind = "nodes",
    ) -> "CTCS":
        """Apply this STL expression over an enforcement interval using CTCS.

        Args:
            interval: An ``Interval`` or a length-2 tuple of numeric bounds.
            interval_type: ``"nodes"`` (default) treats the bounds as
                discretization indices and produces a :class:`NodeInterval`;
                ``"seconds"`` treats them as wall-time and produces a
                :class:`TimeInterval` (lowering not yet implemented).
            penalty: CTCS penalty function name, or a callable ``Expr -> Expr``
                receiving the constraint residual (see :class:`CTCS`)
            idx: Optional grouping index for multiple augmented states
            check_nodally: Whether to also enforce at discrete nodes
            licq_max: Optional upper bound on the augmented state accumulating
                this constraint's violation penalty. Constraints sharing an
                augmented state must agree on the value; groups without one use
                the problem-wide ``licq_max`` (a ``Problem`` argument).

        Returns:
            Continuous-time constraint satisfaction wrapper

        Example::

            visit_either = ox.stl.Or(wp1, wp2)
            constraint = visit_either.over((3, 5))

        See also:
            :meth:`at` for binding the spec to a discrete point set
            instead of a contiguous interval. The two methods seal to
            different constraint types (CTCS vs. NodalConstraint)
            because they pick different region shapes — see the
            class docstring for the full story.
        """
        from .arithmetic import Neg
        from .constraint import CTCS, Inequality

        coerced = Interval.coerce(interval, interval_type)
        if isinstance(coerced, TimeInterval):
            raise NotImplementedError(
                "TimeInterval lowering is not yet implemented. Time-based "
                "intervals will be realised by gating the predicate residual "
                "on the model's explicit `time` state. Use NodeInterval "
                "(int node indices) for now."
            )

        constraint = Inequality(Neg(self), Constant(np.array(0.0)))

        return CTCS(
            constraint,
            penalty=penalty,
            nodes=(coerced.start, coerced.end),
            idx=idx,
            check_nodally=check_nodally,
            licq_max=licq_max,
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

        See also:
            :meth:`over` for binding the spec to a contiguous
            interval instead of a discrete point set. See the class
            docstring for the full story on how the region shape
            determines the lowering.

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
    """Validate that predicates are Constraint or STLExpr instances.

    Also rejects temporal STL nodes that carry their own enforcement
    interval as children of another STL operator. Mixed-interval nesting
    (e.g. ``□[0,5] p ∧ ◇[3,8] q``) has no defined lowering yet, so we
    refuse it loudly at construction time rather than silently dropping
    the inner interval at lowering time. The workaround is to either
    drop the inner interval and let it inherit the ambient window from
    ``.over()``, or to compose the temporal pieces as separate
    top-level constraints.
    """
    if len(predicates) < min_count:
        raise ValueError(f"{cls_name} requires at least {min_count} predicates")
    for pred in predicates:
        if not isinstance(pred, (Constraint, STLExpr)):
            raise TypeError(
                f"{cls_name} requires Constraint or STLExpr predicates, got "
                f"{type(pred).__name__}. "
                f"Did you mean to write a constraint like 'expr <= value'?"
            )
        if isinstance(pred, _TemporalSTLExpr) and pred.interval is not None:
            raise ValueError(
                f"{cls_name}(...) cannot contain a temporal operator that "
                f"already carries an interval (got {type(pred).__name__} with "
                f"interval={pred.interval!r}). Mixed-interval nesting has no "
                f"defined lowering yet, so the inner interval would be "
                f"silently dropped. Either drop the inner interval at "
                f"construction and let it inherit the ambient window from "
                f"the enclosing .over(), or compose the pieces as separate "
                f"top-level constraints: Always(p, I1).over() and "
                f"Always(q, I2).over()."
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

    !!! note
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


class _TemporalSTLExpr(STLExpr):
    """Base class for STL temporal operators (``Always``, ``Eventually``,
    ``Until``).

    Temporal operators are distinguished from purely-logical operators
    (``And``, ``Or``, ``Not``, ``IfThen``) by carrying an enforcement
    :class:`Interval`. Subclasses must set ``self.interval`` (an
    ``Interval`` or ``None``) in their ``__init__``. This marker is what
    :func:`_validate_predicates` checks to refuse mixed-interval nesting.
    """

    interval: Optional[Interval] = None

    def at(self, nodes: Union[list, tuple]) -> "NodalConstraint":
        """Seal this temporal spec at a discrete set of nodes.

        Only valid when the operator does not already carry an
        interval. The interval is part of *what the formula says*
        (``□[a,b] p``); ``.at`` picks a discrete point set instead.
        Specifying both is a contradiction, not an override, and is
        rejected.
        """
        if self.interval is not None:
            raise ValueError(
                f"{type(self).__name__} already carries an interval "
                f"({self.interval!r}); cannot also seal with .at({nodes!r}). "
                f"The interval is part of the formula — pick exactly one of "
                f"a contiguous interval (.over()) or a discrete point set "
                f"(.at()) by leaving the constructor's interval unset."
            )
        return STLExpr.at(self, nodes)


class Always(_TemporalSTLExpr):
    """Enforce ``predicate`` at every point in ``interval`` (STL ``Always``).

    ``Always(p, (a, b))`` is satisfied iff ``p`` holds at *every* time in
    the interval ``[a, b]``. This is the universal temporal quantifier of
    STL and is the natural way to express invariants such as obstacle
    avoidance, box constraints, or speed limits over a window of the
    trajectory.

    Args:
        predicate: A Constraint or STLExpr that should hold throughout
            the interval.
        interval: Optional ``(start, end)`` node indices defining the
            enforcement window. If omitted at construction, must be
            supplied at ``.over()`` time.

    Example::

        import openscvx as ox
        avoid = ox.linalg.Norm(position - obs) >= safety_radius
        constraints.append(ox.stl.Always(avoid, (0, N - 1)).over())

        # Composes naturally with other STL operators:
        spec = ox.stl.Always(avoid, (0, N - 1)) & ox.stl.Always(in_box, (0, N - 1))
        constraints.append(spec.over())

    !!! note
        Standalone, ``Always`` is syntactic sugar around CTCS: enforcing
        the integral of the constraint violation to be zero across the
        interval is equivalent to requiring the predicate to hold
        pointwise. ``.over()`` performs that wrapping.

        When ``Always`` is *nested* inside another STL operator it
        must be interval-free and inherits the ambient window from the
        enclosing ``.over()``. Passing an explicit interval to a nested
        ``Always`` is rejected at construction time, since mixed-
        interval nesting has no defined lowering yet. To compose
        distinct windows, use separate top-level constraints.
    """

    def __init__(
        self,
        predicate: Union[Constraint, "STLExpr"],
        interval: Optional[IntervalLike] = None,
        interval_type: IntervalKind = "nodes",
    ):
        _validate_predicates([predicate], 1, "Always")
        self.predicate = predicate
        self.interval = Interval.coerce(interval, interval_type) if interval is not None else None

    def children(self):
        return [self.predicate]

    def canonicalize(self) -> "Expr":
        result = Always.__new__(Always)
        result.predicate = self.predicate.canonicalize()
        result.interval = self.interval
        return result

    def check_shape(self) -> Tuple[int, ...]:
        self.predicate.check_shape()
        return ()

    def over(
        self,
        interval: Optional[IntervalLike] = None,
        penalty: str = "smooth_relu",
        idx: Optional[int] = None,
        check_nodally: bool = False,
        licq_max: Optional[float] = None,
        interval_type: IntervalKind = "nodes",
    ) -> "CTCS":
        """Convert this ``Always`` to a ``CTCS`` constraint.

        Args:
            interval: Enforcement interval, supplied here *or* at
                construction — but not both. The interval is part of
                what the formula says (``□[a,b] p``); specifying it in
                two places is a contradiction, not an override, and is
                rejected.
            penalty: CTCS penalty function name.
            idx: Optional grouping index for multiple augmented states.
            check_nodally: Whether to additionally enforce at discrete nodes.
            licq_max: Optional upper bound on the augmented state accumulating
                this constraint's violation penalty. Constraints sharing an
                augmented state must agree on the value; groups without one use
                the problem-wide ``licq_max`` (a ``Problem`` argument).

        Returns:
            A ``CTCS`` constraint enforcing the inner predicate over the
            interval.
        """
        if interval is not None and self.interval is not None:
            raise ValueError(
                f"Always already carries an interval ({self.interval!r}); "
                f"cannot also pass interval={interval!r} to .over(). The "
                f"interval is part of the formula — specify it in exactly "
                f"one place: either Always(p, I).over() or Always(p).over(I)."
            )
        chosen = Interval.coerce(interval, interval_type) if interval is not None else self.interval
        if chosen is None:
            raise ValueError(
                "Always requires an interval — pass one to the constructor or to .over()."
            )
        if isinstance(chosen, TimeInterval):
            raise NotImplementedError(
                "TimeInterval lowering is not yet implemented. Time-based "
                "intervals will be realised by gating the predicate residual "
                "on the model's explicit `time` state. Use NodeInterval "
                "(int node indices) for now."
            )

        if isinstance(self.predicate, STLExpr):
            return self.predicate.over(
                chosen, penalty=penalty, idx=idx, check_nodally=check_nodally, licq_max=licq_max
            )

        from .constraint import CTCS

        return CTCS(
            self.predicate,
            penalty=penalty,
            nodes=(chosen.start, chosen.end),
            idx=idx,
            check_nodally=check_nodally,
            licq_max=licq_max,
        )

    def __repr__(self) -> str:
        if self.interval is None:
            return f"Always({self.predicate!r})"
        return f"Always({self.predicate!r}, {self.interval!r})"


class _UnimplementedTemporal(_TemporalSTLExpr):
    """Base class for temporal STL nodes that are not yet implemented.

    !!! note
        Stores the predicate + interval so that future formulations can
        be swapped in without changing the surface API. Construction
        succeeds (so that imports and AST inspection work), but any
        attempt to lower or enforce the node raises NotImplementedError
        with a clear message.
    """

    _operator_name: str = "TemporalOperator"

    def __init__(
        self,
        predicate: Union[Constraint, "STLExpr"],
        interval: Optional[IntervalLike] = None,
        interval_type: IntervalKind = "nodes",
    ):
        _validate_predicates([predicate], 1, self._operator_name)
        self.predicate = predicate
        self.interval = Interval.coerce(interval, interval_type) if interval is not None else None

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
    """STL ``Eventually`` operator: ``p`` holds at some time in the interval.

    Semantically: ``Eventually(p, (a, b))`` is satisfied iff ``p`` holds at
    *some* time in ``[a, b]``. The interval is optional — when omitted,
    the operator inherits the ambient window from the enclosing seal
    (e.g. ``.over()``), the same way ``Always`` does.

    !!! warning "Not yet implemented!"
        Constructing the node is allowed so that downstream
        code can be sketched against the eventual API, but any attempt
        to lower or enforce it raises ``NotImplementedError``.
    """

    _operator_name = "Eventually"


class Until(_UnimplementedTemporal):
    """STL ``Until`` operator: ``left`` holds until ``right`` holds.

    ``Until(left, right, (a, b))`` is satisfied iff there exists a time
    ``t* ∈ [a, b]`` at which ``right`` holds, and ``left`` holds at every
    time in ``[a, t*]``. ``Until`` is the most expressive of the standard
    STL temporal operators; ``Always`` and ``Eventually`` can both be
    derived from it. The interval is optional — when omitted, the
    operator inherits the ambient window from the enclosing seal, the
    same way ``Always`` does.

    !!! warning "Not yet implemented!"
        Constructing the node is allowed so
        downstream code can be sketched against the eventual API, but
        any attempt to lower or enforce it raises ``NotImplementedError``.
    """

    _operator_name = "Until"

    def __init__(
        self,
        left: Union[Constraint, "STLExpr"],
        right: Union[Constraint, "STLExpr"],
        interval: Optional[IntervalLike] = None,
        interval_type: IntervalKind = "nodes",
    ):
        # _validate_predicates runs again inside super().__init__, but it
        # only sees `left`; call it explicitly here to also validate `right`.
        _validate_predicates([left, right], 2, "Until")
        super().__init__(left, interval=interval, interval_type=interval_type)
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def check_shape(self) -> Tuple[int, ...]:
        self.left.check_shape()
        self.right.check_shape()
        return ()

    def __repr__(self) -> str:
        return f"Until({self.left!r}, {self.right!r}, {self.interval!r})"
