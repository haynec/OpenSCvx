"""Tests for constraint parsing.

This module tests parsing of constraints including:
- Comparison operators (<=, >=, ==)
- .at() for NodalConstraint / NodeReference
- .over() for CTCS
- .convex()
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    CTCS,
    Constant,
    Equality,
    Inequality,
    NodalConstraint,
    State,
)
from openscvx.symbolic.expr.expr import NodeReference
from openscvx.symbolic.parser import ExprParser, ParseError

# =============================================================================
# Helper
# =============================================================================


def _parser(**extra):
    x = State("x", shape=(3,))
    symbols = {"x": x}
    symbols.update(extra)
    return ExprParser(symbols)


# =============================================================================
# Comparison Operators
# =============================================================================


def test_parse_less_equal():
    expr = _parser().parse("x <= [1, 2, 3]")
    assert isinstance(expr, Inequality)
    assert isinstance(expr.lhs, State)
    assert isinstance(expr.rhs, Constant)


def test_parse_greater_equal():
    expr = _parser().parse("x >= [0, 0, 0]")
    # >= flips operands: Inequality(rhs, lhs)
    assert isinstance(expr, Inequality)
    assert isinstance(expr.lhs, Constant)
    assert isinstance(expr.rhs, State)


def test_parse_equality():
    expr = _parser().parse("x == [1, 2, 3]")
    assert isinstance(expr, Equality)


def test_parse_scalar_constraint():
    expr = _parser().parse("x[0] <= 5.0")
    assert isinstance(expr, Inequality)


# =============================================================================
# .at() — NodalConstraint
# =============================================================================


def test_parse_constraint_at_single():
    expr = _parser().parse("(x <= [1, 2, 3]).at(0, 10)")
    assert isinstance(expr, NodalConstraint)
    assert expr.nodes == [0, 10]


def test_parse_constraint_at_multiple():
    expr = _parser().parse("(x[0] <= 5.0).at(0, 5, 10, 20)")
    assert isinstance(expr, NodalConstraint)
    assert expr.nodes == [0, 5, 10, 20]


# =============================================================================
# .at() — NodeReference (on plain Expr)
# =============================================================================


def test_parse_expr_at():
    expr = _parser().parse("x.at(5)")
    assert isinstance(expr, NodeReference)


def test_parse_expr_at_in_constraint():
    expr = _parser().parse("x.at(5) - x.at(4) <= [0, 0, 0]")
    assert isinstance(expr, Inequality)


# =============================================================================
# .over() — CTCS
# =============================================================================


def test_parse_over_basic():
    expr = _parser().parse("(x[0] <= 5.0).over(0, 10)")
    assert isinstance(expr, CTCS)
    assert expr.nodes == (0, 10)
    assert expr.penalty == "squared_relu"  # default


def test_parse_over_with_penalty():
    expr = _parser().parse("(x[0] <= 5.0).over(0, 10, penalty='huber')")
    assert isinstance(expr, CTCS)
    assert expr.penalty == "huber"


def test_parse_over_with_check_nodally():
    expr = _parser().parse("(x[0] <= 5.0).over(0, 10, check_nodally=True)")
    assert isinstance(expr, CTCS)
    assert expr.check_nodally is True


def test_over_requires_constraint():
    with pytest.raises(ParseError, match="only be called on a Constraint"):
        _parser().parse("x.over(0, 10)")


def test_over_requires_two_positional_args():
    with pytest.raises(ParseError, match="at least 2"):
        _parser().parse("(x[0] <= 5.0).over(0)")


# =============================================================================
# .convex()
# =============================================================================


def test_parse_convex_on_inequality():
    expr = _parser().parse("(x <= [1, 2, 3]).convex()")
    assert isinstance(expr, Inequality)
    assert expr.is_convex is True


def test_parse_convex_on_equality():
    expr = _parser().parse("(x == [1, 2, 3]).convex()")
    assert isinstance(expr, Equality)
    assert expr.is_convex is True


def test_convex_on_non_constraint_raises():
    with pytest.raises(ParseError, match="only be called on a Constraint"):
        _parser().parse("x.convex()")


# =============================================================================
# Chained Dot Methods
# =============================================================================


def test_parse_at_then_convex():
    expr = _parser().parse("(x <= [1, 2, 3]).at(0, 10).convex()")
    assert isinstance(expr, NodalConstraint)
    assert expr.constraint.is_convex is True


def test_parse_convex_then_at():
    expr = _parser().parse("(x <= [1, 2, 3]).convex().at(0, 10)")
    assert isinstance(expr, NodalConstraint)
    assert expr.constraint.is_convex is True


# =============================================================================
# Realistic Constraint Expressions
# =============================================================================


def test_parse_norm_constraint():
    from openscvx.symbolic.expr import Norm

    obs = Constant(np.array([1.0, 2.0, 3.0]))
    p = _parser(obs=obs)
    expr = p.parse("Norm(x - obs) >= 2.0")
    assert isinstance(expr, Inequality)
    # >= flips: Inequality(Const(2.0), Norm(...))
    assert isinstance(expr.lhs, Constant)
    assert isinstance(expr.rhs, Norm)
