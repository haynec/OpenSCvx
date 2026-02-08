"""Tests for parser logical and control flow handlers.

This module tests parsing of logical operations:
All, Any, Cond
"""

import pytest

from openscvx.symbolic.expr import Constant, State
from openscvx.symbolic.expr.logic import All, Any, Cond
from openscvx.symbolic.parser import ExprParser

# =============================================================================
# Helper
# =============================================================================


def _parser(**extra):
    x = State("x", shape=(3,))
    y = State("y", shape=(3,))
    symbols = {"x": x, "y": y}
    symbols.update(extra)
    return ExprParser(symbols)


# =============================================================================
# All
# =============================================================================


def test_parse_all_single_predicate():
    expr = _parser().parse("All(x[0] <= 5.0)")
    assert isinstance(expr, All)
    assert len(expr.predicates) == 1


def test_parse_all_multiple_predicates():
    expr = _parser().parse("All(x[0] <= 5.0, x[1] >= 0.0)")
    assert isinstance(expr, All)
    assert len(expr.predicates) == 2


def test_parse_all_three_predicates():
    expr = _parser().parse("All(x[0] <= 5.0, x[1] >= 0.0, x[2] <= 10.0)")
    assert isinstance(expr, All)
    assert len(expr.predicates) == 3


def test_all_no_args():
    with pytest.raises(ValueError, match="at least 1"):
        _parser().parse("All()")


# =============================================================================
# Any
# =============================================================================


def test_parse_any_single_predicate():
    expr = _parser().parse("Any(x[0] <= 5.0)")
    assert isinstance(expr, Any)
    assert len(expr.predicates) == 1


def test_parse_any_multiple_predicates():
    expr = _parser().parse("Any(x[0] <= 5.0, x[1] >= 0.0)")
    assert isinstance(expr, Any)
    assert len(expr.predicates) == 2


def test_parse_any_three_predicates():
    expr = _parser().parse("Any(x[0] <= 5.0, x[1] >= 0.0, x[2] <= 10.0)")
    assert isinstance(expr, Any)
    assert len(expr.predicates) == 3


def test_any_no_args():
    with pytest.raises(ValueError, match="at least 1"):
        _parser().parse("Any()")


# =============================================================================
# Cond — basic
# =============================================================================


def test_parse_cond_simple():
    expr = _parser().parse("Cond(x[0] <= 5.0, 1.0, 0.0)")
    assert isinstance(expr, Cond)
    assert isinstance(expr.true_branch, Constant)
    assert isinstance(expr.false_branch, Constant)
    assert expr.node_ranges is None


def test_parse_cond_expr_branches():
    expr = _parser().parse("Cond(x[0] <= 5.0, x[1], y[2])")
    assert isinstance(expr, Cond)


def test_cond_too_few_args():
    with pytest.raises(ValueError, match="at least 3"):
        _parser().parse("Cond(x[0] <= 5.0, 1.0)")


# =============================================================================
# Cond — with All / Any predicates
# =============================================================================


def test_parse_cond_with_all():
    expr = _parser().parse("Cond(All(x[0] <= 5.0, x[1] >= 0.0), 1.0, 0.0)")
    assert isinstance(expr, Cond)
    assert isinstance(expr.predicate, All)
    assert len(expr.predicate.predicates) == 2


def test_parse_cond_with_any():
    expr = _parser().parse("Cond(Any(x[0] <= 5.0, x[1] >= 0.0), 1.0, 0.0)")
    assert isinstance(expr, Cond)
    assert isinstance(expr.predicate, Any)


# =============================================================================
# Cond — node_ranges
# =============================================================================


def test_parse_cond_with_node_ranges():
    expr = _parser().parse("Cond(x[0] <= 5.0, 1.0, 0.0, node_ranges=[0, 10])")
    assert isinstance(expr, Cond)
    assert expr.node_ranges == [(0, 10)]


def test_parse_cond_with_multiple_node_ranges():
    expr = _parser().parse("Cond(x[0] <= 5.0, 1.0, 0.0, node_ranges=[0, 2, 5, 7])")
    assert isinstance(expr, Cond)
    assert expr.node_ranges == [(0, 2), (5, 7)]


def test_parse_cond_node_ranges_odd_length():
    with pytest.raises(ValueError, match="even number"):
        _parser().parse("Cond(x[0] <= 5.0, 1.0, 0.0, node_ranges=[0, 2, 5])")


# =============================================================================
# Cond — None predicate (node-based switching)
# =============================================================================


def test_parse_cond_none_predicate():
    expr = _parser().parse("Cond(None, x[0], y[0], node_ranges=[0, 10])")
    assert isinstance(expr, Cond)
    assert expr.predicate is None
    assert expr.node_ranges == [(0, 10)]


def test_parse_cond_none_without_node_ranges():
    with pytest.raises(ValueError, match="node_ranges"):
        _parser().parse("Cond(None, 1.0, 0.0)")


# =============================================================================
# Composed — Cond in dynamics expression
# =============================================================================


def test_cond_in_arithmetic():
    """Cond used as a subexpression in arithmetic."""
    from openscvx.symbolic.expr import Add

    expr = _parser().parse("x[0] + Cond(x[1] <= 0.0, 1.0, 0.0)")
    assert isinstance(expr, Add)
    assert any(isinstance(t, Cond) for t in expr.terms)
