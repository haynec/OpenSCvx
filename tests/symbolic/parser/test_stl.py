"""Tests for parser STL (Signal Temporal Logic) handlers.

This module tests parsing of STL operations:
Or, And, IfThen, IntegerVariable
"""

import pytest

from openscvx.symbolic.expr import State
from openscvx.symbolic.expr.stl import And, IfThen, IntegerVariable, Or
from openscvx.symbolic.parser import ExprParser

# =============================================================================
# Helper
# =============================================================================


def _parser(**extra):
    x = State("x", shape=(3,))
    symbols = {"x": x}
    symbols.update(extra)
    return ExprParser(symbols)


# =============================================================================
# stl.Or
# =============================================================================


def test_parse_or():
    p = _parser()
    expr = p.parse("Or(x[0] <= 1.0, x[1] <= 2.0)")
    assert isinstance(expr, Or)


def test_or_requires_two_args():
    with pytest.raises(ValueError, match="at least 2"):
        _parser().parse("Or(x[0] <= 1.0)")


def test_or_three_predicates():
    p = _parser()
    expr = p.parse("Or(x[0] <= 1.0, x[1] <= 2.0, x[2] <= 3.0)")
    assert isinstance(expr, Or)


# =============================================================================
# stl.And
# =============================================================================


def test_parse_and():
    p = _parser()
    expr = p.parse("And(x[0] <= 1.0, x[1] <= 2.0)")
    assert isinstance(expr, And)


def test_and_requires_two_args():
    with pytest.raises(ValueError, match="at least 2"):
        _parser().parse("And(x[0] <= 1.0)")


def test_and_three_predicates():
    p = _parser()
    expr = p.parse("And(x[0] <= 1.0, x[1] <= 2.0, x[2] <= 3.0)")
    assert isinstance(expr, And)
    assert len(expr.predicates) == 3


def test_and_preserves_order():
    p = _parser()
    expr = p.parse("And(x[0] <= 1.0, x[1] <= 2.0)")
    assert len(expr.predicates) == 2


# =============================================================================
# stl.IfThen
# =============================================================================


def test_parse_ifthen():
    p = _parser()
    expr = p.parse("IfThen(x[0] <= 1.0, x[1] <= 2.0)")
    assert isinstance(expr, IfThen)


def test_ifthen_requires_exactly_two_args():
    with pytest.raises(ValueError, match="exactly 2"):
        _parser().parse("IfThen(x[0] <= 1.0)")


def test_ifthen_too_many_args():
    with pytest.raises(ValueError, match="exactly 2"):
        _parser().parse("IfThen(x[0] <= 1.0, x[1] <= 2.0, x[2] <= 3.0)")


def test_ifthen_has_condition_and_consequent():
    p = _parser()
    expr = p.parse("IfThen(x[0] <= 1.0, x[1] <= 2.0)")
    assert expr.condition is not None
    assert expr.consequent is not None


# =============================================================================
# stl.IntegerVariable
# =============================================================================


def test_parse_integer_variable():
    p = _parser()
    expr = p.parse("IntegerVariable(x[0], [0.0, 1.0, 2.0])")
    assert isinstance(expr, IntegerVariable)


def test_integer_variable_requires_exactly_two_args():
    with pytest.raises(ValueError, match="exactly 2"):
        _parser().parse("IntegerVariable(x[0])")


def test_integer_variable_values_stored():
    import numpy as np

    p = _parser()
    expr = p.parse("IntegerVariable(x[0], [1.0, 2.0, 3.0])")
    assert isinstance(expr, IntegerVariable)
    assert np.allclose(expr.values, [1.0, 2.0, 3.0])
