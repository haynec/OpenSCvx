"""Tests for parser STL (Signal Temporal Logic) handlers.

This module tests parsing of STL operations:
Or
"""

import pytest

from openscvx.symbolic.expr import State
from openscvx.symbolic.expr.stl import Or
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
# Or
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
