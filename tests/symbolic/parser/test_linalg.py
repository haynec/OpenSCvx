"""Tests for parser linear algebra handlers.

This module tests parsing of linear algebra operations:
Norm, Sum, Diag, Inv, Transpose
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Constant,
    Diag,
    Inv,
    Norm,
    State,
    Sum,
    Transpose,
)
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
# Norm
# =============================================================================


def test_parse_norm_default():
    expr = _parser().parse("Norm(x)")
    assert isinstance(expr, Norm)
    assert expr.ord == "fro"


def test_parse_norm_positional_ord():
    expr = _parser().parse("Norm(x, 2)")
    assert isinstance(expr, Norm)
    assert expr.ord == 2


def test_parse_norm_kwarg_ord():
    expr = _parser().parse("Norm(x, ord=1)")
    assert isinstance(expr, Norm)
    assert expr.ord == 1


def test_parse_norm_ord_string():
    expr = _parser().parse("Norm(x, ord='inf')")
    assert isinstance(expr, Norm)
    assert expr.ord == "inf"


def test_parse_norm_ord_fro():
    expr = _parser().parse("Norm(x, ord='fro')")
    assert isinstance(expr, Norm)
    assert expr.ord == "fro"


def test_norm_wrong_args():
    with pytest.raises(ValueError, match="at least 1"):
        _parser().parse("Norm()")


# =============================================================================
# Sum
# =============================================================================


def test_parse_sum():
    expr = _parser().parse("Sum(x)")
    assert isinstance(expr, Sum)


def test_sum_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("Sum(x, x)")


# =============================================================================
# Diag
# =============================================================================


def test_parse_diag():
    expr = _parser().parse("Diag(x)")
    assert isinstance(expr, Diag)


def test_diag_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("Diag()")


# =============================================================================
# Inv
# =============================================================================


def test_parse_inv():
    M = Constant(np.eye(3))
    p = _parser(M=M)
    expr = p.parse("Inv(M)")
    assert isinstance(expr, Inv)


def test_inv_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("Inv(x, x)")


# =============================================================================
# Transpose
# =============================================================================


def test_parse_transpose_function():
    expr = _parser().parse("Transpose(x)")
    assert isinstance(expr, Transpose)


def test_parse_transpose_dot_T():
    expr = _parser().parse("x.T")
    assert isinstance(expr, Transpose)


def test_transpose_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("Transpose()")


# =============================================================================
# Composed
# =============================================================================


def test_parse_norm_of_sub():
    from openscvx.symbolic.expr import Sub

    p = _parser()
    expr = p.parse("Norm(x - [1, 2, 3])")
    assert isinstance(expr, Norm)
    assert isinstance(expr.operand, Sub)


def test_parse_sum_of_square():
    from openscvx.symbolic.expr import Square

    p = _parser()
    expr = p.parse("Sum(Square(x))")
    assert isinstance(expr, Sum)
    assert isinstance(expr.operand, Square)
