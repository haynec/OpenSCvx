"""Tests for parser math function handlers.

This module tests parsing of mathematical functions:
Sin, Cos, Tan, Tanh, Asin, Acos, Atan, Atan2, Sqrt, Square, Exp, Log, Abs, Max, Min,
PositivePart, Huber, SmoothReLU, LogSumExp, Linterp, Bilerp
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Abs,
    Acos,
    Asin,
    Atan,
    Atan2,
    Bilerp,
    Constant,
    Cos,
    Exp,
    Huber,
    Linterp,
    Log,
    LogSumExp,
    Max,
    Min,
    PositivePart,
    Sin,
    SmoothReLU,
    Sqrt,
    Square,
    State,
    Tan,
    Tanh,
)
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
# Unary Math Functions
# =============================================================================


def test_parse_sin():
    expr = _parser().parse("Sin(x)")
    assert isinstance(expr, Sin)


def test_parse_cos():
    expr = _parser().parse("Cos(x)")
    assert isinstance(expr, Cos)


def test_parse_tan():
    expr = _parser().parse("Tan(x)")
    assert isinstance(expr, Tan)


def test_parse_tanh():
    expr = _parser().parse("Tanh(x)")
    assert isinstance(expr, Tanh)


def test_parse_asin():
    expr = _parser().parse("Asin(x)")
    assert isinstance(expr, Asin)


def test_parse_acos():
    expr = _parser().parse("Acos(x)")
    assert isinstance(expr, Acos)


def test_parse_atan():
    expr = _parser().parse("Atan(x)")
    assert isinstance(expr, Atan)


def test_parse_atan2():
    expr = _parser().parse("Atan2(x[0], x[1])")
    assert isinstance(expr, Atan2)


def test_parse_sqrt():
    expr = _parser().parse("Sqrt(x)")
    assert isinstance(expr, Sqrt)


def test_parse_square():
    expr = _parser().parse("Square(x)")
    assert isinstance(expr, Square)


def test_parse_exp():
    expr = _parser().parse("Exp(x)")
    assert isinstance(expr, Exp)


def test_parse_log():
    expr = _parser().parse("Log(x)")
    assert isinstance(expr, Log)


def test_parse_abs():
    expr = _parser().parse("Abs(x)")
    assert isinstance(expr, Abs)


def test_parse_positive_part():
    expr = _parser().parse("PositivePart(x)")
    assert isinstance(expr, PositivePart)


# =============================================================================
# Unary Functions — Wrong Arg Count
# =============================================================================


def test_sin_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("Sin(x, x)")


def test_cos_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("Cos()")


def test_sqrt_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("Sqrt(x, x)")


def test_atan2_wrong_args():
    with pytest.raises(ValueError, match="exactly 2"):
        _parser().parse("Atan2(x)")


# =============================================================================
# Multi-arg Functions
# =============================================================================


def test_parse_max():
    p = _parser()
    expr = p.parse("Max(x[0], x[1])")
    assert isinstance(expr, Max)


def test_max_requires_two_args():
    with pytest.raises(ValueError, match="at least 2"):
        _parser().parse("Max(x)")


def test_parse_min():
    p = _parser()
    expr = p.parse("Min(x[0], x[1])")
    assert isinstance(expr, Min)


def test_min_requires_two_args():
    with pytest.raises(ValueError, match="at least 2"):
        _parser().parse("Min(x)")


def test_parse_logsumexp():
    p = _parser()
    expr = p.parse("LogSumExp(x[0], x[1])")
    assert isinstance(expr, LogSumExp)


def test_logsumexp_requires_two_args():
    with pytest.raises(ValueError, match="at least 2"):
        _parser().parse("LogSumExp(x[0])")


# =============================================================================
# Functions with Optional Parameters
# =============================================================================


def test_parse_huber_default():
    expr = _parser().parse("Huber(x)")
    assert isinstance(expr, Huber)
    assert expr.delta == 0.25  # default


def test_parse_huber_positional_delta():
    expr = _parser().parse("Huber(x, 0.5)")
    assert isinstance(expr, Huber)
    assert expr.delta == 0.5


def test_parse_huber_kwarg_delta():
    expr = _parser().parse("Huber(x, delta=1.0)")
    assert isinstance(expr, Huber)
    assert expr.delta == 1.0


def test_parse_smooth_relu_default():
    expr = _parser().parse("SmoothReLU(x)")
    assert isinstance(expr, SmoothReLU)


def test_parse_smooth_relu_positional_c():
    expr = _parser().parse("SmoothReLU(x, 0.01)")
    assert isinstance(expr, SmoothReLU)


def test_parse_smooth_relu_kwarg_c():
    expr = _parser().parse("SmoothReLU(x, c=0.01)")
    assert isinstance(expr, SmoothReLU)


# =============================================================================
# Interpolation Functions
# =============================================================================


def test_parse_linterp():
    xp = Constant(np.array([0.0, 1.0, 2.0]))
    fp = Constant(np.array([0.0, 1.0, 4.0]))
    p = _parser(xp=xp, fp=fp)
    expr = p.parse("Linterp(x[0], xp, fp)")
    assert isinstance(expr, Linterp)


def test_linterp_wrong_args():
    with pytest.raises(ValueError, match="exactly 3"):
        _parser().parse("Linterp(x)")


def test_parse_bilerp():
    xp = Constant(np.array([0.0, 1.0]))
    yp = Constant(np.array([0.0, 1.0]))
    fp = Constant(np.array([[0.0, 1.0], [2.0, 3.0]]))
    p = _parser(xp=xp, yp=yp, fp=fp)
    expr = p.parse("Bilerp(x[0], x[1], xp, yp, fp)")
    assert isinstance(expr, Bilerp)


def test_bilerp_wrong_args():
    with pytest.raises(ValueError, match="exactly 5"):
        _parser().parse("Bilerp(x[0], x[1])")


# =============================================================================
# Nested / Composed
# =============================================================================


def test_parse_sin_of_cos():
    expr = _parser().parse("Sin(Cos(x))")
    assert isinstance(expr, Sin)
    assert isinstance(expr.operand, Cos)


def test_parse_sqrt_of_square():
    expr = _parser().parse("Sqrt(Square(x[0]))")
    assert isinstance(expr, Sqrt)
    assert isinstance(expr.operand, Square)
