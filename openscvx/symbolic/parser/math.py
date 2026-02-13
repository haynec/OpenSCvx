"""Parser handlers for mathematical functions.

Handlers: Sin, Cos, Tan, Sqrt, Square, Exp, Log, Abs, Max, Min,
          PositivePart, Huber, SmoothReLU, LogSumExp, Linterp, Bilerp
"""

from openscvx.symbolic.expr.expr import Constant
from openscvx.symbolic.expr.math import (
    Abs,
    Bilerp,
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
    Tan,
)
from openscvx.symbolic.parser._registry import function


def _as_float(val):
    """Extract a Python float from a value or Constant."""
    if isinstance(val, Constant) and val.value.ndim == 0:
        return float(val.value)
    return float(val)


@function("Sin")
def _parse_sin(args, kwargs):
    if len(args) != 1:
        raise ValueError("Sin() takes exactly 1 argument")
    return Sin(args[0])


@function("Cos")
def _parse_cos(args, kwargs):
    if len(args) != 1:
        raise ValueError("Cos() takes exactly 1 argument")
    return Cos(args[0])


@function("Tan")
def _parse_tan(args, kwargs):
    if len(args) != 1:
        raise ValueError("Tan() takes exactly 1 argument")
    return Tan(args[0])


@function("Sqrt")
def _parse_sqrt(args, kwargs):
    if len(args) != 1:
        raise ValueError("Sqrt() takes exactly 1 argument")
    return Sqrt(args[0])


@function("Square")
def _parse_square(args, kwargs):
    if len(args) != 1:
        raise ValueError("Square() takes exactly 1 argument")
    return Square(args[0])


@function("Exp")
def _parse_exp(args, kwargs):
    if len(args) != 1:
        raise ValueError("Exp() takes exactly 1 argument")
    return Exp(args[0])


@function("Log")
def _parse_log(args, kwargs):
    if len(args) != 1:
        raise ValueError("Log() takes exactly 1 argument")
    return Log(args[0])


@function("Abs")
def _parse_abs(args, kwargs):
    if len(args) != 1:
        raise ValueError("Abs() takes exactly 1 argument")
    return Abs(args[0])


@function("Max")
def _parse_max(args, kwargs):
    if len(args) < 2:
        raise ValueError("Max() requires at least 2 arguments")
    return Max(*args)


@function("Min")
def _parse_min(args, kwargs):
    if len(args) < 2:
        raise ValueError("Min() requires at least 2 arguments")
    return Min(*args)


@function("PositivePart")
def _parse_positive_part(args, kwargs):
    if len(args) != 1:
        raise ValueError("PositivePart() takes exactly 1 argument")
    return PositivePart(args[0])


@function("Huber")
def _parse_huber(args, kwargs):
    if len(args) < 1:
        raise ValueError("Huber() requires at least 1 argument")
    x = args[0]
    delta = kwargs.get("delta", 0.25)
    if len(args) > 1:
        delta = _as_float(args[1])
    elif "delta" in kwargs:
        delta = _as_float(kwargs["delta"])
    return Huber(x, delta=float(delta))


@function("SmoothReLU")
def _parse_smooth_relu(args, kwargs):
    if len(args) < 1:
        raise ValueError("SmoothReLU() requires at least 1 argument")
    x = args[0]
    c = kwargs.get("c", 1e-8)
    if len(args) > 1:
        c = _as_float(args[1])
    elif "c" in kwargs:
        c = _as_float(kwargs["c"])
    return SmoothReLU(x, c=float(c))


@function("LogSumExp")
def _parse_logsumexp(args, kwargs):
    if len(args) < 2:
        raise ValueError("LogSumExp() requires at least 2 arguments")
    return LogSumExp(*args)


@function("Linterp")
def _parse_linterp(args, kwargs):
    if len(args) != 3:
        raise ValueError("Linterp() takes exactly 3 arguments (x, xp, fp)")
    return Linterp(args[0], args[1], args[2])


@function("Bilerp")
def _parse_bilerp(args, kwargs):
    if len(args) != 5:
        raise ValueError("Bilerp() takes exactly 5 arguments (x, y, xp, yp, fp)")
    return Bilerp(args[0], args[1], args[2], args[3], args[4])
