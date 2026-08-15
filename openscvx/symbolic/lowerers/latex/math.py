"""LaTeX visitors for math expressions.

Visitors: Sin, Cos, Tan, Asin, Acos, Atan, Atan2, Square, Sqrt, Exp, Log, Abs,
          Max, Min, PositivePart, Huber, SmoothReLU, LogSumExp, Linterp,
          Cinterp, Bilerp

Renders the scalar math AST nodes to LaTeX math strings. The elementary functions
render as ordinary function applications (``\\sin\\left( ... \\right)`` via
``_call``), and the penalty/interpolation atoms render as named operators. Unlike
the CVXPy backend, which admits only the DCP-representable subset, every math node
has a valid LaTeX form, since rendering imposes no convexity requirement.
"""

from openscvx.symbolic.expr.arithmetic import Power
from openscvx.symbolic.expr.math import (
    Abs,
    Acos,
    Asin,
    Atan,
    Atan2,
    Bilerp,
    Cinterp,
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
from openscvx.symbolic.lowerers.latex._lowerer import _PRECEDENCE, wrap
from openscvx.symbolic.lowerers.latex._registry import visitor


def _call(name: str, arg: str) -> str:
    """Render a function application ``\\name\\left( arg \\right)``."""
    return rf"{name}\left( {arg} \right)"


@visitor(Sin)
def _visit_sin(lowerer, node: Sin):
    """Render sine as ``\\sin(a)``."""
    return _call(r"\sin", lowerer.lower(node.operand))


@visitor(Cos)
def _visit_cos(lowerer, node: Cos):
    """Render cosine as ``\\cos(a)``."""
    return _call(r"\cos", lowerer.lower(node.operand))


@visitor(Tan)
def _visit_tan(lowerer, node: Tan):
    """Render tangent as ``\\tan(a)``."""
    return _call(r"\tan", lowerer.lower(node.operand))


@visitor(Asin)
def _visit_asin(lowerer, node: Asin):
    """Render arcsine as ``\\arcsin(a)``."""
    return _call(r"\arcsin", lowerer.lower(node.operand))


@visitor(Acos)
def _visit_acos(lowerer, node: Acos):
    """Render arccosine as ``\\arccos(a)``."""
    return _call(r"\arccos", lowerer.lower(node.operand))


@visitor(Atan)
def _visit_atan(lowerer, node: Atan):
    """Render arctangent as ``\\arctan(a)``."""
    return _call(r"\arctan", lowerer.lower(node.operand))


@visitor(Atan2)
def _visit_atan2(lowerer, node: Atan2):
    """Render two-argument arctangent as ``\\operatorname{atan2}(y, x)``."""
    return _call(r"\operatorname{atan2}", f"{lowerer.lower(node.y)}, {lowerer.lower(node.x)}")


@visitor(Square)
def _visit_square(lowerer, node: Square):
    """Render square as ``a^{2}``."""
    return f"{wrap(lowerer, node.x, _PRECEDENCE[Power])}^{{2}}"


@visitor(Sqrt)
def _visit_sqrt(lowerer, node: Sqrt):
    """Render square root as ``\\sqrt{a}``."""
    return rf"\sqrt{{{lowerer.lower(node.operand)}}}"


@visitor(Exp)
def _visit_exp(lowerer, node: Exp):
    """Render exponential as ``\\exp(a)``."""
    return _call(r"\exp", lowerer.lower(node.operand))


@visitor(Log)
def _visit_log(lowerer, node: Log):
    """Render natural logarithm as ``\\ln(a)``."""
    return _call(r"\ln", lowerer.lower(node.operand))


@visitor(Abs)
def _visit_abs(lowerer, node: Abs):
    """Render absolute value as ``\\left| a \\right|``."""
    return rf"\left| {lowerer.lower(node.operand)} \right|"


@visitor(Max)
def _visit_max(lowerer, node: Max):
    """Render element-wise maximum as ``\\max(a, b, ...)``."""
    return _call(r"\max", ", ".join(lowerer.lower(op) for op in node.operands))


@visitor(Min)
def _visit_min(lowerer, node: Min):
    """Render element-wise minimum as ``\\min(a, b, ...)``."""
    return _call(r"\min", ", ".join(lowerer.lower(op) for op in node.operands))


@visitor(PositivePart)
def _visit_positive_part(lowerer, node: PositivePart):
    """Render the positive part as ``\\left( a \\right)_{+}``."""
    return rf"\left( {lowerer.lower(node.x)} \right)_{{+}}"


@visitor(Huber)
def _visit_huber(lowerer, node: Huber):
    """Render the Huber penalty as ``\\operatorname{huber}_{\\delta}(x)``.

    The threshold ``delta`` (the quadratic-to-linear transition point) is
    carried as the subscript so distinct thresholds render distinctly.
    """
    delta = "%g" % node.delta
    return rf"\operatorname{{huber}}_{{{delta}}}\left( {lowerer.lower(node.x)} \right)"


@visitor(SmoothReLU)
def _visit_smooth_relu(lowerer, node: SmoothReLU):
    """Render the smooth ReLU as ``\\operatorname{smoothrelu}(x)``.

    The smoothing parameter ``c`` is evaluation machinery (it only rounds the
    kink) rather than part of the function's identity, so it is not shown.
    """
    return rf"\operatorname{{smoothrelu}}\left( {lowerer.lower(node.x)} \right)"


@visitor(LogSumExp)
def _visit_logsumexp(lowerer, node: LogSumExp):
    """Render log-sum-exp as ``\\operatorname{logsumexp}(a, b, ...)``."""
    args = ", ".join(lowerer.lower(op) for op in node.operands)
    return _call(r"\operatorname{logsumexp}", args)


@visitor(Linterp)
def _visit_linterp(lowerer, node: Linterp):
    """Render 1-D linear interpolation as ``\\operatorname{linterp}(x, xp, fp)``."""
    args = ", ".join(lowerer.lower(c) for c in node.children())
    return _call(r"\operatorname{linterp}", args)


@visitor(Cinterp)
def _visit_cinterp(lowerer, node: Cinterp):
    """Render 1-D piecewise-cubic interpolation as ``\\operatorname{cinterp}(x)``.

    Only the query point is rendered; the breakpoints and the polynomial table
    (whether baked-in coefficients or a symbolic override) are structural and
    stay implicit.
    """
    return _call(r"\operatorname{cinterp}", lowerer.lower(node.x))


@visitor(Bilerp)
def _visit_bilerp(lowerer, node: Bilerp):
    """Render 2-D bilinear interpolation as ``\\operatorname{bilerp}(x, y, xp, yp, fp)``."""
    args = ", ".join(lowerer.lower(c) for c in node.children())
    return _call(r"\operatorname{bilerp}", args)
