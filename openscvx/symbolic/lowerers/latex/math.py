"""LaTeX visitors for math expressions.

Visitors: Sin, Cos, Tan, Asin, Acos, Atan, Atan2, Square, Sqrt, Exp, Log, Abs,
          Max, Min, PositivePart
"""

from openscvx.symbolic.expr.arithmetic import Power
from openscvx.symbolic.expr.math import (
    Abs,
    Acos,
    Asin,
    Atan,
    Atan2,
    Cos,
    Exp,
    Log,
    Max,
    Min,
    PositivePart,
    Sin,
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
