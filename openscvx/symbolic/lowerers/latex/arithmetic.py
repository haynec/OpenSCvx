"""LaTeX visitors for arithmetic expressions.

Visitors: Add, Sub, Mul, Div, MatMul, Neg, Power
"""

from openscvx.symbolic.expr.arithmetic import Add, Div, MatMul, Mul, Neg, Power, Sub
from openscvx.symbolic.lowerers.latex._lowerer import _PRECEDENCE, wrap
from openscvx.symbolic.lowerers.latex._registry import visitor


@visitor(Add)
def _visit_add(lowerer, node: Add):
    """Render addition as ``a + b + ...``."""
    return " + ".join(wrap(lowerer, t, _PRECEDENCE[Add]) for t in node.terms)


@visitor(Sub)
def _visit_sub(lowerer, node: Sub):
    """Render subtraction as ``a - b``.

    The right operand is wrapped at one precedence level higher so an
    equal-precedence sum/difference is parenthesized (``a - (b + c)``).
    """
    left = wrap(lowerer, node.left, _PRECEDENCE[Sub])
    right = wrap(lowerer, node.right, _PRECEDENCE[Sub] + 1)
    return f"{left} - {right}"


@visitor(Mul)
def _visit_mul(lowerer, node: Mul):
    """Render element-wise multiplication as ``a \\cdot b \\cdot ...``."""
    return r" \cdot ".join(wrap(lowerer, f, _PRECEDENCE[Mul]) for f in node.factors)


@visitor(Div)
def _visit_div(lowerer, node: Div):
    """Render division as a ``\\frac`` (self-delimiting, no wrapping needed)."""
    return rf"\frac{{{lowerer.lower(node.left)}}}{{{lowerer.lower(node.right)}}}"


@visitor(MatMul)
def _visit_matmul(lowerer, node: MatMul):
    """Render matrix multiplication as juxtaposition ``A B``."""
    left = wrap(lowerer, node.left, _PRECEDENCE[MatMul])
    right = wrap(lowerer, node.right, _PRECEDENCE[MatMul])
    return f"{left} {right}"


@visitor(Neg)
def _visit_neg(lowerer, node: Neg):
    """Render negation as ``-a``."""
    return f"-{wrap(lowerer, node.operand, _PRECEDENCE[Neg])}"


@visitor(Power)
def _visit_power(lowerer, node: Power):
    """Render power as ``base^{exp}`` (exponent is self-delimiting in braces)."""
    return f"{wrap(lowerer, node.base, _PRECEDENCE[Power])}^{{{lowerer.lower(node.exponent)}}}"
