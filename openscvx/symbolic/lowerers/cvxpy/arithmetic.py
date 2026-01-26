"""CVXPy visitors for arithmetic expressions.

Visitors: Add, Sub, Mul, Div, MatMul, Neg, Power
"""

import cvxpy as cp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.arithmetic import Add, Div, MatMul, Mul, Neg, Power, Sub
from openscvx.symbolic.lowerers.cvxpy._registry import visitor  # noqa: F401


@visitor(Add)
def _visit_add(lowerer, node: Add) -> cp.Expression:
    """Lower addition to CVXPy expression.

    Recursively lowers all terms and composes them with element-wise addition.
    Addition is affine and always DCP-compliant.

    Args:
        node: Add expression node with multiple terms

    Returns:
        CVXPy expression representing the sum of all terms
    """
    terms = [lowerer.lower(term) for term in node.terms]
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return result


@visitor(Sub)
def _visit_sub(lowerer, node: Sub) -> cp.Expression:
    """Lower subtraction to CVXPy expression (element-wise left - right).

    Subtraction is affine and always DCP-compliant.

    Args:
        node: Sub expression node

    Returns:
        CVXPy expression representing left - right
    """
    left = lowerer.lower(node.left)
    right = lowerer.lower(node.right)
    return left - right


@visitor(Mul)
def _visit_mul(lowerer, node: Mul) -> cp.Expression:
    """Lower element-wise multiplication to CVXPy expression.

    Element-wise multiplication is DCP-compliant when at least one operand
    is constant. For quadratic forms, use MatMul instead.

    Args:
        node: Mul expression node with multiple factors

    Returns:
        CVXPy expression representing element-wise product

    Note:
        For convex optimization, typically one factor should be constant.
        CVXPy will raise a DCP error if the composition violates DCP rules.
    """
    factors = [lowerer.lower(factor) for factor in node.factors]
    result = factors[0]
    for factor in factors[1:]:
        result = result * factor
    return result


@visitor(Div)
def _visit_div(lowerer, node: Div) -> cp.Expression:
    """Lower element-wise division to CVXPy expression.

    Division is DCP-compliant when the denominator is constant or when
    the numerator is constant and the denominator is concave.

    Args:
        node: Div expression node

    Returns:
        CVXPy expression representing left / right

    Note:
        CVXPy will raise a DCP error if the division violates DCP rules.
    """
    left = lowerer.lower(node.left)
    right = lowerer.lower(node.right)
    return left / right


@visitor(MatMul)
def _visit_matmul(lowerer, node: MatMul) -> cp.Expression:
    """Lower matrix multiplication to CVXPy expression using @ operator.

    Matrix multiplication is DCP-compliant when at least one operand is
    constant. Used for quadratic forms like x.T @ Q @ x.

    Args:
        node: MatMul expression node

    Returns:
        CVXPy expression representing left @ right
    """
    left = lowerer.lower(node.left)
    right = lowerer.lower(node.right)
    return left @ right


@visitor(Neg)
def _visit_neg(lowerer, node: Neg) -> cp.Expression:
    """Lower negation (unary minus) to CVXPy expression.

    Negation preserves DCP properties (negating convex gives concave).

    Args:
        node: Neg expression node

    Returns:
        CVXPy expression representing -operand
    """
    operand = lowerer.lower(node.operand)
    return -operand


@visitor(Power)
def _visit_power(lowerer, node: Power) -> cp.Expression:
    """Lower element-wise power (base**exponent) to CVXPy expression.

    Power is DCP-compliant for specific exponent values:
    - exponent >= 1: convex (when base >= 0)
    - 0 <= exponent <= 1: concave (when base >= 0)

    Args:
        node: Power expression node

    Returns:
        CVXPy expression representing base**exponent

    Note:
        CVXPy will verify DCP compliance at problem construction time.
        Common convex cases: x^2, x^3, x^4 (even powers)
    """
    base = lowerer.lower(node.base)
    exponent = lowerer.lower(node.exponent)
    return cp.power(base, exponent)
