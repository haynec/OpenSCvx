"""CVXPy visitors for linear algebra expressions.

Visitors: Sum, Norm, Transpose, Inv
"""

import cvxpy as cp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.linalg import Inv, Norm, Sum, Transpose
from openscvx.symbolic.lowerers.cvxpy._registry import visitor  # noqa: F401


@visitor(Sum)
def _visit_sum(lowerer, node: Sum) -> cp.Expression:
    """Lower sum reduction to CVXPy expression (sums all elements).

    Sum preserves DCP properties (sum of convex is convex).

    Args:
        node: Sum expression node

    Returns:
        CVXPy scalar expression representing the sum of all elements
    """
    operand = lowerer.lower(node.operand)
    return cp.sum(operand)


@visitor(Norm)
def _visit_norm(lowerer, node: Norm) -> cp.Expression:
    """Lower norm operation to CVXPy expression.

    Norms are convex functions and commonly used in convex optimization.
    Supports all CVXPy norm types (1, 2, inf, "fro", etc.).

    Args:
        node: Norm expression node with ord attribute

    Returns:
        CVXPy expression representing the norm of the operand

    Note:
        Common norms: ord=2 (Euclidean), ord=1 (Manhattan), ord="inf"
    """
    operand = lowerer.lower(node.operand)
    return cp.norm(operand, node.ord)


@visitor(Transpose)
def _visit_transpose(lowerer, node: Transpose) -> cp.Expression:
    """Lower matrix transpose to CVXPy expression.

    Transpose preserves DCP properties (transpose of convex is convex).

    Args:
        node: Transpose expression node

    Returns:
        CVXPy expression representing operand.T
    """
    operand = lowerer.lower(node.operand)
    return operand.T


@visitor(Inv)
def _visit_inv(lowerer, node: Inv) -> cp.Expression:
    """Raise NotImplementedError for matrix inverse.

    Matrix inverse is not DCP-compliant in CVXPy as it is neither convex
    nor concave for variable matrices.

    Args:
        node: Inv expression node

    Raises:
        NotImplementedError: Always raised since matrix inverse is not DCP-compliant

    Note:
        For optimization problems requiring matrix inverse:
        - If the matrix is constant/parameter, compute the inverse numerically
            before passing to CVXPy
        - Handle matrix inverse in the JAX dynamics/constraint layer instead
        - Consider reformulating the problem to avoid explicit matrix inverse
    """
    raise NotImplementedError(
        "Matrix inverse (Inv) is not DCP-compliant in CVXPy. "
        "inv(X) is neither convex nor concave for variable matrices. "
        "Consider: (1) computing the inverse numerically if the matrix is constant, "
        "(2) handling this in the JAX layer instead, or "
        "(3) reformulating the problem to avoid explicit matrix inverse."
    )
