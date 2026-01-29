"""CVXPy visitors for logic expressions.

Visitors: All, Any, Cond
"""

import cvxpy as cp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.logic import All, Any, Cond
from openscvx.symbolic.lowerers.cvxpy._registry import visitor  # noqa: F401


@visitor(All)
def _visit_all(lowerer, node: All) -> cp.Expression:
    """Raise NotImplementedError for All expression.

    Logical AND reduction (All) is not DCP-compliant in CVXPy as it introduces
    non-convex boolean logic. All expressions are only supported in JAX lowering.

    Args:
        node: All expression node

    Raises:
        NotImplementedError: Always raised since logical reductions are not DCP-compliant
    """
    raise NotImplementedError(
        "Logical reduction expressions (All) are not DCP-compliant in CVXPy. "
        "Logical operations are only supported in JAX lowering."
    )


@visitor(Any)
def _visit_any(lowerer, node: Any) -> cp.Expression:
    """Raise NotImplementedError for Any expression.

    Logical OR reduction (Any) is not DCP-compliant in CVXPy as it introduces
    non-convex boolean logic. Any expressions are only supported in JAX lowering.

    Args:
        node: Any expression node

    Raises:
        NotImplementedError: Always raised since logical reductions are not DCP-compliant
    """
    raise NotImplementedError(
        "Logical reduction expressions (Any) are not DCP-compliant in CVXPy. "
        "Logical operations are only supported in JAX lowering."
    )


@visitor(Cond)
def _visit_cond(lowerer, node: Cond) -> cp.Expression:
    """Raise NotImplementedError for conditional expression.

    Conditional logic (Cond) is not DCP-compliant in CVXPy as it introduces
    non-convex branching behavior. Conditional expressions are only supported
    in JAX lowering for dynamics and non-convex constraints.

    Args:
        node: Cond expression node

    Raises:
        NotImplementedError: Always raised since conditional logic is not DCP-compliant

    Note:
        For conditional constraints:
        - Use piecewise-linear approximations, or
        - Handle in the JAX dynamics/constraint layer instead of CVXPy
    """
    raise NotImplementedError(
        "Conditional expressions (Cond) are not DCP-compliant in CVXPy. "
        "Conditional logic is only supported in JAX lowering. Consider using "
        "piecewise-linear approximations or handle these constraints in the "
        "dynamics (JAX) layer instead."
    )
