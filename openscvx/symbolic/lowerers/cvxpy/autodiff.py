"""CVXPy visitors for user-supplied Jacobians.

Visitors: WithJacobian

A ``WithJacobian`` override reshapes a *linearization*, and linearization happens
on the JAX path. The convex subproblem sees convex constraints verbatim, with no
derivative taken, so the wrapper has nothing to act on here and raises
``NotImplementedError`` — the same treatment ``CTCS`` gets for being a
JAX-side construct.
"""

import cvxpy as cp

from openscvx.symbolic.expr.autodiff import WithJacobian
from openscvx.symbolic.lowerers.cvxpy._registry import visitor  # noqa: F401


@visitor(WithJacobian)
def _visit_with_jacobian(lowerer, node: WithJacobian) -> cp.Expression:
    """Raise NotImplementedError for a Jacobian override inside a convex constraint.

    Args:
        node: WithJacobian expression node

    Raises:
        NotImplementedError: Always — see the module docstring.
    """
    raise NotImplementedError(
        "`.with_jacobian(...)` overrides the derivative used when a nonconvex "
        "expression is linearized, which only happens on the JAX path (dynamics and "
        "nonconvex constraints). This expression was lowered to CVXPy as part of a "
        "convex constraint, which is passed to the solver as written and never "
        "differentiated, so the override has no meaning here. Apply "
        "`.with_jacobian(...)` to the nonconvex term instead, or drop it from this "
        "constraint."
    )
