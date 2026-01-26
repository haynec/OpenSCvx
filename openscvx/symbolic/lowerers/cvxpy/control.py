"""CVXPy visitors for control expressions.

Visitors: Control
"""

import cvxpy as cp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.lowerers.cvxpy._registry import visitor  # noqa: F401


@visitor(Control)
def _visit_control(lowerer, node: Control) -> cp.Expression:
    """Lower a control variable to a CVXPy expression.

    Extracts the appropriate slice from the unified control vector "u" using
    the slice assigned during unification. The "u" variable must exist in
    the variable_map.

    Args:
        node: Control expression node

    Returns:
        CVXPy expression representing the control slice: u[slice]

    Raises:
        ValueError: If "u" is not found in variable_map
    """
    if "u" not in lowerer.variable_map:
        raise ValueError("Control vector 'u' not found in variable_map.")

    cvx_var = lowerer.variable_map["u"]

    # If the control has a slice assigned, apply it
    if node._slice is not None:
        return cvx_var[node._slice]
    return cvx_var
