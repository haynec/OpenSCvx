"""CVXPy visitors for state/time expressions.

Visitors: State, Time

Lowers a ``State`` leaf to a slice of the unified state variable ``x`` held in the
lowerer's ``variable_map``, using the slice assigned during unification. ``Time``
is a ``State`` subclass and shares the visitor; if the state carries no slice the
whole ``x`` is returned, and a missing ``x`` in the map is a usage error and
raises.
"""

import cvxpy as cp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.state import State
from openscvx.symbolic.expr.time import Time
from openscvx.symbolic.lowerers.cvxpy._registry import visitor  # noqa: F401


@visitor(Time)
@visitor(State)
def _visit_state(lowerer, node: State) -> cp.Expression:
    """Lower a state variable to a CVXPy expression.

    Extracts the appropriate slice from the unified state vector "x" using
    the slice assigned during unification. The "x" variable must exist in
    the variable_map.

    Args:
        node: State expression node (or Time, which is a State subclass)

    Returns:
        CVXPy expression representing the state slice: x[slice]

    Raises:
        ValueError: If "x" is not found in variable_map
    """
    if "x" not in lowerer.variable_map:
        raise ValueError("State vector 'x' not found in variable_map.")

    cvx_var = lowerer.variable_map["x"]

    # If the state has a slice assigned, apply it
    if node._slice is not None:
        return cvx_var[node._slice]
    return cvx_var
