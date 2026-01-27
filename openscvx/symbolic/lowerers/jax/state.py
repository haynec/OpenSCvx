"""JAX visitors for state/time expressions.

Visitors: State, Time
"""

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.state import State
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401
from openscvx.symbolic.time import Time


@visitor(Time)
@visitor(State)
def _visit_state(lowerer, node: State):
    """Lower a state variable to a JAX function.

    Extracts the appropriate slice from the unified state vector x using
    the slice assigned during unification.

    Args:
        node: State expression node (or Time, which is a State subclass)

    Returns:
        Function (x, u, node, params) -> x[slice]

    Raises:
        ValueError: If the state has no slice assigned (unification not run)
    """
    sl = node._slice
    if sl is None:
        raise ValueError(f"State {node.name!r} has no slice assigned")
    return lambda x, u, node, params: x[sl]
