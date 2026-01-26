"""JAX visitors for control expressions.

Visitors: Control
"""

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(Control)
def _visit_control(lowerer, node: Control):
    """Lower a control variable to a JAX function.

    Extracts the appropriate slice from the unified control vector u using
    the slice assigned during unification.

    Args:
        node: Control expression node

    Returns:
        Function (x, u, node, params) -> u[slice]

    Raises:
        ValueError: If the control has no slice assigned (unification not run)
    """
    sl = node._slice
    if sl is None:
        raise ValueError(f"Control {node.name!r} has no slice assigned")
    return lambda x, u, node, params: u[sl]
