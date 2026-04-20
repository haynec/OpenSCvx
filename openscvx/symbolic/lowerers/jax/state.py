"""JAX visitors for state/time expressions.

Visitors: State, Time
"""

import jax

from openscvx.symbolic.expr.state import State
from openscvx.symbolic.expr.time import Time
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(Time)
@visitor(State)
def _visit_state(lowerer, node: State):
    """Lower a state variable to a JAX function.

    Extracts the appropriate slice from the unified state vector x using
    the slice assigned during unification. For STM leaves in ``"approx"``
    mode the read is wrapped in ``jax.lax.stop_gradient`` so the SCP
    Jacobian treats Φ / Φ_imp as frozen inputs (first-order robustification).

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
    if getattr(node, "_is_stm", False) and getattr(node, "mode", "approx") == "approx":
        return lambda x, u, node, params: jax.lax.stop_gradient(x[sl])
    return lambda x, u, node, params: x[sl]
