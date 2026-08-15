"""JAX visitors for state/time expressions.

Visitors: State, Time, Tau

Lowers a ``State`` leaf to a JAX function that slices the corresponding entries
out of the unified state vector ``x``, using the slice assigned during
unification. ``Time`` is a ``State`` subclass and shares the same visitor;
lowering a state whose slice is still unset is a usage error and raises.
``Tau`` — the normalized node coordinate used by initial-guess expressions —
reads its value off the node grid carried in ``params``.
"""

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.state import State
from openscvx.symbolic.expr.time import TAU_PARAM, Tau, Time
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


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


@visitor(Tau)
def _visit_tau(lowerer, node: Tau):
    """Lower the normalized node coordinate to a JAX function.

    The tau grid is supplied by the guess-resolution pass under the reserved
    ``params`` key, so tau is read at the node currently being evaluated.

    Args:
        node: Tau expression node

    Returns:
        Function (x, u, node, params) -> tau at that node
    """
    return lambda x, u, node, params: params[TAU_PARAM][node]
