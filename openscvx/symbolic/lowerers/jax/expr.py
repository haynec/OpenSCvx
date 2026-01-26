"""JAX visitors for core expression types.

Visitors: Constant, Parameter, NodeReference
"""

import jax.numpy as jnp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.expr import Constant, NodeReference, Parameter
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(Constant)
def _visit_constant(lowerer, node: Constant):
    """Lower a constant value to a JAX function.

    Captures the constant value and returns a function that always returns it.
    Scalar constants are squeezed to ensure they're true scalars, not (1,) arrays.

    Args:
        node: Constant expression node

    Returns:
        Function (x, u, node, params) -> constant_value
    """
    # capture the constant value once
    value = jnp.array(node.value)
    # For scalar constants (single element arrays), squeeze to scalar
    # This prevents (1,) shapes in constraint residuals
    if value.size == 1:
        value = value.squeeze()
    return lambda x, u, node, params: value


@visitor(Parameter)
def _visit_parameter(lowerer, node: Parameter):
    """Lower a parameter to a JAX function.

    Parameters are looked up by name in the params dictionary at evaluation time,
    allowing runtime parameter updates without recompilation.

    Args:
        node: Parameter expression node

    Returns:
        Function (x, u, node, params) -> params[name]
    """
    param_name = node.name
    return lambda x, u, node, params: jnp.array(params[param_name])


@visitor(NodeReference)
def _visit_node_reference(lowerer, node: NodeReference):
    """Lower NodeReference - extract value at a specific trajectory node.

    NodeReference extracts a state/control value at a specific node from the
    full trajectory arrays. The node index is baked into the lowered function.

    Args:
        node: NodeReference expression with base and node_idx (integer)

    Returns:
        Function (x, u, node_param, params) that extracts from trajectory
            - x, u: Full trajectories (N, n_x) and (N, n_u)
            - node_param: Unused (kept for signature compatibility)
            - params: Problem parameters

    Example:
        position.at(5) lowers to a function that extracts x[5, position_slice]
        position.at(k-1) where k=7 lowers to extract x[6, position_slice]
    """
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State

    # Node index is baked into the expression at construction time
    fixed_idx = node.node_idx

    if isinstance(node.base, State):
        sl = node.base._slice
        if sl is None:
            raise ValueError(f"State {node.base.name!r} has no slice assigned")

        def state_node_fn(x, u, node_param, params):
            return x[fixed_idx, sl]

        return state_node_fn

    elif isinstance(node.base, Control):
        sl = node.base._slice
        if sl is None:
            raise ValueError(f"Control {node.base.name!r} has no slice assigned")

        def control_node_fn(x, u, node_param, params):
            return u[fixed_idx, sl]

        return control_node_fn

    else:
        # Compound expression (e.g., position[0].at(5))
        base_fn = lowerer.lower(node.base)

        def compound_node_fn(x, u, node_param, params):
            # Extract single-node slices and evaluate base expression
            x_single = x[fixed_idx] if len(x.shape) > 1 else x
            u_single = u[fixed_idx] if len(u.shape) > 1 else u
            return base_fn(x_single, u_single, fixed_idx, params)

        return compound_node_fn
