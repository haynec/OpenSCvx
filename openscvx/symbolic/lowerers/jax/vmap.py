"""JAX visitors for vmap expressions.

Visitors: _Placeholder, Vmap
"""

import jax
import jax.numpy as jnp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.vmap import Vmap, _Placeholder
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(_Placeholder)
def _visit_placeholder(lowerer, node: _Placeholder):
    """Lower Placeholder to params lookup.

    Placeholder is used inside Vmap expressions. During lowering, the Vmap
    visitor injects the current batch element into params, and this visitor
    retrieves it.

    Args:
        node: Placeholder expression node

    Returns:
        Function (x, u, node, params) -> params[placeholder_name]
    """
    name = node.name
    return lambda x, u, node_idx, params: params[name]


@visitor(Vmap)
def _visit_vmap(lowerer, node: Vmap):
    """Lower Vmap to jax.vmap.

    Handles multiple cases based on the type of each data source:

    - **Constant/array**: Data is baked into the closure at lowering time,
        equivalent to closure-captured values in BYOF.
    - **Parameter**: Data is looked up from params dict at runtime,
        allowing updates between SCP iterations.
    - **State**: Data is extracted from the unified state vector x at runtime,
        enabling vectorized operations over state elements.
    - **Control**: Data is extracted from the unified control vector u at runtime,
        enabling vectorized operations over control elements.

    Supports multiple batch arguments, each mapped to a corresponding
    placeholder in the inner expression.

    Args:
        node: Vmap expression node

    Returns:
        Function (x, u, node_idx, params) -> vmapped result

    Example:
        For ox.Vmap(lambda p: ox.linalg.Norm(x - p), batch=points):
        - points has shape (10, 3)
        - Output has shape (10,) - one norm per point

        For ox.Vmap(lambda c, r: r <= norm(x - c), batch=[centers, radii]):
        - centers has shape (100, 3), radii has shape (100,)
        - Output has shape (100,) - one result per center/radius pair

        For ox.Vmap(lambda pos: g(pos), batch=agent_positions):
        - agent_positions is a State with shape (n_agents, 3)
        - Output has shape (n_agents,) - one result per agent
    """
    inner_fn = lowerer.lower(node._child)
    axis = node._axis
    num_batches = node.num_batches

    # Collect placeholder keys and classify batches
    placeholder_keys = tuple(p.name for p in node._placeholders)

    # Check if any batch requires runtime lookup (Parameter, State, or Control)
    any_runtime = any(node._is_parameter) or any(node._is_state) or any(node._is_control)

    if any_runtime:
        # At least one runtime batch: need to gather data at runtime
        # Build lookup info for each batch
        # Format: (kind, key_or_slice, baked_data, original_shape)
        batch_info = []
        for b, is_param, is_state, is_control in zip(
            node._batches, node._is_parameter, node._is_state, node._is_control
        ):
            if is_param:
                batch_info.append(("param", b.name, None, None))
            elif is_state:
                sl = b._slice
                if sl is None:
                    raise ValueError(f"State {b.name!r} has no slice assigned")
                # Store original shape for reshaping after extraction
                batch_info.append(("state", sl, None, b.shape))
            elif is_control:
                sl = b._slice
                if sl is None:
                    raise ValueError(f"Control {b.name!r} has no slice assigned")
                # Store original shape for reshaping after extraction
                batch_info.append(("control", sl, None, b.shape))
            else:
                # Constant: bake the data
                batch_info.append(("constant", None, jnp.array(b.value), None))

        # Freeze for closure
        batch_info = tuple(batch_info)

        def vmapped_fn(x, u, node_idx, params):
            # Gather data from appropriate sources
            data_arrays = []
            for kind, key_or_slice, baked, orig_shape in batch_info:
                if kind == "param":
                    data_arrays.append(params[key_or_slice])
                elif kind == "state":
                    # Extract from unified state and reshape to original shape
                    data = x[key_or_slice].reshape(orig_shape)
                    data_arrays.append(data)
                elif kind == "control":
                    # Extract from unified control and reshape to original shape
                    data = u[key_or_slice].reshape(orig_shape)
                    data_arrays.append(data)
                else:  # constant
                    data_arrays.append(baked)

            def inner(*vs):
                # Inject all placeholder values into params
                new_params = {**params}
                for key, v in zip(placeholder_keys, vs):
                    new_params[key] = v
                return inner_fn(x, u, node_idx, new_params)

            # vmap over all batch arguments along the same axis
            in_axes = tuple(axis for _ in range(num_batches))
            return jax.vmap(inner, in_axes=in_axes)(*data_arrays)

    else:
        # All Constants: bake all data into closure at lowering time
        baked_data = tuple(jnp.array(b.value) for b in node._batches)

        def vmapped_fn(x, u, node_idx, params):
            def inner(*vs):
                # Inject all placeholder values into params
                new_params = {**params}
                for key, v in zip(placeholder_keys, vs):
                    new_params[key] = v
                return inner_fn(x, u, node_idx, new_params)

            # vmap over all batch arguments along the same axis
            in_axes = tuple(axis for _ in range(num_batches))
            return jax.vmap(inner, in_axes=in_axes)(*baked_data)

    return vmapped_fn
