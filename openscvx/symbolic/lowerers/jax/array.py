"""JAX visitors for array expressions.

Visitors: Index, Concat, Stack, Hstack, Vstack, Block
"""

import jax.numpy as jnp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.array import Block, Concat, Hstack, Index, Stack, Vstack
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(Index)
def _visit_index(lowerer, node: Index):
    """Lower indexing/slicing operation to JAX function.

    For multi-dimensional indexing, the base array is reshaped to its
    original shape before applying the index. This is necessary because
    State variables are stored flattened in the state vector.
    """
    f_base = lowerer.lower(node.base)
    idx = node.index
    base_shape = node.base.check_shape()

    def index_fn(x, u, node_arg, params):
        arr = f_base(x, u, node_arg, params)
        # Reshape to original shape for multi-dimensional indexing
        if len(base_shape) > 1:
            arr = arr.reshape(base_shape)
        else:
            arr = jnp.atleast_1d(arr)
        return arr[idx]

    return index_fn


@visitor(Concat)
def _visit_concat(lowerer, node: Concat):
    """Lower concatenation to JAX function (concatenates along axis 0)."""
    # lower each child
    fn_list = [lowerer.lower(child) for child in node.exprs]

    # wrapper that promotes scalars to 1-D and concatenates
    def concat_fn(x, u, node, params):
        parts = [jnp.atleast_1d(fn(x, u, node, params)) for fn in fn_list]
        return jnp.concatenate(parts, axis=0)

    return concat_fn


@visitor(Stack)
def _visit_stack(lowerer, node: Stack):
    """Lower vertical stacking to JAX function (stack along axis 0)."""
    row_fns = [lowerer.lower(row) for row in node.rows]

    def stack_fn(x, u, node, params):
        rows = [jnp.atleast_1d(fn(x, u, node, params)) for fn in row_fns]
        return jnp.stack(rows, axis=0)

    return stack_fn


@visitor(Hstack)
def _visit_hstack(lowerer, node: Hstack):
    """Lower horizontal stacking to JAX function."""
    array_fns = [lowerer.lower(arr) for arr in node.arrays]

    def hstack_fn(x, u, node, params):
        arrays = [jnp.atleast_1d(fn(x, u, node, params)) for fn in array_fns]
        return jnp.hstack(arrays)

    return hstack_fn


@visitor(Vstack)
def _visit_vstack(lowerer, node: Vstack):
    """Lower vertical stacking to JAX function."""
    array_fns = [lowerer.lower(arr) for arr in node.arrays]

    def vstack_fn(x, u, node, params):
        arrays = [jnp.atleast_1d(fn(x, u, node, params)) for fn in array_fns]
        return jnp.vstack(arrays)

    return vstack_fn


@visitor(Block)
def _visit_block(lowerer, node: Block):
    """Lower block matrix construction to JAX function.

    Assembles a block matrix from nested lists of expressions. For 2D blocks,
    uses jnp.block directly. For N-D blocks (3D+), manually assembles along
    the first two dimensions using concatenate, since jnp.block concatenates
    along the last axes (not what we want for block matrix semantics).

    Args:
        node: Block expression node with 2D nested structure of expressions

    Returns:
        Function (x, u, node, params) -> assembled block matrix/tensor
    """
    # Lower each block expression
    block_fns = [[lowerer.lower(block) for block in row] for row in node.blocks]

    def block_fn(x, u, node_arg, params):
        # Evaluate all blocks
        block_values = [
            [jnp.atleast_1d(fn(x, u, node_arg, params)) for fn in row] for row in block_fns
        ]

        # Check if any block is 3D+ (need manual assembly)
        max_ndim = max(arr.ndim for row in block_values for arr in row)

        if max_ndim <= 2:
            # For 2D, jnp.block works correctly
            return jnp.block(block_values)
        else:
            # For N-D, manually assemble along axes 0 and 1
            # First, ensure all blocks have the same number of dimensions
            def promote_to_ndim(arr, target_ndim):
                while arr.ndim < target_ndim:
                    arr = jnp.expand_dims(arr, axis=0)
                return arr

            block_values = [[promote_to_ndim(arr, max_ndim) for arr in row] for row in block_values]

            # Concatenate each row along axis 1 (horizontal)
            row_results = [jnp.concatenate(row, axis=1) for row in block_values]
            # Concatenate rows along axis 0 (vertical)
            return jnp.concatenate(row_results, axis=0)

    return block_fn
