"""CVXPy visitors for array expressions.

Visitors: Index, Concat, Stack, Hstack, Vstack, Block
"""

import cvxpy as cp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.array import Block, Concat, Hstack, Index, Stack, Vstack
from openscvx.symbolic.lowerers.cvxpy._registry import visitor  # noqa: F401


@visitor(Index)
def _visit_index(lowerer, node: Index) -> cp.Expression:
    """Lower indexing/slicing operation to CVXPy expression.

    Indexing preserves DCP properties (indexing into convex is convex).

    Args:
        node: Index expression node

    Returns:
        CVXPy expression representing base[index]
    """
    base = lowerer.lower(node.base)
    return base[node.index]


@visitor(Concat)
def _visit_concat(lowerer, node: Concat) -> cp.Expression:
    """Lower concatenation to CVXPy expression.

    Concatenates expressions horizontally along axis 0. Scalars are
    promoted to 1D arrays before concatenation. Preserves DCP properties.

    Args:
        node: Concat expression node

    Returns:
        CVXPy expression representing horizontal concatenation

    Note:
        Uses cp.hstack for concatenation. Scalars are reshaped to (1,).
    """
    exprs = [lowerer.lower(child) for child in node.exprs]
    # Ensure all expressions are at least 1D for concatenation
    exprs_1d = []
    for expr in exprs:
        if expr.ndim == 0:  # scalar
            exprs_1d.append(cp.reshape(expr, (1,), order="C"))
        else:
            exprs_1d.append(expr)
    return cp.hstack(exprs_1d)


@visitor(Stack)
def _visit_stack(lowerer, node: Stack) -> cp.Expression:
    """Lower vertical stacking to CVXPy expression.

    Stacks expressions vertically using cp.vstack. Preserves DCP properties.

    Args:
        node: Stack expression node with multiple rows

    Returns:
        CVXPy expression representing vertical stack of rows

    Note:
        Each row is stacked along axis 0 to create a 2D array.
    """
    rows = [lowerer.lower(row) for row in node.rows]
    # Stack rows vertically
    return cp.vstack(rows)


@visitor(Hstack)
def _visit_hstack(lowerer, node: Hstack) -> cp.Expression:
    """Lower horizontal stacking to CVXPy expression.

    For 1D arrays, uses cp.hstack (concatenation). For 2D+ arrays, uses
    cp.bmat with a single row to achieve proper horizontal stacking along
    axis 1, matching numpy.hstack semantics.

    Args:
        node: Hstack expression node with multiple arrays

    Returns:
        CVXPy expression representing horizontal stack of arrays
    """
    arrays = [lowerer.lower(arr) for arr in node.arrays]

    # Check dimensionality from the symbolic node's shape
    shape = node.check_shape()
    if len(shape) == 1:
        # 1D: simple concatenation
        return cp.hstack(arrays)
    else:
        # 2D+: use bmat with single row for proper horizontal stacking
        return cp.bmat([arrays])


@visitor(Vstack)
def _visit_vstack(lowerer, node: Vstack) -> cp.Expression:
    """Lower vertical stacking to CVXPy expression.

    Stacks expressions vertically using cp.vstack. Preserves DCP properties.

    Args:
        node: Vstack expression node with multiple arrays

    Returns:
        CVXPy expression representing vertical stack of arrays
    """
    arrays = [lowerer.lower(arr) for arr in node.arrays]
    return cp.vstack(arrays)


@visitor(Block)
def _visit_block(lowerer, node: Block) -> cp.Expression:
    """Lower block matrix construction to CVXPy expression.

    Assembles a block matrix from nested lists of expressions using cp.bmat.
    This is the CVXPy equivalent of numpy.block() for block matrix construction.

    Args:
        node: Block expression node with 2D nested structure of expressions

    Returns:
        CVXPy expression representing the assembled block matrix

    Raises:
        NotImplementedError: If any block has more than 2 dimensions

    Note:
        cp.bmat preserves DCP properties when all blocks are DCP-compliant.
        Block matrices are commonly used for constraint aggregation.
        For 3D+ tensors, use JAX lowering instead.
    """
    # Check for 3D+ blocks - CVXPy's bmat only supports 2D
    for i, row in enumerate(node.blocks):
        for j, block in enumerate(row):
            block_shape = block.check_shape()
            if len(block_shape) > 2:
                raise NotImplementedError(
                    f"CVXPy does not support Block with tensors of dimension > 2. "
                    f"Block[{i}][{j}] has shape {block_shape} ({len(block_shape)}D). "
                    f"For N-D tensor block assembly, use JAX lowering instead."
                )

    # Lower each block expression
    block_exprs = [[lowerer.lower(block) for block in row] for row in node.blocks]
    return cp.bmat(block_exprs)
