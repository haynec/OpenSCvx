"""Static sparsity analysis for Jacobian matrices via AST traversal.

Derives boolean sparsity patterns for dynamics and constraint Jacobians
by walking the expression tree and collecting which State/Control slices
each sub-expression depends on. This is a conservative (superset) analysis:
it may report a dependency that vanishes numerically, but will never miss one.

The primary entry points are:

- ``jacobian_sparsity``: 2-D Jacobian sparsity pattern for any expression
- ``cross_node_sparsity``: per-node patterns for cross-node constraints
"""

from typing import Tuple

import numpy as np

from openscvx.symbolic.expr import (
    Concat,
    Control,
    Expr,
    NodeReference,
    State,
    traverse,
)


def _leaf_masks(
    expr: Expr,
    n_x: int,
    n_u: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flat boolean masks of which state/control indices appear in *expr*.

    Returns 1-D arrays ``(x_mask[n_x], u_mask[n_u])``.
    """
    x_mask = np.zeros(n_x, dtype=bool)
    u_mask = np.zeros(n_u, dtype=bool)

    def _collect(node: Expr) -> None:
        if isinstance(node, State) and node._slice is not None:
            x_mask[node._slice] = True
        elif isinstance(node, Control) and node._slice is not None:
            u_mask[node._slice] = True

    traverse(expr, _collect)
    return x_mask, u_mask


def _output_dim(expr: Expr) -> int:
    """Number of scalar outputs of *expr* (product of its shape, min 1)."""
    shape = expr.check_shape()
    return int(np.prod(shape)) if shape else 1


def jacobian_sparsity(
    expr: Expr,
    n_x: int,
    n_u: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """2-D boolean sparsity patterns for df/dx and df/du.

    For a vector-valued expression ``f`` with *n_out* output elements,
    returns boolean matrices of shape ``(n_out, n_x)`` and ``(n_out, n_u)``
    indicating which partial derivatives may be nonzero.

    When *expr* is a ``Concat``, each child is analysed independently so
    that row blocks corresponding to independent sub-expressions get
    tighter (sparser) patterns.  For all other node types the analysis
    is column-level: every output row receives the same conservative mask.

    Args:
        expr: Root of the expression tree to analyse.
        n_x: Total dimension of the unified state vector.
        n_u: Total dimension of the unified control vector.

    Returns:
        Tuple ``(df_dx, df_du)`` of bool arrays with shapes
        ``(n_out, n_x)`` and ``(n_out, n_u)``.
    """
    if isinstance(expr, Concat):
        children = expr.children()
        blocks_x = []
        blocks_u = []
        for child in children:
            child_dx, child_du = jacobian_sparsity(child, n_x, n_u)
            blocks_x.append(child_dx)
            blocks_u.append(child_du)
        return np.vstack(blocks_x), np.vstack(blocks_u)

    n_out = _output_dim(expr)
    x_mask, u_mask = _leaf_masks(expr, n_x, n_u)
    # Broadcast the 1-D column mask to every output row
    return np.tile(x_mask, (n_out, 1)), np.tile(u_mask, (n_out, 1))


def cross_node_sparsity(
    expr: Expr,
    n_x: int,
    n_u: int,
    N: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-node boolean sparsity for a cross-node constraint expression.

    Cross-node constraints reference specific trajectory nodes via
    ``NodeReference`` wrappers (created by ``.at(k)``).  This function
    walks the AST and records which ``(node, variable-index)`` pairs are
    live.

    Args:
        expr: The cross-node constraint expression.
        n_x: Total dimension of the unified state vector.
        n_u: Total dimension of the unified control vector.
        N: Number of trajectory nodes.

    Returns:
        Tuple ``(x_mask, u_mask)`` where

        - ``x_mask`` has shape ``(N, n_x)``
        - ``u_mask`` has shape ``(N, n_u)``
    """
    x_mask = np.zeros((N, n_x), dtype=bool)
    u_mask = np.zeros((N, n_u), dtype=bool)

    def _collect(node: Expr) -> None:
        if not isinstance(node, NodeReference):
            return
        base = node.base
        k = node.node_idx
        # Normalize negative indices
        k_norm = k if k >= 0 else N + k
        if isinstance(base, State) and base._slice is not None:
            x_mask[k_norm, base._slice] = True
        elif isinstance(base, Control) and base._slice is not None:
            u_mask[k_norm, base._slice] = True

    traverse(expr, _collect)
    return x_mask, u_mask
