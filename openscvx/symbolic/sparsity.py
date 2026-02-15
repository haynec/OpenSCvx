"""Static sparsity analysis for Jacobian matrices.

Derives boolean sparsity patterns for dynamics and constraint Jacobians.
Each ``Expr`` node implements a ``sparsity(n_x, n_u)`` method that
propagates patterns through the AST.  This is a conservative (superset)
analysis: it may report a dependency that vanishes numerically, but will
never miss one.

The primary entry points are:

- ``jacobian_sparsity``: 2-D Jacobian sparsity pattern for any expression
  (delegates to ``Expr.sparsity``)
- ``cross_node_sparsity``: per-node patterns for cross-node constraints
"""

from typing import Tuple

import numpy as np

from openscvx.symbolic.expr import (
    Control,
    Expr,
    NodeReference,
    State,
    traverse,
)


def jacobian_sparsity(
    expr: Expr,
    n_x: int,
    n_u: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """2-D boolean sparsity patterns for df/dx and df/du.

    Delegates to ``expr.sparsity(n_x, n_u)``.  See
    :meth:`Expr.sparsity` for details on the per-node propagation
    rules.

    Args:
        expr: Root of the expression tree to analyse.
        n_x: Total dimension of the unified state vector.
        n_u: Total dimension of the unified control vector.

    Returns:
        Tuple ``(df_dx, df_du)`` of bool arrays with shapes
        ``(n_out, n_x)`` and ``(n_out, n_u)``.
    """
    return expr.sparsity(n_x, n_u)


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
