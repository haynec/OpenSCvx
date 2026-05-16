"""Sparse Jacobian computation via graph coloring and ``jax.experimental.sparse``.

This module provides utilities to compute Jacobians efficiently when the
sparsity pattern is known at compile time.  Instead of computing all n_x
(or n_u) columns via ``jax.jacfwd``, we use **column graph coloring** to
determine the minimum number of directional derivatives (JVPs) needed,
then reconstruct the sparse Jacobian from the compressed results.

The key entry point is :func:`make_sparse_jacobian_fns`, which returns
vmapped Jacobian callables (matching the dense ``jax.vmap(jax.jacfwd(...))``
signature) that internally use the sparse path.
"""

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Graph coloring
# ---------------------------------------------------------------------------


def color_columns(pattern: np.ndarray) -> np.ndarray:
    """Greedy column coloring of a boolean sparsity pattern.

    Two columns ``i`` and ``j`` can share a color when they have no row
    in common where both are nonzero.  This function assigns the fewest
    colors possible via a greedy (first-fit) algorithm over columns
    ordered by decreasing number of nonzeros.

    Args:
        pattern: Boolean ``(m, n)`` array where ``True`` indicates a
            structural nonzero.

    Returns:
        Integer array of length ``n`` mapping each column to its color
        (0-indexed).  The number of distinct colors is ``max(colors) + 1``.
    """
    m, n = pattern.shape
    colors = -np.ones(n, dtype=np.intp)

    col_order = np.argsort(-pattern.sum(axis=0))

    for col in col_order:
        row_set = set(np.where(pattern[:, col])[0])
        forbidden = set()
        for prev_col in range(n):
            if colors[prev_col] < 0:
                continue
            prev_rows = set(np.where(pattern[:, prev_col])[0])
            if row_set & prev_rows:
                forbidden.add(colors[prev_col])
        c = 0
        while c in forbidden:
            c += 1
        colors[col] = c

    return colors


def _build_coloring_data(
    pattern: np.ndarray,
) -> Tuple[jnp.ndarray, int, np.ndarray, np.ndarray]:
    """Pre-compute static coloring artifacts for JIT-friendly sparse Jacobian.

    Returns:
        seeds: ``(n_colors, n)`` float seed matrix — one seed per color.
        n_colors: Number of distinct colors.
        nz_rows: Row indices of structural nonzeros (length nnz).
        nz_cols: Column indices of structural nonzeros (length nnz).
    """
    colors = color_columns(pattern)
    n_colors = int(colors.max()) + 1
    n = pattern.shape[1]

    seeds = np.zeros((n_colors, n), dtype=jnp.float_)
    for col, c in enumerate(colors):
        seeds[c, col] = 1.0

    nz_rows, nz_cols = np.where(pattern)
    return jnp.array(seeds), n_colors, nz_rows, nz_cols


# ---------------------------------------------------------------------------
# Sparse jacfwd via colored JVPs
# ---------------------------------------------------------------------------


def _sparse_jacobian_fn(
    f: Callable,
    argnums: int,
    seeds: jnp.ndarray,
    n_colors: int,
    nz_rows: np.ndarray,
    nz_cols: np.ndarray,
    out_dim: int,
    in_dim: int,
) -> Callable:
    """Build a single-sample sparse Jacobian function using colored JVPs.

    The returned function has the same signature as ``jax.jacfwd(f, argnums)``.
    """
    color_of_col = np.empty(in_dim, dtype=np.intp)
    for c in range(n_colors):
        cols_with_c = np.where(seeds[c] != 0)[0]
        for col in cols_with_c:
            color_of_col[col] = c

    scatter_color = jnp.array(color_of_col[nz_cols])
    scatter_row = jnp.array(nz_rows)
    scatter_col = jnp.array(nz_cols)

    def jac_fn(*args, **kwargs):
        primals = args
        n_args = len(primals)

        def f_of_target(target):
            new_args = tuple(target if i == argnums else primals[i] for i in range(n_args))
            return f(*new_args, **kwargs)

        def single_jvp(seed):
            tangent = seed[:in_dim]
            _, jvp_out = jax.jvp(f_of_target, (primals[argnums],), (tangent,))
            return jvp_out

        # Cast seeds to match the primal dtype.  seeds is built eagerly as
        # float32 (x64 disabled), but jax.export traces with the literal
        # dtype of the dummy arrays (float64 from np.ones), so primals may
        # be float64.  jax.jvp requires matching dtypes.
        typed_seeds = seeds.astype(primals[argnums].dtype)
        compressed = jax.vmap(single_jvp)(typed_seeds)  # (n_colors, out_dim)

        values = compressed[scatter_color, scatter_row]
        jac = jnp.zeros((out_dim, in_dim))
        jac = jac.at[scatter_row, scatter_col].set(values)
        return jac

    return jac_fn


def make_sparse_jacobian_fns(
    f: Callable,
    A_c_pattern: Optional[np.ndarray],
    B_c_pattern: Optional[np.ndarray],
    n_x: int,
    n_u: int,
) -> Tuple[Callable, Callable]:
    """Create vmapped sparse Jacobian functions for df/dx and df/du.

    If a sparsity pattern is fully dense or ``None``, falls back to the
    standard ``jax.jacfwd`` path for that Jacobian.

    Args:
        f: Dynamics function ``f(x, u, node, params) -> x_dot``.
        A_c_pattern: Boolean ``(n_x, n_x)`` sparsity of df/dx, or ``None``.
        B_c_pattern: Boolean ``(n_x, n_u)`` sparsity of df/du, or ``None``.
        n_x: Number of state dimensions.
        n_u: Number of control dimensions.

    Returns:
        ``(A_vmapped, B_vmapped)`` — vmapped Jacobian callables with
        signature ``(x_batch, u_batch, nodes, params) -> J_batch``.
    """
    # --- df/dx ---
    if A_c_pattern is not None and not A_c_pattern.all():
        seeds_A, nc_A, nz_r_A, nz_c_A = _build_coloring_data(A_c_pattern)
        A_fn = _sparse_jacobian_fn(f, 0, seeds_A, nc_A, nz_r_A, nz_c_A, n_x, n_x)
    else:
        A_fn = jax.jacfwd(f, argnums=0)
    A_vmapped = jax.vmap(A_fn, in_axes=(0, 0, 0, None))

    # --- df/du ---
    if B_c_pattern is not None and not B_c_pattern.all():
        seeds_B, nc_B, nz_r_B, nz_c_B = _build_coloring_data(B_c_pattern)
        B_fn = _sparse_jacobian_fn(f, 1, seeds_B, nc_B, nz_r_B, nz_c_B, n_x, n_u)
    else:
        B_fn = jax.jacfwd(f, argnums=1)
    B_vmapped = jax.vmap(B_fn, in_axes=(0, 0, 0, None))

    return A_vmapped, B_vmapped
