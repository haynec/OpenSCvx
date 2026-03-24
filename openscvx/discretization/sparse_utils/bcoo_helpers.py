"""BCOO utilities for fixed sparsity patterns (``jax.experimental.sparse``).

These helpers build sparse matrices from dense batched data by indexing only
precomputed structural nonzeros — useful wherever a sparsity pattern is known
statically (not only for Jacobians).
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO


def precompute_sparse_indices(
    pattern: np.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """Pre-compute BCOO index arrays from a boolean sparsity pattern.

    Args:
        pattern: Boolean ``(m, n)`` array of structural nonzeros.

    Returns:
        ``(nz_rows, nz_cols, nnz)`` — JAX integer arrays and the nonzero count.
    """
    rows, cols = np.where(pattern)
    return jnp.array(rows), jnp.array(cols), len(rows)


def sparse_matmul_batched(
    dense_jac: jnp.ndarray,
    rhs: jnp.ndarray,
    nz_rows: jnp.ndarray,
    nz_cols: jnp.ndarray,
    m: int,
    n: int,
) -> jnp.ndarray:
    """Sparse-dense batched matmul: ``sparse(dense_jac) @ rhs``.

    Extracts nonzero values from the dense batch using pre-computed indices,
    constructs a BCOO matrix per batch item, and multiplies.

    Args:
        dense_jac: ``(batch, m, n)`` dense arrays (only positions at
            ``nz_rows, nz_cols`` are used).
        rhs: ``(batch, n, k)`` right-hand-side matrix.
        nz_rows: Row indices of structural nonzeros.
        nz_cols: Column indices of structural nonzeros.
        m: Number of rows of the sparse matrix.
        n: Number of columns of the sparse matrix.

    Returns:
        ``(batch, m, k)`` result of the sparse matmul.
    """
    data = dense_jac[:, nz_rows, nz_cols]  # (batch, nnz)
    indices = jnp.stack([nz_rows, nz_cols], axis=-1)  # (nnz, 2)

    def _single_matmul(data_i, rhs_i):
        sp = BCOO((data_i, indices), shape=(m, n))
        return sp @ rhs_i

    return jax.vmap(_single_matmul)(data, rhs)
