"""Sparse linear algebra helpers built on ``jax.experimental.sparse``."""

from .sparse_jacobian import (
    color_columns,
    make_sparse_jacobian_fns,
    precompute_sparse_indices,
    sparse_matmul_batched,
)

__all__ = [
    "color_columns",
    "make_sparse_jacobian_fns",
    "precompute_sparse_indices",
    "sparse_matmul_batched",
]
