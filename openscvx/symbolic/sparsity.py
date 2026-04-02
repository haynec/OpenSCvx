"""Sparsity analysis helpers for symbolic Jacobian patterns.

This module contains all sparsity-related utilities used by the symbolic
expression layer:

- **AST helpers** (`_broadcast_sparsity`, `_broadcast_liveness`,
  `_bool_matmul`, `_element_liveness`) support the per-node
  `Expr.sparsity()` method implementations.
- **Discretization helpers** (`transitive_closure`, `discrete_sparsity`)
  lift continuous-time Jacobian sparsity to discrete-time patterns for use
  with CVXPY sparse parameters.

!!! Note
   The actual sparsity patterns are computed by the `sparsity()` method
   defined on each AST node in `expr/`.  This module only provides the
   shared helpers that those per-node implementations depend on.
"""

from typing import TYPE_CHECKING, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from openscvx.symbolic.expr.expr import Expr

# ============================================================================
# AST sparsity helpers — used by Expr.sparsity() implementations
# ============================================================================


def _broadcast_sparsity(
    S: np.ndarray,
    from_shape: Tuple[int, ...],
    to_shape: Tuple[int, ...],
    n_cols: int,
) -> np.ndarray:
    """Broadcast a sparsity matrix from *from_shape* to *to_shape*.

    Reshapes *S* (`prod(from_shape), n_cols`) into the N-D expression
    shape, broadcasts with NumPy rules, and flattens back.
    """
    from_shape = from_shape or (1,)
    to_shape = to_shape or (1,)
    S_nd = S.reshape(from_shape + (n_cols,))
    S_broadcast = np.broadcast_to(S_nd, to_shape + (n_cols,))
    return S_broadcast.reshape(int(np.prod(to_shape)), n_cols).copy()


def _broadcast_liveness(
    live: np.ndarray,
    from_shape: Tuple[int, ...],
    to_shape: Tuple[int, ...],
) -> np.ndarray:
    """Broadcast a 1-D liveness vector from *from_shape* to *to_shape*."""
    from_shape = from_shape or (1,)
    to_shape = to_shape or (1,)
    return np.broadcast_to(live.reshape(from_shape), to_shape).flatten().copy()


def _bool_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Boolean matrix multiply: `C[i,k] = OR_j (A[i,j] AND B[j,k])`."""
    return (A.astype(np.uint8) @ B.astype(np.uint8)) > 0


def _element_liveness(expr: "Expr") -> np.ndarray:
    """Which scalar elements of *expr* might be nonzero at runtime.

    Returns the exact zero pattern for `Constant` nodes; assumes
    all-live for everything else (including `Parameter`, whose value
    may be updated at runtime so its zero structure is not reliable).
    """
    from openscvx.symbolic.expr.expr import Constant

    if isinstance(expr, Constant):
        return np.asarray(expr.value).flatten() != 0
    shape = expr.check_shape()
    n = int(np.prod(shape)) if shape else 1
    return np.ones(n, dtype=bool)


# ============================================================================
# Discretization sparsity — operates on numpy boolean arrays, not Expr nodes
# ============================================================================


def transitive_closure(A: np.ndarray) -> np.ndarray:
    """Reachability matrix of a boolean adjacency matrix.

    Given `A[i,j] = True` meaning *j directly influences i*, returns
    `R[i,j] = True` meaning *j influences i through a path of any
    length* (including zero — the identity is included).

    This is the sparsity pattern of the matrix exponential:
    `expm(A·dt) = I + A·dt + A²·dt²/2 + …`, so `R[i,j]` is nonzero
    iff any power of `A` has a nonzero `(i,j)` entry.

    Used to lift continuous-time Jacobian sparsity to discrete-time
    state-transition sparsity.

    Args:
        A: Boolean matrix of shape `(n, n)`.

    Returns:
        Boolean reachability matrix of shape `(n, n)`.
    """
    n = A.shape[0]
    R = A | np.eye(n, dtype=bool)
    for _ in range(int(np.ceil(np.log2(max(n, 1))))):
        R_new = _bool_matmul(R, R)
        if np.array_equal(R_new, R):
            break
        R = R_new
    return R


def discrete_sparsity(
    A_c: np.ndarray,
    B_c: np.ndarray,
    dis_type: Union[str, Sequence[str], np.ndarray] = "ZOH",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discrete-time sparsity from continuous-time Jacobian patterns.

    The linearize-then-discretize scheme integrates the variational
    equations::

        dΦ/dτ   = A · Φ,              Φ(0) = I
        dB_d/dτ = A · B_d + α · B,    B_d(0) = 0
        dC_d/dτ = A · C_d + β · B,    C_d(0) = 0

    where `α, β` are per-control interpolation weights (ZOH: `α=1, β=0`;
    FOH: linear blend).

    Because the *sparsity pattern* of `A` and `B` is constant along
    the trajectory, the discrete patterns follow from the transitive
    closure::

        A_d  = transitive_closure(A_c)
        B_d  = bool_matmul(A_d, B_c)
        C_d  = per-column: all-False (ZOH) or B_d column (FOH)

    Args:
        A_c: `(n_x, n_x)` boolean continuous-time `df/dx` sparsity.
        B_c: `(n_x, n_u)` boolean continuous-time `df/du` sparsity.
        dis_type: ``"ZOH"`` or ``"FOH"`` (applied to all controls), a
            per-control sequence of length ``n_u``, or a float/bool
            array (``1``/``True`` = FOH, ``0``/``False`` = ZOH).

    Returns:
        `(A_d, B_d, C_d)` boolean sparsity patterns with shapes
        `(n_x, n_x)`, `(n_x, n_u)`, `(n_x, n_u)`.
    """
    A_d = transitive_closure(A_c)
    B_d = _bool_matmul(A_d, B_c)

    n_u = B_c.shape[1]
    if isinstance(dis_type, np.ndarray):
        foh_cols = np.asarray(dis_type).flatten() > 0.5
        C_d = np.zeros_like(B_c)
        C_d[:, foh_cols] = B_d[:, foh_cols]
    elif isinstance(dis_type, str):
        if dis_type == "FOH":
            C_d = B_d.copy()
        else:
            C_d = np.zeros_like(B_c)
    else:
        C_d = np.zeros_like(B_c)
        for i in range(n_u):
            if dis_type[i] == "FOH":
                C_d[:, i] = B_d[:, i]
    return A_d, B_d, C_d
