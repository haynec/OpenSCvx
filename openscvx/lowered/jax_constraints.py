"""JAX-lowered constraint dataclass."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional

import jax.numpy as jnp

if TYPE_CHECKING:
    from openscvx.symbolic.expr import CTCS


@dataclass
class LoweredNodalConstraint:
    """
    Dataclass to hold a lowered symbolic constraint function and its jacobians.

    This is a simplified drop-in replacement for NodalConstraint that holds
    only the essential lowered JAX functions and their jacobians, without
    the complexity of convex/vectorized flags or post-initialization logic.

    Designed for use with symbolic expressions that have been lowered to JAX
    and will be linearized for sequential convex programming.

    Args:
        func (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
            The lowered constraint function g(x, u, ...params) that returns
            constraint residuals. Should follow g(x, u) <= 0 convention.
            - x: 1D array (state at a single node), shape (n_x,)
            - u: 1D array (control at a single node), shape (n_u,)
            - Additional parameters: passed as keyword arguments

        grad_g_x (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            Jacobian of g w.r.t. x. If None, should be computed using jax.jacfwd.

        grad_g_u (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            Jacobian of g w.r.t. u. If None, should be computed using jax.jacfwd.

        nodes (Optional[List[int]]): List of node indices where this constraint applies.
            Set after lowering from NodalConstraint.
    """

    func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    grad_g_x: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    grad_g_u: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    nodes: Optional[List[int]] = None


@dataclass
class LoweredCrossNodeConstraint:
    """Lowered cross-node constraint with trajectory-level evaluation.

    Unlike regular LoweredNodalConstraint which operates on single-node vectors
    and is vmapped across the trajectory, LoweredCrossNodeConstraint operates
    on full trajectory arrays to relate multiple nodes simultaneously.

    This is necessary for constraints like:
    - Rate limits: x[k] - x[k-1] <= max_rate
    - Multi-step dependencies: x[k] = 2*x[k-1] - x[k-2]
    - Periodic boundaries: x[0] = x[N-1]

    The function signatures differ from LoweredNodalConstraint:
    - Regular: f(x, u, node, params) -> scalar (vmapped to handle (N, n_x))
    - Cross-node: f(X, U, params) -> scalar (single constraint with fixed node indices)

    Attributes:
        func: Function (X, U, params) -> scalar residual
            where X: (N, n_x), U: (N, n_u)
            Returns constraint residual following g(X, U) <= 0 convention
            The constraint references fixed trajectory nodes (e.g., X[5] - X[4])
        grad_g_X: Function (X, U, params) -> (N, n_x) Jacobian wrt full state trajectory
            This is typically sparse - most constraints only couple nearby nodes
        grad_g_U: Function (X, U, params) -> (N, n_u) Jacobian wrt full control trajectory
            Often zero or very sparse for cross-node state constraints

    Example:
        For rate constraint x[5] - x[4] <= r:

            func(X, U, params) -> scalar residual
            grad_g_X(X, U, params) -> (N, n_x) sparse Jacobian
                where grad_g_X[5, :] = ∂g/∂x[5] (derivative wrt node 5)
                and grad_g_X[4, :] = ∂g/∂x[4] (derivative wrt node 4)
                all other entries are zero

    Performance Note - Dense Jacobian Storage:
        The Jacobian matrices grad_g_X and grad_g_U are stored as DENSE arrays with
        shape (N, n_x) and (N, n_u), but most cross-node constraints only couple a
        small number of nearby nodes, making these matrices extremely sparse.

        For example, a rate limit constraint x[k] - x[k-1] <= r only has non-zero
        Jacobian entries at positions [k, :] and [k-1, :]. All other N-2 rows are
        zero but still stored in memory.

        Memory impact for large problems:
        - A single constraint with N=100 nodes, n_x=10 states requires ~8KB for
          grad_g_X (compared to ~160 bytes if sparse with 2 non-zero rows)
        - Multiple cross-node constraints multiply this overhead
        - May cause issues for N > 1000 with many constraints

        Performance impact:
        - Slower autodiff (computes many zero gradients)
        - Inefficient constraint linearization in the SCP solver
        - Potential GPU memory limitations for very large problems

        The current implementation prioritizes simplicity and compatibility with
        JAX's autodiff over memory efficiency. Future versions may support sparse
        Jacobian formats (COO, CSR, or custom sparse representations) if this
        becomes a bottleneck in practice.
    """

    func: Callable[[jnp.ndarray, jnp.ndarray, dict], jnp.ndarray]
    grad_g_X: Callable[[jnp.ndarray, jnp.ndarray, dict], jnp.ndarray]
    grad_g_U: Callable[[jnp.ndarray, jnp.ndarray, dict], jnp.ndarray]


@dataclass
class LoweredJaxConstraints:
    """JAX-lowered non-convex constraints with gradient functions.

    Contains constraints that have been lowered to JAX callable functions
    with automatically computed gradients. These are used for linearization
    in the SCP (Sequential Convex Programming) loop.

    Attributes:
        nodal: List of LoweredNodalConstraint objects. Each has `func`,
            `grad_g_x`, `grad_g_u` callables and `nodes` list.
        cross_node: List of LoweredCrossNodeConstraint objects. Each has
            `func`, `grad_g_X`, `grad_g_U` for trajectory-level constraints.
        ctcs: CTCS constraints (unchanged from input, not lowered here).
    """

    nodal: list[LoweredNodalConstraint] = field(default_factory=list)
    cross_node: list[LoweredCrossNodeConstraint] = field(default_factory=list)
    ctcs: list["CTCS"] = field(default_factory=list)
