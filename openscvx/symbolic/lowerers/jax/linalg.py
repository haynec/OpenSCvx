"""JAX visitors for linear algebra expressions.

Visitors: Transpose, Diag, Sum, Inv, Norm
"""

import jax.numpy as jnp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.linalg import Diag, Inv, Norm, Sum, Transpose
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(Transpose)
def _visit_transpose(lowerer, node: Transpose):
    """Lower matrix transpose to JAX function."""
    f = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.transpose(f(x, u, node, params))


@visitor(Diag)
def _visit_diag(lowerer, node: Diag):
    """Lower diagonal matrix construction to JAX function."""
    f = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.diag(f(x, u, node, params))


@visitor(Sum)
def _visit_sum(lowerer, node: Sum):
    """Lower sum reduction to JAX function (sums all elements)."""
    f = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.sum(f(x, u, node, params))


@visitor(Inv)
def _visit_inv(lowerer, node: Inv):
    """Lower matrix inverse to JAX function.

    Computes the inverse of a square matrix using jnp.linalg.inv.
    Supports batched inputs with shape (..., M, M).

    Args:
        node: Inv expression node

    Returns:
        Function (x, u, node, params) -> inverse of operand matrix
    """
    f = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.linalg.inv(f(x, u, node, params))


@visitor(Norm)
def _visit_norm(lowerer, node: Norm):
    """Lower norm operation to JAX function.

    Converts symbolic norm to jnp.linalg.norm with appropriate ord parameter.
    Handles string ord values like "inf", "-inf", "fro".

    Args:
        node: Norm expression node with ord attribute

    Returns:
        Function (x, u, node, params) -> norm of operand
    """
    f = lowerer.lower(node.operand)
    ord_val = node.ord

    # Convert string ord values to appropriate JAX values
    if ord_val == "inf":
        ord_val = jnp.inf
    elif ord_val == "-inf":
        ord_val = -jnp.inf
    elif ord_val == "fro":
        # For vectors, Frobenius norm is the same as 2-norm
        ord_val = None  # Default is 2-norm

    return lambda x, u, node, params: jnp.linalg.norm(f(x, u, node, params), ord=ord_val)
