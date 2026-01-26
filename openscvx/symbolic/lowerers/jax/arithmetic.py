"""JAX visitors for arithmetic expressions.

Visitors: Add, Sub, Mul, Div, MatMul, Neg, Power
"""

import jax.numpy as jnp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.arithmetic import Add, Div, MatMul, Mul, Neg, Power, Sub
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(Add)
def _visit_add(lowerer, node: Add):
    """Lower addition to JAX function.

    Recursively lowers all terms and composes them with element-wise addition.
    Supports broadcasting following NumPy/JAX rules.

    Args:
        node: Add expression node with multiple terms

    Returns:
        Function (x, u, node, params) -> sum of all terms
    """
    fs = [lowerer.lower(term) for term in node.terms]

    def fn(x, u, node, params):
        acc = fs[0](x, u, node, params)
        for f in fs[1:]:
            acc = acc + f(x, u, node, params)
        return acc

    return fn


@visitor(Sub)
def _visit_sub(lowerer, node: Sub):
    """Lower subtraction to JAX function (element-wise left - right)."""
    fL = lowerer.lower(node.left)
    fR = lowerer.lower(node.right)
    return lambda x, u, node, params: fL(x, u, node, params) - fR(x, u, node, params)


@visitor(Mul)
def _visit_mul(lowerer, node: Mul):
    """Lower element-wise multiplication to JAX function (Hadamard product)."""
    fs = [lowerer.lower(factor) for factor in node.factors]

    def fn(x, u, node, params):
        acc = fs[0](x, u, node, params)
        for f in fs[1:]:
            acc = acc * f(x, u, node, params)
        return acc

    return fn


@visitor(Div)
def _visit_div(lowerer, node: Div):
    """Lower element-wise division to JAX function."""
    fL = lowerer.lower(node.left)
    fR = lowerer.lower(node.right)
    return lambda x, u, node, params: fL(x, u, node, params) / fR(x, u, node, params)


@visitor(MatMul)
def _visit_matmul(lowerer, node: MatMul):
    """Lower matrix multiplication to JAX function using jnp.matmul."""
    fL = lowerer.lower(node.left)
    fR = lowerer.lower(node.right)
    return lambda x, u, node, params: jnp.matmul(fL(x, u, node, params), fR(x, u, node, params))


@visitor(Neg)
def _visit_neg(lowerer, node: Neg):
    """Lower negation (unary minus) to JAX function."""
    fO = lowerer.lower(node.operand)
    return lambda x, u, node, params: -fO(x, u, node, params)


@visitor(Power)
def _visit_power(lowerer, node: Power):
    """Lower element-wise power (base**exponent) to JAX function."""
    fB = lowerer.lower(node.base)
    fE = lowerer.lower(node.exponent)
    return lambda x, u, node, params: jnp.power(fB(x, u, node, params), fE(x, u, node, params))
