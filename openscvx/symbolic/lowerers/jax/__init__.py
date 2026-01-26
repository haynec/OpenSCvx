"""JAX backend for lowering symbolic expressions to executable functions.

This package implements the JAX lowering backend that converts symbolic
expression AST nodes into JAX functions with automatic differentiation
support.  The lowering uses a visitor pattern where each expression type
has a corresponding visitor function registered via ``@visitor``.

The visitor functions are split across submodules that mirror the
``openscvx.symbolic.expr`` package structure.  Importing this package
triggers registration of all visitors.

Example::

    from openscvx.symbolic.lowerers.jax import JaxLowerer

    lowerer = JaxLowerer()
    f = lowerer.lower(expr)
    result = f(x_val, u_val, node=0, params={})
"""

# Import visitor modules to trigger @visitor registration.
# Each module populates _JAX_VISITORS as a side effect of import.
from openscvx.symbolic.lowerers.jax import (
    arithmetic,  # noqa: F401
    array,  # noqa: F401
    constraint,  # noqa: F401
    control,  # noqa: F401
    expr,  # noqa: F401
    lie,  # noqa: F401
    linalg,  # noqa: F401
    logic,  # noqa: F401
    math,  # noqa: F401
    spatial,  # noqa: F401
    state,  # noqa: F401
    stl,  # noqa: F401
    vmap,  # noqa: F401
)
from openscvx.symbolic.lowerers.jax._lowerer import JaxLowerer

__all__ = ["JaxLowerer"]
