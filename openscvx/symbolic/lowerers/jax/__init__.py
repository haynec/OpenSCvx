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

from openscvx.symbolic.lowerers.jax._lowerer import JaxLowerer

# Import visitor modules to trigger @visitor registration.
# Each module populates _JAX_VISITORS as a side effect of import.
from openscvx.symbolic.lowerers.jax import arithmetic  # noqa: F401
from openscvx.symbolic.lowerers.jax import array  # noqa: F401
from openscvx.symbolic.lowerers.jax import constraint  # noqa: F401
from openscvx.symbolic.lowerers.jax import control  # noqa: F401
from openscvx.symbolic.lowerers.jax import expr  # noqa: F401
from openscvx.symbolic.lowerers.jax import lie  # noqa: F401
from openscvx.symbolic.lowerers.jax import linalg  # noqa: F401
from openscvx.symbolic.lowerers.jax import logic  # noqa: F401
from openscvx.symbolic.lowerers.jax import math  # noqa: F401
from openscvx.symbolic.lowerers.jax import spatial  # noqa: F401
from openscvx.symbolic.lowerers.jax import state  # noqa: F401
from openscvx.symbolic.lowerers.jax import stl  # noqa: F401
from openscvx.symbolic.lowerers.jax import vmap  # noqa: F401

__all__ = ["JaxLowerer"]
