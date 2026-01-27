"""JaxLowerer class definition.

The visitor methods that populate the registry live in sibling modules
(``arithmetic``, ``math``, ``linalg``, etc.) and are registered via
``@visitor`` at import time.
"""

from typing import Callable

from openscvx.symbolic.expr import Expr
from openscvx.symbolic.lowerers.jax._registry import dispatch


class JaxLowerer:
    """JAX backend for lowering symbolic expressions to executable functions.

    This class implements the visitor pattern for converting symbolic expression
    AST nodes to JAX functions. Each expression type has a corresponding visitor
    function decorated with @visitor that handles the lowering logic.

    The lowering process is recursive: each visitor lowers its child expressions
    first, then composes them into a JAX operation. All lowered functions have
    a standardized signature (x, u, node, params) -> result.

    Note:
        This is a stateless lowerer - all state is in the expression tree.

    Example:
        Set up the JaxLowerer and lower an expression to a JAX function::

            lowerer = JaxLowerer()
            expr = ox.Norm(x)**2 + 0.1 * ox.Norm(u)**2
            f = lowerer.lower(expr)
            result = f(x_val, u_val, node=0, params={})

    Note:
        The lowerer is stateless and can be reused for multiple expressions.
    """

    def lower(self, expr: Expr) -> Callable:
        """Lower a symbolic expression to a JAX function.

        Main entry point for lowering. Delegates to dispatch() which looks up
        the appropriate visitor method based on the expression type.

        Args:
            expr: Symbolic expression to lower (any Expr subclass)

        Returns:
            JAX function with signature (x, u, node, params) -> result

        Raises:
            NotImplementedError: If no visitor exists for the expression type
            ValueError: If the expression is malformed (e.g., State without slice)
        """
        return dispatch(self, expr)
