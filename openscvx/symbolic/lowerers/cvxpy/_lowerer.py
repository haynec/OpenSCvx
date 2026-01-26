"""CvxpyLowerer class definition.

The visitor methods that populate the registry live in sibling modules
(``arithmetic``, ``math``, ``linalg``, etc.) and are registered via
``@visitor`` at import time.
"""

from typing import Dict

import cvxpy as cp

from openscvx.symbolic.expr import Expr
from openscvx.symbolic.lowerers.cvxpy._registry import dispatch


class CvxpyLowerer:
    """CVXPy backend for lowering symbolic expressions to disciplined convex programs.

    This class implements the visitor pattern for converting symbolic expression
    AST nodes to CVXPy expressions and constraints. Each expression type has a
    corresponding visitor function decorated with @visitor that handles the
    lowering logic.

    The lowering process is recursive: each visitor lowers its child expressions
    first, then composes them into a CVXPy operation. CVXPy will validate DCP
    (Disciplined Convex Programming) compliance when the problem is constructed.

    Attributes:
        variable_map (dict): Dictionary mapping variable names to CVXPy expressions.
            Must include "x" for states and "u" for controls. May include parameter
            names mapped to CVXPy Parameter objects or constants.

    Example:
        Lower an expression to CVXPy::

            import cvxpy as cp
            lowerer = CvxpyLowerer(variable_map={
                "x": cp.Variable(3),
                "u": cp.Variable(2),
            })
            expr = ox.Norm(x)**2 + 0.1 * ox.Norm(u)**2
            cvx_expr = lowerer.lower(expr)

    Note:
        The lowerer is stateful (stores variable_map) unlike JaxLowerer which
        is stateless. Variables must be registered before lowering expressions
        that reference them.
    """

    def __init__(self, variable_map: Dict[str, cp.Expression] = None):
        """Initialize the CVXPy lowerer.

        Args:
            variable_map: Dictionary mapping variable names to CVXPy expressions.
                For State/Control objects, keys should be "x" and "u" respectively.
                For Parameter objects, keys should match their names. If None, an
                empty dictionary is created.
        """
        self.variable_map = variable_map or {}

    def register_variable(self, name: str, value: cp.Expression):
        """Register a variable in the variable map.

        Args:
            name: Variable name (e.g., "x", "u", or parameter name)
            value: CVXPy expression to associate with the name
        """
        self.variable_map[name] = value

    def lower(self, expr: Expr) -> cp.Expression:
        """Lower a symbolic expression to a CVXPy expression.

        Main entry point for lowering. Delegates to dispatch() which looks up
        the appropriate visitor method based on the expression type.

        Args:
            expr: Symbolic expression to lower (any Expr subclass)

        Returns:
            CVXPy expression or constraint

        Raises:
            NotImplementedError: If no visitor exists for the expression type
            ValueError: If required variables are missing from variable_map
        """
        return dispatch(self, expr)


def lower_to_cvxpy(expr: Expr, variable_map: Dict[str, cp.Expression] = None) -> cp.Expression:
    """Lower symbolic expression to CVXPy expression or constraint.

    Convenience wrapper that creates a CvxpyLowerer and lowers a single
    symbolic expression to a CVXPy expression. The result can be used in
    CVXPy optimization problems.

    Args:
        expr: Symbolic expression to lower (any Expr subclass)
        variable_map: Dictionary mapping variable names to CVXPy expressions.
            Must include "x" for states and "u" for controls. May include
            parameter names mapped to CVXPy Parameters or constants.

    Returns:
        CVXPy expression for arithmetic expressions (Add, Mul, Norm, etc.)
        or CVXPy constraint for constraint expressions (Equality, Inequality)

    Raises:
        NotImplementedError: If the expression type is not supported
        ValueError: If required variables are missing from variable_map
    """
    lowerer = CvxpyLowerer(variable_map)
    return lowerer.lower(expr)
