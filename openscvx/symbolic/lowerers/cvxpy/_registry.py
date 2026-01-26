"""Visitor registry for the CVXPy lowerer.

This module holds the shared visitor dictionary, the ``@visitor`` decorator,
and the ``dispatch`` function.  It is deliberately kept free of heavyweight
imports (no CVXPy, no expression subclasses beyond the base ``Expr``) so that
the visitor modules can import from it without circular-dependency issues.
"""

from typing import Any, Callable, Dict, Type

from openscvx.symbolic.expr import Expr

_CVXPY_VISITORS: Dict[Type[Expr], Callable] = {}
"""Registry mapping expression types to their visitor functions."""


def visitor(expr_cls: Type[Expr]) -> Callable[[Callable], Callable]:
    """Decorator to register a visitor function for an expression type.

    This decorator registers a visitor method to handle a specific expression
    type during CVXPy lowering. The decorated function is stored in
    _CVXPY_VISITORS and will be called by dispatch() when lowering that
    expression type.

    Args:
        expr_cls: The Expr subclass this visitor handles (e.g., Add, Mul, Norm)

    Returns:
        Decorator function that registers the visitor and returns it unchanged

    Example:
        Register a visitor function for the Add expression::

            @visitor(Add)
            def _visit_add(lowerer, node: Add):
                ...

    Note:
        Multiple expression types can share a visitor by stacking decorators::

            @visitor(Equality)
            @visitor(Inequality)
            def _visit_constraint(lowerer, node: Constraint):
                ...
    """

    def register(fn: Callable[[Any, Expr], Any]):
        _CVXPY_VISITORS[expr_cls] = fn
        return fn

    return register


def dispatch(lowerer: Any, expr: Expr) -> Any:
    """Dispatch an expression to its registered visitor function.

    Looks up the visitor function for the expression's type and calls it.
    This is the core of the visitor pattern implementation.

    Args:
        lowerer: The CvxpyLowerer instance (provides context for visitor methods)
        expr: The expression node to lower

    Returns:
        The result of calling the visitor function (CVXPy expression or constraint)

    Raises:
        NotImplementedError: If no visitor is registered for the expression type
    """
    fn = _CVXPY_VISITORS.get(type(expr))
    if fn is None:
        raise NotImplementedError(
            f"{lowerer.__class__.__name__!r} has no visitor for {type(expr).__name__}"
        )
    return fn(lowerer, expr)
