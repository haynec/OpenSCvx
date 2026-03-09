"""Parser handlers for GMSR-based Signal Temporal Logic operations.

Handlers: Or, And, IfThen
"""

from openscvx.symbolic.expr.stl import And, IfThen, IntegerVariable, Or
from openscvx.symbolic.parser._registry import function


@function("stl.Or")
def _parse_or(args, kwargs):
    if len(args) < 2:
        raise ValueError("stl.Or() requires at least 2 predicate arguments")
    return Or(*args, **kwargs)


@function("stl.And")
def _parse_and(args, kwargs):
    if len(args) < 2:
        raise ValueError("stl.And() requires at least 2 predicate arguments")
    return And(*args, **kwargs)


@function("stl.IfThen")
def _parse_ifthen(args, kwargs):
    if len(args) != 2:
        raise ValueError("stl.IfThen() requires exactly 2 arguments (condition, consequent)")
    return IfThen(*args, **kwargs)


@function("stl.IntegerVariable")
def _parse_integer_variable(args, kwargs):
    if len(args) != 2:
        raise ValueError("stl.IntegerVariable() requires exactly 2 arguments (expr, values)")
    return IntegerVariable(*args, **kwargs)
