"""Parser handlers for GMSR-based Signal Temporal Logic operations.

Handlers: Or, And, IfThen, IntegerVariable
"""

from openscvx.symbolic.expr.expr import Constant
from openscvx.symbolic.expr.stl import And, IfThen, IntegerVariable, Or
from openscvx.symbolic.parser._registry import function


@function("Or")
def _parse_or(args, kwargs):
    if len(args) < 2:
        raise ValueError("Or() requires at least 2 predicate arguments")
    return Or(*args, **kwargs)


@function("And")
def _parse_and(args, kwargs):
    if len(args) < 2:
        raise ValueError("And() requires at least 2 predicate arguments")
    return And(*args, **kwargs)


@function("IfThen")
def _parse_ifthen(args, kwargs):
    if len(args) != 2:
        raise ValueError("IfThen() requires exactly 2 arguments (condition, consequent)")
    return IfThen(*args, **kwargs)


@function("IntegerVariable")
def _parse_integer_variable(args, kwargs):
    if len(args) != 2:
        raise ValueError("IntegerVariable() requires exactly 2 arguments (expr, values)")
    expr, values = args
    # The parser represents array literals as Constant nodes; unwrap to numpy array
    if isinstance(values, Constant):
        values = values.value
    return IntegerVariable(expr, values, **kwargs)
