"""Parser handlers for GMSR-based Signal Temporal Logic operations.

Handlers: Or, And, Not, IfThen, IntegerVariable, Always, Eventually, Until
"""

from openscvx.symbolic.expr.expr import Constant
from openscvx.symbolic.expr.stl import (
    Always,
    And,
    Eventually,
    IfThen,
    IntegerVariable,
    Not,
    Or,
    Until,
)
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


@function("Not")
def _parse_not(args, kwargs):
    if len(args) != 1:
        raise ValueError("Not() requires exactly 1 predicate argument")
    return Not(*args, **kwargs)


@function("Always")
def _parse_always(args, kwargs):
    if len(args) not in (1, 2):
        raise ValueError("Always() requires 1 or 2 arguments (predicate[, interval])")
    if len(args) == 1:
        return Always(args[0], **kwargs)
    predicate, interval = args
    if isinstance(interval, Constant):
        interval = tuple(int(v) for v in interval.value.tolist())
    return Always(predicate, interval, **kwargs)


@function("Eventually")
def _parse_eventually(args, kwargs):
    if len(args) != 2:
        raise ValueError("Eventually() requires exactly 2 arguments (predicate, interval)")
    predicate, interval = args
    if isinstance(interval, Constant):
        interval = tuple(int(v) for v in interval.value.tolist())
    return Eventually(predicate, interval, **kwargs)


@function("Until")
def _parse_until(args, kwargs):
    if len(args) != 3:
        raise ValueError("Until() requires exactly 3 arguments (left, right, interval)")
    left, right, interval = args
    if isinstance(interval, Constant):
        interval = tuple(int(v) for v in interval.value.tolist())
    return Until(left, right, interval, **kwargs)


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
