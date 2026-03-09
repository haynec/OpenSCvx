"""Parser handlers for Signal Temporal Logic operations.

Handlers: Or
"""

from openscvx.symbolic.expr.stljax import Or
from openscvx.symbolic.parser._registry import function


@function("Or")
def _parse_or(args, kwargs):
    if len(args) < 2:
        raise ValueError("Or() requires at least 2 predicate arguments")
    return Or(*args)
