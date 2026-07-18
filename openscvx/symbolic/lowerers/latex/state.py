"""LaTeX visitors for state/time/variable expressions.

Visitors: Variable, State, Time

``State`` and ``Time`` need separate registrations — dispatch is exact-type
and ``Time`` renders as ``t`` regardless of its name.
"""

from openscvx.symbolic.expr.state import State
from openscvx.symbolic.expr.time import Time
from openscvx.symbolic.expr.variable import Variable
from openscvx.symbolic.lowerers.latex._lowerer import latex_symbol, state_symbol
from openscvx.symbolic.lowerers.latex._registry import visitor


@visitor(State)
def _visit_state(lowerer, node: State):
    """Render a state as its role-prefixed symbol, ``x_{<sym>}`` (bare ``x``)."""
    return state_symbol(node.name)


@visitor(Variable)
def _visit_variable(lowerer, node: Variable):
    """Render a generic variable as its symbol (no role prefix)."""
    return latex_symbol(node.name)


@visitor(Time)
def _visit_time(lowerer, node: Time):
    """Render the time state as ``t``."""
    return "t"
