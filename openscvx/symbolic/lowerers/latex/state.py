"""LaTeX visitors for state/time/variable expressions.

Visitors: Variable, State, Time

``State`` and ``Time`` need separate registrations — dispatch is exact-type
and ``Time`` renders as ``t`` regardless of its name.
"""

from openscvx.symbolic.expr.state import State
from openscvx.symbolic.expr.time import Time
from openscvx.symbolic.expr.variable import Variable
from openscvx.symbolic.lowerers.latex._lowerer import latex_symbol
from openscvx.symbolic.lowerers.latex._registry import visitor


@visitor(Variable)
@visitor(State)
def _visit_state(lowerer, node: State):
    """Render a state/variable as its symbol."""
    return latex_symbol(node.name)


@visitor(Time)
def _visit_time(lowerer, node: Time):
    """Render the time state as ``t``."""
    return "t"
