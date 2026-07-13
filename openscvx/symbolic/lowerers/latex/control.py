"""LaTeX visitors for control expressions.

Visitors: Control
"""

from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.lowerers.latex._lowerer import control_symbol
from openscvx.symbolic.lowerers.latex._registry import visitor


@visitor(Control)
def _visit_control(lowerer, node: Control):
    """Render a control as its role-prefixed symbol, ``u_{<sym>}`` (bare ``u``)."""
    return control_symbol(node.name)
