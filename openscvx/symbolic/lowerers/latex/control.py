"""LaTeX visitors for control expressions.

Visitors: Control
"""

from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.lowerers.latex._lowerer import latex_symbol
from openscvx.symbolic.lowerers.latex._registry import visitor


@visitor(Control)
def _visit_control(lowerer, node: Control):
    """Render a control as its symbol."""
    return latex_symbol(node.name)
