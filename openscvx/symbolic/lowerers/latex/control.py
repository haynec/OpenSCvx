"""LaTeX visitors for control expressions.

Visitors: Control

Renders a ``Control`` leaf as its role-prefixed symbol via ``control_symbol``:
a named control becomes ``u_{<sym>}`` while the anonymous unified control renders
as a bare ``u``. Rendering reads the control's name only — no slice into the
unified vector, since the LaTeX form names the variable rather than locating it.
"""

from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.lowerers.latex._lowerer import control_symbol
from openscvx.symbolic.lowerers.latex._registry import visitor


@visitor(Control)
def _visit_control(lowerer, node: Control):
    """Render a control as its role-prefixed symbol, ``u_{<sym>}`` (bare ``u``)."""
    return control_symbol(node.name)
