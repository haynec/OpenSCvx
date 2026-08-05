"""LaTeX visitors for user-supplied Jacobians.

Visitors: WithJacobian

A Jacobian override changes how a term is differentiated, not what it equals, so
the rendered formulation shows the wrapped expression alone — the same choice the
``CTCS`` visitor makes in rendering the constraint it stands for.
"""

from openscvx.symbolic.expr.autodiff import WithJacobian
from openscvx.symbolic.lowerers.latex._registry import visitor


@visitor(WithJacobian)
def _visit_with_jacobian(lowerer, node: WithJacobian):
    """Render the wrapped expression; the override is invisible in the math."""
    return lowerer.lower(node.expr)
