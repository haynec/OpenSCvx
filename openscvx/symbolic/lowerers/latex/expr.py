"""LaTeX visitors for core expression types.

Visitors: Constant, Parameter, NodeReference

Renders the leaf value nodes to LaTeX math strings: ``Constant`` via
``format_constant`` (scalars, vectors, and matrices), ``Parameter`` as its
symbol via ``latex_symbol``, and ``NodeReference`` as its base carrying a node
superscript ``x^{(k)}`` — the notation for "this variable at trajectory node k".
"""

from openscvx.symbolic.expr.expr import Constant, NodeReference
from openscvx.symbolic.expr.parameter import Parameter
from openscvx.symbolic.lowerers.latex._lowerer import format_constant, latex_symbol, wrap
from openscvx.symbolic.lowerers.latex._registry import visitor


@visitor(Constant)
def _visit_constant(lowerer, node: Constant):
    """Render a constant via :func:`format_constant`."""
    return format_constant(node.value)


@visitor(Parameter)
def _visit_parameter(lowerer, node: Parameter):
    """Render a parameter as its symbol."""
    return latex_symbol(node.name)


@visitor(NodeReference)
def _visit_node_reference(lowerer, node: NodeReference):
    """Render a node reference as its base with a node superscript, ``x^{(k)}``."""
    return rf"{wrap(lowerer, node.base, 10)}^{{({node.node_idx})}}"
