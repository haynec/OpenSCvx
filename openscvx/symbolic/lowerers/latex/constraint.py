"""LaTeX visitors for constraint expressions.

Visitors: Equality, Inequality, NodalConstraint, CrossNodeConstraint, CTCS

Renders constraint AST nodes to LaTeX relational rows. ``Equality`` and
``Inequality`` render as ``lhs = rhs`` / ``lhs \\le rhs``; ``NodalConstraint`` adds
a node-set annotation (``k = a``, a contiguous ``k \\in \\{a, \\dots, b\\}``, or an
explicit set elided past six entries); ``CrossNodeConstraint`` renders its inner
constraint with its node-referencing operands; and ``CTCS`` renders as the
continuous-time path constraint it stands for (``\\forall t``), which is how the
Mayer-form formulation presents it.
"""

from openscvx.symbolic.expr.constraint import (
    CTCS,
    CrossNodeConstraint,
    Equality,
    Inequality,
    NodalConstraint,
)
from openscvx.symbolic.lowerers.latex._registry import visitor


def _node_set(indices: list[int]) -> str:
    """Render a set body ``\\{...\\}`` over integer node indices."""
    return r"\{" + ", ".join(str(i) for i in indices) + r"\}"


def _node_annotation(nodes: list[int]) -> str:
    """Render the node-set annotation for a :class:`NodalConstraint`.

    A single node renders as ``k = a``; a contiguous ascending range as
    ``k \\in \\{a, \\dots, b\\}``; otherwise the explicit set, elided with
    ``\\dots`` past six entries.
    """
    nodes = list(nodes)
    if len(nodes) == 1:
        return f"k = {nodes[0]}"
    if nodes == list(range(nodes[0], nodes[-1] + 1)):
        return r"k \in \{" + f"{nodes[0]}, \\dots, {nodes[-1]}" + r"\}"
    if len(nodes) > 6:
        return r"k \in \{" + ", ".join(str(n) for n in nodes[:6]) + r", \dots\}"
    return r"k \in " + _node_set(nodes)


@visitor(Equality)
def _visit_equality(lowerer, node: Equality):
    """Render an equality as ``lhs = rhs``."""
    return f"{lowerer.lower(node.lhs)} = {lowerer.lower(node.rhs)}"


@visitor(Inequality)
def _visit_inequality(lowerer, node: Inequality):
    """Render an inequality as ``lhs \\le rhs``."""
    return rf"{lowerer.lower(node.lhs)} \le {lowerer.lower(node.rhs)}"


@visitor(NodalConstraint)
def _visit_nodal_constraint(lowerer, node: NodalConstraint):
    """Render the inner constraint with a ``\\quad k \\in \\{...\\}`` annotation."""
    return rf"{lowerer.lower(node.constraint)} \quad {_node_annotation(node.nodes)}"


@visitor(CrossNodeConstraint)
def _visit_cross_node_constraint(lowerer, node: CrossNodeConstraint):
    """Render the inner constraint; ``NodeReference`` superscripts carry the nodes."""
    return lowerer.lower(node.constraint)


@visitor(CTCS)
def _visit_ctcs(lowerer, node: CTCS):
    """Render the inner constraint as a continuous-time path constraint.

    Appends ``\\quad \\forall t``, extended with the node interval when
    ``node.nodes`` is set.
    """
    result = rf"{lowerer.lower(node.constraint)} \quad \forall t"
    if node.nodes is not None:
        start, end = node.nodes
        result += rf" \in [t_{{{start}}}, t_{{{end}}}]"
    return result
