"""LaTeX visitors for GMSR Signal Temporal Logic (STL) expressions.

Visitors: And, Or, Not, IfThen, IntegerVariable, Always, Eventually, Until

Notation follows standard STL: propositional connectives ``\\wedge`` /
``\\vee`` / ``\\neg`` / ``\\implies``, and the temporal operators ``\\Box``
(always), ``\\Diamond`` (eventually), and ``\\mathcal{U}`` (until), each carrying
its enforcement interval as a subscript ``_{[a, b]}``.  The abstract bases
(``STLExpr``, ``_TemporalSTLExpr``) are never instantiated, so only the concrete
operators are registered.
"""

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
from openscvx.symbolic.lowerers.latex._lowerer import format_constant
from openscvx.symbolic.lowerers.latex._registry import visitor

# Operators that are self-delimiting when they appear as operands: a prefix
# negation and the temporal operators (which wrap their own argument). Every
# other operand — a bare predicate constraint, a binary/n-ary connective, an
# integer-variable membership — is parenthesized so the surrounding connective
# reads unambiguously.
_SELF_DELIMITING = (Not, Always, Eventually)


def _operand(lowerer, child) -> str:
    """Lower an operand, parenthesizing it unless it is self-delimiting."""
    s = lowerer.lower(child)
    if isinstance(child, _SELF_DELIMITING):
        return s
    return rf"\left( {s} \right)"


def _interval_subscript(interval) -> str:
    """Render a temporal operator's interval as ``_{[a, b]}`` (``""`` if absent).

    ``interval`` is a :class:`NodeInterval` (integer node indices) or
    :class:`TimeInterval` (seconds); both expose ``start``/``end``.  An unbounded
    (``None``) interval — a nested temporal operator inheriting the ambient
    window — renders no subscript.
    """
    if interval is None:
        return ""
    a, b = _fmt_bound(interval.start), _fmt_bound(interval.end)
    return rf"_{{[{a}, {b}]}}"


def _fmt_bound(value) -> str:
    """Format an interval bound: ``%g`` for a float (seconds), bare for an int."""
    if isinstance(value, float):
        return "%g" % value
    return str(value)


@visitor(And)
def _visit_and(lowerer, node: And):
    """Render an n-ary conjunction as ``p_1 \\wedge p_2 \\wedge ...``."""
    return r" \wedge ".join(_operand(lowerer, p) for p in node.predicates)


@visitor(Or)
def _visit_or(lowerer, node: Or):
    """Render an n-ary disjunction as ``p_1 \\vee p_2 \\vee ...``."""
    return r" \vee ".join(_operand(lowerer, p) for p in node.predicates)


@visitor(Not)
def _visit_not(lowerer, node: Not):
    """Render a negation as ``\\neg p``."""
    return rf"\neg {_operand(lowerer, node.predicate)}"


@visitor(IfThen)
def _visit_ifthen(lowerer, node: IfThen):
    """Render an implication as ``p \\implies q``."""
    return rf"{_operand(lowerer, node.condition)} \implies {_operand(lowerer, node.consequent)}"


@visitor(IntegerVariable)
def _visit_integer_variable(lowerer, node: IntegerVariable):
    """Render a discrete-value constraint as ``expr \\in \\{v_1, ..., v_n\\}``.

    ``IntegerVariable`` pins its expression to one of a finite set of allowed
    values; set membership is the faithful reading of that constraint, so it
    renders as ``expr \\in \\{...\\}`` (the value set formatted via
    :func:`format_constant`'s ``%g`` scalars).
    """
    values = r", ".join(format_constant(v) for v in node.values.ravel())
    return rf"{lowerer.lower(node.expr)} \in \left\{{ {values} \right\}}"


@visitor(Always)
def _visit_always(lowerer, node: Always):
    """Render the ``Always`` operator as ``\\Box_{[a, b]} \\left( p \\right)``."""
    sub = _interval_subscript(node.interval)
    return rf"\Box{sub} \left( {lowerer.lower(node.predicate)} \right)"


@visitor(Eventually)
def _visit_eventually(lowerer, node: Eventually):
    """Render the ``Eventually`` operator as ``\\Diamond_{[a, b]} \\left( p \\right)``."""
    sub = _interval_subscript(node.interval)
    return rf"\Diamond{sub} \left( {lowerer.lower(node.predicate)} \right)"


@visitor(Until)
def _visit_until(lowerer, node: Until):
    """Render the ``Until`` operator as ``p \\, \\mathcal{U}_{[a, b]} \\, q``."""
    sub = _interval_subscript(node.interval)
    left = _operand(lowerer, node.left)
    right = _operand(lowerer, node.right)
    return rf"{left} \, \mathcal{{U}}{sub} \, {right}"
