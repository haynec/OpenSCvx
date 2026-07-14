"""LaTeX visitors for logic expressions.

Visitors: All, Any, Cond

Renders the logic AST nodes to LaTeX math strings. ``All`` and ``Any`` render as
big-operator reductions over their predicates (``\\bigwedge`` / ``\\bigvee``); a
single predicate still gets the operator so the reduce-over-elements semantics of
a vector predicate stay visible. ``Cond`` renders as a cases block. These nodes
have no convex form, so where the CVXPy backend rejects them, LaTeX simply
typesets the logical structure the user wrote.
"""

from openscvx.symbolic.expr.logic import All, Any, Cond
from openscvx.symbolic.lowerers.latex._registry import visitor


def _reduction(op: str, predicates) -> str:
    """Render a big-operator reduction ``op \\left( p_1, p_2, ... \\right)``.

    The predicates are the lowered constraint bodies; ``op`` is the n-ary
    logic operator (``\\bigwedge`` for :class:`All`, ``\\bigvee`` for
    :class:`Any`).  A single predicate still gets the operator so the reduce
    semantics — "over every element of the (possibly vector) predicate" — stay
    visible.
    """
    body = ", ".join(predicates)
    return rf"{op} \left( {body} \right)"


@visitor(All)
def _visit_all(lowerer, node: All):
    """Render an AND reduction over predicates as ``\\bigwedge(...)``.

    ``All`` is a hard-boolean conjunction (``jnp.all``); the big-wedge is the
    standard n-ary conjunction and reads as "all predicates hold", including
    the reduce-over-elements case of a single vector predicate.
    """
    return _reduction(r"\bigwedge", [lowerer.lower(p) for p in node.predicates])


@visitor(Any)
def _visit_any(lowerer, node: Any):
    """Render an OR reduction over predicates as ``\\bigvee(...)``.

    ``Any`` is a hard-boolean disjunction (``jnp.any``); the big-vee is the
    standard n-ary disjunction and reads as "some predicate holds".
    """
    return _reduction(r"\bigvee", [lowerer.lower(p) for p in node.predicates])


def _cond_condition(lowerer, node: Cond) -> str:
    """Render the ``\\text{if}`` condition of a :class:`Cond`.

    The predicate (an ``Inequality``/``All``/``Any``) renders as its LaTeX; a
    node-range restriction appends (or, for a pure node switch with no
    predicate, stands alone as) a membership ``k \\in [a, b) \\cup ...`` over the
    half-open ``node_ranges`` intervals.
    """
    parts = []
    if node.predicate is not None:
        parts.append(lowerer.lower(node.predicate))
    if node.node_ranges is not None:
        ranges = r" \cup ".join(rf"[{a}, {b})" for a, b in node.node_ranges)
        parts.append(rf"k \in {ranges}")
    return ", ".join(parts)


@visitor(Cond)
def _visit_cond(lowerer, node: Cond):
    """Render a conditional as a two-branch ``cases`` environment.

    ``\\begin{cases} <true> & \\text{if } <cond> \\\\ <false> & \\text{otherwise}
    \\end{cases}`` — the branch selected when the predicate holds, else the
    fallback.  The condition is the predicate and/or the node-range restriction
    (see :func:`_cond_condition`).
    """
    true_branch = lowerer.lower(node.true_branch)
    false_branch = lowerer.lower(node.false_branch)
    condition = _cond_condition(lowerer, node)
    return (
        r"\begin{cases} "
        rf"{true_branch} & \text{{if }} {condition} \\ "
        rf"{false_branch} & \text{{otherwise}} "
        r"\end{cases}"
    )
