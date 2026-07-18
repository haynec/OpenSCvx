"""LaTeX visitors for vmap expressions.

Visitors: _Placeholder, Vmap

Renders a ``Vmap`` node as ``\\operatorname{vmap}\\left( <body> \\right)`` — the
symbolic analogue of ``jax.vmap`` — by lowering its body once and showing the
mapped expression rather than an opaque repr. Each ``_Placeholder`` (a single
batch element) renders as a neutral ``\\square``; with multiple batch arguments
every placeholder shares that box, so distinct batch sources are not visually
disambiguated.
"""

from openscvx.symbolic.expr.vmap import Vmap, _Placeholder
from openscvx.symbolic.lowerers.latex._registry import visitor


@visitor(_Placeholder)
def _visit_placeholder(lowerer, node: _Placeholder):
    """Render a batch placeholder as ``\\square``.

    A :class:`_Placeholder` stands for a single element of the batched data
    inside a :class:`Vmap` body; its auto-generated uuid name carries no
    meaning, so it renders as a neutral placeholder box ``\\square`` — read as
    "the current batch element".

    Note:
        With multiple batch arguments every placeholder renders as the same
        box, so distinct batch sources are not visually disambiguated; the
        readable win is showing the *body* of the mapped expression rather than
        an opaque ``Vmap(...)`` repr.
    """
    return r"\square"


@visitor(Vmap)
def _visit_vmap(lowerer, node: Vmap):
    """Render a vectorized map as ``\\operatorname{vmap}\\left( <body> \\right)``.

    ``Vmap`` applies a symbolic body to each element of a batch (the symbolic
    analogue of ``jax.vmap``); rendering the lowered body — with its
    placeholders shown as ``\\square`` — inside ``\\operatorname{vmap}(...)`` reads
    as "map this expression over the batch".
    """
    return rf"\operatorname{{vmap}}\left( {lowerer.lower(node._child)} \right)"
