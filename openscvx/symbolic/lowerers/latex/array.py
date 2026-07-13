"""LaTeX visitors for array expressions.

Visitors: Index, Concat, Stack, Hstack, Vstack
"""

from openscvx.symbolic.expr.array import Concat, Hstack, Index, Stack, Vstack
from openscvx.symbolic.lowerers.latex._lowerer import merge_subscript, wrap
from openscvx.symbolic.lowerers.latex._registry import visitor


def _format_index(index) -> str:
    """Render a NumPy-style index (int, slice, or tuple) as a subscript body."""
    if isinstance(index, tuple):
        return ", ".join(_format_index(i) for i in index)
    if isinstance(index, slice):
        start = "" if index.start is None else str(index.start)
        stop = "" if index.stop is None else str(index.stop)
        if index.step is not None:
            return f"{start}:{stop}:{index.step}"
        return f"{start}:{stop}"
    return str(index)


def _bmatrix(entries: list[str], sep: str) -> str:
    """Assemble a ``bmatrix`` from pre-rendered entries joined by ``sep``."""
    return rf"\begin{{bmatrix}} {sep.join(entries)} \end{{bmatrix}}"


@visitor(Index)
def _visit_index(lowerer, node: Index):
    """Render indexing/slicing as a subscript, ``x_{0}`` or ``x_{0:5}``.

    Uses :func:`merge_subscript` so an index on a role-prefixed variable
    comma-merges into the existing group (``x_{\\mathrm{velocity}}`` ->
    ``x_{\\mathrm{velocity},0}``) rather than emitting a double subscript.
    """
    base = wrap(lowerer, node.base, 10)
    return merge_subscript(base, _format_index(node.index))


@visitor(Concat)
def _visit_concat(lowerer, node: Concat):
    """Render concatenation (along axis 0) as a stacked column ``bmatrix``."""
    return _bmatrix([lowerer.lower(e) for e in node.exprs], r" \\ ")


@visitor(Stack)
def _visit_stack(lowerer, node: Stack):
    """Render vertical stacking as a row-per-entry ``bmatrix``."""
    return _bmatrix([lowerer.lower(r) for r in node.rows], r" \\ ")


@visitor(Hstack)
def _visit_hstack(lowerer, node: Hstack):
    """Render horizontal stacking as a single-row ``bmatrix``."""
    return _bmatrix([lowerer.lower(a) for a in node.arrays], " & ")


@visitor(Vstack)
def _visit_vstack(lowerer, node: Vstack):
    """Render vertical stacking as a stacked-column ``bmatrix``."""
    return _bmatrix([lowerer.lower(a) for a in node.arrays], r" \\ ")
