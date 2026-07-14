"""LaTeX visitors for linear algebra expressions.

Visitors: Transpose, Diag, Sum, Inv, Norm

Renders the linear-algebra AST nodes to LaTeX math strings: ``Transpose`` as
``A^{\\top}``, ``Inv`` as ``A^{-1}``, ``Sum`` as ``\\sum``, ``Diag`` as
``\\operatorname{diag}(...)``, and ``Norm`` as ``\\left\\| ... \\right\\|`` with an
order subscript. Euclidean/Frobenius orders (``None``, ``2``, ``"fro"``) render
without a subscript, since the double-bar already reads as the 2-norm.
"""

from openscvx.symbolic.expr.linalg import Diag, Inv, Norm, Sum, Transpose
from openscvx.symbolic.lowerers.latex._lowerer import wrap
from openscvx.symbolic.lowerers.latex._registry import visitor

# Norm orders that render without a subscript (the Euclidean / Frobenius case).
_BARE_NORM_ORDS = (None, 2, "fro")


@visitor(Transpose)
def _visit_transpose(lowerer, node: Transpose):
    """Render transpose as ``A^{\\top}``."""
    return rf"{wrap(lowerer, node.operand, 10)}^{{\top}}"


@visitor(Inv)
def _visit_inv(lowerer, node: Inv):
    """Render matrix inverse as ``A^{-1}``."""
    return rf"{wrap(lowerer, node.operand, 10)}^{{-1}}"


@visitor(Sum)
def _visit_sum(lowerer, node: Sum):
    """Render sum reduction as ``\\sum a``."""
    return rf"\sum {lowerer.lower(node.operand)}"


@visitor(Diag)
def _visit_diag(lowerer, node: Diag):
    """Render diagonal-matrix construction as ``\\operatorname{diag}(v)``."""
    return rf"\operatorname{{diag}}\left( {lowerer.lower(node.operand)} \right)"


@visitor(Norm)
def _visit_norm(lowerer, node: Norm):
    """Render a norm as ``\\left\\| a \\right\\|`` with an order subscript.

    The Euclidean / Frobenius order carries no subscript; ``"inf"`` /
    ``"-inf"`` render as ``\\infty`` / ``-\\infty``; other orders render
    literally (e.g. ``_{1}``).
    """
    ord_val = node.ord
    if ord_val in _BARE_NORM_ORDS:
        sub = ""
    elif ord_val == "inf":
        sub = r"_{\infty}"
    elif ord_val == "-inf":
        sub = r"_{-\infty}"
    else:
        sub = f"_{{{ord_val}}}"
    return rf"\left\| {lowerer.lower(node.operand)} \right\|{sub}"
