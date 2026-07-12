"""LatexLowerer class definition and shared string helpers.

The visitor methods that populate the registry live in sibling modules
(``arithmetic``, ``math``, ``linalg``, etc.) and are registered via
``@visitor`` at import time.  This module also carries the pieces every
visitor leans on: precedence-aware parenthesization (:data:`_PRECEDENCE`,
:func:`wrap`), symbol rendering (:func:`latex_symbol`), and constant rendering
(:func:`format_constant`).
"""

import numpy as np

from openscvx.symbolic.expr import Expr
from openscvx.symbolic.expr.arithmetic import Add, MatMul, Mul, Neg, Power, Sub
from openscvx.symbolic.lowerers.latex._registry import dispatch


class LatexLowerer:
    """LaTeX backend for lowering symbolic expressions to math strings.

    This class implements the visitor pattern for converting symbolic
    expression AST nodes to LaTeX. Each expression type has a corresponding
    visitor function decorated with ``@visitor`` that returns a ``str``.

    The lowering is recursive: each visitor lowers its children first, then
    composes them into a LaTeX fragment, calling :func:`wrap` on children that
    may need parentheses for the surrounding precedence.

    Unlike :class:`~openscvx.symbolic.lowerers.jax.JaxLowerer`, the LaTeX
    lowerer is stateless — rendering is pure string assembly with no trace,
    node cache, or memoization.

    Example:
        Lower an expression to a LaTeX string::

            lowerer = LatexLowerer()
            s = lowerer.lower(ox.Norm(x) - 5.0)
            # '\\left\\| x \\right\\| - 5'
    """

    def lower(self, expr: Expr) -> str:
        """Lower a symbolic expression to a LaTeX string.

        Main entry point for lowering. Delegates to :func:`dispatch`, which
        looks up the appropriate visitor based on the expression type.

        Args:
            expr: Symbolic expression to lower (any Expr subclass)

        Returns:
            LaTeX math string (no ``$`` delimiters — callers add their own)

        Raises:
            NotImplementedError: If no visitor exists for the expression type
        """
        return dispatch(self, expr)


# Precedence-aware parenthesization: visitors call wrap() on children, passing
# the precedence of the surrounding context. Leaves and function-style nodes
# (Norm, Sin, Div-as-\frac, ...) are self-delimiting and default to 10.
_PRECEDENCE = {Add: 1, Sub: 1, Mul: 2, MatMul: 2, Neg: 3, Power: 4}


def wrap(lowerer: LatexLowerer, child: Expr, parent_precedence: int) -> str:
    """Lower ``child`` and parenthesize it if the parent binds tighter.

    A child whose precedence is strictly lower than ``parent_precedence`` is
    wrapped in ``\\left( ... \\right)``. Leaves and function-style nodes
    default to precedence 10 and are never wrapped.

    Args:
        lowerer: The LatexLowerer performing the lowering.
        child: The child expression to render.
        parent_precedence: Precedence of the surrounding operator; pass a value
            one higher than the operator's own precedence for the right operand
            of a non-associative operator (e.g. ``Sub``) so equal-precedence
            children are still parenthesized.

    Returns:
        The child's LaTeX, parenthesized when required.
    """
    s = lowerer.lower(child)
    if _PRECEDENCE.get(type(child), 10) < parent_precedence:
        return rf"\left( {s} \right)"
    return s


# Greek-word names that map to a LaTeX command of the same name.
_GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau",
    "upsilon", "phi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
    "Phi", "Psi", "Omega",
}


def latex_symbol(name: str) -> str:
    """Render a variable name as a LaTeX symbol.

    Rules, in order:

    - A Greek-word name renders as its command: ``'alpha'`` -> ``'\\alpha'``.
    - A single character renders bare: ``'x'`` -> ``'x'``.
    - A ``base_sub`` name splits on the first underscore into a subscript:
      ``'x_pos'`` -> ``'x_{\\mathrm{pos}}'`` (base and subscript are rendered
      recursively).
    - Any other multi-letter name renders upright: ``'position'`` ->
      ``'\\mathrm{position}'``, with underscores escaped.

    Args:
        name: The variable name to render.

    Returns:
        The LaTeX fragment for the name.
    """
    if name in _GREEK:
        return rf"\{name}"
    if len(name) == 1:
        return name
    # base_sub -> subscript, when the split yields non-empty halves.
    if "_" in name and not name.startswith("_") and not name.endswith("_"):
        base, sub = name.split("_", 1)
        return rf"{latex_symbol(base)}_{{{latex_symbol(sub)}}}"
    escaped = name.replace("_", r"\_")
    return rf"\mathrm{{{escaped}}}"


def _format_scalar(value: float) -> str:
    """Render a scalar with ``%g`` (compact, no trailing zeros)."""
    return "%g" % value


def format_constant(value: np.ndarray) -> str:
    """Render a numeric constant as LaTeX.

    Scalars render via ``%g``. Vectors render as a column ``bmatrix`` and
    matrices as a 2-D ``bmatrix``, up to 6 entries per axis. Larger arrays
    render as a placeholder ``\\mathrm{const} \\in \\mathbb{R}^{...}`` rather
    than an unreadable wall of numbers.

    Args:
        value: The constant value (scalar, vector, matrix, or higher-rank
            array).

    Returns:
        The LaTeX fragment for the constant.
    """
    arr = np.asarray(value)

    if arr.ndim == 0:
        return _format_scalar(arr.item())

    if arr.ndim == 1:
        if arr.shape[0] > 6:
            return rf"\mathrm{{const}} \in \mathbb{{R}}^{{{arr.shape[0]}}}"
        body = r" \\ ".join(_format_scalar(v) for v in arr)
        return rf"\begin{{bmatrix}} {body} \end{{bmatrix}}"

    if arr.ndim == 2:
        m, n = arr.shape
        if m > 6 or n > 6:
            return rf"\mathrm{{const}} \in \mathbb{{R}}^{{{m} \times {n}}}"
        rows = [" & ".join(_format_scalar(v) for v in row) for row in arr]
        body = r" \\ ".join(rows)
        return rf"\begin{{bmatrix}} {body} \end{{bmatrix}}"

    shape = r" \times ".join(str(d) for d in arr.shape)
    return rf"\mathrm{{const}} \in \mathbb{{R}}^{{{shape}}}"
