"""LaTeX backend for lowering symbolic expressions to math strings.

This package implements the LaTeX lowering backend that converts symbolic
expression AST nodes into LaTeX math strings.  Like the JAX and CVXPy
backends, lowering uses a visitor pattern where each expression type has a
corresponding visitor function registered via ``@visitor``.

The visitor functions are split across submodules that mirror the
``openscvx.symbolic.expr`` package structure.  Importing this package
triggers registration of all visitors, which now cover every shipped node
type — arithmetic, linear algebra, elementwise math, arrays, constraints,
logic, spatial, Lie-group, STL, and ``Vmap``.  A future node added without a
visitor raises ``NotImplementedError`` — the designed fallback until its
visitor lands.  (The legacy ``stljax`` operators are intentionally left
unregistered; the GMSR ``ox.stl`` surface is the rendered one.)

Example::

    from openscvx.symbolic.lowerers.latex import LatexLowerer

    lowerer = LatexLowerer()
    s = lowerer.lower(ox.Norm(x) - 5.0)  # '\\left\\| x \\right\\| - 5'
"""

# Import visitor modules to trigger @visitor registration.
# Each module populates _LATEX_VISITORS as a side effect of import.
from openscvx.symbolic.lowerers.latex import (
    arithmetic,  # noqa: F401
    array,  # noqa: F401
    autodiff,  # noqa: F401
    constraint,  # noqa: F401
    control,  # noqa: F401
    expr,  # noqa: F401
    lie,  # noqa: F401
    linalg,  # noqa: F401
    logic,  # noqa: F401
    math,  # noqa: F401
    spatial,  # noqa: F401
    state,  # noqa: F401
    stl,  # noqa: F401
    vmap,  # noqa: F401
)
from openscvx.symbolic.lowerers.latex._lowerer import (
    LatexLowerer,
    format_constant,
    latex_symbol,
)
from openscvx.symbolic.lowerers.latex.formulation import problem_to_latex

__all__ = ["LatexLowerer", "latex_symbol", "format_constant", "problem_to_latex"]
