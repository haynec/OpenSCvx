"""Parser handlers for Signal Temporal Logic operations.

Handlers: stljax.Or

Registers the ``stljax``-backed disjunction under its namespaced name
``"stljax.Or"`` via ``@function``, turning the call-syntax form the Pratt parser
encounters in an expression string into an
:class:`openscvx.symbolic.expr.stljax.Or` node. The ``stljax.`` prefix keeps this
external-library operator distinct from the in-house GMSR ``Or`` handled in
:mod:`openscvx.symbolic.parser.stl`.
"""

from openscvx.symbolic.expr.stljax import Or
from openscvx.symbolic.parser._registry import function


@function("stljax.Or")
def _parse_or(args, kwargs):
    if len(args) < 2:
        raise ValueError("stljax.Or() requires at least 2 predicate arguments")
    return Or(*args)
