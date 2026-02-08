"""Tokenizer, parser, and loader for symbolic expression strings.

This package converts expression strings (e.g. from YAML/JSON problem
definitions) into the ``Expr`` AST used by the rest of openscvx.  It
mirrors the structure of :mod:`openscvx.symbolic.lowerers.jax`:

- A **registry** (``_registry.py``) maps function names to handler
  callables via the ``@function`` decorator, analogous to ``@visitor``.
- **Handler modules** (``math``, ``linalg``, ``array``, ``spatial``,
  ``lie``, ``stl``) register handlers as a side-effect of import,
  analogous to the JAX visitor modules.
- A **parser** (``parser.py``) implements a Pratt (precedence-climbing)
  parser that builds ``Expr`` trees from token streams.
- A **loader** (``load.py``) reads YAML / JSON files and returns the
  keyword arguments needed to construct a :class:`openscvx.problem.Problem`.

Example::

    from openscvx.symbolic.parser import ExprParser

    parser = ExprParser({"pos": pos_state, "vel": vel_state})
    expr = parser.parse("Norm(pos[:2] - vel[:2]) <= 5.0")

Example (YAML)::

    from openscvx.symbolic.parser import load_yaml

    problem_kwargs = load_yaml("my_problem.yaml")
    problem = Problem(**problem_kwargs)
"""

# Import handler modules to trigger @function registration.
# Each module populates _PARSE_FUNCTIONS as a side effect of import.
from openscvx.symbolic.parser import (
    array,  # noqa: F401
    linalg,  # noqa: F401
    math,  # noqa: F401
    spatial,  # noqa: F401
    stl,  # noqa: F401
)

# Lie handlers may fail if jaxlie is not installed; that's fine —
# the rest of the parser still works.
try:
    from openscvx.symbolic.parser import lie  # noqa: F401
except ImportError:
    pass

from openscvx.symbolic.parser.load import load_dict, load_json, load_yaml
from openscvx.symbolic.parser.parser import ExprParser, ParseError
from openscvx.symbolic.parser.tokenizer import TokenizeError, tokenize

__all__ = [
    "ExprParser",
    "ParseError",
    "TokenizeError",
    "tokenize",
    "load_yaml",
    "load_json",
    "load_dict",
]
