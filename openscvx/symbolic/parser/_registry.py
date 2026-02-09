"""Parser function registry for expression construction.

This module holds the shared function dictionary and the ``@function``
decorator, mirroring the ``@visitor`` / ``_JAX_VISITORS`` pattern used by
the JAX lowerer in ``openscvx.symbolic.lowerers.jax._registry``.

Visitor modules (``math``, ``linalg``, etc.) populate ``_PARSE_FUNCTIONS``
as a side-effect of import.
"""

from typing import Callable, Dict, Optional

_PARSE_FUNCTIONS: Dict[str, Callable] = {}
"""Registry mapping function names (e.g. ``"sin"``, ``"norm"``) to handler
callables with signature ``(args: list, kwargs: dict) -> Expr``.

All keys are stored in lowercase so that lookups are case-insensitive:
``Sin``, ``sin``, and ``SIN`` all resolve to the same handler."""


def function(name: str) -> Callable[[Callable], Callable]:
    """Decorator to register a parser handler for a named expression.

    The *name* is normalised to lowercase before storage so that lookups
    via :func:`lookup` are case-insensitive.

    Args:
        name: The function name as it appears in expression strings.

    Returns:
        Decorator that registers the handler and returns it unchanged.

    Example::

        @function("Sin")
        def _parse_sin(args, kwargs):
            return Sin(args[0])
    """

    def register(fn: Callable):
        _PARSE_FUNCTIONS[name.lower()] = fn
        return fn

    return register


def lookup(name: str) -> Optional[Callable]:
    """Look up a registered function handler by name (case-insensitive).

    Args:
        name: Function name to look up.

    Returns:
        The handler callable, or ``None`` if not registered.
    """
    return _PARSE_FUNCTIONS.get(name.lower())
