"""Parser function registry for expression construction.

This module holds the shared function dictionary and the ``@function``
decorator, mirroring the ``@visitor`` / ``_JAX_VISITORS`` pattern used by
the JAX lowerer in ``openscvx.symbolic.lowerers.jax._registry``.

Visitor modules (``math``, ``linalg``, etc.) populate ``_PARSE_FUNCTIONS``
as a side-effect of import.
"""

from typing import Callable, Dict, Optional

_PARSE_FUNCTIONS: Dict[str, Callable] = {}
"""Registry mapping function names (e.g. ``"Sin"``, ``"Norm"``) to handler
callables with signature ``(args: list, kwargs: dict) -> Expr``."""


def function(name: str) -> Callable[[Callable], Callable]:
    """Decorator to register a parser handler for a named expression.

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
        _PARSE_FUNCTIONS[name] = fn
        return fn

    return register


def lookup(name: str) -> Optional[Callable]:
    """Look up a registered function handler by name.

    Args:
        name: Function name to look up (case-sensitive).

    Returns:
        The handler callable, or ``None`` if not registered.
    """
    return _PARSE_FUNCTIONS.get(name)
