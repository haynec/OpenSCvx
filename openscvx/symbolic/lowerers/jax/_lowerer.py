"""JaxLowerer class definition.

The visitor methods that populate the registry live in sibling modules
(``arithmetic``, ``math``, ``linalg``, etc.) and are registered via
``@visitor`` at import time.
"""

from typing import Callable

import jax

from openscvx.symbolic.expr import Expr
from openscvx.symbolic.lowerers.jax._registry import dispatch

_MISSING = object()

# Value-memoization is only valid within a single JAX trace. Visitors bracket
# their nested trace with :func:`pause_memo` / :func:`resume_memo` so the wrapped
# closures recompute instead of returning a cross-trace value.
_memo_paused = [0]


def pause_memo() -> None:
    """Disable value-memoization for the duration of a nested sub-trace."""
    _memo_paused[0] += 1


def resume_memo() -> None:
    """Re-enable value-memoization after a nested sub-trace finishes tracing."""
    _memo_paused[0] -= 1


def _memoize_call(fn: Callable) -> Callable:
    """Cache a lowered closure's result so it emits its subgraph once per trace.

    Within a trace every node receives the same argument objects, so caching on
    argument identity (``is``) lets a shared subexpression emit its subgraph once
    instead of once per consumer. The cache only applies while tracing (at least
    one positional argument is a JAX tracer); eager calls, keyword-argument
    calls, and paused sections (nested sub-traces, see module note) bypass it.
    """
    last_args = None
    last_val = _MISSING

    def wrapped(*args, **kwargs):
        nonlocal last_args, last_val
        if kwargs or _memo_paused[0] or not any(isinstance(a, jax.core.Tracer) for a in args):
            return fn(*args, **kwargs)
        if (
            last_val is not _MISSING
            and last_args is not None
            and len(last_args) == len(args)
            and all(a is b for a, b in zip(last_args, args))
        ):
            return last_val
        last_val = fn(*args)
        last_args = args
        return last_val

    return wrapped


class JaxLowerer:
    """JAX backend for lowering symbolic expressions to executable functions.

    This class implements the visitor pattern for converting symbolic expression
    AST nodes to JAX functions. Each expression type has a corresponding visitor
    function decorated with @visitor that handles the lowering logic.

    The lowering process is recursive: each visitor lowers its child expressions
    first, then composes them into a JAX operation. All lowered functions have
    a standardized signature (x, u, node, params) -> result.

    Shared subexpressions are lowered once: :meth:`lower` caches the closure for
    each AST node by object identity, and each closure is wrapped so it emits its
    subgraph only once per trace (see :func:`_memoize_call`).

    Example:
        Set up the JaxLowerer and lower an expression to a JAX function::

            lowerer = JaxLowerer()
            expr = ox.Norm(x)**2 + 0.1 * ox.Norm(u)**2
            f = lowerer.lower(expr)
            result = f(x_val, u_val, node=0, params={})

    Note:
        A fresh lowerer is created per lowering pass; its node cache is keyed by
        ``id(expr)`` and is valid only while the expression tree is alive.
    """

    def __init__(self) -> None:
        # Maps id(expr) -> (expr, lowered closure) so each node is lowered once,
        # no matter how many parents share it. Storing the expr itself keeps it
        # alive: if it were garbage-collected, Python could reuse its id for a
        # new node and the cache would return the wrong closure. The identity
        # check on lookup guards against exactly that.
        self._node_cache: dict[int, tuple[Expr, Callable]] = {}

    def lower(self, expr: Expr) -> Callable:
        """Lower a symbolic expression to a JAX function.

        Main entry point for lowering. Delegates to dispatch() which looks up
        the appropriate visitor method based on the expression type. Results are
        memoized by node identity so shared subexpressions are lowered once.

        Args:
            expr: Symbolic expression to lower (any Expr subclass)

        Returns:
            JAX function with signature (x, u, node, params) -> result

        Raises:
            NotImplementedError: If no visitor exists for the expression type
            ValueError: If the expression is malformed (e.g., State without slice)
        """
        eid = id(expr)
        cached = self._node_cache.get(eid)
        if cached is not None and cached[0] is expr:
            return cached[1]
        fn = _memoize_call(dispatch(self, expr))
        self._node_cache[eid] = (expr, fn)
        return fn
