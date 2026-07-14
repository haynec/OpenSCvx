"""Parser handlers for logical and control flow operations.

Handlers: All, Any, Cond

Each handler is registered under its function name via ``@function`` and turns the
call-syntax form (e.g. ``Cond(pred, a, b)``) that the Pratt parser encounters in
an expression string into the corresponding logic ``Expr`` node — the boolean
reductions and the conditional branch.
"""

from openscvx.symbolic.expr.constraint import Inequality
from openscvx.symbolic.expr.expr import Constant, Expr
from openscvx.symbolic.expr.logic import All, Any, Cond
from openscvx.symbolic.parser._registry import function


def _to_predicate_list(args):
    """Convert positional args to a list of Inequality predicates."""
    preds = []
    for a in args:
        if not isinstance(a, Inequality):
            raise ValueError(
                f"Expected an Inequality predicate (e.g. x <= 5), got {type(a).__name__}"
            )
        preds.append(a)
    return preds


def _parse_node_ranges(val):
    """Convert a flat Constant array or list into a list of (start, end) tuples.

    Accepts ``node_ranges=[0, 2, 5, 7]`` and produces ``[(0, 2), (5, 7)]``.
    """
    if isinstance(val, Constant):
        flat = [int(v) for v in val.value.ravel()]
    elif isinstance(val, list):
        flat = [int(v) for v in val]
    else:
        raise ValueError(f"node_ranges must be an array literal, got {type(val).__name__}")
    if len(flat) % 2 != 0:
        raise ValueError("node_ranges must have an even number of elements (start/end pairs)")
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


@function("All")
def _parse_all(args, kwargs):
    if len(args) == 1:
        return All(args[0])
    if len(args) < 1:
        raise ValueError("All() requires at least 1 predicate argument")
    return All(_to_predicate_list(args))


@function("Any")
def _parse_any(args, kwargs):
    if len(args) == 1:
        return Any(args[0])
    if len(args) < 1:
        raise ValueError("Any() requires at least 1 predicate argument")
    return Any(_to_predicate_list(args))


@function("Cond")
def _parse_cond(args, kwargs):
    if len(args) < 3:
        raise ValueError(
            "Cond() requires at least 3 arguments (predicate, true_branch, false_branch)"
        )

    pred = args[0]
    true_branch = args[1]
    false_branch = args[2]

    # Handle None predicate (purely node-based switching)
    if pred is None:
        pass
    # Handle multiple predicates passed as extra positional args:
    #   Cond(p1, p2, true_branch, false_branch) is not supported — use All/Any.
    elif not isinstance(pred, (Inequality, All, Any, Expr)):
        raise ValueError(
            f"Cond predicate must be an Inequality, All, or Any, got {type(pred).__name__}"
        )

    node_ranges = None
    if "node_ranges" in kwargs:
        node_ranges = _parse_node_ranges(kwargs["node_ranges"])

    return Cond(pred, true_branch, false_branch, node_ranges=node_ranges)
