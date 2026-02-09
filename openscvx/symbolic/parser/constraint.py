"""Parser handlers for constraint operations.

Handlers: CrossNodeConstraint, NodalConstraint, ctcs

Note: ``Equality`` / ``Inequality`` are produced by infix operators
(``==``, ``<=``, ``>=``) in ``parser.py`` and have no function-call form.
``NodalConstraint`` and ``CTCS`` can also be produced via dot-access
(``.at()`` and ``.over()``) in ``parser.py``.
"""

from openscvx.symbolic.expr.constraint import (
    CTCS,
    Constraint,
    CrossNodeConstraint,
    NodalConstraint,
)
from openscvx.symbolic.parser._registry import function
from openscvx.symbolic.parser.parser import ExprParser


@function("CrossNodeConstraint")
def _parse_cross_node_constraint(args, kwargs):
    if len(args) != 1:
        raise ValueError("CrossNodeConstraint() takes exactly 1 argument (a Constraint)")
    if not isinstance(args[0], Constraint):
        raise ValueError("CrossNodeConstraint() argument must be a Constraint (e.g. expr <= val)")
    return CrossNodeConstraint(args[0])


@function("NodalConstraint")
def _parse_nodal_constraint(args, kwargs):
    if len(args) < 2:
        raise ValueError("NodalConstraint() requires at least 2 arguments (constraint, node, ...)")
    constraint = args[0]
    if not isinstance(constraint, Constraint):
        raise ValueError("NodalConstraint() first argument must be a Constraint")
    nodes = ExprParser._args_to_int_list(args[1:])
    return NodalConstraint(constraint, nodes)


@function("ctcs")
def _parse_ctcs(args, kwargs):
    if len(args) < 1:
        raise ValueError("ctcs() requires at least 1 argument (a Constraint)")
    constraint = args[0]
    if not isinstance(constraint, Constraint):
        raise ValueError("ctcs() first argument must be a Constraint")

    penalty = str(kwargs.get("penalty", "squared_relu"))
    nodes = None
    if "nodes" in kwargs:
        nodes_val = kwargs["nodes"]
        if isinstance(nodes_val, list):
            nodes = tuple(int(n) for n in nodes_val)
        elif isinstance(nodes_val, tuple):
            nodes = tuple(int(n) for n in nodes_val)
    idx = None
    if "idx" in kwargs:
        idx = ExprParser._arg_to_int(kwargs["idx"])
    check_nodally = bool(kwargs.get("check_nodally", False))

    return CTCS(constraint, penalty=penalty, nodes=nodes, idx=idx, check_nodally=check_nodally)
