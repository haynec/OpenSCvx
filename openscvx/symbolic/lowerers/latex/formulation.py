"""Render a whole :class:`~openscvx.problem.Problem` as a Mayer-form formulation.

Where the expression visitors (``arithmetic``, ``math``, ...) turn a single AST
node into LaTeX, this module assembles the *problem* — objective, dynamics,
path constraints, box bounds, and boundary conditions — into one classic
optimal-control block::

    minimize    <Mayer terms>
    subject to  <dynamics>, <path constraints>, <box bounds>, <boundary conditions>

A **Mayer form** puts the entire objective on the boundary of the trajectory
(terms like ``t(t_f)`` or ``x_i(t_0)``) rather than as a running integral; this
is exactly how OpenSCvx problems are posed, since ``initial``/``final`` boundary
types are the only source of cost.

The block renders the **pre-augmentation, user-authored** problem.  The solver
augments dynamics with a time-dilation control and per-CTCS penalty states
(``_``-prefixed), but those are machinery, not math the user wrote — so anything
named with a leading underscore is filtered out.  CTCS constraints render as the
continuous-time path constraints they stand for, which is the whole point of the
Mayer story.

Each of the two configurable sections (``dynamics`` and ``constraints``) renders
at one of three detail levels:

- ``"inline"``   — full expressions sit in the formulation.
- ``"symbolic"`` — structure only: ``\\dot{x} = f(x, u)`` for dynamics, numbered
  ``g_i(x, u) \\le 0`` / ``h_j(x, u) = 0`` references for constraints, with their
  ``\\forall t`` / node annotations kept visible.
- ``"separate"`` — symbolic in the formulation, with the bodies appended as their
  own ``\\text{where}`` equation blocks.  The ``g_i`` / ``h_j`` numbering lines up
  between the references and the definitions.

The whole thing is one string with no ``$`` delimiters — callers add their own.
"""

from typing import List, Sequence, Tuple

import numpy as np

from openscvx.symbolic.expr import CTCS, Equality, Expr, NodalConstraint, traverse, to_expr
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State
from openscvx.symbolic.expr.variable import Variable
from openscvx.symbolic.lowerers.latex._lowerer import (
    LatexLowerer,
    format_constant,
    latex_symbol,
)

_MODES = ("inline", "symbolic", "separate")

# Boundary-condition types that pin a value (Variable.initial/final setters
# store plain numbers as "Fix" and ("fixed", v) tuples as "Fixed").
_FIXED_TYPES = ("Fix", "Fixed")


def problem_to_latex(
    symbolic,
    dynamics_dict: dict,
    lam_cost,
    *,
    dynamics: str,
    constraints: str,
) -> str:
    """Assemble a Mayer-form LaTeX formulation of a problem.

    Args:
        symbolic: The preprocessed :class:`SymbolicProblem` (source of states,
            controls, categorized constraints, and horizon ``N``).
        dynamics_dict: The user-authored ``{state_name: expr}`` dynamics, before
            time/CTCS augmentation.
        lam_cost: Objective weights — a scalar or an ``(n_states,)`` array indexed
            by each state's ``_slice`` (the resolved ``algorithm.lam_cost``).
        dynamics: Detail level for the dynamics section — one of ``"inline"``,
            ``"symbolic"``, ``"separate"``.
        constraints: Detail level for the constraint section — same choices.

    Returns:
        One LaTeX string: the ``\\begin{aligned}...\\end{aligned}`` formulation,
        with ``\\text{where}`` definition blocks appended in ``"separate"`` modes.
        No ``$`` delimiters.

    Raises:
        ValueError: If ``dynamics`` or ``constraints`` is not a valid mode.
    """
    _validate_mode("dynamics", dynamics)
    _validate_mode("constraints", constraints)

    lowerer = LatexLowerer()
    N = symbolic.N

    objective = _objective(symbolic.states, lam_cost, lowerer)

    dyn_pairs = _dynamics_pairs(dynamics_dict, lowerer)
    if dynamics == "inline":
        dyn_rows = [rf"\dot{{{nm}}} = {ex}" for nm, ex in dyn_pairs]
    else:
        dyn_rows = [r"\dot{x} = f(x, u)"]

    con_refs, con_defs = _constraint_rows(symbolic, N, constraints, lowerer)

    box_rows = _box_bound_rows(list(symbolic.states) + list(symbolic.controls), lowerer)
    bc_rows = _boundary_rows(symbolic.states, lowerer)

    st_rows = dyn_rows + con_refs + box_rows + bc_rows
    main = _aligned_formulation(objective, st_rows)

    where_blocks: List[str] = []
    if dynamics == "separate":
        where_blocks.append(
            _aligned_block(rf"\dot{{{nm}}} &= {ex}" for nm, ex in dyn_pairs)
        )
    if constraints == "separate":
        where_blocks.append(
            _aligned_block(rf"{label}(x, u) &= {residual}" for label, residual in con_defs)
        )

    if not where_blocks:
        return main
    return (
        main
        + "\n\\\\[1ex]\n\\text{where}\\\\[0.5ex]\n"
        + "\n\\\\[1ex]\n".join(where_blocks)
    )


# --- validation -------------------------------------------------------------


def _validate_mode(name: str, value: str) -> None:
    """Reject an unknown section mode, naming the argument and valid choices."""
    if value not in _MODES:
        raise ValueError(
            f"{name} must be one of {_MODES}, got {value!r}."
        )


# --- objective --------------------------------------------------------------


def _objective(states: Sequence[State], lam_cost, lowerer: LatexLowerer) -> str:
    """Build the Mayer objective from minimize/maximize boundary types.

    A ``minimize`` element adds ``+w x(t)`` and a ``maximize`` element ``-w x(t)``
    (matching the solver's cost sign convention), weighted by ``lam_cost``.
    """
    terms: List[Tuple[str, str]] = []
    for state in states:
        if _is_augmented(state.name):
            continue
        base = lowerer.lower(state)
        n = state.shape[0]
        for types, when in ((state.initial_type, "t_0"), (state.final_type, "t_f")):
            if types is None:
                continue
            for i in range(n):
                sign = {"Minimize": "+", "Maximize": "-"}.get(types[i])
                if sign is None:
                    continue
                weight = lam_cost if np.ndim(lam_cost) == 0 else lam_cost[state._slice.start + i]
                sym = base + (rf"_{{{i}}}" if n > 1 else "")
                terms.append((sign, rf"{_coefficient(weight)}{sym}({when})"))

    if not terms:
        return "0"
    head_sign, head_body = terms[0]
    parts = [head_body if head_sign == "+" else f"-{head_body}"]
    parts.extend(f" {sign} {body}" for sign, body in terms[1:])
    return "".join(parts)


def _coefficient(weight) -> str:
    """Render a weight prefix, omitting it entirely when the weight is 1."""
    if float(weight) == 1.0:
        return ""
    return rf"{_scalar(weight)}\, "


# --- dynamics ---------------------------------------------------------------


def _dynamics_pairs(
    dynamics_dict: dict, lowerer: LatexLowerer
) -> List[Tuple[str, str]]:
    """Return ``(\\dot{}-symbol, rhs-latex)`` pairs for the user dynamics dict."""
    return [
        (latex_symbol(name), lowerer.lower(to_expr(expr)))
        for name, expr in dynamics_dict.items()
    ]


# --- path constraints -------------------------------------------------------


def _constraint_rows(
    symbolic, N: int, mode: str, lowerer: LatexLowerer
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Render path constraints as formulation rows and (separate) definitions.

    Constraints are walked in bucket order (ctcs, nodal, cross-node) and numbered
    ``g_i`` (inequalities) / ``h_j`` (equalities); the same label indexes both the
    formulation reference and the ``"separate"`` definition, so they line up.

    Returns:
        ``(reference_rows, definitions)`` where ``definitions`` is a list of
        ``(label, residual)`` pairs (empty unless ``mode == "separate"``).
    """
    buckets = (
        list(symbolic.constraints.ctcs)
        + list(symbolic.constraints.nodal)
        + list(symbolic.constraints.nodal_convex)
        + list(symbolic.constraints.cross_node)
        + list(symbolic.constraints.cross_node_convex)
    )

    refs: List[str] = []
    defs: List[Tuple[str, str]] = []
    n_ineq = n_eq = 0
    for con in buckets:
        inner = con.constraint
        if _only_augmented(inner):
            continue
        body = lowerer.lower(inner)
        annotation = _annotation(con, N)
        suffix = rf" \quad {annotation}" if annotation else ""

        if mode == "inline":
            refs.append(f"{body}{suffix}")
            continue

        if isinstance(inner, Equality):
            n_eq += 1
            label, op = rf"h_{{{n_eq}}}", "="
        else:
            n_ineq += 1
            label, op = rf"g_{{{n_ineq}}}", r"\le"
        refs.append(rf"{label}(x, u) {op} 0{suffix}")
        if mode == "separate":
            # Constraints are canonicalized to ``residual <op> 0``; define the
            # numbered symbol as that residual so it lines up with the reference.
            defs.append((label, lowerer.lower(inner.lhs)))

    return refs, defs


def _annotation(con, N: int) -> str:
    """Render the temporal annotation for a path constraint (``""`` if none)."""
    if isinstance(con, CTCS):
        start, end = con.nodes
        if (start, end) == (0, N):
            return r"\forall t"
        return rf"\forall t \in [t_{{{start}}}, t_{{{end}}}]"
    if isinstance(con, NodalConstraint):
        return _node_annotation(con.nodes, N)
    return ""


def _node_annotation(nodes: Sequence[int], N: int) -> str:
    """Render a node-set annotation, collapsing full coverage to ``\\forall k``."""
    nodes = list(nodes)
    if nodes == list(range(N)):
        return r"\forall k"
    if len(nodes) == 1:
        return f"k = {nodes[0]}"
    if nodes == list(range(nodes[0], nodes[-1] + 1)):
        return r"k \in \{" + f"{nodes[0]}, \\dots, {nodes[-1]}" + r"\}"
    if len(nodes) > 6:
        return r"k \in \{" + ", ".join(str(n) for n in nodes[:6]) + r", \dots\}"
    return r"k \in \{" + ", ".join(str(n) for n in nodes) + r"\}"


# --- box bounds -------------------------------------------------------------


def _box_bound_rows(
    variables: Sequence[Variable], lowerer: LatexLowerer
) -> List[str]:
    """One ``lb \\le v \\le ub`` row per variable with any finite bound."""
    rows: List[str] = []
    for var in variables:
        if _is_augmented(var.name):
            continue
        lo, hi = var.min, var.max
        has_lo = lo is not None and bool(np.any(np.isfinite(lo)))
        has_hi = hi is not None and bool(np.any(np.isfinite(hi)))
        if not (has_lo or has_hi):
            continue
        sym = lowerer.lower(var)
        if has_lo and has_hi:
            rows.append(rf"{_const(lo)} \le {sym} \le {_const(hi)}")
        elif has_lo:
            rows.append(rf"{_const(lo)} \le {sym}")
        else:
            rows.append(rf"{sym} \le {_const(hi)}")
    return rows


# --- boundary conditions ----------------------------------------------------


def _boundary_rows(states: Sequence[State], lowerer: LatexLowerer) -> List[str]:
    """Render fixed initial/final conditions (free/minimize/maximize omitted).

    A whole vector of fixed elements renders as a single ``x(t_0) = [...]`` row;
    a mix of fixed and non-fixed elements renders one row per fixed element.
    """
    rows: List[str] = []
    for state in states:
        if _is_augmented(state.name):
            continue
        sym = lowerer.lower(state)
        n = state.shape[0]
        for values, types, when in (
            (state.initial, state.initial_type, "t_0"),
            (state.final, state.final_type, "t_f"),
        ):
            if types is None:
                continue
            fixed = [t in _FIXED_TYPES for t in types]
            if not any(fixed):
                continue
            if all(fixed):
                rows.append(rf"{sym}({when}) = {_const(values)}")
            else:
                for i in range(n):
                    if fixed[i]:
                        elem = sym + (rf"_{{{i}}}" if n > 1 else "")
                        rows.append(rf"{elem}({when}) = {_scalar(values[i])}")
    return rows


# --- assembly & shared helpers ----------------------------------------------


def _aligned_formulation(objective: str, st_rows: Sequence[str]) -> str:
    """Wrap the objective and subject-to rows in an ``aligned`` environment."""
    rows: List[Tuple[str, str]] = [(r"\min_{x,\,u} \quad", objective)]
    for i, row in enumerate(st_rows):
        rows.append((r"\text{s.t.} \quad" if i == 0 else "", row))
    body = " \\\\\n".join(f"{lead} & {content}" for lead, content in rows)
    return "\\begin{aligned}\n" + body + "\n\\end{aligned}"


def _aligned_block(rows) -> str:
    """Wrap already-``&``-aligned rows in a bare ``aligned`` environment."""
    body = " \\\\\n".join(rows)
    return "\\begin{aligned}\n" + body + "\n\\end{aligned}"


def _is_augmented(name: str) -> bool:
    """True for solver-augmented variables (``_time_dilation``, ``_ctcs_aug_*``)."""
    return name.startswith("_")


def _variable_names(expr: Expr) -> List[str]:
    """Collect the names of every State/Control referenced by ``expr``."""
    names: List[str] = []

    def _visit(node: Expr) -> None:
        if isinstance(node, (State, Control)):
            names.append(node.name)

    traverse(expr, _visit)
    return names


def _only_augmented(expr: Expr) -> bool:
    """True if ``expr`` references variables and every one is ``_``-prefixed."""
    names = _variable_names(expr)
    return bool(names) and all(_is_augmented(n) for n in names)


def _const(value: np.ndarray) -> str:
    """Render a bound/value, collapsing a length-1 vector to a bare scalar."""
    arr = np.asarray(value)
    if arr.ndim == 1 and arr.shape[0] == 1:
        return format_constant(arr[0])
    return format_constant(arr)


def _scalar(value) -> str:
    """Render a scalar value via ``format_constant`` (``%g``)."""
    return format_constant(np.asarray(value))
