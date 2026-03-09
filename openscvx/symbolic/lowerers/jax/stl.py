"""JAX visitors for GMSR-based STL (Signal Temporal Logic) expressions.

Visitors: Or, And, IfThen

Lowers symbolic STL expression nodes to JAX functions using Generalized
Mean-based Smooth Robustness (GMSR) parameterizations.
"""

import jax.numpy as jnp

from openscvx.symbolic.expr.stl import And, IfThen, IntegerVariable, Or, STLExpr
from openscvx.symbolic.lowerers.jax._registry import visitor
from openscvx.symbolic.lowerers.jax.gmsr import (
    AND,
    OR,
    AND_lite,
    OR_lite,
)
from openscvx.symbolic.lowerers.jax.gmsr import IfThen as gmsr_IfThen
from openscvx.symbolic.lowerers.jax.gmsr import IfThen_lite as gmsr_IfThen_lite
from openscvx.symbolic.lowerers.jax.gmsr import integer_variable as gmsr_integer_variable


def _lower_predicate_residuals(lowerer, predicates):
    """Extract constraint residuals from predicates and lower them to JAX functions.

    For Constraint (lhs <= rhs): residual = lhs - rhs (negative when satisfied)
    For STLExpr: negate the robustness to get residual (negative when satisfied)

    Returns a list of lowered JAX functions, each returning a scalar residual.
    """
    from openscvx.symbolic.expr.arithmetic import Neg, Sub
    from openscvx.symbolic.expr.constraint import Constraint

    residual_fns = []
    for pred in predicates:
        if isinstance(pred, Constraint):
            residual_expr = Sub(pred.lhs, pred.rhs)
            residual_fns.append(lowerer.lower(residual_expr))
        elif isinstance(pred, STLExpr):
            # STLExpr returns robustness (positive when satisfied);
            # negate to get residual (negative when satisfied)
            residual_fns.append(lowerer.lower(Neg(pred)))
        else:
            raise TypeError(f"Unexpected predicate type: {type(pred)}")
    return residual_fns


@visitor(Or)
def _visit_or(lowerer, node: Or):
    """Lower GMSR disjunction (Or) to JAX.

    GMSR OR(y) <= 0 when some y_i <= 0. We negate the output to get
    robustness (positive when at least one predicate is satisfied).
    """
    residual_fns = _lower_predicate_residuals(lowerer, node.predicates)
    gmsr_fn = OR_lite if node.lite else OR
    c = node.c

    def or_fn(x, u, node_idx, params):
        residuals = jnp.array([fn(x, u, node_idx, params) for fn in residual_fns])
        return -gmsr_fn(residuals, c=c)

    return or_fn


@visitor(And)
def _visit_and(lowerer, node: And):
    """Lower GMSR conjunction (And) to JAX.

    GMSR AND(y) <= 0 when all y_i <= 0. We negate the output to get
    robustness (positive when all predicates are satisfied).
    """
    residual_fns = _lower_predicate_residuals(lowerer, node.predicates)
    gmsr_fn = AND_lite if node.lite else AND
    c = node.c

    def and_fn(x, u, node_idx, params):
        residuals = jnp.array([fn(x, u, node_idx, params) for fn in residual_fns])
        return -gmsr_fn(residuals, c=c)

    return and_fn


@visitor(IfThen)
def _visit_ifthen(lowerer, node: IfThen):
    """Lower GMSR implication (IfThen) to JAX.

    GMSR IfThen([y0, y1]) <= 0 when (y0 <= 0 => y1 <= 0). We negate the
    output to get robustness (positive when the implication holds).
    """
    cond_fns = _lower_predicate_residuals(lowerer, [node.condition])
    conseq_fns = _lower_predicate_residuals(lowerer, [node.consequent])
    gmsr_fn = gmsr_IfThen_lite if node.lite else gmsr_IfThen
    c = node.c

    def ifthen_fn(x, u, node_idx, params):
        cond_residual = cond_fns[0](x, u, node_idx, params)
        conseq_residual = conseq_fns[0](x, u, node_idx, params)
        return -gmsr_fn(jnp.array([cond_residual, conseq_residual]), c=c)

    return ifthen_fn


@visitor(IntegerVariable)
def _visit_integer_variable(lowerer, node: IntegerVariable):
    """Lower GMSR IntegerVariable to JAX.

    GMSR integer_variable(y, values) >= 0 always, and equals 0 when y
    matches one of the allowed values. We negate for the robustness convention.
    """
    expr_fn = lowerer.lower(node.expr)
    values = jnp.asarray(node.values)
    c = node.c

    def integer_var_fn(x, u, node_idx, params):
        y = expr_fn(x, u, node_idx, params)
        return -gmsr_integer_variable(y, values, c=c)

    return integer_var_fn
