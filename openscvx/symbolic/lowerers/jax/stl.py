"""JAX visitors and GMSR math for STL (Signal Temporal Logic) expressions.

Visitors: Or, And, IfThen, IntegerVariable

Lowers symbolic STL expression nodes to JAX functions using Generalized
Mean-based Smooth Robustness (GMSR) parameterizations.

The GMSR helper functions (AND, OR, IfThen, etc.) are pure JAX math
implementations used by the visitor functions below.

Author: Samet Uzun
Reference:  [https://doi.org/10.48550/arxiv.2405.10996]
            [https://doi.org/10.2514/6.2025-1895]
"""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from openscvx.symbolic.expr.stl import And, IfThen, IntegerVariable, Or, STLExpr
from openscvx.symbolic.lowerers.jax._registry import visitor


# ---------------------------------------------------------------------------
# GMSR primitive functions
# ---------------------------------------------------------------------------


def _root_sum_of_product_terms(terms: ArrayLike, c: float) -> Array:
    """Compute (prod(terms) + c**K)**(1/K) in log-space, guarding against NaNs.

    The original formulation,

        prod(terms) ** (1 / K),

    can overflow/underflow or introduce NaNs when any term is zero (log(0)),
    when K = 0, or when c is extremely small. We:

    - Clamp terms to be non-negative and add an epsilon before log.
    - Handle the K = 0 case explicitly (return 0.0).
    - Clamp c away from zero before logging.
    """
    terms = jnp.ravel(jnp.asarray(terms))
    k = terms.size

    def _empty_case():
        return jnp.array(0.0, dtype=terms.dtype if terms.size > 0 else jnp.float32)

    def _nonempty_case():
        eps = jnp.finfo(terms.dtype).tiny
        safe_terms = jnp.maximum(terms, 0.0)
        log_terms = jnp.log(jnp.maximum(safe_terms, eps))

        safe_c = jnp.maximum(jnp.asarray(c, dtype=terms.dtype), eps)
        log_prod = jnp.sum(log_terms)
        log_c_term = k * jnp.log(safe_c)

        return jnp.exp(jnp.logaddexp(log_prod, log_c_term) / k)

    return jnp.where(k == 0, _empty_case(), _nonempty_case())


def _smooth_equality(y: ArrayLike, c: float = 1e-8) -> Array:
    """Smooth penalty for equality constraints: returns 0 iff y = 0."""
    y = jnp.asarray(y)
    return jnp.sqrt(y**2 + c**2) - c


def AND(y: ArrayLike, c: float = 1e-8) -> Array:
    """Smooth conjunction: AND(y) <= 0  iff  y_i <= 0 for all i."""
    y = jnp.asarray(y)

    positive_part = jnp.maximum(y, 0.0)
    negative_part = jnp.maximum(-y, 0.0)

    mp = jnp.mean(positive_part**2) + c
    m0 = _root_sum_of_product_terms(negative_part**2, c)

    return jnp.sqrt(mp) - jnp.sqrt(m0)


def OR(y: ArrayLike, c: float = 1e-8) -> Array:
    """Smooth disjunction: OR(y) <= 0  iff  y_i <= 0 for some i."""
    y = jnp.asarray(y)
    return -AND(-y, c=c)


def gmsr_IfThen(y: ArrayLike, c: float = 1e-8) -> Array:
    """Smooth implication: IfThen(y) <= 0  iff  (y_0 <= 0 => y_1 <= 0)."""
    y = jnp.asarray(y)
    return OR(jnp.array([-y[0], y[1]]), c=c)


def integer_variable(y: ArrayLike, values: ArrayLike, c: float = 1e-8) -> Array:
    """Smooth discrete constraint: returns 0 iff y equals one of values."""
    y = jnp.asarray(y)
    values = jnp.asarray(values)
    return OR(_smooth_equality(y - values, c=c), c=c)


def AND_lite(y: ArrayLike, c: float = 1e-8) -> Array:
    """Lite conjunction (positive part only): AND_lite(y) = 0  iff  y_i <= 0 for all i."""
    y = jnp.asarray(y)

    mp = jnp.mean(jnp.maximum(y, 0.0) ** 2) + c
    return jnp.sqrt(mp) - jnp.sqrt(c)


def OR_lite(y: ArrayLike, c: float = 1e-8) -> Array:
    """Lite disjunction (positive part only): OR_lite(y) = 0  iff  y_i <= 0 for some i."""
    y = jnp.asarray(y)

    m0 = _root_sum_of_product_terms(jnp.maximum(y, 0.0) ** 2, c)
    return jnp.sqrt(m0) - jnp.sqrt(c)


def gmsr_IfThen_lite(y: ArrayLike, c: float = 1e-8) -> Array:
    """Lite implication: IfThen_lite(y) = 0  iff  (y_0 <= 0 => y_1 <= 0).

    Can enforce continuous-time implication via periodic auxiliary state:
        z_dot(t) = IfThen_lite([y_0(t), y_1(t)])
        z(0) = z(T)
    """
    y = jnp.asarray(y)
    return OR_lite(jnp.array([-y[0], y[1]]), c=c)


# ---------------------------------------------------------------------------
# Visitor functions
# ---------------------------------------------------------------------------


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
        return -integer_variable(y, values, c=c)

    return integer_var_fn
