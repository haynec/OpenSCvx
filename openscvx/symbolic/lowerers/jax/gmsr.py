"""Generalized Mean-based Smooth Robustness (GMSR) helper functions.

Pure JAX math implementations for smooth logical operators. These are used
by the STL visitor module (stl.py) to lower symbolic STL expressions.

Author: Samet Uzun
Reference:  [https://doi.org/10.48550/arxiv.2405.10996]
            [https://doi.org/10.2514/6.2025-1895]
"""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


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

    # If there are no terms, return 0.0 (neutral element for our use-cases)
    def _empty_case():
        return jnp.array(0.0, dtype=terms.dtype if terms.size > 0 else jnp.float32)

    def _nonempty_case():
        # Ensure non-negative and avoid log(0) exactly
        eps = jnp.finfo(terms.dtype).tiny
        safe_terms = jnp.maximum(terms, 0.0)
        log_terms = jnp.log(jnp.maximum(safe_terms, eps))

        # Guard c away from zero as well
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


def IfThen(y: ArrayLike, c: float = 1e-8) -> Array:
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


def IfThen_lite(y: ArrayLike, c: float = 1e-8) -> Array:
    """Lite implication: IfThen_lite(y) = 0  iff  (y_0 <= 0 => y_1 <= 0).

    Can enforce continuous-time implication via periodic auxiliary state:
        z_dot(t) = IfThen_lite([y_0(t), y_1(t)])
        z(0) = z(T)
    """
    y = jnp.asarray(y)
    return OR_lite(jnp.array([-y[0], y[1]]), c=c)
