'''
Logical Operators parameterized with Generalized Mean-based Smooth Robustness (GMSR)

Author: Samet Uzun
Reference:  [https://doi.org/10.48550/arxiv.2405.10996]
            [https://doi.org/10.2514/6.2025-1895]
'''

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def _root_sum_of_product_terms(terms: ArrayLike, c: float) -> Array:
    """
    Compute the quantity

        (prod(terms) + c**K)**(1/K)

    in a numerically safer way using log-space operations, where K is the
    number of terms.

    This helper is used in the smooth logical operators below.
    """
    terms = jnp.ravel(jnp.asarray(terms))
    k = terms.shape[0]

    log_prod = jnp.sum(jnp.log(terms))
    log_c_term = k * jnp.log(c)

    return jnp.exp(jnp.logaddexp(log_prod, log_c_term) / k)


def smooth_inequality(y: ArrayLike, c: float = 1e-4) -> Array:
    """
    Smooth penalty for inequality constraints.

    smooth_inequality(y) = 0  <==>  y <= 0
    """
    y = jnp.asarray(y)
    return jnp.sqrt(jnp.maximum(y, 0.0) ** 2 + c**2) - c


def smooth_equality(y: ArrayLike, c: float = 1e-4) -> Array:
    """
    Smooth penalty for equality constraints.

    smooth_equality(y) = 0  <==>  y = 0
    """
    y = jnp.asarray(y)
    return jnp.sqrt(y**2 + c**2) - c


def AND(y: ArrayLike, c: float = 1e-4) -> Array:
    """
    Smooth parameterization for conjunction.

    AND(y) <= 0  <==>  y_i <= 0 for all i
    """
    y = jnp.asarray(y)

    positive_part = jnp.maximum(y, 0.0)
    negative_part = jnp.maximum(-y, 0.0)

    mp = jnp.mean(positive_part**2) + c
    m0 = _root_sum_of_product_terms(negative_part**2, c)

    return jnp.sqrt(mp) - jnp.sqrt(m0)


def OR(y: ArrayLike, c: float = 1e-4) -> Array:
    """
    Smooth parameterization for disjunction.

    OR(y) <= 0  <==>  y_i <= 0 for some i
    """
    y = jnp.asarray(y)
    return -AND(-y, c=c)


def IfThen(y: ArrayLike, c: float = 1e-4) -> Array:
    """
    Smooth parameterization of implication.

    IfThen(y) <= 0  <==>  (y_0 <= 0 ==> y_1 <= 0)
                    <==>  (y_0 > 0 OR y_1 <= 0)
    """
    y = jnp.asarray(y)
    return OR(jnp.array([-y[0], y[1]]), c=c)


def integer_variable(y: ArrayLike, values: ArrayLike, c: float = 1e-4) -> Array:
    """
    Smooth parameterization for discrete/integer-valued variables.

    integer_variable(y, values) = 0  <==>  y is equal to one of `values`
    """
    y = jnp.asarray(y)
    values = jnp.asarray(values)
    return OR(smooth_equality(y - values, c=c), c=c)


def AND_lite(y: ArrayLike, c: float = 1e-4) -> Array:
    """
    Lite version of the conjunction (AND) operator.
    Considers only the positive part of the AND function.

    AND_lite(y) = 0  <==>  y_i <= 0 for all i
    """
    y = jnp.asarray(y)

    mp = jnp.mean(jnp.maximum(y, 0.0) ** 2) + c
    return jnp.sqrt(mp) - jnp.sqrt(c)


def OR_lite(y: ArrayLike, c: float = 1e-4) -> Array:
    """
    Lite version of the disjunction (OR) operator.
    Considers only the positive part of the OR function.

    OR_lite(y) = 0  <==>  y_i <= 0 for some i
    """
    y = jnp.asarray(y)

    m0 = _root_sum_of_product_terms(jnp.maximum(y, 0.0) ** 2, c)
    return jnp.sqrt(m0) - jnp.sqrt(c)


def IfThen_lite(y: ArrayLike, c: float = 1e-4) -> Array:
    """
    Lite version of implication (IfThen) operator.
    Considers only the positive part of the IfThen function.

    IfThen_lite(y) = 0  <==>  (y_0 <= 0 ==> y_1 <= 0)
                        <==>  (y_0 > 0 OR y_1 <= 0)

    Notes
    -----
    This operator can also be used to enforce continuous-time satisfaction
    of an implication specification through a periodic auxiliary-state
    construction, e.g.,

        z_dot(t) = IfThen_lite([y_0(t), y_1(t)])
        z(0) = z(T)
    """
    y = jnp.asarray(y)
    return OR_lite(jnp.array([-y[0], y[1]]), c=c)
