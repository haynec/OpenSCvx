"""JAX visitors for math expressions.

Visitors: Sin, Cos, Tan, Asin, Acos, Atan, Atan2, Square, Sqrt, Exp, Log, Abs,
          Max, Min, PositivePart, Huber, SmoothReLU, LogSumExp, Linterp, Cinterp,
          Bilerp

Lowers the scalar math AST nodes to JAX functions over ``jnp`` and
``jax.scipy``. Unlike the CVXPy backend, every node here is admissible: the
trigonometric, elementary, and reduction functions map straight onto their
``jnp`` counterparts, the smooth penalties (``PositivePart``, ``Huber``,
``SmoothReLU``, ``LogSumExp``) are the differentiable surrogates the SCP loop
relies on, and the interpolators (``Linterp``, ``Cinterp``, ``Bilerp``) evaluate
baked tabulated data while staying differentiable through JAX autodiff.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.math import (
    Abs,
    Acos,
    Asin,
    Atan,
    Atan2,
    Bilerp,
    Cinterp,
    Cos,
    Exp,
    Huber,
    Linterp,
    Log,
    LogSumExp,
    Max,
    Min,
    PositivePart,
    Sin,
    SmoothReLU,
    Sqrt,
    Square,
    Tan,
)
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(Sin)
def _visit_sin(lowerer, node: Sin):
    """Lower sine function to JAX function."""
    fO = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.sin(fO(x, u, node, params))


@visitor(Cos)
def _visit_cos(lowerer, node: Cos):
    """Lower cosine function to JAX function."""
    fO = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.cos(fO(x, u, node, params))


@visitor(Tan)
def _visit_tan(lowerer, node: Tan):
    """Lower tangent function to JAX function."""
    fO = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.tan(fO(x, u, node, params))


@visitor(Asin)
def _visit_asin(lowerer, node: Asin):
    """Lower arcsine function to JAX function."""
    fO = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.arcsin(fO(x, u, node, params))


@visitor(Acos)
def _visit_acos(lowerer, node: Acos):
    """Lower arccosine function to JAX function."""
    fO = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.arccos(fO(x, u, node, params))


@visitor(Atan)
def _visit_atan(lowerer, node: Atan):
    """Lower arctangent function to JAX function."""
    fO = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.arctan(fO(x, u, node, params))


@visitor(Atan2)
def _visit_atan2(lowerer, node: Atan2):
    """Lower two-argument arctangent function to JAX function."""
    f_y = lowerer.lower(node.y)
    f_x = lowerer.lower(node.x)
    return lambda x, u, node, params: jnp.arctan2(f_y(x, u, node, params), f_x(x, u, node, params))


@visitor(Square)
def _visit_square(lowerer, node):
    """Lower square function to JAX.

    Computes x^2 element-wise. Used in quadratic penalty methods.

    Args:
        node: Square expression node

    Returns:
        Function (x, u, node, params) -> operand^2
    """
    f = lowerer.lower(node.x)
    return lambda x, u, node, params: f(x, u, node, params) * f(x, u, node, params)


@visitor(Sqrt)
def _visit_sqrt(lowerer, node: Sqrt):
    """Lower square root to JAX function."""
    f = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.sqrt(f(x, u, node, params))


@visitor(Exp)
def _visit_exp(lowerer, node: Exp):
    """Lower exponential function to JAX function."""
    fO = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.exp(fO(x, u, node, params))


@visitor(Log)
def _visit_log(lowerer, node: Log):
    """Lower natural logarithm to JAX function."""
    fO = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.log(fO(x, u, node, params))


@visitor(Abs)
def _visit_abs(lowerer, node: Abs):
    """Lower absolute value to JAX function."""
    fO = lowerer.lower(node.operand)
    return lambda x, u, node, params: jnp.abs(fO(x, u, node, params))


@visitor(Max)
def _visit_max(lowerer, node: Max):
    """Lower element-wise maximum to JAX function."""
    fs = [lowerer.lower(op) for op in node.operands]

    def fn(x, u, node, params):
        values = [f(x, u, node, params) for f in fs]
        # jnp.maximum can take multiple arguments
        result = values[0]
        for val in values[1:]:
            result = jnp.maximum(result, val)
        return result

    return fn


@visitor(Min)
def _visit_min(lowerer, node: Min):
    """Lower element-wise minimum to JAX function."""
    fs = [lowerer.lower(op) for op in node.operands]

    def fn(x, u, node, params):
        values = [f(x, u, node, params) for f in fs]
        # jnp.minimum can take multiple arguments
        result = values[0]
        for val in values[1:]:
            result = jnp.minimum(result, val)
        return result

    return fn


@visitor(PositivePart)
def _visit_pos(lowerer, node):
    """Lower positive part function to JAX.

    Computes max(x, 0), used in penalty methods for inequality constraints.

    Args:
        node: PositivePart expression node

    Returns:
        Function (x, u, node, params) -> max(operand, 0)
    """
    f = lowerer.lower(node.x)
    return lambda x, u, node, params: jnp.maximum(f(x, u, node, params), 0.0)


@visitor(Huber)
def _visit_huber(lowerer, node):
    """Lower Huber penalty function to JAX.

    Huber penalty is quadratic for small values and linear for large values:
    - |x| <= delta: 0.5 * x^2
    - |x| > delta: delta * (|x| - 0.5 * delta)

    Args:
        node: Huber expression node with delta parameter

    Returns:
        Function (x, u, node, params) -> Huber penalty
    """
    f = lowerer.lower(node.x)
    delta = node.delta
    return lambda x, u, node, params: jnp.where(
        jnp.abs(f(x, u, node, params)) <= delta,
        0.5 * f(x, u, node, params) ** 2,
        delta * (jnp.abs(f(x, u, node, params)) - 0.5 * delta),
    )


@visitor(SmoothReLU)
def _visit_srelu(lowerer, node):
    """Lower smooth ReLU penalty function to JAX.

    Smooth approximation to ReLU: sqrt(max(x, 0)^2 + c^2) - c
    Differentiable everywhere, approaches ReLU as c -> 0.

    Args:
        node: SmoothReLU expression node with smoothing parameter c

    Returns:
        Function (x, u, node, params) -> smooth ReLU penalty
    """
    f = lowerer.lower(node.x)
    c = node.c
    # smooth_relu(pos(x)) = sqrt(pos(x)^2 + c^2) - c ; here f already includes pos inside node
    return lambda x, u, node, params: (
        jnp.sqrt(jnp.maximum(f(x, u, node, params), 0.0) ** 2 + c**2) - c
    )


@visitor(LogSumExp)
def _visit_logsumexp(lowerer, node: LogSumExp):
    """Lower log-sum-exp to JAX function.

    Computes log(sum(exp(x_i))) for multiple operands, which is a smooth
    approximation to the maximum function. Uses JAX's numerically stable
    logsumexp implementation. Performs element-wise log-sum-exp with
    broadcasting support.
    """
    fs = [lowerer.lower(op) for op in node.operands]

    def fn(x, u, node, params):
        values = [f(x, u, node, params) for f in fs]
        # Broadcast all values to the same shape, then stack along new axis
        # and compute logsumexp along that axis for element-wise operation
        broadcasted = jnp.broadcast_arrays(*values)
        stacked = jnp.stack(list(broadcasted), axis=0)
        return logsumexp(stacked, axis=0)

    return fn


@visitor(Linterp)
def _visit_linterp(lowerer, node: Linterp):
    """Lower 1D linear interpolation to JAX function.

    Uses jnp.interp which performs piecewise linear interpolation.
    For query points outside the data range, boundary values are returned.

    Args:
        node: Linterp expression node with xp, fp, and x

    Returns:
        Function (x, u, node, params) -> interpolated value(s)

    Note:
        The xp and fp arrays are typically constants (tabulated data),
        while x is typically a symbolic expression (state or derived value).
        jnp.interp is differentiable through JAX's autodiff.
    """
    f_xp = lowerer.lower(node.xp)
    f_fp = lowerer.lower(node.fp)
    f_x = lowerer.lower(node.x)

    def linterp_fn(x, u, node_idx, params):
        xp_val = f_xp(x, u, node_idx, params)
        fp_val = f_fp(x, u, node_idx, params)
        x_val = f_x(x, u, node_idx, params)
        return jnp.interp(x_val, xp_val, fp_val)

    return linterp_fn


@visitor(Cinterp)
def _visit_cinterp(lowerer, node: Cinterp):
    """Lower 1D cubic spline interpolation to JAX function.

    Evaluates a precomputed cubic spline via segment lookup and Horner's
    method.  Coefficients are baked in as constants; only the query point
    is symbolic.  For queries outside the data range the boundary segment
    polynomial is extrapolated.

    Args:
        node: Cinterp expression node with precomputed coeffs and xp_np.

    Returns:
        Function (x, u, node_idx, params) -> interpolated value(s)
    """
    f_x = lowerer.lower(node.x)
    xp = jnp.asarray(node.xp_np)
    coeffs = jnp.asarray(node.coeffs)  # (4, n_seg)

    def cinterp_fn(x, u, node_idx, params):
        x_val = f_x(x, u, node_idx, params)
        # Segment index (clamp to valid range)
        idx = jnp.searchsorted(xp, x_val, side="right") - 1
        idx = jnp.clip(idx, 0, xp.shape[0] - 2)
        t = x_val - xp[idx]
        # Horner evaluation: c0 + t*(c1 + t*(c2 + t*c3))
        c = coeffs[:, idx]
        return c[0] + t * (c[1] + t * (c[2] + t * c[3]))

    return cinterp_fn


@visitor(Bilerp)
def _visit_bilerp(lowerer, node: Bilerp):
    """Lower 2D bilinear interpolation to JAX function.

    Uses jax.scipy.ndimage.map_coordinates for bilinear interpolation on a
    regular grid. For query points outside the grid, boundary values are
    returned (clamping via mode='nearest').

    Args:
        node: Bilerp expression node with x, y, xp, yp, fp

    Returns:
        Function (x, u, node, params) -> interpolated scalar value

    Note:
        The grid arrays (xp, yp, fp) are typically constants (tabulated data),
        while x and y are symbolic expressions (state or derived values).
        Physical coordinates are converted to fractional indices before
        interpolation. The implementation is differentiable through JAX's autodiff.
    """
    f_x = lowerer.lower(node.x)
    f_y = lowerer.lower(node.y)
    f_xp = lowerer.lower(node.xp)
    f_yp = lowerer.lower(node.yp)
    f_fp = lowerer.lower(node.fp)

    def bilerp_fn(x, u, node_idx, params):
        x_val = f_x(x, u, node_idx, params)
        y_val = f_y(x, u, node_idx, params)
        xp_val = f_xp(x, u, node_idx, params)
        yp_val = f_yp(x, u, node_idx, params)
        fp_val = f_fp(x, u, node_idx, params)

        # Convert physical coordinates to fractional indices
        # jnp.interp maps physical coords to index space (handles non-uniform grids)
        idx_x = jnp.interp(x_val, xp_val, jnp.arange(len(xp_val)))
        idx_y = jnp.interp(y_val, yp_val, jnp.arange(len(yp_val)))

        # Use map_coordinates with order=1 (bilinear) and mode='nearest' (clamp)
        coords = jnp.array([[idx_x], [idx_y]])
        return jax.scipy.ndimage.map_coordinates(fp_val, coords, order=1, mode="nearest")[0]

    return bilerp_fn
