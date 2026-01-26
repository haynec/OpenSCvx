"""CVXPy visitors for math expressions.

Visitors: Sin, Cos, Tan, Exp, Log, Abs, PositivePart, Square, Huber,
          SmoothReLU, Sqrt, Max, LogSumExp, Linterp, Bilerp
"""

import cvxpy as cp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.math import (
    Abs,
    Bilerp,
    Cos,
    Exp,
    Huber,
    Linterp,
    Log,
    LogSumExp,
    Max,
    PositivePart,
    Sin,
    SmoothReLU,
    Sqrt,
    Square,
    Tan,
)
from openscvx.symbolic.lowerers.cvxpy._registry import visitor  # noqa: F401


@visitor(Sin)
def _visit_sin(lowerer, node: Sin) -> cp.Expression:
    """Raise NotImplementedError for sine function.

    Sine is not DCP-compliant in CVXPy as it is neither convex nor concave.

    Args:
        node: Sin expression node

    Raises:
        NotImplementedError: Always raised since sine is not DCP-compliant

    Note:
        For constraints involving trigonometric functions:
        - Use piecewise-linear approximations, or
        - Handle in the JAX dynamics/constraint layer instead of CVXPy
    """
    raise NotImplementedError(
        "Trigonometric functions like Sin are not DCP-compliant in CVXPy. "
        "Consider using piecewise-linear approximations or handle these constraints "
        "in the dynamics (JAX) layer instead."
    )


@visitor(Cos)
def _visit_cos(lowerer, node: Cos) -> cp.Expression:
    """Raise NotImplementedError for cosine function.

    Cosine is not DCP-compliant in CVXPy as it is neither convex nor concave.

    Args:
        node: Cos expression node

    Raises:
        NotImplementedError: Always raised since cosine is not DCP-compliant

    Note:
        For constraints involving trigonometric functions:
        - Use piecewise-linear approximations, or
        - Handle in the JAX dynamics/constraint layer instead of CVXPy
    """
    raise NotImplementedError(
        "Trigonometric functions like Cos are not DCP-compliant in CVXPy. "
        "Consider using piecewise-linear approximations or handle these constraints "
        "in the dynamics (JAX) layer instead."
    )


@visitor(Tan)
def _visit_tan(lowerer, node: Tan) -> cp.Expression:
    """Raise NotImplementedError for tangent function.

    Tangent is not DCP-compliant in CVXPy as it is neither convex nor concave.

    Args:
        node: Tan expression node

    Raises:
        NotImplementedError: Always raised since tangent is not DCP-compliant

    Note:
        For constraints involving trigonometric functions:
        - Use piecewise-linear approximations, or
        - Handle in the JAX dynamics/constraint layer instead of CVXPy
    """
    raise NotImplementedError(
        "Trigonometric functions like Tan are not DCP-compliant in CVXPy. "
        "Consider using piecewise-linear approximations or handle these constraints "
        "in the dynamics (JAX) layer instead."
    )


@visitor(Exp)
def _visit_exp(lowerer, node: Exp) -> cp.Expression:
    """Lower exponential function to CVXPy expression.

    Exponential is a convex function and DCP-compliant when used in
    appropriate contexts (e.g., minimizing exp(x) or constraints like
    exp(x) <= c).

    Args:
        node: Exp expression node

    Returns:
        CVXPy expression representing exp(operand)

    Note:
        Exponential is convex increasing, so it's valid in:
        - Objective: minimize exp(x)
        - Constraints: exp(x) <= c (convex constraint)
    """
    operand = lowerer.lower(node.operand)
    return cp.exp(operand)


@visitor(Log)
def _visit_log(lowerer, node: Log) -> cp.Expression:
    """Lower natural logarithm to CVXPy expression.

    Logarithm is a concave function and DCP-compliant when used in
    appropriate contexts (e.g., maximizing log(x) or constraints like
    log(x) >= c).

    Args:
        node: Log expression node

    Returns:
        CVXPy expression representing log(operand)

    Note:
        Logarithm is concave increasing, so it's valid in:
        - Objective: maximize log(x)
        - Constraints: log(x) >= c (concave constraint, or equivalently c <= log(x))
    """
    operand = lowerer.lower(node.operand)
    return cp.log(operand)


@visitor(Abs)
def _visit_abs(lowerer, node: Abs) -> cp.Expression:
    """Lower absolute value to CVXPy expression.

    Absolute value is a convex function and DCP-compliant when used in
    appropriate contexts (e.g., minimizing |x| or constraints like |x| <= c).

    Args:
        node: Abs expression node

    Returns:
        CVXPy expression representing |operand|

    Note:
        Absolute value is convex, so it's valid in:
        - Objective: minimize abs(x)
        - Constraints: abs(x) <= c (convex constraint)
    """
    operand = lowerer.lower(node.operand)
    return cp.abs(operand)


@visitor(PositivePart)
def _visit_pos(lowerer, node: PositivePart) -> cp.Expression:
    """Lower positive part function to CVXPy.

    Computes max(x, 0), which is convex. Used in penalty methods for
    inequality constraints.

    Args:
        node: PositivePart expression node

    Returns:
        CVXPy expression representing max(operand, 0)

    Note:
        Positive part is convex and commonly used in hinge loss and
        penalty methods for inequality constraints.
    """
    operand = lowerer.lower(node.x)
    return cp.maximum(operand, 0.0)


@visitor(Square)
def _visit_square(lowerer, node: Square) -> cp.Expression:
    """Lower square function to CVXPy.

    Computes x^2, which is convex. Used in quadratic penalty methods
    and least-squares objectives.

    Args:
        node: Square expression node

    Returns:
        CVXPy expression representing operand^2

    Note:
        Square is convex increasing for x >= 0 and convex decreasing for
        x <= 0. It's always convex overall.
    """
    operand = lowerer.lower(node.x)
    return cp.square(operand)


@visitor(Sqrt)
def _visit_sqrt(lowerer, node: Sqrt) -> cp.Expression:
    """Lower square root to CVXPy expression.

    Square root is concave and DCP-compliant when used appropriately
    (e.g., maximizing sqrt(x) or constraints like sqrt(x) >= c).

    Args:
        node: Sqrt expression node

    Returns:
        CVXPy expression representing sqrt(operand)

    Note:
        Square root is concave increasing for x > 0. Valid in:
        - Objective: maximize sqrt(x)
        - Constraints: sqrt(x) >= c (concave constraint)
    """
    operand = lowerer.lower(node.operand)
    return cp.sqrt(operand)


@visitor(Huber)
def _visit_huber(lowerer, node: Huber) -> cp.Expression:
    """Lower Huber penalty function to CVXPy.

    Huber penalty is quadratic for small values and linear for large values,
    providing robustness to outliers. It is convex and DCP-compliant.

    The Huber function is defined as:
    - |x| <= delta: 0.5 * x^2
    - |x| > delta: delta * (|x| - 0.5 * delta)

    Args:
        node: Huber expression node with delta parameter

    Returns:
        CVXPy expression representing Huber penalty

    Note:
        Huber loss is convex and combines the benefits of squared error
        (smooth, differentiable) and absolute error (robust to outliers).
    """
    operand = lowerer.lower(node.x)
    return cp.huber(operand, M=node.delta)


@visitor(SmoothReLU)
def _visit_srelu(lowerer, node: SmoothReLU) -> cp.Expression:
    """Lower smooth ReLU penalty function to CVXPy.

    Smooth approximation to ReLU: sqrt(max(x, 0)^2 + c^2) - c
    Differentiable everywhere, approaches ReLU as c -> 0. Convex.

    Args:
        node: SmoothReLU expression node with smoothing parameter c

    Returns:
        CVXPy expression representing smooth ReLU penalty

    Note:
        This provides a smooth, convex approximation to the ReLU function
        max(x, 0). The parameter c controls the smoothness: smaller c gives
        a better approximation but less smoothness.
    """
    operand = lowerer.lower(node.x)
    c = node.c
    # smooth_relu(x) = sqrt(max(x, 0)^2 + c^2) - c
    pos_part = cp.maximum(operand, 0.0)
    # For SmoothReLU, we use the 2-norm formulation
    return cp.sqrt(cp.sum_squares(pos_part) + c**2) - c


@visitor(Max)
def _visit_max(lowerer, node: Max) -> cp.Expression:
    """Lower element-wise maximum to CVXPy expression.

    Maximum is convex (pointwise max of convex functions is convex).

    Args:
        node: Max expression node with multiple operands

    Returns:
        CVXPy expression representing element-wise maximum

    Note:
        For multiple operands, chains binary maximum operations.
        Maximum preserves convexity.
    """
    operands = [lowerer.lower(op) for op in node.operands]
    # CVXPy's maximum can take multiple arguments
    if len(operands) == 2:
        return cp.maximum(operands[0], operands[1])
    else:
        # For more than 2 operands, chain maximum calls
        result = cp.maximum(operands[0], operands[1])
        for op in operands[2:]:
            result = cp.maximum(result, op)
        return result


@visitor(LogSumExp)
def _visit_logsumexp(lowerer, node: LogSumExp) -> cp.Expression:
    """Lower log-sum-exp to CVXPy expression.

    Log-sum-exp is convex and is a smooth approximation to the maximum function.
    CVXPy's log_sum_exp atom computes log(sum(exp(x_i))) for stacked operands.

    Args:
        node: LogSumExp expression node with multiple operands

    Returns:
        CVXPy expression representing log-sum-exp

    Note:
        Log-sum-exp is convex and DCP-compliant. It satisfies:
        max(x₁, ..., xₙ) ≤ logsumexp(x₁, ..., xₙ) ≤ max(x₁, ..., xₙ) + log(n)
    """
    operands = [lowerer.lower(op) for op in node.operands]

    # CVXPy's log_sum_exp expects a stacked expression with an axis parameter
    # For element-wise log-sum-exp, we stack along a new axis and reduce along it
    if len(operands) == 1:
        return operands[0]

    # Stack operands along a new axis (axis 0) and compute log_sum_exp along that axis
    stacked = cp.vstack(operands)
    return cp.log_sum_exp(stacked, axis=0)


@visitor(Linterp)
def _visit_linterp(lowerer, node: Linterp) -> cp.Expression:
    """Raise NotImplementedError for linear interpolation.

    Linear interpolation (Linterp) is not DCP-compliant in CVXPy as it
    represents a piecewise-linear function that is neither convex nor
    concave in general.

    Args:
        node: Linterp expression node

    Raises:
        NotImplementedError: Always raised since Linterp is not DCP-compliant
    """
    raise NotImplementedError("Linear interpolation (Linterp) is not DCP-compliant in CVXPy.")


@visitor(Bilerp)
def _visit_bilerp(lowerer, node: Bilerp) -> cp.Expression:
    """Raise NotImplementedError for bilinear interpolation.

    Bilinear interpolation (Bilerp) is not DCP-compliant in CVXPy as it
    represents a nonlinear function that is neither convex nor concave.

    Args:
        node: Bilerp expression node

    Raises:
        NotImplementedError: Always raised since Bilerp is not DCP-compliant
    """
    raise NotImplementedError("Bilinear interpolation (Bilerp) is not DCP-compliant in CVXPy.")
