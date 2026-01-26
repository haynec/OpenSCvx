"""Mathematical functions for symbolic expressions.

This module provides common mathematical operations used in optimization problems,
including trigonometric functions, exponential functions, and smooth approximations
of non-differentiable operations. All functions are element-wise and preserve the
shape of their inputs.

Function Categories:
    - **Trigonometric:** `Sin`, `Cos`, `Tan` - Standard trigonometric functions
    - **Exponential and Roots:** `Exp`, `Log`, `Sqrt`, `Square` - Exponential, logarithm, square
        root, and squaring operations
    - **Absolute Value:** `Abs` - Element-wise absolute value function
    - **Smooth Approximations:** `PositivePart`, `Huber`, `SmoothReLU` - Smooth, differentiable
        approximations of non-smooth functions like max(0, x) and absolute value
    - **Reductions:** `Max` - Maximum over elements
    - **Smooth Maximum:** `LogSumExp` - Log-sum-exp function, a smooth approximation to maximum

Example:
    Using trigonometric functions in dynamics::

        import openscvx as ox

        # Pendulum dynamics: theta_ddot = -g/L * sin(theta)
        theta = ox.State("theta", shape=(1,))
        theta_dot = ox.State("theta_dot", shape=(1,))
        g, L = 9.81, 1.0

        theta_ddot = -(g / L) * ox.Sin(theta)

    Smooth penalty functions for constraints::

        # Soft constraint using smooth ReLU
        x = ox.Variable("x", shape=(3,))
        penalty = ox.SmoothReLU(ox.Norm(x) - 1.0)  # Penalize norm > 1
"""

import hashlib
import struct
from typing import Tuple, Union

import numpy as np

from .expr import Expr, to_expr


class Sin(Expr):
    """Element-wise sine function for symbolic expressions.

    Computes the sine of each element in the operand. Preserves the shape
    of the input expression.

    Attributes:
        operand: Expression to apply sine function to

    Example:
        Define a Sin expression:

            theta = Variable("theta", shape=(3,))
            sin_theta = Sin(theta)
    """

    def __init__(self, operand: Union[Expr, float, int, np.ndarray]):
        """Initialize a sine operation.

        Args:
            operand: Expression to apply sine function to
        """
        self.operand = operand

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        operand = self.operand.canonicalize()
        return Sin(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Sin preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self) -> str:
        return f"(sin({self.operand!r}))"


class Cos(Expr):
    """Element-wise cosine function for symbolic expressions.

    Computes the cosine of each element in the operand. Preserves the shape
    of the input expression.

    Attributes:
        operand: Expression to apply cosine function to

    Example:
        Define a Cos expression:

            theta = Variable("theta", shape=(3,))
            cos_theta = Cos(theta)
    """

    def __init__(self, operand: Union[Expr, float, int, np.ndarray]):
        """Initialize a cosine operation.

        Args:
            operand: Expression to apply cosine function to
        """
        self.operand = operand

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        operand = self.operand.canonicalize()
        return Cos(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Cos preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self) -> str:
        return f"(cos({self.operand!r}))"


class Tan(Expr):
    """Element-wise tangent function for symbolic expressions.

    Computes the tangent of each element in the operand. Preserves the shape
    of the input expression.

    Attributes:
        operand: Expression to apply tangent function to

    Example:
        Define a Tan expression:

            theta = Variable("theta", shape=(3,))
            tan_theta = Tan(theta)

    Note:
        Tan is only supported for JAX lowering. CVXPy lowering will raise
        NotImplementedError since tangent is not DCP-compliant.
    """

    def __init__(self, operand: Union[Expr, float, int, np.ndarray]):
        """Initialize a tangent operation.

        Args:
            operand: Expression to apply tangent function to
        """
        self.operand = operand

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        operand = self.operand.canonicalize()
        return Tan(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Tan preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self) -> str:
        return f"(tan({self.operand!r}))"


class Square(Expr):
    """Element-wise square function for symbolic expressions.

    Computes the square (x^2) of each element in the operand. Preserves the
    shape of the input expression. This is more efficient than using Power(x, 2)
    for some optimization backends.

    Attributes:
        x: Expression to square

    Example:
        Define a Square expression:

            v = Variable("v", shape=(3,))
            v_squared = Square(v)  # Equivalent to v ** 2
    """

    def __init__(self, x: Union[Expr, float, int, np.ndarray]):
        """Initialize a square operation.

        Args:
            x: Expression to square
        """
        self.x = to_expr(x)

    def children(self):
        return [self.x]

    def canonicalize(self) -> "Expr":
        x = self.x.canonicalize()
        return Square(x)

    def check_shape(self) -> Tuple[int, ...]:
        """x^2 preserves the shape of x."""
        return self.x.check_shape()

    def __repr__(self) -> str:
        return f"({self.x!r})^2"


class Sqrt(Expr):
    """Element-wise square root function for symbolic expressions.

    Computes the square root of each element in the operand. Preserves the
    shape of the input expression.

    Attributes:
        operand: Expression to apply square root to

    Example:
        Define a Sqrt expression:

            x = Variable("x", shape=(3,))
            sqrt_x = Sqrt(x)
    """

    def __init__(self, operand: Union[Expr, float, int, np.ndarray]):
        """Initialize a square root operation.

        Args:
            operand: Expression to apply square root to
        """
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        operand = self.operand.canonicalize()
        return Sqrt(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Sqrt preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self) -> str:
        return f"sqrt({self.operand!r})"


class Exp(Expr):
    """Element-wise exponential function for symbolic expressions.

    Computes e^x for each element in the operand, where e is Euler's number.
    Preserves the shape of the input expression.

    Attributes:
        operand: Expression to apply exponential function to

    Example:
        Define an Exp expression:

            x = Variable("x", shape=(3,))
            exp_x = Exp(x)
    """

    def __init__(self, operand: Union[Expr, float, int, np.ndarray]):
        """Initialize an exponential operation.

        Args:
            operand: Expression to apply exponential function to
        """
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        operand = self.operand.canonicalize()
        return Exp(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Exp preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self) -> str:
        return f"exp({self.operand!r})"


class Log(Expr):
    """Element-wise natural logarithm function for symbolic expressions.

    Computes the natural logarithm (base e) of each element in the operand.
    Preserves the shape of the input expression.

    Attributes:
        operand: Expression to apply logarithm to

    Example:
        Define a Log expression:

            x = Variable("x", shape=(3,))
            log_x = Log(x)
    """

    def __init__(self, operand: Union[Expr, float, int, np.ndarray]):
        """Initialize a natural logarithm operation.

        Args:
            operand: Expression to apply logarithm to
        """
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        operand = self.operand.canonicalize()
        return Log(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Log preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self) -> str:
        return f"log({self.operand!r})"


class Abs(Expr):
    """Element-wise absolute value function for symbolic expressions.

    Computes the absolute value (|x|) of each element in the operand. Preserves
    the shape of the input expression. The absolute value function is convex
    and DCP-compliant in CVXPy.

    Attributes:
        operand: Expression to apply absolute value to

    Example:
        Define an Abs expression:

            x = Variable("x", shape=(3,))
            abs_x = Abs(x)  # Element-wise |x|
    """

    def __init__(self, operand: Union[Expr, float, int, np.ndarray]):
        """Initialize an absolute value operation.

        Args:
            operand: Expression to apply absolute value to
        """
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        operand = self.operand.canonicalize()
        return Abs(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Abs preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self) -> str:
        return f"abs({self.operand!r})"


class Max(Expr):
    """Element-wise maximum function for symbolic expressions.

    Computes the element-wise maximum across two or more operands. Supports
    broadcasting following NumPy rules. During canonicalization, nested Max
    operations are flattened and constants are folded.

    Attributes:
        operands: List of expressions to compute maximum over

    Example:
        Define a Max expression:

            x = Variable("x", shape=(3,))
            y = Variable("y", shape=(3,))
            max_xy = Max(x, y, 0)  # Element-wise max(x, y, 0)
    """

    def __init__(self, *args: Union[Expr, float, int, np.ndarray]):
        """Initialize a maximum operation.

        Args:
            *args: Two or more expressions to compute maximum over

        Raises:
            ValueError: If fewer than two operands are provided
        """
        if len(args) < 2:
            raise ValueError("Max requires two or more operands")
        self.operands = [to_expr(a) for a in args]

    def children(self):
        return list(self.operands)

    def canonicalize(self) -> "Expr":
        """Canonicalize max: flatten nested Max, fold constants."""
        from .expr import Constant

        operands = []
        const_vals = []

        for op in self.operands:
            c = op.canonicalize()
            if isinstance(c, Max):
                operands.extend(c.operands)
            elif isinstance(c, Constant):
                const_vals.append(c.value)
            else:
                operands.append(c)

        # If we have constants, compute their max and keep it
        if const_vals:
            max_const = np.maximum.reduce(const_vals)
            operands.append(Constant(max_const))

        if not operands:
            raise ValueError("Max must have at least one operand after canonicalization")
        if len(operands) == 1:
            return operands[0]
        return Max(*operands)

    def check_shape(self) -> Tuple[int, ...]:
        """Max broadcasts shapes like NumPy."""
        shapes = [child.check_shape() for child in self.children()]
        try:
            return np.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"Max shapes not broadcastable: {shapes}") from e

    def __repr__(self) -> str:
        inner = ", ".join(repr(op) for op in self.operands)
        return f"max({inner})"


# Penalty function building blocks
class PositivePart(Expr):
    """Positive part function for symbolic expressions.

    Computes max(x, 0) element-wise, effectively zeroing out negative values
    while preserving positive values. This is also known as the ReLU (Rectified
    Linear Unit) function and is commonly used as a penalty function building
    block in optimization.

    Attributes:
        x: Expression to apply positive part function to

    Example:
        Define a PositivePart expression:

            constraint_violation = x - 10
            penalty = PositivePart(constraint_violation)  # Penalizes x > 10
    """

    def __init__(self, x: Union[Expr, float, int, np.ndarray]):
        """Initialize a positive part operation.

        Args:
            x: Expression to apply positive part function to
        """
        self.x = to_expr(x)

    def children(self):
        return [self.x]

    def canonicalize(self) -> "Expr":
        x = self.x.canonicalize()
        return PositivePart(x)

    def check_shape(self) -> Tuple[int, ...]:
        """pos(x) = max(x, 0) preserves the shape of x."""
        return self.x.check_shape()

    def __repr__(self) -> str:
        return f"pos({self.x!r})"


class Huber(Expr):
    """Huber penalty function for symbolic expressions.

    The Huber penalty is a smooth approximation to the absolute value function
    that is quadratic for small values (|x| < delta) and linear for large values
    (|x| >= delta). This makes it more robust to outliers than squared penalties
    while maintaining smoothness.

    The Huber function is defined as:
    - (x^2) / (2*delta)           for |x| <= delta
    - |x| - delta/2               for |x| > delta

    Attributes:
        x: Expression to apply Huber penalty to
        delta: Threshold parameter controlling the transition point (default: 0.25)

    Example:
        Define a Huber penalty expression:

            residual = y_measured - y_predicted
            penalty = Huber(residual, delta=0.5)
    """

    def __init__(self, x: Union[Expr, float, int, np.ndarray], delta: float = 0.25):
        """Initialize a Huber penalty operation.

        Args:
            x: Expression to apply Huber penalty to
            delta: Threshold parameter for quadratic-to-linear transition (default: 0.25)
        """
        self.x = to_expr(x)
        self.delta = float(delta)

    def children(self):
        return [self.x]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand but preserve delta parameter."""
        x = self.x.canonicalize()
        return Huber(x, delta=self.delta)

    def check_shape(self) -> Tuple[int, ...]:
        """Huber penalty preserves the shape of x."""
        return self.x.check_shape()

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Hash Huber including its delta parameter.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"Huber")
        # Hash delta as bytes
        hasher.update(struct.pack(">d", self.delta))
        # Hash the operand
        self.x._hash_into(hasher)

    def __repr__(self) -> str:
        return f"huber({self.x!r}, delta={self.delta})"


class SmoothReLU(Expr):
    """Smooth approximation to the ReLU (positive part) function.

    Computes a smooth, differentiable approximation to max(x, 0) using the formula:
    sqrt(max(x, 0)^2 + c^2) - c

    The parameter c controls the smoothness: smaller values give a sharper
    transition, while larger values produce a smoother approximation. As c
    approaches 0, this converges to the standard ReLU function.

    This is particularly useful in optimization contexts where smooth gradients
    are required, such as in penalty methods for constraint handling (CTCS).

    Attributes:
        x: Expression to apply smooth ReLU to
        c: Smoothing parameter (default: 1e-8)

    Example:
        Define a smooth ReLU expression:

            constraint_violation = x - 10
            penalty = SmoothReLU(constraint_violation, c=1e-6)
    """

    def __init__(self, x: Union[Expr, float, int, np.ndarray], c: float = 1e-8):
        """Initialize a smooth ReLU operation.

        Args:
            x: Expression to apply smooth ReLU to
            c: Smoothing parameter controlling transition sharpness (default: 1e-8)
        """
        self.x = to_expr(x)
        self.c = float(c)

    def children(self):
        return [self.x]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand but preserve c parameter."""
        x = self.x.canonicalize()
        return SmoothReLU(x, c=self.c)

    def check_shape(self) -> Tuple[int, ...]:
        """Smooth ReLU preserves the shape of x."""
        return self.x.check_shape()

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Hash SmoothReLU including its c parameter.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"SmoothReLU")
        # Hash c as bytes
        hasher.update(struct.pack(">d", self.c))
        # Hash the operand
        self.x._hash_into(hasher)

    def __repr__(self) -> str:
        return f"smooth_relu({self.x!r}, c={self.c})"


class LogSumExp(Expr):
    """Log-sum-exp function for symbolic expressions.

    Computes the log-sum-exp (LSE) of multiple operands, which is a smooth,
    differentiable approximation to the maximum function. The log-sum-exp is
    defined as:

        logsumexp(x₁, x₂, ..., xₙ) = log(exp(x₁) + exp(x₂) + ... + exp(xₙ))

    This function is numerically stable and is commonly used in optimization
    as a smooth alternative to the non-differentiable maximum function. It
    satisfies the inequality:

        max(x₁, x₂, ..., xₙ) ≤ logsumexp(x₁, x₂, ..., xₙ) ≤ max(x₁, x₂, ..., xₙ) + log(n)

    The log-sum-exp is convex and is particularly useful for:
    - Smooth approximations of maximum constraints
    - Soft maximum operations in neural networks
    - Relaxing logical OR operations in STL specifications

    Attributes:
        operands: List of expressions to compute log-sum-exp over

    Example:
        Define a LogSumExp expression:

            x = Variable("x", shape=(3,))
            y = Variable("y", shape=(3,))
            z = Variable("z", shape=(3,))
            lse = LogSumExp(x, y, z)  # Smooth approximation to max(x, y, z)

        Use in STL relaxation:

            import openscvx as ox
            # Relax: Or(φ₁, φ₂) using log-sum-exp
            phi1 = ox.Norm(x - goal1) - 0.5
            phi2 = ox.Norm(x - goal2) - 0.5
            relaxed_or = LogSumExp(phi1, phi2) >= 0
    """

    def __init__(self, *args: Union[Expr, float, int, np.ndarray]):
        """Initialize a log-sum-exp operation.

        Args:
            *args: Two or more expressions to compute log-sum-exp over

        Raises:
            ValueError: If fewer than two operands are provided
        """
        if len(args) < 2:
            raise ValueError("LogSumExp requires two or more operands")
        self.operands = [to_expr(a) for a in args]

    def children(self):
        return list(self.operands)

    def canonicalize(self) -> "Expr":
        """Canonicalize log-sum-exp: flatten nested LogSumExp, fold constants."""
        from .expr import Constant

        operands = []
        const_vals = []

        for op in self.operands:
            c = op.canonicalize()
            if isinstance(c, LogSumExp):
                operands.extend(c.operands)
            elif isinstance(c, Constant):
                const_vals.append(c.value)
            else:
                operands.append(c)

        # If we have constants, compute their log-sum-exp and keep it
        if const_vals:
            # For constants, we can compute logsumexp directly
            # logsumexp(c1, c2, ..., cn) = log(sum(exp(ci)))
            exp_vals = [np.exp(v) for v in const_vals]
            lse_const = np.log(np.sum(exp_vals))
            operands.append(Constant(lse_const))

        if not operands:
            raise ValueError("LogSumExp must have at least one operand after canonicalization")
        if len(operands) == 1:
            return operands[0]
        return LogSumExp(*operands)

    def check_shape(self) -> Tuple[int, ...]:
        """LogSumExp broadcasts shapes like NumPy, preserving element-wise shape."""
        shapes = [child.check_shape() for child in self.children()]
        try:
            return np.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"LogSumExp shapes not broadcastable: {shapes}") from e

    def __repr__(self) -> str:
        inner = ", ".join(repr(op) for op in self.operands)
        return f"logsumexp({inner})"


class Linterp(Expr):
    """1D linear interpolation for symbolic expressions.

    Computes the linear interpolant of data points (xp, fp) evaluated at x,
    equivalent to jax.numpy.interp(x, xp, fp). For values outside the data range,
    the boundary values are returned (no extrapolation).

    This is useful for incorporating tabulated data (e.g., atmospheric properties,
    engine thrust curves, aerodynamic coefficients) into trajectory optimization
    dynamics and constraints.

    Attributes:
        x: Query point(s) at which to evaluate the interpolant (symbolic expression)
        xp: 1D array of x-coordinates of data points (must be increasing)
        fp: 1D array of y-coordinates of data points (same length as xp)

    Example:
        Interpolate atmospheric density from altitude table::

            import openscvx as ox
            import numpy as np

            # US 1976 Standard Atmosphere data
            alt_data = np.array([0, 5000, 10000, 15000, 20000])  # meters
            rho_data = np.array([1.225, 0.736, 0.414, 0.195, 0.089])  # kg/m^3

            altitude = ox.State("altitude", shape=(1,))
            rho = ox.Linterp(altitude[0], alt_data, rho_data)

            # rho can now be used in dynamics expressions
            drag = 0.5 * rho * v**2 * Cd * S

    Note:
        - xp must be strictly increasing
        - For query points outside [xp[0], xp[-1]], boundary values are returned
    """

    def __init__(
        self,
        x: Union[Expr, float, int, np.ndarray],
        xp: Union[Expr, float, int, np.ndarray],
        fp: Union[Expr, float, int, np.ndarray],
    ):
        """Initialize a 1D linear interpolation node.

        Args:
            x: Query point(s) at which to evaluate the interpolant.
                Can be a scalar or array symbolic expression.
            xp: 1D array of x-coordinates of data points. Must be increasing.
                Can be a numpy array or Constant expression.
            fp: 1D array of y-coordinates of data points. Must have same length as xp.
                Can be a numpy array or Constant expression.
        """
        self.x = to_expr(x)
        self.xp = to_expr(xp)
        self.fp = to_expr(fp)

    def children(self):
        return [self.x, self.xp, self.fp]

    def canonicalize(self) -> "Expr":
        """Canonicalize by canonicalizing all operands."""
        x = self.x.canonicalize()
        xp = self.xp.canonicalize()
        fp = self.fp.canonicalize()
        return Linterp(x, xp, fp)

    def check_shape(self) -> Tuple[int, ...]:
        """Output shape matches the query point shape.

        The interpolation is element-wise over x, so the output has
        the same shape as the query points.

        Returns:
            tuple: Shape of the query point x

        Raises:
            ValueError: If xp and fp have different lengths or are not 1D
        """
        xp_shape = self.xp.check_shape()
        fp_shape = self.fp.check_shape()

        if len(xp_shape) != 1:
            raise ValueError(f"Linterp xp must be 1D, got shape {xp_shape}")
        if len(fp_shape) != 1:
            raise ValueError(f"Linterp fp must be 1D, got shape {fp_shape}")
        if xp_shape != fp_shape:
            raise ValueError(
                f"Linterp xp and fp must have same length, got {xp_shape} vs {fp_shape}"
            )

        return self.x.check_shape()

    def __repr__(self) -> str:
        return f"linterp({self.x!r}, {self.xp!r}, {self.fp!r})"


class Bilerp(Expr):
    """2D bilinear interpolation for symbolic expressions.

    Performs bilinear interpolation on a regular 2D grid. Given grid points
    (xp, yp) and corresponding values fp, computes the bilinearly interpolated
    value at query point (x, y). For values outside the grid, boundary values
    are returned (clamping, no extrapolation).

    This is useful for incorporating 2D tabulated data (e.g., engine thrust
    as a function of altitude and Mach number, aerodynamic coefficients as
    a function of angle of attack and sideslip) into trajectory optimization.

    Attributes:
        x: Query x-coordinate (symbolic expression)
        y: Query y-coordinate (symbolic expression)
        xp: 1D array of x grid coordinates (must be increasing), length N
        yp: 1D array of y grid coordinates (must be increasing), length M
        fp: 2D array of values with shape (N, M), where fp[i, j] is the
            value at grid point (xp[i], yp[j])

    Example:
        Interpolate engine thrust from altitude and Mach number::

            import openscvx as ox
            import numpy as np

            # Grid coordinates
            alt_grid = np.array([0, 5000, 10000, 15000, 20000])  # meters
            mach_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

            # Thrust values: thrust_table[i, j] = thrust at (alt_grid[i], mach_grid[j])
            thrust_table = np.array([...])  # shape (5, 5)

            altitude = ox.State("altitude", shape=(1,))
            mach = ox.State("mach", shape=(1,))

            thrust = ox.Bilerp(altitude[0], mach[0], alt_grid, mach_grid, thrust_table)

    Note:
        - xp and yp must be strictly increasing
        - fp must have shape (len(xp), len(yp))
        - For query points outside the grid, boundary values are returned
        - This node is only supported in JAX lowering (dynamics/cost), not CVXPy
    """

    def __init__(
        self,
        x: Union[Expr, float, int, np.ndarray],
        y: Union[Expr, float, int, np.ndarray],
        xp: Union[Expr, float, int, np.ndarray],
        yp: Union[Expr, float, int, np.ndarray],
        fp: Union[Expr, float, int, np.ndarray],
    ):
        """Initialize a 2D bilinear interpolation node.

        Args:
            x: Query x-coordinate. Can be a scalar symbolic expression.
            y: Query y-coordinate. Can be a scalar symbolic expression.
            xp: 1D array of x grid coordinates. Must be increasing.
            yp: 1D array of y grid coordinates. Must be increasing.
            fp: 2D array of values with shape (len(xp), len(yp)).
        """
        self.x = to_expr(x)
        self.y = to_expr(y)
        self.xp = to_expr(xp)
        self.yp = to_expr(yp)
        self.fp = to_expr(fp)

    def children(self):
        return [self.x, self.y, self.xp, self.yp, self.fp]

    def canonicalize(self) -> "Expr":
        """Canonicalize by canonicalizing all operands."""
        x = self.x.canonicalize()
        y = self.y.canonicalize()
        xp = self.xp.canonicalize()
        yp = self.yp.canonicalize()
        fp = self.fp.canonicalize()
        return Bilerp(x, y, xp, yp, fp)

    def check_shape(self) -> Tuple[int, ...]:
        """Output shape is scalar (single interpolated value).

        Returns:
            tuple: Empty tuple (scalar output)

        Raises:
            ValueError: If grid arrays have invalid shapes
        """
        xp_shape = self.xp.check_shape()
        yp_shape = self.yp.check_shape()
        fp_shape = self.fp.check_shape()
        x_shape = self.x.check_shape()
        y_shape = self.y.check_shape()

        if len(xp_shape) != 1:
            raise ValueError(f"Bilerp xp must be 1D, got shape {xp_shape}")
        if len(yp_shape) != 1:
            raise ValueError(f"Bilerp yp must be 1D, got shape {yp_shape}")
        if len(fp_shape) != 2:
            raise ValueError(f"Bilerp fp must be 2D, got shape {fp_shape}")
        if fp_shape != (xp_shape[0], yp_shape[0]):
            raise ValueError(
                f"Bilerp fp shape {fp_shape} must match (len(xp), len(yp)) = "
                f"({xp_shape[0]}, {yp_shape[0]})"
            )
        if x_shape != ():
            raise ValueError(f"Bilerp x must be scalar, got shape {x_shape}")
        if y_shape != ():
            raise ValueError(f"Bilerp y must be scalar, got shape {y_shape}")

        return ()

    def __repr__(self) -> str:
        return f"bilerp({self.x!r}, {self.y!r}, {self.xp!r}, {self.yp!r}, {self.fp!r})"
