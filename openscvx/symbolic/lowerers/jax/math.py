"""JAX visitors for math expressions.

Visitors: Sin, Cos, Tan, Square, Sqrt, Exp, Log, Abs, Max,
          PositivePart, Huber, SmoothReLU, LogSumExp, Linterp, Bilerp
"""

from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401

# Expression types to handle — uncomment as you paste visitors:
# from openscvx.symbolic.expr.math import (
#     Sin, Cos, Tan, Square, Sqrt, Exp, Log, Abs, Max,
#     PositivePart, Huber, SmoothReLU, LogSumExp, Linterp, Bilerp,
# )
