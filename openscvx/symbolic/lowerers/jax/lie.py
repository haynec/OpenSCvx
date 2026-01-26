"""JAX visitors for Lie group expressions.

Visitors: AdjointDual, Adjoint, SE3Adjoint, SE3AdjointDual,
          SO3Exp, SO3Log, SE3Exp, SE3Log
"""

from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401

# Expression types to handle — uncomment as you paste visitors:
# from openscvx.symbolic.expr.lie.adjoint import (
#     AdjointDual,
#     Adjoint,
#     SE3Adjoint,
#     SE3AdjointDual,
# )
# from openscvx.symbolic.expr.lie.so3 import SO3Exp, SO3Log
# from openscvx.symbolic.expr.lie.se3 import SE3Exp, SE3Log
