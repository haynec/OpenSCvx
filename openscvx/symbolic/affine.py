"""Static affine-in-state/control checker for symbolic expressions.

Provides :func:`is_affine_in_state_control` and :func:`is_constant`, which
are used by :mod:`openscvx.symbolic.canonicalize` to verify that user
``.convex()`` constraints can be lowered to a :class:`~openscvx.solvers.cones.ConeConstraint`
without requiring a non-linear solver.

Both functions walk the expression AST without executing any JAX code, so
they are safe to call before JIT compilation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openscvx.symbolic.expr.expr import Expr


def is_constant(expr: "Expr") -> bool:
    """Return ``True`` iff *expr* has no :class:`~openscvx.symbolic.expr.State`
    or :class:`~openscvx.symbolic.expr.Control` leaves.

    Parameters that are constant across a trajectory are treated as constant;
    :class:`~openscvx.symbolic.expr.Variable` objects are treated as
    non-constant (conservative assumption).
    """
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.expr import Constant
    from openscvx.symbolic.expr.parameter import Parameter
    from openscvx.symbolic.expr.state import State
    from openscvx.symbolic.expr.variable import Variable

    if isinstance(expr, Constant):
        return True
    if isinstance(expr, Parameter):
        return True
    if isinstance(expr, (State, Control, Variable)):
        return False
    # Compound nodes: constant iff every child is constant.
    return all(is_constant(c) for c in expr.children())


def is_affine_in_state_control(expr: "Expr") -> bool:
    """Return ``True`` iff *expr* is affine in
    :class:`~openscvx.symbolic.expr.State` and
    :class:`~openscvx.symbolic.expr.Control` variables.

    Recognises:

    * Leaf nodes: :class:`Constant`, :class:`Parameter` → constant (affine).
    * :class:`State`, :class:`Control` → linear.
    * :class:`Add`, :class:`Sub` → affine iff all operands are affine.
    * :class:`Neg` → affine iff operand is affine.
    * :class:`Mul` → affine iff at most one factor is non-constant
      (and the non-constant factor, if present, is itself affine).
    * :class:`MatMul` → affine iff one side is constant and the other is
      affine.
    * :class:`Div` → affine iff numerator is affine and denominator is
      constant.
    * :class:`Transpose`, :class:`Sum`, :class:`Diag` → affine iff operand
      is affine.
    * :class:`Index`, :class:`Concat`, :class:`Stack`, :class:`Hstack`,
      :class:`Vstack`, :class:`Block` → affine iff all children are affine.
    * Anything else (norms, trig, powers, …) → **not affine**.
    """
    from openscvx.symbolic.expr.arithmetic import Div, MatMul, Mul, Neg, Sub
    from openscvx.symbolic.expr.array import Block, Concat, Hstack, Index, Stack, Vstack
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.expr import Constant
    from openscvx.symbolic.expr.linalg import Diag, Sum, Transpose
    from openscvx.symbolic.expr.parameter import Parameter
    from openscvx.symbolic.expr.state import State
    from openscvx.symbolic.expr.variable import Variable

    # -----------------------------------------------------------------
    # Leaf cases
    # -----------------------------------------------------------------
    if isinstance(expr, (Constant, Parameter)):
        return True
    if isinstance(expr, (State, Control)):
        return True
    if isinstance(expr, Variable):
        return False

    # -----------------------------------------------------------------
    # Add: n-ary via .terms
    # -----------------------------------------------------------------
    # Import locally to avoid circular imports at module level.
    try:
        from openscvx.symbolic.expr.arithmetic import Add

        if isinstance(expr, Add):
            return all(is_affine_in_state_control(t) for t in expr.terms)
    except ImportError:
        pass

    # -----------------------------------------------------------------
    # Sub: .left, .right
    # -----------------------------------------------------------------
    if isinstance(expr, Sub):
        return is_affine_in_state_control(expr.left) and is_affine_in_state_control(
            expr.right
        )

    # -----------------------------------------------------------------
    # Neg: .operand
    # -----------------------------------------------------------------
    if isinstance(expr, Neg):
        return is_affine_in_state_control(expr.operand)

    # -----------------------------------------------------------------
    # Mul (element-wise, n-ary via .factors):
    # affine iff at most one factor is non-constant AND that factor is
    # itself affine.
    # -----------------------------------------------------------------
    if isinstance(expr, Mul):
        non_const = [f for f in expr.factors if not is_constant(f)]
        if len(non_const) == 0:
            return True
        if len(non_const) == 1:
            return is_affine_in_state_control(non_const[0])
        return False

    # -----------------------------------------------------------------
    # MatMul: .left, .right
    # -----------------------------------------------------------------
    if isinstance(expr, MatMul):
        left_const = is_constant(expr.left)
        right_const = is_constant(expr.right)
        if left_const:
            return is_affine_in_state_control(expr.right)
        if right_const:
            return is_affine_in_state_control(expr.left)
        return False

    # -----------------------------------------------------------------
    # Div: .left (numerator), .right (denominator)
    # -----------------------------------------------------------------
    if isinstance(expr, Div):
        return is_affine_in_state_control(expr.left) and is_constant(expr.right)

    # -----------------------------------------------------------------
    # Linear-algebra ops that preserve affinity
    # -----------------------------------------------------------------
    if isinstance(expr, (Transpose, Sum, Diag)):
        return is_affine_in_state_control(expr.operand)

    # -----------------------------------------------------------------
    # Array ops: affine iff all children are affine
    # -----------------------------------------------------------------
    if isinstance(expr, Index):
        return is_affine_in_state_control(expr.base)

    if isinstance(expr, (Concat, Stack, Hstack, Vstack)):
        return all(is_affine_in_state_control(c) for c in expr.children())

    if isinstance(expr, Block):
        return all(
            is_affine_in_state_control(cell)
            for row in expr.blocks
            for cell in row
        )

    # -----------------------------------------------------------------
    # Everything else (Norm, Power, Sin, Cos, …) → not affine.
    # -----------------------------------------------------------------
    return False
