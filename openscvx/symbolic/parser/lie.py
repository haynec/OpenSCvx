"""Parser handlers for Lie algebra operations.

Handlers: AdjointDual, Adjoint, SE3Adjoint, SE3AdjointDual,
          SO3Exp, SO3Log, SE3Exp, SE3Log

Each handler is registered under its function name via ``@function`` and turns the
call-syntax form (e.g. ``SE3Exp(twist)``) that the Pratt parser encounters in an
expression string into the corresponding SO(3)/SE(3) ``Expr`` node — the group
exp/log maps and the (co)adjoint operators used in rigid-body dynamics.
"""

from openscvx.symbolic.expr.lie import (
    Adjoint,
    AdjointDual,
    SE3Adjoint,
    SE3AdjointDual,
    SE3Exp,
    SE3Log,
    SO3Exp,
    SO3Log,
)
from openscvx.symbolic.parser._registry import function


@function("AdjointDual")
def _parse_adjoint_dual(args, kwargs):
    if len(args) != 2:
        raise ValueError("AdjointDual() takes exactly 2 arguments (twist, momentum)")
    return AdjointDual(args[0], args[1])


@function("Adjoint")
def _parse_adjoint(args, kwargs):
    if len(args) != 2:
        raise ValueError("Adjoint() takes exactly 2 arguments (twist, vector)")
    return Adjoint(args[0], args[1])


@function("SE3Adjoint")
def _parse_se3_adjoint(args, kwargs):
    if len(args) != 1:
        raise ValueError("SE3Adjoint() takes exactly 1 argument (transform)")
    return SE3Adjoint(args[0])


@function("SE3AdjointDual")
def _parse_se3_adjoint_dual(args, kwargs):
    if len(args) != 1:
        raise ValueError("SE3AdjointDual() takes exactly 1 argument (transform)")
    return SE3AdjointDual(args[0])


@function("SO3Exp")
def _parse_so3_exp(args, kwargs):
    if len(args) != 1:
        raise ValueError("SO3Exp() takes exactly 1 argument (rotation vector)")
    return SO3Exp(args[0])


@function("SO3Log")
def _parse_so3_log(args, kwargs):
    if len(args) != 1:
        raise ValueError("SO3Log() takes exactly 1 argument (rotation matrix)")
    return SO3Log(args[0])


@function("SE3Exp")
def _parse_se3_exp(args, kwargs):
    if len(args) != 1:
        raise ValueError("SE3Exp() takes exactly 1 argument (twist vector)")
    return SE3Exp(args[0])


@function("SE3Log")
def _parse_se3_log(args, kwargs):
    if len(args) != 1:
        raise ValueError("SE3Log() takes exactly 1 argument (transformation matrix)")
    return SE3Log(args[0])
