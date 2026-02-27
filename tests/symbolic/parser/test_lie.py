"""Tests for parser Lie algebra handlers.

This module tests parsing of Lie algebra operations:
AdjointDual, Adjoint, SE3Adjoint, SE3AdjointDual,
SO3Exp, SO3Log, SE3Exp, SE3Log

Note: These tests are skipped if jaxlie is not installed.
"""

import pytest

from openscvx.symbolic.expr import State

try:
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
    from openscvx.symbolic.parser import lie as _lie_module  # noqa: F401

    HAS_LIE = True
except ImportError:
    HAS_LIE = False

from openscvx.symbolic.parser import ExprParser

pytestmark = pytest.mark.skipif(not HAS_LIE, reason="jaxlie not installed")


# =============================================================================
# Helper
# =============================================================================


def _parser(**extra):
    twist = State("twist", shape=(6,))
    momentum = State("momentum", shape=(6,))
    vec = State("vec", shape=(6,))
    rotvec = State("rotvec", shape=(3,))
    rotmat = State("rotmat", shape=(3, 3))
    T = State("T", shape=(4, 4))
    symbols = {
        "twist": twist,
        "momentum": momentum,
        "vec": vec,
        "rotvec": rotvec,
        "rotmat": rotmat,
        "T": T,
    }
    symbols.update(extra)
    return ExprParser(symbols)


# =============================================================================
# AdjointDual & Adjoint
# =============================================================================


def test_parse_adjoint_dual():
    expr = _parser().parse("AdjointDual(twist, momentum)")
    assert isinstance(expr, AdjointDual)


def test_adjoint_dual_wrong_args():
    with pytest.raises(ValueError, match="exactly 2"):
        _parser().parse("AdjointDual(twist)")


def test_parse_adjoint():
    expr = _parser().parse("Adjoint(twist, vec)")
    assert isinstance(expr, Adjoint)


def test_adjoint_wrong_args():
    with pytest.raises(ValueError, match="exactly 2"):
        _parser().parse("Adjoint(twist)")


# =============================================================================
# SE3Adjoint & SE3AdjointDual
# =============================================================================


def test_parse_se3_adjoint():
    expr = _parser().parse("SE3Adjoint(T)")
    assert isinstance(expr, SE3Adjoint)


def test_se3_adjoint_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("SE3Adjoint(T, vec)")


def test_parse_se3_adjoint_dual():
    expr = _parser().parse("SE3AdjointDual(T)")
    assert isinstance(expr, SE3AdjointDual)


def test_se3_adjoint_dual_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("SE3AdjointDual(T, vec)")


# =============================================================================
# SO3 Exp & Log
# =============================================================================


def test_parse_so3_exp():
    expr = _parser().parse("SO3Exp(rotvec)")
    assert isinstance(expr, SO3Exp)


def test_so3_exp_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("SO3Exp(rotvec, rotvec)")


def test_parse_so3_log():
    expr = _parser().parse("SO3Log(rotmat)")
    assert isinstance(expr, SO3Log)


def test_so3_log_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("SO3Log()")


# =============================================================================
# SE3 Exp & Log
# =============================================================================


def test_parse_se3_exp():
    expr = _parser().parse("SE3Exp(twist)")
    assert isinstance(expr, SE3Exp)


def test_se3_exp_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("SE3Exp(twist, twist)")


def test_parse_se3_log():
    expr = _parser().parse("SE3Log(T)")
    assert isinstance(expr, SE3Log)


def test_se3_log_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("SE3Log()")
