"""Tests for parser spatial operation handlers.

This module tests parsing of spatial / 6-DOF operations:
QDCM, SSM, SSMP
"""

import pytest

from openscvx.symbolic.expr import State
from openscvx.symbolic.expr.spatial import QDCM, SSM, SSMP
from openscvx.symbolic.parser import ExprParser

# =============================================================================
# Helper
# =============================================================================


def _parser(**extra):
    q = State("q", shape=(4,))
    omega = State("omega", shape=(3,))
    symbols = {"q": q, "omega": omega}
    symbols.update(extra)
    return ExprParser(symbols)


# =============================================================================
# QDCM
# =============================================================================


def test_parse_qdcm():
    expr = _parser().parse("QDCM(q)")
    assert isinstance(expr, QDCM)


def test_qdcm_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("QDCM(q, omega)")


# =============================================================================
# SSM
# =============================================================================


def test_parse_ssm():
    expr = _parser().parse("SSM(omega)")
    assert isinstance(expr, SSM)


def test_ssm_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("SSM()")


# =============================================================================
# SSMP
# =============================================================================


def test_parse_ssmp():
    expr = _parser().parse("SSMP(omega)")
    assert isinstance(expr, SSMP)


def test_ssmp_wrong_args():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("SSMP(q, omega)")


# =============================================================================
# Composed — QDCM @ thrust pattern
# =============================================================================


def test_parse_qdcm_matmul():
    from openscvx.symbolic.expr import MatMul

    thrust = State("thrust", shape=(3,))
    p = _parser(thrust=thrust)
    expr = p.parse("QDCM(q) @ thrust")
    assert isinstance(expr, MatMul)
    assert isinstance(expr.left, QDCM)
