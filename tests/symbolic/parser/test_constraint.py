"""Tests for parser constraint handlers.

This module tests parsing of constraint operations:
CrossNodeConstraint, NodalConstraint, ctcs
"""

import pytest

from openscvx.symbolic.expr import State
from openscvx.symbolic.expr.constraint import CTCS, CrossNodeConstraint, NodalConstraint
from openscvx.symbolic.parser import ExprParser

# =============================================================================
# Helper
# =============================================================================


def _parser(**extra):
    pos = State("pos", shape=(3,))
    vel = State("vel", shape=(3,))
    symbols = {"pos": pos, "vel": vel}
    symbols.update(extra)
    return ExprParser(symbols)


# =============================================================================
# CrossNodeConstraint
# =============================================================================


def test_parse_cross_node_constraint():
    expr = _parser().parse("CrossNodeConstraint(pos.at(5) - pos.at(4) <= 0.1)")
    assert isinstance(expr, CrossNodeConstraint)


def test_cross_node_constraint_preserves_inner():
    expr = _parser().parse("CrossNodeConstraint(pos.at(1)[0] - pos.at(0)[0] <= 1.0)")
    assert isinstance(expr, CrossNodeConstraint)
    assert hasattr(expr, "constraint")


def test_cross_node_constraint_wrong_arg_count():
    with pytest.raises(ValueError, match="exactly 1"):
        _parser().parse("CrossNodeConstraint(pos[0] <= 1.0, pos[1] <= 2.0)")


def test_cross_node_constraint_requires_constraint():
    with pytest.raises(ValueError, match="must be a Constraint"):
        _parser().parse("CrossNodeConstraint(pos)")


# =============================================================================
# NodalConstraint
# =============================================================================


def test_parse_nodal_constraint():
    expr = _parser().parse("NodalConstraint(pos[0] <= 5.0, 0, 10, 20)")
    assert isinstance(expr, NodalConstraint)
    assert expr.nodes == [0, 10, 20]


def test_nodal_constraint_single_node():
    expr = _parser().parse("NodalConstraint(vel[2] >= 0.0, 0)")
    assert isinstance(expr, NodalConstraint)
    assert expr.nodes == [0]


def test_nodal_constraint_equality():
    expr = _parser().parse("NodalConstraint(pos[0] == 0.0, 0)")
    assert isinstance(expr, NodalConstraint)
    assert expr.nodes == [0]


def test_nodal_constraint_wrong_arg_count():
    with pytest.raises(ValueError, match="at least 2"):
        _parser().parse("NodalConstraint(pos[0] <= 1.0)")


def test_nodal_constraint_requires_constraint():
    with pytest.raises(ValueError, match="must be a Constraint"):
        _parser().parse("NodalConstraint(pos, 0)")


# =============================================================================
# ctcs — defaults
# =============================================================================


def test_parse_ctcs_default():
    expr = _parser().parse("ctcs(pos[2] >= 0.0)")
    assert isinstance(expr, CTCS)
    assert expr.penalty == "squared_relu"
    assert expr.nodes is None
    assert expr.idx is None
    assert expr.check_nodally is False


# =============================================================================
# ctcs — penalty kwarg
# =============================================================================


def test_ctcs_huber_penalty():
    expr = _parser().parse('ctcs(pos[2] >= 0.0, penalty="huber")')
    assert isinstance(expr, CTCS)
    assert expr.penalty == "huber"


def test_ctcs_smooth_relu_penalty():
    expr = _parser().parse('ctcs(vel[0] <= 10.0, penalty="smooth_relu")')
    assert isinstance(expr, CTCS)
    assert expr.penalty == "smooth_relu"


# =============================================================================
# ctcs — check_nodally kwarg
# =============================================================================


def test_ctcs_check_nodally():
    expr = _parser().parse("ctcs(pos[2] >= 0.0, check_nodally=True)")
    assert isinstance(expr, CTCS)
    assert expr.check_nodally is True


def test_ctcs_check_nodally_false():
    expr = _parser().parse("ctcs(pos[2] >= 0.0, check_nodally=False)")
    assert isinstance(expr, CTCS)
    assert expr.check_nodally is False


# =============================================================================
# ctcs — idx kwarg
# =============================================================================


def test_ctcs_idx():
    expr = _parser().parse("ctcs(pos[2] >= 0.0, idx=3)")
    assert isinstance(expr, CTCS)
    assert expr.idx == 3


# =============================================================================
# ctcs — combined kwargs
# =============================================================================


def test_ctcs_all_kwargs():
    expr = _parser().parse('ctcs(vel[0] <= 5.0, penalty="huber", idx=1, check_nodally=True)')
    assert isinstance(expr, CTCS)
    assert expr.penalty == "huber"
    assert expr.idx == 1
    assert expr.check_nodally is True


# =============================================================================
# ctcs — error cases
# =============================================================================


def test_ctcs_no_args():
    with pytest.raises(ValueError, match="at least 1"):
        _parser().parse("ctcs()")


def test_ctcs_requires_constraint():
    with pytest.raises(ValueError, match="must be a Constraint"):
        _parser().parse("ctcs(pos)")
