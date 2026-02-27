"""Tests for parser array operation handlers.

This module tests parsing of array manipulation operations:
Concat, Stack, Hstack, Vstack, Block
"""

import pytest

from openscvx.symbolic.expr import State
from openscvx.symbolic.expr.array import Block, Concat, Hstack, Stack, Vstack
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.parser import ExprParser

# =============================================================================
# Helper
# =============================================================================


def _parser(**extra):
    x = State("x", shape=(3,))
    y = State("y", shape=(3,))
    u = Control("u", shape=(2,))
    symbols = {"x": x, "y": y, "u": u}
    symbols.update(extra)
    return ExprParser(symbols)


# =============================================================================
# Concat
# =============================================================================


def test_parse_concat():
    expr = _parser().parse("Concat(x, y)")
    assert isinstance(expr, Concat)


def test_concat_wrong_args():
    with pytest.raises(ValueError, match="at least 1"):
        _parser().parse("Concat()")


# =============================================================================
# Stack
# =============================================================================


def test_parse_stack():
    expr = _parser().parse("Stack(x, y)")
    assert isinstance(expr, Stack)


def test_stack_wrong_args():
    with pytest.raises(ValueError, match="at least 1"):
        _parser().parse("Stack()")


# =============================================================================
# Hstack
# =============================================================================


def test_parse_hstack():
    expr = _parser().parse("Hstack(x, y)")
    assert isinstance(expr, Hstack)


def test_hstack_wrong_args():
    with pytest.raises(ValueError, match="at least 1"):
        _parser().parse("Hstack()")


# =============================================================================
# Vstack
# =============================================================================


def test_parse_vstack():
    expr = _parser().parse("Vstack(x, y)")
    assert isinstance(expr, Vstack)


def test_vstack_wrong_args():
    with pytest.raises(ValueError, match="at least 1"):
        _parser().parse("Vstack()")


# =============================================================================
# Block
# =============================================================================


def test_parse_block():
    expr = _parser().parse("Block(x, y)")
    assert isinstance(expr, Block)


def test_block_wrong_args_empty():
    # Block with a single list arg is valid; empty args produce an empty list
    expr = _parser().parse("Block(x)")
    assert isinstance(expr, Block)
