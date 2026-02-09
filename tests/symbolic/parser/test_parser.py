"""Tests for the core Pratt parser.

This module tests the core parser infrastructure including:
- Arithmetic operators and precedence
- Unary minus
- Parenthesized expressions
- Symbol table lookup
- Array literals
- Built-in constants (pi, True, False)
- Indexing and slicing
- Error cases
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Add,
    Constant,
    Div,
    MatMul,
    Mul,
    Neg,
    Power,
    State,
    Sub,
)
from openscvx.symbolic.expr.array import Index
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.parser import ExprParser, ParseError

# =============================================================================
# Helper
# =============================================================================


def _parser(**extra):
    """Create a parser with default symbols (x, u) plus extras."""
    x = State("x", shape=(3,))
    u = Control("u", shape=(2,))
    symbols = {"x": x, "u": u}
    symbols.update(extra)
    return ExprParser(symbols)


# =============================================================================
# Arithmetic Operators
# =============================================================================


def test_parse_add():
    p = _parser()
    expr = p.parse("x + u")
    assert isinstance(expr, Add)


def test_parse_sub():
    p = _parser()
    expr = p.parse("x - [1, 2, 3]")
    assert isinstance(expr, Sub)


def test_parse_mul():
    p = _parser()
    expr = p.parse("x * 2.0")
    assert isinstance(expr, Mul)


def test_parse_div():
    p = _parser()
    expr = p.parse("x / 3.0")
    assert isinstance(expr, Div)


def test_parse_power():
    p = _parser()
    expr = p.parse("x ** 2")
    assert isinstance(expr, Power)


def test_parse_matmul():
    A = Constant(np.eye(3))
    p = _parser(A=A)
    expr = p.parse("A @ x")
    assert isinstance(expr, MatMul)


def test_parse_unary_minus():
    p = _parser()
    expr = p.parse("-x")
    assert isinstance(expr, Neg)


def test_parse_double_neg():
    p = _parser()
    expr = p.parse("--x")
    assert isinstance(expr, Neg)
    assert isinstance(expr.operand, Neg)


# =============================================================================
# Precedence
# =============================================================================


def test_mul_binds_tighter_than_add():
    a = Constant(np.array(1.0))
    b = Constant(np.array(2.0))
    c = Constant(np.array(3.0))
    p = _parser(a=a, b=b, c=c)
    # a + b * c  should parse as  Add(a, Mul(b, c))
    expr = p.parse("a + b * c")
    assert isinstance(expr, Add)
    assert isinstance(expr.terms[1], Mul)


def test_power_right_associative():
    a = Constant(np.array(2.0))
    b = Constant(np.array(3.0))
    c = Constant(np.array(4.0))
    p = _parser(a=a, b=b, c=c)
    # a ** b ** c  should parse as  Power(a, Power(b, c))
    expr = p.parse("a ** b ** c")
    assert isinstance(expr, Power)
    assert isinstance(expr.exponent, Power)


def test_parens_override_precedence():
    a = Constant(np.array(1.0))
    b = Constant(np.array(2.0))
    c = Constant(np.array(3.0))
    p = _parser(a=a, b=b, c=c)
    # (a + b) * c  should parse as  Mul(Add(a, b), c)
    expr = p.parse("(a + b) * c")
    assert isinstance(expr, Mul)
    children = expr.children()
    assert isinstance(children[0], Add)


def test_unary_minus_binds_tighter_than_mul():
    a = Constant(np.array(2.0))
    b = Constant(np.array(3.0))
    p = _parser(a=a, b=b)
    # -a * b  should parse as  Mul(Neg(a), b), NOT  Neg(Mul(a, b))
    expr = p.parse("-a * b")
    assert isinstance(expr, Mul)
    assert isinstance(expr.children()[0], Neg)


# =============================================================================
# Symbol Table
# =============================================================================


def test_symbol_lookup_returns_same_object():
    x = State("x", shape=(3,))
    p = ExprParser({"x": x})
    expr = p.parse("x")
    assert expr is x


def test_unknown_identifier_raises():
    p = _parser()
    with pytest.raises(ParseError, match="Unknown identifier"):
        p.parse("unknown_var")


def test_unknown_identifier_suggests_close_match():
    """Typo in a symbol name should produce a 'did you mean?' hint."""
    from openscvx.symbolic.expr import Parameter

    obs = Parameter("obs_center", shape=(3,), value=[0, 0, 0])
    p = _parser(obs_center=obs)
    with pytest.raises(ParseError, match="did you mean 'obs_center'"):
        p.parse("obs_centr")


def test_unknown_identifier_no_suggestion_when_distant():
    p = _parser()
    with pytest.raises(ParseError, match="Unknown identifier") as exc_info:
        p.parse("zzzzzzz")
    assert "did you mean" not in str(exc_info.value)


def test_unknown_function_raises():
    p = _parser()
    with pytest.raises(ParseError, match="Unknown function"):
        p.parse("NotAFunction(x)")


def test_unknown_function_suggests_close_match():
    """Typo in a function name should produce a 'did you mean?' hint."""
    p = _parser()
    with pytest.raises(ParseError, match="did you mean 'norm'"):
        p.parse("Nrom(x)")


def test_unknown_function_includes_position():
    p = _parser()
    with pytest.raises(ParseError, match="at position"):
        p.parse("Nrom(x)")


# =============================================================================
# Built-in Constants
# =============================================================================


def test_parse_pi():
    p = _parser()
    expr = p.parse("pi")
    assert isinstance(expr, Constant)
    assert np.isclose(float(expr.value), np.pi)


def test_parse_true():
    p = _parser()
    expr = p.parse("True")
    assert isinstance(expr, Constant)
    assert float(expr.value) == 1.0


def test_parse_false():
    p = _parser()
    expr = p.parse("False")
    assert isinstance(expr, Constant)
    assert float(expr.value) == 0.0


# =============================================================================
# Number Literals
# =============================================================================


def test_parse_integer():
    p = _parser()
    expr = p.parse("42")
    assert isinstance(expr, Constant)
    assert float(expr.value) == 42.0


def test_parse_float():
    p = _parser()
    expr = p.parse("3.14")
    assert isinstance(expr, Constant)
    assert np.isclose(float(expr.value), 3.14)


def test_parse_scientific_notation():
    p = _parser()
    expr = p.parse("1e-3")
    assert isinstance(expr, Constant)
    assert np.isclose(float(expr.value), 1e-3)


# =============================================================================
# Array Literals
# =============================================================================


def test_parse_constant_array():
    p = _parser()
    expr = p.parse("[1, 2, 3]")
    assert isinstance(expr, Constant)
    assert np.array_equal(expr.value, np.array([1.0, 2.0, 3.0]))


def test_parse_constant_array_with_negative():
    p = _parser()
    expr = p.parse("[1, -2, 3]")
    assert isinstance(expr, Constant)
    assert np.array_equal(expr.value, np.array([1.0, -2.0, 3.0]))


def test_parse_mixed_array_produces_concat():
    from openscvx.symbolic.expr.array import Concat

    p = _parser()
    expr = p.parse("[x, u]")
    assert isinstance(expr, Concat)


def test_parse_empty_array():
    p = _parser()
    expr = p.parse("[]")
    assert isinstance(expr, Constant)
    assert expr.value.shape == (0,)


# =============================================================================
# Indexing & Slicing
# =============================================================================


def test_parse_integer_index():
    p = _parser()
    expr = p.parse("x[0]")
    assert isinstance(expr, Index)


def test_parse_slice():
    p = _parser()
    expr = p.parse("x[0:2]")
    assert isinstance(expr, Index)
    assert expr.index == slice(0, 2)


def test_parse_slice_open_start():
    p = _parser()
    expr = p.parse("x[:2]")
    assert isinstance(expr, Index)
    assert expr.index == slice(None, 2)


def test_parse_slice_open_end():
    p = _parser()
    expr = p.parse("x[1:]")
    assert isinstance(expr, Index)
    assert expr.index == slice(1, None)


def test_parse_slice_with_step():
    p = _parser()
    expr = p.parse("x[::2]")
    assert isinstance(expr, Index)
    assert expr.index == slice(None, None, 2)


def test_parse_negative_index():
    p = _parser()
    expr = p.parse("x[-1]")
    assert isinstance(expr, Index)
    assert expr.index == -1


def test_parse_multidim_index():
    M = Constant(np.eye(3))
    p = _parser(M=M)
    expr = p.parse("M[0, 1]")
    assert isinstance(expr, Index)
    assert expr.index == (0, 1)


# =============================================================================
# Dot Access
# =============================================================================


def test_parse_dot_T():
    from openscvx.symbolic.expr.linalg import Transpose

    M = Constant(np.eye(3))
    p = _parser(M=M)
    expr = p.parse("M.T")
    assert isinstance(expr, Transpose)


def test_parse_unknown_dot_raises():
    p = _parser()
    with pytest.raises(ParseError, match="Unknown method"):
        p.parse("x.unknown_method")


# =============================================================================
# Keyword Arguments
# =============================================================================


def test_parse_kwargs():
    from openscvx.symbolic.expr.linalg import Norm

    p = _parser()
    expr = p.parse("Norm(x, ord=2)")
    assert isinstance(expr, Norm)
    assert expr.ord == 2


def test_parse_positional_after_keyword_raises():
    p = _parser()
    with pytest.raises(ParseError, match="Positional argument follows keyword"):
        p.parse("Norm(ord=2, x)")


# =============================================================================
# Complex Expressions
# =============================================================================


def test_parse_nested_expression():
    from openscvx.symbolic.expr.linalg import Norm
    from openscvx.symbolic.expr.math import Sin

    p = _parser()
    expr = p.parse("Norm(Sin(x) + [1, 2, 3])")
    assert isinstance(expr, Norm)
    inner = expr.operand
    assert isinstance(inner, Add)
    assert isinstance(inner.terms[0], Sin)


def test_parse_chained_arithmetic():
    p = _parser()
    # x + u should work since both are symbols (though shapes differ)
    expr = p.parse("x[0] + x[1] - x[2]")
    assert isinstance(expr, Sub)


def test_parser_reuse():
    """Parser can be reused for multiple expressions."""
    p = _parser()
    e1 = p.parse("x + [1, 2, 3]")
    e2 = p.parse("x * 2.0")
    assert isinstance(e1, Add)
    assert isinstance(e2, Mul)


# =============================================================================
# Error Cases
# =============================================================================


def test_trailing_tokens_raise():
    p = _parser()
    with pytest.raises(ParseError, match="Unexpected token"):
        p.parse("x y")


def test_unexpected_token_in_prefix():
    p = _parser()
    with pytest.raises(ParseError, match="Unexpected token"):
        p.parse(")")


def test_missing_closing_paren_raises():
    p = _parser()
    with pytest.raises(ParseError, match="Expected RPAREN"):
        p.parse("(x + [1, 2, 3]")


def test_missing_closing_bracket_raises():
    p = _parser()
    with pytest.raises(ParseError, match="Expected RBRACKET"):
        p.parse("x[0:2")


# =============================================================================
# Case-Insensitive Function Calls
# =============================================================================


def test_lowercase_function_call():
    from openscvx.symbolic.expr import Sin

    expr = _parser().parse("sin(x)")
    assert isinstance(expr, Sin)


def test_uppercase_function_call():
    from openscvx.symbolic.expr import Cos

    expr = _parser().parse("Cos(x)")
    assert isinstance(expr, Cos)


def test_mixed_case_function_call():
    from openscvx.symbolic.expr import Norm

    expr = _parser().parse("norm(x)")
    assert isinstance(expr, Norm)


def test_lowercase_nested():
    from openscvx.symbolic.expr import Norm, Sub

    expr = _parser().parse("norm(x - [1, 2, 3])")
    assert isinstance(expr, Norm)
    assert isinstance(expr.operand, Sub)


def test_lowercase_vmap():
    from openscvx.symbolic.expr import Parameter
    from openscvx.symbolic.expr.vmap import Vmap

    obs = Parameter("obs", shape=(3, 2), value=np.zeros((3, 2)))
    p = _parser(obs=obs)
    expr = p.parse("vmap(o: obs -> norm(x - o))")
    assert isinstance(expr, Vmap)
