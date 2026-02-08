"""Tests for the symbolic expression tokenizer.

This module tests the tokenizer that converts expression strings into
token streams for consumption by the Pratt parser, including:
- Number literals (integer, float, scientific notation)
- Identifiers
- String literals
- Arithmetic and comparison operators
- Delimiters and punctuation
- Whitespace handling
- Error cases
"""

import pytest

from openscvx.symbolic.parser.tokenizer import TokenizeError, TokenType, tokenize

# =============================================================================
# Helper
# =============================================================================


def _types(tokens):
    """Extract just the TokenType sequence (excluding trailing EOF)."""
    return [t.type for t in tokens if t.type != TokenType.EOF]


def _values(tokens):
    """Extract just the value sequence (excluding trailing EOF)."""
    return [t.value for t in tokens if t.type != TokenType.EOF]


# =============================================================================
# EOF Sentinel
# =============================================================================


def test_empty_string_produces_only_eof():
    tokens = tokenize("")
    assert len(tokens) == 1
    assert tokens[0].type == TokenType.EOF


def test_whitespace_only_produces_only_eof():
    tokens = tokenize("   \t\n\r  ")
    assert len(tokens) == 1
    assert tokens[0].type == TokenType.EOF


def test_last_token_is_always_eof():
    tokens = tokenize("x + 1")
    assert tokens[-1].type == TokenType.EOF


# =============================================================================
# Number Literals
# =============================================================================


def test_integer_literal():
    tokens = tokenize("42")
    assert _types(tokens) == [TokenType.NUMBER]
    assert _values(tokens) == ["42"]


def test_float_literal():
    tokens = tokenize("3.14")
    assert _types(tokens) == [TokenType.NUMBER]
    assert _values(tokens) == ["3.14"]


def test_float_leading_dot():
    tokens = tokenize(".5")
    assert _types(tokens) == [TokenType.NUMBER]
    assert _values(tokens) == [".5"]


def test_float_trailing_dot():
    tokens = tokenize("5.")
    assert _types(tokens) == [TokenType.NUMBER]
    assert _values(tokens) == ["5."]


def test_scientific_notation():
    tokens = tokenize("1e10")
    assert _types(tokens) == [TokenType.NUMBER]
    assert _values(tokens) == ["1e10"]


def test_scientific_notation_with_sign():
    for src in ("1e+10", "1e-10", "2.5E-3", "3E+2"):
        tokens = tokenize(src)
        assert _types(tokens) == [TokenType.NUMBER], f"failed for {src!r}"
        assert _values(tokens) == [src]


def test_scientific_notation_float():
    tokens = tokenize("6.022e23")
    assert _types(tokens) == [TokenType.NUMBER]
    assert _values(tokens) == ["6.022e23"]


def test_zero():
    tokens = tokenize("0")
    assert _types(tokens) == [TokenType.NUMBER]
    assert _values(tokens) == ["0"]


def test_multiple_numbers():
    tokens = tokenize("1, 2.0, 3e5")
    assert _types(tokens) == [
        TokenType.NUMBER,
        TokenType.COMMA,
        TokenType.NUMBER,
        TokenType.COMMA,
        TokenType.NUMBER,
    ]
    assert _values(tokens) == ["1", ",", "2.0", ",", "3e5"]


# =============================================================================
# Identifiers
# =============================================================================


def test_simple_identifier():
    tokens = tokenize("pos")
    assert _types(tokens) == [TokenType.IDENT]
    assert _values(tokens) == ["pos"]


def test_identifier_with_underscores():
    tokens = tokenize("obs_center_2")
    assert _types(tokens) == [TokenType.IDENT]
    assert _values(tokens) == ["obs_center_2"]


def test_identifier_leading_underscore():
    tokens = tokenize("_private")
    assert _types(tokens) == [TokenType.IDENT]
    assert _values(tokens) == ["_private"]


def test_identifier_all_caps():
    tokens = tokenize("QDCM")
    assert _types(tokens) == [TokenType.IDENT]
    assert _values(tokens) == ["QDCM"]


def test_multiple_identifiers():
    tokens = tokenize("pos vel thrust")
    assert _types(tokens) == [TokenType.IDENT] * 3
    assert _values(tokens) == ["pos", "vel", "thrust"]


# =============================================================================
# String Literals
# =============================================================================


def test_double_quoted_string():
    tokens = tokenize('"hello"')
    assert _types(tokens) == [TokenType.STRING]
    assert _values(tokens) == ["hello"]


def test_single_quoted_string():
    tokens = tokenize("'world'")
    assert _types(tokens) == [TokenType.STRING]
    assert _values(tokens) == ["world"]


def test_string_with_escaped_quote():
    tokens = tokenize(r'"say \"hi\""')
    assert _types(tokens) == [TokenType.STRING]
    assert _values(tokens) == [r"say \"hi\""]


def test_unterminated_string_raises():
    with pytest.raises(TokenizeError, match="Unterminated string"):
        tokenize('"oops')


def test_empty_string_literal():
    tokens = tokenize('""')
    assert _types(tokens) == [TokenType.STRING]
    assert _values(tokens) == [""]


# =============================================================================
# Arithmetic Operators
# =============================================================================


def test_plus():
    tokens = tokenize("a + b")
    assert _types(tokens) == [TokenType.IDENT, TokenType.PLUS, TokenType.IDENT]


def test_minus():
    tokens = tokenize("a - b")
    assert _types(tokens) == [TokenType.IDENT, TokenType.MINUS, TokenType.IDENT]


def test_star():
    tokens = tokenize("a * b")
    assert _types(tokens) == [TokenType.IDENT, TokenType.STAR, TokenType.IDENT]


def test_slash():
    tokens = tokenize("a / b")
    assert _types(tokens) == [TokenType.IDENT, TokenType.SLASH, TokenType.IDENT]


def test_doublestar():
    tokens = tokenize("a ** b")
    assert _types(tokens) == [TokenType.IDENT, TokenType.DOUBLESTAR, TokenType.IDENT]


def test_at_operator():
    tokens = tokenize("A @ x")
    assert _types(tokens) == [TokenType.IDENT, TokenType.AT, TokenType.IDENT]


# =============================================================================
# Comparison Operators
# =============================================================================


def test_less_equal():
    tokens = tokenize("x <= 5")
    assert _types(tokens) == [TokenType.IDENT, TokenType.LE, TokenType.NUMBER]


def test_greater_equal():
    tokens = tokenize("x >= 0")
    assert _types(tokens) == [TokenType.IDENT, TokenType.GE, TokenType.NUMBER]


def test_double_equal():
    tokens = tokenize("x == y")
    assert _types(tokens) == [TokenType.IDENT, TokenType.EQEQ, TokenType.IDENT]


def test_single_equal():
    tokens = tokenize("key = val")
    assert _types(tokens) == [TokenType.IDENT, TokenType.EQ, TokenType.IDENT]


def test_equal_vs_double_equal():
    tokens = tokenize("a = b == c")
    assert _types(tokens) == [
        TokenType.IDENT,
        TokenType.EQ,
        TokenType.IDENT,
        TokenType.EQEQ,
        TokenType.IDENT,
    ]


# =============================================================================
# Delimiters
# =============================================================================


def test_parens():
    tokens = tokenize("(a)")
    assert _types(tokens) == [TokenType.LPAREN, TokenType.IDENT, TokenType.RPAREN]


def test_brackets():
    tokens = tokenize("[0]")
    assert _types(tokens) == [TokenType.LBRACKET, TokenType.NUMBER, TokenType.RBRACKET]


def test_comma():
    tokens = tokenize("a, b")
    assert _types(tokens) == [TokenType.IDENT, TokenType.COMMA, TokenType.IDENT]


def test_colon():
    tokens = tokenize("0:3")
    assert _types(tokens) == [TokenType.NUMBER, TokenType.COLON, TokenType.NUMBER]


def test_dot():
    tokens = tokenize("x.T")
    assert _types(tokens) == [TokenType.IDENT, TokenType.DOT, TokenType.IDENT]


# =============================================================================
# Token Positions
# =============================================================================


def test_positions_track_source_offset():
    tokens = tokenize("ab + cd")
    # ab at 0, + at 3, cd at 5
    assert tokens[0].pos == 0
    assert tokens[1].pos == 3
    assert tokens[2].pos == 5


def test_eof_position_is_source_length():
    src = "hello"
    tokens = tokenize(src)
    assert tokens[-1].pos == len(src)


# =============================================================================
# Whitespace Handling
# =============================================================================


def test_no_whitespace():
    tokens = tokenize("a+b*c")
    assert _types(tokens) == [
        TokenType.IDENT,
        TokenType.PLUS,
        TokenType.IDENT,
        TokenType.STAR,
        TokenType.IDENT,
    ]


def test_extra_whitespace():
    tokens = tokenize("  a   +   b  ")
    assert _types(tokens) == [TokenType.IDENT, TokenType.PLUS, TokenType.IDENT]
    assert _values(tokens) == ["a", "+", "b"]


def test_tabs_and_newlines():
    tokens = tokenize("a\t+\nb")
    assert _types(tokens) == [TokenType.IDENT, TokenType.PLUS, TokenType.IDENT]


# =============================================================================
# Realistic Expressions
# =============================================================================


def test_function_call():
    tokens = tokenize("Sin(x)")
    assert _types(tokens) == [
        TokenType.IDENT,
        TokenType.LPAREN,
        TokenType.IDENT,
        TokenType.RPAREN,
    ]
    assert _values(tokens) == ["Sin", "(", "x", ")"]


def test_indexing_with_slice():
    tokens = tokenize("pos[0:3]")
    assert _types(tokens) == [
        TokenType.IDENT,
        TokenType.LBRACKET,
        TokenType.NUMBER,
        TokenType.COLON,
        TokenType.NUMBER,
        TokenType.RBRACKET,
    ]


def test_constraint_expression():
    tokens = tokenize("Norm(pos[:2] - obs) >= 2.0")
    expected_types = [
        TokenType.IDENT,  # Norm
        TokenType.LPAREN,  # (
        TokenType.IDENT,  # pos
        TokenType.LBRACKET,  # [
        TokenType.COLON,  # :
        TokenType.NUMBER,  # 2
        TokenType.RBRACKET,  # ]
        TokenType.MINUS,  # -
        TokenType.IDENT,  # obs
        TokenType.RPAREN,  # )
        TokenType.GE,  # >=
        TokenType.NUMBER,  # 2.0
    ]
    assert _types(tokens) == expected_types


def test_dot_method_chain():
    tokens = tokenize("(x <= 5.0).at(0, 10)")
    expected_types = [
        TokenType.LPAREN,  # (
        TokenType.IDENT,  # x
        TokenType.LE,  # <=
        TokenType.NUMBER,  # 5.0
        TokenType.RPAREN,  # )
        TokenType.DOT,  # .
        TokenType.IDENT,  # at
        TokenType.LPAREN,  # (
        TokenType.NUMBER,  # 0
        TokenType.COMMA,  # ,
        TokenType.NUMBER,  # 10
        TokenType.RPAREN,  # )
    ]
    assert _types(tokens) == expected_types


def test_keyword_argument():
    tokens = tokenize("Norm(x, ord=2)")
    expected_types = [
        TokenType.IDENT,  # Norm
        TokenType.LPAREN,  # (
        TokenType.IDENT,  # x
        TokenType.COMMA,  # ,
        TokenType.IDENT,  # ord
        TokenType.EQ,  # =
        TokenType.NUMBER,  # 2
        TokenType.RPAREN,  # )
    ]
    assert _types(tokens) == expected_types


def test_array_literal():
    tokens = tokenize("[1, 2.0, -3]")
    expected_types = [
        TokenType.LBRACKET,  # [
        TokenType.NUMBER,  # 1
        TokenType.COMMA,  # ,
        TokenType.NUMBER,  # 2.0
        TokenType.COMMA,  # ,
        TokenType.MINUS,  # -
        TokenType.NUMBER,  # 3
        TokenType.RBRACKET,  # ]
    ]
    assert _types(tokens) == expected_types


def test_power_expression():
    tokens = tokenize("x ** 2 + y ** 2")
    expected_types = [
        TokenType.IDENT,  # x
        TokenType.DOUBLESTAR,  # **
        TokenType.NUMBER,  # 2
        TokenType.PLUS,  # +
        TokenType.IDENT,  # y
        TokenType.DOUBLESTAR,  # **
        TokenType.NUMBER,  # 2
    ]
    assert _types(tokens) == expected_types


def test_matmul_expression():
    tokens = tokenize("QDCM(q) @ thrust")
    expected_types = [
        TokenType.IDENT,  # QDCM
        TokenType.LPAREN,  # (
        TokenType.IDENT,  # q
        TokenType.RPAREN,  # )
        TokenType.AT,  # @
        TokenType.IDENT,  # thrust
    ]
    assert _types(tokens) == expected_types


def test_nested_function_calls():
    tokens = tokenize("Norm(Sin(x) + Cos(y))")
    expected_types = [
        TokenType.IDENT,  # Norm
        TokenType.LPAREN,  # (
        TokenType.IDENT,  # Sin
        TokenType.LPAREN,  # (
        TokenType.IDENT,  # x
        TokenType.RPAREN,  # )
        TokenType.PLUS,  # +
        TokenType.IDENT,  # Cos
        TokenType.LPAREN,  # (
        TokenType.IDENT,  # y
        TokenType.RPAREN,  # )
        TokenType.RPAREN,  # )
    ]
    assert _types(tokens) == expected_types


# =============================================================================
# Error Cases
# =============================================================================


def test_unexpected_character_raises():
    with pytest.raises(TokenizeError, match="Unexpected character"):
        tokenize("x & y")


def test_unexpected_character_reports_position():
    with pytest.raises(TokenizeError, match="position 2"):
        tokenize("x & y")


def test_hash_raises():
    with pytest.raises(TokenizeError):
        tokenize("# comment")


def test_backtick_raises():
    with pytest.raises(TokenizeError):
        tokenize("`x`")
