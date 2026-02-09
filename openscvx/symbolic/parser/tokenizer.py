"""Tokenizer for symbolic expression strings.

Converts expression strings like ``"Sin(pos[0:2] - obs) >= 2.0"`` into a
flat list of typed tokens for consumption by the Pratt parser.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class TokenType(Enum):
    """Token types produced by the tokenizer."""

    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENT = auto()

    # Arithmetic operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    DOUBLESTAR = auto()
    AT = auto()

    # Comparison operators
    LE = auto()  # <=
    GE = auto()  # >=
    EQEQ = auto()  # ==

    # Assignment (for keyword arguments)
    EQ = auto()  # =

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    COLON = auto()
    DOT = auto()
    ARROW = auto()  # ->

    # Sentinel
    EOF = auto()


@dataclass
class Token:
    """A single token from the expression source."""

    type: TokenType
    value: str
    pos: int


class TokenizeError(Exception):
    """Raised when the tokenizer encounters an invalid character sequence."""


_SINGLE: dict[str, TokenType] = {
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "@": TokenType.AT,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    ",": TokenType.COMMA,
    ":": TokenType.COLON,
    ".": TokenType.DOT,
    "=": TokenType.EQ,
}


def tokenize(source: str) -> List[Token]:
    """Tokenize an expression string into a list of tokens.

    Args:
        source: The expression string to tokenize.

    Returns:
        List of Token objects, always terminated by an EOF token.

    Raises:
        TokenizeError: On unexpected characters or unterminated strings.
    """
    tokens: List[Token] = []
    i = 0
    n = len(source)

    while i < n:
        c = source[i]

        # Skip whitespace
        if c in " \t\n\r":
            i += 1
            continue

        # Number literal (integer, float, scientific notation)
        if c.isdigit() or (c == "." and i + 1 < n and source[i + 1].isdigit()):
            start = i
            while i < n and source[i].isdigit():
                i += 1
            if i < n and source[i] == ".":
                i += 1
                while i < n and source[i].isdigit():
                    i += 1
            if i < n and source[i] in "eE":
                i += 1
                if i < n and source[i] in "+-":
                    i += 1
                while i < n and source[i].isdigit():
                    i += 1
            tokens.append(Token(TokenType.NUMBER, source[start:i], start))
            continue

        # Identifier
        if c.isalpha() or c == "_":
            start = i
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
            tokens.append(Token(TokenType.IDENT, source[start:i], start))
            continue

        # String literal (single or double quotes)
        if c in "\"'":
            quote = c
            start = i
            i += 1
            while i < n and source[i] != quote:
                if source[i] == "\\":
                    i += 1  # skip escaped character
                i += 1
            if i >= n:
                raise TokenizeError(f"Unterminated string at position {start}")
            i += 1  # skip closing quote
            tokens.append(Token(TokenType.STRING, source[start + 1 : i - 1], start))
            continue

        # Two-character operators (must be checked before single-character)
        if i + 1 < n:
            two = source[i : i + 2]
            if two == "**":
                tokens.append(Token(TokenType.DOUBLESTAR, two, i))
                i += 2
                continue
            if two == "<=":
                tokens.append(Token(TokenType.LE, two, i))
                i += 2
                continue
            if two == ">=":
                tokens.append(Token(TokenType.GE, two, i))
                i += 2
                continue
            if two == "==":
                tokens.append(Token(TokenType.EQEQ, two, i))
                i += 2
                continue
            if two == "->":
                tokens.append(Token(TokenType.ARROW, two, i))
                i += 2
                continue

        # Single-character operators and delimiters
        if c in _SINGLE:
            tokens.append(Token(_SINGLE[c], c, i))
            i += 1
            continue

        raise TokenizeError(f"Unexpected character {c!r} at position {i}")

    tokens.append(Token(TokenType.EOF, "", n))
    return tokens
