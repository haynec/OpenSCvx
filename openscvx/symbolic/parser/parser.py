"""Pratt parser for symbolic expression strings.

Converts a token stream into an ``Expr`` AST using precedence climbing.
The parser resolves named identifiers via a user-supplied symbol table and
delegates function-call syntax (``Name(args...)``) to handlers registered
in :mod:`openscvx.symbolic.parser._registry`.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from openscvx.symbolic.expr.arithmetic import Add, Div, MatMul, Mul, Neg, Power, Sub
from openscvx.symbolic.expr.array import Concat
from openscvx.symbolic.expr.constraint import (
    CTCS,
    Constraint,
    Equality,
    Inequality,
    NodalConstraint,
)
from openscvx.symbolic.expr.expr import Constant, Expr, NodeReference
from openscvx.symbolic.expr.linalg import Transpose
from openscvx.symbolic.parser._registry import lookup
from openscvx.symbolic.parser.tokenizer import Token, TokenType, tokenize

# ---------------------------------------------------------------------------
# Precedence levels (higher binds tighter)
# ---------------------------------------------------------------------------
_PREC_COMPARISON = 10
_PREC_ADD = 20
_PREC_MUL = 30
_PREC_POWER = 40
_PREC_UNARY = 50
_PREC_POSTFIX = 60

# ---------------------------------------------------------------------------
# Infix rule table:  TokenType → (precedence, associativity, constructor)
#
# The constructor is called as ``constructor(left, right)`` where *left*
# and *right* are already-parsed ``Expr`` nodes.
# ---------------------------------------------------------------------------
_INFIX_RULES: Dict[TokenType, Tuple[int, str, Any]] = {
    # arithmetic
    TokenType.PLUS: (_PREC_ADD, "left", Add),
    TokenType.MINUS: (_PREC_ADD, "left", Sub),
    TokenType.STAR: (_PREC_MUL, "left", Mul),
    TokenType.SLASH: (_PREC_MUL, "left", Div),
    TokenType.AT: (_PREC_MUL, "left", MatMul),
    TokenType.DOUBLESTAR: (_PREC_POWER, "right", Power),
    # comparison → constraint nodes
    TokenType.LE: (_PREC_COMPARISON, "left", Inequality),
    TokenType.GE: (_PREC_COMPARISON, "left", lambda left, right: Inequality(right, left)),
    TokenType.EQEQ: (_PREC_COMPARISON, "left", Equality),
}


class ParseError(Exception):
    """Raised when the parser encounters a syntactic or semantic error."""


class ExprParser:
    """Pratt parser that converts expression strings to ``Expr`` AST nodes.

    Args:
        symbols: Dict mapping identifier names to live ``Expr`` objects
            (``State``, ``Control``, ``Parameter``, etc.) that should be
            available in the expression namespace.

    Example::

        parser = ExprParser({"pos": pos_state, "vel": vel_state})
        expr = parser.parse("vel + [0, 0, -9.81]")
    """

    def __init__(self, symbols: Dict[str, Expr]):
        self.symbols = symbols
        self._tokens: List[Token] = []
        self._pos: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, source: str) -> Expr:
        """Parse an expression string into an ``Expr`` AST.

        Args:
            source: Expression string to parse.

        Returns:
            The parsed ``Expr``.

        Raises:
            ParseError: On syntax errors or unknown identifiers.
        """
        self._tokens = tokenize(source)
        self._pos = 0
        expr = self._parse_expr(0)
        if self._peek().type != TokenType.EOF:
            tok = self._peek()
            raise ParseError(f"Unexpected token {tok.value!r} at position {tok.pos}")
        return expr

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, tt: TokenType) -> Token:
        tok = self._advance()
        if tok.type != tt:
            raise ParseError(
                f"Expected {tt.name}, got {tok.type.name} ({tok.value!r}) at position {tok.pos}"
            )
        return tok

    # ------------------------------------------------------------------
    # Core Pratt expression parser
    # ------------------------------------------------------------------

    def _parse_expr(self, min_prec: int) -> Expr:
        """Parse an expression with minimum binding power *min_prec*."""
        left = self._parse_prefix()

        while True:
            tok = self._peek()

            # --- table-driven infix operators ---
            rule = _INFIX_RULES.get(tok.type)
            if rule is not None:
                prec, assoc, constructor = rule
                if prec < min_prec:
                    break
                self._advance()
                right_prec = prec if assoc == "right" else prec + 1
                left = constructor(left, self._parse_expr(right_prec))
                continue

            # --- postfix: indexing ---
            if tok.type == TokenType.LBRACKET and _PREC_POSTFIX >= min_prec:
                left = self._parse_index(left)
            elif tok.type == TokenType.DOT and _PREC_POSTFIX >= min_prec:
                left = self._parse_dot(left)
            else:
                break

        return left

    # ------------------------------------------------------------------
    # Prefix (nud) parsing
    # ------------------------------------------------------------------

    def _parse_prefix(self) -> Expr:
        tok = self._peek()

        # Unary minus
        if tok.type == TokenType.MINUS:
            self._advance()
            return Neg(self._parse_expr(_PREC_UNARY))

        # Parenthesised expression
        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expr(0)
            self._expect(TokenType.RPAREN)
            return expr

        # Array literal  [a, b, c]
        if tok.type == TokenType.LBRACKET:
            return self._parse_array_literal()

        # Number literal
        if tok.type == TokenType.NUMBER:
            self._advance()
            return Constant(np.array(float(tok.value)))

        # Identifier: function call **or** symbol lookup
        if tok.type == TokenType.IDENT:
            self._advance()
            name = tok.value

            # Function call: Name(...)
            if self._peek().type == TokenType.LPAREN:
                return self._parse_function_call(name)

            # Built-in constants
            if name == "True":
                return Constant(np.array(1.0))
            if name == "False":
                return Constant(np.array(0.0))
            if name == "pi":
                return Constant(np.array(np.pi))

            # Symbol table lookup
            if name in self.symbols:
                return self.symbols[name]

            raise ParseError(f"Unknown identifier {name!r} at position {tok.pos}")

        raise ParseError(f"Unexpected token {tok.type.name} ({tok.value!r}) at position {tok.pos}")

    # ------------------------------------------------------------------
    # Function calls:  Name(arg, ..., key=val, ...)
    # ------------------------------------------------------------------

    def _parse_function_call(self, name: str) -> Expr:
        self._expect(TokenType.LPAREN)
        args, kwargs = self._parse_call_args()
        self._expect(TokenType.RPAREN)

        handler = lookup(name)
        if handler is None:
            raise ParseError(f"Unknown function {name!r}")
        return handler(args, kwargs)

    def _parse_call_args(self) -> Tuple[list, dict]:
        """Parse ``arg, ..., key=val, ...`` returning ``(args, kwargs)``."""
        args: list = []
        kwargs: dict = {}

        if self._peek().type == TokenType.RPAREN:
            return args, kwargs

        while True:
            # Keyword argument?  IDENT =  (single =, not ==)
            if (
                self._peek().type == TokenType.IDENT
                and self._pos + 1 < len(self._tokens)
                and self._tokens[self._pos + 1].type == TokenType.EQ
            ):
                key = self._advance().value  # consume IDENT
                self._advance()  # consume =
                kwargs[key] = self._parse_arg_value()
            else:
                if kwargs:
                    raise ParseError("Positional argument follows keyword argument")
                args.append(self._parse_arg_value())

            if self._peek().type == TokenType.COMMA:
                self._advance()
            else:
                break

        return args, kwargs

    def _parse_arg_value(self) -> Any:
        """Parse a single argument value (Expr, string, bool, or None)."""
        tok = self._peek()

        # String literal
        if tok.type == TokenType.STRING:
            self._advance()
            return tok.value

        # Boolean / None keywords
        if tok.type == TokenType.IDENT and tok.value in ("True", "False", "None"):
            self._advance()
            if tok.value == "True":
                return True
            if tok.value == "False":
                return False
            return None

        return self._parse_expr(0)

    # ------------------------------------------------------------------
    # Indexing:  expr[spec]
    # ------------------------------------------------------------------

    def _parse_index(self, base: Expr) -> Expr:
        from openscvx.symbolic.expr.array import Index

        self._expect(TokenType.LBRACKET)
        indices: list = []

        while True:
            indices.append(self._parse_index_element())
            if self._peek().type == TokenType.COMMA:
                self._advance()
            else:
                break

        self._expect(TokenType.RBRACKET)

        idx = indices[0] if len(indices) == 1 else tuple(indices)
        return Index(base, idx)

    def _parse_index_element(self) -> Union[int, slice]:
        """Parse a single index dimension: ``int``, or ``[start]:[stop][:step]``."""
        # Leading `:`  →  slice starting from None
        if self._peek().type == TokenType.COLON:
            return self._parse_slice_from_colon(None)

        # Expression (likely a constant integer)
        expr = self._parse_expr(0)

        # Followed by `:`  →  it was *start* of a slice
        if self._peek().type == TokenType.COLON:
            return self._parse_slice_from_colon(self._const_to_int(expr))

        # Plain integer index
        return self._const_to_int(expr)

    def _parse_slice_from_colon(self, start: Optional[int]) -> slice:
        """Parse ``:[stop][:step]`` having already consumed *start*."""
        self._advance()  # consume ':'

        stop: Optional[int] = None
        step: Optional[int] = None

        # stop?
        if self._peek().type not in (
            TokenType.COLON,
            TokenType.RBRACKET,
            TokenType.COMMA,
        ):
            stop = self._const_to_int(self._parse_expr(0))

        # step?
        if self._peek().type == TokenType.COLON:
            self._advance()
            if self._peek().type not in (TokenType.RBRACKET, TokenType.COMMA):
                step = self._const_to_int(self._parse_expr(0))

        return slice(start, stop, step)

    @staticmethod
    def _const_to_int(expr: Expr) -> int:
        """Extract a Python ``int`` from a Constant (or negated Constant)."""
        if isinstance(expr, Constant) and expr.value.ndim == 0:
            return int(expr.value)
        if (
            isinstance(expr, Neg)
            and isinstance(expr.operand, Constant)
            and expr.operand.value.ndim == 0
        ):
            return -int(expr.operand.value)
        raise ParseError(f"Expected constant integer, got {type(expr).__name__}")

    # ------------------------------------------------------------------
    # Dot access:  .T, .at(...), .over(...), .convex()
    # ------------------------------------------------------------------

    def _parse_dot(self, left: Expr) -> Expr:
        self._advance()  # consume '.'
        name_tok = self._expect(TokenType.IDENT)
        name = name_tok.value

        if name == "T":
            return Transpose(left)

        if name == "at":
            return self._parse_dot_at(left)

        if name == "over":
            return self._parse_dot_over(left)

        if name == "convex":
            self._expect(TokenType.LPAREN)
            self._expect(TokenType.RPAREN)
            if isinstance(left, (Constraint, NodalConstraint)):
                return left.convex()
            raise ParseError(".convex() can only be called on a Constraint")

        raise ParseError(f"Unknown method/property {name!r} at position {name_tok.pos}")

    # -- .at() ---------------------------------------------------------

    def _parse_dot_at(self, left: Expr) -> Expr:
        self._expect(TokenType.LPAREN)
        args, _ = self._parse_call_args()
        self._expect(TokenType.RPAREN)

        if isinstance(left, Constraint):
            # Constraint.at(nodes) → NodalConstraint
            nodes = self._args_to_int_list(args)
            return NodalConstraint(left, nodes)

        # Expr.at(k) → NodeReference
        if len(args) != 1:
            raise ParseError(".at() on an expression requires exactly 1 integer argument")
        return NodeReference(left, self._arg_to_int(args[0]))

    # -- .over() -------------------------------------------------------

    def _parse_dot_over(self, left: Expr) -> Expr:
        self._expect(TokenType.LPAREN)
        args, kwargs = self._parse_call_args()
        self._expect(TokenType.RPAREN)

        if not isinstance(left, Constraint):
            raise ParseError(".over() can only be called on a Constraint")

        if len(args) < 2:
            raise ParseError(".over() requires at least 2 positional args (start, end)")

        start = self._arg_to_int(args[0])
        end = self._arg_to_int(args[1])
        penalty = str(kwargs.get("penalty", "squared_relu"))
        idx = kwargs.get("idx", None)
        if idx is not None:
            idx = int(idx) if not isinstance(idx, int) else idx
        check_nodally = bool(kwargs.get("check_nodally", False))

        return CTCS(
            left,
            penalty=penalty,
            nodes=(start, end),
            idx=idx,
            check_nodally=check_nodally,
        )

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _arg_to_int(val: Any) -> int:
        """Coerce an argument value to a Python int."""
        if isinstance(val, int):
            return val
        if isinstance(val, float) and val == int(val):
            return int(val)
        if isinstance(val, Constant) and val.value.ndim == 0:
            return int(val.value)
        if (
            isinstance(val, Neg)
            and isinstance(val.operand, Constant)
            and val.operand.value.ndim == 0
        ):
            return -int(val.operand.value)
        raise ParseError(f"Expected integer, got {type(val).__name__}")

    @classmethod
    def _args_to_int_list(cls, args: list) -> List[int]:
        """Coerce a list of argument values to a list of Python ints.

        Supports both ``at(0, 10, 20)`` (multiple args) and
        ``at([0, 10, 20])`` (single Constant array arg).
        """
        # Single array-constant argument  →  extract elements
        if len(args) == 1 and isinstance(args[0], Constant) and args[0].value.ndim == 1:
            return [int(v) for v in args[0].value]

        return [cls._arg_to_int(a) for a in args]

    # ------------------------------------------------------------------
    # Array literals:  [a, b, c]
    # ------------------------------------------------------------------

    def _parse_array_literal(self) -> Expr:
        self._advance()  # consume '['
        elements: list = []

        if self._peek().type != TokenType.RBRACKET:
            while True:
                elements.append(self._parse_expr(0))
                if self._peek().type == TokenType.COMMA:
                    self._advance()
                else:
                    break

        self._expect(TokenType.RBRACKET)

        if not elements:
            return Constant(np.array([]))

        # All-constant  →  fold into a single Constant
        if all(isinstance(e, Constant) for e in elements):
            return Constant(
                np.array([e.value.item() if e.value.ndim == 0 else e.value for e in elements])
            )

        # Mixed  →  Concat (each element treated as at-least-1D)
        return Concat(*elements)
