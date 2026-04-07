"""Pratt parser for symbolic expression strings.

Converts a token stream into an ``Expr`` AST using precedence climbing.
The parser resolves named identifiers via a user-supplied symbol table and
delegates function-call syntax (``Name(args...)``) to handlers registered
in :mod:`openscvx.symbolic.parser._registry`.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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
from openscvx.symbolic.parser._registry import _PARSE_FUNCTIONS, lookup
from openscvx.symbolic.parser.tokenizer import Token, TokenType, tokenize

# ---------------------------------------------------------------------------
# "Did you mean?" helper
# ---------------------------------------------------------------------------


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def _suggest(name: str, candidates: Iterable[str], max_distance: int = 3) -> Optional[str]:
    """Return the closest candidate within *max_distance*, or ``None``."""
    best, best_dist = None, max_distance + 1
    for c in candidates:
        d = _edit_distance(name, c)
        if d < best_dist:
            best, best_dist = c, d
    return best


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
                return self._parse_function_call(name, tok.pos)

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

            msg = f"Unknown identifier {name!r} at position {tok.pos}"
            hint = _suggest(name, self.symbols)
            if hint:
                msg += f"; did you mean {hint!r}?"
            raise ParseError(msg)

        raise ParseError(f"Unexpected token {tok.type.name} ({tok.value!r}) at position {tok.pos}")

    # ------------------------------------------------------------------
    # Function calls:  Name(arg, ..., key=val, ...)
    # ------------------------------------------------------------------

    def _parse_function_call(self, name: str, pos: int = 0) -> Expr:
        if name.lower() == "vmap":
            return self._parse_vmap_call()

        self._expect(TokenType.LPAREN)
        args, kwargs = self._parse_call_args()
        self._expect(TokenType.RPAREN)

        handler = lookup(name)  # case-insensitive
        if handler is None:
            msg = f"Unknown function {name!r} at position {pos}"
            hint = _suggest(name.lower(), _PARSE_FUNCTIONS)
            if hint:
                msg += f"; did you mean {hint!r}?"
            raise ParseError(msg)
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

    # ------------------------------------------------------------------
    # Vmap:  Vmap(name: source, ... -> body_expr)
    # ------------------------------------------------------------------

    def _parse_vmap_call(self) -> Expr:
        """Parse ``Vmap(name: source, ... [, axis=N] -> body)``.

        Bindings (``name: source``) map placeholder names to batch sources
        in the symbol table.  The optional ``axis=N`` keyword sets the vmap
        axis (default 0).  Everything after ``->`` is the body expression,
        parsed with the placeholders temporarily added to the symbol table.
        """
        from openscvx.symbolic.expr.control import Control
        from openscvx.symbolic.expr.parameter import Parameter
        from openscvx.symbolic.expr.state import State
        from openscvx.symbolic.expr.vmap import Vmap, _Placeholder

        self._expect(TokenType.LPAREN)

        # -- Parse bindings and kwargs before '->' -----------------------
        bindings: List[Tuple[str, str]] = []  # (placeholder_name, source_name)
        axis = 0

        while True:
            if self._peek().type == TokenType.ARROW:
                self._advance()
                break

            name_tok = self._expect(TokenType.IDENT)
            next_tok = self._peek()

            if next_tok.type == TokenType.COLON:
                # Binding: name : source
                self._advance()  # consume ':'
                source_tok = self._expect(TokenType.IDENT)
                bindings.append((name_tok.value, source_tok.value))
            elif next_tok.type == TokenType.EQ:
                # Keyword arg: axis = N
                self._advance()  # consume '='
                val = self._parse_arg_value()
                if name_tok.value == "axis":
                    axis = self._arg_to_int(val) if not isinstance(val, int) else val
                else:
                    raise ParseError(f"Unknown Vmap keyword {name_tok.value!r} (expected 'axis')")
            else:
                raise ParseError(f"Expected ':' or '=' after {name_tok.value!r} in Vmap bindings")

            if self._peek().type == TokenType.COMMA:
                self._advance()

        if not bindings:
            raise ParseError("Vmap requires at least one binding (name: source)")

        # -- Resolve batch sources and create placeholders ---------------
        batches = []
        is_parameter = []
        is_state = []
        is_control = []
        placeholders = []
        saved_symbols: Dict[str, Expr] = {}

        for ph_name, source_name in bindings:
            source = self.symbols.get(source_name)
            if source is None:
                raise ParseError(f"Unknown batch source {source_name!r}")
            if isinstance(source, np.ndarray):
                source = Constant(source)
            elif not isinstance(source, (Constant, Parameter, State, Control)):
                raise ParseError(
                    f"Batch source {source_name!r} must be a Parameter, State, Control, or Constant"
                )

            is_p = isinstance(source, Parameter)
            is_s = isinstance(source, State)
            is_c = isinstance(source, Control)
            batches.append(source)
            is_parameter.append(is_p)
            is_state.append(is_s)
            is_control.append(is_c)

            shape = Vmap._get_batch_shape(source, is_p, is_s, is_c)
            if axis < 0 or axis >= len(shape):
                raise ParseError(
                    f"Vmap axis {axis} out of bounds for {source_name!r} with shape {shape}"
                )
            per_elem_shape = tuple(s for i, s in enumerate(shape) if i != axis)
            ph = _Placeholder(shape=per_elem_shape)
            placeholders.append(ph)

            # Temporarily shadow any existing symbol
            if ph_name in self.symbols:
                saved_symbols[ph_name] = self.symbols[ph_name]
            self.symbols[ph_name] = ph

        # Validate batch sizes match along the vmap axis
        first_shape = Vmap._get_batch_shape(batches[0], is_parameter[0], is_state[0], is_control[0])
        batch_size = first_shape[axis]
        for i, (b, is_p, is_s, is_c) in enumerate(zip(batches, is_parameter, is_state, is_control)):
            s = Vmap._get_batch_shape(b, is_p, is_s, is_c)[axis]
            if s != batch_size:
                raise ParseError(
                    f"Batch size mismatch: binding 0 has size {batch_size} "
                    f"along axis {axis}, but binding {i} has size {s}"
                )

        # -- Parse body expression ---------------------------------------
        try:
            body = self._parse_expr(0)
        finally:
            # Restore symbol table (even on parse errors)
            for ph_name, _ in bindings:
                if ph_name in saved_symbols:
                    self.symbols[ph_name] = saved_symbols[ph_name]
                else:
                    del self.symbols[ph_name]

        self._expect(TokenType.RPAREN)

        # -- Construct Vmap (bypass __init__ like canonicalize does) ------
        vmap = Vmap.__new__(Vmap)
        vmap._batches = tuple(batches)
        vmap._axis = axis
        vmap._placeholders = tuple(placeholders)
        vmap._child = body
        vmap._is_parameter = tuple(is_parameter)
        vmap._is_state = tuple(is_state)
        vmap._is_control = tuple(is_control)
        return vmap

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

        # Fold Neg(Constant(scalar)) so that [1, -2, 3] becomes a single Constant
        elements = [self._fold_neg_constant(e) for e in elements]

        # All-constant  →  fold into a single Constant
        if all(isinstance(e, Constant) for e in elements):
            return Constant(
                np.array([e.value.item() if e.value.ndim == 0 else e.value for e in elements])
            )

        # Mixed  →  Concat (each element treated as at-least-1D)
        return Concat(*elements)

    @staticmethod
    def _fold_neg_constant(expr: Expr) -> Expr:
        """Fold ``Neg(Constant(scalar))`` into ``Constant(-scalar)``."""
        if (
            isinstance(expr, Neg)
            and isinstance(expr.operand, Constant)
            and expr.operand.value.ndim == 0
        ):
            return Constant(-expr.operand.value)
        return expr
