"""Arithmetic operations for symbolic expressions.

This module provides fundamental arithmetic operations that form the building blocks
of symbolic expressions in openscvx. These operations are created automatically through
operator overloading on Expr objects.

Arithmetic Operations:

- **Binary operations:** `Add`, `Sub`, `Mul`, `Div`, `MatMul`, `Power` - Standard arithmetic
- **Unary operations:** `Neg` - Negation (unary minus)

All arithmetic operations support:
- Automatic canonicalization (constant folding, identity elimination, flattening)
- Broadcasting following NumPy rules (except MatMul which follows linear algebra rules)
- Shape checking and validation

Example:
    Arithmetic operations are created via operator overloading::

        import openscvx as ox

        x = ox.State("x", shape=(3,))
        y = ox.State("y", shape=(3,))

        # Element-wise operations
        z = x + y           # Creates Add(x, y)
        w = x * 2           # Creates Mul(x, Constant(2))
        neg_x = -x          # Creates Neg(x)

        # Matrix multiplication
        A = ox.State("A", shape=(3, 3))
        b = A @ x           # Creates MatMul(A, x)
"""

from typing import Tuple, Union

import numpy as np

from .expr import Constant, Expr, to_expr


class Add(Expr):
    """Addition operation for symbolic expressions.

    Represents element-wise addition of two or more expressions. Supports broadcasting
    following NumPy rules. Can be created using the + operator on Expr objects.

    Attributes:
        terms: List of expression operands to add together

    Example:
        Define an Add expression:

            x = ox.State("x", shape=(3,))
            y = ox.State("y", shape=(3,))
            z = x + y + 5  # Creates Add(x, y, Constant(5))
    """

    def __init__(self, *args: Union[Expr, float, int, np.ndarray]):
        """Initialize an addition operation.

        Args:
            *args: Two or more expressions to add together

        Raises:
            ValueError: If fewer than two operands are provided
        """
        if len(args) < 2:
            raise ValueError("Add requires two or more operands")
        self.terms = [to_expr(a) for a in args]

    def children(self):
        return list(self.terms)

    def canonicalize(self) -> "Expr":
        """Canonicalize addition: flatten, fold constants, and eliminate zeros.

        Returns:
            Expr: Canonical form of the addition expression
        """
        terms = []
        const_vals = []

        for t in self.terms:
            c = t.canonicalize()
            if isinstance(c, Add):
                terms.extend(c.terms)
            elif isinstance(c, Constant):
                const_vals.append(c.value)
            else:
                terms.append(c)

        if const_vals:
            total = sum(const_vals)
            # If not all-zero, keep it
            if not (isinstance(total, np.ndarray) and np.all(total == 0)):
                terms.append(Constant(total))

        if not terms:
            return Constant(np.array(0))
        if len(terms) == 1:
            return terms[0]
        return Add(*terms)

    def check_shape(self) -> Tuple[int, ...]:
        """Check shape compatibility and compute broadcasted result shape like NumPy.

        Returns:
            tuple: The broadcasted shape of all operands

        Raises:
            ValueError: If operand shapes are not broadcastable
        """
        shapes = [child.check_shape() for child in self.children()]
        try:
            return np.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"Add shapes not broadcastable: {shapes}") from e

    def __repr__(self) -> str:
        inner = " + ".join(repr(e) for e in self.terms)
        return f"({inner})"


class Sub(Expr):
    """Subtraction operation for symbolic expressions.

    Represents element-wise subtraction (left - right). Supports broadcasting
    following NumPy rules. Can be created using the - operator on Expr objects.

    Attributes:
        left: Left-hand side expression (minuend)
        right: Right-hand side expression (subtrahend)

    Example:
        Define a Sub expression:

            x = ox.State("x", shape=(3,))
            y = ox.State("y", shape=(3,))
            z = x - y  # Creates Sub(x, y)
    """

    def __init__(
        self,
        left: Union[Expr, float, int, np.ndarray],
        right: Union[Expr, float, int, np.ndarray],
    ):
        """Initialize a subtraction operation.

        Args:
            left: Expression to subtract from (minuend)
            right: Expression to subtract (subtrahend)
        """
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def canonicalize(self) -> "Expr":
        """Canonicalize subtraction: fold constants if both sides are constants.

        Returns:
            Expr: Canonical form of the subtraction expression
        """
        left = self.left.canonicalize()
        right = self.right.canonicalize()
        if isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value - right.value)
        return Sub(left, right)

    def check_shape(self) -> Tuple[int, ...]:
        """Check shape compatibility and compute broadcasted result shape like NumPy.

        Returns:
            tuple: The broadcasted shape of all operands

        Raises:
            ValueError: If operand shapes are not broadcastable
        """
        shapes = [child.check_shape() for child in self.children()]
        try:
            return np.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"Sub shapes not broadcastable: {shapes}") from e

    def __repr__(self) -> str:
        return f"({self.left!r} - {self.right!r})"


class Mul(Expr):
    """Element-wise multiplication operation for symbolic expressions.

    Represents element-wise (Hadamard) multiplication of two or more expressions.
    Supports broadcasting following NumPy rules. Can be created using the * operator
    on Expr objects. For matrix multiplication, use MatMul or the @ operator.

    Attributes:
        factors: List of expression operands to multiply together

    Example:
        Define a Mul expression:

            x = ox.State("x", shape=(3,))
            y = ox.State("y", shape=(3,))
            z = x * y * 2  # Creates Mul(x, y, Constant(2))
    """

    def __init__(self, *args: Union[Expr, float, int, np.ndarray]):
        """Initialize an element-wise multiplication operation.

        Args:
            *args: Two or more expressions to multiply together

        Raises:
            ValueError: If fewer than two operands are provided
        """
        if len(args) < 2:
            raise ValueError("Mul requires two or more operands")
        self.factors = [to_expr(a) for a in args]

    def children(self):
        return list(self.factors)

    def canonicalize(self) -> "Expr":
        """Canonicalize multiplication: flatten, fold constants, and eliminating ones.

        Returns:
            Expr: Canonical form of the multiplication expression
        """
        factors = []
        const_vals = []

        for f in self.factors:
            c = f.canonicalize()
            if isinstance(c, Mul):
                factors.extend(c.factors)
            elif isinstance(c, Constant):
                const_vals.append(c.value)
            else:
                factors.append(c)

        if const_vals:
            # Multiply constants element-wise (broadcasting), not reducing with prod
            prod = const_vals[0]
            for val in const_vals[1:]:
                prod = prod * val

            # If prod != 1, keep it
            # Check both scalar and array cases
            is_identity = False
            if isinstance(prod, np.ndarray):
                is_identity = np.all(prod == 1)
            else:
                is_identity = prod == 1

            if not is_identity:
                factors.append(Constant(prod))

        if not factors:
            return Constant(np.array(1))
        if len(factors) == 1:
            return factors[0]
        return Mul(*factors)

    def check_shape(self) -> Tuple[int, ...]:
        """Check shape compatibility and compute broadcasted result shape like NumPy.


        Returns:
            tuple: The broadcasted shape of all operands

        Raises:
            ValueError: If operand shapes are not broadcastable
        """
        shapes = [child.check_shape() for child in self.children()]
        try:
            return np.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"Mul shapes not broadcastable: {shapes}") from e

    def __repr__(self) -> str:
        inner = " * ".join(repr(e) for e in self.factors)
        return f"({inner})"


class Div(Expr):
    """Element-wise division operation for symbolic expressions.

    Represents element-wise division (left / right). Supports broadcasting
    following NumPy rules. Can be created using the / operator on Expr objects.

    Attributes:
        left: Numerator expression
        right: Denominator expression

    Example:
        Define a Div expression

            x = ox.State("x", shape=(3,))
            y = ox.State("y", shape=(3,))
            z = x / y  # Creates Div(x, y)
    """

    def __init__(
        self,
        left: Union[Expr, float, int, np.ndarray],
        right: Union[Expr, float, int, np.ndarray],
    ):
        """Initialize a division operation.

        Args:
            left: Expression for the numerator
            right: Expression for the denominator
        """
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def canonicalize(self) -> "Expr":
        """Canonicalize division: fold constants if both sides are constants.

        Returns:
            Expr: Canonical form of the division expression
        """
        lhs = self.left.canonicalize()
        rhs = self.right.canonicalize()
        if isinstance(lhs, Constant) and isinstance(rhs, Constant):
            return Constant(lhs.value / rhs.value)
        return Div(lhs, rhs)

    def check_shape(self) -> Tuple[int, ...]:
        """Check shape compatibility and compute broadcasted result shape like NumPy.

        Returns:
            tuple: The broadcasted shape of both operands

        Raises:
            ValueError: If operand shapes are not broadcastable
        """
        shapes = [child.check_shape() for child in self.children()]
        try:
            return np.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"Div shapes not broadcastable: {shapes}") from e

    def __repr__(self) -> str:
        return f"({self.left!r} / {self.right!r})"


class MatMul(Expr):
    """Matrix multiplication operation for symbolic expressions.

    Represents matrix multiplication following standard linear algebra rules.
    Can be created using the @ operator on Expr objects. Handles:
    - Matrix @ Matrix: (m,n) @ (n,k) -> (m,k)
    - Matrix @ Vector: (m,n) @ (n,) -> (m,)
    - Vector @ Matrix: (m,) @ (m,n) -> (n,)
    - Vector @ Vector: (m,) @ (m,) -> scalar

    Attributes:
        left: Left-hand side expression
        right: Right-hand side expression

    Example:
        Define a MatMul expression:

            A = ox.State("A", shape=(3, 4))
            x = ox.State("x", shape=(4,))
            y = A @ x  # Creates MatMul(A, x), result shape (3,)
    """

    def __init__(
        self,
        left: Union[Expr, float, int, np.ndarray],
        right: Union[Expr, float, int, np.ndarray],
    ):
        """Initialize a matrix multiplication operation.

        Args:
            left: Left-hand side expression for matrix multiplication
            right: Right-hand side expression for matrix multiplication
        """
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def canonicalize(self) -> "Expr":
        left = self.left.canonicalize()
        right = self.right.canonicalize()
        return MatMul(left, right)

    def check_shape(self) -> Tuple[int, ...]:
        """Check matrix multiplication shape compatibility and return result shape."""
        L, R = self.left.check_shape(), self.right.check_shape()

        # Handle different matmul cases:
        # Matrix @ Matrix: (m,n) @ (n,k) -> (m,k)
        # Matrix @ Vector: (m,n) @ (n,) -> (m,)
        # Vector @ Matrix: (m,) @ (m,n) -> (n,)
        # Vector @ Vector: (m,) @ (m,) -> ()

        if len(L) == 0 or len(R) == 0:
            raise ValueError(f"MatMul requires at least 1D operands: {L} @ {R}")

        if len(L) == 1 and len(R) == 1:
            # Vector @ Vector -> scalar
            if L[0] != R[0]:
                raise ValueError(f"MatMul incompatible: {L} @ {R}")
            return ()
        elif len(L) == 1:
            # Vector @ Matrix: (m,) @ (m,n) -> (n,)
            if len(R) < 2 or L[0] != R[-2]:
                raise ValueError(f"MatMul incompatible: {L} @ {R}")
            return R[-1:]
        elif len(R) == 1:
            # Matrix @ Vector: (m,n) @ (n,) -> (m,)
            if len(L) < 2 or L[-1] != R[0]:
                raise ValueError(f"MatMul incompatible: {L} @ {R}")
            return L[:-1]
        else:
            # Matrix @ Matrix: (...,m,n) @ (...,n,k) -> (...,m,k)
            if len(L) < 2 or len(R) < 2 or L[-1] != R[-2]:
                raise ValueError(f"MatMul incompatible: {L} @ {R}")
            return L[:-1] + (R[-1],)

    def __repr__(self) -> str:
        return f"({self.left!r} * {self.right!r})"


class Neg(Expr):
    """Negation operation for symbolic expressions.

    Represents element-wise negation (unary minus). Can be created using the
    unary - operator on Expr objects.

    Attributes:
        operand: Expression to negate

    Example:
        Define a Neg expression:

            x = ox.State("x", shape=(3,))
            y = -x  # Creates Neg(x)
    """

    def __init__(self, operand: Union[Expr, float, int, np.ndarray]):
        """Initialize a negation operation.

        Args:
            operand: Expression to negate
        """
        self.operand = operand

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize negation: fold constant negations.

        Returns:
            Expr: Canonical form of the negation expression
        """
        o = self.operand.canonicalize()
        if isinstance(o, Constant):
            return Constant(-o.value)
        return Neg(o)

    def check_shape(self) -> Tuple[int, ...]:
        """Negation preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self) -> str:
        return f"(-{self.operand!r})"


class Power(Expr):
    """Element-wise power operation for symbolic expressions.

    Represents element-wise exponentiation (base ** exponent). Supports broadcasting
    following NumPy rules. Can be created using the ** operator on Expr objects.

    Attributes:
        base: Base expression
        exponent: Exponent expression

    Example:
        Define a Power expression:

            x = ox.State("x", shape=(3,))
            y = x ** 2  # Creates Power(x, Constant(2))
    """

    def __init__(
        self,
        base: Union[Expr, float, int, np.ndarray],
        exponent: Union[Expr, float, int, np.ndarray],
    ):
        """Initialize a power operation.

        Args:
            base: Base expression
            exponent: Exponent expression
        """
        self.base = to_expr(base)
        self.exponent = to_expr(exponent)

    def children(self):
        return [self.base, self.exponent]

    def canonicalize(self) -> "Expr":
        """Canonicalize power by canonicalizing base and exponent.

        Returns:
            Expr: Canonical form of the power expression
        """
        base = self.base.canonicalize()
        exponent = self.exponent.canonicalize()
        return Power(base, exponent)

    def check_shape(self) -> Tuple[int, ...]:
        shapes = [child.check_shape() for child in self.children()]
        try:
            return np.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"Power shapes not broadcastable: {shapes}") from e

    def __repr__(self) -> str:
        return f"({self.base!r})**({self.exponent!r})"
