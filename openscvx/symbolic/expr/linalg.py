"""Linear algebra operations for symbolic expressions.

This module provides essential linear algebra operations for matrix and vector
manipulation in optimization problems. Operations follow NumPy/JAX conventions
for shapes and broadcasting behavior.

Key Operations:
    - **Matrix Operations:**
        - `Transpose` - Matrix/tensor transposition (swaps last two dimensions)
        - `Diag` - Construct diagonal matrix from vector
        - `Inv` - Matrix inverse (square matrices only, JAX lowering only)
    - **Reductions:**
        - `Sum` - Sum all elements of an array (reduces to scalar)
        - `Norm` - Euclidean (L2) norm and other norms of vectors/matrices

Note:
    For array manipulation operations like stacking and concatenation, see the
    `array` module.

Example:
    Matrix transposition and diagonal matrices::

        import openscvx as ox
        import numpy as np

        # Transpose a matrix
        A = ox.State("A", shape=(3, 4))
        A_T = A.T  # Result shape (4, 3)

        # Create a diagonal matrix
        v = ox.State("v", shape=(5,))
        D = ox.Diag(v)  # Result shape (5, 5)

    Reduction operations::

        x = ox.State("x", shape=(3, 4))

        # Sum all elements
        total = ox.Sum(x)  # Result is scalar

        # Compute norm
        magnitude = ox.Norm(x)  # Result is scalar

    Computing kinetic energy with norms::

        v = ox.State("v", shape=(3,))  # Velocity vector
        m = 10.0  # Mass
        kinetic_energy = 0.5 * m * ox.Norm(v)**2
"""

import hashlib
from typing import Tuple, Union

import numpy as np

from .expr import Constant, Expr, to_expr


class Transpose(Expr):
    """Matrix transpose operation for symbolic expressions.

    Transposes the last two dimensions of an expression. For matrices, this swaps
    rows and columns. For higher-dimensional arrays, it swaps the last two axes.
    Scalars and vectors are unchanged by transposition.

    The canonicalization includes an optimization that eliminates double transposes:
    (A.T).T simplifies to A.

    Attributes:
        operand: Expression to transpose

    Example:
        Define Tranpose expressions:

            A = Variable("A", shape=(3, 4))
            A_T = Transpose(A)  # or A.T, result shape (4, 3)
            v = Variable("v", shape=(5,))
            v_T = Transpose(v)  # result shape (5,) - vectors unchanged
    """

    def __init__(self, operand: Union[Expr, float, int, np.ndarray]):
        """Initialize a transpose operation.

        Args:
            operand: Expression to transpose
        """
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand with double transpose optimization."""
        operand = self.operand.canonicalize()

        # Double transpose optimization: (A.T).T = A
        if isinstance(operand, Transpose):
            return operand.operand

        return Transpose(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Matrix transpose operation swaps the last two dimensions."""
        operand_shape = self.operand.check_shape()

        if len(operand_shape) == 0:
            # Scalar transpose is the scalar itself
            return ()
        elif len(operand_shape) == 1:
            # Vector transpose is the vector itself (row vector remains row vector)
            return operand_shape
        elif len(operand_shape) == 2:
            # Matrix transpose: (m,n) -> (n,m)
            return (operand_shape[1], operand_shape[0])
        else:
            # Higher-dimensional array: transpose last two dimensions
            # (..., m, n) -> (..., n, m)
            return operand_shape[:-2] + (operand_shape[-1], operand_shape[-2])

    def __repr__(self) -> str:
        return f"({self.operand!r}).T"


class Diag(Expr):
    """Diagonal matrix construction from a vector.

    Creates a square diagonal matrix from a 1D vector. The vector elements become
    the diagonal entries, with all off-diagonal entries set to zero. This is
    analogous to numpy.diag() or jax.numpy.diag().

    Note:
        Currently only supports creating diagonal matrices from vectors.
        Extracting diagonals from matrices is not yet implemented.

    Attributes:
        operand: 1D vector expression to place on the diagonal

    Example:
        Define a Diag:

            v = Variable("v", shape=(3,))
            D = Diag(v)  # Creates a (3, 3) diagonal matrix
    """

    def __init__(self, operand: Union[Expr, float, int, np.ndarray]):
        """Initialize a diagonal matrix operation.

        Args:
            operand: 1D vector expression to place on the diagonal
        """
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        operand = self.operand.canonicalize()
        return Diag(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Diag converts a vector (n,) to a diagonal matrix (n,n)."""
        operand_shape = self.operand.check_shape()
        if len(operand_shape) != 1:
            raise ValueError(f"Diag expects a 1D vector, got shape {operand_shape}")
        n = operand_shape[0]
        return (n, n)

    def __repr__(self) -> str:
        return f"diag({self.operand!r})"


class Sum(Expr):
    """Sum reduction operation for symbolic expressions.

    Sums all elements of an expression, reducing it to a scalar. This is a
    reduction operation that collapses all dimensions.

    Attributes:
        operand: Expression whose elements will be summed

    Example:
        Define a Sum expression::

            x = ox.State("x", shape=(3, 4))
            total = Sum(x)  # Creates Sum(x), result shape ()
    """

    def __init__(self, operand: Union[Expr, float, int, np.ndarray]):
        """Initialize a sum reduction operation.

        Args:
            operand: Expression to sum over all elements
        """
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize sum: canonicalize the operand.

        Returns:
            Expr: Canonical form of the sum expression
        """
        operand = self.operand.canonicalize()
        return Sum(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Sum reduces any shape to a scalar."""
        # Validate that the operand has a valid shape
        self.operand.check_shape()
        # Sum always produces a scalar regardless of input shape
        return ()

    def __repr__(self) -> str:
        return f"sum({self.operand!r})"


class Inv(Expr):
    """Matrix inverse operation for symbolic expressions.

    Computes the inverse of a square matrix. For batched inputs with shape
    (..., M, M), inverts the last two dimensions following jax.numpy.linalg.inv
    conventions.

    The canonicalization includes an optimization that eliminates double inverses:
    Inv(Inv(A)) simplifies to A.

    Attributes:
        operand: Square matrix expression to invert

    Example:
        Define matrix inverse expressions::

            M = Variable("M", shape=(3, 3))
            M_inv = Inv(M)  # Result shape (3, 3)

            # Batched case
            M_batch = Variable("M_batch", shape=(5, 3, 3))
            M_batch_inv = Inv(M_batch)  # Result shape (5, 3, 3)

    Note:
        Matrix inverse is non-convex and only supported in JAX lowering.
        CVXPy lowering will raise NotImplementedError since inv(X) is neither
        convex nor concave for variable matrices.

    !!! warning
        Solving a matrix inverse inside an optimization loop can be somewhat
        of an oxymoron and performance may be severly impacted.
        Consider whether your problem can be reformulated to avoid the inverse.
    """

    def __init__(self, operand: Union[Expr, float, int, np.ndarray]):
        """Initialize a matrix inverse operation.

        Args:
            operand: Square matrix expression to invert. Must have shape
                (..., M, M) where the last two dimensions are equal.
        """
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand with double inverse optimization and constant folding."""
        operand = self.operand.canonicalize()

        # Double inverse optimization: Inv(Inv(A)) = A
        if isinstance(operand, Inv):
            return operand.operand

        # Constant folding: compute inverse at canonicalization time
        if isinstance(operand, Constant):
            return Constant(np.linalg.inv(operand.value))

        return Inv(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Matrix inverse preserves shape; validates square matrix."""
        operand_shape = self.operand.check_shape()

        if len(operand_shape) < 2:
            raise ValueError(f"Inv requires at least a 2D matrix, got shape {operand_shape}")

        if operand_shape[-1] != operand_shape[-2]:
            raise ValueError(
                f"Inv requires a square matrix (last two dims must be equal), "
                f"got shape {operand_shape}"
            )

        return operand_shape

    def __repr__(self) -> str:
        return f"inv({self.operand!r})"


class Norm(Expr):
    """Norm operation for symbolic expressions (reduction to scalar).

    Computes the norm of an expression according to the specified order parameter.
    This is a reduction operation that always produces a scalar result regardless
    of the input shape. Supports various norm types following NumPy/SciPy conventions.

    Attributes:
        operand: Expression to compute norm of
        ord: Norm order specification (default: "fro" for Frobenius norm)
            - "fro": Frobenius norm (default)
            - "inf": Infinity norm
            - 1: L1 norm (sum of absolute values)
            - 2: L2 norm (Euclidean norm)
            - Other values as supported by the backend

    Example:
        Define Norms:

            x = Variable("x", shape=(3,))
            euclidean_norm = Norm(x, ord=2)  # L2 norm, result is scalar
            A = Variable("A", shape=(3, 4))
            frobenius_norm = Norm(A)  # Frobenius norm, result is scalar
    """

    def __init__(
        self,
        operand: Union[Expr, float, int, np.ndarray],
        ord: Union[str, int, float] = "fro",
    ):
        """Initialize a norm operation.

        Args:
            operand: Expression to compute norm of
            ord: Norm order specification (default: "fro")
        """
        self.operand = to_expr(operand)
        self.ord = ord  # Can be "fro", "inf", 1, 2, etc.

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand but preserve the ord parameter."""
        canon_operand = self.operand.canonicalize()
        return Norm(canon_operand, ord=self.ord)

    def check_shape(self) -> Tuple[int, ...]:
        """Norm reduces any shape to a scalar."""
        # Validate that the operand has a valid shape
        self.operand.check_shape()
        # Norm always produces a scalar regardless of input shape
        return ()

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Hash Norm including its ord parameter.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"Norm")
        # Hash the ord parameter
        hasher.update(repr(self.ord).encode())
        # Hash the operand
        self.operand._hash_into(hasher)

    def __repr__(self) -> str:
        return f"norm({self.operand!r}, ord={self.ord!r})"
