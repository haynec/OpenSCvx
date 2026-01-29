"""Core symbolic expression system for trajectory optimization.

This module provides the foundation for openscvx's symbolic expression framework,
implementing an Abstract Syntax Tree (AST) representation for mathematical expressions
used in optimization problems. The expression system enables:

- Declarative problem specification: Write optimization problems using familiar
    mathematical notation with operator overloading (+, -, *, /, @, **, etc.)
- Automatic differentiation: Expressions are automatically differentiated during
    compilation to solver-specific formats
- Shape checking: Static validation of tensor dimensions before optimization
- Canonicalization: Algebraic simplification for more efficient compilation
- Multiple backends: Expressions can be compiled to CVXPy, JAX, or custom solvers

Architecture:
    The expression system is built around an AST where each node is an `Expr` subclass:

    - Leaf nodes: `Parameter`, `Variable`, `State`, `Control` - symbolic values
    - Arithmetic operations: `Add`, `Sub`, `Mul`, `Div`, `MatMul`, `Power`, `Neg`
    - Array operations: `Index`, `Concat`, `Stack`, `Hstack`, `Vstack`
    - Linear algebra: `Transpose`, `Diag`, `Sum`, `Norm`
    - Constraints: `Equality`, `Inequality`
    - Functions: `Sin`, `Cos`, `Exp`, `Log`, `Sqrt`, etc.

    Each expression node implements:

    - `children()`: Returns child expressions in the AST
    - `canonicalize()`: Returns a simplified/normalized version
    - `check_shape()`: Validates and returns the output shape

Example:
    Creating symbolic variables and expressions::

        import openscvx as ox

        # Define symbolic variables
        x = ox.State("x", shape=(3,))
        A = ox.Parameter("A", shape=(3, 3), value=np.eye(3))

        # Build expressions using natural syntax
        expr = A @ x + 5
        constraint = ox.Norm(x) <= 1.0

        # Expressions form an AST
        print(expr.pretty())  # Visualize the tree structure

    Shape checking with automatic validation::

        x = ox.State("x", shape=(3,))
        y = ox.State("y", shape=(4,))

        # This will raise ValueError during shape checking
        try:
            expr = x + y  # Shapes (3,) and (4,) not broadcastable
            expr.check_shape()
        except ValueError as e:
            print(f"Shape error: {e}")

    Algebraic canonicalization::

        x = ox.State("x", shape=(3,))
        expr = x + 0 + (1 * x)
        canonical = expr.canonicalize()  # Simplifies to: x + x
"""

import hashlib
import struct
from typing import TYPE_CHECKING, Callable, List, Tuple, Union

if TYPE_CHECKING:
    from .linalg import Transpose

import numpy as np


class Expr:
    """Base class for symbolic expressions in optimization problems.

    Expr is the foundation of the symbolic expression system in openscvx. It represents
    nodes in an abstract syntax tree (AST) for mathematical expressions. Expressions
    support:

    - Arithmetic operations: +, -, *, /, @, **
    - Comparison operations: ==, <=, >=
    - Indexing and slicing: []
    - Transposition: .T property
    - Shape checking and validation
    - Canonicalization (algebraic simplification)

    All Expr subclasses implement a tree structure where each node can have child
    expressions accessed via the children() method.

    Attributes:
        __array_priority__: Priority for operations with numpy arrays (set to 1000)

    Note:
        When used in operations with numpy arrays, Expr objects take precedence,
        allowing symbolic expressions to wrap numeric values automatically.
    """

    # Give Expr objects higher priority than numpy arrays in operations
    __array_priority__ = 1000

    def __le__(self, other):
        from .constraint import Inequality

        return Inequality(self, to_expr(other))

    def __ge__(self, other):
        from .constraint import Inequality

        return Inequality(to_expr(other), self)

    def __eq__(self, other):
        from .constraint import Equality

        return Equality(self, to_expr(other))

    def __add__(self, other):
        from .arithmetic import Add

        return Add(self, to_expr(other))

    def __radd__(self, other):
        from .arithmetic import Add

        return Add(to_expr(other), self)

    def __sub__(self, other):
        from .arithmetic import Sub

        return Sub(self, to_expr(other))

    def __rsub__(self, other):
        # e.g. 5 - a  ⇒ Sub(Constant(5), a)
        from .arithmetic import Sub

        return Sub(to_expr(other), self)

    def __truediv__(self, other):
        from .arithmetic import Div

        return Div(self, to_expr(other))

    def __rtruediv__(self, other):
        # e.g. 10 / a
        from .arithmetic import Div

        return Div(to_expr(other), self)

    def __mul__(self, other):
        from .arithmetic import Mul

        return Mul(self, to_expr(other))

    def __rmul__(self, other):
        from .arithmetic import Mul

        return Mul(to_expr(other), self)

    def __matmul__(self, other):
        from .arithmetic import MatMul

        return MatMul(self, to_expr(other))

    def __rmatmul__(self, other):
        from .arithmetic import MatMul

        return MatMul(to_expr(other), self)

    def __rle__(self, other):
        # other <= self  =>  Inequality(other, self)
        from .constraint import Inequality

        return Inequality(to_expr(other), self)

    def __rge__(self, other):
        # other >= self  =>  Inequality(self, other)
        from .constraint import Inequality

        return Inequality(self, to_expr(other))

    def __req__(self, other):
        # other == self  =>  Equality(other, self)
        from .constraint import Equality

        return Equality(to_expr(other), self)

    def __neg__(self):
        from .arithmetic import Neg

        return Neg(self)

    def __pow__(self, other):
        from .arithmetic import Power

        return Power(self, to_expr(other))

    def __rpow__(self, other):
        from .arithmetic import Power

        return Power(to_expr(other), self)

    def __getitem__(self, idx):
        from .array import Index

        return Index(self, idx)

    @property
    def T(self) -> "Transpose":
        """Transpose property for matrix expressions.

        Returns:
            Transpose: A Transpose expression wrapping this expression

        Example:
            Create a transpose:

                A = ox.State("A", shape=(3, 4))
                A_T = A.T  # Creates Transpose(A), result shape (4, 3)
        """
        from .linalg import Transpose

        return Transpose(self)

    def at(self, k: int) -> "NodeReference":
        """Reference this expression at a specific trajectory node.

        This method enables inter-node constraints where you can reference
        the value of an expression at different time steps. Common patterns
        include rate limits and multi-step dependencies.

        Args:
            k: Absolute node index (integer) in the trajectory.
                Can be positive (0, 1, 2, ...) or negative (-1 for last node).

        Returns:
            NodeReference: An expression representing this expression at node k

        Example:
            Rate limit constraint (applied across trajectory using a loop):

                position = State("pos", shape=(3,))

                # Create rate limit for each node
                constraints = [
                    (ox.linalg.Norm(position.at(k) - position.at(k-1)) <= 0.1).at([k])
                    for k in range(1, N)
                ]

            Multi-step dependency:

                state = State("x", shape=(1,))

                # Fibonacci-like recurrence
                constraints = [
                    (state.at(k) == state.at(k-1) + state.at(k-2)).at([k])
                    for k in range(2, N)
                ]

        Performance Note:
            Cross-node constraints use dense Jacobian storage which can be memory-intensive
            for large N (>100 nodes). See LoweredCrossNodeConstraint documentation for
            details on memory usage and future sparse Jacobian support.
        """
        return NodeReference(self, k)

    def children(self) -> List["Expr"]:
        """Return the child expressions of this node.

        Returns:
            list: List of child Expr objects. Empty list for leaf nodes.
        """
        return []

    def canonicalize(self) -> "Expr":
        """
        Return a canonical (simplified) form of this expression.

        Canonicalization performs algebraic simplifications such as:
        - Constant folding (e.g., 2 + 3 → 5)
        - Identity elimination (e.g., x + 0 → x, x * 1 → x)
        - Flattening nested operations (e.g., Add(Add(a, b), c) → Add(a, b, c))
        - Algebraic rewrites (e.g., constraints to standard form)

        Returns:
            Expr: A canonical version of this expression

        Raises:
            NotImplementedError: If canonicalization is not implemented for this node type
        """
        raise NotImplementedError(f"canonicalize() not implemented for {self.__class__.__name__}")

    def check_shape(self) -> Tuple[int, ...]:
        """
        Compute and validate the shape of this expression.

        This method:
        1. Recursively checks shapes of all child expressions
        2. Validates that operations are shape-compatible (e.g., broadcasting rules)
        3. Returns the output shape of this expression

        For example:
        - A Parameter with shape (3, 4) returns (3, 4)
        - MatMul of (3, 4) @ (4, 5) returns (3, 5)
        - Sum of any shape returns () (scalar)
        - Add broadcasts shapes like NumPy

        Returns:
            tuple: The shape of this expression as a tuple of integers.
                   Empty tuple () represents a scalar.

        Raises:
            NotImplementedError: If shape checking is not implemented for this node type
            ValueError: If the expression has invalid shapes (e.g., incompatible dimensions)
        """
        raise NotImplementedError(f"check_shape() not implemented for {self.__class__.__name__}")

    def pretty(self, indent: int = 0) -> str:
        """Generate a pretty-printed string representation of the expression tree.

        Creates an indented, hierarchical view of the expression tree structure,
        useful for debugging and visualization.

        Args:
            indent: Current indentation level (default: 0)

        Returns:
            str: Multi-line string representation of the expression tree

        Example:
            Pretty print an expression:

                expr = (x + y) * z
                print(expr.pretty())
                # Mul
                #   Add
                #     State
                #     State
                #   State
        """
        pad = "  " * indent
        pad = "  " * indent
        lines = [f"{pad}{self.__class__.__name__}"]
        for child in self.children():
            lines.append(child.pretty(indent + 1))
        return "\n".join(lines)

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Contribute this expression's structural identity to a hash.

        This method is used to compute a structural hash of the expression tree
        that is name-invariant (same structure = same hash regardless of variable names).

        The default implementation hashes the class name and recursively hashes all
        children. Subclasses with additional attributes (like Norm.ord, Index.index)
        should override this to include those attributes.

        Args:
            hasher: A hashlib hash object to update
        """
        # Hash the class name to distinguish different node types
        hasher.update(self.__class__.__name__.encode())
        # Recursively hash all children
        for child in self.children():
            child._hash_into(hasher)

    def structural_hash(self) -> bytes:
        """Compute a structural hash of this expression.

        Returns a hash that depends only on the mathematical structure of the
        expression, not on variable names. Two expressions that are structurally
        equivalent (same operations, same variable positions) will have the same hash.

        Returns:
            bytes: SHA-256 digest of the expression structure
        """
        hasher = hashlib.sha256()
        self._hash_into(hasher)
        return hasher.digest()


class Leaf(Expr):
    """
    Base class for leaf nodes (terminal expressions) in the symbolic expression tree.

    Leaf nodes represent named symbolic variables that don't have child expressions.
    This includes Parameters, Variables, States, and Controls.

    Attributes:
        name (str): Name identifier for the leaf node
        _shape (tuple): Shape of the leaf node
    """

    def __init__(self, name: str, shape: tuple = ()):
        """Initialize a Leaf node.

        Args:
            name (str): Name identifier for the leaf node
            shape (tuple): Shape of the leaf node
        """
        super().__init__()
        self.name = name
        self._shape = shape

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the leaf node.

        Returns:
            tuple: Shape of the leaf node
        """
        return self._shape

    def children(self) -> List["Expr"]:
        """Leaf nodes have no children.

        Returns:
            Empty list since leaf nodes are terminal
        """
        return []

    def canonicalize(self) -> "Expr":
        """Leaf nodes are already in canonical form.

        Returns:
            Expr: Returns self since leaf nodes are already canonical
        """
        return self

    def check_shape(self) -> Tuple[int, ...]:
        """Return the shape of this leaf node.

        Returns:
            tuple: The shape of the leaf node
        """
        return self._shape

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Hash leaf node by class name and shape.

        This base implementation hashes the class name and shape. Subclasses
        like Variable and Parameter override this to add their specific
        canonical identifiers (_slice for Variables, value for Parameters).

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(self.__class__.__name__.encode())
        hasher.update(str(self._shape).encode())

    def __repr__(self) -> str:
        """String representation of the leaf node.

        Returns:
            str: A string describing the leaf node
        """
        return f"{self.__class__.__name__}('{self.name}', shape={self.shape})"


class Parameter(Leaf):
    """Parameter that can be changed at runtime without recompilation.

    Parameters are symbolic variables with initial values that can be updated
    through the problem's parameter dictionary. They allow for efficient
    parameter sweeps without needing to recompile the optimization problem.

    Example:
        obs_center = ox.Parameter("obs_center", shape=(3,), value=np.array([1.0, 0.0, 0.0]))
        # Later: problem.parameters["obs_center"] = new_value
    """

    def __init__(
        self,
        name: str,
        shape: tuple = (),
        value: Union[float, int, np.ndarray, None] = None,
    ):
        """Initialize a Parameter node.

        Args:
            name: Name identifier for the parameter
            shape: Shape of the parameter (default: scalar)
            value: Initial value for the parameter (required)
        """
        super().__init__(name, shape)
        if value is None:
            raise ValueError(f"Parameter '{name}' requires an initial value")
        self.value = np.asarray(value, dtype=float)

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Hash Parameter by its shape only (value-invariant).

        Parameters are hashed by shape only, not by value. This allows the same
        compiled solver to be reused across parameter sweeps - only the structure
        matters for compilation, not the actual values.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"Parameter")
        hasher.update(str(self._shape).encode())


def to_expr(x: Union[Expr, float, int, np.ndarray]) -> Expr:
    """Convert a value to an Expr if it is not already one.

    This is a convenience function that wraps numeric values and arrays as Constant
    expressions, while leaving Expr instances unchanged. Used internally by operators
    to ensure operands are proper Expr objects.

    Args:
        x: Value to convert - can be an Expr, numeric scalar, or numpy array

    Returns:
        The input if it's already an Expr, otherwise a Constant wrapping the value
    """
    return x if isinstance(x, Expr) else Constant(np.array(x))


def traverse(expr: Expr, visit: Callable[[Expr], None]):
    """Depth-first traversal of an expression tree.

    Visits each node in the expression tree by applying the visit function to the
    current node, then recursively visiting all children.

    Args:
        expr: Root expression node to start traversal from
        visit: Callback function applied to each node during traversal
    """
    visit(expr)
    for child in expr.children():
        traverse(child, visit)


class Constant(Expr):
    """Constant value expression.

    Represents a constant numeric value in the expression tree. Constants are
    automatically normalized (squeezed) upon construction to ensure consistency.

    Attributes:
        value: The numpy array representing the constant value (squeezed)

    Example:
        Define constants:

            c1 = Constant(5.0)        # Scalar constant
            c2 = Constant([1, 2, 3])  # Vector constant
            c3 = to_expr(10)          # Also creates a Constant
    """

    def __init__(self, value: np.ndarray):
        """Initialize a constant expression.

        Args:
            value: Numeric value or numpy array to wrap as a constant.
                   Will be converted to numpy array and squeezed.
        """
        # Normalize immediately upon construction to ensure consistency
        # This ensures Constant(5.0) and Constant([5.0]) create identical objects
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=float)
        self.value = np.squeeze(value)

    def canonicalize(self) -> "Expr":
        """Constants are already in canonical form.

        Returns:
            Expr: Returns self since constants are already canonical
        """
        return self

    def check_shape(self) -> Tuple[int, ...]:
        """Return the shape of this constant's value.

        Returns:
            tuple: The shape of the constant's numpy array value
        """
        # Verify the invariant: constants should already be squeezed during construction
        original_shape = self.value.shape
        squeezed_shape = np.squeeze(self.value).shape
        if original_shape != squeezed_shape:
            raise ValueError(
                f"Constant not properly normalized: has shape {original_shape} "
                "but should have shape {squeezed_shape}. "
                "Constants should be squeezed during construction."
            )
        return self.value.shape

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Hash constant by its value.

        Constants are hashed by their actual numeric value, ensuring that
        expressions with the same constant values produce the same hash.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"Constant")
        hasher.update(str(self.value.shape).encode())
        hasher.update(self.value.tobytes())

    def __repr__(self) -> str:
        # Show clean representation - always show as Python values, not numpy arrays
        if self.value.ndim == 0:
            # Scalar: show as plain number
            return f"Const({self.value.item()!r})"
        else:
            # Array: show as Python list for readability
            return f"Const({self.value.tolist()!r})"


class NodeReference(Expr):
    """Reference to a variable at a specific trajectory node.

    NodeReference enables inter-node constraints by allowing you to reference
    the value of a state or control variable at a specific discrete time point
    (node) in the trajectory. This is essential for expressing temporal relationships
    such as:

    - Rate limits and smoothness constraints
    - Multi-step dependencies and recurrence relations
    - Constraints coupling specific nodes

    Attributes:
        base: The expression (typically a Leaf like State or Control) being referenced
        node_idx: Trajectory node index (integer, can be negative for end-indexing)

    Example:
        Rate limit across trajectory:

            position = State("pos", shape=(3,))

            # Create rate limit constraints for all nodes
            constraints = [
                (ox.linalg.Norm(position.at(k) - position.at(k-1)) <= 0.1).at([k])
                for k in range(1, N)
            ]

        Multi-step dependency:

            state = State("x", shape=(1,))

            # Fibonacci-like recurrence at each node
            constraints = [
                (state.at(k) == state.at(k-1) + state.at(k-2)).at([k])
                for k in range(2, N)
            ]

        Coupling specific nodes:

            # Constrain distance between nodes 5 and 10
            coupling = (position.at(10) - position.at(5) <= threshold).at([10])

    Performance Note:
        Cross-node constraints use dense Jacobian storage. For details on memory
        usage and performance implications, see LoweredCrossNodeConstraint documentation.

    Note:
        NodeReference is typically created via the `.at(k)` method on expressions
        rather than constructed directly.
    """

    def __init__(self, base: Expr, node_idx: int):
        """Initialize a NodeReference.

        Args:
            base: Expression to reference at a specific node (typically a Leaf)
            node_idx: Absolute trajectory node index (integer)
                     Supports negative indexing (e.g., -1 for last node)

        Raises:
            TypeError: If node_idx is not an integer
        """
        if not isinstance(node_idx, int):
            raise TypeError(f"Node index must be an integer, got {type(node_idx).__name__}")

        self.node_idx = node_idx
        self.base = base

    def children(self) -> List["Expr"]:
        """Return the base expression as the only child.

        Returns:
            list: Single-element list containing the base expression
        """
        return [self.base]

    def canonicalize(self) -> "Expr":
        """Canonicalize by canonicalizing the base expression.

        Returns:
            NodeReference: A new NodeReference with canonicalized base
        """
        canon_base = self.base.canonicalize()
        return NodeReference(canon_base, self.node_idx)

    def check_shape(self) -> Tuple[int, ...]:
        """Return the shape of the base expression.

        NodeReference doesn't change the shape of the underlying expression,
        it just references it at a specific time point.

        Returns:
            tuple: The shape of the base expression
        """
        return self.base.check_shape()

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Hash NodeReference including its node index.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"NodeReference")
        # Hash the node index (signed int)
        hasher.update(struct.pack(">i", self.node_idx))
        # Hash the base expression
        self.base._hash_into(hasher)

    def __repr__(self) -> str:
        """String representation of the NodeReference.

        Returns:
            str: String showing the base expression and node index
        """
        return f"{self.base!r}.at({self.node_idx})"
