"""Specialized constraint types for trajectory optimization.

This module provides advanced constraint specification mechanisms that extend the
basic Equality and Inequality constraints. These specialized constraint types enable
precise control over when and how constraints are enforced in discretized trajectory
optimization problems.

Key constraint types:
    - **NodalConstraint:** Enforces constraints only at specific discrete time points (nodes) along
    the trajectory. Useful for waypoint constraints, boundary conditions, and reducing computational
    cost by selective enforcement.
    - **CTCS (Continuous-Time Constraint Satisfaction):** Guarantees strict constraint satisfaction
    throughout the entire continuous trajectory, not just at discrete nodes. Works by augmenting the
    state vector with additional states whose dynamics integrate constraint violation penalties.
    Essential for safety-critical applications where inter-node violations could be catastrophic.

Example:
    Nodal constraints for waypoints::

        import openscvx as ox

        x = ox.State("x", shape=(3,))
        target = [10, 5, 0]

        # Enforce position constraint only at specific nodes
        waypoint_constraint = (x == target).at([0, 10, 20])

    Continuous-time constraint for obstacle avoidance::

        obstacle_center = ox.Parameter("obs", shape=(2,), value=[5, 5])
        obstacle_radius = 2.0

        # Distance from obstacle must be > radius for ALL time
        distance = ox.Norm(x[:2] - obstacle_center)
        safety_constraint = (distance >= obstacle_radius).over((0, 100))
"""

import hashlib
import struct
from typing import List, Optional, Tuple, Union

import numpy as np

from .arithmetic import Sub
from .expr import Constant, Expr
from .linalg import Sum


class Constraint(Expr):
    """Abstract base class for optimization constraints.

    Constraints represent relationships between expressions that must be satisfied
    in the optimization problem. This base class provides common functionality for
    both equality and inequality constraints.

    Attributes:
        lhs: Left-hand side expression
        rhs: Right-hand side expression
        is_convex: Flag indicating if the constraint is known to be convex

    Note:
        Constraints are canonicalized to standard form: (lhs - rhs) {op} 0
    """

    def __init__(self, lhs: Expr, rhs: Expr):
        """Initialize a constraint.

        Args:
            lhs: Left-hand side expression
            rhs: Right-hand side expression
        """
        self.lhs = lhs
        self.rhs = rhs
        self.is_convex = False

    def children(self) -> List["Expr"]:
        return [self.lhs, self.rhs]

    def canonicalize(self) -> "Expr":
        """Canonicalize constraint to standard form: (lhs - rhs) {op} 0.

        This works for both Equality and Inequality by using type(self) to
        construct the appropriate subclass type.
        """
        diff = Sub(self.lhs, self.rhs)
        canon_diff = diff.canonicalize()
        new_constraint = type(self)(canon_diff, Constant(np.array(0)))
        new_constraint.is_convex = self.is_convex  # Preserve convex flag
        return new_constraint

    def check_shape(self) -> Tuple[int, ...]:
        """Check that constraint operands are broadcastable and return the shape.

        Returns the broadcasted shape of lhs and rhs, which represents the shape
        of the constraint residual (lhs - rhs). Vector constraints are interpreted
        element-wise.

        Returns:
            tuple: The broadcasted shape of lhs and rhs
        """
        L_shape = self.lhs.check_shape()
        R_shape = self.rhs.check_shape()

        # Figure out their broadcasted shape (or error if incompatible)
        try:
            return np.broadcast_shapes(L_shape, R_shape)
        except ValueError as e:
            constraint_type = type(self).__name__
            raise ValueError(f"{constraint_type} not broadcastable: {L_shape} vs {R_shape}") from e

    def at(self, nodes: Union[list, tuple]) -> "NodalConstraint":
        """Apply this constraint only at specific discrete nodes.

        Args:
            nodes: List of node indices where the constraint should be enforced

        Returns:
            NodalConstraint wrapping this constraint with node specification
        """
        if isinstance(nodes, int):
            nodes = [nodes]
        return NodalConstraint(self, list(nodes))

    def over(
        self,
        interval: tuple[int, int],
        penalty: str = "squared_relu",
        idx: Optional[int] = None,
        check_nodally: bool = False,
    ) -> "CTCS":
        """Apply this constraint over a continuous interval using CTCS.

        Args:
            interval: Tuple of (start, end) node indices for the continuous interval
            penalty: Penalty function type ("squared_relu", "huber", "smooth_relu")
            idx: Optional grouping index for multiple augmented states
            check_nodally: Whether to also enforce this constraint nodally

        Returns:
            CTCS constraint wrapping this constraint with interval specification
        """
        return CTCS(self, penalty=penalty, nodes=interval, idx=idx, check_nodally=check_nodally)

    def convex(self) -> "Constraint":
        """Mark this constraint as convex for CVXPy lowering.

        Returns:
            Self with convex flag set to True (enables method chaining)
        """
        self.is_convex = True
        return self


class Equality(Constraint):
    """Equality constraint for optimization problems.

    Represents an equality constraint: lhs == rhs. Can be created using the ==
    operator on Expr objects.

    Example:
        Define an Equality constraint:

            x = ox.State("x", shape=(3,))
            constraint = x == 0  # Creates Equality(x, Constant(0))
    """

    def __repr__(self) -> str:
        return f"{self.lhs!r} == {self.rhs!r}"


class Inequality(Constraint):
    """Inequality constraint for optimization problems.

    Represents an inequality constraint: lhs <= rhs. Can be created using the <=
    operator on Expr objects.

    Example:
        Define an Inequality constraint:

            x = ox.State("x", shape=(3,))
            constraint = x <= 10  # Creates Inequality(x, Constant(10))
    """

    def __repr__(self) -> str:
        return f"{self.lhs!r} <= {self.rhs!r}"


class NodalConstraint(Expr):
    """Wrapper for constraints enforced only at specific discrete trajectory nodes.

    NodalConstraint allows selective enforcement of constraints at specific time points
    (nodes) in a discretized trajectory, rather than enforcing them at every node.
    This is useful for:

    - Specifying waypoint constraints (e.g., pass through point X at node 10)
    - Boundary conditions at non-standard locations
    - Reducing computational cost by checking constraints less frequently
    - Enforcing periodic constraints (e.g., every 5th node)

    The wrapper maintains clean separation between the constraint's mathematical
    definition and the specification of where it should be applied during optimization.

    Note:
        Bare Constraint objects (without .at() or .over()) are automatically converted
        to NodalConstraints applied at all nodes during preprocessing.

    Attributes:
        constraint: The wrapped Constraint (Equality or Inequality) to enforce
        nodes: List of integer node indices where the constraint is enforced

    Example:
        Enforce position constraint only at nodes 0, 10, and 20:

            x = State("x", shape=(3,))
            target = [10, 5, 0]
            constraint = (x == target).at([0, 10, 20])

        Equivalent using NodalConstraint directly:

            constraint = NodalConstraint(x == target, nodes=[0, 10, 20])

        Periodic constraint enforcement (every 10th node):

            velocity_limit = (vel <= 100).at(list(range(0, 100, 10)))

        Bare constraints are automatically applied at all nodes.
        These are equivalent:

            constraint1 = vel <= 100  # Auto-converted to all nodes
            constraint2 = (vel <= 100).at(list(range(n_nodes)))
    """

    def __init__(self, constraint: Constraint, nodes: list[int]):
        """Initialize a NodalConstraint.

        Args:
            constraint: The Constraint (Equality or Inequality) to enforce at specified nodes
            nodes: List of integer node indices where the constraint should be enforced.
                Automatically converts numpy integers to Python integers.

        Raises:
            TypeError: If constraint is not a Constraint instance
            TypeError: If nodes is not a list
            TypeError: If any node index is not an integer

        Note:
            Bounds checking for cross-node constraints (those containing NodeReference)
            is performed later in the pipeline when N is known, via
            validate_cross_node_constraint_bounds() in preprocessing.py.
        """
        if not isinstance(constraint, Constraint):
            raise TypeError("NodalConstraint must wrap a Constraint")
        if not isinstance(nodes, list):
            raise TypeError("nodes must be a list of integers")

        # Convert numpy integers to Python integers
        converted_nodes = []
        for n in nodes:
            if isinstance(n, np.integer):
                converted_nodes.append(int(n))
            elif isinstance(n, int):
                converted_nodes.append(n)
            else:
                raise TypeError("all node indices must be integers")

        self.constraint = constraint
        self.nodes = converted_nodes

    def children(self) -> List["Expr"]:
        """Return the wrapped constraint as the only child.

        Returns:
            list: Single-element list containing the wrapped constraint
        """
        return [self.constraint]

    def canonicalize(self) -> "Expr":
        """Canonicalize the wrapped constraint while preserving node specification.

        Returns:
            NodalConstraint: A new NodalConstraint with canonicalized inner constraint
        """
        canon_constraint = self.constraint.canonicalize()
        return NodalConstraint(canon_constraint, self.nodes)

    def check_shape(self) -> Tuple[int, ...]:
        """Validate the wrapped constraint's shape.

        NodalConstraint wraps a constraint without changing its computational meaning,
        only specifying where it should be applied. Like all constraints, it produces
        a scalar result.

        Returns:
            tuple: Empty tuple () representing scalar shape
        """
        # Validate the wrapped constraint's shape
        self.constraint.check_shape()

        # NodalConstraint produces a scalar like any constraint
        return ()

    def convex(self) -> "NodalConstraint":
        """Mark the underlying constraint as convex for CVXPy lowering.

        Returns:
            Self with underlying constraint's convex flag set to True (enables method chaining)

        Example:
            Mark a constraint as convex:
                constraint = (x <= 10).at([0, 5, 10]).convex()
        """
        self.constraint.convex()
        return self

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Hash NodalConstraint including its node list.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"NodalConstraint")
        # Hash the nodes list
        for node in self.nodes:
            hasher.update(struct.pack(">i", node))
        hasher.update(b"|")  # Separator to distinguish node counts
        # Hash the wrapped constraint
        self.constraint._hash_into(hasher)

    def __repr__(self) -> str:
        """String representation of the NodalConstraint.

        Returns:
            str: String showing the wrapped constraint and node indices
        """
        return f"NodalConstraint({self.constraint!r}, nodes={self.nodes})"


class CrossNodeConstraint(Expr):
    """A constraint that couples specific trajectory nodes via .at(k) references.

    Unlike NodalConstraint which applies a constraint pattern at multiple nodes
    (via vmapping), CrossNodeConstraint is a single constraint with fixed node
    indices embedded in the expression via NodeReference nodes.

    CrossNodeConstraint is created automatically when a bare Constraint contains
    NodeReference nodes (from .at(k) calls). Users should NOT manually wrap
    cross-node constraints - they are auto-detected during constraint separation.

    **Key differences from NodalConstraint:**

    - **NodalConstraint**: Same constraint evaluated at multiple nodes via vmapping.
      Signature: (x, u, node, params) → scalar, vmapped to (N, n_x) inputs.
    - **CrossNodeConstraint**: Single constraint coupling specific fixed nodes.
      Signature: (X, U, params) → scalar, operates on full trajectory arrays.

    **Lowering:**

    - **Non-convex**: Lowered to JAX with automatic differentiation for SCP linearization
    - **Convex**: Lowered to CVXPy and solved directly by the convex solver

    Attributes:
        constraint: The wrapped Constraint containing NodeReference nodes

    Example:
        Rate limit constraint (auto-detected as CrossNodeConstraint):

            position = State("pos", shape=(3,))

            # This creates a CrossNodeConstraint automatically:
            rate_limit = position.at(5) - position.at(4) <= 0.1

            # Mark as convex if the constraint is convex:
            rate_limit_convex = (position.at(5) - position.at(4) <= 0.1).convex()

        Creating multiple cross-node constraints with a loop:

            constraints = []
            for k in range(1, N):
                # Each iteration creates one CrossNodeConstraint
                rate_limit = position.at(k) - position.at(k-1) <= max_step
                constraints.append(rate_limit)

    Note:
        Do NOT use .at([...]) on cross-node constraints. The nodes are already
        specified via .at(k) inside the expression. Using .at([...]) will raise
        an error during constraint separation.
    """

    def __init__(self, constraint: Constraint):
        """Initialize a CrossNodeConstraint.

        Args:
            constraint: The Constraint containing NodeReference nodes.
                Must contain at least one NodeReference (from .at(k) calls).

        Raises:
            TypeError: If constraint is not a Constraint instance
        """
        if not isinstance(constraint, Constraint):
            raise TypeError("CrossNodeConstraint must wrap a Constraint")

        self.constraint = constraint

    @property
    def is_convex(self) -> bool:
        """Whether the underlying constraint is marked as convex.

        Returns:
            bool: True if the constraint is convex, False otherwise
        """
        return self.constraint.is_convex

    def children(self) -> List["Expr"]:
        """Return the wrapped constraint as the only child.

        Returns:
            Single-element list containing the wrapped constraint
        """
        return [self.constraint]

    def canonicalize(self) -> "Expr":
        """Canonicalize the wrapped constraint.

        Returns:
            CrossNodeConstraint: A new CrossNodeConstraint with canonicalized inner constraint
        """
        canon_constraint = self.constraint.canonicalize()
        return CrossNodeConstraint(canon_constraint)

    def check_shape(self) -> Tuple[int, ...]:
        """Validate the wrapped constraint's shape.

        Returns:
            tuple: Empty tuple () representing scalar shape
        """
        self.constraint.check_shape()
        return ()

    def convex(self) -> "CrossNodeConstraint":
        """Mark the underlying constraint as convex for CVXPy lowering.

        Returns:
            Self with underlying constraint's convex flag set to True
        """
        self.constraint.convex()
        return self

    def __repr__(self) -> str:
        """String representation of the CrossNodeConstraint.

        Returns:
            str: String showing the wrapped constraint
        """
        return f"CrossNodeConstraint({self.constraint!r})"


# CTCS STUFF


class CTCS(Expr):
    """Continuous-Time Constraint Satisfaction using augmented state dynamics.

    CTCS enables strict continuous-time constraint enforcement in discretized trajectory
    optimization by augmenting the state vector with additional states whose dynamics
    are the constraint violation penalties. By constraining these augmented states to remain
    at zero throughout the trajectory, the original constraints are guaranteed to be satisfied
    continuously, not just at discrete nodes.

    **How it works:**

    1. Each constraint (in canonical form: lhs <= 0) is wrapped in a penalty function
    2. Augmented states s_aug_i are added with dynamics: ds_aug_i/dt = sum(penalty_j(lhs_j))
       for all CTCS constraints j in group i
    3. Each augmented state is constrained: s_aug_i(t) = 0 for all t (strictly enforced)
    4. Since s_aug_i integrates the penalties, s_aug_i = 0 implies all penalties in the
       group are zero, which means all constraints in the group are satisfied continuously

    **Grouping and augmented states:**

    - CTCS constraints with the **same node interval** are grouped into a single augmented
      state by default (their penalties are summed)
    - CTCS constraints with **different node intervals** create separate augmented states
    - Using the `idx` parameter explicitly assigns constraints to specific augmented states,
      allowing manual control over grouping
    - Each unique group creates one augmented state named `_ctcs_aug_0`, `_ctcs_aug_1`, etc.

    This is particularly useful for:

    - Path constraints that must hold throughout the entire trajectory (not just at nodes)
    - Obstacle avoidance where constraint violation between nodes could be catastrophic
    - State limits that should be respected continuously (e.g., altitude > 0 for aircraft)
    - Ensuring smooth, feasible trajectories between discretization points

    **Penalty functions** (applied to constraint violations):

    - **squared_relu**: Square(PositivePart(lhs)) - smooth, differentiable (default)
    - **huber**: Huber(PositivePart(lhs)) - less sensitive to outliers than squared
    - **smooth_relu**: SmoothReLU(lhs) - smooth approximation of ReLU

    Attributes:
        constraint: The wrapped Constraint (typically Inequality) to enforce continuously
        penalty: Penalty function type ('squared_relu', 'huber', or 'smooth_relu')
        nodes: Optional (start, end) tuple specifying the interval for enforcement,
            or None to enforce over the entire trajectory
        idx: Optional grouping index for managing multiple augmented states.
            CTCS constraints with the same idx and nodes are grouped together, sharing
            an augmented state. If None, auto-assigned based on node intervals.
        check_nodally: Whether to also enforce the constraint at discrete nodes for
            additional numerical robustness (creates both continuous and nodal constraints)

    Example:
        Single augmented state (default behavior - same node interval):

            altitude = State("alt", shape=(1,))
            constraints = [
                (altitude >= 10).over((0, 10)),  # Both constraints share
                (altitude <= 1000).over((0, 10))  # one augmented state
            ]

        Multiple augmented states (different node intervals):

            constraints = [
                (altitude >= 10).over((0, 5)),  # Creates _ctcs_aug_0
                (altitude >= 20).over((5, 10))  # Creates _ctcs_aug_1
            ]

        Manual grouping with idx parameter:

            constraints = [
                (altitude >= 10).over((0, 10), idx=0),    # Group 0
                (velocity <= 100).over((0, 10), idx=1),   # Group 1 (separate state)
                (altitude <= 1000).over((0, 10), idx=0)   # Also group 0
            ]
    """

    def __init__(
        self,
        constraint: Constraint,
        penalty: str = "squared_relu",
        nodes: Optional[Tuple[int, int]] = None,
        idx: Optional[int] = None,
        check_nodally: bool = False,
    ):
        """Initialize a CTCS constraint.

        Args:
            constraint: The Constraint to enforce continuously (typically an Inequality)
            penalty: Penalty function type. Options:
                - 'squared_relu': Square(PositivePart(lhs)) - default, smooth, differentiable
                - 'huber': Huber(PositivePart(lhs)) - robust to outliers
                - 'smooth_relu': SmoothReLU(lhs) - smooth ReLU approximation
            nodes: Optional (start, end) tuple of node indices defining the enforcement interval.
                None means enforce over the entire trajectory. Must satisfy start < end.
                CTCS constraints with the same nodes are automatically grouped together.
            idx: Optional grouping index for multiple augmented states. Allows organizing
                multiple CTCS constraints with separate augmented state variables.
                If None, constraints are auto-grouped by their node intervals.
                Explicitly setting idx allows manual control over which constraints
                share an augmented state.
            check_nodally: If True, also enforce the constraint at discrete nodes for
                numerical stability (creates both continuous and nodal constraints).
                Defaults to False.

        Raises:
            TypeError: If constraint is not a Constraint instance
            ValueError: If nodes is not None or a 2-tuple of integers
            ValueError: If nodes[0] >= nodes[1] (invalid interval)
        """
        if not isinstance(constraint, Constraint):
            raise TypeError("CTCS must wrap a Constraint")

        # Validate nodes parameter for CTCS
        if nodes is not None:
            if not isinstance(nodes, tuple) or len(nodes) != 2:
                raise ValueError(
                    "CTCS constraints must specify nodes as a tuple of (start, end) or None "
                    "for all nodes"
                )
            if not all(isinstance(n, int) for n in nodes):
                raise ValueError("CTCS node indices must be integers")
            if nodes[0] >= nodes[1]:
                raise ValueError("CTCS node range must have start < end")

        self.constraint = constraint
        self.penalty = penalty
        self.nodes = nodes  # (start, end) node range or None for all nodes
        self.idx = idx  # Optional grouping index for multiple augmented states
        # Whether to also enforce this constraint nodally for numerical stability
        self.check_nodally = check_nodally

    def children(self) -> List[Expr]:
        """Return the wrapped constraint as the only child.

        Returns:
            list: Single-element list containing the wrapped constraint
        """
        return [self.constraint]

    def canonicalize(self) -> "Expr":
        """Canonicalize the inner constraint while preserving CTCS parameters.

        Returns:
            CTCS: A new CTCS with canonicalized inner constraint and same parameters
        """
        canon_constraint = self.constraint.canonicalize()
        return CTCS(
            canon_constraint,
            penalty=self.penalty,
            nodes=self.nodes,
            idx=self.idx,
            check_nodally=self.check_nodally,
        )

    def check_shape(self) -> Tuple[int, ...]:
        """Validate the constraint and penalty expression shapes.

        CTCS transforms the wrapped constraint into a penalty expression that is
        summed (integrated) over the trajectory, always producing a scalar result.

        Returns:
            tuple: Empty tuple () representing scalar shape

        Raises:
            ValueError: If the wrapped constraint has invalid shape
            ValueError: If the generated penalty expression is not scalar
        """
        # First validate the wrapped constraint's shape
        self.constraint.check_shape()

        # Also validate the penalty expression that would be generated
        try:
            penalty_expr = self.penalty_expr()
            penalty_shape = penalty_expr.check_shape()

            # The penalty expression should always be scalar due to Sum wrapper
            if penalty_shape != ():
                raise ValueError(
                    f"CTCS penalty expression should be scalar, but got shape {penalty_shape}"
                )
        except Exception as e:
            # Re-raise with more context about which CTCS node failed
            raise ValueError(f"CTCS penalty expression validation failed: {e}") from e

        # CTCS always produces a scalar due to the Sum in penalty_expr
        return ()

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Hash CTCS including all its parameters.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"CTCS")
        # Hash penalty type
        hasher.update(self.penalty.encode())
        # Hash nodes interval
        if self.nodes is not None:
            hasher.update(struct.pack(">ii", self.nodes[0], self.nodes[1]))
        else:
            hasher.update(b"None")
        # Hash idx
        if self.idx is not None:
            hasher.update(struct.pack(">i", self.idx))
        else:
            hasher.update(b"None")
        # Hash check_nodally
        hasher.update(b"1" if self.check_nodally else b"0")
        # Hash the wrapped constraint
        self.constraint._hash_into(hasher)

    def over(self, interval: tuple[int, int]) -> "CTCS":
        """Set or update the continuous interval for this CTCS constraint.

        Args:
            interval: Tuple of (start, end) node indices defining the enforcement interval

        Returns:
            CTCS: New CTCS constraint with the specified interval

        Example:
            Define constraint over range:

                constraint = (altitude >= 10).over((0, 50))

            Update interval to cover different range:

                constraint_updated = constraint.over((50, 100))
        """
        return CTCS(
            self.constraint,
            penalty=self.penalty,
            nodes=interval,
            idx=self.idx,
            check_nodally=self.check_nodally,
        )

    def __repr__(self) -> str:
        """String representation of the CTCS constraint.

        Returns:
            str: String showing constraint, penalty type, and optional parameters
        """
        parts = [f"{self.constraint!r}", f"penalty={self.penalty!r}"]
        if self.nodes is not None:
            parts.append(f"nodes={self.nodes}")
        if self.idx is not None:
            parts.append(f"idx={self.idx}")
        if self.check_nodally:
            parts.append(f"check_nodally={self.check_nodally}")
        return f"CTCS({', '.join(parts)})"

    def penalty_expr(self) -> Expr:
        """Build the penalty expression for this CTCS constraint.

        Transforms the constraint's left-hand side (in canonical form: lhs <= 0)
        into a penalty expression using the specified penalty function. The penalty
        is zero when the constraint is satisfied and positive when violated.

        This penalty expression becomes part of the dynamics of an augmented state.
        Multiple CTCS constraints in the same group (same idx) have their penalties
        summed: ds_aug_i/dt = sum(penalty_j) for all j in group i. By constraining
        s_aug_i(t) = 0 for all t, we ensure all penalties in the group are zero,
        which strictly enforces all constraints in the group continuously.

        Returns:
            Expr: Sum of the penalty function applied to the constraint violation

        Raises:
            ValueError: If an unknown penalty type is specified

        Note:
            This method is used internally during problem compilation to create
            augmented state dynamics. Multiple penalty expressions with the same
            idx are summed together before being added to the dynamics vector via Concat.
        """
        lhs = self.constraint.lhs

        if self.penalty == "squared_relu":
            from openscvx.symbolic.expr.math import PositivePart, Square

            penalty = Square(PositivePart(lhs))
        elif self.penalty == "huber":
            from openscvx.symbolic.expr.math import Huber, PositivePart

            penalty = Huber(PositivePart(lhs))
        elif self.penalty == "smooth_relu":
            from openscvx.symbolic.expr.math import SmoothReLU

            penalty = SmoothReLU(lhs)
        else:
            raise ValueError(f"Unknown penalty {self.penalty!r}")

        return Sum(penalty)


def ctcs(
    constraint: Constraint,
    penalty: str = "squared_relu",
    nodes: Optional[Tuple[int, int]] = None,
    idx: Optional[int] = None,
    check_nodally: bool = False,
) -> CTCS:
    """Helper function to create CTCS (Continuous-Time Constraint Satisfaction) constraints.

    This is a convenience function that creates a CTCS constraint with the same
    parameters as the CTCS constructor. Useful for functional-style constraint building.

    Args:
        constraint: The Constraint to enforce continuously
        penalty: Penalty function type ('squared_relu', 'huber', or 'smooth_relu').
            Defaults to 'squared_relu'.
        nodes: Optional (start, end) tuple of node indices for enforcement interval.
            None enforces over entire trajectory.
        idx: Optional grouping index for multiple augmented states
        check_nodally: Whether to also enforce constraint at discrete nodes.
            Defaults to False.

    Returns:
        CTCS: A CTCS constraint wrapping the input constraint

    Example:
        Using the helper function:

            from openscvx.symbolic.expr.constraint import ctcs
            altitude_constraint = ctcs(
                altitude >= 10,
                penalty="huber",
                nodes=(0, 100),
                check_nodally=True
            )

        Equivalent to using CTCS constructor:

            altitude_constraint = CTCS(altitude >= 10, penalty="huber", nodes=(0, 100))

        Also equivalent to using .over() method on constraint:

            altitude_constraint = (altitude >= 10).over((0, 100), penalty="huber")
    """
    return CTCS(constraint, penalty, nodes, idx, check_nodally)
