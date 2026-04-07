"""Bring-Your-Own-Functions (BYOF) - Expert User Mode.

This module provides type definitions and documentation for expert users who want
to bypass the symbolic layer and directly provide raw JAX functions.

Important:
    The unified state/control vectors include ALL states/controls in the order
    they were provided, plus any augmented states from CTCS constraints. You are
    responsible for correct indexing. Consider inspecting the symbolic problem
    to understand the layout.

Warning:
    **Constraint Sign Convention**: All constraints follow g(x,u) <= 0 convention.
    Return **negative when satisfied**, **positive when violated**.
    Example: for x <= 10 return ``x - 10``, for x >= 5 return ``5 - x``.

Function Signatures:
    All byof functions must be JAX-compatible (use jax.numpy, avoid side effects).

    - dynamics: ``(x, u, node, params) -> xdot_component``
        - x: Full unified state vector (1D array)
        - u: Full unified control vector (1D array)
        - node: Integer node index
        - params: Dict of parameters
        - Returns: State derivative component (array matching state shape)

    - dynamics_discrete: ``(x, u, node, params) -> x_next_component``
        - Same arguments as dynamics
        - Returns: Next-state component for impulsive/discrete update (array matching state shape)
        - Used when controls include impulsive (e.g. delta-V).
        - Can mix with symbolic dynamics_discrete.

    - nodal_constraints: ``(x, u, node, params) -> residual``
        - Same arguments as dynamics
        - Returns: Constraint residual (g <= 0: negative=satisfied, positive=violated)

    - cross_nodal_constraints: ``(X, U, params) -> residual``
        - X: State trajectory (N, n_x) where N is number of trajectory nodes,
            n_x is unified state dimension
        - U: Control trajectory (N, n_u) where N is number of trajectory nodes,
            n_u is unified control dimension
        - params: Dict of parameters
        - Returns: Constraint residual (g <= 0: negative=satisfied, positive=violated)

    - ctcs constraint_fn: ``(x, u, node, params) -> scalar``
        - Same as nodal_constraints but MUST return scalar
        - Returns: Constraint residual (g <= 0: negative=satisfied, positive=violated)

    - ctcs penalty: ``(residual) -> penalty_value``
        - residual: Scalar constraint residual
        - Returns: Non-negative penalty value

Example:
    Basic usage mixing symbolic and byof::

        import jax.numpy as jnp
        import openscvx as ox
        from openscvx import ByofSpec

        # Define states
        position = ox.State("position", shape=(2,))
        velocity = ox.State("velocity", shape=(1,))
        theta = ox.Control("theta", shape=(1,))

        # Unified state: [position[0], position[1], velocity[0], time, augmented...]
        # Unified control: [theta[0], time_dilation]

        # Tip: Use the .slice property on State/Control objects for cleaner,
        # more maintainable indexing instead of hardcoded indices.
        byof: ByofSpec = {
            "nodal_constraints": [
                # Velocity bounds (applied to all nodes)
                {
                    "constraint_fn": lambda x, u, node, params: x[velocity.slice][0] - 10.0,
                },
                {
                    "constraint_fn": lambda x, u, node, params: -x[velocity.slice][0],
                },
                # Velocity must be exactly 0 at start (selective enforcement)
                {
                    "constraint_fn": lambda x, u, node, params: x[velocity.slice][0],
                    "nodes": [0],  # Only at first node
                },
            ],
            "ctcs_constraints": [
                {
                    "constraint_fn": lambda x, u, node, params: x[position.slice][0] - 10.0,
                    "penalty": "square",
                    "bounds": (0.0, 1e-4),
                }
            ],
        }

        problem = ox.Problem(..., byof=byof)
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict

from openscvx.symbolic.expr.parameter import Parameter

if TYPE_CHECKING:
    from jax import Array as JaxArray
else:
    JaxArray = Any

__all__ = ["ByofSpec", "CtcsConstraintSpec", "NodalConstraintSpec", "PenaltyFunction"]


# Type aliases for clarity
DynamicsFunction = Callable[[JaxArray, JaxArray, int, dict], JaxArray]
NodalConstraintFunction = Callable[[JaxArray, JaxArray, int, dict], JaxArray]
CrossNodalConstraintFunction = Callable[[JaxArray, JaxArray, dict], JaxArray]
CtcsConstraintFunction = Callable[[JaxArray, JaxArray, int, dict], float]
PenaltyFunction = Union[Literal["square", "l1", "huber"], Callable[[float], float]]


class NodalConstraintSpec(BaseModel):
    """Specification for nodal constraint with optional node selection.

    Nodal constraints are point-wise constraints evaluated at specific trajectory nodes.
    By default, constraints apply to all nodes, but you can restrict enforcement to
    specific nodes for boundary conditions, waypoints, or computational efficiency.

    Attributes:
        constraint_fn: Constraint function with signature ``(x, u, node, params) -> residual``.
            Follows g(x,u) <= 0 convention (negative = satisfied). Required field.
        nodes: List of integer node indices where constraint is enforced.
            If omitted, applies to all nodes. Negative indices supported (e.g., -1 for last).
            Optional field.

    Example:
        Boundary constraint only at first and last nodes::

            nodal_spec: NodalConstraintSpec = {
                "constraint_fn": lambda x, u, node, params: x[velocity.slice][0],
                "nodes": [0, -1],  # Only at start and end
            }

        Waypoint constraint at middle of trajectory::

            nodal_spec: NodalConstraintSpec = {
                "constraint_fn": lambda x, u, node, params: jnp.linalg.norm(
                    x[position.slice] - jnp.array([5.0, 7.5])
                ) - 0.1,
                "nodes": [N // 2],
            }
    """

    constraint_fn: NodalConstraintFunction
    nodes: Optional[List[int]] = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class CtcsConstraintSpec(BaseModel):
    """Specification for CTCS (Continuous-Time Constraint Satisfaction) constraint.

    CTCS constraints are enforced by augmenting the dynamics with a penalty term that
    accumulates violations over time. Useful for path constraints that must be satisfied
    continuously, not just at discrete nodes.

    Attributes:
        constraint_fn: Function computing constraint residual with signature
            ``(x, u, node, params) -> scalar``. Must return scalar.
            Follows g(x,u) <= 0 convention (negative = satisfied). Required field.
        penalty: Penalty function for positive residuals (violations).
            Built-in options: "square" (max(r,0)^2, default), "l1" (max(r,0)),
            "huber" (Huber loss). Custom: Callable ``(r) -> penalty`` (non-negative,
            differentiable).
        bounds: (min, max) bounds for augmented state accumulating penalties.
            Default: (0.0, 1e-4). Max acts as soft constraint on total violation.
        initial: Initial value for augmented state. Default: bounds[0] (usually 0.0).
        over: Node interval (start, end) where constraint is active. The constraint
            is enforced for nodes in [start, end). If omitted, constraint is active
            over all nodes. Matches symbolic `.over()` method behavior.
        idx: Constraint group index for sharing augmented states (default: 0).
            All CTCS constraints (symbolic and byof) with the same idx share a single
            augmented state. Their penalties are summed together. Use different idx values
            to track different types of violations separately.

    Warning:
        If symbolic CTCS constraints exist with idx values [0, 1, 2], then byof idx **must** either:

        - Match an existing idx (e.g., 0, 1, or 2) to add to that augmented state
        - Be sequential after them (e.g., 3, 4, 5) to create new augmented states

        You cannot use idx values that create gaps (e.g., if symbolic has [0, 1],
        you cannot use byof idx=3 without also using idx=2).

    Example:
        Enforce position[0] <= 10.0 continuously::

            # Assuming position = ox.State("position", shape=(2,))
            ctcs_spec: CtcsConstraintSpec = {
                "constraint_fn": lambda x, u, node, params: x[position.slice][0] - 10.0,
                "penalty": "square",
                "bounds": (0.0, 1e-4),
                "initial": 0.0,
                "idx": 0,  # Groups with other constraints having idx=0
            }

        Enforce constraint only over specific node range::

            ctcs_spec: CtcsConstraintSpec = {
                "constraint_fn": lambda x, u, node, params: x[position.slice][0] - 10.0,
                "over": (10, 50),  # Active only for nodes 10-49
                "penalty": "square",
            }

        Multiple constraints sharing an augmented state::

            # If symbolic CTCS already has idx=[0, 1], then:

            byof = {
                "ctcs_constraints": [
                    # Add to existing symbolic idx=0 augmented state
                    {
                        "constraint_fn": lambda x, u, node, params: x[pos.slice][0] - 10.0,
                        "idx": 0,  # Shares with symbolic idx=0
                    },
                    # Add to existing symbolic idx=1 augmented state
                    {
                        "constraint_fn": lambda x, u, node, params: x[vel.slice][0] - 5.0,
                        "idx": 1,  # Shares with symbolic idx=1
                    },
                    # Create NEW augmented state (sequential after symbolic)
                    {
                        "constraint_fn": lambda x, u, node, params: x[pos.slice][1] - 8.0,
                        "idx": 2,  # New state (symbolic has 0,1, so next is 2)
                    },
                ]
            }
    """

    constraint_fn: CtcsConstraintFunction
    penalty: PenaltyFunction = "square"
    bounds: Tuple[float, float] = (0.0, 1e-4)
    initial: Optional[float] = None
    over: Optional[Tuple[int, int]] = None
    idx: int = 0

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class ByofSpec(BaseModel):
    """Bring-Your-Own-Functions specification for expert users.

    Allows bypassing the symbolic layer and directly providing raw JAX functions.
    All fields are optional - you can mix symbolic and byof as needed.

    Warning:
        You are responsible for:

        - Correct indexing into unified state/control vectors
        - Ensuring functions are JAX-compatible (use jax.numpy, no side effects)
        - Ensuring functions are differentiable
        - Following g(x,u) <= 0 convention for constraints

    Tip:
        Use the ``.slice`` property on State/Control objects for cleaner, more
        maintainable indexing instead of hardcoded indices. For example, use
        ``x[velocity.slice]`` instead of ``x[2:3]``. The slice property is set
        after preprocessing and provides the correct indices into the unified
        state/control vectors.

    Attributes:
        parameters: List of :class:`~openscvx.symbolic.expr.expr.Parameter` objects that
            are only used inside byof functions (not in any symbolic expression). Parameters
            already referenced in symbolic dynamics or constraints are collected automatically
            and do not need to be listed here. Duplicates are ignored.
        dynamics: Raw JAX functions for state derivatives. Maps state names to functions
            with signature ``(x, u, node, params) -> xdot_component``. States here should
            NOT appear in symbolic dynamics dict. You can mix: some states symbolic,
            some in byof.
        dynamics_discrete: Raw JAX functions for discrete/impulsive state updates. Maps state
            names to functions with signature ``(x, u, node, params) -> x_next_component``.
            States here should NOT appear in symbolic dynamics_discrete dict. Use when you need
            custom impulsive updates (e.g. delta-V) alongside or instead of symbolic ones.
        nodal_constraints: Point-wise constraints applied at specific nodes.
            Each item is a :class:`NodalConstraintSpec` dict with:

            - ``func``: Constraint function ``(x, u, node, params) -> residual`` (required)
            - ``nodes``: List of node indices (optional, defaults to all nodes)

            Follows g(x,u) <= 0 convention.
        cross_nodal_constraints: Constraints coupling multiple nodes (smoothness, rate limits).
            Signature: ``(X, U, params) -> residual`` where X is (N, n_x) and U is (N, n_u).
            N is the number of trajectory nodes, n_x is state dimension, n_u is control dimension.
            Follows g(X,U) <= 0 convention.
        ctcs_constraints: Continuous-time constraint satisfaction via dynamics augmentation.
            Each adds an augmented state accumulating violation penalties.
            See :class:`CtcsConstraintSpec` for details.

    Example:
        Custom dynamics and constraints::

            import jax.numpy as jnp
            import openscvx as ox
            from openscvx import ByofSpec

            # Define states and controls
            position = ox.State("position", shape=(2,))
            velocity = ox.State("velocity", shape=(1,))
            theta = ox.Control("theta", shape=(1,))

            # Parameters used by byof functions
            g = ox.Parameter("g", shape=(), value=9.81)

            # Custom dynamics for one state using .slice property
            def custom_velocity_dynamics(x, u, node, params):
                # Use .slice property for clean indexing
                return params["g"] * jnp.cos(u[theta.slice][0])

            byof: ByofSpec = {
                "parameters": [g],
                "dynamics": {
                    "velocity": custom_velocity_dynamics,
                },
                "dynamics_discrete": {
                    # Optional: e.g. "velocity": lambda x, u, node, params: x[vel_sl] + u[dv_sl]
                },
                "nodal_constraints": [
                    # Applied to all nodes (no "nodes" field)
                    {
                        "constraint_fn": lambda x, u, node, params: x[velocity.slice][0] - 10.0,
                    },
                    {
                        "constraint_fn": lambda x, u, node, params: -x[velocity.slice][0],
                    },
                    # Specify nodes for selective enforcement
                    {
                        "constraint_fn": lambda x, u, node, params: x[velocity.slice][0],
                        "nodes": [0],  # Velocity must be exactly 0 at start
                    },
                ],
                "cross_nodal_constraints": [
                    # Constrain total velocity across trajectory: sum(velocities) >= 5
                    # X.shape = (N, n_x), extract velocity column using slice
                    lambda X, U, params: 5.0 - jnp.sum(X[:, velocity.slice]),
                ],
                "ctcs_constraints": [
                    {
                        "constraint_fn": lambda x, u, node, params: x[position.slice][0] - 5.0,
                        "penalty": "square",
                    }
                ],
            }
    """

    parameters: List[Parameter] = []
    dynamics: Dict[str, DynamicsFunction] = {}
    dynamics_discrete: Dict[str, DynamicsFunction] = {}
    nodal_constraints: List[NodalConstraintSpec] = []
    cross_nodal_constraints: List[CrossNodalConstraintFunction] = []
    ctcs_constraints: List[CtcsConstraintSpec] = []

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
