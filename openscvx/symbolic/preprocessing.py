"""Validation and preprocessing utilities for symbolic expressions.

This module provides preprocessing and validation functions for symbolic expressions
in trajectory optimization problems. These utilities ensure that expressions are
well-formed and constraints are properly specified before compilation to solvers.

The preprocessing pipeline includes:
    - Shape validation: Ensure all expressions have compatible shapes
    - Variable name validation: Check for unique, non-reserved variable names
    - Constraint validation: Verify constraints appear only at root level
    - Dynamics validation: Check that dynamics match state dimensions
    - Time parameter validation: Validate time configuration
    - Slice assignment: Assign contiguous memory slices to variables

These functions are typically called automatically during problem construction,
but can also be used manually for debugging or custom problem setups.

Example:
    Validating expressions before problem construction::

        import openscvx as ox

        x = ox.State("x", shape=(3,))
        u = ox.Control("u", shape=(2,))

        # Build dynamics and constraints
        dynamics = {
            "x": u  # Will fail validation - dimension mismatch!
        }

        # Validate dimensions before creating problem
        from openscvx.symbolic.preprocessing import validate_dynamics_dict_dimensions

        try:
            validate_dynamics_dict_dimensions(dynamics, [x])
        except ValueError as e:
            print(f"Validation error: {e}")
"""

from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from openscvx.symbolic.expr import (
    CTCS,
    Concat,
    Constant,
    Constraint,
    Control,
    CrossNodeConstraint,
    Expr,
    NodalConstraint,
    State,
    Variable,
    traverse,
)


def validate_shapes(exprs: Union[Expr, list[Expr]]) -> None:
    """Validate shapes for a single expression or list of expressions.

    Args:
        exprs: Single expression or list of expressions to validate

    Raises:
        ValueError: If any expression has invalid shapes
    """
    exprs = exprs if isinstance(exprs, (list, tuple)) else [exprs]
    for e in exprs:
        e.check_shape()  # will raise ValueError if anything's wrong


# TODO: (norrisg) allow `traverse` to take a list of visitors, that way we can combine steps
def validate_variable_names(
    exprs: Iterable[Expr],
    *,
    reserved_prefix: str = "_",
    reserved_names: Set[str] = None,
) -> None:
    """Validate variable names for uniqueness and reserved name conflicts.

    This function ensures that all State and Control variable names are:
    1. Unique across distinct variable instances
    2. Not starting with the reserved prefix (default: "_")
    3. Not colliding with explicitly reserved names

    Args:
        exprs: Iterable of expression trees to scan for variables
        reserved_prefix: Prefix that user variables cannot start with (default: "_")
        reserved_names: Set of explicitly reserved names that cannot be used (default: None)

    Raises:
        ValueError: If any variable name violates uniqueness or reserved name rules

    Example:
            x1 = ox.State("x", shape=(3,))
            x2 = ox.State("x", shape=(2,))  # Same name, different object
            validate_variable_names([x1 + x2])  # Raises ValueError: Duplicate name 'x'

            bad = ox.State("_internal", shape=(2,))
            validate_variable_names([bad])  # Raises ValueError: Reserved prefix '_'
    """
    seen_names = set()
    seen_ids = set()
    reserved = set(reserved_names or ())

    def visitor(node):
        if not isinstance(node, (State, Control)):
            return

        node_id = id(node)
        if node_id in seen_ids:
            # we already checked this exact object
            return

        name = node.name

        # 1) uniqueness across *different* variables
        if name in seen_names:
            raise ValueError(f"Duplicate variable name: {name!r}")

        # 2) no leading underscore
        if name.startswith(reserved_prefix):
            raise ValueError(
                f"Variable name {name!r} is reserved (cannot start with {reserved_prefix!r})"
            )

        # 3) no collision with explicit reserved set
        if name in reserved:
            raise ValueError(f"Variable name {name!r} collides with reserved name")

        seen_names.add(name)
        seen_ids.add(node_id)

    for e in exprs:
        traverse(e, visitor)


def collect_and_assign_slices(
    states: List[State], controls: List[Control], *, start_index: int = 0
) -> Tuple[list[State], list[Control]]:
    """Assign contiguous memory slices to states and controls.

    This function assigns slice objects to states and controls that determine their
    positions in the flat decision variable vector. Variables can have either:
    - Auto-assigned slices: Automatically assigned contiguously based on order
    - Manual slices: User-specified slices that must be contiguous and non-overlapping

    If any variables have manual slices, they must:
    - Start at index 0 (or start_index if specified)
    - Be contiguous and non-overlapping
    - Match the variable's flattened dimension

    Args:
        states: List of State objects in canonical order
        controls: List of Control objects in canonical order
        start_index: Starting index for slice assignment (default: 0)

    Returns:
        Tuple of (states, controls) with slice attributes assigned

    Raises:
        ValueError: If manual slices are invalid (wrong size, overlapping, not starting at 0)

    Example:
            x = ox.State("x", shape=(3,))
            u = ox.Control("u", shape=(2,))
            states, controls = collect_and_assign_slices([x], [u])
            print(x._slice)  # slice(0, 3)
            print(u._slice)  # slice(0, 2)
    """

    def assign(vars_list, start_index):
        # split into manual vs auto
        manual = [v for v in vars_list if v._slice is not None]
        auto = [v for v in vars_list if v._slice is None]

        if manual:
            # 1) shape‐match check
            for v in manual:
                dim = int(np.prod(v.shape))
                sl = v._slice
                if (sl.stop - sl.start) != dim:
                    raise ValueError(
                        f"Manual slice for {v.name!r} is length {sl.stop - sl.start}, "
                        f"but variable has shape {v.shape} (dim {dim})"
                    )
            # sort by the start of their slices
            manual.sort(key=lambda v: v._slice.start)
            # 2a) must start at 0
            if manual[0]._slice.start != start_index:
                raise ValueError("User-defined slices must start at index 0")
            # 2b) check contiguity & no overlaps
            cursor = start_index
            for v in manual:
                sl = v._slice
                dim = int(np.prod(v.shape))
                if sl.start != cursor or sl.stop != cursor + dim:
                    raise ValueError(
                        f"Manual slice for {v.name!r} must be contiguous and non-overlapping"
                    )
                cursor += dim
            offset = cursor
        else:
            offset = start_index

        # 3) auto-assign the rest
        for v in auto:
            dim = int(np.prod(v.shape))
            v._slice = slice(offset, offset + dim)
            offset += dim

    # run separately on states (x) and controls (u)
    assign(states, start_index)
    assign(controls, start_index)

    # Return the collected variables
    return states, controls


def _traverse_with_depth(expr: Expr, visit: Callable[[Expr, int], None], depth: int = 0):
    """Depth-first traversal of an expression tree with depth tracking.

    Internal helper function that extends the standard traverse function to track
    the depth of each node in the tree. Used for constraint validation.

    Args:
        expr: Root expression node to start traversal from
        visit: Callback function applied to each (node, depth) pair during traversal
        depth: Current depth level (default: 0)
    """
    visit(expr, depth)
    for child in expr.children():
        _traverse_with_depth(child, visit, depth + 1)


def validate_constraints_at_root(exprs: Union[Expr, list[Expr]]):
    """Validate that constraints only appear at the root level of expression trees.

    Constraints and constraint wrappers (CTCS, NodalConstraint, CrossNodeConstraint)
    must only appear as top-level expressions, not nested within other expressions.
    However, constraints inside constraint wrappers are allowed (e.g., the constraint
    inside CTCS(x <= 5)).

    This ensures constraints are properly processed during problem compilation and
    prevents ambiguous constraint specifications.

    Args:
        exprs: Single expression or list of expressions to validate

    Raises:
        ValueError: If any constraint or constraint wrapper is found at depth > 0

    Example:
            x = ox.State("x", shape=(3,))
            constraint = x <= 5
            validate_constraints_at_root([constraint])  # OK - constraint at root

            bad_expr = ox.Sum(x <= 5)  # Constraint nested inside Sum
            validate_constraints_at_root([bad_expr])  # Raises ValueError
    """

    # Define constraint wrappers that must also be at root level
    CONSTRAINT_WRAPPERS = (CTCS, NodalConstraint, CrossNodeConstraint)

    # normalize to list
    expr_list = exprs if isinstance(exprs, (list, tuple)) else [exprs]

    for expr in expr_list:

        def visit(node: Expr, depth: int):
            if depth > 0:
                if isinstance(node, CONSTRAINT_WRAPPERS):
                    raise ValueError(
                        f"Nested constraint wrapper found at depth {depth!r}: {node!r}; "
                        "constraint wrappers must only appear as top-level roots"
                    )
                elif isinstance(node, Constraint):
                    raise ValueError(
                        f"Nested Constraint found at depth {depth!r}: {node!r}; "
                        "constraints must only appear as top-level roots"
                    )

            # If this is a constraint wrapper, don't validate its children
            # (we allow constraints inside constraint wrappers)
            if isinstance(node, CONSTRAINT_WRAPPERS):
                return  # Skip traversing children

            # Otherwise, continue traversing children
            for child in node.children():
                visit(child, depth + 1)

        # Start traversal
        visit(expr, 0)


def validate_and_normalize_constraint_nodes(exprs: Union[Expr, list[Expr]], n_nodes: int):
    """Validate and normalize constraint node specifications.

    This function validates and normalizes node specifications for constraint wrappers:

    For NodalConstraint:
        - nodes should be a list of specific node indices: [2, 4, 6, 8]
        - Validates all nodes are within the valid range [0, n_nodes)

    For CTCS (Continuous-Time Constraint Satisfaction) constraints:
        - nodes should be a tuple of (start, end): (0, 10)
        - None is replaced with (0, n_nodes) to apply over entire trajectory
        - Validation ensures tuple has exactly 2 elements and start < end
        - Validates indices are within trajectory bounds

    Args:
        exprs: Single expression or list of expressions to validate
        n_nodes: Total number of nodes in the trajectory

    Raises:
        ValueError: If node specifications are invalid (out of range, malformed, etc.)

    Example:
            x = ox.State("x", shape=(3,))
            constraint = (x <= 5).at([0, 10, 20])  # NodalConstraint
            validate_and_normalize_constraint_nodes([constraint], n_nodes=50)  # OK

            ctcs_constraint = (x <= 5).over((0, 100))  # CTCS
            validate_and_normalize_constraint_nodes([ctcs_constraint], n_nodes=50)
        # Raises ValueError: Range exceeds trajectory length
    """

    # Normalize to list
    expr_list = exprs if isinstance(exprs, (list, tuple)) else [exprs]

    for expr in expr_list:
        if isinstance(expr, CTCS):
            # CTCS constraint validation (already done in __init__, but normalize None)
            if expr.nodes is None:
                expr.nodes = (0, n_nodes)
            elif expr.nodes[0] >= n_nodes or expr.nodes[1] > n_nodes:
                raise ValueError(
                    f"CTCS node range {expr.nodes} exceeds trajectory length {n_nodes}"
                )

        elif isinstance(expr, NodalConstraint):
            # NodalConstraint validation - nodes are already validated in __init__
            # Just need to check they're within trajectory range
            for node in expr.nodes:
                if node < 0 or node >= n_nodes:
                    raise ValueError(f"NodalConstraint node {node} is out of range [0, {n_nodes})")


def validate_cross_node_constraint(
    cross_node_constraint: CrossNodeConstraint, n_nodes: int
) -> None:
    """Validate cross-node constraint bounds and variable consistency.

    This function performs two validations in a single tree traversal:

    1. **Bounds checking**: Ensures all NodeReference indices are within [0, n_nodes).
       Cross-node constraints reference fixed trajectory nodes (e.g., position.at(5)),
       and this validates those indices are valid. Negative indices are normalized
       (e.g., -1 becomes n_nodes-1) before checking.

    2. **Variable consistency**: Ensures that if ANY variable uses .at(), then ALL
       state/control variables must use .at(). Mixing causes shape mismatches during
       lowering because:
       - Variables with .at(k) extract single-node values: X[k, :] → shape (n_x,)
       - Variables without .at() expect full trajectory: X[:, :] → shape (N, n_x)

    Args:
        cross_node_constraint: The CrossNodeConstraint to validate
        n_nodes: Total number of trajectory nodes

    Raises:
        ValueError: If any NodeReference accesses nodes outside [0, n_nodes)
        ValueError: If constraint mixes .at() and non-.at() variables

    Example:
        Valid cross-node constraint:

            from openscvx.symbolic.expr import CrossNodeConstraint

            position = State("pos", shape=(3,))

            # Valid: all variables use .at(), indices in bounds
            constraint = CrossNodeConstraint(position.at(5) - position.at(4) <= 0.1)
            validate_cross_node_constraint(constraint, n_nodes=10)  # OK

        Invalid - out of bounds:

            # Invalid: node 10 is out of bounds for n_nodes=10
            bad_bounds = CrossNodeConstraint(position.at(0) == position.at(10))
            validate_cross_node_constraint(bad_bounds, n_nodes=10)  # Raises ValueError

        Invalid - mixed .at() usage:

            velocity = State("vel", shape=(3,))
            # Invalid: position uses .at(), velocity doesn't
            bad_mixed = CrossNodeConstraint(position.at(5) - velocity <= 0.1)
            validate_cross_node_constraint(bad_mixed, n_nodes=10)  # Raises ValueError
    """
    from openscvx.symbolic.expr import Control, CrossNodeConstraint, NodeReference, State

    if not isinstance(cross_node_constraint, CrossNodeConstraint):
        raise TypeError(
            f"Expected CrossNodeConstraint, got {type(cross_node_constraint).__name__}. "
            f"Bare constraints with NodeReferences should be wrapped in CrossNodeConstraint "
            f"by separate_constraints() before validation."
        )

    constraint = cross_node_constraint.constraint

    # Collect information in a single traversal
    node_refs = []  # List of (node_idx, normalized_idx) tuples
    unwrapped_vars = []  # List of variable names without .at()

    def traverse(expr):
        if isinstance(expr, NodeReference):
            # Normalize negative indices
            idx = expr.node_idx
            normalized_idx = idx if idx >= 0 else n_nodes + idx
            node_refs.append((idx, normalized_idx))
            # Don't traverse into children - NodeReference wraps the variable
            return

        if isinstance(expr, (State, Control)):
            # Found a bare State/Control not wrapped in NodeReference
            unwrapped_vars.append(expr.name)
            return

        # Recurse on children
        for child in expr.children():
            traverse(child)

    # Traverse the constraint expression (both sides)
    traverse(constraint.lhs)
    traverse(constraint.rhs)

    # Check 1: Bounds validation
    for orig_idx, normalized_idx in node_refs:
        if normalized_idx < 0 or normalized_idx >= n_nodes:
            raise ValueError(
                f"Cross-node constraint references invalid node index {orig_idx}. "
                f"Node indices must be in range [0, {n_nodes}) "
                f"(or negative indices in range [-{n_nodes}, -1]). "
                f"Constraint: {constraint}"
            )

    # Check 2: Variable consistency - if we have NodeReferences, all vars must use .at()
    if node_refs and unwrapped_vars:
        raise ValueError(
            f"Cross-node constraint contains NodeReferences (variables with .at(k)) "
            f"but also has variables without .at(): {unwrapped_vars}. "
            f"All state/control variables in cross-node constraints must use .at(k). "
            f"For example, if you use 'position.at(5)', you must also use 'velocity.at(4)' "
            f"instead of just 'velocity'. "
            f"Constraint: {constraint}"
        )


def validate_dynamics_dimension(
    dynamics_expr: Union[Expr, list[Expr]], states: Union[State, list[State]]
) -> None:
    """Validate that dynamics expression dimensions match state dimensions.

    Ensures that the total dimension of all dynamics expressions matches the total
    dimension of all states. Each dynamics expression must be a 1D vector, and their
    combined dimension must equal the sum of all state dimensions.

    This is essential for ensuring the ODE system x_dot = f(x, u, t) is well-formed.

    Args:
        dynamics_expr: Single dynamics expression or list of dynamics expressions.
                      Combined, they represent x_dot = f(x, u, t) for all states.
        states: Single state variable or list of state variables that the dynamics describe.

    Raises:
        ValueError: If dimensions don't match or if any dynamics is not a 1D vector

    Example:
            x = ox.State("x", shape=(3,))
            y = ox.State("y", shape=(2,))
            dynamics = ox.Concat(x * 2, y + 1)  # Shape (5,) - matches total state dim
            validate_dynamics_dimension(dynamics, [x, y])  # OK

            bad_dynamics = x  # Shape (3,) - doesn't match total dim of 5
            validate_dynamics_dimension(bad_dynamics, [x, y])  # Raises ValueError
    """
    # Normalize inputs to lists
    dynamics_list = dynamics_expr if isinstance(dynamics_expr, (list, tuple)) else [dynamics_expr]
    states_list = states if isinstance(states, (list, tuple)) else [states]

    # Calculate total state dimension
    total_state_dim = sum(int(np.prod(state.shape)) for state in states_list)

    # Validate each dynamics expression and calculate total dynamics dimension
    total_dynamics_dim = 0

    for i, dyn_expr in enumerate(dynamics_list):
        # Get the shape of this dynamics expression
        dynamics_shape = dyn_expr.check_shape()

        # Dynamics should be a 1D vector
        if len(dynamics_shape) != 1:
            prefix = f"Dynamics expression {i}" if len(dynamics_list) > 1 else "Dynamics expression"
            raise ValueError(
                f"{prefix} must be 1-dimensional (vector), but got shape {dynamics_shape}"
            )

        total_dynamics_dim += dynamics_shape[0]

    # Check that total dynamics dimension matches total state dimension
    if total_dynamics_dim != total_state_dim:
        if len(dynamics_list) == 1:
            raise ValueError(
                f"Dynamics dimension mismatch: dynamics has dimension {total_dynamics_dim}, "
                f"but total state dimension is {total_state_dim}. "
                f"States: {[(s.name, s.shape) for s in states_list]}"
            )
        else:
            dynamics_dims = [dyn.check_shape()[0] for dyn in dynamics_list]
            raise ValueError(
                f"Dynamics dimension mismatch: {len(dynamics_list)} dynamics expressions "
                f"have combined dimension {total_dynamics_dim} {dynamics_dims}, "
                f"but total state dimension is {total_state_dim}. "
                f"States: {[(s.name, s.shape) for s in states_list]}"
            )


def validate_dynamics_dict(
    dynamics: Dict[str, Expr],
    states: List[State],
    byof_dynamics: Optional[Dict[str, callable]] = None,
) -> None:
    """Validate that dynamics dictionary keys match state names exactly.

    Ensures that the dynamics dictionary (combined with optional byof dynamics) has
    exactly the same keys as the state names, with no missing states, no extra keys,
    and no overlap between symbolic and byof dynamics.

    Args:
        dynamics: Dictionary mapping state names to their dynamics expressions
        states: List of State objects
        byof_dynamics: Optional dictionary mapping state names to raw JAX functions.
            States in byof_dynamics should NOT appear in dynamics dict.

    Raises:
        ValueError: If there's a mismatch between state names and dynamics keys,
            or if a state appears in both dynamics and byof_dynamics.

    Example:
            x = ox.State("x", shape=(3,))
            y = ox.State("y", shape=(2,))
            dynamics = {"x": x * 2, "y": y + 1}
            validate_dynamics_dict(dynamics, [x, y])  # OK

            bad_dynamics = {"x": x * 2}  # Missing "y"
            validate_dynamics_dict(bad_dynamics, [x, y])  # Raises ValueError

            # With byof_dynamics (expert user mode)
            dynamics = {"x": x * 2}  # Only symbolic for x
            byof_dynamics = {"y": some_jax_fn}  # Raw JAX for y
            validate_dynamics_dict(dynamics, [x, y], byof_dynamics)  # OK
    """
    state_names_set = set(state.name for state in states)
    symbolic_keys = set(dynamics.keys())
    byof_keys = set(byof_dynamics.keys()) if byof_dynamics else set()

    # Check for overlap - a state can't be defined in both
    overlap = symbolic_keys & byof_keys
    if overlap:
        raise ValueError(
            f"States defined in both symbolic and byof dynamics: {overlap}\n"
            "Each state must have dynamics in exactly one place."
        )

    # Check coverage - all states must be covered
    covered = symbolic_keys | byof_keys
    missing = state_names_set - covered
    extra = covered - state_names_set

    if missing or extra:
        error_msg = "Mismatch between state names and dynamics keys.\n"
        if missing:
            error_msg += f"  States missing from dynamics: {missing}\n"
        if extra:
            error_msg += f"  Extra keys in dynamics: {extra}\n"
        raise ValueError(error_msg)


def validate_dynamics_dict_dimensions(dynamics: Dict[str, Expr], states: List[State]) -> None:
    """Validate that each dynamics expression matches its corresponding state shape.

    For dictionary-based dynamics specification, ensures that each state's dynamics
    expression has the same shape as the state itself. This validates that each
    component of x_dot = f(x, u, t) has the correct dimension.

    Scalars are normalized to shape (1,) for comparison, matching Concat behavior.

    Args:
        dynamics: Dictionary mapping state names to their dynamics expressions
        states: List of State objects

    Raises:
        ValueError: If any dynamics expression dimension doesn't match its state shape

    Example:
            x = ox.State("x", shape=(3,))
            y = ox.State("y", shape=(2,))
            u = ox.Control("u", shape=(3,))
            dynamics = {"x": u, "y": y + 1}
            validate_dynamics_dict_dimensions(dynamics, [x, y])  # OK

            bad_dynamics = {"x": u, "y": u}  # y dynamics has wrong shape
            validate_dynamics_dict_dimensions(bad_dynamics, [x, y])  # Raises ValueError
    """

    def normalize_scalars(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Normalize shape: scalar () becomes (1,)"""
        return (1,) if len(shape) == 0 else shape

    for state in states:
        dyn_expr = dynamics[state.name]
        expected_shape = state.shape

        # Handle raw Python numbers (which will be converted to Constant later)
        if isinstance(dyn_expr, (int, float)):
            actual_shape = ()  # Scalars have shape ()
        else:
            # Compute the shape of the dynamics expression
            actual_shape = dyn_expr.check_shape()

        # Normalize both shapes for comparison (consistent with Concat behavior)
        if normalize_scalars(actual_shape) != normalize_scalars(expected_shape):
            raise ValueError(
                f"Dynamics for state '{state.name}' has shape {actual_shape}, "
                f"but state has shape {expected_shape}"
            )


def convert_dynamics_dict_to_expr(
    dynamics: Dict[str, Expr], states: List[State]
) -> Tuple[Dict[str, Expr], Expr]:
    """Convert dynamics dictionary to concatenated expression in canonical order.

    Converts a dictionary-based dynamics specification to a single concatenated expression
    that represents the full ODE system x_dot = f(x, u, t). The dynamics are ordered
    according to the states list to ensure consistent variable ordering.

    This function also normalizes scalar values (int, float) to Constant expressions.

    Args:
        dynamics: Dictionary mapping state names to their dynamics expressions
        states: List of State objects defining the canonical order

    Returns:
        Tuple of:
            - Updated dynamics dictionary (with scalars converted to Constant expressions)
            - Concatenated dynamics expression ordered by states list

    Example:
        Convert dynamics dict to a single expression:

            x = ox.State("x", shape=(3,))
            y = ox.State("y", shape=(2,))
            dynamics_dict = {"x": x * 2, "y": 1.0}  # Scalar for y
            converted_dict, concat_expr = convert_dynamics_dict_to_expr(
                dynamics_dict, [x, y]
            )
            # converted_dict["y"] is now Constant(1.0)
            # concat_expr is Concat(x * 2, Constant(1.0))
    """
    # Create a copy to avoid mutating the input
    dynamics_converted = dict(dynamics)

    # Convert scalar values to Constant expressions
    for state_name, dyn_expr in dynamics_converted.items():
        if isinstance(dyn_expr, (int, float)):
            dynamics_converted[state_name] = Constant(dyn_expr)

    # Create concatenated expression ordered by states list
    dynamics_exprs = [dynamics_converted[state.name] for state in states]
    dynamics_concat = Concat(*dynamics_exprs)

    return dynamics_converted, dynamics_concat


def fill_default_guesses(states: List[State], N: int) -> None:
    """Fill in default linspace guesses for states with guess=None.

    For states with both initial and final conditions set, generates a linear
    interpolation from initial to final values.

    This function modifies states in-place.

    Args:
        states: List of State objects to fill guesses for
        N: Number of discretization nodes
    """
    from openscvx.init import linspace

    for state in states:
        if state.guess is None and state.initial is not None and state.final is not None:
            # state.initial and state.final are already numpy arrays of values
            # (the setter handles parsing tuples like ("free", value))
            state.guess = linspace(
                keyframes=[state.initial, state.final],
                nodes=[0, N - 1],
            )


def validate_boundary_conditions(states: List[State]) -> None:
    """Validate that all states have initial and final boundary conditions set.

    Args:
        states: List of State objects to validate

    Raises:
        ValueError: If any state is missing initial or final conditions
    """
    for state in states:
        if state.initial is None:
            raise ValueError(
                f"State '{state.name}' is missing initial condition. "
                f"Please set {state.name}.initial"
            )
        if state.final is None:
            raise ValueError(
                f"State '{state.name}' is missing final condition. Please set {state.name}.final"
            )


def validate_bounds(variables: List[Variable]) -> None:
    """Validate that all variables have min and max bounds set.

    Args:
        variables: List of Variable objects (State or Control) to validate

    Raises:
        ValueError: If any variable is missing min or max bounds
    """
    for var in variables:
        if var.min is None:
            raise ValueError(
                f"Variable '{var.name}' is missing min bound. Please set {var.name}.min"
            )
        if var.max is None:
            raise ValueError(
                f"Variable '{var.name}' is missing max bound. Please set {var.name}.max"
            )


def validate_input_types(
    dynamics: any,
    states: any,
    controls: any,
    constraints: any,
    N: any,
    time: any,
) -> None:
    """Validate that all user-facing inputs have correct types.

    This catches common user errors like passing a single State or Control
    instead of a list, or passing wrong types for dynamics, N, or time.
    Should be called before any other validation in the preprocessing pipeline.

    Raises:
        TypeError: If any input has the wrong type
        ValueError: If N is not positive
    """
    from openscvx.symbolic.expr import CTCS, Constraint, CrossNodeConstraint, NodalConstraint
    from openscvx.symbolic.time import Time

    if not isinstance(dynamics, dict):
        raise TypeError(
            f"'dynamics' must be a dict mapping state names to expressions, "
            f"got {type(dynamics).__name__}"
        )

    if not isinstance(states, list):
        hint = ""
        if isinstance(states, State):
            hint = f" Hint: use states=[{states.name}] instead of states={states.name}"
        raise TypeError(
            f"'states' must be a list of State objects, got {type(states).__name__}.{hint}"
        )

    for i, s in enumerate(states):
        if not isinstance(s, State):
            raise TypeError(f"states[{i}] must be a State, got {type(s).__name__}")

    if not isinstance(controls, list):
        hint = ""
        if isinstance(controls, Control):
            hint = f" Hint: use controls=[{controls.name}] instead of controls={controls.name}"
        raise TypeError(
            f"'controls' must be a list of Control objects, got {type(controls).__name__}.{hint}"
        )

    for i, c in enumerate(controls):
        if not isinstance(c, Control):
            raise TypeError(f"controls[{i}] must be a Control, got {type(c).__name__}")

    if not isinstance(constraints, list):
        raise TypeError(
            f"'constraints' must be a list of Constraint objects, got {type(constraints).__name__}"
        )

    valid_constraint_types = (Constraint, NodalConstraint, CrossNodeConstraint, CTCS)
    for i, c in enumerate(constraints):
        if not isinstance(c, valid_constraint_types):
            raise TypeError(
                f"constraints[{i}] must be a Constraint, NodalConstraint, "
                f"CrossNodeConstraint, or CTCS, got {type(c).__name__}"
            )

    if not isinstance(N, int):
        raise TypeError(f"'N' must be an integer, got {type(N).__name__}")

    if N < 1:
        raise ValueError(f"'N' must be positive, got {N}")

    if not isinstance(time, Time):
        raise TypeError(f"'time' must be a Time object, got {type(time).__name__}")


def validate_propagation_input_types(
    dynamics_prop_extra: any,
    states_prop_extra: any,
) -> None:
    """Validate types for optional propagation inputs.

    These parameters must either both be None or both be provided.
    When provided, dynamics_prop_extra must be a dict and states_prop_extra
    must be a list of State objects.

    Args:
        dynamics_prop_extra: Should be None or a dict mapping state names to expressions
        states_prop_extra: Should be None or a list of State objects

    Raises:
        TypeError: If either input has the wrong type
        ValueError: If only one of the two is provided

    Example:
            distance = ox.State("distance", shape=(1,))

            # Wrong: passing bare State instead of list
            validate_propagation_input_types({"distance": expr}, distance)
            # Raises TypeError: 'states_prop_extra' must be a list ...
    """
    both_none = dynamics_prop_extra is None and states_prop_extra is None
    both_set = dynamics_prop_extra is not None and states_prop_extra is not None

    if not both_none and not both_set:
        provided = "dynamics_prop" if dynamics_prop_extra is not None else "states_prop"
        missing = "states_prop" if dynamics_prop_extra is not None else "dynamics_prop"
        raise ValueError(
            f"'{provided}' was provided but '{missing}' was not. "
            f"Both must be provided together, or both omitted."
        )

    if both_none:
        return

    if not isinstance(dynamics_prop_extra, dict):
        raise TypeError(
            f"'dynamics_prop' must be a dict mapping state names to expressions, "
            f"got {type(dynamics_prop_extra).__name__}"
        )

    if not isinstance(states_prop_extra, list):
        hint = ""
        if isinstance(states_prop_extra, State):
            hint = (
                f" Hint: use states_prop=[{states_prop_extra.name}]"
                f" instead of states_prop={states_prop_extra.name}"
            )
        raise TypeError(
            f"'states_prop' must be a list of State objects, "
            f"got {type(states_prop_extra).__name__}.{hint}"
        )

    for i, s in enumerate(states_prop_extra):
        if not isinstance(s, State):
            raise TypeError(f"states_prop[{i}] must be a State, got {type(s).__name__}")


def validate_guesses(variables: List[Variable]) -> None:
    """Validate that all variables have initial guesses set.

    Args:
        variables: List of Variable objects (State or Control) to validate

    Raises:
        ValueError: If any variable is missing a guess
    """
    for var in variables:
        if var.guess is None:
            if isinstance(var, Control):
                raise ValueError(
                    f"Control '{var.name}' is missing initial guess. "
                    f"Please set {var.name}.guess (controls require explicit guesses)"
                )
            raise ValueError(
                f"State '{var.name}' is missing initial guess. Please set {var.name}.guess"
            )
