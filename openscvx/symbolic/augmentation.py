"""State and dynamics augmentation for continuous-time constraint satisfaction.

This module provides utilities for augmenting trajectory optimization problems with
additional states and dynamics to handle continuous-time constraint satisfaction (CTCS).
The CTCS method enforces path constraints continuously along the trajectory rather than
just at discretization nodes.

Key functionality:
    - CTCS constraint grouping: Sort and group CTCS constraints by time intervals
    - Constraint separation: Separate CTCS, nodal, and convex constraints
    - Vector decomposition: Decompose vector constraints into scalar components
    - Time augmentation: Add time state with appropriate dynamics and constraints
    - CTCS dynamics augmentation: Add augmented states and time dilation control

The augmentation process transforms the original dynamics x_dot = f(x, u) into an
augmented system with additional states for constraint satisfaction and time dilation.

Architecture:
    The CTCS method works by:

    1. Grouping constraints by time interval and assigning index (idx)
    2. Creating augmented states (one per constraint group)
    3. Adding penalty dynamics: aug_dot = penalty(constraint_violation)
    4. Adding time dilation control to slow down near constraint boundaries

Example:
    Augmenting dynamics with CTCS constraints::

        import openscvx as ox

        # Define problem
        x = ox.State("x", shape=(3,))
        u = ox.Control("u", shape=(2,))

        # Create dynamics
        xdot = u @ A  # Some dynamics expression

        # Define path constraint
        path_constraint = (ox.Norm(x) <= 1.0).over((0, 50))  # CTCS constraint

        # Augment dynamics with CTCS
        from openscvx.symbolic.augmentation import augment_dynamics_with_ctcs

        xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
            xdot=xdot,
            states=[x],
            controls=[u],
            constraints_ctcs=[path_constraint],
            N=50
        )
        # xdot_aug now includes augmented state dynamics
        # states_aug includes original states + augmented states
        # controls_aug includes original controls + time dilation
"""

from typing import Dict, List, Tuple

import numpy as np

from openscvx.symbolic.constraint_set import ConstraintSet
from openscvx.symbolic.expr import (
    CTCS,
    Add,
    Concat,
    Constraint,
    CrossNodeConstraint,
    Expr,
    Index,
    NodalConstraint,
)
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State


def sort_ctcs_constraints(
    constraints_ctcs: List[CTCS],
) -> Tuple[List[CTCS], List[Tuple[int, int]], int]:
    """Sort and group CTCS constraints by time interval and assign indices.

    Groups CTCS constraints by their time intervals (nodes) and assigns a unique
    index (idx) to each group. Constraints with the same time interval can share
    an augmented state (same idx), while constraints with different intervals must
    have different augmented states.

    Grouping rules:
        - Constraints with the same node interval can share an idx
        - Constraints with different node intervals must have different idx values
        - idx values must form a contiguous block starting from 0
        - Unspecified idx values are automatically assigned
        - User-specified idx values are validated for consistency

    Args:
        constraints_ctcs: List of CTCS constraints to sort and group

    Returns:
        Tuple of:
            - List of CTCS constraints with idx assigned to each
            - List of node intervals (start, end) in ascending idx order
            - Number of augmented states needed (number of unique idx values)

    Raises:
        ValueError: If user-specified idx values are inconsistent or non-contiguous

    Example:
        Sort CTCS constraints by interval and index:

            constraint1 = (x <= 5).over((0, 50))  # Auto-assigned idx
            constraint2 = (y <= 10).over((0, 50))  # Same interval, same idx
            constraint3 = (z <= 15).over((20, 80))  # Different interval, different idx
            sorted_ctcs, intervals, n_aug = sort_ctcs_constraints([c1, c2, c3])
            # constraint1.idx = 0, constraint2.idx = 0, constraint3.idx = 1
            # intervals = [(0, 50), (20, 80)]
            # n_aug = 2
    """
    idx_to_nodes: Dict[int, Tuple[int, int]] = {}
    next_idx = 0

    for c in constraints_ctcs:
        key = c.nodes

        if c.idx is not None:
            # User supplied an identifier: ensure it always points to the same interval
            if c.idx in idx_to_nodes:
                if idx_to_nodes[c.idx] != key:
                    raise ValueError(
                        f"idx={c.idx} was first used with interval={idx_to_nodes[c.idx]}, "
                        f"but now you gave it interval={key}"
                    )
            else:
                # When idx is explicitly provided, always create a separate group
                # even if nodes are the same - this allows multiple constraint groups
                # with the same node interval but different idx values
                idx_to_nodes[c.idx] = key
        else:
            # No identifier: see if this interval already has one
            for existing_id, nodes in idx_to_nodes.items():
                if nodes == key:
                    c.idx = existing_id
                    break
            else:
                # Brand-new interval: pick the next free auto-id
                while next_idx in idx_to_nodes:
                    next_idx += 1
                c.idx = next_idx
                idx_to_nodes[next_idx] = key
                next_idx += 1

    # Validate that idx values form a contiguous block starting from 0
    ordered_ids = sorted(idx_to_nodes.keys())
    expected_ids = list(range(len(ordered_ids)))
    if ordered_ids != expected_ids:
        raise ValueError(
            f"CTCS constraint idx values must form a contiguous block starting from 0. "
            f"Got {ordered_ids}, expected {expected_ids}"
        )

    # Extract intervals in ascending idx order
    node_intervals = [idx_to_nodes[i] for i in ordered_ids]
    num_augmented_states = len(ordered_ids)

    return constraints_ctcs, node_intervals, num_augmented_states


def separate_constraints(constraint_set: ConstraintSet, n_nodes: int) -> ConstraintSet:
    """Separate and categorize constraints by type and convexity.

    Moves constraints from `constraint_set.unsorted` into their appropriate
    category fields (ctcs, nodal, nodal_convex, cross_node, cross_node_convex).

    Bare Constraint objects are automatically categorized:
    - If they contain NodeReferences (from .at(k) calls), they become CrossNodeConstraint
    - Otherwise, they become NodalConstraint applied at all nodes

    Constraints within CTCS wrappers that have check_nodally=True are also extracted
    and added to the nodal constraint lists.

    Args:
        constraint_set: ConstraintSet with raw constraints in `unsorted` field
        n_nodes: Total number of nodes in the trajectory

    Returns:
        The same ConstraintSet with `unsorted` drained and categories populated

    Raises:
        ValueError: If a constraint is not one of the expected types
        ValueError: If a NodalConstraint contains NodeReferences (use bare Constraint instead)
        ValueError: If a CTCS constraint contains NodeReferences

    Example:
        Separate and categorize constraints::

            x = ox.State("x", shape=(3,))
            constraint_set = ConstraintSet(unsorted=[
                (x <= 5).over((0, 50)),           # CTCS
                (x >= 0).at([0, 10, 20]),         # NodalConstraint
                ox.Norm(x) <= 1,                  # Bare -> all nodes
                x.at(5) - x.at(4) <= 0.1,         # Bare with NodeRef -> cross-node
            ])
            separate_constraints(constraint_set, n_nodes=50)
            assert constraint_set.is_categorized
            # Access via: constraint_set.ctcs, constraint_set.nodal, etc.
    """
    from openscvx.symbolic.lower import _contains_node_reference

    # Process all constraints from unsorted
    for c in constraint_set.unsorted:
        if isinstance(c, CTCS):
            # Validate that CTCS constraints don't contain NodeReferences
            if _contains_node_reference(c.constraint):
                raise ValueError(
                    "CTCS constraints cannot contain NodeReferences (.at(k)). "
                    "Cross-node constraints should be specified as bare Constraint objects. "
                    f"Constraint: {c.constraint}"
                )
            # Normalize None to full horizon
            c.nodes = c.nodes or (0, n_nodes)
            constraint_set.ctcs.append(c)

        elif isinstance(c, NodalConstraint):
            # NodalConstraint means user explicitly called .at([...])
            # Cross-node constraints should NOT use .at([...]) wrapper
            if _contains_node_reference(c.constraint):
                raise ValueError(
                    f"Cross-node constraints should not use .at([...]) wrapper. "
                    f"The constraint already references specific nodes via .at(k) inside the "
                    f"expression. Remove the outer .at([...]) wrapper and use the bare "
                    f"constraint directly. "
                    f"Constraint: {c.constraint}"
                )

            # Regular nodal constraint - categorize by convexity
            if c.constraint.is_convex:
                constraint_set.nodal_convex.append(c)
            else:
                constraint_set.nodal.append(c)

        elif isinstance(c, Constraint):
            # Bare constraint - check if it's a cross-node constraint
            if _contains_node_reference(c):
                # Cross-node constraint: wrap in CrossNodeConstraint
                cross_node = CrossNodeConstraint(c)
                if c.is_convex:
                    constraint_set.cross_node_convex.append(cross_node)
                else:
                    constraint_set.cross_node.append(cross_node)
            else:
                # Regular constraint: apply at all nodes
                all_nodes = list(range(n_nodes))
                nodal_constraint = NodalConstraint(c, all_nodes)
                if c.is_convex:
                    constraint_set.nodal_convex.append(nodal_constraint)
                else:
                    constraint_set.nodal.append(nodal_constraint)

        else:
            raise ValueError(
                "Constraints must be `Constraint`, `NodalConstraint`, or `CTCS`, "
                f"got {type(c).__name__}"
            )

    # Clear unsorted now that all have been categorized
    constraint_set.unsorted = []

    # Add nodal constraints from CTCS constraints that have check_nodally=True
    ctcs_nodal_constraints = get_nodal_constraints_from_ctcs(constraint_set.ctcs)
    for constraint, interval in ctcs_nodal_constraints:
        # CTCS check_nodally constraints cannot have NodeReferences (validated above)
        # Convert CTCS interval (start, end) to list of nodes [start, start+1, ..., end-1]
        interval_nodes = list(range(interval[0], interval[1]))
        nodal_constraint = NodalConstraint(constraint, interval_nodes)

        if constraint.is_convex:
            constraint_set.nodal_convex.append(nodal_constraint)
        else:
            constraint_set.nodal.append(nodal_constraint)

    # Validate cross-node constraints (bounds and variable consistency)
    from openscvx.symbolic.preprocessing import validate_cross_node_constraint

    for cross_node_constraint in constraint_set.cross_node + constraint_set.cross_node_convex:
        validate_cross_node_constraint(cross_node_constraint, n_nodes)

    return constraint_set


def decompose_vector_nodal_constraints(
    constraints_nodal: List[NodalConstraint],
) -> List[NodalConstraint]:
    """Decompose vector-valued nodal constraints into scalar constraints.

    Decomposes vector constraints into individual scalar constraints, which is necessary
    for nonconvex nodal constraints that are lowered to JAX functions. The JAX-to-CVXPY
    interface expects scalar constraint values at each node.

    For example, a constraint with shape (3,) is decomposed into 3 separate scalar
    constraints using indexing. CTCS constraints don't need decomposition since they
    handle vector values internally.

    Args:
        constraints_nodal (List[NodalConstraint]): List of NodalConstraint objects
            (must be canonicalized)

    Returns:
        List of NodalConstraint objects with vector constraints decomposed into scalars.
        Scalar constraints are passed through unchanged.

    Note:
        Constraints are assumed to be in canonical form: residual <= 0 or residual == 0,
        where residual is the lhs of the constraint.

    Example:
        Decompose vector constraint into 3 constraints:

            x = ox.State("x", shape=(3,))
            constraint = (x <= 5).at([0, 10, 20])  # Vector constraint, shape (3,)
            decomposed = decompose_vector_nodal_constraints([constraint])
            # Returns 3 constraints: x[0] <= 5, x[1] <= 5, x[2] <= 5
    """
    decomposed_constraints = []

    for nodal_constraint in constraints_nodal:
        constraint = nodal_constraint.constraint
        nodes = nodal_constraint.nodes

        try:
            # Get the shape of the constraint residual
            # Canonicalized constraints are in form: residual <= 0 or residual == 0
            residual_shape = constraint.lhs.check_shape()

            # Check if this is a vector constraint
            # Decompose ALL vector-shaped constraints (including shape=(1,)) to avoid
            # vmap adding an extra dimension when stacking results
            if len(residual_shape) > 0:
                # Vector constraint - decompose into scalar constraints
                total_elements = int(np.prod(residual_shape))

                for i in range(total_elements):
                    # Create indexed version: residual[i] <= 0 or residual[i] == 0
                    indexed_lhs = Index(constraint.lhs, i)
                    indexed_rhs = constraint.rhs  # Should be Constant(0)
                    indexed_constraint = constraint.__class__(indexed_lhs, indexed_rhs)
                    decomposed_constraints.append(NodalConstraint(indexed_constraint, nodes))
            else:
                # Scalar constraint - keep as is
                decomposed_constraints.append(nodal_constraint)

        except Exception:
            # If shape analysis fails, keep original constraint for backward compatibility
            decomposed_constraints.append(nodal_constraint)

    return decomposed_constraints


def get_nodal_constraints_from_ctcs(
    constraints_ctcs: List[CTCS],
) -> List[tuple[Constraint, tuple[int, int]]]:
    """Extract constraints from CTCS wrappers that should be checked nodally.

    Some CTCS constraints have the check_nodally flag set, indicating that the
    underlying constraint should be enforced both continuously (via CTCS) and
    discretely at the nodes. This function extracts those underlying constraints
    along with their node intervals.

    Args:
        constraints_ctcs: List of CTCS constraint wrappers

    Returns:
        List of tuples (constraint, nodes) where:
            - constraint: The underlying Constraint object from CTCS with check_nodally=True
            - nodes: The (start, end) interval from the CTCS wrapper

    Example:
        Extract CTCS constraint that should also be checked at nodes:

            x = ox.State("x", shape=(3,))
            constraint = (x <= 5).over((10, 50), check_nodally=True)
            nodal = get_nodal_constraints_from_ctcs([constraint])

        Returns [(x <= 5, (10, 50))] to be enforced at nodes 10 through 49
    """
    nodal_ctcs = []
    for ctcs in constraints_ctcs:
        if ctcs.check_nodally:
            nodal_ctcs.append((ctcs.constraint, ctcs.nodes))
    return nodal_ctcs


def augment_dynamics_with_ctcs(
    xdot: Expr,
    states: List[State],
    controls: List[Control],
    constraints_ctcs: List[CTCS],
    N: int,
    licq_min: float = 0.0,
    licq_max: float = 1e-4,
    time_dilation_factor_min: float = 0.3,
    time_dilation_factor_max: float = 3.0,
) -> Tuple[Expr, List[State], List[Control]]:
    """Augment dynamics with continuous-time constraint satisfaction states.

    Implements the CTCS method by adding augmented states and time dilation control
    to the original dynamics. For each group of CTCS constraints, an augmented state
    is created whose dynamics are the penalty function of constraint violations.

    The CTCS method enforces path constraints continuously by:
    1. Creating augmented states with dynamics = penalty(constraint_violation)
    2. Constraining augmented states to stay near zero (LICQ condition)
    3. Adding time dilation control to slow down near constraint boundaries

    The augmented dynamics become:
        x_dot = f(x, u)
        aug_dot = penalty(g(x, u))  # For each constraint group
        time_dot = time_dilation

    Args:
        xdot: Original dynamics expression for states
        states: List of state variables (must include a state named "time")
        controls: List of control variables
        constraints_ctcs: List of CTCS constraints (should be sorted and grouped)
        N: Number of discretization nodes
        licq_min: Minimum bound for augmented states (default: 0.0)
        licq_max: Maximum bound for augmented states (default: 1e-4)
        time_dilation_factor_min: Minimum time dilation factor (default: 0.3)
        time_dilation_factor_max: Maximum time dilation factor (default: 3.0)

    Returns:
        Tuple of:
            - Augmented dynamics expression (original + augmented state dynamics)
            - Updated states list (original + augmented states)
            - Updated controls list (original + time dilation control)

    Raises:
        ValueError: If no state named "time" is found in the states list

    Example:
        Augment dynamics with CTCS penalty states:

            x = ox.State("x", shape=(3,))
            u = ox.Control("u", shape=(2,))
            time = ox.State("time", shape=(1,))
            xdot = u @ A  # Some dynamics
            constraint = (ox.Norm(x) <= 1.0).over((0, 50))
            xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
                xdot=xdot,
                states=[x, time],
                controls=[u],
                constraints_ctcs=[constraint],
                N=50
            )

        states_aug includes x, time, and _ctcs_aug_0,
        controls_aug includes u and _time_dilation
    """
    # Copy the original states and controls lists
    states_augmented = list(states)
    controls_augmented = list(controls)

    if constraints_ctcs:
        # Group penalty expressions by idx (constraints should already be sorted)
        penalty_groups: Dict[int, List[Expr]] = {}

        for ctcs in constraints_ctcs:
            # Keep the CTCS wrapper intact to preserve node interval information
            # The JAX lowerer's visit_ctcs() method will handle the conditional logic

            # TODO: In the future, apply scaling here if ctcs has a scaling attribute
            # if hasattr(ctcs, 'scaling') and ctcs.scaling != 1.0:
            #     ctcs = scale_ctcs(ctcs, scaling_factor)

            if ctcs.idx not in penalty_groups:
                penalty_groups[ctcs.idx] = []
            penalty_groups[ctcs.idx].append(ctcs)

        # Create augmented state expressions for each group
        augmented_state_exprs = []
        for idx in sorted(penalty_groups.keys()):
            penalty_terms = penalty_groups[idx]
            if len(penalty_terms) == 1:
                augmented_state_expr = penalty_terms[0]
            else:
                augmented_state_expr = Add(*penalty_terms)
            augmented_state_exprs.append(augmented_state_expr)

        # Calculate number of augmented states from the penalty groups
        num_augmented_states = len(penalty_groups)

        # Create augmented state variables
        for idx in range(num_augmented_states):
            aug_var = State(f"_ctcs_aug_{idx}", shape=(1,))
            aug_var.initial = np.array([licq_min])  # Set initial to respect bounds
            aug_var.final = [("free", 0)]
            aug_var.min = np.array([licq_min])
            aug_var.max = np.array([licq_max])
            # Set guess to licq_min as well
            aug_var.guess = np.full([N, 1], licq_min)  # N x num augmented states
            states_augmented.append(aug_var)

        # Concatenate with original dynamics
        xdot_aug = Concat(xdot, *augmented_state_exprs)
    else:
        xdot_aug = xdot

    time_dilation = Control("_time_dilation", shape=(1,))

    # Set up time dilation bounds and initial guess
    # Find the time state by name
    time_state = None
    for state in states:
        if state.name == "time":
            time_state = state
            break

    if time_state is None:
        raise ValueError("No state named 'time' found in states list")

    time_final = time_state.final[0]
    time_dilation.min = np.array([time_dilation_factor_min * time_final])
    time_dilation.max = np.array([time_dilation_factor_max * time_final])

    # Compute initial guess for time_dilation from time.guess using finite differences
    # The relationship is: dt/dtau = time_dilation, where tau is normalized time [0,1]
    # With N nodes, dtau = 1/(N-1) between consecutive nodes
    if time_state.guess is None:
        raise ValueError("time state must have a guess set before augmentation")

    if N > 1:
        time_guess = time_state.guess.flatten()  # Shape (N,)
        time_dilation_guess = np.zeros(N)
        dtau = 1.0 / (N - 1)  # Normalized time step between nodes
        # Compute finite difference: time_dilation[k] = (time[k+1] - time[k]) / dtau
        for k in range(N - 1):
            time_dilation_guess[k] = (time_guess[k + 1] - time_guess[k]) / dtau
        # For the last node, use the previous value (extrapolate)
        time_dilation_guess[N - 1] = time_dilation_guess[N - 2]
        time_dilation.guess = time_dilation_guess.reshape(-1, 1)
    else:
        # Single node case: use time_final as guess
        time_dilation.guess = np.ones([N, 1]) * time_final

    controls_augmented.append(time_dilation)

    return xdot_aug, states_augmented, controls_augmented
