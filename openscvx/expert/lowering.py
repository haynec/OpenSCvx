"""Lowering logic for bring-your-own-functions (byof).

This module handles integration of user-provided JAX functions into the
lowered problem representation, including dynamics splicing and constraint
addition.
"""

from typing import TYPE_CHECKING, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd
from jax.lax import cond

if TYPE_CHECKING:
    from openscvx.lowered.unified import UnifiedState
    from openscvx.symbolic.expr.state import State

from openscvx.lowered import (
    Dynamics,
    LoweredCrossNodeConstraint,
    LoweredJaxConstraints,
    LoweredNodalConstraint,
)

__all__ = ["apply_byof"]


def apply_byof(
    byof: dict,
    dynamics: Dynamics,
    dynamics_prop: Dynamics,
    jax_constraints: LoweredJaxConstraints,
    x_unified: "UnifiedState",
    x_prop_unified: "UnifiedState",
    u_unified: "UnifiedState",
    states: List["State"],
    states_prop: List["State"],
    N: int,
) -> Tuple[Dynamics, Dynamics, LoweredJaxConstraints, "UnifiedState", "UnifiedState"]:
    """Apply bring-your-own-functions (byof) to augment lowered problem.

    Handles raw JAX functions provided by expert users, including:
    - dynamics: Raw JAX functions for specific state derivatives
    - nodal_constraints: Point-wise constraints at each node
    - cross_nodal_constraints: Constraints coupling multiple nodes
    - ctcs_constraints: Continuous-time constraint satisfaction via dynamics augmentation

    Args:
        byof: Dict with keys "dynamics", "nodal_constraints", "cross_nodal_constraints",
            "ctcs_constraints"
        dynamics: Lowered optimization dynamics to potentially augment
        dynamics_prop: Lowered propagation dynamics to potentially augment
        jax_constraints: Lowered JAX constraints to append to
        x_unified: Unified optimization state interface to potentially augment
        x_prop_unified: Unified propagation state interface to potentially augment
        u_unified: Unified control interface for validation
        states: List of State objects for optimization (with _slice attributes)
        states_prop: List of State objects for propagation (with _slice attributes)
        N: Number of nodes in the trajectory

    Returns:
        Tuple of (dynamics, dynamics_prop, jax_constraints, x_unified, x_prop_unified)

    Example:
        >>> dynamics, dynamics_prop, constraints, x_unified, x_prop_unified = apply_byof(
        ...     byof, dynamics, dynamics_prop, jax_constraints,
        ...     x_unified, x_prop_unified, u_unified, states, states_prop, N
        ... )
    """

    # Note: byof validation happens earlier in Problem.__init__ to fail fast
    # Handle byof dynamics by splicing in raw JAX functions at the correct slices
    byof_dynamics = byof.get("dynamics", {})
    if byof_dynamics:
        # Build mapping from state name to slice for optimization states
        state_slices = {state.name: state._slice for state in states}
        state_slices_prop = {state.name: state._slice for state in states_prop}

        # Time-dilation slice for multiplying byof outputs by s
        td_slice = u_unified.time_dilation_slice

        def _make_composite_dynamics(orig_f, byof_fns, slices_map, td_sl):
            """Create composite dynamics combining symbolic and byof state derivatives.

            This factory splices user-provided byof dynamics into the unified dynamics
            function at the appropriate slice indices, replacing the symbolic dynamics
            for specific states while preserving the rest. The byof outputs are
            multiplied by the time-dilation factor s to match the symbolic dynamics
            which already include s * f(x, u) via the Mul node.

            Args:
                orig_f: Original unified dynamics (x, u, node, params) -> xdot
                byof_fns: Dict mapping state names to byof dynamics functions
                slices_map: Dict mapping state names to slice objects for indexing
                td_sl: Slice for the time-dilation control in the unified u vector

            Returns:
                Composite dynamics function with byof derivatives spliced in
            """

            def composite_f(x, u, node, params):
                # Start with symbolic/default dynamics for all states
                xdot = orig_f(x, u, node, params)

                # Time-dilation factor (symbolic dynamics already include s *)
                s = u[td_sl]

                # Splice in byof dynamics for specific states, multiplied by s
                for state_name, byof_fn in byof_fns.items():
                    sl = slices_map[state_name]
                    # Replace the derivative for this state with s * byof result
                    xdot = xdot.at[sl].set(s * byof_fn(x, u, node, params))

                return xdot

            return composite_f

        # Create composite optimization dynamics
        # Jacobians are computed by the discretizer, not here.
        composite_f = _make_composite_dynamics(dynamics.f, byof_dynamics, state_slices, td_slice)
        dynamics = Dynamics(f=composite_f)

        # Create composite propagation dynamics
        composite_f_prop = _make_composite_dynamics(
            dynamics_prop.f, byof_dynamics, state_slices_prop, td_slice
        )
        dynamics_prop = Dynamics(f=composite_f_prop)

    # Handle nodal constraints
    # Note: Validation happens earlier in Problem.__init__ via validate_byof
    for constraint_spec in byof.get("nodal_constraints", []):
        fn = constraint_spec["constraint_fn"]
        nodes = constraint_spec.get("nodes", list(range(N)))  # Default: all nodes

        # Normalize negative node indices (validation already done in validate_byof)
        normalized_nodes = [node if node >= 0 else N + node for node in nodes]

        constraint = LoweredNodalConstraint(
            func=jax.vmap(fn, in_axes=(0, 0, None, None)),
            grad_g_x=jax.vmap(jacfwd(fn, argnums=0), in_axes=(0, 0, None, None)),
            grad_g_u=jax.vmap(jacfwd(fn, argnums=1), in_axes=(0, 0, None, None)),
            nodes=normalized_nodes,
        )
        jax_constraints.nodal.append(constraint)

    # Handle cross-nodal constraints
    for fn in byof.get("cross_nodal_constraints", []):
        constraint = LoweredCrossNodeConstraint(
            func=fn,
            grad_g_X=jacfwd(fn, argnums=0),
            grad_g_U=jacfwd(fn, argnums=1),
        )
        jax_constraints.cross_node.append(constraint)

    # Handle CTCS constraints by augmenting dynamics
    # Built-in penalty functions
    def _penalty_square(r):
        return jnp.maximum(r, 0.0) ** 2

    def _penalty_l1(r):
        return jnp.maximum(r, 0.0)

    def _penalty_huber(r, delta=1.0):
        abs_r = jnp.maximum(r, 0.0)
        return jnp.where(abs_r <= delta, 0.5 * abs_r**2, delta * (abs_r - 0.5 * delta))

    _PENALTY_FUNCTIONS = {
        "square": _penalty_square,
        "l1": _penalty_l1,
        "huber": _penalty_huber,
    }

    # Determine which symbolic CTCS idx values already exist
    # Symbolic augmented states are named "_ctcs_aug_{i}" where i is sequential
    # and corresponds to sorted symbolic idx values (0, 1, 2, ...)
    symbolic_ctcs_idx = []
    for state in states:
        if state.name.startswith("_ctcs_aug_"):
            try:
                aug_idx = int(state.name.split("_")[-1])
                symbolic_ctcs_idx.append(aug_idx)
            except (ValueError, IndexError):
                pass

    # Symbolic CTCS creates augmented states with sequential idx: 0, 1, 2, ...
    # so max_symbolic_idx = len(symbolic_ctcs_idx) - 1 (or -1 if none exist)
    max_symbolic_idx = len(symbolic_ctcs_idx) - 1 if symbolic_ctcs_idx else -1

    # Build idx -> augmented_state_slice mapping for existing symbolic CTCS
    # Augmented states appear after regular states in the unified vector
    # We'll determine the slice by finding the state in the states list
    idx_to_aug_slice = {}
    for state in states:
        if state.name.startswith("_ctcs_aug_"):
            try:
                aug_idx = int(state.name.split("_")[-1])
                # The actual idx value IS the sequential index for symbolic CTCS
                # (they're created with idx 0, 1, 2, ... in sorted order)
                idx_to_aug_slice[aug_idx] = state._slice
            except (ValueError, IndexError, AttributeError):
                pass

    # Group BYOF CTCS constraints by idx (default to 0)
    byof_ctcs_groups = {}
    for ctcs_spec in byof.get("ctcs_constraints", []):
        idx = ctcs_spec.get("idx", 0)
        if idx not in byof_ctcs_groups:
            byof_ctcs_groups[idx] = []
        byof_ctcs_groups[idx].append(ctcs_spec)

    # Validate that byof idx values don't create gaps
    # All idx must form contiguous sequence: [0, 1, 2, ..., max_idx]
    if byof_ctcs_groups:
        all_idx = sorted(set(range(max_symbolic_idx + 1)) | set(byof_ctcs_groups.keys()))
        expected_idx = list(range(len(all_idx)))
        if all_idx != expected_idx:
            raise ValueError(
                f"BYOF CTCS idx values create non-contiguous sequence. "
                f"Symbolic CTCS has idx=[{', '.join(map(str, range(max_symbolic_idx + 1)))}], "
                f"combined with byof idx={sorted(byof_ctcs_groups.keys())} gives {all_idx}. "
                f"Expected contiguous sequence {expected_idx}. "
                f"Byof idx must either match existing symbolic idx or be sequential after them."
            )

    # Process each idx group
    for idx in sorted(byof_ctcs_groups.keys()):
        specs = byof_ctcs_groups[idx]

        # Collect all penalty functions for this idx
        penalty_fns = []
        for spec in specs:
            constraint_fn = spec["constraint_fn"]
            penalty_spec = spec.get("penalty", "square")
            over_interval = spec.get("over", None)  # Node interval (start, end) or None

            if callable(penalty_spec):
                penalty_func = penalty_spec
            else:
                penalty_func = _PENALTY_FUNCTIONS[penalty_spec]

            # Create a combined constraint+penalty function
            def _make_penalty_fn(cons_fn, pen_func, over):
                """Factory to capture constraint, penalty functions, and node interval.

                Args:
                    cons_fn: Constraint function (x, u, node, params) -> scalar residual
                    pen_func: Penalty function (residual) -> penalty value
                    over: Optional (start, end) tuple for conditional activation

                Returns:
                    Penalty function that conditionally activates based on node interval
                """

                def penalty_fn(x, u, node, params):
                    # Compute penalty for the constraint violation
                    residual = cons_fn(x, u, node, params)
                    penalty_value = pen_func(residual)

                    # Apply conditional logic if over interval is specified
                    if over is not None:
                        start_node, end_node = over
                        # Extract scalar from node (which may be array or scalar)
                        # Keep as JAX array for tracing compatibility
                        node_scalar = jnp.atleast_1d(node)[0]
                        is_active = (start_node <= node_scalar) & (node_scalar < end_node)

                        # Use jax.lax.cond for JAX-traceable conditional evaluation
                        # Penalty is active only when node is in [start, end)
                        return cond(
                            is_active,
                            lambda _: penalty_value,
                            lambda _: 0.0,
                            operand=None,
                        )
                    else:
                        # Always active if no interval specified
                        return penalty_value

                return penalty_fn

            penalty_fns.append(_make_penalty_fn(constraint_fn, penalty_func, over_interval))

        # Time-dilation slice for multiplying byof CTCS penalties by s
        td_slice = u_unified.time_dilation_slice

        if idx in idx_to_aug_slice:
            # This idx already exists from symbolic CTCS - add penalties to existing state
            aug_slice = idx_to_aug_slice[idx]

            def _make_ctcs_addition(orig_f, pen_fns, aug_sl, td_sl):
                """Create dynamics that adds penalties to existing augmented state.

                The penalty is multiplied by the time-dilation factor s to match
                the symbolic dynamics which already include s * f(x, u).

                Args:
                    orig_f: Original dynamics function
                    pen_fns: List of penalty functions to add
                    aug_sl: Slice of the augmented state to modify
                    td_sl: Slice for the time-dilation control

                Returns:
                    Modified dynamics function
                """

                def modified_f(x, u, node, params):
                    xdot = orig_f(x, u, node, params)

                    # Sum all penalties for this idx, scaled by time-dilation
                    s = u[td_sl]
                    total_penalty = s * sum(pen_fn(x, u, node, params) for pen_fn in pen_fns)

                    # Add to existing augmented state derivative
                    current_deriv = xdot[aug_sl]
                    xdot = xdot.at[aug_sl].set(current_deriv + total_penalty)

                    return xdot

                return modified_f

            # Modify both optimization and propagation dynamics
            # Jacobians are computed by the discretizer, not here.
            dynamics.f = _make_ctcs_addition(dynamics.f, penalty_fns, aug_slice, td_slice)
            dynamics_prop.f = _make_ctcs_addition(dynamics_prop.f, penalty_fns, aug_slice, td_slice)

        else:
            # New idx - create new augmented state
            # Use bounds/initial from first spec in this group
            first_spec = specs[0]
            bounds = first_spec.get("bounds", (0.0, 1e-4))
            initial = first_spec.get("initial", bounds[0])

            def _make_ctcs_new_state(orig_f, pen_fns, td_sl):
                """Create dynamics augmented with new CTCS state.

                The penalty is multiplied by the time-dilation factor s to match
                the symbolic dynamics which already include s * f(x, u).

                Args:
                    orig_f: Original dynamics function
                    pen_fns: List of penalty functions to sum
                    td_sl: Slice for the time-dilation control

                Returns:
                    Augmented dynamics function
                """

                def augmented_f(x, u, node, params):
                    xdot = orig_f(x, u, node, params)

                    # Sum all penalties for this new idx, scaled by time-dilation
                    s = u[td_sl]
                    total_penalty = s * sum(pen_fn(x, u, node, params) for pen_fn in pen_fns)

                    # Append as new augmented state derivative
                    return jnp.concatenate([xdot, jnp.atleast_1d(total_penalty)])

                return augmented_f

            # Augment optimization dynamics
            # Jacobians are computed by the discretizer, not here.
            aug_f = _make_ctcs_new_state(dynamics.f, penalty_fns, td_slice)
            dynamics = Dynamics(f=aug_f)

            # Augment propagation dynamics
            aug_f_prop = _make_ctcs_new_state(dynamics_prop.f, penalty_fns, td_slice)
            dynamics_prop = Dynamics(f=aug_f_prop)

            # Create State objects for the new augmented states
            # This is necessary for CVXPy variable creation and other bookkeeping
            from openscvx.symbolic.expr.state import State

            # Create augmented state for optimization
            aug_state = State(f"_ctcs_aug_{idx}", shape=(1,))
            aug_state.min = np.array([bounds[0]])
            aug_state.max = np.array([bounds[1]])
            aug_state.initial = np.array([initial])
            aug_state.final = [("free", 0.0)]
            aug_state.guess = np.full((N, 1), initial)

            # Set _slice attribute for the new state
            current_dim = x_unified.shape[0]
            aug_state._slice = slice(current_dim, current_dim + 1)

            # Append to states list (in-place modification visible to caller)
            states.append(aug_state)

            # Create augmented state for propagation
            aug_state_prop = State(f"_ctcs_aug_{idx}", shape=(1,))
            aug_state_prop.min = np.array([bounds[0]])
            aug_state_prop.max = np.array([bounds[1]])
            aug_state_prop.initial = np.array([initial])
            aug_state_prop.final = [("free", 0.0)]
            aug_state_prop.guess = np.full((N, 1), initial)

            # Set _slice attribute for the propagation state
            current_dim_prop = x_prop_unified.shape[0]
            aug_state_prop._slice = slice(current_dim_prop, current_dim_prop + 1)

            # Append to states_prop list
            states_prop.append(aug_state_prop)

            # Add new augmented states to both unified state interfaces
            x_unified.append(
                min=bounds[0],
                max=bounds[1],
                guess=initial,
                initial=initial,
                final=0.0,
                augmented=True,
            )
            x_prop_unified.append(
                min=bounds[0],
                max=bounds[1],
                guess=initial,
                initial=initial,
                final=0.0,
                augmented=True,
            )

    return dynamics, dynamics_prop, jax_constraints, x_unified, x_prop_unified
