"""Symbolic problem preprocessing and augmentation pipeline.

This module provides the main preprocessing pipeline for trajectory optimization problems,
transforming user-specified symbolic dynamics and constraints into an augmented form
ready for compilation to executable code.

The preprocessing pipeline is purely symbolic - no code generation occurs here. Instead,
it performs validation, canonicalization, and augmentation to prepare the problem for
efficient numerical solution.

Key functionality:
    - Problem validation: Check shapes, variable names, constraint placement
    - Time handling: Auto-create time state or validate user-provided time
    - Canonicalization: Simplify expressions algebraically
    - Parameter collection: Extract parameter values from expressions
    - Constraint separation: Categorize constraints by type (CTCS, nodal, convex)
    - CTCS augmentation: Add augmented states and time dilation for path constraints
    - Propagation dynamics: Optionally extend dynamics for post-solution propagation

The preprocessing pipeline is purely symbolic - no code generation occurs here.

Pipeline stages:
    1. Time handling & validation
    2. Expression validation (shapes, names, constraint structure)
    3. Canonicalization & parameter collection
    4. Constraint separation & CTCS augmentation
    5. Propagation dynamics creation

See `preprocess_symbolic_problem()` for the main entry point.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from openscvx.symbolic.augmentation import (
    augment_dynamics_with_ctcs,
    decompose_vector_nodal_constraints,
    separate_constraints,
    sort_ctcs_constraints,
)
from openscvx.symbolic.constraint_set import ConstraintSet
from openscvx.symbolic.expr import Constant, Parameter, traverse
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State
from openscvx.symbolic.preprocessing import (
    collect_and_assign_slices,
    convert_dynamics_dict_to_expr,
    fill_default_guesses,
    validate_and_normalize_constraint_nodes,
    validate_boundary_conditions,
    validate_bounds,
    validate_constraints_at_root,
    validate_dynamics_dict,
    validate_dynamics_dict_dimensions,
    validate_dynamics_dimension,
    validate_guesses,
    validate_input_types,
    validate_propagation_input_types,
    validate_shapes,
    validate_variable_names,
)
from openscvx.symbolic.problem import SymbolicProblem
from openscvx.symbolic.time import Time


def preprocess_symbolic_problem(
    dynamics: dict,
    constraints: list,
    states: List[State],
    controls: List[Control],
    N: int,
    time: Time,
    licq_min: float = 0.0,
    licq_max: float = 1e-4,
    time_dilation_factor_min: float = 0.3,
    time_dilation_factor_max: float = 3.0,
    dynamics_prop_extra: dict = None,
    states_prop_extra: List[State] = None,
    algebraic_prop: dict = None,
    byof: Optional[dict] = None,
) -> SymbolicProblem:
    """Preprocess and augment symbolic trajectory optimization problem.

    This is the main preprocessing pipeline that transforms a user-specified symbolic
    problem into an augmented form ready for compilation. It performs validation,
    canonicalization, constraint separation, and CTCS augmentation in a series of
    well-defined phases.

    The function is purely symbolic - no code generation or compilation occurs. The
    output is a SymbolicProblem dataclass that can be lowered to JAX or CVXPy by
    downstream compilation functions.

    Pipeline phases:
        1. Time handling & validation: Auto-create or validate time state
        2. Expression validation: Validate shapes, names, constraints
        3. Canonicalization & parameter collection: Simplify and extract parameters
        4. Constraint separation & augmentation: Sort constraints and add CTCS states
        5. Propagation dynamics creation: Optionally add extra states for simulation

    Args:
        dynamics: Dictionary mapping state names to dynamics expressions.
            Example: {"x": v, "v": u}
        constraints: List of constraint objects (Constraint, NodalConstraint,
            CrossNodeConstraint, or CTCS).
        states: List of user-defined State objects (should NOT include time or CTCS states)
        controls: List of user-defined Control objects (should NOT include time dilation)
        N: Number of discretization nodes in the trajectory
        time: Time configuration object specifying time bounds and constraints
        licq_min: Minimum bound for CTCS augmented states (default: 0.0)
        licq_max: Maximum bound for CTCS augmented states (default: 1e-4)
        time_dilation_factor_min: Minimum factor for time dilation control (default: 0.3)
        time_dilation_factor_max: Maximum factor for time dilation control (default: 3.0)
        dynamics_prop_extra: Optional dictionary of additional dynamics for propagation-only
            states (default: None)
        states_prop_extra: Optional list of additional State objects for propagation only
            (default: None)
        algebraic_prop: Optional dictionary of algebraic outputs for propagation
            (evaluated, not integrated). (default: None)
        byof: Optional dict of raw JAX functions for expert users. If byof contains
            a "dynamics" key, it should map state names to raw JAX functions with
            signature f(x, u, node, params) -> xdot_component. States in byof["dynamics"]
            should NOT appear in the symbolic dynamics dict.

    Returns:
        SymbolicProblem dataclass with:
            - dynamics: Augmented dynamics (user + time + CTCS penalties)
            - states: Augmented states (user + time + CTCS augmented)
            - controls: Augmented controls (user + time dilation)
            - constraints: ConstraintSet with is_categorized=True
            - parameters: Dict of extracted parameter values
            - node_intervals: List of (start, end) tuples for CTCS intervals
            - dynamics_prop: Propagation dynamics
            - states_prop: Propagation states
            - controls_prop: Propagation controls
            - algebraic_prop: Algebraic outputs (validated and canonicalized)

    Raises:
        ValueError: If validation fails at any stage

    Example:
        Basic usage with CTCS constraint::

            import openscvx as ox

            x = ox.State("x", shape=(2,))
            v = ox.State("v", shape=(2,))
            u = ox.Control("u", shape=(2,))

            dynamics = {"x": v, "v": u}
            constraints = [(ox.Norm(x) <= 5.0).over((0, 50))]

            problem = preprocess_symbolic_problem(
                dynamics=dynamics,
                constraints=constraints,
                states=[x, v],
                controls=[u],
                N=50,
                time=ox.Time(initial=0.0, final=10.0)
            )

            assert problem.is_preprocessed
            # problem.dynamics: augmented dynamics expression
            # problem.states: [x, v, time, _ctcs_aug_0]
            # problem.controls: [u, _time_dilation]
            print([s.name for s in problem.states])
            # ['x', 'v', 'time', '_ctcs_aug_0']

        With propagation-only states::

            distance = ox.State("distance", shape=(1,))
            dynamics_extra = {"distance": ox.Norm(v)}

            problem = preprocess_symbolic_problem(
                dynamics=dynamics,
                constraints=constraints,
                states=[x, v],
                controls=[u],
                N=50,
                time=ox.Time(initial=0.0, final=10.0),
                dynamics_prop_extra=dynamics_extra,
                states_prop_extra=[distance]
            )

            # Propagation states include distance for post-solve simulation
            print([s.name for s in problem.states_prop])
    """

    # Validate input types before anything else
    validate_input_types(dynamics, states, controls, constraints, N, time)
    validate_propagation_input_types(dynamics_prop_extra, states_prop_extra)

    # Wrap validated constraints into a ConstraintSet
    constraints = ConstraintSet(unsorted=list(constraints))

    # Validate user-provided variables have required attributes
    validate_boundary_conditions(states)
    validate_bounds(states + controls)

    # Fill in default guesses for user-provided states
    # (augmented states get their guesses set by augmentation code)
    fill_default_guesses(states, N)

    # Validate that all user-provided variables have guesses
    validate_guesses(states + controls)

    # ==================== PHASE 1: Time Handling ====================

    # Check if time state already in states list
    time_state = next((s for s in states if s.name == "time"), None)

    if time_state is None:
        # Add the Time object to states (Time is a State subclass)
        states = list(states) + [time]
        time_state = time

    # Fill in default guess if needed (for Time instances)
    if isinstance(time_state, Time) and time_state.guess is None:
        time_state.guess = time_state._generate_default_guess(N)

    # Add CTCS constraints for time bounds
    from openscvx.symbolic.expr import CTCS

    constraints.unsorted.append(CTCS(time_state <= time_state.max))
    constraints.unsorted.append(CTCS(time_state.min <= time_state))

    # Add time derivative to dynamics dict (if not already present)
    # Time derivative is always 1.0 when using Time object
    dynamics = dict(dynamics)  # Make a copy to avoid mutating the input
    if "time" not in dynamics:
        dynamics["time"] = 1.0

    # Extract byof dynamics for validation
    byof_dynamics = byof.get("dynamics", {}) if byof else {}

    # Validate dynamics dict matches state names and dimensions
    # byof_dynamics states should not be in symbolic dynamics dict
    validate_dynamics_dict(dynamics, states, byof_dynamics=byof_dynamics)

    # Inject zero placeholders for byof dynamics states
    # These will be replaced with the actual byof functions at lowering time
    for state in states:
        if state.name in byof_dynamics:
            dynamics[state.name] = Constant(np.zeros(state.shape))

    # Validate dynamics dimensions AFTER injecting placeholders
    validate_dynamics_dict_dimensions(dynamics, states)

    # Convert dynamics dict to concatenated expression
    dynamics, dynamics_concat = convert_dynamics_dict_to_expr(dynamics, states)

    # ==================== PHASE 2: Expression Validation ====================

    # Validate all expressions (use unsorted constraints)
    all_exprs = [dynamics_concat] + constraints.unsorted
    validate_variable_names(all_exprs)
    collect_and_assign_slices(states, controls)
    validate_shapes(all_exprs)
    validate_constraints_at_root(constraints.unsorted)
    validate_and_normalize_constraint_nodes(constraints.unsorted, N)
    validate_dynamics_dimension(dynamics_concat, states)

    # ==================== PHASE 3: Canonicalization & Parameter Collection ====================

    # Canonicalize all expressions after validation
    dynamics_concat = dynamics_concat.canonicalize()
    constraints.unsorted = [expr.canonicalize() for expr in constraints.unsorted]

    # Collect parameter values from all constraints and dynamics
    parameters = {}

    def collect_param_values(expr):
        if isinstance(expr, Parameter):
            if expr.name not in parameters:
                parameters[expr.name] = expr.value

    # Collect from dynamics
    traverse(dynamics_concat, collect_param_values)

    # Collect from constraints
    for constraint in constraints.unsorted:
        traverse(constraint, collect_param_values)

    # ==================== PHASE 4: Constraint Separation & Augmentation ====================

    # Sort and separate constraints by type (drains unsorted -> fills categories)
    separate_constraints(constraints, N)

    # Decompose vector-valued nodal constraints into scalar constraints
    # This is necessary for non-convex nodal constraints that get lowered to JAX
    constraints.nodal = decompose_vector_nodal_constraints(constraints.nodal)

    # Sort CTCS constraints by their idx to get node_intervals
    constraints.ctcs, node_intervals, _ = sort_ctcs_constraints(constraints.ctcs)

    # Augment dynamics, states, and controls with CTCS constraints, time dilation
    dynamics_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        dynamics_concat,
        states,
        controls,
        constraints.ctcs,
        N,
        licq_min=licq_min,
        licq_max=licq_max,
        time_dilation_factor_min=time_dilation_factor_min,
        time_dilation_factor_max=time_dilation_factor_max,
    )

    # Assign slices to augmented states and controls in canonical order
    collect_and_assign_slices(states_aug, controls_aug)

    # ==================== PHASE 5: Create Propagation Dynamics ====================

    # By default, propagation dynamics are the same as optimization dynamics
    # Use deepcopy to avoid reference issues when lowering
    from copy import deepcopy

    dynamics_prop = deepcopy(dynamics_aug)
    states_prop = list(states_aug)  # Shallow copy of list is fine for states
    controls_prop = list(controls_aug)

    # If user provided extra propagation states, extend propagation dynamics
    if dynamics_prop_extra is not None and states_prop_extra is not None:
        (
            dynamics_prop,
            states_prop,
            controls_prop,
            parameters,
        ) = add_propagation_states(
            dynamics_extra=dynamics_prop_extra,
            states_extra=states_prop_extra,
            dynamics_opt=dynamics_prop,
            states_opt=states_prop,
            controls_opt=controls_prop,
            parameters=parameters,
        )

    # ==================== PHASE 6: Process Algebraic Outputs ====================

    # Validate and canonicalize algebraic_prop expressions
    algebraic_prop_processed = None
    if algebraic_prop is not None:
        algebraic_prop_processed = {}
        for name, expr in algebraic_prop.items():
            # Validate shape inference works
            validate_shapes([expr])

            # Canonicalize the expression
            expr_canonical = expr.canonicalize()

            # Collect any parameter values from output expressions
            def collect_param_values(e):
                if isinstance(e, Parameter):
                    if e.name not in parameters:
                        parameters[e.name] = e.value

            traverse(expr_canonical, collect_param_values)

            algebraic_prop_processed[name] = expr_canonical

    # ==================== Return SymbolicProblem ====================

    return SymbolicProblem(
        dynamics=dynamics_aug,
        states=states_aug,
        controls=controls_aug,
        constraints=constraints,
        parameters=parameters,
        N=N,
        node_intervals=node_intervals,
        dynamics_prop=dynamics_prop,
        states_prop=states_prop,
        controls_prop=controls_prop,
        algebraic_prop=algebraic_prop_processed,
    )


def add_propagation_states(
    dynamics_extra: dict,
    states_extra: List[State],
    dynamics_opt: any,
    states_opt: List[State],
    controls_opt: List[Control],
    parameters: Dict[str, any],
) -> Tuple:
    """Extend optimization dynamics with additional propagation-only states.

    This function augments the optimization dynamics with extra states that are only
    needed for post-solution trajectory propagation and simulation. These states
    don't affect the optimization but are useful for computing derived quantities
    like distance traveled, energy consumed, or accumulated cost.

    Propagation-only states are NOT part of the optimization problem - they are
    integrated forward after solving using the optimized state and control trajectories.
    This is more efficient than including them as optimization variables.

    The user specifies only the ADDITIONAL states and their dynamics. These are
    appended after all optimization states (user states + time + CTCS augmented states).

    State ordering in propagation dynamics:
        [user_states, time, ctcs_aug_states, extra_prop_states]

    Args:
        dynamics_extra: Dictionary mapping extra state names to dynamics expressions.
            Only specify NEW states, not optimization states. Example: {"distance": speed}
        states_extra: List of extra State objects for propagation only
        dynamics_opt: Augmented optimization dynamics expression (from preprocessing)
        states_opt: Augmented optimization states (user + time + CTCS augmented)
        controls_opt: Augmented optimization controls (user + time dilation)
        parameters: Dictionary of parameter values from optimization preprocessing

    Returns:
        Tuple containing:
            - dynamics_prop (Expr): Extended dynamics (optimization + extra)
            - states_prop (List[State]): Extended states (optimization + extra)
            - controls_prop (List[Control]): Same as controls_opt
            - parameters_updated (Dict): Updated parameters including any from extra dynamics

    Raises:
        ValueError: If extra states conflict with optimization state names or if
                   validation fails

    Example:
        Adding distance and energy tracking for propagation::

                # After preprocessing, add propagation states
                import openscvx as ox
                import numpy as np

                # Define extra states for tracking
                distance = ox.State("distance", shape=(1,))
                distance.initial = np.array([0.0])

                energy = ox.State("energy", shape=(1,))
                energy.initial = np.array([0.0])

                # Define their dynamics (using optimization states/controls)
                # Assume v and u are optimization states/controls
                dynamics_extra = {
                    "distance": ox.Norm(v),  # Integrate velocity magnitude
                    "energy": ox.Norm(u)**2  # Integrate squared control
                }

                dyn_prop, states_prop, controls_prop, params = add_propagation_states(
                    dynamics_extra=dynamics_extra,
                    states_extra=[distance, energy],
                    dynamics_opt=dynamics_aug,
                    states_opt=states_aug,
                    controls_opt=controls_aug,
                    parameters=parameters
                )

                # Now states_prop includes all states for forward simulation
                # distance and energy will be integrated during propagation

    Note:
        The extra states should have initial conditions set, as they will be
        integrated from these initial values during propagation.
    """

    # Make copies to avoid mutating inputs
    states_extra = list(states_extra)
    dynamics_extra = dict(dynamics_extra)
    parameters = dict(parameters)

    # ==================== PHASE 1: Validate Extra States ====================

    # Validate that extra states don't conflict with optimization state names
    opt_state_names = {s.name for s in states_opt}
    extra_state_names = {s.name for s in states_extra}
    conflicts = opt_state_names & extra_state_names
    if conflicts:
        raise ValueError(
            f"Extra propagation states conflict with optimization states: {conflicts}. "
            f"Only specify additional states, not optimization states."
        )

    # Validate dynamics dict for extra states
    validate_dynamics_dict(dynamics_extra, states_extra)
    validate_dynamics_dict_dimensions(dynamics_extra, states_extra)

    # ==================== PHASE 2: Process Extra Dynamics ====================

    # Convert extra dynamics to expression
    _, dynamics_extra_concat = convert_dynamics_dict_to_expr(dynamics_extra, states_extra)

    # Validate and canonicalize
    validate_variable_names([dynamics_extra_concat])

    # Temporarily assign slices for validation (will be recalculated below)
    collect_and_assign_slices(states_extra, controls_opt)
    validate_shapes([dynamics_extra_concat])
    validate_dynamics_dimension(dynamics_extra_concat, states_extra)
    dynamics_extra_concat = dynamics_extra_concat.canonicalize()

    # Collect any new parameter values from extra dynamics
    def collect_param_values(expr):
        if isinstance(expr, Parameter):
            if expr.name not in parameters:
                parameters[expr.name] = expr.value

    traverse(dynamics_extra_concat, collect_param_values)

    # ==================== PHASE 3: Concatenate with Optimization Dynamics ====================

    # Concatenate: {opt dynamics, extra dynamics}
    from openscvx.symbolic.expr import Concat

    dynamics_prop = Concat(dynamics_opt, dynamics_extra_concat)

    # Manually assign slices to extra states ONLY (don't modify optimization state slices)
    # Extra states are appended after all optimization states
    n_opt_states = states_opt[-1]._slice.stop if states_opt else 0
    start_idx = n_opt_states
    for state in states_extra:
        end_idx = start_idx + state.shape[0]
        state._slice = slice(start_idx, end_idx)
        start_idx = end_idx

    # Append extra states to optimization states
    states_prop = states_opt + states_extra

    # Propagation uses same controls as optimization
    controls_prop = controls_opt

    # ==================== Return Symbolic Outputs ====================

    return (
        dynamics_prop,
        states_prop,
        controls_prop,
        parameters,
    )
