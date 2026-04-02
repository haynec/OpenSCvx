"""Unification functions for aggregating symbolic State and Control objects.

This module provides the unification layer that transforms multiple symbolic State
and Control objects into unified representations for numerical optimization.

The unification process:
    1. **Collection**: Gathers all State and Control objects from expression trees
    2. **Sorting**: Organizes variables (user-defined first, then augmented)
    3. **Aggregation**: Concatenates bounds, guesses, and boundary conditions
    4. **Slice Assignment**: Assigns each State/Control a slice for indexing
    5. **Unified Representation**: Creates UnifiedState/UnifiedControl objects

This separation allows users to define problems with natural variable names
while maintaining efficient vectorized operations during optimization.

Example:
    Creating and unifying multiple states::

        import openscvx as ox
        from openscvx.symbolic.unified import unify_states

        # Define separate symbolic states
        position = ox.State("position", shape=(3,), min=-10, max=10)
        velocity = ox.State("velocity", shape=(3,), min=-5, max=5)
        mass = ox.State("mass", shape=(1,), min=0.1, max=10.0)

        # Unify into single state vector
        unified_x = unify_states([position, velocity, mass], name="x")

        # Access unified properties
        print(unified_x.shape)    # (7,) - combined shape
        print(unified_x.min)      # Combined bounds: [-10, -10, -10, -5, -5, -5, 0.1]
        print(unified_x.true)     # Access only user-defined states

    Accessing slices after unification::

        # After unification, each State has a slice assigned
        print(position._slice)    # slice(0, 3)
        print(velocity._slice)    # slice(3, 6)
        print(mass._slice)        # slice(6, 7)

        # During lowering, these slices extract values from unified vector
        x_unified = jnp.array([1, 2, 3, 4, 5, 6, 7])
        position_val = x_unified[position._slice]  # [1, 2, 3]

See Also:
    - UnifiedState: Dataclass for unified state representation (in openscvx.lowered.unified)
    - UnifiedControl: Dataclass for unified control representation (in openscvx.lowered.unified)
    - State: Individual symbolic state variable (symbolic/expr/state.py)
    - Control: Individual symbolic control variable (symbolic/expr/control.py)
"""

from typing import List

import numpy as np

from openscvx.lowered.unified import UnifiedControl, UnifiedState
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State

# Re-export for backwards compatibility
__all__ = ["unify_states", "unify_controls", "UnifiedState", "UnifiedControl"]


def unify_states(states: List[State], name: str = "unified_state") -> UnifiedState:
    """Create a UnifiedState from a list of State objects.

    This function is the primary way to aggregate multiple symbolic State objects into
    a single unified state vector for numerical optimization. It:

    1. Sorts states (user-defined first, augmented states second)
    2. Concatenates all state properties (bounds, guesses, boundary conditions)
    3. Assigns slices to each State for extracting values from unified vector
    4. Identifies special states (time, CTCS augmented states)
    5. Returns a UnifiedState with all aggregated data

    Args:
        states (List[State]): List of State objects to unify. Can include both
            user-defined states and augmented states (names starting with '_').
        name (str): Name identifier for the unified state vector (default: "unified_state")

    Returns:
        UnifiedState: Unified state object containing:
            - Aggregated bounds, guesses, and boundary conditions
            - Shape equal to sum of all state shapes
            - Slices for extracting individual state components
            - Properties for accessing true vs augmented states

    Example:
        Basic unification::

            import openscvx as ox
            from openscvx.symbolic.unified import unify_states

            position = ox.State("pos", shape=(3,), min=-10, max=10)
            velocity = ox.State("vel", shape=(3,), min=-5, max=5)

            unified = unify_states([position, velocity], name="x")
            print(unified.shape)       # (6,)
            print(unified._true_dim)   # 6 (all are user states)
            print(position._slice)     # slice(0, 3) - assigned during unification
            print(velocity._slice)     # slice(3, 6)

        With augmented states::

            # CTCS or other features may add augmented states
            time_state = ox.State("time", shape=(1,))
            ctcs_aug = ox.State("_ctcs_aug_0", shape=(2,))  # Augmented state

            unified = unify_states([position, velocity, time_state, ctcs_aug])
            print(unified._true_dim)         # 7 (pos + vel + time)
            print(unified.true.shape)        # (7,)
            print(unified.augmented.shape)   # (2,) - only CTCS augmented

    Note:
        After unification, each State object has its `_slice` attribute set,
        which is used during JAX lowering to extract the correct values from
        the unified state vector.

    See Also:
        - UnifiedState: Return type with detailed documentation
        - unify_controls(): Analogous function for Control objects
        - State: Individual symbolic state variable
    """
    if not states:
        return UnifiedState(name=name, shape=(0,))

    # Sort states: true states (not starting with '_') first, then augmented states
    # (starting with '_')
    true_states = [state for state in states if not state.name.startswith("_")]
    augmented_states = [state for state in states if state.name.startswith("_")]
    sorted_states = true_states + augmented_states

    # Calculate total shape
    total_shape = sum(state.shape[0] for state in sorted_states)

    # Concatenate all arrays, handling None values properly
    min_arrays = []
    max_arrays = []
    guess_arrays = []
    initial_arrays = []
    final_arrays = []
    _initial_arrays = []
    _final_arrays = []
    initial_type_arrays = []
    final_type_arrays = []

    for state in sorted_states:
        if state.min is not None:
            min_arrays.append(state.min)
        else:
            # If min is None, fill with -inf for this state's dimensions
            min_arrays.append(np.full(state.shape[0], -np.inf))

        if state.max is not None:
            max_arrays.append(state.max)
        else:
            # If max is None, fill with +inf for this state's dimensions
            max_arrays.append(np.full(state.shape[0], np.inf))

        if state.guess is not None:
            guess_arrays.append(state.guess)
        if state.initial is not None:
            initial_arrays.append(state.initial)
        if state.final is not None:
            final_arrays.append(state.final)
        if state._initial is not None:
            _initial_arrays.append(state._initial)
        if state._final is not None:
            _final_arrays.append(state._final)
        if state.initial_type is not None:
            initial_type_arrays.append(state.initial_type)
        else:
            # If initial_type is None, fill with "Free" for this state's dimensions
            initial_type_arrays.append(np.full(state.shape[0], "Free", dtype=object))

        if state.final_type is not None:
            final_type_arrays.append(state.final_type)
        else:
            # If final_type is None, fill with "Free" for this state's dimensions
            final_type_arrays.append(np.full(state.shape[0], "Free", dtype=object))

    # Concatenate arrays if they exist
    unified_min = np.concatenate(min_arrays) if min_arrays else None
    unified_max = np.concatenate(max_arrays) if max_arrays else None
    unified_guess = np.concatenate(guess_arrays, axis=1) if guess_arrays else None
    unified_initial = np.concatenate(initial_arrays) if initial_arrays else None
    unified_final = np.concatenate(final_arrays) if final_arrays else None
    unified__initial = np.concatenate(_initial_arrays) if _initial_arrays else None
    unified__final = np.concatenate(_final_arrays) if _final_arrays else None
    unified_initial_type = np.concatenate(initial_type_arrays) if initial_type_arrays else None
    unified_final_type = np.concatenate(final_type_arrays) if final_type_arrays else None

    # Calculate true dimension (only from user-defined states, not augmented ones)
    # Since we simplified State/Control classes, all user states are "true" dimensions
    true_dim = sum(state.shape[0] for state in true_states)

    # Find time state slice
    time_state = next((s for s in sorted_states if s.name == "time"), None)
    time_slice = time_state._slice if time_state else None

    # Find CTCS augmented states slice
    ctcs_states = [s for s in sorted_states if s.name.startswith("_ctcs_aug_")]
    ctcs_slice = (
        slice(ctcs_states[0]._slice.start, ctcs_states[-1]._slice.stop) if ctcs_states else None
    )

    # Aggregate scaling_min and scaling_max from individual states
    # Build full arrays using scaling where available, min/max otherwise
    unified_scaling_min = None
    unified_scaling_max = None

    # Check if any state has scaling
    has_any_scaling = any(
        state.scaling_min is not None or state.scaling_max is not None for state in sorted_states
    )

    if has_any_scaling:
        # Build full scaling arrays
        scaling_min_list = []
        scaling_max_list = []
        for state in sorted_states:
            if state.scaling_min is not None:
                scaling_min_list.append(state.scaling_min)
            else:
                # Use min as fallback
                if state.min is not None:
                    scaling_min_list.append(state.min)
                else:
                    scaling_min_list.append(np.full(state.shape[0], -np.inf))

            if state.scaling_max is not None:
                scaling_max_list.append(state.scaling_max)
            else:
                # Use max as fallback
                if state.max is not None:
                    scaling_max_list.append(state.max)
                else:
                    scaling_max_list.append(np.full(state.shape[0], np.inf))

        unified_scaling_min = np.concatenate(scaling_min_list)
        unified_scaling_max = np.concatenate(scaling_max_list)

    return UnifiedState(
        name=name,
        shape=(total_shape,),
        min=unified_min,
        max=unified_max,
        guess=unified_guess,
        initial=unified_initial,
        final=unified_final,
        _initial=unified__initial,
        _final=unified__final,
        initial_type=unified_initial_type,
        final_type=unified_final_type,
        _true_dim=true_dim,
        _true_slice=slice(0, true_dim),
        _augmented_slice=slice(true_dim, total_shape),
        time_slice=time_slice,
        ctcs_slice=ctcs_slice,
        scaling_min=unified_scaling_min,
        scaling_max=unified_scaling_max,
    )


def unify_controls(controls: List[Control], name: str = "unified_control") -> UnifiedControl:
    """Create a UnifiedControl from a list of Control objects.

    This function is the primary way to aggregate multiple symbolic Control objects into
    a single unified control vector for numerical optimization. It:

    1. Reorders controls by actuation type: continuous first, impulsive second
    2. Within each actuation-type group, places user-defined controls first and
       augmented controls (names starting with ``"_"``) second
    3. Concatenates all control properties (bounds, guesses)
    4. Assigns slices to each Control for extracting values from unified vector
    5. Identifies special controls (time dilation, impulsive masks/nodes)
    6. Returns a UnifiedControl with all aggregated data

    Args:
        controls (List[Control]): List of Control objects to unify. Can include both
            user-defined controls and augmented controls (names starting with '_').
        name (str): Name identifier for the unified control vector (default: "unified_control")

    Returns:
        UnifiedControl: Unified control object containing:
            - Aggregated bounds and guesses
            - Shape equal to sum of all control shapes
            - Slices for extracting individual control components
            - Properties for accessing true vs augmented controls

    Example:
        Basic unification::

            import openscvx as ox
            from openscvx.symbolic.unified import unify_controls

            thrust = ox.Control("thrust", shape=(3,), min=0, max=10)
            torque = ox.Control("torque", shape=(3,), min=-1, max=1)

            unified = unify_controls([thrust, torque], name="u")
            print(unified.shape)       # (6,)
            print(unified._true_dim)   # 6 (all are user controls)
            print(thrust._slice)       # slice(0, 3) - assigned during unification
            print(torque._slice)       # slice(3, 6)

        With augmented controls::

            # Time-optimal problems may add time dilation control
            time_dilation = ox.Control("_time_dilation", shape=(1,))

            unified = unify_controls([thrust, torque, time_dilation])
            print(unified._true_dim)         # 6 (thrust + torque)
            print(unified.true.shape)        # (6,)
            print(unified.augmented.shape)   # (1,) - time dilation

        With both impulsive and augmented controls::

            thrust = ox.Control("thrust", shape=(2,))  # continuous, user-defined
            time_dilation = ox.Control("_time_dilation", shape=(1,))  # continuous, augmented
            delta_v = ox.Control("delta_v", shape=(2,), parameterization="impulsive", nodes=[0, 20])
            delta_v_aug = ox.Control("_delta_v_bias", shape=(1,), parameterization="impulsive", nodes=[0, 20])

            unified = unify_controls([time_dilation, delta_v_aug, thrust, delta_v], name="u")

            # Unified ordering is:
            # [continuous true, continuous augmented | impulsive true, impulsive augmented]
            # [thrust, _time_dilation | delta_v, _delta_v_bias]
            print(thrust._slice)         # slice(0, 2)
            print(time_dilation._slice)  # slice(2, 3)
            print(delta_v._slice)        # slice(3, 5)
            print(delta_v_aug._slice)    # slice(5, 6)
            print(unified.slice_continuous)  # slice(0, 3)
            print(unified.slice_impulsive)   # slice(3, 6)

    Note:
        After unification, each Control object has its `_slice` attribute set,
        which is used during JAX lowering to extract the correct values from
        the unified control vector.

    See Also:
        - UnifiedControl: Return type with detailed documentation
        - unify_states(): Analogous function for State objects
        - Control: Individual symbolic control variable
    """
    if not controls:
        return UnifiedControl(name=name, shape=(0,))

    def _is_impulsive_control(ctrl: Control) -> bool:
        is_imp = getattr(ctrl, "is_impulsive", False)
        is_imp_arr = np.asarray(is_imp, dtype=bool).reshape(-1)
        return bool(np.any(is_imp_arr))

    # Reorder controls to mirror unified control layout assumptions:
    # 1) continuous controls first, 2) impulsive controls second.
    # Inside each group keep user controls before augmented controls.
    controls_continuous = [c for c in controls if not _is_impulsive_control(c)]
    controls_impulsive = [c for c in controls if _is_impulsive_control(c)]

    def _stable_true_then_aug(group: list[Control]) -> list[Control]:
        true_controls = [c for c in group if not c.name.startswith("_")]
        augmented_controls = [c for c in group if c.name.startswith("_")]
        return true_controls + augmented_controls

    sorted_controls = _stable_true_then_aug(controls_continuous) + _stable_true_then_aug(
        controls_impulsive
    )

    # Assign slices in the new unified order.
    offset = 0
    for control in sorted_controls:
        dim = int(control.shape[0])
        control._slice = slice(offset, offset + dim)
        offset += dim

    # Calculate total shape
    total_shape = sum(control.shape[0] for control in sorted_controls)

    # Concatenate all arrays, handling None values properly
    min_arrays = []
    max_arrays = []
    guess_arrays = []

    for control in sorted_controls:
        if control.min is not None:
            min_arrays.append(control.min)
        else:
            # If min is None, fill with -inf for this control's dimensions
            min_arrays.append(np.full(control.shape[0], -np.inf))

        if control.max is not None:
            max_arrays.append(control.max)
        else:
            # If max is None, fill with +inf for this control's dimensions
            max_arrays.append(np.full(control.shape[0], np.inf))

        if control.guess is not None:
            guess_arrays.append(control.guess)

    # Concatenate arrays if they exist
    unified_min = np.concatenate(min_arrays) if min_arrays else None
    unified_max = np.concatenate(max_arrays) if max_arrays else None
    unified_guess = np.concatenate(guess_arrays, axis=1) if guess_arrays else None

    # Compute a true-control mask from the reordered controls.
    true_mask_parts = []
    for control in sorted_controls:
        is_true_control = not control.name.startswith("_")
        true_mask_parts.append(np.full(control.shape[0], is_true_control, dtype=bool))
    true_mask = np.concatenate(true_mask_parts) if true_mask_parts else np.zeros((0,), dtype=bool)
    true_dim = int(np.sum(true_mask))
    true_slice = UnifiedControl._mask_to_selector(true_mask, empty_at_end=False)
    augmented_slice = UnifiedControl._mask_to_selector(~true_mask, empty_at_end=True)

    # Find time dilation control slice
    time_dilation_control = next((c for c in sorted_controls if c.name == "_time_dilation"), None)
    time_dilation_slice = time_dilation_control._slice if time_dilation_control else None

    # Aggregate scaling_min and scaling_max from individual controls
    # Build full arrays using scaling where available, min/max otherwise
    unified_scaling_min = None
    unified_scaling_max = None
    unified_is_impulsive = np.zeros((total_shape,), dtype=bool)

    # Check if any control has scaling
    has_any_scaling = any(
        control.scaling_min is not None or control.scaling_max is not None
        for control in sorted_controls
    )
    if has_any_scaling:
        # Build full scaling arrays
        scaling_min_list = []
        scaling_max_list = []
        for control in sorted_controls:
            if control.scaling_min is not None:
                scaling_min_list.append(control.scaling_min)
            else:
                # Use min as fallback
                if control.min is not None:
                    scaling_min_list.append(control.min)
                else:
                    scaling_min_list.append(np.full(control.shape[0], -np.inf))

            if control.scaling_max is not None:
                scaling_max_list.append(control.scaling_max)
            else:
                # Use max as fallback
                if control.max is not None:
                    scaling_max_list.append(control.max)
                else:
                    scaling_max_list.append(np.full(control.shape[0], np.inf))

        unified_scaling_min = np.concatenate(scaling_min_list)
        unified_scaling_max = np.concatenate(scaling_max_list)

    nodes = {}
    is_impulsive_list = []
    for control in sorted_controls:
        is_impulsive_block = np.asarray(control.is_impulsive, dtype=bool).reshape(-1)
        if is_impulsive_block.size != int(control.shape[0]):
            raise ValueError(
                f"Control '{control.name}' impulsive mask has size {is_impulsive_block.size}, "
                f"expected {control.shape[0]}."
            )

        if np.any(is_impulsive_block):
            if not np.all(is_impulsive_block):
                raise ValueError(
                    f"Control '{control.name}' mixes continuous and impulsive components. "
                    "Use separate Control objects."
                )
            if control.nodes is not None:
                nodes[control.name] = list(control.nodes)

        is_impulsive_list.append(is_impulsive_block)

    if is_impulsive_list:
        unified_is_impulsive = np.concatenate(is_impulsive_list)
    if not nodes:
        nodes = None

    # Build per-element FOH mask from control-level hold settings.
    # 1.0 = FOH, 0.0 = ZOH, nan = unset (defer to discretizer dis_type).
    any_hold_set = any(getattr(c, "hold", None) is not None for c in sorted_controls)
    if any_hold_set:
        foh_parts: list[np.ndarray] = []
        for control in sorted_controls:
            n = control.shape[0]
            hold = getattr(control, "hold", None)
            if hold == "FOH":
                foh_parts.append(np.ones(n))
            elif hold == "ZOH":
                foh_parts.append(np.zeros(n))
            else:
                foh_parts.append(np.full(n, np.nan))
        unified_foh_mask: np.ndarray | None = np.concatenate(foh_parts)
    else:
        unified_foh_mask = None

    return UnifiedControl(
        name=name,
        shape=(total_shape,),
        min=unified_min,
        max=unified_max,
        guess=unified_guess,
        _true_dim=true_dim,
        _true_slice=true_slice,
        _augmented_slice=augmented_slice,
        time_dilation_slice=time_dilation_slice,
        scaling_min=unified_scaling_min,
        scaling_max=unified_scaling_max,
        is_impulsive=unified_is_impulsive,
        nodes=nodes,
        foh_mask=unified_foh_mask,
    )
