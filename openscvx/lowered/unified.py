"""Unified state and control dataclasses for the lowered representation.

This module contains the UnifiedState and UnifiedControl dataclasses that describe
the structure of the monolithic state and control vectors used in numerical optimization.

In the symbolic world, users define many named State and Control objects (position,
velocity, thrust, etc.). In the lowered world, these are aggregated into single
monolithic x and u vectors. UnifiedState and UnifiedControl hold the metadata
describing this aggregation: bounds, guesses, boundary conditions, and slices
for extracting individual components.

See Also:
    - openscvx.symbolic.unified: Contains unify_states() and unify_controls()
      functions that create these dataclasses from symbolic State/Control objects.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State


@dataclass
class UnifiedState:
    """Unified state vector aggregating multiple State objects.

    UnifiedState is a drop-in replacement for individual State objects that holds
    aggregated data from multiple State instances. It maintains compatibility with
    optimization infrastructure while providing access to individual state components
    through slicing.

    The unified state separates user-defined "true" states from augmented states
    added internally (e.g., for CTCS constraints or time variables). This separation
    allows clean access to physical states while supporting advanced features.

    Attributes:
        name (str): Name identifier for the unified state vector
        shape (tuple): Combined shape (total_dim,) of all aggregated states
        min (np.ndarray): Lower bounds for all state variables, shape (total_dim,)
        max (np.ndarray): Upper bounds for all state variables, shape (total_dim,)
        guess (np.ndarray): Initial guess trajectory, shape (num_nodes, total_dim)
        initial (np.ndarray): Initial boundary conditions, shape (total_dim,)
        final (np.ndarray): Final boundary conditions, shape (total_dim,)
        _initial (np.ndarray): Internal initial values, shape (total_dim,)
        _final (np.ndarray): Internal final values, shape (total_dim,)
        initial_type (np.ndarray): Boundary condition types at t0 ("Fix" or "Free"),
            shape (total_dim,), dtype=object
        final_type (np.ndarray): Boundary condition types at tf ("Fix" or "Free"),
            shape (total_dim,), dtype=object
        _true_dim (int): Number of user-defined state dimensions (excludes augmented)
        _true_slice (slice): Slice for extracting true states from unified vector
        _augmented_slice (slice): Slice for extracting augmented states
        time_slice (Optional[slice]): Slice for time state variable, if present
        ctcs_slice (Optional[slice]): Slice for CTCS augmented states, if present

    Properties:
        true: Returns UnifiedState view containing only true (user-defined) states
        augmented: Returns UnifiedState view containing only augmented states

    Example:
        Creating a unified state from multiple State objects::

            from openscvx.symbolic.unified import unify_states

            position = ox.State("pos", shape=(3,), min=-10, max=10)
            velocity = ox.State("vel", shape=(3,), min=-5, max=5)

            unified = unify_states([position, velocity], name="x")
            print(unified.shape)        # (6,)
            print(unified.min)          # [-10, -10, -10, -5, -5, -5]
            print(unified.true.shape)   # (6,) - all are true states
            print(unified.augmented.shape)  # (0,) - no augmented states

        Appending states dynamically::

            unified = UnifiedState(name="x", shape=(0,), _true_dim=0)
            unified.append(min=-1, max=1, guess=0.5)  # Add scalar state
            print(unified.shape)  # (1,)

    See Also:
        - unify_states(): Factory function for creating UnifiedState from State list
        - State: Individual symbolic state variable
        - UnifiedControl: Analogous unified control vector
    """

    name: str
    shape: tuple
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    guess: Optional[np.ndarray] = None
    initial: Optional[np.ndarray] = None
    final: Optional[np.ndarray] = None
    _initial: Optional[np.ndarray] = None
    _final: Optional[np.ndarray] = None
    initial_type: Optional[np.ndarray] = None
    final_type: Optional[np.ndarray] = None
    _true_dim: int = 0
    _true_slice: Optional[slice] = None
    _augmented_slice: Optional[slice] = None
    time_slice: Optional[slice] = None  # Slice for time state
    ctcs_slice: Optional[slice] = None  # Slice for CTCS augmented states
    scaling_min: Optional[np.ndarray] = None  # Scaling minimum bounds for unified state
    scaling_max: Optional[np.ndarray] = None  # Scaling maximum bounds for unified state

    def __post_init__(self):
        """Initialize slices after dataclass creation."""
        if self._true_slice is None:
            self._true_slice = slice(0, self._true_dim)
        if self._augmented_slice is None:
            self._augmented_slice = slice(self._true_dim, self.shape[0])

    @property
    def true(self) -> "UnifiedState":
        """Get the true (user-defined) state variables.

        Returns a view of the unified state containing only user-defined states,
        excluding internal augmented states added for CTCS, time, etc.

        Returns:
            UnifiedState: Sliced view containing only true state variables

        Example:
            Get true user-defined state::

                unified = unify_states([position, velocity, ctcs_aug], name="x")
                true_states = unified.true  # Only position and velocity
                true_states.shape  # (6,) if position and velocity are 3D each
        """
        return self[self._true_slice]

    @property
    def augmented(self) -> "UnifiedState":
        """Get the augmented (internal) state variables.

        Returns a view of the unified state containing only augmented states
        added internally by the optimization framework (e.g., CTCS penalty states,
        time variables).

        Returns:
            UnifiedState: Sliced view containing only augmented state variables

        Example:
            Get augmented state::

                unified = unify_states([position, ctcs_aug], name="x")
                aug_states = unified.augmented  # Only CTCS states
        """
        return self[self._augmented_slice]

    def append(
        self,
        other: "Optional[State | UnifiedState]" = None,
        *,
        min=-np.inf,
        max=np.inf,
        guess=0.0,
        initial=0.0,
        final=0.0,
        augmented=False,
    ) -> None:
        """Append another state or create a new state variable.

        This method allows dynamic extension of the unified state, either by appending
        another State/UnifiedState object or by creating a new scalar state variable
        with specified properties. Modifies the unified state in-place.

        Args:
            other (Optional[State | UnifiedState]): State object to append. If None,
                creates a new scalar state variable with properties from keyword args.
            min (float): Lower bound for new scalar state (default: -inf)
            max (float): Upper bound for new scalar state (default: inf)
            guess (float): Initial guess value for new scalar state (default: 0.0)
            initial (float): Initial boundary condition for new scalar state (default: 0.0)
            final (float): Final boundary condition for new scalar state (default: 0.0)
            augmented (bool): Whether the appended state is augmented (internal) rather
                than true (user-defined). Affects _true_dim tracking. Default: False

        Returns:
            None: Modifies the unified state in-place

        Example:
            Appending a State object::

                unified = unify_states([position], name="x")
                velocity = ox.State("vel", shape=(3,), min=-5, max=5)
                unified.append(velocity)
                print(unified.shape)  # (6,) - position (3) + velocity (3)

            Creating new scalar state variables::

                unified = UnifiedState(name="x", shape=(0,), _true_dim=0)
                unified.append(min=-1, max=1, guess=0.5)  # Add scalar state
                unified.append(min=-2, max=2, augmented=True)  # Add augmented state
                print(unified.shape)  # (2,)
                print(unified._true_dim)  # 1 (only first is true)

        Note:
            Maintains the invariant that true states appear before augmented states
            in the unified vector. When appending augmented states, they are added
            to the end but don't increment _true_dim.
        """
        # Import here to avoid circular imports at module level
        from openscvx.symbolic.expr.state import State

        if isinstance(other, (State, UnifiedState)):
            # Append another state object
            new_shape = (self.shape[0] + other.shape[0],)

            # Update bounds
            if self.min is not None and other.min is not None:
                new_min = np.concatenate([self.min, other.min])
            else:
                new_min = self.min

            if self.max is not None and other.max is not None:
                new_max = np.concatenate([self.max, other.max])
            else:
                new_max = self.max

            # Update guess
            if self.guess is not None and other.guess is not None:
                new_guess = np.concatenate([self.guess, other.guess], axis=1)
            else:
                new_guess = self.guess

            # Update initial/final conditions
            if self.initial is not None and other.initial is not None:
                new_initial = np.concatenate([self.initial, other.initial])
            else:
                new_initial = self.initial

            if self.final is not None and other.final is not None:
                new_final = np.concatenate([self.final, other.final])
            else:
                new_final = self.final

            # Update internal arrays
            if self._initial is not None and other._initial is not None:
                new__initial = np.concatenate([self._initial, other._initial])
            else:
                new__initial = self._initial

            if self._final is not None and other._final is not None:
                new__final = np.concatenate([self._final, other._final])
            else:
                new__final = self._final

            # Update types
            if self.initial_type is not None and other.initial_type is not None:
                new_initial_type = np.concatenate([self.initial_type, other.initial_type])
            else:
                new_initial_type = self.initial_type

            if self.final_type is not None and other.final_type is not None:
                new_final_type = np.concatenate([self.final_type, other.final_type])
            else:
                new_final_type = self.final_type

            # Update scaling bounds (if present)
            if (
                self.scaling_min is not None
                and hasattr(other, "scaling_min")
                and other.scaling_min is not None
            ):
                new_scaling_min = np.concatenate([self.scaling_min, other.scaling_min])
            else:
                new_scaling_min = self.scaling_min

            if (
                self.scaling_max is not None
                and hasattr(other, "scaling_max")
                and other.scaling_max is not None
            ):
                new_scaling_max = np.concatenate([self.scaling_max, other.scaling_max])
            else:
                new_scaling_max = self.scaling_max

            # Update true dimension
            if not augmented:
                new_true_dim = self._true_dim + getattr(other, "_true_dim", other.shape[0])
            else:
                new_true_dim = self._true_dim

            # Update all attributes in place
            self.shape = new_shape
            self.min = new_min
            self.max = new_max
            self.guess = new_guess
            self.initial = new_initial
            self.final = new_final
            self._initial = new__initial
            self._final = new__final
            self.initial_type = new_initial_type
            self.final_type = new_final_type
            self.scaling_min = new_scaling_min
            self.scaling_max = new_scaling_max
            self._true_dim = new_true_dim
            self._true_slice = slice(0, self._true_dim)
            self._augmented_slice = slice(self._true_dim, self.shape[0])

        else:
            # Create a single new variable
            new_shape = (self.shape[0] + 1,)

            # Extend arrays
            if self.min is not None:
                self.min = np.concatenate([self.min, np.array([min])])
            if self.max is not None:
                self.max = np.concatenate([self.max, np.array([max])])
            if self.guess is not None:
                guess_arr = np.full((self.guess.shape[0], 1), guess)
                self.guess = np.concatenate([self.guess, guess_arr], axis=1)
            if self.initial is not None:
                self.initial = np.concatenate([self.initial, np.array([initial])])
            if self.final is not None:
                self.final = np.concatenate([self.final, np.array([final])])
            if self._initial is not None:
                self._initial = np.concatenate([self._initial, np.array([initial])])
            if self._final is not None:
                self._final = np.concatenate([self._final, np.array([final])])
            if self.initial_type is not None:
                self.initial_type = np.concatenate(
                    [self.initial_type, np.array(["Fix"], dtype=object)]
                )
            if self.final_type is not None:
                self.final_type = np.concatenate([self.final_type, np.array(["Fix"], dtype=object)])
            if self.scaling_min is not None:
                self.scaling_min = np.concatenate([self.scaling_min, np.array([min])])
            if self.scaling_max is not None:
                self.scaling_max = np.concatenate([self.scaling_max, np.array([max])])

            # Update dimensions
            self.shape = new_shape
            if not augmented:
                self._true_dim += 1
            self._true_slice = slice(0, self._true_dim)
            self._augmented_slice = slice(self._true_dim, self.shape[0])

    def __getitem__(self, idx):
        """Get a subset of the unified state variables.

        Enables slicing of the unified state to extract subsets of state variables.
        Returns a new UnifiedState containing only the sliced dimensions.

        Args:
            idx (slice): Slice object specifying which state dimensions to extract.
                Only simple slices with step=1 are supported.

        Returns:
            UnifiedState: New unified state containing only the sliced dimensions

        Raises:
            NotImplementedError: If idx is not a slice, or if step != 1

        Example:
            Generate unified state object::

                unified = unify_states([position, velocity], name="x")

            position has shape (3,), velocity has shape (3,)::

                first_three = unified[0:3]  # Extract position only
                print(first_three.shape)  # (3,)
                last_three = unified[3:6]  # Extract velocity only
                print(last_three.shape)  # (3,)

        Note:
            The sliced state maintains all properties (bounds, guesses, etc.) for
            the selected dimensions. The _true_dim is recalculated based on which
            dimensions fall within the original true state range.
        """
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.shape[0])
            if step != 1:
                raise NotImplementedError("Step slicing not supported")

            new_shape = (stop - start,)
            new_name = f"{self.name}[{start}:{stop}]"

            # Slice all arrays
            new_min = self.min[idx] if self.min is not None else None
            new_max = self.max[idx] if self.max is not None else None
            new_guess = self.guess[:, idx] if self.guess is not None else None
            new_initial = self.initial[idx] if self.initial is not None else None
            new_final = self.final[idx] if self.final is not None else None
            new__initial = self._initial[idx] if self._initial is not None else None
            new__final = self._final[idx] if self._final is not None else None
            new_initial_type = self.initial_type[idx] if self.initial_type is not None else None
            new_final_type = self.final_type[idx] if self.final_type is not None else None

            # Calculate new true dimension
            new_true_dim = max(0, min(stop, self._true_dim) - max(start, 0))

            return UnifiedState(
                name=new_name,
                shape=new_shape,
                min=new_min,
                max=new_max,
                guess=new_guess,
                initial=new_initial,
                final=new_final,
                _initial=new__initial,
                _final=new__final,
                initial_type=new_initial_type,
                final_type=new_final_type,
                _true_dim=new_true_dim,
                _true_slice=slice(0, new_true_dim),
                _augmented_slice=slice(new_true_dim, new_shape[0]),
            )
        else:
            raise NotImplementedError("Only slice indexing is supported")

    def __repr__(self):
        """String representation of the UnifiedState object."""
        return f"UnifiedState('{self.name}', shape={self.shape})"


@dataclass
class UnifiedControl:
    """Unified control vector aggregating multiple Control objects.

    UnifiedControl is a drop-in replacement for individual Control objects that holds
    aggregated data from multiple Control instances. It maintains compatibility with
    optimization infrastructure while providing access to individual control components
    through slicing.

    The unified control separates user-defined "true" controls from augmented controls
    added internally (e.g., for time dilation). This separation allows clean access to
    physical control inputs while supporting advanced features.

    Attributes:
        name (str): Name identifier for the unified control vector
        shape (tuple): Combined shape (total_dim,) of all aggregated controls
        min (np.ndarray): Lower bounds for all control variables, shape (total_dim,)
        max (np.ndarray): Upper bounds for all control variables, shape (total_dim,)
        guess (np.ndarray): Initial guess trajectory, shape (num_nodes, total_dim)
        _true_dim (int): Number of user-defined control dimensions (excludes augmented)
        _true_slice (slice): Slice for extracting true controls from unified vector
        _augmented_slice (slice): Slice for extracting augmented controls
        time_dilation_slice (Optional[slice]): Slice for time dilation control, if present

    Properties:
        true: Returns UnifiedControl view containing only true (user-defined) controls
        augmented: Returns UnifiedControl view containing only augmented controls
        slice_continuous: Slice covering continuous controls in unified ordering
        slice_impulsive: Slice covering impulsive controls in unified ordering

    Example:
        Creating a unified control from multiple Control objects::

            from openscvx.symbolic.unified import unify_controls

            thrust = ox.Control("thrust", shape=(3,), min=0, max=10)
            torque = ox.Control("torque", shape=(3,), min=-1, max=1)

            unified = unify_controls([thrust, torque], name="u")
            print(unified.shape)        # (6,)
            print(unified.min)          # [0, 0, 0, -1, -1, -1]
            print(unified.true.shape)   # (6,) - all are true controls
            print(unified.augmented.shape)  # (0,) - no augmented controls

        Appending controls dynamically::

            unified = UnifiedControl(name="u", shape=(0,), _true_dim=0)
            unified.append(min=-1, max=1, guess=0.0)  # Add scalar control
            print(unified.shape)  # (1,)

    See Also:
        - unify_controls(): Factory function for creating UnifiedControl from Control list
        - Control: Individual symbolic control variable
        - UnifiedState: Analogous unified state vector
    """

    name: str
    shape: tuple
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    guess: Optional[np.ndarray] = None
    _true_dim: int = 0
    _true_slice: Optional[slice | tuple[slice, ...]] = None
    _augmented_slice: Optional[slice | tuple[slice, ...]] = None
    time_dilation_slice: Optional[slice] = None  # Slice for time dilation control
    scaling_min: Optional[np.ndarray] = None  # Scaling minimum bounds for unified control
    scaling_max: Optional[np.ndarray] = None  # Scaling maximum bounds for unified control
    is_impulsive: Optional[bool] = False  # Default toggle for 'impulsivity' of the unified control
    nodes: Optional[dict[str, list[int]]] = None
    foh_mask: Optional[np.ndarray] = None  # Per-element: 1.0=FOH, 0.0=ZOH, nan=unset

    def __post_init__(self):
        """Initialize slices after dataclass creation."""
        if self._true_slice is None:
            self._true_slice = slice(0, self._true_dim)
        if self._augmented_slice is None:
            true_mask = self._selector_to_mask(self._true_slice, self.shape[0])
            self._augmented_slice = self._mask_to_selector(~true_mask, empty_at_end=True)
        true_mask = self._selector_to_mask(self._true_slice, self.shape[0])
        self._true_dim = int(np.sum(true_mask))

    @staticmethod
    def _mask_to_selector(
        mask: np.ndarray,
        *,
        empty_at_end: bool,
    ) -> slice | tuple[slice, ...]:
        """Convert a boolean mask to one slice (contiguous) or tuple of slices."""
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        n = int(mask.size)
        if n == 0:
            return slice(0, 0)
        if not np.any(mask):
            return slice(n, n) if empty_at_end else slice(0, 0)

        idx = np.flatnonzero(mask)
        run_starts = [int(idx[0])]
        run_stops = []
        prev = int(idx[0])
        for value in idx[1:]:
            current = int(value)
            if current != prev + 1:
                run_stops.append(prev + 1)
                run_starts.append(current)
            prev = current
        run_stops.append(prev + 1)

        slices = tuple(slice(start, stop) for start, stop in zip(run_starts, run_stops))
        if len(slices) == 1:
            return slices[0]
        return slices

    @staticmethod
    def _selector_to_mask(selector: slice | tuple[slice, ...], size: int) -> np.ndarray:
        """Convert selector (slice or tuple of slices) to boolean mask."""
        mask = np.zeros((size,), dtype=bool)
        selectors = selector if isinstance(selector, tuple) else (selector,)
        for sl in selectors:
            if not isinstance(sl, slice):
                raise TypeError(f"Expected slice selector, got {type(sl).__name__}")
            start, stop, step = sl.indices(size)
            if step != 1:
                raise NotImplementedError("Step slicing not supported")
            if stop > start:
                mask[start:stop] = True
        return mask

    @staticmethod
    def _selector_to_indices(selector: slice | tuple[slice, ...], size: int) -> np.ndarray:
        """Convert selector to ordered integer indices."""
        selectors = selector if isinstance(selector, tuple) else (selector,)
        idx_parts = []
        for sl in selectors:
            if not isinstance(sl, slice):
                raise TypeError(f"Expected slice selector, got {type(sl).__name__}")
            start, stop, step = sl.indices(size)
            if step != 1:
                raise NotImplementedError("Step slicing not supported")
            if stop > start:
                idx_parts.append(np.arange(start, stop, dtype=int))
        if not idx_parts:
            return np.array([], dtype=int)
        return np.concatenate(idx_parts)

    def _impulsive_mask(self) -> np.ndarray:
        """Return impulsive mask aligned to unified control dimension."""
        if self.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        if self.is_impulsive is None:
            return np.zeros((self.shape[0],), dtype=bool)
        mask = np.asarray(self.is_impulsive, dtype=bool).reshape(-1)
        if mask.size == 1:
            return np.repeat(mask, self.shape[0])
        if mask.size != self.shape[0]:
            raise ValueError(
                f"is_impulsive mask size {mask.size} does not match unified "
                f"control size {self.shape[0]}"
            )
        return mask

    def _subset(self, indices: np.ndarray, *, name: str, true_mask: np.ndarray) -> "UnifiedControl":
        """Build a sliced UnifiedControl by explicit indices."""
        indices = np.asarray(indices, dtype=int).reshape(-1)
        true_mask = np.asarray(true_mask, dtype=bool).reshape(-1)
        if indices.size != true_mask.size:
            raise ValueError("Subset indices and true_mask must have the same size.")

        new_shape = (int(indices.size),)
        new_min = self.min[indices] if self.min is not None else None
        new_max = self.max[indices] if self.max is not None else None
        new_guess = self.guess[:, indices] if self.guess is not None else None
        new_scaling_min = self.scaling_min[indices] if self.scaling_min is not None else None
        new_scaling_max = self.scaling_max[indices] if self.scaling_max is not None else None

        parent_impulsive_mask = self._impulsive_mask()
        new_is_impulsive = (
            parent_impulsive_mask[indices]
            if parent_impulsive_mask.size
            else np.zeros((new_shape[0],), dtype=bool)
        )
        new_foh_mask = self.foh_mask[indices] if self.foh_mask is not None else None

        true_selector = self._mask_to_selector(true_mask, empty_at_end=False)
        augmented_selector = self._mask_to_selector(~true_mask, empty_at_end=True)

        return UnifiedControl(
            name=name,
            shape=new_shape,
            min=new_min,
            max=new_max,
            guess=new_guess,
            _true_dim=int(np.sum(true_mask)),
            _true_slice=true_selector,
            _augmented_slice=augmented_selector,
            scaling_min=new_scaling_min,
            scaling_max=new_scaling_max,
            is_impulsive=new_is_impulsive,
            nodes=None,
            foh_mask=new_foh_mask,
        )

    @property
    def true(self) -> "UnifiedControl":
        """Get the true (user-defined) control variables.

        Returns a view of the unified control containing only user-defined controls,
        excluding internal augmented controls added for time dilation, etc.

        Returns:
            UnifiedControl: Sliced view containing only true control variables

        Example:
            Get true user defined controls::

                unified = unify_controls([thrust, torque, time_dilation], name="u")
                true_controls = unified.true  # Only thrust and torque
        """
        true_indices = self._selector_to_indices(self._true_slice, self.shape[0])
        true_mask = np.ones((true_indices.size,), dtype=bool)
        return self._subset(true_indices, name=f"{self.name}[true]", true_mask=true_mask)

    @property
    def augmented(self) -> "UnifiedControl":
        """Get the augmented (internal) control variables.

        Returns a view of the unified control containing only augmented controls
        added internally by the optimization framework (e.g., time dilation control).

        Returns:
            UnifiedControl: Sliced view containing only augmented control variables

        Example:
            Get augmented controls::

                unified = unify_controls([thrust, time_dilation], name="u")
                aug_controls = unified.augmented  # Only time dilation
        """
        augmented_indices = self._selector_to_indices(self._augmented_slice, self.shape[0])
        true_mask = np.zeros((augmented_indices.size,), dtype=bool)
        return self._subset(augmented_indices, name=f"{self.name}[augmented]", true_mask=true_mask)

    @property
    def slice_continuous(self) -> slice:
        """Slice for continuous controls in the unified vector."""
        if self.shape[0] == 0:
            return slice(0, 0)

        mask = self._impulsive_mask()

        if not np.any(mask):
            return slice(0, self.shape[0])

        first_impulsive_idx = int(np.flatnonzero(mask)[0])
        if np.any(~mask[first_impulsive_idx:]):
            raise ValueError(
                "Unified controls are not ordered as [continuous | impulsive], "
                "cannot represent continuous controls with one slice."
            )
        return slice(0, first_impulsive_idx)

    @property
    def slice_impulsive(self) -> slice:
        """Slice for impulsive controls in the unified vector."""
        if self.shape[0] == 0:
            return slice(0, 0)

        mask = self._impulsive_mask()

        if not np.any(mask):
            return slice(self.shape[0], self.shape[0])

        first_impulsive_idx = int(np.flatnonzero(mask)[0])
        if np.any(~mask[first_impulsive_idx:]):
            raise ValueError(
                "Unified controls are not ordered as [continuous | impulsive], "
                "cannot represent impulsive controls with one slice."
            )
        return slice(first_impulsive_idx, self.shape[0])

    def append(
        self,
        other: "Optional[Control | UnifiedControl]" = None,
        *,
        min=-np.inf,
        max=np.inf,
        is_impulsive=False,
        guess=0.0,
        augmented=False,
    ) -> None:
        """Append another control or create a new control variable.

        This method allows dynamic extension of the unified control, either by appending
        another Control/UnifiedControl object or by creating a new scalar control variable
        with specified properties. Modifies the unified control in-place.

        Args:
            other (Optional[Control | UnifiedControl]): Control object to append. If None,
                creates a new scalar control variable with properties from keyword args.
            min (float): Lower bound for new scalar control (default: -inf)
            max (float): Upper bound for new scalar control (default: inf)
            guess (float): Initial guess value for new scalar control (default: 0.0)
            augmented (bool): Whether the appended control is augmented (internal) rather
                than true (user-defined). Affects _true_dim tracking. Default: False

        Returns:
            None: Modifies the unified control in-place

        Example:
            Appending a Control object::

                unified = unify_controls([thrust], name="u")
                torque = ox.Control("torque", shape=(3,), min=-1, max=1)
                unified.append(torque)
                print(unified.shape)  # (6,) - thrust (3) + torque (3)

            Creating new scalar control variables::

                unified = UnifiedControl(name="u", shape=(0,), _true_dim=0)
                unified.append(min=-1, max=1, guess=0.0)  # Add scalar control
                print(unified.shape)  # (1,)
        """
        # Import here to avoid circular imports at module level
        from openscvx.symbolic.expr.control import Control

        current_true_mask = self._selector_to_mask(self._true_slice, self.shape[0])

        if isinstance(other, (Control, UnifiedControl)):
            # Append another control object
            new_shape = (self.shape[0] + other.shape[0],)

            # Update bounds
            if self.min is not None and other.min is not None:
                new_min = np.concatenate([self.min, other.min])
            else:
                new_min = self.min

            if self.max is not None and other.max is not None:
                new_max = np.concatenate([self.max, other.max])
            else:
                new_max = self.max

            # Update guess
            if self.guess is not None and other.guess is not None:
                new_guess = np.concatenate([self.guess, other.guess], axis=1)
            else:
                new_guess = self.guess

            # Update scaling bounds (if present)
            if (
                self.scaling_min is not None
                and hasattr(other, "scaling_min")
                and other.scaling_min is not None
            ):
                new_scaling_min = np.concatenate([self.scaling_min, other.scaling_min])
            else:
                new_scaling_min = self.scaling_min

            if (
                self.scaling_max is not None
                and hasattr(other, "scaling_max")
                and other.scaling_max is not None
            ):
                new_scaling_max = np.concatenate([self.scaling_max, other.scaling_max])
            else:
                new_scaling_max = self.scaling_max

            if augmented:
                appended_true_mask = np.zeros((other.shape[0],), dtype=bool)
            elif isinstance(other, UnifiedControl):
                appended_true_mask = self._selector_to_mask(other._true_slice, other.shape[0])
            else:
                appended_true_mask = np.ones((other.shape[0],), dtype=bool)
            new_true_mask = np.concatenate([current_true_mask, appended_true_mask])

            # Per-element FOH mask for the appended block: UnifiedControl.foh_mask,
            # or Control ``parameterization`` (FOH/ZOH) converted to 1.0/0.0/nan.
            n_o = int(other.shape[0])
            if isinstance(other, UnifiedControl):
                if other.foh_mask is not None:
                    other_foh_part = other.foh_mask
                else:
                    other_foh_part = np.full(n_o, np.nan)
            else:
                param = getattr(other, "parameterization", None)
                if param == "FOH":
                    other_foh_part = np.ones(n_o)
                elif param == "ZOH":
                    other_foh_part = np.zeros(n_o)
                else:
                    other_foh_part = np.full(n_o, np.nan)

            needs_foh_mask = self.foh_mask is not None or not np.all(np.isnan(other_foh_part))
            if needs_foh_mask:
                self_part = (
                    self.foh_mask if self.foh_mask is not None else np.full(self.shape[0], np.nan)
                )
                new_foh_mask = np.concatenate([self_part, other_foh_part])
            else:
                new_foh_mask = None

            # Update all attributes in place
            self.shape = new_shape
            self.min = new_min
            self.max = new_max
            self.guess = new_guess
            self.scaling_min = new_scaling_min
            self.scaling_max = new_scaling_max
            self.foh_mask = new_foh_mask
            self._true_dim = int(np.sum(new_true_mask))
            self._true_slice = self._mask_to_selector(new_true_mask, empty_at_end=False)
            self._augmented_slice = self._mask_to_selector(~new_true_mask, empty_at_end=True)

        else:
            # Create a single new variable
            new_shape = (self.shape[0] + 1,)

            # Extend arrays
            if self.min is not None:
                self.min = np.concatenate([self.min, np.array([min])])
            if self.max is not None:
                self.max = np.concatenate([self.max, np.array([max])])
            if self.guess is not None:
                guess_arr = np.full((self.guess.shape[0], 1), guess)
                self.guess = np.concatenate([self.guess, guess_arr], axis=1)
            if self.scaling_min is not None:
                self.scaling_min = np.concatenate([self.scaling_min, np.array([min])])
            if self.scaling_max is not None:
                self.scaling_max = np.concatenate([self.scaling_max, np.array([max])])
            if self.is_impulsive is not None:
                self.is_impulsive = np.concatenate([self.is_impulsive, np.array([is_impulsive])])
            if self.foh_mask is not None:
                self.foh_mask = np.concatenate([self.foh_mask, np.array([np.nan])])

            # Update dimensions
            self.shape = new_shape
            new_true_mask = np.concatenate(
                [current_true_mask, np.array([not augmented], dtype=bool)]
            )
            self._true_dim = int(np.sum(new_true_mask))
            self._true_slice = self._mask_to_selector(new_true_mask, empty_at_end=False)
            self._augmented_slice = self._mask_to_selector(~new_true_mask, empty_at_end=True)

    def __getitem__(self, idx):
        """Get a subset of the unified control variables.

        Enables slicing of the unified control to extract subsets of control variables.
        Returns a new UnifiedControl containing only the sliced dimensions.

        Args:
            idx (slice): Slice object specifying which control dimensions to extract.
                Only simple slices with step=1 are supported.

        Returns:
            UnifiedControl: New unified control containing only the sliced dimensions

        Raises:
            NotImplementedError: If idx is not a slice, or if step != 1

        Example:
            Generate unified control object::

                unified = unify_controls([thrust, torque], name="u")

            thrust has shape (3,), torque has shape (3,)::

                first_three = unified[0:3]  # Extract thrust only
                print(first_three.shape)  # (3,)

        Note:
            The sliced control maintains all properties (bounds, guesses, etc.) for
            the selected dimensions. The _true_dim is recalculated based on which
            dimensions fall within the original true control range.
        """
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.shape[0])
            if step != 1:
                raise NotImplementedError("Step slicing not supported")

            subset_indices = np.arange(start, stop, dtype=int)
            parent_true_mask = self._selector_to_mask(self._true_slice, self.shape[0])
            subset_true_mask = parent_true_mask[start:stop]
            return self._subset(
                subset_indices,
                name=f"{self.name}[{start}:{stop}]",
                true_mask=subset_true_mask,
            )
        else:
            raise NotImplementedError("Only slice indexing is supported")

    def __repr__(self):
        """String representation of the UnifiedControl object."""
        return f"UnifiedControl('{self.name}', shape={self.shape})"
