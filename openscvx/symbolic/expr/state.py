import hashlib
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .variable import Variable


class BoundaryType(str, Enum):
    """Enumeration of boundary condition types for state variables.

    This enum allows users to specify boundary conditions using plain strings
    while maintaining type safety internally. Boundary conditions control how
    the optimizer handles initial and final state values.

    Attributes:
        FIXED (str): State value is fixed to a specific value
        FREE (str): State value is free to be optimized within bounds
        MINIMIZE (str): Objective term to minimize the state value
        MAXIMIZE (str): Objective term to maximize the state value

    Example:
        Can use either enum or string:

            BoundaryType.FIXED
            "fixed"  # Equivalent
    """

    FIXED = "fixed"
    FREE = "free"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


def Free(guess: float) -> Tuple[str, float]:
    """Create a free boundary condition tuple.

    This is a convenience function that returns a tuple ("free", guess) which
    can be used to specify free boundary conditions for State or Time objects.

    Args:
        guess: Initial guess value for the free variable.

    Returns:
        tuple: ("free", guess) tuple suitable for use in State.initial, State.final,
            or Time.initial, Time.final.

    Example:
        ```python
        pos = ox.State("pos", (3,))
        pos.final = [ox.Free(5.0), ox.Free(3.0), 10]  # First two free, third fixed

        time = ox.Time(
            initial=0.0,
            final=ox.Free(10.0),
            min=0.0,
            max=20.0
        )
        ```
    """
    return ("free", guess)


def Fixed(value: float) -> Tuple[str, float]:
    """Create a fixed boundary condition tuple.

    This is a convenience function that returns a tuple ("fixed", value) which
    can be used to explicitly specify fixed boundary conditions for State or Time objects.
    Note that plain numbers default to fixed, so this is mainly for clarity.

    Args:
        value: Fixed value for the boundary condition.

    Returns:
        tuple: ("fixed", value) tuple suitable for use in State.initial, State.final,
            or Time.initial, Time.final.

    Example:
        ```python
        pos = ox.State("pos", (3,))
        pos.final = [ox.Fixed(10.0), ox.Free(5.0), ox.Fixed(2.0)]

        # Equivalent to:
        pos.final = [10.0, ox.Free(5.0), 2.0]  # Plain numbers default to fixed
        ```
    """
    return ("fixed", value)


def Minimize(guess: float) -> Tuple[str, float]:
    """Create a minimize boundary condition tuple.

    This is a convenience function that returns a tuple ("minimize", guess) which
    can be used to specify that a boundary value should be minimized in the objective
    function for State or Time objects.

    Args:
        guess: Initial guess value for the variable to be minimized.

    Returns:
        tuple: ("minimize", guess) tuple suitable for use in State.initial, State.final,
            or Time.initial, Time.final.

    Example:
        ```python
        time = ox.Time(
            initial=0.0,
            final=ox.Minimize(10.0),  # Minimize final time
            min=0.0,
            max=20.0
        )

        fuel = ox.State("fuel", (1,))
        fuel.final = [ox.Minimize(0)]  # Minimize final fuel consumption
        ```
    """
    return ("minimize", guess)


def Maximize(guess: float) -> Tuple[str, float]:
    """Create a maximize boundary condition tuple.

    This is a convenience function that returns a tuple ("maximize", guess) which
    can be used to specify that a boundary value should be maximized in the objective
    function for State or Time objects.

    Args:
        guess: Initial guess value for the variable to be maximized.

    Returns:
        tuple: ("maximize", guess) tuple suitable for use in State.initial, State.final,
            or Time.initial, Time.final.

    Example:
        ```python
        altitude = ox.State("altitude", (1,))
        altitude.final = [ox.Maximize(100.0)]  # Maximize final altitude

        time = ox.Time(
            initial=ox.Maximize(0.0),  # Maximize initial time
            final=10.0,
            min=0.0,
            max=20.0
        )
        ```
    """
    return ("maximize", guess)


class State(Variable):
    """State variable with boundary conditions for trajectory optimization.

    State represents a dynamic state variable in a trajectory optimization problem.
    Unlike control inputs, states evolve according to dynamics constraints and can
    have boundary conditions specified at the initial and final time points.
    Like all Variables, States also support min/max bounds and initial trajectory
    guesses to help guide the optimization solver toward good solutions.

    States support four types of boundary conditions:

    - **fixed**: State value is constrained to a specific value
    - **free**: State value is optimized within the specified bounds
    - **minimize**: Adds a term to the objective function to minimize the state value
    - **maximize**: Adds a term to the objective function to maximize the state value

    Each element of a multi-dimensional state can have different boundary condition
    types, allowing for fine-grained control over the optimization.

    Attributes:
        name (str): Unique name identifier for this state variable
        _shape (tuple[int, ...]): Shape of the state vector (typically 1D like (3,) for 3D position)
        _slice (slice | None): Internal slice information for variable indexing
        _min (np.ndarray | None): Minimum bounds for state variables
        _max (np.ndarray | None): Maximum bounds for state variables
        _guess (np.ndarray | None): Initial trajectory guess
        _initial (np.ndarray | None): Initial state values with boundary condition types
        initial_type (np.ndarray | None): Array of boundary condition types for initial state
        _final (np.ndarray | None): Final state values with boundary condition types
        final_type (np.ndarray | None): Array of boundary condition types for final state

    Example:
        Scalar time state with fixed initial time, minimize final time:

            time = State("time", (1,))
            time.min = [0.0]
            time.max = [10.0]
            time.initial = [("fixed", 0.0)]
            time.final = [("minimize", 5.0)]

        3D position state with mixed boundary conditions:

            pos = State("pos", (3,))
            pos.min = [0, 0, 10]
            pos.max = [10, 10, 200]
            pos.initial = [0, ("free", 1), 50]  # x fixed, y free, z fixed
            pos.final = [10, ("free", 5), ("maximize", 150)]  # Maximize final altitude
    """

    def __init__(self, name: str, shape: Tuple[int, ...]):
        """Initialize a State object.

        Args:
            name: Name identifier for the state variable
            shape: Shape of the state vector (typically 1D tuple)
        """
        super().__init__(name, shape)
        self._initial = None
        self.initial_type = None
        self._final = None
        self.final_type = None
        self._scaling_min = None
        self._scaling_max = None

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Hash State including boundary condition types.

        Extends Variable._hash_into to include the structural metadata that
        affects the compiled problem: boundary condition types (fixed, free,
        minimize, maximize). Values are not hashed as they are runtime parameters.

        Args:
            hasher: A hashlib hash object to update
        """
        # Hash the base Variable attributes (class name, shape, slice)
        super()._hash_into(hasher)
        # Hash boundary condition types (these affect constraint structure)
        if self.initial_type is not None:
            hasher.update(b"initial_type:")
            hasher.update(str(self.initial_type.tolist()).encode())
        if self.final_type is not None:
            hasher.update(b"final_type:")
            hasher.update(str(self.final_type.tolist()).encode())

    @property
    def min(self) -> Optional[np.ndarray]:
        """Get the minimum bounds for the state variables.

        Returns:
            Array of minimum values for each state variable element.

        Example:
            Get lower bounds:

                pos = State("pos", (3,))
                pos.min = [0, 0, 10]
                print(pos.min)  # [0. 0. 10.]
        """
        return self._min

    @min.setter
    def min(self, val):
        """Set the minimum bounds for the state variables.

        Bounds are validated against any fixed initial/final conditions to ensure
        consistency.

        Args:
            val: Array of minimum values, must match the state shape exactly

        Raises:
            ValueError: If the shape doesn't match the state shape, or if fixed
                boundary conditions violate the bounds

        Example:
            Set lower bounds:

                pos = State("pos", (3,))
                pos.min = [0, 0, 10]
                pos.initial = [0, 5, 15]  # Must satisfy: 0>=0, 5>=0, 15>=10
        """
        val = np.asarray(val, dtype=float)
        if val.shape != self.shape:
            raise ValueError(f"Min shape {val.shape} does not match State shape {self.shape}")
        self._min = val
        self._check_bounds_against_initial_final()

    @property
    def max(self) -> Optional[np.ndarray]:
        """Get the maximum bounds for the state variables.

        Returns:
            Array of maximum values for each state variable element.

        Example:
            Get upper bounds:

                vel = State("vel", (3,))
                vel.max = [10, 10, 5]
                print(vel.max)  # [10. 10. 5.]
        """
        return self._max

    @max.setter
    def max(self, val):
        """Set the maximum bounds for the state variables.

        Bounds are validated against any fixed initial/final conditions to ensure
        consistency.

        Args:
            val: Array of maximum values, must match the state shape exactly

        Raises:
            ValueError: If the shape doesn't match the state shape, or if fixed
                boundary conditions violate the bounds

        Example:
            Set upper bounds:

                vel = State("vel", (3,))
                vel.max = [10, 10, 5]
                vel.final = [8, 9, 4]  # Must satisfy: 8<=10, 9<=10, 4<=5
        """
        val = np.asarray(val, dtype=float)
        if val.shape != self.shape:
            raise ValueError(f"Max shape {val.shape} does not match State shape {self.shape}")
        self._max = val
        self._check_bounds_against_initial_final()

    def _check_bounds_against_initial_final(self):
        """Validate that fixed boundary conditions respect min/max bounds.

        This internal method is automatically called when bounds or boundary
        conditions are set to ensure consistency.

        Raises:
            ValueError: If any fixed initial or final value violates the min/max bounds
        """
        for field_name, data, types in [
            ("initial", self._initial, self.initial_type),
            ("final", self._final, self.final_type),
        ]:
            if data is None or types is None:
                continue
            for i, val in np.ndenumerate(data):
                if types[i] != "Fix":
                    continue
                min_i = self._min[i] if self._min is not None else -np.inf
                max_i = self._max[i] if self._max is not None else np.inf
                if val < min_i:
                    raise ValueError(
                        f"{field_name.capitalize()} Fixed value at index {i[0]} is lower then the "
                        f"min: {val} < {min_i}"
                    )
                if val > max_i:
                    raise ValueError(
                        f"{field_name.capitalize()} Fixed value at index {i[0]} is greater then "
                        f"the max: {val} > {max_i}"
                    )

    @property
    def initial(self) -> Optional[np.ndarray]:
        """Get the initial state boundary condition values.

        Returns:
            Array of initial state values (regardless of boundary condition type),
            or None if not set.

        Note:
            Use `initial_type` to see the boundary condition types for each element.

        Example:
            Get initial state boundary conditions:

                x = State("x", (2,))
                x.initial = [0, ("free", 1)]
                print(x.initial)  # [0. 1.]
                print(x.initial_type)  # ['Fix' 'Free']
        """
        return self._initial

    @initial.setter
    def initial(self, arr):
        """Set the initial state boundary conditions.

        Each element can be specified as either a simple number (defaults to "fixed")
        or a tuple of (type, value) where type specifies the boundary condition.

        Args:
            arr: Array-like of initial conditions. Each element can be:
                - A number: Defaults to fixed boundary condition at that value
                - A tuple (type, value): Where type is one of:
                    - "fixed": Constrain state to this exact value
                    - "free": Let optimizer choose within bounds, initialize at value
                    - "minimize": Add objective term to minimize, initialize at value
                    - "maximize": Add objective term to maximize, initialize at value

        Raises:
            ValueError: If the shape doesn't match the state shape, if boundary
                condition type is invalid, or if fixed values violate bounds

        Example:
            Set initial state boundary conditions:

                pos = State("pos", (3,))
                pos.min = [0, 0, 0]
                pos.max = [10, 10, 10]
                # x fixed at 0, y free (starts at 5), z fixed at 2
                pos.initial = [0, ("free", 5), 2]

            Can also minimize/maximize boundary values:

                time = State("t", (1,))
                time.initial = [("minimize", 0)]  # Minimize initial time
        """
        # Convert to list first to handle mixed types properly
        if not isinstance(arr, (list, tuple)):
            arr = np.asarray(arr)
            if arr.shape != self.shape:
                raise ValueError(f"Shape mismatch: {arr.shape} != {self.shape}")
            arr = arr.tolist()

        # Ensure we have the right number of elements
        if len(arr) != self.shape[0]:
            raise ValueError(f"Length mismatch: got {len(arr)} elements, expected {self.shape[0]}")

        self._initial = np.zeros(self.shape, dtype=float)
        self.initial_type = np.full(self.shape, "Fix", dtype=object)

        for i, v in enumerate(arr):
            if isinstance(v, tuple) and len(v) == 2:
                # Tuple API: (type, value)
                bc_type_str, bc_value = v
                try:
                    bc_type = BoundaryType(bc_type_str)  # Validates the string
                except ValueError:
                    valid_types = [t.value for t in BoundaryType]
                    raise ValueError(
                        f"Invalid boundary condition type: {bc_type_str}. "
                        f"Valid types are: {valid_types}"
                    )
                self._initial[i] = float(bc_value)
                self.initial_type[i] = bc_type.value.capitalize()
            elif isinstance(v, (int, float, np.number)):
                # Simple number defaults to fixed
                self._initial[i] = float(v)
                self.initial_type[i] = "Fix"
            else:
                raise ValueError(
                    f"Invalid boundary condition format: {v}. "
                    f"Use a number (defaults to fixed) or tuple ('type', value) "
                    f"where type is 'fixed', 'free', 'minimize', or 'maximize'."
                )

        self._check_bounds_against_initial_final()

    @property
    def final(self) -> Optional[np.ndarray]:
        """Get the final state boundary condition values.

        Returns:
            Array of final state values (regardless of boundary condition type),
            or None if not set.

        Note:
            Use `final_type` to see the boundary condition types for each element.

        Example:
            Get final state boundary conditions:

                x = State("x", (2,))
                x.final = [10, ("minimize", 0)]
                print(x.final)  # [10. 0.]
                print(x.final_type)  # ['Fix' 'Minimize']
        """
        return self._final

    @final.setter
    def final(self, arr):
        """Set the final state boundary conditions.

        Each element can be specified as either a simple number (defaults to "fixed")
        or a tuple of (type, value) where type specifies the boundary condition.

        Args:
            arr: Array-like of final conditions. Each element can be:
                - A number: Defaults to fixed boundary condition at that value
                - A tuple (type, value): Where type is one of:
                    - "fixed": Constrain state to this exact value
                    - "free": Let optimizer choose within bounds, initialize at value
                    - "minimize": Add objective term to minimize, initialize at value
                    - "maximize": Add objective term to maximize, initialize at value

        Raises:
            ValueError: If the shape doesn't match the state shape, if boundary
                condition type is invalid, or if fixed values violate bounds

        Example:
            Set final state boundary conditionis:

                pos = State("pos", (3,))
                pos.min = [0, 0, 0]
                pos.max = [10, 10, 10]
                # x fixed at 10, y free (starts at 5), z maximize altitude
                pos.final = [10, ("free", 5), ("maximize", 8)]

            Minimize final time in time-optimal problem:

                time = State("t", (1,))
                time.final = [("minimize", 10)]
        """
        # Convert to list first to handle mixed types properly
        if not isinstance(arr, (list, tuple)):
            arr = np.asarray(arr)
            if arr.shape != self.shape:
                raise ValueError(f"Shape mismatch: {arr.shape} != {self.shape}")
            arr = arr.tolist()

        # Ensure we have the right number of elements
        if len(arr) != self.shape[0]:
            raise ValueError(f"Length mismatch: got {len(arr)} elements, expected {self.shape[0]}")

        self._final = np.zeros(self.shape, dtype=float)
        self.final_type = np.full(self.shape, "Fix", dtype=object)

        for i, v in enumerate(arr):
            if isinstance(v, tuple) and len(v) == 2:
                # Tuple API: (type, value)
                bc_type_str, bc_value = v
                try:
                    bc_type = BoundaryType(bc_type_str)  # Validates the string
                except ValueError:
                    valid_types = [t.value for t in BoundaryType]
                    raise ValueError(
                        f"Invalid boundary condition type: {bc_type_str}. "
                        f"Valid types are: {valid_types}"
                    )
                self._final[i] = float(bc_value)
                self.final_type[i] = bc_type.value.capitalize()
            elif isinstance(v, (int, float, np.number)):
                # Simple number defaults to fixed
                self._final[i] = float(v)
                self.final_type[i] = "Fix"
            else:
                raise ValueError(
                    f"Invalid boundary condition format: {v}. "
                    f"Use a number (defaults to fixed) or tuple ('type', value) "
                    f"where type is 'fixed', 'free', 'minimize', or 'maximize'."
                )

        self._check_bounds_against_initial_final()

    @property
    def scaling_min(self) -> Optional[np.ndarray]:
        """Get the scaling minimum bounds for the state variables.

        Returns:
            Array of scaling minimum values for each state variable element, or None if not set.
        """
        return self._scaling_min

    @scaling_min.setter
    def scaling_min(self, val):
        """Set the scaling minimum bounds for the state variables.

        Args:
            val: Array of scaling minimum values, must match the state shape exactly

        Raises:
            ValueError: If the shape doesn't match the state shape
        """
        if val is None:
            self._scaling_min = None
            return
        val = np.asarray(val, dtype=float)
        if val.shape != self.shape:
            raise ValueError(
                f"Scaling min shape {val.shape} does not match State shape {self.shape}"
            )
        self._scaling_min = val

    @property
    def scaling_max(self) -> Optional[np.ndarray]:
        """Get the scaling maximum bounds for the state variables.

        Returns:
            Array of scaling maximum values for each state variable element, or None if not set.
        """
        return self._scaling_max

    @scaling_max.setter
    def scaling_max(self, val):
        """Set the scaling maximum bounds for the state variables.

        Args:
            val: Array of scaling maximum values, must match the state shape exactly

        Raises:
            ValueError: If the shape doesn't match the state shape
        """
        if val is None:
            self._scaling_max = None
            return
        val = np.asarray(val, dtype=float)
        if val.shape != self.shape:
            raise ValueError(
                f"Scaling max shape {val.shape} does not match State shape {self.shape}"
            )
        self._scaling_max = val

    def __repr__(self) -> str:
        """String representation of the State object.

        Returns:
            Concise string showing the state name and shape.
        """
        return f"State('{self.name}', shape={self.shape})"
