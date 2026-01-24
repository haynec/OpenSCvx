from typing import Union

import numpy as np

from openscvx.symbolic.expr.state import State


class Time(State):
    """Time state variable for trajectory optimization problems.

    Time is a specialized State that represents physical time along the trajectory.
    It is used for time-optimal control problems and problems with time-dependent
    dynamics or constraints.

    Time inherits from State and can be:
    - Added directly to the states list
    - Used in constraint expressions (e.g., `time[0] <= 5.0`)
    - Given a custom initial guess via the `guess` property

    By default, Time generates a linear interpolation guess from initial to final
    time values. This can be overridden by setting `time.guess = custom_array`.

    The time derivative is always 1.0 (i.e., `d(time)/d(tau) = 1` where tau is
    the normalized time variable).

    Attributes:
        All State attributes, plus:
        derivative (float): Always 1.0 - the time derivative in normalized coordinates.

    Example:
        Simple usage (auto-added to states):

            time = ox.Time(initial=0.0, final=10.0, min=0.0, max=20.0)
            problem = Problem(dynamics=dynamics, states=states, time=time, ...)

        Time-optimal problem (minimize final time):

            time = ox.Time(
                initial=0.0,
                final=("minimize", 10.0),  # Minimize final time, guess=10.0
                min=0.0,
                max=20.0,
            )

        Using time in constraints (add to states list):

            time = ox.Time(initial=0.0, final=10.0, min=0.0, max=20.0)
            states = [position, velocity, time]  # Include time in states

            # Now time can be used in expressions
            constraint = (time[0] <= 5.0).at([0, 1, 2])

        Custom initial guess:

            time = ox.Time(initial=0.0, final=10.0, min=0.0, max=20.0)
            time.guess = my_custom_time_trajectory  # Shape (N, 1)
    """

    def __init__(
        self,
        initial: Union[float, tuple],
        final: Union[float, tuple],
        min: float,
        max: float,
    ):
        """Initialize a Time object.

        Args:
            initial: Initial time boundary condition (float or tuple).
                Tuple format: ("free", value), ("minimize", value), or ("maximize", value).
            final: Final time boundary condition (float or tuple).
                Tuple format: ("free", value), ("minimize", value), or ("maximize", value).
            min: Minimum bound for time variable (required).
            max: Maximum bound for time variable (required).

        Raises:
            ValueError: If tuple format is invalid.
        """
        # Validate tuple format before passing to State
        for name, value in [("initial", initial), ("final", final)]:
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(f"{name} tuple must have exactly 2 elements: (type, value)")
                bc_type, bc_value = value
                if bc_type not in ["free", "minimize", "maximize"]:
                    raise ValueError(
                        f"{name} boundary condition type must be 'free', "
                        f"'minimize', or 'maximize', got '{bc_type}'"
                    )
                if not isinstance(bc_value, (int, float)):
                    raise ValueError(
                        f"{name} boundary condition value must be a number, "
                        f"got {type(bc_value).__name__}"
                    )

        # Initialize as a State with name "time" and shape (1,)
        super().__init__("time", shape=(1,))

        # Set bounds using State's property setters (expects arrays)
        self._min = np.array([min])
        self._max = np.array([max])

        # Set boundary conditions using State's property setters (expects lists)
        # The State setter handles tuple parsing
        super(Time, type(self)).initial.fset(self, [initial])
        super(Time, type(self)).final.fset(self, [final])

        # Store original values for convenience access
        self._time_initial_raw = initial
        self._time_final_raw = final
        self._time_min = min
        self._time_max = max

        # Time derivative is always 1.0 internally
        self.derivative = 1.0

        # Flag to track if user set a custom guess
        self._use_default_time_guess = True

    @property
    def guess(self):
        """Get the initial trajectory guess for time.

        If no custom guess was set, returns None and preprocessing will
        generate a linear interpolation from initial to final time.

        Returns:
            Array of shape (N, 1) if custom guess was set, else None.
        """
        if self._use_default_time_guess:
            return None
        return self._guess

    @guess.setter
    def guess(self, value):
        """Set a custom initial trajectory guess for time.

        Args:
            value: Array of shape (N, 1) with time values at each node.
        """
        if value is None:
            self._use_default_time_guess = True
            self._guess = None
        else:
            self._use_default_time_guess = False
            self._guess = np.asarray(value)

    def _generate_default_guess(self, N: int) -> np.ndarray:
        """Generate the default linear interpolation guess.

        Args:
            N: Number of discretization nodes.

        Returns:
            Array of shape (N, 1) with linear interpolation from initial to final.
        """
        # Extract guess values from boundary conditions
        if isinstance(self._time_initial_raw, tuple):
            start = self._time_initial_raw[1]
        else:
            start = self._time_initial_raw

        if isinstance(self._time_final_raw, tuple):
            end = self._time_final_raw[1]
        else:
            end = self._time_final_raw

        return np.linspace(start, end, N).reshape(-1, 1)

    # Override initial/final to return raw values for backward compatibility
    @property
    def initial(self):
        """Get the initial time boundary condition (raw value)."""
        return self._time_initial_raw

    @initial.setter
    def initial(self, val):
        """Set the initial time boundary condition."""
        # Accept either raw value or list with single element
        if isinstance(val, (list, tuple)) and len(val) == 1:
            val = val[0]
        self._time_initial_raw = val
        # Also set the State's internal representation
        super(Time, type(self)).initial.fset(self, [val])

    @property
    def final(self):
        """Get the final time boundary condition (raw value)."""
        return self._time_final_raw

    @final.setter
    def final(self, val):
        """Set the final time boundary condition."""
        # Accept either raw value or list with single element
        if isinstance(val, (list, tuple)) and len(val) == 1:
            val = val[0]
        self._time_final_raw = val
        # Also set the State's internal representation
        super(Time, type(self)).final.fset(self, [val])

    # Override min/max to return scalar values for backward compatibility
    @property
    def min(self):
        """Get the minimum bound for the time variable (scalar)."""
        return self._time_min

    @min.setter
    def min(self, val):
        """Set the minimum bound for the time variable."""
        self._time_min = float(val) if not hasattr(val, "__len__") else float(val[0])
        self._min = np.array([self._time_min])

    @property
    def max(self):
        """Get the maximum bound for the time variable (scalar)."""
        return self._time_max

    @max.setter
    def max(self, val):
        """Set the maximum bound for the time variable."""
        self._time_max = float(val) if not hasattr(val, "__len__") else float(val[0])
        self._max = np.array([self._time_max])

    # Override scaling_min/max to accept scalar values for backward compatibility
    @property
    def scaling_min(self):
        """Get the scaling minimum bound for the time variable (scalar or None)."""
        if self._scaling_min is None:
            return None
        return float(self._scaling_min[0])

    @scaling_min.setter
    def scaling_min(self, val):
        """Set the scaling minimum bound for the time variable."""
        if val is None:
            self._scaling_min = None
        else:
            self._scaling_min = np.array([float(val)])

    @property
    def scaling_max(self):
        """Get the scaling maximum bound for the time variable (scalar or None)."""
        if self._scaling_max is None:
            return None
        return float(self._scaling_max[0])

    @scaling_max.setter
    def scaling_max(self, val):
        """Set the scaling maximum bound for the time variable."""
        if val is None:
            self._scaling_max = None
        else:
            self._scaling_max = np.array([float(val)])

    def __repr__(self):
        """String representation of the Time object."""
        return f"Time(initial={self._time_initial_raw}, final={self._time_final_raw})"
