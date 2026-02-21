from typing import Optional, Union

import numpy as np

from openscvx.symbolic.expr.state import State


class Time(State):
    """Time state variable for trajectory optimization.

    Time is a State representing physical time along the trajectory. Used for
    time-optimal control and problems with time-dependent dynamics/constraints.

    Since Time is a State, it can be:
    - Used directly in constraint expressions (e.g., `time[0] <= 5.0`)
    - Added to the states list, or auto-added via the `time=` argument

    The constructor accepts scalar values for convenience, which are converted
    to arrays internally to match State's API. All parameters can also be set
    via property setters after construction.

    Attributes:
        derivative (float): Always 1.0 - time derivative in normalized coordinates.

    Example:
        Constructor style::

            time = ox.Time(initial=0.0, final=10.0, min=0.0, max=20.0)
            problem = Problem(..., time=time)

        Setter style::

            time = ox.Time()
            time.min = 0.0
            time.max = 20.0
            time.initial = 0.0
            time.final = ox.Minimize(10.0)

        Time-optimal (minimize final time)::

            time = ox.Time(
                initial=0.0,
                final=("minimize", 10.0),
                min=0.0,
                max=20.0,
            )

        Using time in constraints::

            time = ox.Time(initial=0.0, final=10.0, min=0.0, max=20.0)
            states = [position, velocity, time]
            constraint = ox.ctcs(time[0] <= 5.0)
    """

    def __init__(
        self,
        initial: Optional[Union[float, tuple]] = None,
        final: Optional[Union[float, tuple]] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ):
        """Initialize a Time state.

        All parameters are optional and can be set later via property setters.

        Args:
            initial: Initial time. Either a float (fixed) or tuple like
                ("free", value), ("minimize", value), ("maximize", value).
            final: Final time. Same format as initial.
            min: Minimum time bound.
            max: Maximum time bound.
        """
        # Skip State.__init__'s kwarg handling — we wrap scalars ourselves
        State.__init__(self, "time", shape=(1,))

        self.derivative = 1.0

        if min is not None:
            self.min = min
        if max is not None:
            self.max = max
        if initial is not None:
            self.initial = initial
        if final is not None:
            self.final = final

    @State.min.setter
    def min(self, val):
        """Set the minimum time bound. Accepts a scalar or array."""
        if isinstance(val, (int, float, np.number)):
            val = np.array([val], dtype=float)
        State.min.fset(self, val)

    @State.max.setter
    def max(self, val):
        """Set the maximum time bound. Accepts a scalar or array."""
        if isinstance(val, (int, float, np.number)):
            val = np.array([val], dtype=float)
        State.max.fset(self, val)

    @State.initial.setter
    def initial(self, val):
        """Set the initial time. Accepts a scalar, tuple, or array."""
        if not isinstance(val, (list, np.ndarray)):
            val = [val]
        State.initial.fset(self, val)

    @State.final.setter
    def final(self, val):
        """Set the final time. Accepts a scalar, tuple, or array."""
        if not isinstance(val, (list, np.ndarray)):
            val = [val]
        State.final.fset(self, val)

    def _generate_default_guess(self, N: int) -> np.ndarray:
        """Generate linear interpolation guess from initial to final time.

        Args:
            N: Number of discretization nodes.

        Returns:
            Array of shape (N, 1) with linear interpolation.
        """
        # _initial and _final hold the numeric values (State parses tuples)
        return np.linspace(self._initial[0], self._final[0], N).reshape(-1, 1)

    def __repr__(self):
        if self._initial is not None and self._final is not None:
            return f"Time(initial={self._initial[0]}, final={self._final[0]})"
        return "Time()"
