from typing import Union

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
    to arrays internally to match State's API.

    Attributes:
        derivative (float): Always 1.0 - time derivative in normalized coordinates.

    Example:
        Basic usage::

            time = ox.Time(initial=0.0, final=10.0, min=0.0, max=20.0)
            problem = Problem(..., time=time)

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
        initial: Union[float, tuple],
        final: Union[float, tuple],
        min: float,
        max: float,
    ):
        """Initialize a Time state.

        Args:
            initial: Initial time. Either a float (fixed) or tuple like
                ("free", value), ("minimize", value), ("maximize", value).
            final: Final time. Same format as initial.
            min: Minimum time bound.
            max: Maximum time bound.
        """
        super().__init__("time", shape=(1,))

        self.min = np.array([min])
        self.max = np.array([max])
        self.initial = [initial]  # State's setter handles tuple parsing
        self.final = [final]

        self.derivative = 1.0

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
        return f"Time(initial={self._initial[0]}, final={self._final[0]})"
