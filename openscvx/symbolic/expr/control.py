from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .variable import Variable


class Control(Variable):
    """Control input variable for trajectory optimization problems.

    Control represents control input variables (actuator commands) in a trajectory
    optimization problem. Unlike State variables which evolve according to dynamics,
    Controls are direct decision variables that the optimizer can freely adjust
    (within specified bounds) at each time step to influence the system dynamics.

    Controls are conceptually similar to State variables but simpler - they don't
    have boundary conditions (initial/final specifications) since controls are
    typically not constrained at the endpoints. Like States, Controls support:

    - Min/max bounds to enforce actuator limits
    - Initial trajectory guesses to help the optimizer converge

    Common examples of control inputs include:

    - Thrust magnitude and direction for spacecraft/rockets
    - Throttle settings for engines
    - Steering angles for vehicles
    - Torques for robotic manipulators
    - Force/acceleration commands

    Attributes:
        name (str): Unique name identifier for this control variable
        _shape (tuple[int, ...]): Shape of the control vector (typically 1D like (3,) for 3D thrust)
        _slice (slice | None): Internal slice information for variable indexing
        _min (np.ndarray | None): Minimum bounds for each element of the control
        _max (np.ndarray | None): Maximum bounds for each element of the control
        _guess (np.ndarray | None): Initial guess for the control trajectory (n_points, n_controls)

    Example:
        Scalar throttle control bounded [0, 1]:

            throttle = Control("throttle", shape=(1,))
            throttle.min = [0.0]
            throttle.max = [1.0]
            throttle.guess = np.full((50, 1), 0.5)  # Start at 50% throttle

        3D thrust vector for spacecraft:

            thrust = Control("thrust", shape=(3,))
            thrust.min = [-10, -10, 0]    # No downward thrust
            thrust.max = [10, 10, 50]     # Limited thrust
            thrust.guess = np.zeros((50, 3))  # Initialize with zero thrust

        2D steering control (left/right, forward/backward):

            steer = Control("steer", shape=(2,))
            steer.min = [-1, -1]
            steer.max = [1, 1]
            steer.guess = np.linspace([0, 0], [0, 1], 50)  # Gradual acceleration
    """

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        *,
        min: Optional[np.ndarray] = None,
        max: Optional[np.ndarray] = None,
        impulsive: bool = False,
        allocation_matrix: np.ndarray = None,
        impulsive_nodes: Optional[list[int]] = None,
    ):
        """Initialize a Control object.

        Args:
            name: Name identifier for the control variable
            shape: Shape of the control vector (typically 1D tuple like (3,))
            min: Optional minimum bounds array (keyword-only)
            max: Optional maximum bounds array (keyword-only)
            impulsive: Whether this control is treated as impulsive
        """
        super().__init__(name, shape)
        self._scaling_min = None
        self._scaling_max = None
        self._is_impulsive = np.repeat(impulsive, shape[0])
        if impulsive_nodes is not None and not impulsive:
            raise ValueError("impulsive_nodes provided for a non-impulsive control.")
        if impulsive_nodes is not None:
            self._impulsive_nodes = [int(idx) for idx in impulsive_nodes]
        else:
            self._impulsive_nodes = None
        ## TODO: (fabio) maybe this check could be moved to ''builder.py''
        if impulsive:
            if allocation_matrix.shape[1] != shape[0]:
                raise ValueError(
                    (
                        "Allocation matrix dimensions not consistent with control dimensions.",
                        "Number of rows shall be equal to control dimension.",
                    )
                )
        self._allocation_matrix = allocation_matrix

        if min is not None:
            self.min = min
        if max is not None:
            self.max = max

    def sparsity(self, n_x: int, n_u: int) -> Tuple[np.ndarray, np.ndarray]:
        """Element-level exact sparsity: diagonal block at ``_slice``."""
        n = self._shape[0]
        S_x = np.zeros((n, n_x), dtype=bool)
        S_u = np.zeros((n, n_u), dtype=bool)
        if self._slice is not None:
            for i in range(n):
                S_u[i, self._slice.start + i] = True
        return S_x, S_u

    @property
    def scaling_min(self) -> Optional[np.ndarray]:
        """Get the scaling minimum bounds for the control variables.

        Returns:
            Array of scaling minimum values for each control variable element, or None if not set.
        """
        return self._scaling_min

    @scaling_min.setter
    def scaling_min(self, val):
        """Set the scaling minimum bounds for the control variables.

        Args:
            val: Array of scaling minimum values, must match the control shape exactly

        Raises:
            ValueError: If the shape doesn't match the control shape
        """
        if val is None:
            self._scaling_min = None
            return
        val = np.asarray(val, dtype=float)
        if val.shape != self.shape:
            raise ValueError(
                f"Control '{self.name}': scaling_min expected shape {self.shape}, got {val.shape}"
            )
        self._scaling_min = val

    @property
    def scaling_max(self) -> Optional[np.ndarray]:
        """Get the scaling maximum bounds for the control variables.

        Returns:
            Array of scaling maximum values for each control variable element, or None if not set.
        """
        return self._scaling_max

    @scaling_max.setter
    def scaling_max(self, val):
        """Set the scaling maximum bounds for the control variables.

        Args:
            val: Array of scaling maximum values, must match the control shape exactly

        Raises:
            ValueError: If the shape doesn't match the control shape
        """
        if val is None:
            self._scaling_max = None
            return
        val = np.asarray(val, dtype=float)
        if val.shape != self.shape:
            raise ValueError(
                f"Control '{self.name}': scaling_max expected shape {self.shape}, got {val.shape}"
            )
        self._scaling_max = val

    @property
    def is_impulsive(self) -> Optional[bool]:
        return self._is_impulsive

    @is_impulsive.setter
    def is_impulsive(self, val):
        if val is None:
            self._is_impulsive = np.repeat(False, self.shape)
            return
        val = np.repeat(val, self.shape)
        if val.shape != self.shape:
            raise ValueError(
                (
                    f"Impulsive controls toggles shape {val.shape} ",
                    f" does not match Control shape {self.shape}",
                )
            )
        self._is_impulsive = val

    @property
    def allocation_matrix(self) -> Optional[np.ndarray]:
        return self._allocation_matrix

    @allocation_matrix.setter
    def allocation_matrix(self, val):
        # TODO @haynec find a better way to check shape including augmentation
        self._allocation_matrix = val

    @property
    def impulsive_nodes(self) -> Optional[list[int]]:
        return self._impulsive_nodes

    @impulsive_nodes.setter
    def impulsive_nodes(self, val: Optional[list[int]]):
        if val is None:
            self._impulsive_nodes = None
            return
        if not np.any(self._is_impulsive):
            raise ValueError("impulsive_nodes can only be set for impulsive controls.")
        self._impulsive_nodes = [int(idx) for idx in val]

    def __repr__(self) -> str:
        """String representation of the Control object.

        Returns:
            Concise string showing the control name, shape and type.
        """
        return (
            "Control("
            f"'{self.name}', shape={self.shape}, impulsive={self._is_impulsive}, "
            f"allocation_matrix={self._allocation_matrix}, impulsive_nodes={self._impulsive_nodes})"
        )
