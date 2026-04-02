from typing import Literal, Optional, Tuple, Union

import numpy as np

from .variable import Variable

Parameterization = Union[Literal["ZOH", "FOH", "impulsive"], None]


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
    - A single ``parameterization`` choice: ``"FOH"`` / ``"ZOH"`` for continuous
      hold type, or ``"impulsive"`` for discrete/impulsive actuation (same role
      as the former ``impulsive=True`` flag)

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
        _parameterization: ``"FOH"``, ``"ZOH"``, ``"impulsive"``, or ``None`` (defer hold to the
            discretizer's ``dis_type`` when not impulsive)

    Example:
        Scalar throttle control bounded [0, 1]:

            throttle = Control("throttle", shape=(1,))
            throttle.min = [0.0]
            throttle.max = [1.0]
            throttle.guess = np.full((50, 1), 0.5)  # Start at 50% throttle

        3D thrust vector with zero-order hold:

            thrust = Control("thrust", shape=(3,), parameterization="ZOH")
            thrust.min = [-10, -10, 0]    # No downward thrust
            thrust.max = [10, 10, 50]     # Limited thrust
            thrust.guess = np.zeros((50, 3))  # Initialize with zero thrust

        Impulsive delta-v at selected nodes:

            dv = Control("delta_v", shape=(2,), parameterization="impulsive", nodes=[0, n - 1])

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
        parameterization: Parameterization = None,
        nodes: Optional[list[int]] = None,
    ):
        """Initialize a Control object.

        Args:
            name: Name identifier for the control variable
            shape: Shape of the control vector (typically 1D tuple like (3,))
            min: Optional minimum bounds array (keyword-only)
            max: Optional maximum bounds array (keyword-only)
            parameterization: ``"FOH"`` or ``"ZOH"`` for continuous hold,
                ``"impulsive"`` for discrete/impulsive actuation (requires
                ``dynamics_discrete`` in the problem), or ``None`` to defer hold
                to the discretizer when not impulsive.
            nodes: Optional list of node indices where impulsive control is
                applied (only valid with ``parameterization="impulsive"``).
        """
        super().__init__(name, shape)
        self._scaling_min = None
        self._scaling_max = None

        if parameterization is not None and parameterization not in ("FOH", "ZOH", "impulsive"):
            raise ValueError(
                "parameterization must be 'FOH', 'ZOH', 'impulsive', or None; "
                f"got {parameterization!r}"
            )
        if nodes is not None and parameterization != "impulsive":
            raise ValueError("nodes are only valid when parameterization='impulsive'.")

        self._parameterization: Parameterization = parameterization
        if nodes is not None:
            self._nodes = [int(idx) for idx in nodes]
        else:
            self._nodes = None

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
    def parameterization(self) -> Parameterization:
        """``"FOH"``, ``"ZOH"``, ``"impulsive"``, or ``None`` (defer hold / non-impulsive)."""
        return self._parameterization

    @parameterization.setter
    def parameterization(self, val: Parameterization) -> None:
        if val is not None and val not in ("FOH", "ZOH", "impulsive"):
            raise ValueError(
                f"parameterization must be 'FOH', 'ZOH', 'impulsive', or None; got {val!r}"
            )
        if val != "impulsive" and self._nodes is not None:
            raise ValueError(
                "Cannot change parameterization away from 'impulsive' while nodes is set; "
                "clear nodes first."
            )
        self._parameterization = val

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
    def is_impulsive(self) -> np.ndarray:
        return np.repeat(self._parameterization == "impulsive", self.shape[0])

    @is_impulsive.setter
    def is_impulsive(self, val) -> None:
        if val is None:
            if self._parameterization == "impulsive":
                self._parameterization = None
                self._nodes = None
            return
        arr = np.asarray(val, dtype=bool).reshape(-1)
        if arr.size == 1:
            arr = np.repeat(bool(arr.item()), self.shape[0])
        if arr.shape != self.shape:
            raise ValueError(
                f"Impulsive controls toggles shape {arr.shape} "
                f"does not match Control shape {self.shape}"
            )
        if np.all(~arr):
            if self._parameterization == "impulsive":
                self._parameterization = None
                self._nodes = None
            return
        if np.all(arr):
            if self._parameterization in ("FOH", "ZOH"):
                raise ValueError(
                    "Cannot mark control impulsive while parameterization is FOH/ZOH; "
                    "set parameterization='impulsive'."
                )
            self._parameterization = "impulsive"
            return
        raise ValueError(
            "Per-element impulsive mask is not supported; use separate Control objects."
        )

    @property
    def nodes(self) -> Optional[list[int]]:
        return self._nodes

    @nodes.setter
    def nodes(self, val: Optional[list[int]]) -> None:
        if val is None:
            self._nodes = None
            return
        if self._parameterization != "impulsive":
            raise ValueError("nodes can only be set when parameterization='impulsive'.")
        self._nodes = [int(idx) for idx in val]

    def __repr__(self) -> str:
        """String representation of the Control object.

        Returns:
            Concise string showing the control name, shape and type.
        """
        parts = [
            f"'{self.name}'",
            f"shape={self.shape}",
            f"parameterization={self._parameterization!r}",
            f"nodes={self._nodes}",
        ]
        return f"Control({', '.join(parts)})"
