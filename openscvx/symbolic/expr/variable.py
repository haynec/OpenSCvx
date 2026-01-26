import hashlib
from typing import Optional, Tuple

import numpy as np

from .expr import Leaf


class Variable(Leaf):
    """Base class for decision variables in optimization problems.

    Variable represents decision variables (free parameters) in an optimization problem.
    These are values that the optimizer can adjust to minimize the objective function
    while satisfying constraints. Variables can have bounds (min/max) and initial guesses
    to guide the optimization process.

    Unlike Parameters (which are fixed values that can be changed between solves),
    Variables are optimized by the solver. In trajectory optimization, Variables typically
    represent discretized state or control trajectories.

    Note:
        Variable is typically not instantiated directly. Instead, use the specialized
        subclasses State (for state variables with boundary conditions) or Control
        (for control inputs). These provide additional functionality specific to
        trajectory optimization.

    Attributes:
        name (str): Name identifier for the variable
        _shape (tuple[int, ...]): Shape of the variable as a tuple (typically 1D)
        _slice (slice | None): Internal slice information for variable indexing
        _min (np.ndarray | None): Minimum bounds for each element of the variable
        _max (np.ndarray | None): Maximum bounds for each element of the variable
        _guess (np.ndarray | None): Initial guess for the variable trajectory (n_points, n_vars)

    Example:
            # Typically, use State or Control instead of Variable directly:
            pos = openscvx.State("pos", shape=(3,))
            u = openscvx.Control("u", shape=(2,))
    """

    def __init__(self, name: str, shape: Tuple[int, ...]):
        """Initialize a Variable object.

        Args:
            name: Name identifier for the variable
            shape: Shape of the variable as a tuple (typically 1D like (3,) for 3D vector)
        """
        super().__init__(name, shape)
        self._slice = None
        self._min = None
        self._max = None
        self._guess = None

    def __repr__(self) -> str:
        return f"Var({self.name!r})"

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Hash Variable using its slice (canonical position, name-invariant).

        Instead of hashing the variable name, we hash the _slice attribute
        which represents the variable's canonical position in the unified
        state/control vector. This ensures that two problems with the same
        structure but different variable names produce the same hash.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(self.__class__.__name__.encode())
        hasher.update(str(self._shape).encode())
        # Hash the slice (canonical position) - this is name-invariant
        if self._slice is not None:
            hasher.update(f"slice:{self._slice.start}:{self._slice.stop}".encode())
        else:
            raise RuntimeError(
                f"Cannot hash Variable '{self.name}' without _slice attribute. "
                "Hashing should only be called on preprocessed problems where "
                "all Variables have been assigned canonical slice positions."
            )

    @property
    def min(self) -> Optional[np.ndarray]:
        """Get the minimum bounds (lower bounds) for the variable.

        Returns:
            Array of minimum values for each element of the variable, or None if unbounded.

        Example:
                pos = Variable("pos", shape=(3,))
                pos.min = [-10, -10, 0]
                print(pos.min)  # [-10., -10., 0.]
        """
        return self._min

    @min.setter
    def min(self, arr):
        """Set the minimum bounds (lower bounds) for the variable.

        The bounds are applied element-wise to each component of the variable.
        Scalars will be broadcast to match the variable shape.

        Args:
            arr: Array of minimum values, must be broadcastable to shape (n,)
                where n is the variable dimension

        Raises:
            ValueError: If the shape of arr doesn't match the variable shape

        Example:
                pos = Variable("pos", shape=(3,))
                pos.min = -10  # Broadcasts to [-10, -10, -10]
                pos.min = [-5, -10, 0]  # Element-wise bounds
        """
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != self.shape[0]:
            raise ValueError(
                f"{self.__class__.__name__} min must be 1D with shape ({self.shape[0]},), got"
                f" {arr.shape}"
            )
        self._min = arr

    @property
    def max(self) -> Optional[np.ndarray]:
        """Get the maximum bounds (upper bounds) for the variable.

        Returns:
            Array of maximum values for each element of the variable, or None if unbounded.

        Example:
                vel = Variable("vel", shape=(3,))
                vel.max = [10, 10, 5]
                print(vel.max)  # [10., 10., 5.]
        """
        return self._max

    @max.setter
    def max(self, arr):
        """Set the maximum bounds (upper bounds) for the variable.

        The bounds are applied element-wise to each component of the variable.
        Scalars will be broadcast to match the variable shape.

        Args:
            arr: Array of maximum values, must be broadcastable to shape (n,)
                where n is the variable dimension

        Raises:
            ValueError: If the shape of arr doesn't match the variable shape

        Example:
                vel = Variable("vel", shape=(3,))
                vel.max = 10  # Broadcasts to [10, 10, 10]
                vel.max = [15, 10, 5]  # Element-wise bounds
        """
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != self.shape[0]:
            raise ValueError(
                f"{self.__class__.__name__} max must be 1D with shape ({self.shape[0]},), got"
                f" {arr.shape}"
            )
        self._max = arr

    @property
    def slice(self) -> Optional[slice]:
        """Get the slice indexing this variable in the unified state/control vector.

        After preprocessing, each variable is assigned a canonical position in the
        unified optimization vector. This property returns the slice object that
        extracts this variable's values from the unified vector.

        This is particularly useful for expert users working with byof (bring-your-own
        functions) who need to manually index into the unified x and u vectors.

        Returns:
            slice: Slice object for indexing into unified vector, or None if the
                variable hasn't been preprocessed yet.

        Example:
                velocity = ox.State("velocity", shape=(3,))
                # ... after Problem construction ...
                print(velocity.slice)  # slice(2, 5) (for example)

                # Use in byof functions
                def my_constraint(x, u, node, params):
                    vel = x[velocity.slice]  # Extract velocity from unified state
                    return jnp.sum(vel**2) - 100  # |v|^2 <= 100
        """
        return self._slice

    @property
    def guess(self) -> Optional[np.ndarray]:
        """Get the initial guess for the variable trajectory.

        The guess provides a starting point for the optimizer. A good initial guess
        can significantly improve convergence speed and help avoid local minima.

        Returns:
            2D array of shape (n_points, n_vars) representing the variable trajectory
            over time, or None if no guess is provided.

        Example:
                x = Variable("x", shape=(2,))
                # Linear interpolation from [0,0] to [10,10] over 50 points
                x.guess = np.linspace([0, 0], [10, 10], 50)
                print(x.guess.shape)  # (50, 2)
        """
        return self._guess

    @guess.setter
    def guess(self, arr):
        """Set the initial guess for the variable trajectory.

        The guess should be a 2D array where each row represents the variable value
        at a particular time point or trajectory node.

        Args:
            arr: 2D array of shape (n_points, n_vars) where n_vars matches the
                variable dimension. Can be fewer points than the final trajectory -
                the solver will interpolate as needed.

        Raises:
            ValueError: If the array is not 2D or if the second dimension doesn't
                match the variable dimension

        Example:
                pos = Variable("pos", shape=(3,))
                # Create a straight-line trajectory from origin to target
                n_points = 50
                pos.guess = np.linspace([0, 0, 0], [10, 5, 3], n_points)
        """
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"Guess must be a 2D array of shape (n_guess_points, {self.shape[0]}), got shape"
                f" {arr.shape}"
            )
        if arr.shape[1] != self.shape[0]:
            raise ValueError(
                f"Guess must have second dimension equal to variable dimension {self.shape[0]}, got"
                f" {arr.shape[1]}"
            )
        self._guess = arr

    def append(
        self,
        other: Optional["Variable"] = None,
        *,
        min: float = -np.inf,
        max: float = np.inf,
        guess: float = 0.0,
    ) -> None:
        """Append a new dimension to this variable or merge with another variable.

        This method extends the variable's dimension by either:
        1. Appending another Variable object (concatenating their dimensions)
        2. Adding a single new scalar dimension with specified bounds and guess

        The bounds and guesses of both variables are concatenated appropriately.

        Args:
            other: Another Variable object to append. If None, adds a single scalar
                dimension with the specified min/max/guess values.
            min: Minimum bound for the new dimension (only used if other is None).
                Defaults to -np.inf (unbounded below).
            max: Maximum bound for the new dimension (only used if other is None).
                Defaults to np.inf (unbounded above).
            guess: Initial guess value for the new dimension (only used if other is None).
                Defaults to 0.0.

        Example:
            Create a 2D variable and extend it to 3D:

                pos_xy = Variable("pos", shape=(2,))
                pos_xy.min = [-10, -10]
                pos_xy.max = [10, 10]
                pos_xy.append(min=0, max=100)  # Add z dimension
                print(pos_xy.shape)  # (3,)
                print(pos_xy.min)  # [-10., -10., 0.]
                print(pos_xy.max)  # [10., 10., 100.]

            Merge two variables:

                pos = Variable("pos", shape=(3,))
                vel = Variable("vel", shape=(3,))
                pos.append(vel)  # Now pos has shape (6,)
        """

        def process_array(val, is_guess=False):
            """Process input array to ensure correct shape and type.

            Args:
                val: Input value to process
                is_guess: Whether the value is a guess array

            Returns:
                Processed array with correct shape and type
            """
            arr = np.asarray(val, dtype=float)
            if is_guess:
                return np.atleast_2d(arr)
            return np.atleast_1d(arr)

        if isinstance(other, Variable):
            self._shape = (self.shape[0] + other.shape[0],)

            if self._min is not None and other._min is not None:
                self._min = np.concatenate([self._min, process_array(other._min)], axis=0)

            if self._max is not None and other._max is not None:
                self._max = np.concatenate([self._max, process_array(other._max)], axis=0)

            if self._guess is not None and other._guess is not None:
                self._guess = np.concatenate(
                    [self._guess, process_array(other._guess, is_guess=True)], axis=1
                )

        else:
            self._shape = (self.shape[0] + 1,)

            if self._min is not None:
                self._min = np.concatenate([self._min, process_array(min)], axis=0)

            if self._max is not None:
                self._max = np.concatenate([self._max, process_array(max)], axis=0)

            if self._guess is not None:
                guess_arr = process_array(guess, is_guess=True)
                if guess_arr.shape[1] != 1:
                    guess_arr = guess_arr.T
                self._guess = np.concatenate([self._guess, guess_arr], axis=1)
