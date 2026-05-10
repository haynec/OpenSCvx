import hashlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict

from .expr import Leaf

# Reserved parameter names usable in guess callables. Currently only
# ``tau`` (normalized [0, 1] grid). Any state/control with one of these
# names would shadow the reserved meaning; preprocessing validates that.
RESERVED_GUESS_PARAMS: frozenset = frozenset({"tau"})


def _inspect_guess_callable(fn: Callable, owner_name: str, owner_class: str):
    """Inspect a guess callable's signature, rejecting unsupported forms.

    Returns the list of (name, has_default) tuples for parameters the
    dispatcher will try to fill in. Errors are raised eagerly at assignment
    time so they point at the user's callable, not at preprocessing.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{owner_class} '{owner_name}': could not introspect guess callable "
            f"signature ({e}). If you used a decorator, ensure it preserves the "
            "wrapped signature (e.g. functools.wraps)."
        ) from e

    params: List[Tuple[str, bool]] = []
    for name, p in sig.parameters.items():
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            raise ValueError(
                f"{owner_class} '{owner_name}': guess callable uses '*{name}', which "
                "is incompatible with name-based dispatch. Declare each parameter "
                "explicitly."
            )
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            raise ValueError(
                f"{owner_class} '{owner_name}': guess callable uses '**{name}', which "
                "is incompatible with name-based dispatch. Declare each parameter "
                "explicitly."
            )
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ValueError(
                f"{owner_class} '{owner_name}': guess callable parameter '{name}' is "
                "positional-only; the dispatcher calls by keyword. Remove the '/' "
                "marker or rewrite as a regular parameter."
            )
        params.append((name, p.default is not inspect.Parameter.empty))
    return params


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
        self._guess_callable: Optional[Callable] = None
        self._guess_callable_params: Optional[List[Tuple[str, bool]]] = None

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
                f"{self.__class__.__name__} '{self.name}': min expected shape"
                f" ({self.shape[0]},), got {arr.shape}"
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
                f"{self.__class__.__name__} '{self.name}': max expected shape"
                f" ({self.shape[0]},), got {arr.shape}"
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
        """Get the resolved initial guess for the variable trajectory.

        The guess provides a starting point for the optimizer. A good initial guess
        can significantly improve convergence speed and help avoid local minima.

        Returns:
            2D array of shape ``(N, n)`` representing the variable trajectory,
            or ``None`` if no guess has been set, or if a callable was assigned
            but has not yet been resolved (resolution happens during problem
            build and on each ``Problem.sync`` / ``solve`` / ``reset``).

        Example:
                x = Variable("x", shape=(2,))
                # Linear interpolation from [0,0] to [10,10] over 50 points
                x.guess = np.linspace([0, 0], [10, 10], 50)
                print(x.guess.shape)  # (50, 2)
        """
        return self._guess

    @guess.setter
    def guess(self, val):
        """Set the initial guess for the variable trajectory.

        Accepts either:

        - A 2D array of shape ``(N, n)`` — assigned directly.
        - A callable ``f(variables, ...) -> array``, deferred until the
          discretization size is known. Each parameter name is dispatched
          against the other states and controls in the problem (so e.g.
          ``lambda pos: np.gradient(pos, axis=0)`` declares "this guess
          depends on the resolved ``pos`` guess"). The reserved name
          ``tau`` is the one special parameter — it receives a normalized
          ``[0, 1]`` grid of length N for shape-only callables that don't
          reference any other variable. Resolution happens inside problem
          build / sync, so the array form of ``.guess`` reads back as
          ``None`` until then.

        Assigning either form clears the other. Callable signatures are
        validated immediately so errors point at the user's lambda rather
        than at preprocessing.

        When a callable is set, the resolved array on ``.guess`` is derived
        — it is recomputed on every solve / sync, so in-place edits won't
        stick. Reassign ``.guess`` (with an array or a new callable) to
        change what's used.

        Args:
            val: 2D array of shape ``(N, n)`` where ``n`` matches the variable
                dimension, OR a callable returning such an array.

        Raises:
            ValueError: If the array is not 2D, the second dimension doesn't
                match the variable dimension, or the callable signature uses
                ``*args`` / ``**kwargs`` / positional-only parameters.

        Example:
                pos = Variable("pos", shape=(3,))
                # Eager array form
                pos.guess = np.linspace([0, 0, 0], [10, 5, 3], 50)
                # Lazy form referencing another state
                vel = Variable("vel", shape=(3,))
                vel.guess = lambda pos: np.gradient(pos, axis=0)
                # Lazy form using the reserved tau grid (no cross-var dep)
                pos.guess = lambda tau: np.outer(tau, [10, 5, 3])
        """
        if callable(val) and not isinstance(val, np.ndarray):
            self._guess_callable_params = _inspect_guess_callable(
                val, self.name, self.__class__.__name__
            )
            self._guess_callable = val
            self._guess = None
            return

        self._assign_guess_array(val)
        self._guess_callable = None
        self._guess_callable_params = None

    def _assign_guess_array(self, arr) -> None:
        """Validate and store an explicit guess array. Shared by setter and resolver."""
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"{self.__class__.__name__} '{self.name}': guess expected 2D array of shape"
                f" (N, {self.shape[0]}), got {arr.shape}"
            )
        if arr.shape[1] != self.shape[0]:
            raise ValueError(
                f"{self.__class__.__name__} '{self.name}': guess expected second dimension"
                f" {self.shape[0]}, got {arr.shape[1]}"
            )
        self._guess = arr

    def _guess_dependencies(self) -> List[str]:
        """Names this variable's guess callable depends on, excluding reserved
        names and parameters with defaults. Empty if no callable is set.
        """
        if self._guess_callable_params is None:
            return []
        return [
            name
            for name, has_default in self._guess_callable_params
            if name not in RESERVED_GUESS_PARAMS and not has_default
        ]

    def _resolve_guess(self, N: int, tau: np.ndarray, resolved_vars: Dict[str, np.ndarray]) -> None:
        """Resolve a deferred guess callable into a concrete ``(N, n)`` array.

        No-op if no callable was registered (an explicit array stays in place).
        Dispatches parameters by name: ``tau`` gets the normalized grid, names
        matching known states/controls get their resolved guess arrays. Params
        with defaults are skipped permissively when no match is found.
        """
        if self._guess_callable is None:
            return

        kwargs: Dict[str, Any] = {}
        for name, has_default in self._guess_callable_params or []:
            if name == "tau":
                kwargs[name] = tau
            elif name in resolved_vars:
                kwargs[name] = resolved_vars[name]
            elif has_default:
                continue
            else:
                raise ValueError(
                    f"{self.__class__.__name__} '{self.name}': guess callable parameter "
                    f"'{name}' is not a reserved name (tau) and does not match any "
                    f"state or control in the problem."
                )

        try:
            result = self._guess_callable(**kwargs)
        except Exception as e:
            raise type(e)(
                f"{self.__class__.__name__} '{self.name}': guess callable raised: {e}"
            ) from e

        arr = np.asarray(result, dtype=float)
        # Subclasses (e.g. Time) may override array shape coercion; route
        # through the same helper they use for the array path.
        try:
            self._assign_guess_array(arr)
        except ValueError as e:
            raise ValueError(
                f"{self.__class__.__name__} '{self.name}': guess callable returned shape "
                f"{arr.shape}, expected ({N}, {self.shape[0]})."
            ) from e

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


# =============================================================================
# Pydantic spec for YAML / JSON / dict validation
# =============================================================================


class VariableSpec(BaseModel):
    """Validates Variable configuration from YAML/JSON/dict input."""

    name: str
    shape: List[int]
    min: Optional[List[float]] = None
    max: Optional[List[float]] = None
    guess: Optional[List[List[float]]] = None
    scaling_min: Optional[List[float]] = None
    scaling_max: Optional[List[float]] = None

    model_config = ConfigDict(extra="forbid")
