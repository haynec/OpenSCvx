from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import ConfigDict, field_validator

from openscvx.symbolic.expr.state import State, StateSpec
from openscvx.symbolic.expr.variable import (
    RESERVED_GUESS_PARAMS,
    Variable,
    _inspect_guess_callable,
)


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

    Time dilation bounds and guesses can also be configured here via
    `time_dilation_min`, `time_dilation_max`, and `time_dilation_guess`.
    These control the augmented time dilation variable that the solver adds
    internally, and can be updated between solves (e.g. in a receding horizon
    loop) without reconstructing the problem.

    !!! note
       `time_dilation_min` and `time_dilation_max` are **absolute** bounds
       on the time dilation control variable.

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

        With guess and time-dilation overrides::

            time = ox.Time(
                initial=0.0,
                final=("minimize", 10.0),
                min=0.0,
                max=20.0,
                guess=np.linspace(0, 10, 50).reshape(-1, 1),
                time_dilation_min=1.0,
                time_dilation_max=30.0,
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
        *,
        guess: Optional[np.ndarray] = None,
        time_dilation_min: Optional[float] = None,
        time_dilation_max: Optional[float] = None,
        time_dilation_guess: Optional[np.ndarray] = None,
        uniform_time_grid: bool = False,
    ):
        """Initialize a Time state.

        All parameters are optional and can be set later via property setters.

        Args:
            initial: Initial time. Either a float (fixed) or tuple like
                ("free", value), ("minimize", value), ("maximize", value).
            final: Final time. Same format as initial.
            min: Minimum time bound.
            max: Maximum time bound.
            guess: Initial guess for the time trajectory. Either an array of
                shape ``(N,)`` or ``(N, 1)``, or a callable
                ``f(variables, ...) -> array`` that returns one once N is
                known. Parameter names are matched against the problem's
                states and controls; the reserved name ``tau`` (a normalized
                ``[0, 1]`` grid) is also available for shape-only callables.
                If not provided, a linear interpolation from initial to final
                is generated.
            time_dilation_min: Absolute minimum bound for time dilation.
                Defaults to `0.3 * time_final` if not set.
            time_dilation_max: Absolute maximum bound for time dilation.
                Defaults to `3.0 * time_final` if not set.
            time_dilation_guess: Initial guess for time dilation.
                1D array of shape (N,) or 2D of shape (N, 1). Overrides
                the default finite-difference computation.
            uniform_time_grid: If True, constrain the time dilation to be
                the same across all nodes (uniform time steps). Defaults
                to False.
        """
        self.derivative = 1.0
        self.uniform_time_grid = uniform_time_grid
        self._time_dilation_min = None
        self._time_dilation_max = None
        self._time_dilation_guess = None
        self._time_dilation_guess_callable: Optional[Callable] = None
        self._time_dilation_guess_callable_params: Optional[List[Tuple[str, bool]]] = None
        self._time_dilation_control = None  # set by augmentation for live propagation

        # Wrap scalars into the array forms that State/Variable setters expect.
        # Time is always shape (1,), so a bare scalar is unambiguous.
        if isinstance(min, (int, float, np.number)):
            min = np.array([min], dtype=float)
        if isinstance(max, (int, float, np.number)):
            max = np.array([max], dtype=float)
        if initial is not None and not isinstance(initial, (list, np.ndarray)):
            initial = [initial]
        if final is not None and not isinstance(final, (list, np.ndarray)):
            final = [final]

        super().__init__("time", shape=(1,), min=min, max=max, initial=initial, final=final)

        if guess is not None:
            # The setter handles arrays (with 1D → 2D reshape) and callables uniformly.
            self.guess = guess
        if time_dilation_min is not None:
            self.time_dilation_min = time_dilation_min
        if time_dilation_max is not None:
            self.time_dilation_max = time_dilation_max
        if time_dilation_guess is not None:
            self.time_dilation_guess = time_dilation_guess

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

    @Variable.guess.setter
    def guess(self, val):
        """Set the time guess. Accepts 1D ``(N,)`` or 2D ``(N, 1)`` arrays,
        or a callable that returns either form."""
        if callable(val) and not isinstance(val, np.ndarray):
            Variable.guess.fset(self, val)
            return
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        Variable.guess.fset(self, arr)

    def _assign_guess_array(self, arr) -> None:
        """Time accepts ``(N,)`` or ``(N, 1)`` arrays — coerce before validating."""
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        super()._assign_guess_array(arr)

    def _install_default_guess_callable(self) -> None:
        """Install a default ``tau``-driven linspace from initial to final time.

        Called by preprocessing when the user has not supplied a guess.
        Captures current ``_initial`` / ``_final`` by closure so a later
        change still flows through on re-resolution.
        """
        time_self = self

        def _default_time_guess(tau):
            return np.linspace(time_self._initial[0], time_self._final[0], len(tau))

        self.guess = _default_time_guess

    @property
    def time_dilation_min(self) -> Optional[float]:
        """Absolute minimum bound for time dilation."""
        return self._time_dilation_min

    @time_dilation_min.setter
    def time_dilation_min(self, val: float):
        val = float(val)
        if val < 0:
            raise ValueError(f"time_dilation_min must be non-negative, got {val}")
        self._time_dilation_min = val
        self._sync_time_dilation_control()

    @property
    def time_dilation_max(self) -> Optional[float]:
        """Absolute maximum bound for time dilation."""
        return self._time_dilation_max

    @time_dilation_max.setter
    def time_dilation_max(self, val: float):
        self._time_dilation_max = float(val)
        self._sync_time_dilation_control()

    @property
    def time_dilation_guess(self) -> Optional[np.ndarray]:
        """Initial guess for time dilation, shape (N, 1)."""
        return self._time_dilation_guess

    @time_dilation_guess.setter
    def time_dilation_guess(self, val):
        """Set the time dilation guess. Accepts an array of shape ``(N,)`` or
        ``(N, 1)``, or a callable ``f(variables, ...) -> array`` using the
        same name-dispatch rules as ``Variable.guess`` (parameters match the
        problem's states/controls; the reserved name ``tau`` is the
        normalized ``[0, 1]`` grid)."""
        if callable(val) and not isinstance(val, np.ndarray):
            self._time_dilation_guess_callable_params = _inspect_guess_callable(
                val, "time_dilation_guess", "Time"
            )
            self._time_dilation_guess_callable = val
            self._time_dilation_guess = None
            return

        arr = np.asarray(val, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError(f"time_dilation_guess expected shape (N, 1), got {arr.shape}")
        self._time_dilation_guess = arr
        self._time_dilation_guess_callable = None
        self._time_dilation_guess_callable_params = None
        self._sync_time_dilation_control()

    def _time_dilation_dependencies(self) -> List[str]:
        """State/control names that the time-dilation guess callable depends on."""
        if self._time_dilation_guess_callable_params is None:
            return []
        return [
            name
            for name, has_default in self._time_dilation_guess_callable_params
            if name not in RESERVED_GUESS_PARAMS and not has_default
        ]

    def _resolve_time_dilation_guess(
        self, N: int, tau: np.ndarray, resolved_vars: Dict[str, np.ndarray]
    ) -> None:
        """Resolve the time-dilation guess callable, if any, into an ``(N, 1)`` array."""
        if self._time_dilation_guess_callable is None:
            return

        kwargs: Dict[str, Any] = {}
        for name, has_default in self._time_dilation_guess_callable_params or []:
            if name == "tau":
                kwargs[name] = tau
            elif name in resolved_vars:
                kwargs[name] = resolved_vars[name]
            elif has_default:
                continue
            else:
                raise ValueError(
                    f"Time 'time_dilation_guess': callable parameter '{name}' is not "
                    f"a reserved name (tau) and does not match any state or control."
                )

        try:
            result = self._time_dilation_guess_callable(**kwargs)
        except Exception as e:
            raise type(e)(f"Time 'time_dilation_guess': callable raised: {e}") from e

        arr = np.asarray(result, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape != (N, 1):
            raise ValueError(
                f"Time 'time_dilation_guess': callable returned shape {arr.shape}, "
                f"expected ({N}, 1)."
            )
        self._time_dilation_guess = arr
        self._sync_time_dilation_control()

    def _sync_time_dilation_control(self):
        """Push current time_dilation_* values to the linked Control."""
        ctrl = self._time_dilation_control
        if ctrl is None:
            return
        if self._time_dilation_min is not None:
            ctrl.min = np.array([self._time_dilation_min])
        if self._time_dilation_max is not None:
            ctrl.max = np.array([self._time_dilation_max])
        if self._time_dilation_guess is not None:
            ctrl.guess = self._time_dilation_guess

    def __repr__(self):
        parts = []
        if self._initial is not None:
            parts.append(f"initial={self._initial[0]}")
        if self._final is not None:
            parts.append(f"final={self._final[0]}")
        if self._min is not None:
            parts.append(f"min={self._min[0]}")
        if self._max is not None:
            parts.append(f"max={self._max[0]}")
        return f"Time({', '.join(parts)})"


# =============================================================================
# Pydantic spec for YAML / JSON / dict validation
# =============================================================================


def _parse_time_boundary(val: Any) -> Any:
    """Convert a YAML time boundary to the Time constructor format."""
    if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
        return (str(val[0]), float(val[1]))
    return val


class TimeSpec(StateSpec):
    """Validates Time configuration from YAML/JSON/dict input.

    Extends :class:`StateSpec` with time-specific fields.  ``name`` and
    ``shape`` are fixed (``"time"`` and ``[1]``), and ``initial``/``final``
    are required (a Time must have boundary conditions).

    YAML scalars and ``[tag, value]`` pairs are normalised into the
    ``List`` forms that :class:`StateSpec` expects via ``field_validator``
    so that no field type overrides are needed.
    """

    # Time is always named "time" with shape (1,)
    name: str = "time"
    shape: List[int] = [1]

    # Required for Time (StateSpec has these Optional; re-declared without
    # a default so pydantic treats them as required).
    initial: Optional[List[Any]]
    final: Optional[List[Any]]

    # Required for Time
    min: Optional[List[float]]
    max: Optional[List[float]]

    # Time-specific fields
    uniform_time_grid: bool = False
    time_dilation_min: Optional[float] = None
    time_dilation_max: Optional[float] = None
    time_dilation_guess: Optional[List[float]] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("initial", "final", mode="before")
    @classmethod
    def _wrap_scalar_boundary(cls, v: Any) -> Any:
        """Wrap a scalar or ``[tag, value]`` pair into a single-element list."""
        if v is None:
            return v
        if not isinstance(v, list):
            return [v]
        # bare [tag, value] pair like ["minimize", 10.0]
        if len(v) == 2 and isinstance(v[0], str) and not isinstance(v[1], list):
            return [v]
        return v

    @field_validator("min", "max", mode="before")
    @classmethod
    def _wrap_scalar_bound(cls, v: Any) -> Any:
        """Wrap a scalar float into a single-element list."""
        if v is None:
            return v
        if not isinstance(v, list):
            return [float(v)]
        return v

    def to_time(self) -> "Time":
        return Time(
            initial=_parse_time_boundary(self.initial[0]) if self.initial else None,
            final=_parse_time_boundary(self.final[0]) if self.final else None,
            min=self.min[0] if self.min else None,
            max=self.max[0] if self.max else None,
            uniform_time_grid=self.uniform_time_grid,
            time_dilation_min=self.time_dilation_min,
            time_dilation_max=self.time_dilation_max,
            time_dilation_guess=(
                np.asarray(self.time_dilation_guess, dtype=float)
                if self.time_dilation_guess is not None
                else None
            ),
        )
