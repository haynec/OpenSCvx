from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from openscvx.lowered.unified import UnifiedControl, UnifiedState


def get_affine_scaling_matrices(n, minimum, maximum):
    S = np.diag(np.maximum(np.ones(n), abs(minimum - maximum) / 2))
    c = (maximum + minimum) / 2
    return S, c


class DevConfig(BaseModel):
    """Configuration class for development settings.

    This class defines the parameters used for development and debugging
    purposes.

    Args:
        profiling: Whether to enable profiling for performance
            analysis. Defaults to False.
        debug: Disables all precompilation so you can place
            breakpoints and inspect values. Defaults to False.
        printing: Whether to enable printing during development.
            Defaults to True.
        verbosity: Verbosity level for iteration output.
            1 (MINIMAL): Core metrics only (iter, cost, status)
            2 (STANDARD): + timing, penalty terms (default)
            3 (FULL): + autotuning diagnostics
    """

    profiling: bool = False
    debug: bool = False
    printing: bool = True
    verbosity: int = 2

    model_config = ConfigDict(extra="forbid")


class PropagationConfig(BaseModel):
    """Configuration class for propagation settings.

    This class defines the parameters required for propagating the nonlinear
    system dynamics using the optimal control sequence.

    Args:
        inter_sample: How dense the propagation within multishot
            discretization should be. Defaults to 30.
        dt: The time step for propagation. Defaults to 0.01.
        solver: The numerical solver to use for propagation
            (e.g., "Dopri8"). Defaults to "Dopri8".
        max_tau_len: The maximum length of the time vector for
            propagation. Defaults to 1000.
        args: Additional arguments to pass to the solver.
            Defaults to an empty dictionary.
        atol: Absolute tolerance for the solver. Defaults to 1e-3.
        rtol: Relative tolerance for the solver. Defaults to 1e-6.
    """

    inter_sample: int = 30
    dt: float = 0.01
    solver: str = "Dopri8"
    max_tau_len: int = 1000
    args: Dict[str, Any] = {}
    atol: float = 1e-3
    rtol: float = 1e-6

    model_config = ConfigDict(extra="forbid")


@dataclass(init=False)
class SimConfig:
    # No class-level field declarations

    def __init__(
        self,
        x: UnifiedState,
        x_prop: UnifiedState,
        u: UnifiedControl,
        total_time: float,
        n: int,
        save_compiled: bool = False,
        ctcs_node_intervals: Optional[list] = None,
        n_states: Optional[int] = None,
        n_states_prop: Optional[int] = None,
        n_controls: Optional[int] = None,
    ):
        """
        Configuration class for simulation settings.

        This class defines the parameters required for simulating a trajectory
        optimization problem.

        Main arguments:
        These are the arguments most commonly used day-to-day.

        Args:
            x (State): State object, must have .min and .max attributes for bounds.
            x_prop (State): Propagation state object, must have .min and .max
                attributes for bounds.
            u (Control): Control object, must have .min and .max attributes for
                bounds.
            total_time (float): The total simulation time.
            n (int): Number of discretization nodes.
            save_compiled (bool): If True, save and reuse compiled solver
                functions. Defaults to False.
            ctcs_node_intervals (list, optional): Node intervals for CTCS
                constraints.
            n_states (int, optional): The number of state variables. Defaults to
                `None` (inferred from x.max).
            n_states_prop (int, optional): The number of propagation state
                variables. Defaults to `None` (inferred from x_prop.max).
            n_controls (int, optional): The number of control variables. Defaults
                to `None` (inferred from u.max).

        Note:
            You can specify custom scaling for specific states/controls using
            the `scaling_min` and `scaling_max` attributes on State, Control, and Time objects.
            If not set, the default min/max bounds will be used for scaling.
        """
        # Assign core arguments to self
        self.x = x
        self.x_prop = x_prop
        self.u = u
        self.total_time = total_time
        self.n = n
        self.save_compiled = save_compiled
        self.ctcs_node_intervals = ctcs_node_intervals
        self.n_states = n_states
        self.n_states_prop = n_states_prop
        self.n_controls = n_controls
        self._uniform_time_grid = False

        # Call post init logic
        self.__post_init__()

    def __post_init__(self):
        self.n_states = len(self.x.max)
        self.n_controls = len(self.u.max)

        # State scaling
        # Use scaling_min/max if provided, otherwise use regular min/max
        min_x = np.array(self.x.min, dtype=float)
        max_x = np.array(self.x.max, dtype=float)

        # UnifiedState now always provides full-size scaling arrays when any state has scaling
        if self.x.scaling_min is not None:
            lower_x = np.array(self.x.scaling_min, dtype=float)
        else:
            lower_x = min_x

        if self.x.scaling_max is not None:
            upper_x = np.array(self.x.scaling_max, dtype=float)
        else:
            upper_x = max_x

        S_x, c_x = get_affine_scaling_matrices(self.n_states, lower_x, upper_x)
        self.S_x = S_x
        self.c_x = c_x
        self.inv_S_x = np.diag(1 / np.diag(self.S_x))

        # Control scaling
        # Use scaling_min/max if provided, otherwise use regular min/max
        min_u = np.array(self.u.min, dtype=float)
        max_u = np.array(self.u.max, dtype=float)

        # UnifiedControl now always provides full-size scaling arrays when any control has scaling
        if self.u.scaling_min is not None:
            lower_u = np.array(self.u.scaling_min, dtype=float)
        else:
            lower_u = min_u

        if self.u.scaling_max is not None:
            upper_u = np.array(self.u.scaling_max, dtype=float)
        else:
            upper_u = max_u

        S_u, c_u = get_affine_scaling_matrices(self.n_controls, lower_u, upper_u)
        self.S_u = S_u
        self.c_u = c_u
        self.inv_S_u = np.diag(1 / np.diag(self.S_u))

    # Properties for accessing slices from unified objects
    @property
    def time_slice(self):
        """Slice for accessing time in the state vector."""
        return self.x.time_slice

    @property
    def ctcs_slice(self):
        """Slice for accessing CTCS augmented states."""
        return self.x.ctcs_slice

    @property
    def ctcs_slice_prop(self):
        """Slice for accessing CTCS augmented states in propagation."""
        return self.x_prop.ctcs_slice

    @property
    def time_dilation_slice(self):
        """Slice for accessing time dilation in the control vector."""
        return self.u.time_dilation_slice

    @property
    def true_state_slice(self):
        """Slice for accessing true (non-augmented) states."""
        return self.x._true_slice

    @property
    def true_state_slice_prop(self):
        """Slice for accessing true (non-augmented) propagation states."""
        return self.x_prop._true_slice

    @property
    def true_control_slice(self):
        """Slice for accessing true (non-augmented) controls."""
        return self.u._true_slice


@dataclass
class Config:
    sim: SimConfig
    prp: PropagationConfig
    dev: DevConfig

    # Subsections derived from the lowered problem — not user-configurable.
    _INTERNAL_FIELDS: ClassVar[frozenset] = frozenset({"sim"})

    # Maps configurable section names to their pydantic model classes.
    _SECTION_MODELS: ClassVar[Dict[str, type]] = {
        "prp": PropagationConfig,
        "dev": DevConfig,
    }

    def apply_dict(self, settings: dict) -> None:
        """Apply a nested settings dict to this :class:`Config`.

        Valid subsection names are discovered automatically from the
        dataclass fields (minus internal ones like ``sim``).

        Each subsection is validated through its pydantic model, so
        typos and wrong types are caught immediately (``extra="forbid"``).

        Example::

            config.apply_dict({
                "dev": {"printing": False, "verbosity": 1},
            })

        Args:
            settings: ``{section: {key: value, ...}, ...}``

        Raises:
            ValueError: On unknown section names.
            ValidationError: On unknown keys or wrong types within a section.
        """
        configurable = set(self._SECTION_MODELS)

        for section, values in settings.items():
            if section not in configurable:
                raise ValueError(
                    f"Unknown settings section {section!r}; expected one of {sorted(configurable)}"
                )
            if not isinstance(values, dict):
                raise ValueError(
                    f"Expected a mapping for settings.{section}, got {type(values).__name__}"
                )
            # Merge current values with overrides and re-validate through pydantic.
            current = getattr(self, section)
            model_cls = self._SECTION_MODELS[section]
            updated = model_cls.model_validate({**current.model_dump(), **values})
            setattr(self, section, updated)
