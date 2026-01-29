from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from openscvx.algorithms.autotuning import AutotuningBase

import numpy as np

from openscvx.lowered.unified import UnifiedControl, UnifiedState


def get_affine_scaling_matrices(n, minimum, maximum):
    S = np.diag(np.maximum(np.ones(n), abs(minimum - maximum) / 2))
    c = (maximum + minimum) / 2
    return S, c


@dataclass
class DiscretizationConfig:
    def __init__(
        self,
        dis_type: str = "FOH",
        custom_integrator: bool = False,
        solver: str = "Tsit5",
        args: Optional[dict] = None,
        atol: float = 1e-3,
        rtol: float = 1e-6,
    ):
        """
        Configuration class for discretization settings.

        This class defines the parameters required for discretizing system dynamics.

        Main arguments:
        These are the arguments most commonly used day-to-day.

        Args:
            dis_type (str): The type of discretization to use (e.g., "FOH" for
                First-Order Hold). Defaults to "FOH".
            custom_integrator (bool): This enables our custom fixed-step RK45
                algorithm. This tends to be faster than Diffrax but unless you're
                going for speed, it's recommended to stick with Diffrax for
                robustness and other solver options. Defaults to False.
            solver (str): Not used if custom_integrator is enabled. Any choice of
                solver in Diffrax is valid, please refer here,
                [How to Choose a Solver](https://docs.kidger.site/diffrax/usage/
                how-to-choose-a-solver/). Defaults to "Tsit5".

        Other arguments:
        These arguments are less frequently used, and for most purposes you
        shouldn't need to understand these.

        Args:
            args (Dict): Additional arguments to pass to the solver which can be
                found [here](https://docs.kidger.site/diffrax/api/diffeqsolve/).
                Defaults to an empty dictionary.
            atol (float): Absolute tolerance for the solver. Defaults to 1e-3.
            rtol (float): Relative tolerance for the solver. Defaults to 1e-6.
        """
        self.dis_type = dis_type
        self.custom_integrator = custom_integrator
        self.solver = solver
        self.args = args if args is not None else {}
        self.atol = atol
        self.rtol = rtol


@dataclass
class DevConfig:
    def __init__(
        self,
        profiling: bool = False,
        debug: bool = False,
        printing: bool = True,
        verbosity: int = 2,
    ):
        """
        Configuration class for development settings.

        This class defines the parameters used for development and debugging
        purposes.

        Main arguments:
        These are the arguments most commonly used day-to-day.

        Args:
            profiling (bool): Whether to enable profiling for performance
                analysis. Defaults to False.
            debug (bool): Disables all precompilation so you can place
                breakpoints and inspect values. Defaults to False.
            printing (bool): Whether to enable printing during development.
                Defaults to True.
            verbosity (int): Verbosity level for iteration output.
                1 (MINIMAL): Core metrics only (iter, cost, status)
                2 (STANDARD): + timing, penalty terms (default)
                3 (FULL): + autotuning diagnostics
        """
        self.profiling = profiling
        self.debug = debug
        self.printing = printing
        self.verbosity = verbosity


@dataclass
class ConvexSolverConfig:
    def __init__(
        self,
        solver: str = "QOCO",
        solver_args: Optional[dict] = None,
        cvxpygen: bool = False,
        cvxpygen_override: bool = False,
    ):
        """
        Configuration class for convex solver settings.

        This class defines the parameters required for configuring a convex solver.

        These are the arguments most commonly used day-to-day. Generally I have
        found [QOCO](https://qoco-org.github.io/qoco/index.html) to be the most
        performant of the CVXPY solvers for these types of problems (I do have a
        bias as the author is from my group) and can handle up to SOCP's.
        [CLARABEL](https://clarabel.org/stable/) is also a great option with
        feasibility checking and can handle a few more problem types.
        [CVXPYGen](https://github.com/cvxgrp/cvxpygen) is also great if your
        problem isn't too large. I have found qocogen to be the most performant
        of the CVXPYGen solvers.

        Args:
            solver (str): The name of the CVXPY solver to use. A list of options
                can be found [here](https://www.cvxpy.org/tutorial/solvers/
                index.html). Defaults to "QOCO".
            solver_args (dict, optional): Ensure you are using the correct
                arguments for your solver as they are not all common. Additional
                arguments to configure the solver, such as tolerances. Defaults
                to {"abstol": 1e-6, "reltol": 1e-9}.
            cvxpygen (bool): Whether to enable CVXPY code generation for the
                solver. Defaults to False.
        """
        if solver_args is None:
            solver_args = {"abstol": 1e-06, "reltol": 1e-09, "enforce_dpp": True}
        self.solver = solver
        self.solver_args = (
            solver_args if solver_args is not None else {"abstol": 1e-6, "reltol": 1e-9}
        )
        self.cvxpygen = cvxpygen
        self.cvxpygen_override = cvxpygen_override


@dataclass
class PropagationConfig:
    def __init__(
        self,
        inter_sample: int = 30,
        dt: float = 0.01,
        solver: str = "Dopri8",
        max_tau_len: int = 1000,
        args: Optional[dict] = None,
        atol: float = 1e-3,
        rtol: float = 1e-6,
    ):
        """
        Configuration class for propagation settings.

        This class defines the parameters required for propagating the nonlinear
        system dynamics using the optimal control sequence.

        Main arguments:
        These are the arguments most commonly used day-to-day.

        Other arguments:
        The solver should likely not be changed as it is a high accuracy 8th-order
        Runge-Kutta method.

        Args:
            inter_sample (int): How dense the propagation within multishot
                discretization should be. Defaults to 30.
            dt (float): The time step for propagation. Defaults to 0.1.
            solver (str): The numerical solver to use for propagation
                (e.g., "Dopri8"). Defaults to "Dopri8".
            max_tau_len (int): The maximum length of the time vector for
                propagation. Defaults to 1000.
            args (Dict, optional): Additional arguments to pass to the solver.
                Defaults to an empty dictionary.
            atol (float): Absolute tolerance for the solver. Defaults to 1e-3.
            rtol (float): Relative tolerance for the solver. Defaults to 1e-6.
        """
        self.inter_sample = inter_sample
        self.dt = dt
        self.solver = solver
        self.max_tau_len = max_tau_len
        self.args = args if args is not None else {}
        self.atol = atol
        self.rtol = rtol


@dataclass(init=False)
class SimConfig:
    # No class-level field declarations

    def __init__(
        self,
        x: UnifiedState,
        x_prop: UnifiedState,
        u: UnifiedControl,
        total_time: float,
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
        self.save_compiled = save_compiled
        self.ctcs_node_intervals = ctcs_node_intervals
        self.n_states = n_states
        self.n_states_prop = n_states_prop
        self.n_controls = n_controls

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
class ScpConfig:
    def __init__(
        self,
        n: Optional[int] = None,
        n_states: Optional[int] = None,
        k_max: int = 200,
        lam_prox: float = 1e0,
        lam_vc: float = 1e1,
        lam_cost: float = 1e-1,
        lam_vb: float = 0.0,
        ep_tr: float = 1e-4,
        ep_vb: float = 1e-4,
        ep_vc: float = 1e-8,
        uniform_time_grid: bool = False,
        autotuner: Optional["AutotuningBase"] = None,
    ):
        """
        Configuration class for Sequential Convex Programming (SCP).

        This class defines the parameters used to configure the SCP solver. You
        will very likely need to modify the weights for your problem. Please
        refer to my guide [here](https://openscvx.github.io/openscvx/
        hyperparameter_tuning) for more information.

        Attributes:
            n (int): The number of discretization nodes. Defaults to `None`.
            k_max (int): The maximum number of SCP iterations. Defaults to 200.
            lam_prox (float): The trust region weight. Defaults to 1.0.
            lam_vc (float): The penalty weight for virtual control. Defaults to 1.0.
            ep_tr (float): The trust region convergence tolerance. Defaults to 1e-4.
            ep_vb (float): The boundary constraint convergence tolerance.
                Defaults to 1e-4.
            ep_vc (float): The virtual constraint convergence tolerance.
                Defaults to 1e-8.
            lam_cost (float): The weight for original cost. Defaults to 0.0.
            lam_vb (float): The weight for virtual buffer. This is only used if
                there are nonconvex nodal constraints present. Defaults to 0.0.
            uniform_time_grid (bool): Whether to use a uniform time grid.
                Defaults to `False`.
            autotuner: Optional custom autotuner instance. If not provided, defaults
                to ``AugmentedLagrangian()`` with default parameters. Useful for
                customizing parameters:

                .. code-block:: python

                    auto_tuner = ox.AugmentedLagrangian()
                    auto_tuner.rho_max = 1e7
                    problem.settings.scp.autotuner = auto_tuner

        Note:
            Autotuner parameters can be accessed and modified via the autotuner
            instance (e.g., ``problem.algorithm.autotuner.rho_max``) after
            initialization. Default values are set in the AugmentedLagrangian class.
        """
        # Initialize references first (before any setters that might use them)
        self._parent_config = None  # Will be set by Config.__post_init__
        self._n_states = n_states

        self.n = n
        self.k_max = k_max
        self.lam_prox = lam_prox
        # Store raw lam_vc; convert to array later once n_states is known
        self._lam_vc = lam_vc
        self.ep_tr = ep_tr
        self.ep_vb = ep_vb
        self.ep_vc = ep_vc
        self.lam_cost = lam_cost
        self.lam_vb = lam_vb
        self.uniform_time_grid = uniform_time_grid
        # Store autotuner via property to support lazy default construction
        self.autotuner = autotuner

        # Internal storage for autotuner instance (may be lazily created)
        # Initialized here to ensure attribute exists even if setter logic changes
        if not hasattr(self, "_autotuner"):
            self._autotuner = None

    @property
    def autotuner(self) -> Optional["AutotuningBase"]:
        """Return the configured autotuner, defaulting to AugmentedLagrangian.

        If no custom autotuner instance has been provided, this property lazily
        constructs a default :class:`AugmentedLagrangian` instance and caches it
        on the config object. This keeps the configuration as the single source
        of truth for the autotuning strategy while avoiding circular imports by
        importing inside the method body.
        """
        if self._autotuner is None:
            # Local import avoids circular dependency:
            # - autotuning imports Config
            # - Config should not eagerly import autotuning at module import time
            from openscvx.algorithms.autotuning import AugmentedLagrangian

            self._autotuner = AugmentedLagrangian()
        return self._autotuner

    @autotuner.setter
    def autotuner(self, value: Optional["AutotuningBase"]) -> None:
        """Set a custom autotuner instance or reset to default when None.

        Passing ``None`` clears the cached autotuner so that the next access
        will recreate the default :class:`AugmentedLagrangian` instance.
        """
        self._autotuner = value

    @property
    def lam_vc(self):
        """Getter for lam_vc."""
        return self._lam_vc

    @lam_vc.setter
    def lam_vc(self, value):
        """Setter for lam_vc that converts scalar to array if possible.

        If a scalar is provided and both `n` and `n_states` (from parent config)
        are available, the scalar is converted to an array of shape (n-1, n_states).
        This setter assumes it is called only after `n_states` is available.
        """
        # If it's already an array, store it directly
        if isinstance(value, np.ndarray):
            self._lam_vc = value
            return

        # If it's a scalar, require dimensions to be available
        if isinstance(value, (int, float)):
            # Both dimensions available, convert to array
            self._lam_vc = np.ones((self.n - 1, self._n_states)) * value
            return

        raise ValueError(f"Invalid value for lam_vc: {value}")

    def __post_init__(self):
        keys_to_scale = ["lam_prox", "lam_vc", "lam_cost", "lam_vb"]
        # Handle lam_vc which might be scalar or array
        scale_values = []
        for key in keys_to_scale:
            val = getattr(self, key)
            if isinstance(val, np.ndarray):
                scale_values.append(np.max(val))
            else:
                scale_values.append(val)
        scale = max(scale_values)
        for key in keys_to_scale:
            val = getattr(self, key)
            if isinstance(val, np.ndarray):
                setattr(self, key, val / scale)
            else:
                setattr(self, key, val / scale)


@dataclass
class Config:
    sim: SimConfig
    scp: ScpConfig
    cvx: ConvexSolverConfig
    dis: DiscretizationConfig
    prp: PropagationConfig
    dev: DevConfig

    def __post_init__(self):
        # Set parent config reference in scp
        self.scp._parent_config = self
