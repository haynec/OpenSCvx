"""Base class for successive convexification algorithms.

This module defines the abstract interface that all SCP algorithm implementations
must follow, along with the AlgorithmState dataclass that holds mutable state
during SCP iterations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints
    from openscvx.solvers import ConvexSolver


class _IterateView:
    """Lightweight attribute view into an AlgorithmState iterate.

    This is used to provide ergonomic access like ``state.candidate.x`` while
    the algorithm stores the actual data on the state (e.g. ``state.inter_x``).
    Supports both reading and writing.
    """

    def __init__(self, state: "AlgorithmState", prefix: str):
        self._state = state
        self._prefix = prefix

    def _get(self, name: str) -> Optional[object]:
        return getattr(self._state, f"{self._prefix}{name}", None)

    def _set(self, name: str, value: object) -> None:
        setattr(self._state, f"{self._prefix}{name}", value)

    @property
    def x(self) -> Optional[np.ndarray]:
        return self._get("x")

    @x.setter
    def x(self, value: Optional[np.ndarray]) -> None:
        self._set("x", value)

    @property
    def u(self) -> Optional[np.ndarray]:
        return self._get("u")

    @u.setter
    def u(self, value: Optional[np.ndarray]) -> None:
        self._set("u", value)

    @property
    def V(self) -> Optional[np.ndarray]:
        return self._get("V")

    @V.setter
    def V(self, value: Optional[np.ndarray]) -> None:
        self._set("V", value)

    @property
    def x_prop(self) -> Optional[np.ndarray]:
        return self._get("x_prop")

    @x_prop.setter
    def x_prop(self, value: Optional[np.ndarray]) -> None:
        self._set("x_prop", value)

    @property
    def VC(self) -> Optional[np.ndarray]:
        return self._get("VC")

    @VC.setter
    def VC(self, value: Optional[np.ndarray]) -> None:
        self._set("VC", value)

    @property
    def TR(self) -> Optional[np.ndarray]:
        return self._get("TR")

    @TR.setter
    def TR(self, value: Optional[np.ndarray]) -> None:
        self._set("TR", value)

    @property
    def lam_vc(self) -> Optional[Union[float, np.ndarray]]:
        return self._get("lam_vc")

    @lam_vc.setter
    def lam_vc(self, value: Optional[Union[float, np.ndarray]]) -> None:
        self._set("lam_vc", value)

    @property
    def lam_cost(self) -> Optional[float]:
        return self._get("lam_cost")

    @lam_cost.setter
    def lam_cost(self, value: Optional[float]) -> None:
        self._set("lam_cost", value)

    @property
    def lam_vb(self) -> Optional[float]:
        return self._get("lam_vb")

    @lam_vb.setter
    def lam_vb(self, value: Optional[float]) -> None:
        self._set("lam_vb", value)

    @property
    def J_lin(self) -> Optional[float]:
        return self._get("J_lin")

    @J_lin.setter
    def J_lin(self, value: Optional[float]) -> None:
        self._set("J_lin", value)

    @property
    def J_nonlin(self) -> Optional[float]:
        return self._get("J_nonlin")

    @J_nonlin.setter
    def J_nonlin(self, value: Optional[float]) -> None:
        self._set("J_nonlin", value)


@dataclass
class AlgorithmState:
    """Mutable state for SCP iterations.

    This dataclass holds all state that changes during the solve process.
    It stores only the evolving trajectory arrays, not the full State/Control
    objects which contain immutable configuration metadata.

    Trajectory arrays are stored in history lists, with the current guess
    accessed via properties that return the latest entry.

    A fresh instance is created for each solve, enabling easy reset functionality.

    Attributes:
        k: Current iteration number (starts at 1)
        J_tr: Current trust region cost
        J_vb: Current virtual buffer cost
        J_vc: Current virtual control cost
        lam_prox: Current trust region weight (may adapt during solve)
        lam_cost: Current cost weight (may relax during solve)
        lam_vc: Current virtual control penalty weight
        lam_vb: Current virtual buffer penalty weight
        n_x: Number of states (for unpacking V vectors)
        n_u: Number of controls (for unpacking V vectors)
        N: Number of trajectory nodes (for unpacking V vectors)
        X: List of state trajectory iterates
        U: List of control trajectory iterates
        V_history: List of discretization history
        VC_history: List of virtual control history
        TR_history: List of trust region history
        A_bar_history: List of state transition matrices
        B_bar_history: List of control influence matrices
        C_bar_history: List of control influence matrices for next node
        x_prop_history: List of propagated states
    """

    k: int
    J_tr: float
    J_vb: float
    J_vc: float
    n_x: int
    n_u: int
    N: int
    J_nonlin_history: float
    J_lin_history: float
    pred_reduction_history: float
    actual_reduction_history: float
    acceptance_ratio_history: float
    X: List[np.ndarray] = field(default_factory=list)
    U: List[np.ndarray] = field(default_factory=list)
    V_history: List[np.ndarray] = field(default_factory=list)
    VC_history: List[np.ndarray] = field(default_factory=list)
    TR_history: List[np.ndarray] = field(default_factory=list)
    lam_vc_history: List[Union[float, np.ndarray]] = field(default_factory=list)
    lam_cost_history: List[float] = field(default_factory=list)
    lam_vb_history: List[float] = field(default_factory=list)
    lam_prox_history: List[float] = field(default_factory=list)
    x_full: List[np.ndarray] = field(default_factory=list)
    x_prop_full: List[np.ndarray] = field(default_factory=list)

    @property
    def candidate(self) -> _IterateView:
        """Convenience view of the current *candidate* iterate.

        Algorithms typically store the in-progress iterate on fields prefixed
        with ``inter_`` (e.g. ``inter_x``, ``inter_u``). This view lets you access
        them ergonomically:

            - ``state.candidate.x`` -> ``state.inter_x``
            - ``state.candidate.u`` -> ``state.inter_u``

        Any missing fields return None.
        """

        return _IterateView(self, "inter_")

    def accept_solution(self) -> None:
        """Accept the current SCP iterate by updating the state in place."""
        cand = self.candidate

        if cand.x is None or cand.u is None:
            raise ValueError(
                "No candidate iterate to accept. Expected algorithm to set "
                "`state.inter_x` and `state.inter_u` before calling accept_solution()."
            )

        self.X.append(cand.x)
        self.U.append(cand.u)

        if cand.V is not None:
            self.V_history.append(cand.V)
        if cand.VC is not None:
            self.VC_history.append(cand.VC)
        if cand.TR is not None:
            self.TR_history.append(cand.TR)

        if cand.lam_vc is not None:
            self.lam_vc_history.append(cand.lam_vc)
        if cand.lam_cost is not None:
            self.lam_cost_history.append(cand.lam_cost)
        if cand.lam_vb is not None:
            self.lam_vb_history.append(cand.lam_vb)

        if cand.J_nonlin is not None:
            self.J_nonlin_history.append(cand.J_nonlin)
        if cand.J_lin is not None:
            self.J_lin_history.append(cand.J_lin)

    def clear_candidate(self) -> None:
        """Clear the candidate data after accepting."""
        self.candidate.x = None
        self.candidate.u = None
        self.candidate.V = None
        self.candidate.x_prop = None
        self.candidate.VC = None
        self.candidate.TR = None
        self.candidate.lam_vc = None
        self.candidate.lam_cost = None
        self.candidate.lam_vb = None
        self.candidate.J_lin = None
        self.candidate.J_nonlin = None

    @property
    def x(self) -> np.ndarray:
        """Get current state trajectory array.

        Returns:
            Current state trajectory guess (latest entry in history), shape (N, n_states)
        """
        return self.X[-1]

    @property
    def u(self) -> np.ndarray:
        """Get current control trajectory array.

        Returns:
            Current control trajectory guess (latest entry in history), shape (N, n_controls)
        """
        return self.U[-1]

    def x_prop(self, index: int = -1) -> np.ndarray:
        """Extract propagated state trajectory from V_history.

        Args:
            index: Index into V_history (default: -1 for latest entry)

        Returns:
            Propagated state trajectory x_prop with shape (N-1, n_x), or None if no V_history

        Example:
            After running an iteration, access the propagated states::

                problem.step()
                x_prop = problem.state.x_prop()  # Shape (N-1, n_x), latest
                x_prop_prev = problem.state.x_prop(-2)  # Previous iteration
        """
        if not self.V_history:
            return None

        # V_history contains Vmulti from discretization
        # Shape: (flattened_size, n_timesteps) where flattened_size = (N-1) * i4
        V = self.V_history[index]

        # Take final timestep and reshape to (N-1, i4)
        i4 = self.n_x + self.n_x * self.n_x + 2 * self.n_x * self.n_u
        V_final = V[:, -1].reshape(-1, i4)

        # Extract propagated state (first n_x elements of each row)
        return V_final[:, : self.n_x]

    def A_d(self, index: int = -1) -> np.ndarray:
        """Extract discretized state transition matrix from V_history.

        Args:
            index: Index into V_history (default: -1 for latest entry)

        Returns:
            Discretized state Jacobian A_d with shape (N-1, n_x, n_x), or None if no V_history

        Example:
            After running an iteration, access linearization matrices::

                problem.step()
                A_d = problem.state.A_d()  # Shape (N-1, n_x, n_x), latest
                A_d_prev = problem.state.A_d(-2)  # Previous iteration
        """
        if not self.V_history:
            return None

        # Extract indices for unpacking V vector
        i1 = self.n_x
        i2 = i1 + self.n_x * self.n_x

        # V_history contains Vmulti from discretization
        # Shape: (flattened_size, n_timesteps) where flattened_size = (N-1) * i4
        V = self.V_history[index]

        # Take final timestep and reshape to (N-1, i4)
        i4 = self.n_x + self.n_x * self.n_x + 2 * self.n_x * self.n_u
        V_final = V[:, -1].reshape(-1, i4)

        # Extract and reshape A_d matrix
        return V_final[:, i1:i2].reshape(self.N - 1, self.n_x, self.n_x)

    def B_d(self, index: int = -1) -> np.ndarray:
        """Extract discretized control influence matrix (current node) from V_history.

        Args:
            index: Index into V_history (default: -1 for latest entry)

        Returns:
            Discretized control Jacobian B_d with shape (N-1, n_x, n_u), or None if no V_history

        Example:
            After running an iteration, access linearization matrices::

                problem.step()
                B_d = problem.state.B_d()  # Shape (N-1, n_x, n_u), latest
                B_d_prev = problem.state.B_d(-2)  # Previous iteration
        """
        if not self.V_history:
            return None

        # Extract indices for unpacking V vector
        i1 = self.n_x
        i2 = i1 + self.n_x * self.n_x
        i3 = i2 + self.n_x * self.n_u

        # V_history contains Vmulti from discretization
        V = self.V_history[index]

        # Take final timestep and reshape to (N-1, i4)
        i4 = self.n_x + self.n_x * self.n_x + 2 * self.n_x * self.n_u
        V_final = V[:, -1].reshape(-1, i4)

        # Extract and reshape B_d matrix
        return V_final[:, i2:i3].reshape(self.N - 1, self.n_x, self.n_u)

    def C_d(self, index: int = -1) -> np.ndarray:
        """Extract discretized control influence matrix (next node) from V_history.

        Args:
            index: Index into V_history (default: -1 for latest entry)

        Returns:
            Discretized control Jacobian C_d with shape (N-1, n_x, n_u), or None if no V_history

        Example:
            After running an iteration, access linearization matrices::

                problem.step()
                C_d = problem.state.C_d()  # Shape (N-1, n_x, n_u), latest
                C_d_prev = problem.state.C_d(-2)  # Previous iteration
        """
        if not self.V_history:
            return None

        # Extract indices for unpacking V vector
        i2 = self.n_x + self.n_x * self.n_x
        i3 = i2 + self.n_x * self.n_u
        i4 = i3 + self.n_x * self.n_u

        # V_history contains Vmulti from discretization
        V = self.V_history[index]

        # Take final timestep and reshape to (N-1, i4)
        V_final = V[:, -1].reshape(-1, i4)

        # Extract and reshape C_d matrix
        return V_final[:, i3:i4].reshape(self.N - 1, self.n_x, self.n_u)

    @property
    def lam_prox(self) -> float:
        """Get current trust region weight.

        Returns:
            Current trust region weight (latest entry in lam_prox_history)
        """
        if not self.lam_prox_history:
            raise ValueError("lam_prox_history is empty. Initialize state using from_settings().")
        return self.lam_prox_history[-1]

    @property
    def lam_cost(self) -> float:
        """Get current cost weight.

        Returns:
            Current cost weight (latest entry in lam_cost_history)
        """
        if not self.lam_cost_history:
            raise ValueError("lam_cost_history is empty. Initialize state using from_settings().")
        return self.lam_cost_history[-1]

    @property
    def lam_vc(self) -> Union[float, np.ndarray]:
        """Get current virtual control penalty weight.

        Returns:
            Current virtual control penalty weight (latest entry in lam_vc_history)
        """
        if not self.lam_vc_history:
            raise ValueError("lam_vc_history is empty. Initialize state using from_settings().")
        return self.lam_vc_history[-1]

    @property
    def lam_vb(self) -> float:
        """Get current virtual buffer penalty weight.

        Returns:
            Current virtual buffer penalty weight (latest entry in lam_vb_history)
        """
        if not self.lam_vb_history:
            raise ValueError("lam_vb_history is empty. Initialize state using from_settings().")
        return self.lam_vb_history[-1]

    @classmethod
    def from_settings(cls, settings: "Config") -> "AlgorithmState":
        """Create initial algorithm state from configuration.

        Copies only the trajectory arrays from settings, leaving all metadata
        (bounds, boundary conditions, etc.) in the original settings object.

        Args:
            settings: Configuration object containing initial guesses and SCP parameters

        Returns:
            Fresh AlgorithmState initialized from settings with copied arrays
        """
        return cls(
            k=1,
            J_tr=1e2,
            J_vb=1e2,
            J_vc=1e2,
            n_x=settings.sim.n_states,
            n_u=settings.sim.n_controls,
            N=settings.scp.n,
            J_nonlin_history=[],
            J_lin_history=[],
            pred_reduction_history=[],
            actual_reduction_history=[],
            acceptance_ratio_history=[],
            X=[settings.sim.x.guess.copy()],
            U=[settings.sim.u.guess.copy()],
            V_history=[],
            VC_history=[],
            TR_history=[],
            lam_vc_history=[settings.scp.lam_vc],
            lam_cost_history=[settings.scp.lam_cost],
            lam_vb_history=[settings.scp.lam_vb],
            lam_prox_history=[settings.scp.lam_prox],
        )


class Algorithm(ABC):
    """Abstract base class for successive convexification algorithms.

    This class defines the interface for SCP algorithms used in trajectory
    optimization. Implementations should remain minimal and functional,
    delegating state management to the AlgorithmState dataclass.

    The two core methods mirror the SCP workflow:

    - initialize: Store compiled infrastructure and warm-start solvers
    - step: Execute one convex subproblem iteration

    Immutable components (ocp, discretization_solver, jax_constraints, etc.) are
    stored during initialize(). Mutable configuration (params, settings) is passed
    per-step to support runtime parameter updates and tolerance tuning.

    !!! tip "Statefullness"
        Avoid storing mutable iteration state (costs, weights, trajectories) on
        ``self``. All iteration state should live in :class:`AlgorithmState` or
        a subclass thereof, passed explicitly to ``step()``. This keeps algorithm
        classes stateless w.r.t. iteration, making data flow explicit and staying
        close to functional programming principles where possible.

    Example:
        Implementing a custom algorithm::

            class MyAlgorithm(Algorithm):
                def initialize(self, solver, discretization_solver,
                               jax_constraints, emitter,
                               params, settings):
                    # Store compiled infrastructure
                    self._solver = solver
                    self._discretization_solver = discretization_solver
                    self._jax_constraints = jax_constraints
                    self._emitter = emitter
                    # Warm-start with initial params/settings...

                def step(self, state, params, settings):
                    # Run one iteration using self._* and per-step params/settings
                    return converged
    """

    @abstractmethod
    def initialize(
        self,
        solver: "ConvexSolver",
        discretization_solver: callable,
        jax_constraints: "LoweredJaxConstraints",
        emitter: callable,
        params: dict,
        settings: "Config",
    ) -> None:
        """Initialize the algorithm and store compiled infrastructure.

        This method stores immutable components and performs any setup required
        before the SCP loop begins (e.g., warm-starting solvers). The params and
        settings are passed for warm-start but may change between steps.

        Args:
            solver: Convex subproblem solver (e.g., CVXPySolver)
            discretization_solver: Compiled discretization solver function
            jax_constraints: JIT-compiled JAX constraint functions
            emitter: Callback for emitting iteration progress data
            params: Problem parameters dictionary (for warm-start only)
            settings: Configuration object (for warm-start only)
        """
        ...

    @abstractmethod
    def step(
        self,
        state: AlgorithmState,
        params: dict,
        settings: "Config",
    ) -> bool:
        """Execute one iteration of the SCP algorithm.

        This method solves a single convex subproblem, updates the algorithm
        state in place, and returns whether convergence criteria are met.

        Uses stored infrastructure (ocp, discretization_solver, etc.) with
        per-step params and settings to support runtime modifications.

        Args:
            state: Mutable algorithm state (modified in place)
            params: Problem parameters dictionary (may change between steps)
            settings: Configuration object (may change between steps)

        Returns:
            True if convergence criteria are satisfied, False otherwise.
        """
        ...

    @abstractmethod
    def citation(self) -> List[str]:
        """Return BibTeX citations for this algorithm.

        Implementations should return a list of BibTeX entry strings for the
        papers that should be cited when using this algorithm.

        Returns:
            List of BibTeX citation strings.

        Example:
            Getting citations for an algorithm::

                algorithm = PenalizedTrustRegion()
                for bibtex in algorithm.citation():
                    print(bibtex)
        """
        ...
