"""Base class for successive convexification algorithms.

This module defines the abstract interface that all SCP algorithm implementations
must follow, along with the AlgorithmState dataclass that holds mutable state
during SCP iterations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from openscvx.utils.printing import Column

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints
    from openscvx.solvers import ConvexSolver
    from openscvx.symbolic.expr.state import State


def _expand_lam_cost_dict(
    lam_cost_dict: Dict[str, Union[float, list, np.ndarray]],
    states: List["State"],
) -> np.ndarray:
    """Expand a ``{state_name: weight}`` dict to a per-state weight array.

    Maps user-provided per-state cost weights to a dense array of shape
    ``(n_states,)`` using each state's ``_slice``.  States without a
    minimize/maximize objective receive weight 0.  States **with** a
    minimize/maximize objective **must** appear in the dict.

    Values may be scalars (broadcast to every component of that state) or
    arrays matching the state's shape for per-component weighting, e.g.
    ``{"position": [0, 0, 1e-6]}``.

    Args:
        lam_cost_dict: Mapping from state names to cost weights (scalar or
            array matching the state's shape).
        states: List of State objects (must already have ``_slice`` assigned).

    Returns:
        np.ndarray of shape ``(n_states,)`` with per-index weights.

    Raises:
        ValueError: If the dict contains unknown state names or is missing
            entries for states that have minimize/maximize objectives.
    """
    n_states = sum(s.shape[0] if len(s.shape) > 0 else 1 for s in states)
    lam_arr = np.zeros(n_states)

    valid_names = {s.name for s in states}

    # Check for unknown keys
    unknown = set(lam_cost_dict.keys()) - valid_names
    if unknown:
        raise ValueError(
            f"lam_cost dict contains unknown state name(s): {unknown}. "
            f"Valid state names: {sorted(valid_names)}"
        )

    # Identify states that have minimize/maximize objectives.
    # initial_type/final_type are set on symbolic State objects during property
    # assignment (e.g. state.initial = ...), so they are available before lowering.
    cost_states: Set[str] = set()
    for state in states:
        if state.initial_type is not None:
            for t in state.initial_type:
                if t in ("Minimize", "Maximize"):
                    cost_states.add(state.name)
                    break
        if state.final_type is not None:
            for t in state.final_type:
                if t in ("Minimize", "Maximize"):
                    cost_states.add(state.name)
                    break

    # Check that all cost states are in the dict
    missing = cost_states - set(lam_cost_dict.keys())
    if missing:
        raise ValueError(
            f"lam_cost dict is missing weight(s) for state(s) with "
            f"minimize/maximize objectives: {missing}. All states with "
            f"cost terms must have a weight in the dict."
        )

    # Fill the array.  _slice is assigned by preprocess_symbolic_problem
    # (via collect_and_assign_slices), which runs before algorithm construction.
    for state in states:
        if state.name in lam_cost_dict:
            val = np.asarray(lam_cost_dict[state.name], dtype=float)
            n_components = state.shape[0] if len(state.shape) > 0 else 1
            if val.ndim > 0 and val.shape[0] != n_components:
                raise ValueError(
                    f"lam_cost['{state.name}'] has length {val.shape[0]}, "
                    f"expected scalar or length {n_components}"
                )
            lam_arr[state._slice] = val

    return lam_arr


@dataclass
class Weights:
    """Normalized SCP weights used internally by the algorithm and autotuner.

    This dataclass is an **internal** representation. Users should read and
    write weights through the algorithm's properties (e.g.
    ``algorithm.lam_cost``) which are the source of truth for user-facing
    values. The autotuner may mutate the normalized fields on this object
    during SCP iteration; those mutations are reflected in the weight
    histories on :class:`AlgorithmState` but do **not** alter the raw
    (user-specified) values.

    The public fields (``lam_prox``, ``lam_vc``, ``lam_cost``, ``lam_vb``)
    always hold **normalized** values (largest == 1.0).  The original
    user-specified values are preserved in private ``_raw_*`` attributes so
    that :meth:`normalize` can be re-invoked without accumulating rounding
    drift.

    Attributes:
        lam_prox: Trust region (proximal) weight (normalized).
        lam_vc: Virtual control penalty weight (normalized).
        lam_cost: Cost weight per state (normalized). Scalar or array of
            shape ``(n_states,)`` for per-state weighting.
        lam_vb: Global virtual buffer penalty weight (normalized). Scalar
            default applied to every constraint. Use ``.weight()`` on
            individual constraints for per-constraint or per-node overrides.
        lam_vb_nodal: Virtual buffer penalty weights for nodal constraints
            (normalized), shape ``(N, n_nodal)``. Set by
            :meth:`set_vb_arrays`.
        lam_vb_cross: Virtual buffer penalty weights for cross-node
            constraints (normalized), shape ``(n_cross,)``. Set by
            :meth:`set_vb_arrays`.
    """

    lam_prox: float = 1e0
    lam_vc: float = 1e1
    lam_cost: Union[float, np.ndarray] = 1e-1
    lam_vb: float = 0.0
    lam_vb_nodal: Optional[np.ndarray] = None
    lam_vb_cross: Optional[np.ndarray] = None

    def __post_init__(self):
        # Coerce lists/lists-of-lists to numpy arrays.
        if isinstance(self.lam_cost, (list, tuple)):
            self.lam_cost = np.asarray(self.lam_cost, dtype=float)

        # Snapshot the user-specified values so normalize() is idempotent.
        self._raw_lam_prox = self.lam_prox
        self._raw_lam_vc = self.lam_vc
        self._raw_lam_cost = (
            self.lam_cost.copy() if isinstance(self.lam_cost, np.ndarray) else self.lam_cost
        )
        self._raw_lam_vb = float(self.lam_vb)
        self._raw_lam_vb_nodal: Optional[np.ndarray] = None
        self._raw_lam_vb_cross: Optional[np.ndarray] = None

    def normalize(self) -> None:
        """Normalize weights so the largest equals 1.0.

        Always re-derives from the stored raw (user-specified) values,
        making this method idempotent and safe to call after updating
        any individual raw weight.
        """
        raw_cost_max = (
            float(np.max(self._raw_lam_cost))
            if isinstance(self._raw_lam_cost, np.ndarray)
            else self._raw_lam_cost
        )
        if self._raw_lam_vb_nodal is not None:
            raw_vb_max = max(
                float(np.max(self._raw_lam_vb_nodal)),
                float(np.max(self._raw_lam_vb_cross)),
            )
        else:
            raw_vb_max = self._raw_lam_vb
        scale = max(self._raw_lam_prox, self._raw_lam_vc, raw_cost_max, raw_vb_max)
        if scale > 0:
            self.lam_prox = self._raw_lam_prox / scale
            self.lam_vc = self._raw_lam_vc / scale
            self.lam_cost = self._raw_lam_cost / scale
            self.lam_vb = self._raw_lam_vb / scale
            if self._raw_lam_vb_nodal is not None:
                self.lam_vb_nodal = self._raw_lam_vb_nodal / scale
                self.lam_vb_cross = self._raw_lam_vb_cross / scale

    def set_vb_arrays(
        self,
        lam_vb_nodal: np.ndarray,
        lam_vb_cross: np.ndarray,
    ) -> None:
        """Set pre-built virtual buffer weight arrays.

        Stores the given arrays as both the current (normalized) and raw
        (pre-normalization) values. Call :meth:`normalize` afterwards to
        rescale all weights consistently.

        Args:
            lam_vb_nodal: Weight array of shape ``(N, n_nodal)``.
            lam_vb_cross: Weight array of shape ``(n_cross,)``.
        """
        self.lam_vb_nodal = lam_vb_nodal
        self.lam_vb_cross = lam_vb_cross
        self._raw_lam_vb_nodal = lam_vb_nodal.copy()
        self._raw_lam_vb_cross = lam_vb_cross.copy()


@dataclass
class CandidateIterate:
    x: Optional[np.ndarray] = None
    u: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None
    W: Optional[np.ndarray] = None
    x_prop: Optional[np.ndarray] = None
    x_prop_plus: Optional[np.ndarray] = None
    D_d: Optional[np.ndarray] = None
    E_d: Optional[np.ndarray] = None
    VC: Optional[np.ndarray] = None
    TR: Optional[np.ndarray] = None
    lam_vc: Optional[Union[float, np.ndarray]] = None
    lam_cost: Optional[Union[float, np.ndarray]] = None
    lam_vb_nodal: Optional[np.ndarray] = None
    lam_vb_cross: Optional[np.ndarray] = None
    J_lin: Optional[float] = None
    J_nonlin: Optional[float] = None


@dataclass(frozen=True, slots=True)
class DiscretizationResult:
    """Unpacked discretization data from a multi-shot discretization matrix.

    The discretization solver returns a matrix ``V`` that stores multiple blocks
    (propagated state and linearization matrices) across nodes/time. Historically,
    we stored the raw ``V`` matrices and re-unpacked them repeatedly via slicing.
    This dataclass unpacks once and makes access trivial.
    """

    V: np.ndarray  # raw V matrix, shape: (flattened_size, n_timesteps)
    x_prop: np.ndarray  # (N-1, n_x)
    A_d: np.ndarray  # (N-1, n_x, n_x)
    B_d: np.ndarray  # (N-1, n_x, n_u)
    C_d: np.ndarray  # (N-1, n_x, n_u)
    x_prop_plus: Optional[np.ndarray] = None  # (N, n_x), discrete dynamics on node states
    D_d: Optional[np.ndarray] = None  # (N, n_x, n_x), d(x_prop_plus)/d(x_node)
    E_d: Optional[np.ndarray] = None  # (N, n_x, n_u), d(x_prop_plus)/d(u_node)

    @classmethod
    def from_V(
        cls,
        V: np.ndarray,
        n_x: int,
        n_u: int,
        N: int,
    ) -> "DiscretizationResult":
        """Unpack the final timestep of a raw discretization matrix ``V``."""
        i1, i2 = n_x, n_x + n_x * n_x
        i3, i4 = i2 + n_x * n_u, i2 + 2 * n_x * n_u
        V_final = V[:, -1].reshape(-1, i4)
        return cls(
            V=np.asarray(V),
            x_prop=V_final[:, :i1],
            A_d=V_final[:, i1:i2].reshape(N - 1, n_x, n_x),
            B_d=V_final[:, i2:i3].reshape(N - 1, n_x, n_u),
            C_d=V_final[:, i3:i4].reshape(N - 1, n_x, n_u),
        )

    @classmethod
    def from_VW(
        cls,
        V: np.ndarray,
        W: np.ndarray,
        n_x: int,
        n_u: int,
        N: int,
    ) -> "DiscretizationResult":
        """Unpack continuous and impulsive discretization blocks from ``V`` and ``W``."""
        base = cls.from_V(V=V, n_x=n_x, n_u=n_u, N=N)

        W_arr = np.asarray(W)
        i_w = n_x + n_x * n_x + n_x * n_u
        i1 = n_x
        i2 = i1 + n_x * n_x
        i3 = i2 + n_x * n_u

        W_final = W_arr[:, -1].reshape(-1, i_w)

        return cls(
            V=base.V,
            x_prop=base.x_prop,
            A_d=base.A_d,
            B_d=base.B_d,
            C_d=base.C_d,
            x_prop_plus=W_final[:, :i1],
            D_d=W_final[:, i1:i2].reshape(W_final.shape[0], n_x, n_x),
            E_d=W_final[:, i2:i3].reshape(W_final.shape[0], n_x, n_u),
        )


class AutotuningBase(ABC):
    """Base class for autotuning methods in SCP algorithms.

    This class provides common functionality for calculating costs and penalties
    that are shared across different autotuning strategies (e.g., Penalized Trust
    Region, Augmented Lagrangian).

    Subclasses should implement the `update_weights` method to define their specific
    weight update strategy.

    Class Attributes:
        COLUMNS: List of Column specs for autotuner-specific metrics to display.
            Subclasses override this to add their own columns.
    """

    COLUMNS: List[Column] = []

    @staticmethod
    def calculate_cost_from_state(
        x: np.ndarray,
        settings: "Config",
        lam_cost: Union[float, np.ndarray] = 1.0,
    ) -> float:
        """Calculate cost from state vector based on final_type and initial_type.

        Args:
            x: State trajectory array (n_nodes, n_states)
            settings: Configuration object containing state types
            lam_cost: Per-state cost weight. Scalar (applied uniformly) or
                array of shape ``(n_states,)`` for per-state weighting.

        Returns:
            float: Computed cost (weighted by lam_cost)
        """
        scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
        lam = np.asarray(lam_cost)
        cost = 0.0
        for i in range(settings.sim.n_states):
            w = float(lam[i]) if lam.ndim > 0 else float(lam)
            if settings.sim.x.final_type[i] == "Minimize":
                cost += w * scaled_x[-1, i]
            if settings.sim.x.final_type[i] == "Maximize":
                cost -= w * scaled_x[-1, i]
            if settings.sim.x.initial_type[i] == "Minimize":
                cost += w * scaled_x[0, i]
            if settings.sim.x.initial_type[i] == "Maximize":
                cost -= w * scaled_x[0, i]
        return cost

    @staticmethod
    def calculate_nonlinear_penalty(
        x_prop: np.ndarray,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        lam_vc: np.ndarray,
        lam_vb_nodal: np.ndarray,
        lam_vb_cross: np.ndarray,
        lam_cost: Union[float, np.ndarray],
        nodal_constraints: "LoweredJaxConstraints",
        params: dict,
        settings: "Config",
    ) -> Tuple[float, float, float]:
        """Calculate nonlinear penalty components.

        This method computes three penalty components:
        1. Cost penalty: weighted original cost
        2. Virtual control penalty: penalty for dynamics violations
        3. Nodal penalty: penalty for constraint violations

        Args:
            x_prop: Propagated state (n_nodes-1, n_states)
            x_bar: Previous iteration state (n_nodes, n_states)
            u_bar: Solution control (n_nodes, n_controls)
            lam_vc: Virtual control weight (scalar or matrix)
            lam_vb_nodal: Virtual buffer penalty weights for nodal
                constraints, shape ``(N, n_nodal)``.
            lam_vb_cross: Virtual buffer penalty weights for cross-node
                constraints, shape ``(n_cross,)``.
            lam_cost: Cost weight. Scalar (applied uniformly) or
                array of shape ``(n_states,)`` for per-state weighting.
            nodal_constraints: Lowered JAX constraints
            params: Dictionary of problem parameters
            settings: Configuration object

        Returns:
            Tuple of (nonlinear_cost, nonlinear_penalty, nodal_penalty):
                - nonlinear_cost: Weighted cost component
                - nonlinear_penalty: Virtual control penalty
                - nodal_penalty: Constraint violation penalty
        """
        nodal_penalty = 0.0

        # Evaluate nodal constraints
        for idx, constraint in enumerate(nodal_constraints.nodal):
            # Nodal constraint function is vmapped: func(x, u, node, params)
            # When called with arrays, it evaluates at all nodes
            g = constraint.func(x_bar, u_bar, 0, params)
            # Only sum violations at nodes where constraint is enforced
            if constraint.nodes is not None:
                nodes_array = np.array(constraint.nodes)
                g_filtered = g[nodes_array]
                w = lam_vb_nodal[nodes_array, idx]
            else:
                g_filtered = g
                w = lam_vb_nodal[:, idx]
            nodal_penalty += np.sum(w * np.maximum(0, g_filtered))

        # Evaluate cross-node constraints
        for idx, constraint in enumerate(nodal_constraints.cross_node):
            w = lam_vb_cross[idx]
            # Cross-node constraint function signature: func(X, U, params)
            # No node argument - operates on full trajectory
            g = constraint.func(x_bar, u_bar, params)
            # Cross-node constraints return scalar or array, sum all violations
            nodal_penalty += w * np.sum(np.maximum(0, g))

        # lam_cost weighting is applied inside calculate_cost_from_state,
        # so the returned cost is already weighted (no outer multiplication).
        cost = AutotuningBase.calculate_cost_from_state(x_bar, settings, lam_cost)
        x_diff = settings.sim.inv_S_x @ (x_bar[1:, :] - x_prop).T

        return cost, np.sum(lam_vc * np.abs(x_diff.T)), nodal_penalty

    @abstractmethod
    def update_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        settings: "Config",
        params: dict,
        weights: "Weights",
    ) -> str:
        """Update SCP weights and cost parameters based on iteration state.

        This method is called each iteration to adapt weights based on the
        current solution quality and constraint satisfaction.

        Args:
            state: Solver state containing current weight values (mutated in place)
            candidate: Candidate iterate from the current subproblem solve
            nodal_constraints: Lowered JAX constraints
            settings: Configuration object containing adaptation parameters
            params: Dictionary of problem parameters
            weights: Normalized initial weights from the algorithm

        Returns:
            str: Adaptive state string describing the update action (e.g., "Accept Lower")
        """

        pass


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
        lam_vb_nodal: Current per-node nodal virtual buffer penalty weights
        lam_vb_cross: Current cross-node virtual buffer penalty weights
        n_x: Number of states (for unpacking V vectors)
        n_u: Number of controls (for unpacking V vectors)
        N: Number of trajectory nodes (for unpacking V vectors)
        X: List of state trajectory iterates
        U: List of control trajectory iterates
        discretizations: List of unpacked discretization results
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
    J_nonlin_history: List[float]
    J_lin_history: List[float]
    pred_reduction_history: List[float]
    actual_reduction_history: List[float]
    acceptance_ratio_history: List[float]
    X: List[np.ndarray] = field(default_factory=list)
    U: List[np.ndarray] = field(default_factory=list)
    discretizations: List[DiscretizationResult] = field(default_factory=list)
    VC_history: List[np.ndarray] = field(default_factory=list)
    TR_history: List[np.ndarray] = field(default_factory=list)
    lam_vc_history: List[Union[float, np.ndarray]] = field(default_factory=list)
    lam_cost_history: List[Union[float, np.ndarray]] = field(default_factory=list)
    lam_vb_nodal_history: List[np.ndarray] = field(default_factory=list)
    lam_vb_cross_history: List[np.ndarray] = field(default_factory=list)
    lam_prox_history: List[float] = field(default_factory=list)
    x_full: List[np.ndarray] = field(default_factory=list)
    x_prop_full: List[np.ndarray] = field(default_factory=list)

    def accept_solution(self, cand: CandidateIterate) -> None:
        """Accept the given candidate iterate by updating the state in place."""

        if cand.x is None or cand.u is None:
            raise ValueError(
                "No candidate iterate to accept. Expected algorithm to set "
                "`cand.x` and `cand.u` before calling accept_solution()."
            )

        self.X.append(cand.x)
        self.U.append(cand.u)

        if cand.V is not None:
            if cand.W is not None:
                self.discretizations.append(
                    DiscretizationResult.from_VW(
                        cand.V,
                        cand.W,
                        n_x=self.n_x,
                        n_u=self.n_u,
                        N=self.N,
                    )
                )
            else:
                self.discretizations.append(
                    DiscretizationResult.from_V(
                        cand.V,
                        n_x=self.n_x,
                        n_u=self.n_u,
                        N=self.N,
                    )
                )
        if cand.VC is not None:
            self.VC_history.append(cand.VC)
        if cand.TR is not None:
            self.TR_history.append(cand.TR)

        if cand.lam_vc is not None:
            self.lam_vc_history.append(cand.lam_vc)
        if cand.lam_cost is not None:
            self.lam_cost_history.append(cand.lam_cost)
        if cand.lam_vb_nodal is not None:
            self.lam_vb_nodal_history.append(cand.lam_vb_nodal)
        if cand.lam_vb_cross is not None:
            self.lam_vb_cross_history.append(cand.lam_vb_cross)

        if cand.J_nonlin is not None:
            self.J_nonlin_history.append(cand.J_nonlin)
        if cand.J_lin is not None:
            self.J_lin_history.append(cand.J_lin)

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

    def add_discretization(self, V: np.ndarray) -> None:
        """Append a raw discretization matrix as an unpacked result."""
        self.discretizations.append(
            DiscretizationResult.from_V(V, n_x=self.n_x, n_u=self.n_u, N=self.N)
        )

    def add_impulsive_discretization(
        self,
        W: np.ndarray,
    ) -> None:
        """Attach impulsive discretization data to the latest discretization entry."""
        if not self.discretizations:
            raise ValueError(
                "Cannot attach impulsive discretization before adding the base discretization."
            )
        last = self.discretizations[-1]
        self.discretizations[-1] = DiscretizationResult.from_VW(
            V=last.V,
            W=W,
            n_x=self.n_x,
            n_u=self.n_u,
            N=self.N,
        )

    @property
    def V_history(self) -> List[np.ndarray]:
        """Backward-compatible view of raw discretization matrices.

        Note:
            This is a read-only view. Internal code should prefer
            ``state.discretizations``.
        """
        return [d.V for d in self.discretizations]

    def x_prop(self, index: int = -1) -> np.ndarray:
        """Extract propagated state trajectory from the discretization history.

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
        if not self.discretizations:
            return None
        return self.discretizations[index].x_prop

    def A_d(self, index: int = -1) -> np.ndarray:
        """Extract discretized state transition matrix from discretizations.

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
        if not self.discretizations:
            return None
        return self.discretizations[index].A_d

    def B_d(self, index: int = -1) -> np.ndarray:
        """Extract discretized control influence matrix (current node).

        Args:
            index: Index into discretization history (default: -1 for latest entry)

        Returns:
            Discretized control Jacobian B_d with shape (N-1, n_x, n_u), or None if empty.

        Example:
            After running an iteration, access linearization matrices::

                problem.step()
                B_d = problem.state.B_d()  # Shape (N-1, n_x, n_u), latest
                B_d_prev = problem.state.B_d(-2)  # Previous iteration
        """
        if not self.discretizations:
            return None
        return self.discretizations[index].B_d

    def C_d(self, index: int = -1) -> np.ndarray:
        """Extract discretized control influence matrix (next node).

        Args:
            index: Index into discretization history (default: -1 for latest entry)

        Returns:
            Discretized control Jacobian C_d with shape (N-1, n_x, n_u), or None if empty.

        Example:
            After running an iteration, access linearization matrices::

                problem.step()
                C_d = problem.state.C_d()  # Shape (N-1, n_x, n_u), latest
                C_d_prev = problem.state.C_d(-2)  # Previous iteration
        """
        if not self.discretizations:
            return None
        return self.discretizations[index].C_d

    def x_prop_plus(self, index: int = -1) -> np.ndarray:
        """Extract discrete dynamics evaluated at x_prop."""
        if not self.discretizations:
            return None
        return self.discretizations[index].x_prop_plus

    def D_d(self, index: int = -1) -> np.ndarray:
        """Extract Jacobian of x_prop_plus w.r.t. x_prop."""
        if not self.discretizations:
            return None
        return self.discretizations[index].D_d

    def E_d(self, index: int = -1) -> np.ndarray:
        """Extract Jacobian of x_prop_plus w.r.t. discrete controls."""
        if not self.discretizations:
            return None
        return self.discretizations[index].E_d

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
    def lam_cost(self) -> Union[float, np.ndarray]:
        """Get current cost weight.

        Returns:
            Current cost weight (latest entry in lam_cost_history).
            Scalar or array of shape ``(n_states,)`` for per-state weighting.
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
    def lam_vb_nodal(self) -> np.ndarray:
        """Get current virtual buffer penalty weights for nodal constraints.

        Returns:
            Array of shape ``(N, n_nodal_constraints)``.
        """
        if not self.lam_vb_nodal_history:
            raise ValueError(
                "lam_vb_nodal_history is empty. Initialize state using from_settings()."
            )
        return self.lam_vb_nodal_history[-1]

    @property
    def lam_vb_cross(self) -> np.ndarray:
        """Get current virtual buffer penalty weights for cross-node constraints.

        Returns:
            Array of shape ``(n_cross_node_constraints,)``.
        """
        if not self.lam_vb_cross_history:
            raise ValueError(
                "lam_vb_cross_history is empty. Initialize state using from_settings()."
            )
        return self.lam_vb_cross_history[-1]

    @classmethod
    def from_settings(
        cls,
        settings: "Config",
        weights: "Weights",
    ) -> "AlgorithmState":
        """Create initial algorithm state from configuration.

        Copies only the trajectory arrays from settings, leaving all metadata
        (bounds, boundary conditions, etc.) in the original settings object.

        Args:
            settings: Configuration object containing initial guesses and SCP parameters
            weights: Normalized initial weights from the algorithm. The scalar
                ``lam_vc`` is expanded to an ``(N-1, n_states)`` array here.

        Returns:
            Fresh AlgorithmState initialized from settings with copied arrays
        """
        n = settings.sim.n
        n_states = settings.sim.n_states
        lam_vc_array = np.ones((n - 1, n_states)) * weights.lam_vc

        # Expand scalar lam_cost to per-state array
        if isinstance(weights.lam_cost, np.ndarray):
            lam_cost_init = weights.lam_cost.copy()
        else:
            lam_cost_init = np.full(n_states, weights.lam_cost)

        return cls(
            k=1,
            J_tr=1e2,
            J_vb=1e2,
            J_vc=1e2,
            n_x=n_states,
            n_u=settings.sim.n_controls,
            N=n,
            J_nonlin_history=[],
            J_lin_history=[],
            pred_reduction_history=[],
            actual_reduction_history=[],
            acceptance_ratio_history=[],
            X=[settings.sim.x.guess.copy()],
            U=[settings.sim.u.guess.copy()],
            discretizations=[],
            VC_history=[],
            TR_history=[],
            lam_vc_history=[lam_vc_array],
            lam_cost_history=[lam_cost_init],
            lam_vb_nodal_history=[weights.lam_vb_nodal.copy()],
            lam_vb_cross_history=[weights.lam_vb_cross.copy()],
            lam_prox_history=[weights.lam_prox],
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

    Attributes:
        weights: Normalized SCP weights used by the algorithm and autotuner.
            Subclasses must set this in ``__init__``.
        k_max: Maximum number of SCP iterations.
            Subclasses must set this in ``__init__``.
    """

    #: Normalized SCP weights. Subclasses must set this in ``__init__``.
    weights: Weights

    #: Maximum number of SCP iterations. Subclasses must set this in ``__init__``.
    k_max: int

    @staticmethod
    def _resolve_lam_cost(
        lam_cost: Union[float, Dict[str, float]],
        states: Optional[List["State"]] = None,
    ) -> Union[float, np.ndarray]:
        """Resolve a ``lam_cost`` spec to a numeric value.

        If *lam_cost* is a float it is returned as-is.  If it is a dict
        mapping state names to weights, *states* must be provided so the
        dict can be expanded to a per-state array via
        :func:`_expand_lam_cost_dict`.

        Args:
            lam_cost: Scalar weight or ``{state_name: weight}`` dict.
            states: Symbolic State objects (required when *lam_cost* is a dict).

        Returns:
            float or np.ndarray of shape ``(n_states,)``.

        Raises:
            ValueError: If *lam_cost* is a dict and *states* is ``None``.
        """
        if isinstance(lam_cost, dict):
            if states is None:
                raise ValueError(
                    "lam_cost was specified as a dict but no states were "
                    "provided. Pass states so the dict can be expanded to "
                    "a per-state weight array."
                )
            return _expand_lam_cost_dict(lam_cost, states)
        return lam_cost

    def _resolve_lam_vb(
        self,
        N: int,
        nodal_constraints: list,
        cross_node_constraints: list,
    ) -> None:
        """Resolve per-constraint virtual buffer weight arrays and re-normalize.

        Inspects each symbolic constraint's shape (to account for vector
        decomposition) and ``.weight()`` overrides, populates
        ``weights.lam_vb_nodal`` and ``weights.lam_vb_cross``, then
        re-normalizes all weights so the overrides participate in the scale.

        Args:
            N: Number of trajectory nodes.
            nodal_constraints: Symbolic ``NodalConstraint`` objects (post-
                preprocessing, pre-lowering).
            cross_node_constraints: Symbolic ``CrossNodeConstraint`` objects.
        """
        default_vb = float(self.weights._raw_lam_vb)

        # Count decomposed nodal constraints (vector → multiple scalars).
        # Vector constraints are decomposed element-wise during lowering
        # (see decompose_vector_nodal_constraints), so each element gets its
        # own column.  We mirror that here via check_shape() to ensure the
        # array dimensions match the post-decomposition constraint count.
        n_nodal = 0
        for nc in nodal_constraints:
            try:
                shape = nc.constraint.lhs.check_shape()
                n_nodal += int(np.prod(shape)) if len(shape) > 0 else 1
            except Exception:
                n_nodal += 1

        n_cross = len(cross_node_constraints)

        # max(..., 1) avoids size-0 CVXPy parameters.
        n_nodal_param = max(n_nodal, 1)
        n_cross_param = max(n_cross, 1)

        lam_vb_nodal = np.full((N, n_nodal_param), default_vb)
        lam_vb_cross = np.full(n_cross_param, default_vb)

        # Apply per-constraint .weight() overrides for nodal constraints.
        col = 0
        for nc in nodal_constraints:
            try:
                shape = nc.constraint.lhs.check_shape()
                n_elem = int(np.prod(shape)) if len(shape) > 0 else 1
            except Exception:
                n_elem = 1

            w = nc._lam_vb
            if w is not None:
                nodes = nc.nodes if nc.nodes is not None else list(range(N))
                if isinstance(w, (int, float)):
                    lam_vb_nodal[nodes, col : col + n_elem] = float(w)
                elif isinstance(w, np.ndarray):
                    if w.ndim == 1:
                        # (n_elem,) — broadcast across nodes
                        for i in range(n_elem):
                            val = float(w[0]) if len(w) == 1 else float(w[i])
                            lam_vb_nodal[nodes, col + i] = val
                    elif w.ndim == 2:
                        # (n_nodes, n_elem) — per-node-per-element
                        for i in range(n_elem):
                            c_i = 0 if w.shape[1] == 1 else i
                            lam_vb_nodal[nodes, col + i] = w[:, c_i]

            col += n_elem

        # Apply per-constraint .weight() overrides for cross-node constraints.
        for idx, cc in enumerate(cross_node_constraints):
            if cc._lam_vb is not None:
                lam_vb_cross[idx] = float(cc._lam_vb)

        self.weights.set_vb_arrays(lam_vb_nodal, lam_vb_cross)
        self.weights.normalize()

    @abstractmethod
    def initialize(
        self,
        solver: "ConvexSolver",
        discretization_solver: callable,
        jax_constraints: "LoweredJaxConstraints",
        emitter: callable,
        params: dict,
        settings: "Config",
        discretization_solver_impulsive: Optional[Callable] = None,
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
            discretization_solver_impulsive: Optional solver for discrete/impulsive
                dynamics evaluated on ``(x_prop, u_discrete)``
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
