"""Base classes for successive convexification algorithms.

This module defines the abstract interface that all SCP algorithm implementations
follow, along with two state containers used during the SCP iteration:

* :class:`AlgorithmState` — a JAX-registered, frozen pytree holding the
  *current iterate* (state/control/weights/diagnostics). Every autotuner's
  :py:meth:`AutotuningBase.update_weights` is a pure functional update on this
  pytree, so the SCP body composes with ``jax.jit``, ``jax.vmap``, and
  ``jax.grad``.
* :class:`AlgorithmHistory` — a CPU-side, mutable container for the append-only
  iteration histories (per-iteration trajectories, discretizations, costs,
  weight snapshots). Lives outside the JAX boundary; populated by the SCP loop
  via :py:meth:`AlgorithmHistory.record_iteration`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from openscvx.utils.printing import Column

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints
    from openscvx.solvers import ConvexSolver

    from .weights import Weights


@dataclass
class CandidateIterate:
    """Per-iteration candidate produced by the convex subproblem.

    Mutable on purpose: the discretizer / subproblem code fills fields
    incrementally (``x``, ``u`` first, then ``V``/``W``/``x_prop``/etc.) before
    handing the candidate to :py:meth:`AutotuningBase.update_weights`. Treat
    this object as a structured-but-numpy input to ``update_weights``; the
    autotuner does not mutate it.
    """

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
    J_lin: Optional[float] = None


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


# ---------------------------------------------------------------------------
# Adaptive-state enum
# ---------------------------------------------------------------------------


class AdaptiveStateCode(IntEnum):
    """Outcome of one autotuner ``update_weights`` call.

    The autotuner stores its decision as an ``int32`` on
    :py:attr:`AlgorithmState.adaptive_state_code` so the pytree remains
    JAX-traceable. The SCP printing path maps the code back to a human-readable
    string via :func:`adaptive_state_code_to_str`.
    """

    REJECT = 0
    ACCEPT_HIGHER = 1
    ACCEPT_CONSTANT = 2
    ACCEPT_LOWER = 3
    INITIAL = 4


_ADAPTIVE_STATE_NAMES = {
    AdaptiveStateCode.REJECT: "Reject Higher",
    AdaptiveStateCode.ACCEPT_HIGHER: "Accept Higher",
    AdaptiveStateCode.ACCEPT_CONSTANT: "Accept Constant",
    AdaptiveStateCode.ACCEPT_LOWER: "Accept Lower",
    AdaptiveStateCode.INITIAL: "Initial",
}


def adaptive_state_code_to_str(code: Union[int, jnp.ndarray, np.ndarray]) -> str:
    """Map an :class:`AdaptiveStateCode` value (int or 0-d array) to its label."""
    return _ADAPTIVE_STATE_NAMES[AdaptiveStateCode(int(code))]


# ---------------------------------------------------------------------------
# Autotuning base class
# ---------------------------------------------------------------------------


class AutotuningBase(ABC):
    """Base class for autotuning strategies in SCP algorithms.

    Subclasses implement :py:meth:`update_weights` as a **pure functional
    update** on an :class:`AlgorithmState` pytree. Concretely, an autotuner
    must:

    * Not mutate ``state`` or ``candidate``.
    * Not return strings, raise on data-dependent conditions, or append to
      Python lists at trace time.
    * Express all branching on iterate values via ``jax.lax.cond`` /
      ``jnp.where``.

    The autotuner's per-iteration outcome is encoded as an
    :class:`AdaptiveStateCode` int32 on the returned state; the SCP loop
    converts that back to a human-readable label only on the Python-loop
    printing path.

    Class Attributes:
        COLUMNS: List of Column specs for autotuner-specific metrics to display.
            Subclasses override this to add their own columns.
    """

    COLUMNS: List[Column] = []

    @staticmethod
    def calculate_cost_from_state(
        x: jnp.ndarray,
        settings: "Config",
        lam_cost: Union[float, jnp.ndarray] = 1.0,
    ) -> jnp.ndarray:
        """Compute the boundary-condition-weighted cost contribution for ``x``.

        Args:
            x: State trajectory, shape ``(N, n_states)``.
            settings: Configuration object carrying scaling matrices and
                boundary-condition types.
            lam_cost: Per-state cost weight. Scalar or array of shape
                ``(n_states,)``.

        Returns:
            Scalar cost (jnp), weighted by ``lam_cost``.
        """
        scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
        lam = jnp.asarray(lam_cost)
        cost = jnp.asarray(0.0)
        for i in range(settings.sim.n_states):
            w = lam[i] if lam.ndim > 0 else lam
            if settings.sim.x.final_type[i] == "Minimize":
                cost = cost + w * scaled_x[-1, i]
            if settings.sim.x.final_type[i] == "Maximize":
                cost = cost - w * scaled_x[-1, i]
            if settings.sim.x.initial_type[i] == "Minimize":
                cost = cost + w * scaled_x[0, i]
            if settings.sim.x.initial_type[i] == "Maximize":
                cost = cost - w * scaled_x[0, i]
        return cost

    @staticmethod
    def calculate_nonlinear_penalty(
        x_prop: jnp.ndarray,
        x_bar: jnp.ndarray,
        u_bar: jnp.ndarray,
        lam_vc: jnp.ndarray,
        lam_vb_nodal: jnp.ndarray,
        lam_vb_cross: jnp.ndarray,
        lam_cost: Union[float, jnp.ndarray],
        nodal_constraints: "LoweredJaxConstraints",
        params: dict,
        settings: "Config",
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute the three components of the nonlinear penalty.

        This is JAX-traceable: the Python loops over
        ``nodal_constraints.nodal`` / ``cross_node`` unroll at trace time
        (the lists are static-length lists of compiled closures built by
        :class:`Problem`).

        Args:
            x_prop: Propagated state, shape ``(N-1, n_states)``.
            x_bar: Nodal state, shape ``(N, n_states)``.
            u_bar: Nodal control, shape ``(N, n_controls)``.
            lam_vc: Virtual control weight, scalar or matrix.
            lam_vb_nodal: Nodal virtual-buffer weights, shape ``(N, n_nodal)``.
            lam_vb_cross: Cross-node virtual-buffer weights, shape ``(n_cross,)``.
            lam_cost: Cost weight, scalar or shape ``(n_states,)``.
            nodal_constraints: Lowered JAX constraints.
            params: Problem parameter dictionary.
            settings: Configuration object.

        Returns:
            ``(nonlinear_cost, nonlinear_penalty, nodal_penalty)`` — all
            scalar jnp arrays.
        """
        nodal_penalty = jnp.asarray(0.0)

        for idx, constraint in enumerate(nodal_constraints.nodal):
            g = constraint.func(x_bar, u_bar, 0, params)
            if constraint.nodes is not None:
                nodes_array = jnp.asarray(constraint.nodes)
                g_filtered = g[nodes_array]
                w = lam_vb_nodal[nodes_array, idx]
            else:
                g_filtered = g
                w = lam_vb_nodal[:, idx]
            nodal_penalty = nodal_penalty + jnp.sum(w * jnp.maximum(0.0, g_filtered))

        for idx, constraint in enumerate(nodal_constraints.cross_node):
            w = lam_vb_cross[idx]
            g = constraint.func(x_bar, u_bar, params)
            nodal_penalty = nodal_penalty + w * jnp.sum(jnp.maximum(0.0, g))

        cost = AutotuningBase.calculate_cost_from_state(x_bar, settings, lam_cost)
        x_diff = settings.sim.inv_S_x @ (x_bar[1:, :] - x_prop).T

        return cost, jnp.sum(lam_vc * jnp.abs(x_diff.T)), nodal_penalty

    @abstractmethod
    def update_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        settings: "Config",
        params: dict,
        weights: "Weights",
    ) -> "AlgorithmState":
        """Return the next-iterate :class:`AlgorithmState`.

        Must be JAX-traceable. See the class docstring for the contract.

        Args:
            state: Current-iterate pytree.
            candidate: Subproblem result (read-only here).
            nodal_constraints: Lowered JAX constraints.
            settings: Configuration object.
            params: Problem parameter dictionary.
            weights: Initial weights from the algorithm.

        Returns:
            The next-iterate :class:`AlgorithmState`. Its
            :py:attr:`AlgorithmState.adaptive_state_code` encodes the
            autotuner's decision; the SCP loop records that into history and
            maps it to a printable label.
        """


# ---------------------------------------------------------------------------
# AlgorithmState — JAX-traceable current iterate
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AlgorithmState:
    """Current-iterate pytree for SCP iterations.

    Every field is a JAX array (or scalar). The state moves between iterations
    by way of pure functional updates: :py:meth:`replace` returns a new state
    rather than mutating in place. Registered as a JAX pytree so the SCP body
    composes with ``jax.jit`` / ``jax.vmap`` / ``jax.grad``.

    Append-only iteration histories (X, U, discretizations, ...) live on
    :class:`AlgorithmHistory`, which is CPU-side and grown by the SCP loop.

    Attributes:
        x: Nodal state trajectory, shape ``(N, n_states)``.
        u: Nodal control trajectory, shape ``(N, n_controls)``.
        x_prop: Continuous-time propagation of ``x``, shape ``(N-1, n_states)``.
        x_prop_plus: Impulsive/discrete dynamics evaluated at ``x_prop``,
            shape ``(N, n_states)``. Zeros when no impulsive component is
            present.
        lam_prox: Trust-region weight, shape ``(N, n_states + n_controls)``.
        lam_vc: Virtual-control penalty weight, shape ``(N-1, n_states)``.
        lam_cost: Cost weight, shape ``(n_states,)``.
        lam_vb_nodal: Nodal virtual-buffer weights, shape ``(N, n_nodal)``.
        lam_vb_cross: Cross-node virtual-buffer weights, shape ``(n_cross,)``.
        k: Iteration counter (starts at 1).
        J_tr: Current trust-region cost (scalar).
        J_vb: Current virtual-buffer cost (scalar).
        J_vc: Current virtual-control cost (scalar).
        J_nonlin: Nonlinear-objective value for the current accepted iterate.
        predicted_reduction: Predicted reduction in ``J_nonlin`` for this iter.
        actual_reduction: Actual reduction in ``J_nonlin`` for this iter.
        acceptance_ratio: ``actual_reduction / predicted_reduction``.
        adaptive_state_code: :class:`AdaptiveStateCode` value as ``int32``.
    """

    x: jnp.ndarray
    u: jnp.ndarray
    x_prop: jnp.ndarray
    x_prop_plus: jnp.ndarray
    lam_prox: jnp.ndarray
    lam_vc: jnp.ndarray
    lam_cost: jnp.ndarray
    lam_vb_nodal: jnp.ndarray
    lam_vb_cross: jnp.ndarray
    k: jnp.ndarray
    J_tr: jnp.ndarray
    J_vb: jnp.ndarray
    J_vc: jnp.ndarray
    J_nonlin: jnp.ndarray
    predicted_reduction: jnp.ndarray
    actual_reduction: jnp.ndarray
    acceptance_ratio: jnp.ndarray
    adaptive_state_code: jnp.ndarray

    # Field order is the source of truth for tree_flatten / tree_unflatten;
    # keep _FIELDS in sync with the dataclass field declarations above.
    _FIELDS = (
        "x",
        "u",
        "x_prop",
        "x_prop_plus",
        "lam_prox",
        "lam_vc",
        "lam_cost",
        "lam_vb_nodal",
        "lam_vb_cross",
        "k",
        "J_tr",
        "J_vb",
        "J_vc",
        "J_nonlin",
        "predicted_reduction",
        "actual_reduction",
        "acceptance_ratio",
        "adaptive_state_code",
    )

    def replace(self, **changes) -> "AlgorithmState":
        """Return a new state with ``changes`` applied (functional update)."""
        return dc_replace(self, **changes)

    def tree_flatten(self):
        children = tuple(getattr(self, name) for name in self._FIELDS)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(**dict(zip(cls._FIELDS, children)))

    @classmethod
    def from_settings(
        cls,
        settings: "Config",
        weights: "Weights",
    ) -> "AlgorithmState":
        """Construct the initial iterate from configuration.

        Copies trajectory guesses, expands weights to dense arrays, and seeds
        the diagnostic scalars to zero / :py:attr:`AdaptiveStateCode.INITIAL`.
        ``x_prop`` and ``x_prop_plus`` are zero-initialized; the SCP algorithm
        fills them at the first discretization step.
        """
        n = settings.sim.n
        n_states = settings.sim.n_states
        n_controls = settings.sim.n_controls
        n_total = n_states + n_controls

        lam_vc_array = np.ones((n - 1, n_states)) * weights.lam_vc
        lam_prox_array = np.ones((n, n_total)) * weights.lam_prox

        if isinstance(weights.lam_cost, np.ndarray):
            lam_cost_init = weights.lam_cost.copy()
        else:
            lam_cost_init = np.full(n_states, weights.lam_cost)

        return cls(
            x=jnp.asarray(settings.sim.x.guess),
            u=jnp.asarray(settings.sim.u.guess),
            x_prop=jnp.zeros((n - 1, n_states)),
            x_prop_plus=jnp.zeros((n, n_states)),
            lam_prox=jnp.asarray(lam_prox_array),
            lam_vc=jnp.asarray(lam_vc_array),
            lam_cost=jnp.asarray(lam_cost_init),
            lam_vb_nodal=jnp.asarray(weights.lam_vb_nodal),
            lam_vb_cross=jnp.asarray(weights.lam_vb_cross),
            k=jnp.asarray(1, dtype=jnp.int32),
            J_tr=jnp.asarray(1e2),
            J_vb=jnp.asarray(1e2),
            J_vc=jnp.asarray(1e2),
            J_nonlin=jnp.asarray(0.0),
            predicted_reduction=jnp.asarray(0.0),
            actual_reduction=jnp.asarray(0.0),
            acceptance_ratio=jnp.asarray(0.0),
            adaptive_state_code=jnp.asarray(int(AdaptiveStateCode.INITIAL), dtype=jnp.int32),
        )


# ---------------------------------------------------------------------------
# AlgorithmHistory — CPU-side append-only iteration log
# ---------------------------------------------------------------------------


@dataclass
class AlgorithmHistory:
    """Append-only iteration log, populated by the SCP loop.

    Mirrors the lists that previously lived directly on ``AlgorithmState``.
    Never appears on the JAX boundary; the SCP loop appends to it after each
    iteration via :py:meth:`record_iteration`. Mostly diagnostic, but also
    used by the convex subproblem solver to look up the most recent
    discretization for linearization.
    """

    n_x: int
    n_u: int
    N: int
    X: List[np.ndarray] = field(default_factory=list)
    U: List[np.ndarray] = field(default_factory=list)
    discretizations: List[DiscretizationResult] = field(default_factory=list)
    VC: List[np.ndarray] = field(default_factory=list)
    TR: List[np.ndarray] = field(default_factory=list)
    lam_prox: List[np.ndarray] = field(default_factory=list)
    lam_vc: List[Union[float, np.ndarray]] = field(default_factory=list)
    lam_cost: List[Union[float, np.ndarray]] = field(default_factory=list)
    lam_vb_nodal: List[np.ndarray] = field(default_factory=list)
    lam_vb_cross: List[np.ndarray] = field(default_factory=list)
    J_nonlin: List[float] = field(default_factory=list)
    J_lin: List[float] = field(default_factory=list)
    pred_reduction: List[float] = field(default_factory=list)
    actual_reduction: List[float] = field(default_factory=list)
    acceptance_ratio: List[float] = field(default_factory=list)
    adaptive_state: List[str] = field(default_factory=list)
    x_full: List[np.ndarray] = field(default_factory=list)
    x_prop_full: List[np.ndarray] = field(default_factory=list)

    @classmethod
    def from_settings(cls, settings: "Config") -> "AlgorithmHistory":
        """Construct an empty history sized to the problem."""
        return cls(
            n_x=settings.sim.n_states,
            n_u=settings.sim.n_controls,
            N=settings.sim.n,
        )

    # -- Discretization plumbing --------------------------------------------

    def add_discretization(self, V: np.ndarray) -> None:
        """Append a raw continuous-time discretization matrix."""
        self.discretizations.append(
            DiscretizationResult.from_V(V, n_x=self.n_x, n_u=self.n_u, N=self.N)
        )

    def add_impulsive_discretization(self, W: np.ndarray) -> None:
        """Attach impulsive data to the most recently added discretization."""
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

    # -- Discretization accessors -------------------------------------------
    #
    # Subproblem solvers ask for the *latest* (or i-th) linearization; the
    # accessors below mirror the old ``state.A_d()`` / ``state.B_d()`` /
    # ``state.x_prop()`` API.

    @property
    def V_history(self) -> List[np.ndarray]:
        """Backward-compatible view of raw discretization matrices."""
        return [d.V for d in self.discretizations]

    def x_prop(self, index: int = -1) -> Optional[np.ndarray]:
        """Return the i-th propagated state trajectory."""
        if not self.discretizations:
            return None
        return self.discretizations[index].x_prop

    def x_prop_plus(self, index: int = -1) -> Optional[np.ndarray]:
        """Return the i-th impulsive output (or ``None`` if not present)."""
        if not self.discretizations:
            return None
        return self.discretizations[index].x_prop_plus

    def A_d(self, index: int = -1) -> Optional[np.ndarray]:
        if not self.discretizations:
            return None
        return self.discretizations[index].A_d

    def B_d(self, index: int = -1) -> Optional[np.ndarray]:
        if not self.discretizations:
            return None
        return self.discretizations[index].B_d

    def C_d(self, index: int = -1) -> Optional[np.ndarray]:
        if not self.discretizations:
            return None
        return self.discretizations[index].C_d

    def D_d(self, index: int = -1) -> Optional[np.ndarray]:
        if not self.discretizations:
            return None
        return self.discretizations[index].D_d

    def E_d(self, index: int = -1) -> Optional[np.ndarray]:
        if not self.discretizations:
            return None
        return self.discretizations[index].E_d

    # -- Per-iteration recording -------------------------------------------

    def record_iteration(
        self,
        state: AlgorithmState,
        candidate: CandidateIterate,
        record_diagnostics: bool = True,
    ) -> Tuple[dict, np.ndarray]:
        """Append per-iteration data based on ``state.adaptive_state_code``.

        Reproduces the old ``accept_solution`` / ``reject_solution`` behavior:

        * **REJECT**: append only ``lam_prox`` (the bumped weight that drives
          the next subproblem).
        * **INITIAL**: append every trajectory / weight history entry, but
          skip the predicted/actual/acceptance diagnostics (the autotuner
          didn't compute them on iter 1).
        * **ACCEPT_***: append everything, including diagnostics when
          ``record_diagnostics`` is set (the SCP loop turns this off for
          autotuners that don't compute them — :class:`ConstantProximalWeight`,
          :class:`RampProximalWeight`).

        Every device-resident leaf this method needs is pulled with one
        ``jax.device_get`` call (a single CPU<->device round trip) and the
        host-side numpy arrays are then partitioned onto the history lists.
        The SCP loop also reads the same scalars for printing + convergence,
        so the bundle is returned alongside the lam_prox array to avoid a
        second sync.

        Returns:
            ``(scalars, lam_prox_np)`` — a dict of host-side scalar floats
            (``J_tr``, ``J_vb``, ``J_vc``, ``J_nonlin``,
            ``predicted_reduction``, ``actual_reduction``,
            ``acceptance_ratio``) plus the int ``adaptive_state_code``, and
            the numpy copy of ``state.lam_prox``. The SCP emitter consumes
            these directly without re-syncing.
        """
        # Coalesce every device read into one transfer. ``jax.device_get`` on
        # a tuple lets XLA dispatch the whole bundle as a single host copy,
        # which is the entire point on tiny problems where dozens of independent
        # ``float(state.scalar)`` calls would each issue a sync.
        leaves = jax.device_get(
            (
                state.adaptive_state_code,
                state.lam_prox,
                state.J_tr,
                state.J_vb,
                state.J_vc,
                state.J_nonlin,
                state.predicted_reduction,
                state.actual_reduction,
                state.acceptance_ratio,
                state.x,
                state.u,
                state.lam_vc,
                state.lam_cost,
                state.lam_vb_nodal,
                state.lam_vb_cross,
            )
        )
        (
            asc_np,
            lam_prox_np,
            J_tr_np,
            J_vb_np,
            J_vc_np,
            J_nonlin_np,
            pred_np,
            actual_np,
            ratio_np,
            x_np,
            u_np,
            lam_vc_np,
            lam_cost_np,
            lam_vb_nodal_np,
            lam_vb_cross_np,
        ) = leaves

        adaptive_code = int(asc_np)
        code = AdaptiveStateCode(adaptive_code)

        scalars = {
            "J_tr": float(J_tr_np),
            "J_vb": float(J_vb_np),
            "J_vc": float(J_vc_np),
            "J_nonlin": float(J_nonlin_np),
            "predicted_reduction": float(pred_np),
            "actual_reduction": float(actual_np),
            "acceptance_ratio": float(ratio_np),
            "adaptive_state_code": adaptive_code,
        }

        self.lam_prox.append(np.asarray(lam_prox_np))
        self.adaptive_state.append(adaptive_state_code_to_str(code))

        if code is AdaptiveStateCode.REJECT:
            if record_diagnostics:
                self.pred_reduction.append(scalars["predicted_reduction"])
                self.actual_reduction.append(scalars["actual_reduction"])
                self.acceptance_ratio.append(scalars["acceptance_ratio"])
            return scalars, lam_prox_np

        # INITIAL and any ACCEPT_*: full record of the accepted iterate.
        self.X.append(np.asarray(x_np))
        self.U.append(np.asarray(u_np))

        if candidate.V is not None:
            if candidate.W is not None:
                self.discretizations.append(
                    DiscretizationResult.from_VW(
                        candidate.V,
                        candidate.W,
                        n_x=self.n_x,
                        n_u=self.n_u,
                        N=self.N,
                    )
                )
            else:
                self.discretizations.append(
                    DiscretizationResult.from_V(
                        candidate.V,
                        n_x=self.n_x,
                        n_u=self.n_u,
                        N=self.N,
                    )
                )
        if candidate.VC is not None:
            self.VC.append(np.asarray(candidate.VC))
        if candidate.TR is not None:
            self.TR.append(np.asarray(candidate.TR))

        self.lam_vc.append(np.asarray(lam_vc_np))
        self.lam_cost.append(np.asarray(lam_cost_np))
        self.lam_vb_nodal.append(np.asarray(lam_vb_nodal_np))
        self.lam_vb_cross.append(np.asarray(lam_vb_cross_np))

        self.J_nonlin.append(scalars["J_nonlin"])
        if candidate.J_lin is not None:
            self.J_lin.append(float(candidate.J_lin))

        # Diagnostics: only meaningful for iterations after the initial one.
        if record_diagnostics and code is not AdaptiveStateCode.INITIAL:
            self.pred_reduction.append(scalars["predicted_reduction"])
            self.actual_reduction.append(scalars["actual_reduction"])
            self.acceptance_ratio.append(scalars["acceptance_ratio"])

        return scalars, lam_prox_np


# ---------------------------------------------------------------------------
# Algorithm base class
# ---------------------------------------------------------------------------


class Algorithm(ABC):
    """Abstract base class for successive convexification algorithms.

    Subclasses implement :py:meth:`initialize` and :py:meth:`step`. ``step``
    takes the current :class:`AlgorithmState` and a CPU-side
    :class:`AlgorithmHistory`, returns the next state plus a convergence flag,
    and appends to ``history`` for diagnostics.

    !!! tip "Statefullness"
        Avoid storing mutable iteration state on ``self``. All iteration
        state lives on :class:`AlgorithmState` (JAX-traceable) and
        :class:`AlgorithmHistory` (Python-side), threaded explicitly through
        ``step()``.

    Attributes:
        weights: SCP weights used by the algorithm and autotuner.
            Subclasses must set this in ``__init__``.
        k_max: Maximum number of SCP iterations.
            Subclasses must set this in ``__init__``.
        t_max: Optional wall-clock time limit in seconds. ``None`` means no
            limit. Subclasses must set this in ``__init__``.
    """

    weights: "Weights"
    k_max: int
    t_max: Optional[float]

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
        """Initialize the algorithm and store compiled infrastructure."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        state: AlgorithmState,
        history: AlgorithmHistory,
        params: dict,
        settings: "Config",
    ) -> Tuple[AlgorithmState, bool]:
        """Execute one iteration.

        Args:
            state: Current-iterate pytree.
            history: CPU-side append-only iteration log.
            params: Problem parameters (may change between steps).
            settings: Configuration object (may change between steps).

        Returns:
            ``(next_state, converged)`` — the post-iteration state pytree
            and a flag indicating whether the SCP convergence criteria are
            satisfied.
        """
        raise NotImplementedError

    @abstractmethod
    def citation(self) -> List[str]:
        """Return BibTeX citations for this algorithm."""
        raise NotImplementedError
