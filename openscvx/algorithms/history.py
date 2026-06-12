"""CPU-side append-only iteration log for SCP algorithms.

:class:`AlgorithmHistory` is the mutable, host-side container the SCP loop grows
after each iteration via :py:meth:`~AlgorithmHistory.record_iteration` — it
never crosses the JAX boundary. :class:`DiscretizationResult` unpacks a raw
discretization matrix once so history reads are trivial slicing-free access.

Last in the algorithms import order: it depends on
:mod:`openscvx.algorithms.state` and nothing here depends on it.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import jax
import numpy as np

from .state import AdaptiveStateCode, AlgorithmState, adaptive_state_code_to_str

if TYPE_CHECKING:
    from openscvx.config import Config


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
# AlgorithmHistory — CPU-side append-only iteration log
# ---------------------------------------------------------------------------


@dataclass
class AlgorithmHistory:
    """Append-only iteration log, populated by the SCP loop.

    Mirrors the lists that previously lived directly on ``AlgorithmState``.
    Never appears on the JAX boundary; the SCP loop appends to it after each
    iteration via :py:meth:`record_iteration`. Purely diagnostic — the
    subproblem solver consumes ``SubproblemData`` in-trace, not the history.
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

    @classmethod
    def from_settings(cls, settings: "Config") -> "AlgorithmHistory":
        """Construct an empty history sized to the problem."""
        return cls(
            n_x=settings.sim.n_states,
            n_u=settings.sim.n_controls,
            N=settings.sim.n,
        )

    @property
    def V_history(self) -> List[np.ndarray]:
        """View of the raw discretization matrices recorded so far."""
        return [d.V for d in self.discretizations]

    # -- Per-iteration recording -------------------------------------------

    def record_iteration(
        self,
        state: AlgorithmState,
        *,
        V: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
        VC: Optional[np.ndarray] = None,
        TR: Optional[np.ndarray] = None,
        J_lin: Optional[float] = None,
        record_diagnostics: bool = True,
    ) -> Tuple[dict, np.ndarray]:
        """Append per-iteration data based on ``state.adaptive_state_code``.

        The accepted iterate (``x`` / ``u`` / weights / convergence scalars) is
        read off ``state``; the raw host-side diagnostics the SCP loop already
        synced — discretization matrices (``V`` / ``W``), virtual control
        (``VC``), trust region (``TR``), and the linearized cost (``J_lin``) —
        come in as explicit keyword arrays so the generic history never has to
        know the algorithm's diagnostics type.

        Reproduces the old ``accept_solution`` / ``reject_solution`` behavior:

        * **REJECT**: append ``lam_prox`` (the bumped weight that drives the
          next subproblem), plus the predicted/actual/acceptance diagnostics
          when ``record_diagnostics`` is set.
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

        if V is not None:
            if W is not None:
                self.discretizations.append(
                    DiscretizationResult.from_VW(
                        V,
                        W,
                        n_x=self.n_x,
                        n_u=self.n_u,
                        N=self.N,
                    )
                )
            else:
                self.discretizations.append(
                    DiscretizationResult.from_V(
                        V,
                        n_x=self.n_x,
                        n_u=self.n_u,
                        N=self.N,
                    )
                )
        if VC is not None:
            self.VC.append(np.asarray(VC))
        if TR is not None:
            self.TR.append(np.asarray(TR))

        self.lam_vc.append(np.asarray(lam_vc_np))
        self.lam_cost.append(np.asarray(lam_cost_np))
        self.lam_vb_nodal.append(np.asarray(lam_vb_nodal_np))
        self.lam_vb_cross.append(np.asarray(lam_vb_cross_np))

        self.J_nonlin.append(scalars["J_nonlin"])
        if J_lin is not None:
            self.J_lin.append(float(J_lin))

        # Diagnostics: only meaningful for iterations after the initial one.
        if record_diagnostics and code is not AdaptiveStateCode.INITIAL:
            self.pred_reduction.append(scalars["predicted_reduction"])
            self.actual_reduction.append(scalars["actual_reduction"])
            self.acceptance_ratio.append(scalars["acceptance_ratio"])

        return scalars, lam_prox_np
