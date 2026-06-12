"""Penalized Trust Region (PTR) successive convexification algorithm.

This module implements the PTR algorithm for solving non-convex trajectory
optimization problems through iterative convex approximation.
"""

import time
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Union

import jax
import numpy as np

from openscvx.config import Config
from openscvx.solvers.ptr_solver import StatusCode, status_code_to_str
from openscvx.utils.printing import (
    Column,
    Verbosity,
    color_J_tr,
    color_J_vb,
    color_J_vc,
    color_prob_stat,
)

from ..autotuner.augmented_lagrangian import AugmentedLagrangian
from ..base import Algorithm
from ..history import AlgorithmHistory
from ..state import AlgorithmState, adaptive_state_code_to_str
from ..weights import Weights
from .iteration import make_scp_iteration

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State

    from ..autotuner.base import AutotuningBase


class PenalizedTrustRegion(Algorithm):
    """Penalized Trust Region (PTR) successive convexification algorithm.

    PTR solves non-convex trajectory optimization problems through iterative
    convex approximation. Each subproblem balances competing cost terms:

    - **Trust region penalty**: Discourages large deviations from the previous
      iterate, keeping the solution within the region where linearization is valid.
    - **Virtual control**: Relaxes dynamics constraints, penalized to drive
      defects toward zero as the algorithm converges.
    - **Virtual buffer**: Relaxes non-convex constraints, similarly penalized
      to enforce feasibility at convergence.
    - **Problem objective and other terms**: The user-defined cost (e.g., minimum
      fuel, minimum time) and any additional penalty terms.

    The interplay between these terms guides the optimization: the trust region
    anchors the solution near the linearization point while virtual terms allow
    temporary constraint violations that shrink over iterations.

    Example:
        Using PTR with a Problem::

            from openscvx.algorithms import PenalizedTrustRegion

            problem = Problem(dynamics, constraints, states, controls, N, time)
            problem.initialize()
            result = problem.solve()
    """

    # Base columns emitted by PTR algorithm (before autotuner columns)
    BASE_COLUMNS: List[Column] = [
        Column("iter", "Iter", 4, "{:4d}"),
        Column("subprop_time", "Step (ms)", 10, "{:6.2f}", min_verbosity=Verbosity.STANDARD),
        Column("cost", "Cost", 8, "{: .1e}"),
        Column("J_tr", "J_tr", 8, "{: .1e}", color_J_tr, Verbosity.STANDARD),
        Column("J_vb", "J_vb", 8, "{: .1e}", color_J_vb, Verbosity.STANDARD),
        Column("J_vc", "J_vc", 8, "{: .1e}", color_J_vc, Verbosity.STANDARD),
    ]

    # Columns that always appear last (after autotuner columns)
    TAIL_COLUMNS: List[Column] = [
        Column("prob_stat", "Cvx Status", 11, "{}", color_prob_stat),
    ]

    def __init__(
        self,
        autotuner: "AutotuningBase" = None,
        k_max: int = 200,
        t_max: float | None = None,
        lam_prox: Union[float, Dict[str, Union[float, list]]] = 1e-1,
        lam_vc: Union[float, Dict[str, Union[float, list]]] = 1e0,
        lam_cost: Union[float, Dict[str, float]] = 1e-2,
        lam_vb: float = 0.0,
        ep_tr: float = 1e-4,
        ep_vb: float = 1e-4,
        ep_vc: float = 1e-8,
        states: List["State"] = None,
        controls: List["Control"] = None,
    ):
        """Initialize PTR with algorithm parameters and optional autotuner."""
        # Compiled infrastructure (set by initialize()). ``_iteration_fn`` is
        # the fused JAX-pure SCP body; ``step()`` is a thin Python wrapper that
        # calls it, records history, and emits.
        self._iteration_fn: Callable | None = None
        self._emitter: Callable | None = None

        # Store states/controls for later re-resolution.
        self._states: List["State"] = states
        self._controls: List["Control"] = controls

        super().__init__(
            # SCP weights (grouped dataclass, dict inputs expanded to arrays).
            weights=Weights.build(
                lam_prox=lam_prox,
                lam_vc=lam_vc,
                lam_cost=lam_cost,
                lam_vb=lam_vb,
                states=states,
                controls=controls,
            ),
            autotuner=autotuner if autotuner is not None else AugmentedLagrangian(),
            k_max=k_max,
            t_max=t_max,
            ep_tr=ep_tr,
            ep_vb=ep_vb,
            ep_vc=ep_vc,
        )

    @property
    def lam_prox(self) -> Union[float, np.ndarray]:
        return self.weights.lam_prox

    @lam_prox.setter
    def lam_prox(self, value: Union[float, Dict[str, Union[float, list]]]) -> None:
        self.weights.lam_prox = Weights.resolve_lam_prox(value, self._states, self._controls)

    @property
    def lam_vc(self) -> Union[float, np.ndarray]:
        return self.weights.lam_vc

    @lam_vc.setter
    def lam_vc(self, value: Union[float, Dict[str, Union[float, list]]]) -> None:
        self.weights.lam_vc = Weights.resolve_lam_vc(value, self._states)

    @property
    def lam_cost(self) -> Union[float, np.ndarray]:
        return self.weights.lam_cost

    @lam_cost.setter
    def lam_cost(self, value: Union[float, Dict[str, float]]) -> None:
        self.weights.lam_cost = Weights.resolve_lam_cost(value, self._states)

    @property
    def lam_vb(self) -> float:
        return self.weights.lam_vb

    @lam_vb.setter
    def lam_vb(self, value: float) -> None:
        self.weights.lam_vb = float(value)

    def build_iteration(
        self,
        dis_continuous: Callable,
        dis_impulsive: Callable,
        jax_constraints: "LoweredJaxConstraints",
        solver_callback: Callable,
        settings: Config,
    ) -> Callable:
        """Fuse the discretizers, constraints, and solver into the PTR step.

        Thin wrapper around
        :func:`~openscvx.algorithms.scvx.iteration.make_scp_iteration`,
        threading this algorithm's :attr:`autotuner` into the fused body.
        """
        return make_scp_iteration(
            dis_continuous=dis_continuous,
            dis_impulsive=dis_impulsive,
            jax_constraints=jax_constraints,
            solver_callback=solver_callback,
            autotuner=self.autotuner,
            settings=settings,
        )

    def get_columns(self, verbosity: int = Verbosity.STANDARD) -> List[Column]:
        """Get the columns to display for iteration output."""
        all_columns = self.BASE_COLUMNS + self.autotuner.COLUMNS + self.TAIL_COLUMNS
        return [col for col in all_columns if col.min_verbosity <= verbosity]

    def initialize(self, iteration_fn: Callable, emitter: Callable) -> None:
        """Store the fused SCP iteration body and per-iteration infrastructure.

        ``iteration_fn`` is built and JIT-warmed by :meth:`Problem.initialize`;
        :meth:`step` is a thin Python wrapper that calls it, records history,
        and emits diagnostics. The boundary conditions the subproblem pins are
        carried on :class:`AlgorithmState` (``x_init_pin`` / ``x_term_pin``), so
        there is no solver state to prime here.
        """
        self._iteration_fn = iteration_fn
        self._emitter = emitter

    def step(
        self,
        state: AlgorithmState,
        history: AlgorithmHistory,
        params: dict,
        settings: Config,
    ) -> Tuple[AlgorithmState, bool]:
        """Execute one PTR iteration and return ``(next_state, converged)``.

        Calls the fused JAX iteration body, records the per-iteration
        diagnostics it returns into ``history``, emits progress, and reports
        convergence from the metrics on the next state.
        """
        if self._iteration_fn is None:
            raise RuntimeError(
                "PenalizedTrustRegion.step() called before initialize(). "
                "Call initialize() first to set up the iteration body."
            )

        iter_index = int(state.k)

        t0 = time.time()
        next_state, diag = self._iteration_fn(state, params)
        jax.block_until_ready((next_state, diag))
        step_time = time.time() - t0

        # Fail loudly on a bad subproblem solve before it becomes the next
        # linearization point. Two gates: a non-OPTIMAL status from the backend
        # (e.g. QPAX divergence, CVXPy infeasibility) and a non-finite iterate
        # (NaN/Inf from anywhere in the body). Either would silently corrupt the
        # rest of the SCP run.
        if int(diag.status) != int(StatusCode.OPTIMAL):
            raise RuntimeError(
                f"Convex subproblem did not solve to optimality "
                f"(status={status_code_to_str(int(diag.status))!r}). The backend failed "
                f"to converge or the subproblem is infeasible — adjust solver tolerances, "
                f"rescale the problem, or use float_dtype='float64'."
            )
        if not bool(np.all(np.isfinite(np.asarray(next_state.x)))):
            raise RuntimeError(
                "Subproblem solve produced a non-finite iterate (NaN/Inf in the state)."
            )

        use_full_metrics = self.autotuner.COMPUTES_ACCEPTANCE_METRICS
        scalars, lam_prox_np = history.record_iteration(
            next_state,
            V=np.asarray(diag.V),
            W=np.asarray(diag.W),
            VC=np.asarray(diag.VC),
            TR=np.asarray(diag.TR),
            J_lin=float(diag.J_lin),
            record_diagnostics=use_full_metrics,
        )

        emission_data = {
            "iter": iter_index,
            "subprop_time": step_time * 1000.0,
            "J_tr": scalars["J_tr"],
            "J_vb": scalars["J_vb"],
            "J_vc": scalars["J_vc"],
            "cost": float(diag.cost),
            # TODO: (haynec) log per-variable lam_prox detail (e.g. min/max range)
            "lam_prox": float(np.max(lam_prox_np)),
            "prob_stat": status_code_to_str(int(diag.status)),
            "adaptive_state": adaptive_state_code_to_str(scalars["adaptive_state_code"]),
            "ep_tr": self.ep_tr,
            "ep_vb": self.ep_vb,
            "ep_vc": self.ep_vc,
        }

        if use_full_metrics:
            emission_data.update(
                {
                    "J_nonlin": scalars["J_nonlin"],
                    "J_lin": float(diag.J_lin),
                    "pred_reduction": scalars["predicted_reduction"],
                    "actual_reduction": scalars["actual_reduction"],
                    "acceptance_ratio": scalars["acceptance_ratio"],
                }
            )

        self._emitter(emission_data)

        # Same convergence test the lax.while_loop path uses, reading the
        # thresholds off the state pytree (synced from this algorithm's
        # ep_* attributes by Problem._sync_scp_constants at solve() time).
        return next_state, bool(self.converged(next_state))

    def citation(self) -> List[str]:
        """Return BibTeX citations for the PTR algorithm."""
        return [
            r"""@article{drusvyatskiy2018error,
  title={Error bounds, quadratic growth, and linear convergence of proximal methods},
  author={Drusvyatskiy, Dmitriy and Lewis, Adrian S},
  journal={Mathematics of operations research},
  volume={43},
  number={3},
  pages={919--948},
  year={2018},
  publisher={INFORMS}
}""",
            r"""@article{szmuk2020successive,
  title={Successive convexification for real-time six-degree-of-freedom powered descent guidance
    with state-triggered constraints},
  author={Szmuk, Michael and Reynolds, Taylor P and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
  journal={Journal of Guidance, Control, and Dynamics},
  volume={43},
  number={8},
  pages={1399--1413},
  year={2020},
  publisher={American Institute of Aeronautics and Astronautics}
}""",
            r"""@article{reynolds2020dual,
  title={Dual quaternion-based powered descent guidance with state-triggered constraints},
  author={Reynolds, Taylor P and Szmuk, Michael and Malyuta, Danylo and Mesbahi, Mehran and
    A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et and Carson III, John M},
  journal={Journal of Guidance, Control, and Dynamics},
  volume={43},
  number={9},
  pages={1584--1599},
  year={2020},
  publisher={American Institute of Aeronautics and Astronautics}
}""",
        ]
