"""ProxConvex algorithm — Uzun et al. (arXiv:2512.20602v1).

Minimizes F(x) = g(x) + h(C(x)) + s(R(x)) by keeping convex ``r_i``
components exact in the CVXPy subproblem when ∇_i s(R(x_k)) ≥ 0, and
linearizing them when ∇_i s(R(x_k)) < 0.  The scalar proximal metric
Q_k = µ_k I is adapted via an acceptance-ratio test.

Usage::

    from openscvx.algorithms.scvx.prox_convex import ProxConvex, SRComposite
    from openscvx.solvers.cvxpy_ptr_solver import CVXPyProxConvexSolver
    from openscvx.symbolic.lowerers.jax.stl import OR
    import jax.numpy as jnp

    composite = SRComposite(
        s=lambda R, p: -OR(R, c=1e-4),   # JAX-only outer function
        r=[r0_fn, r1_fn],                 # convex; duck-types to CVXPy
    )
    problem = Problem(
        ...,
        algorithm=ProxConvex(composite=composite),
        solver=CVXPyProxConvexSolver(composite=composite),
    )
    problem.initialize()
    result = problem.solve()
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
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

from ..autotuner.adaptive_proximal_weight import AdaptiveProximalWeight
from ..base import Algorithm
from ..history import AlgorithmHistory
from ..state import AlgorithmState, adaptive_state_code_to_str
from ..weights import Weights
from .iteration import make_proxconvex_iteration

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State

    from ..autotuner.base import AutotuningBase


@dataclass
class SRComposite:
    """SR composite: s(R(x)) where R = [r_0(x,u), …, r_{n_r-1}(x,u)].

    Attributes:
        s: ``s(R, params) -> scalar``.  JAX-only; never passed to CVXPy.
            GMSR functions (``AND``, ``OR``, ``IfThen_lite``, …) and any
            composition thereof are natural choices — they are pure JAX and
            differentiated via ``jax.grad`` without special handling.
        r: List of ``r_i(x, u, params) -> scalar`` callables.  Each must:

            * Duck-type to JAX arrays AND CVXPy Variables via plain Python
              operators (``**``, ``+``, ``-``, ``.sum()``, …).
            * Be DCP-convex when evaluated with CVXPy Variables.
            * Have **params-independent DCP structure** — params may affect
              numerical values but not the convexity graph.
    """

    s: Callable
    r: List[Callable]

    def eval(
        self, x: jnp.ndarray, u: jnp.ndarray, params: dict
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evaluate R(x,u), ∇s(R), sign mask, and ∇R at the current iterate.

        Returns:
            R_val: shape ``(n_r,)``
            ds_val: shape ``(n_r,)`` — gradient of ``s`` w.r.t. ``R``
            I_neg_mask: shape ``(n_r,)`` bool — ``True`` where ``ds_val < 0``
            grad_R: shape ``(n_r, N, n_x)`` — Jacobian of each ``r_i`` w.r.t. ``x``
        """
        R_val = jnp.stack([ri(x, u, params) for ri in self.r])
        ds_val = jax.grad(lambda R: self.s(R, params))(R_val)
        I_neg_mask = ds_val < 0
        grad_R = jnp.stack([jax.jacrev(ri, argnums=0)(x, u, params) for ri in self.r])
        return R_val, ds_val, I_neg_mask, grad_R


def check_sr_composite(composite: "SRComposite", x_var, u_var, params: dict) -> None:
    """Validate an :class:`SRComposite` at initialize-time.

    Checks:

    1. ``s(R, params)`` returns a scalar for a test ``R`` of shape ``(n_r,)``
       — required for ``jax.grad`` to succeed.
    2. Each ``r_i(x_var, u_var, params)`` is DCP-compliant when called with
       CVXPy Variables — required so exact channels can be embedded in the
       subproblem as-is.

    Args:
        composite: The :class:`SRComposite` to validate.
        x_var: ``x_nonscaled`` list from the solver (N CVXPy expressions of
            shape ``(n_x,)`` each), or any CVXPy Variable-like sequence.
        u_var: ``u_nonscaled`` list from the solver (N CVXPy expressions of
            shape ``(n_u,)`` each).
        params: Passed through to ``r_i`` for structural validation.
            ``r_i`` must not require specific keys for its DCP structure
            (``{}`` or any dict must work).

    Raises:
        ValueError: If ``s`` does not return a scalar, or if any ``r_i`` is
            not DCP-compliant.
    """
    n_r = len(composite.r)
    R_test = jnp.zeros(n_r)
    s_out = composite.s(R_test, params)
    if jnp.asarray(s_out).ndim != 0:
        raise ValueError(f"s(R, params) must return a scalar; got shape {jnp.asarray(s_out).shape}")
    for i, ri in enumerate(composite.r):
        try:
            expr = ri(x_var, u_var, params)
        except Exception as exc:
            raise ValueError(
                f"r[{i}] raised an exception when called with CVXPy Variables: {exc}"
            ) from exc
        if hasattr(expr, "is_dcp") and not expr.is_dcp():
            raise ValueError(
                f"r[{i}] is not DCP-compliant when evaluated with CVXPy Variables. "
                "Rewrite using plain Python operators (**, +, -, .sum()) that work "
                "for both JAX arrays and CVXPy Variables."
            )


class ProxConvex(Algorithm):
    """ProxConvex successive convexification algorithm (Uzun et al., 2025).

    Minimizes F(x) = g(x) + h(C(x)) + s(R(x)).  The ``s∘R`` term is an
    :class:`SRComposite`: each ``r_i`` is kept exact in the CVXPy subproblem
    when ``∇_i s(R(x_k)) ≥ 0``, and linearized when ``∇_i s(R(x_k)) < 0``.
    The scalar proximal metric ``Q_k = µ_k I`` is adapted by an
    acceptance-ratio test identical to Algorithm 1 of the paper.

    Pair this algorithm with :class:`~openscvx.solvers.cvxpy_ptr_solver.CVXPyProxConvexSolver`
    (pass both the same ``composite`` instance).

    Args:
        composite: :class:`SRComposite` encoding the ``s`` and ``r`` functions.
        autotuner: Weight-update rule.  Defaults to
            :class:`~openscvx.algorithms.autotuner.adaptive_proximal_weight.AdaptiveProximalWeight`
            configured with the ``alpha_1 / alpha_2 / nu_inc / nu_dec`` knobs
            so the four-bucket rule collapses to Algorithm 1's three-bucket rule.
        k_max: SCP iteration cap.
        t_max: Optional wall-clock time limit in seconds.
        mu_0: Initial scalar proximal weight ``µ_0``.
        alpha_1: Acceptance-ratio threshold for the lower bucket (reject /
            accept-higher boundary) *and* the accept-higher / accept-constant
            boundary — setting both equal collapses the four-bucket rule to
            three buckets as in Algorithm 1.
        alpha_2: Accept-constant / accept-lower threshold.
        nu_inc: Multiplicative increase factor for ``µ_k`` on reject / weak accept.
        nu_dec: Multiplicative decrease factor for ``µ_k`` on strong accept.
        mu_min: Lower clip for ``µ_k``.
        mu_max: Upper clip for ``µ_k``.
        lam_vc: Virtual-control penalty weight.
        lam_cost: Boundary-condition cost weight.
        lam_vb: Virtual-buffer penalty weight.
        ep_tr, ep_vb, ep_vc: Convergence thresholds.
        states, controls: Optional state / control lists for per-variable weight
            resolution (forwarded to :class:`~openscvx.algorithms.weights.Weights`).
    """

    BASE_COLUMNS: List[Column] = [
        Column("iter", "Iter", 4, "{:4d}"),
        Column("subprop_time", "Step (ms)", 10, "{:6.2f}", min_verbosity=Verbosity.STANDARD),
        Column("cost", "Cost", 8, "{: .1e}"),
        Column("J_tr", "J_tr", 8, "{: .1e}", color_J_tr, Verbosity.STANDARD),
        Column("J_vb", "J_vb", 8, "{: .1e}", color_J_vb, Verbosity.STANDARD),
        Column("J_vc", "J_vc", 8, "{: .1e}", color_J_vc, Verbosity.STANDARD),
    ]

    TAIL_COLUMNS: List[Column] = [
        Column("prob_stat", "Cvx Status", 11, "{}", color_prob_stat),
    ]

    def __init__(
        self,
        composite: SRComposite,
        autotuner: "AutotuningBase" = None,
        k_max: int = 200,
        t_max: Optional[float] = None,
        mu_0: float = 1e-8,
        alpha_1: float = 1e-2,
        alpha_2: float = 0.8,
        nu_inc: float = 2.0,
        nu_dec: float = 0.5,
        mu_min: float = 1e-8,
        mu_max: float = 1e4,
        lam_vc: Union[float, Dict[str, Union[float, list]]] = 1e0,
        lam_cost: Union[float, Dict[str, float]] = 1e-2,
        lam_vb: float = 0.0,
        ep_tr: float = 1e-4,
        ep_vb: float = 1e-4,
        ep_vc: float = 1e-8,
        states: List["State"] = None,
        controls: List["Control"] = None,
    ):
        self._composite = composite
        self._iteration_fn: Optional[Callable] = None
        self._emitter: Optional[Callable] = None
        self._states: List["State"] = states
        self._controls: List["Control"] = controls

        if autotuner is None:
            # eta_0 == eta_1 == alpha_1 collapses the four-bucket to three-bucket
            # (reject / accept-higher share the same boundary), matching Algorithm 1.
            autotuner = AdaptiveProximalWeight(
                eta_0=alpha_1,
                eta_1=alpha_1,
                eta_2=alpha_2,
                gamma_1=nu_inc,
                gamma_2=nu_dec,
                lam_prox_min=mu_min,
                lam_prox_max=mu_max,
            )

        super().__init__(
            weights=Weights.build(
                lam_prox=mu_0,
                lam_vc=lam_vc,
                lam_cost=lam_cost,
                lam_vb=lam_vb,
                states=states,
                controls=controls,
            ),
            autotuner=autotuner,
            k_max=k_max,
            t_max=t_max,
            ep_tr=ep_tr,
            ep_vb=ep_vb,
            ep_vc=ep_vc,
        )

    def build_iteration(
        self,
        dis_continuous: Callable,
        dis_impulsive: Callable,
        jax_constraints: "LoweredJaxConstraints",
        solver_callback: Callable,
        settings: Config,
    ) -> Callable:
        """Fuse the discretizers, constraints, solver, and composite into one step."""
        return make_proxconvex_iteration(
            composite=self._composite,
            dis_continuous=dis_continuous,
            dis_impulsive=dis_impulsive,
            jax_constraints=jax_constraints,
            solver_callback=solver_callback,
            autotuner=self.autotuner,
            settings=settings,
        )

    def get_columns(self, verbosity: int = Verbosity.STANDARD) -> List[Column]:
        """Return iteration-table columns at the requested verbosity."""
        all_columns = self.BASE_COLUMNS + self.autotuner.COLUMNS + self.TAIL_COLUMNS
        return [col for col in all_columns if col.min_verbosity <= verbosity]

    def initialize(self, iteration_fn: Callable, emitter: Callable) -> None:
        """Store the fused iteration body and emitter (called by Problem.initialize)."""
        self._iteration_fn = iteration_fn
        self._emitter = emitter

    def step(
        self,
        state: AlgorithmState,
        history: AlgorithmHistory,
        params: dict,
        settings: Config,
    ) -> Tuple[AlgorithmState, bool]:
        """Execute one ProxConvex iteration and return ``(next_state, converged)``."""
        if self._iteration_fn is None:
            raise RuntimeError(
                "ProxConvex.step() called before initialize(). Call problem.initialize() first."
            )

        iter_index = int(state.k)

        t0 = time.time()
        next_state, diag = self._iteration_fn(state, params)
        jax.block_until_ready((next_state, diag))
        step_time = time.time() - t0

        if int(diag.status) != int(StatusCode.OPTIMAL):
            raise RuntimeError(
                f"Convex subproblem did not solve to optimality "
                f"(status={status_code_to_str(int(diag.status))!r}). Adjust solver "
                f"tolerances, rescale the problem, or use float_dtype='float64'."
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
        return next_state, bool(self.converged(next_state))

    def citation(self) -> List[str]:
        """Return BibTeX citation for the ProxConvex algorithm."""
        return [
            r"""@misc{uzun2025proxconvex,
  title={A Proximal Method for Composite Optimization with Smooth and Convex Components},
  author={Uzun, Samet and Luo, Dayou and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et and Aravkin, Aleksandr Y.},
  year={2025},
  eprint={2512.20602},
  archivePrefix={arXiv},
  primaryClass={math.OC}
}""",
        ]
