"""Penalized Trust Region (PTR) successive convexification algorithm.

This module implements the PTR algorithm for solving non-convex trajectory
optimization problems through iterative convex approximation.
"""

import time
import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpy.linalg as la

from openscvx.config import Config
from openscvx.utils.printing import (
    Column,
    Verbosity,
    color_J_tr,
    color_J_vb,
    color_J_vc,
    color_prob_stat,
)

from ..autotuner.augmented_lagrangian import AugmentedLagrangian
from ..autotuner.constant_proximal_weight import ConstantProximalWeight
from ..autotuner.ramp_proximal_weight import RampProximalWeight
from ..base import (
    Algorithm,
    AlgorithmHistory,
    AlgorithmState,
    CandidateIterate,
    adaptive_state_code_to_str,
)
from ..weights import Weights

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints
    from openscvx.solvers import ConvexSolver
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State

    from ..base import AutotuningBase

warnings.filterwarnings("ignore")


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
        Column("dis_time", "Dis (ms)", 8, "{:6.2f}", min_verbosity=Verbosity.STANDARD),
        Column("subprop_time", "Solve (ms)", 10, "{:6.2f}", min_verbosity=Verbosity.STANDARD),
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
        # Compiled infrastructure (set by initialize())
        self._solver: "ConvexSolver" = None
        self._discretization_solver: callable = None
        self._discretization_solver_impulsive: callable = None
        self._jax_constraints: "LoweredJaxConstraints" = None
        self._emitter: callable = None
        # The three private fields below — ``_jit_update_weights``,
        # ``_iter_index``, ``_last_scalars`` — are scaffolding for the
        # Python-driven SCP loop: they keep the inner JIT cache stable and
        # dodge per-iteration host/device syncs. Removable once the SCP loop
        # body is a single JAX trace.
        #
        # JIT'd autotuner step (built lazily in initialize() to fuse the
        # hundreds of small jnp ops in `update_weights` into a single dispatch
        # per iteration). Captures `nodal_constraints`, `settings`, `weights`
        # as closure constants — they don't change within a solve.
        self._jit_update_weights: Callable | None = None

        # Python-side mirror of ``state.k``. Kept in lockstep with the pytree
        # field so the Python loop can read the iteration index without a
        # ``int(state.k)`` device sync each step.
        self._iter_index: int = 1

        # Host-side cache of the diagnostic scalars produced by the last
        # ``step()`` call. ``record_iteration`` pulls these in one coalesced
        # ``jax.device_get`` and stashes them here so external readers
        # (``problem.step()``'s return dict) don't re-sync per iteration.
        self._last_scalars: dict = {}

        # Autotuner
        self.autotuner: "AutotuningBase" = (
            autotuner if autotuner is not None else AugmentedLagrangian()
        )

        # Store states/controls for later re-resolution.
        self._states: List["State"] = states
        self._controls: List["Control"] = controls

        # SCP weights (grouped dataclass, dict inputs expanded to arrays)
        self.weights = Weights.build(
            lam_prox=lam_prox,
            lam_vc=lam_vc,
            lam_cost=lam_cost,
            lam_vb=lam_vb,
            states=states,
            controls=controls,
        )

        # SCP convergence parameters
        self.k_max = k_max
        self.t_max = t_max
        self.ep_tr = ep_tr
        self.ep_vb = ep_vb
        self.ep_vc = ep_vc

    @staticmethod
    def _invoke_solver(solver: callable, *args):
        """Call either a compiled solver wrapper (.call) or a plain callable."""
        if hasattr(solver, "call"):
            return solver.call(*args)
        return solver(*args)

    def _build_jit_update_weights(
        self,
        jax_constraints: "LoweredJaxConstraints",
        settings: Config,
    ) -> Callable:
        """Compile a JIT'd autotuner step bound to the current solve context.

        ``update_weights`` issues hundreds of small ``jnp`` ops on per-iterate
        scalars and arrays; outside ``jax.jit`` each one round-trips through
        the dispatcher. Folding them into a single compiled function gives an
        order-of-magnitude wall-clock improvement per SCP iteration.

        Closure captures ``jax_constraints`` / ``settings`` only: they're fixed
        for the duration of a solve and contain non-pytree leaves (dataclasses
        of compiled closures, pydantic config) that JAX cannot route through
        positional arguments. The autotuners' reset weights live on
        ``state.lam_cost_init``, so weight mutations between solves propagate
        through the pytree without rebuilding the closure.

        Removable once the SCP loop body is a single JAX trace — the loop-level
        JIT subsumes this inner one and the dict ↔ ``CandidateIterate`` adapter
        below can be deleted along with the duplicate adapter in ``step()``.
        """
        autotuner = self.autotuner

        def fn(state, candidate_dict, params):
            cand = CandidateIterate()
            cand.x = candidate_dict["x"]
            cand.u = candidate_dict["u"]
            cand.x_prop = candidate_dict["x_prop"]
            cand.x_prop_plus = candidate_dict["x_prop_plus"]
            cand.J_lin = candidate_dict["J_lin"]
            return autotuner.update_weights(state, cand, jax_constraints, settings, params)

        return jax.jit(fn)

    @staticmethod
    def _block_until_ready_outputs(outputs: Tuple[object, ...]) -> None:
        """Finish any pending XLA work from discretization exports (warm-up helper).

        Removable once the SCP loop body is a single JAX trace — the warm-up
        path goes away with it.
        """
        jax.block_until_ready(outputs)

    def _recover_prior_node_from_initial(
        self,
        settings: Config,
        x0_fallback: np.ndarray,
    ) -> np.ndarray:
        """Build node-0 prior state from initial conditions (fixed entries exact)."""
        x0_prior = np.asarray(x0_fallback, dtype=float).reshape(-1).copy()
        x0_init = np.asarray(settings.sim.x.initial, dtype=float).reshape(-1)
        is_fixed = np.asarray(settings.sim.x.initial_type) == "Fix"
        x0_prior[is_fixed] = x0_init[is_fixed]
        return x0_prior.reshape(1, -1)

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

    def get_columns(self, verbosity: int = Verbosity.STANDARD) -> List[Column]:
        """Get the columns to display for iteration output."""
        all_columns = self.BASE_COLUMNS + self.autotuner.COLUMNS + self.TAIL_COLUMNS
        return [col for col in all_columns if col.min_verbosity <= verbosity]

    def initialize(
        self,
        solver: "ConvexSolver",
        discretization_solver: callable,
        jax_constraints: "LoweredJaxConstraints",
        emitter: callable,
        params: dict,
        settings: Config,
        discretization_solver_impulsive: Callable[..., object] | None = None,
    ) -> None:
        """Initialize PTR and warm up the compiled infrastructure."""
        self._solver = solver
        self._discretization_solver = discretization_solver
        self._discretization_solver_impulsive = discretization_solver_impulsive
        self._jax_constraints = jax_constraints
        self._emitter = emitter
        # Cheap autotuners (ConstantProximalWeight / RampProximalWeight) opt out
        # of the JIT wrapper via ``JIT_UPDATE_WEIGHTS=False`` on the class: their
        # bodies are a handful of ``jnp`` ops, and JAX's eager dispatch is cheaper
        # than the pytree-flatten / dict-pack overhead the JIT'd closure adds.
        if self.autotuner.JIT_UPDATE_WEIGHTS:
            self._jit_update_weights = self._build_jit_update_weights(jax_constraints, settings)
        else:
            self._jit_update_weights = None
        self._iter_index = 1

        self._solver.update_boundary_conditions(
            x_init=settings.sim.x.initial,
            x_term=settings.sim.x.final,
        )

        # Throwaway state/history for warm-up only.
        init_state = AlgorithmState.from_settings(settings, self.weights)
        init_history = AlgorithmHistory.from_settings(settings)

        # Warm up the dynamics discretization on the initial guess.
        x_init = np.asarray(init_state.x)
        u_init = np.asarray(init_state.u, dtype=float)
        _, _, _, x_prop, V_multi_shoot = self._invoke_solver(
            self._discretization_solver, x_init, u_init, params
        )
        init_history.add_discretization(V_multi_shoot.__array__())
        slice_imp = settings.sim.u.slice_impulsive
        has_impulsive = bool(slice_imp.stop > slice_imp.start)
        if has_impulsive and self._discretization_solver_impulsive is not None:
            x0_prior = self._recover_prior_node_from_initial(settings, x_init[0])
            x_nodes_prior = np.vstack((x0_prior, np.asarray(x_prop)))
            _, _, _, W_multi_shoot = self._invoke_solver(
                self._discretization_solver_impulsive,
                x_nodes_prior,
                u_init,
                params,
            )
            init_history.add_impulsive_discretization(W_multi_shoot.__array__())

        # Warm up the subproblem solver (DPP cache, JAX jacobians).
        (x_sol, u_sol, *_) = self._subproblem(params, init_state, init_history, settings)

        # Prime the post-CVX discretization path so the first real step() does
        # not pay an XLA cache miss on (x_sol, u_sol).
        cont_out = self._invoke_solver(
            self._discretization_solver, x_sol, u_sol.astype(float), params
        )
        x_prop_c = cont_out[3]
        u_candidate = u_sol.astype(float)
        x0_prior_c = self._recover_prior_node_from_initial(settings, x_sol[0])
        x_nodes_prior_c = np.vstack((x0_prior_c, np.asarray(x_prop_c)))
        if self._discretization_solver_impulsive is not None:
            imp_out = self._invoke_solver(
                self._discretization_solver_impulsive,
                x_nodes_prior_c,
                u_candidate,
                params,
            )
            self._block_until_ready_outputs(cont_out + imp_out)
        else:
            self._block_until_ready_outputs(cont_out)

        # Warm up the JIT'd autotuner step so the first real iteration pays
        # only the cached-dispatch cost. We construct a dummy candidate from
        # the warm-up subproblem result so the trace sees the actual array
        # shapes. Eager-dispatch autotuners (``JIT_UPDATE_WEIGHTS=False``) skip
        # this — there is no JIT cache to prime.
        if self._jit_update_weights is not None:
            if self._discretization_solver_impulsive is not None:
                x_prop_plus_c = imp_out[0]
            else:
                x_prop_plus_c = jnp.zeros((settings.sim.n, settings.sim.n_states))
            warm_candidate = {
                "x": jnp.asarray(x_sol),
                "u": jnp.asarray(u_sol),
                "x_prop": jnp.asarray(x_prop_c),
                "x_prop_plus": jnp.asarray(x_prop_plus_c),
                "J_lin": jnp.asarray(0.0),
            }
            # Trace under both k=1 (INITIAL branch) and k>1 (acceptance-ratio
            # branch) so neither path recompiles inside the SCP loop. The first
            # warm-up traces against an uncommitted state (matching the iter-1
            # input from ``AlgorithmState.from_settings``); the second traces
            # against the *returned* state (committed, matching iter ≥ 2 where
            # the state comes back from the JIT). Without this two-phase warm-
            # up, JAX caches by argument committedness and the iter-2 dispatch
            # misses.
            #
            # Removable once the SCP loop body is a single JAX trace — the
            # entire warm-up block below becomes unnecessary.
            warm_state = init_state.replace(
                x_prop=jnp.asarray(x_prop_c),
                x_prop_plus=jnp.asarray(x_prop_plus_c),
            )
            warm_out = self._jit_update_weights(warm_state, warm_candidate, params)
            jax.block_until_ready(warm_out)
            warm_state_k2 = warm_out.replace(k=jnp.asarray(2, dtype=jnp.int32))
            jax.block_until_ready(self._jit_update_weights(warm_state_k2, warm_candidate, params))

    def step(
        self,
        state: AlgorithmState,
        history: AlgorithmHistory,
        params: dict,
        settings: Config,
    ) -> Tuple[AlgorithmState, bool]:
        """Execute one PTR iteration and return ``(next_state, converged)``.

        Discretizes the current iterate (on iter 1 only — subsequent iters
        reuse the candidate discretization stored on history), solves the
        convex subproblem, discretizes the candidate, hands everything to
        the autotuner, records history, and emits diagnostics.
        """
        if self._solver is None:
            raise RuntimeError(
                "PenalizedTrustRegion.step() called before initialize(). "
                "Call initialize() first to set up compiled infrastructure."
            )

        x_state = np.asarray(state.x)
        u_state = np.asarray(state.u, dtype=float)

        # Parallel Python iteration counter. ``state.k`` is a jnp scalar; on
        # tiny problems the ``int(state.k)`` cast dominates each step because
        # it forces a CPU<->device sync. We track the same value as a Python
        # int and only re-sync into ``state.k`` once per step.
        iter_index = self._iter_index

        # Iter 1: discretize the initial guess so the subproblem and the
        # autotuner have something to linearize about. Subsequent iters
        # reuse the candidate discretization from the previous iter.
        if iter_index == 1:
            t0 = time.time()
            _, _, _, x_prop_init, V_multi_shoot = self._invoke_solver(
                self._discretization_solver, x_state, u_state, params
            )
            x0_prior = self._recover_prior_node_from_initial(settings, x_state[0])
            x_nodes_prior = np.vstack((x0_prior, np.asarray(x_prop_init)))
            x_prop_plus_init, _, _, W_multi_shoot = self._invoke_solver(
                self._discretization_solver_impulsive, x_nodes_prior, u_state, params
            )
            history.add_discretization(V_multi_shoot.__array__())
            history.add_impulsive_discretization(W_multi_shoot.__array__())

            # Mirror the discretization onto the state pytree so the autotuner
            # can read it as the "previous iterate" propagation on iter 2. Keep
            # the JAX arrays as-is (committed sharding) so the JIT cache hit
            # established by the ``initialize()`` warm-up survives.
            state = state.replace(
                x_prop=x_prop_init,
                x_prop_plus=x_prop_plus_init,
            )
            iter1_dis_time = time.time() - t0
        else:
            iter1_dis_time = 0.0

        # Subproblem.
        (
            x_sol,
            u_sol,
            cost,
            J_total,
            J_vb_vec,
            J_vc_vec,
            J_tr_vec,
            prob_stat,
            subprop_time,
            vc_mat,
            tr_mat,
        ) = self._subproblem(params, state, history, settings)

        candidate = CandidateIterate()
        candidate.x = x_sol
        candidate.u = u_sol
        candidate.J_lin = J_total

        # Discretize candidate.
        t0 = time.time()
        _, _, _, x_prop, V_multi_shoot = self._invoke_solver(
            self._discretization_solver, candidate.x, candidate.u.astype(float), params
        )
        u_candidate = candidate.u.astype(float)
        x0_prior = self._recover_prior_node_from_initial(settings, candidate.x[0])
        x_nodes_prior = np.vstack((x0_prior, np.asarray(x_prop)))
        x_prop_plus, D_d, E_d, W_multi_shoot = self._invoke_solver(
            self._discretization_solver_impulsive, x_nodes_prior, u_candidate, params
        )
        dis_time = iter1_dis_time + (time.time() - t0)

        # V, W, D_d, E_d are consumed host-side by ``history.record_iteration``
        # only — pull them off the device once here so the recorder doesn't have
        # to. x_prop / x_prop_plus stay as JAX arrays: they flow straight back
        # into the JIT'd autotuner closure, and round-tripping them through a
        # numpy array would lose their committed sharding and miss the JIT cache
        # established by the ``initialize()`` warm-up.
        candidate.V = V_multi_shoot.__array__()
        candidate.W = W_multi_shoot.__array__()
        candidate.x_prop = x_prop
        candidate.x_prop_plus = x_prop_plus
        candidate.D_d = D_d.__array__()
        candidate.E_d = E_d.__array__()
        candidate.VC = vc_mat
        candidate.TR = tr_mat

        # Roll J_* scalars onto the pytree before the autotuner runs; the
        # autotuner returns the next-iterate state, which we then thread back
        # to history and the emitter. On the JIT path the values are strong-
        # typed and committed (matching the rest of the pytree, which comes
        # back from the JIT'd ``update_weights`` already committed) so the JIT
        # cache key is stable across iterations.
        #
        # The numpy → float → jnp → device_put round-trip is removable once the
        # SCP loop body is a single JAX trace; the J_* sums can then live on
        # the pytree directly with no cache-key gymnastics.
        if self._jit_update_weights is not None:
            dev = state.x.sharding
            state = state.replace(
                J_tr=jax.device_put(
                    jnp.asarray(float(np.sum(np.array(J_tr_vec))), dtype=state.J_tr.dtype), dev
                ),
                J_vb=jax.device_put(
                    jnp.asarray(float(np.sum(np.array(J_vb_vec))), dtype=state.J_vb.dtype), dev
                ),
                J_vc=jax.device_put(
                    jnp.asarray(float(np.sum(np.array(J_vc_vec))), dtype=state.J_vc.dtype), dev
                ),
            )

            # Autotuner: pure functional update on the pytree. We dispatch
            # through the JIT'd closure built in initialize() so the hundreds
            # of small jnp ops inside `update_weights` fuse into a single
            # compiled call. The candidate is bounced through a dict because
            # `CandidateIterate` is a mutable dataclass not registered as a JAX
            # pytree. Match the JIT-closure trace signature established in
            # initialize()'s warm-up: pre-converted JAX f32 arrays for the
            # array leaves and a weak-typed jnp scalar for J_lin. CVXPy
            # returns x/u as f64 numpy and J_lin as a Python float; passing
            # them raw would miss the warm-up's cache key (f32/strong vs
            # f64/weak). x_prop / x_prop_plus already come off the JAX
            # discretizer with committed sharding — keep them as-is.
            #
            # Removable once the SCP loop body is a single JAX trace:
            # ``candidate`` flows in as a registered pytree and the dict
            # adapter (duplicated in ``_build_jit_update_weights``)
            # disappears along with the wrapping JIT.
            candidate_dict = {
                "x": jnp.asarray(candidate.x),
                "u": jnp.asarray(candidate.u),
                "x_prop": candidate.x_prop,
                "x_prop_plus": candidate.x_prop_plus,
                "J_lin": jnp.asarray(float(candidate.J_lin)),
            }
            state = self._jit_update_weights(state, candidate_dict, params)
        else:
            # Eager-dispatch path for cheap autotuners (a few jnp ops; the
            # JIT'd closure's pytree-flatten overhead would dominate). Skip
            # the JIT-cache-stability gymnastics: the J_* fields are written
            # as plain jnp scalars (``record_iteration`` pulls them off the
            # device in a single coalesced ``jax.device_get`` either way), the
            # candidate stays a ``CandidateIterate`` (no dict pack), and
            # ``update_weights`` is called directly.
            state = state.replace(
                J_tr=jnp.asarray(np.sum(np.asarray(J_tr_vec)), dtype=state.J_tr.dtype),
                J_vb=jnp.asarray(np.sum(np.asarray(J_vb_vec)), dtype=state.J_vb.dtype),
                J_vc=jnp.asarray(np.sum(np.asarray(J_vc_vec)), dtype=state.J_vc.dtype),
            )
            state = self.autotuner.update_weights(
                state, candidate, self._jax_constraints, settings, params
            )

        # History bookkeeping (Python-side, never traced). The recorder
        # transfers everything it needs (diagnostic scalars, weight arrays,
        # adaptive-state code) off the device in one coalesced
        # ``jax.device_get`` so we only pay one CPU<->device synchronization
        # per iteration, no matter how many fields it reads.
        use_full_metrics = not isinstance(
            self.autotuner, (ConstantProximalWeight, RampProximalWeight)
        )
        scalars, lam_prox_np = history.record_iteration(
            state, candidate, record_diagnostics=use_full_metrics
        )
        self._last_scalars = scalars

        emission_data = {
            "iter": iter_index,
            "dis_time": dis_time * 1000.0,
            "subprop_time": subprop_time * 1000.0,
            "J_tr": scalars["J_tr"],
            "J_vb": scalars["J_vb"],
            "J_vc": scalars["J_vc"],
            "cost": cost[-1],
            # TODO: (haynec) log per-variable lam_prox detail (e.g. min/max range)
            "lam_prox": float(np.max(lam_prox_np)),
            "prob_stat": prob_stat,
            "adaptive_state": adaptive_state_code_to_str(scalars["adaptive_state_code"]),
            "ep_tr": self.ep_tr,
            "ep_vb": self.ep_vb,
            "ep_vc": self.ep_vc,
        }

        if use_full_metrics:
            emission_data.update(
                {
                    "J_nonlin": scalars["J_nonlin"],
                    "J_lin": float(candidate.J_lin),
                    "pred_reduction": scalars["predicted_reduction"],
                    "actual_reduction": scalars["actual_reduction"],
                    "acceptance_ratio": scalars["acceptance_ratio"],
                }
            )

        self._emitter(emission_data)

        # Advance the Python iteration mirror and bump ``state.k`` so external
        # readers (``problem.py``'s loop, ``OptimizationResults.scp_k``) stay
        # in sync. ``state.k + 1`` dispatches a JIT'd add — compiled once at
        # iter 1 (~10 ms one-time) and then a cheap kernel; doing this on the
        # device keeps the result committed and matches the rest of the pytree
        # so the autotuner's cache key is stable.
        self._iter_index = iter_index + 1
        state = state.replace(k=state.k + 1)

        converged = (
            (scalars["J_tr"] < self.ep_tr)
            and (scalars["J_vb"] < self.ep_vb)
            and (scalars["J_vc"] < self.ep_vc)
        )
        return state, converged

    def _subproblem(
        self,
        params: dict,
        state: AlgorithmState,
        history: AlgorithmHistory,
        settings: Config,
    ):
        """Solve a single convex subproblem against the latest linearization.

        Reads the dynamics linearization from ``history.discretizations[-1]``
        and the iterate / weight values from ``state``.
        """
        param_dict = params

        x_bar = np.asarray(state.x)
        u_bar = np.asarray(state.u)

        self._solver.update_dynamics_linearization(
            x_bar=x_bar,
            u_bar=u_bar,
            A_d=history.A_d(),
            B_d=history.B_d(),
            C_d=history.C_d(),
            x_prop=history.x_prop(),
            x_prop_plus=history.x_prop_plus(),
            D_d=history.D_d(),
            E_d=history.E_d(),
        )

        nodal_linearizations = []
        if self._jax_constraints.nodal:
            for constraint in self._jax_constraints.nodal:
                g_full = np.asarray(constraint.func(x_bar, u_bar, 0, param_dict))
                grad_g_x_full = np.asarray(constraint.grad_g_x(x_bar, u_bar, 0, param_dict))
                grad_g_u_full = np.asarray(constraint.grad_g_u(x_bar, u_bar, 0, param_dict))

                g_full = np.squeeze(g_full)
                if g_full.ndim == 0:
                    g_full = np.broadcast_to(g_full, (x_bar.shape[0],))
                elif g_full.ndim > 1:
                    g_full = g_full.reshape(g_full.shape[0], -1).sum(axis=1)

                if grad_g_x_full.ndim == 1:
                    grad_g_x_full = np.broadcast_to(
                        grad_g_x_full, (x_bar.shape[0], grad_g_x_full.shape[0])
                    )
                elif grad_g_x_full.ndim > 2:
                    grad_g_x_full = grad_g_x_full.reshape(grad_g_x_full.shape[0], -1)
                    n_x = x_bar.shape[1]
                    if grad_g_x_full.shape[1] > n_x:
                        grad_g_x_full = grad_g_x_full[:, :n_x]

                if grad_g_u_full.ndim == 1:
                    grad_g_u_full = np.broadcast_to(
                        grad_g_u_full, (u_bar.shape[0], grad_g_u_full.shape[0])
                    )
                elif grad_g_u_full.ndim > 2:
                    grad_g_u_full = grad_g_u_full.reshape(grad_g_u_full.shape[0], -1)
                    n_u = u_bar.shape[1]
                    if grad_g_u_full.shape[1] > n_u:
                        grad_g_u_full = grad_g_u_full[:, :n_u]

                nodal_linearizations.append(
                    {
                        "g": g_full,
                        "grad_g_x": grad_g_x_full,
                        "grad_g_u": grad_g_u_full,
                    }
                )

        cross_node_linearizations = []
        if self._jax_constraints.cross_node:
            for constraint in self._jax_constraints.cross_node:
                cross_node_linearizations.append(
                    {
                        "g": np.asarray(constraint.func(x_bar, u_bar, param_dict)),
                        "grad_g_X": np.asarray(constraint.grad_g_X(x_bar, u_bar, param_dict)),
                        "grad_g_U": np.asarray(constraint.grad_g_U(x_bar, u_bar, param_dict)),
                    }
                )

        self._solver.update_constraint_linearizations(
            nodal=nodal_linearizations if nodal_linearizations else None,
            cross_node=cross_node_linearizations if cross_node_linearizations else None,
        )

        self._solver.update_penalties(
            lam_prox=np.asarray(state.lam_prox),
            lam_cost=np.asarray(state.lam_cost),
            lam_vc=np.asarray(state.lam_vc),
            lam_vb_nodal=np.asarray(state.lam_vb_nodal),
            lam_vb_cross=np.asarray(state.lam_vb_cross),
        )
        self._solver.update_proximal_terms()

        t0 = time.time()
        result = self._solver.solve()
        subprop_time = time.time() - t0

        x_new_guess = result.x
        u_new_guess = result.u

        costs = [0]
        for i, bc_type in enumerate(settings.sim.x.final_type):
            if bc_type == "Minimize":
                costs += x_new_guess[:, i]
            elif bc_type == "Maximize":
                costs -= x_new_guess[:, i]

        inv_block_diag = np.block(
            [
                [
                    settings.sim.inv_S_x,
                    np.zeros((settings.sim.inv_S_x.shape[0], settings.sim.inv_S_u.shape[1])),
                ],
                [
                    np.zeros((settings.sim.inv_S_u.shape[0], settings.sim.inv_S_x.shape[1])),
                    settings.sim.inv_S_u,
                ],
            ]
        )

        tr_mat = inv_block_diag @ np.hstack((x_new_guess - x_bar, u_new_guess - u_bar)).T
        J_tr_vec = la.norm(tr_mat, axis=0) ** 2
        vc_mat = np.abs(settings.sim.inv_S_x @ result.nu.T).T
        J_vc_vec = np.sum(vc_mat, axis=1)

        J_vb_vec = 0
        for nu_vb_arr in result.nu_vb:
            J_vb_vec += np.maximum(0, nu_vb_arr)

        for nu_vb_cross_val in result.nu_vb_cross:
            J_vb_vec += np.maximum(0, nu_vb_cross_val)

        return (
            x_new_guess,
            u_new_guess,
            costs,
            result.cost,
            J_vb_vec,
            J_vc_vec,
            J_tr_vec,
            result.status,
            subprop_time,
            vc_mat,
            tr_mat,
        )

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
