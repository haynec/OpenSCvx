"""Penalized Trust Region (PTR) successive convexification algorithm.

This module implements the PTR algorithm for solving non-convex trajectory
optimization problems through iterative convex approximation.
"""

import time
import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Union

import jax
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

from .augmented_lagrangian import AugmentedLagrangian
from .base import Algorithm, AlgorithmState, CandidateIterate
from .constant_proximal_weight import ConstantProximalWeight
from .ramp_proximal_weight import RampProximalWeight
from .weights import Weights

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints
    from openscvx.solvers import ConvexSolver
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State

    from .base import AutotuningBase

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
        """Initialize PTR with algorithm parameters and optional autotuner.

        Args:
            autotuner: Weight adaptation strategy. Defaults to
                :class:`AugmentedLagrangian` when ``None``.
            k_max: Maximum SCP iterations. Defaults to 200.
            t_max: Wall-clock time limit in seconds. ``None`` (default)
                disables the time limit.
            lam_prox: Trust region (proximal) weight. Either a float
                (applied uniformly to all states and controls) or a dict
                mapping state/control names to weights, e.g.
                ``{"velocity": 1e0, "thrust": 5e-1}``.  Dict values may
                be scalars, 1-D arrays for per-component weighting, or
                2-D arrays of shape ``(n_nodes, n_components)`` for
                per-node-per-component weighting.  Variables not in the
                dict default to ``1.0``. Defaults to 0.1.
            lam_vc: Virtual control penalty weight. Either a float
                (applied uniformly to all states) or a dict mapping state
                names to per-state weights, e.g.
                ``{"velocity": 1e1, "position": 5e0}``.  Dict values may
                be scalars, 1-D arrays for per-component weighting, or
                2-D arrays of shape ``(n_nodes-1, n_components)`` for
                per-node-per-component weighting.  States not in the dict
                default to ``1.0``. Defaults to 1.0.
            lam_cost: Cost weight. Either a float (applied to all
                minimize/maximize states) or a dict mapping state names
                to per-state weights, e.g.
                ``{"velocity": 1e-1, "time": 1e0}``.  Dict values may
                be arrays for per-component weighting, e.g.
                ``{"position": [0, 0, 1e-6]}``. Defaults to 0.01.
            lam_vb: Virtual buffer penalty weight. Defaults to 0.0.
            ep_tr: Trust region convergence tolerance. Defaults to 1e-4.
            ep_vb: Virtual buffer convergence tolerance. Defaults to 1e-4.
            ep_vc: Virtual control convergence tolerance. Defaults to 1e-8.
            states: Symbolic State objects (required when *lam_cost* or
                *lam_prox* is a dict). Normally provided automatically by
                :class:`Problem`.
            controls: Symbolic Control objects (required when *lam_prox*
                is a dict). Normally provided automatically by
                :class:`Problem`.
        """
        # Compiled infrastructure (set by initialize())
        self._solver: "ConvexSolver" = None
        self._discretization_solver: callable = None
        self._discretization_solver_impulsive: callable = None
        self._jax_constraints: "LoweredJaxConstraints" = None
        self._emitter: callable = None

        # Autotuner
        self.autotuner: "AutotuningBase" = (
            autotuner if autotuner is not None else AugmentedLagrangian()
        )

        # Store states/controls for later re-resolution (e.g. user sets
        # lam_cost or lam_prox to a new dict via the property setter after
        # construction).
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

    # -- Weight properties ---------------------------------------------------
    # These properties resolve dict-valued inputs (e.g. {"velocity": 1e0})
    # to arrays via Weights.resolve_lam_* helpers, then store the result
    # on self.weights.  During SCP iteration the autotuner may mutate the
    # values on self.weights directly (e.g. ramping lam_prox).  Those
    # in-flight changes are tracked in AlgorithmState weight histories.

    @staticmethod
    def _invoke_solver(solver: callable, *args):
        """Call either a compiled solver wrapper (.call) or a plain callable."""
        if hasattr(solver, "call"):
            return solver.call(*args)
        return solver(*args)

    @staticmethod
    def _block_until_ready_outputs(outputs: Tuple[object, ...]) -> None:
        """Finish any pending XLA work from discretization exports (warm-up helper)."""
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
        """Trust region (proximal) weight.

        May be a scalar or array for per-variable / per-node weighting.
        """
        return self.weights.lam_prox

    @lam_prox.setter
    def lam_prox(self, value: Union[float, Dict[str, Union[float, list]]]) -> None:
        self.weights.lam_prox = Weights.resolve_lam_prox(value, self._states, self._controls)

    @property
    def lam_vc(self) -> Union[float, np.ndarray]:
        """Virtual control penalty weight.

        Returns a float when a uniform scalar was provided, or an ndarray
        when per-state weights were given (via dict or array).
        """
        return self.weights.lam_vc

    @lam_vc.setter
    def lam_vc(self, value: Union[float, Dict[str, Union[float, list]]]) -> None:
        self.weights.lam_vc = Weights.resolve_lam_vc(value, self._states)

    @property
    def lam_cost(self) -> Union[float, np.ndarray]:
        """Cost weight.

        Returns a float when a uniform scalar was provided, or an ndarray
        of shape ``(n_states,)`` when per-state weights were given (via dict
        or array).
        """
        return self.weights.lam_cost

    @lam_cost.setter
    def lam_cost(self, value: Union[float, Dict[str, float]]) -> None:
        self.weights.lam_cost = Weights.resolve_lam_cost(value, self._states)

    @property
    def lam_vb(self) -> float:
        """Global virtual buffer penalty weight.

        Per-constraint overrides are set via ``.weight()`` on individual
        constraints.
        """
        return self.weights.lam_vb

    @lam_vb.setter
    def lam_vb(self, value: float) -> None:
        self.weights.lam_vb = float(value)

    def get_columns(self, verbosity: int = Verbosity.STANDARD) -> List[Column]:
        """Get the columns to display for iteration output.

        Combines base PTR columns with autotuner-specific columns,
        filtered by the requested verbosity level.

        Args:
            verbosity: Minimum verbosity level for columns to include.
                MINIMAL (1): Core metrics only (iter, cost, status)
                STANDARD (2): + timing, penalty terms
                FULL (3): + autotuning diagnostics

        Returns:
            List of Column specs filtered by verbosity level.

        Raises:
            AttributeError: If algorithm has not been initialized yet.
        """
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
        """Initialize PTR algorithm.

        Stores compiled infrastructure and performs a warm-start solve to
        initialize DPP and JAX jacobians. Also runs post-subproblem discretization
        on the throwaway CVX solution so XLA/export caches match the first ``step()``.

        Args:
            solver: Convex subproblem solver (e.g., CVXPySolver)
            discretization_solver: Compiled discretization solver
            jax_constraints: JIT-compiled constraint functions
            emitter: Callback for emitting iteration progress
            params: Problem parameters dictionary (for warm-start)
            settings: Configuration object (for warm-start)
            discretization_solver_impulsive: Optional impulsive/discrete
                discretization solver used to populate W/x_prop_plus/D_d/E_d.
        """
        # Store immutable infrastructure
        self._solver = solver
        self._discretization_solver = discretization_solver
        self._discretization_solver_impulsive = discretization_solver_impulsive
        self._jax_constraints = jax_constraints
        self._emitter = emitter

        # Set boundary conditions
        self._solver.update_boundary_conditions(
            x_init=settings.sim.x.initial,
            x_term=settings.sim.x.final,
        )

        # Create temporary state for initialization solve
        init_state = AlgorithmState.from_settings(settings, self.weights)

        # Solve a dumb problem to initialize DPP and JAX jacobians
        _, _, _, x_prop, V_multi_shoot = self._invoke_solver(
            self._discretization_solver, init_state.x, init_state.u.astype(float), params
        )

        init_state.add_discretization(V_multi_shoot.__array__())
        slice_imp = settings.sim.u.slice_impulsive
        has_impulsive = bool(slice_imp.stop > slice_imp.start)
        if has_impulsive and self._discretization_solver_impulsive is not None:
            u_init = init_state.u.astype(float)
            x0_prior = self._recover_prior_node_from_initial(settings, init_state.x[0])
            x_nodes_prior = np.vstack((x0_prior, np.asarray(x_prop)))
            _, _, _, W_multi_shoot = self._invoke_solver(
                self._discretization_solver_impulsive,
                x_nodes_prior,
                u_init,
                params,
            )
            init_state.add_impulsive_discretization(W_multi_shoot.__array__())
        (
            x_sol,
            u_sol,
            *_
        ) = self._subproblem(params, init_state, settings)

        # Prime the same exported discretization calls used after every subproblem in
        # step() (candidate trajectory). initialize() previously only discretized the
        # initial guess, so the first step() in solve() could still hit an XLA cache_miss
        # on post-CVX (x_sol, u_sol). Running that path here moves compilation into init.
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

    def step(
        self,
        state: AlgorithmState,
        params: dict,
        settings: Config,
    ) -> bool:
        """Execute one PTR iteration.

        Solves the convex subproblem, updates state in place, and checks
        convergence based on trust region, virtual buffer, and virtual
        control costs.

        Args:
            state: Mutable solver state (modified in place)
            params: Problem parameters dictionary (may change between steps)
            settings: Configuration object (may change between steps)

        Returns:
            True if J_tr, J_vb, and J_vc are all below their thresholds.

        Raises:
            RuntimeError: If initialize() has not been called.
        """
        if self._solver is None:
            raise RuntimeError(
                "PenalizedTrustRegion.step() called before initialize(). "
                "Call initialize() first to set up compiled infrastructure."
            )

        # Compute discretization before subproblem only for the first iteration
        if state.k == 1:
            t0 = time.time()
            _, _, _, x_prop, V_multi_shoot = self._invoke_solver(
                self._discretization_solver, state.x, state.u.astype(float), params
            )

            u_state = state.u.astype(float)
            x0_prior = self._recover_prior_node_from_initial(settings, state.x[0])
            x_nodes_prior = np.vstack((x0_prior, np.asarray(x_prop)))
            _, _, _, W_multi_shoot = self._invoke_solver(
                self._discretization_solver_impulsive, x_nodes_prior, u_state, params
            )
            dis_time = time.time() - t0

            state.add_discretization(V_multi_shoot.__array__())
            state.add_impulsive_discretization(W_multi_shoot.__array__())

        # Run the subproblem
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
        ) = self._subproblem(params, state, settings)

        candidate = CandidateIterate()
        candidate.x = x_sol
        candidate.u = u_sol
        candidate.J_lin = J_total

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

        dis_time = time.time() - t0

        candidate.V = V_multi_shoot.__array__()
        candidate.W = W_multi_shoot.__array__()
        candidate.x_prop = x_prop.__array__()
        candidate.x_prop_plus = x_prop_plus.__array__()
        candidate.D_d = D_d.__array__()
        candidate.E_d = E_d.__array__()

        # Update state in place by appending to history
        # The x_guess/u_guess properties will automatically return the latest entry
        candidate.VC = vc_mat
        candidate.TR = tr_mat

        state.J_tr = np.sum(np.array(J_tr_vec))
        state.J_vb = np.sum(np.array(J_vb_vec))
        state.J_vc = np.sum(np.array(J_vc_vec))

        # Update weights in state using configured autotuning method
        adaptive_state = self.autotuner.update_weights(
            state, candidate, self._jax_constraints, settings, params, self.weights
        )

        # Build emission data - only include nonlinear/reduction metrics when
        # the autotuner actually uses them (constant/ramp methods don't)
        use_full_metrics = not isinstance(
            self.autotuner, (ConstantProximalWeight, RampProximalWeight)
        )

        emission_data = {
            "iter": state.k,
            "dis_time": dis_time * 1000.0,
            "subprop_time": subprop_time * 1000.0,
            "J_tr": state.J_tr,
            "J_vb": state.J_vb,
            "J_vc": state.J_vc,
            "cost": cost[-1],
            # TODO: (haynec) log per-variable lam_prox detail (e.g. min/max range)
            "lam_prox": float(np.max(state.lam_prox)),
            "prob_stat": prob_stat,
            "adaptive_state": adaptive_state,
            "ep_tr": self.ep_tr,
            "ep_vb": self.ep_vb,
            "ep_vc": self.ep_vc,
        }

        # Only include nonlinear/reduction metrics when autotuner uses them
        # (constant/ramp methods don't compute these, so we don't emit them)
        if use_full_metrics:
            if len(state.pred_reduction_history) == 0:
                pred_reduction = 0.0
            else:
                pred_reduction = state.pred_reduction_history[-1]
            if len(state.actual_reduction_history) == 0:
                actual_reduction = 0.0
            else:
                actual_reduction = state.actual_reduction_history[-1]
            if len(state.acceptance_ratio_history) == 0:
                acceptance_ratio = 0.0
            else:
                acceptance_ratio = state.acceptance_ratio_history[-1]

            emission_data.update(
                {
                    "J_nonlin": candidate.J_nonlin,
                    "J_lin": candidate.J_lin,
                    "pred_reduction": pred_reduction,
                    "actual_reduction": actual_reduction,
                    "acceptance_ratio": acceptance_ratio,
                }
            )

        # Emit data
        self._emitter(emission_data)

        # Increment iteration counter
        state.k += 1

        # Return convergence status
        return (state.J_tr < self.ep_tr) and (state.J_vb < self.ep_vb) and (state.J_vc < self.ep_vc)

    def _subproblem(
        self,
        params: dict,
        state: AlgorithmState,
        settings: Config,
    ):
        """Solve a single convex subproblem.

        Uses stored infrastructure (solver, discretization_solver, jax_constraints)
        with per-step params and settings.

        Args:
            params: Problem parameters dictionary
            state: Current solver state
            settings: Configuration object

        Returns:
            Tuple containing solution data, costs, and timing information.
        """
        param_dict = params

        # Update solver with dynamics linearization
        self._solver.update_dynamics_linearization(
            x_bar=state.x,
            u_bar=state.u,
            A_d=state.A_d(),
            B_d=state.B_d(),
            C_d=state.C_d(),
            x_prop=state.x_prop(),
            x_prop_plus=state.x_prop_plus(),
            D_d=state.D_d(),
            E_d=state.E_d(),
        )

        # Build constraint linearization data
        # TODO: (norrisg) investigate why we are passing `0` for the node here
        nodal_linearizations = []
        if self._jax_constraints.nodal:
            for constraint in self._jax_constraints.nodal:
                # Evaluate constraint at all nodes (vmapped function returns shape (N,))
                g_full = np.asarray(constraint.func(state.x, state.u, 0, param_dict))
                grad_g_x_full = np.asarray(constraint.grad_g_x(state.x, state.u, 0, param_dict))
                grad_g_u_full = np.asarray(constraint.grad_g_u(state.x, state.u, 0, param_dict))

                # Ensure g is 1D with shape (N,) - squeeze any extra dimensions
                # This handles cases where constraint might return shape (N, 1) or similar
                g_full = np.squeeze(g_full)
                if g_full.ndim == 0:
                    # Scalar result - expand to (N,)
                    g_full = np.broadcast_to(g_full, (state.x.shape[0],))
                elif g_full.ndim > 1:
                    # Multi-dimensional result - flatten to (N,)
                    # This should not happen for properly decomposed constraints,
                    # but handle it gracefully
                    g_full = g_full.reshape(g_full.shape[0], -1).sum(axis=1)

                # Ensure grad_g_x and grad_g_u have correct shapes
                # grad_g_x should be (N, n_x), grad_g_u should be (N, n_u)
                if grad_g_x_full.ndim == 1:
                    # If 1D, it should be (n_x,) - broadcast to (N, n_x)
                    grad_g_x_full = np.broadcast_to(
                        grad_g_x_full, (state.x.shape[0], grad_g_x_full.shape[0])
                    )
                elif grad_g_x_full.ndim > 2:
                    # Flatten extra dimensions
                    grad_g_x_full = grad_g_x_full.reshape(grad_g_x_full.shape[0], -1)
                    # Take only first n_x columns
                    n_x = state.x.shape[1]
                    if grad_g_x_full.shape[1] > n_x:
                        grad_g_x_full = grad_g_x_full[:, :n_x]

                if grad_g_u_full.ndim == 1:
                    # If 1D, it should be (n_u,) - broadcast to (N, n_u)
                    grad_g_u_full = np.broadcast_to(
                        grad_g_u_full, (state.u.shape[0], grad_g_u_full.shape[0])
                    )
                elif grad_g_u_full.ndim > 2:
                    # Flatten extra dimensions
                    grad_g_u_full = grad_g_u_full.reshape(grad_g_u_full.shape[0], -1)
                    # Take only first n_u columns
                    n_u = state.u.shape[1]
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
                        "g": np.asarray(constraint.func(state.x, state.u, param_dict)),
                        "grad_g_X": np.asarray(constraint.grad_g_X(state.x, state.u, param_dict)),
                        "grad_g_U": np.asarray(constraint.grad_g_U(state.x, state.u, param_dict)),
                    }
                )

        # Update solver with constraint linearizations
        self._solver.update_constraint_linearizations(
            nodal=nodal_linearizations if nodal_linearizations else None,
            cross_node=cross_node_linearizations if cross_node_linearizations else None,
        )

        # Update solver with penalty weights
        self._solver.update_penalties(
            lam_prox=state.lam_prox,
            lam_cost=state.lam_cost,
            lam_vc=state.lam_vc,
            lam_vb_nodal=state.lam_vb_nodal,
            lam_vb_cross=state.lam_vb_cross,
        )

        # Solve the convex subproblem
        t0 = time.time()
        result = self._solver.solve()
        subprop_time = time.time() - t0

        # Extract unscaled trajectories from result
        x_new_guess = result.x
        u_new_guess = result.u

        # Calculate costs from boundary conditions using utility function
        # Note: The original code only considered final_type, but the utility handles both
        # Here we maintain backward compatibility by only using final_type
        costs = [0]
        for i, bc_type in enumerate(settings.sim.x.final_type):
            if bc_type == "Minimize":
                costs += x_new_guess[:, i]
            elif bc_type == "Maximize":
                costs -= x_new_guess[:, i]

        # Create the block diagonal matrix using jax.numpy.block
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

        # Calculate J_tr_vec using the JAX-compatible block diagonal matrix
        tr_mat = inv_block_diag @ np.hstack((x_new_guess - state.x, u_new_guess - state.u)).T
        J_tr_vec = la.norm(tr_mat, axis=0) ** 2
        vc_mat = np.abs(settings.sim.inv_S_x @ result.nu.T).T
        J_vc_vec = np.sum(vc_mat, axis=1)

        # Sum nodal constraint violations
        J_vb_vec = 0
        for nu_vb_arr in result.nu_vb:
            J_vb_vec += np.maximum(0, nu_vb_arr)

        # Add cross-node constraint violations
        for nu_vb_cross_val in result.nu_vb_cross:
            J_vb_vec += np.maximum(0, nu_vb_cross_val)

        # Convex constraints are already handled in the OCP, no processing needed here
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
        """Return BibTeX citations for the PTR algorithm.

        Returns:
            List containing the BibTeX entry for the PTR paper.
        """
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
