"""Penalized Trust Region (PTR) successive convexification algorithm.

This module implements the PTR algorithm for solving non-convex trajectory
optimization problems through iterative convex approximation.
"""

import time
import warnings
from typing import TYPE_CHECKING, Dict, List, Set, Union

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
from .base import Algorithm, AlgorithmState, CandidateIterate, Weights
from .constant_proximal_weight import ConstantProximalWeight
from .ramp_proximal_weight import RampProximalWeight

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints
    from openscvx.solvers import ConvexSolver
    from openscvx.symbolic.expr.state import State

    from .base import AutotuningBase


def _expand_lam_cost_dict(
    lam_cost_dict: Dict[str, float],
    states: List["State"],
) -> np.ndarray:
    """Expand a ``{state_name: weight}`` dict to a per-state weight array.

    Maps user-provided per-state cost weights to a dense array of shape
    ``(n_states,)`` using each state's ``_slice``.  States without a
    minimize/maximize objective receive weight 0.  States **with** a
    minimize/maximize objective **must** appear in the dict.

    Args:
        lam_cost_dict: Mapping from state names to scalar cost weights.
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

    # Identify states that have minimize/maximize objectives
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

    # Fill the array
    for state in states:
        if state.name in lam_cost_dict:
            lam_arr[state._slice] = lam_cost_dict[state.name]

    return lam_arr


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
        lam_prox: float = 1e0,
        lam_vc: float = 1e1,
        lam_cost: Union[float, Dict[str, float]] = 1e-1,
        lam_vb: float = 0.0,
        ep_tr: float = 1e-4,
        ep_vb: float = 1e-4,
        ep_vc: float = 1e-8,
    ):
        """Initialize PTR with algorithm parameters and optional autotuner.

        Args:
            autotuner: Weight adaptation strategy. Defaults to
                :class:`AugmentedLagrangian` when ``None``.
            k_max: Maximum SCP iterations. Defaults to 200.
            lam_prox: Trust region (proximal) weight. Defaults to 1.0.
            lam_vc: Virtual control penalty weight. Defaults to 10.0.
            lam_cost: Cost weight. Either a float (applied to all
                minimize/maximize states) or a dict mapping state names
                to per-state weights, e.g.
                ``{"velocity": 1e-1, "time": 1e0}``. Defaults to 0.1.
            lam_vb: Virtual buffer penalty weight. Defaults to 0.0.
            ep_tr: Trust region convergence tolerance. Defaults to 1e-4.
            ep_vb: Virtual buffer convergence tolerance. Defaults to 1e-4.
            ep_vc: Virtual control convergence tolerance. Defaults to 1e-8.
        """
        # Compiled infrastructure (set by initialize())
        self._solver: "ConvexSolver" = None
        self._discretization_solver: callable = None
        self._jax_constraints: "LoweredJaxConstraints" = None
        self._emitter: callable = None

        # Autotuner
        self.autotuner: "AutotuningBase" = (
            autotuner if autotuner is not None else AugmentedLagrangian()
        )

        # Raw user-specified lam_cost (may be dict, expanded in initialize).
        self._lam_cost_spec = lam_cost
        self._states: List["State"] = None

        # SCP weights (grouped dataclass, normalized for numerical conditioning)
        # If lam_cost is a dict, defer to initialize(); use 0.0 as placeholder.
        lam_cost_init = lam_cost if not isinstance(lam_cost, dict) else 0.0
        self.weights = Weights(
            lam_prox=lam_prox,
            lam_vc=lam_vc,
            lam_cost=lam_cost_init,
            lam_vb=lam_vb,
        )
        if not isinstance(lam_cost, dict):
            self.weights.normalize()

        # SCP convergence parameters
        self.k_max = k_max
        self.ep_tr = ep_tr
        self.ep_vb = ep_vb
        self.ep_vc = ep_vc

    def _expand_lam_cost(self) -> None:
        """Expand dict ``_lam_cost_spec`` into weights and normalize."""
        lam_arr = _expand_lam_cost_dict(self._lam_cost_spec, self._states)
        self.weights._raw_lam_cost = lam_arr.copy()
        self.weights.lam_cost = lam_arr.copy()
        self.weights.normalize()

    # -- Weight properties ---------------------------------------------------
    # These properties are the **source of truth** for user-facing weight
    # values.  Getters return the raw (user-specified, pre-normalization)
    # value.  Setters update the raw value and re-normalize ``self.weights``.
    #
    # During SCP iteration the autotuner may mutate the *normalized* values
    # on ``self.weights`` directly (e.g. ramping ``lam_prox``).  Those
    # in-flight changes are tracked in ``AlgorithmState`` weight histories
    # but do not affect the raw values returned here.

    @property
    def lam_prox(self) -> float:
        """Trust region (proximal) weight.

        This is the user-specified value before normalization. Setting this
        property triggers automatic re-normalization of all weights.

        !!! note
            The autotuner may modify the normalized weight in
            ``self.weights.lam_prox`` during iteration. Those changes are
            internal and do not alter the value returned here.
        """
        return self.weights._raw_lam_prox

    @lam_prox.setter
    def lam_prox(self, value: float) -> None:
        self.weights._raw_lam_prox = value
        self.weights.normalize()

    @property
    def lam_vc(self) -> float:
        """Virtual control penalty weight.

        This is the user-specified value before normalization. Setting this
        property triggers automatic re-normalization of all weights.

        !!! note
            The autotuner may modify the normalized weight in
            ``self.weights.lam_vc`` during iteration. Those changes are
            internal and do not alter the value returned here.
        """
        return self.weights._raw_lam_vc

    @lam_vc.setter
    def lam_vc(self, value: float) -> None:
        self.weights._raw_lam_vc = value
        self.weights.normalize()

    @property
    def lam_cost(self) -> Union[float, Dict[str, float], np.ndarray]:
        """Cost weight.

        This is the user-specified value before normalization. Setting this
        property triggers automatic re-normalization of all weights.

        Can be a float (uniform weight for all cost states), a dict mapping
        state names to per-state weights, or an ndarray after expansion.

        !!! note
            The autotuner may modify the normalized weight in
            ``self.weights.lam_cost`` during iteration. Those changes are
            internal and do not alter the value returned here.
        """
        return self._lam_cost_spec

    @lam_cost.setter
    def lam_cost(self, value: Union[float, Dict[str, float]]) -> None:
        self._lam_cost_spec = value
        if isinstance(value, dict):
            if self._states is None:
                return  # Will be expanded in initialize()
            self._expand_lam_cost()
        else:
            self.weights._raw_lam_cost = value
            self.weights.normalize()

    @property
    def lam_vb(self) -> float:
        """Virtual buffer penalty weight.

        This is the user-specified value before normalization. Setting this
        property triggers automatic re-normalization of all weights.

        !!! note
            The autotuner may modify the normalized weight in
            ``self.weights.lam_vb`` during iteration. Those changes are
            internal and do not alter the value returned here.
        """
        return self.weights._raw_lam_vb

    @lam_vb.setter
    def lam_vb(self, value: float) -> None:
        self.weights._raw_lam_vb = value
        self.weights.normalize()

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
        states: list = None,
    ) -> None:
        """Initialize PTR algorithm.

        Stores compiled infrastructure and performs a warm-start solve to
        initialize DPP and JAX jacobians.

        Args:
            solver: Convex subproblem solver (e.g., CVXPySolver)
            discretization_solver: Compiled discretization solver
            jax_constraints: JIT-compiled constraint functions
            emitter: Callback for emitting iteration progress
            params: Problem parameters dictionary (for warm-start)
            settings: Configuration object (for warm-start)
            states: List of symbolic State objects (for per-state weight expansion)
        """
        # Expand dict lam_cost now that states are available
        if states is not None:
            self._states = states
        if isinstance(self._lam_cost_spec, dict):
            if self._states is None:
                raise ValueError(
                    "lam_cost was specified as a dict but no states were "
                    "provided to initialize(). Pass states so the dict can "
                    "be expanded to a per-state weight array."
                )
            self._expand_lam_cost()

        # Store immutable infrastructure
        self._solver = solver
        self._discretization_solver = discretization_solver
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
        _, _, _, x_prop, V_multi_shoot = self._discretization_solver.call(
            init_state.x, init_state.u.astype(float), params
        )

        init_state.add_discretization(V_multi_shoot.__array__())
        _ = self._subproblem(params, init_state, settings)

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
            _, _, _, x_prop, V_multi_shoot = self._discretization_solver.call(
                state.x, state.u.astype(float), params
            )
            dis_time = time.time() - t0

            state.add_discretization(V_multi_shoot.__array__())

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
        _, _, _, x_prop, V_multi_shoot = self._discretization_solver.call(
            candidate.x, candidate.u.astype(float), params
        )
        dis_time = time.time() - t0

        candidate.V = V_multi_shoot.__array__()
        candidate.x_prop = x_prop.__array__()

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
            "lam_prox": state.lam_prox,
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
            lam_vb=state.lam_vb,
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
