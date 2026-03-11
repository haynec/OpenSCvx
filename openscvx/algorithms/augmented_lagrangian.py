"""Autotuning functions for SCP (Successive Convex Programming) parameters."""

from copy import deepcopy
from typing import TYPE_CHECKING, List

import numpy as np

from openscvx.config import Config
from openscvx.utils.printing import (
    Column,
    Verbosity,
    color_acceptance_ratio,
    color_adaptive_state,
    color_J_nonlin,
)

from .base import AutotuningBase

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

    from .base import AlgorithmState, CandidateIterate, Weights


class AugmentedLagrangian(AutotuningBase):
    """Augmented Lagrangian method for autotuning SCP weights.

    This method uses Lagrange multipliers and penalty parameters to handle
    constraints. The method:
    - Updates Lagrange multipliers based on constraint violations
    - Increases penalty parameters when constraints are violated
    - Decreases penalty parameters when constraints are satisfied
    """

    COLUMNS: List[Column] = [
        Column("J_nonlin", "J_nonlin", 8, "{: .1e}", color_J_nonlin, Verbosity.STANDARD),
        Column("J_lin", "J_lin", 8, "{: .1e}", color_J_nonlin, Verbosity.STANDARD),
        Column("pred_reduction", "pred_red", 9, "{: .1e}", min_verbosity=Verbosity.FULL),
        Column("actual_reduction", "act_red", 9, "{: .1e}", min_verbosity=Verbosity.FULL),
        Column(
            "acceptance_ratio",
            "acc_ratio",
            9,
            "{: .2e}",
            color_acceptance_ratio,
            Verbosity.STANDARD,
        ),
        Column("lam_prox", "lam_prox", 8, "{: .1e}", min_verbosity=Verbosity.FULL),
        Column("adaptive_state", "Adaptive", 16, "{}", color_adaptive_state, Verbosity.FULL),
    ]

    def __init__(
        self,
        rho_init: float = 1.0,
        rho_max: float = 1e6,
        gamma_1: float = 2.0,
        gamma_2: float = 0.5,
        eta_0: float = 1e-2,
        eta_1: float = 1e-1,
        eta_2: float = 0.8,
        ep: float = 0.5,
        eta_lambda: float = 1e1,
        lam_vc_max: float = 1e5,
        lam_prox_min: float = 1e-3,
        lam_prox_max: float = 2e5,
        lam_cost_drop: int = -1,
        lam_cost_relax: float = 1.0,
    ):
        """Initialize Augmented Lagrangian autotuning parameters.

        All parameters have defaults and can be modified after instantiation
        via attribute access (e.g., ``autotuner.rho_max = 1e7``).

        Args:
            rho_init: Initial penalty parameter for constraints. Defaults to 1.0.
            rho_max: Maximum penalty parameter. Defaults to 1e6.
            gamma_1: Factor to increase trust region weight when ratio is low.
                Defaults to 2.0.
            gamma_2: Factor to decrease trust region weight when ratio is high.
                Defaults to 0.5.
            eta_0: Acceptance ratio threshold below which solution is rejected.
                Defaults to 1e-2.
            eta_1: Threshold above which solution is accepted with constant weight.
                Defaults to 1e-1.
            eta_2: Threshold above which solution is accepted with lower weight.
                Defaults to 0.8.
            ep: Threshold for virtual control weight update (nu > ep vs nu <= ep).
                Defaults to 0.5.
            eta_lambda: Step size for virtual control weight update. Defaults to 1e0.
            lam_vc_max: Maximum virtual control penalty weight. Defaults to 1e5.
            lam_prox_min: Minimum trust region (proximal) weight. Defaults to 1e-3.
            lam_prox_max: Maximum trust region (proximal) weight. Defaults to 2e5.
            lam_cost_drop: Iteration after which cost relaxation applies (-1 = never).
                Defaults to -1.
            lam_cost_relax: Factor applied to lam_cost after lam_cost_drop.
                Defaults to 1.0.
        """
        self.rho_init = rho_init
        self.rho_max = rho_max
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.eta_0 = eta_0
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.ep = ep
        self.eta_lambda = eta_lambda
        self.lam_vc_max = lam_vc_max
        self.lam_prox_min = lam_prox_min
        self.lam_prox_max = lam_prox_max
        self.lam_cost_drop = lam_cost_drop
        self.lam_cost_relax = lam_cost_relax

    def _update_virtual_control_weights(
        self,
        candidate: "CandidateIterate",
        candidate_x_prop: np.ndarray,
        settings: Config,
        lam_vc: np.ndarray,
        lam_prox: np.ndarray,
    ) -> np.ndarray:
        """Update virtual control penalty weights from state violation.

        Computes scaled violation nu = inv_S_x @ |x[1:] - x_prop|, then applies
        two update rules: when nu > ep (linear in nu), else quadratic in nu.
        Result is clipped to lam_vc_max.
        """
        nu = (settings.sim.inv_S_x @ abs(candidate.x[1:] - candidate_x_prop).T).T
        mask = nu > self.ep
        # TODO: (haynec) use per-variable lam_prox to scale VC updates proportionally
        lam_prox_scalar = float(np.max(lam_prox))
        scale = self.eta_lambda * (1 / (2 * lam_prox_scalar))
        case1 = lam_vc + nu * scale
        case2 = lam_vc + (nu**2) / self.ep * scale
        vc_new = np.where(mask, case1, case2)
        return np.minimum(self.lam_vc_max, vc_new)

    def _update_virtual_buffer_nodal_weights(
        self,
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        params: dict,
        lam_vb_nodal: np.ndarray,
        lam_prox: np.ndarray,
    ) -> np.ndarray:
        """Update virtual buffer penalty weights for nodal constraints.

        Evaluates each nodal constraint to obtain violation
        nu = max(0, g(x, u)), then applies the same two-case update rule
        used for virtual control weights: linear when nu > ep, quadratic
        otherwise. Result is clipped to lam_vc_max.
        """
        lam_vb_new = lam_vb_nodal.copy()
        lam_prox_scalar = float(np.max(lam_prox))
        scale = self.eta_lambda * (1 / (2 * lam_prox_scalar))

        for idx, constraint in enumerate(nodal_constraints.nodal):
            g = constraint.func(candidate.x, candidate.u, 0, params)
            nu = np.maximum(0, g)

            if constraint.nodes is not None:
                nodes_array = np.array(constraint.nodes)
                nu_slice = nu[nodes_array]
                current = lam_vb_nodal[nodes_array, idx]
            else:
                nu_slice = nu
                current = lam_vb_nodal[:, idx]

            mask = nu_slice > self.ep
            case1 = current + nu_slice * scale
            case2 = current + (nu_slice**2) / self.ep * scale
            updated = np.where(mask, case1, case2)

            if constraint.nodes is not None:
                lam_vb_new[nodes_array, idx] = updated
            else:
                lam_vb_new[:, idx] = updated

        return np.minimum(self.lam_vc_max, lam_vb_new)

    def _update_virtual_buffer_cross_weights(
        self,
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        params: dict,
        lam_vb_cross: np.ndarray,
        lam_prox: np.ndarray,
    ) -> np.ndarray:
        """Update virtual buffer penalty weights for cross-node constraints.

        Evaluates each cross-node constraint to obtain total violation
        nu = sum(max(0, g(X, U))), then applies the same two-case update
        rule used for virtual control weights. Result is clipped to lam_vc_max.
        """
        lam_vb_new = lam_vb_cross.copy()
        lam_prox_scalar = float(np.max(lam_prox))
        scale = self.eta_lambda * (1 / (2 * lam_prox_scalar))

        for idx, constraint in enumerate(nodal_constraints.cross_node):
            g = constraint.func(candidate.x, candidate.u, params)
            nu = np.sum(np.maximum(0, g))

            current = lam_vb_cross[idx]
            if nu > self.ep:
                lam_vb_new[idx] = current + nu * scale
            else:
                lam_vb_new[idx] = current + (nu**2) / self.ep * scale

        return np.minimum(self.lam_vc_max, lam_vb_new)

    def update_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        settings: Config,
        params: dict,
        weights: "Weights",
    ) -> str:
        """Update SCP weights and cost parameters based on iteration number.

        Args:
            state: Solver state containing current weight values (mutated in place)
            candidate: Candidate iterate from the current subproblem solve
            nodal_constraints: Lowered JAX constraints
            settings: Configuration object containing adaptation parameters
            params: Dictionary of problem parameters
            weights: Normalized initial weights from the algorithm
        """
        # Calculate nonlinear penalty for current candidate
        candidate_x_prop = (
            candidate.x_prop_plus[1:] if candidate.x_prop_plus is not None else candidate.x_prop
        )
        (
            nonlinear_cost,
            nonlinear_penalty,
            nodal_penalty,
        ) = self.calculate_nonlinear_penalty(
            candidate_x_prop,
            candidate.x,
            candidate.u,
            state.lam_vc,
            state.lam_vb_nodal,
            state.lam_vb_cross,
            state.lam_cost,
            nodal_constraints,
            params,
            settings,
        )

        candidate.J_nonlin = nonlinear_cost + nonlinear_penalty + nodal_penalty

        # Update cost relaxation parameter after cost_drop iterations.
        # When lam_cost is a per-state array, scalar lam_cost_relax scales
        # uniformly, preserving the user-specified per-state weight ratios.
        if state.k > self.lam_cost_drop:
            candidate.lam_cost = state.lam_cost * self.lam_cost_relax
        else:
            candidate.lam_cost = weights.lam_cost

        lam_prox_k = deepcopy(state.lam_prox)

        if state.k > 1:
            state_x_prop_plus = state.x_prop_plus()
            state_x_prop = (
                state_x_prop_plus[1:] if state_x_prop_plus is not None else state.x_prop()
            )
            (
                prev_nonlinear_cost,
                prev_nonlinear_penalty,
                prev_nodal_penalty,
            ) = self.calculate_nonlinear_penalty(
                state_x_prop,
                state.x,
                state.u,
                state.lam_vc,
                state.lam_vb_nodal,
                state.lam_vb_cross,
                state.lam_cost,
                nodal_constraints,
                params,
                settings,
            )

            J_nonlin_prev = prev_nonlinear_cost + prev_nonlinear_penalty + prev_nodal_penalty

            actual_reduction = J_nonlin_prev - candidate.J_nonlin
            predicted_reduction = J_nonlin_prev - candidate.J_lin

            if predicted_reduction == 0:
                raise ValueError("Predicted reduction is 0.")

            rho = actual_reduction / predicted_reduction

            state.pred_reduction_history.append(predicted_reduction)
            state.actual_reduction_history.append(actual_reduction)
            state.acceptance_ratio_history.append(rho)

            if rho < self.eta_0:
                # Reject Solution and higher weight
                lam_prox_k1 = np.minimum(self.lam_prox_max, self.gamma_1 * lam_prox_k)
                candidate.lam_prox = lam_prox_k1
                state.reject_solution(candidate)
                adaptive_state = "Reject Higher"
            elif rho >= self.eta_0 and rho < self.eta_1:
                # Accept Solution with heigher weight
                lam_prox_k1 = np.minimum(self.lam_prox_max, self.gamma_1 * lam_prox_k)
                candidate.lam_prox = lam_prox_k1
                # Update virtual control weight matrix
                candidate.lam_vc = self._update_virtual_control_weights(
                    candidate, candidate_x_prop, settings, state.lam_vc, candidate.lam_prox
                )
                # candidate.lam_vb_nodal = self._update_virtual_buffer_nodal_weights(
                #     candidate, nodal_constraints, params, state.lam_vb_nodal, candidate.lam_prox
                # )
                # candidate.lam_vb_cross = self._update_virtual_buffer_cross_weights(
                #     candidate, nodal_constraints, params, state.lam_vb_cross, candidate.lam_prox
                # )

                state.accept_solution(candidate)
                adaptive_state = "Accept Higher"
            elif rho >= self.eta_1 and rho < self.eta_2:
                # Accept Solution with constant weight
                lam_prox_k1 = lam_prox_k
                candidate.lam_prox = lam_prox_k1
                # Update virtual control weight matrix
                candidate.lam_vc = self._update_virtual_control_weights(
                    candidate, candidate_x_prop, settings, state.lam_vc, candidate.lam_prox
                )
                # candidate.lam_vb_nodal = self._update_virtual_buffer_nodal_weights(
                #     candidate, nodal_constraints, params, state.lam_vb_nodal, candidate.lam_prox
                # )
                # candidate.lam_vb_cross = self._update_virtual_buffer_cross_weights(
                #     candidate, nodal_constraints, params, state.lam_vb_cross, candidate.lam_prox
                # )

                state.accept_solution(candidate)
                adaptive_state = "Accept Constant"
            else:
                # Accept Solution with lower weight
                lam_prox_k1 = np.maximum(self.lam_prox_min, self.gamma_2 * lam_prox_k)
                candidate.lam_prox = lam_prox_k1
                # Update virtual control weight matrix
                candidate.lam_vc = self._update_virtual_control_weights(
                    candidate, candidate_x_prop, settings, state.lam_vc, candidate.lam_prox
                )
                # candidate.lam_vb_nodal = self._update_virtual_buffer_nodal_weights(
                #     candidate, nodal_constraints, params, state.lam_vb_nodal, candidate.lam_prox
                # )
                # candidate.lam_vb_cross = self._update_virtual_buffer_cross_weights(
                #     candidate, nodal_constraints, params, state.lam_vb_cross, candidate.lam_prox
                # )
                state.accept_solution(candidate)
                adaptive_state = "Accept Lower"

        else:
            candidate.lam_prox = lam_prox_k
            candidate.lam_vc = state.lam_vc
            candidate.lam_vb_nodal = state.lam_vb_nodal
            candidate.lam_vb_cross = state.lam_vb_cross
            state.accept_solution(candidate)
            adaptive_state = "Initial"

        return adaptive_state
