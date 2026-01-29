"""Autotuning functions for SCP (Successive Convex Programming) parameters."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from openscvx.config import Config
from openscvx.utils.printing import (
    Column,
    Verbosity,
    color_acceptance_ratio,
    color_adaptive_state,
    color_J_nonlin,
)

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

    from .base import AlgorithmState, CandidateIterate


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
    def calculate_cost_from_state(x: np.ndarray, settings: Config) -> float:
        """Calculate cost from state vector based on final_type and initial_type.

        Args:
            x: State trajectory array (n_nodes, n_states)
            settings: Configuration object containing state types

        Returns:
            float: Computed cost
        """
        scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
        cost = 0.0
        for i in range(settings.sim.n_states):
            if settings.sim.x.final_type[i] == "Minimize":
                cost += scaled_x[-1, i]
            if settings.sim.x.final_type[i] == "Maximize":
                cost -= scaled_x[-1, i]
            if settings.sim.x.initial_type[i] == "Minimize":
                cost += scaled_x[0, i]
            if settings.sim.x.initial_type[i] == "Maximize":
                cost -= scaled_x[0, i]
        return cost

    @staticmethod
    def calculate_nonlinear_penalty(
        x_prop: np.ndarray,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        lam_vc: np.ndarray,
        lam_vb: float,
        lam_cost: float,
        nodal_constraints: "LoweredJaxConstraints",
        params: dict,
        settings: Config,
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
            lam_vb: Virtual buffer penalty weight (scalar)
            lam_cost: Cost relaxation parameter (scalar)
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
        for constraint in nodal_constraints.nodal:
            # Nodal constraint function is vmapped: func(x, u, node, params)
            # When called with arrays, it evaluates at all nodes
            g = constraint.func(x_bar, u_bar, 0, params)
            # Only sum violations at nodes where constraint is enforced
            if constraint.nodes is not None:
                # Filter to only specified nodes
                # Convert to numpy array for JAX compatibility
                nodes_array = np.array(constraint.nodes)
                g_filtered = g[nodes_array]
            else:
                # If no nodes specified, check all nodes
                g_filtered = g
            nodal_penalty += lam_vb * np.sum(np.maximum(0, g_filtered))

        # Evaluate cross-node constraints
        for constraint in nodal_constraints.cross_node:
            # Cross-node constraint function signature: func(X, U, params)
            # No node argument - operates on full trajectory
            g = constraint.func(x_bar, u_bar, params)
            # Cross-node constraints return scalar or array, sum all violations
            nodal_penalty += lam_vb * np.sum(np.maximum(0, g))

        cost = AutotuningBase.calculate_cost_from_state(x_bar, settings)
        x_diff = settings.sim.inv_S_x @ (x_bar[1:, :] - x_prop).T

        return lam_cost * cost, np.sum(lam_vc * np.abs(x_diff.T)), nodal_penalty

    @abstractmethod
    def update_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        settings: Config,
        params: dict,
    ) -> str:
        """Update SCP weights and cost parameters based on iteration state.

        This method is called each iteration to adapt weights based on the
        current solution quality and constraint satisfaction.

        Args:
            state: Solver state containing current weight values (mutated in place)
            nodal_constraints: Lowered JAX constraints
            settings: Configuration object containing adaptation parameters
            params: Dictionary of problem parameters

        Returns:
            str: Adaptive state string describing the update action (e.g., "Accept Lower")
        """

        pass


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
        eta_lambda: float = 1e0,
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

    def update_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        settings: Config,
        params: dict,
    ) -> str:
        """Update SCP weights and cost parameters based on iteration number.

        Args:
            state: Solver state containing current weight values (mutated in place)
            nodal_constraints: Lowered JAX constraints
            settings: Configuration object containing adaptation parameters
            params: Dictionary of problem parameters
        """
        # Calculate nonlinear penalty for current candidate
        nonlinear_cost, nonlinear_penalty, nodal_penalty = self.calculate_nonlinear_penalty(
            candidate.x_prop,
            candidate.x,
            candidate.u,
            state.lam_vc,
            state.lam_vb,
            state.lam_cost,
            nodal_constraints,
            params,
            settings,
        )

        candidate.J_nonlin = nonlinear_cost + nonlinear_penalty + nodal_penalty

        # Update cost relaxation parameter after cost_drop iterations
        if state.k > self.lam_cost_drop:
            candidate.lam_cost = state.lam_cost * self.lam_cost_relax
        else:
            candidate.lam_cost = settings.scp.lam_cost

        lam_prox_k = deepcopy(state.lam_prox)

        if state.k > 1:
            prev_nonlinear_cost, prev_nonlinear_penalty, prev_nodal_penalty = (
                self.calculate_nonlinear_penalty(
                    state.x_prop(),
                    state.x,
                    state.u,
                    state.lam_vc,
                    state.lam_vb,
                    state.lam_cost,
                    nodal_constraints,
                    params,
                    settings,
                )
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
                lam_prox_k1 = min(self.lam_prox_max, self.gamma_1 * lam_prox_k)
                state.lam_prox_history.append(lam_prox_k1)
                adaptive_state = "Reject Higher"
            elif rho >= self.eta_0 and rho < self.eta_1:
                # Accept Solution with heigher weight
                lam_prox_k1 = min(self.lam_prox_max, self.gamma_1 * lam_prox_k)
                state.lam_prox_history.append(lam_prox_k1)
                state.accept_solution(candidate)
                adaptive_state = "Accept Higher"
            elif rho >= self.eta_1 and rho < self.eta_2:
                # Accept Solution with constant weight
                lam_prox_k1 = lam_prox_k
                state.lam_prox_history.append(lam_prox_k1)
                state.accept_solution(candidate)
                adaptive_state = "Accept Constant"
            else:
                # Accept Solution with lower weight
                lam_prox_k1 = max(self.lam_prox_min, self.gamma_2 * lam_prox_k)
                state.lam_prox_history.append(lam_prox_k1)
                state.accept_solution(candidate)
                adaptive_state = "Accept Lower"

            # Update virtual control weight matrix
            nu = (settings.sim.inv_S_x @ abs(candidate.x[1:] - candidate.x_prop).T).T

            # Vectorized update: use mask to select between two update rules
            mask = nu > self.ep
            # when abs(nu) > ep
            scale = self.eta_lambda * (1 / (2 * state.lam_prox))
            case1 = state.lam_vc + nu * scale
            # when abs(nu) <= ep
            case2 = state.lam_vc + (nu**2) / self.ep * scale
            vc_new = np.where(mask, case1, case2)
            vc_new = np.minimum(self.lam_vc_max, vc_new)
            candidate.lam_vc = vc_new
            candidate.lam_vb = settings.scp.lam_vb

        else:
            state.lam_prox_history.append(lam_prox_k)
            candidate.lam_vc = settings.scp.lam_vc
            candidate.lam_vb = settings.scp.lam_vb
            state.accept_solution(candidate)
            adaptive_state = "Initial"

        return adaptive_state


class ConstantProximalWeight(AutotuningBase):
    """Constant Proximal Weight method.

    This method keeps the trust region weight constant throughout the optimization,
    while still updating virtual control weights and handling cost relaxation.
    Useful when you want a fixed trust region size without adaptation.
    """

    def __init__(
        self,
        lam_cost_drop: int = -1,
        lam_cost_relax: float = 1.0,
    ):
        self.lam_cost_drop = lam_cost_drop
        self.lam_cost_relax = lam_cost_relax

    def update_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        settings: Config,
        params: dict,
    ) -> str:
        """Update SCP weights keeping trust region constant.

        Args:
            state: Solver state containing current weight values (mutated in place)
            nodal_constraints: Lowered JAX constraints
            settings: Configuration object containing adaptation parameters
            params: Dictionary of problem parameters

        Returns:
            str: Adaptive state string (e.g., "Accept", "Reject")
        """
        # Update cost relaxation parameter after cost_drop iterations
        if state.k > self.lam_cost_drop:
            candidate.lam_cost = state.lam_cost * self.lam_cost_relax
        else:
            candidate.lam_cost = settings.scp.lam_cost

        state.lam_prox_history.append(state.lam_prox)
        state.accept_solution(candidate)
        return "Accept Constant"


class RampProximalWeight(AutotuningBase):
    """Ramp Proximal Weight method.

    This method ramps the proximal weight up linearly over the first few iterations,
    then keeps it constant.
    """

    def __init__(
        self,
        ramp_factor: float = 1.0,
        lam_prox_max: float = 1e3,
        lam_cost_drop: int = -1,
        lam_cost_relax: float = 1.0,
    ):
        self.ramp_factor = ramp_factor
        self.lam_prox_max = lam_prox_max
        self.lam_cost_drop = lam_cost_drop
        self.lam_cost_relax = lam_cost_relax

    def update_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        settings: Config,
        params: dict,
    ) -> str:
        """Update SCP weights keeping trust region constant.

        Args:
            state: Solver state containing current weight values (mutated in place)
            nodal_constraints: Lowered JAX constraints
            settings: Configuration object containing adaptation parameters
            params: Dictionary of problem parameters

        Returns:
            str: Adaptive state string (e.g., "Accept", "Reject")
        """
        # Update cost relaxation parameter after cost_drop iterations
        if state.k > self.lam_cost_drop:
            candidate.lam_cost = state.lam_cost * self.lam_cost_relax
        else:
            candidate.lam_cost = settings.scp.lam_cost

        # Check if we're already at max before updating
        was_at_max = state.lam_prox >= self.lam_prox_max

        # Calculate and append new value
        new_lam_prox = min(state.lam_prox * self.ramp_factor, self.lam_prox_max)
        state.lam_prox_history.append(new_lam_prox)

        # If we were already at max, or if we just reached it and it's staying constant
        if was_at_max:
            state.accept_solution(candidate)
            return "Accept Constant"
        else:
            state.accept_solution(candidate)
            return "Accept Higher"
