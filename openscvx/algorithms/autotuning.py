"""Autotuning functions for SCP (Successive Convex Programming) parameters."""

from typing import TYPE_CHECKING

import numpy as np

from copy import deepcopy

from openscvx.config import Config

if TYPE_CHECKING:
    from .base import AlgorithmState


def update_scp_weights(state: "AlgorithmState", settings: Config, params: dict):
    """Update SCP weights and cost parameters based on iteration number.

    Args:
        state: Solver state containing current weight values (mutated in place)
        settings: Configuration object containing adaptation parameters
        scp_k: Current SCP iteration number
    """
    # Update trust region weight in state
    # state.w_tr = min(state.w_tr * settings.scp.w_tr_adapt, settings.scp.w_tr_max)

    nonlinear_cost, nonlinear_penalty = calculate_nonlinear_penalty(state.candidate.x_prop,
                                                   state.candidate.x,
                                                   state.lam_vc,
                                                   state.lam_cost,
                                                   params,
                                                   settings)

    state.candidate.J_nonlin = nonlinear_cost + nonlinear_penalty

    # Update cost relaxation parameter after cost_drop iterations
    if state.k > settings.scp.cost_drop:
        state.candidate.lam_cost = state.lam_cost * settings.scp.cost_relax
    else:
        state.candidate.lam_cost = settings.scp.lam_cost

    eta_0 = 1E-2
    eta_1 = 1E-1
    eta_2 = 0.8

    gamma_1 = 2.0
    gamma_2 = 0.5

    w_tr_min = 1E-3
    w_tr_max = 2E5

    w_tr_k = deepcopy(state.w_tr)

    if state.k > 1:
        prev_nonlinear_cost, prev_nonlinear_penalty = calculate_nonlinear_penalty(state.x_prop(),
                                                    state.x,
                                                    state.lam_vc,
                                                    state.lam_cost,
                                                    params,
                                                    settings)

        J_nonlin_prev = prev_nonlinear_cost + prev_nonlinear_penalty

        J_lin_current = calculate_linear_penalty(state.candidate.x,
                                                 state.candidate.TR,
                                                 state.candidate.VC,
                                                 state.lam_cost,
                                                 state.lam_vc,
                                                 state.w_tr,
                                                 settings)
 
        actual_reduction = J_nonlin_prev - state.candidate.J_nonlin
        predicted_reduction = J_nonlin_prev - state.candidate.J_lin
        rho = actual_reduction / predicted_reduction

        state.pred_reduction_history.append(predicted_reduction)
        state.actual_reduction_history.append(actual_reduction)
        state.acceptance_ratio_history.append(rho)


        if rho < eta_0:
            # Reject Solution and higher weight
            w_tr_k1 = min(w_tr_max, gamma_1 * w_tr_k)
            state.w_tr_history.append(w_tr_k1)
            adaptive_state = "Reject Higher"
        elif rho >= eta_0 and rho < eta_1:
            # Accept Solution with heigher weight
            w_tr_k1 = min(w_tr_max, gamma_1 * w_tr_k)
            state.w_tr_history.append(w_tr_k1)
            state.accept_solution()
            adaptive_state = "Accept Higher"
        elif rho >= eta_1 and rho < eta_2:
            # Accept Solution with constant weight
            w_tr_k1 = w_tr_k
            state.w_tr_history.append(w_tr_k1)
            state.accept_solution()
            adaptive_state = "Accept Constant"
        else:
            # Accept Solution with lower weight
            w_tr_k1 = max(w_tr_min, gamma_2 * w_tr_k)
            state.w_tr_history.append(w_tr_k1)
            state.accept_solution()
            adaptive_state = "Accept Lower"

        # Update virtual control weight matrix
        ep = 0.5
        nu = abs(state.candidate.x[1:] - state.candidate.x_prop)
        vc_max = 1E2
        eta_lambda = 1E0
        
        # Vectorized update: use mask to select between two update rules
        mask = nu > ep
        case1 = state.lam_vc + nu * eta_lambda * (1 / (2 * state.w_tr))  # when abs(nu) > ep
        case2 = state.lam_vc + (nu**2) / ep * eta_lambda * (1 / (2 * state.w_tr))  # when abs(nu) <= ep
        vc_new = np.where(mask, case1, case2)
        vc_new = np.minimum(vc_max, vc_new)
        # state.lam_vc_history.append(vc_new)

    else:
        state.w_tr_history.append(w_tr_k)
        state.candidate.lam_vc = settings.scp.lam_vc
        state.accept_solution()
        adaptive_state = "Initial"
    
    return adaptive_state
    


def calculate_cost_from_state(x, settings: Config):
    """Calculate cost from state vector based on final_type.

    Args:
        state: Solver state containing current state vector

    Returns:
        float: Computed cost
    """
    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:,None])).T
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

def calculate_nonlinear_penalty(x_prop: np.ndarray, 
                                x_bar: np.ndarray,
                                lam_vc: np.ndarray, 
                                lam_cost: float,
                                params: dict, 
                                settings: Config):
    """Calculate nonlinear penalty

    Args:
        x_prop: Propagated state (n_nodes-1, n_states)
        x: Previous iteration state (n_nodes, n_states)
        u: Solution control (n_nodes, n_controls)
        lam_vc: Virtual control weight (scalar or matrix)
        lam_cost: Cost relaxation parameter (scalar)
        param_dict: Dictionary of problem parameters
        settings: Configuration object

    Returns:
        float: Nonlinear penalty value
    """
    # TODO: Implement nonconvex nodal constraint violations

    cost = calculate_cost_from_state(x_bar, settings)
    x_diff = settings.sim.inv_S_x @ (x_bar[1:, :] - x_prop).T

    return lam_cost * cost, np.sum(lam_vc * np.abs(x_diff.T))


def calculate_linear_penalty(x_bar: np.ndarray,
                             TR_matrix: np.ndarray, 
                             VC_matrix: np.ndarray,
                             lam_cost: float,
                             lam_vc: np.ndarray,
                             w_tr: float,
                             settings: Config):
    """Calculate linear penalty

    Args:
        TR_matrix: Trust region matrix
        VC_matrix: Virtual control matrix
        cost: Cost
    """
    cost = calculate_cost_from_state(x_bar, settings)
    TR_matrix = np.linalg.norm(TR_matrix, axis=0) ** 2
    return lam_cost * cost + np.sum(lam_vc * (settings.sim.inv_S_x @ VC_matrix.T).T) + w_tr * np.linalg.norm(TR_matrix) ** 2