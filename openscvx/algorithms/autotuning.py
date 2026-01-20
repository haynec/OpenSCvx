"""Autotuning functions for SCP (Successive Convex Programming) parameters."""

from typing import TYPE_CHECKING

import numpy as np

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

    update_penalties(state, settings, params)

    # Update cost relaxation parameter after cost_drop iterations
    # if state.k > settings.scp.cost_drop:
    #     state.lam_cost = state.lam_cost * settings.scp.cost_relax

    eta_1 = 0.1
    eta_2 = 0.9

    gamma_1 = 2.0
    gamma_2 = 0.5

    w_tr_min = 1E-3

    w_tr_k = state.w_tr

    if state.k > 1:
        actual_reduction = state.J_nonlin_history[-2] - state.J_nonlin_history[-1]
        predicted_reduction = state.J_nonlin_history[-2] - state.J_lin_history[-1]
        rho = actual_reduction / predicted_reduction
        state.acceptance_ratio_history.append(rho)

        if state.acceptance_ratio_history[-1] >= eta_1:
            if state.acceptance_ratio_history[-1] < eta_2:
                state.w_tr = w_tr_k
            else:
                state.w_tr = max(w_tr_min, gamma_2 * w_tr_k)
            
            state.lam_cost_history.append(settings.scp.lam_cost)
        else:
            state.w_tr = gamma_1 * w_tr_k
            state.reject_last_solution()

    else:
        state.w_tr = w_tr_k
        state.lam_vc_history.append(settings.scp.lam_vc)
        state.lam_cost_history.append(settings.scp.lam_cost)
    
    # Update trust region weight history
    state.w_tr_history.append(state.w_tr)
    


def calculate_cost_from_state(x, settings: Config):
    """Calculate cost from state vector based on final_type.

    Args:
        state: Solver state containing current state vector

    Returns:
        float: Computed cost
    """
    cost = 0.0
    for i in range(settings.sim.n_states):
        if settings.sim.x.final_type[i] == "Minimize":
            cost += x[-1, i]
        if settings.sim.x.final_type[i] == "Maximize":
            cost -= x[-1, i]
        if settings.sim.x.initial_type[i] == "Minimize":
            cost += x[0, i]
        if settings.sim.x.initial_type[i] == "Maximize":
            cost -= x[0, i]
    return cost

def calculate_nonlinear_penalty(x_prop: np.ndarray, 
                                x_bar: np.ndarray,
                                lam_vc: np.ndarray, 
                                lam_cost: float,
                                params: dict, 
                                settings: Config):
    """Calculate nonlinear penalty J_nonlin = x_prop[cost] + lam_vc(x_prop-x_sol) + lam_vb(g(x_prop)).

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
    x_diff = x_bar[1:, :] - x_prop

    return lam_cost * cost + np.sum(lam_vc * np.abs(x_diff))

def update_penalties(state: "AlgorithmState", 
                               settings: Config, 
                               params: dict):
    """Calculate acceptance ratio for trust region method.
    
    Args:
        state: Solver state containing current weight values
        settings: Configuration object containing adaptation parameters
        params: Dictionary of problem parameters
    """

    x = state.x
    x_prop = state.x_prop_history[-1]
    lam_vc = state.lam_vc
    lam_cost = state.lam_cost
    
    J_nonlin_current = calculate_nonlinear_penalty(x_prop,
                                                   x,
                                                   lam_vc,
                                                   lam_cost,
                                                   params,
                                                   settings)

    state.J_nonlin_history.append(J_nonlin_current)