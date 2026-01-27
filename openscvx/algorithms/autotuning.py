"""Autotuning functions for SCP (Successive Convex Programming) parameters."""

from typing import TYPE_CHECKING

import numpy as np

from copy import deepcopy

from openscvx.config import Config

if TYPE_CHECKING:
    from .base import AlgorithmState
    from openscvx.lowered import LoweredJaxConstraints


def update_scp_weights(state: "AlgorithmState", nodal_constraints: "LoweredJaxConstraints", settings: Config, params: dict):
    """Update SCP weights and cost parameters based on iteration number.

    Args:
        state: Solver state containing current weight values (mutated in place)
        settings: Configuration object containing adaptation parameters
        scp_k: Current SCP iteration number
    """
    # Update trust region weight in state
    # state.w_tr = min(state.w_tr * settings.scp.w_tr_adapt, settings.scp.w_tr_max)

    nonlinear_cost, nonlinear_penalty, nodal_penalty = calculate_nonlinear_penalty(state.candidate.x_prop,
                                                   state.candidate.x,
                                                   state.candidate.u,
                                                   state.lam_vc,
                                                   state.lam_vb,
                                                   state.lam_cost,
                                                   nodal_constraints,
                                                   params,
                                                   settings)

    state.candidate.J_nonlin = nonlinear_cost + nonlinear_penalty + nodal_penalty

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
        prev_nonlinear_cost, prev_nonlinear_penalty, prev_nodal_penalty = calculate_nonlinear_penalty(state.x_prop(),
                                                    state.x,
                                                    state.u,
                                                    state.lam_vc,
                                                    state.lam_vb,
                                                    state.lam_cost,
                                                    nodal_constraints,
                                                    params,
                                                    settings)

        J_nonlin_prev = prev_nonlinear_cost + prev_nonlinear_penalty + prev_nodal_penalty
 
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
        nu = (settings.sim.inv_S_x @ abs(state.candidate.x[1:] - state.candidate.x_prop).T).T
        vc_max = 1E5
        eta_lambda = 1E0
        
        # Vectorized update: use mask to select between two update rules
        mask = nu > ep
        case1 = state.lam_vc + nu * eta_lambda * (1 / (2 * state.w_tr))  # when abs(nu) > ep
        case2 = state.lam_vc + (nu**2) / ep * eta_lambda * (1 / (2 * state.w_tr))  # when abs(nu) <= ep
        vc_new = np.where(mask, case1, case2)
        vc_new = np.minimum(vc_max, vc_new)
        state.candidate.lam_vc = vc_new
        state.candidate.lam_vb = settings.scp.lam_vb

    else:
        state.w_tr_history.append(w_tr_k)
        state.candidate.lam_vc = settings.scp.lam_vc
        state.candidate.lam_vb = settings.scp.lam_vb
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
                                u_bar: np.ndarray,
                                lam_vc: np.ndarray, 
                                lam_vb: float,
                                lam_cost: float,
                                nodal_constraints: "LoweredJaxConstraints",
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

    cost = calculate_cost_from_state(x_bar, settings)
    x_diff = settings.sim.inv_S_x @ (x_bar[1:, :] - x_prop).T

    return lam_cost * cost, np.sum(lam_vc * np.abs(x_diff.T)), nodal_penalty