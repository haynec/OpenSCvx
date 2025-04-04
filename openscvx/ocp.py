import os
import numpy.linalg as la
import scipy.linalg as sla
import cvxpy as cp
from cvxpygen import cpg
from openscvx.config import Config
from cvxpygen import cpg


def OCP(params):
    ########################
    # VARIABLES & PARAMETERS
    ########################

    # Parameters
    w_tr = cp.Parameter(nonneg = True, name='w_tr')
    lam_cost = cp.Parameter(nonneg=True, name='lam_cost')

    # State
    x = cp.Variable((params.scp.n, params.sim.n_states), name='x') # Current State
    dx = cp.Variable((params.scp.n, params.sim.n_states), name='dx') # State Error
    x_bar = cp.Parameter((params.scp.n, params.sim.n_states), name='x_bar') # Previous SCP State

    # Affine Scaling for State
    S_x = params.sim.S_x
    c_x = params.sim.c_x

    # Control
    u = cp.Variable((params.scp.n, params.sim.n_controls), name='u') # Current Control
    du = cp.Variable((params.scp.n, params.sim.n_controls), name='du') # Control Error
    u_bar = cp.Parameter((params.scp.n, params.sim.n_controls), name='u_bar') # Previous SCP Control

    # Affine Scaling for Control
    S_u = params.sim.S_u
    c_u = params.sim.c_u

    # Linearized Augmented Dynamics Constraints for CTCS
    A_d = cp.Parameter((params.scp.n - 1, (params.sim.n_states)*(params.sim.n_states)), name='A_d')
    B_d = cp.Parameter((params.scp.n - 1, params.sim.n_states*params.sim.n_controls), name='B_d')
    C_d = cp.Parameter((params.scp.n - 1, params.sim.n_states*params.sim.n_controls), name='C_d')
    z_d = cp.Parameter((params.scp.n - 1, params.sim.n_states), name='z_d')
    nu  = cp.Variable((params.scp.n - 1, params.sim.n_states), name='nu') # Virtual Control

    # Applying the affine scaling to state and control
    x_nonscaled = []
    u_nonscaled = []
    for k in range(params.scp.n):
        x_nonscaled.append(S_x @ x[k] + c_x)
        u_nonscaled.append(S_u @ u[k] + c_u)

    constr = []
    cost = lam_cost * 0

    #############
    # CONSTRAINTS
    #############
    if hasattr(params.veh, 'g_cvx_nodal'):
        constr += params.veh.g_cvx_nodal(x_nonscaled) # Nodal Convex Inequality Constraints
    
    if hasattr(params.veh, 'h_cvx_nodal'):
        constr += params.veh.h_cvx_nodal(x_nonscaled) # Nodal Convex Equality Constraints
    
    if hasattr(params.veh, 'g_ncvx_nodal'):
        constr += params.veh.g_ncvx_nodal(x_nonscaled) # Nodal Nonconvex Inequality Constraints
    
    if hasattr(params.veh, 'h_ncvx_nodal'):
        constr += params.veh.h_ncvx_nodal(x_nonscaled) # Nodal Nonconvex Equality Constraints
    

    for i in range(params.sim.n_states-1):
        if params.sim.initial_state['type'][i] == 'Fix':
            constr += [x_nonscaled[0][i] == params.sim.initial_state['value'][i]]  # Initial Boundary Conditions
        if params.sim.final_state['type'][i] == 'Fix':
            constr += [x_nonscaled[-1][i] == params.sim.final_state['value'][i]]   # Final Boundary Conditions
        if params.sim.initial_state['type'][i] == 'Minimize':
            cost += lam_cost * x_nonscaled[0][i]
        if params.sim.final_state['type'][i] == 'Minimize':
            cost += lam_cost * x_nonscaled[-1][i]

    if params.scp.uniform_time_grid:
        constr += [x_nonscaled[i][params.veh.t_inds] - x_nonscaled[i-1][params.veh.t_inds] == x_nonscaled[i-1][params.veh.t_inds] - x_nonscaled[i-2][params.veh.t_inds] for i in range(2, params.scp.n)] # Uniform Time Step

    constr += [0 == la.inv(S_x) @ (x_nonscaled[i] - x_bar[i] - dx[i]) for i in range(params.scp.n)] # State Error
    constr += [0 == la.inv(S_u) @ (u_nonscaled[i] - u_bar[i] - du[i]) for i in range(params.scp.n)] # Control Error

    constr += [x_nonscaled[i] == \
                      cp.reshape(A_d[i-1], (params.sim.n_states, params.sim.n_states)) @ x_nonscaled[i-1] \
                    + cp.reshape(B_d[i-1], (params.sim.n_states, params.sim.n_controls)) @ u_nonscaled[i-1] \
                    + cp.reshape(C_d[i-1], (params.sim.n_states, params.sim.n_controls)) @ u_nonscaled[i] \
                    + z_d[i-1] \
                    + nu[i-1] for i in range(1, params.scp.n)] # Dynamics Constraint
    
    constr += [u_nonscaled[i] <= params.sim.max_control for i in range(params.scp.n)]
    constr += [u_nonscaled[i] >= params.sim.min_control for i in range(params.scp.n)] # Control Constraints

    constr += [x_nonscaled[i][:-1] <= params.sim.max_state[:-1] for i in range(params.scp.n)]
    constr += [x_nonscaled[i][:-1] >= params.sim.min_state[:-1] for i in range(params.scp.n)] # State Constraints (Also implemented in CTCS but included for numerical stability)

    ########
    # COSTS
    ########
    
    cost += sum(w_tr * cp.sum_squares(sla.block_diag(la.inv(S_x), la.inv(S_u)) @ cp.hstack((dx[i], du[i]))) for i in range(params.scp.n)) # Trust Region Cost
    cost += sum(params.scp.lam_vc * cp.sum(cp.abs(nu[i-1])) for i in range(1, params.scp.n)) # Virtual Control Slack
    
    constr += [cp.abs(x_nonscaled[i][-1] - x_nonscaled[i-1][-1]) <= params.sim.max_state[-1] for i in range(1, params.scp.n)] # LICQ Constraint
    constr += [x_nonscaled[0][-1] == 0]
    
    #########
    # PROBLEM
    #########
    prob = cp.Problem(cp.Minimize(cost), constr)
    if params.sim.cvxpygen:
        # Check to see if solver directory exists
        if not os.path.exists('solver'):
            cpg.generate_code(prob, solver = params.sim.solver, code_dir='solver', wrapper = True)
        else:
            # Prompt the use to indicate if they wish to overwrite the solver directory or use the existing compiled solver
            overwrite = input("Solver directory already exists. Overwrite? (y/n): ")
            if overwrite.lower() == 'y':
                cpg.generate_code(prob, solver = params.sim.solver, code_dir='solver', wrapper = True)
            else:
                pass
    return prob