import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
import cvxpy as cp
import pickle
import time

import sys
from termcolor import colored

from openscvx.discretization import ExactDis
from openscvx.config import Config
from openscvx.propagation import full_subject_traj_time, subject_traj, u_lambda, simulate_nonlinear_time
from openscvx.ocp import OCP
from openscvx.plotting import plot_initial_guess

import warnings
warnings.filterwarnings("ignore")

def PTR_init(params: Config) -> tuple[cp.Problem, ExactDis]:
    intro()

    t_0_while = time.time()

    ocp = OCP(params) # Initialize the problem

    if params.sim.cvxpygen:
        from solver.cpg_solver import cpg_solve
        with open('solver/problem.pickle', 'rb') as f:
            prob = pickle.load(f)
    else:
        cpg_solve = None

    aug_dy = ExactDis(params)

    # Solve a dumb problem to intilize DPP and JAX jacobians
    _ = PTR_subproblem(cpg_solve, params.sim.x_bar, params.sim.u_bar, aug_dy, ocp, params)

    t_f_while = time.time()
    print("Total Initialization Time: ", t_f_while - t_0_while)
    return ocp, aug_dy, cpg_solve

def PTR_main(params: Config, prob: cp.Problem, aug_dy: ExactDis, cpg_solve) -> dict:
    J_vb = 1E2
    J_vc = 1E2
    J_tr = 1E2

    x_bar = params.sim.x_bar
    u_bar = params.sim.u_bar

    scp_trajs = [x_bar]
    scp_controls = [u_bar]
    V_multi_shoot_traj = []

    # Define colors for printing
    col_main = "blue"
    col_pos = "green"
    col_neg = "red"

    print("{:^4} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} |  {:^7} | {:^14}".format(
            "Iter", "Dis Time (ms)", "Solve Time (ms)", "J_total", "J_tr", "J_vb", "J_vc", "Cost", "Solver Status"))
    print(colored("---------------------------------------------------------------------------------------------------------"))

    k = 1

    if params.sim.profiling:
        import cProfile
        pr = cProfile.Profile()
        
        # Enable the profiler
        pr.enable()

    t_0_while = time.time()
    while k <= params.scp.k_max and ((J_tr >= params.scp.ep_tr) or (J_vb >= params.scp.ep_vb) or (J_vc >= params.scp.ep_vc)):
        x, u, t, J_total, J_vb_vec, J_vc_vec, J_tr_vec, prob_stat, V_multi_shoot, subprop_time, dis_time = PTR_subproblem(cpg_solve, x_bar, u_bar, aug_dy, prob, params)
        
        V_multi_shoot_traj.append(V_multi_shoot)

        x_bar = x
        u_bar = u

        J_tr = np.sum(np.array(J_tr_vec))
        J_vb = np.sum(np.array(J_vb_vec))
        J_vc = np.sum(np.array(J_vc_vec))
        scp_trajs.append(x)
        scp_controls.append(u)

        params.scp.w_tr = min(params.scp.w_tr * params.scp.w_tr_adapt, params.scp.w_tr_max)
        if k > params.scp.cost_drop:
            params.scp.lam_cost = params.scp.lam_cost * params.scp.cost_relax
        
        # remove bottom labels and line
        if not k == 1:
            sys.stdout.write('\x1b[1A\x1b[2K\x1b[1A\x1b[2K')
        
        if prob_stat[3] == 'f':
            # Only show the first element of the string
            prob_stat = prob_stat[0]

        # Determine color for each value
        iter_colored = colored("{:4d}".format(k))
        J_tot_colored = colored("{:.1e}".format(J_total))
        J_tr_colored = colored("{:.1e}".format(J_tr), col_pos if J_tr <= params.scp.ep_tr else col_neg)
        J_vb_colored = colored("{:.1e}".format(J_vb), col_pos if J_vb <= params.scp.ep_vb else col_neg)
        J_vc_colored = colored("{:.1e}".format(J_vc), col_pos if J_vc <= params.scp.ep_vc else col_neg)
        cost_colored = colored("{:.1e}".format(t[-1]))
        prob_stat_colored = colored(prob_stat, col_pos if prob_stat == 'optimal' else col_neg)

        # Print with colors
        print("{:^4} |     {:^6.2f}    |      {:^6.2F}     | {:^7} | {:^7} | {:^7} | {:^7} |  {:^7} | {:^14}".format(
            iter_colored, dis_time*1000.0, subprop_time*1000.0, J_tot_colored, J_tr_colored, J_vb_colored, J_vc_colored, cost_colored, prob_stat_colored))

        print(colored("---------------------------------------------------------------------------------------------------------"))
        print("{:^4} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} |  {:^7} | {:^14}".format(
            "Iter", "Dis Time (ms)", "Solve Time (ms)", "J_total", "J_tr", "J_vb", "J_vc", "Cost", "Solver Status"))
        
        k += 1

    t_f_while = time.time()
    # Disable the profiler
    if params.sim.profiling:
        pr.disable()
        
        # Save results so it can be viusualized with snakeviz
        pr.dump_stats('profiling_results.prof')


    print(colored("---------------------------------------------------------------------------------------------------------"))
    # Define ANSI color codes
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Print with bold text
    print("------------------------------------------------ " + BOLD + "RESULTS" + RESET + " ------------------------------------------------")
    print("Total Computation Time: ", t_f_while - t_0_while)
    
    t = np.array(aug_dy.s_to_t(u, params))

    params.sim.total_time = t[-1]

    result = dict(
        converged = k <= params.scp.k_max,
        tof = t[-1],
        control_scp = u,
        state_scp = x,
        scp_trajs = scp_trajs,
        scp_controls = scp_controls,
        scp_multi_shoot = V_multi_shoot_traj,
        J_tr_vec = J_tr_vec,
        J_vb_vec = J_vb_vec,
        J_vc_vec = J_vc_vec,
    )
    return result

def PTR_post(params: Config, result: dict, aug_dy: ExactDis) -> dict:
    t_0_while = time.time()
    x = result['state_scp']
    u = result['control_scp']
    scp_trajs = result['scp_trajs']

    t = np.array(aug_dy.s_to_t(u, params))

    u_lam = u_lambda(u, t, params)
    t_full = np.arange(0, t[-1], params.sim.dt)

    tau_vals, u_full = aug_dy.t_to_tau(u_lam, t_full, u, t, params)

    x_full = simulate_nonlinear_time(x[0], u_lam, tau_vals, t, aug_dy, params)

    x_sub_full = []
    if hasattr(params.veh, 'init_poses') or hasattr(params.veh, 'get_kp_pose'):
        x_sub_full, _, x_sub_sen = full_subject_traj_time(x_full, params, False)
        x_sub_sen_node = subject_traj(x, params)
    else:
        x_sub_sen = []
        x_sub_sen_node = []

    print("Total CTCS Constraint Violation:", x_full[-1, params.veh.y_inds])
    i = 0
    cost = np.zeros_like(x[-1, i])
    for type in params.veh.initial_state.type:
        if type == 'Minimize':
            cost += x[0, i]
        i +=1
    i = 0
    for type in params.veh.final_state.type:
        if type == 'Minimize':
            cost += x[-1, i]
        i +=1
    print("Cost: ", cost)
    
    scp_trajs_interp = scp_traj_interp(scp_trajs, params)\

    more_result = dict(
        t_full = t_full,
        state = x_full,
        control = u_full,
        sub_positions = x_sub_full,
        sub_positions_sen = x_sub_sen,
        sub_positions_sen_node = x_sub_sen_node,
        scp_interp = scp_trajs_interp,
    )

    t_f_while = time.time()
    print("Total Post Processing Time: ", t_f_while - t_0_while)
    result.update(more_result)
    return result


def PTR_subproblem(cpg_solve, x_bar, u_bar, aug_dy, prob, params: Config):
    J_vb_vec = []
    J_vc_vec = []
    J_tr_vec = []
    
    prob.param_dict['x_bar'].value = x_bar
    prob.param_dict['u_bar'].value = u_bar
    
    t0 = time.time()
    A_bar, B_bar, C_bar, z_bar, V_multi_shoot = aug_dy.calculate_discretization(x_bar, u_bar.astype(float))
    dis_time = time.time() - t0
    
    prob.param_dict['A_d'].value = np.asarray(A_bar)
    prob.param_dict['B_d'].value = np.asarray(B_bar)
    prob.param_dict['C_d'].value = np.asarray(C_bar)
    prob.param_dict['z_d'].value = np.asarray(z_bar)
    dis_time = time.time() - t0
    
    prob.param_dict['w_tr'].value = params.scp.w_tr
    prob.param_dict['lam_cost'].value = params.scp.lam_cost

    if params.sim.cvxpygen:
        t0 = time.time()
        prob.register_solve('CPG', cpg_solve)
        prob.solve(method = 'CPG', abstol = 1E-6, reltol = 1E-9)
        subprop_time = time.time() - t0
    else:
        t0 = time.time()
        prob.solve(solver = params.sim.solver, enforce_dpp = True, abstol = 1E-6, reltol = 1E-9)
        subprop_time = time.time() - t0

    x = (params.sim.S_x @ prob.var_dict['x'].value.T + np.expand_dims(params.sim.c_x, axis = 1)).T
    u = (params.sim.S_u @ prob.var_dict['u'].value.T + np.expand_dims(params.sim.c_u, axis = 1)).T

    i = 0
    costs = 0
    for type in params.veh.final_state.type:
        if type == 'Minimize':
            costs = x[:,i]
        i += 1

    J_tr_vec.append(la.norm(la.inv(params.sim.S_x) @ (x[0] - x_bar[0])))
    for k in range(params.scp.n):
        J_tr_vec.append(la.norm(la.inv(sla.block_diag(params.sim.S_x, params.sim.S_u)) @ (np.hstack((x[k], u[k-1])) - np.hstack((x_bar[k], u_bar[k-1]))))**2)

        J_vb = 0
        J_vb_vec.append(J_vb)
        J_vc_vec.append(np.sum(np.abs(prob.var_dict['nu'].value[k-1])))
    return x, u, costs, prob.value, J_vb_vec, J_vc_vec, J_tr_vec, prob.status, V_multi_shoot, subprop_time, dis_time

def scp_traj_interp(scp_trajs, params):
    scp_prop_trajs = []
    for traj in scp_trajs:
        states = []
        for k in range(params.scp.n):
            traj_temp = np.repeat(np.expand_dims(traj[k], axis = 1), params.sim.inter_sample - 1, axis = 1)
            for i in range(1, params.sim.inter_sample - 1):
                states.append(traj_temp[:,i])
        scp_prop_trajs.append(np.array(states))
    return scp_prop_trajs

def intro():
    # Silence syntax warnings
    warnings.filterwarnings("ignore")
    ascii_art = '''
                             
                               ____                    _____  _____           
                              / __ \                  / ____|/ ____|          
                             | |  | |_ __   ___ _ __ | (___ | |  __   ____  __
                             | |  | | '_ \ / _ \ '_ \ \___ \| |  \ \ / /\ \/ /
                             | |__| | |_) |  __/ | | |____) | |___\ V /  >  < 
                              \____/| .__/ \___|_| |_|_____/ \_____\_/  /_/\_\ 
                                    | |                                       
                                    |_|                                       
---------------------------------------------------------------------------------------------------------
                                Author: Chris Hayner and Griffin Norris
                                    Autonomous Controls Laboratory
                                       University of Washington
---------------------------------------------------------------------------------------------------------
'''
    print(ascii_art)
