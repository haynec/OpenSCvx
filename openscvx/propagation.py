import numpy as np
import jax.numpy as jnp
import scipy.integrate as itg
from scipy.interpolate import interp1d

from openscvx.utils import qdcm
from openscvx.config import Config

def simulate_nonlinear_time(x_0, u_lam, tau_vals, t, aug_dy, params: Config):
    states = []
    tau = np.linspace(0, 1, params.scp.n)
    
    # Bin the tau_vals into with respect to the uniform tau grid, tau
    tau_inds = np.digitize(tau_vals, tau) - 1

    # Force the last indice to be in the same bin as the previous ones
    tau_inds[tau_inds == params.scp.n-1] = params.scp.n-2

    for k in range(params.scp.n-1):
        controls_current = np.squeeze(u_lam(t[k]))[None,:]
        controls_next = np.squeeze(u_lam(t[k+1]))[None,:]
        
        # Obtain those values from tau_vals
        tau_cur = tau_vals[(tau_inds >= k) & (tau_inds < k+1)]

        sol = itg.solve_ivp(aug_dy.prop_aug_dy, (tau[k], tau[k+1]), x_0, args=(np.array(controls_current), np.array(controls_next), np.array([[tau[k]]]), params.veh.s_inds), method='DOP853', dense_output=True)
        x = sol.y
        x_time = sol.sol(tau_cur)
        for i in range(x_time.shape[1]):
            states.append(x_time[:,i])
        x_0 = x[:,-1]
    
    return np.array(states)

def u_lambda(u, t, params: Config):
    """
    Generate a lambda function that linearly interpolates between the control input given a time.

    Parameters:
    u (np.ndarray): Array of control inputs (shape: m x n).
    t (np.ndarray): Array of time points corresponding to the control inputs (shape: n).
    params (dict): Additional parameters if needed.

    Returns:
    function: A lambda function that interpolates the control input at a given time.
    """

    # Ensure t is a 1D array
    t = t.flatten()

    # Determine the interpolation method based on params
    if params.scp.dis_type == 'ZOH':
        kind = 'previous'
    else:
        kind = 'linear'

    # Create the interpolator
    interpolators = [interp1d(t, u_row, kind=kind, fill_value="extrapolate") for u_row in u.T]

    # Return the lambda function
    def interpolate(time):
        time = np.atleast_1d(time)
        u_interp = np.array([interp(time) for interp in interpolators])
        return u_interp.reshape(-1, 1)

    return interpolate


def full_subject_traj(x_full, params, init):
    t_full = x_full[params.veh.t_inds]
    subs_traj = []
    subs_traj_sen = []
    # # t = s_to_t(u, params)
    # t_full = []
    # for i in range(params.scp.n-1):
    #     t_interp = np.linspace(t[i], t[i+1], params.sim.inter_sample)
    #     t_interp = t_interp[:-1]
    #     t_full.append(t_interp)
    # t_full = np.array(t_full).flatten()
    # # Add the last element of t
    # t_full = np.append(t_full, t[-1])
    if params.vp.tracking:
        subs_traj = [params.veh.get_kp_pose(t_full)]
    else:
        for pose in params.veh.init_poses:
            subs_traj.append(pose)
    
    if not init:
        R_sb = params.vp.R_sb
        for sub_traj in subs_traj:
            sub_traj_sen = []
            for i in range(x_full.shape[0]):
                sub_pose = sub_traj[i]
                sub_traj_sen.append(R_sb @ qdcm(x_full[i, 6:10]).T @ (sub_pose - x_full[i, 0:3]))
            subs_traj_sen.append(sub_traj_sen)
    else:
        subs_traj_sen = None
    
    return subs_traj, np.array(t_full).flatten(), subs_traj_sen

def full_subject_traj_time(x_full, params, init):
    t_full = x_full[:,params.veh.t_inds]

    subs_traj = []
    subs_traj_sen = []
    
    if hasattr(params.veh, 'get_kp_pose'):
        subs_traj.append(params.veh.get_kp_pose(t_full))
    else:
        for pose in params.veh.init_poses:
            # repeat the pose for all time steps
            pose = np.repeat(pose[:,np.newaxis], x_full.shape[0], axis=1).T
            subs_traj.append(pose)
        
    if not init:
        R_sb = params.veh.R_sb
        for sub_traj in subs_traj:
            sub_traj_sen = []
            for i in range(x_full.shape[0]):
                sub_pose = sub_traj[i]
                sub_traj_sen.append(R_sb @ qdcm(x_full[i, 6:10]).T @ (sub_pose - x_full[i, 0:3]))
            subs_traj_sen.append(sub_traj_sen)
    else:
        subs_traj_sen = None
    return subs_traj, np.array(t_full).flatten(), subs_traj_sen

def subject_traj(x, params: Config):
    subs_traj = []
    subs_traj_sen = []
    t = x[:,params.veh.t_inds]
    if hasattr(params.veh, 'get_kp_pose'):
        subs_traj = [params.veh.get_kp_pose(t)]
    else:
        for pose in params.veh.init_poses:
            sub_traj = []
            pose = np.repeat(pose[:,np.newaxis], x.shape[0], axis=1).T
            subs_traj.append(pose)

    R_sb = params.veh.R_sb
    for sub_traj in subs_traj:
        sub_traj_sen = []
        for i in range(x.shape[0]):
            sub_pose = sub_traj[i]
            sub_traj_sen.append(R_sb @ qdcm(x[i, 6:10]).T @ (sub_pose - x[i, 0:3]))
        subs_traj_sen.append(sub_traj_sen)
    return subs_traj_sen