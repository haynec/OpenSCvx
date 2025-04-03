import numpy as np
import numpy.linalg as la
from jax import vmap, jit, jacfwd
import jax.numpy as jnp
from openscvx.config import (
    SimConfig,
    ScpConfig,
    Config,
)

from openscvx.dynamics import Dynamics
from openscvx.utils import qdcm, SSMP, SSM

n = 12 # Number of Nodes
total_time = 40.0  # Total time for the simulation

class CinemaVPDynamics(Dynamics):
    def __init__(self):
        self.t_inds = -3          # Time Index in State
        self.fuel_inds = -2       # Fuel Index in State
        self.y_inds = -1          # Constraint Violation Index in State
        self.s_inds = -1          # Time dilation index in Control

        self.max_state = np.array([ 200,  100,  50,  100,  100,  100,  1,  1,  1,  1,  10,  10,  10,  40, 2000, 1E-8])  # Upper Bound on the states
        self.min_state = np.array([-100, -100, -10, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10,   0,   0,     0])  # Lower Bound on the states

        self.initial_state= {'value' : [    8,  -0.2,   2.2,     0,     0,     0,      1,      0,      0,      0,      0,      0,      0,        0,       0],
                             'type' :  ['Fix', 'Fix', 'Fix', 'Fix', 'Fix', 'Fix', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free',    'Fix',   'Fix']}  # Initial State
        
        self.final_state= {'value' : [   -10,      0,      2,      0,      0,      0,      1,      0,      0,      0,      0,      0,      0,    40,          0],
                           'type'  : ['Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Fix', 'Minimize']}


        self.initial_control=np.array([0, 0, 10, 0, 0, 0, 1])

        self.m = 1.0  # Mass of the drone
        self.g_const = -9.18
        self.J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone
        
    
        self.get_kp_pose = self.get_kp_pose

        self.init_pose = np.array([13.0, 0.0, 2.0])
        self.min_range = 4.0
        self.max_range = 16.0

        ### View Planning Params ###
        self.n_subs = 1  # Number of Subjects
        self.alpha_x = 6.0  # Angle for the x-axis of Sensor Cone
        self.alpha_y = 8.0  # Angle for the y-axis of Sensor Cone
        self.A_cone = np.diag(
            [
                1 / np.tan(np.pi / self.alpha_x),
                1 / np.tan(np.pi / self.alpha_y),
                0,
            ]
        )  # Conic Matrix in Sensor Frame
        self.c = jnp.array([0, 0, 1]) # Boresight Vector in Sensor Frame
        self.norm_type = np.inf  # Norm Type
        self.R_sb=jnp.array([[0, 1, 0], 
                             [0, 0, 1], 
                             [1, 0, 0]]
                             )
        super().__post_init__()

    def dynamics(self, x, u):
        # Unpack the state and control vectors
        v = x[3:6]
        q = x[6:10]
        w = x[10:13]

        f = u[:3]
        tau = u[3:]

        q_norm = jnp.linalg.norm(q)
        q = q / q_norm

        # Compute the time derivatives of the state variables
        r_dot = v
        v_dot = (1 / self.m) * qdcm(q) @ f + jnp.array([0, 0, self.g_const])
        q_dot = 0.5 * SSMP(w) @ q
        w_dot = jnp.diag(1/self.J_b) @ (
            tau - SSM(w) @ jnp.diag(self.J_b) @ w
        )
        t_dot = 1
        fuel_dot = jnp.linalg.norm(u)[None]
        y_dot = self.g_jit(x)
        return jnp.hstack([r_dot, v_dot, q_dot, w_dot, t_dot, fuel_dot, y_dot])

    def get_kp_pose(self, t):
        loop_time = 40.0
        loop_radius = 20.0

        t_angle = t / loop_time * (2 * jnp.pi)
        x = loop_radius * jnp.sin(t_angle)
        y = x * jnp.cos(t_angle)
        z = 0.5 * x * jnp.sin(t_angle)
        return jnp.array([x, y, z]).T + self.init_pose

    def g_vp(self, x):
        p_s_I = self.get_kp_pose(x[self.t_inds])
        p_s_s = self.R_sb @ qdcm(x[6:10]).T @ (p_s_I - x[:3])
        return jnp.linalg.norm(self.A_cone @ p_s_s, ord=self.norm_type) - (self.c.T @ p_s_s)
    
    def g_min(self, x):
        p_s_I = self.get_kp_pose(x[self.t_inds])
        return self.min_range - jnp.linalg.norm(p_s_I - x[:3])
    
    def g_max(self, x):
        p_s_I = self.get_kp_pose(x[self.t_inds])
        return jnp.linalg.norm(p_s_I - x[:3]) - self.max_range

    def g_func(self, x):
        return 2E1 * jnp.maximum(0, self.g_vp(x)) ** 2 + jnp.sum(jnp.maximum(0, (x[:-1] - self.max_state[:-1])) ** 2) + jnp.sum(jnp.maximum(0, (self.min_state[:-1] - x[:-1])) ** 2) + jnp.maximum(0, self.g_min(x)) ** 2 + jnp.maximum(0, self.g_max(x)) ** 2

class Initial_Guess():
    def __init__(self, dy):
        self.dy = dy
        self.x_bar, self.u_bar = self.initial_guess(dy)
    
    def initial_guess(self, dy):
        u_bar = np.repeat(np.expand_dims(dy.initial_control, axis = 0), n, axis = 0)
        s = total_time
        u_bar[:,-1] = np.repeat(s, n)

        x_bar = np.repeat(np.expand_dims(np.zeros_like(dy.max_state), axis=0), n, axis = 0)
        x_bar[:,:dy.y_inds] = np.linspace(dy.initial_state['value'], dy.final_state['value'], n)

        x_bar[:,:3] = dy.get_kp_pose(x_bar[:,dy.t_inds]) + jnp.array([-5, 0.2, 0.2])[None, :]
        
        R_sb = dy.R_sb # Sensor to body frame
        b = R_sb @ np.array([0, 1, 0])
        for k in range(n):
            kp = dy.get_kp_pose(x_bar[k, dy.t_inds])
            a = kp - x_bar[k,:3]
            # Determine the direction cosine matrix that aligns the z-axis of the sensor frame with the relative position vector
            q_xyz = np.cross(b, a)
            q_w = np.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2) + np.dot(a,b)
            q_no_norm = np.hstack((q_w, q_xyz))
            q = q_no_norm / la.norm(q_no_norm)
            x_bar[k,6:10] = q
        return x_bar, u_bar

dy = CinemaVPDynamics()
initial_guess = Initial_Guess(dy)
sim = SimConfig(
    x_bar=initial_guess.x_bar,  # Initial Guess State Trajectory
    u_bar=initial_guess.u_bar,  # Initial Guess Control Sequence
    initial_state=dy.initial_state,  # Initial State
    final_state=dy.final_state,  # Final State
    initial_control=dy.initial_control,  # Initial Control
    max_state=dy.max_state,  # Upper Bound on the states
    min_state=dy.min_state,  # Lower Bound on the states
    max_control=np.array(
        [0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562, 3.0 * total_time]
    ),  # Upper Bound on the controls
    min_control=np.array(
        [0, 0, 0, -18.665, -18.665, -0.55562, 0.3 * total_time]
    ),  # Lower Bound on the controls
    total_time=total_time,  # Total time for the simulation
    n_states = len(dy.max_state),  # Number of States
    dt=0.1
)
scp = ScpConfig(
    n=n, # Number of Nodes
    w_tr=4E0,  # Weight on the Trust Reigon
    lam_cost=1E-2,  # Weight on the Minimal Fuel Objective
    lam_vc=1E1,  # Weight on the Virtual Control Objective (not including CTCS Augmentation)
    ep_tr=5e-6,  # Trust Region Tolerance
    ep_vb=1e-4,  # Virtual Control Tolerance
    ep_vc=1e-8,  # Virtual Control Tolerance for CTCS
    w_tr_adapt=1.3,  # Trust Region Adaptation Factor
    w_tr_max_scaling_factor=1e3,  # Maximum Trust Region Weight
    gen_code=False,
)
params = Config(sim=sim, scp=scp, veh=dy)