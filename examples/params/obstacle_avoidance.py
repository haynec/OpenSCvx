import numpy as np
from jax import vmap, jit, jacfwd
import jax.numpy as jnp
import jax
from openscvx.config import (
    SimConfig,
    ScpConfig,
    Config,
)

from openscvx.dynamics import Dynamics
from openscvx.utils import qdcm, SSMP, SSM, generate_orthogonal_unit_vectors

n = 6 # Discretization Nodes
total_time = 4.0  # Total time for the simulation

class ObstacleAvoidanceDynamics(Dynamics):
    def __init__(self):
        self.t_inds = -2          # Time Index in State
        self.y_inds = -1          # Constraint Violation Index in State
        self.s_inds = -1          # Time dilation index in Control

        self.m = 1.0  # Mass of the drone
        self.g_const = -9.18
        self.J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone

        self.max_state = np.array([200, 10, 20, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 100, 1E-4])
        self.min_state = np.array([-200, -100, 0, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0, 0])

        self.initial_state= {'value' : [   10,     0,     2,     0,     0,     0,      1,      0,      0,      0,      0,      0,      0,     0],
                             'type'  : ['Fix', 'Fix', 'Fix', 'Fix', 'Fix', 'Fix', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Fix']}  # Initial State
        
        self.final_state= {'value' : [  -10,     0,     2,      0,      0,      0,      1,      0,      0,      0,      0,       0,       0,  total_time],
                           'type' :  ['Fix', 'Fix', 'Fix', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free',   'Free',  'Minimize']}

        self.initial_control = np.array([0, 0, 50, 0, 0, 0, 1])


        ### Ellipsoidal Obstacle Params ###
        self.obstacle_centers=[
        np.array([-5.1, 0.1, 2]),
        np.array([0.1, 0.1, 2]),
        np.array([5.1, 0.1, 2])]

        self.A_obs = []
        self.radius = []
        self.axes = []
        np.random.seed(0)
        for _ in self.obstacle_centers:
            ax = generate_orthogonal_unit_vectors()
            self.axes.append(generate_orthogonal_unit_vectors())
            rad = np.random.rand(3) + 0.1 * np.ones(3)
            self.radius.append(rad)
            self.A_obs.append(ax @ np.diag(rad**2) @ ax.T)

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
        return jnp.hstack([r_dot, v_dot, q_dot, w_dot, t_dot])

    def g_obs(self, center, A, x):
        return 1 - (x[:3] - center).T @ A @ (x[:3] - center)

    def g_func(self, x):
        g = 0
        for center, A in zip(self.obstacle_centers, self.A_obs):
            g += jnp.maximum(0, self.g_obs(center, A, x))**2
        return g + jnp.sum(jnp.maximum(0, (x[:-1] - self.max_state[:-1])) ** 2) + jnp.sum(jnp.maximum(0, (self.min_state[:-1] - x[:-1])) ** 2)
    
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
        return x_bar, u_bar

dy = ObstacleAvoidanceDynamics()
initial_guess = Initial_Guess(dy)

sim = SimConfig(
    x_bar=initial_guess.x_bar,  # Initial Guess for the States
    u_bar=initial_guess.u_bar,  # Initial Guess for the Controls
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
    max_dt=1e1,  # Maximum Time Step
    min_dt=1e-2,  # Minimum Time Step
    total_time=total_time,
    n_states=len(dy.max_state),  # Number of States
    dt = 0.01,
)
scp = ScpConfig(
    n=n,
    w_tr=1E1,                     # Weight on the Trust Reigon
    lam_cost=1E1,                 # Weight on the Nonlinear Cost
    lam_vc=1E2,                   # Weight on the Virtual Control Objective (not including CTCS Augmentation)
    ep_tr=1e-4,                   # Trust Region Tolerance
    ep_vb=1e-4,                   # Virtual Control Tolerance
    ep_vc=1e-8,                   # Virtual Control Tolerance for CTCS
    cost_drop=4,                  # SCP iteration to relax minimal final time objective
    cost_relax=0.5,               # Minimal Time Relaxation Factor
    w_tr_adapt=1.2,               # Trust Region Adaptation Factor
    w_tr_max_scaling_factor=1E2,  # Maximum Trust Region Weight
    gen_code=False,               # Generate the code
)

params = Config(sim=sim, scp=scp, veh=dy)
