import numpy as np
import numpy.linalg as la
import cvxpy as cp
from jax import vmap, jit, jacfwd
import jax.numpy as jnp
from openscvx.config import (
    SimConfig,
    ScpConfig,
    Config,
)

from openscvx.dynamics import Dynamics
from openscvx.utils import qdcm, SSMP, SSM

n = 33 # Number of Nodes
total_time = 40.0  # Total time for the simulation

class DrVpDynamics(Dynamics):
    def __init__(self):
        self.t_inds = -2          # Time Index in State
        self.y_inds = -1          # Constraint Violation Index in State
        self.s_inds = -1          # Time dilation index in Control

        self.max_state=np.array([200, 100, 50, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 100, 1e-4])  # Upper Bound on the states
        self.min_state=np.array([-200, -100, 15, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0, 0])  # Lower Bound on the states

        self.initial_state= {'value' : [10, 0, 20, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                             'type'  : ['Fix', 'Fix', 'Fix', 'Fix', 'Fix', 'Fix', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Fix']}  # Initial State
        
        self.final_state= {'value' : [10, 0, 20, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, total_time],
                            'type' : ['Fix', 'Fix', 'Fix', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Minimize']}

        self.initial_control = np.array([0, 0, 10, 0, 0, 0, 1])

        self.m = 1.0  # Mass of the drone
        self.g_const = -9.18
        self.J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone
        
        ### Sensor Params ###
        self.alpha_x = 6.0  # Angle for the x-axis of Sensor Cone
        self.alpha_y = 6.0  # Angle for the y-axis of Sensor Cone
        self.A_cone = np.diag(
            [
                1 / np.tan(np.pi / self.alpha_x),
                1 / np.tan(np.pi / self.alpha_y),
                0,
            ]
        )  # Conic Matrix in Sensor Frame
        self.c = jnp.array([0, 0, 1]) # Boresight Vector in Sensor Frame
        self.norm_type = 2  # Norm Type
        self.R_sb=jnp.array([[0, 1, 0], 
                             [0, 0, 1], 
                             [1, 0, 0]]
                             )
        ### End Sensor Params ###

        ### Gate Parameters ###
        self.n_gates = 10
        self.gate_centers =[np.array([59.436, 0.0000, 20.0000]),
                            np.array([92.964, -23.750, 25.5240]),
                            np.array([92.964, -29.274, 20.0000]),
                            np.array([92.964, -23.750, 20.0000]),
                            np.array([130.150, -23.750, 20.0000]),
                            np.array([152.400, -73.152, 20.0000]),
                            np.array([92.964, -75.080, 20.0000]),
                            np.array([92.964, -68.556, 20.0000]),
                            np.array([59.436, -81.358, 20.0000]),
                            np.array([22.250, -42.672, 20.0000]),
                        ]
        self.rot = np.array([
                            [np.cos(np.pi / 2), np.sin(np.pi / 2), 0],
                            [-np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
                            [0, 0, 1],
                            ])
        
        self.radii = np.array([2.5, 1E-4, 2.5])
        self.A_gate = self.rot @ np.diag(1/self.radii) @ self.rot.T
        self.A_gate_cen = []
        for center in self.gate_centers:
            center[0] = center[0] + 2.5
            center[2] = center[2] + 2.5
            self.A_gate_cen.append(self.A_gate @ center)
        self.nodes_per_gate = 3
        self.gate_nodes = np.arange(self.nodes_per_gate,n,self.nodes_per_gate)
        self.vertices = []
        for center in self.gate_centers:
            self.vertices.append(self.gen_vertices(center))
        ### End Gate Parameters ### 

        n_subs = 10
        init_poses = []
        np.random.seed(0)
        for i in range(n_subs):
            init_pose = np.array([100.0, -60.0, 20.0])
            init_pose[:2] = init_pose[:2] + np.random.random(2) * 20.0
            init_poses.append(init_pose)
        
        self.init_poses = init_poses

        super().__post_init__()

    def g_vp(self, p_s_I, x):
        p_s_s = self.R_sb @ qdcm(x[6:10]).T @ (p_s_I - x[0:3])
        return jnp.linalg.norm(self.A_cone @ p_s_s, ord=self.norm_type) - (self.c.T @ p_s_s)
    

    def huber_loss(self, x, delta=1.0):
        abs_x = jnp.abs(x)
        quadratic = jnp.minimum(abs_x, delta)
        linear = abs_x - quadratic
        return 0.5 * quadratic ** 2 + delta * linear

    def g_func(self, x):
        g = 0
        for pose in self.init_poses:
            g += jnp.maximum(0, self.g_vp(pose, x)) ** 2
        g += jnp.sum(jnp.maximum(0, (x[:-1] - self.max_state[:-1])) ** 2) + jnp.sum(jnp.maximum(0, (self.min_state[:-1] - x[:-1])) ** 2)
        return g
    
    def g_cvx_nodal(self, x): # Nodal Convex Inequality Constraints
        constr = []
        for node, cen in zip(self.gate_nodes, self.A_gate_cen):
            constr += [cp.norm(self.A_gate @ x[node][:3] - cen, "inf") <= 1]
        return constr
    
    def gen_vertices(self, center):
        """
        Obtains the vertices of the gate.
        """
        vertices = []
        vertices.append(center + self.rot @ [self.radii[0], 0, self.radii[2]])
        vertices.append(center + self.rot @ [-self.radii[0], 0, self.radii[2]])
        vertices.append(center + self.rot @ [-self.radii[0], 0, -self.radii[2]])
        vertices.append(center + self.rot @ [self.radii[0], 0, -self.radii[2]])
        return vertices
 
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

        i = 0
        origins = [dy.initial_state['value'][:3]]
        ends = []
        for center in  dy.gate_centers:
            origins.append(center)
            ends.append(center)
        ends.append(dy.final_state['value'][:3])
        gate_idx = 0
        for _ in range(dy.n_gates + 1):
            for k in range(n//(dy.n_gates + 1)):
                x_bar[i,:3] = origins[gate_idx] + (k/(n//(dy.n_gates + 1))) * (ends[gate_idx] - origins[gate_idx])
                i += 1
            gate_idx += 1
        
        R_sb = dy.R_sb # Sensor to body frame
        b = R_sb @ np.array([0, 1, 0])
        for k in range(n):
            kp = []
            for pose in dy.init_poses:
                kp.append(pose)
            kp = np.mean(kp, axis = 0)
            a = kp - x_bar[k,:3]
            # Determine the direction cosine matrix that aligns the z-axis of the sensor frame with the relative position vector
            q_xyz = np.cross(b, a)
            q_w = np.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2) + np.dot(a,b)
            q_no_norm = np.hstack((q_w, q_xyz))
            q = q_no_norm / la.norm(q_no_norm)
            x_bar[k,6:10] = q
        return x_bar, u_bar

dy = DrVpDynamics()
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
    max_dt=1e2,  # Maximum Time Step
    min_dt=1e-2,  # Minimum Time Step
    total_time=total_time,
    n_states=len(dy.max_state),  # Number of States
    dt=0.1
)
scp = ScpConfig(
    k_max=200,
    n=n,
    w_tr=2E0,  # Weight on the Trust Reigon
    lam_cost=1E-1, #0e-1,  # Weight on the Minimal Time Objective
    lam_vc=1E1, #1e1,  # Weight on the Virtual Control Objective (not including CTCS Augmentation)
    ep_tr=1e-3,  # Trust Region Tolerance
    ep_vb=1e-4,  # Virtual Control Tolerance
    ep_vc=1e-8,  # Virtual Control Tolerance
    cost_drop=10,  # SCP iteration to relax minimal final time objective
    cost_relax=0.8,  # Minimal Time Relaxation Factor
    w_tr_adapt=1.4,  # Trust Region Adaptation Factor
    w_tr_max_scaling_factor=1e2,  # Maximum Trust Region Weight
    dis_type='FOH',  # Discretization Type
    gen_code=False,
)
params = Config(sim=sim, scp=scp,veh=dy)