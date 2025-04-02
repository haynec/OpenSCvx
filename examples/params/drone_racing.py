import numpy as np
from jax import vmap, jit, jacfwd
import jax.numpy as jnp
import cvxpy as cp
from openscvx.config import (
    SimConfig,
    ScpConfig,
    Config,
)

n = 22 # Number of Nodes
total_time = 24.0  # Total time for the simulation

class Dynamics:
    def __init__(self):
        self.t_inds = -2          # Time Index in State
        self.y_inds = -1          # Constraint Violation Index in State

        self.max_state=np.array([200, 100, 50, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 100, 1e-4])  # Upper Bound on the states
        self.min_state=np.array([-200, -100, 15, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0, 0])  # Lower Bound on the states

        self.initial_state= {'value' : [   10,     0,    20,     0,     0,     0,      1,      0,      0,      0,      0,      0,      0,     0],
                             'type'  : ['Fix', 'Fix', 'Fix', 'Fix', 'Fix', 'Fix', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Fix']}  # Initial State
        
        self.final_state= {'value' : [   10,     0,    20,      0,      0,      0,      1,      0,      0,      0,      0,      0,      0, total_time],
                           'type'  : ['Fix', 'Fix', 'Fix', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Free', 'Minimize']}

        self.initial_control=np.array([0, 0, 10, 0, 0, 0, 1])

        self.m = 1.0  # Mass of the drone
        self.g_const = -9.18
        self.J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone


        ### Gate Parameters ###
        self.n_gates = 10
        self.gate_centers =[np.array([ 59.436,   0.000, 20.0000]),
                            np.array([ 92.964, -23.750, 25.5240]),
                            np.array([ 92.964, -29.274, 20.0000]),
                            np.array([ 92.964, -23.750, 20.0000]),
                            np.array([130.150, -23.750, 20.0000]),
                            np.array([152.400, -73.152, 20.0000]),
                            np.array([ 92.964, -75.080, 20.0000]),
                            np.array([ 92.964, -68.556, 20.0000]),
                            np.array([ 59.436, -81.358, 20.0000]),
                            np.array([ 22.250, -42.672, 20.0000]),
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
        self.nodes_per_gate = 2
        self.gate_nodes = np.arange(self.nodes_per_gate,n,self.nodes_per_gate)
        self.vertices = []
        for center in self.gate_centers:
            self.vertices.append(self.gen_vertices(center))
        ### End Gate Parameters ### 

        
        self.g = self.g
        self.g_vec = vmap(self.g, in_axes=(0))
  
        self.state_dot = vmap(self.state_dot_func)
        self.A = vmap(jacfwd(self.state_dot_func, argnums=0), in_axes=(0, 0))
        self.B = vmap(jacfwd(self.state_dot_func, argnums=1), in_axes=(0, 0))

    def g(self, x): # CTCS Inequality Constraints
        return jnp.sum(jnp.maximum(0, (x[:-1] - self.max_state[:-1])) ** 2) + jnp.sum(jnp.maximum(0, (self.min_state[:-1] - x[:-1])) ** 2)

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
    
    def qdcm(self, q: jnp.ndarray) -> jnp.ndarray:
        # Convert a quaternion to a direction cosine matrix
        q_norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
        w, x, y, z = q / q_norm
        return jnp.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )

    def SSMP(self, w: jnp.ndarray):
        # Convert an angular rate to a 4 x 4 skew symetric matrix
        x, y, z = w
        return jnp.array([[0, -x, -y, -z], [x, 0, z, -y], [y, -z, 0, x], [z, y, -x, 0]])

    def SSM(self, w: jnp.ndarray):
        # Convert an angular rate to a 3 x 3 skew symetric matrix
        x, y, z = w
        return jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
 
    def state_dot_func(self, x, u):
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
        v_dot = (1 / self.m) * self.qdcm(q) @ f + jnp.array([0, 0, self.g_const])
        q_dot = 0.5 * self.SSMP(w) @ q
        w_dot = jnp.diag(1/self.J_b) @ (
            tau - self.SSM(w) @ jnp.diag(self.J_b) @ w
        )
        t_dot = 1
        y_dot = self.g(x)
        return jnp.hstack([r_dot, v_dot, q_dot, w_dot, t_dot, y_dot])
    
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
        return x_bar, u_bar

dy = Dynamics()
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
    dt=0.01
)
scp = ScpConfig(
    k_max=200,
    n=n,
    w_tr=2E0,  # Weight on the Trust Reigon
    lam_cost=1E-1, #0e-1,  # Weight on the Minimal Time Objective
    lam_vc=1E1, #1e1,  # Weight on the Virtual Control Objective (not including CTCS Augmentation)
    ep_tr=1e-3,  # Trust Region Tolerance
    ep_vb=1e-4,  # Virtual Control Tolerance
    ep_vc=1e-8,  # Virtual Control Tolerance for CTCS
    cost_drop=10,  # SCP iteration to relax minimal final time objective
    cost_relax=0.8,  # Minimal Time Relaxation Factor
    w_tr_adapt=1.4,  # Trust Region Adaptation Factor
    w_tr_max_scaling_factor=1e2,  # Maximum Trust Region Weight
    dis_type='FOH',  # Discretization Type
    gen_code=False,
)
params = Config(sim=sim, scp=scp,veh=dy)