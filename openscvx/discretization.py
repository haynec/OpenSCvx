import jax.numpy as jnp
from jax import jit
import numpy as np
import scipy.integrate as itg

class AugmentedDynamics:
    def __init__(self, params) -> None:
        self.params = params

        # Extract the number of states and controls from the parameters
        n_x = self.params.sim.n_states
        n_u = self.params.sim.n_controls

        # Define indices for slicing the augmented state vector
        self.i0 = 0
        self.i1 = n_x
        self.i2 = self.i1 + n_x * n_x
        self.i3 = self.i2 + n_x * n_u
        self.i4 = self.i3 + n_x * n_u
        self.i5 = self.i4 + n_x

        self.tau_grid = jnp.linspace(0, 1, self.params.scp.n)

        dVdt_lower = jit(self.dVdt_fun).lower(0.0, np.ones(int(self.i5*(self.params.scp.n-1))), np.ones((self.params.scp.n-1, self.params.sim.n_controls)), np.ones((self.params.scp.n-1, self.params.sim.n_controls)))
        self.dVdt = dVdt_lower.compile()
        # self.dVdt = self.dVdt_fun
    
    def s_to_t(self, u, params):
        t = [0]
        tau = np.linspace(0, 1, params.scp.n)
        for k in range(1, params.scp.n):
            s_kp = u[k-1,-1]
            s_k = u[k,-1]
            if params.scp.dis_type == 'ZOH':
                t.append(t[k-1] + (tau[k] - tau[k-1])*(s_kp))
            else:
                t.append(t[k-1] + 0.5 * (s_k + s_kp) * (tau[k] - tau[k-1]))
        return t
    
    def t_to_tau(self, u_lam, t, u_nodal, t_nodal, params):
        u = np.array([u_lam(t_i) for t_i in t])

        tau = np.zeros(len(t))
        tau_nodal = np.linspace(0, 1, params.scp.n)
        for k in range(1, len(t)):
            k_nodal = np.where(t_nodal < t[k])[0][-1]
            s_kp = u_nodal[k_nodal, -1]
            tp = t_nodal[k_nodal]
            tau_p = tau_nodal[k_nodal]

            s_k = u[k, -1]
            if params.scp.dis_type == 'ZOH':
                tau[k] = tau_p + (t[k] - tp) / s_kp
            else:
                tau[k] = tau_p + 2 * (t[k] - tp) / (s_k + s_kp)
        return tau, u
    
    def calculate_discretization(self,
                                 x: np.ndarray,
                                 u: np.ndarray):
        """
        Calculate discretization for given states, inputs and total time.
        x: Matrix of states for all time points
        u: Matrix of inputs for all time points
        return: A_k, B_k, C_k, z_k
        """

        # Extract the number of states and controls from the parameters
        n_x = self.params.sim.n_states
        n_u = self.params.sim.n_controls

        # Initialize the augmented state vector
        V0 = np.zeros((x.shape[0]-1, self.i5))
        V0[:, self.i1:self.i2] = np.tile(np.eye(n_x).reshape(-1), (x.shape[0]-1, 1))
        
        # Initialize the augmented Jacobians
        A_bar = np.zeros((n_x * n_x, self.params.scp.n - 1))
        B_bar = np.zeros((n_x * n_u, self.params.scp.n - 1))
        C_bar = np.zeros((n_x * n_u, self.params.scp.n - 1))
        z_bar = np.zeros((n_x, self.params.scp.n - 1))

        # Vectorized integration
        V0[:, self.i0:self.i1] = x[:-1]
        int_result = itg.solve_ivp(self.dVdt, (self.tau_grid[0], self.tau_grid[1]), V0.flatten(), args=(u[:-1, :].astype(float), u[1:, :].astype(float)), method='RK45', t_eval=np.linspace(self.tau_grid[0], self.tau_grid[1], 50))
        V = int_result.y[:,-1].reshape(-1, self.i5)

        V_multi_shoot = int_result.y

        # Flatten matrices in column-major (Fortran) order for cvxpy
        # A_bar = V[:, self.i1:self.i2].reshape((self.params.scp.n - 1, n_x, n_x)).transpose(1, 2, 0).reshape(n_x * n_x, -1, order='F')
        # B_bar = V[:, self.i2:self.i3].reshape((self.params.scp.n - 1, n_x, n_u)).transpose(1, 2, 0).reshape(n_x * n_u, -1, order='F')
        # C_bar = V[:, self.i3:self.i4].reshape((self.params.scp.n - 1, n_x, n_u)).transpose(1, 2, 0).reshape(n_x * n_u, -1, order='F')
        # z_bar = V[:, self.i4:self.i5].T
        A_bar = V[:, self.i1:self.i2].reshape((self.params.scp.n - 1, n_x, n_x)).transpose(1, 2, 0).reshape(n_x * n_x, -1, order='F').T
        B_bar = V[:, self.i2:self.i3].reshape((self.params.scp.n - 1, n_x, n_u)).transpose(1, 2, 0).reshape(n_x * n_u, -1, order='F').T
        C_bar = V[:, self.i3:self.i4].reshape((self.params.scp.n - 1, n_x, n_u)).transpose(1, 2, 0).reshape(n_x * n_u, -1, order='F').T
        z_bar = V[:, self.i4:self.i5]
        return A_bar, B_bar, C_bar, z_bar, V_multi_shoot

    def dVdt_fun(self,
             tau: float,
             V: jnp.ndarray,
             u_cur: np.ndarray,
             u_next: np.ndarray
             ) -> jnp.ndarray:
        """
        Computes the time derivative of the augmented state vector for the system for a sequence of states.

        Parameters:
        tau (float): Current time.
        V (np.ndarray): Sequence of augmented state vectors.
        u_cur (np.ndarray): Sequence of current control inputs.
        u_next (np.ndarray): Sequence of next control inputs.
        A: Function that computes the Jacobian of the system dynamics with respect to the state.
        B: Function that computes the Jacobian of the system dynamics with respect to the control input.
        obstacles: List of obstacles in the environment.
        params (dict): Parameters of the system.

        Returns:
        np.ndarray: Time derivatives of the augmented state vectors.
        """
        
        # Extract the number of states and controls from the parameters
        n_x = self.params.sim.n_states
        n_u = self.params.sim.n_controls

        # Unflatten V
        V = V.reshape(-1, self.i5)

        # Compute the interpolation factor based on the discretization type
        if self.params.scp.dis_type == 'ZOH':
            beta = 0.
        elif self.params.scp.dis_type == 'FOH':
            beta = (tau) * self.params.scp.n
        alpha = 1 - beta

        # Interpolate the control input
        u = u_cur + beta * (u_next - u_cur)
        s = u[:,-1]

        # Initialize the augmented Jacobians
        dfdx = jnp.zeros((V.shape[0], n_x, n_x))
        dfdu = jnp.zeros((V.shape[0], n_x, n_u))

        # Ensure x_seq and u have the same batch size
        x = V[:,:self.params.sim.n_states]
        u = u[:x.shape[0]]

        # Compute the nonlinear propagation term
        f = self.params.veh.state_dot(x, u[:,:-1])
        F = s[:, None] * f

        # Evaluate the State Jacobian
        dfdx = self.params.veh.A(x, u[:,:-1])
        sdfdx = s[:, None, None] * dfdx

        # Evaluate the Control Jacobian
        dfdu_veh = self.params.veh.B(x, u[:,:-1])
        dfdu = dfdu.at[:, :, :-1].set(s[:, None, None] * dfdu_veh)
        dfdu = dfdu.at[:, :, -1].set(f)
        
        # Compute the defect
        z = F - jnp.einsum('ijk,ik->ij', sdfdx, x) - jnp.einsum('ijk,ik->ij', dfdu, u)

        # Stack up the results into the augmented state vector
        dVdt = jnp.zeros_like(V)
        dVdt = dVdt.at[:, self.i0:self.i1].set(F)
        dVdt = dVdt.at[:, self.i1:self.i2].set(jnp.matmul(sdfdx, V[:, self.i1:self.i2].reshape(-1, n_x, n_x)).reshape(-1, n_x * n_x))
        dVdt = dVdt.at[:, self.i2:self.i3].set((jnp.matmul(sdfdx, V[:, self.i2:self.i3].reshape(-1, n_x, n_u)) + dfdu * alpha).reshape(-1, n_x * n_u))
        dVdt = dVdt.at[:, self.i3:self.i4].set((jnp.matmul(sdfdx, V[:, self.i3:self.i4].reshape(-1, n_x, n_u)) + dfdu * beta).reshape(-1, n_x * n_u))
        dVdt = dVdt.at[:, self.i4:self.i5].set((jnp.matmul(sdfdx, V[:, self.i4:self.i5].reshape(-1, n_x)[..., None]).squeeze(-1) + z).reshape(-1, n_x))
        return dVdt.flatten()
    
    def prop_aug_dy(self,
                    tau: float,
                    x: np.ndarray,
                    u_current: np.ndarray,
                    u_next: np.ndarray,
                    tau_init: float,
                    idx_s: int) -> np.ndarray:
        x = x[None, :]
        
        if self.params.scp.dis_type == "ZOH":
            beta = 0.0
        elif self.params.scp.dis_type == "FOH":
            beta = (tau - tau_init) * self.params.scp.n
        u = u_current + beta * (u_next - u_current)
        
        return  u[:, idx_s] * self.params.veh.state_dot(x, u[:,:-1]).squeeze()