import jax.numpy as jnp
from jax import jit, lax
import numpy as np
import diffrax  as dfx
import scipy.integrate as itg

from openscvx.config import Config

SOLVER_MAP = {
    "Tsit5": dfx.Tsit5,
    "Euler": dfx.Euler,
    "Heun": dfx.Heun,
    "Midpoint": dfx.Midpoint,
    "Ralston": dfx.Ralston,
    "Dopri5": dfx.Dopri5,
    "Dopri8": dfx.Dopri8,
    "Bosh3": dfx.Bosh3,
    "ReversibleHeun": dfx.ReversibleHeun,
    "ImplicitEuler": dfx.ImplicitEuler,
    "KenCarp3": dfx.KenCarp3,
    "KenCarp4": dfx.KenCarp4,
    "KenCarp5": dfx.KenCarp5
}

class RK45_Custom:
    def __init__(self):
        pass

    def rk45_step(self, f, t, y, h, *args):
        def compute_k(f, t, y, h, *args):
            k1 = f(t, y, *args)
            k2 = f(t + h/4, y + h*k1/4, *args)
            k3 = f(t + 3*h/8, y + 3*h*k1/32 + 9*h*k2/32, *args)
            k4 = f(t + 12*h/13, y + 1932*h*k1/2197 - 7200*h*k2/2197 + 7296*h*k3/2197, *args)
            k5 = f(t + h, y + 439*h*k1/216 - 8*h*k2 + 3680*h*k3/513 - 845*h*k4/4104, *args)
            return k1, k2, k3, k4, k5
        
        k1, k2, k3, k4, k5 = compute_k(f, t, y, h, *args)
        y_next = y + h * (25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
        return y_next

    def solve_ivp(self, dVdt, tau_grid, V0, args, method='RK45', t_eval=None):
        if method != 'RK45':
            raise ValueError("Currently, only 'RK45' method is supported.")
        
        if t_eval is None:
            t_eval = jnp.linspace(tau_grid[0], tau_grid[1], 50)
        
        h = (tau_grid[1] - tau_grid[0]) / (len(t_eval) - 1)
        V_result = jnp.zeros((len(t_eval), len(V0)))
        V_result = V_result.at[0].set(V0)
        
        def body_fun(i, val):
            t, y, V_result = val
            y_next = self.rk45_step(dVdt, t, y, h, *args)
            V_result = V_result.at[i].set(y_next)
            return (t + h, y_next, V_result)

        _, _, V_result = lax.fori_loop(1, len(t_eval), body_fun, (tau_grid[0], V0, V_result))
        
        return V_result


class Diffrax:
    def __init__(self, params):
        self.params = params
    
    def solve_ivp(self, dVdt, tau_grid, V0, args, t_eval=None):
        # if t_eval is None:
        t_eval = jnp.linspace(tau_grid[0], tau_grid[1], 50)

        solver_class = SOLVER_MAP.get(self.params.dis.diffrax_solver)
        if solver_class is None:
            raise ValueError(f"Unknown solver: {self.params.cvx.solver_name}")
        solver = solver_class()

        term = dfx.ODETerm(lambda t, y, args: dVdt(t, y, *args))
        stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-8)
        solution = dfx.diffeqsolve(
            term,
            solver = solver,
            t0=tau_grid[0],
            t1=tau_grid[1],
            dt0=(tau_grid[1] - tau_grid[0]) / (len(t_eval) - 1),
            y0=V0,
            args=args,
            stepsize_controller=stepsize_controller,
            saveat=dfx.SaveAt(ts=t_eval),
            **self.params.dis.diffrax_args
        )

        return solution.ys


class ExactDis:
    def __init__(self, params: Config) -> None:
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

        if self.params.dis.diffrax:
            self.integrator = Diffrax(self.params)
        else:
            self.integrator = RK45_Custom()

        self.tau_grid = jnp.linspace(0, 1, self.params.scp.n)

    def s_to_t(self, u, params: Config):
        t = [0]
        tau = np.linspace(0, 1, params.scp.n)
        for k in range(1, params.scp.n):
            s_kp = u[k-1,-1]
            s_k = u[k,-1]
            if params.dis.dis_type == 'ZOH':
                t.append(t[k-1] + (tau[k] - tau[k-1])*(s_kp))
            else:
                t.append(t[k-1] + 0.5 * (s_k + s_kp) * (tau[k] - tau[k-1]))
        return t
    
    def t_to_tau(self, u_lam, t, u_nodal, t_nodal, params: Config):
        u = np.array([u_lam(t_i) for t_i in t])

        tau = np.zeros(len(t))
        tau_nodal = np.linspace(0, 1, params.scp.n)
        for k in range(1, len(t)):
            k_nodal = np.where(t_nodal < t[k])[0][-1]
            s_kp = u_nodal[k_nodal, -1]
            tp = t_nodal[k_nodal]
            tau_p = tau_nodal[k_nodal]

            s_k = u[k, -1]
            if params.dis.dis_type == 'ZOH':
                tau[k] = tau_p + (t[k] - tp) / s_kp
            else:
                tau[k] = tau_p + 2 * (t[k] - tp) / (s_k + s_kp)
        return tau, u
    
    def calculate_discretization(self,
                                 x: jnp.ndarray,
                                 u: jnp.ndarray):
        """
        Calculate discretization for given states, inputs and total time.
        x: Matrix of states for all time points
        u: Matrix of inputs for all time points
        return: A_k, B_k, C_k, z_k
        """

        # Extract the number of states and controls from the parameters
        n_x = self.params.sim.n_states
        n_u = self.params.sim.n_controls
        
        if self.params.dis.custom_integrator:
            # Initialize the augmented state vector
            V0 = jnp.zeros((x.shape[0]-1, self.i5))

            # Vectorized integration
            V0 = V0.at[:, self.i0:self.i1].set(x[:-1, :].astype(float))
            V0 = V0.at[:, self.i1:self.i2].set(np.eye(n_x).reshape(1, n_x * n_x).repeat(self.params.scp.n - 1, axis=0))
            
            int_result = self.integrator.solve_ivp(self.dVdt, (self.tau_grid[0], self.tau_grid[1]), V0.flatten(), args=(u[:-1, :].astype(float), u[1:, :].astype(float)), t_eval=self.tau_grid)
            

            V = int_result[-1].T.reshape(-1, self.i5)
            V_multi_shoot = int_result.T
        else:
            V0 = np.zeros((x.shape[0]-1, self.i5))

            V0[:, self.i0:self.i1] = x[:-1, :].astype(float)
            V0[:, self.i1:self.i2] = np.eye(n_x).reshape(1, n_x * n_x).repeat(self.params.scp.n - 1, axis=0)

            int_result = itg.solve_ivp(self.dVdt, (self.tau_grid[0], self.tau_grid[1]), V0.flatten(), args=(u[:-1, :].astype(float), u[1:, :].astype(float)), method='RK45', t_eval=jnp.linspace(self.tau_grid[0], self.tau_grid[1], 50))
            V = int_result.y[:,-1].reshape(-1, self.i5)
            V_multi_shoot = int_result.y
    
        # Flatten matrices in column-major (Fortran) order for cvxpy
        A_bar = V[:, self.i1:self.i2].reshape((self.params.scp.n - 1, n_x, n_x)).transpose(1, 2, 0).reshape(n_x * n_x, -1, order='F').T
        B_bar = V[:, self.i2:self.i3].reshape((self.params.scp.n - 1, n_x, n_u)).transpose(1, 2, 0).reshape(n_x * n_u, -1, order='F').T
        C_bar = V[:, self.i3:self.i4].reshape((self.params.scp.n - 1, n_x, n_u)).transpose(1, 2, 0).reshape(n_x * n_u, -1, order='F').T
        z_bar = V[:, self.i4:self.i5]
        return A_bar, B_bar, C_bar, z_bar, V_multi_shoot

    def dVdt(self,
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
        if self.params.dis.dis_type == 'ZOH':
            beta = 0.
        elif self.params.dis.dis_type == 'FOH':
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
        f = self.params.dyn.state_dot(x, u[:,:-1])
        F = s[:, None] * f

        # Evaluate the State Jacobian
        dfdx = self.params.dyn.A(x, u[:,:-1])
        sdfdx = s[:, None, None] * dfdx

        # Evaluate the Control Jacobian
        dfdu_veh = self.params.dyn.B(x, u[:,:-1])
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
        
        if self.params.dis.dis_type == "ZOH":
            beta = 0.0
        elif self.params.dis.dis_type == "FOH":
            beta = (tau - tau_init) * self.params.scp.n
        u = u_current + beta * (u_next - u_current)
        
        return  u[:, idx_s] * self.params.dyn.state_dot(x, u[:,:-1]).squeeze()