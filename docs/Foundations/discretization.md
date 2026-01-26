# Exact Discretization

!!! Warning
    This page is still under development :construction:.

``` py title="dVdt.py"
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
```
    