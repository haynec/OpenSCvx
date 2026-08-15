---
title: Exact Discretization
description: >-
  How OpenSCvx discretizes and linearizes continuous-time dynamics by
  integrating an augmented state vector to build the ZOH/FOH transition matrices.
---

# Exact Discretization

``` py title="dVdt.py"
def dVdt(self, tau: float, V: jnp.ndarray, u_cur: np.ndarray, u_next: np.ndarray) -> jnp.ndarray:
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
    if self.dis_type == "ZOH":
        beta = 0.0
    elif self.dis_type == "FOH":
        beta = (tau) * self.params.sim.n
    alpha = 1 - beta

    # Interpolate the control input
    u = u_cur + beta * (u_next - u_cur)
    s = u[:, -1]

    # Initialize the augmented Jacobians
    dfdx = jnp.zeros((V.shape[0], n_x, n_x))
    dfdu = jnp.zeros((V.shape[0], n_x, n_u))

    # Ensure x_seq and u have the same batch size
    x = V[:, : self.params.sim.n_states]
    u = u[: x.shape[0]]

    # Compute the nonlinear propagation term
    f = self.params.dyn.state_dot(x, u[:, :-1])
    F = s[:, None] * f

    # Evaluate the State Jacobian
    dfdx = self.params.dyn.A(x, u[:, :-1])
    sdfdx = s[:, None, None] * dfdx

    # Evaluate the Control Jacobian
    dfdu_veh = self.params.dyn.B(x, u[:, :-1])
    dfdu = dfdu.at[:, :, :-1].set(s[:, None, None] * dfdu_veh)
    dfdu = dfdu.at[:, :, -1].set(f)

    # Compute the defect
    z = F - jnp.einsum("ijk,ik->ij", sdfdx, x) - jnp.einsum("ijk,ik->ij", dfdu, u)

    # Stack up the results into the augmented state vector
    dVdt = jnp.zeros_like(V)
    dVdt = dVdt.at[:, self.i0 : self.i1].set(F)
    dVdt = dVdt.at[:, self.i1 : self.i2].set(
        jnp.matmul(sdfdx, V[:, self.i1 : self.i2].reshape(-1, n_x, n_x)).reshape(-1, n_x * n_x)
    )
    dVdt = dVdt.at[:, self.i2 : self.i3].set(
        (jnp.matmul(sdfdx, V[:, self.i2 : self.i3].reshape(-1, n_x, n_u)) + dfdu * alpha).reshape(
            -1, n_x * n_u
        )
    )
    dVdt = dVdt.at[:, self.i3 : self.i4].set(
        (jnp.matmul(sdfdx, V[:, self.i3 : self.i4].reshape(-1, n_x, n_u)) + dfdu * beta).reshape(
            -1, n_x * n_u
        )
    )
    dVdt = dVdt.at[:, self.i4 : self.i5].set(
        (
            jnp.matmul(sdfdx, V[:, self.i4 : self.i5].reshape(-1, n_x)[..., None]).squeeze(-1) + z
        ).reshape(-1, n_x)
    )
    return dVdt.flatten()
```

## Supplying your own Jacobian

The state and control Jacobians above (`dfdx`, `dfdu`) come from automatic
differentiation of the dynamics you wrote. Occasionally the exact Jacobian is
not the one you want: a drag law, a contact model, or a lookup table can be
smooth enough to integrate while its true derivative makes the convex
subproblem badly conditioned, so the trust region collapses and the SCP loop
crawls. The standard remedy is to linearize with a deliberately simplified
(inexact) Jacobian.

`Expr.with_jacobian` applies that remedy to a single term:

``` py
drag = -0.5 * rho * ox.Norm(vel) * vel
J_drag = -0.5 * rho * ox.Norm(vel) * np.eye(3)  # drops the d||v||/dv term

dynamics = {"pos": vel, "vel": thrust / m + drag.with_jacobian({vel: J_drag})}
```

The keys are the `State` or `Control` objects the derivative is taken with
respect to, and each value is an expression (or constant array) of shape
`(*term.shape, *variable.shape)`. Directions you do not name are still
differentiated automatically, so the example above hands over `∂/∂vel` while
`∂/∂thrust` still comes from autodiff.

The *value* of the term is untouched — only the search direction changes — so a
converged trajectory still satisfies the original nonlinear dynamics; the
defect and the propagated solution are computed with the term as written. An
inexact Jacobian trades linearization accuracy (and hence convergence rate) for
conditioning, so reach for it when a term is a known source of stiffness, not
as a default.

The override lives inside the lowered JAX function, so every downstream
derivative picks it up: the discretizer's `A`/`B`, nonconvex constraint
linearization, and the sparsity pattern used for coloring. It has no meaning
inside a constraint that is handed to the convex solver as written, and lowering
one there raises.
