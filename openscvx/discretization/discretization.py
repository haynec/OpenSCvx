import jax
import jax.numpy as jnp
import numpy as np

from openscvx.config import Config
from openscvx.discretization.base import Discretizer
from openscvx.integrators import solve_ivp_diffrax, solve_ivp_rk45
from openscvx.lowered import Dynamics


def dVdt(
    tau: float,
    V: jnp.ndarray,
    u_cur: np.ndarray,
    u_next: np.ndarray,
    state_dot: callable,
    A: callable,
    B: callable,
    n_x: int,
    n_u: int,
    N: int,
    dis_type: str,
    S_x: np.ndarray,
    c_x: np.ndarray,
    S_u: np.ndarray,
    c_u: np.ndarray,
    inv_S_x: np.ndarray,
    inv_S_u: np.ndarray,
    params: dict,
) -> jnp.ndarray:
    """Compute the time derivative of the augmented state vector.

    This function computes the time derivative of the augmented state vector V,
    which includes the state, state transition matrix, and control influence matrix.

    Args:
        tau (float): Current normalized time in [0,1].
        V (jnp.ndarray): Augmented state vector.
        u_cur (np.ndarray): Control input at current node.
        u_next (np.ndarray): Control input at next node.
        state_dot (callable): Function computing state derivatives.
        A (callable): Function computing state Jacobian.
        B (callable): Function computing control Jacobian.
        n_x (int): Number of states.
        n_u (int): Number of controls.
        N (int): Number of nodes in trajectory.
        dis_type (str): Discretization type ("ZOH" or "FOH").
        S_x: State scaling matrix.
        c_x: State offset vector.
        S_u: Control scaling matrix.
        c_u: Control offset vector.
        inv_S_x: Inverse state scaling matrix.
        inv_S_u: Inverse control scaling matrix.
        params: Additional parameters passed to state_dot, A, and B.

    Returns:
        jnp.ndarray: Time derivative of augmented state vector.
    """

    # TODO Implement scaling of V vector

    # Define the nodes
    nodes = jnp.arange(0, N - 1)

    # Define indices for slicing the augmented state vector
    i0 = 0
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u

    # Unflatten V
    V = V.reshape(-1, i4)

    # Compute the interpolation factor based on the discretization type
    if dis_type == "ZOH":
        beta = 0.0
    elif dis_type == "FOH":
        beta = (tau) * N
    alpha = 1 - beta

    # Interpolate the control input
    u = u_cur + beta * (u_next - u_cur)
    s = u[:, -1]

    # Initialize the augmented Jacobians
    dfdx = jnp.zeros((V.shape[0], n_x, n_x))
    dfdu = jnp.zeros((V.shape[0], n_x, n_u))

    # Ensure x_seq and u have the same batch size
    x = V[:, :n_x]
    u = u[: x.shape[0]]

    # Compute the nonlinear propagation term
    f = state_dot(x, u[:, :-1], nodes, params)
    F = s[:, None] * f

    # Evaluate the State Jacobian
    dfdx = A(x, u[:, :-1], nodes, params)
    sdfdx = s[:, None, None] * dfdx

    # Evaluate the Control Jacobian
    dfdu_veh = B(x, u[:, :-1], nodes, params)
    dfdu = dfdu.at[:, :, :-1].set(s[:, None, None] * dfdu_veh)
    dfdu = dfdu.at[:, :, -1].set(f)

    # Stack up the results into the augmented state vector
    # fmt: off
    dVdt = jnp.zeros_like(V)
    dVdt = dVdt.at[:, i0:i1].set(F)
    dVdt = dVdt.at[:, i1:i2].set(
        jnp.matmul(sdfdx, V[:, i1:i2].reshape(-1, n_x, n_x)).reshape(-1, n_x * n_x)
    )
    dVdt = dVdt.at[:, i2:i3].set(
        (jnp.matmul(sdfdx, V[:, i2:i3].reshape(-1, n_x, n_u)) + dfdu * alpha).reshape(-1, n_x * n_u)
    )
    dVdt = dVdt.at[:, i3:i4].set(
        (jnp.matmul(sdfdx, V[:, i3:i4].reshape(-1, n_x, n_u)) + dfdu * beta).reshape(-1, n_x * n_u)
    )
    # fmt: on

    # TODO Implement scaling of V vector

    return dVdt.reshape(-1)


def calculate_discretization(
    x: np.ndarray,
    u: np.ndarray,
    state_dot: callable,
    A: callable,
    B: callable,
    settings: Config,
    params: dict,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate the discretized system matrices.

    This function computes the discretized system matrices (A_bar, B_bar, C_bar)
    and defect vector (z_bar) using numerical integration.

    Args:
        x: State trajectory.
        u: Control trajectory.
        state_dot (callable): Function computing state derivatives.
        A (callable): Function computing state Jacobian.
        B (callable): Function computing control Jacobian.
        settings: Configuration settings for OpenSCvx.
        params: Additional parameters passed to state_dot, A, and B.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            (A_bar, B_bar, C_bar, x_prop, Vmulti) where:
            - A_bar: Discretized state transition matrix
            - B_bar: Discretized control influence matrix
            - C_bar: Discretized control influence matrix for next node
            - x_prop: Propagated state
            - Vmulti: Full augmented state trajectory
    """
    # Unpack settings
    n_x = settings.sim.n_states
    n_u = settings.sim.n_controls

    N = settings.scp.n

    # Define indices for slicing the augmented state vector
    i0 = 0
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u

    # Initial augmented state
    V0 = jnp.zeros((N - 1, i4))
    V0 = V0.at[:, :n_x].set(x[:-1].astype(float))
    V0 = V0.at[:, n_x : n_x + n_x * n_x].set(jnp.eye(n_x).reshape(1, -1).repeat(N - 1, axis=0))
    V0 = V0.reshape(-1)

    # TODO Implement scaling of V vector

    # Choose integrator
    integrator_args = dict(
        u_cur=u[:-1].astype(float),
        u_next=u[1:].astype(float),
        state_dot=state_dot,
        A=A,
        B=B,
        n_x=n_x,
        n_u=n_u,
        N=N,
        dis_type=settings.dis.dis_type,
        S_x=settings.sim.S_x,
        c_x=settings.sim.c_x,
        S_u=settings.sim.S_u,
        c_u=settings.sim.c_u,
        inv_S_x=settings.sim.inv_S_x,
        inv_S_u=settings.sim.inv_S_u,
        params=params,  # Pass params as single dict
    )

    # Define dVdt wrapper using named arguments
    def dVdt_wrapped(t, y):
        return dVdt(t, y, **integrator_args)

    # Choose integrator
    if settings.dis.custom_integrator:
        sol = solve_ivp_rk45(
            dVdt_wrapped,
            1.0 / (N - 1),
            V0,
            args=(),
            is_not_compiled=settings.dev.debug,
        )
    else:
        sol = solve_ivp_diffrax(
            dVdt_wrapped,
            1.0 / (N - 1),
            V0,
            solver_name=settings.dis.solver,
            rtol=settings.dis.rtol,
            atol=settings.dis.atol,
            args=(),
            extra_kwargs=settings.dis.args,
        )

    Vend = sol[-1].T.reshape(-1, i4)
    Vmulti = sol.T

    x_prop = Vend[:, i0:i1]

    # Return as 3D arrays: (N-1, n_x, n_x) for A_bar, (N-1, n_x, n_u) for B_bar/C_bar
    A_bar = Vend[:, i1:i2].reshape(N - 1, n_x, n_x)
    B_bar = Vend[:, i2:i3].reshape(N - 1, n_x, n_u)
    C_bar = Vend[:, i3:i4].reshape(N - 1, n_x, n_u)

    return A_bar, B_bar, C_bar, x_prop, Vmulti


class MultiShootDiscretizer(Discretizer):
    """Linearize-then-discretize via augmented ODE integration.

    Computes continuous-time Jacobians (df/dx, df/du) via JAX forward-mode
    autodiff, then integrates them alongside the nonlinear dynamics through
    an augmented state vector to produce discrete-time matrices.

    Supports ZOH (zero-order hold) and FOH (first-order hold) control
    interpolation between nodes, configurable via ``settings.dis.dis_type``.

    This is the default discretization scheme in OpenSCvx.
    """

    def get_solver(self, dynamics: Dynamics, settings: Config) -> callable:
        """Create a multi-shoot discretization solver.

        Computes Jacobians of ``dynamics.f`` via ``jax.jacfwd``, vmaps all
        functions for batch evaluation across nodes, and returns a callable
        that integrates the augmented variational equations.

        Args:
            dynamics: System dynamics. Only ``dynamics.f`` is used; Jacobians
                are computed internally via JAX autodiff.
            settings: Problem configuration.

        Returns:
            Callable ``(x, u, params) -> (A_d, B_d, C_d, x_prop, V)``.
        """
        # Compute continuous-time Jacobians from dynamics.f
        A_fn = jax.jacfwd(dynamics.f, argnums=0)
        B_fn = jax.jacfwd(dynamics.f, argnums=1)

        # Vmap for batch evaluation across nodes
        f_vmapped = jax.vmap(dynamics.f, in_axes=(0, 0, 0, None))
        A_vmapped = jax.vmap(A_fn, in_axes=(0, 0, 0, None))
        B_vmapped = jax.vmap(B_fn, in_axes=(0, 0, 0, None))

        return lambda x, u, params: calculate_discretization(
            x=x,
            u=u,
            state_dot=f_vmapped,
            A=A_vmapped,
            B=B_vmapped,
            settings=settings,
            params=params,
        )


def get_discretization_solver(dyn: Dynamics, settings: Config) -> callable:
    """Create a discretization solver function.

    .. deprecated::
        Use :class:`MultiShootDiscretizer` instead. This function is kept
        for backward compatibility.

    Args:
        dyn (Dynamics): System dynamics object (must have f, A, B).
        settings (Config): Configuration settings for discretization.

    Returns:
        callable: A function that computes the discretized system matrices.
    """
    return lambda x, u, params: calculate_discretization(
        x=x,
        u=u,
        state_dot=dyn.f,
        A=dyn.A,
        B=dyn.B,
        settings=settings,
        params=params,
    )
