from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from openscvx.config import Config
from openscvx.discretization.base import Discretizer
from openscvx.integrators import solve_ivp_diffrax, solve_ivp_rk45
from openscvx.lowered import Dynamics


def _dVdt(
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
    """Time derivative of the augmented state vector for variational integration.

    The augmented state ``V`` packs four blocks per segment:

    - ``V[0:n_x]``          — state x
    - ``V[n_x:n_x+n_x²]``  — state transition matrix Φ (flattened)
    - ``V[...:+n_x·n_u]``   — control influence B_d for current node (flattened)
    - ``V[...:+n_x·n_u]``   — control influence C_d for next node (flattened)

    Their derivatives follow from the variational equations:

    - ``dx/dτ     = s · f(x, u)``
    - ``dΦ/dτ     = s · A(x, u) · Φ``
    - ``dB_d/dτ   = s · A(x, u) · B_d  +  α · s · B(x, u)``
    - ``dC_d/dτ   = s · A(x, u) · C_d  +  β · s · B(x, u)``

    where ``s`` is the time-dilation factor (last column of ``u``),
    ``A = ∂f/∂x``, ``B = ∂f/∂u``, and ``α, β`` are interpolation weights
    determined by the hold type (ZOH: α=1, β=0; FOH: linear blend).

    Args:
        tau: Normalized time in [0, 1] within the current segment.
        V: Flattened augmented state vector, shape ``((N-1) * aug_dim,)``.
        u_cur: Control at current node, shape ``(N-1, n_u+1)``.
        u_next: Control at next node, shape ``(N-1, n_u+1)``.
        state_dot: Vmapped dynamics ``f(x, u, node, params) -> x_dot``.
        A: Vmapped state Jacobian ``∂f/∂x(x, u, node, params)``.
        B: Vmapped control Jacobian ``∂f/∂u(x, u, node, params)``.
        n_x: Number of states.
        n_u: Number of controls (excluding time-dilation slack).
        N: Number of trajectory nodes.
        dis_type: ``"ZOH"`` (zero-order hold) or ``"FOH"`` (first-order hold).
        S_x: State scaling matrix (unused, reserved for future scaling).
        c_x: State offset vector (unused, reserved for future scaling).
        S_u: Control scaling matrix (unused, reserved for future scaling).
        c_u: Control offset vector (unused, reserved for future scaling).
        inv_S_x: Inverse state scaling matrix (unused, reserved for future scaling).
        inv_S_u: Inverse control scaling matrix (unused, reserved for future scaling).
        params: Parameters forwarded to ``state_dot``, ``A``, and ``B``.

    Returns:
        Flattened time derivative of the augmented state, same shape as ``V``.
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


def _calculate_discretization(
    x: np.ndarray,
    u: np.ndarray,
    state_dot: callable,
    A: callable,
    B: callable,
    settings: Config,
    params: dict,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Integrate the augmented variational equations to produce discrete-time matrices.

    Initializes the augmented state vector (state + identity Φ + zero B_d/C_d)
    at each segment, then integrates ``_dVdt`` from node k to node k+1 using the
    configured ODE solver. The final values of the augmented state yield the
    discretized linearization matrices.

    Args:
        x: Reference state trajectory, shape ``(N, n_x)``.
        u: Reference control trajectory, shape ``(N, n_u+1)`` (includes
            time-dilation slack as last column).
        state_dot: Vmapped dynamics ``f(x, u, node, params) -> x_dot``.
        A: Vmapped state Jacobian ``∂f/∂x``.
        B: Vmapped control Jacobian ``∂f/∂u``.
        settings: Configuration (integrator choice, tolerances, hold type, etc.).
        params: Parameters forwarded to ``state_dot``, ``A``, and ``B``.

    Returns:
        Tuple ``(A_d, B_d, C_d, x_prop, V)`` where:

        - ``A_d``: ``(N-1, n_x, n_x)`` discretized state transition matrix
        - ``B_d``: ``(N-1, n_x, n_u)`` control influence (current node)
        - ``C_d``: ``(N-1, n_x, n_u)`` control influence (next node)
        - ``x_prop``: ``(N-1, n_x)`` nonlinearly propagated state
        - ``V``: full augmented state trajectory from the integrator
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
        return _dVdt(t, y, **integrator_args)

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


class LinearizeDiscretize(Discretizer):
    """Linearize-then-discretize via augmented ODE integration.

    Computes continuous-time Jacobians (df/dx, df/du) via JAX forward-mode
    autodiff, then integrates them alongside the nonlinear dynamics through
    an augmented state vector using a multi-shooting approach to produce
    discrete-time matrices.

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

        return lambda x, u, params: _calculate_discretization(
            x=x,
            u=u,
            state_dot=f_vmapped,
            A=A_vmapped,
            B=B_vmapped,
            settings=settings,
            params=params,
        )

    def citation(self) -> List[str]:
        """Return BibTeX citations for the linearize-then-discretize algorithm.

        Returns:
            List containing the BibTeX entries
        """
        return [
            r"""@article{kamath2023real,
  title={Real-time sequential conic optimization for multi-phase rocket landing guidance},
  author={Kamath, Abhinav G and Elango, Purnanand and Yu, Yue and Mceowen, Skye and Chari, Govind M
    and Carson III, John M and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
  journal={IFAC-PapersOnLine},
  volume={56},
  number={2},
  pages={3118--3125},
  year={2023},
  publisher={Elsevier}
}""",
        ]
