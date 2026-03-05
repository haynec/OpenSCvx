from typing import TYPE_CHECKING, List, Optional

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np

from openscvx.discretization.base import Discretizer
from openscvx.integrators import solve_ivp_diffrax, solve_ivp_rk45

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered import Dynamics


class DiscretizeLinearize(Discretizer):
    """Discretize-then-linearize by differentiating through the integrator.

    Integrates the nonlinear dynamics over a batch of trajectory segments,
    then computes Jacobians of the integrated solutions (dF/dx, dF/du) via
    JAX forward-mode to produce discrete-time Jacobians.

    Supports ZOH (zero-order hold) and FOH (first-order hold) control
    interpolation between nodes.

    Use this integration scheme when the nonlinear dynamics are challenging
    (e.g. stiff/sensitive, badly scaled, or with long time horizons) or when
    tight tolerances are desired.

    Args:
        dis_type: Control hold type. ``"FOH"`` (first-order hold) or
            ``"ZOH"`` (zero-order hold). Defaults to ``"FOH"``.
        ode_solver: Diffrax solver name. Any solver from
            `Diffrax <https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/>`_
            is valid. Defaults to ``"Tsit5"``.
        custom_integrator: Use the built-in fixed-step RK45 integrator
            instead of Diffrax. Faster but less robust. Defaults to ``False``.
        atol: Absolute tolerance for the ODE solver. Defaults to ``1e-3``.
        rtol: Relative tolerance for the ODE solver. Defaults to ``1e-6``.
        vectorize_last: Integrate and linearize the dynamics before batching
            across nodes. Slower but more accurate. Defaults to ``False``.
        args: Extra keyword arguments forwarded to
            :func:`diffrax.diffeqsolve`. Defaults to ``{}``.
    """

    def __init__(
        self,
        dis_type: str = "FOH",
        ode_solver: str = "Tsit5",
        custom_integrator: bool = False,
        atol: float = 1e-3,
        rtol: float = 1e-6,
        vectorize_last = False,
        args: Optional[dict] = None,
    ):
        self.dis_type = dis_type
        self.ode_solver = ode_solver
        self.custom_integrator = custom_integrator
        self.atol = atol
        self.rtol = rtol
        self.vectorize_last = vectorize_last
        self.args = args | {"adjoint": dfx.ForwardMode()} if args is not None else {"adjoint": dfx.ForwardMode()}

    def get_solver(self, dynamics: "Dynamics", settings: "Config") -> callable:
        """Create a multi-shoot discretization solver.

        Integrates ``dynamics.f`` and computes Jacobians of the discretized
        function directly, without using the variational equations. Outputs are
        vmapped for batch evaluation across nodes.

        Args:
            dynamics: System dynamics. Only ``dynamics.f`` is used; Jacobians
                are computed internally via JAX autodiff.
            settings: Problem configuration.

        Returns:
            Callable ``(x, u, params) -> (A_d, B_d, C_d, x_prop, V)``.
        """

        discretization = get_discretize_then_vectorize_solver if self.vectorize_last else get_vectorize_then_discretize_solver

        # Capture discretizer settings for the returned closure
        discretizer = self

        return discretization(
            state_dot=dynamics.f,
            settings=settings,
            discretizer=discretizer,
        )

    def citation(self) -> List[str]:
        return []


def get_discretize_then_vectorize_solver(
    state_dot: callable,
    settings: "Config",
    discretizer: "DiscretizeLinearize",
):

    N = settings.sim.n
    n_x = settings.sim.n_states
    n_u = settings.sim.n_controls

    def single_dxdt(
        tau: float,
        x: jnp.ndarray,
        u_cur: np.ndarray,
        u_next: np.ndarray,
        node: int,
        params: dict,
    ) -> jnp.ndarray:

        # Compute the interpolation factor based on the discretization type
        if discretizer.dis_type == "ZOH":
            beta = 0.0
        elif discretizer.dis_type == "FOH":
            beta = (tau) * N

        # Interpolate the control input
        u = u_cur + beta * (u_next - u_cur)

        # Compute the nonlinear propagation term
        F = state_dot(x, u, node, params)

        return F

    def single_shot(
        x: jnp.ndarray,
        u_cur: np.ndarray,
        u_next: np.ndarray,
        node: int,
        params: dict,
    ) -> jnp.ndarray:

        if discretizer.custom_integrator:
            sol = solve_ivp_rk45(
                single_dxdt,
                1.0 / (N - 1),
                x,
                args=(u_cur, u_next, node, params),
                is_not_compiled=settings.dev.debug,
            )
        else:
            sol = solve_ivp_diffrax(
                single_dxdt,
                1.0 / (N - 1),
                x,
                solver_name=discretizer.ode_solver,
                rtol=discretizer.rtol,
                atol=discretizer.atol,
                args=(u_cur, u_next, node, params),
                extra_kwargs=discretizer.args,
            )
        return sol[-1]

    discretize_then_vectorize = jax.vmap(single_shot, in_axes=(0, 0, 0, 0, None))
    discretize_then_linearize = jax.jacfwd(single_shot, argnums=(0, 1, 2))
    discretize_then_linearize_then_vectorize = jax.vmap(discretize_then_linearize, in_axes=(0, 0, 0, 0, None))

    nodes = jnp.arange(0, N - 1)

    def solver(x, u, params):
        A_d, B_d, C_d = discretize_then_linearize_then_vectorize(x[:-1], u[:-1], u[1:], nodes, params)
        x_prop = discretize_then_vectorize(x[:-1], u[:-1], u[1:], nodes, params)

        V_multi = jnp.concatenate([x_prop, A_d.reshape(N-1, n_x*n_x), B_d.reshape(N-1, n_x*n_u), C_d.reshape(N-1, n_x*n_u)], axis=1).reshape(-1, 1)  # TODO: return full nonlinear propagation of state

        return A_d, B_d, C_d, x_prop, V_multi

    return solver


def get_vectorize_then_discretize_solver(
    state_dot: callable,
    settings: "Config",
    discretizer: "DiscretizeLinearize",
):

    N = settings.sim.n
    n_x = settings.sim.n_states
    n_u = settings.sim.n_controls
    nodes = jnp.arange(0, N - 1)

    multiple_state_dot = jax.vmap(state_dot, in_axes=(0, 0, 0, None))

    def multiple_dxdt(
        tau: float,
        x: jnp.ndarray,
        u_cur: np.ndarray,
        u_next: np.ndarray,
        params: dict,
    ) -> jnp.ndarray:

        # Compute the interpolation factor based on the discretization type
        if discretizer.dis_type == "ZOH":
            beta = 0.0
        elif discretizer.dis_type == "FOH":
            beta = (tau) * N

        x = x.reshape(N - 1, n_x)

        # Interpolate the control input
        u = u_cur + beta * (u_next - u_cur)

        # Compute the nonlinear propagation term
        F = multiple_state_dot(x, u, nodes, params)

        return F.flatten()

    def vectorize_then_discretize(
        x: jnp.ndarray,
        u_cur: np.ndarray,
        u_next: np.ndarray,
        params: dict,
    ) -> jnp.ndarray:
        """
        Propagates all segments of a multiple-shooting trajectory.

        Parameters:
            x (jax.Array, shape (N-1, n_x)): Reference value of the vehicle state at the start of each
                segment. Here N is the number of nodes and n_x is the number of states.
            u_cur (jax.Array, shape (N-1, n_u)): Reference value of the vehicle control at the start of
                each segment. Here m is the number of controls.
            u_next (jax.Array, shape(N-1, n_u)): Reference value of the vehicle control at the end of
                each segment.
            params: Parameters forwarded to ``state_dot``.

        Returns:
            x_prop (jax.Array, shape (N-1, n_x)): Stack of end states of each propagated segment.
        """

        if discretizer.custom_integrator:
            sol = solve_ivp_rk45(
                multiple_dxdt,
                1.0 / (N - 1),
                x.flatten(),
                args=(u_cur, u_next, params),
                is_not_compiled=settings.dev.debug,
            )
        else:
            sol = solve_ivp_diffrax(
                multiple_dxdt,
                1.0 / (N - 1),
                x.flatten(),
                solver_name=discretizer.ode_solver,
                rtol=discretizer.rtol,
                atol=discretizer.atol,
                args=(u_cur, u_next, params),
                extra_kwargs=discretizer.args,
            )
        return sol[-1].reshape(N - 1, n_x)

    def get_propagation_history(
        x: jnp.ndarray,
        u_cur: np.ndarray,
        u_next: np.ndarray,
        params: dict,
    ) -> jnp.ndarray:
        sol = solve_ivp_diffrax(
            multiple_dxdt,
            1.0 / (N - 1),
            x.flatten(),
            solver_name=discretizer.ode_solver,
            rtol=discretizer.rtol,
            atol=discretizer.atol,
            args=(u_cur, u_next, params),
            extra_kwargs=discretizer.args,
        )
        return sol.reshape(-1, N - 1, n_x)

    def vectorize_and_discretize_then_linearize(
        x: jnp.ndarray,
        u_cur: np.ndarray,
        u_next: np.ndarray,
        params: dict,
    ):

        partial_in_x = lambda x : vectorize_then_discretize(x, u_cur, u_next, params)
        partial_in_u_cur = lambda u_cur : vectorize_then_discretize(x, u_cur, u_next, params)
        partial_in_u_next = lambda u_next : vectorize_then_discretize(x, u_cur, u_next, params)

        mapped_jvp = lambda f, primal : jax.vmap(lambda tangent : jax.jvp(f, (primal,), (tangent,))[1], in_axes=0, out_axes=-1)  # zeroth output of jvp is function value, which can be discarded

        x_tangents = jnp.repeat(jnp.eye(n_x)[:, None, :], N - 1, axis=1)  # array of (repeated arrays of (one-hot vectors))
        u_tangents = jnp.repeat(jnp.eye(n_u)[:, None, :], N - 1, axis=1)

        A_d = mapped_jvp(partial_in_x, x)(x_tangents)
        B_d = mapped_jvp(partial_in_u_cur, u_cur)(u_tangents)
        C_d = mapped_jvp(partial_in_u_next, u_next)(u_tangents)

        return A_d, B_d, C_d

    def solver(x, u, params):
        A_d, B_d, C_d = vectorize_and_discretize_then_linearize(x[:-1], u[:-1], u[1:], params)
        x_prop = vectorize_then_discretize(x[:-1], u[:-1], u[1:], params)
        x_prop_history = get_propagation_history(x[:-1], u[:-1], u[1:], params)

        # TODO: This is a kludge. The V_multi output doesn't make sense for discretize-then-linearize, since the time
        # histories of the Jacobians are never constructed. In fact, nothing downstream even uses those time histories,
        # which is why I can get away with this. V_multi should be replaced with an output that only includes the
        # propagation history of the state. Furthermore, A_d, B_d, C_d, and x_prop may then be used instead of V_multi.
        # This is preferable because they are more descriptively named.
        i1 = n_x
        i2 = i1 + n_x * n_x
        i3 = i2 + n_x * n_u
        i4 = i3 + n_x * n_u
        V_multi = jnp.pad(x_prop_history, ((0, 0), (0, 0), (0, i4 - i1))).reshape(-1, (N - 1) * i4)
        V_multi = V_multi.at[-1].set(jnp.concatenate([x_prop, A_d.reshape(N-1, n_x*n_x), B_d.reshape(N-1, n_x*n_u), C_d.reshape(N-1, n_x*n_u)], axis=1).flatten())
        V_multi = V_multi.T
        # </kludge>

        return A_d, B_d, C_d, x_prop, V_multi

    return solver
