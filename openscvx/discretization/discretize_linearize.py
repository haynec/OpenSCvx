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

    def __init__(
        self,
        dis_type: str = "FOH",
        ode_solver: str = "Dopri8",
        custom_integrator: bool = False,
        atol: float = 1e-3,
        rtol: float = 1e-6,
        do_discretize_then_vectorize=False,
        args: Optional[dict] = None,
    ):
        self.dis_type = dis_type
        self.ode_solver = ode_solver
        self.custom_integrator = custom_integrator
        self.atol = atol
        self.rtol = rtol
        self.do_discretize_then_vectorize = do_discretize_then_vectorize
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

        discretization = get_discretize_then_vectorize_solver if self.do_discretize_then_vectorize else get_vectorize_then_discretize_solver

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
            sol = solve_ivp_rk45(single_dxdt,
                                 1.0 / (N - 1),
                                 x,
                                 args=(u_cur, u_next, node, params),
                                 is_not_compiled=settings.dev.debug,
            )
        else:
            sol = solve_ivp_diffrax(single_dxdt,
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
        A_bar, B_bar, C_bar = discretize_then_linearize_then_vectorize(x[:-1], u[:-1], u[1:], nodes, params)
        x_prop = discretize_then_vectorize(x[:-1], u[:-1], u[1:], nodes, params)

        V_multi = jnp.concatenate([x_prop, A_bar.reshape(N-1, n_x*n_x), B_bar.reshape(N-1, n_x*n_u), C_bar.reshape(N-1, n_x*n_u)], axis=1).reshape(-1, 1)

        return A_bar, B_bar, C_bar, x_prop, V_multi

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

        x = x.reshape(N - 1, -1)

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

        if discretizer.custom_integrator:
            sol = solve_ivp_rk45(multiple_dxdt,
                                 1.0 / (N - 1),
                                 x.flatten(),
                                 args=(u_cur, u_next, params),
                                 is_not_compiled=settings.dev.debug,
            )
        else:
            sol = solve_ivp_diffrax(multiple_dxdt,
                                    1.0 / (N - 1),
                                    x.flatten(),
                                    solver_name=discretizer.ode_solver,
                                    rtol=discretizer.rtol,
                                    atol=discretizer.atol,
                                    args=(u_cur, u_next, params),
                                    extra_kwargs=discretizer.args,
            )
        return sol[-1].reshape(N - 1, -1)

    def vectorize_and_discretize_then_linearize(
        x: jnp.ndarray,
        u_cur: np.ndarray,
        u_next: np.ndarray,
        params: dict,
    ):

        # TODO: Reduce code duplication using partial functions (jax.argnums_partial?)
        multiple_shot_partial_wrt_x = lambda x : vectorize_then_discretize(x, u_cur, u_next, params)
        multiple_shot_partial_wrt_u_cur = lambda u_cur : vectorize_then_discretize(x, u_cur, u_next, params)
        multiple_shot_partial_wrt_u_next = lambda u_next : vectorize_then_discretize(x, u_cur, u_next, params)

        multiple_shot_jvp_wrt_x = jax.vmap(lambda x_tangent : jax.jvp(multiple_shot_partial_wrt_x, (x,), (x_tangent,))[1], in_axes=0, out_axes=-1)  # zeroth output of jvp is function value, which can be discarded
        multiple_shot_jvp_wrt_u_cur = jax.vmap(lambda u_cur_tangent : jax.jvp(multiple_shot_partial_wrt_u_cur, (u_cur,), (u_cur_tangent,))[1], in_axes=0, out_axes=-1)
        multiple_shot_jvp_wrt_u_next = jax.vmap(lambda u_next_tangent : jax.jvp(multiple_shot_partial_wrt_u_next, (u_next,), (u_next_tangent,))[1], in_axes=0, out_axes=-1)

        x_tangents = jnp.tile(jnp.eye(n_x)[:, None, :], (1, N - 1, 1))
        u_tangents = jnp.tile(jnp.eye(n_u)[:, None, :], (1, N - 1, 1))

        A_bar = multiple_shot_jvp_wrt_x(x_tangents)
        B_bar = multiple_shot_jvp_wrt_u_cur(u_tangents)
        C_bar = multiple_shot_jvp_wrt_u_next(u_tangents)

        return A_bar, B_bar, C_bar

    def solver(x, u, params):
        A_bar, B_bar, C_bar = vectorize_and_discretize_then_linearize(x[:-1], u[:-1], u[1:], params)
        x_prop = vectorize_then_discretize(x[:-1], u[:-1], u[1:], params)

        V_multi = jnp.concatenate([x_prop, A_bar.reshape(N-1, n_x*n_x), B_bar.reshape(N-1, n_x*n_u), C_bar.reshape(N-1, n_x*n_u)], axis=1).reshape(-1, 1)

        return A_bar, B_bar, C_bar, x_prop, V_multi

    return solver
