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


class VectorizeDiscretizeLinearize(Discretizer):
    """Discretization via differentiating through the integrator for all segments simultaneously.

    Propagates the nonlinear dynamics over all trajectory segments at once, then directly computes
    Jacobians of the propagated solutions (dF/dx, dF/du) via JAX forward-mode autodiff to produce
    discrete-time Jacobians.

    Supports ZOH (zero-order hold) and FOH (first-order hold) control interpolation between nodes.

    Use this integration scheme when the nonlinear dynamics are challenging
    (e.g. stiff/sensitive, badly scaled, or with long time horizons) or when
    tight tolerances are desired.

    Args:
        dis_type: Control hold type. ``"FOH"`` (first-order hold) or ``"ZOH"`` (zero-order hold).
            Defaults to ``"FOH"``.
        ode_solver: Diffrax solver name. Any solver from
            `Diffrax <https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/>`_
            is valid. Defaults to ``"Tsit5"``.
        custom_integrator: Use the built-in fixed-step RK45 integrator instead of Diffrax.
            Faster but less robust. Defaults to ``False``.
        atol: Absolute tolerance of the ODE solver for all segments combined. Defaults to ``1e-3``.
        rtol: Relative tolerance of the ODE solver for all segments combined. Defaults to ``1e-6``.
        args: Extra keyword arguments forwarded to :func:`diffrax.diffeqsolve`. Defaults to ``{}``.
    """

    def __init__(
        self,
        dis_type: str = "FOH",
        ode_solver: str = "Tsit5",
        custom_integrator: bool = False,
        atol: float = 1e-3,
        rtol: float = 1e-6,
        args: Optional[dict] = None,
    ):
        self.dis_type = dis_type
        self.ode_solver = ode_solver
        self.custom_integrator = custom_integrator
        self.atol_combined = atol
        self.rtol_combined = rtol
        self.extra_kwargs = args | {"adjoint": dfx.ForwardMode()} if args is not None else {"adjoint": dfx.ForwardMode()}

    def get_solver(self, dynamics: "Dynamics", settings: "Config") -> callable:
        """Create a multiple-shooting vectorize-then-discretize-then-linearize solver.

        Batches ``dynamics.f`` across all nodes, integrates it, and computes Jacobians of the
        solution directly (i.e. without using the variational equations).

        Args:
            dynamics: System dynamics. Only ``dynamics.f`` is used; Jacobians are computed
                internally via JAX autodiff.
            settings: Problem configuration.

        Returns:
            Callable ``(x, u, params) -> (A_d, B_d, C_d, x_prop, V)``.
        """

        N = settings.sim.n
        n_x = settings.sim.n_states
        n_u = settings.sim.n_controls
        nodes = jnp.arange(0, N - 1)

        # Provided tolerances are for state and all three Jacobians combined, but only state will
        # be integrated. Tighten tolerances by factor of (state size) / (state and Jacobians size)
        # to make up for this
        rtol = self.rtol_combined / (1 + n_x + 2*n_u)
        atol = self.atol_combined / (1 + n_x + 2*n_u)

        multiple_state_dot = jax.vmap(dynamics.f, in_axes=(0, 0, 0, None))

        def multiple_dxdt(
            tau: float,
            x: jnp.ndarray,
            u_cur: np.ndarray,
            u_next: np.ndarray,
            params: dict,
        ) -> jnp.ndarray:

            if self.dis_type == "ZOH":
                beta = 0.0
            elif self.dis_type == "FOH":
                beta = (tau) * N

            x = x.reshape(N - 1, n_x)
            u = u_cur + beta * (u_next - u_cur)
            F = multiple_state_dot(x, u, nodes, params)

            return F.flatten()

        def vectorize_then_discretize(
            x: jnp.ndarray,
            u_cur: np.ndarray,
            u_next: np.ndarray,
            params: dict,
        ) -> jnp.ndarray:

            if self.custom_integrator:
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
                    solver_name=self.ode_solver,
                    rtol=rtol,
                    atol=atol,
                    args=(u_cur, u_next, params),
                    extra_kwargs=self.extra_kwargs,
                )
            return sol.reshape(-1, N - 1, n_x)

        def vectorize_then_discretize_then_linearize(
            x: jnp.ndarray,
            u_cur: np.ndarray,
            u_next: np.ndarray,
            params: dict,
        ):

            partial_in_x = lambda x : vectorize_then_discretize(x, u_cur, u_next, params)
            partial_in_u_cur = lambda u_cur : vectorize_then_discretize(x, u_cur, u_next, params)
            partial_in_u_next = lambda u_next : vectorize_then_discretize(x, u_cur, u_next, params)

            mapped_jvp = lambda f, primal : jax.vmap(lambda tangent : jax.jvp(f, (primal,), (tangent,))[1], in_axes=0, out_axes=-1)  # zeroth output of jvp is function value, which can be discarded

            x_tangents = jnp.repeat(jnp.eye(n_x)[:, None, :], N - 1, axis=1)  # stack of (repeats over all nodes of (standard basis vectors of state))
            u_tangents = jnp.repeat(jnp.eye(n_u)[:, None, :], N - 1, axis=1)  # stack of (repeats over all nodes of (standard basis vectors of control))

            A_d = mapped_jvp(partial_in_x, x)(x_tangents)
            B_d = mapped_jvp(partial_in_u_cur, u_cur)(u_tangents)
            C_d = mapped_jvp(partial_in_u_next, u_next)(u_tangents)

            return A_d, B_d, C_d

        def solver(x, u, params):
            A_d, B_d, C_d = vectorize_then_discretize_then_linearize(x[:-1], u[:-1], u[1:], params)
            x_prop = vectorize_then_discretize(x[:-1], u[:-1], u[1:], params)

            V_multi = jnp.concatenate([x_prop, A_d.reshape(-1, N-1, n_x*n_x), B_d.reshape(-1, N-1, n_x*n_u), C_d.reshape(-1, N-1, n_x*n_u)], axis=2)
            i4 = V_multi.shape[2]
            V_multi = V_multi.reshape(-1, (N-1)*i4).T

            return A_d[-1], B_d[-1], C_d[-1], x_prop[-1], V_multi

        return solver

    def citation(self) -> List[str]:
        return []


class DiscretizeLinearizeVectorize(Discretizer):
    """Discretization via differentiating through the integrator for each segment individually.

    Propagates the nonlinear dynamics for each trajectory segment on its own, then directly computes
    Jacobians of the propagated solutions (dF/dx, dF/du) via JAX forward-mode autodiff to produce
    discrete-time Jacobians for each node.

    Supports ZOH (zero-order hold) and FOH (first-order hold) control interpolation between nodes.

    More accurate than ``VectorizeDiscretizeLinearize``, but also more expensive. Use this
    integration scheme when the nonlinear dynamics are challenging (e.g. stiff/sensitive,
    badly scaled, or long time horizons) or when very tight tolerances are desired.

    Args:
        dis_type: Control hold type. ``"FOH"`` (first-order hold) or ``"ZOH"`` (zero-order hold).
            Defaults to ``"FOH"``.
        ode_solver: Diffrax solver name. Any solver from
            `Diffrax <https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/>`_
            is valid. Defaults to ``"Tsit5"``.
        custom_integrator: Use the built-in fixed-step RK45 integrator instead of Diffrax.
            Faster but less robust. Defaults to ``False``.
        atol: Absolute tolerance of the ODE solver for all segments combined. Defaults to ``1e-3``.
        rtol: Relative tolerance of the ODE solver for all segments combined. Defaults to ``1e-6``.
        args: Extra keyword arguments forwarded to :func:`diffrax.diffeqsolve`. Defaults to ``{}``.
    """

    def __init__(
        self,
        dis_type: str = "FOH",
        ode_solver: str = "Tsit5",
        custom_integrator: bool = False,
        atol: float = 1e-3,
        rtol: float = 1e-6,
        args: Optional[dict] = None,
    ):
        self.dis_type = dis_type
        self.ode_solver = ode_solver
        self.custom_integrator = custom_integrator
        self.atol_combined = atol
        self.rtol_combined = rtol
        self.extra_kwargs = args | {"adjoint": dfx.ForwardMode()} if args is not None else {"adjoint": dfx.ForwardMode()}

    def get_solver(self, dynamics: "Dynamics", settings: "Config") -> callable:
        """Create a multiple-shooting discretize-then-linearize-then-vectorize solver.

        Integrates ``dynamics.f`` and computes Jacobians of the discretized function directly
        (i.e. without using the variational equations). Outputs are vmapped for batch evaluation
        across nodes.

        Args:
            dynamics: System dynamics. Only ``dynamics.f`` is used; Jacobians are computed
                internally via JAX autodiff.
            settings: Problem configuration.

        Returns:
            Callable ``(x, u, params) -> (A_d, B_d, C_d, x_prop, V)``.
        """

        N = settings.sim.n
        n_x = settings.sim.n_states
        n_u = settings.sim.n_controls

        # Provided tolerances are for state and all three Jacobians for all nodes combined, but only
        # state will be integrated and on only one segment. Tighten tolerances by factor of
        # (state size) / (state and Jacobians size for all nodes) to make up for this
        rtol = self.rtol_combined / (N - 1) / (1 + n_x + 2*n_u)
        atol = self.atol_combined / (N - 1) / (1 + n_x + 2*n_u)

        single_state_dot = dynamics.f

        def single_dxdt(
            tau: float,
            x: jnp.ndarray,
            u_cur: np.ndarray,
            u_next: np.ndarray,
            node: int,
            params: dict,
        ) -> jnp.ndarray:

            if self.dis_type == "ZOH":
                beta = 0.0
            elif self.dis_type == "FOH":
                beta = (tau) * N

            u = u_cur + beta * (u_next - u_cur)
            F = single_state_dot(x, u, node, params)

            return F

        def single_shot(
            x: jnp.ndarray,
            u_cur: np.ndarray,
            u_next: np.ndarray,
            node: int,
            params: dict,
        ) -> jnp.ndarray:

            if self.custom_integrator:
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
                    solver_name=self.ode_solver,
                    rtol=rtol,
                    atol=atol,
                    args=(u_cur, u_next, node, params),
                    extra_kwargs=self.extra_kwargs,
                )
            return sol

        discretize_then_vectorize = jax.vmap(single_shot, in_axes=(0, 0, 0, 0, None), out_axes=1)
        discretize_then_linearize = jax.jacfwd(single_shot, argnums=(0, 1, 2))
        discretize_then_linearize_then_vectorize = jax.vmap(discretize_then_linearize, in_axes=(0, 0, 0, 0, None), out_axes=1)

        nodes = jnp.arange(0, N - 1)

        def solver(x, u, params):
            A_d, B_d, C_d = discretize_then_linearize_then_vectorize(x[:-1], u[:-1], u[1:], nodes, params)
            x_prop = discretize_then_vectorize(x[:-1], u[:-1], u[1:], nodes, params)

            V_multi = jnp.concatenate([x_prop, A_d.reshape(-1, N-1, n_x*n_x), B_d.reshape(-1, N-1, n_x*n_u), C_d.reshape(-1, N-1, n_x*n_u)], axis=2)
            i4 = V_multi.shape[2]
            V_multi = V_multi.reshape(-1, (N-1)*i4).T

            return A_d[-1], B_d[-1], C_d[-1], x_prop[-1], V_multi

        return solver

    def citation(self) -> List[str]:
        return []
