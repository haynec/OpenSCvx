from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from openscvx.discretization.linearize_discretize import LinearizeDiscretize
from openscvx.integrators import solve_ivp_diffrax, solve_ivp_rk45

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered import Dynamics


class LinearizeDiscretizeSparse(LinearizeDiscretize):
    """Sparse variant of linearize-then-discretize.

    Uses graph-coloring-based sparse Jacobian computation and a compact
    augmented state vector that only integrates the structurally nonzero
    entries of Φ, B_d and C_d.  This reduces the ODE system dimension
    from ``n_x + n_x² + 2·n_x·n_u`` to ``n_x + nnz_Ad + nnz_Bd + nnz_Cd``
    per segment.

    Requires ``A_c_sparsity`` and ``B_c_sparsity`` boolean arrays on the
    :class:`Dynamics` object (set automatically when using the symbolic
    problem interface).  Falls back to the dense
    :class:`LinearizeDiscretize` path when sparsity patterns are
    unavailable or fully dense.

    Args:
        dis_type: Control hold type. ``"FOH"`` or ``"ZOH"``.
            Defaults to ``"FOH"``.
        ode_solver: Diffrax solver name. Defaults to ``"Tsit5"``.
        custom_integrator: Use built-in fixed-step RK45 instead of Diffrax.
            Defaults to ``False``.
        atol: Absolute tolerance for the ODE solver. Defaults to ``1e-3``.
        rtol: Relative tolerance for the ODE solver. Defaults to ``1e-6``.
        args: Extra keyword arguments forwarded to
            :func:`diffrax.diffeqsolve`. Defaults to ``{}``.
    """

    def get_solver(self, dynamics: "Dynamics", settings: "Config") -> callable:
        """Create a sparse multi-shoot discretization solver.

        When ``dynamics.A_c_sparsity`` and ``dynamics.B_c_sparsity`` are
        available and the pattern is not fully dense, builds a compact-V
        integration path with graph-coloring Jacobians.  Otherwise
        delegates to the dense parent implementation.

        Args:
            dynamics: System dynamics with optional sparsity annotations.
            settings: Problem configuration.

        Returns:
            Callable ``(x, u, params) -> (A_d, B_d, C_d, x_prop, V)``.
        """
        from openscvx.sparse import make_sparse_jacobian_fns
        from openscvx.symbolic.sparsity import discrete_sparsity

        A_c_pat = getattr(dynamics, "A_c_sparsity", None)
        B_c_pat = getattr(dynamics, "B_c_sparsity", None)
        has_sparsity = A_c_pat is not None and B_c_pat is not None

        if not has_sparsity or A_c_pat.all():
            return super().get_solver(dynamics, settings)

        f_vmapped = jax.vmap(dynamics.f, in_axes=(0, 0, 0, None))
        discretizer = self
        n_x = settings.sim.n_states
        n_u = settings.sim.n_controls

        A_vmapped, B_vmapped = make_sparse_jacobian_fns(
            dynamics.f,
            A_c_pat,
            B_c_pat,
            n_x,
            n_u,
        )

        Ad_pat, Bd_pat, Cd_pat = discrete_sparsity(
            A_c_pat,
            B_c_pat,
            self.dis_type,
        )
        Ad_r, Ad_c = np.where(Ad_pat)
        Bd_r, Bd_c = np.where(Bd_pat)
        Cd_r, Cd_c = np.where(Cd_pat)

        sparse_layout = (
            jnp.array(Ad_r),
            jnp.array(Ad_c),
            len(Ad_r),
            jnp.array(Bd_r),
            jnp.array(Bd_c),
            len(Bd_r),
            jnp.array(Cd_r),
            jnp.array(Cd_c),
            len(Cd_r),
        )

        return lambda x, u, params: _calculate_discretization_sparse(
            x=x,
            u=u,
            state_dot=f_vmapped,
            A=A_vmapped,
            B=B_vmapped,
            settings=settings,
            discretizer=discretizer,
            params=params,
            sparse_layout=sparse_layout,
        )


def _dVdt_sparse(
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
    Ad_rows: jnp.ndarray,
    Ad_cols: jnp.ndarray,
    nnz_Ad: int,
    Bd_rows: jnp.ndarray,
    Bd_cols: jnp.ndarray,
    nnz_Bd: int,
    Cd_rows: jnp.ndarray,
    Cd_cols: jnp.ndarray,
    nnz_Cd: int,
) -> jnp.ndarray:
    """Time derivative of the *compact* augmented state for sparse variational integration.

    Instead of storing the full flattened Φ, B_d, C_d matrices (``n_x²`` and
    ``n_x·n_u`` entries each), this function only tracks the structurally
    nonzero entries as determined by ``discrete_sparsity``.

    The compact layout per segment is::

        V = [x(n_x), Φ_nz(nnz_Ad), B_d_nz(nnz_Bd), C_d_nz(nnz_Cd)]

    At each evaluation the compact values are scattered into dense matrices
    for the matmul, and the derivative is gathered back at the nonzero
    positions.

    Args:
        tau: Normalized time in [0, 1] within the current segment.
        V: Flattened compact augmented state, shape
            ``((N-1) * (n_x + nnz_Ad + nnz_Bd + nnz_Cd),)``.
        Ad_rows, Ad_cols: Row/column indices of A_d structural nonzeros.
        nnz_Ad: Number of A_d structural nonzeros.
        Bd_rows, Bd_cols: Row/column indices of B_d structural nonzeros.
        nnz_Bd: Number of B_d structural nonzeros.
        Cd_rows, Cd_cols: Row/column indices of C_d structural nonzeros.
        nnz_Cd: Number of C_d structural nonzeros.
        (remaining args identical to :func:`_dVdt`)

    Returns:
        Flattened time derivative of the compact augmented state.
    """
    nodes = jnp.arange(0, N - 1)

    aug_dim = n_x + nnz_Ad + nnz_Bd + nnz_Cd
    i_phi = n_x
    i_bd = n_x + nnz_Ad
    i_cd = n_x + nnz_Ad + nnz_Bd

    V = V.reshape(-1, aug_dim)

    if dis_type == "ZOH":
        beta = 0.0
    elif dis_type == "FOH":
        beta = tau * N
    alpha = 1 - beta

    u = u_cur + beta * (u_next - u_cur)
    x = V[:, :n_x]
    u = u[: x.shape[0]]
    batch = x.shape[0]

    F = state_dot(x, u, nodes, params)
    dfdx = A(x, u, nodes, params)
    dfdu = B(x, u, nodes, params)

    # Φ: scatter compact → dense, matmul, gather back
    phi_nz = V[:, i_phi:i_bd]
    Phi = jnp.zeros((batch, n_x, n_x)).at[:, Ad_rows, Ad_cols].set(phi_nz)
    dPhi_nz = jnp.matmul(dfdx, Phi)[:, Ad_rows, Ad_cols]

    # B_d: scatter, matmul + forcing, gather
    bd_nz = V[:, i_bd:i_cd]
    Bd = jnp.zeros((batch, n_x, n_u)).at[:, Bd_rows, Bd_cols].set(bd_nz)
    dBd_nz = (jnp.matmul(dfdx, Bd) + dfdu * alpha)[:, Bd_rows, Bd_cols]

    # C_d: scatter, matmul + forcing, gather
    cd_nz = V[:, i_cd:]
    Cd = jnp.zeros((batch, n_x, n_u)).at[:, Cd_rows, Cd_cols].set(cd_nz)
    dCd_nz = (jnp.matmul(dfdx, Cd) + dfdu * beta)[:, Cd_rows, Cd_cols]

    return jnp.concatenate([F, dPhi_nz, dBd_nz, dCd_nz], axis=-1).reshape(-1)


def _calculate_discretization_sparse(
    x: np.ndarray,
    u: np.ndarray,
    state_dot: callable,
    A: callable,
    B: callable,
    settings: "Config",
    discretizer: "LinearizeDiscretizeSparse",
    params: dict,
    sparse_layout: tuple,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Integrate the compact variational equations to produce discrete-time matrices.

    Uses the structurally nonzero entries of Φ, B_d and C_d to form a
    reduced augmented state, integrates with :func:`_dVdt_sparse`, then
    scatters the results back into dense matrices for downstream
    compatibility.

    Args:
        x: Reference state trajectory, shape ``(N, n_x)``.
        u: Reference control trajectory, shape ``(N, n_u)``.
        state_dot: Vmapped time-dilated dynamics.
        A: Vmapped state Jacobian ``∂F/∂x``.
        B: Vmapped control Jacobian ``∂F/∂u``.
        settings: Problem configuration.
        discretizer: Discretizer instance with integrator settings.
        params: Parameters forwarded to dynamics callables.
        sparse_layout: Tuple
            ``(Ad_rows, Ad_cols, nnz_Ad, Bd_rows, Bd_cols, nnz_Bd,
            Cd_rows, Cd_cols, nnz_Cd)``.

    Returns:
        Tuple ``(A_d, B_d, C_d, x_prop, V)``.
    """
    n_x = settings.sim.n_states
    n_u = settings.sim.n_controls
    N = settings.sim.n

    (
        Ad_rows,
        Ad_cols,
        nnz_Ad,
        Bd_rows,
        Bd_cols,
        nnz_Bd,
        Cd_rows,
        Cd_cols,
        nnz_Cd,
    ) = sparse_layout

    aug_dim = n_x + nnz_Ad + nnz_Bd + nnz_Cd

    V0 = jnp.zeros((N - 1, aug_dim))
    V0 = V0.at[:, :n_x].set(x[:-1].astype(float))
    phi0_nz = (Ad_rows == Ad_cols).astype(x.dtype)
    V0 = V0.at[:, n_x : n_x + nnz_Ad].set(jnp.broadcast_to(phi0_nz[None], (N - 1, nnz_Ad)))
    V0 = V0.reshape(-1)

    integrator_args = dict(
        u_cur=u[:-1].astype(float),
        u_next=u[1:].astype(float),
        state_dot=state_dot,
        A=A,
        B=B,
        n_x=n_x,
        n_u=n_u,
        N=N,
        dis_type=discretizer.dis_type,
        S_x=settings.sim.S_x,
        c_x=settings.sim.c_x,
        S_u=settings.sim.S_u,
        c_u=settings.sim.c_u,
        inv_S_x=settings.sim.inv_S_x,
        inv_S_u=settings.sim.inv_S_u,
        params=params,
        Ad_rows=Ad_rows,
        Ad_cols=Ad_cols,
        nnz_Ad=nnz_Ad,
        Bd_rows=Bd_rows,
        Bd_cols=Bd_cols,
        nnz_Bd=nnz_Bd,
        Cd_rows=Cd_rows,
        Cd_cols=Cd_cols,
        nnz_Cd=nnz_Cd,
    )

    def dVdt_wrapped(t, y):
        return _dVdt_sparse(t, y, **integrator_args)

    if discretizer.custom_integrator:
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
            solver_name=discretizer.ode_solver,
            rtol=discretizer.rtol,
            atol=discretizer.atol,
            args=(),
            extra_kwargs=discretizer.args,
        )

    Vend = sol[-1].T.reshape(-1, aug_dim)

    x_prop = Vend[:, :n_x]
    phi_nz = Vend[:, n_x : n_x + nnz_Ad]
    bd_nz = Vend[:, n_x + nnz_Ad : n_x + nnz_Ad + nnz_Bd]
    cd_nz = Vend[:, n_x + nnz_Ad + nnz_Bd :]

    A_bar = jnp.zeros((N - 1, n_x, n_x)).at[:, Ad_rows, Ad_cols].set(phi_nz)
    B_bar = jnp.zeros((N - 1, n_x, n_u)).at[:, Bd_rows, Bd_cols].set(bd_nz)
    C_bar = jnp.zeros((N - 1, n_x, n_u))
    if nnz_Cd > 0:
        C_bar = C_bar.at[:, Cd_rows, Cd_cols].set(cd_nz)

    # Reconstruct a dense-layout Vmulti so that
    # DiscretizationResult.from_V can unpack it unchanged.
    Vend_dense = jnp.concatenate(
        [
            x_prop,
            A_bar.reshape(N - 1, n_x * n_x),
            B_bar.reshape(N - 1, n_x * n_u),
            C_bar.reshape(N - 1, n_x * n_u),
        ],
        axis=-1,
    )
    Vmulti = Vend_dense.reshape(-1, 1)

    return A_bar, B_bar, C_bar, x_prop, Vmulti
