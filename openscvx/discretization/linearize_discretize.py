from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from openscvx.discretization.base import Discretizer, _resolve_foh_mask
from openscvx.integrators import (
    DEFAULT_DIFFRAX_ATOL,
    DEFAULT_DIFFRAX_RTOL,
    solve_ivp_diffrax,
    solve_ivp_rk45,
)
from openscvx.lowered.stm_meta import StmMeta

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered import Dynamics


class LinearizeDiscretize(Discretizer):
    """Linearize-then-discretize via augmented ODE integration.

    Computes continuous-time Jacobians (df/dx, df/du) via JAX forward-mode
    autodiff, then integrates them alongside the nonlinear dynamics through
    an augmented state vector using a multi-shooting approach to produce
    discrete-time matrices.

    Supports ZOH (zero-order hold) and FOH (first-order hold) control
    interpolation between nodes.

    This is the default discretization scheme in OpenSCvx.

    Args:
        dis_type: Control hold type. ``"FOH"`` (first-order hold) or
            ``"ZOH"`` (zero-order hold) applies the same hold to every
            control.  A per-control sequence (e.g. ``["FOH", "ZOH", "FOH"]``)
            sets the hold independently for each control, merged with any
            per-control ``parameterization`` (``"FOH"`` / ``"ZOH"``).
            Defaults to ``"FOH"``.
        ode_solver: Diffrax solver name. Any solver from
            `Diffrax <https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/>`_
            is valid. Defaults to ``"Tsit5"``.
        diffrax_kwargs: Preferred Diffrax keyword overrides. These map to
            :func:`openscvx.integrators.solve_ivp_diffrax` kwargs, and unknown
            keys are forwarded to :func:`diffrax.diffeqsolve` (e.g.
            ``stepsize_controller``). Set ``rtol``/``atol`` here when using
            the default PID controller. Defaults to ``{}``.
    """

    def __init__(
        self,
        dis_type: Union[str, Sequence[str]] = "FOH",
        ode_solver: str = "Tsit5",
        diffrax_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.dis_type = dis_type
        self.ode_solver = ode_solver
        self.diffrax_kwargs = dict(diffrax_kwargs) if diffrax_kwargs is not None else {}

    def _resolve_diffrax_kwargs(self) -> dict[str, Any]:
        """Build kwargs for :func:`solve_ivp_diffrax`.

        Unknown keys from ``self.diffrax_kwargs`` are forwarded to
        :func:`diffrax.diffeqsolve` via ``extra_kwargs`` so users can pass
        objects like ``stepsize_controller`` directly from discretizer settings.
        """
        kwargs: dict[str, Any] = {
            "solver_name": self.ode_solver,
            "rtol": DEFAULT_DIFFRAX_RTOL,
            "atol": DEFAULT_DIFFRAX_ATOL,
        }

        user_kwargs = dict(self.diffrax_kwargs)
        extra_kwargs: dict[str, Any] = {}

        nested_extra = user_kwargs.pop("extra_kwargs", None)
        if nested_extra is not None:
            extra_kwargs.update(dict(nested_extra))

        direct_keys = {"tau_0", "num_substeps", "solver_name", "rtol", "atol"}
        for key, value in user_kwargs.items():
            if key in direct_keys:
                kwargs[key] = value
            else:
                extra_kwargs[key] = value

        kwargs["extra_kwargs"] = extra_kwargs
        return kwargs

    def _resolve_rk45_kwargs(self, *, is_not_compiled: bool) -> dict[str, Any]:
        """Build kwargs for :func:`solve_ivp_rk45`."""
        kwargs: dict[str, Any] = {"is_not_compiled": is_not_compiled}
        direct_keys = {"tau_0", "num_substeps", "is_not_compiled"}
        for key, value in self.diffrax_kwargs.items():
            if key in direct_keys:
                kwargs[key] = value
        return kwargs

    def get_solver(self, dynamics: "Dynamics", settings: "Config") -> callable:
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

        # Capture discretizer settings for the returned closure
        discretizer = self

        stm_meta = getattr(dynamics, "stm_meta", None) or StmMeta()

        # Hessian H_ilm = ∂²f_i/∂x_l ∂x_m only needed for exact-mode Ψ integration.
        H_vmapped = None
        if stm_meta.has_exact:
            H_fn = jax.jacfwd(A_fn, argnums=0)
            H_vmapped = jax.vmap(H_fn, in_axes=(0, 0, 0, None))

        return lambda x, u, params: _calculate_discretization(
            x=x,
            u=u,
            state_dot=f_vmapped,
            A=A_vmapped,
            B=B_vmapped,
            H=H_vmapped,
            settings=settings,
            discretizer=discretizer,
            params=params,
            stm_meta=stm_meta,
        )

    def citation(self) -> List[str]:
        """Return BibTeX citations for the linearize-then-discretize discretization method.

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


def _dVdt(
    tau: float,
    V: jnp.ndarray,
    u_cur: np.ndarray,
    u_next: np.ndarray,
    state_dot: callable,
    A: callable,
    B: callable,
    H: Optional[callable],
    n_x: int,
    n_u: int,
    N: int,
    foh_mask: np.ndarray,
    S_x: np.ndarray,
    c_x: np.ndarray,
    S_u: np.ndarray,
    c_u: np.ndarray,
    inv_S_x: np.ndarray,
    inv_S_u: np.ndarray,
    params: dict,
    stm_meta: StmMeta,
) -> jnp.ndarray:
    """Time derivative of the augmented state vector for variational integration.

    The augmented state ``V`` packs four blocks per segment:

    - ``V[0:n_x]``          — state x
    - ``V[n_x:n_x+n_x²]``  — state transition matrix Φ (flattened)
    - ``V[...:+n_x·n_u]``   — control influence B_d for current node (flattened)
    - ``V[...:+n_x·n_u]``   — control influence C_d for next node (flattened)

    Their derivatives follow from the variational equations:

    - ``dx/dτ     = F(x, u)``          where F = s · f(x, u) is the time-dilated dynamics
    - ``dΦ/dτ     = A(x, u) · Φ``
    - ``dB_d/dτ   = A(x, u) · B_d  +  α · B(x, u)``
    - ``dC_d/dτ   = A(x, u) · C_d  +  β · B(x, u)``

    where ``A = ∂F/∂x`` and ``B = ∂F/∂u`` are Jacobians of the
    time-dilated dynamics (which include the time-dilation factor ``s``
    symbolically), and ``α, β`` are per-control interpolation weights
    determined by the hold type (ZOH: α=1, β=0; FOH: linear blend).

    Args:
        tau: Normalized time in [0, 1] within the current segment.
        V: Flattened augmented state vector, shape ``((N-1) * aug_dim,)``.
        u_cur: Control at current node, shape ``(N-1, n_u)``.
        u_next: Control at next node, shape ``(N-1, n_u)``.
        state_dot: Vmapped time-dilated dynamics ``F(x, u, node, params) -> x_dot``.
        A: Vmapped state Jacobian ``∂F/∂x(x, u, node, params)``.
        B: Vmapped control Jacobian ``∂F/∂u(x, u, node, params)``.
        n_x: Number of states.
        n_u: Number of controls (including time-dilation).
        N: Number of trajectory nodes.
        foh_mask: Float array of shape ``(n_u,)`` — ``1.0`` for FOH controls,
            ``0.0`` for ZOH controls.
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
    # Ψ region (exact-mode only). Lives past the B_d/C_d blocks.
    i5 = i4 + stm_meta.psi_size

    # Unflatten V
    V = V.reshape(-1, i5)

    # Per-control interpolation weights: beta_i = tau*N for FOH, 0 for ZOH
    beta = tau * N * foh_mask
    alpha = 1 - beta

    # Interpolate the control input
    u = u_cur + beta * (u_next - u_cur)

    # Ensure x_seq and u have the same batch size
    x = V[:, :n_x]
    u = u[: x.shape[0]]

    # Compute the time-dilated dynamics (s * f already included symbolically)
    F = state_dot(x, u, nodes, params)

    # Evaluate the Jacobians (already include time-dilation derivatives via autodiff)
    dfdx = A(x, u, nodes, params)
    dfdu = B(x, u, nodes, params)

    # Stack up the results into the augmented state vector
    # fmt: off
    dVdt = jnp.zeros_like(V)
    dVdt = dVdt.at[:, i0:i1].set(F)
    # Inject STM variational RHS (physical-state block of A only; one-way invariant).
    if not stm_meta.is_empty:
        n_phys = stm_meta.n_phys
        A_phys = dfdx[:, :n_phys, :n_phys]
        for slot in stm_meta.slots:
            block = V[:, slot.slice]
            if slot.kind == "physical":
                phi = block.reshape(-1, n_phys, n_phys)
                dphi = jnp.matmul(A_phys, phi).reshape(-1, n_phys * n_phys)
            else:  # "impulse"
                dphi = jnp.einsum("nij,nj->ni", A_phys, block)
            dVdt = dVdt.at[:, slot.slice].set(dphi)
    dVdt = dVdt.at[:, i1:i2].set(
        jnp.matmul(dfdx, V[:, i1:i2].reshape(-1, n_x, n_x)).reshape(-1, n_x * n_x)
    )
    dVdt = dVdt.at[:, i2:i3].set(
        (jnp.matmul(dfdx, V[:, i2:i3].reshape(-1, n_x, n_u)) + dfdu * alpha).reshape(-1, n_x * n_u)
    )
    dVdt = dVdt.at[:, i3:i4].set(
        (jnp.matmul(dfdx, V[:, i3:i4].reshape(-1, n_x, n_u)) + dfdu * beta).reshape(-1, n_x * n_u)
    )
    # Ψ variational RHS (exact-mode physical slots):
    #   dΨ_ijk/dτ = A_il · Ψ_ljk + H_ilm · Φ_mk · Φ_lj
    # where H_ilm = ∂²f_i/∂x_l ∂x_m; all indices range over [0, n_phys).
    if stm_meta.psi_size > 0 and H is not None:
        n_phys = stm_meta.n_phys
        A_phys = dfdx[:, :n_phys, :n_phys]
        H_phys = H(x, u, nodes, params)[:, :n_phys, :n_phys, :n_phys]
        for slot in stm_meta.slots:
            if slot.psi_slice is None or slot.kind != "physical":
                continue
            psi_abs = slice(i4 + slot.psi_slice.start, i4 + slot.psi_slice.stop)
            phi = V[:, slot.slice].reshape(-1, n_phys, n_phys)
            psi = V[:, psi_abs].reshape(-1, n_phys, n_phys, n_phys)
            dpsi = jnp.einsum("bil,bljk->bijk", A_phys, psi) + jnp.einsum(
                "bilm,bmk,blj->bijk", H_phys, phi, phi
            )
            dVdt = dVdt.at[:, psi_abs].set(dpsi.reshape(-1, n_phys ** 3))
    # fmt: on

    # TODO Implement scaling of V vector

    return dVdt.reshape(-1)


def _calculate_discretization(
    x: np.ndarray,
    u: np.ndarray,
    state_dot: callable,
    A: callable,
    B: callable,
    H: Optional[callable],
    settings: "Config",
    discretizer: "LinearizeDiscretize",
    params: dict,
    stm_meta: StmMeta,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Integrate the augmented variational equations to produce discrete-time matrices.

    Initializes the augmented state vector (state + identity Φ + zero B_d/C_d)
    at each segment, then integrates ``_dVdt`` from node k to node k+1 using the
    configured ODE solver. The final values of the augmented state yield the
    discretized linearization matrices.

    Args:
        x: Reference state trajectory, shape ``(N, n_x)``.
        u: Reference control trajectory, shape ``(N, n_u)`` (includes
            time-dilation as part of the unified control vector).
        state_dot: Vmapped time-dilated dynamics ``F(x, u, node, params) -> x_dot``.
        A: Vmapped state Jacobian ``∂F/∂x``.
        B: Vmapped control Jacobian ``∂F/∂u``.
        settings: Problem configuration (node count, scaling matrices, debug flag).
        discretizer: Discretizer instance holding integrator settings (hold type,
            solver, tolerances, etc.).
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

    N = settings.sim.n

    # Define indices for slicing the augmented state vector
    i0 = 0
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u
    i5 = i4 + stm_meta.psi_size  # Ψ region (exact-mode only; 0 otherwise)

    # Initial augmented state
    V0 = jnp.zeros((N - 1, i5))
    V0 = V0.at[:, :n_x].set(x[:-1].astype(float))
    # Reset STM slots at each segment start.
    #   - physical (anchor_node=None): identity reset on every segment row.
    #   - physical (anchor_node=j): identity injected only on row j; on every
    #     other row keep V0[k, slot.slice] = x[k, slot.slice] (the reference
    #     trajectory's chained Φ value that the SCP iterates toward).
    #   - impulse: zero-reset every segment.
    if not stm_meta.is_empty:
        n_phys = stm_meta.n_phys
        eye_flat = jnp.eye(n_phys).reshape(-1)
        for slot in stm_meta.slots:
            if slot.kind == "physical":
                if slot.anchor_node is None:
                    V0 = V0.at[:, slot.slice].set(
                        jnp.broadcast_to(eye_flat, (N - 1, n_phys * n_phys))
                    )
                else:
                    j = slot.anchor_node
                    if 0 <= j < N - 1:
                        V0 = V0.at[j, slot.slice].set(eye_flat)
                    # rows k != j already carry x[k, slot.slice] from the
                    # initial copy of x[:-1] above — leave them as-is.
            else:
                V0 = V0.at[:, slot.slice].set(0.0)
    V0 = V0.at[:, n_x : n_x + n_x * n_x].set(jnp.eye(n_x).reshape(1, -1).repeat(N - 1, axis=0))
    V0 = V0.reshape(-1)

    # TODO Implement scaling of V vector

    u_foh_mask = getattr(settings.sim.u, "foh_mask", None)
    foh_mask = _resolve_foh_mask(discretizer.dis_type, n_u, u_foh_mask)

    # Choose integrator
    integrator_args = dict(
        u_cur=u[:-1].astype(float),
        u_next=u[1:].astype(float),
        state_dot=state_dot,
        A=A,
        B=B,
        H=H,
        n_x=n_x,
        n_u=n_u,
        N=N,
        foh_mask=foh_mask,
        S_x=settings.sim.S_x,
        c_x=settings.sim.c_x,
        S_u=settings.sim.S_u,
        c_u=settings.sim.c_u,
        inv_S_x=settings.sim.inv_S_x,
        inv_S_u=settings.sim.inv_S_u,
        params=params,  # Pass params as single dict
        stm_meta=stm_meta,
    )

    # Define dVdt wrapper using named arguments
    def dVdt_wrapped(t, y):
        return _dVdt(t, y, **integrator_args)

    if settings.dev.debug:
        rk45_kwargs = discretizer._resolve_rk45_kwargs(is_not_compiled=settings.dev.debug)
        sol = solve_ivp_rk45(
            dVdt_wrapped,
            1.0 / (N - 1),
            V0,
            args=(),
            **rk45_kwargs,
        )
    else:
        diffrax_kwargs = discretizer._resolve_diffrax_kwargs()
        sol = solve_ivp_diffrax(
            dVdt_wrapped,
            1.0 / (N - 1),
            V0,
            args=(),
            **diffrax_kwargs,
        )

    Vend = sol[-1].T.reshape(-1, i5)

    x_prop = Vend[:, i0:i1]

    # Return as 3D arrays: (N-1, n_x, n_x) for A_bar, (N-1, n_x, n_u) for B_bar/C_bar
    A_bar = Vend[:, i1:i2].reshape(N - 1, n_x, n_x)
    B_bar = Vend[:, i2:i3].reshape(N - 1, n_x, n_u)
    C_bar = Vend[:, i3:i4].reshape(N - 1, n_x, n_u)

    # Exact-mode STM rows of A_bar come from Ψ. Φ resets to I at each segment
    # start, so ∂Φ(τ_{k+1})/∂Φ(τ_k) = 0 (zero whole row first) and
    # ∂Φ(τ_{k+1})/∂x_phys(τ_k) = Ψ (flatten (i,j) → row index within slot).
    if stm_meta.psi_size > 0:
        n_phys = stm_meta.n_phys
        for slot in stm_meta.slots:
            if slot.psi_slice is None or slot.kind != "physical":
                continue
            psi_abs = slice(i4 + slot.psi_slice.start, i4 + slot.psi_slice.stop)
            psi_end = Vend[:, psi_abs].reshape(N - 1, n_phys * n_phys, n_phys)
            A_bar = A_bar.at[:, slot.slice, :].set(0.0)
            A_bar = A_bar.at[:, slot.slice, :n_phys].set(psi_end)

    # Strip Ψ from V before returning; downstream consumers expect i4 per segment.
    # Patch the final column's Φ block with the Ψ-corrected A_bar rows so that
    # ``DiscretizationResult.from_V`` yields the exact-mode linearization.
    sol_mat = sol.reshape(sol.shape[0], N - 1, i5)[:, :, :i4]
    if stm_meta.psi_size > 0:
        sol_mat = sol_mat.at[-1, :, i1:i2].set(A_bar.reshape(N - 1, n_x * n_x))
    Vmulti = sol_mat.reshape(sol.shape[0], (N - 1) * i4).T

    return A_bar, B_bar, C_bar, x_prop, Vmulti


def calculate_impulsive_discretization(
    x_nodes: np.ndarray,
    u_nodes: np.ndarray,
    state_dot_discrete: callable,
    A_discrete: callable,
    B_discrete: callable,
    params: dict,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate discrete/impulsive dynamics and Jacobians at all nodes."""
    n_nodes = x_nodes.shape[0]
    nodes = jnp.arange(n_nodes)
    n_x = x_nodes.shape[1]
    n_u = u_nodes.shape[1]

    x_prop_plus = state_dot_discrete(x_nodes, u_nodes, nodes, params)
    D_d = A_discrete(x_nodes, u_nodes, nodes, params)
    E_d = B_discrete(x_nodes, u_nodes, nodes, params)

    W_end = jnp.concatenate(
        (
            x_prop_plus,
            D_d.reshape(n_nodes, n_x * n_x),
            E_d.reshape(n_nodes, n_x * n_u),
        ),
        axis=1,
    )
    W = W_end.reshape(-1, 1)
    return x_prop_plus, D_d, E_d, W


def get_impulsive_discretization_solver(dyn_discrete: "Dynamics") -> callable:
    """Create a solver for discrete/impulsive dynamics linearization."""
    A_fn = jax.jacfwd(dyn_discrete.f, argnums=0)
    B_fn = jax.jacfwd(dyn_discrete.f, argnums=1)

    f_vmapped = jax.vmap(dyn_discrete.f, in_axes=(0, 0, 0, None))
    A_vmapped = jax.vmap(A_fn, in_axes=(0, 0, 0, None))
    B_vmapped = jax.vmap(B_fn, in_axes=(0, 0, 0, None))

    return lambda x_nodes, u_nodes, params: calculate_impulsive_discretization(
        x_nodes=x_nodes,
        u_nodes=u_nodes,
        state_dot_discrete=f_vmapped,
        A_discrete=A_vmapped,
        B_discrete=B_vmapped,
        params=params,
    )
