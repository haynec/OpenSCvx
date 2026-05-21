from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np  # noqa: F401

from openscvx.config import Config
from openscvx.discretization import Discretizer
from openscvx.discretization.base import _resolve_foh_mask
from openscvx.integrators import solve_ivp_diffrax_prop
from openscvx.lowered import Dynamics
from openscvx.lowered.stm_meta import StmMeta


def _invoke_solver(solver: callable, *args):
    """Call either a compiled solver wrapper (.call) or a plain callable."""
    if hasattr(solver, "call"):
        return solver.call(*args)
    return solver(*args)


def _time_dilation_index(settings: Config, n_controls: int) -> int:
    """Return time-dilation control index, falling back to last control."""
    td_slice = getattr(settings.sim, "time_dilation_slice", None)
    if td_slice is None:
        return n_controls - 1
    return int(td_slice.start)


def _stm_carry_size(stm_meta: Optional[StmMeta]) -> int:
    if stm_meta is None or stm_meta.is_empty:
        return 0
    return max(slot.slice.stop for slot in stm_meta.slots)


def prop_aug_dy(
    tau: float,
    x: np.ndarray,
    u_current: np.ndarray,
    u_next: np.ndarray,
    tau_init: float,
    node: int,
    state_dot: callable,
    foh_mask: np.ndarray,
    N: int,
    params: dict,
    stm_meta: Optional[StmMeta] = None,
    A: Optional[callable] = None,
    n_states: Optional[int] = None,
) -> np.ndarray:
    """Compute the augmented dynamics for propagation.

    This function computes the time-dilated dynamics for propagating the system
    state, taking into account the per-control hold type (ZOH or FOH). The
    time-dilation multiplication is already included in ``state_dot``
    symbolically.

    Args:
        tau (float): Current normalized time in [0,1].
        x (np.ndarray): Current state vector.
        u_current (np.ndarray): Control input at current node.
        u_next (np.ndarray): Control input at next node.
        tau_init (float): Initial normalized time.
        node (int): Current node index.
        state_dot (callable): Function computing time-dilated state derivatives.
        foh_mask (np.ndarray): Float array of shape ``(n_u,)`` — ``1.0`` for
            FOH controls, ``0.0`` for ZOH controls.
        N (int): Number of nodes in trajectory.
        params: Dictionary of additional parameters passed to state_dot.

    Returns:
        np.ndarray: Time-dilated state derivatives.
    """
    stm_size = _stm_carry_size(stm_meta)
    if stm_size == 0:
        x_state = x[None, :]
        beta = (tau - tau_init) * N * foh_mask
        u = u_current + beta * (u_next - u_current)
        dx = state_dot(x_state, u, node, params)
        return dx.squeeze()

    # Augmented carry: x = [x_state, Φ_stm_block].
    if n_states is None:
        n_states = x.shape[0] - stm_size
    x_state = x[:n_states][None, :]
    phi_block = x[n_states:]
    n_phys = stm_meta.n_phys

    beta = (tau - tau_init) * N * foh_mask
    u = u_current + beta * (u_next - u_current)

    # Inject live Φ into params so the CTCS RHS sees the value being
    # integrated, not the constant per-node value from the prior pass.
    transient_phi = dict(params.get("__stm_phi__", {}))
    for slot in stm_meta.slots:
        block = phi_block[slot.slice]
        if slot.kind == "physical":
            phi_val = block.reshape(n_phys, n_phys)
        else:
            phi_val = block
        # Pad to (N, ...) to match the JIT trace signature used by the QP
        # context. ``node`` is shape (1, 1) so node_idx becomes a scalar; the
        # visitor's ``arr[node_idx]`` picks one row, so we replicate.
        if slot.kind == "physical":
            transient_phi[slot.name] = jnp.broadcast_to(
                phi_val[None, ...], (N, n_phys, n_phys)
            )
        else:
            transient_phi[slot.name] = jnp.broadcast_to(phi_val[None, ...], (N, n_phys))
    local_params = {**params, "__stm_phi__": transient_phi}

    dx_state = state_dot(x_state, u, node, local_params)

    # Φ variational RHS: dΦ/dτ = A_phys · Φ. A is vmapped; index batch=0.
    if A is None:
        dphi_block = jnp.zeros_like(phi_block)
    else:
        A_phys = A(x_state, u, node, local_params)[0, :n_phys, :n_phys]
        dphi_block = jnp.zeros_like(phi_block)
        for slot in stm_meta.slots:
            block = phi_block[slot.slice]
            if slot.kind == "physical":
                phi_val = block.reshape(n_phys, n_phys)
                dphi = (A_phys @ phi_val).reshape(-1)
            else:
                dphi = A_phys @ block
            dphi_block = dphi_block.at[slot.slice].set(dphi)

    return jnp.concatenate([dx_state.squeeze(), dphi_block])


def get_propagation_solver(
    state_dot: Dynamics,
    settings: Config,
    discretizer: Discretizer,
    stm_meta: Optional[StmMeta] = None,
    f_scalar: Optional[Callable] = None,
) -> callable:
    """Create a propagation solver function.

    This function creates a solver that propagates the system state using the
    specified dynamics and settings.

    Args:
        state_dot: Dynamics object containing state derivative function.
        settings: Configuration settings for propagation.
        discretizer: Discretizer instance (used for ``dis_type``).

    Returns:
        callable: A function that solves the propagation problem.
    """

    u_foh_mask = getattr(settings.sim.u, "foh_mask", None)
    foh_mask = _resolve_foh_mask(discretizer.dis_type, settings.sim.n_controls, u_foh_mask)

    A_vmapped: Optional[Callable] = None
    if stm_meta is not None and not stm_meta.is_empty and f_scalar is not None:
        A_vmapped = jax.vmap(jax.jacfwd(f_scalar, argnums=0), in_axes=(0, 0, 0, None))

    n_states_prop = settings.sim.n_states_prop

    def propagation_solver(V0, tau_grid, u_cur, u_next, tau_init, node, save_time, mask, params):
        param_map_update = params
        return solve_ivp_diffrax_prop(
            f=prop_aug_dy,
            tau_final=tau_grid[1],  # scalar
            y_0=V0,  # shape (n_states + stm_size,)
            args=(
                u_cur,  # shape (1, n_controls)
                u_next,  # shape (1, n_controls)
                tau_init,  # shape (1, 1)
                node,  # shape (1, 1)
                state_dot,  # function or array
                foh_mask,
                settings.sim.n,
                param_map_update,
                stm_meta,
                A_vmapped,
                n_states_prop,
            ),
            tau_0=tau_grid[0],  # scalar
            solver_name=settings.prp.solver,
            rtol=settings.prp.rtol,
            atol=settings.prp.atol,
            extra_kwargs=settings.prp.args,
            save_time=save_time,  # shape (MAX_TAU_LEN,)
            mask=mask,  # shape (MAX_TAU_LEN,), dtype=bool
        )

    return propagation_solver


def s_to_t(x: np.ndarray, u: np.ndarray, settings: Config, discretizer: Discretizer) -> list[float]:
    """Convert normalized time s to real time t.

    This function converts the normalized time variable s to real time t
    based on the hold type of the time-dilation control.

    Args:
        x: State trajectory array, shape (N, n_states).
        u: Control trajectory array, shape (N, n_controls).
        settings (Config): Configuration settings.
        discretizer: Discretizer instance (used for ``dis_type``).

    Returns:
        list[float]: List of real time points.
    """
    t = [x[:, settings.sim.time_slice][0]]
    tau = np.linspace(0, 1, settings.sim.n)
    idx_s = _time_dilation_index(settings, u.shape[1])
    u_foh_mask = getattr(settings.sim.u, "foh_mask", None)
    foh_mask = _resolve_foh_mask(discretizer.dis_type, u.shape[1], u_foh_mask)
    td_is_foh = foh_mask[idx_s] > 0.5
    for k in range(1, settings.sim.n):
        s_kp = u[k - 1, idx_s]
        s_k = u[k, idx_s]
        if td_is_foh:
            t.append(t[k - 1] + 0.5 * (s_k + s_kp) * (tau[k] - tau[k - 1]))
        else:
            t.append(t[k - 1] + (tau[k] - tau[k - 1]) * (s_kp))
    return t


def t_to_tau(
    u: np.ndarray, t: np.ndarray, t_nodal: np.ndarray, settings: Config, discretizer: Discretizer
) -> tuple[np.ndarray, np.ndarray]:
    """Convert real time t to normalized time tau.

    This function converts real time t to normalized time tau and interpolates
    the control inputs according to each control's hold type.

    Args:
        u (np.ndarray): Control trajectory array, shape (N, n_controls).
        t (np.ndarray): Real time points.
        t_nodal (np.ndarray): Nodal time points.
        settings (Config): Configuration settings.
        discretizer: Discretizer instance (used for ``dis_type``).

    Returns:
        tuple[np.ndarray, np.ndarray]: (tau, u_interp) where tau is normalized time and u_interp is
            interpolated controls.
    """
    u_foh_mask = getattr(settings.sim.u, "foh_mask", None)
    foh_mask = _resolve_foh_mask(discretizer.dis_type, u.shape[1], u_foh_mask)
    foh_mask_bool = foh_mask > 0.5

    def u_lam(new_t):
        idx = np.searchsorted(t_nodal, new_t, side="right") - 1
        idx = np.clip(idx, 0, len(t_nodal) - 1)
        zoh_vals = u[idx, :]
        foh_vals = np.array([np.interp(new_t, t_nodal, u[:, i]) for i in range(u.shape[1])])
        return np.where(foh_mask_bool, foh_vals, zoh_vals)

    u_interp = np.array([u_lam(t_i) for t_i in t])

    tau = np.zeros(len(t))
    tau_nodal = np.linspace(0, 1, settings.sim.n)
    idx_s = _time_dilation_index(settings, u.shape[1])
    td_is_foh = foh_mask[idx_s] > 0.5
    for k in range(1, len(t)):
        k_nodal = np.where(t_nodal < t[k])[0][-1]
        s_kp = u[k_nodal, idx_s]
        tp = t_nodal[k_nodal]
        tau_p = tau_nodal[k_nodal]

        s_k = u[k_nodal + 1, idx_s]
        if td_is_foh:
            tau[k] = tau_p + 2 * (t[k] - tp) / (s_k + s_kp)
        else:
            tau[k] = tau_p + (t[k] - tp) / s_kp
    return tau, u_interp


def simulate_nonlinear_time(
    params: dict,
    x: np.ndarray,
    u: np.ndarray,
    tau_vals: np.ndarray,
    t: np.ndarray,
    settings: Config,
    propagation_solver: callable,
    dynamics_discrete: Optional[Callable] = None,
    stm_meta: Optional[StmMeta] = None,
) -> np.ndarray:
    """Simulate the nonlinear system dynamics over time.

    This function simulates the system dynamics using the optimal control sequence
    and returns the resulting state trajectory.

    Args:
        params: System parameters.
        x: State trajectory array, shape (N, n_states).
        u: Control trajectory array, shape (N, n_controls).
        tau_vals (np.ndarray): Normalized time points for simulation.
        t (np.ndarray): Real time points.
        settings: Configuration settings.
        propagation_solver (callable): Function for propagating the system state.
        dynamics_discrete: Optional discrete dynamics map f_discrete(x, u, node, params)
            used to apply impulsive/discrete updates at each node before continuous propagation.

    Returns:
        np.ndarray: Simulated state trajectory.
    """
    x_0 = settings.sim.x_prop.initial

    n_segments = settings.sim.n - 1
    n_states = x_0.shape[0]
    n_tau = len(tau_vals)

    states = np.empty((n_states, n_tau))
    tau = np.linspace(0, 1, settings.sim.n)

    stm_size = _stm_carry_size(stm_meta)

    # Precompute control interpolation
    u_interp = np.stack([np.interp(t, t, u[:, i]) for i in range(u.shape[1])], axis=-1)
    _time_dilation_index(settings, u.shape[1])

    has_u_d = np.any(settings.sim.u._impulsive_mask())

    # Bin tau_vals into segments of tau
    tau_inds = np.digitize(tau_vals, tau) - 1
    tau_inds = np.where(tau_inds == settings.sim.n - 1, settings.sim.n - 2, tau_inds)

    prev_count = 0
    out_idx = 0

    for k in range(n_segments):
        controls_current = u_interp[k][None, :]
        controls_next = u_interp[k + 1][None, :]

        # Mask for tau_vals in current segment
        mask = (tau_inds >= k) & (tau_inds < k + 1)
        count = np.sum(mask)

        tau_cur = tau_vals[prev_count : prev_count + count]
        # Ensure integration reaches the segment endpoint.
        # If tau_cur already contains tau[k+1], avoid duplicating it.
        append_endpoint = tau_cur.size == 0 or not np.isclose(tau_cur[-1], tau[k + 1])
        if append_endpoint:
            tau_cur = np.concatenate([tau_cur, np.array([tau[k + 1]])])
            count += 1

        # Pad to fixed length
        pad_len = settings.prp.max_tau_len - count
        tau_cur_padded = np.pad(tau_cur, (0, pad_len), constant_values=tau[k + 1])
        mask_padded = np.concatenate([np.ones(count), np.zeros(pad_len)]).astype(bool)

        # Map prior node state to posterior using discrete dynamics when available.
        if has_u_d and dynamics_discrete is not None:
            x_post = np.asarray(
                dynamics_discrete(
                    np.asarray(x_0),
                    np.asarray(u[k]),
                    int(k),
                    params,
                )
            ).reshape(-1)
        else:
            x_post = np.asarray(x_0).copy()

        # STMs are propagated parameters: extend x with a Φ_stm carry block so
        # the propagator integrates Φ alongside x. The block is seeded with
        # the per-segment Φ (identity for non-anchored / for anchor row at k;
        # otherwise the prior-iterate chained value from params).
        if stm_size > 0 and stm_meta is not None:
            n_phys = stm_meta.n_phys
            stm_phi_prev = params.get("__stm_phi__", {})
            phi_carry = np.zeros(stm_size, dtype=float)
            for slot in stm_meta.slots:
                if slot.kind == "physical":
                    if slot.anchor_node is None:
                        phi_carry[slot.slice] = np.eye(n_phys).reshape(-1)
                    elif slot.anchor_node == k:
                        phi_carry[slot.slice] = np.eye(n_phys).reshape(-1)
                    else:
                        prev = stm_phi_prev.get(slot.name)
                        if prev is not None and k < len(prev):
                            phi_carry[slot.slice] = np.asarray(prev[k]).reshape(-1)
                else:
                    phi_carry[slot.slice] = 0.0
            x_post_carry = np.concatenate([np.asarray(x_post), phi_carry])
        else:
            x_post_carry = x_post

        # Call the continuous propagation solver with padded tau_cur and mask
        sol = _invoke_solver(
            propagation_solver,
            x_post_carry,
            (tau[k], tau[k + 1]),
            controls_current,
            controls_next,
            np.array([[tau[k]]]),
            np.array([[k]]),
            tau_cur_padded,
            mask_padded,
            params,
        )

        # Store requested samples (state portion only); exclude endpoint only
        # when it was appended solely for continuity propagation to the next
        # segment.
        n_store = count - 1 if append_endpoint else count
        sol_state = sol[:, :n_states]
        states[:, out_idx : out_idx + n_store] = sol_state[:n_store].T
        out_idx += n_store
        x_0 = sol_state[count - 1]  # Last x value used as next x_0

        prev_count += n_store

    return states.T
