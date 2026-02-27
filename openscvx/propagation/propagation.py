from typing import Callable, Optional

import numpy as np

from openscvx.config import Config
from openscvx.discretization import Discretizer
from openscvx.integrators import solve_ivp_diffrax_prop
from openscvx.lowered import Dynamics


def _time_dilation_index(settings: Config, n_controls: int) -> int:
    """Return time-dilation control index, falling back to last control."""
    td_slice = getattr(settings.sim, "time_dilation_slice", None)
    if td_slice is None:
        return n_controls - 1
    return int(td_slice.start)


def prop_aug_dy(
    tau: float,
    x: np.ndarray,
    u_current: np.ndarray,
    u_next: np.ndarray,
    tau_init: float,
    node: int,
    state_dot: callable,
    dis_type: str,
    N: int,
    params: dict,
) -> np.ndarray:
    """Compute the augmented dynamics for propagation.

    This function computes the time-dilated dynamics for propagating the system
    state, taking into account the discretization type (ZOH or FOH). The
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
        dis_type (str): Discretization type ("ZOH" or "FOH").
        N (int): Number of nodes in trajectory.
        params: Dictionary of additional parameters passed to state_dot.

    Returns:
        np.ndarray: Time-dilated state derivatives.
    """
    x = x[None, :]

    if dis_type == "ZOH":
        beta = 0.0
    elif dis_type == "FOH":
        beta = (tau - tau_init) * N
    u = u_current + beta * (u_next - u_current)

    return state_dot(x, u, node, params).squeeze()


def get_propagation_solver(
    state_dot: Dynamics, settings: Config, discretizer: Discretizer
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

    def propagation_solver(V0, tau_grid, u_cur, u_next, tau_init, node, save_time, mask, params):
        param_map_update = params
        return solve_ivp_diffrax_prop(
            f=prop_aug_dy,
            tau_final=tau_grid[1],  # scalar
            y_0=V0,  # shape (n_states,)
            args=(
                u_cur,  # shape (1, n_controls)
                u_next,  # shape (1, n_controls)
                tau_init,  # shape (1, 1)
                node,  # shape (1, 1)
                state_dot,  # function or array
                discretizer.dis_type,
                settings.sim.n,
                param_map_update,
                # additional named parameters as **kwargs
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
    based on the discretization type and time dilation factors.

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
    for k in range(1, settings.sim.n):
        s_kp = u[k - 1, idx_s]
        s_k = u[k, idx_s]
        if discretizer.dis_type == "ZOH":
            t.append(t[k - 1] + (tau[k] - tau[k - 1]) * (s_kp))
        else:
            t.append(t[k - 1] + 0.5 * (s_k + s_kp) * (tau[k] - tau[k - 1]))
    return t


def t_to_tau(
    u: np.ndarray, t: np.ndarray, t_nodal: np.ndarray, settings: Config, discretizer: Discretizer
) -> tuple[np.ndarray, np.ndarray]:
    """Convert real time t to normalized time tau.

    This function converts real time t to normalized time tau and interpolates
    the control inputs accordingly.

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
    if discretizer.dis_type == "ZOH":
        # Zero-Order Hold: step interpolation (hold previous value)
        def u_lam(new_t):
            # Find the index of the last nodal time <= new_t
            idx = np.searchsorted(t_nodal, new_t, side="right") - 1
            idx = np.clip(idx, 0, len(t_nodal) - 1)
            return u[idx, :]
    elif discretizer.dis_type == "FOH":
        # First-Order Hold: linear interpolation
        def u_lam(new_t):
            return np.array([np.interp(new_t, t_nodal, u[:, i]) for i in range(u.shape[1])]).T
    else:
        raise ValueError("Currently unsupported discretization type")

    u_interp = np.array([u_lam(t_i) for t_i in t])

    tau = np.zeros(len(t))
    tau_nodal = np.linspace(0, 1, settings.sim.n)
    idx_s = _time_dilation_index(settings, u.shape[1])
    for k in range(1, len(t)):
        k_nodal = np.where(t_nodal < t[k])[0][-1]
        s_kp = u[k_nodal, idx_s]
        tp = t_nodal[k_nodal]
        tau_p = tau_nodal[k_nodal]

        s_k = u[k_nodal + 1, idx_s]
        if discretizer.dis_type == "ZOH":
            tau[k] = tau_p + (t[k] - tp) / s_kp
        else:
            tau[k] = tau_p + 2 * (t[k] - tp) / (s_k + s_kp)
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

    # Precompute control interpolation
    u_interp = np.stack([np.interp(t, t, u[:, i]) for i in range(u.shape[1])], axis=-1)
    _time_dilation_index(settings, u.shape[1])

    has_u_d = np.any(settings.sim.u.is_impulsive)

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
        tau_cur = np.concatenate([tau_cur, np.array([tau[k + 1]])])  # Always include final point
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
            x_post = x_0

        # Call the continuous propagation solver with padded tau_cur and mask
        sol = propagation_solver.call(
            x_post,
            (tau[k], tau[k + 1]),
            controls_current,
            controls_next,
            np.array([[tau[k]]]),
            np.array([[k]]),
            tau_cur_padded,
            mask_padded,
            params,
        )

        # Only store the valid portion (excluding the final point which becomes next x_0)
        states[:, out_idx : out_idx + count - 1] = sol[: count - 1].T
        out_idx += count - 1
        x_0 = sol[count - 1]  # Last value used as next x_0

        prev_count += count - 1

    return states.T
