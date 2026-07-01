"""Shared SCP cost / penalty math.

These two functions are the cost machinery the SCP iteration and the autotuners
both lean on: :func:`calculate_cost_from_state` weights a state trajectory by
its boundary-condition objective, and :func:`calculate_nonlinear_penalty`
assembles the three-component nonlinear penalty (cost, virtual control, nodal
constraint violation). They are plumbing, not autotuning policy, so they live
as module functions with no dependency beyond ``jnp`` and the config typing.
"""

from typing import TYPE_CHECKING, Tuple, Union

import jax.numpy as jnp

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints


def calculate_cost_from_state(
    x: jnp.ndarray,
    settings: "Config",
    lam_cost: Union[float, jnp.ndarray] = 1.0,
) -> jnp.ndarray:
    """Compute the boundary-condition-weighted cost contribution for ``x``.

    Args:
        x: State trajectory, shape ``(N, n_states)``.
        settings: Configuration object carrying scaling matrices and
            boundary-condition types.
        lam_cost: Per-state cost weight. Scalar or array of shape
            ``(n_states,)``.

    Returns:
        Scalar cost (jnp), weighted by ``lam_cost``.
    """
    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    lam = jnp.asarray(lam_cost)
    cost = jnp.asarray(0.0)
    for i in range(settings.sim.n_states):
        w = lam[i] if lam.ndim > 0 else lam
        if settings.sim.x.final_type[i] == "Minimize":
            cost = cost + w * scaled_x[-1, i]
        if settings.sim.x.final_type[i] == "Maximize":
            cost = cost - w * scaled_x[-1, i]
        if settings.sim.x.initial_type[i] == "Minimize":
            cost = cost + w * scaled_x[0, i]
        if settings.sim.x.initial_type[i] == "Maximize":
            cost = cost - w * scaled_x[0, i]
    return cost


def calculate_nonlinear_penalty(
    x_prop: jnp.ndarray,
    x_bar: jnp.ndarray,
    u_bar: jnp.ndarray,
    lam_vc: jnp.ndarray,
    lam_vb_nodal: jnp.ndarray,
    lam_vb_cross: jnp.ndarray,
    lam_cost: Union[float, jnp.ndarray],
    nodal_constraints: "LoweredJaxConstraints",
    params: dict,
    settings: "Config",
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the three components of the nonlinear penalty.

    This is JAX-traceable: the Python loops over
    ``nodal_constraints.nodal`` / ``cross_node`` unroll at trace time
    (the lists are static-length lists of compiled closures built by
    :class:`Problem`).

    Args:
        x_prop: Propagated state, shape ``(N-1, n_states)``.
        x_bar: Nodal state, shape ``(N, n_states)``.
        u_bar: Nodal control, shape ``(N, n_controls)``.
        lam_vc: Virtual control weight, scalar or matrix.
        lam_vb_nodal: Nodal virtual-buffer weights, shape ``(N, n_nodal)``.
        lam_vb_cross: Cross-node virtual-buffer weights, shape ``(n_cross,)``.
        lam_cost: Cost weight, scalar or shape ``(n_states,)``.
        nodal_constraints: Lowered JAX constraints.
        params: Problem parameter dictionary.
        settings: Configuration object.

    Returns:
        ``(nonlinear_cost, nonlinear_penalty, nodal_penalty)`` — all
        scalar jnp arrays.
    """
    nodal_penalty = jnp.asarray(0.0)

    for idx, constraint in enumerate(nodal_constraints.nodal):
        g = constraint.func(x_bar, u_bar, 0, params)
        if constraint.nodes is not None:
            nodes_array = jnp.asarray(constraint.nodes)
            g_filtered = g[nodes_array]
            w = lam_vb_nodal[nodes_array, idx]
        else:
            g_filtered = g
            w = lam_vb_nodal[:, idx]
        viol = jnp.abs(g_filtered) if constraint.is_equality else jnp.maximum(0.0, g_filtered)
        nodal_penalty = nodal_penalty + jnp.sum(w * viol)

    for idx, constraint in enumerate(nodal_constraints.cross_node):
        w = lam_vb_cross[idx]
        g = constraint.func(x_bar, u_bar, params)
        viol = jnp.abs(g) if constraint.is_equality else jnp.maximum(0.0, g)
        nodal_penalty = nodal_penalty + w * jnp.sum(viol)

    cost = calculate_cost_from_state(x_bar, settings, lam_cost)
    x_diff = settings.sim.inv_S_x @ (x_bar[1:, :] - x_prop).T

    return cost, jnp.sum(lam_vc * jnp.abs(x_diff.T)), nodal_penalty
