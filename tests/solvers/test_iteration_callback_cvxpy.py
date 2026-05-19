"""Tests for ``CVXPyPTRSolver.iteration_callback``.

The CVXPy callback wraps the existing NumPy ``solve()`` in
:func:`jax.pure_callback`; the assertion is that running the callback
end-to-end on a fixed brachistochrone iterate produces the same primal
trajectory as a direct :meth:`solve` call (the spike in
``test_cvxpy_callback_jit_spike.py`` already confirms the
``pure_callback`` + ``jit`` plumbing works at the toy level — this test
exercises the real PTR pipeline).
"""

import os
import sys
import tempfile
import types

import jax
import jax.numpy as jnp
import numpy as np

from openscvx.solvers.ptr_solver import (
    StatusCode,
    SubproblemData,
    SubproblemSolution,
)
from tests.solvers._iteration_callback_helpers import build_brachistochrone

# ============================================================================
# Fixtures
# ============================================================================


def _subproblem_data_from_solver(prob) -> SubproblemData:
    """Build a :class:`SubproblemData` from the CVXPy params after one solve.

    Reads back the exact linearization the last ``update_*`` cycle wrote into
    the CVXPy parameters. This means ``solver.solve()`` and
    ``iteration_callback()(None, data)`` are guaranteed to see the same
    inputs (the post-SCP-iter ``state.x`` is *not* the same as the
    last-iteration ``x_bar``, so reading from the params is the only way to
    get matched inputs without re-running an iteration).
    """
    settings = prob.settings
    lowered = prob._lowered
    solver = prob.solver
    ocp = solver._ocp_vars

    N = settings.sim.n
    n_x = settings.sim.n_states
    n_u = settings.sim.n_controls

    x_bar = np.asarray(ocp.x_bar.value)
    u_bar = np.asarray(ocp.u_bar.value)
    A_d = np.asarray(ocp.A_d.value)
    B_d = np.asarray(ocp.B_d.value)
    C_d = np.asarray(ocp.C_d.value)
    # New nomenclature: x_prop/x_prop_plus are not stored directly on CVXPyVariables.
    # Reconstruct continuous x_prop from affine dynamics bias:
    # dyn_bias[k] = x_prop[k] - A[k]x_bar[k] - B[k]u_bar[k] - C[k]u_bar[k+1]
    if hasattr(ocp, "x_prop") and ocp.x_prop is not None and ocp.x_prop.value is not None:
        x_prop = np.asarray(ocp.x_prop.value)
    else:
        dyn_bias = np.asarray(ocp.dyn_bias.value)
        x_prop = np.zeros((N - 1, n_x))
        for k in range(N - 1):
            x_prop[k] = (
                dyn_bias[k]
                + A_d[k] @ x_bar[k]
                + B_d[k] @ u_bar[k]
                + C_d[k] @ u_bar[k + 1]
            )

    x_prop_plus = (
        np.asarray(getattr(ocp, "x_prop_plus").value)
        if hasattr(ocp, "x_prop_plus")
        and getattr(ocp, "x_prop_plus") is not None
        and getattr(ocp, "x_prop_plus").value is not None
        else np.zeros((N, n_x))
    )
    E_d = (
        np.asarray(ocp.E_d.value)
        if ocp.E_d is not None and ocp.E_d.value is not None
        else np.zeros((N, n_x, n_u))
    )
    # D_d is absorbed into A/B/C at update time on the CVXPy path; passing
    # zeros tells the callback "no further absorption" — it just calls
    # update_dynamics_linearization with the already-absorbed A/B/C.
    D_d = np.zeros((N, n_x, n_x))

    jax_constraints = lowered.jax_constraints
    n_nodal = len(jax_constraints.nodal)
    n_cross = len(jax_constraints.cross_node)
    nodal_g = np.zeros((N, max(n_nodal, 1)))
    nodal_grad_x = np.zeros((N, max(n_nodal, 1), n_x))
    nodal_grad_u = np.zeros((N, max(n_nodal, 1), n_u))
    for c_idx, constraint in enumerate(jax_constraints.nodal):
        g_full = np.asarray(ocp.g[c_idx].value)
        grad_x_full = np.asarray(ocp.grad_g_x[c_idx].value)
        grad_u_full = np.asarray(ocp.grad_g_u[c_idx].value)
        for node in constraint.nodes:
            nodal_g[node, c_idx] = g_full[node]
            nodal_grad_x[node, c_idx] = grad_x_full[node]
            nodal_grad_u[node, c_idx] = grad_u_full[node]
    if n_nodal == 0:
        nodal_g = np.zeros((N, 0))
        nodal_grad_x = np.zeros((N, 0, n_x))
        nodal_grad_u = np.zeros((N, 0, n_u))

    cross_g = np.zeros((n_cross,))
    cross_grad_X = np.zeros((n_cross, N, n_x))
    cross_grad_U = np.zeros((n_cross, N, n_u))
    for c_idx in range(n_cross):
        cross_g[c_idx] = float(np.asarray(ocp.g_cross[c_idx].value))
        cross_grad_X[c_idx] = np.asarray(ocp.grad_g_X_cross[c_idx].value)
        cross_grad_U[c_idx] = np.asarray(ocp.grad_g_U_cross[c_idx].value)

    lam_prox = np.asarray(ocp.lam_prox.value)
    lam_cost = np.asarray(ocp.lam_cost.value)
    lam_vc = np.asarray(ocp.lam_vc.value)
    lam_vb_nodal = np.asarray(ocp.lam_vb_nodal.value)
    lam_vb_cross = np.asarray(ocp.lam_vb_cross.value)

    x_init = (
        np.asarray(lowered.x_unified.initial, dtype=float)
        if lowered.x_unified.initial is not None
        else np.full(n_x, np.nan)
    )
    x_term = (
        np.asarray(lowered.x_unified.final, dtype=float)
        if lowered.x_unified.final is not None
        else np.full(n_x, np.nan)
    )

    return SubproblemData(
        x_bar=jnp.asarray(x_bar),
        u_bar=jnp.asarray(u_bar),
        A_d=jnp.asarray(A_d),
        B_d=jnp.asarray(B_d),
        C_d=jnp.asarray(C_d),
        x_prop=jnp.asarray(x_prop),
        x_prop_plus=jnp.asarray(x_prop_plus),
        D_d=jnp.asarray(D_d),
        E_d=jnp.asarray(E_d),
        nodal_g=jnp.asarray(nodal_g),
        nodal_grad_x=jnp.asarray(nodal_grad_x),
        nodal_grad_u=jnp.asarray(nodal_grad_u),
        cross_g=jnp.asarray(cross_g),
        cross_grad_X=jnp.asarray(cross_grad_X),
        cross_grad_U=jnp.asarray(cross_grad_U),
        lam_prox=jnp.asarray(lam_prox),
        lam_cost=jnp.asarray(lam_cost),
        lam_vc=jnp.asarray(lam_vc),
        lam_vb_nodal=jnp.asarray(lam_vb_nodal),
        lam_vb_cross=jnp.asarray(lam_vb_cross),
        x_init=jnp.asarray(x_init),
        x_term=jnp.asarray(x_term),
    )


# ============================================================================
# Solution parity
# ============================================================================


def test_iteration_callback_matches_solve_on_brachistochrone():
    """``iteration_callback()(state, data)`` must produce the same primal
    trajectory as ``solver.solve()`` on the same iterate. The callback wraps
    the same NumPy ``solve()`` we compare against, so parity should be
    exact modulo CLARABEL re-solve noise.
    """
    prob = build_brachistochrone("cvxpy", n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    reference = solver.solve()

    data = _subproblem_data_from_solver(prob)
    callback = solver.iteration_callback()
    solution = callback(None, data)

    assert isinstance(solution, SubproblemSolution)
    np.testing.assert_allclose(np.asarray(solution.x), reference.x, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(solution.u), reference.u, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(solution.nu), reference.nu, atol=1e-6, rtol=1e-6)
    # nu_vb is stacked (N, n_nodal) in the callback output, list-of-arrays on NumPy.
    assert solution.nu_vb.shape == (prob.settings.sim.n, len(prob._lowered.jax_constraints.nodal))
    np.testing.assert_allclose(float(solution.cost), reference.cost, atol=1e-6, rtol=1e-6)
    # CVXPy + CLARABEL on a feasible iterate reports "optimal".
    assert int(solution.status_code) == int(StatusCode.OPTIMAL)


def test_iteration_callback_composes_with_jit():
    """``jax.jit(cb)(state, data)`` matches the bare call.

    The spike test already covers ``pure_callback`` + ``jit`` at the toy
    level; this asserts the real CVXPy solver also composes under ``jit``
    without per-call retracing.
    """
    prob = build_brachistochrone("cvxpy", n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    data = _subproblem_data_from_solver(prob)
    callback = solver.iteration_callback()
    jitted = jax.jit(callback)

    bare = callback(None, data)
    jitt = jitted(None, data)

    np.testing.assert_allclose(np.asarray(jitt.x), np.asarray(bare.x), atol=1e-8)
    np.testing.assert_allclose(np.asarray(jitt.u), np.asarray(bare.u), atol=1e-8)
    np.testing.assert_allclose(np.asarray(jitt.nu_vb), np.asarray(bare.nu_vb), atol=1e-8)


def test_iteration_callback_composes_with_vmap_sequential():
    """``jax.vmap(cb)`` fires the callback once per batch element.

    CVXPy can't ingest a batched parameter set, so the callback declares
    ``vmap_method="sequential"`` — under vmap the host is invoked B times in
    sequence. Stacking the same ``SubproblemData`` four times should yield
    four identical ``SubproblemSolution`` slices that each match the bare
    call.
    """
    prob = build_brachistochrone("cvxpy", n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    data = _subproblem_data_from_solver(prob)
    callback = solver.iteration_callback()
    bare = callback(None, data)

    batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (3,) + x.shape), data)
    batched = jax.vmap(callback, in_axes=(None, 0))(None, batch)

    for i in range(3):
        np.testing.assert_allclose(np.asarray(batched.x[i]), np.asarray(bare.x), atol=1e-8)
        np.testing.assert_allclose(np.asarray(batched.u[i]), np.asarray(bare.u), atol=1e-8)
