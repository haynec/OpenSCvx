"""``Problem.solve_jax()`` must match ``Problem.solve()`` on brachistochrone.

The Python-loop ``solve()`` and the ``lax.while_loop``-driven ``solve_jax()``
share the same fused ``iteration_fn`` body — the only divergence is the
floating-point reordering between Python-loop and ``lax.while_loop``-compiled
execution, plus any termination-criterion timing differences. Both should
converge to the same iterate within tolerance on a fixed-seed brachistochrone.
Parametrized over the two backends that work today (CVXPy / QPAX); Moreau is
excluded because its warm-start carry is host-side state that neither path
threads (see ``plans/jax-pure-solve.md`` Decision Log 2026-05-27).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.solvers._iteration_callback_helpers import build_brachistochrone


@pytest.mark.parametrize("backend", ["cvxpy", pytest.param("qpax", marks=pytest.mark.qpax)])
def test_solve_jax_matches_solve(backend):
    prob = build_brachistochrone(backend, n=8, k_max=20)
    prob.initialize()

    prob.solve()
    solve_x = np.asarray(prob.state.x)
    solve_u = np.asarray(prob.state.u)

    prob.reset()
    result = prob.solve_jax()

    np.testing.assert_allclose(np.asarray(result.x), solve_x, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(result.u), solve_u, atol=1e-5, rtol=1e-5)

    # ``result.X`` is the single-element wrapper documented in
    # ``OptimizationResults.from_final_state`` — same final iterate, lists
    # exist so ``result.x`` continues to return ``X[-1]``.
    assert len(result.X) == 1
    assert len(result.U) == 1
    # History fields are empty on the JAX-pure path.
    assert result.J_tr_history == []
    assert result.J_vb_history == []
    assert result.J_vc_history == []
    assert result.discretization_history == []

    jax.clear_caches()


@pytest.mark.qpax
def test_solve_jax_returns_pytree():
    """The result registers as a JAX pytree (children flow through transforms)."""

    prob = build_brachistochrone("qpax", n=4, k_max=2)
    prob.initialize()
    result = prob.solve_jax()

    leaves, treedef = jax.tree_util.tree_flatten(result)
    # Round-trip
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(np.asarray(rebuilt.x), np.asarray(result.x))
    assert bool(rebuilt.converged) == bool(result.converged)

    # ``result.t_final`` is a jnp 1-vector (matches the host-path shape).
    assert isinstance(result.t_final, jnp.ndarray)

    jax.clear_caches()
