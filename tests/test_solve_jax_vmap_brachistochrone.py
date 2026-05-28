"""``jax.vmap(problem.solve_jax)`` matches per-element ``solve_jax``.

The batched solve over four stacked initial conditions should produce, for each
batch element, the same trajectory as if that single problem were solved alone.
CVXPy's :func:`jax.pure_callback` runs sequentially under vmap (host CVXPy
isn't thread-safe — see ``solve_jax``'s docstring); QPAX runs in parallel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.solvers._iteration_callback_helpers import build_brachistochrone


@pytest.mark.parametrize("backend", ["cvxpy", "qpax"])
def test_vmap_solve_jax_matches_per_element(backend):
    if backend == "qpax":
        pytest.importorskip("qpax")

    prob = build_brachistochrone(backend, n=8, k_max=20)
    prob.initialize()

    # Default pin (full unified vector with ``nan`` at non-Fix entries).
    x_init_default = prob.state.x_init_pin

    # Stack four ICs by varying the x-coordinate of position (component 0).
    shifts = jnp.array([0.0, 0.3, -0.3, 0.6])
    stacked = jnp.stack(
        [x_init_default.at[0].set(x_init_default[0] + s) for s in shifts]
    )

    # Per-element reference.
    bare_xs = []
    bare_us = []
    for i in range(stacked.shape[0]):
        res = prob.solve_jax(x_initial=stacked[i])
        bare_xs.append(np.asarray(res.x))
        bare_us.append(np.asarray(res.u))
    bare_xs = np.stack(bare_xs)
    bare_us = np.stack(bare_us)

    # Batched solve.
    batched = jax.vmap(prob.solve_jax, in_axes=(0, None, None))(stacked, None, None)

    assert batched.x.shape == bare_xs.shape
    np.testing.assert_allclose(np.asarray(batched.x), bare_xs, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(batched.u), bare_us, atol=1e-5, rtol=1e-5)
    # ``converged`` is a per-batch jnp.bool_ under vmap.
    assert batched.converged.shape == (stacked.shape[0],)

    jax.clear_caches()
