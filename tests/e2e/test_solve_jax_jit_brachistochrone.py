"""``jax.jit(problem.solve_jax)`` matches the bare call.

``solve_jax`` is JIT'd internally via the cached ``make_solve_loop`` closure,
so wrapping again with ``jax.jit`` is a near no-op — but it's the canonical
way users write "compile once, solve many times" (the MPC inner-loop shape
from ``solve_jax``'s docstring), so the round-trip is worth a regression
test.
"""

import jax
import numpy as np
import pytest

from tests.solvers._iteration_callback_helpers import build_brachistochrone


@pytest.mark.parametrize(
    "backend", ["cvxpy", pytest.param("qpax", marks=pytest.mark.qpax)]
)
def test_jit_solve_jax_matches_bare(backend):
    prob = build_brachistochrone(backend, n=8, k_max=20)
    prob.initialize()

    result_bare = prob.solve_jax()
    fast_solve = jax.jit(prob.solve_jax)
    result_jit = fast_solve()

    np.testing.assert_allclose(
        np.asarray(result_jit.x), np.asarray(result_bare.x), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(result_jit.u), np.asarray(result_bare.u), atol=1e-5, rtol=1e-5
    )

    jax.clear_caches()
