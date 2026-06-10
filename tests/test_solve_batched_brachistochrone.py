"""``Problem.solve_batched`` matches ``jax.vmap(solve_jax)`` element-wise.

``solve_batched`` owns the batch axis internally (``jax.vmap`` applied inside
the method) where :func:`jax.vmap` over :meth:`Problem.solve_jax` leaves it to
the caller. With no export wired up (Phase 1) the two are just different
spellings of the same batched solve, so over a stack of trajectory guesses
each batch element must agree with the corresponding ``jax.vmap(solve_jax)``
result. CVXPy runs the ``B`` solves sequentially (host CVXPy isn't
thread-safe); QPAX runs them in parallel under vmap. Parallels
``tests/test_solve_jax_vmap_brachistochrone.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.solvers._iteration_callback_helpers import build_brachistochrone


@pytest.mark.parametrize("backend", ["cvxpy", "qpax"])
def test_solve_batched_matches_vmap_solve_jax(backend):
    if backend == "qpax":
        pytest.importorskip("qpax")

    prob = build_brachistochrone(backend, n=8, k_max=20)
    prob.initialize()

    base_x = prob.state.x

    # Stack four guesses by varying the x-coordinate of position (component 0).
    shifts = jnp.array([0.0, 0.3, -0.3, 0.6])
    x_guess_stack = jnp.stack([base_x.at[0, 0].set(base_x[0, 0] + s) for s in shifts])

    # Per-element reference.
    bare_xs = []
    bare_us = []
    for i in range(x_guess_stack.shape[0]):
        res = prob.solve_jax(x_guess=x_guess_stack[i])
        bare_xs.append(np.asarray(res.x))
        bare_us.append(np.asarray(res.u))
    bare_xs = np.stack(bare_xs)
    bare_us = np.stack(bare_us)

    # Internal-vmap batched solve.
    batched = prob.solve_batched(x_guess=x_guess_stack)

    assert batched.x.shape == bare_xs.shape == (x_guess_stack.shape[0], 8, base_x.shape[1])
    np.testing.assert_allclose(np.asarray(batched.x), bare_xs, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(batched.u), bare_us, atol=1e-5, rtol=1e-5)
    assert batched.converged.shape == (x_guess_stack.shape[0],)

    jax.clear_caches()


def test_solve_batched_before_initialize_raises():
    prob = build_brachistochrone("qpax" if _has_qpax() else "cvxpy", n=8, k_max=1)
    N = prob.settings.sim.n
    n_x = prob.settings.sim.n_states
    x_stack = jnp.zeros((2, N, n_x))
    with pytest.raises(ValueError, match="initialize"):
        prob.solve_batched(x_guess=x_stack)


def _has_qpax() -> bool:
    try:
        import qpax  # noqa: F401

        return True
    except ImportError:
        return False
