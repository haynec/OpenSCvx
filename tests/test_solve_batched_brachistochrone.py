"""``Problem.solve_batched`` matches ``jax.vmap(solve_jax)`` element-wise.

``solve_batched`` owns the batch axis internally (``jax.vmap`` applied inside
the method) where :func:`jax.vmap` over :meth:`Problem.solve_jax` leaves it to
the caller. With no export wired up (Phase 1) the two are just different
spellings of the same batched solve, so over a stack of boundary conditions
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

    # Default pins (full unified vectors with ``nan`` at non-Fix entries).
    x_init_default = prob.state.x_init_pin
    x_term_default = prob.state.x_term_pin

    # Stack four ICs by varying the x-coordinate of position (component 0);
    # the terminal pin is shared, so broadcast it to the same leading axis.
    shifts = jnp.array([0.0, 0.3, -0.3, 0.6])
    x0_stack = jnp.stack([x_init_default.at[0].set(x_init_default[0] + s) for s in shifts])
    xf_stack = jnp.broadcast_to(x_term_default, x0_stack.shape)

    # Reference: caller-owned vmap over solve_jax across both stacks.
    reference = jax.vmap(prob.solve_jax, in_axes=(0, 0, None))(x0_stack, xf_stack, None)

    # Internal-vmap batched solve.
    batched = prob.solve_batched(x0_stack, xf_stack)

    assert batched.x.shape == reference.x.shape == (x0_stack.shape[0], 8, x0_stack.shape[1])
    np.testing.assert_allclose(np.asarray(batched.x), np.asarray(reference.x), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(batched.u), np.asarray(reference.u), atol=1e-5, rtol=1e-5)
    # ``converged`` carries the per-batch leading axis.
    assert batched.converged.shape == (x0_stack.shape[0],)

    jax.clear_caches()


def test_solve_batched_before_initialize_raises():
    prob = build_brachistochrone("qpax" if _has_qpax() else "cvxpy", n=8, k_max=1)
    x_stack = jnp.zeros((2, prob.settings.sim.n_states))
    with pytest.raises(ValueError, match="initialize"):
        prob.solve_batched(x_stack, x_stack)


def _has_qpax() -> bool:
    try:
        import qpax  # noqa: F401

        return True
    except ImportError:
        return False
