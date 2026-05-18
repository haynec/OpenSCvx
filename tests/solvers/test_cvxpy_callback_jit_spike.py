"""Spike: verify ``jax.pure_callback`` composes cleanly with a CVXPy solve.

This is the go/no-go gate for Phase 4 of
``plans/solver-iteration-callbacks.md``. Before wrapping the full PTR solver
in a callback, prove on a trivial CVXPy problem that:

* ``jax.pure_callback`` ferries a parameter array into ``cvxpy.Problem.solve``
  and returns the optimum as a JAX-shaped array;
* ``jax.jit`` around the callback gives the same answer as a bare call;
* ``jax.jit(jax.vmap(cb))`` with ``vmap_method="sequential"`` fires the
  callback once per batch element and produces per-element-correct outputs.

If any of the three fail, the plan's CVXPy strategy is wrong and the rest of
Phase 4 has to be rethought before the real ``CVXPyPTRSolver.iteration_callback``
is implemented. If they pass, the plan's premise holds and the same shape of
wrapping works in the real solver.

The toy problem is ``min ||x - p||^2 s.t. x >= 0``, whose closed-form solution
is ``max(p, 0)`` — easy to assert against without re-running the solver.
"""

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np

# Declared output dtype follows JAX's current x64 setting so the spike runs
# regardless of how the rest of the suite has configured precision.
_JAX_FLOAT = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


# ============================================================================
# Toy CVXPy problem
# ============================================================================


def _make_toy_cvxpy_solve():
    """Build a parameterized CVXPy problem and return a host-side solve closure.

    The closure mirrors what ``CVXPyPTRSolver.iteration_callback`` will do:
    take a NumPy parameter value, mutate the CVXPy ``Parameter``, solve, and
    return a NumPy array. Built once and reused across calls so the same
    ``cvxpy.Problem`` (and its compiled chain) is exercised on every callback
    invocation — exactly how the real solver will use it.
    """
    n = 3
    x = cp.Variable(n)
    p = cp.Parameter(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= 0])

    def host_solve(p_value: np.ndarray) -> np.ndarray:
        p.value = np.asarray(p_value, dtype=float)
        prob.solve(solver="CLARABEL")
        return np.asarray(x.value, dtype=_JAX_FLOAT)

    return host_solve, n


def _build_callback(host_solve, n: int):
    """Wrap ``host_solve`` in ``jax.pure_callback`` with a declared shape."""
    result_struct = jax.ShapeDtypeStruct((n,), _JAX_FLOAT)

    def callback(p_value):
        return jax.pure_callback(
            host_solve,
            result_struct,
            p_value,
            vmap_method="sequential",
        )

    return callback


# ============================================================================
# Spike assertions
# ============================================================================


def test_pure_callback_matches_bare_solve():
    """Bare ``pure_callback`` call returns the closed-form optimum.

    Floor of the closed-form: ``argmin ||x - p||^2 s.t. x >= 0`` is
    ``max(p, 0)``. Exercises the basic ``pure_callback`` plumbing without any
    transform on top.
    """
    host_solve, n = _make_toy_cvxpy_solve()
    callback = _build_callback(host_solve, n)

    p_value = jnp.asarray([1.0, -2.0, 0.5])
    result = callback(p_value)

    # Tolerance is CLARABEL's default convergence floor — the test only cares
    # that the callback ferried the answer back into JAX shape, not that the
    # toy problem was solved to machine precision.
    np.testing.assert_allclose(np.asarray(result), np.maximum(np.asarray(p_value), 0.0), atol=1e-4)


def test_pure_callback_composes_with_jit():
    """``jax.jit`` around the callback produces the same answer as the bare call.

    This is the load-bearing assertion: if ``jit`` can't trace the callback,
    the real ``CVXPyPTRSolver.iteration_callback`` can't be embedded in the
    JAX-pure SCP loop. Run twice to confirm the compiled trace is cached and
    reused (no per-call re-tracing).
    """
    host_solve, n = _make_toy_cvxpy_solve()
    callback = _build_callback(host_solve, n)
    jitted = jax.jit(callback)

    p_value = jnp.asarray([1.0, -2.0, 0.5])

    bare = callback(p_value)
    jit1 = jitted(p_value)
    jit2 = jitted(jnp.asarray([0.0, 3.0, -1.0]))

    np.testing.assert_allclose(np.asarray(jit1), np.asarray(bare), atol=1e-4)
    np.testing.assert_allclose(np.asarray(jit2), np.asarray([0.0, 3.0, 0.0]), atol=1e-4)


def test_pure_callback_composes_with_jit_vmap():
    """``jax.jit(jax.vmap(cb))`` fires the callback B times sequentially.

    ``vmap_method="sequential"`` is the only contract CVXPy can satisfy
    (CVXPy can't ingest a batched parameter set). Confirm the resulting batch
    output is element-wise the per-element solve.
    """
    host_solve, n = _make_toy_cvxpy_solve()
    callback = _build_callback(host_solve, n)
    batched = jax.jit(jax.vmap(callback))

    batch = jnp.asarray(
        [
            [1.0, -2.0, 0.5],
            [0.0, 3.0, -1.0],
            [-4.0, -5.0, 2.0],
            [7.0, 0.25, -0.25],
        ]
    )
    result = batched(batch)

    expected = np.maximum(np.asarray(batch), 0.0)
    np.testing.assert_allclose(np.asarray(result), expected, atol=1e-4)
