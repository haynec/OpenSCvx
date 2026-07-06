"""``make_solve_loop`` must match the Python-loop ``.solve()`` trajectory.

The ``lax.while_loop`` wrapper over ``iteration_fn`` should reproduce what the
Python ``while`` loop in ``Problem.solve()`` produces. ``k_max`` is kept below
the convergence point (``J_tr`` stays well above ``ep_tr`` throughout), so both
loops run the identical iteration count and the only divergence is the
floating-point reordering between the ``lax.while_loop``-compiled body and the
per-iteration ``jax.jit``'d body the Python loop drives.
"""

import numpy as np
import pytest

from openscvx.algorithms.loop import make_solve_loop
from tests.algorithms._iteration_helpers import build_iteration_fn
from tests.solvers._iteration_callback_helpers import build_brachistochrone

pytestmark = [pytest.mark.e2e, pytest.mark.qpax]


def test_make_solve_loop_matches_python_solve():
    prob = build_brachistochrone("qpax", n=8, k_max=5)
    prob.initialize()

    # Reference: the legacy Python-driven solve.
    prob.solve()
    solve_x = np.asarray(prob.state.x)
    solve_u = np.asarray(prob.state.u)
    solve_k = int(prob.state.k)

    # lax.while_loop over the fused body, from the same fresh initial iterate.
    # The convergence thresholds and iteration cap ride the state pytree
    # (snapshotted from the algorithm at reset()), so the loop takes no
    # constants — only the algorithm's convergence predicate.
    prob.reset()
    state0 = prob.state
    iteration_fn = build_iteration_fn(prob)
    solve_loop = make_solve_loop(iteration_fn, prob.algorithm.converged)
    loop_state = solve_loop(state0, prob._parameters)

    assert int(loop_state.k) == solve_k
    np.testing.assert_allclose(np.asarray(loop_state.x), solve_x, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(loop_state.u), solve_u, atol=1e-5, rtol=1e-5)


def test_make_solve_loop_honors_custom_predicate():
    """A custom convergence predicate, not PTR's default, terminates the loop.

    ``make_solve_loop`` takes the predicate as a parameter, so an algorithm that
    overrides :meth:`Algorithm.converged` has its policy honored on the JAX
    path. Here ``k > 3`` halts the loop strictly before the ``k_max`` cap and
    well before the J-metrics test would fire, proving the passed predicate —
    not a hard-coded one — drives termination.
    """
    prob = build_brachistochrone("qpax", n=8, k_max=20)
    prob.initialize()
    prob.reset()

    iteration_fn = build_iteration_fn(prob)
    solve_loop = make_solve_loop(iteration_fn, lambda state: state.k > 3)
    loop_state = solve_loop(prob.state, prob._parameters)

    # The cond is ``k <= k_max & not converged``; with the body advancing k by
    # one each step, ``k > 3`` first holds at k == 4, so the loop exits there.
    assert int(loop_state.k) == 4
