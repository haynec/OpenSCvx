"""``make_solve_loop`` must match the Python-loop ``.solve()`` trajectory.

The ``lax.while_loop`` wrapper over ``iteration_fn`` should reproduce what the
Python ``while`` loop in ``Problem.solve()`` produces. ``k_max`` is kept below
the convergence point (``J_tr`` stays well above ``ep_tr`` throughout), so both
loops run the identical iteration count and the only divergence is QPAX's
``solve_qp_primal`` (JAX path) vs ``solve_qp`` (NumPy path).
"""

import numpy as np
import pytest

pytest.importorskip("qpax")

from openscvx.algorithms.scvx.iteration import make_scp_iteration, make_solve_loop
from tests.solvers._iteration_callback_helpers import build_brachistochrone


def test_make_solve_loop_matches_python_solve():
    prob = build_brachistochrone("qpax", n=8, k_max=5)
    prob.initialize()

    # Reference: the legacy Python-driven solve.
    prob.solve()
    solve_x = np.asarray(prob.state.x)
    solve_u = np.asarray(prob.state.u)
    solve_k = int(prob.state.k)

    # lax.while_loop over the fused body, from the same fresh initial iterate.
    prob.reset()
    state0 = prob.state
    iteration_fn = make_scp_iteration(
        dynamics=prob.lowered.dynamics,
        dynamics_discrete=prob.lowered.dynamics_discrete,
        jax_constraints=prob._compiled_constraints,
        discretizer=prob.discretizer,
        solver_callback=prob.solver.iteration_callback(),
        autotuner=prob.algorithm.autotuner,
        settings=prob.settings,
    )
    algo = prob.algorithm
    solve_loop = make_solve_loop(iteration_fn, algo.ep_tr, algo.ep_vb, algo.ep_vc, algo.k_max)
    loop_state = solve_loop(state0, prob._parameters)

    assert int(loop_state.k) == solve_k
    np.testing.assert_allclose(np.asarray(loop_state.x), solve_x, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(loop_state.u), solve_u, atol=1e-5, rtol=1e-5)
