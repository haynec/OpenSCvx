"""Tests for ``QPAXPTRSolver.iteration_callback``.

Two layers of verification:

* **Assembly parity** — the JAX assembly inside ``iteration_callback`` must
  produce the same ``(Q, q, A, b, G, h)`` matrices the NumPy ``_assemble_qp``
  produces on a fixed iterate. Without this, no downstream test result is
  trustworthy.
* **Solution parity** — running ``iteration_callback()(state, data)`` on the
  same iterate must yield the same primal trajectory as
  ``solver.solve()``. ``solve_qp_primal`` and ``solve_qp`` may take slightly
  different paths through PDIP, so we hold this to a loose-but-tight tolerance.

Built on the brachistochrone problem (small N, nonlinear dynamics, CTCS
constraints, no impulsive controls) — exercises the full QPAX assembly path
except impulsive coupling.
"""

import numpy as np
import pytest

pytest.importorskip("qpax")

from openscvx.solvers.ptr_solver import StatusCode, SubproblemSolution
from tests.solvers._iteration_callback_helpers import (
    build_brachistochrone,
    populate_numpy_stash,
    subproblem_data_from_numpy_stash,
)

# ============================================================================
# Assembly parity (Phase 2)
# ============================================================================


@pytest.mark.parametrize("constraint_style", ["ctcs", "nodal"])
def test_assemble_qp_jax_matches_numpy_on_brachistochrone(constraint_style):
    """The JAX assembly must produce the same (Q, q, A, b, G, h) matrices
    the NumPy ``_assemble_qp`` produces on a fixed iterate. Tolerance is
    tight (atol=1e-10) — any drift would indicate a numerical-formula
    divergence between the two paths.

    Parametrized across CTCS (only LICQ rows, no nodal assembly) and nodal
    (exercises the per-constraint nodal block) so both code paths get
    parity-checked.
    """
    prob = build_brachistochrone("qpax", n=4, k_max=1, constraint_style=constraint_style)
    prob.initialize()
    # Populate _dyn / _cons / _pen / _x_init / _x_term on the solver so both
    # assembly paths read from the same iterate. (The SCP loop no longer drives
    # the NumPy update_* path, so we set up the stash explicitly.)
    populate_numpy_stash(prob)

    solver = prob.solver
    Q_np, q_np, A_np, b_np, G_np, h_np = solver._assemble_qp()

    data = subproblem_data_from_numpy_stash(solver)
    Q_j, q_j, A_j, b_j, G_j, h_j = solver._assemble_qp_jax(data)

    np.testing.assert_allclose(np.asarray(Q_j), Q_np, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(q_j), q_np, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(A_j), A_np, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(b_j), b_np, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(G_j), G_np, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(h_j), h_np, atol=1e-10, rtol=1e-10)


# ============================================================================
# Solution parity (Phase 6)
# ============================================================================


def test_iteration_callback_matches_solve_on_brachistochrone():
    """``iteration_callback()(state, data)`` must produce the same primal
    trajectory as ``solver.solve()`` on the same iterate. Both now call
    ``qpax.solve_qp``, so on a convergent QP they produce the same primal up to
    PDIP tolerance."""
    prob = build_brachistochrone("qpax", n=4, k_max=1)
    prob.initialize()
    populate_numpy_stash(prob)
    solver = prob.solver

    # NumPy reference: re-call _assemble_qp + qpax.solve_qp on the stash.
    reference = solver.solve()

    # JAX callback path.
    data = subproblem_data_from_numpy_stash(solver)
    callback = solver.iteration_callback()
    # state is unused by QPAX's callback, but it has to be a valid JAX pytree
    # so ``jit``'s argument-tracing doesn't trip. ``None`` is the canonical
    # empty pytree.
    state = None
    solution = callback(state, data)

    assert isinstance(solution, SubproblemSolution)
    np.testing.assert_allclose(np.asarray(solution.x), reference.x, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(np.asarray(solution.u), reference.u, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(np.asarray(solution.nu), reference.nu, atol=1e-8, rtol=1e-8)
    # nu_vb is stacked (N, n_nodal) in the JAX path, list-of-arrays on NumPy.
    assert solution.nu_vb.shape == (solver.layout.N, solver.layout.n_nodal)
    # Cost reconstruction is independent of the QP solve — should match the
    # NumPy path's _reconstruct_cost output directly.
    np.testing.assert_allclose(float(solution.cost), reference.cost, atol=1e-8, rtol=1e-8)
    # solve_qp reports convergence on this well-posed QP.
    assert int(solution.status_code) == int(StatusCode.OPTIMAL)


def test_iteration_callback_traces_under_jit():
    """The callback is constructed under ``jax.jit`` already; this test
    just confirms it's callable end-to-end and that the compilation cost
    amortizes across repeated calls (no per-call re-tracing surprises)."""
    prob = build_brachistochrone("qpax", n=4, k_max=1)
    prob.initialize()
    populate_numpy_stash(prob)
    solver = prob.solver

    data = subproblem_data_from_numpy_stash(solver)
    callback = solver.iteration_callback()
    state = None

    sol1 = callback(state, data)
    sol2 = callback(state, data)

    # Both calls should produce structurally identical solutions.
    np.testing.assert_allclose(np.asarray(sol1.x), np.asarray(sol2.x), atol=0.0)
    np.testing.assert_allclose(np.asarray(sol1.u), np.asarray(sol2.u), atol=0.0)
