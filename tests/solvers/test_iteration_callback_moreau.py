"""Tests for ``MoreauPTRSolver.iteration_callback``.

Two layers of verification:

* **Assembly parity** — the JAX assembly inside ``iteration_callback`` must
  produce the same ``(P_data, coo_vals, q, b)`` quadruple the NumPy
  ``_assemble_conic`` produces on a fixed iterate.
* **Solution parity** — ``iteration_callback()(state, data)`` must produce
  the same primal trajectory as ``solver.solve()`` on the same iterate.
  Moreau's functional API is cold-start (no warm-start support per docs),
  while ``solver.solve()`` uses the OO Solver with warm-start — but on a
  **single-iterate** call the warm_start is None either way, so the two
  paths solve identical conic programs and should agree to PDIP tolerance.

These tests gate on ``_MOREAU_OK`` because Moreau is an optional dependency
with a license requirement; the gate keeps the suite green on machines
without a license while still exercising on CI / dev hosts that have one.
"""

import numpy as np
import pytest

from tests._marks import _MOREAU_OK, requires_moreau

pytestmark = requires_moreau

# Imports below are only reached when _MOREAU_OK is True.
if _MOREAU_OK:
    from openscvx.solvers.ptr_solver import StatusCode, SubproblemSolution
    from tests.solvers._iteration_callback_helpers import (
        build_brachistochrone,
        subproblem_data_from_numpy_stash,
    )


# ============================================================================
# Assembly parity (Phase 3)
# ============================================================================


@pytest.mark.parametrize("constraint_style", ["ctcs", "nodal"])
def test_assemble_conic_jax_matches_numpy_on_brachistochrone(constraint_style):
    """The JAX assembly must produce the same (P_data, coo_vals, q, b) the
    NumPy ``_assemble_conic`` produces on a fixed iterate.

    ``coo_vals`` is emitted in COO traversal order on both sides; the
    NumPy path applies scipy's CSR sort downstream, the JAX path applies the
    precomputed ``csr_to_coo_perm`` — both compose to the same A_data passed
    to Moreau's solver.
    """
    prob = build_brachistochrone("moreau", n=4, k_max=1, constraint_style=constraint_style)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    P_np, coo_np, q_np, b_np = solver._assemble_conic()

    data = subproblem_data_from_numpy_stash(solver)
    P_j, coo_j, q_j, b_j = solver._assemble_conic_jax(data)

    np.testing.assert_allclose(np.asarray(P_j), P_np, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(coo_j), coo_np, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(q_j), q_np, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(b_j), b_np, atol=1e-10, rtol=1e-10)


def test_csr_to_coo_perm_reconstructs_numpy_A_data():
    """``coo_vals[csr_to_coo_perm]`` must equal the ``A_data`` that
    ``solver.solve()`` passes to Moreau. Without this, the iteration_callback
    path would silently send a permuted system to Moreau."""
    import scipy.sparse as sp

    prob = build_brachistochrone("moreau", n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    _, coo_np, _, _ = solver._assemble_conic()
    # Match scipy's CSR sort the way solver.solve() does it.
    A_csr = sp.csr_matrix(
        (coo_np, (solver._coo_rows, solver._coo_cols)),
        shape=(solver._n_con, solver.layout.n_z),
    )
    A_csr.sort_indices()

    A_data_via_perm = coo_np[solver._csr_to_coo_perm]
    np.testing.assert_allclose(A_data_via_perm, A_csr.data, atol=0.0)


# ============================================================================
# Solution parity (Phase 6)
# ============================================================================


def test_iteration_callback_matches_solve_on_brachistochrone():
    """``iteration_callback()(state, data)`` must produce the same primal
    trajectory as ``solver.solve()`` on the same iterate.

    On a single-iterate call the OO Solver's warm-start is ``None`` (initial
    state), so the OO path and the functional path solve identical conic
    programs and must agree to PDIP tolerance.
    """
    prob = build_brachistochrone("moreau", n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    reference = solver.solve()

    data = subproblem_data_from_numpy_stash(solver)
    callback = solver.iteration_callback()
    solution = callback(None, data)

    assert isinstance(solution, SubproblemSolution)
    np.testing.assert_allclose(np.asarray(solution.x), reference.x, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(np.asarray(solution.u), reference.u, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(np.asarray(solution.nu), reference.nu, atol=1e-7, rtol=1e-7)
    assert solution.nu_vb.shape == (solver.layout.N, solver.layout.n_nodal)
    np.testing.assert_allclose(float(solution.cost), reference.cost, atol=1e-7, rtol=1e-7)
    # Optimal solves should map to StatusCode.OPTIMAL.
    assert int(solution.status_code) == int(StatusCode.OPTIMAL)


def test_iteration_callback_traces_under_jit():
    """The callback is constructed under ``jax.jit`` already; this confirms
    it's callable end-to-end and that repeated calls share the compiled
    trace (no per-call retrace surprises)."""
    prob = build_brachistochrone("moreau", n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    data = subproblem_data_from_numpy_stash(solver)
    callback = solver.iteration_callback()

    sol1 = callback(None, data)
    sol2 = callback(None, data)

    np.testing.assert_allclose(np.asarray(sol1.x), np.asarray(sol2.x), atol=0.0)
    np.testing.assert_allclose(np.asarray(sol1.u), np.asarray(sol2.u), atol=0.0)
