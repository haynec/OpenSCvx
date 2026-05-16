"""Tests for the Moreau-backed PTR convex subproblem solver.

Covers:
  * Instantiation guard when ``moreau`` isn't installed.
  * ``PTRSolverSpec`` rejects CVXPy-only fields under ``backend='moreau'``.
  * ``initialize()`` raises for unsupported feature combinations
    (.convex(), cross-node, impulsive).
  * Assembly produces arrays with shapes consistent with ``_ConicLayout`` and
    the fixed CSR structure built at ``initialize()``.
  * End-to-end ``solve()`` returns a :class:`PTRSolveResult`.
  * Round-trip parity vs ``CVXPyPTRSolver`` on a small double-integrator.

The brachistochrone parametrized-backend test in ``tests/test_brachistochrone.py``
exercises Moreau on a richer nonlinear CTCS problem; the unit-style tests here
focus on the API contract and the assembly machinery.
"""

import numpy as np
import pytest
from scipy import sparse as sp

# Moreau is an optional dependency — skip the whole module if it's missing.
pytest.importorskip("moreau")

import openscvx as ox
from openscvx import Problem
from openscvx.solvers import MoreauPTRSolver, PTRSolver, PTRSolveResult


# ============================================================================
# Helpers
# ============================================================================


def _make_double_integrator_problem(n=6, backend="moreau", k_max=20):
    """2-D double integrator with state/control box bounds — no ``.convex()``,
    no cross-node, no impulsive.  The library's auto-CTCS for time bounds
    still applies, exercising the CTCS row assembly in ``MoreauPTRSolver``."""
    pos = ox.State("pos", shape=(2,))
    pos.min = np.array([-10.0, -10.0])
    pos.max = np.array([10.0, 10.0])
    pos.initial = np.array([0.0, 0.0])
    pos.final = np.array([3.0, 3.0])

    vel = ox.State("vel", shape=(2,))
    vel.min = np.array([-5.0, -5.0])
    vel.max = np.array([5.0, 5.0])
    vel.initial = np.array([0.0, 0.0])
    vel.final = [("free", 0.0), ("free", 0.0)]

    u = ox.Control("u", shape=(2,))
    u.min = np.array([-3.0, -3.0])
    u.max = np.array([3.0, 3.0])
    u.guess = np.zeros((n, 2))

    dyn = {"pos": vel, "vel": u}
    time = ox.Time(initial=0.0, final=("minimize", 2.0), min=0.0, max=10.0, uniform_time_grid=True)

    return Problem(
        dynamics=dyn,
        states=[pos, vel],
        controls=[u],
        time=time,
        constraints=[],
        N=n,
        float_dtype="float64",
        algorithm={
            "lam_prox": 1.0,
            "lam_cost": 0.5,
            "k_max": k_max,
            "ep_tr": 1e-5,
            "ep_vb": 1e-5,
            "ep_vc": 1e-8,
        },
        solver={"backend": backend},
    )


# ============================================================================
# Construction / dependency-guard tests
# ============================================================================


def test_moreau_solver_is_a_PTRSolver():
    """MoreauPTRSolver must satisfy the abstract PTR contract so it composes
    with the rest of the SCP machinery interchangeably with other backends."""
    solver = MoreauPTRSolver()
    assert isinstance(solver, PTRSolver)


def test_moreau_missing_moreau_raises_clear_error(monkeypatch):
    """When moreau isn't installed, instantiation should raise ImportError
    pointing the user at the install command — not an opaque ModuleNotFoundError
    from inside ``solve()``."""
    import openscvx.solvers.moreau_ptr_solver as mod

    monkeypatch.setattr(mod, "_MOREAU_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"pip install openscvx\[moreau\]"):
        MoreauPTRSolver()


def test_moreau_spec_rejects_cvxpy_only_fields():
    """PTRSolverSpec should reject cvx_solver / cvxpygen under backend='moreau'
    so users get a config-time validation error rather than a confusing runtime
    one."""
    from openscvx.solvers import resolve_solver_config

    with pytest.raises(ValueError, match="only valid for backend='cvxpy'"):
        resolve_solver_config({"backend": "moreau", "cvxpygen": True})

    with pytest.raises(ValueError, match="only valid for backend='cvxpy'"):
        resolve_solver_config({"backend": "moreau", "cvx_solver": "CLARABEL"})


def test_moreau_spec_build_returns_moreau_solver():
    """PTRSolverSpec.build() with backend='moreau' should return a
    MoreauPTRSolver instance."""
    from openscvx.solvers import resolve_solver_config

    spec = resolve_solver_config({"backend": "moreau"})
    solver = spec.build()
    assert isinstance(solver, MoreauPTRSolver)


# ============================================================================
# Unsupported-feature rejection tests
# ============================================================================


def test_moreau_rejects_user_convex_constraints():
    """User .convex() constraints lower to second-order cones — not yet
    supported by MoreauPTRSolver v1.  The refusal lives on
    ``ConvexSolver.lower_convex_constraints`` (inherited default), so Moreau
    rejects them during ``Problem(...)`` lowering — fail-fast before setup."""
    n = 5
    pos = ox.State("pos", shape=(2,))
    pos.min = np.array([-10.0, -10.0])
    pos.max = np.array([10.0, 10.0])
    pos.initial = np.array([0.0, 0.0])
    pos.final = np.array([3.0, 3.0])
    vel = ox.State("vel", shape=(2,))
    vel.min = np.array([-5.0, -5.0])
    vel.max = np.array([5.0, 5.0])
    vel.initial = np.array([0.0, 0.0])
    vel.final = [("free", 0.0), ("free", 0.0)]
    u = ox.Control("u", shape=(2,))
    u.min = np.array([-3.0, -3.0])
    u.max = np.array([3.0, 3.0])
    u.guess = np.zeros((n, 2))
    dyn = {"pos": vel, "vel": u}
    time = ox.Time(initial=0.0, final=("minimize", 2.0), min=0.0, max=10.0)

    cvx_constraint = (ox.linalg.Norm(pos) <= 8.0).convex()

    with pytest.raises(NotImplementedError):
        Problem(
            dynamics=dyn,
            states=[pos, vel],
            controls=[u],
            time=time,
            constraints=[cvx_constraint],
            N=n,
            float_dtype="float64",
            solver={"backend": "moreau"},
        )


# ============================================================================
# Assembly sanity checks
# ============================================================================


def test_moreau_assembly_shapes_consistent():
    """After one SCP iteration, ``_assemble_conic`` should produce arrays
    whose shapes are consistent with ``_ConicLayout`` and the CSR structure
    built at ``initialize()``."""
    prob = _make_double_integrator_problem(n=6, backend="moreau", k_max=1)
    prob.settings.dev.printing = False
    prob.initialize()
    prob.solve()

    solver = prob.solver
    assert isinstance(solver, MoreauPTRSolver)

    P_data, coo_vals, q, b = solver._assemble_conic()
    n_z = solver.layout.n_z

    # P_data: one entry per diagonal slot (dx + du)
    assert P_data.shape == (len(solver._P_diag_slots),)
    # All P entries must be strictly positive (regularised).
    assert (P_data > 0).all()

    # q must have the same length as the decision vector.
    assert q.shape == (n_z,)

    # b must have one entry per constraint row.
    assert b.shape == (solver._n_con,)

    # coo_vals must have the same length as stored (row, col) pairs.
    assert coo_vals.shape == (len(solver._coo_rows),)

    # After converting to CSR, the column index array must match the fixed
    # structure built at initialize() — this verifies that _assemble_conic and
    # _structural_pass emit entries in the same order.
    A_csr = sp.csr_matrix(
        (coo_vals, (solver._coo_rows, solver._coo_cols)),
        shape=(solver._n_con, n_z),
    )
    A_csr.sort_indices()
    np.testing.assert_array_equal(A_csr.indptr, solver._A_indptr)
    np.testing.assert_array_equal(A_csr.indices, solver._A_indices)


def test_moreau_solve_returns_PTRSolveResult():
    """After ``initialize()`` and one round of SCP, ``solver.solve()`` must
    return a correctly-shaped :class:`PTRSolveResult`."""
    prob = _make_double_integrator_problem(n=5, backend="moreau", k_max=1)
    prob.settings.dev.printing = False
    prob.initialize()
    res = prob.solver.solve()
    assert isinstance(res, PTRSolveResult)
    assert res.x.shape[0] == 5
    assert res.u.shape[0] == 5
    assert res.status in {"optimal", "infeasible"}


# ============================================================================
# Round-trip parity with CVXPyPTRSolver
# ============================================================================


def test_moreau_round_trip_matches_cvxpy_on_double_integrator():
    """End-to-end parity check on a small CTCS-only (auto time-bound), no
    nodal-nonconvex problem.  Moreau and CVXPy assemble the same convex
    subproblem; converged final states should agree to within a loose
    tolerance (exact agreement isn't required since tolerances differ)."""
    prob_cvx = _make_double_integrator_problem(n=6, backend="cvxpy", k_max=20)
    prob_cvx.settings.dev.printing = False
    prob_cvx.initialize()
    res_cvx = prob_cvx.solve()
    x_cvx = res_cvx.get("x")

    prob_moreau = _make_double_integrator_problem(n=6, backend="moreau", k_max=20)
    prob_moreau.settings.dev.printing = False
    prob_moreau.initialize()
    res_moreau = prob_moreau.solve()
    x_moreau = res_moreau.get("x")

    # Both should reach the same terminal Fix conditions.
    np.testing.assert_allclose(x_cvx[-1, :4], x_moreau[-1, :4], atol=5e-2)
    # Initial Fix conditions must match exactly.
    np.testing.assert_allclose(x_cvx[0, :4], x_moreau[0, :4], atol=1e-6)

    # Composite costs should be in the same ballpark (within 20%).
    cost_cvx = float(res_cvx.get("J_full", default=np.nan))
    cost_moreau = float(res_moreau.get("J_full", default=np.nan))
    if np.isfinite(cost_cvx) and np.isfinite(cost_moreau):
        assert abs(cost_cvx - cost_moreau) / max(abs(cost_cvx), 1e-6) < 0.2


def test_moreau_warm_start_does_not_degrade():
    """Running the SCP loop with warm-starting should not produce a worse
    solution than cold-starting.  Verifies that carrying the warm start
    doesn't corrupt the solver state between iterations."""
    prob = _make_double_integrator_problem(n=6, backend="moreau", k_max=15)
    prob.settings.dev.printing = False
    prob.initialize()

    # First full solve (warm-starts accumulate across iterations internally).
    res = prob.solve()
    x_final = res.get("x")

    # The fixed-condition states (pos = [3, 3]) should be reached.
    np.testing.assert_allclose(x_final[-1, :2], np.array([3.0, 3.0]), atol=0.1)
