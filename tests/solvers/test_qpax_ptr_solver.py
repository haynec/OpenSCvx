"""Tests for the QPAX-backed PTR convex subproblem solver.

Covers:
  * Instantiation guard when ``qpax`` isn't installed.
  * ``initialize()`` raises for unsupported feature combinations
    (.convex(), cross-node).
  * End-to-end SCP loop converges on a small CTCS-free LQR-style problem.
  * Round-trip parity vs ``CVXPyPTRSolver`` on the same problem: same final
    trajectory and same final cost to within a loose tolerance.

The brachistochrone parametrized-backend test in ``tests/e2e/test_brachistochrone.py``
exercises QPAX on a richer (nonlinear, CTCS) problem; the unit-style tests
here focus on the API contract and the assembly machinery.
"""

import numpy as np
import pytest

import openscvx as ox
from openscvx import Problem
from openscvx.solvers import PTRSolver, PTRSolveResult, QPAXPTRSolver
from tests.solvers._iteration_callback_helpers import populate_numpy_stash

pytestmark = [pytest.mark.e2e, pytest.mark.qpax]

# ============================================================================
# Helpers
# ============================================================================


def _make_double_integrator_problem(n=6, backend="qpax", k_max=20):
    """2-D double integrator with state/control box bounds — no `.convex()`,
    no cross-node, no impulsive. The library's auto-CTCS for time bounds
    still applies, which is one of the things we're explicitly testing
    QPAX handles."""
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


def test_qpax_solver_is_a_PTRSolver():
    """QPAXPTRSolver must satisfy the abstract PTR contract so it composes
    with the rest of the SCP machinery interchangeably with CVXPyPTRSolver."""
    solver = QPAXPTRSolver()
    assert isinstance(solver, PTRSolver)


def test_qpax_missing_qpax_raises_clear_error(monkeypatch):
    """When qpax isn't installed, instantiation should raise ImportError
    pointing the user at the install command — not an opaque ModuleNotFoundError
    from inside ``solve()``."""
    import openscvx.solvers.qpax_ptr_solver as mod

    monkeypatch.setattr(mod, "_QPAX_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"pip install openscvx\[qpax\]"):
        QPAXPTRSolver()


def test_qpax_spec_rejects_cvxpy_only_fields():
    """The PTRSolverSpec validator should reject cvx_solver/cvxpygen under
    backend='qpax' so users get a config-time error rather than a confusing
    runtime one."""
    from openscvx.solvers import resolve_solver_config

    with pytest.raises(ValueError, match="only valid for backend='cvxpy'"):
        resolve_solver_config({"backend": "qpax", "cvxpygen": True})

    with pytest.raises(ValueError, match="only valid for backend='cvxpy'"):
        resolve_solver_config({"backend": "qpax", "cvx_solver": "CLARABEL"})


def test_qpax_named_params_populate_solver_args():
    """Named constructor params should be merged into solver_args so the
    existing qpax.solve_qp(**self.solver_args) dispatch picks them up."""
    solver = QPAXPTRSolver(solver_tol=1e-8, max_iter=50)
    assert solver.solver_args["solver_tol"] == 1e-8
    assert solver.solver_args["max_iter"] == 50


def test_qpax_named_param_overlap_with_solver_args_raises():
    """Passing the same key as a named arg and inside solver_args is a user
    error — the constructor should raise immediately rather than silently
    discarding one value."""
    with pytest.raises(ValueError, match="solver_tol"):
        QPAXPTRSolver(solver_tol=1e-8, solver_args={"solver_tol": 1e-7})

    with pytest.raises(ValueError, match="max_iter"):
        QPAXPTRSolver(max_iter=50, solver_args={"max_iter": 100})


def test_qpax_solver_args_escape_hatch():
    """Keys not covered by named params (e.g. backend='e') should pass
    through the solver_args escape hatch unchanged."""
    solver = QPAXPTRSolver(solver_tol=1e-6, solver_args={"backend": "e"})
    assert solver.solver_args["backend"] == "e"
    assert solver.solver_args["solver_tol"] == 1e-6


def test_qpax_spec_named_fields_build_correctly():
    """PTRSolverSpec with QPAX named fields should build a QPAXPTRSolver
    with the right solver_args."""
    from openscvx.solvers import QPAXPTRSolver, resolve_solver_config

    spec = resolve_solver_config({"backend": "qpax", "solver_tol": 1e-8, "max_iter": 75})
    solver = spec.build()
    assert isinstance(solver, QPAXPTRSolver)
    assert solver.solver_args["solver_tol"] == 1e-8
    assert solver.solver_args["max_iter"] == 75


def test_qpax_spec_rejects_moreau_only_fields():
    """Moreau-specific fields (verbose, device, tol_gap_abs, tol_feas) must
    not be accepted under backend='qpax'."""
    from openscvx.solvers import resolve_solver_config

    with pytest.raises(ValueError, match="only valid for backend='moreau'"):
        resolve_solver_config({"backend": "qpax", "verbose": True})

    with pytest.raises(ValueError, match="only valid for backend='moreau'"):
        resolve_solver_config({"backend": "qpax", "tol_gap_abs": 1e-10})


def test_qpax_spec_rejects_solver_tol_under_moreau():
    """solver_tol must not be accepted under backend='moreau'."""
    from openscvx.solvers import resolve_solver_config

    with pytest.raises(ValueError, match="solver_tol"):
        resolve_solver_config({"backend": "moreau", "solver_tol": 1e-8})


def test_cvxpy_spec_rejects_jax_backend_fields():
    """QPAX/Moreau named fields (solver_tol, max_iter, verbose, device, …)
    must not be accepted under backend='cvxpy'."""
    from openscvx.solvers import resolve_solver_config

    with pytest.raises(ValueError, match="not valid for backend='cvxpy'"):
        resolve_solver_config({"backend": "cvxpy", "solver_tol": 1e-8})

    with pytest.raises(ValueError, match="not valid for backend='cvxpy'"):
        resolve_solver_config({"backend": "cvxpy", "max_iter": 100})


# ============================================================================
# Unsupported-feature rejection tests
# ============================================================================


def test_qpax_rejects_user_convex_constraints():
    """User .convex() norm constraints canonicalise to SOCConstraint.
    QPAXPTRSolver's ``SUPPORTED_CONE_TYPES`` excludes SOC, so lowering must
    raise ``NotImplementedError`` with a message naming the unsupported cone
    and suggesting MoreauPTRSolver as an alternative."""
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

    # User-defined convex (nodal) — a 2-norm ball, which lowers to a
    # second-order cone and therefore can't fit in QP form. We use a Norm
    # rather than an affine inequality because the .convex() categorizer
    # may inline trivially-affine constraints into the box pipeline.
    cvx_constraint = (ox.linalg.Norm(pos) <= 8.0).convex()

    with pytest.raises(
        NotImplementedError,
        match=r"QPAXPTRSolver does not support SOCConstraint",
    ):
        Problem(
            dynamics=dyn,
            states=[pos, vel],
            controls=[u],
            time=time,
            constraints=[cvx_constraint],
            N=n,
            float_dtype="float64",
            solver={"backend": "qpax"},
        )


def test_qpax_rejects_nodal_equality():
    """L1-penalized nodal equalities need a two-sided slack penalty, which the
    QPAX one-sided positive-part reformulation can't express; lowering must
    raise ``NotImplementedError`` naming CVXPyPTRSolver."""
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

    eq_constraint = (pos == np.array([1.0, 1.0])).at([2])

    with pytest.raises(
        NotImplementedError,
        match=r"QPAXPTRSolver does not support L1-penalized equality",
    ):
        Problem(
            dynamics=dyn,
            states=[pos, vel],
            controls=[u],
            time=time,
            constraints=[eq_constraint],
            N=n,
            float_dtype="float64",
            solver={"backend": "qpax"},
        )


# ============================================================================
# Round-trip parity with CVXPyPTRSolver
# ============================================================================


def test_qpax_round_trip_matches_cvxpy_on_double_integrator():
    """End-to-end parity check on a small CTCS-only (auto time-bound), no
    nodal-nonconvex problem. The QP backend assembles the same convex
    subproblem the CVXPy backend does, so converged costs and final states
    should agree to a few significant digits."""
    prob_cvx = _make_double_integrator_problem(n=6, backend="cvxpy", k_max=20)
    prob_cvx.settings.dev.printing = False
    prob_cvx.initialize()
    res_cvx = prob_cvx.solve()
    x_cvx = res_cvx.get("x")
    cost_cvx = float(res_cvx.get("J_full", default=np.nan))  # final composite cost

    prob_qpax = _make_double_integrator_problem(n=6, backend="qpax", k_max=20)
    prob_qpax.settings.dev.printing = False
    prob_qpax.initialize()
    res_qpax = prob_qpax.solve()
    x_qpax = res_qpax.get("x")
    cost_qpax = float(res_qpax.get("J_full", default=np.nan))

    # Final state should hit the same target (loose tol; QPAX uses 1e-5
    # default tolerance and we don't pass tighter args here).
    np.testing.assert_allclose(x_cvx[-1, :4], x_qpax[-1, :4], atol=1e-2)
    # Initial state should match (both pinned to Fix initial).
    np.testing.assert_allclose(x_cvx[0, :4], x_qpax[0, :4], atol=1e-6)

    # Costs may differ by a few percent because each backend's SCP path
    # accepts/rejects iterations independently. Sanity-check they're in
    # the same ballpark rather than asserting equality.
    if np.isfinite(cost_cvx) and np.isfinite(cost_qpax):
        assert abs(cost_cvx - cost_qpax) / max(abs(cost_cvx), 1e-6) < 0.2


# ============================================================================
# QP-assembly sanity checks
# ============================================================================


def test_qpax_assembly_produces_consistent_shapes():
    """After one SCP iteration's worth of updates, _assemble_qp should
    produce (Q, q, A, b, G, h) of shapes consistent with the declared
    decision-vector layout."""
    prob = _make_double_integrator_problem(n=6, backend="qpax", k_max=1)
    prob.settings.dev.printing = False
    prob.initialize()
    # Populate _dyn / _cons / _pen / _x_init / _x_term for the NumPy assembly
    # (the SCP loop no longer drives the update_* path).
    populate_numpy_stash(prob)

    solver = prob.solver
    Q, q, A, b, G, h = solver._assemble_qp()
    n_z = solver.layout.n_z

    assert Q.shape == (n_z, n_z)
    assert q.shape == (n_z,)
    assert A.shape[1] == n_z and A.shape[0] == b.shape[0]
    assert G.shape[1] == n_z and G.shape[0] == h.shape[0]

    # Q should be symmetric (it's diagonal — only the trust-region terms
    # contribute) and have only nonneg diagonal entries.
    assert np.allclose(Q, Q.T)
    assert (np.diag(Q) >= 0).all()


def test_qpax_solve_returns_PTRSolveResult():
    prob = _make_double_integrator_problem(n=5, backend="qpax", k_max=1)
    prob.settings.dev.printing = False
    prob.initialize()
    # One direct solver.solve() call after the NumPy stash is populated.
    populate_numpy_stash(prob)
    res = prob.solver.solve()
    assert isinstance(res, PTRSolveResult)
    assert res.x.shape[0] == 5
    assert res.u.shape[0] == 5
    assert res.status in {"optimal", "infeasible"}


# ============================================================================
# Convergence-failure guard
# ============================================================================


def test_qpax_solve_raises_on_nonconvergence(monkeypatch):
    """When ``qpax.solve_qp`` returns ``converged=False`` (typical under
    float32 ill-conditioning) the backend must raise rather than unpack a
    NaN-filled primal. Without the guard the next SCP linearization point
    is NaN-poisoned and every subsequent iteration produces garbage. The
    end-to-end coverage in ``tests/e2e/test_brachistochrone.py::test_backend_float32_raises``
    exercises the same path through the SCvx loop; this unit test pins the
    behavior at the solver boundary directly."""
    import jax.numpy as jnp

    import openscvx.solvers.qpax_ptr_solver as mod

    prob = _make_double_integrator_problem(n=5, backend="qpax", k_max=1)
    prob.settings.dev.printing = False
    prob.initialize()
    # Populate the stash so solver.solve() reaches the qpax.solve_qp guard
    # (rather than the "update_* not called" precondition error).
    populate_numpy_stash(prob)

    n_z = prob.solver.layout.n_z
    nan_z = jnp.full((n_z,), jnp.nan)

    def fake_solve_qp(Q, q, A, b, G, h, **kwargs):
        del Q, q, A, b, G, h, kwargs
        return nan_z, nan_z, nan_z, nan_z, jnp.bool_(False), jnp.int32(5)

    monkeypatch.setattr(mod.qpax, "solve_qp", fake_solve_qp)

    with pytest.raises(RuntimeError, match=r"qpax\.solve_qp failed"):
        prob.solver.solve()
