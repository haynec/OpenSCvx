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

from tests._marks import requires_moreau, _MOREAU_OK

pytestmark = requires_moreau

# Imports below are only reached when _MOREAU_OK is True.
if _MOREAU_OK:
    import jax
    import jax.numpy as jnp

    from openscvx import Problem
    from openscvx.solvers.ptr_solver import (
        StatusCode,
        SubproblemData,
        SubproblemSolution,
    )


# ============================================================================
# Fixtures
# ============================================================================


def _build_brachistochrone(n: int = 4, k_max: int = 1, constraint_style: str = "ctcs"):
    """Build the brachistochrone problem with the Moreau backend.

    Mirrors the QPAX iteration-callback test fixture so the two backends are
    exercised on the same problem shape — the only delta is ``backend="moreau"``
    and the constraint-row encoding choice (Moreau uses SOC epigraphs for
    ``|nu|`` where QPAX uses paired nonneg slacks).
    """
    import openscvx as ox

    g = 9.81

    position = ox.State("position", shape=(2,))
    position.max = np.array([10.0, 10.0])
    position.min = np.array([0.0, 0.0])
    position.initial = np.array([0.0, 10.0])
    position.final = [10.0, 5.0]

    velocity = ox.State("velocity", shape=(1,))
    velocity.max = np.array([10.0])
    velocity.min = np.array([0.0])
    velocity.initial = np.array([0.0])
    velocity.final = [("free", 10.0)]

    theta = ox.Control("theta", shape=(1,))
    theta.max = np.array([100.5 * jnp.pi / 180])
    theta.min = np.array([0.0])
    theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    states = [position, velocity]
    controls = [theta]

    dynamics = {
        "position": ox.Concat(
            velocity[0] * ox.Sin(theta[0]),
            -velocity[0] * ox.Cos(theta[0]),
        ),
        "velocity": g * ox.Cos(theta[0]),
    }

    constraint_exprs = []
    if constraint_style == "ctcs":
        for state in states:
            constraint_exprs.extend(
                [ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)]
            )
    elif constraint_style == "nodal":
        for state in states:
            constraint_exprs.extend([state <= state.max, state.min <= state])
    else:
        raise ValueError(f"unknown constraint_style {constraint_style!r}")

    time = ox.Time(
        initial=0.0,
        final=("minimize", 2.0),
        min=0.0,
        max=2.0,
        uniform_time_grid=True,
    )

    prob = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraint_exprs,
        N=n,
        float_dtype="float64",
        algorithm={
            "autotuner": "ConstantProximalWeight",
            "lam_prox": 1e0,
            "lam_cost": 6e-1,
            "k_max": k_max,
        },
        solver={"backend": "moreau"},
    )
    prob.settings.dev.printing = False
    return prob


def _subproblem_data_from_solver(solver) -> "SubproblemData":
    """Reconstruct the JAX-pure :class:`SubproblemData` from the NumPy stash.

    Mirrors the QPAX-side helper: after one ``update_*`` cycle the solver
    has every per-iteration array on hand, and the stacked-array layout
    required by the callback is built here so both assembly paths see the
    identical iterate.
    """
    L = solver.layout
    N, n_x, n_u = L.N, L.n_x, L.n_u
    sim = solver._settings.sim
    has_impulsive = sim.u.slice_impulsive.stop > sim.u.slice_impulsive.start

    dyn = solver._dyn
    cons = solver._cons
    pen = solver._pen
    n_nodal = L.n_nodal

    nodal_g = np.zeros((N, max(n_nodal, 1)), dtype=float)
    nodal_grad_x = np.zeros((N, max(n_nodal, 1), n_x), dtype=float)
    nodal_grad_u = np.zeros((N, max(n_nodal, 1), n_u), dtype=float)
    for c_idx, (constraint, entry) in enumerate(
        zip(solver._jax_constraints.nodal, cons.get("nodal", []))
    ):
        for node in constraint.nodes:
            nodal_g[node, c_idx] = entry["g"][node]
            nodal_grad_x[node, c_idx] = entry["grad_g_x"][node]
            nodal_grad_u[node, c_idx] = entry["grad_g_u"][node]
    if n_nodal == 0:
        nodal_g = np.zeros((N, 0))
        nodal_grad_x = np.zeros((N, 0, n_x))
        nodal_grad_u = np.zeros((N, 0, n_u))

    x_prop_plus = (
        dyn["x_prop_plus"] if dyn["x_prop_plus"] is not None else np.zeros((N, n_x))
    )
    E_d = dyn["E_d"] if dyn["E_d"] is not None else np.zeros((N, n_x, n_u))
    # NumPy path already absorbed D_d into A_d/B_d/C_d; pass zero D_d so the
    # JAX path's einsum (gated on has_impulsive) is skipped via the matching
    # static-Python branch — has_impulsive is False for this problem.
    D_d = np.zeros((N, n_x, n_x))

    x_init = solver._x_init if solver._x_init is not None else np.full(n_x, np.nan)
    x_term = solver._x_term if solver._x_term is not None else np.full(n_x, np.nan)

    return SubproblemData(
        x_bar=jnp.asarray(dyn["x_bar"]),
        u_bar=jnp.asarray(dyn["u_bar"]),
        A_d=jnp.asarray(dyn["A_d"]),
        B_d=jnp.asarray(dyn["B_d"]),
        C_d=jnp.asarray(dyn["C_d"]),
        x_prop=jnp.asarray(dyn["x_prop"]),
        x_prop_plus=jnp.asarray(x_prop_plus),
        D_d=jnp.asarray(D_d),
        E_d=jnp.asarray(E_d),
        nodal_g=jnp.asarray(nodal_g),
        nodal_grad_x=jnp.asarray(nodal_grad_x),
        nodal_grad_u=jnp.asarray(nodal_grad_u),
        cross_g=jnp.zeros((0,)),
        cross_grad_X=jnp.zeros((0, N, n_x)),
        cross_grad_U=jnp.zeros((0, N, n_u)),
        lam_prox=jnp.asarray(pen["lam_prox"]),
        lam_cost=jnp.asarray(pen["lam_cost"]),
        lam_vc=jnp.asarray(pen["lam_vc"]),
        lam_vb_nodal=jnp.asarray(pen["lam_vb_nodal"]),
        lam_vb_cross=jnp.zeros((0,)),
        x_init=jnp.asarray(x_init),
        x_term=jnp.asarray(x_term),
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
    prob = _build_brachistochrone(n=4, k_max=1, constraint_style=constraint_style)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    P_np, coo_np, q_np, b_np = solver._assemble_conic()

    data = _subproblem_data_from_solver(solver)
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

    prob = _build_brachistochrone(n=4, k_max=1)
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
    prob = _build_brachistochrone(n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    reference = solver.solve()

    data = _subproblem_data_from_solver(solver)
    callback = solver.iteration_callback()
    solution = callback(None, data)

    assert isinstance(solution, SubproblemSolution)
    np.testing.assert_allclose(
        np.asarray(solution.x), reference.x, atol=1e-7, rtol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(solution.u), reference.u, atol=1e-7, rtol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(solution.nu), reference.nu, atol=1e-7, rtol=1e-7
    )
    assert solution.nu_vb.shape == (solver.layout.N, solver.layout.n_nodal)
    np.testing.assert_allclose(
        float(solution.cost), reference.cost, atol=1e-7, rtol=1e-7
    )
    # Optimal solves should map to StatusCode.OPTIMAL.
    assert int(solution.status_code) == int(StatusCode.OPTIMAL)
    # moreau_carry is a zero-length triple — placeholder, not consumed.
    for leaf in solution.moreau_carry:
        assert leaf.shape == (0,)


def test_iteration_callback_traces_under_jit():
    """The callback is constructed under ``jax.jit`` already; this confirms
    it's callable end-to-end and that repeated calls share the compiled
    trace (no per-call retrace surprises)."""
    prob = _build_brachistochrone(n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    data = _subproblem_data_from_solver(solver)
    callback = solver.iteration_callback()

    sol1 = callback(None, data)
    sol2 = callback(None, data)

    np.testing.assert_allclose(np.asarray(sol1.x), np.asarray(sol2.x), atol=0.0)
    np.testing.assert_allclose(np.asarray(sol1.u), np.asarray(sol2.u), atol=0.0)
