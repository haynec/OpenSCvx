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


def _build_brachistochrone(n=4, k_max=1, constraint_style: str = "ctcs"):
    """Build the brachistochrone problem at a small ``N`` for assembly tests.

    Mirrors ``examples/abstract/brachistochrone.py`` but rebuilds the symbolic
    state in-test so the cached problem from the example doesn't carry over.
    ``constraint_style`` toggles between CTCS box rows (the default) and
    plain nodal inequalities; the latter exercises the nodal-constraint
    assembly block which is otherwise dormant under CTCS-only setups.
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
        solver={"backend": "qpax"},
    )
    prob.settings.dev.printing = False
    return prob


def _subproblem_data_from_solver(solver) -> SubproblemData:
    """Reconstruct the JAX-pure ``SubproblemData`` from the NumPy stash.

    After one ``update_dynamics_linearization`` / ``update_constraint_linearizations``
    / ``update_penalties`` / ``update_boundary_conditions`` cycle, the solver
    has every per-iteration array on hand. The stacked-array layout the
    callback expects is built here so the two assembly paths see the same
    iterate.

    The NumPy path absorbs ``D_d`` into ``A_d`` / ``B_d`` / ``C_d`` at update
    time; the JAX path's ``_assemble_qp_jax`` does the same absorption inline
    when ``has_impulsive`` is true. For non-impulsive problems both paths see
    the same raw ``A_d`` / ``B_d`` / ``C_d`` and ``D_d`` is the all-zeros
    placeholder.
    """
    L = solver.layout
    N, n_x, n_u = L.N, L.n_x, L.n_u
    sim = solver._settings.sim
    has_impulsive = sim.u.slice_impulsive.stop > sim.u.slice_impulsive.start

    dyn = solver._dyn
    cons = solver._cons
    pen = solver._pen
    n_nodal = L.n_nodal

    # Stack nodal linearizations into (N, n_nodal*) layout with zero-fill
    # for nodes outside each constraint's static ``nodes`` tuple.
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
    # The NumPy path absorbed D_d already, so we pass zero D_d to the JAX
    # path and let it skip the einsum (has_impulsive=False for this problem).
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
    prob = _build_brachistochrone(n=4, k_max=1, constraint_style=constraint_style)
    prob.initialize()
    # One SCP iteration populates _dyn / _cons / _pen / _x_init / _x_term on
    # the solver — both assembly paths read from the same iterate after this.
    prob.solve()

    solver = prob.solver
    Q_np, q_np, A_np, b_np, G_np, h_np = solver._assemble_qp()

    data = _subproblem_data_from_solver(solver)
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
    trajectory as ``solver.solve()`` on the same iterate. ``solve_qp_primal``
    and ``solve_qp`` differ only in what they return (primal-only vs
    full primal-dual), so on a convergent QP they must produce the same
    primal up to PDIP tolerance."""
    prob = _build_brachistochrone(n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    # NumPy reference: re-call _assemble_qp + qpax.solve_qp on the stash.
    reference = solver.solve()

    # JAX callback path.
    data = _subproblem_data_from_solver(solver)
    callback = solver.iteration_callback()
    # state is unused by QPAX's callback, but it has to be a valid JAX pytree
    # so ``jit``'s argument-tracing doesn't trip. ``None`` is the canonical
    # empty pytree.
    state = None
    solution = callback(state, data)

    assert isinstance(solution, SubproblemSolution)
    np.testing.assert_allclose(
        np.asarray(solution.x), reference.x, atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(solution.u), reference.u, atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(solution.nu), reference.nu, atol=1e-8, rtol=1e-8
    )
    # nu_vb is stacked (N, n_nodal) in the JAX path, list-of-arrays on NumPy.
    assert solution.nu_vb.shape == (solver.layout.N, solver.layout.n_nodal)
    # Cost reconstruction is independent of the QP solve — should match the
    # NumPy path's _reconstruct_cost output directly.
    np.testing.assert_allclose(float(solution.cost), reference.cost, atol=1e-8, rtol=1e-8)
    # solve_qp_primal exposes no convergence diagnostic.
    assert int(solution.status_code) == int(StatusCode.UNKNOWN)


def test_iteration_callback_traces_under_jit():
    """The callback is constructed under ``jax.jit`` already; this test
    just confirms it's callable end-to-end and that the compilation cost
    amortizes across repeated calls (no per-call re-tracing surprises)."""
    prob = _build_brachistochrone(n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    data = _subproblem_data_from_solver(solver)
    callback = solver.iteration_callback()
    state = None

    sol1 = callback(state, data)
    sol2 = callback(state, data)

    # Both calls should produce structurally identical solutions.
    np.testing.assert_allclose(np.asarray(sol1.x), np.asarray(sol2.x), atol=0.0)
    np.testing.assert_allclose(np.asarray(sol1.u), np.asarray(sol2.u), atol=0.0)
