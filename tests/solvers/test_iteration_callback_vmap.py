"""``jax.vmap`` composition for QPAX and Moreau iteration callbacks.

Each backend's per-iteration callback must compose with ``jax.vmap`` and
produce per-element-correct outputs over a stack of distinct
``SubproblemData`` inputs — the precondition for the downstream
``jax.vmap(problem.solve)`` path in ``plans/batchable-problem.md``.

Per-element correctness is verified by perturbing ``lam_prox`` across batch
elements (the cheapest knob that meaningfully changes the QP / cone solution
without altering its sparsity pattern). Each batched output slice must match
a bare call on the corresponding unbatched ``SubproblemData``.

CVXPy's vmap composition lives in ``test_iteration_callback_cvxpy.py`` with
``vmap_method="sequential"`` — it can't ingest a batched parameter set, so
the pattern there is structurally different from QPAX / Moreau and stays in
its own module.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from openscvx import Problem
from openscvx.solvers.ptr_solver import SubproblemData, SubproblemSolution

from tests._marks import requires_moreau


# ============================================================================
# Fixtures (mirror tests/solvers/test_iteration_callback_{qpax,moreau}.py)
# ============================================================================


def _build_brachistochrone(backend: str, n: int = 4, k_max: int = 1):
    """Build the brachistochrone problem with the named backend.

    Kept in-test rather than imported from the per-backend test modules so
    each file remains independently runnable.
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
    for state in states:
        constraint_exprs.extend(
            [ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)]
        )

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
        solver={"backend": backend},
    )
    prob.settings.dev.printing = False
    return prob


def _subproblem_data_from_solver(solver) -> SubproblemData:
    """Reconstruct the JAX-pure ``SubproblemData`` from the NumPy stash.

    Identical to the helper in ``test_iteration_callback_{qpax,moreau}.py``;
    duplicated here to keep this module independently runnable.
    """
    L = solver.layout
    N, n_x, n_u = L.N, L.n_x, L.n_u
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


def _make_batch(data: SubproblemData, scales) -> SubproblemData:
    """Stack ``data`` ``B`` times, perturbing ``lam_prox`` by ``scales[b]``.

    Perturbing a penalty weight rather than a structural array keeps every
    batch element a well-posed subproblem with the same sparsity pattern,
    while guaranteeing the optimum genuinely differs across batch elements
    (otherwise the test would only verify broadcasting, not per-element
    correctness under vmap).
    """
    scales = jnp.asarray(scales)
    B = scales.shape[0]

    def stack_lam_prox(lam_prox):
        # (B, 1, 1) * (N, n_x+n_u) -> (B, N, n_x+n_u)
        return scales[:, None, None] * jnp.broadcast_to(lam_prox, (B,) + lam_prox.shape)

    def stack_other(leaf):
        return jnp.broadcast_to(leaf, (B,) + leaf.shape)

    leaves = {
        f.name: getattr(data, f.name) for f in data.__dataclass_fields__.values()
    }
    leaves["lam_prox"] = stack_lam_prox(leaves["lam_prox"])
    for name in list(leaves):
        if name == "lam_prox":
            continue
        leaves[name] = stack_other(leaves[name])
    return SubproblemData(**leaves)


# ============================================================================
# QPAX
# ============================================================================


pytest.importorskip("qpax")


def test_qpax_iteration_callback_composes_with_vmap():
    """``jax.vmap(cb)`` over a batch of distinct ``SubproblemData`` must
    yield per-element-correct ``SubproblemSolution`` slices.

    Batch elements differ by a ``lam_prox`` scale — the cheapest perturbation
    that meaningfully changes the QP solution. Each batched output slice
    must match a bare call on the corresponding unbatched data within PDIP
    tolerance.
    """
    prob = _build_brachistochrone("qpax", n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    base = _subproblem_data_from_solver(solver)
    scales = jnp.array([0.5, 1.0, 1.5, 2.0])
    batch = _make_batch(base, scales)

    callback = solver.iteration_callback()
    batched = jax.vmap(callback, in_axes=(None, 0))(None, batch)

    assert isinstance(batched, SubproblemSolution)
    assert batched.x.shape[0] == scales.shape[0]

    for i, s in enumerate(scales):
        per_element_data = SubproblemData(
            **{
                **{f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()},
                "lam_prox": float(s) * base.lam_prox,
            }
        )
        bare = callback(None, per_element_data)
        np.testing.assert_allclose(
            np.asarray(batched.x[i]), np.asarray(bare.x), atol=1e-8, rtol=1e-8
        )
        np.testing.assert_allclose(
            np.asarray(batched.u[i]), np.asarray(bare.u), atol=1e-8, rtol=1e-8
        )
        np.testing.assert_allclose(
            float(batched.cost[i]), float(bare.cost), atol=1e-8, rtol=1e-8
        )


# ============================================================================
# Moreau (gated on license availability)
# ============================================================================


@requires_moreau
def test_moreau_iteration_callback_composes_with_vmap():
    """Same contract as the QPAX variant — ``jax.vmap`` over distinct
    ``SubproblemData`` instances yields per-element-correct solutions.

    Moreau's functional API supports batched solves natively (per the docs),
    so unlike CVXPy this path is genuinely vectorized rather than
    sequentially fanned out. Each batch element solves an independent conic
    program; per-element results must match unbatched calls.
    """
    prob = _build_brachistochrone("moreau", n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    base = _subproblem_data_from_solver(solver)
    scales = jnp.array([0.5, 1.0, 1.5, 2.0])
    batch = _make_batch(base, scales)

    callback = solver.iteration_callback()
    batched = jax.vmap(callback, in_axes=(None, 0))(None, batch)

    assert isinstance(batched, SubproblemSolution)
    assert batched.x.shape[0] == scales.shape[0]

    for i, s in enumerate(scales):
        per_element_data = SubproblemData(
            **{
                **{f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()},
                "lam_prox": float(s) * base.lam_prox,
            }
        )
        bare = callback(None, per_element_data)
        np.testing.assert_allclose(
            np.asarray(batched.x[i]), np.asarray(bare.x), atol=1e-7, rtol=1e-7
        )
        np.testing.assert_allclose(
            np.asarray(batched.u[i]), np.asarray(bare.u), atol=1e-7, rtol=1e-7
        )
        np.testing.assert_allclose(
            float(batched.cost[i]), float(bare.cost), atol=1e-7, rtol=1e-7
        )
