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

from openscvx.solvers.ptr_solver import SubproblemData, SubproblemSolution

from tests._marks import requires_moreau
from tests.solvers._iteration_callback_helpers import (
    build_brachistochrone,
    subproblem_data_from_numpy_stash,
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
    prob = build_brachistochrone("qpax", n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    base = subproblem_data_from_numpy_stash(solver)
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
    prob = build_brachistochrone("moreau", n=4, k_max=1)
    prob.initialize()
    prob.solve()
    solver = prob.solver

    base = subproblem_data_from_numpy_stash(solver)
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
