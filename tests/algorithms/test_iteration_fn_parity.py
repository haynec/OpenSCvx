"""A factory-built ``make_scp_iteration`` body matches the production ``.step()``.

One call to a standalone ``iteration_fn`` must reproduce one
:meth:`PenalizedTrustRegion.step` — same next-iterate trajectory, same SCP
convergence metrics, and the same per-iteration diagnostics (raw
discretization matrix, trust-region / virtual-control matrices) the step
records into history. Since ``.step()`` now drives the same fused body, this is
a consistency gate between the test-helper build (bare body, no-disk
``jax.jit`` discretization) and the production wiring (``jax.jit``'d body built
in :meth:`Problem.initialize`), plus a check that the diagnostics aux-return
feeds history correctly.

QPAX matches only to ~1e-6 because its iterative ``qpax.solve_qp`` is sensitive
to the jit-vs-bare floating-point reordering between the two builds; CVXPy
matches to 1e-8 because its ``pure_callback`` host QOCO solve is bit-identical
across both.
"""

import numpy as np
import pytest

from tests.algorithms._iteration_helpers import build_iteration_fn
from tests.solvers._iteration_callback_helpers import build_brachistochrone


@pytest.mark.parametrize("backend, primal_atol", [("qpax", 1e-6), ("cvxpy", 1e-8)])
def test_iteration_fn_matches_production_step(backend, primal_atol):
    if backend == "qpax":
        pytest.importorskip("qpax")
    prob = build_brachistochrone(backend, n=4, k_max=1)
    prob.initialize()

    # Both paths start from the same fresh initial iterate. ``step`` does not
    # mutate the (frozen) state it is handed, so we can reuse it.
    state0 = prob.state

    iteration_fn = build_iteration_fn(prob)
    it_next, it_diag = iteration_fn(state0, prob._parameters)

    step_next, _ = prob.algorithm.step(state0, prob.history, prob._parameters, prob.settings)

    # Next-iterate trajectory + metrics.
    np.testing.assert_allclose(
        np.asarray(it_next.x), np.asarray(step_next.x), atol=primal_atol, rtol=primal_atol
    )
    np.testing.assert_allclose(
        np.asarray(it_next.u), np.asarray(step_next.u), atol=primal_atol, rtol=primal_atol
    )
    np.testing.assert_allclose(
        np.asarray(it_next.x_prop), np.asarray(step_next.x_prop), atol=1e-5, rtol=1e-5
    )
    for metric in ("J_tr", "J_vb", "J_vc"):
        np.testing.assert_allclose(
            float(getattr(it_next, metric)), float(getattr(step_next, metric)), atol=1e-4
        )
    assert int(it_next.k) == int(step_next.k)

    # Diagnostics must match what the step recorded into history
    # (ConstantProximalWeight always accepts, so TR / VC / V are appended).
    np.testing.assert_allclose(np.asarray(it_diag.TR), prob.history.TR[-1], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(it_diag.VC), prob.history.VC[-1], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(it_diag.V), prob.history.V_history[-1], atol=1e-6, rtol=1e-6
    )
