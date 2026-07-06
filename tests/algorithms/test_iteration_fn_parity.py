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

import jax
import numpy as np
import pytest

from openscvx.discretization import get_impulsive_discretization_solver
from tests.algorithms._iteration_helpers import build_iteration_fn
from tests.solvers._iteration_callback_helpers import build_brachistochrone

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize(
    "backend, primal_atol",
    [pytest.param("qpax", 1e-6, marks=pytest.mark.qpax), ("cvxpy", 1e-8)],
)
def test_iteration_fn_matches_production_step(backend, primal_atol):
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


@pytest.mark.parametrize(
    "backend, primal_atol",
    [pytest.param("qpax", 1e-6, marks=pytest.mark.qpax), ("cvxpy", 1e-8)],
)
def test_build_iteration_matches_direct_factory(backend, primal_atol):
    """``Algorithm.build_iteration`` produces the same body as the bare factory.

    Phase 4 moved iteration construction behind the ABC: ``Problem`` no longer
    calls ``make_scp_iteration`` directly but ``self._algorithm.build_iteration``.
    Since that method is a thin wrapper threading ``self.autotuner`` into the
    same factory, the body it returns must be the same function modulo
    construction path. Build both ways from one problem and assert the outputs
    agree on a shared initial iterate.
    """
    prob = build_brachistochrone(backend, n=4, k_max=1)
    prob.initialize()
    state0 = prob.state

    # Bare factory (the test helper calls make_scp_iteration directly).
    direct = build_iteration_fn(prob)

    # Same components routed through the Algorithm ABC contract.
    dis_continuous = jax.jit(prob.discretizer.get_solver(prob.lowered.dynamics, prob.settings))
    dis_impulsive = jax.jit(get_impulsive_discretization_solver(prob.lowered.dynamics_discrete))
    via_abc = prob.algorithm.build_iteration(
        dis_continuous=dis_continuous,
        dis_impulsive=dis_impulsive,
        jax_constraints=prob._compiled_constraints,
        solver_callback=prob.solver.iteration_callback(),
        settings=prob.settings,
    )

    direct_next, direct_diag = direct(state0, prob._parameters)
    abc_next, abc_diag = via_abc(state0, prob._parameters)

    np.testing.assert_allclose(
        np.asarray(abc_next.x), np.asarray(direct_next.x), atol=primal_atol, rtol=primal_atol
    )
    np.testing.assert_allclose(
        np.asarray(abc_next.u), np.asarray(direct_next.u), atol=primal_atol, rtol=primal_atol
    )
    np.testing.assert_allclose(
        np.asarray(abc_diag.V), np.asarray(direct_diag.V), atol=1e-6, rtol=1e-6
    )
    assert int(abc_next.k) == int(direct_next.k)
