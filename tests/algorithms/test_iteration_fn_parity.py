"""Per-iteration parity between ``make_scp_iteration`` and the legacy path.

One call to the fused JAX iteration body must reproduce one
:meth:`PenalizedTrustRegion.step` — same next-iterate trajectory, same SCP
convergence metrics — on a fixed brachistochrone. This is the acceptance gate
for the body before it replaces the Python-side ``_subproblem`` stitching.

QPAX exercises the fully JAX-native solve (``solve_qp_primal`` vs the NumPy
``solve_qp``, hence the looser primal tolerance); CVXPy exercises the
``pure_callback`` host solve, which runs the identical QOCO solve as the legacy
path and so matches tightly.
"""

import numpy as np
import pytest

from openscvx.algorithms.scvx.iteration import make_scp_iteration
from tests.solvers._iteration_callback_helpers import build_brachistochrone


def _build_iteration_fn(prob):
    """Assemble the iteration body from an initialized problem's components."""
    return make_scp_iteration(
        dynamics=prob.lowered.dynamics,
        dynamics_discrete=prob.lowered.dynamics_discrete,
        jax_constraints=prob._compiled_constraints,
        discretizer=prob.discretizer,
        solver_callback=prob.solver.iteration_callback(),
        autotuner=prob.algorithm.autotuner,
        settings=prob.settings,
    )


@pytest.mark.parametrize("backend, primal_atol", [("qpax", 1e-6), ("cvxpy", 1e-8)])
def test_iteration_fn_matches_subproblem(backend, primal_atol):
    if backend == "qpax":
        pytest.importorskip("qpax")
    prob = build_brachistochrone(backend, n=4, k_max=1)
    prob.initialize()

    # Both paths start from the same fresh initial iterate. ``step`` does not
    # mutate the (frozen) state it is handed, so we can reuse it.
    state0 = prob.state

    iteration_fn = _build_iteration_fn(prob)
    it_next = iteration_fn(state0, prob._parameters)

    legacy_next, _ = prob.algorithm.step(
        state0, prob.history, prob._parameters, prob.settings
    )

    np.testing.assert_allclose(
        np.asarray(it_next.x), np.asarray(legacy_next.x), atol=primal_atol, rtol=primal_atol
    )
    np.testing.assert_allclose(
        np.asarray(it_next.u), np.asarray(legacy_next.u), atol=primal_atol, rtol=primal_atol
    )
    np.testing.assert_allclose(
        np.asarray(it_next.x_prop), np.asarray(legacy_next.x_prop), atol=1e-5, rtol=1e-5
    )
    for metric in ("J_tr", "J_vb", "J_vc"):
        np.testing.assert_allclose(
            float(getattr(it_next, metric)), float(getattr(legacy_next, metric)), atol=1e-4
        )
    assert int(it_next.k) == int(legacy_next.k)
