"""``jax.jit(iteration_fn)`` must round-trip equivalently to the bare body.

A divergence here would mean the body either retraces unexpectedly or breaks
pytree structure across the ``jit`` boundary — both fatal for the
``lax.while_loop``-driven solve in follow-up work.
"""

import jax
import numpy as np
import pytest

pytest.importorskip("qpax")

from openscvx.algorithms import AlgorithmState
from openscvx.algorithms.scvx.iteration import make_scp_iteration
from tests.solvers._iteration_callback_helpers import build_brachistochrone


def test_jit_matches_bare():
    prob = build_brachistochrone("qpax", n=4, k_max=1)
    prob.initialize()
    state0 = prob.state

    iteration_fn = make_scp_iteration(
        dynamics=prob.lowered.dynamics,
        dynamics_discrete=prob.lowered.dynamics_discrete,
        jax_constraints=prob._compiled_constraints,
        discretizer=prob.discretizer,
        solver_callback=prob.solver.iteration_callback(),
        autotuner=prob.algorithm.autotuner,
        settings=prob.settings,
    )

    bare = iteration_fn(state0, prob._parameters)
    jitted = jax.jit(iteration_fn)(state0, prob._parameters)

    assert isinstance(jitted, AlgorithmState)
    for field in AlgorithmState._FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(jitted, field)),
            np.asarray(getattr(bare, field)),
            atol=1e-8,
            rtol=1e-8,
            equal_nan=True,
        )
