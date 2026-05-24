"""``jax.jit(iteration_fn)`` must round-trip equivalently to the bare body.

A divergence here would mean the body either retraces unexpectedly or breaks
pytree structure across the ``jit`` boundary — both fatal for the
``lax.while_loop``-driven solve in follow-up work. Both the next state and the
diagnostics pytree are checked.
"""

import jax
import numpy as np
import pytest

pytest.importorskip("qpax")

from openscvx.algorithms import AlgorithmState
from openscvx.algorithms.scvx.iteration import IterationDiagnostics
from tests.algorithms._iteration_helpers import build_iteration_fn
from tests.solvers._iteration_callback_helpers import build_brachistochrone


def test_jit_matches_bare():
    prob = build_brachistochrone("qpax", n=4, k_max=1)
    prob.initialize()
    state0 = prob.state

    iteration_fn = build_iteration_fn(prob)

    bare_state, bare_diag = iteration_fn(state0, prob._parameters)
    jit_state, jit_diag = jax.jit(iteration_fn)(state0, prob._parameters)

    assert isinstance(jit_state, AlgorithmState)
    assert isinstance(jit_diag, IterationDiagnostics)
    for field in AlgorithmState._FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(jit_state, field)),
            np.asarray(getattr(bare_state, field)),
            atol=1e-8,
            rtol=1e-8,
            equal_nan=True,
        )
    for field in ("cost", "J_lin", "V", "W", "TR", "VC"):
        np.testing.assert_allclose(
            np.asarray(getattr(jit_diag, field)),
            np.asarray(getattr(bare_diag, field)),
            atol=1e-8,
            rtol=1e-8,
        )
