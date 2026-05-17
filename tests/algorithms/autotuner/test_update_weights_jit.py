"""``jax.jit(update_weights)`` returns the same state as the bare call.

This is the JAX-traceability acceptance gate for the autotuners: if any
autotuner introduced a Python ``if`` on a tracer value, a list append at
trace time, or a string return, ``jax.jit`` would either error at trace
or silently take a wrong branch. We check structural equality of every
pytree leaf produced by jit'd vs. bare calls.
"""

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.algorithms import (
    AdaptiveProximalWeight,
    AlgorithmState,
    AugmentedLagrangian,
    ConstantProximalWeight,
    RampProximalWeight,
)
from openscvx.algorithms.base import CandidateIterate
from openscvx.algorithms.weights import Weights
from openscvx.config import Config, DevConfig, PropagationConfig, SimConfig
from openscvx.lowered.jax_constraints import LoweredJaxConstraints


# -- Tiny problem fixture ---------------------------------------------------


class _DummyState:
    initial = np.array([0.0, 0.0])
    guess = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    min = np.array([-10.0, -10.0])
    max = np.array([10.0, 10.0])
    final_type = ["Minimize", "None"]
    initial_type = ["None", "None"]
    time_slice = 0
    scaling_min = None
    scaling_max = None


class _DummyControl:
    guess = np.array([[0.0], [0.5], [1.0]])
    min = np.array([-1.0])
    max = np.array([1.0])
    scaling_min = None
    scaling_max = None


@pytest.fixture
def settings():
    sim = SimConfig(
        x=_DummyState(),
        x_prop=_DummyState(),
        u=_DummyControl(),
        total_time=1.0,
        n=3,
        n_states=2,
        n_controls=1,
    )
    return Config(sim=sim, prp=PropagationConfig(), dev=DevConfig())


@pytest.fixture
def weights():
    w = Weights(lam_prox=1.0, lam_vc=1.0, lam_vb=1.0, lam_cost=1.0)
    w.lam_vb_nodal = np.ones((3, 1))
    w.lam_vb_cross = np.ones(1)
    return w


@pytest.fixture
def state(settings, weights):
    # Pre-populate x_prop / x_prop_plus so the autotuner's "previous iterate"
    # branch (k > 1) has finite values to compare against. The actual numbers
    # are arbitrary — we only check jit vs. bare equivalence.
    base = AlgorithmState.from_settings(settings, weights)
    return base.replace(
        x_prop=jnp.asarray(np.array([[0.1, 0.1], [0.9, 0.9]])),
        x_prop_plus=jnp.asarray(np.array([[0.0, 0.0], [0.1, 0.1], [0.9, 0.9]])),
    )


@pytest.fixture
def candidate():
    c = CandidateIterate()
    c.x = jnp.asarray(np.array([[0.0, 0.0], [1.1, 1.1], [2.1, 2.1]]))
    c.u = jnp.asarray(np.array([[0.1], [0.6], [1.1]]))
    c.x_prop = jnp.asarray(np.array([[1.05, 1.05], [2.05, 2.05]]))
    c.x_prop_plus = jnp.asarray(np.array([[0.05, 0.05], [1.05, 1.05], [2.05, 2.05]]))
    c.J_lin = jnp.asarray(1.0)
    return c


@pytest.fixture
def empty_constraints():
    return LoweredJaxConstraints(nodal=[], cross_node=[], ctcs=[])


# -- Pure-vs-jit equivalence ------------------------------------------------


def _states_match(a: AlgorithmState, b: AlgorithmState) -> None:
    for name in AlgorithmState._FIELDS:
        if name == "_FIELDS":
            continue
        np.testing.assert_allclose(
            np.asarray(getattr(a, name)),
            np.asarray(getattr(b, name)),
            err_msg=f"field {name!r} diverged between jit'd and bare call",
            rtol=1e-7,
            atol=1e-7,
        )


def _candidate_to_dict(c: CandidateIterate) -> dict:
    """Lower the mutable candidate dataclass into a pytree-friendly dict.

    The autotuner only reads ``x``, ``u``, ``x_prop``, ``x_prop_plus``, and
    ``J_lin`` from the candidate. Bouncing through a dict keeps
    :class:`CandidateIterate` itself out of the JAX trace boundary (we don't
    want to register it as a pytree yet — that's a downstream concern).
    """
    return {
        "x": c.x,
        "u": c.u,
        "x_prop": c.x_prop,
        "x_prop_plus": c.x_prop_plus,
        "J_lin": c.J_lin,
    }


def _dict_to_candidate(d: dict) -> CandidateIterate:
    c = CandidateIterate()
    for name, value in d.items():
        setattr(c, name, value)
    return c


def _make_jit_target(autotuner, constraints, settings, weights):
    """Wrap update_weights so jit only sees pytree-friendly arguments."""

    def fn(state, candidate_dict):
        cand = _dict_to_candidate(candidate_dict)
        return autotuner.update_weights(state, cand, constraints, settings, {}, weights)

    return jax.jit(fn)


@pytest.mark.parametrize(
    "make_autotuner",
    [
        pytest.param(lambda: AugmentedLagrangian(), id="augmented_lagrangian"),
        pytest.param(lambda: AdaptiveProximalWeight(), id="adaptive_proximal"),
        pytest.param(lambda: ConstantProximalWeight(), id="constant_proximal"),
        pytest.param(
            lambda: RampProximalWeight(ramp_factor=1.2, lam_prox_max=10.0),
            id="ramp_proximal",
        ),
    ],
)
def test_jit_matches_bare_iter1(
    make_autotuner, state, candidate, empty_constraints, settings, weights
):
    """Iteration 1 (INITIAL branch) traces and matches the bare call."""
    autotuner = make_autotuner()
    bare = autotuner.update_weights(
        state, candidate, empty_constraints, settings, {}, weights
    )

    jit_target = _make_jit_target(autotuner, empty_constraints, settings, weights)
    jitted = jit_target(state, _candidate_to_dict(candidate))

    _states_match(bare, jitted)


@pytest.mark.parametrize(
    "make_autotuner",
    [
        pytest.param(lambda: AugmentedLagrangian(), id="augmented_lagrangian"),
        pytest.param(lambda: AdaptiveProximalWeight(), id="adaptive_proximal"),
    ],
)
def test_jit_matches_bare_iter2(
    make_autotuner, state, candidate, empty_constraints, settings, weights
):
    """Iteration k>1 (acceptance-ratio branch) traces and matches the bare call.

    Constant / Ramp autotuners are excluded because they take the same branch
    regardless of ``k``; their iter-1 test above already covers the trace.
    """
    autotuner = make_autotuner()
    state_k2 = state.replace(k=jnp.asarray(2, dtype=jnp.int32))

    bare = autotuner.update_weights(
        state_k2, candidate, empty_constraints, settings, {}, weights
    )

    jit_target = _make_jit_target(autotuner, empty_constraints, settings, weights)
    jitted = jit_target(state_k2, _candidate_to_dict(candidate))

    _states_match(bare, jitted)
