"""JAX pytree contract tests for :class:`AlgorithmState`.

These tests are the structural acceptance gate for the pytree split: the SCP
loop body (and any future ``jax.vmap`` / ``lax.while_loop`` wrapping) treats
``AlgorithmState`` as an opaque pytree. If flatten/unflatten or ``replace``
breaks, the higher-level JAX composition silently breaks too.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.algorithms import AdaptiveStateCode, AlgorithmState
from openscvx.algorithms.weights import Weights
from openscvx.config import Config, DevConfig, PropagationConfig, SimConfig


class _DummyState:
    initial = np.array([0.0, 0.0])
    final = np.array([0.0, 0.0])
    guess = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    min = np.array([-10.0, -10.0])
    max = np.array([10.0, 10.0])
    final_type = ["None", "None"]
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
def state():
    sim = SimConfig(
        x=_DummyState(),
        x_prop=_DummyState(),
        u=_DummyControl(),
        total_time=1.0,
        n=3,
        n_states=2,
        n_controls=1,
    )
    settings = Config(sim=sim, prp=PropagationConfig(), dev=DevConfig())
    weights = Weights(lam_prox=1.0, lam_vc=1.0, lam_vb=1.0, lam_cost=1.0)
    weights.lam_vb_nodal = np.ones((3, 1))
    weights.lam_vb_cross = np.ones(1)
    return AlgorithmState.from_settings(
        settings,
        weights,
        ep_tr=1e-4,
        ep_vb=1e-4,
        ep_vc=1e-8,
        k_max=200,
        lam_cost_drop=-1,
    )


# === pytree registration ===================================================


def test_flatten_unflatten_roundtrip(state):
    leaves, treedef = jax.tree_util.tree_flatten(state)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, AlgorithmState)
    for name in AlgorithmState._FIELDS:
        original = getattr(state, name)
        if name == "_FIELDS":  # not a leaf — sanity guard
            continue
        np.testing.assert_array_equal(np.asarray(getattr(rebuilt, name)), np.asarray(original))


def test_tree_map_preserves_shape(state):
    doubled = jax.tree_util.tree_map(lambda leaf: leaf * 2, state)
    assert isinstance(doubled, AlgorithmState)
    # Shape preserved across all leaves
    leaves_orig = jax.tree_util.tree_leaves(state)
    leaves_doubled = jax.tree_util.tree_leaves(doubled)
    for a, b in zip(leaves_orig, leaves_doubled):
        assert a.shape == b.shape
        assert a.dtype == b.dtype


# === replace() semantics ===================================================


def test_replace_returns_new_instance(state):
    new = state.replace(k=jnp.asarray(42, dtype=jnp.int32))
    assert new is not state
    assert int(new.k) == 42
    # Original is untouched (frozen).
    assert int(state.k) == 1


def test_replace_preserves_unchanged_fields(state):
    new = state.replace(J_tr=jnp.asarray(1.5))
    np.testing.assert_array_equal(np.asarray(new.x), np.asarray(state.x))
    np.testing.assert_array_equal(np.asarray(new.lam_prox), np.asarray(state.lam_prox))
    assert float(new.J_tr) == 1.5


def test_frozen_dataclass_disallows_mutation(state):
    with pytest.raises((AttributeError, Exception)):
        state.k = jnp.asarray(99, dtype=jnp.int32)


# === Initial state seeded with INITIAL adaptive-state code ==================


def test_from_settings_initial_adaptive_state(state):
    assert int(state.adaptive_state_code) == int(AdaptiveStateCode.INITIAL)
    assert int(state.k) == 1
