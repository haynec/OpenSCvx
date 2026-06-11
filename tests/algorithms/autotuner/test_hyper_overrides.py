"""Every promoted autotuner knob reaches the solve through all three channels.

After the knob-promotion phase, *all* numeric knobs of the built-in autotuners
are declared :class:`HyperParams` fields living on the frozen ``hyper``
container and riding ``AlgorithmState.hyper``. This pins that a representative
knob per autotuner actually changes behavior through each of the three ways a
user reaches it:

* **(a) constructor arg** — ``AugmentedLagrangian(gamma_1=...)`` lands in
  ``autotuner.hyper.gamma_1``;
* **(b) attribute mutation** — ``autotuner.gamma_1 = ...`` routes through the
  :class:`AutotuningBase` proxy into ``autotuner.hyper``;
* **(c) per-solve / batched override** — the channel ``solve_jax(algorithm=
  {...})`` uses: seed an :class:`AlgorithmState` from the autotuner's ``hyper``,
  ``state.replace(hyper=dataclasses.replace(hyper, knob=...))``, and one
  ``update_weights`` call diverges from the default-knob run.

The synthetic state/candidate machinery is the same as
``tests/test_autotuning.py`` / ``tests/algorithms/autotuner/test_update_weights_jit.py``;
no full solves run here — it stays fast.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.algorithms import (
    AdaptiveProximalWeight,
    AugmentedLagrangian,
    ConstantProximalWeight,
    RampProximalWeight,
)
from openscvx.algorithms.base import (
    AdaptiveStateCode,
    AlgorithmState,
    AutotuningBase,
    CandidateIterate,
    HyperParams,
)
from openscvx.algorithms.weights import Weights
from openscvx.config import Config, DevConfig, PropagationConfig, SimConfig
from openscvx.lowered.jax_constraints import LoweredJaxConstraints

# --- Synthetic fixtures (mirrors tests/test_autotuning.py) ------------------


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
def empty_constraints():
    return LoweredJaxConstraints(nodal=[], cross_node=[], ctcs=[])


@pytest.fixture
def candidate():
    c = CandidateIterate()
    c.x = jnp.asarray(np.array([[0.0, 0.0], [1.1, 1.1], [2.1, 2.1]]))
    c.u = jnp.asarray(np.array([[0.1], [0.6], [1.1]]))
    c.x_prop = jnp.asarray(np.array([[1.05, 1.05], [2.05, 2.05]]))
    c.x_prop_plus = jnp.asarray(np.array([[0.05, 0.05], [1.05, 1.05], [2.05, 2.05]]))
    # Large linear objective forces predicted reduction < 0 -> the
    # acceptance-ratio tuners reject and scale lam_prox by gamma_1.
    c.J_lin = jnp.asarray(1e6)
    return c


def _state(settings, weights, hyper, k):
    """Build an AlgorithmState carrying ``hyper`` at iteration ``k``.

    Seeds ``x_prop`` / ``x_prop_plus`` with finite values so the autotuner's
    ``k > 1`` branch has a well-defined previous iterate.
    """
    base = AlgorithmState.from_settings(
        settings, weights, ep_tr=1e-4, ep_vb=1e-4, ep_vc=1e-8, k_max=200, hyper=hyper
    )
    return base.replace(
        k=jnp.asarray(k, dtype=jnp.int32),
        x_prop=jnp.asarray(np.array([[0.1, 0.1], [0.9, 0.9]])),
        x_prop_plus=jnp.asarray(np.array([[0.0, 0.0], [0.1, 0.1], [0.9, 0.9]])),
    )


def _override_run(autotuner, state, candidate, constraints, settings, knob, value):
    """Run update_weights with ``knob`` overridden on ``state.hyper`` (channel c)."""
    overridden = state.replace(hyper=dataclasses.replace(state.hyper, **{knob: value}))
    return autotuner.update_weights(overridden, candidate, constraints, settings, {})


# --- Per-autotuner knob sweeps through every channel ------------------------
#
# Each row: (factory, knob, default_value, override_value, k, lam_prox_field).
# The acceptance-ratio tuners change ``lam_prox`` on reject via ``gamma_1`` at
# k>1; the simple tuners change a different observable at k=1.


@pytest.mark.parametrize(
    "make_autotuner, knob, override_value, k, observe",
    [
        pytest.param(AugmentedLagrangian, "gamma_1", 5.0, 2, "lam_prox", id="auglag_gamma_1"),
        pytest.param(AdaptiveProximalWeight, "gamma_1", 5.0, 2, "lam_prox", id="adaptive_gamma_1"),
        pytest.param(RampProximalWeight, "ramp_factor", 3.0, 1, "lam_prox", id="ramp_ramp_factor"),
    ],
)
def test_constructor_and_proxy_channels(
    make_autotuner,
    knob,
    override_value,
    k,
    observe,
    settings,
    weights,
    candidate,
    empty_constraints,
):
    """Channels (a) constructor and (b) proxy mutation both reach update_weights."""
    # Default run.
    default = make_autotuner()
    default_state = _state(settings, weights, default.hyper, k)
    default_out = default.update_weights(default_state, candidate, empty_constraints, settings, {})

    # (a) constructor arg.
    ctor = make_autotuner(**{knob: override_value})
    assert getattr(ctor.hyper, knob) == override_value
    ctor_state = _state(settings, weights, ctor.hyper, k)
    ctor_out = ctor.update_weights(ctor_state, candidate, empty_constraints, settings, {})
    assert not np.allclose(np.asarray(ctor_out.lam_prox), np.asarray(default_out.lam_prox))

    # (b) attribute mutation via the proxy lands on hyper.
    proxy = make_autotuner()
    setattr(proxy, knob, override_value)
    assert getattr(proxy.hyper, knob) == override_value  # landed on hyper, not shadowed
    proxy_state = _state(settings, weights, proxy.hyper, k)
    proxy_out = proxy.update_weights(proxy_state, candidate, empty_constraints, settings, {})
    np.testing.assert_allclose(np.asarray(proxy_out.lam_prox), np.asarray(ctor_out.lam_prox))


@pytest.mark.parametrize(
    "make_autotuner, knob, override_value, k, observe",
    [
        pytest.param(AugmentedLagrangian, "gamma_1", 5.0, 2, "lam_prox", id="auglag_gamma_1"),
        pytest.param(AdaptiveProximalWeight, "gamma_1", 5.0, 2, "lam_prox", id="adaptive_gamma_1"),
        pytest.param(RampProximalWeight, "ramp_factor", 3.0, 1, "lam_prox", id="ramp_ramp_factor"),
    ],
)
def test_solve_override_channel(
    make_autotuner,
    knob,
    override_value,
    k,
    observe,
    settings,
    weights,
    candidate,
    empty_constraints,
):
    """Channel (c): a state.hyper override diverges from the default-knob run."""
    autotuner = make_autotuner()
    state = _state(settings, weights, autotuner.hyper, k)

    default_out = autotuner.update_weights(state, candidate, empty_constraints, settings, {})
    override_out = _override_run(
        autotuner, state, candidate, empty_constraints, settings, knob, override_value
    )

    assert not np.allclose(
        np.asarray(getattr(override_out, observe)), np.asarray(getattr(default_out, observe))
    )


def test_constant_proximal_weight_lam_cost_relax_channels(
    settings, weights, candidate, empty_constraints
):
    """ConstantProximalWeight's only behavioral knob (lam_cost_relax) across channels.

    ConstantProximalWeight never touches ``lam_prox``; its observable knob is
    ``lam_cost_relax``, applied once ``state.k > lam_cost_drop``. Seed ``k`` past
    the (default ``-1``) drop so the relaxation is live.
    """
    k = 2

    # (a) constructor + (c) override baseline.
    default = ConstantProximalWeight()
    default_state = _state(settings, weights, default.hyper, k)
    default_out = default.update_weights(default_state, candidate, empty_constraints, settings, {})

    ctor = ConstantProximalWeight(lam_cost_relax=0.5)
    assert ctor.hyper.lam_cost_relax == 0.5
    ctor_state = _state(settings, weights, ctor.hyper, k)
    ctor_out = ctor.update_weights(ctor_state, candidate, empty_constraints, settings, {})
    assert not np.allclose(np.asarray(ctor_out.lam_cost), np.asarray(default_out.lam_cost))

    # (b) proxy mutation.
    proxy = ConstantProximalWeight()
    proxy.lam_cost_relax = 0.5
    assert proxy.hyper.lam_cost_relax == 0.5
    proxy_state = _state(settings, weights, proxy.hyper, k)
    proxy_out = proxy.update_weights(proxy_state, candidate, empty_constraints, settings, {})
    np.testing.assert_allclose(np.asarray(proxy_out.lam_cost), np.asarray(ctor_out.lam_cost))

    # (c) per-solve override.
    override_out = _override_run(
        default, default_state, candidate, empty_constraints, settings, "lam_cost_relax", 0.5
    )
    np.testing.assert_allclose(np.asarray(override_out.lam_cost), np.asarray(ctor_out.lam_cost))


# --- Capability flag: COMPUTES_ACCEPTANCE_METRICS gates the diagnostics -----


class _NoMetricsAutotuner(AutotuningBase):
    """Minimal custom autotuner that declares it computes no acceptance metrics."""

    COMPUTES_ACCEPTANCE_METRICS = False

    def __init__(self):
        self.hyper = HyperParams()

    def update_weights(self, state, candidate, nodal_constraints, settings, params):
        # Always accept; never compute predicted/actual/acceptance.
        return state.replace(
            x=candidate.x,
            u=candidate.u,
            x_prop=candidate.x_prop,
            x_prop_plus=candidate.x_prop_plus,
            adaptive_state_code=jnp.asarray(
                int(AdaptiveStateCode.ACCEPT_CONSTANT), dtype=jnp.int32
            ),
        )


def _history(settings):
    from openscvx.algorithms.base import AlgorithmHistory

    return AlgorithmHistory.from_settings(settings)


def _accepted_state(settings, weights, candidate):
    """A post-update state in the ACCEPT_CONSTANT bucket, as record_iteration sees it."""
    state = _state(settings, weights, HyperParams(), k=2)
    return state.replace(
        x=candidate.x,
        u=candidate.u,
        adaptive_state_code=jnp.asarray(int(AdaptiveStateCode.ACCEPT_CONSTANT), dtype=jnp.int32),
        predicted_reduction=jnp.asarray(1.0),
        actual_reduction=jnp.asarray(0.5),
        acceptance_ratio=jnp.asarray(0.5),
    )


def test_capability_flag_gates_acceptance_diagnostics(settings, weights, candidate):
    """COMPUTES_ACCEPTANCE_METRICS routes the acceptance diagnostics in record_iteration.

    ``PenalizedTrustRegion.step`` reads the flag and passes
    ``record_diagnostics=COMPUTES_ACCEPTANCE_METRICS`` into
    :meth:`AlgorithmHistory.record_iteration`. With the flag ``False`` the
    predicted / actual / acceptance lists stay empty; with the default ``True``
    they gain one entry. This pins the Phase-2 behavior at the recording seam
    without a full solve.
    """
    next_state = _accepted_state(settings, weights, candidate)
    # The acceptance-diagnostics routing is independent of the discretization
    # matrices, so leave V / VC / TR / J_lin unset (None) — record_iteration
    # skips those appends and we isolate the diagnostics gate.
    cand = CandidateIterate()

    # Flag False (the toy autotuner) -> step would pass record_diagnostics=False.
    flag = _NoMetricsAutotuner().COMPUTES_ACCEPTANCE_METRICS
    assert flag is False
    hist_off = _history(settings)
    hist_off.record_iteration(next_state, cand, record_diagnostics=flag)
    assert hist_off.acceptance_ratio == []
    assert hist_off.pred_reduction == []
    assert hist_off.actual_reduction == []

    # Default subclass (flag True) -> diagnostics recorded.
    default_flag = AugmentedLagrangian().COMPUTES_ACCEPTANCE_METRICS
    assert default_flag is True
    hist_on = _history(settings)
    hist_on.record_iteration(next_state, cand, record_diagnostics=default_flag)
    assert hist_on.acceptance_ratio == [0.5]
    assert hist_on.pred_reduction == [1.0]
    assert hist_on.actual_reduction == [0.5]
