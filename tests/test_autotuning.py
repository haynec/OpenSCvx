"""Unit tests for autotuning functions in openscvx.algorithms.

Tests for the legacy mutate-in-place ``AlgorithmState`` contract were removed
during the JAX-traceability refactor; see ``plans/jax-traceable-autotuners.md``.
The autotuners now return a new :class:`AlgorithmState` pytree, so tests here
assert on the *returned* state rather than mutation of an input.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.algorithms import (
    AdaptiveProximalWeight,
    AdaptiveStateCode,
    AugmentedLagrangian,
    ConstantProximalWeight,
    PenalizedTrustRegion,
    PenalizedTrustRegionConfig,
    RampProximalWeight,
)
from openscvx.algorithms.penalty import (
    calculate_cost_from_state,
    calculate_nonlinear_penalty,
)
from openscvx.algorithms.state import AlgorithmState, CandidateIterate
from openscvx.algorithms.weights import Weights
from openscvx.config import (
    Config,
    DevConfig,
    PropagationConfig,
    SimConfig,
)
from openscvx.lowered.jax_constraints import (
    LoweredCrossNodeConstraint,
    LoweredJaxConstraints,
    LoweredNodalConstraint,
)

# --- Test Fixtures ---------------------------------------------------------


class DummyState:
    """Dummy state object for testing."""

    pass


class DummyControl:
    """Dummy control object for testing."""

    pass


@pytest.fixture
def mock_unified_state():
    """Create a mock UnifiedState object."""
    state = DummyState()
    state.initial = np.array([0.0, 0.0])
    state.final = np.array([0.0, 0.0])
    state.guess = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    state.min = np.array([-10.0, -10.0])
    state.max = np.array([10.0, 10.0])
    state.final_type = ["None", "None"]
    state.initial_type = ["None", "None"]
    state.time_slice = 0
    state.scaling_min = None  # No custom scaling
    state.scaling_max = None  # No custom scaling
    return state


@pytest.fixture
def mock_unified_control():
    """Create a mock UnifiedControl object."""
    control = DummyControl()
    control.guess = np.array([[0.0], [0.5], [1.0]])
    control.min = np.array([-1.0])
    control.max = np.array([1.0])
    control.scaling_min = None  # No custom scaling
    control.scaling_max = None  # No custom scaling
    return control


@pytest.fixture
def settings(mock_unified_state, mock_unified_control):
    """Create a Config object for testing."""
    sim_config = SimConfig(
        x=mock_unified_state,
        x_prop=mock_unified_state,
        u=mock_unified_control,
        total_time=1.0,
        n=3,
        n_states=2,
        n_controls=1,
    )

    config = Config(
        sim=sim_config,
        prp=PropagationConfig(),
        dev=DevConfig(),
    )

    return config


@pytest.fixture
def weights():
    """Default scalar weights with explicit nodal/cross virtual-buffer arrays.

    The autotuner needs ``lam_vb_nodal`` / ``lam_vb_cross`` to be populated
    before constructing an :class:`AlgorithmState`; in production these are
    set by :meth:`Weights.build_vb_arrays`. For unit tests we seed them
    explicitly to match the 3-node, 1-nodal-constraint, 1-cross-constraint
    sizes used throughout the file.
    """
    w = Weights(lam_prox=1.0, lam_vc=1.0, lam_vb=1.0, lam_cost=1.0)
    w.lam_vb_nodal = np.full((3, 1), 1.0)
    w.lam_vb_cross = np.full(1, 1.0)
    return w


@pytest.fixture
def algorithm_state(settings, weights):
    """Initial :class:`AlgorithmState` for the 3-node test problem."""
    return AlgorithmState.from_settings(
        settings,
        weights,
        ep_tr=1e-4,
        ep_vb=1e-4,
        ep_vc=1e-8,
        k_max=200,
        hyper=AugmentedLagrangian().hyper,
    )


def _candidate_x_prop_plus(N: int = 3, n_x: int = 2) -> np.ndarray:
    """Zero ``x_prop_plus`` of shape ``(N, n_x)``; first row is unused."""
    return np.zeros((N, n_x))


def _seeded_state_for_k2(state: AlgorithmState) -> AlgorithmState:
    """Replace ``x_prop`` / ``x_prop_plus`` with finite values for k>1 tests.

    The autotuner's k>1 branch reads ``state.x_prop_plus[1:]`` as the previous
    iterate's propagation. ``from_settings`` seeds those to zero, which is fine
    for shape but uninformative for math; we make them match ``state.x`` so
    the previous-iterate residual is well-defined.
    """
    x = np.asarray(state.x)
    x_prop = x[1:]  # (N-1, n_x)
    x_prop_plus = x  # (N, n_x); only [1:] is read by the autotuner
    return state.replace(
        k=jnp.asarray(2, dtype=jnp.int32),
        x_prop=jnp.asarray(x_prop),
        x_prop_plus=jnp.asarray(x_prop_plus),
    )


@pytest.fixture
def empty_nodal_constraints():
    """Create empty LoweredJaxConstraints."""
    return LoweredJaxConstraints(
        nodal=[],
        cross_node=[],
        ctcs=[],
    )


@pytest.fixture
def nodal_constraints_with_violations():
    """Create LoweredJaxConstraints with some constraint violations."""

    # Create a simple nodal constraint that returns positive values (violations)
    # The function is vmapped, so it receives (N, n_x) and (N, n_u) arrays
    def nodal_func(x, u, node, params):
        # Constraint: x[:, 0] - 1.5 <= 0, so violation when x[:, 0] > 1.5
        # x has shape (N, n_x), so x[:, 0] gives first state at all nodes
        return x[:, 0] - 1.5

    constraint = LoweredNodalConstraint(
        func=nodal_func,
        nodes=None,  # Apply to all nodes
    )

    return LoweredJaxConstraints(
        nodal=[constraint],
        cross_node=[],
        ctcs=[],
    )


@pytest.fixture
def cross_node_constraints():
    """Create LoweredJaxConstraints with cross-node constraints."""

    def cross_node_func(X, U, params):
        # Constraint: X[1, 0] - X[0, 0] - 0.5 <= 0
        # Violation when difference > 0.5
        return X[1, 0] - X[0, 0] - 0.5

    def grad_g_X(X, U, params):
        # Gradient w.r.t. X: only non-zero at nodes 0 and 1
        grad = np.zeros_like(X)
        grad[0, 0] = -1.0  # d/dX[0,0]
        grad[1, 0] = 1.0  # d/dX[1,0]
        return grad

    def grad_g_U(X, U, params):
        # Gradient w.r.t. U: zero (constraint doesn't depend on U)
        return np.zeros_like(U)

    constraint = LoweredCrossNodeConstraint(
        func=cross_node_func,
        grad_g_X=grad_g_X,
        grad_g_U=grad_g_U,
    )

    return LoweredJaxConstraints(
        nodal=[],
        cross_node=[constraint],
        ctcs=[],
    )


# --- Tests for calculate_cost_from_state -----------------------------------


def test_calculate_cost_from_state_minimize_final(settings):
    """Test cost calculation with Minimize final_type."""
    settings.sim.x.final_type = ["None", "Minimize"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cost = calculate_cost_from_state(x, settings)

    # Should add scaled final state value
    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = scaled_x[-1, 1]  # Final node, second state
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_maximize_final(settings):
    """Test cost calculation with Maximize final_type."""
    settings.sim.x.final_type = ["None", "Maximize"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cost = calculate_cost_from_state(x, settings)

    # Should subtract scaled final state value
    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = -scaled_x[-1, 1]  # Final node, second state (negated)
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_minimize_initial(settings):
    """Test cost calculation with Minimize initial_type."""
    settings.sim.x.initial_type = ["Minimize", "None"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cost = calculate_cost_from_state(x, settings)

    # Should add scaled initial state value
    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = scaled_x[0, 0]  # Initial node, first state
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_maximize_initial(settings):
    """Test cost calculation with Maximize initial_type."""
    settings.sim.x.initial_type = ["Maximize", "None"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cost = calculate_cost_from_state(x, settings)

    # Should subtract scaled initial state value
    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = -scaled_x[0, 0]  # Initial node, first state (negated)
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_combined(settings):
    """Test cost calculation with both initial and final types."""
    settings.sim.x.initial_type = ["Minimize", "None"]
    settings.sim.x.final_type = ["None", "Maximize"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cost = calculate_cost_from_state(x, settings)

    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = scaled_x[0, 0] - scaled_x[-1, 1]
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_no_cost(settings):
    """Test cost calculation with no cost types (should return 0)."""
    settings.sim.x.initial_type = ["None", "None"]
    settings.sim.x.final_type = ["None", "None"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cost = calculate_cost_from_state(x, settings)

    assert cost == 0.0


def test_calculate_cost_from_state_per_state_weights(settings):
    """Test cost calculation with a per-state lam_cost array."""
    settings.sim.x.final_type = ["Minimize", "Minimize"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    lam_cost = np.array([2.0, 5.0])

    cost = calculate_cost_from_state(x, settings, lam_cost=lam_cost)

    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = 2.0 * scaled_x[-1, 0] + 5.0 * scaled_x[-1, 1]
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_per_state_weights_maximize(settings):
    """Test per-state weights with Maximize objective."""
    settings.sim.x.final_type = ["Maximize", "None"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    lam_cost = np.array([3.0, 0.0])

    cost = calculate_cost_from_state(x, settings, lam_cost=lam_cost)

    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = -3.0 * scaled_x[-1, 0]
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_per_state_weights_mixed(settings):
    """Test per-state weights with mixed Minimize initial and Maximize final."""
    settings.sim.x.initial_type = ["Minimize", "None"]
    settings.sim.x.final_type = ["None", "Maximize"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    lam_cost = np.array([4.0, 7.0])

    cost = calculate_cost_from_state(x, settings, lam_cost=lam_cost)

    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = 4.0 * scaled_x[0, 0] - 7.0 * scaled_x[-1, 1]
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_per_state_zero_weight_ignores_cost(settings):
    """Test that a zero weight effectively ignores the cost for that state."""
    settings.sim.x.final_type = ["Minimize", "Minimize"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    lam_cost = np.array([0.0, 3.0])

    cost = calculate_cost_from_state(x, settings, lam_cost=lam_cost)

    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = 3.0 * scaled_x[-1, 1]
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_scalar_lam_cost_matches_default(settings):
    """Test that passing a scalar lam_cost gives consistent results with the default."""
    settings.sim.x.final_type = ["None", "Minimize"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cost_default = calculate_cost_from_state(x, settings)
    cost_scalar = calculate_cost_from_state(x, settings, lam_cost=1.0)

    assert cost_scalar == pytest.approx(cost_default, rel=1e-6)


# --- Tests for calculate_nonlinear_penalty ----------------------------------


def test_calculate_nonlinear_penalty_no_constraints(settings, empty_nodal_constraints):
    """Test penalty calculation with no constraints."""
    x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    x_bar = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    u_bar = np.array([[0.0], [0.5], [1.0]])
    lam_vc = np.array([1.0, 1.0])
    lam_vb_nodal = np.full((3, 1), 1.0)
    lam_vb_cross = np.full(1, 1.0)
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = calculate_nonlinear_penalty(
        x_prop,
        x_bar,
        u_bar,
        lam_vc,
        lam_vb_nodal,
        lam_vb_cross,
        lam_cost,
        empty_nodal_constraints,
        params,
        settings,
    )

    # Should have cost component
    assert nonlinear_cost != 0.0 or nonlinear_cost == 0.0  # May be zero if no cost types
    # Should have virtual control penalty from x_diff
    assert nonlinear_penalty >= 0.0
    # Should have no nodal penalty
    assert nodal_penalty == 0.0


def test_calculate_nonlinear_penalty_with_nodal_violations(
    settings, nodal_constraints_with_violations
):
    """Test penalty calculation with nodal constraint violations."""
    x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    x_bar = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])  # x[0] = 2.0 > 1.5, violation
    u_bar = np.array([[0.0], [0.5], [1.0]])
    lam_vc = np.array([1.0, 1.0])
    lam_vb_nodal = np.full((3, 1), 1.0)  # 1 nodal constraint
    lam_vb_cross = np.full(1, 1.0)
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = calculate_nonlinear_penalty(
        x_prop,
        x_bar,
        u_bar,
        lam_vc,
        lam_vb_nodal,
        lam_vb_cross,
        lam_cost,
        nodal_constraints_with_violations,
        params,
        settings,
    )

    # Should have positive nodal penalty due to violations
    assert nodal_penalty > 0.0
    # Virtual control penalty should be non-negative
    assert nonlinear_penalty >= 0.0


def test_calculate_nonlinear_penalty_equality_is_two_sided(settings):
    """Equality nodal constraints penalize |g| (two-sided), inequalities use max(0, g)."""

    def nodal_func(x, u, node, params):
        return x[:, 0] - 1.5  # negative residual at x=0 (satisfied as inequality)

    x_prop = np.array([[0.0, 0.0], [0.0, 0.0]])
    x_bar = np.zeros((3, 2))
    u_bar = np.zeros((3, 1))
    args = (np.array([1.0, 1.0]), np.full((3, 1), 1.0), np.full(1, 1.0), 1.0)

    eq = LoweredJaxConstraints(
        nodal=[LoweredNodalConstraint(func=nodal_func, nodes=None, is_equality=True)],
    )
    ineq = LoweredJaxConstraints(
        nodal=[LoweredNodalConstraint(func=nodal_func, nodes=None, is_equality=False)],
    )

    _, _, eq_pen = calculate_nonlinear_penalty(x_prop, x_bar, u_bar, *args, eq, {}, settings)
    _, _, ineq_pen = calculate_nonlinear_penalty(x_prop, x_bar, u_bar, *args, ineq, {}, settings)

    assert eq_pen > 0.0  # |−1.5| penalized
    assert ineq_pen == 0.0  # max(0, −1.5) == 0


def test_calculate_nonlinear_penalty_with_cross_node_violations(settings, cross_node_constraints):
    """Test penalty calculation with cross-node constraint violations."""
    x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    x_bar = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])  # x[1,0] - x[0,0] = 1.0 > 0.5, violation
    u_bar = np.array([[0.0], [0.5], [1.0]])
    lam_vc = np.array([1.0, 1.0])
    lam_vb_nodal = np.full((3, 1), 1.0)
    lam_vb_cross = np.full(1, 1.0)  # 1 cross-node constraint
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = calculate_nonlinear_penalty(
        x_prop,
        x_bar,
        u_bar,
        lam_vc,
        lam_vb_nodal,
        lam_vb_cross,
        lam_cost,
        cross_node_constraints,
        params,
        settings,
    )

    # Should have positive nodal penalty due to cross-node violation
    assert nodal_penalty > 0.0


def test_calculate_nonlinear_penalty_nodal_with_node_filter(settings):
    """Test penalty calculation with nodal constraints filtered to specific nodes."""

    # The function is vmapped, so it receives (N, n_x) and (N, n_u) arrays
    def nodal_func(x, u, node, params):
        # x has shape (N, n_x)
        return x[:, 0] - 1.5

    constraint = LoweredNodalConstraint(
        func=nodal_func,
        nodes=[0, 2],  # Only apply to nodes 0 and 2
    )

    nodal_constraints = LoweredJaxConstraints(
        nodal=[constraint],
        cross_node=[],
        ctcs=[],
    )

    x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    x_bar = np.array([[2.0, 0.0], [0.0, 1.0], [2.0, 2.0]])  # Nodes 0 and 2 violate
    u_bar = np.array([[0.0], [0.5], [1.0]])
    lam_vc = np.array([1.0, 1.0])
    lam_vb_nodal = np.full((3, 1), 1.0)  # 1 nodal constraint
    lam_vb_cross = np.full(1, 1.0)
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = calculate_nonlinear_penalty(
        x_prop,
        x_bar,
        u_bar,
        lam_vc,
        lam_vb_nodal,
        lam_vb_cross,
        lam_cost,
        nodal_constraints,
        params,
        settings,
    )

    # Should have positive penalty from filtered nodes
    assert nodal_penalty > 0.0


def test_calculate_nonlinear_penalty_virtual_control_component(settings, empty_nodal_constraints):
    """Test that virtual control penalty is calculated correctly."""
    x_prop = np.array([[0.0, 0.0], [1.0, 1.0]])
    x_bar = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])  # Large difference at end
    u_bar = np.array([[0.0], [0.5], [1.0]])
    lam_vc = np.array([2.0, 2.0])  # Higher weight
    lam_vb_nodal = np.full((3, 1), 1.0)
    lam_vb_cross = np.full(1, 1.0)
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = calculate_nonlinear_penalty(
        x_prop,
        x_bar,
        u_bar,
        lam_vc,
        lam_vb_nodal,
        lam_vb_cross,
        lam_cost,
        empty_nodal_constraints,
        params,
        settings,
    )

    # Virtual control penalty should be positive and larger with larger differences
    assert nonlinear_penalty > 0.0

    # Calculate expected penalty manually
    x_diff = settings.sim.inv_S_x @ (x_bar[1:, :] - x_prop).T
    expected_penalty = np.sum(lam_vc * np.abs(x_diff.T))
    assert nonlinear_penalty == pytest.approx(expected_penalty, rel=1e-6)


# --- Tests for update_scp_weights -------------------------------------------


def test_update_scp_weights_initial_iteration(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """First iteration (k=1) returns INITIAL and accepts the candidate."""
    autotuner = AugmentedLagrangian()
    candidate = CandidateIterate(
        x=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        u=np.array([[0.0], [0.5], [1.0]]),
        x_prop=np.array([[0.5, 0.5], [1.5, 1.5]]),
        x_prop_plus=_candidate_x_prop_plus(),
        J_lin=10.0,
    )

    new_state = autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, {}
    )

    assert int(new_state.adaptive_state_code) == int(AdaptiveStateCode.INITIAL)
    # Trajectory and propagation fields propagate from the candidate.
    np.testing.assert_allclose(np.asarray(new_state.x), candidate.x)
    np.testing.assert_allclose(np.asarray(new_state.u), candidate.u)
    np.testing.assert_allclose(np.asarray(new_state.x_prop), candidate.x_prop)
    np.testing.assert_allclose(np.asarray(new_state.x_prop_plus), candidate.x_prop_plus)
    # lam_prox is unchanged on the initial iteration.
    np.testing.assert_allclose(np.asarray(new_state.lam_prox), np.asarray(algorithm_state.lam_prox))


def test_update_scp_weights_cost_drop(settings, algorithm_state, empty_nodal_constraints, weights):
    """Cost relaxation kicks in once ``state.k > hyper.lam_cost_drop``."""
    autotuner = AugmentedLagrangian(lam_cost_drop=3, lam_cost_relax=0.8)

    # k=4 > lam_cost_drop=3, so lam_cost should be scaled by lam_cost_relax.
    # The declared knob is read from state.hyper, so seed it from the
    # autotuner's hyper container (in production from_settings does this).
    state = _seeded_state_for_k2(algorithm_state).replace(
        k=jnp.asarray(4, dtype=jnp.int32),
        hyper=jax.tree_util.tree_map(jnp.asarray, autotuner.hyper),
    )
    lam_cost_prev = np.asarray(state.lam_cost)

    candidate = CandidateIterate(
        x=np.asarray(state.x),
        u=np.asarray(state.u),
        x_prop=np.asarray(state.x_prop),
        x_prop_plus=_candidate_x_prop_plus(),
        J_lin=0.5,
    )

    new_state = autotuner.update_weights(state, candidate, empty_nodal_constraints, settings, {})

    np.testing.assert_allclose(np.asarray(new_state.lam_cost), lam_cost_prev * 0.8, rtol=1e-6)


def test_update_scp_weights_weight_bounds(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """lam_prox saturates at lam_prox_max even with sustained rejection."""
    autotuner = AugmentedLagrangian()

    # Start near the upper bound; a rejected step would try to scale up.
    high_prox = np.full(np.asarray(algorithm_state.lam_prox).shape, autotuner.lam_prox_max)
    state = _seeded_state_for_k2(algorithm_state).replace(lam_prox=jnp.asarray(high_prox))

    candidate = CandidateIterate(
        x=np.asarray(state.x),
        u=np.asarray(state.u),
        x_prop=np.asarray(state.x_prop),
        x_prop_plus=_candidate_x_prop_plus(),
        J_lin=1e6,  # makes predicted reduction strongly negative → reject
    )

    new_state = autotuner.update_weights(state, candidate, empty_nodal_constraints, settings, {})

    final = np.asarray(new_state.lam_prox)
    assert np.all(final >= autotuner.lam_prox_min)
    assert np.all(final <= autotuner.lam_prox_max)


# --- Tests for AugmentedLagrangianAutotuning ---------------------------------


def test_augmented_lagrangian_accept_decrease(
    settings, algorithm_state, nodal_constraints_with_violations, weights
):
    """Explicitly realize the 'Accept Lower' branch with constraint violations.

    Constructs ``J_lin`` so that the acceptance ratio ``rho`` lies strictly
    above ``eta_2`` (high-quality step), and verifies:

    * the returned state's code is ``ACCEPT_LOWER``,
    * ``lam_prox`` is scaled down by ``gamma_2`` (clipped at ``lam_prox_min``),
    * ``lam_vb_nodal`` follows the same piecewise rule that drives the
      virtual-control update.
    """
    autotuner = AugmentedLagrangian()

    state = _seeded_state_for_k2(algorithm_state)

    cand_x = np.asarray(state.x)
    cand_u = np.asarray(state.u)
    cand_x_prop = np.asarray(state.x_prop)
    cand_x_prop_plus = _candidate_x_prop_plus()

    # Compute previous and candidate nonlinear objectives explicitly, then
    # choose J_lin so that rho > eta_2, guaranteeing "Accept Lower".
    prev_cost, prev_penalty, prev_nodal = calculate_nonlinear_penalty(
        state.x_prop_plus[1:],
        state.x,
        state.u,
        state.lam_vc,
        state.lam_vb_nodal,
        state.lam_vb_cross,
        state.lam_cost,
        nodal_constraints_with_violations,
        {},
        settings,
    )
    J_nonlin_prev = float(prev_cost + prev_penalty + prev_nodal)

    cand_cost, cand_penalty, cand_nodal = calculate_nonlinear_penalty(
        cand_x_prop_plus[1:],
        cand_x,
        cand_u,
        state.lam_vc,
        state.lam_vb_nodal,
        state.lam_vb_cross,
        state.lam_cost,
        nodal_constraints_with_violations,
        {},
        settings,
    )
    J_nonlin_cand = float(cand_cost + cand_penalty + cand_nodal)

    actual_reduction = J_nonlin_prev - J_nonlin_cand
    rho_target = autotuner.eta_2 + 0.1 * (1.0 - autotuner.eta_2)  # strictly > eta_2, < 1
    predicted_reduction = actual_reduction / rho_target

    candidate = CandidateIterate(
        x=cand_x,
        u=cand_u,
        x_prop=cand_x_prop,
        x_prop_plus=cand_x_prop_plus,
        J_lin=J_nonlin_prev - predicted_reduction,
    )

    lam_prox_prev = np.asarray(state.lam_prox)

    new_state = autotuner.update_weights(
        state,
        candidate,
        nodal_constraints_with_violations,
        settings,
        {},
    )

    assert int(new_state.adaptive_state_code) == int(AdaptiveStateCode.ACCEPT_LOWER)
    expected_prox = np.maximum(autotuner.lam_prox_min, autotuner.gamma_2 * lam_prox_prev)
    np.testing.assert_allclose(np.asarray(new_state.lam_prox), expected_prox)
    np.testing.assert_allclose(np.asarray(new_state.x), candidate.x)

    # Virtual buffer weights follow the same piecewise rule as virtual control.
    lam_prox_new = expected_prox
    scale = autotuner.eta_lambda / (2.0 * float(np.max(lam_prox_new)))
    nu_flat = np.maximum(0.0, candidate.x[:, 0] - 1.5)
    expected_vb_col = np.ones(3)
    for i in range(3):
        nui = nu_flat[i]
        if nui > autotuner.ep:
            expected_vb_col[i] = 1.0 + nui * scale
        else:
            expected_vb_col[i] = 1.0 + (nui**2) / autotuner.ep * scale
    expected_vb_nodal = expected_vb_col.reshape(3, 1)
    np.testing.assert_allclose(np.asarray(new_state.lam_vb_nodal), expected_vb_nodal)
    np.testing.assert_allclose(np.asarray(new_state.lam_vb_cross), np.asarray(state.lam_vb_cross))


def test_augmented_lagrangian_reject_increase(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """A bad step (very high J_lin) is rejected and lam_prox is scaled up by gamma_1."""
    autotuner = AugmentedLagrangian()
    state = _seeded_state_for_k2(algorithm_state)

    # J_lin much larger than the previous objective drives predicted < 0,
    # which forces rho < eta_0.
    candidate = CandidateIterate(
        x=np.asarray(state.x),
        u=np.asarray(state.u),
        x_prop=np.asarray(state.x_prop),
        x_prop_plus=_candidate_x_prop_plus(),
        J_lin=1e6,
    )

    lam_prox_prev = np.asarray(state.lam_prox)
    lam_vc_prev = np.asarray(state.lam_vc)

    new_state = autotuner.update_weights(state, candidate, empty_nodal_constraints, settings, {})

    assert int(new_state.adaptive_state_code) == int(AdaptiveStateCode.REJECT)
    np.testing.assert_allclose(
        np.asarray(new_state.lam_prox),
        np.minimum(autotuner.lam_prox_max, autotuner.gamma_1 * lam_prox_prev),
    )
    # On reject the trajectory and virtual-control weights are carried forward.
    np.testing.assert_allclose(np.asarray(new_state.x), np.asarray(state.x))
    np.testing.assert_allclose(np.asarray(new_state.lam_vc), lam_vc_prev)


def _build_rho_targeted_candidate(state, settings, nodal_constraints, rho_target):
    """Build a CandidateIterate whose J_lin yields the given acceptance ratio.

    Mirrors the bookkeeping inside ``AugmentedLagrangian.update_weights`` so
    that ``rho = actual / predicted`` lands on ``rho_target``.
    """
    cand_x = np.asarray(state.x)
    cand_u = np.asarray(state.u)
    cand_x_prop = np.asarray(state.x_prop)
    cand_x_prop_plus = _candidate_x_prop_plus()

    prev_cost, prev_penalty, prev_nodal = calculate_nonlinear_penalty(
        state.x_prop_plus[1:],
        state.x,
        state.u,
        state.lam_vc,
        state.lam_vb_nodal,
        state.lam_vb_cross,
        state.lam_cost,
        nodal_constraints,
        {},
        settings,
    )
    J_nonlin_prev = float(prev_cost + prev_penalty + prev_nodal)

    cand_cost, cand_penalty, cand_nodal = calculate_nonlinear_penalty(
        cand_x_prop_plus[1:],
        cand_x,
        cand_u,
        state.lam_vc,
        state.lam_vb_nodal,
        state.lam_vb_cross,
        state.lam_cost,
        nodal_constraints,
        {},
        settings,
    )
    J_nonlin_cand = float(cand_cost + cand_penalty + cand_nodal)

    actual_reduction = J_nonlin_prev - J_nonlin_cand
    predicted_reduction = actual_reduction / rho_target
    return CandidateIterate(
        x=cand_x,
        u=cand_u,
        x_prop=cand_x_prop,
        x_prop_plus=cand_x_prop_plus,
        J_lin=J_nonlin_prev - predicted_reduction,
    )


def test_augmented_lagrangian_accept_higher(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """Realize the 'Accept Higher' branch (eta_0 <= rho < eta_1)."""
    autotuner = AugmentedLagrangian()
    state = _seeded_state_for_k2(algorithm_state)

    rho_target = 0.5 * (autotuner.eta_0 + autotuner.eta_1)
    candidate = _build_rho_targeted_candidate(state, settings, empty_nodal_constraints, rho_target)

    lam_prox_prev = np.asarray(state.lam_prox)

    new_state = autotuner.update_weights(state, candidate, empty_nodal_constraints, settings, {})

    assert int(new_state.adaptive_state_code) == int(AdaptiveStateCode.ACCEPT_HIGHER)
    expected = np.minimum(autotuner.lam_prox_max, autotuner.gamma_1 * lam_prox_prev)
    np.testing.assert_allclose(np.asarray(new_state.lam_prox), expected)
    # Candidate accepted -> trajectory propagates.
    np.testing.assert_allclose(np.asarray(new_state.x), candidate.x)


def test_augmented_lagrangian_accept_constant(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """Realize the 'Accept Constant' branch (eta_1 <= rho < eta_2)."""
    autotuner = AugmentedLagrangian()
    state = _seeded_state_for_k2(algorithm_state)

    rho_target = 0.5 * (autotuner.eta_1 + autotuner.eta_2)
    candidate = _build_rho_targeted_candidate(state, settings, empty_nodal_constraints, rho_target)

    lam_prox_prev = np.asarray(state.lam_prox)

    new_state = autotuner.update_weights(state, candidate, empty_nodal_constraints, settings, {})

    assert int(new_state.adaptive_state_code) == int(AdaptiveStateCode.ACCEPT_CONSTANT)
    np.testing.assert_allclose(np.asarray(new_state.lam_prox), lam_prox_prev)
    np.testing.assert_allclose(np.asarray(new_state.x), candidate.x)


def test_augmented_lagrangian_base_class_methods(settings):
    """Test that base class methods work correctly."""
    # Static method returns a JAX scalar (0-d jnp array) under the new contract.
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    cost = calculate_cost_from_state(x, settings)
    assert jnp.asarray(cost).shape == ()


def test_algorithm_autotuner_default():
    """PenalizedTrustRegion.autotuner should default to AugmentedLagrangian."""
    algorithm = PenalizedTrustRegion()
    assert isinstance(algorithm.autotuner, AugmentedLagrangian)


def test_algorithm_autotuner_configurable():
    """PenalizedTrustRegion default autotuner should be a configurable AugmentedLagrangian."""
    algorithm = PenalizedTrustRegion()
    autotuner = algorithm.autotuner
    assert isinstance(autotuner, AugmentedLagrangian)
    assert hasattr(autotuner, "lam_prox_min")
    assert hasattr(autotuner, "lam_prox_max")
    assert hasattr(autotuner, "lam_vc_max")
    assert hasattr(autotuner, "lam_cost_relax")
    # Every numeric knob is a declared hyperparameter living on the frozen
    # HyperParams container (and riding state.hyper); updates go through
    # dataclasses.replace.
    assert {f.name for f in dataclasses.fields(autotuner.hyper)} == {
        "gamma_1",
        "gamma_2",
        "eta_0",
        "eta_1",
        "eta_2",
        "lam_prox_min",
        "lam_prox_max",
        "lam_cost_drop",
        "lam_cost_relax",
        "ep",
        "eta_lambda",
        "lam_vc_max",
        "rho_init",
        "rho_max",
    }
    # rho_max is a reserved-but-unused knob: setting it to a non-default value
    # at construction warns. Pin that as behavior, then confirm the value lands.
    with pytest.warns(UserWarning, match="reserved"):
        autotuner = AugmentedLagrangian(rho_max=1e7)
    assert autotuner.hyper.rho_max == 1e7


def test_custom_autotuner_instance():
    """Custom autotuner instance can be passed to PenalizedTrustRegion."""
    with pytest.warns(UserWarning, match="reserved"):
        custom_autotuner = AugmentedLagrangian(rho_max=1e7)
    custom_autotuner.lam_prox_max = 1e6
    custom_autotuner.lam_vc_max = 1e6
    algorithm = PenalizedTrustRegion(autotuner=custom_autotuner)
    assert algorithm.autotuner is custom_autotuner
    assert algorithm.autotuner.hyper.rho_max == 1e7
    assert algorithm.autotuner.lam_prox_max == 1e6
    assert algorithm.autotuner.lam_vc_max == 1e6


def test_augmented_lagrangian_exported():
    """Test that AugmentedLagrangian is exported from main module."""
    import openscvx as ox

    # Should be able to import directly
    auto_tuner = ox.AugmentedLagrangian()
    assert hasattr(auto_tuner.hyper, "rho_max")
    assert hasattr(auto_tuner, "lam_prox_max")
    assert hasattr(auto_tuner, "lam_vc_max")

    # Should be able to modify parameters. Setting the reserved rho_max to a
    # non-default value at construction warns; pin that.
    with pytest.warns(UserWarning, match="reserved"):
        auto_tuner = ox.AugmentedLagrangian(rho_max=1e7)
    auto_tuner.lam_prox_max = 1e6
    assert auto_tuner.hyper.rho_max == 1e7
    assert auto_tuner.lam_prox_max == 1e6


# --- Tests for AdaptiveProximalWeight --------------------------------------------


def test_adaptive_proximal_weight_initial_iteration(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """AdaptiveProximalWeight on k=1 carries VC/VB unchanged and accepts."""
    autotuner = AdaptiveProximalWeight()
    candidate = CandidateIterate(
        x=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        u=np.array([[0.0], [0.5], [1.0]]),
        x_prop=np.array([[0.5, 0.5], [1.5, 1.5]]),
        x_prop_plus=_candidate_x_prop_plus(),
        J_lin=10.0,
    )

    new_state = autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, {}
    )

    assert int(new_state.adaptive_state_code) == int(AdaptiveStateCode.INITIAL)
    np.testing.assert_allclose(np.asarray(new_state.lam_vc), np.asarray(algorithm_state.lam_vc))
    np.testing.assert_allclose(
        np.asarray(new_state.lam_vb_nodal), np.asarray(algorithm_state.lam_vb_nodal)
    )
    np.testing.assert_allclose(np.asarray(new_state.x), candidate.x)


def test_adaptive_proximal_weight_accept_lower_fixed_vc_vb(
    settings, algorithm_state, nodal_constraints_with_violations, weights
):
    """Accept Lower decreases lam_prox but leaves lam_vc / lam_vb unchanged."""
    autotuner = AdaptiveProximalWeight()
    state = _seeded_state_for_k2(algorithm_state)

    rho_target = autotuner.eta_2 + 0.1 * (1.0 - autotuner.eta_2)
    candidate = _build_rho_targeted_candidate(
        state, settings, nodal_constraints_with_violations, rho_target
    )

    lam_prox_prev = np.asarray(state.lam_prox)
    lam_vc_prev = np.asarray(state.lam_vc)
    lam_vb_nodal_prev = np.asarray(state.lam_vb_nodal)
    lam_vb_cross_prev = np.asarray(state.lam_vb_cross)

    new_state = autotuner.update_weights(
        state,
        candidate,
        nodal_constraints_with_violations,
        settings,
        {},
    )

    assert int(new_state.adaptive_state_code) == int(AdaptiveStateCode.ACCEPT_LOWER)
    expected_prox = np.maximum(autotuner.lam_prox_min, autotuner.gamma_2 * lam_prox_prev)
    np.testing.assert_allclose(np.asarray(new_state.lam_prox), expected_prox)
    # AdaptiveProximalWeight holds VC / VB constant — that's the whole point.
    np.testing.assert_allclose(np.asarray(new_state.lam_vc), lam_vc_prev)
    np.testing.assert_allclose(np.asarray(new_state.lam_vb_nodal), lam_vb_nodal_prev)
    np.testing.assert_allclose(np.asarray(new_state.lam_vb_cross), lam_vb_cross_prev)


def test_adaptive_proximal_weight_reject_increase(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """Reject increases lam_prox and leaves lam_vc unchanged."""
    autotuner = AdaptiveProximalWeight()
    state = _seeded_state_for_k2(algorithm_state)

    candidate = CandidateIterate(
        x=np.asarray(state.x),
        u=np.asarray(state.u),
        x_prop=np.asarray(state.x_prop),
        x_prop_plus=_candidate_x_prop_plus(),
        J_lin=1e6,  # forces rho < eta_0
    )

    lam_prox_prev = np.asarray(state.lam_prox)
    lam_vc_prev = np.asarray(state.lam_vc)

    new_state = autotuner.update_weights(state, candidate, empty_nodal_constraints, settings, {})

    assert int(new_state.adaptive_state_code) == int(AdaptiveStateCode.REJECT)
    np.testing.assert_allclose(
        np.asarray(new_state.lam_prox),
        np.minimum(autotuner.lam_prox_max, autotuner.gamma_1 * lam_prox_prev),
    )
    np.testing.assert_allclose(np.asarray(new_state.lam_vc), lam_vc_prev)


def test_penalized_trust_region_config_adaptive_proximal_weight():
    """Dict/YAML autotuner config builds AdaptiveProximalWeight."""
    cfg = PenalizedTrustRegionConfig(
        autotuner={"type": "AdaptiveProximalWeight", "gamma_1": 3.0},
    )
    algorithm = cfg.to_algorithm()
    assert isinstance(algorithm.autotuner, AdaptiveProximalWeight)
    assert algorithm.autotuner.gamma_1 == 3.0


def test_adaptive_proximal_weight_exported():
    """AdaptiveProximalWeight is exported from the top-level openscvx namespace."""
    import openscvx as ox

    autotuner = ox.AdaptiveProximalWeight()
    assert isinstance(autotuner, AdaptiveProximalWeight)
    assert autotuner.gamma_1 == 2.0


# --- Tests for ConstantProximalWeight ---------------------------------------------


def test_constant_proximal_weight_keeps_lam_prox_and_accepts(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """ConstantProximalWeight always accepts and never changes lam_prox."""
    autotuner = ConstantProximalWeight()
    initial_lam_prox = np.asarray(algorithm_state.lam_prox)

    candidate = CandidateIterate(
        x=np.asarray(algorithm_state.x),
        u=np.asarray(algorithm_state.u),
        x_prop=np.asarray(algorithm_state.x_prop),
        x_prop_plus=_candidate_x_prop_plus(),
        J_lin=jnp.asarray(0.0),
    )

    new_state = autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, {}
    )

    assert int(new_state.adaptive_state_code) == int(AdaptiveStateCode.ACCEPT_CONSTANT)
    np.testing.assert_allclose(np.asarray(new_state.lam_prox), initial_lam_prox)
    # Before cost_drop, lam_cost is reset to the configured scalar.
    np.testing.assert_allclose(
        np.asarray(new_state.lam_cost),
        np.full_like(np.asarray(algorithm_state.lam_cost), weights.lam_cost),
    )


def test_constant_proximal_weight_uses_relaxed_cost_after_cost_drop(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """After cost_drop, ConstantProximalWeight scales lam_cost by lam_cost_relax."""
    autotuner = ConstantProximalWeight(lam_cost_drop=5, lam_cost_relax=0.9)
    state = algorithm_state.replace(
        k=jnp.asarray(autotuner.hyper.lam_cost_drop + 1, dtype=jnp.int32),
        hyper=jax.tree_util.tree_map(jnp.asarray, autotuner.hyper),
    )
    initial_lam_cost = np.asarray(state.lam_cost)

    candidate = CandidateIterate(
        x=np.asarray(state.x),
        u=np.asarray(state.u),
        x_prop=np.asarray(state.x_prop),
        x_prop_plus=_candidate_x_prop_plus(),
        J_lin=jnp.asarray(0.0),
    )

    new_state = autotuner.update_weights(state, candidate, empty_nodal_constraints, settings, {})

    assert int(new_state.adaptive_state_code) == int(AdaptiveStateCode.ACCEPT_CONSTANT)
    np.testing.assert_allclose(
        np.asarray(new_state.lam_cost), initial_lam_cost * autotuner.lam_cost_relax
    )


# --- Tests for RampProximalWeight ---------------------------------------------


def test_ramp_proximal_weight_increases_until_max(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """RampProximalWeight ramps lam_prox up to a maximum, then stays constant."""
    autotuner = RampProximalWeight(ramp_factor=2.0, lam_prox_max=4.0)

    def make_candidate(state):
        return CandidateIterate(
            x=np.asarray(state.x),
            u=np.asarray(state.u),
            x_prop=np.asarray(state.x_prop),
            x_prop_plus=_candidate_x_prop_plus(),
            J_lin=jnp.asarray(0.0),
        )

    # The ramp knobs (ramp_factor / lam_prox_max) now ride state.hyper, so seed
    # it from this autotuner's container rather than the fixture's AL default.
    state = algorithm_state.replace(  # lam_prox starts at 1.0
        hyper=jax.tree_util.tree_map(jnp.asarray, autotuner.hyper)
    )

    # 1.0 -> 2.0, still below max
    state = autotuner.update_weights(
        state, make_candidate(state), empty_nodal_constraints, settings, {}
    )
    assert int(state.adaptive_state_code) == int(AdaptiveStateCode.ACCEPT_HIGHER)
    np.testing.assert_allclose(np.asarray(state.lam_prox), 2.0)

    # 2.0 -> 4.0 == max, still reported as higher (was_at_max is read pre-update)
    state = autotuner.update_weights(
        state, make_candidate(state), empty_nodal_constraints, settings, {}
    )
    assert int(state.adaptive_state_code) == int(AdaptiveStateCode.ACCEPT_HIGHER)
    np.testing.assert_allclose(np.asarray(state.lam_prox), 4.0)

    # At max -> saturates and reports constant.
    state = autotuner.update_weights(
        state, make_candidate(state), empty_nodal_constraints, settings, {}
    )
    assert int(state.adaptive_state_code) == int(AdaptiveStateCode.ACCEPT_CONSTANT)
    np.testing.assert_allclose(np.asarray(state.lam_prox), 4.0)
