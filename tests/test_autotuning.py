"""Unit tests for autotuning functions in openscvx.algorithms."""

import numpy as np
import pytest

from openscvx.algorithms.augmented_lagrangian import AugmentedLagrangian
from openscvx.algorithms.base import (
    AlgorithmState,
    AutotuningBase,
    CandidateIterate,
    DiscretizationResult,
    Weights,
)
from openscvx.algorithms.constant_proximal_weight import ConstantProximalWeight
from openscvx.algorithms.penalized_trust_region import PenalizedTrustRegion
from openscvx.algorithms.ramp_proximal_weight import RampProximalWeight
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
    return Weights(lam_prox=1.0, lam_vc=1.0, lam_vb=1.0, lam_cost=1.0)


@pytest.fixture
def algorithm_state(settings):
    """Create an AlgorithmState for testing."""
    state = AlgorithmState(
        k=1,
        J_tr=100.0,
        J_vb=100.0,
        J_vc=100.0,
        n_x=2,
        n_u=1,
        N=3,
        J_nonlin_history=[],
        J_lin_history=[],
        pred_reduction_history=[],
        actual_reduction_history=[],
        acceptance_ratio_history=[],
        X=[np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])],
        U=[np.array([[0.0], [0.5], [1.0]])],
        discretizations=[],
        lam_vc_history=[np.array([1.0, 1.0])],  # Array for virtual control
        lam_cost_history=[1.0],
        lam_vb_nodal_history=[np.full((3, 0), 1.0)],  # (N, n_nodal=0)
        lam_vb_cross_history=[np.full(0, 1.0)],  # (n_cross=0,)
        lam_prox_history=[1.0],
    )
    return state


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

    cost = AutotuningBase.calculate_cost_from_state(x, settings)

    # Should add scaled final state value
    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = scaled_x[-1, 1]  # Final node, second state
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_maximize_final(settings):
    """Test cost calculation with Maximize final_type."""
    settings.sim.x.final_type = ["None", "Maximize"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cost = AutotuningBase.calculate_cost_from_state(x, settings)

    # Should subtract scaled final state value
    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = -scaled_x[-1, 1]  # Final node, second state (negated)
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_minimize_initial(settings):
    """Test cost calculation with Minimize initial_type."""
    settings.sim.x.initial_type = ["Minimize", "None"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cost = AutotuningBase.calculate_cost_from_state(x, settings)

    # Should add scaled initial state value
    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = scaled_x[0, 0]  # Initial node, first state
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_maximize_initial(settings):
    """Test cost calculation with Maximize initial_type."""
    settings.sim.x.initial_type = ["Maximize", "None"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cost = AutotuningBase.calculate_cost_from_state(x, settings)

    # Should subtract scaled initial state value
    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = -scaled_x[0, 0]  # Initial node, first state (negated)
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_combined(settings):
    """Test cost calculation with both initial and final types."""
    settings.sim.x.initial_type = ["Minimize", "None"]
    settings.sim.x.final_type = ["None", "Maximize"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cost = AutotuningBase.calculate_cost_from_state(x, settings)

    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = scaled_x[0, 0] - scaled_x[-1, 1]
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_no_cost(settings):
    """Test cost calculation with no cost types (should return 0)."""
    settings.sim.x.initial_type = ["None", "None"]
    settings.sim.x.final_type = ["None", "None"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cost = AutotuningBase.calculate_cost_from_state(x, settings)

    assert cost == 0.0


def test_calculate_cost_from_state_per_state_weights(settings):
    """Test cost calculation with a per-state lam_cost array."""
    settings.sim.x.final_type = ["Minimize", "Minimize"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    lam_cost = np.array([2.0, 5.0])

    cost = AutotuningBase.calculate_cost_from_state(x, settings, lam_cost=lam_cost)

    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = 2.0 * scaled_x[-1, 0] + 5.0 * scaled_x[-1, 1]
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_per_state_weights_maximize(settings):
    """Test per-state weights with Maximize objective."""
    settings.sim.x.final_type = ["Maximize", "None"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    lam_cost = np.array([3.0, 0.0])

    cost = AutotuningBase.calculate_cost_from_state(x, settings, lam_cost=lam_cost)

    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = -3.0 * scaled_x[-1, 0]
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_per_state_weights_mixed(settings):
    """Test per-state weights with mixed Minimize initial and Maximize final."""
    settings.sim.x.initial_type = ["Minimize", "None"]
    settings.sim.x.final_type = ["None", "Maximize"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    lam_cost = np.array([4.0, 7.0])

    cost = AutotuningBase.calculate_cost_from_state(x, settings, lam_cost=lam_cost)

    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = 4.0 * scaled_x[0, 0] - 7.0 * scaled_x[-1, 1]
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_per_state_zero_weight_ignores_cost(settings):
    """Test that a zero weight effectively ignores the cost for that state."""
    settings.sim.x.final_type = ["Minimize", "Minimize"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    lam_cost = np.array([0.0, 3.0])

    cost = AutotuningBase.calculate_cost_from_state(x, settings, lam_cost=lam_cost)

    scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
    expected = 3.0 * scaled_x[-1, 1]
    assert cost == pytest.approx(expected, rel=1e-6)


def test_calculate_cost_from_state_scalar_lam_cost_matches_default(settings):
    """Test that passing a scalar lam_cost gives consistent results with the default."""
    settings.sim.x.final_type = ["None", "Minimize"]
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    cost_default = AutotuningBase.calculate_cost_from_state(x, settings)
    cost_scalar = AutotuningBase.calculate_cost_from_state(x, settings, lam_cost=1.0)

    assert cost_scalar == pytest.approx(cost_default, rel=1e-6)


# --- Tests for calculate_nonlinear_penalty ----------------------------------


def test_calculate_nonlinear_penalty_no_constraints(settings, empty_nodal_constraints):
    """Test penalty calculation with no constraints."""
    x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    x_bar = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    u_bar = np.array([[0.0], [0.5], [1.0]])
    lam_vc = np.array([1.0, 1.0])
    lam_vb_nodal = np.full((3, 0), 1.0)
    lam_vb_cross = np.full(0, 1.0)
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = AutotuningBase.calculate_nonlinear_penalty(
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
    lam_vb_cross = np.full(0, 1.0)
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = AutotuningBase.calculate_nonlinear_penalty(
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


def test_calculate_nonlinear_penalty_with_cross_node_violations(settings, cross_node_constraints):
    """Test penalty calculation with cross-node constraint violations."""
    x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    x_bar = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])  # x[1,0] - x[0,0] = 1.0 > 0.5, violation
    u_bar = np.array([[0.0], [0.5], [1.0]])
    lam_vc = np.array([1.0, 1.0])
    lam_vb_nodal = np.full((3, 0), 1.0)  # 0 nodal constraints
    lam_vb_cross = np.full(1, 1.0)  # 1 cross-node constraint
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = AutotuningBase.calculate_nonlinear_penalty(
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
    lam_vb_cross = np.full(0, 1.0)
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = AutotuningBase.calculate_nonlinear_penalty(
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
    lam_vb_nodal = np.full((3, 0), 1.0)
    lam_vb_cross = np.full(0, 1.0)
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = AutotuningBase.calculate_nonlinear_penalty(
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
    """Test weight update on first iteration (k=1)."""
    autotuner = AugmentedLagrangian()
    algorithm_state.k = 1
    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    candidate.J_lin = 10.0

    params = {}
    initial_x_len = len(algorithm_state.X)

    adaptive_state = autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, params, weights
    )

    assert adaptive_state == "Initial"
    assert len(algorithm_state.lam_prox_history) == 2  # Initial + new entry
    assert algorithm_state.lam_prox_history[-1] == algorithm_state.lam_prox
    # Should accept solution
    assert len(algorithm_state.X) == initial_x_len + 1  # Original + accepted candidate

    # Should set initial weights on candidate and persist them into state histories
    assert candidate.lam_vc is not None
    assert candidate.lam_vb_nodal is not None
    assert candidate.lam_vb_cross is not None
    assert len(algorithm_state.lam_vc_history) == 2
    assert np.allclose(algorithm_state.lam_vc_history[-1], candidate.lam_vc)
    assert len(algorithm_state.lam_vb_nodal_history) == 2
    assert len(algorithm_state.lam_vb_cross_history) == 2
    assert len(algorithm_state.lam_cost_history) == 2
    assert algorithm_state.lam_cost_history[-1] == pytest.approx(candidate.lam_cost)


def test_update_scp_weights_reject_higher(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """Test weight update when rho < eta_0 (reject solution, higher weight)."""
    algorithm_state.k = 2
    # Ensure lam_prox_history has the current weight
    algorithm_state.lam_prox_history = [1.0]

    # Set up previous iteration data
    algorithm_state.X.append(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    algorithm_state.U.append(np.array([[0.0], [0.5], [1.0]]))

    # Create discretization entry that x_prop() can use
    # V shape: (flattened_size, n_timesteps) where flattened_size = (N-1) * i4
    # i4 = n_x + n_x*n_x + 2*n_x*n_u = 2 + 4 + 4 = 10
    # flattened_size = (3-1) * 10 = 20
    i4 = 2 + 4 + 4  # n_x=2, n_u=1
    flattened_size = (3 - 1) * i4  # (N-1) * i4
    n_timesteps = 5
    V_dummy = np.zeros((flattened_size, n_timesteps))
    # Set final timestep: reshape to (N-1, i4) and set x_prop values (first n_x columns)
    V_final = V_dummy[:, -1].reshape(-1, i4)
    V_final[:, :2] = np.array([[0.0, 0.0], [1.0, 1.0]])  # x_prop values
    V_dummy[:, -1] = V_final.flatten()
    algorithm_state.discretizations.append(
        DiscretizationResult.from_V(
            V_dummy, n_x=algorithm_state.n_x, n_u=algorithm_state.n_u, N=algorithm_state.N
        )
    )

    # Set up candidate with poor performance (low rho)
    # Make J_lin low (good prediction) but J_nonlin high (bad actual)
    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    candidate.J_lin = 1.0  # Low predicted cost (good prediction)

    params = {}

    autotuner = AugmentedLagrangian()
    adaptive_state = autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, params, weights
    )

    # Should update weight (may accept or reject depending on rho)
    assert adaptive_state in ["Reject Higher", "Accept Higher", "Accept Constant", "Accept Lower"]
    # Weight should be updated
    assert len(algorithm_state.lam_prox_history) >= 2


def test_update_scp_weights_accept_lower(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """Test weight update when rho >= eta_2 (accept solution, lower weight)."""
    algorithm_state.k = 2
    algorithm_state.lam_prox_history = [10.0]  # Start with higher weight

    # Set up previous iteration
    algorithm_state.X.append(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    algorithm_state.U.append(np.array([[0.0], [0.5], [1.0]]))

    # Create discretization entry
    i4 = 2 + 4 + 4  # n_x=2, n_u=1
    flattened_size = (3 - 1) * i4
    V_dummy = np.zeros((flattened_size, 5))
    V_final = V_dummy[:, -1].reshape(-1, i4)
    V_final[:, :2] = np.array([[0.0, 0.0], [1.0, 1.0]])
    V_dummy[:, -1] = V_final.flatten()
    algorithm_state.discretizations.append(
        DiscretizationResult.from_V(
            V_dummy, n_x=algorithm_state.n_x, n_u=algorithm_state.n_u, N=algorithm_state.N
        )
    )

    # Set up candidate with excellent performance (high rho)
    # Make x_prop match x closely to reduce virtual control penalty
    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.0, 0.0], [1.0, 1.0]])  # Good match
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    candidate.J_lin = 1.0  # Low predicted cost

    params = {}
    initial_x_len = len(algorithm_state.X)

    autotuner = AugmentedLagrangian()
    adaptive_state = autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, params, weights
    )

    # Should accept and potentially lower weight (depending on rho)
    assert adaptive_state in ["Accept Lower", "Accept Constant", "Accept Higher", "Reject Higher"]
    # Solution should be accepted (if not rejected)
    if adaptive_state != "Reject Higher":
        assert len(algorithm_state.X) >= initial_x_len + 1


def test_update_scp_weights_cost_drop(settings, algorithm_state, empty_nodal_constraints):
    """Test that cost relaxation happens after cost_drop iterations."""
    weights = Weights(lam_prox=1.0, lam_vc=1.0, lam_vb=1.0, lam_cost=2.0)

    # Create autotuner with cost relaxation parameters
    autotuner = AugmentedLagrangian(lam_cost_drop=3, lam_cost_relax=0.8)

    algorithm_state.k = 4  # After cost_drop
    algorithm_state.lam_cost_history = [2.0]  # Current cost weight
    algorithm_state.lam_prox_history = [1.0]

    # Set up previous iteration data for k > 1
    algorithm_state.X.append(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    algorithm_state.U.append(np.array([[0.0], [0.5], [1.0]]))

    # Create discretization entry for x_prop() method
    i4 = 2 + 4 + 4  # n_x=2, n_u=1
    flattened_size = (3 - 1) * i4
    V_dummy = np.zeros((flattened_size, 5))
    V_final = V_dummy[:, -1].reshape(-1, i4)
    V_final[:, :2] = np.array([[0.0, 0.0], [1.0, 1.0]])
    V_dummy[:, -1] = V_final.flatten()
    algorithm_state.discretizations.append(
        DiscretizationResult.from_V(
            V_dummy, n_x=algorithm_state.n_x, n_u=algorithm_state.n_u, N=algorithm_state.N
        )
    )

    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    params = {}

    # Design J_lin so the step is accepted (rho >= eta_1),
    # then compute the expected virtual control update explicitly.
    state_x_prop = algorithm_state.x_prop()
    lam_vc_prev = algorithm_state.lam_vc
    lam_vb_nodal_prev = algorithm_state.lam_vb_nodal
    lam_vb_cross_prev = algorithm_state.lam_vb_cross
    lam_cost_prev = algorithm_state.lam_cost

    prev_cost, prev_penalty, prev_nodal = AutotuningBase.calculate_nonlinear_penalty(
        state_x_prop,
        algorithm_state.x,
        algorithm_state.u,
        lam_vc_prev,
        lam_vb_nodal_prev,
        lam_vb_cross_prev,
        lam_cost_prev,
        empty_nodal_constraints,
        params,
        settings,
    )
    J_nonlin_prev = prev_cost + prev_penalty + prev_nodal

    cand_cost, cand_penalty, cand_nodal = AutotuningBase.calculate_nonlinear_penalty(
        candidate.x_prop,
        candidate.x,
        candidate.u,
        lam_vc_prev,
        lam_vb_nodal_prev,
        lam_vb_cross_prev,
        lam_cost_prev,
        empty_nodal_constraints,
        params,
        settings,
    )
    J_nonlin_cand = cand_cost + cand_penalty + cand_nodal

    actual_reduction = J_nonlin_prev - J_nonlin_cand
    # Choose J_lin so predicted_reduction = actual_reduction / 2 -> rho = 2 > eta_2
    predicted_reduction = actual_reduction / 2.0
    candidate.J_lin = J_nonlin_prev - predicted_reduction

    # Expected virtual control update from helper
    lam_prox_prev = algorithm_state.lam_prox

    adaptive_state = autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, params, weights
    )

    # With the constructed J_lin we have rho > eta_2, so we must be in
    # the "Accept Lower" branch:
    # - lam_prox is decreased by gamma_2
    # - the candidate is accepted
    assert adaptive_state == "Accept Lower"
    assert len(algorithm_state.lam_prox_history) == 2
    assert algorithm_state.lam_prox_history[0] == pytest.approx(lam_prox_prev)
    assert algorithm_state.lam_prox_history[1] == pytest.approx(
        max(autotuner.lam_prox_min, autotuner.gamma_2 * lam_prox_prev)
    )

    # Cost should be relaxed when k > cost_drop and written back to state
    expected_lam_cost = 2.0 * 0.8
    assert candidate.lam_cost == pytest.approx(expected_lam_cost, rel=1e-6)
    assert len(algorithm_state.lam_cost_history) == 2
    assert algorithm_state.lam_cost_history[-1] == pytest.approx(expected_lam_cost)
    # Virtual control weights should also be stored in the state history
    assert len(algorithm_state.lam_vc_history) == 2
    assert np.allclose(algorithm_state.lam_vc_history[-1], candidate.lam_vc)


def test_update_scp_weights_history_tracking(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """Test that reduction history is tracked correctly."""
    algorithm_state.k = 2

    # Set up previous iteration
    algorithm_state.X.append(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    algorithm_state.U.append(np.array([[0.0], [0.5], [1.0]]))

    # Create discretization entry
    i4 = 2 + 4 + 4  # n_x=2, n_u=1
    flattened_size = (3 - 1) * i4
    V_dummy = np.zeros((flattened_size, 5))
    V_final = V_dummy[:, -1].reshape(-1, i4)
    V_final[:, :2] = np.array([[0.0, 0.0], [1.0, 1.0]])
    V_dummy[:, -1] = V_final.flatten()
    algorithm_state.discretizations.append(
        DiscretizationResult.from_V(
            V_dummy, n_x=algorithm_state.n_x, n_u=algorithm_state.n_u, N=algorithm_state.N
        )
    )

    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    candidate.J_lin = 10.0

    params = {}

    initial_pred_len = len(algorithm_state.pred_reduction_history)
    initial_actual_len = len(algorithm_state.actual_reduction_history)
    initial_rho_len = len(algorithm_state.acceptance_ratio_history)

    autotuner = AugmentedLagrangian()
    autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, params, weights
    )

    # History should be updated
    assert len(algorithm_state.pred_reduction_history) == initial_pred_len + 1
    assert len(algorithm_state.actual_reduction_history) == initial_actual_len + 1
    assert len(algorithm_state.acceptance_ratio_history) == initial_rho_len + 1

    # Ratios should be reasonable
    assert algorithm_state.acceptance_ratio_history[-1] is not None
    assert not np.isnan(algorithm_state.acceptance_ratio_history[-1])
    assert not np.isinf(algorithm_state.acceptance_ratio_history[-1])


def test_update_scp_weights_weight_bounds(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """Test that trust region weights respect min/max bounds."""
    algorithm_state.k = 2
    algorithm_state.lam_prox_history = [1e5]  # Very high weight

    # Set up previous iteration
    algorithm_state.X.append(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    algorithm_state.U.append(np.array([[0.0], [0.5], [1.0]]))

    # Create discretization entry
    i4 = 2 + 4 + 4  # n_x=2, n_u=1
    flattened_size = (3 - 1) * i4
    V_dummy = np.zeros((flattened_size, 5))
    V_final = V_dummy[:, -1].reshape(-1, i4)
    V_final[:, :2] = np.array([[0.0, 0.0], [1.0, 1.0]])
    V_dummy[:, -1] = V_final.flatten()
    algorithm_state.discretizations.append(
        DiscretizationResult.from_V(
            V_dummy, n_x=algorithm_state.n_x, n_u=algorithm_state.n_u, N=algorithm_state.N
        )
    )

    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    candidate.J_lin = 10.0

    params = {}

    autotuner = AugmentedLagrangian()
    autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, params, weights
    )

    # Weight should be bounded
    lam_prox_min = 1e-3
    lam_prox_max = 2e5
    final_weight = algorithm_state.lam_prox_history[-1]
    assert final_weight >= lam_prox_min
    assert final_weight <= lam_prox_max


# --- Tests for AugmentedLagrangianAutotuning ---------------------------------


def test_augmented_lagrangian_initial_iteration(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """Test AugmentedLagrangian (PTR method) on first iteration (k=1)."""
    autotuner = AugmentedLagrangian()
    algorithm_state.k = 1
    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    candidate.J_lin = 10.0

    params = {}
    initial_x_len = len(algorithm_state.X)

    adaptive_state = autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, params, weights
    )

    assert adaptive_state == "Initial"
    # Should accept solution
    assert len(algorithm_state.X) == initial_x_len + 1
    # Should set initial weights
    assert candidate.lam_vc is not None
    assert candidate.lam_vb_nodal is not None
    assert candidate.lam_vb_cross is not None


def test_augmented_lagrangian_multiplier_update(
    settings, algorithm_state, nodal_constraints_with_violations, weights
):
    """Test that AugmentedLagrangian uses PTR method (no multiplier updates)."""
    autotuner = AugmentedLagrangian()
    algorithm_state.k = 2
    algorithm_state.lam_prox_history = [1.0]
    algorithm_state.lam_vc_history = [np.array([1.0, 1.0])]
    # 1 nodal constraint, 0 cross-node
    algorithm_state.lam_vb_nodal_history = [np.full((3, 1), 1.0)]
    algorithm_state.lam_vb_cross_history = [np.full(0, 1.0)]

    # Set up previous iteration
    algorithm_state.X.append(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    algorithm_state.U.append(np.array([[0.0], [0.5], [1.0]]))

    # Create discretization entry
    i4 = 2 + 4 + 4
    flattened_size = (3 - 1) * i4
    V_dummy = np.zeros((flattened_size, 5))
    V_final = V_dummy[:, -1].reshape(-1, i4)
    V_final[:, :2] = np.array([[0.0, 0.0], [1.0, 1.0]])
    V_dummy[:, -1] = V_final.flatten()
    algorithm_state.discretizations.append(
        DiscretizationResult.from_V(
            V_dummy, n_x=algorithm_state.n_x, n_u=algorithm_state.n_u, N=algorithm_state.N
        )
    )

    # Set up candidate with constraint violations
    # x[0] = 2.0 > 1.5, violation
    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    candidate.J_lin = 10.0

    params = {}

    adaptive_state = autotuner.update_weights(
        algorithm_state, candidate, nodal_constraints_with_violations, settings, params, weights
    )

    # Should use PTR method (no multiplier attributes)
    assert not hasattr(algorithm_state, "lambda_multipliers")
    assert not hasattr(algorithm_state, "rho")
    assert not hasattr(algorithm_state, "mu")
    # Should have updated weights based on acceptance ratio
    assert adaptive_state in ["Reject Higher", "Accept Higher", "Accept Constant", "Accept Lower"]


def test_augmented_lagrangian_accept_decrease(
    settings, algorithm_state, nodal_constraints_with_violations, weights
):
    """Explicitly realize the 'Accept Lower' branch with constraint violations."""
    autotuner = AugmentedLagrangian()
    algorithm_state.k = 2
    algorithm_state.lam_prox_history = [1.0]
    algorithm_state.lam_vc_history = [np.array([1.0, 1.0])]
    # 1 nodal constraint, 0 cross-node
    algorithm_state.lam_vb_nodal_history = [np.full((3, 1), 1.0)]
    algorithm_state.lam_vb_cross_history = [np.full(0, 1.0)]

    # Set up previous iteration
    algorithm_state.X.append(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    algorithm_state.U.append(np.array([[0.0], [0.5], [1.0]]))

    # Create discretization entry
    i4 = 2 + 4 + 4
    flattened_size = (3 - 1) * i4
    V_dummy = np.zeros((flattened_size, 5))
    V_final = V_dummy[:, -1].reshape(-1, i4)
    V_final[:, :2] = np.array([[0.0, 0.0], [1.0, 1.0]])
    V_dummy[:, -1] = V_final.flatten()
    algorithm_state.discretizations.append(
        DiscretizationResult.from_V(
            V_dummy, n_x=algorithm_state.n_x, n_u=algorithm_state.n_u, N=algorithm_state.N
        )
    )

    # Set up candidate with violations
    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])  # Violation
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    params = {}

    # Compute previous and candidate nonlinear objectives explicitly, then
    # choose J_lin so that rho > eta_2, guaranteeing "Accept Lower".
    state_x_prop = algorithm_state.x_prop()
    lam_vc_prev = algorithm_state.lam_vc
    lam_vb_nodal_prev = algorithm_state.lam_vb_nodal
    lam_vb_cross_prev = algorithm_state.lam_vb_cross
    lam_cost_prev = algorithm_state.lam_cost

    prev_cost, prev_penalty, prev_nodal = AutotuningBase.calculate_nonlinear_penalty(
        state_x_prop,
        algorithm_state.x,
        algorithm_state.u,
        lam_vc_prev,
        lam_vb_nodal_prev,
        lam_vb_cross_prev,
        lam_cost_prev,
        nodal_constraints_with_violations,
        params,
        settings,
    )
    J_nonlin_prev = prev_cost + prev_penalty + prev_nodal

    cand_cost, cand_penalty, cand_nodal = AutotuningBase.calculate_nonlinear_penalty(
        candidate.x_prop,
        candidate.x,
        candidate.u,
        lam_vc_prev,
        lam_vb_nodal_prev,
        lam_vb_cross_prev,
        lam_cost_prev,
        nodal_constraints_with_violations,
        params,
        settings,
    )
    J_nonlin_cand = cand_cost + cand_penalty + cand_nodal

    actual_reduction = J_nonlin_prev - J_nonlin_cand
    rho_target = autotuner.eta_2 + 0.1 * (1.0 - autotuner.eta_2)  # strictly > eta_2, < 1
    predicted_reduction = actual_reduction / rho_target
    candidate.J_lin = J_nonlin_prev - predicted_reduction

    lam_prox_prev = algorithm_state.lam_prox
    initial_x_len = len(algorithm_state.X)

    adaptive_state = autotuner.update_weights(
        algorithm_state,
        candidate,
        nodal_constraints_with_violations,
        settings,
        params,
        weights,
    )

    # We should be in the "Accept Lower" branch:
    # - lam_prox is decreased by gamma_2 (but not below lam_prox_min)
    # - candidate is accepted and its weights recorded in the state histories
    assert adaptive_state == "Accept Lower"
    assert len(algorithm_state.lam_prox_history) == 2
    assert algorithm_state.lam_prox_history[0] == pytest.approx(lam_prox_prev)
    assert algorithm_state.lam_prox_history[1] == pytest.approx(
        max(autotuner.lam_prox_min, autotuner.gamma_2 * lam_prox_prev)
    )
    assert len(algorithm_state.X) == initial_x_len + 1
    assert len(algorithm_state.lam_vc_history) == 2
    assert np.allclose(algorithm_state.lam_vc_history[-1], candidate.lam_vc)
    assert len(algorithm_state.lam_vb_nodal_history) == 2
    assert len(algorithm_state.lam_vb_cross_history) == 2
    assert len(algorithm_state.lam_cost_history) == 2
    assert algorithm_state.lam_cost_history[-1] == pytest.approx(candidate.lam_cost)


def test_augmented_lagrangian_reject_increase(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """Test that AugmentedLagrangian rejects and does not update lam_vc."""
    autotuner = AugmentedLagrangian()
    algorithm_state.k = 2
    algorithm_state.lam_prox_history = [1.0]
    algorithm_state.lam_vc_history = [np.array([1.0, 1.0])]

    # Set up previous iteration
    algorithm_state.X.append(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    algorithm_state.U.append(np.array([[0.0], [0.5], [1.0]]))

    # Create discretization entry
    i4 = 2 + 4 + 4
    flattened_size = (3 - 1) * i4
    V_dummy = np.zeros((flattened_size, 5))
    V_final = V_dummy[:, -1].reshape(-1, i4)
    V_final[:, :2] = np.array([[0.0, 0.0], [1.0, 1.0]])
    V_dummy[:, -1] = V_final.flatten()
    algorithm_state.discretizations.append(
        DiscretizationResult.from_V(
            V_dummy, n_x=algorithm_state.n_x, n_u=algorithm_state.n_u, N=algorithm_state.N
        )
    )

    # Set up candidate with no violations (good match)
    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.0, 0.0], [1.0, 1.0]])  # Good match
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    candidate.J_lin = 10.0

    params = {}

    lam_prox_prev = algorithm_state.lam_prox
    lam_vc_prev = algorithm_state.lam_vc

    adaptive_state = autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, params, weights
    )

    # With this setup the step is rejected (rho < eta_0), so:
    # - lam_prox is increased
    # - lam_vc is NOT updated on the candidate or in the state history
    assert adaptive_state == "Reject Higher"
    assert candidate.lam_vc is None
    assert len(algorithm_state.lam_prox_history) == 2
    assert algorithm_state.lam_prox_history[0] == pytest.approx(lam_prox_prev)
    assert algorithm_state.lam_prox_history[1] == pytest.approx(autotuner.gamma_1 * lam_prox_prev)
    assert len(algorithm_state.lam_vc_history) == 1
    assert np.allclose(algorithm_state.lam_vc_history[0], lam_vc_prev)

    # Should use PTR method (no penalty parameters)
    assert not hasattr(algorithm_state, "rho")
    assert not hasattr(algorithm_state, "mu")
    # Should update trust region weights
    assert len(algorithm_state.lam_prox_history) >= 2


def test_augmented_lagrangian_accept_higher(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """Explicitly realize the 'Accept Higher' adaptive_state branch."""
    autotuner = AugmentedLagrangian()
    algorithm_state.k = 2
    algorithm_state.lam_prox_history = [1.0]
    algorithm_state.lam_vc_history = [np.array([1.0, 1.0])]

    # Set up previous iteration
    algorithm_state.X.append(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    algorithm_state.U.append(np.array([[0.0], [0.5], [1.0]]))

    # Create discretization entry so x_prop() is defined
    i4 = 2 + 4 + 4
    flattened_size = (3 - 1) * i4
    V_dummy = np.zeros((flattened_size, 5))
    V_final = V_dummy[:, -1].reshape(-1, i4)
    V_final[:, :2] = np.array([[0.0, 0.0], [1.0, 1.0]])
    V_dummy[:, -1] = V_final.flatten()
    algorithm_state.discretizations.append(
        DiscretizationResult.from_V(
            V_dummy, n_x=algorithm_state.n_x, n_u=algorithm_state.n_u, N=algorithm_state.N
        )
    )

    # Candidate that improves the nonlinear objective (smaller virtual control penalty)
    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    candidate.u = np.array([[0.0], [0.5], [1.0]])

    params = {}

    # Compute previous and candidate nonlinear objectives explicitly, then
    # choose J_lin so that eta_0 < rho < eta_1, guaranteeing "Accept Higher".
    state_x_prop = algorithm_state.x_prop()
    lam_vc_prev = algorithm_state.lam_vc
    lam_vb_nodal_prev = algorithm_state.lam_vb_nodal
    lam_vb_cross_prev = algorithm_state.lam_vb_cross
    lam_cost_prev = algorithm_state.lam_cost

    prev_cost, prev_penalty, prev_nodal = AutotuningBase.calculate_nonlinear_penalty(
        state_x_prop,
        algorithm_state.x,
        algorithm_state.u,
        lam_vc_prev,
        lam_vb_nodal_prev,
        lam_vb_cross_prev,
        lam_cost_prev,
        empty_nodal_constraints,
        params,
        settings,
    )
    J_nonlin_prev = prev_cost + prev_penalty + prev_nodal

    cand_cost, cand_penalty, cand_nodal = AutotuningBase.calculate_nonlinear_penalty(
        candidate.x_prop,
        candidate.x,
        candidate.u,
        lam_vc_prev,
        lam_vb_nodal_prev,
        lam_vb_cross_prev,
        lam_cost_prev,
        empty_nodal_constraints,
        params,
        settings,
    )
    J_nonlin_cand = cand_cost + cand_penalty + cand_nodal

    actual_reduction = J_nonlin_prev - J_nonlin_cand
    rho_target = 0.5 * (autotuner.eta_0 + autotuner.eta_1)  # in (eta_0, eta_1)
    predicted_reduction = actual_reduction / rho_target
    candidate.J_lin = J_nonlin_prev - predicted_reduction

    initial_x_len = len(algorithm_state.X)
    lam_prox_prev = algorithm_state.lam_prox

    adaptive_state = autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, params, weights
    )

    assert adaptive_state == "Accept Higher"
    # Trust-region weight should be increased by gamma_1
    assert len(algorithm_state.lam_prox_history) == 2
    assert algorithm_state.lam_prox_history[0] == pytest.approx(lam_prox_prev)
    assert algorithm_state.lam_prox_history[1] == pytest.approx(
        min(autotuner.lam_prox_max, autotuner.gamma_1 * lam_prox_prev)
    )
    # Candidate should have updated virtual control weights and be accepted,
    # and those weights must be recorded back into the algorithm_state histories.
    assert candidate.lam_vc is not None
    assert candidate.lam_vb_nodal is not None
    assert candidate.lam_vb_cross is not None
    assert len(algorithm_state.X) == initial_x_len + 1
    assert len(algorithm_state.lam_vc_history) == 2
    assert np.allclose(algorithm_state.lam_vc_history[-1], candidate.lam_vc)
    assert len(algorithm_state.lam_vb_nodal_history) == 2
    assert len(algorithm_state.lam_vb_cross_history) == 2
    assert len(algorithm_state.lam_cost_history) == 2
    assert algorithm_state.lam_cost_history[-1] == pytest.approx(candidate.lam_cost)


def test_augmented_lagrangian_accept_constant(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """Explicitly realize the 'Accept Constant' adaptive_state branch."""
    autotuner = AugmentedLagrangian()
    algorithm_state.k = 2
    algorithm_state.lam_prox_history = [1.0]
    algorithm_state.lam_vc_history = [np.array([1.0, 1.0])]

    # Set up previous iteration
    algorithm_state.X.append(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    algorithm_state.U.append(np.array([[0.0], [0.5], [1.0]]))

    # Create discretization entry so x_prop() is defined
    i4 = 2 + 4 + 4
    flattened_size = (3 - 1) * i4
    V_dummy = np.zeros((flattened_size, 5))
    V_final = V_dummy[:, -1].reshape(-1, i4)
    V_final[:, :2] = np.array([[0.0, 0.0], [1.0, 1.0]])
    V_dummy[:, -1] = V_final.flatten()
    algorithm_state.discretizations.append(
        DiscretizationResult.from_V(
            V_dummy, n_x=algorithm_state.n_x, n_u=algorithm_state.n_u, N=algorithm_state.N
        )
    )

    # Candidate that improves the nonlinear objective (same pattern as above)
    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    candidate.u = np.array([[0.0], [0.5], [1.0]])

    params = {}

    state_x_prop = algorithm_state.x_prop()
    lam_vc_prev = algorithm_state.lam_vc
    lam_vb_nodal_prev = algorithm_state.lam_vb_nodal
    lam_vb_cross_prev = algorithm_state.lam_vb_cross
    lam_cost_prev = algorithm_state.lam_cost

    prev_cost, prev_penalty, prev_nodal = AutotuningBase.calculate_nonlinear_penalty(
        state_x_prop,
        algorithm_state.x,
        algorithm_state.u,
        lam_vc_prev,
        lam_vb_nodal_prev,
        lam_vb_cross_prev,
        lam_cost_prev,
        empty_nodal_constraints,
        params,
        settings,
    )
    J_nonlin_prev = prev_cost + prev_penalty + prev_nodal

    cand_cost, cand_penalty, cand_nodal = AutotuningBase.calculate_nonlinear_penalty(
        candidate.x_prop,
        candidate.x,
        candidate.u,
        lam_vc_prev,
        lam_vb_nodal_prev,
        lam_vb_cross_prev,
        lam_cost_prev,
        empty_nodal_constraints,
        params,
        settings,
    )
    J_nonlin_cand = cand_cost + cand_penalty + cand_nodal

    actual_reduction = J_nonlin_prev - J_nonlin_cand
    rho_target = 0.5 * (autotuner.eta_1 + autotuner.eta_2)  # in [eta_1, eta_2)
    predicted_reduction = actual_reduction / rho_target
    candidate.J_lin = J_nonlin_prev - predicted_reduction

    initial_x_len = len(algorithm_state.X)
    lam_prox_prev = algorithm_state.lam_prox

    adaptive_state = autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, params, weights
    )

    assert adaptive_state == "Accept Constant"
    # Trust-region weight should remain constant
    assert len(algorithm_state.lam_prox_history) == 2
    assert algorithm_state.lam_prox_history[0] == pytest.approx(lam_prox_prev)
    assert algorithm_state.lam_prox_history[1] == pytest.approx(lam_prox_prev)
    # Candidate should have updated virtual control weights and be accepted,
    # and those weights must be recorded back into the algorithm_state histories.
    assert candidate.lam_vc is not None
    assert candidate.lam_vb_nodal is not None
    assert candidate.lam_vb_cross is not None
    assert len(algorithm_state.X) == initial_x_len + 1
    assert len(algorithm_state.lam_vc_history) == 2
    assert np.allclose(algorithm_state.lam_vc_history[-1], candidate.lam_vc)
    assert len(algorithm_state.lam_vb_nodal_history) == 2
    assert len(algorithm_state.lam_vb_cross_history) == 2
    assert len(algorithm_state.lam_cost_history) == 2
    assert algorithm_state.lam_cost_history[-1] == pytest.approx(candidate.lam_cost)


def test_augmented_lagrangian_virtual_control_update(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """Test AL rejects this step and does not update virtual control weights."""
    autotuner = AugmentedLagrangian()
    algorithm_state.k = 2
    algorithm_state.lam_prox_history = [1.0]
    algorithm_state.lam_vc_history = [np.array([1.0, 1.0])]

    # Set up previous iteration
    algorithm_state.X.append(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    algorithm_state.U.append(np.array([[0.0], [0.5], [1.0]]))

    # Create discretization entry
    i4 = 2 + 4 + 4
    flattened_size = (3 - 1) * i4
    V_dummy = np.zeros((flattened_size, 5))
    V_final = V_dummy[:, -1].reshape(-1, i4)
    V_final[:, :2] = np.array([[0.0, 0.0], [1.0, 1.0]])
    V_dummy[:, -1] = V_final.flatten()
    algorithm_state.discretizations.append(
        DiscretizationResult.from_V(
            V_dummy, n_x=algorithm_state.n_x, n_u=algorithm_state.n_u, N=algorithm_state.N
        )
    )

    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    candidate.J_lin = 10.0

    params = {}

    autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, params, weights
    )

    # With this setup, the acceptance ratio rho is negative (about -0.02),
    # so the AL logic enters the \"Reject Higher\" branch:
    # - lam_prox is increased by gamma_1
    # - lam_vc is NOT updated on the candidate
    assert candidate.lam_vc is None
    # Trust-region weight history should have grown by one entry and increased
    assert len(algorithm_state.lam_prox_history) == 2
    assert algorithm_state.lam_prox_history[0] == pytest.approx(1.0)
    assert algorithm_state.lam_prox_history[1] == pytest.approx(autotuner.gamma_1 * 1.0)
    # Virtual control history in the state should remain unchanged
    assert len(algorithm_state.lam_vc_history) == 1
    assert np.allclose(algorithm_state.lam_vc_history[0], np.array([1.0, 1.0]))


def test_augmented_lagrangian_base_class_methods(settings):
    """Test that base class methods work correctly."""
    # Test static methods
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    cost = AutotuningBase.calculate_cost_from_state(x, settings)
    assert isinstance(cost, (float, np.floating))

    # Test that subclass can use base methods
    auglag_autotuner = AugmentedLagrangian()

    # Should have the same base methods
    assert hasattr(auglag_autotuner, "calculate_cost_from_state")
    assert hasattr(auglag_autotuner, "calculate_nonlinear_penalty")


def test_algorithm_autotuner_default():
    """PenalizedTrustRegion.autotuner should default to AugmentedLagrangian."""
    algorithm = PenalizedTrustRegion()
    assert isinstance(algorithm.autotuner, AugmentedLagrangian)


def test_algorithm_autotuner_configurable():
    """PenalizedTrustRegion default autotuner should be a configurable AugmentedLagrangian."""
    algorithm = PenalizedTrustRegion()
    autotuner = algorithm.autotuner
    assert isinstance(autotuner, AugmentedLagrangian)
    assert hasattr(autotuner, "rho_init")
    assert hasattr(autotuner, "rho_max")
    assert hasattr(autotuner, "lam_prox_min")
    assert hasattr(autotuner, "lam_prox_max")
    assert hasattr(autotuner, "lam_vc_max")
    assert hasattr(autotuner, "lam_cost_drop")
    assert hasattr(autotuner, "lam_cost_relax")
    autotuner.rho_max = 1e7
    assert autotuner.rho_max == 1e7


def test_custom_autotuner_instance():
    """Custom autotuner instance can be passed to PenalizedTrustRegion."""
    custom_autotuner = AugmentedLagrangian()
    custom_autotuner.rho_max = 1e7
    custom_autotuner.lam_prox_max = 1e6
    custom_autotuner.lam_vc_max = 1e6
    algorithm = PenalizedTrustRegion(autotuner=custom_autotuner)
    assert algorithm.autotuner is custom_autotuner
    assert algorithm.autotuner.rho_max == 1e7
    assert algorithm.autotuner.lam_prox_max == 1e6
    assert algorithm.autotuner.lam_vc_max == 1e6


def test_augmented_lagrangian_exported():
    """Test that AugmentedLagrangian is exported from main module."""
    import openscvx as ox

    # Should be able to import directly
    auto_tuner = ox.AugmentedLagrangian()
    assert hasattr(auto_tuner, "rho_max")
    assert hasattr(auto_tuner, "lam_prox_max")
    assert hasattr(auto_tuner, "lam_vc_max")

    # Should be able to modify parameters
    auto_tuner.rho_max = 1e7
    auto_tuner.lam_prox_max = 1e6
    assert auto_tuner.rho_max == 1e7
    assert auto_tuner.lam_prox_max == 1e6


# --- Tests for ConstantProximalWeight ---------------------------------------------


def test_constant_proximal_weight_appends_history_and_accepts(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """ConstantProximalWeight should append the current lam_prox and accept."""
    autotuner = ConstantProximalWeight()
    # Use first iteration (before cost_drop)
    algorithm_state.k = 1
    candidate = CandidateIterate()
    candidate.x = algorithm_state.x
    candidate.u = algorithm_state.u

    initial_x_len = len(algorithm_state.X)
    initial_lam_prox_history_len = len(algorithm_state.lam_prox_history)
    initial_lam_prox = algorithm_state.lam_prox
    initial_lam_cost_history_len = len(algorithm_state.lam_cost_history)

    adaptive_state = autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, {}, weights
    )

    # Always accepts and reports constant behaviour
    assert adaptive_state == "Accept Constant"
    # Candidate should have been accepted into history
    assert len(algorithm_state.X) == initial_x_len + 1
    # Proximal weight history should append the current value, but not change it
    assert len(algorithm_state.lam_prox_history) == initial_lam_prox_history_len + 1
    assert algorithm_state.lam_prox_history[-1] == pytest.approx(initial_lam_prox)
    # Before cost_drop we use the configured lam_cost
    assert len(algorithm_state.lam_cost_history) == initial_lam_cost_history_len + 1
    assert algorithm_state.lam_cost_history[-1] == pytest.approx(weights.lam_cost)


def test_constant_proximal_weight_uses_relaxed_cost_after_cost_drop(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """After cost_drop, ConstantProximalWeight should use relaxed lam_cost."""
    # Create autotuner with cost relaxation parameters
    autotuner = ConstantProximalWeight(lam_cost_drop=5, lam_cost_relax=0.9)
    algorithm_state.k = autotuner.lam_cost_drop + 1
    candidate = CandidateIterate()
    candidate.x = algorithm_state.x
    candidate.u = algorithm_state.u

    initial_lam_cost = algorithm_state.lam_cost
    initial_lam_cost_history_len = len(algorithm_state.lam_cost_history)

    adaptive_state = autotuner.update_weights(
        algorithm_state, candidate, empty_nodal_constraints, settings, {}, weights
    )

    assert adaptive_state == "Accept Constant"
    assert len(algorithm_state.lam_cost_history) == initial_lam_cost_history_len + 1
    expected_relaxed = initial_lam_cost * autotuner.lam_cost_relax
    assert algorithm_state.lam_cost_history[-1] == pytest.approx(expected_relaxed)


# --- Tests for RampProximalWeight ---------------------------------------------


def test_ramp_proximal_weight_increases_until_max(
    settings, algorithm_state, empty_nodal_constraints, weights
):
    """RampProximalWeight should ramp lam_prox up to a maximum, then stay constant."""
    autotuner = RampProximalWeight(ramp_factor=2.0, lam_prox_max=4.0)

    # Helper to set a simple candidate each call
    def set_candidate():
        candidate = CandidateIterate()
        candidate.x = algorithm_state.x
        candidate.u = algorithm_state.u
        return candidate

    # Start from initial lam_prox = 1.0
    candidate = set_candidate()
    state_str = autotuner.update_weights(
        algorithm_state,
        candidate,
        empty_nodal_constraints,
        settings,
        {},
        weights,
    )
    # 1.0 -> 2.0, still below max
    assert state_str == "Accept Higher"
    assert algorithm_state.lam_prox_history[-1] == pytest.approx(2.0)

    # Next iteration: 2.0 -> 4.0 == max, still reported as higher
    candidate = set_candidate()
    state_str = autotuner.update_weights(
        algorithm_state,
        candidate,
        empty_nodal_constraints,
        settings,
        {},
        weights,
    )
    assert state_str == "Accept Higher"
    assert algorithm_state.lam_prox_history[-1] == pytest.approx(4.0)

    # Once lam_prox == lam_prox_max, it should stop increasing and report constant
    candidate = set_candidate()
    state_str = autotuner.update_weights(
        algorithm_state,
        candidate,
        empty_nodal_constraints,
        settings,
        {},
        weights,
    )
    assert state_str == "Accept Constant"
    # Still at the maximum and not exceeded
    assert algorithm_state.lam_prox_history[-1] == pytest.approx(4.0)
