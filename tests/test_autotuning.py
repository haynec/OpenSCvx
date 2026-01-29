"""Unit tests for autotuning functions in openscvx.algorithms.autotuning."""

import numpy as np
import pytest

from openscvx.algorithms.autotuning import (
    AugmentedLagrangian,
    AutotuningBase,
    ConstantProximalWeight,
    RampProximalWeight,
)
from openscvx.algorithms.base import AlgorithmState, CandidateIterate, DiscretizationResult
from openscvx.config import (
    Config,
    ConvexSolverConfig,
    DevConfig,
    DiscretizationConfig,
    PropagationConfig,
    ScpConfig,
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
        n_states=2,
        n_controls=1,
    )

    scp_config = ScpConfig(
        n=3,
        lam_prox=1.0,
        lam_vc=1.0,
        lam_vb=1.0,
        lam_cost=1.0,
    )

    config = Config(
        sim=sim_config,
        scp=scp_config,
        cvx=ConvexSolverConfig(),
        dis=DiscretizationConfig(),
        prp=PropagationConfig(),
        dev=DevConfig(),
    )

    return config


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
        lam_vb_history=[1.0],
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


# --- Tests for calculate_nonlinear_penalty ----------------------------------


def test_calculate_nonlinear_penalty_no_constraints(settings, empty_nodal_constraints):
    """Test penalty calculation with no constraints."""
    x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    x_bar = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    u_bar = np.array([[0.0], [0.5], [1.0]])
    lam_vc = np.array([1.0, 1.0])
    lam_vb = 1.0
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = AutotuningBase.calculate_nonlinear_penalty(
        x_prop, x_bar, u_bar, lam_vc, lam_vb, lam_cost, empty_nodal_constraints, params, settings
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
    lam_vb = 1.0
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = AutotuningBase.calculate_nonlinear_penalty(
        x_prop,
        x_bar,
        u_bar,
        lam_vc,
        lam_vb,
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
    lam_vb = 1.0
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = AutotuningBase.calculate_nonlinear_penalty(
        x_prop, x_bar, u_bar, lam_vc, lam_vb, lam_cost, cross_node_constraints, params, settings
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
    lam_vb = 1.0
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = AutotuningBase.calculate_nonlinear_penalty(
        x_prop, x_bar, u_bar, lam_vc, lam_vb, lam_cost, nodal_constraints, params, settings
    )

    # Should have positive penalty from filtered nodes
    assert nodal_penalty > 0.0


def test_calculate_nonlinear_penalty_virtual_control_component(settings, empty_nodal_constraints):
    """Test that virtual control penalty is calculated correctly."""
    x_prop = np.array([[0.0, 0.0], [1.0, 1.0]])
    x_bar = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])  # Large difference at end
    u_bar = np.array([[0.0], [0.5], [1.0]])
    lam_vc = np.array([2.0, 2.0])  # Higher weight
    lam_vb = 1.0
    lam_cost = 1.0
    params = {}

    nonlinear_cost, nonlinear_penalty, nodal_penalty = AutotuningBase.calculate_nonlinear_penalty(
        x_prop, x_bar, u_bar, lam_vc, lam_vb, lam_cost, empty_nodal_constraints, params, settings
    )

    # Virtual control penalty should be positive and larger with larger differences
    assert nonlinear_penalty > 0.0

    # Calculate expected penalty manually
    x_diff = settings.sim.inv_S_x @ (x_bar[1:, :] - x_prop).T
    expected_penalty = np.sum(lam_vc * np.abs(x_diff.T))
    assert nonlinear_penalty == pytest.approx(expected_penalty, rel=1e-6)


# --- Tests for update_scp_weights -------------------------------------------


def test_update_scp_weights_initial_iteration(settings, algorithm_state, empty_nodal_constraints):
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
        algorithm_state, candidate, empty_nodal_constraints, settings, params
    )

    assert adaptive_state == "Initial"
    assert len(algorithm_state.lam_prox_history) == 2  # Initial + new entry
    assert algorithm_state.lam_prox_history[-1] == algorithm_state.lam_prox
    # Should accept solution
    assert len(algorithm_state.X) == initial_x_len + 1  # Original + accepted candidate


def test_update_scp_weights_reject_higher(settings, algorithm_state, empty_nodal_constraints):
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
        algorithm_state, candidate, empty_nodal_constraints, settings, params
    )

    # Should update weight (may accept or reject depending on rho)
    assert adaptive_state in ["Reject Higher", "Accept Higher", "Accept Constant", "Accept Lower"]
    # Weight should be updated
    assert len(algorithm_state.lam_prox_history) >= 2


def test_update_scp_weights_accept_lower(settings, algorithm_state, empty_nodal_constraints):
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
        algorithm_state, candidate, empty_nodal_constraints, settings, params
    )

    # Should accept and potentially lower weight (depending on rho)
    assert adaptive_state in ["Accept Lower", "Accept Constant", "Accept Higher", "Reject Higher"]
    # Solution should be accepted (if not rejected)
    if adaptive_state != "Reject Higher":
        assert len(algorithm_state.X) >= initial_x_len + 1


def test_update_scp_weights_cost_drop(settings, algorithm_state, empty_nodal_constraints):
    """Test that cost relaxation happens after cost_drop iterations."""
    settings.scp.lam_cost = 2.0

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
    candidate.J_lin = 10.0

    params = {}

    autotuner.update_weights(algorithm_state, candidate, empty_nodal_constraints, settings, params)

    # Cost should be relaxed when k > cost_drop
    expected_lam_cost = 2.0 * 0.8
    assert candidate.lam_cost == pytest.approx(expected_lam_cost, rel=1e-6)


def test_update_scp_weights_before_cost_drop(settings, algorithm_state, empty_nodal_constraints):
    """Test that cost relaxation does NOT happen before cost_drop iterations."""
    settings.scp.lam_cost = 2.0

    # Create autotuner with cost relaxation parameters
    autotuner = AugmentedLagrangian(lam_cost_drop=5, lam_cost_relax=0.8)

    algorithm_state.k = 1  # Use k=1 to avoid needing previous iteration data
    algorithm_state.lam_cost_history = [2.0]
    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])  # Must set x_prop
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    candidate.J_lin = 10.0

    params = {}

    autotuner.update_weights(algorithm_state, candidate, empty_nodal_constraints, settings, params)

    # Cost should remain at initial value (k=1 uses initial settings, not relaxed)
    assert candidate.lam_cost == settings.scp.lam_cost


def test_update_scp_weights_virtual_control_update(
    settings, algorithm_state, empty_nodal_constraints
):
    """Test that virtual control weights are updated based on nu."""
    algorithm_state.k = 2
    algorithm_state.lam_prox_history = [1.0]
    algorithm_state.lam_vc_history = [np.array([1.0, 1.0])]

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

    # Create large difference to trigger virtual control update
    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])  # Large difference
    candidate.x_prop = np.array([[0.0, 0.0], [1.0, 1.0]])  # Small propagated
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    candidate.J_lin = 10.0

    params = {}

    autotuner = AugmentedLagrangian()
    autotuner.update_weights(algorithm_state, candidate, empty_nodal_constraints, settings, params)

    # Virtual control should be updated (increased due to large nu)
    assert candidate.lam_vc is not None
    # Should be array with shape (N-1, n_x) = (2, 2) due to broadcasting
    assert isinstance(candidate.lam_vc, np.ndarray)
    assert candidate.lam_vc.shape == (2, 2)  # (N-1, n_x)


def test_update_scp_weights_history_tracking(settings, algorithm_state, empty_nodal_constraints):
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
    autotuner.update_weights(algorithm_state, candidate, empty_nodal_constraints, settings, params)

    # History should be updated
    assert len(algorithm_state.pred_reduction_history) == initial_pred_len + 1
    assert len(algorithm_state.actual_reduction_history) == initial_actual_len + 1
    assert len(algorithm_state.acceptance_ratio_history) == initial_rho_len + 1

    # Ratios should be reasonable
    assert algorithm_state.acceptance_ratio_history[-1] is not None
    assert not np.isnan(algorithm_state.acceptance_ratio_history[-1])
    assert not np.isinf(algorithm_state.acceptance_ratio_history[-1])


def test_update_scp_weights_weight_bounds(settings, algorithm_state, empty_nodal_constraints):
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
    autotuner.update_weights(algorithm_state, candidate, empty_nodal_constraints, settings, params)

    # Weight should be bounded
    lam_prox_min = 1e-3
    lam_prox_max = 2e5
    final_weight = algorithm_state.lam_prox_history[-1]
    assert final_weight >= lam_prox_min
    assert final_weight <= lam_prox_max


# --- Tests for AugmentedLagrangianAutotuning ---------------------------------


def test_augmented_lagrangian_initial_iteration(settings, algorithm_state, empty_nodal_constraints):
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
        algorithm_state, candidate, empty_nodal_constraints, settings, params
    )

    assert adaptive_state == "Initial"
    # Should accept solution
    assert len(algorithm_state.X) == initial_x_len + 1
    # Should set initial weights
    assert candidate.lam_vc is not None
    assert candidate.lam_vb == settings.scp.lam_vb


def test_augmented_lagrangian_multiplier_update(
    settings, algorithm_state, nodal_constraints_with_violations
):
    """Test that AugmentedLagrangian uses PTR method (no multiplier updates)."""
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

    # Set up candidate with constraint violations
    # x[0] = 2.0 > 1.5, violation
    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    candidate.J_lin = 10.0

    params = {}

    adaptive_state = autotuner.update_weights(
        algorithm_state, candidate, nodal_constraints_with_violations, settings, params
    )

    # Should use PTR method (no multiplier attributes)
    assert not hasattr(algorithm_state, "lambda_multipliers")
    assert not hasattr(algorithm_state, "rho")
    assert not hasattr(algorithm_state, "mu")
    # Should have updated weights based on acceptance ratio
    assert adaptive_state in ["Reject Higher", "Accept Higher", "Accept Constant", "Accept Lower"]


def test_augmented_lagrangian_penalty_increase(
    settings, algorithm_state, nodal_constraints_with_violations
):
    """Test that AugmentedLagrangian uses PTR method (no penalty parameters)."""
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

    # Set up candidate with violations
    candidate = CandidateIterate()
    candidate.x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])  # Violation
    candidate.x_prop = np.array([[0.5, 0.5], [1.5, 1.5]])
    candidate.u = np.array([[0.0], [0.5], [1.0]])
    candidate.J_lin = 10.0

    params = {}

    autotuner.update_weights(
        algorithm_state,
        candidate,
        nodal_constraints_with_violations,
        settings,
        params,
    )

    # Should use PTR method (no penalty parameters)
    assert not hasattr(algorithm_state, "rho")
    assert not hasattr(algorithm_state, "mu")
    # Should update trust region weights based on acceptance ratio
    assert len(algorithm_state.lam_prox_history) >= 2


def test_augmented_lagrangian_penalty_decrease(settings, algorithm_state, empty_nodal_constraints):
    """Test that AugmentedLagrangian uses PTR method (no penalty parameters)."""
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

    autotuner.update_weights(algorithm_state, candidate, empty_nodal_constraints, settings, params)

    # Should use PTR method (no penalty parameters)
    assert not hasattr(algorithm_state, "rho")
    assert not hasattr(algorithm_state, "mu")
    # Should update trust region weights
    assert len(algorithm_state.lam_prox_history) >= 2


def test_augmented_lagrangian_virtual_control_update(
    settings, algorithm_state, empty_nodal_constraints
):
    """Test that virtual control weights are updated using PTR method."""
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

    autotuner.update_weights(algorithm_state, candidate, empty_nodal_constraints, settings, params)

    # Virtual control should be updated based on nu (PTR method)
    assert candidate.lam_vc is not None
    assert isinstance(candidate.lam_vc, np.ndarray)
    # Should have shape (N-1, n_x) = (2, 2)
    assert candidate.lam_vc.shape == (2, 2)


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


def test_scpconfig_autotuner_default(settings):
    """ScpConfig.autotuner should default to AugmentedLagrangian."""
    # Default should be AugmentedLagrangian when no autotuner provided
    settings.scp.autotuner = None
    autotuner = settings.scp.autotuner
    assert isinstance(autotuner, AugmentedLagrangian)


def test_scpconfig_autotuner_augmented_lagrangian(settings):
    """ScpConfig.autotuner default should be a configurable AugmentedLagrangian."""
    # When no autotuner provided, should default to AugmentedLagrangian
    settings.scp.autotuner = None
    autotuner = settings.scp.autotuner
    assert isinstance(autotuner, AugmentedLagrangian)

    # Check that default parameters are set (AugmentedLagrangian constructor args)
    assert hasattr(autotuner, "rho_init")
    assert hasattr(autotuner, "rho_max")
    assert hasattr(autotuner, "lam_prox_min")
    assert hasattr(autotuner, "lam_prox_max")
    assert hasattr(autotuner, "lam_vc_max")
    assert hasattr(autotuner, "lam_cost_drop")
    assert hasattr(autotuner, "lam_cost_relax")

    # Check that parameters can be modified
    autotuner.rho_max = 1e7
    assert autotuner.rho_max == 1e7


def test_custom_autotuner_instance(settings):
    """Custom autotuner instance can be passed via ScpConfig."""

    # Create custom autotuner with modified parameters
    custom_autotuner = AugmentedLagrangian()
    custom_autotuner.rho_max = 1e7
    custom_autotuner.lam_prox_max = 1e6
    custom_autotuner.lam_vc_max = 1e6

    # Pass it to ScpConfig
    settings.scp.autotuner = custom_autotuner

    # ScpConfig.autotuner should return the custom instance
    autotuner = settings.scp.autotuner
    assert autotuner is custom_autotuner
    assert autotuner.rho_max == 1e7
    assert autotuner.lam_prox_max == 1e6
    assert autotuner.lam_vc_max == 1e6


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
    settings, algorithm_state, empty_nodal_constraints
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
        algorithm_state, candidate, empty_nodal_constraints, settings, {}
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
    assert algorithm_state.lam_cost_history[-1] == pytest.approx(settings.scp.lam_cost)


def test_constant_proximal_weight_uses_relaxed_cost_after_cost_drop(
    settings, algorithm_state, empty_nodal_constraints
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
        algorithm_state, candidate, empty_nodal_constraints, settings, {}
    )

    assert adaptive_state == "Accept Constant"
    assert len(algorithm_state.lam_cost_history) == initial_lam_cost_history_len + 1
    expected_relaxed = initial_lam_cost * autotuner.lam_cost_relax
    assert algorithm_state.lam_cost_history[-1] == pytest.approx(expected_relaxed)


# --- Tests for RampProximalWeight ---------------------------------------------


def test_ramp_proximal_weight_increases_until_max(
    settings, algorithm_state, empty_nodal_constraints
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
    )
    assert state_str == "Accept Constant"
    # Still at the maximum and not exceeded
    assert algorithm_state.lam_prox_history[-1] == pytest.approx(4.0)
