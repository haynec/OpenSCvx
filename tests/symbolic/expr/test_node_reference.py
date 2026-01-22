"""Tests for NodeReference - inter-node constraint functionality.

This module tests the NodeReference expression class which enables users to define
constraints across different trajectory nodes, such as:
- Rate limits: (position.at(k) - position.at(k-1)) <= threshold
- Multi-step dependencies: state.at(k) == 2*state.at(k-1) - state.at(k-2)
"""

import pytest

from openscvx.symbolic.expr import (
    Control,
    Equality,
    Inequality,
    NodeReference,
    State,
)

# =============================================================================
# Core Functionality Tests
# =============================================================================


def test_node_reference_creation():
    """Test creating NodeReference from State and Control variables."""
    position = State("pos", shape=(3,))
    thrust = Control("thrust", shape=(2,))

    # Create node references using .at() method
    pos_ref = position.at(5)
    thrust_ref = thrust.at(10)

    assert isinstance(pos_ref, NodeReference)
    assert pos_ref.base is position
    assert pos_ref.node_idx == 5

    assert isinstance(thrust_ref, NodeReference)
    assert thrust_ref.base is thrust
    assert thrust_ref.node_idx == 10


def test_node_reference_type_validation():
    """Test that node index must be an integer."""
    position = State("pos", shape=(3,))

    # Valid: positive integers
    ref_positive = position.at(5)
    assert ref_positive.node_idx == 5

    # Valid: negative integers (for end-indexing)
    ref_negative = position.at(-1)
    assert ref_negative.node_idx == -1

    # Invalid: non-integer types
    with pytest.raises(TypeError, match="Node index must be an integer"):
        position.at(1.5)

    with pytest.raises(TypeError, match="Node index must be an integer"):
        position.at("k")

    with pytest.raises(TypeError, match="Node index must be an integer"):
        position.at([1, 2])


def test_node_reference_shape_preservation():
    """Test that NodeReference preserves the shape of its base expression."""
    # Vector state
    vector_state = State("pos", shape=(3,))
    assert vector_state.at(5).check_shape() == (3,)

    # Scalar state
    scalar_state = State("x", shape=(1,))
    assert scalar_state.at(0).check_shape() == (1,)

    # NodeReference on spatially-indexed expression
    position = State("pos", shape=(3,))
    x_component = position[0]  # Shape () - scalar
    x_at_k = x_component.at(5)
    assert x_at_k.check_shape() == ()


def test_node_reference_tree_structure():
    """Test that NodeReference correctly reports its base as a child."""
    state = State("x", shape=(2,))
    ref = state.at(3)

    children = ref.children()
    assert len(children) == 1
    assert children[0] is state


# =============================================================================
# NodeReference in Expressions
# =============================================================================


def test_node_reference_in_arithmetic():
    """Test that NodeReference works in arithmetic expressions."""
    velocity = State("vel", shape=(3,))
    state = State("x", shape=(1,))

    # Simple subtraction (rate limit pattern)
    vel_k = velocity.at(10)
    vel_k_minus_1 = velocity.at(9)
    delta_v = vel_k - vel_k_minus_1

    from openscvx.symbolic.expr import Sub

    assert isinstance(delta_v, Sub)
    assert delta_v.check_shape() == (3,)
    assert isinstance(delta_v.left, NodeReference)
    assert isinstance(delta_v.right, NodeReference)

    # Multi-step expression (Fibonacci-like)
    x_k = state.at(10)
    x_k_minus_1 = state.at(9)
    x_k_minus_2 = state.at(8)
    recurrence = x_k - x_k_minus_1 - x_k_minus_2

    assert recurrence.check_shape() == (1,)


def test_node_reference_in_constraints():
    """Test creating constraints with NodeReference."""
    velocity = State("vel", shape=(3,))
    position = State("pos", shape=(2,))

    # Inequality constraint
    vel_k = velocity.at(10)
    vel_k_minus_1 = velocity.at(9)
    max_accel = 0.5
    inequality = (vel_k - vel_k_minus_1) <= max_accel

    assert isinstance(inequality, Inequality)
    assert inequality.check_shape() == (3,)  # vector constraint

    # Equality constraint
    pos_start = position.at(0)
    pos_end = position.at(100)
    equality = pos_start == pos_end

    assert isinstance(equality, Equality)
    assert equality.check_shape() == (2,)  # vector constraint

    # Note: Cross-node constraints should NOT use .at([...]) wrapper
    # The constraint is auto-detected as cross-node due to NodeReferences
    # Just use the bare constraint:
    assert isinstance(inequality, Inequality)


# =============================================================================
# Real-World Usage Patterns
# =============================================================================


def test_rate_limit_pattern():
    """Test realistic rate limiting constraint pattern.

    This is the most common cross-node constraint pattern:
    limit the change between consecutive nodes.
    """
    position = State("pos", shape=(3,))
    max_step = 0.1

    # Cross-node constraints reference specific trajectory nodes
    # Here we test one instance of the pattern
    pos_k = position.at(10)
    pos_k_prev = position.at(9)

    constraint = (pos_k - pos_k_prev) <= max_step
    assert constraint.check_shape() == (3,)  # vector constraint

    # Cross-node constraints are auto-detected - no .at([...]) wrapper needed
    # The constraint is a bare Constraint object
    assert isinstance(constraint, Inequality)


def test_multi_step_dependencies():
    """Test multi-step dependencies involving 3+ nodes.

    Examples: second-order differences, Fibonacci-like recurrences.
    """
    state = State("x", shape=(1,))

    # Second-order finite difference (acceleration)
    x_next = state.at(11)
    x_curr = state.at(10)
    x_prev = state.at(9)

    dt = 0.1
    accel = (x_next - 2 * x_curr + x_prev) / (dt**2)

    # Should be able to constrain the second derivative
    max_accel = 5.0
    constraint = accel <= max_accel

    assert constraint.check_shape() == (1,)  # shape (1,) from state
    assert isinstance(constraint, Inequality)


def test_spatial_and_temporal_indexing():
    """Test combining spatial indexing with node references.

    Useful for constraining individual components of vector states.
    """
    velocity = State("vel", shape=(3,))

    # Rate limit only on z-component (index 2)
    z_k = velocity[2].at(10)
    z_k_prev = velocity[2].at(9)

    max_z_rate = 0.05
    constraint = (z_k - z_k_prev) <= max_z_rate

    # z component is scalar after spatial indexing
    assert z_k.check_shape() == ()
    assert constraint.check_shape() == ()


def test_boundary_coupling():
    """Test coupling constraints between specific nodes.

    Useful for periodic boundary conditions or linking distant nodes.
    """
    state = State("x", shape=(2,))

    # Periodic boundary condition: state at end equals state at start
    x_start = state.at(0)
    x_end = state.at(100)

    periodicity_constraint = x_start == x_end

    assert isinstance(periodicity_constraint, Equality)
    assert periodicity_constraint.check_shape() == (2,)  # vector constraint

    # Cross-node constraints are auto-detected - no .at([...]) wrapper needed
    # Just use the bare constraint directly
    assert isinstance(periodicity_constraint, Equality)


def test_loop_pattern_for_trajectory_constraints():
    """Test the recommended pattern: using Python loops to create constraints.

    This demonstrates how users should apply cross-node constraints across
    a trajectory using standard Python iteration.
    """
    position = State("pos", shape=(3,))
    max_step = 0.1
    N = 50

    # Recommended pattern: use a Python loop to create cross-node constraints
    # No .at([...]) wrapper needed - constraints are auto-detected as cross-node
    constraints = []
    for k in range(1, N):
        rate_limit = position.at(k) - position.at(k - 1) <= max_step
        constraints.append(rate_limit)

    # Should have N-1 constraints
    assert len(constraints) == N - 1

    # Each should be a bare Constraint (Inequality)
    for constraint in constraints:
        assert isinstance(constraint, Inequality)

    # List comprehension also works
    constraints_v2 = [position.at(k) - position.at(k - 1) <= max_step for k in range(1, N)]

    assert len(constraints_v2) == N - 1
