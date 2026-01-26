"""Tests for scaling_min and scaling_max attributes on State, Control, and Time.

This module tests the scaling feature that allows custom scaling bounds
for numerical optimization, separate from the actual min/max bounds.
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import Control, State
from openscvx.symbolic.time import Time
from openscvx.symbolic.unified import unify_controls, unify_states

# =============================================================================
# State scaling tests
# =============================================================================


def test_state_scaling_min_max():
    """Test setting scaling_min and scaling_max on State."""
    state = State("pos", shape=(3,))
    state.min = [-10.0, -10.0, 0.0]
    state.max = [10.0, 10.0, 100.0]

    # Set scaling bounds
    state.scaling_min = [-5.0, -5.0, 10.0]
    state.scaling_max = [5.0, 5.0, 50.0]

    assert np.allclose(state.scaling_min, [-5.0, -5.0, 10.0])
    assert np.allclose(state.scaling_max, [5.0, 5.0, 50.0])
    # Original bounds should be unchanged
    assert np.allclose(state.min, [-10.0, -10.0, 0.0])
    assert np.allclose(state.max, [10.0, 10.0, 100.0])


def test_state_scaling_none():
    """Test that scaling_min/max can be set to None."""
    state = State("pos", shape=(2,))
    state.scaling_min = [1.0, 2.0]
    state.scaling_max = [3.0, 4.0]

    state.scaling_min = None
    state.scaling_max = None

    assert state.scaling_min is None
    assert state.scaling_max is None


def test_state_scaling_shape_validation():
    """Test that scaling_min/max validate shape."""
    state = State("pos", shape=(3,))

    # Should raise ValueError for wrong shape
    with pytest.raises(ValueError, match="does not match State shape"):
        state.scaling_min = [1.0, 2.0]  # Wrong length

    with pytest.raises(ValueError, match="does not match State shape"):
        state.scaling_max = [1.0]  # Wrong length

    # Should work with correct shape
    state.scaling_min = [1.0, 2.0, 3.0]
    state.scaling_max = [4.0, 5.0, 6.0]


# =============================================================================
# Control scaling tests
# =============================================================================


def test_control_scaling_min_max():
    """Test setting scaling_min and scaling_max on Control."""
    control = Control("thrust", shape=(2,))
    control.min = [-10.0, 0.0]
    control.max = [10.0, 20.0]

    # Set scaling bounds
    control.scaling_min = [-5.0, 5.0]
    control.scaling_max = [5.0, 15.0]

    assert np.allclose(control.scaling_min, [-5.0, 5.0])
    assert np.allclose(control.scaling_max, [5.0, 15.0])
    # Original bounds should be unchanged
    assert np.allclose(control.min, [-10.0, 0.0])
    assert np.allclose(control.max, [10.0, 20.0])


def test_control_scaling_none():
    """Test that scaling_min/max can be set to None."""
    control = Control("u", shape=(1,))
    control.scaling_min = [1.0]
    control.scaling_max = [2.0]

    control.scaling_min = None
    control.scaling_max = None

    assert control.scaling_min is None
    assert control.scaling_max is None


def test_control_scaling_shape_validation():
    """Test that scaling_min/max validate shape."""
    control = Control("u", shape=(2,))

    # Should raise ValueError for wrong shape
    with pytest.raises(ValueError, match="does not match Control shape"):
        control.scaling_min = [1.0]  # Wrong length

    with pytest.raises(ValueError, match="does not match Control shape"):
        control.scaling_max = [1.0, 2.0, 3.0]  # Wrong length


# =============================================================================
# Time scaling tests
# =============================================================================


def test_time_scaling_min_max():
    """Test setting scaling_min and scaling_max on Time."""
    time = Time(initial=0.0, final=10.0, min=0.0, max=20.0)

    # Set scaling bounds (Time is a State with shape=(1,), so use arrays)
    time.scaling_min = [1.0]
    time.scaling_max = [15.0]

    assert np.allclose(time.scaling_min, [1.0])
    assert np.allclose(time.scaling_max, [15.0])
    # Original bounds should be unchanged
    assert np.allclose(time.min, [0.0])
    assert np.allclose(time.max, [20.0])


def test_time_scaling_none():
    """Test that scaling_min/max can be set to None."""
    time = Time(initial=0.0, final=10.0, min=0.0, max=20.0)
    time.scaling_min = [1.0]
    time.scaling_max = [15.0]

    time.scaling_min = None
    time.scaling_max = None

    assert time.scaling_min is None
    assert time.scaling_max is None


# =============================================================================
# UnifiedState scaling aggregation tests
# =============================================================================


def test_unify_states_aggregates_scaling():
    """Test that UnifiedState aggregates scaling_min/max from individual states."""
    state1 = State("pos", shape=(2,))
    state1.min = [0.0, 0.0]
    state1.max = [10.0, 10.0]
    state1.scaling_min = [1.0, 2.0]
    state1.scaling_max = [9.0, 8.0]

    state2 = State("vel", shape=(2,))
    state2.min = [-5.0, -5.0]
    state2.max = [5.0, 5.0]
    state2.scaling_min = [-4.0, -3.0]
    state2.scaling_max = [4.0, 3.0]

    unified = unify_states([state1, state2])

    # Should concatenate scaling arrays
    expected_scaling_min = np.array([1.0, 2.0, -4.0, -3.0])
    expected_scaling_max = np.array([9.0, 8.0, 4.0, 3.0])
    np.testing.assert_array_equal(unified.scaling_min, expected_scaling_min)
    np.testing.assert_array_equal(unified.scaling_max, expected_scaling_max)


def test_unify_states_scaling_partial():
    """Test that UnifiedState handles partial scaling (some states have it, some don't)."""
    state1 = State("pos", shape=(2,))
    state1.min = [0.0, 0.0]
    state1.max = [10.0, 10.0]
    state1.scaling_min = [1.0, 2.0]
    state1.scaling_max = [9.0, 8.0]

    state2 = State("vel", shape=(2,))
    state2.min = [-5.0, -5.0]
    state2.max = [5.0, 5.0]
    # state2 has no scaling_min/max

    unified = unify_states([state1, state2])

    # Should use scaling for state1, min/max for state2
    expected_scaling_min = np.array([1.0, 2.0, -5.0, -5.0])
    expected_scaling_max = np.array([9.0, 8.0, 5.0, 5.0])
    np.testing.assert_array_equal(unified.scaling_min, expected_scaling_min)
    np.testing.assert_array_equal(unified.scaling_max, expected_scaling_max)


def test_unify_states_scaling_none():
    """Test that UnifiedState has None scaling when no states have it."""
    state1 = State("pos", shape=(2,))
    state1.min = [0.0, 0.0]
    state1.max = [10.0, 10.0]

    state2 = State("vel", shape=(2,))
    state2.min = [-5.0, -5.0]
    state2.max = [5.0, 5.0]

    unified = unify_states([state1, state2])

    assert unified.scaling_min is None
    assert unified.scaling_max is None


# =============================================================================
# UnifiedControl scaling aggregation tests
# =============================================================================


def test_unify_controls_aggregates_scaling():
    """Test that UnifiedControl aggregates scaling_min/max from individual controls."""
    control1 = Control("thrust", shape=(2,))
    control1.min = [-10.0, 0.0]
    control1.max = [10.0, 20.0]
    control1.scaling_min = [-5.0, 5.0]
    control1.scaling_max = [5.0, 15.0]

    control2 = Control("torque", shape=(1,))
    control2.min = [-1.0]
    control2.max = [1.0]
    control2.scaling_min = [-0.5]
    control2.scaling_max = [0.5]

    unified = unify_controls([control1, control2])

    # Should concatenate scaling arrays
    expected_scaling_min = np.array([-5.0, 5.0, -0.5])
    expected_scaling_max = np.array([5.0, 15.0, 0.5])
    np.testing.assert_array_equal(unified.scaling_min, expected_scaling_min)
    np.testing.assert_array_equal(unified.scaling_max, expected_scaling_max)


def test_unify_controls_scaling_partial():
    """Test that UnifiedControl handles partial scaling."""
    control1 = Control("thrust", shape=(2,))
    control1.min = [-10.0, 0.0]
    control1.max = [10.0, 20.0]
    control1.scaling_min = [-5.0, 5.0]
    control1.scaling_max = [5.0, 15.0]

    control2 = Control("torque", shape=(1,))
    control2.min = [-1.0]
    control2.max = [1.0]
    # control2 has no scaling_min/max

    unified = unify_controls([control1, control2])

    # Should use scaling for control1, min/max for control2
    expected_scaling_min = np.array([-5.0, 5.0, -1.0])
    expected_scaling_max = np.array([5.0, 15.0, 1.0])
    np.testing.assert_array_equal(unified.scaling_min, expected_scaling_min)
    np.testing.assert_array_equal(unified.scaling_max, expected_scaling_max)


def test_unify_controls_scaling_none():
    """Test that UnifiedControl has None scaling when no controls have it."""
    control1 = Control("thrust", shape=(2,))
    control1.min = [-10.0, 0.0]
    control1.max = [10.0, 20.0]

    control2 = Control("torque", shape=(1,))
    control2.min = [-1.0]
    control2.max = [1.0]

    unified = unify_controls([control1, control2])

    assert unified.scaling_min is None
    assert unified.scaling_max is None


# =============================================================================
# SimConfig integration tests
# =============================================================================


def test_simconfig_uses_scaling_when_provided():
    """Test that SimConfig uses scaling_min/max when provided, otherwise falls back to min/max."""
    from openscvx.config import SimConfig

    # Create unified state with scaling
    state = State("x", shape=(2,))
    state.min = [0.0, 0.0]
    state.max = [10.0, 10.0]
    state.scaling_min = [2.0, 3.0]
    state.scaling_max = [8.0, 7.0]

    unified_state = unify_states([state])
    unified_control = unify_controls([Control("u", shape=(1,))])

    # Create SimConfig
    sim_config = SimConfig(
        x=unified_state,
        x_prop=unified_state,
        u=unified_control,
        total_time=1.0,
    )

    # Check that scaling matrices use scaling_min/max
    # S_x and c_x should be computed from scaling bounds, not regular bounds
    # The scaling matrix S is computed as diag(max(abs(min - max) / 2, 1))
    # So for scaling_min=[2,3], scaling_max=[8,7]:
    # S should be diag([max(1, abs(2-8)/2), max(1, abs(3-7)/2)]) = diag([3, 2])
    # c should be ([2+8]/2, [3+7]/2) = (5, 5)

    # Verify the scaling center c_x uses scaling bounds
    expected_c = np.array([5.0, 5.0])  # (scaling_min + scaling_max) / 2
    np.testing.assert_array_almost_equal(sim_config.c_x, expected_c)

    # Verify scaling matrix uses scaling bounds
    # S = diag(max(1, abs(scaling_min - scaling_max) / 2))
    expected_S_diag = np.array([3.0, 2.0])  # max(1, abs(2-8)/2), max(1, abs(3-7)/2)
    np.testing.assert_array_almost_equal(np.diag(sim_config.S_x), expected_S_diag)


def test_simconfig_falls_back_to_min_max():
    """Test that SimConfig falls back to min/max when scaling is not provided."""
    from openscvx.config import SimConfig

    # Create unified state without scaling
    state = State("x", shape=(2,))
    state.min = [0.0, 0.0]
    state.max = [10.0, 10.0]
    # No scaling_min/max

    unified_state = unify_states([state])
    unified_control = unify_controls([Control("u", shape=(1,))])

    # Create SimConfig
    sim_config = SimConfig(
        x=unified_state,
        x_prop=unified_state,
        u=unified_control,
        total_time=1.0,
    )

    # Should use regular min/max for scaling
    expected_c = np.array([5.0, 5.0])  # (min + max) / 2
    np.testing.assert_array_almost_equal(sim_config.c_x, expected_c)

    expected_S_diag = np.array([5.0, 5.0])  # max(1, abs(0-10)/2)
    np.testing.assert_array_almost_equal(np.diag(sim_config.S_x), expected_S_diag)


def test_simconfig_partial_scaling():
    """Test SimConfig uses scaling where available, min/max elsewhere."""
    from openscvx.config import SimConfig

    # Create states - one with scaling, one without
    state1 = State("pos", shape=(2,))
    state1.min = [0.0, 0.0]
    state1.max = [10.0, 10.0]
    state1.scaling_min = [2.0, 3.0]
    state1.scaling_max = [8.0, 7.0]

    state2 = State("vel", shape=(1,))
    state2.min = [-5.0]
    state2.max = [5.0]
    # No scaling

    unified_state = unify_states([state1, state2])
    unified_control = unify_controls([Control("u", shape=(1,))])

    # unified_state.scaling_min/max should have full size: scaling for state1, min/max for state2
    sim_config = SimConfig(
        x=unified_state,
        x_prop=unified_state,
        u=unified_control,
        total_time=1.0,
    )

    # Should use scaling for state1, min/max for state2
    # c_x[0:2] from scaling: (2+8)/2=5, (3+7)/2=5
    # c_x[2] from regular: (-5+5)/2=0
    expected_c = np.array([5.0, 5.0, 0.0])
    np.testing.assert_array_almost_equal(sim_config.c_x, expected_c)

    # S_x[0:2] from scaling: max(1, abs(2-8)/2)=3, max(1, abs(3-7)/2)=2
    # S_x[2] from regular: max(1, abs(-5-5)/2)=5
    expected_S_diag = np.array([3.0, 2.0, 5.0])
    np.testing.assert_array_almost_equal(np.diag(sim_config.S_x), expected_S_diag)
