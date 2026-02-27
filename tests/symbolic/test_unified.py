import numpy as np

from openscvx.symbolic.expr import Control, State
from openscvx.symbolic.preprocessing import collect_and_assign_slices
from openscvx.symbolic.unified import unify_controls, unify_states


# Test unify_states function
def test_unify_states_sorting():
    """Test that true states come before augmented states."""
    true_state1 = State("pos", (2,))
    true_state1.min = np.array([0.0, 1.0])

    aug_state = State("_slack", (1,))
    aug_state.min = np.array([5.0])

    true_state2 = State("vel", (2,))
    true_state2.min = np.array([2.0, 3.0])

    # Pass in mixed order - augmented state in the middle
    unified = unify_states([true_state1, aug_state, true_state2])

    # Should have total shape 5, true dim 4
    assert unified.shape == (5,)
    assert unified._true_dim == 4

    # Check that true states come first, then augmented
    expected_min = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    np.testing.assert_array_equal(unified.min, expected_min)


def test_unify_states_none_handling():
    """Test that None values are handled properly with defaults."""
    state1 = State("x", (2,))
    # Don't set min/max - they should default

    state2 = State("_aug", (1,))
    state2.min = np.array([1.0])

    unified = unify_states([state1, state2])

    # Should have -inf for state1's min values, 1.0 for state2
    expected_min = np.array([-np.inf, -np.inf, 1.0])
    np.testing.assert_array_equal(unified.min, expected_min)

    # Should have +inf for state1's max values
    expected_max = np.array([np.inf, np.inf, np.inf])
    np.testing.assert_array_equal(unified.max, expected_max)


def test_unify_states_underscore_naming():
    """Test various underscore naming patterns."""
    normal = State("position", (1,))
    normal.min = np.array([0.0])

    underscore_start = State("_slack_var", (1,))
    underscore_start.min = np.array([1.0])

    underscore_middle = State("some_var", (1,))  # Should be treated as normal
    underscore_middle.min = np.array([2.0])

    double_underscore = State("__private", (1,))  # Should be treated as augmented
    double_underscore.min = np.array([3.0])

    unified = unify_states([underscore_start, normal, double_underscore, underscore_middle])

    # True states: normal, underscore_middle (2 total)
    assert unified._true_dim == 2

    # Order should be: normal, underscore_middle, underscore_start, double_underscore
    expected_min = np.array([0.0, 2.0, 1.0, 3.0])
    np.testing.assert_array_equal(unified.min, expected_min)


# Test unify_controls function
def test_unify_controls_sorting():
    """Test that true controls come before augmented controls."""
    true_control1 = Control("thrust", (1,))
    true_control1.min = np.array([-10.0])

    aug_control = Control("_auxiliary", (1,))
    aug_control.min = np.array([0.0])

    true_control2 = Control("torque", (1,))
    true_control2.min = np.array([-5.0])

    # Pass in mixed order
    unified = unify_controls([aug_control, true_control1, true_control2])

    assert unified.shape == (3,)
    assert unified._true_dim == 2

    # Should be: true_control1, true_control2, aug_control
    expected_min = np.array([-10.0, -5.0, 0.0])
    np.testing.assert_array_equal(unified.min, expected_min)


# Test UnifiedState properties and methods
def test_unified_state_properties():
    """Test true and augmented properties of UnifiedState."""
    true_state = State("x", (2,))
    true_state.min = np.array([0.0, 1.0])

    aug_state = State("_slack", (1,))
    aug_state.min = np.array([5.0])

    unified = unify_states([true_state, aug_state])

    # Test true property
    true_part = unified.true
    assert true_part.shape == (2,)
    np.testing.assert_array_equal(true_part.min, np.array([0.0, 1.0]))

    # Test augmented property
    aug_part = unified.augmented
    assert aug_part.shape == (1,)
    np.testing.assert_array_equal(aug_part.min, np.array([5.0]))


def test_unified_state_append():
    """Test appending to UnifiedState."""
    state1 = State("x", (2,))
    state1.min = np.array([0.0, 1.0])

    unified = unify_states([state1])

    # Append as augmented state
    state2 = State("_aug", (1,))
    state2.min = np.array([5.0])
    unified.append(state2, augmented=True)

    assert unified.shape == (3,)
    assert unified._true_dim == 2  # Should not change
    np.testing.assert_array_equal(unified.min, np.array([0.0, 1.0, 5.0]))

    # Append scalar variable
    unified.append(min=-1.0, max=1.0, augmented=False)
    assert unified.shape == (4,)
    assert unified._true_dim == 3


def test_unified_state_slicing():
    """Test slicing UnifiedState."""
    state1 = State("x", (2,))
    state1.min = np.array([0.0, 1.0])

    state2 = State("_aug", (2,))
    state2.min = np.array([5.0, 6.0])

    unified = unify_states([state1, state2])

    # Get first 3 elements
    subset = unified[0:3]
    assert subset.shape == (3,)
    assert subset._true_dim == 2
    np.testing.assert_array_equal(subset.min, np.array([0.0, 1.0, 5.0]))


# Test UnifiedControl properties and methods
def test_unified_control_properties():
    """Test true and augmented properties of UnifiedControl."""
    true_control = Control("u", (1,))
    true_control.min = np.array([-1.0])

    aug_control = Control("_aux", (1,))
    aug_control.min = np.array([5.0])

    unified = unify_controls([true_control, aug_control])

    # Test true property
    true_part = unified.true
    assert true_part.shape == (1,)
    np.testing.assert_array_equal(true_part.min, np.array([-1.0]))

    # Test augmented property
    aug_part = unified.augmented
    assert aug_part.shape == (1,)
    np.testing.assert_array_equal(aug_part.min, np.array([5.0]))


# Test integration with arrays
def test_state_with_guess_arrays():
    """Test that guess arrays and initial/final conditions are handled properly."""
    state1 = State("x", (2,))
    state1.initial = np.array([1.0, 2.0])
    guess1 = np.random.rand(100, 2)
    state1.guess = guess1

    state2 = State("_aug", (1,))
    state2.initial = np.array([5.0])
    guess2 = np.random.rand(100, 1)
    state2.guess = guess2

    unified = unify_states([state1, state2])

    # Check guess arrays
    assert unified.guess.shape == (100, 3)
    np.testing.assert_array_equal(unified.guess[:, :2], guess1)
    np.testing.assert_array_equal(unified.guess[:, 2:], guess2)

    # Check initial conditions ordering: true states first, then augmented
    expected_initial = np.array([1.0, 2.0, 5.0])
    np.testing.assert_array_equal(unified.initial, expected_initial)


# Test metadata slice capabilities
def test_unified_state_time_slice():
    """Test that time_slice is correctly identified in unified states."""
    pos = State("pos", (3,))
    pos.min = np.array([0.0, 0.0, 0.0])

    time = State("time", (1,))
    time.min = np.array([0.0])

    vel = State("vel", (3,))
    vel.min = np.array([-10.0, -10.0, -10.0])

    # Assign slices first
    states, _ = collect_and_assign_slices([pos, time, vel], [])

    # Pass in mixed order
    unified = unify_states(states)

    # time_slice should point to the time state
    assert unified.time_slice is not None
    assert unified.time_slice.start == 3  # After pos (3 dims)
    assert unified.time_slice.stop == 4
    assert unified.min[unified.time_slice][0] == 0.0


def test_unified_state_time_slice_none():
    """Test that time_slice is None when no time state exists."""
    pos = State("pos", (3,))
    vel = State("vel", (3,))

    # Assign slices first
    states, _ = collect_and_assign_slices([pos, vel], [])
    unified = unify_states(states)

    assert unified.time_slice is None


def test_unified_state_ctcs_slice():
    """Test that ctcs_slice is correctly identified for CTCS augmented states."""
    pos = State("pos", (2,))
    pos.min = np.array([0.0, 0.0])

    ctcs1 = State("_ctcs_aug_0", (1,))
    ctcs1.min = np.array([0.0])

    ctcs2 = State("_ctcs_aug_1", (1,))
    ctcs2.min = np.array([0.0])

    slack = State("_slack", (1,))
    slack.min = np.array([0.0])

    # Assign slices first
    states, _ = collect_and_assign_slices([pos, ctcs1, ctcs2, slack], [])

    # CTCS states should be grouped together
    unified = unify_states(states)

    # ctcs_slice should span both CTCS augmented states
    assert unified.ctcs_slice is not None
    assert unified.ctcs_slice.start == 2  # After pos (2 dims)
    assert unified.ctcs_slice.stop == 4  # Covers both ctcs states
    assert unified.min[unified.ctcs_slice][0] == 0.0
    assert unified.min[unified.ctcs_slice][1] == 0.0


def test_unified_state_ctcs_slice_none():
    """Test that ctcs_slice is None when no CTCS states exist."""
    pos = State("pos", (2,))
    slack = State("_slack", (1,))

    # Assign slices first
    states, _ = collect_and_assign_slices([pos, slack], [])
    unified = unify_states(states)

    assert unified.ctcs_slice is None


def test_unified_state_multiple_metadata_slices():
    """Test that both time and CTCS slices coexist properly."""
    pos = State("pos", (2,))
    pos.min = np.array([0.0, 0.0])

    time = State("time", (1,))
    time.min = np.array([0.0])

    ctcs1 = State("_ctcs_aug_0", (1,))
    ctcs1.min = np.array([0.0])

    ctcs2 = State("_ctcs_aug_1", (1,))
    ctcs2.min = np.array([0.0])

    # Assign slices first
    states, _ = collect_and_assign_slices([pos, time, ctcs1, ctcs2], [])
    unified = unify_states(states)

    # Both slices should be present and non-overlapping
    assert unified.time_slice is not None
    assert unified.ctcs_slice is not None

    # time comes before ctcs in true states
    assert unified.time_slice.start == 2  # After pos
    assert unified.time_slice.stop == 3

    # ctcs are augmented states
    assert unified.ctcs_slice.start == 3  # After true states
    assert unified.ctcs_slice.stop == 5  # Both ctcs states


def test_unified_control_time_dilation_slice():
    """Test that time_dilation_slice is correctly identified in unified controls."""
    thrust = Control("thrust", (2,))
    thrust.min = np.array([0.0, 0.0])

    time_dilation = Control("_time_dilation", (1,))
    time_dilation.min = np.array([0.5])
    time_dilation.max = np.array([2.0])

    torque = Control("torque", (1,))
    torque.min = np.array([-5.0])

    # Assign slices in the canonical sorted order (true first, then augmented)
    # This is the order that unify_controls will use
    _, controls = collect_and_assign_slices([], [thrust, torque, time_dilation])

    unified = unify_controls(controls)

    # time_dilation_slice should point to the time dilation control
    assert unified.time_dilation_slice is not None
    # time_dilation is augmented, so comes after true controls
    assert unified.time_dilation_slice.start == 3  # After thrust (2) + torque (1)
    assert unified.time_dilation_slice.stop == 4
    assert unified.min[unified.time_dilation_slice][0] == 0.5
    assert unified.max[unified.time_dilation_slice][0] == 2.0


def test_unified_control_time_dilation_slice_none():
    """Test that time_dilation_slice is None when no time dilation control exists."""
    thrust = Control("thrust", (2,))
    torque = Control("torque", (1,))

    # Assign slices first
    _, controls = collect_and_assign_slices([], [thrust, torque])
    unified = unify_controls(controls)

    assert unified.time_dilation_slice is None


def test_unified_control_impulsive_order_and_true_selector():
    """Continuous controls must precede impulsive controls in unified ordering."""
    thrust = Control("thrust", (1,))
    thrust.min = np.array([0.0])

    time_dilation = Control("_time_dilation", (1,))
    time_dilation.min = np.array([0.5])
    time_dilation.max = np.array([2.0])

    impulse = Control(
        "impulse",
        (1,),
        impulsive=True,
        nodes=[0],
    )
    impulse.min = np.array([-10.0])

    # Pass controls in mixed order and verify canonical reordering.
    _, controls = collect_and_assign_slices([], [impulse, time_dilation, thrust])
    unified = unify_controls(controls)

    # New order should be: thrust (continuous true), time_dilation (continuous augmented), impulse.
    np.testing.assert_array_equal(unified.min, np.array([0.0, 0.5, -10.0]))
    assert unified.slice_continuous == slice(0, 2)
    assert unified.slice_impulsive == slice(2, 3)

    # True controls are non-contiguous in this layout: [0] U [2].
    assert isinstance(unified._true_slice, tuple)
    assert unified._true_slice == (slice(0, 1), slice(2, 3))
    assert unified._true_dim == 2
    np.testing.assert_array_equal(unified.true.min, np.array([0.0, -10.0]))


def test_metadata_slices_indexing():
    """Test that metadata slices can be used to index into arrays correctly."""
    pos = State("pos", (2,))
    pos.initial = np.array([1.0, 2.0])

    time = State("time", (1,))
    time.initial = np.array([5.0])

    ctcs = State("_ctcs_aug_0", (1,))
    ctcs.initial = np.array([0.0])

    # Assign slices first
    states, _ = collect_and_assign_slices([pos, time, ctcs], [])
    unified = unify_states(states)

    # Use time_slice to extract time value
    time_value = unified.initial[unified.time_slice]
    assert time_value.shape == (1,)
    assert time_value[0] == 5.0

    # Use ctcs_slice to extract CTCS values
    ctcs_value = unified.initial[unified.ctcs_slice]
    assert ctcs_value.shape == (1,)
    assert ctcs_value[0] == 0.0

    # Create a trajectory array
    traj = np.ones((10, unified.shape[0]))
    traj[:, unified.time_slice] = np.linspace(0, 1, 10).reshape(-1, 1)
    traj[:, unified.ctcs_slice] = np.zeros((10, 1))

    # Verify slicing works on 2D arrays
    time_traj = traj[:, unified.time_slice]
    assert time_traj.shape == (10, 1)
    np.testing.assert_array_almost_equal(time_traj.flatten(), np.linspace(0, 1, 10))

    ctcs_traj = traj[:, unified.ctcs_slice]
    assert ctcs_traj.shape == (10, 1)
    np.testing.assert_array_equal(ctcs_traj, np.zeros((10, 1)))
