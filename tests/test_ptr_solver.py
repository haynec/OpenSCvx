import numpy as np

from openscvx.solvers.cvxpy_ptr_solver import _ctcs_impulse_effect


def test_ctcs_impulse_effect_scattered_only_to_ctcs_rows():
    n_states = 4
    ctcs_slice = slice(2, 4)
    slice_imp = slice(1, 2)

    B_d = np.zeros((n_states, 3))
    C_d = np.zeros((n_states, 3))

    # These physical-row impulse sensitivities must be ignored by this correction.
    B_d[0, 1] = 99.0
    C_d[1, 1] = -88.0

    # Only CTCS rows are scattered into the full state vector.
    B_d[2, 1] = 2.0
    B_d[3, 1] = -3.0
    C_d[2, 1] = 5.0
    C_d[3, 1] = 7.0

    du_prev = np.array([10.0, 0.25, 11.0])
    du_next = np.array([12.0, -0.5, 13.0])

    effect = _ctcs_impulse_effect(
        n_states,
        ctcs_slice,
        slice_imp,
        B_d,
        C_d,
        du_prev,
        du_next,
    )

    np.testing.assert_allclose(effect, np.array([0.0, 0.0, -2.0, -4.25]))


def test_ctcs_impulse_effect_is_zero_without_ctcs_or_impulses():
    B_d = np.ones((3, 2))
    C_d = np.ones((3, 2))
    du_prev = np.ones(2)
    du_next = np.ones(2)

    assert _ctcs_impulse_effect(3, None, slice(1, 2), B_d, C_d, du_prev, du_next) == 0
    assert _ctcs_impulse_effect(3, slice(2, 2), slice(1, 2), B_d, C_d, du_prev, du_next) == 0
    assert _ctcs_impulse_effect(3, slice(2, 3), slice(1, 1), B_d, C_d, du_prev, du_next) == 0
