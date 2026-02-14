"""Tests for static Jacobian sparsity analysis.

Covers the two public functions in openscvx.symbolic.sparsity:
- jacobian_sparsity  (2-D Jacobian sparsity for any expression)
- cross_node_sparsity (per-node masks for cross-node constraints)
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Concat,
    Constant,
    Control,
    Norm,
    Parameter,
    State,
)
from openscvx.symbolic.preprocessing import collect_and_assign_slices
from openscvx.symbolic.sparsity import (
    cross_node_sparsity,
    jacobian_sparsity,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_rocket_vars():
    """Standard 3-DOF rocket: pos(3), vel(3), mass(1), thrust(3)."""
    pos = State("pos", (3,))
    vel = State("vel", (3,))
    mass = State("mass", (1,))
    thrust = Control("thrust", (3,))

    states = [pos, vel, mass]
    controls = [thrust]
    collect_and_assign_slices(states, controls)

    n_x = sum(s.shape[0] for s in states)   # 7
    n_u = sum(c.shape[0] for c in controls)  # 3
    return pos, vel, mass, thrust, states, controls, n_x, n_u


# =============================================================================
# jacobian_sparsity — non-Concat expressions (column-level broadcast)
# =============================================================================


def test_single_state_leaf():
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()
    df_dx, df_du = jacobian_sparsity(vel, n_x, n_u)

    # vel has shape (3,) => 3 output rows
    assert df_dx.shape == (3, n_x)
    assert df_du.shape == (3, n_u)
    # vel occupies indices 3:6 — every row should have the same mask
    expected_row = np.array([False, False, False, True, True, True, False])
    for i in range(3):
        np.testing.assert_array_equal(df_dx[i], expected_row)
    np.testing.assert_array_equal(df_du, False)


def test_single_control_leaf():
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()
    df_dx, df_du = jacobian_sparsity(thrust, n_x, n_u)

    assert df_dx.shape == (3, n_x)
    np.testing.assert_array_equal(df_dx, False)
    np.testing.assert_array_equal(df_du, True)


def test_constant_has_no_dependence():
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()
    c = Constant(np.array([9.81]))
    df_dx, df_du = jacobian_sparsity(c, n_x, n_u)

    assert df_dx.shape == (1, n_x)
    assert not df_dx.any()
    assert not df_du.any()


def test_parameter_has_no_dependence():
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()
    g = Parameter("g", (3,), value=np.array([0.0, 0.0, -9.81]))
    df_dx, df_du = jacobian_sparsity(g, n_x, n_u)

    assert df_dx.shape == (3, n_x)
    assert not df_dx.any()
    assert not df_du.any()


def test_mixed_state_and_control():
    """thrust / mass depends on both u[0:3] and x[6:7]."""
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()
    expr = thrust / mass
    df_dx, df_du = jacobian_sparsity(expr, n_x, n_u)

    assert df_dx.shape == (3, n_x)
    # x dependence: only mass (index 6) — same mask on every row
    expected_row = np.array([False, False, False, False, False, False, True])
    for i in range(3):
        np.testing.assert_array_equal(df_dx[i], expected_row)
    # u dependence: all of thrust
    np.testing.assert_array_equal(df_du, True)


def test_scalar_expression():
    """Norm reduces to scalar — output should be (1, n_x) and (1, n_u)."""
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()
    expr = Norm(thrust)
    df_dx, df_du = jacobian_sparsity(expr, n_x, n_u)

    assert df_dx.shape == (1, n_x)
    assert df_du.shape == (1, n_u)
    assert not df_dx.any()
    np.testing.assert_array_equal(df_du, True)


# =============================================================================
# jacobian_sparsity — Concat decomposition (row-block analysis)
# =============================================================================


def test_rocket_dynamics_sparsity():
    """Classic rocket: pos_dot=vel, vel_dot=thrust/mass - g, mass_dot=-alpha*||thrust||."""
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()

    g = Constant(np.array([0.0, 0.0, -9.81]))
    alpha = Constant(np.array([0.01]))

    dynamics_dict = {
        "pos": vel,
        "vel": thrust / mass - g,
        "mass": -alpha * Norm(thrust),
    }
    dynamics_exprs = [dynamics_dict[s.name] for s in states]
    dynamics_concat = Concat(*dynamics_exprs)

    A, B = jacobian_sparsity(dynamics_concat, n_x, n_u)

    assert A.shape == (n_x, n_x)
    assert B.shape == (n_x, n_u)

    # --- df/dx (A pattern) ---
    # pos_dot = vel  =>  row 0:3 depends on x[3:6] only
    np.testing.assert_array_equal(A[0:3, 0:3], False)  # no pos dep
    np.testing.assert_array_equal(A[0:3, 3:6], True)   # vel dep
    np.testing.assert_array_equal(A[0:3, 6:7], False)  # no mass dep

    # vel_dot = thrust/mass - g  =>  row 3:6 depends on x[6:7] (mass) only
    np.testing.assert_array_equal(A[3:6, 0:3], False)
    np.testing.assert_array_equal(A[3:6, 3:6], False)
    np.testing.assert_array_equal(A[3:6, 6:7], True)

    # mass_dot = -alpha * ||thrust||  =>  row 6:7 has no state dep
    np.testing.assert_array_equal(A[6:7, :], False)

    # --- df/du (B pattern) ---
    # pos_dot = vel  =>  no control dep
    np.testing.assert_array_equal(B[0:3, :], False)

    # vel_dot = thrust/mass - g  =>  depends on thrust
    np.testing.assert_array_equal(B[3:6, :], True)

    # mass_dot = -alpha * ||thrust||  =>  depends on thrust
    np.testing.assert_array_equal(B[6:7, :], True)


def test_decoupled_dynamics():
    """Two states with fully independent dynamics produce a block-diagonal A."""
    x = State("x", (2,))
    y = State("y", (2,))
    u = Control("u", (1,))
    states = [x, y]
    controls = [u]
    collect_and_assign_slices(states, controls)

    # x_dot = x, y_dot = y + u
    dynamics_concat = Concat(x, y + u)
    A, B = jacobian_sparsity(dynamics_concat, 4, 1)

    # x_dot depends only on x (cols 0:2)
    np.testing.assert_array_equal(A[0:2, 0:2], True)
    np.testing.assert_array_equal(A[0:2, 2:4], False)

    # y_dot depends only on y (cols 2:4)
    np.testing.assert_array_equal(A[2:4, 0:2], False)
    np.testing.assert_array_equal(A[2:4, 2:4], True)

    # Only y_dot depends on u
    np.testing.assert_array_equal(B[0:2, :], False)
    np.testing.assert_array_equal(B[2:4, :], True)


def test_constant_dynamics_row():
    """A state with constant dynamics (e.g. mass_dot = 0) has all-False rows."""
    x = State("x", (2,))
    states = [x]
    controls = []
    collect_and_assign_slices(states, controls)

    dynamics_concat = Concat(Constant(np.zeros(2)))
    A, B = jacobian_sparsity(dynamics_concat, 2, 0)

    np.testing.assert_array_equal(A, False)
    assert B.shape == (2, 0)


def test_nested_concat():
    """Concat inside Concat is decomposed recursively."""
    x = State("x", (1,))
    y = State("y", (1,))
    z = State("z", (1,))
    collect_and_assign_slices([x, y, z], [])

    inner = Concat(x, y)   # rows 0-1
    outer = Concat(inner, z)  # row 2
    A, _ = jacobian_sparsity(outer, 3, 0)

    # Row 0 (x_dot=x) depends on col 0 only
    np.testing.assert_array_equal(A[0], [True, False, False])
    # Row 1 (y_dot=y) depends on col 1 only
    np.testing.assert_array_equal(A[1], [False, True, False])
    # Row 2 (z_dot=z) depends on col 2 only
    np.testing.assert_array_equal(A[2], [False, False, True])


# =============================================================================
# cross_node_sparsity — NodeReference-based analysis
# =============================================================================


def test_rate_constraint_sparsity():
    """pos.at(k) - pos.at(k-1) should mark exactly nodes k and k-1 for pos."""
    pos = State("pos", (3,))
    vel = State("vel", (3,))
    states = [pos, vel]
    controls = []
    collect_and_assign_slices(states, controls)
    n_x = 6
    N = 10

    expr = pos.at(5) - pos.at(4)
    x_mask, u_mask = cross_node_sparsity(expr, n_x, 0, N)

    assert x_mask.shape == (N, n_x)
    assert u_mask.shape == (N, 0)

    # Only nodes 4 and 5 should be marked, and only for pos (cols 0:3)
    for k in range(N):
        if k in (4, 5):
            np.testing.assert_array_equal(x_mask[k, 0:3], True)
        else:
            np.testing.assert_array_equal(x_mask[k, :], False)
    # vel columns always False
    np.testing.assert_array_equal(x_mask[:, 3:6], False)


def test_cross_node_negative_index():
    """Negative node indices are normalized correctly."""
    x = State("x", (2,))
    states = [x]
    controls = []
    collect_and_assign_slices(states, controls)
    N = 5

    # x.at(-1) references node N-1 = 4
    expr = x.at(0) - x.at(-1)
    x_mask, u_mask = cross_node_sparsity(expr, 2, 0, N)

    np.testing.assert_array_equal(x_mask[0, :], True)
    np.testing.assert_array_equal(x_mask[4, :], True)
    # nodes 1-3 untouched
    np.testing.assert_array_equal(x_mask[1:4, :], False)


def test_cross_node_with_control():
    """Cross-node constraint referencing both state and control."""
    x = State("x", (2,))
    u = Control("u", (1,))
    states = [x]
    controls = [u]
    collect_and_assign_slices(states, controls)
    N = 5

    expr = x.at(3) + u.at(3)
    x_mask, u_mask = cross_node_sparsity(expr, 2, 1, N)

    # Only node 3 is live
    for k in range(N):
        if k == 3:
            np.testing.assert_array_equal(x_mask[k, :], True)
            np.testing.assert_array_equal(u_mask[k, :], True)
        else:
            np.testing.assert_array_equal(x_mask[k, :], False)
            np.testing.assert_array_equal(u_mask[k, :], False)
