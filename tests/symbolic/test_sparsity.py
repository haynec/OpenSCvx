"""Tests for static Jacobian sparsity analysis.

Covers:
- Expr.sparsity() method (element-level propagation through AST)
- jacobian_sparsity() wrapper in openscvx.symbolic.sparsity
- cross_node_sparsity() for cross-node constraints
"""

import numpy as np

from openscvx.symbolic.expr import (
    Concat,
    Constant,
    Control,
    MatMul,
    Norm,
    Parameter,
    Sin,
    State,
    Sum,
    Transpose,
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

    n_x = sum(s.shape[0] for s in states)  # 7
    n_u = sum(c.shape[0] for c in controls)  # 3
    return pos, vel, mass, thrust, states, controls, n_x, n_u


# =============================================================================
# Leaf-level element-level exact sparsity
# =============================================================================


def test_state_leaf_element_level():
    """State leaf gives diagonal identity block at its slice."""
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()
    df_dx, df_du = vel.sparsity(n_x, n_u)

    assert df_dx.shape == (3, n_x)
    assert df_du.shape == (3, n_u)

    # vel occupies x[3:6] — element-level diagonal
    expected = np.zeros((3, n_x), dtype=bool)
    expected[0, 3] = True
    expected[1, 4] = True
    expected[2, 5] = True
    np.testing.assert_array_equal(df_dx, expected)
    np.testing.assert_array_equal(df_du, False)


def test_control_leaf_element_level():
    """Control leaf gives diagonal identity block at its slice."""
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()
    df_dx, df_du = thrust.sparsity(n_x, n_u)

    assert df_dx.shape == (3, n_x)
    assert df_du.shape == (3, n_u)
    np.testing.assert_array_equal(df_dx, False)

    expected = np.eye(3, dtype=bool)
    np.testing.assert_array_equal(df_du, expected)


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


# =============================================================================
# Unary pass-through preserves element-level precision
# =============================================================================


def test_sin_preserves_element_level():
    """Sin(state) preserves diagonal sparsity from the leaf."""
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()
    expr = Sin(vel)
    df_dx, _ = expr.sparsity(n_x, n_u)

    # Should be identical to vel's own sparsity (diagonal)
    expected = np.zeros((3, n_x), dtype=bool)
    expected[0, 3] = True
    expected[1, 4] = True
    expected[2, 5] = True
    np.testing.assert_array_equal(df_dx, expected)


# =============================================================================
# Binary union with broadcasting
# =============================================================================


def test_mixed_state_and_control():
    """thrust / mass depends on both u[0:3] and x[6:7]."""
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()
    expr = thrust / mass  # shape (3,) / shape (1,) => shape (3,)
    df_dx, df_du = jacobian_sparsity(expr, n_x, n_u)

    assert df_dx.shape == (3, n_x)
    # x dependence: mass (index 6) broadcasts to all 3 rows
    expected_x = np.zeros((3, n_x), dtype=bool)
    expected_x[:, 6] = True
    np.testing.assert_array_equal(df_dx, expected_x)

    # u dependence: thrust is diagonal + mass is zero for u => diagonal
    expected_u = np.eye(3, dtype=bool)
    np.testing.assert_array_equal(df_du, expected_u)


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
# Concat decomposition (row-block analysis)
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
    # pos_dot = vel => element-level diagonal in x[3:6]
    np.testing.assert_array_equal(A[0:3, 0:3], False)
    expected_vel_block = np.eye(3, dtype=bool)
    np.testing.assert_array_equal(A[0:3, 3:6], expected_vel_block)
    np.testing.assert_array_equal(A[0:3, 6:7], False)

    # vel_dot = thrust/mass - g => depends on mass (x[6]) for all 3 rows
    np.testing.assert_array_equal(A[3:6, 0:3], False)
    np.testing.assert_array_equal(A[3:6, 3:6], False)
    np.testing.assert_array_equal(A[3:6, 6:7], True)

    # mass_dot = -alpha * ||thrust|| => no state dep
    np.testing.assert_array_equal(A[6:7, :], False)

    # --- df/du (B pattern) ---
    # pos_dot = vel => no control dep
    np.testing.assert_array_equal(B[0:3, :], False)

    # vel_dot = thrust/mass - g => thrust diagonal + mass broadcast
    expected_vel_u = np.eye(3, dtype=bool)
    np.testing.assert_array_equal(B[3:6, :], expected_vel_u)

    # mass_dot = -alpha * ||thrust|| => Norm reduces, so all u cols True
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

    # x_dot depends only on x (cols 0:2) — element-level diagonal
    np.testing.assert_array_equal(A[0:2, 0:2], np.eye(2, dtype=bool))
    np.testing.assert_array_equal(A[0:2, 2:4], False)

    # y_dot depends only on y (cols 2:4) — element-level diagonal
    np.testing.assert_array_equal(A[2:4, 0:2], False)
    np.testing.assert_array_equal(A[2:4, 2:4], np.eye(2, dtype=bool))

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

    inner = Concat(x, y)  # rows 0-1
    outer = Concat(inner, z)  # row 2
    A, _ = jacobian_sparsity(outer, 3, 0)

    np.testing.assert_array_equal(A[0], [True, False, False])
    np.testing.assert_array_equal(A[1], [False, True, False])
    np.testing.assert_array_equal(A[2], [False, False, True])


# =============================================================================
# Index — row selection from base sparsity
# =============================================================================


def test_index_selects_sparsity_rows():
    """Indexing a state selects the correct sparsity rows."""
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()
    # vel[1] should give the single row for x[4]
    expr = vel[1]
    df_dx, _ = expr.sparsity(n_x, n_u)

    assert df_dx.shape == (1, n_x)
    expected = np.zeros(n_x, dtype=bool)
    expected[4] = True
    np.testing.assert_array_equal(df_dx[0], expected)


def test_index_slice():
    """Slicing a state selects a contiguous block of sparsity rows."""
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()
    # vel[0:2] should give rows for x[3] and x[4]
    expr = vel[0:2]
    df_dx, _ = expr.sparsity(n_x, n_u)

    assert df_dx.shape == (2, n_x)
    expected = np.zeros((2, n_x), dtype=bool)
    expected[0, 3] = True
    expected[1, 4] = True
    np.testing.assert_array_equal(df_dx, expected)


# =============================================================================
# Transpose — row permutation
# =============================================================================


def test_transpose_2d():
    """Transpose of a 2D expression permutes sparsity rows."""
    x = State("x", (2,))
    y = State("y", (3,))
    collect_and_assign_slices([x, y], [])
    # Stack creates (2, 5) from two (5,) rows — but we need a 2D expr.
    # Use a simpler test: Concat gives (5,), transpose of (5,) is (5,) — identity.
    # Instead, create a (2,3) matrix via Block or Stack.
    from openscvx.symbolic.expr import Stack

    # Stack([x, x]) gives shape (2, 2). Transpose gives (2, 2).
    mat = Stack([x, x])  # shape (2, 2)
    t = Transpose(mat)
    S_x, _ = t.sparsity(5, 0)

    # mat flattened: [x[0], x[1], x[0], x[1]]
    # transposed (2,2).T = (2,2), flattened: [x[0], x[0], x[1], x[1]]
    mat_S, _ = mat.sparsity(5, 0)
    # mat_S rows: 0->x[0], 1->x[1], 2->x[0], 3->x[1]
    # After transpose, output flattened: 0->(0,0)=x[0], 1->(1,0)=x[0], 2->(0,1)=x[1], 3->(1,1)=x[1]
    assert S_x.shape == (4, 5)
    # Row 0: depends on x[0]
    assert S_x[0, 0] and not S_x[0, 1]
    # Row 1: depends on x[0]
    assert S_x[1, 0] and not S_x[1, 1]
    # Row 2: depends on x[1]
    assert S_x[2, 1] and not S_x[2, 0]
    # Row 3: depends on x[1]
    assert S_x[3, 1] and not S_x[3, 0]


# =============================================================================
# Sum / Norm — reduction
# =============================================================================


def test_sum_reduces_sparsity():
    """Sum collapses all rows into one via any()."""
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()
    expr = Sum(vel)  # scalar
    df_dx, _ = expr.sparsity(n_x, n_u)

    assert df_dx.shape == (1, n_x)
    # Should have True at indices 3,4,5 (vel's columns)
    expected = np.zeros(n_x, dtype=bool)
    expected[3:6] = True
    np.testing.assert_array_equal(df_dx[0], expected)


# =============================================================================
# Default fallback (conservative) for exotic ops
# =============================================================================


def test_default_fallback_is_conservative():
    """Ops without explicit sparsity override get conservative union+tile."""
    from openscvx.symbolic.expr import QDCM

    q = State("q", (4,))
    collect_and_assign_slices([q], [])
    expr = QDCM(q)  # shape (3,3), uses default Expr.sparsity

    S_x, _ = expr.sparsity(4, 0)
    assert S_x.shape == (9, 4)
    # Conservative: every output element depends on all 4 quaternion components
    np.testing.assert_array_equal(S_x, True)


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

    for k in range(N):
        if k in (4, 5):
            np.testing.assert_array_equal(x_mask[k, 0:3], True)
        else:
            np.testing.assert_array_equal(x_mask[k, :], False)
    np.testing.assert_array_equal(x_mask[:, 3:6], False)


def test_cross_node_negative_index():
    """Negative node indices are normalized correctly."""
    x = State("x", (2,))
    states = [x]
    controls = []
    collect_and_assign_slices(states, controls)
    N = 5

    expr = x.at(0) - x.at(-1)
    x_mask, u_mask = cross_node_sparsity(expr, 2, 0, N)

    np.testing.assert_array_equal(x_mask[0, :], True)
    np.testing.assert_array_equal(x_mask[4, :], True)
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

    for k in range(N):
        if k == 3:
            np.testing.assert_array_equal(x_mask[k, :], True)
            np.testing.assert_array_equal(u_mask[k, :], True)
        else:
            np.testing.assert_array_equal(x_mask[k, :], False)
            np.testing.assert_array_equal(u_mask[k, :], False)


# =============================================================================
# MatMul — boolean matrix multiply sparsity
# =============================================================================


def test_matmul_constant_matrix_times_state():
    """Constant @ State filters through the constant's zero structure."""
    x = State("x", (3,))
    collect_and_assign_slices([x], [])

    # A has a sparse structure: row 0 uses x[0] only, row 1 uses x[1,2]
    A = Constant(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]]))
    expr = MatMul(A, x)  # (2,3) @ (3,) -> (2,)
    S_x, S_u = expr.sparsity(3, 0)

    assert S_x.shape == (2, 3)
    # Row 0: only x[0] is live (A[0,:] = [1,0,0])
    np.testing.assert_array_equal(S_x[0], [True, False, False])
    # Row 1: x[1] and x[2] are live (A[1,:] = [0,2,3])
    np.testing.assert_array_equal(S_x[1], [False, True, True])


def test_matmul_identity_constant():
    """Identity matrix @ State gives exact diagonal sparsity."""
    x = State("x", (3,))
    collect_and_assign_slices([x], [])

    eye = Constant(np.eye(3))
    expr = MatMul(eye, x)
    S_x, _ = expr.sparsity(3, 0)

    np.testing.assert_array_equal(S_x, np.eye(3, dtype=bool))


def test_matmul_parameter_times_state_is_conservative():
    """Parameter @ State gives column-level (not element-level) sparsity."""
    x = State("x", (3,))
    collect_and_assign_slices([x], [])

    # Parameter has a sparse value, but we don't use it for liveness
    P = Parameter("P", (2, 3), value=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    expr = MatMul(P, x)
    S_x, _ = expr.sparsity(3, 0)

    assert S_x.shape == (2, 3)
    # Conservative: every output depends on all state elements
    np.testing.assert_array_equal(S_x, True)


def test_matmul_state_times_constant_vector():
    """State-derived matrix @ constant vector uses vector liveness."""
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()

    # vel (3,) viewed as left operand, constant (3,) right operand with a zero
    # This is a dot product: (3,) @ (3,) -> scalar
    c = Constant(np.array([1.0, 0.0, 1.0]))
    expr = MatMul(vel, c)
    S_x, _ = expr.sparsity(n_x, n_u)

    assert S_x.shape == (1, n_x)
    # vel is x[3:6]; c has zeros at index 1 => vel[1] (x[4]) is filtered out
    expected = np.zeros(n_x, dtype=bool)
    expected[3] = True  # vel[0] * c[0]=1
    expected[5] = True  # vel[2] * c[2]=1
    np.testing.assert_array_equal(S_x[0], expected)


def test_matmul_vector_times_matrix():
    """Vector @ Matrix: (n,) @ (n,k) -> (k,)."""
    x = State("x", (2,))
    collect_and_assign_slices([x], [])

    # (2,) @ (2, 3) -> (3,)
    M = Constant(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]))
    expr = MatMul(x, M)
    S_x, _ = expr.sparsity(2, 0)

    assert S_x.shape == (3, 2)
    # col 0 of M: [1,0] -> uses x[0] only
    np.testing.assert_array_equal(S_x[0], [True, False])
    # col 1 of M: [0,3] -> uses x[1] only
    np.testing.assert_array_equal(S_x[1], [False, True])
    # col 2 of M: [2,0] -> uses x[0] only
    np.testing.assert_array_equal(S_x[2], [True, False])


def test_matmul_matrix_times_matrix():
    """Matrix @ Matrix: (m,n) @ (n,k) -> (m,k)."""
    x = State("x", (4,))
    collect_and_assign_slices([x], [])

    # Build a (2,4) "state matrix" from x via Concat + reshaping is complex,
    # so test with Constant @ Constant @ State to verify matrix-matrix path.
    # A (2,2) @ (2,4_state) but we need a (2,4) state-dependent expr.
    # Simpler: use Constant (2,3) @ Constant (3,4) — both constant, result all-False.
    A = Constant(np.ones((2, 3)))
    B = Constant(np.ones((3, 4)))
    expr = MatMul(A, B)
    S_x, _ = expr.sparsity(4, 0)

    assert S_x.shape == (8, 4)
    np.testing.assert_array_equal(S_x, False)


def test_matmul_in_rocket_dynamics():
    """MatMul with sparse constant in realistic dynamics: R @ thrust / mass."""
    pos, vel, mass, thrust, states, controls, n_x, n_u = _make_rocket_vars()

    # Rotation-like selector: only first two thrust components affect vel
    R = Constant(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    accel = MatMul(R, thrust) / mass  # (3,) / (1,)

    S_x, S_u = accel.sparsity(n_x, n_u)
    assert S_x.shape == (3, n_x)
    assert S_u.shape == (3, n_u)

    # x dependence: mass (x[6]) broadcasts to all 3 rows
    expected_x = np.zeros((3, n_x), dtype=bool)
    expected_x[:, 6] = True
    np.testing.assert_array_equal(S_x, expected_x)

    # u dependence: R filters thrust — only u[0] and u[1] are live
    # Row 0: R[0,:] = [1,0,0] -> u[0]
    # Row 1: R[1,:] = [0,1,0] -> u[1]
    # Row 2: R[2,:] = [0,0,0] -> nothing
    np.testing.assert_array_equal(S_u[0], [True, False, False])
    np.testing.assert_array_equal(S_u[1], [False, True, False])
    np.testing.assert_array_equal(S_u[2], [False, False, False])
