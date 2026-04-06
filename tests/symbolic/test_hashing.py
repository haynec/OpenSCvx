"""Tests for structural hashing of symbolic expressions.

This module tests the name-invariant structural hashing system used for caching
compiled solvers. The key properties being tested:

1. **Structural equivalence**: Same AST structure produces same hash
2. **Name invariance**: Different variable names with same structure produce same hash
3. **Structural differentiation**: Different structures produce different hashes
4. **Parameter handling**: Parameters hashed by shape, not value
5. **Constant handling**: Constants hashed by value
6. **Boundary conditions**: State boundary condition types affect hash
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Concat,
    Constant,
    Control,
    State,
)
from openscvx.symbolic.expr.constraint import CTCS
from openscvx.symbolic.expr.linalg import Norm
from openscvx.symbolic.expr.math import Cos, Huber, Sin, SmoothReLU
from openscvx.symbolic.hashing import hash_symbolic_problem


def hash_expr(expr):
    """Compute the structural hash of an expression."""
    return expr.structural_hash()


# =============================================================================
# Constant Hashing Tests
# =============================================================================


def test_constant_same_value_same_hash():
    """Constants with identical values should hash the same."""
    c1 = Constant(np.array([1.0, 2.0, 3.0]))
    c2 = Constant(np.array([1.0, 2.0, 3.0]))

    assert hash_expr(c1) == hash_expr(c2)


def test_constant_different_value_different_hash():
    """Constants with different values should hash differently."""
    c1 = Constant(np.array([1.0, 2.0, 3.0]))
    c2 = Constant(np.array([1.0, 2.0, 4.0]))

    assert hash_expr(c1) != hash_expr(c2)


def test_constant_different_shape_different_hash():
    """Constants with different shapes should hash differently."""
    c1 = Constant(np.array([1.0, 2.0]))
    c2 = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert hash_expr(c1) != hash_expr(c2)


def test_constant_scalar_hash():
    """Scalar constants should hash consistently."""
    c1 = Constant(5.0)
    c2 = Constant(5.0)
    c3 = Constant(6.0)

    assert hash_expr(c1) == hash_expr(c2)
    assert hash_expr(c1) != hash_expr(c3)


# =============================================================================
# Variable Hashing Tests
# =============================================================================


def test_state_requires_slice_for_hashing():
    """State without _slice should raise RuntimeError when hashing."""
    x = State("x", (3,))

    with pytest.raises(RuntimeError, match="Cannot hash Variable.*without _slice"):
        hash_expr(x)


def test_control_requires_slice_for_hashing():
    """Control without _slice should raise RuntimeError when hashing."""
    u = Control("u", (2,))

    with pytest.raises(RuntimeError, match="Cannot hash Variable.*without _slice"):
        hash_expr(u)


def test_same_slice_same_hash_different_names():
    """Variables with same slice but different names should hash the same."""
    x = State("position", (3,))
    x._slice = slice(0, 3)

    y = State("pos", (3,))
    y._slice = slice(0, 3)

    assert hash_expr(x) == hash_expr(y)


def test_different_slice_different_hash():
    """Variables with different slices should hash differently."""
    x = State("x", (3,))
    x._slice = slice(0, 3)

    y = State("y", (3,))
    y._slice = slice(3, 6)

    assert hash_expr(x) != hash_expr(y)


def test_state_vs_control_different_hash():
    """State and Control with same slice should hash differently (different class)."""
    x = State("x", (3,))
    x._slice = slice(0, 3)

    u = Control("u", (3,))
    u._slice = slice(0, 3)

    assert hash_expr(x) != hash_expr(u)


def test_variable_different_shape_different_hash():
    """Variables with different shapes should hash differently."""
    x = State("x", (3,))
    x._slice = slice(0, 3)

    y = State("y", (4,))
    y._slice = slice(0, 4)

    assert hash_expr(x) != hash_expr(y)


# =============================================================================
# State Boundary Condition Hashing Tests
# =============================================================================


def test_state_same_boundary_types_same_hash():
    """States with same boundary condition types should hash the same."""
    x = State("x", (2,))
    x._slice = slice(0, 2)
    x.min = [0.0, 0.0]
    x.max = [10.0, 10.0]
    x.initial = [1.0, 2.0]
    x.final = [5.0, 6.0]

    y = State("y", (2,))
    y._slice = slice(0, 2)
    y.min = [0.0, 0.0]
    y.max = [10.0, 10.0]
    y.initial = [3.0, 4.0]  # Different values, same types
    y.final = [7.0, 8.0]

    assert hash_expr(x) == hash_expr(y)


def test_state_different_boundary_types_different_hash():
    """States with different boundary condition types should hash differently."""
    x = State("x", (2,))
    x._slice = slice(0, 2)
    x.min = [0.0, 0.0]
    x.max = [10.0, 10.0]
    x.initial = [1.0, 2.0]  # Fixed

    y = State("y", (2,))
    y._slice = slice(0, 2)
    y.min = [0.0, 0.0]
    y.max = [10.0, 10.0]
    y.initial = [("free", 1.0), ("free", 2.0)]  # Free instead of Fixed

    assert hash_expr(x) != hash_expr(y)


def test_state_mixed_boundary_types():
    """States with mixed boundary types should hash based on type pattern."""
    x = State("x", (3,))
    x._slice = slice(0, 3)
    x.min = [0.0, 0.0, 0.0]
    x.max = [10.0, 10.0, 10.0]
    x.initial = [0.0, ("free", 1.0), ("minimize", 2.0)]

    y = State("y", (3,))
    y._slice = slice(0, 3)
    y.min = [0.0, 0.0, 0.0]
    y.max = [10.0, 10.0, 10.0]
    y.initial = [5.0, ("free", 6.0), ("minimize", 7.0)]  # Same types, different values

    z = State("z", (3,))
    z._slice = slice(0, 3)
    z.min = [0.0, 0.0, 0.0]
    z.max = [10.0, 10.0, 10.0]
    z.initial = [0.0, ("free", 1.0), ("maximize", 2.0)]  # Different: maximize vs minimize

    assert hash_expr(x) == hash_expr(y)
    assert hash_expr(x) != hash_expr(z)


# =============================================================================
# Arithmetic Expression Hashing Tests
# =============================================================================


def test_add_same_structure():
    """Add expressions with same structure should hash the same."""
    x = State("x", (2,))
    x._slice = slice(0, 2)
    y = State("y", (2,))
    y._slice = slice(2, 4)

    a = State("a", (2,))
    a._slice = slice(0, 2)
    b = State("b", (2,))
    b._slice = slice(2, 4)

    expr1 = x + y
    expr2 = a + b

    assert hash_expr(expr1) == hash_expr(expr2)


def test_add_vs_sub_different_hash():
    """Add and Sub with same operands should hash differently."""
    x = State("x", (2,))
    x._slice = slice(0, 2)
    c = Constant(np.array([1.0, 2.0]))

    add_expr = x + c
    sub_expr = x - c

    assert hash_expr(add_expr) != hash_expr(sub_expr)


def test_operand_order_matters():
    """Operand order should affect hash (non-commutative in AST)."""
    x = State("x", (2,))
    x._slice = slice(0, 2)
    y = State("y", (2,))
    y._slice = slice(2, 4)

    expr1 = x - y
    expr2 = y - x

    assert hash_expr(expr1) != hash_expr(expr2)


def test_nested_arithmetic():
    """Nested arithmetic expressions should hash based on structure."""
    x = State("x", (2,))
    x._slice = slice(0, 2)
    y = State("y", (2,))
    y._slice = slice(2, 4)
    c = Constant(np.array([1.0, 1.0]))

    expr1 = (x + y) * c

    a = State("a", (2,))
    a._slice = slice(0, 2)
    b = State("b", (2,))
    b._slice = slice(2, 4)
    c2 = Constant(np.array([1.0, 1.0]))

    expr2 = (a + b) * c2

    assert hash_expr(expr1) == hash_expr(expr2)


def test_negation():
    """Negation should be part of hash."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    expr1 = x
    expr2 = -x

    assert hash_expr(expr1) != hash_expr(expr2)


# =============================================================================
# Math Function Hashing Tests
# =============================================================================


def test_norm_same_ord():
    """Norm expressions with same ord should hash the same."""
    x = State("x", (3,))
    x._slice = slice(0, 3)

    y = State("y", (3,))
    y._slice = slice(0, 3)

    norm1 = Norm(x, ord=2)
    norm2 = Norm(y, ord=2)

    assert hash_expr(norm1) == hash_expr(norm2)


def test_norm_different_ord():
    """Norm expressions with different ord should hash differently."""
    x = State("x", (3,))
    x._slice = slice(0, 3)

    norm_l2 = Norm(x, ord=2)
    norm_l1 = Norm(x, ord=1)
    norm_linf = Norm(x, ord=float("inf"))

    assert hash_expr(norm_l2) != hash_expr(norm_l1)
    assert hash_expr(norm_l2) != hash_expr(norm_linf)
    assert hash_expr(norm_l1) != hash_expr(norm_linf)


def test_huber_same_delta():
    """Huber expressions with same delta should hash the same."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    y = State("y", (2,))
    y._slice = slice(0, 2)

    h1 = Huber(x, delta=1.0)
    h2 = Huber(y, delta=1.0)

    assert hash_expr(h1) == hash_expr(h2)


def test_huber_different_delta():
    """Huber expressions with different delta should hash differently."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    h1 = Huber(x, delta=1.0)
    h2 = Huber(x, delta=2.0)

    assert hash_expr(h1) != hash_expr(h2)


def test_smooth_relu_same_c():
    """SmoothReLU expressions with same c should hash the same."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    y = State("y", (2,))
    y._slice = slice(0, 2)

    sr1 = SmoothReLU(x, c=0.1)
    sr2 = SmoothReLU(y, c=0.1)

    assert hash_expr(sr1) == hash_expr(sr2)


def test_smooth_relu_different_c():
    """SmoothReLU expressions with different c should hash differently."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    sr1 = SmoothReLU(x, c=0.1)
    sr2 = SmoothReLU(x, c=0.5)

    assert hash_expr(sr1) != hash_expr(sr2)


def test_sin_cos_different_hash():
    """Sin and Cos of same operand should hash differently."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    sin_x = Sin(x)
    cos_x = Cos(x)

    assert hash_expr(sin_x) != hash_expr(cos_x)


# =============================================================================
# Indexing Hashing Tests
# =============================================================================


def test_same_index_same_hash():
    """Index expressions with same index should hash the same."""
    x = State("x", (3,))
    x._slice = slice(0, 3)

    y = State("y", (3,))
    y._slice = slice(0, 3)

    idx1 = x[0]
    idx2 = y[0]

    assert hash_expr(idx1) == hash_expr(idx2)


def test_different_index_different_hash():
    """Index expressions with different indices should hash differently."""
    x = State("x", (3,))
    x._slice = slice(0, 3)

    idx0 = x[0]
    idx1 = x[1]
    idx_slice = x[0:2]

    assert hash_expr(idx0) != hash_expr(idx1)
    assert hash_expr(idx0) != hash_expr(idx_slice)


def test_slice_range():
    """Slice expressions should hash based on slice parameters."""
    x = State("x", (5,))
    x._slice = slice(0, 5)

    s1 = x[0:2]
    s2 = x[1:3]
    s3 = x[0:2]

    assert hash_expr(s1) == hash_expr(s3)
    assert hash_expr(s1) != hash_expr(s2)


# =============================================================================
# NodeReference Hashing Tests
# =============================================================================


def test_node_reference_same_node_same_hash():
    """NodeReference with same node index should hash the same."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    y = State("y", (2,))
    y._slice = slice(0, 2)

    ref1 = x.at(5)
    ref2 = y.at(5)

    assert hash_expr(ref1) == hash_expr(ref2)


def test_node_reference_different_node_different_hash():
    """NodeReference with different node indices should hash differently."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    ref1 = x.at(5)
    ref2 = x.at(6)

    assert hash_expr(ref1) != hash_expr(ref2)


# =============================================================================
# Constraint Hashing Tests
# =============================================================================


def test_inequality_same_structure():
    """Inequality constraints with same structure should hash the same."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    y = State("y", (2,))
    y._slice = slice(0, 2)

    c1 = x <= 5
    c2 = y <= 5

    assert hash_expr(c1) == hash_expr(c2)


def test_inequality_direction_matters():
    """<= and >= constraints should hash differently."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    le = x <= 5
    ge = x >= 5

    assert hash_expr(le) != hash_expr(ge)


def test_equality_vs_inequality():
    """Equality and inequality constraints should hash differently."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    eq = x == 5
    le = x <= 5

    assert hash_expr(eq) != hash_expr(le)


def test_nodal_constraint_same_nodes():
    """NodalConstraint with same nodes should hash the same."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    y = State("y", (2,))
    y._slice = slice(0, 2)

    nc1 = (x <= 5).at([0, 5, 10])
    nc2 = (y <= 5).at([0, 5, 10])

    assert hash_expr(nc1) == hash_expr(nc2)


def test_nodal_constraint_different_nodes():
    """NodalConstraint with different nodes should hash differently."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    nc1 = (x <= 5).at([0, 5, 10])
    nc2 = (x <= 5).at([0, 5, 15])

    assert hash_expr(nc1) != hash_expr(nc2)


# =============================================================================
# CTCS Hashing Tests
# =============================================================================


def test_ctcs_same_params():
    """CTCS with same parameters should hash the same."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    y = State("y", (2,))
    y._slice = slice(0, 2)

    ctcs1 = CTCS(x <= 5, penalty="l2")
    ctcs2 = CTCS(y <= 5, penalty="l2")

    assert hash_expr(ctcs1) == hash_expr(ctcs2)


def test_ctcs_different_penalty():
    """CTCS with different penalty types should hash differently."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    ctcs_l2 = CTCS(x <= 5, penalty="l2")
    ctcs_l1 = CTCS(x <= 5, penalty="l1")

    assert hash_expr(ctcs_l2) != hash_expr(ctcs_l1)


def test_ctcs_with_nodes():
    """CTCS with node intervals should include them in hash."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    ctcs1 = CTCS(x <= 5).over((0, 50))
    ctcs2 = CTCS(x <= 5).over((0, 50))
    ctcs3 = CTCS(x <= 5).over((0, 100))

    assert hash_expr(ctcs1) == hash_expr(ctcs2)
    assert hash_expr(ctcs1) != hash_expr(ctcs3)


# =============================================================================
# Parameter Hashing Tests
# =============================================================================


def test_parameter_hash_by_shape_not_value():
    """Parameters should hash by shape, not value."""
    from openscvx.symbolic.expr import Parameter

    p1 = Parameter("mass", shape=(), value=1.0)
    p2 = Parameter("mass", shape=(), value=2.0)

    assert hash_expr(p1) == hash_expr(p2)


def test_parameter_different_shape_different_hash():
    """Parameters with different shapes should hash differently."""
    from openscvx.symbolic.expr import Parameter

    p1 = Parameter("vec2", shape=(2,), value=np.array([1.0, 2.0]))
    p2 = Parameter("vec3", shape=(3,), value=np.array([1.0, 2.0, 3.0]))

    assert hash_expr(p1) != hash_expr(p2)


# =============================================================================
# Full Problem Hashing Tests
# =============================================================================


def _make_simple_problem(state_name="x", control_name="u", N=50):
    """Create a simple SymbolicProblem for testing with slices pre-assigned."""
    from openscvx.symbolic.constraint_set import ConstraintSet
    from openscvx.symbolic.problem import SymbolicProblem

    x = State(state_name, (2,))
    x._slice = slice(0, 2)
    x.min = [-10.0, -10.0]
    x.max = [10.0, 10.0]
    x.initial = [0.0, 0.0]
    x.final = [5.0, 5.0]

    u = Control(control_name, (1,))
    u._slice = slice(0, 1)
    u.min = [-1.0]
    u.max = [1.0]

    dynamics = Concat(u, u)

    problem = SymbolicProblem(
        states=[x],
        controls=[u],
        dynamics=dynamics,
        constraints=ConstraintSet(),
        parameters={},
        N=N,
    )
    return problem


def test_problem_same_structure_same_hash():
    """Problems with identical structure should have same hash."""
    p1 = _make_simple_problem(state_name="position", control_name="thrust")
    p2 = _make_simple_problem(state_name="x", control_name="u")

    hash1 = hash_symbolic_problem(p1)
    hash2 = hash_symbolic_problem(p2)

    assert hash1 == hash2


def test_problem_different_n_different_hash():
    """Problems with different N should have different hash."""
    p1 = _make_simple_problem(N=50)
    p2 = _make_simple_problem(N=100)

    hash1 = hash_symbolic_problem(p1)
    hash2 = hash_symbolic_problem(p2)

    assert hash1 != hash2


def test_problem_different_dynamics_different_hash():
    """Problems with different dynamics structure should have different hash."""
    from openscvx.symbolic.constraint_set import ConstraintSet
    from openscvx.symbolic.problem import SymbolicProblem

    # Problem 1: simple linear dynamics
    x1 = State("x", (2,))
    x1._slice = slice(0, 2)
    x1.min = [-10.0, -10.0]
    x1.max = [10.0, 10.0]
    x1.initial = [0.0, 0.0]
    x1.final = [5.0, 5.0]

    u1 = Control("u", (2,))
    u1._slice = slice(0, 2)
    u1.min = [-1.0, -1.0]
    u1.max = [1.0, 1.0]

    p1 = SymbolicProblem(
        states=[x1],
        controls=[u1],
        dynamics=u1,  # dx/dt = u
        constraints=ConstraintSet(),
        parameters={},
        N=50,
    )

    # Problem 2: nonlinear dynamics
    x2 = State("x", (2,))
    x2._slice = slice(0, 2)
    x2.min = [-10.0, -10.0]
    x2.max = [10.0, 10.0]
    x2.initial = [0.0, 0.0]
    x2.final = [5.0, 5.0]

    u2 = Control("u", (2,))
    u2._slice = slice(0, 2)
    u2.min = [-1.0, -1.0]
    u2.max = [1.0, 1.0]

    p2 = SymbolicProblem(
        states=[x2],
        controls=[u2],
        dynamics=u2 * x2,  # dx/dt = u * x (nonlinear)
        constraints=ConstraintSet(),
        parameters={},
        N=50,
    )

    hash1 = hash_symbolic_problem(p1)
    hash2 = hash_symbolic_problem(p2)

    assert hash1 != hash2


def test_problem_different_constraints_different_hash():
    """Problems with different constraints should have different hash."""
    from openscvx.symbolic.constraint_set import ConstraintSet
    from openscvx.symbolic.problem import SymbolicProblem

    # Problem 1: no constraints
    x1 = State("x", (2,))
    x1._slice = slice(0, 2)
    x1.min = [-10.0, -10.0]
    x1.max = [10.0, 10.0]
    x1.initial = [0.0, 0.0]
    x1.final = [5.0, 5.0]

    u1 = Control("u", (2,))
    u1._slice = slice(0, 2)
    u1.min = [-1.0, -1.0]
    u1.max = [1.0, 1.0]

    p1 = SymbolicProblem(
        states=[x1],
        controls=[u1],
        dynamics=u1,
        constraints=ConstraintSet(),
        parameters={},
        N=50,
    )

    # Problem 2: with constraint
    x2 = State("x", (2,))
    x2._slice = slice(0, 2)
    x2.min = [-10.0, -10.0]
    x2.max = [10.0, 10.0]
    x2.initial = [0.0, 0.0]
    x2.final = [5.0, 5.0]

    u2 = Control("u", (2,))
    u2._slice = slice(0, 2)
    u2.min = [-1.0, -1.0]
    u2.max = [1.0, 1.0]

    p2 = SymbolicProblem(
        states=[x2],
        controls=[u2],
        dynamics=u2,
        constraints=ConstraintSet(nodal=[Norm(x2) <= 5.0]),
        parameters={},
        N=50,
    )

    hash1 = hash_symbolic_problem(p1)
    hash2 = hash_symbolic_problem(p2)

    assert hash1 != hash2


def test_problem_bound_values_dont_affect_hash():
    """Different bound values (same structure) should produce same hash."""
    from openscvx.symbolic.constraint_set import ConstraintSet
    from openscvx.symbolic.problem import SymbolicProblem

    # Problem 1: bounds [-10, 10]
    x1 = State("x", (2,))
    x1._slice = slice(0, 2)
    x1.min = [-10.0, -10.0]
    x1.max = [10.0, 10.0]
    x1.initial = [0.0, 0.0]
    x1.final = [5.0, 5.0]

    u1 = Control("u", (2,))
    u1._slice = slice(0, 2)
    u1.min = [-1.0, -1.0]
    u1.max = [1.0, 1.0]

    p1 = SymbolicProblem(
        states=[x1],
        controls=[u1],
        dynamics=u1,
        constraints=ConstraintSet(),
        parameters={},
        N=50,
    )

    # Problem 2: bounds [-100, 100] (different values)
    x2 = State("x", (2,))
    x2._slice = slice(0, 2)
    x2.min = [-100.0, -100.0]
    x2.max = [100.0, 100.0]
    x2.initial = [1.0, 1.0]
    x2.final = [50.0, 50.0]

    u2 = Control("u", (2,))
    u2._slice = slice(0, 2)
    u2.min = [-10.0, -10.0]
    u2.max = [10.0, 10.0]

    p2 = SymbolicProblem(
        states=[x2],
        controls=[u2],
        dynamics=u2,
        constraints=ConstraintSet(),
        parameters={},
        N=50,
    )

    hash1 = hash_symbolic_problem(p1)
    hash2 = hash_symbolic_problem(p2)

    assert hash1 == hash2


# =============================================================================
# Edge Cases
# =============================================================================


def test_deeply_nested_expression():
    """Deeply nested expressions should hash correctly."""
    x = State("x", (2,))
    x._slice = slice(0, 2)

    expr = x
    for _ in range(10):
        expr = expr + Constant(np.array([0.1, 0.1]))
        expr = expr * Constant(np.array([0.9, 0.9]))

    h1 = hash_expr(expr)
    h2 = hash_expr(expr)
    assert h1 == h2


def test_hash_is_bytes():
    """structural_hash should return bytes."""
    c = Constant(1.0)
    h = hash_expr(c)
    assert isinstance(h, bytes)


def test_hash_length():
    """structural_hash should return SHA-256 (32 bytes)."""
    c = Constant(1.0)
    h = hash_expr(c)
    assert len(h) == 32


def test_empty_concat():
    """Empty Concat should hash consistently."""
    c1 = Concat()
    c2 = Concat()
    assert hash_expr(c1) == hash_expr(c2)


def test_concat_order_matters():
    """Order of expressions in Concat should affect hash."""
    x = State("x", (2,))
    x._slice = slice(0, 2)
    y = State("y", (2,))
    y._slice = slice(2, 4)

    c1 = Concat(x, y)
    c2 = Concat(y, x)

    assert hash_expr(c1) != hash_expr(c2)


# =============================================================================
# Constraint Order Invariance Tests
# =============================================================================


def test_problem_nodal_constraint_order_invariant():
    """Problem hash should be the same regardless of nodal constraint order."""
    from openscvx.symbolic.constraint_set import ConstraintSet
    from openscvx.symbolic.problem import SymbolicProblem

    def make_problem_with_constraints(constraint_order):
        x = State("x", (3,))
        x._slice = slice(0, 3)
        x.min = [-10.0, -10.0, -10.0]
        x.max = [10.0, 10.0, 10.0]
        x.initial = [0.0, 0.0, 0.0]
        x.final = [5.0, 5.0, 5.0]

        u = Control("u", (2,))
        u._slice = slice(0, 2)
        u.min = [-1.0, -1.0]
        u.max = [1.0, 1.0]

        # Create constraints in the specified order
        constraints_map = {
            "norm_x": Norm(x) <= 5.0,
            "x0_positive": x[0] >= 0,
            "x1_bounded": x[1] <= 3.0,
        }
        ordered_constraints = [constraints_map[name] for name in constraint_order]

        return SymbolicProblem(
            states=[x],
            controls=[u],
            dynamics=Concat(u, Constant(np.array([0.0]))),
            constraints=ConstraintSet(nodal=ordered_constraints),
            parameters={},
            N=50,
        )

    # Create problems with constraints in different orders
    p1 = make_problem_with_constraints(["norm_x", "x0_positive", "x1_bounded"])
    p2 = make_problem_with_constraints(["x1_bounded", "norm_x", "x0_positive"])
    p3 = make_problem_with_constraints(["x0_positive", "x1_bounded", "norm_x"])

    hash1 = hash_symbolic_problem(p1)
    hash2 = hash_symbolic_problem(p2)
    hash3 = hash_symbolic_problem(p3)

    assert hash1 == hash2
    assert hash2 == hash3


def test_problem_ctcs_constraint_order_invariant():
    """Problem hash should be the same regardless of CTCS constraint order."""
    from openscvx.symbolic.constraint_set import ConstraintSet
    from openscvx.symbolic.problem import SymbolicProblem

    def make_problem_with_ctcs(constraint_order):
        x = State("x", (2,))
        x._slice = slice(0, 2)
        x.min = [-10.0, -10.0]
        x.max = [10.0, 10.0]
        x.initial = [0.0, 0.0]
        x.final = [5.0, 5.0]

        u = Control("u", (2,))
        u._slice = slice(0, 2)
        u.min = [-1.0, -1.0]
        u.max = [1.0, 1.0]

        # Create CTCS constraints in the specified order
        constraints_map = {
            "ctcs_norm": CTCS(Norm(x) <= 3.0, penalty="l2"),
            "ctcs_x0": CTCS(x[0] >= -1.0, penalty="l1"),
            "ctcs_x1": CTCS(x[1] <= 2.0, penalty="l2"),
        }
        ordered_constraints = [constraints_map[name] for name in constraint_order]

        return SymbolicProblem(
            states=[x],
            controls=[u],
            dynamics=u,
            constraints=ConstraintSet(ctcs=ordered_constraints),
            parameters={},
            N=50,
        )

    p1 = make_problem_with_ctcs(["ctcs_norm", "ctcs_x0", "ctcs_x1"])
    p2 = make_problem_with_ctcs(["ctcs_x1", "ctcs_norm", "ctcs_x0"])
    p3 = make_problem_with_ctcs(["ctcs_x0", "ctcs_x1", "ctcs_norm"])

    hash1 = hash_symbolic_problem(p1)
    hash2 = hash_symbolic_problem(p2)
    hash3 = hash_symbolic_problem(p3)

    assert hash1 == hash2
    assert hash2 == hash3


def test_problem_mixed_constraint_order_invariant():
    """Problem hash should be order-invariant across all constraint categories."""
    from openscvx.symbolic.constraint_set import ConstraintSet
    from openscvx.symbolic.problem import SymbolicProblem

    def make_problem(nodal_order, ctcs_order):
        x = State("x", (2,))
        x._slice = slice(0, 2)
        x.min = [-10.0, -10.0]
        x.max = [10.0, 10.0]
        x.initial = [0.0, 0.0]
        x.final = [5.0, 5.0]

        u = Control("u", (2,))
        u._slice = slice(0, 2)
        u.min = [-1.0, -1.0]
        u.max = [1.0, 1.0]

        nodal_map = {
            "a": Norm(x) <= 5.0,
            "b": x[0] >= -2.0,
        }
        ctcs_map = {
            "c": CTCS(x[1] <= 1.0),
            "d": CTCS(Norm(u) <= 0.5),
        }

        return SymbolicProblem(
            states=[x],
            controls=[u],
            dynamics=u,
            constraints=ConstraintSet(
                nodal=[nodal_map[k] for k in nodal_order],
                ctcs=[ctcs_map[k] for k in ctcs_order],
            ),
            parameters={},
            N=50,
        )

    # All permutations should hash the same
    p1 = make_problem(["a", "b"], ["c", "d"])
    p2 = make_problem(["b", "a"], ["c", "d"])
    p3 = make_problem(["a", "b"], ["d", "c"])
    p4 = make_problem(["b", "a"], ["d", "c"])

    hashes = [hash_symbolic_problem(p) for p in [p1, p2, p3, p4]]

    assert all(h == hashes[0] for h in hashes)


def test_duplicate_constraints_same_hash():
    """Duplicate constraints should produce consistent hashes."""
    from openscvx.symbolic.constraint_set import ConstraintSet
    from openscvx.symbolic.problem import SymbolicProblem

    def make_problem(num_duplicates):
        x = State("x", (2,))
        x._slice = slice(0, 2)
        x.min = [-10.0, -10.0]
        x.max = [10.0, 10.0]
        x.initial = [0.0, 0.0]
        x.final = [5.0, 5.0]

        u = Control("u", (2,))
        u._slice = slice(0, 2)
        u.min = [-1.0, -1.0]
        u.max = [1.0, 1.0]

        # Create identical constraints
        constraints = [Norm(x) <= 5.0 for _ in range(num_duplicates)]

        return SymbolicProblem(
            states=[x],
            controls=[u],
            dynamics=u,
            constraints=ConstraintSet(nodal=constraints),
            parameters={},
            N=50,
        )

    # Same number of duplicates should hash the same
    p1 = make_problem(3)
    p2 = make_problem(3)
    assert hash_symbolic_problem(p1) == hash_symbolic_problem(p2)

    # Different number of duplicates should hash differently
    p3 = make_problem(2)
    assert hash_symbolic_problem(p1) != hash_symbolic_problem(p3)


# =============================================================================
# BYOF Hashing Tests
# =============================================================================


def test_byof_hash_none():
    """None byof should return empty bytes."""
    from openscvx.utils.caching import _hash_byof

    assert _hash_byof(None) == b""


def test_byof_hash_empty():
    """Empty ByofSpec should return empty bytes."""
    from openscvx.expert.byof import ByofSpec
    from openscvx.utils.caching import _hash_byof

    assert _hash_byof(ByofSpec()) == b""


def test_byof_hash_changes_with_function():
    """Different lambda implementations should produce different hashes."""
    from openscvx.expert.byof import ByofSpec, NodalConstraintSpec
    from openscvx.utils.caching import _hash_byof

    byof1 = ByofSpec(
        nodal_constraints=[NodalConstraintSpec(constraint_fn=lambda x, u, n, p: x[0] - 10.0)]
    )
    byof2 = ByofSpec(
        nodal_constraints=[NodalConstraintSpec(constraint_fn=lambda x, u, n, p: x[0] - 20.0)]
    )

    assert _hash_byof(byof1) != _hash_byof(byof2)


def test_byof_hash_same_function_same_hash():
    """Identical lambda implementations should produce same hash."""
    from openscvx.expert.byof import ByofSpec, NodalConstraintSpec
    from openscvx.utils.caching import _hash_byof

    byof1 = ByofSpec(
        nodal_constraints=[NodalConstraintSpec(constraint_fn=lambda x, u, n, p: x[0] - 10.0)]
    )
    byof2 = ByofSpec(
        nodal_constraints=[NodalConstraintSpec(constraint_fn=lambda x, u, n, p: x[0] - 10.0)]
    )

    assert _hash_byof(byof1) == _hash_byof(byof2)
