"""Tests for JAX lowering infrastructure.

This module tests the JAX lowering framework itself, not specific node types.
Node-specific lowering tests have been moved to their respective node test files:
- test_parameters.py: Parameter lowering
- test_arithmetic.py: Arithmetic operation lowering
- test_array.py: Array operation lowering
- test_math.py: Math function lowering
- test_linalg.py: Linear algebra lowering
- test_spatial.py: Spatial/6DOF lowering
- test_constraint.py: Constraint lowering
- test_variable.py: Variable (State/Control) lowering

This file contains:
- Lowering framework infrastructure tests
- Integration tests combining multiple node types
- Constant normalization tests (cross-cutting concerns)
- Implicit conversion equivalence tests
"""

import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Add,
    Concat,
    Constant,
    Control,
    Expr,
    MatMul,
    State,
)
from openscvx.symbolic.lower import lower, lower_to_jax
from openscvx.symbolic.lowerers.jax import JaxLowerer

# =============================================================================
# Lowering Framework Infrastructure Tests
# =============================================================================


class UnregisteredExpr(Expr):
    pass


def test_jaxlowerer_raises_when_no_visitor_registered():
    jl = JaxLowerer()
    node = UnregisteredExpr()
    with pytest.raises(NotImplementedError) as excinfo:
        # this should internally call dispatch() and fail
        jl.lower(node)

    msg = str(excinfo.value)
    assert "JaxLowerer" in msg, "should mention the lowerer class name"
    assert "UnregisteredExpr" in msg, "should mention the Expr subclass name"


def test_top_level_lower_raises_for_unregistered_expr():
    jl = JaxLowerer()
    node = UnregisteredExpr()
    # our top-level lower() simply forwards to jl.lower(...)
    with pytest.raises(NotImplementedError):
        lower(node, jl)


def test_jax_lower_constant():
    const = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    jl = JaxLowerer()
    f = jl.lower(const)
    out = f(None, None, None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2, 2)
    assert jnp.allclose(out, jnp.array([[1, 2], [3, 4]]))


def test_lower_to_jax_constant_produces_callable():
    c = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    fns = lower_to_jax([c])
    assert isinstance(fns, list) and len(fns) == 1
    fn = fns[0]
    out = fn(None, None, None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2, 2)
    assert jnp.allclose(out, jnp.array([[1.0, 2.0], [3.0, 4.0]]))


def test_lower_to_jax_add_with_slices():
    x = jnp.arange(6.0)
    a = State("a", (3,))
    a._slice = slice(0, 3)
    b = State("b", (3,))
    b._slice = slice(3, 6)
    expr = Add(a, b)

    fn = lower_to_jax(expr)
    out = fn(x, None, None, None)
    expected = x[0:3] + x[3:6]
    assert jnp.allclose(out, expected)


def test_lower_to_jax_multiple_exprs_returns_in_order():
    x = jnp.array([10.0, 20.0, 30.0])
    u = jnp.array([1.0, 2.0, 3.0])
    # expr1: constant, expr2: identity of x
    c = Constant(np.array([1.0, 2.0, 3.0]))
    v = State("v", (3,))
    v._slice = slice(0, 3)
    exprs = [c, v]

    fns = lower_to_jax(exprs)
    assert len(fns) == 2

    f_const, f_x = fns
    assert jnp.allclose(f_const(x, None, None, None), jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(f_x(x, u, None, None), x)


# =============================================================================
# Integration Tests
# =============================================================================


def test_lower_to_jax_double_integrator():
    x = jnp.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
    u = jnp.array([1.0, 1.0, 1.0])
    g = 9.81
    m = 1.0
    pos = State("pos", (3,))
    pos._slice = slice(0, 3)
    vel = State("vel", (3,))
    vel._slice = slice(3, 6)

    acc = Control("acc", (3,))
    acc._slice = slice(0, 3)

    pos_dot = vel
    vel_dot = acc / m + Constant(np.array([0.0, 0.0, g]))

    dynamics_expr = Concat(pos_dot, vel_dot)
    fn = lower_to_jax(dynamics_expr)
    xdot = fn(x, u, None, None)

    expected = jnp.concatenate([x[3:6], u / m + jnp.array([0.0, 0.0, g])], axis=0)

    assert jnp.allclose(xdot, expected)
    assert xdot.shape == (6,)


def test_lower_to_jax_double_integrator_indexed():
    # numeric inputs
    x_jax = jnp.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
    u_jax = jnp.array([1.0, 1.0, 1.0])
    g = 9.81
    m = 1.0

    # one 6-vector state
    x = State("x", (6,))
    x._slice = slice(0, 6)

    # 3-vector control
    u = Control("u", (3,))
    u._slice = slice(0, 3)

    pos_dot = x[3:6]
    vel_dot = u / m + Constant(np.array([0.0, 0.0, g]))
    dynamics_expr = Concat(pos_dot, vel_dot)

    # lower and execute
    fn = lower_to_jax(dynamics_expr)
    xdot = fn(x_jax, u_jax, None, None)

    # expected by hand
    expected = jnp.concatenate([x_jax[3:6], u_jax / m + jnp.array([0.0, 0.0, g])], axis=0)

    assert jnp.allclose(xdot, expected)
    assert xdot.shape == (6,)


# =============================================================================
# Constant Normalization Tests (Cross-Cutting Preprocessing)
# =============================================================================


def test_normalized_constants_lower_correctly():
    """Test that normalized constants work correctly with JAX lowering"""

    jl = JaxLowerer()

    # Test scalar constant that was squeezed from higher dimensions
    scalar_squeezed = Constant(np.array([[5.0]]))  # (1,1) -> () after squeeze
    assert scalar_squeezed.value.shape == ()  # Verify normalization happened

    fn_scalar = jl.lower(scalar_squeezed)
    result_scalar = fn_scalar(None, None, None, None)

    assert isinstance(result_scalar, jnp.ndarray)
    assert result_scalar.shape == ()
    assert jnp.allclose(result_scalar, 5.0)

    # Test vector constant that was squeezed
    vector_squeezed = Constant(np.array([[1.0, 2.0, 3.0]]))  # (1,3) -> (3,) after squeeze
    assert vector_squeezed.value.shape == (3,)  # Verify normalization happened

    fn_vector = jl.lower(vector_squeezed)
    result_vector = fn_vector(None, None, None, None)

    assert isinstance(result_vector, jnp.ndarray)
    assert result_vector.shape == (3,)
    assert jnp.allclose(result_vector, jnp.array([1.0, 2.0, 3.0]))

    # Test matrix that had singleton dimensions removed
    matrix_squeezed = Constant(
        np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    )  # (1,2,1,2) -> (2,2) after squeeze
    assert matrix_squeezed.value.shape == (2, 2)  # Verify normalization happened

    fn_matrix = jl.lower(matrix_squeezed)
    result_matrix = fn_matrix(None, None, None, None)

    assert isinstance(result_matrix, jnp.ndarray)
    assert result_matrix.shape == (2, 2)
    assert jnp.allclose(result_matrix, jnp.array([[1.0, 2.0], [3.0, 4.0]]))


def test_normalized_constants_in_complex_expressions():
    """Test that normalized constants work correctly in complex expressions that get lowered"""

    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.5, 1.0])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (2,))
    control._slice = slice(0, 2)

    # Use constants that were created with extra dimensions and got squeezed
    scalar_const = Constant(np.array([[2.0]]))  # (1,1) -> () after squeeze
    vector_const = Constant(np.array([[1.0, 1.0, 1.0]]))  # (1,3) -> (3,) after squeeze

    # Verify normalization happened
    assert scalar_const.value.shape == ()
    assert vector_const.value.shape == (3,)

    # Create expression: (x + vector_const) * scalar_const
    from openscvx.symbolic.expr import Mul

    expr = Mul(Add(state, vector_const), scalar_const)

    jl = JaxLowerer()
    fn = jl.lower(expr)
    result = fn(x, u, None, None)

    # Expected: ([1,2,3] + [1,1,1]) * 2 = [2,3,4] * 2 = [4,6,8]
    expected = jnp.array([4.0, 6.0, 8.0])
    assert jnp.allclose(result, expected)


def test_normalized_constants_preserve_dtype_in_lowering():
    """Test that JAX lowering preserves dtypes from normalized constants"""

    jl = JaxLowerer()

    # Test different dtypes with normalization
    int32_const = Constant(np.array([[42]], dtype=np.int32))  # (1,1) -> (), dtype preserved
    float32_const = Constant(np.array([[3.14]], dtype=np.float32))  # (1,1) -> (), dtype preserved

    # Verify normalization and dtype preservation
    assert int32_const.value.shape == ()
    assert float32_const.value.shape == ()
    assert int32_const.value.dtype == np.int32
    assert float32_const.value.dtype == np.float32

    # Test lowering
    fn_int = jl.lower(int32_const)
    fn_float = jl.lower(float32_const)

    result_int = fn_int(None, None, None, None)
    result_float = fn_float(None, None, None, None)

    # JAX should preserve the dtypes
    assert result_int.dtype == jnp.int32
    assert result_float.dtype == jnp.float32
    assert result_int == 42
    assert jnp.allclose(result_float, 3.14)


# =============================================================================
# Implicit Conversion Equivalence Tests (Cross-Cutting Behavior)
# =============================================================================


def test_constant_vs_implicit_conversion_equivalence():
    """Test that expressions with explicit Constant() and implicit conversion via to_expr produce
    identical results.
    """
    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.5, 1.0])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (2,))
    control._slice = slice(0, 2)

    # Test various expression types with both explicit and implicit constants
    from openscvx.symbolic.expr import Inequality, Mul

    # 1. Arithmetic operations
    scalar_value = 2.5
    vector_value = np.array([1.0, 1.0, 1.0])

    # Explicit vs implicit multiplication
    expr_explicit_mul = Mul(state, Constant(scalar_value))
    expr_implicit_mul = state * scalar_value

    fn_explicit_mul = lower_to_jax(expr_explicit_mul)
    fn_implicit_mul = lower_to_jax(expr_implicit_mul)

    result_explicit_mul = fn_explicit_mul(x, u, None, None)
    result_implicit_mul = fn_implicit_mul(x, u, None, None)

    assert jnp.allclose(result_explicit_mul, result_implicit_mul)
    assert jnp.allclose(result_explicit_mul, x * scalar_value)

    # Explicit vs implicit addition with vector
    expr_explicit_add = Add(state, Constant(vector_value))
    expr_implicit_add = state + vector_value

    fn_explicit_add = lower_to_jax(expr_explicit_add)
    fn_implicit_add = lower_to_jax(expr_implicit_add)

    result_explicit_add = fn_explicit_add(x, u, None, None)
    result_implicit_add = fn_implicit_add(x, u, None, None)

    assert jnp.allclose(result_explicit_add, result_implicit_add)
    assert jnp.allclose(result_explicit_add, x + vector_value)

    # 2. Matrix operations
    matrix_value = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])

    # Explicit vs implicit matrix multiplication
    expr_explicit_matmul = MatMul(Constant(matrix_value), state)
    expr_implicit_matmul = matrix_value @ state

    fn_explicit_matmul = lower_to_jax(expr_explicit_matmul)
    fn_implicit_matmul = lower_to_jax(expr_implicit_matmul)

    result_explicit_matmul = fn_explicit_matmul(x, u, None, None)
    result_implicit_matmul = fn_implicit_matmul(x, u, None, None)

    assert jnp.allclose(result_explicit_matmul, result_implicit_matmul)
    assert jnp.allclose(result_explicit_matmul, matrix_value @ x)

    # 3. Comparison operations (constraints)
    bounds_value = np.array([0.5, 1.5, 2.5])

    # Explicit vs implicit inequality
    constraint_explicit = Inequality(Constant(bounds_value), state)
    constraint_implicit = bounds_value <= state

    fn_constraint_explicit = lower_to_jax(constraint_explicit)
    fn_constraint_implicit = lower_to_jax(constraint_implicit)

    result_constraint_explicit = fn_constraint_explicit(x, u, None, None)
    result_constraint_implicit = fn_constraint_implicit(x, u, None, None)

    assert jnp.allclose(result_constraint_explicit, result_constraint_implicit)
    assert jnp.allclose(result_constraint_explicit, bounds_value - x)


def test_complex_expression_constant_equivalence():
    """Test equivalence in complex expressions that mix explicit and implicit constants."""
    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.5])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (1,))
    control._slice = slice(0, 1)

    # Complex dynamics-like expression with both explicit and implicit constants
    from openscvx.symbolic.expr import Div

    m = 2.0
    g = np.array([0.0, 0.0, -9.81])
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Version 1: Fully explicit constants
    expr_explicit = Add(Div(MatMul(Constant(A), state), Constant(m)), Constant(g))

    # Version 2: Mixed explicit/implicit (what users might write)
    expr_mixed = (A @ state) / m + g

    # Version 3: All implicit (most natural)
    expr_implicit = A @ state / m + g

    # All should produce identical results
    fn_explicit = lower_to_jax(expr_explicit)
    fn_mixed = lower_to_jax(expr_mixed)
    fn_implicit = lower_to_jax(expr_implicit)

    result_explicit = fn_explicit(x, u, None, None)
    result_mixed = fn_mixed(x, u, None, None)
    result_implicit = fn_implicit(x, u, None, None)

    # All versions should be identical
    assert jnp.allclose(result_explicit, result_mixed)
    assert jnp.allclose(result_mixed, result_implicit)
    assert jnp.allclose(result_explicit, result_implicit)

    # And match the expected mathematical result
    expected = A @ x / m + g
    assert jnp.allclose(result_explicit, expected)
    assert jnp.allclose(result_mixed, expected)
    assert jnp.allclose(result_implicit, expected)


# =============================================================================
# Cross-Node Constraint Lowering Tests
# =============================================================================


def test_contains_node_reference():
    """Test detection of NodeReference in expressions."""
    from openscvx.symbolic.lower import _contains_node_reference as contains_node_reference

    position = State("pos", shape=(3,))

    # Regular expression - no NodeReference
    regular_expr = position + 1.0
    assert not contains_node_reference(regular_expr)

    # With NodeReference
    cross_node_expr = position.at(5) - position.at(4)
    assert contains_node_reference(cross_node_expr)

    # Deeply nested
    nested_expr = (position.at(5) - position.at(4)) * 2.0 + 1.0
    assert contains_node_reference(nested_expr)


def test_absolute_node_reference_semantics():
    """Test that absolute indexing references the correct trajectory nodes.

    Tests that CrossNodeConstraint lowering produces a function with
    trajectory-level signature (X, U, params) -> result.
    """
    from openscvx.symbolic.expr import CrossNodeConstraint
    from openscvx.symbolic.lower import lower_to_jax

    position = State("pos", shape=(2,))
    position._slice = slice(0, 2)  # Manually assign slice for testing

    # Expression: position[5] - position[3]
    expr = position.at(5) - position.at(3)

    # Wrap in CrossNodeConstraint and lower - visitor handles wrapping
    cross_node = CrossNodeConstraint(expr <= 0)  # Need a constraint for CrossNodeConstraint
    constraint_fn = lower_to_jax(cross_node)

    # Create fake trajectory
    X = jnp.arange(20).reshape(10, 2).astype(float)  # 10 nodes, 2-dim state
    U = jnp.zeros((10, 0))
    params = {}

    # Evaluate - CrossNodeConstraint visitor provides (X, U, params) signature
    results = constraint_fn(X, U, params)

    # Expected: X[5] - X[3] - 0 = [10, 11] - [6, 7] = [4, 4]
    # Inequality lowers to lhs - rhs, so result is position[5] - position[3] - 0
    # Shape is (n_x,) = (2,) since constraint evaluates once
    expected = jnp.array([4.0, 4.0])

    assert results.shape == (2,)  # 2-dim state (single evaluation)
    assert jnp.allclose(results, expected)
