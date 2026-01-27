"""Tests for arithmetic operation nodes.

This module tests arithmetic operation nodes: Add, Sub, Mul, Div, MatMul, Neg, Power.
Tests are organized by node/node-group, with each section covering:

- Node creation and tree structure
- Shape Checking
- Canonicalization patterns
- JAX lowering
- CVXPY lowering
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Add,
    Constant,
    Div,
    Inequality,
    MatMul,
    Mul,
    Neg,
    Power,
    Sub,
    Variable,
)

# =============================================================================
# Add & Sub
# =============================================================================


def test_add_sub_basic_nodes_and_children():
    """Test basic Add and Sub node creation, children, and repr."""
    a, b = Constant(2), Constant(3)
    add = a + b
    sub = a - b

    # types
    assert isinstance(add, Add)
    assert isinstance(sub, Sub)

    # children
    assert add.children() == [a, b]
    assert sub.children() == [a, b]

    # repr should nest correctly
    assert repr(add) == "(Const(2.0) + Const(3.0))"
    assert repr(sub) == "(Const(2.0) - Const(3.0))"


def test_add_accepts_many_terms():
    """Test that Add can accept multiple terms."""
    a, b, c, d = Constant(5), Constant(3), Constant(1), Constant(2)
    add = Add(a, b, c, d)

    assert add.children() == [a, b, c, d]
    assert repr(add) == "(Const(5.0) + Const(3.0) + Const(1.0) + Const(2.0))"


def test_add_requires_at_least_two_terms():
    """Test that Add requires at least two terms."""
    with pytest.raises(ValueError):
        Add(Constant(1))


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((2,), (2,)),  # vector + vector
        ((2, 2), (2, 2)),  # matrix + matrix
    ],
)
def test_add_elementwise_children_for_arrays(shape_a, shape_b):
    """Test that Add captures children correctly for arrays."""
    A = Constant(np.ones(shape_a))
    B = Constant(np.full(shape_b, 2.0))
    expr = A + B

    # children captured correctly
    left, right = expr.children()
    assert left is A and right is B

    # repr mentions both shapes
    rep = repr(expr)
    assert "Const" in rep


# --- Add & Sub: Shape Checking ---


def test_add_same_shape_passes():
    """Test that Add with same shapes passes."""
    a = Constant(np.zeros((2, 3)))
    b = Constant(np.ones((2, 3)))
    add = a + b
    shape = add.check_shape()
    assert shape == (2, 3)


def test_add_broadcasting_passes():
    """Test that Add with broadcastable shapes passes."""
    a = Constant(np.zeros((2, 3)))
    b = Constant(np.array(5.0))  # scalar broadcasts
    add = a + b
    shape = add.check_shape()
    assert shape == (2, 3)


def test_add_vector_broadcast_passes():
    """Test that Add broadcasts vector to matrix correctly."""
    a = Constant(np.zeros((2, 3)))
    b = Constant(np.ones((3,)))  # broadcasts to (2, 3)
    add = a + b
    shape = add.check_shape()
    assert shape == (2, 3)


def test_add_shape_mismatch_raises():
    """Test that Add with incompatible shapes raises."""
    a = Constant(np.zeros((2, 3)))
    b = Constant(np.ones((3, 2)))
    add = a + b
    with pytest.raises(ValueError):
        add.check_shape()


def test_sub_same_shape_passes():
    """Test that Sub with same shapes passes."""
    a = Constant(np.zeros((4,)))
    b = Constant(np.ones((4,)))
    sub = a - b
    shape = sub.check_shape()
    assert shape == (4,)


def test_sub_broadcasting_passes():
    """Test that Sub with broadcastable shapes passes."""
    a = Constant(np.zeros((2, 3)))
    b = Constant(np.array(1.0))  # scalar broadcasts
    sub = a - b
    shape = sub.check_shape()
    assert shape == (2, 3)


def test_sub_shape_mismatch_raises():
    """Test that Sub with incompatible shapes raises."""
    a = Constant(np.zeros((4,)))
    b = Constant(np.ones((5,)))
    sub = a - b
    with pytest.raises(ValueError):
        sub.check_shape()


# --- Add & Sub: Canonicalization ---


def test_add_flatten_and_fold():
    """Test that nested Add nodes flatten and constants fold during canonicalization."""
    a = Constant(1)
    b = Constant(2)
    c = Constant(3)
    nested = Add(Add(a, b), c, Constant(4))
    result = nested.canonicalize()
    # should be Add(1,2,3,4) then folded to Constant(10)
    assert isinstance(result, Constant)
    assert result.value == 10


def test_add_eliminate_zero_and_singleton():
    """Test that Add eliminates zero terms and reduces singleton to just the value."""
    x = Constant(5)
    zero = Constant(0)
    # 5 + 0 + 0 ⇒ 5
    expr = Add(zero, x, zero)
    result = expr.canonicalize()
    assert not isinstance(result, Add)
    assert isinstance(result, Constant)
    assert result.value == 5


def test_sub_constant_folding():
    """Test that Sub folds constants during canonicalization."""
    expr = Sub(Constant(10), Constant(4))
    result = expr.canonicalize()
    assert isinstance(result, Constant)
    assert result.value == 6


# --- Add & Sub: JAX Lowering ---


def test_add_jax_lowering():
    """Test JAX lowering for Add operation."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.arange(8.0)
    a = State("a", (3,))
    a._slice = slice(0, 3)
    b = State("b", (3,))
    b._slice = slice(3, 6)
    expr_add = Add(a, b)

    jl = JaxLowerer()
    f_res_add = jl.lower(expr_add)
    res_add = f_res_add(x, None, None, None)

    assert jnp.allclose(res_add, x[0:3] + x[3:6])


def test_sub_jax_lowering():
    """Test JAX lowering for Sub operation."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Control, State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.arange(8.0)
    u = jnp.arange(8.0) * 3.0
    a = State("a", (3,))
    a._slice = slice(0, 3)
    b = Control("b", (3,))
    b._slice = slice(0, 3)
    expr_sub = Sub(a, b)

    jl = JaxLowerer()
    f_res_sub = jl.lower(expr_sub)
    res_sub = f_res_sub(x, u, None, None)

    assert jnp.allclose(res_sub, x[0:3] - u[0:3])


# --- Add & Sub: CVXPY Lowering ---


def test_add_cvxpy_lowering():
    """Test CVXPY lowering for Add operation."""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    const = Constant(np.array(2.0))
    expr = Add(x, const)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


def test_sub_cvxpy_lowering():
    """Test CVXPY lowering for Sub operation."""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    const = Constant(np.array(1.0))
    expr = Sub(x, const)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


# =============================================================================
# Mul & Div
# =============================================================================


def test_mul_div_basic_nodes_and_children():
    """Test basic Mul and Div node creation, children, and repr."""
    a, b = Constant(2), Constant(3)
    mul = a * b
    div = a / b

    # types
    assert isinstance(mul, Mul)
    assert isinstance(div, Div)

    # children
    assert mul.children() == [a, b]
    assert div.children() == [a, b]

    # repr should nest correctly
    assert repr(mul) == "(Const(2.0) * Const(3.0))"
    assert repr(div) == "(Const(2.0) / Const(3.0))"


def test_mul_accepts_many_terms():
    """Test that Mul can accept multiple terms."""
    a, b, c, d = Constant(5), Constant(3), Constant(1), Constant(2)
    mul = Mul(a, b, c, d)

    assert mul.children() == [a, b, c, d]
    assert repr(mul) == "(Const(5.0) * Const(3.0) * Const(1.0) * Const(2.0))"


def test_mul_requires_at_least_two_terms():
    """Test that Mul requires at least two terms."""
    with pytest.raises(ValueError):
        Mul(Constant(2))


# --- Mul & Div: Shape Checking ---


def test_mul_same_shape_passes():
    """Test that Mul with same shapes passes."""
    a = Constant(np.zeros((2, 2)))
    b = Constant(np.ones((2, 2)))
    mul = a * b
    shape = mul.check_shape()
    assert shape == (2, 2)


def test_mul_broadcasting_passes():
    """Test that Mul with broadcastable shapes passes."""
    a = Constant(np.zeros((2, 3)))
    b = Constant(np.array(2.0))  # scalar broadcasts
    mul = a * b
    shape = mul.check_shape()
    assert shape == (2, 3)


def test_mul_vector_broadcast_passes():
    """Test that Mul broadcasts vector correctly."""
    a = Constant(np.zeros((2, 3)))
    b = Constant(np.ones((3,)))  # broadcasts to (2, 3)
    mul = a * b
    shape = mul.check_shape()
    assert shape == (2, 3)


def test_mul_shape_mismatch_raises():
    """Test that Mul with incompatible shapes raises."""
    a = Constant(np.zeros((2, 2)))
    b = Constant(np.ones((2, 3)))
    mul = a * b
    with pytest.raises(ValueError):
        mul.check_shape()


def test_div_array_by_scalar_passes():
    """Test that Div with array divided by scalar passes."""
    a = Constant(np.zeros((3,)))
    b = Constant(np.array(2.0))
    div = a / b
    shape = div.check_shape()
    assert shape == (3,)


def test_div_broadcasting_passes():
    """Test that Div with broadcastable shapes passes."""
    a = Constant(np.zeros((2, 3)))
    b = Constant(np.ones((3,)))  # broadcasts
    div = a / b
    shape = div.check_shape()
    assert shape == (2, 3)


def test_div_shape_mismatch_raises():
    """Test that Div with incompatible shapes raises."""
    a = Constant(np.zeros((3,)))
    b = Constant(np.zeros((2,)))
    div = a / b
    with pytest.raises(ValueError):
        div.check_shape()


# --- Mul & Div: Canonicalization ---


def test_mul_flatten_and_fold():
    """Test that nested Mul nodes flatten and constants fold during canonicalization."""
    a = Constant(2)
    b = Constant(3)
    c = Constant(4)
    nested = Mul(Mul(a, b), c, Constant(5))
    result = nested.canonicalize()
    # 2*3*4*5 = 120
    assert isinstance(result, Constant)
    assert result.value == 120


def test_mul_eliminate_one_and_singleton():
    """Test that Mul eliminates identity (1) terms and reduces singleton."""
    x = Constant(7)
    one = Constant(1)
    expr = Mul(one, x, one)
    result = expr.canonicalize()
    assert not isinstance(result, Mul)
    assert isinstance(result, Constant)
    assert result.value == 7


def test_div_constant_folding():
    """Test that Div folds constants during canonicalization."""
    expr = Div(Constant(20), Constant(5))
    result = expr.canonicalize()
    assert isinstance(result, Constant)
    assert result.value == 4


def test_mul_preserves_vector_structure_with_parameter():
    """Test that multiplying a Parameter by a vector preserves the vector structure.

    Regression test for bug where Parameter * vector was incorrectly reduced to
    Parameter * scalar during canonicalization.
    """
    from openscvx.symbolic.expr import Parameter

    # Create a parameter
    g = Parameter("g", value=3.7114)

    # Create a vector
    g_vec = np.array([0.0, 0.0, 1.0])

    # Multiply parameter by vector
    expr = g * g_vec

    # Canonicalize
    result = expr.canonicalize()

    # The result should still be a Mul expression
    assert isinstance(result, Mul)

    # It should contain the parameter and a Constant with the FULL vector
    has_param = False
    has_vector_const = False

    for factor in result.factors:
        if isinstance(factor, Parameter) and factor.name == "g":
            has_param = True
        if isinstance(factor, Constant):
            # The constant should be a vector [0, 0, 1], NOT a scalar 0
            assert factor.value.shape == (3,), (
                f"Expected vector shape (3,), got {factor.value.shape}"
            )
            assert np.allclose(factor.value, [0.0, 0.0, 1.0]), (
                f"Expected [0, 0, 1], got {factor.value}"
            )
            has_vector_const = True

    assert has_param, "Parameter 'g' should be in the canonicalized expression"
    assert has_vector_const, "Vector constant should be preserved in canonicalized expression"


def test_mul_vector_constant_folding():
    """Test that multiplying multiple vector constants correctly performs element-wise mult."""
    # Create vector constants
    vec1 = Constant(np.array([2.0, 3.0, 4.0]))
    vec2 = Constant(np.array([5.0, 6.0, 7.0]))

    # Multiply them
    expr = Mul(vec1, vec2)

    # Canonicalize
    result = expr.canonicalize()

    # Should fold to a single constant with element-wise product
    assert isinstance(result, Constant)
    expected = np.array([10.0, 18.0, 28.0])
    assert np.allclose(result.value, expected), f"Expected {expected}, got {result.value}"


def test_mul_matrix_constant_folding():
    """Test that multiplying matrix constants correctly performs element-wise multiplication."""
    # Create matrix constants
    mat1 = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    mat2 = Constant(np.array([[2.0, 3.0], [4.0, 5.0]]))

    # Multiply them
    expr = Mul(mat1, mat2)

    # Canonicalize
    result = expr.canonicalize()

    # Should fold to a single constant with element-wise product
    assert isinstance(result, Constant)
    expected = np.array([[2.0, 6.0], [12.0, 20.0]])
    assert np.allclose(result.value, expected), f"Expected {expected}, got {result.value}"


def test_mul_scalar_times_vector():
    """Test that multiplying a scalar constant by a vector constant works correctly."""
    scalar = Constant(2.0)
    vector = Constant(np.array([1.0, 2.0, 3.0]))

    # Multiply
    expr = Mul(scalar, vector)

    # Canonicalize
    result = expr.canonicalize()

    # Should fold to a single vector constant
    assert isinstance(result, Constant)
    expected = np.array([2.0, 4.0, 6.0])
    assert np.allclose(result.value, expected), f"Expected {expected}, got {result.value}"


def test_mul_multiple_constants_with_parameter():
    """Test that multiplying multiple constants with a parameter preserves structure correctly."""
    from openscvx.symbolic.expr import Parameter

    # This mimics the real-world case: g_vec * g where g_vec = [0, 0, 1] * g_scalar
    g = Parameter("g", value=3.7114)
    base_vec = np.array([0.0, 0.0, 1.0])
    scalar = 9.807

    # Create expression: Parameter * vector * scalar
    expr = Mul(g, Constant(base_vec), Constant(scalar))

    # Canonicalize
    result = expr.canonicalize()

    # Should have Parameter and a single folded constant (vector * scalar)
    assert isinstance(result, Mul)

    has_param = False
    has_const = False

    for factor in result.factors:
        if isinstance(factor, Parameter):
            has_param = True
        if isinstance(factor, Constant):
            has_const = True
            # The two constants should be multiplied: [0, 0, 1] * 9.807 = [0, 0, 9.807]
            expected = np.array([0.0, 0.0, 9.807])
            assert factor.value.shape == (3,), (
                f"Expected vector shape (3,), got {factor.value.shape}"
            )
            assert np.allclose(factor.value, expected), f"Expected {expected}, got {factor.value}"

    assert has_param, "Parameter should be preserved"
    assert has_const, "Folded constant should be present"


def test_mul_gravity_vector_case():
    """Regression test for the exact gravity vector bug case.

    This tests: g_vec = np.array([0, 0, 1]) * Parameter('g', value=3.7114)
    which should canonicalize to Parameter('g') * Const([0, 0, 1])
    NOT Parameter('g') * Const(0.0)
    """
    from openscvx.symbolic.expr import Parameter

    g = Parameter("g", value=3.7114)
    g_vec_array = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # This is how it appears in the dynamics
    g_vec = g_vec_array * g

    # Canonicalize
    g_vec_canon = g_vec.canonicalize()

    # Should be Mul(Parameter('g'), Const([0, 0, 1]))
    assert isinstance(g_vec_canon, Mul), "Should be a Mul expression"

    # Extract factors
    param_factor = None
    const_factor = None

    for factor in g_vec_canon.factors:
        if isinstance(factor, Parameter):
            param_factor = factor
        elif isinstance(factor, Constant):
            const_factor = factor

    assert param_factor is not None, "Should have a Parameter factor"
    assert param_factor.name == "g", "Parameter should be 'g'"

    assert const_factor is not None, "Should have a Constant factor"
    assert const_factor.value.shape == (3,), (
        f"Constant should be a 3D vector, got shape {const_factor.value.shape}"
    )
    assert np.allclose(const_factor.value, [0.0, 0.0, 1.0]), (
        f"Expected [0, 0, 1], got {const_factor.value}"
    )

    # The bug would have produced Const(0.0) because np.prod([0, 0, 1]) = 0
    # Verify this doesn't happen
    assert not (const_factor.value.shape == () and const_factor.value == 0.0), (
        "Bug detected: vector was reduced to scalar 0!"
    )


def test_mul_broadcasting_with_parameters():
    """Test that broadcasting works correctly with parameters and arrays of different shapes."""
    from openscvx.symbolic.expr import Parameter

    param = Parameter("alpha", value=2.0)

    # Scalar * vector
    vec = Constant(np.array([1.0, 2.0, 3.0]))
    expr1 = Mul(param, vec)
    result1 = expr1.canonicalize()
    assert isinstance(result1, Mul)

    # Scalar * matrix
    mat = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    expr2 = Mul(param, mat)
    result2 = expr2.canonicalize()
    assert isinstance(result2, Mul)

    # Both should preserve the parameter and the array structure
    for result in [result1, result2]:
        has_param = any(isinstance(f, Parameter) for f in result.factors)
        assert has_param, "Parameter should be preserved in canonicalized expression"


# --- Mul & Div: JAX Lowering ---


def test_mul_jax_lowering():
    """Test JAX lowering for Mul operation."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.arange(8.0)
    a = State("a", (3,))
    a._slice = slice(0, 3)
    b = State("b", (3,))
    b._slice = slice(3, 6)
    expr_mul = Mul(a, b)

    jl = JaxLowerer()
    f_res_mul = jl.lower(expr_mul)
    res_mul = f_res_mul(x, None, None, None)

    assert jnp.allclose(res_mul, x[0:3] * x[3:6])


def test_div_jax_lowering():
    """Test JAX lowering for Div operation."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.arange(8.0)
    a = State("a", (3,))
    a._slice = slice(0, 3)
    c = Constant(2.0)
    expr_div = Div(a, c)

    jl = JaxLowerer()
    f_res_div = jl.lower(expr_div)
    res_div = f_res_div(x, None, None, None)

    assert jnp.allclose(res_div, x[0:3] / c.value)


# --- Mul & Div: CVXPY Lowering ---


def test_mul_cvxpy_lowering():
    """Test CVXPY lowering for Mul operation."""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    const = Constant(np.array(2.0))
    expr = Mul(x, const)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


def test_div_cvxpy_lowering():
    """Test CVXPY lowering for Div operation."""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    const = Constant(np.array(2.0))
    expr = Div(x, const)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


# =============================================================================
# Neg
# =============================================================================


def test_neg_basic_node_and_children():
    """Test basic Neg node creation, children, and repr."""
    a = Constant(2)
    neg = -a

    # type
    assert isinstance(neg, Neg)

    # children
    assert neg.children() == [a]

    # repr should nest correctly
    assert repr(neg) == "(-Const(2.0))"


# --- Neg: Shape Checking ---


def test_neg_preserves_shape():
    """Test that Neg preserves the shape of its operand."""
    a = Constant(np.zeros((2, 3)))
    neg = -a
    shape = neg.check_shape()
    assert shape == (2, 3)


def test_neg_scalar_shape():
    """Test that Neg preserves scalar shape."""
    a = Constant(5)
    neg = -a
    shape = neg.check_shape()
    assert shape == ()


# --- Neg: Canonicalization ---


def test_neg_constant_folding():
    """Test that Neg folds constants during canonicalization."""
    expr = Neg(Constant(8))
    result = expr.canonicalize()
    assert isinstance(result, Constant)
    assert result.value == -8


# --- Neg: JAX Lowering ---


def test_neg_jax_lowering():
    """Test JAX lowering for Neg operation with composite expression."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Control, State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.arange(6.0)
    u = jnp.arange(6.0) * 3
    a = State("a", (2,))
    a._slice = slice(0, 2)
    b = Control("b", (2,))
    b._slice = slice(0, 2)
    c = Constant(np.array([1.0, 1.0]))

    # expr = -((a + b) * c)
    expr = Neg(Mul(Add(a, b), c))
    jl = JaxLowerer()
    f = jl.lower(expr)
    out = f(x, u, None, None)

    expected = -((x[0:2] + u[0:2]) * jnp.array([1.0, 1.0]))
    assert jnp.allclose(out, expected)


# --- Neg: CVXPY Lowering ---


def test_neg_cvxpy_lowering():
    """Test CVXPY lowering for Neg operation."""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Neg(x)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


# =============================================================================
# Power
# =============================================================================

# --- Power: Basic Usage ---


def test_power_operator_and_node():
    """Test power operation using ** operator and Power node."""
    a, b = Constant(2), Constant(3)

    # Test ** operator
    pow1 = a**b
    assert isinstance(pow1, Power)
    assert pow1.children() == [a, b]
    assert repr(pow1) == "(Const(2.0))**(Const(3.0))"

    # Test direct Power node creation
    pow2 = Power(a, b)
    assert isinstance(pow2, Power)
    assert pow2.children() == [a, b]
    assert repr(pow2) == "(Const(2.0))**(Const(3.0))"


def test_power_with_mixed_types():
    """Test power operation with mixed numeric and expression types."""
    x = Variable("x", shape=(1,))

    # Expression ** numeric
    pow1 = x**2
    assert isinstance(pow1, Power)
    assert pow1.base is x
    assert isinstance(pow1.exponent, Constant)
    assert pow1.exponent.value == 2
    assert repr(pow1) == "(Var('x'))**(Const(2))"

    # Numeric ** expression (rpow)
    pow2 = 10**x
    assert isinstance(pow2, Power)
    assert isinstance(pow2.base, Constant)
    assert pow2.base.value == 10
    assert pow2.exponent is x
    assert repr(pow2) == "(Const(10))**(Var('x'))"


def test_power_arrays():
    """Test power operation with arrays."""
    base = Constant(np.array([2.0, 3.0, 4.0]))
    exp = Constant(2)
    pow_expr = base**exp

    assert isinstance(pow_expr, Power)
    assert pow_expr.children() == [base, exp]


# --- Power: Shape Checking ---


def test_power_same_shape_passes():
    """Test that Power with same shapes passes."""
    a = Constant(np.ones((2, 3)))
    b = Constant(np.ones((2, 3)))
    pow_expr = a**b
    shape = pow_expr.check_shape()
    assert shape == (2, 3)


def test_power_broadcasting_passes():
    """Test that Power with broadcastable shapes passes."""
    a = Constant(np.ones((2, 3)))
    b = Constant(2.0)  # scalar broadcasts
    pow_expr = a**b
    shape = pow_expr.check_shape()
    assert shape == (2, 3)


def test_power_shape_mismatch_raises():
    """Test that Power with incompatible shapes raises."""
    a = Constant(np.ones((2, 3)))
    b = Constant(np.ones((3, 2)))
    pow_expr = a**b
    with pytest.raises(ValueError):
        pow_expr.check_shape()


# --- Power: Canonicalization ---


def test_power_constant_folding():
    """Test that Power folds constants during canonicalization."""
    expr = Power(Constant(2), Constant(3))
    result = expr.canonicalize()
    # Note: Current implementation doesn't fold power constants,
    # it just canonicalizes children
    assert isinstance(result, Power)
    assert isinstance(result.base, Constant)
    assert isinstance(result.exponent, Constant)


def test_power_canonicalize_children():
    """Test that Power canonicalizes its children."""
    # Create nested add that will canonicalize
    base = Add(Constant(2), Constant(3))  # will become Constant(5)
    exp = Add(Constant(1), Constant(1))  # will become Constant(2)
    pow_expr = Power(base, exp)

    result = pow_expr.canonicalize()
    assert isinstance(result, Power)
    assert isinstance(result.base, Constant)
    assert result.base.value == 5
    assert isinstance(result.exponent, Constant)
    assert result.exponent.value == 2


# =============================================================================
# MatMul
# =============================================================================

# --- MatMul: Basic Usage ---


def test_matmul_basic_node_creation():
    """Test basic MatMul node creation with @ operator."""
    a = Constant(np.eye(3))
    b = Constant(np.ones((3, 2)))
    mm = a @ b

    assert isinstance(mm, MatMul)
    assert mm.children() == [a, b]
    # repr uses * for MatMul (not @)
    assert "(" in repr(mm)
    assert "*" in repr(mm)


def test_matmul_vector_and_matrix():
    """Test MatMul with vector and matrix."""
    # 2×2 identity matrix × 2-vector
    M = Constant(np.eye(2))
    v = Constant(np.array([1.0, 2.0]))
    mm = M @ v

    assert isinstance(mm, MatMul)
    children = mm.children()
    assert children[0] is M and children[1] is v

    # repr should reflect operator
    assert "MatMul" in mm.pretty()  # tree form contains the node name
    assert "(" in repr(mm) and "@" not in repr(mm)  # repr is Python‐safe


def test_matmul_matrix_matrix():
    """Test MatMul with two matrices."""
    A = Constant(np.ones((3, 4)))
    B = Constant(np.ones((4, 2)))
    mm = A @ B

    assert isinstance(mm, MatMul)
    assert mm.children() == [A, B]


def test_matmul_vector_vector():
    """Test MatMul with two vectors (dot product)."""
    v1 = Constant(np.array([1.0, 2.0, 3.0]))
    v2 = Constant(np.array([4.0, 5.0, 6.0]))
    mm = v1 @ v2

    assert isinstance(mm, MatMul)
    assert mm.children() == [v1, v2]


# --- MatMul: Shape Checking ---


def test_matmul_matrix_matrix_shape():
    """Test MatMul shape checking for matrix @ matrix."""
    a = Constant(np.zeros((4, 5)))
    b = Constant(np.zeros((5, 2)))
    matmul = a @ b
    shape = matmul.check_shape()
    assert shape == (4, 2)


def test_matmul_matrix_vector_shape():
    """Test MatMul shape checking for matrix @ vector."""
    a = Constant(np.zeros((3, 4)))
    b = Constant(np.zeros((4,)))
    matmul = a @ b
    shape = matmul.check_shape()
    assert shape == (3,)


def test_matmul_vector_matrix_shape():
    """Test MatMul shape checking for vector @ matrix."""
    a = Constant(np.zeros((3,)))
    b = Constant(np.zeros((3, 5)))
    matmul = a @ b
    shape = matmul.check_shape()
    assert shape == (5,)


def test_matmul_vector_vector_shape():
    """Test MatMul shape checking for vector @ vector (dot product -> scalar)."""
    a = Constant(np.zeros((4,)))
    b = Constant(np.zeros((4,)))
    matmul = a @ b
    shape = matmul.check_shape()
    assert shape == ()  # scalar result


def test_matmul_incompatible_raises():
    """Test that MatMul with incompatible shapes raises."""
    a = Constant(np.zeros((4, 5)))
    b = Constant(np.zeros((4, 2)))  # inner dimensions don't match
    matmul = a @ b
    with pytest.raises(ValueError):
        matmul.check_shape()


def test_matmul_scalar_raises():
    """Test that MatMul with scalar operands raises."""
    a = Constant(5.0)
    b = Constant(3.0)
    matmul = a @ b
    with pytest.raises(ValueError):
        matmul.check_shape()


# --- MatMul: Canonicalization ---


def test_matmul_canonicalize_children():
    """Test that MatMul canonicalizes its children."""
    # Create expressions that will canonicalize
    left = Add(Constant(np.ones((2, 3))), Constant(np.zeros((2, 3))))  # -> ones
    right = Add(Constant(np.ones((3, 2))), Constant(np.zeros((3, 2))))  # -> ones
    mm = MatMul(left, right)

    result = mm.canonicalize()
    assert isinstance(result, MatMul)
    # Children should be canonicalized
    assert isinstance(result.left, Constant)
    assert isinstance(result.right, Constant)


def test_matmul_preserves_structure():
    """Test that MatMul doesn't fold constants (just canonicalizes children)."""
    # MatMul doesn't perform constant folding at canonicalization
    a = Constant(np.eye(2))
    b = Constant(np.array([1.0, 2.0]))
    mm = MatMul(a, b)

    result = mm.canonicalize()
    assert isinstance(result, MatMul)
    assert isinstance(result.left, Constant)
    assert isinstance(result.right, Constant)


# --- MatMul: JAX Lowering ---


def test_matmul_jax_lowering():
    """Test JAX lowering for MatMul operation."""
    import jax.numpy as jnp

    from openscvx.symbolic.lowerers.jax import JaxLowerer

    # (2×2 matrix) @ (2-vector)
    M = Constant(np.array([[1.0, 0.0], [0.0, 2.0]]))
    v = Constant(np.array([3.0, 4.0]))
    expr = MatMul(M, v)

    jl = JaxLowerer()
    f = jl.lower(expr)
    out = f(None, None, None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2,)
    assert jnp.allclose(out, jnp.array([3.0, 8.0]))


# --- MatMul: CVXPY Lowering ---


def test_matmul_cvxpy_lowering():
    """Test CVXPY lowering for MatMul operation."""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable(3, name="x")  # Single vector, not time series
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    A = Constant(np.eye(3))
    expr = MatMul(A, x)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


# =============================================================================
# Integration Tests
# =============================================================================


def test_combined_ops_produce_correct_constraint_tree():
    """Test that combined operations produce correct constraint tree structure."""
    # (x + y) @ z >= 5
    x = Variable("x", (3,))
    y = Variable("y", (3,))
    z = Variable("z", (3,))

    # note: MatMul between two 3-vectors is allowed at AST level
    expr = (x + y) @ z <= 5
    # root is Constraint
    assert isinstance(expr, Inequality)
    # check tree structure via pretty()
    p = expr.pretty().splitlines()
    assert p[0].strip().startswith("Inequality")
    # next line is MatMul
    assert "MatMul" in p[1]

    # children of the constraint:
    assert isinstance(expr.lhs, MatMul)
    assert isinstance(expr.rhs, Constant)
