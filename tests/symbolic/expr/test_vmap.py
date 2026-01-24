"""Tests for Vmap and Placeholder expression nodes.

This module tests the Vmap vectorized map operation and its associated Placeholder
node for data-parallel operations within the symbolic expression framework.

Tests are organized by node/node-group, with each section containing:

1. Node creation and properties
2. Shape Checking
3. Canonicalization
4. JAX lowering tests

Note: CVXPy lowering tests are not included as Vmap is JAX-specific
(vmap has no direct CVXPy equivalent).
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Constant,
    Control,
    Norm,
    Parameter,
    State,
    Variable,
    Vmap,
    _Placeholder,
)

# =============================================================================
# Placeholder
# =============================================================================


def test_placeholder_creation_and_uniqueness():
    """Test Placeholder creation, shapes, and unique name generation."""
    # Basic creation with various shapes
    p_scalar = _Placeholder(shape=())
    p_vector = _Placeholder(shape=(3,))
    p_matrix = _Placeholder(shape=(3, 4))

    assert p_scalar.shape == ()
    assert p_vector.shape == (3,)
    assert p_matrix.shape == (3, 4)

    # All should have auto-generated unique names
    assert p_scalar.name.startswith("_vmap_placeholder_")
    assert p_vector.name.startswith("_vmap_placeholder_")
    assert len({p_scalar.name, p_vector.name, p_matrix.name}) == 3  # All unique


# =============================================================================
# Vmap
# =============================================================================

# --- Vmap: Creation & Tree Structure ---


def test_vmap_creation_with_numpy_array():
    """Test Vmap node creation with numpy array (baked-in)."""
    x = Variable("x", shape=(3,))
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)

    vmap_expr = Vmap(lambda p: Norm(x - p), batch=data)

    assert isinstance(vmap_expr, Vmap)
    assert vmap_expr.axis == 0
    assert vmap_expr.is_parameter == (False,)
    assert vmap_expr.num_batches == 1
    assert isinstance(vmap_expr.batch, Constant)
    assert isinstance(vmap_expr.placeholder, _Placeholder)
    assert vmap_expr.placeholder.shape == (3,)  # Per-element shape


def test_vmap_creation_with_constant():
    """Test Vmap node creation with explicit Constant."""
    x = Variable("x", shape=(3,))
    data = Constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    vmap_expr = Vmap(lambda p: Norm(x - p), batch=data)

    assert isinstance(vmap_expr, Vmap)
    assert vmap_expr.is_parameter == (False,)
    assert vmap_expr.batch is data


def test_vmap_creation_with_parameter():
    """Test Vmap node creation with Parameter (runtime lookup)."""
    x = Variable("x", shape=(3,))
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    refs = Parameter("refs", shape=(2, 3), value=data)

    vmap_expr = Vmap(lambda p: Norm(x - p), batch=refs)

    assert isinstance(vmap_expr, Vmap)
    assert vmap_expr.is_parameter == (True,)
    assert vmap_expr.batch is refs


def test_vmap_children_constant():
    """Test Vmap children() for Constant (baked-in) case."""
    x = Variable("x", shape=(3,))
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    vmap_expr = Vmap(lambda p: Norm(x - p), batch=data)
    children = vmap_expr.children()

    # Constant case: only the inner expression is a child
    assert len(children) == 1
    assert isinstance(children[0], Norm)


def test_vmap_children_parameter():
    """Test Vmap children() for Parameter (runtime lookup) case."""
    x = Variable("x", shape=(3,))
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    refs = Parameter("refs", shape=(2, 3), value=data)

    vmap_expr = Vmap(lambda p: Norm(x - p), batch=refs)
    children = vmap_expr.children()

    # Parameter case: inner expression AND Parameter are children
    # (so traverse() finds the Parameter for preprocessing)
    assert len(children) == 2
    assert isinstance(children[0], Norm)
    assert children[1] is refs


def test_vmap_repr():
    """Test Vmap repr for Constant and Parameter cases."""
    x = Variable("x", shape=(3,))
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Constant case
    vmap_const = Vmap(lambda p: Norm(x - p), batch=data)
    assert "Vmap" in repr(vmap_const)
    assert "Constant" in repr(vmap_const)
    assert "(2, 3)" in repr(vmap_const)

    # Parameter case
    refs = Parameter("refs", shape=(2, 3), value=data)
    vmap_param = Vmap(lambda p: Norm(x - p), batch=refs)
    assert "Vmap" in repr(vmap_param)
    assert "Parameter" in repr(vmap_param)
    assert "refs" in repr(vmap_param)


# --- Vmap: Shape Checking ---


def test_vmap_shape_vector_to_scalar():
    """Test Vmap shape: vmapping over vectors, producing scalars."""
    x = Variable("x", shape=(3,))
    data = np.random.randn(10, 3)  # 10 vectors of size 3

    vmap_expr = Vmap(lambda p: Norm(x - p), batch=data)

    # Each Norm produces a scalar (), vmap over 10 elements -> (10,)
    assert vmap_expr.check_shape() == (10,)


def test_vmap_shape_vector_to_vector():
    """Test Vmap shape: vmapping over vectors, producing vectors."""
    x = Variable("x", shape=(3,))
    data = np.random.randn(5, 3)  # 5 vectors of size 3

    # Return a vector difference instead of norm
    vmap_expr = Vmap(lambda p: x - p, batch=data)

    # Each x - p produces (3,), vmap over 5 elements -> (5, 3)
    assert vmap_expr.check_shape() == (5, 3)


def test_vmap_shape_with_parameter():
    """Test Vmap shape with Parameter source."""
    x = Variable("x", shape=(3,))
    data = np.random.randn(7, 3)
    refs = Parameter("refs", shape=(7, 3), value=data)

    vmap_expr = Vmap(lambda p: Norm(x - p), batch=refs)

    assert vmap_expr.check_shape() == (7,)


def test_vmap_shape_scalar_per_element():
    """Test Vmap shape when per-element is a scalar."""
    x = Variable("x", shape=())
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 scalars

    vmap_expr = Vmap(lambda p: x - p, batch=data)

    # Per-element is scalar, result is scalar, vmap over 5 -> (5,)
    assert vmap_expr.check_shape() == (5,)


def test_vmap_axis_out_of_bounds():
    """Test that Vmap raises error for invalid axis."""
    x = Variable("x", shape=(3,))
    data = np.random.randn(10, 3)  # shape (10, 3), valid axes: 0, 1

    with pytest.raises(ValueError, match="axis.*out of bounds"):
        Vmap(lambda p: Norm(x - p), batch=data, axis=5)


def test_vmap_axis_negative_invalid():
    """Test that Vmap raises error for negative axis."""
    x = Variable("x", shape=(3,))
    data = np.random.randn(10, 3)

    with pytest.raises(ValueError, match="axis.*out of bounds"):
        Vmap(lambda p: Norm(x - p), batch=data, axis=-1)


# --- Vmap: Canonicalization ---


def test_vmap_canonicalize():
    """Test that canonicalize recurses and preserves Vmap properties."""
    x = Variable("x", shape=(3,))
    data = np.random.randn(5, 3)
    refs = Parameter("refs", shape=(5, 3), value=data)

    vmap_expr = Vmap(lambda p: Norm((x + 0) - p), batch=refs, axis=0)

    canonical = vmap_expr.canonicalize()

    # Should still be a Vmap with preserved properties
    assert isinstance(canonical, Vmap)
    assert canonical.axis == 0
    assert canonical.is_parameter == (True,)
    assert canonical.placeholder is vmap_expr.placeholder


# --- Vmap: JAX Lowering (Constant/baked-in) ---


def test_vmap_jax_constant():
    """Test JAX lowering of Vmap with Constant data."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(3,))
    x._slice = slice(0, 3)

    refs = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ]
    )

    # Test scalar output (norms)
    vmap_norm = Vmap(lambda p: Norm(x - p, ord=2), batch=refs)
    fn_norm = lower_to_jax(vmap_norm)

    x_val = jnp.array([0.0, 0.0, 0.0])
    result = fn_norm(x_val, None, None, {})
    expected = jnp.array([0.0, 5.0])  # 0 to origin, 5 to (3,4,0)
    assert jnp.allclose(result, expected, atol=1e-12)
    assert result.shape == (2,)

    # Test vector output (differences)
    vmap_diff = Vmap(lambda p: x - p, batch=refs)
    fn_diff = lower_to_jax(vmap_diff)

    x_val = jnp.array([1.0, 1.0, 1.0])
    result = fn_diff(x_val, None, None, {})
    expected = jnp.array([[1.0, 1.0, 1.0], [-2.0, -3.0, 1.0]])
    assert jnp.allclose(result, expected, atol=1e-12)
    assert result.shape == (2, 3)


def test_vmap_jax_axis_nonzero():
    """Test JAX lowering of Vmap with axis != 0."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(2,))
    x._slice = slice(0, 2)

    # Data shape (2, 3): axis=0 gives 2 elements of shape (3,),
    # axis=1 gives 3 elements of shape (2,)
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    # Vmap over axis=1: iterate over 3 columns, each of shape (2,)
    vmap_expr = Vmap(lambda col: Norm(x - col, ord=2), batch=data, axis=1)

    assert vmap_expr.check_shape() == (3,)  # 3 columns
    assert vmap_expr.placeholder.shape == (2,)  # Each column is (2,)

    fn = lower_to_jax(vmap_expr)

    x_val = jnp.array([1.0, 4.0])  # Matches first column exactly
    result = fn(x_val, None, None, {})

    # Distances to each column: [1,4], [2,5], [3,6]
    # dist to col0 = |[1,4] - [1,4]| = 0
    # dist to col1 = |[1,4] - [2,5]| = |[-1,-1]| = sqrt(2)
    # dist to col2 = |[1,4] - [3,6]| = |[-2,-2]| = sqrt(8)
    expected = jnp.array([0.0, jnp.sqrt(2.0), jnp.sqrt(8.0)])
    assert jnp.allclose(result, expected, atol=1e-12)
    assert result.shape == (3,)


# --- Vmap: JAX Lowering (Parameter/runtime lookup) ---


def test_vmap_jax_parameter():
    """Test JAX lowering of Vmap with Parameter data and runtime updates."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(3,))
    x._slice = slice(0, 3)

    refs_data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    refs = Parameter("refs", shape=(2, 3), value=refs_data)

    vmap_expr = Vmap(lambda p: Norm(x - p, ord=2), batch=refs)
    fn = lower_to_jax(vmap_expr)

    x_val = jnp.array([0.0, 0.0, 0.0])

    # First call with original data
    params1 = {"refs": jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])}
    result1 = fn(x_val, None, None, params1)
    assert jnp.allclose(result1, jnp.array([1.0, 1.0]), atol=1e-12)

    # Second call with DIFFERENT runtime data (same compiled function!)
    params2 = {"refs": jnp.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])}
    result2 = fn(x_val, None, None, params2)
    assert jnp.allclose(result2, jnp.array([2.0, 2.0]), atol=1e-12)


# --- Vmap: Constant vs Parameter Equivalence ---


def test_vmap_constant_vs_parameter_same_values():
    """Test that Constant and Parameter produce same results with identical data.

    This is the key test validating that the symbolic layer faithfully
    reproduces BYOF closure-captured behavior.
    """
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(3,))
    x._slice = slice(0, 3)

    # Same underlying data
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    # Constant version (baked-in)
    vmap_const = Vmap(lambda p: Norm(x - p, ord=2), batch=data)
    fn_const = lower_to_jax(vmap_const)

    # Parameter version (runtime lookup)
    refs = Parameter("refs", shape=(3, 3), value=data)
    vmap_param = Vmap(lambda p: Norm(x - p, ord=2), batch=refs)
    fn_param = lower_to_jax(vmap_param)

    # Test with several positions
    test_positions = [
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array([5.0, 5.0, 5.0]),
    ]

    params = {"refs": jnp.array(data)}

    for x_val in test_positions:
        result_const = fn_const(x_val, None, None, {})
        result_param = fn_param(x_val, None, None, params)

        assert jnp.allclose(result_const, result_param, atol=1e-12), (
            f"Constant and Parameter versions differ for x={x_val}"
        )


# --- Vmap: Hash ---


def test_vmap_hash():
    """Test structural hashing behavior for Vmap and Placeholder."""
    x = Variable("x", shape=(3,))
    x._slice = slice(0, 3)  # Required for hashing
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    vmap_expr = Vmap(lambda p: Norm(x - p), batch=data)

    # Same instance is deterministic
    assert vmap_expr.structural_hash() == vmap_expr.structural_hash()

    # Different instances have different hashes (unique Placeholders)
    vmap2 = Vmap(lambda p: Norm(x - p), batch=data.copy())
    assert vmap_expr.structural_hash() != vmap2.structural_hash()

    # Placeholders also have unique hashes
    p1, p2 = _Placeholder(shape=(3,)), _Placeholder(shape=(3,))
    assert p1.structural_hash() != p2.structural_hash()


# =============================================================================
# Vmap: Multi-Batch Support
# =============================================================================


def test_vmap_multi_batch_creation():
    """Test Vmap creation with multiple batch arguments."""
    x = Variable("x", shape=(3,))
    centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    radii = np.array([0.5, 0.7, 0.9])

    vmap_expr = Vmap(
        lambda c, r: Norm(x - c) - r,
        batch=[centers, radii],
    )

    assert vmap_expr.num_batches == 2
    assert vmap_expr.is_parameter == (False, False)
    assert len(vmap_expr.placeholders) == 2
    assert vmap_expr.placeholders[0].shape == (3,)  # center shape
    assert vmap_expr.placeholders[1].shape == ()  # radius shape (scalar)


def test_vmap_multi_batch_with_parameters():
    """Test Vmap with multiple Parameter batches."""
    x = Variable("x", shape=(3,))
    centers_data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    radii_data = np.array([0.5, 0.7])

    centers = Parameter("centers", shape=(2, 3), value=centers_data)
    radii = Parameter("radii", shape=(2,), value=radii_data)

    vmap_expr = Vmap(
        lambda c, r: Norm(x - c) - r,
        batch=[centers, radii],
    )

    assert vmap_expr.num_batches == 2
    assert vmap_expr.is_parameter == (True, True)
    assert vmap_expr.batches[0] is centers
    assert vmap_expr.batches[1] is radii


def test_vmap_multi_batch_mixed_types():
    """Test Vmap with mixed Constant and Parameter batches."""
    x = Variable("x", shape=(3,))
    centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Will become Constant
    radii = Parameter("radii", shape=(2,), value=np.array([0.5, 0.7]))

    vmap_expr = Vmap(
        lambda c, r: Norm(x - c) - r,
        batch=[centers, radii],
    )

    assert vmap_expr.num_batches == 2
    assert vmap_expr.is_parameter == (False, True)
    assert isinstance(vmap_expr.batches[0], Constant)
    assert vmap_expr.batches[1] is radii


def test_vmap_multi_batch_shape():
    """Test shape checking for multi-batch Vmap."""
    x = Variable("x", shape=(3,))
    centers = np.random.randn(10, 3)
    radii = np.random.randn(10)

    vmap_expr = Vmap(
        lambda c, r: Norm(x - c) - r,
        batch=[centers, radii],
    )

    # Inner expression produces scalar, vmap over 10 -> (10,)
    assert vmap_expr.check_shape() == (10,)


def test_vmap_multi_batch_size_mismatch():
    """Test that Vmap raises error when batch sizes don't match."""
    x = Variable("x", shape=(3,))
    centers = np.random.randn(10, 3)
    radii = np.random.randn(5)  # Different size!

    with pytest.raises(ValueError, match="Batch size mismatch"):
        Vmap(lambda c, r: Norm(x - c) - r, batch=[centers, radii])


def test_vmap_multi_batch_children():
    """Test children() for multi-batch Vmap with mixed types."""
    x = Variable("x", shape=(3,))
    centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Constant
    radii = Parameter("radii", shape=(2,), value=np.array([0.5, 0.7]))

    vmap_expr = Vmap(
        lambda c, r: Norm(x - c) - r,
        batch=[centers, radii],
    )

    children = vmap_expr.children()

    # Should include inner expr and Parameter (but not Constant)
    assert len(children) == 2
    assert children[1] is radii


def test_vmap_multi_batch_repr():
    """Test repr for multi-batch Vmap."""
    x = Variable("x", shape=(3,))
    centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    radii = Parameter("radii", shape=(2,), value=np.array([0.5, 0.7]))

    vmap_expr = Vmap(
        lambda c, r: Norm(x - c) - r,
        batch=[centers, radii],
    )

    repr_str = repr(vmap_expr)
    assert "Vmap" in repr_str
    assert "Constant" in repr_str
    assert "Parameter" in repr_str
    assert "radii" in repr_str


# --- Multi-Batch JAX Lowering ---


def test_vmap_multi_batch_jax_constants():
    """Test JAX lowering of multi-batch Vmap with all Constants."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(3,))
    x._slice = slice(0, 3)

    centers = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    radii = np.array([0.0, 1.0])

    # distance - radius (obstacle avoidance style)
    vmap_expr = Vmap(
        lambda c, r: Norm(x - c, ord=2) - r,
        batch=[centers, radii],
    )

    fn = lower_to_jax(vmap_expr)

    x_val = jnp.array([0.0, 0.0, 0.0])
    result = fn(x_val, None, None, {})

    # At origin: dist to [0,0,0] is 0, minus 0 = 0
    #            dist to [3,4,0] is 5, minus 1 = 4
    expected = jnp.array([0.0, 4.0])
    assert jnp.allclose(result, expected, atol=1e-12)


def test_vmap_multi_batch_jax_parameters():
    """Test JAX lowering of multi-batch Vmap with all Parameters."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(3,))
    x._slice = slice(0, 3)

    centers_data = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    radii_data = np.array([0.0, 1.0])

    centers = Parameter("centers", shape=(2, 3), value=centers_data)
    radii = Parameter("radii", shape=(2,), value=radii_data)

    vmap_expr = Vmap(
        lambda c, r: Norm(x - c, ord=2) - r,
        batch=[centers, radii],
    )

    fn = lower_to_jax(vmap_expr)

    x_val = jnp.array([0.0, 0.0, 0.0])
    params = {
        "centers": jnp.array(centers_data),
        "radii": jnp.array(radii_data),
    }
    result = fn(x_val, None, None, params)

    expected = jnp.array([0.0, 4.0])
    assert jnp.allclose(result, expected, atol=1e-12)

    # Test runtime update
    params2 = {
        "centers": jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "radii": jnp.array([0.5, 0.5]),
    }
    result2 = fn(x_val, None, None, params2)

    # dist to [1,0,0] is 1, minus 0.5 = 0.5
    # dist to [0,1,0] is 1, minus 0.5 = 0.5
    expected2 = jnp.array([0.5, 0.5])
    assert jnp.allclose(result2, expected2, atol=1e-12)


def test_vmap_multi_batch_jax_mixed():
    """Test JAX lowering of multi-batch Vmap with mixed Constant/Parameter."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(3,))
    x._slice = slice(0, 3)

    # Centers are baked in (Constant), radii are runtime (Parameter)
    centers = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    radii = Parameter("radii", shape=(2,), value=np.array([0.0, 1.0]))

    vmap_expr = Vmap(
        lambda c, r: Norm(x - c, ord=2) - r,
        batch=[centers, radii],
    )

    fn = lower_to_jax(vmap_expr)

    x_val = jnp.array([0.0, 0.0, 0.0])

    # First call
    params1 = {"radii": jnp.array([0.0, 1.0])}
    result1 = fn(x_val, None, None, params1)
    expected1 = jnp.array([0.0, 4.0])
    assert jnp.allclose(result1, expected1, atol=1e-12)

    # Update radii at runtime (centers stay baked)
    params2 = {"radii": jnp.array([1.0, 2.0])}
    result2 = fn(x_val, None, None, params2)
    # dist to [0,0,0] is 0, minus 1 = -1
    # dist to [3,4,0] is 5, minus 2 = 3
    expected2 = jnp.array([-1.0, 3.0])
    assert jnp.allclose(result2, expected2, atol=1e-12)


# =============================================================================
# Vmap: State/Control Batching Support
# =============================================================================


def test_vmap_state_batch():
    """Test Vmap with State batch source: creation, properties, and canonicalization."""
    positions = State("positions", shape=(5, 3))

    vmap_expr = Vmap(lambda pos: Norm(pos), batch=positions)

    # Creation and properties
    assert vmap_expr.num_batches == 1
    assert vmap_expr.is_state == (True,)
    assert vmap_expr.is_parameter == (False,)
    assert vmap_expr.is_control == (False,)
    assert vmap_expr.batch is positions
    assert vmap_expr.placeholder.shape == (3,)
    assert vmap_expr.check_shape() == (5,)

    # Children includes State
    children = vmap_expr.children()
    assert len(children) == 2
    assert children[1] is positions

    # Repr
    assert "State" in repr(vmap_expr) and "positions" in repr(vmap_expr)

    # Canonicalization preserves state info
    canonical = vmap_expr.canonicalize()
    assert canonical.is_state == (True,)
    assert canonical.batch is positions


def test_vmap_control_batch():
    """Test Vmap with Control batch source: creation and properties."""
    thrusters = Control("thrusters", shape=(4,))

    vmap_expr = Vmap(lambda t: t * 2, batch=thrusters)

    assert vmap_expr.num_batches == 1
    assert vmap_expr.is_control == (True,)
    assert vmap_expr.is_state == (False,)
    assert vmap_expr.batch is thrusters
    assert vmap_expr.placeholder.shape == ()
    assert "Control" in repr(vmap_expr) and "thrusters" in repr(vmap_expr)


def test_vmap_state_control_jax_lowering():
    """Test JAX lowering with State, Control, and mixed batches."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    # --- State batch ---
    positions = State("positions", shape=(3, 2))
    positions._slice = slice(0, 6)

    vmap_state = Vmap(lambda pos: Norm(pos, ord=2), batch=positions)
    fn_state = lower_to_jax(vmap_state)

    x_val = jnp.array([1.0, 0.0, 0.0, 1.0, 3.0, 4.0])  # [1,0], [0,1], [3,4]
    result = fn_state(x_val, None, None, {})
    assert jnp.allclose(result, jnp.array([1.0, 1.0, 5.0]), atol=1e-12)

    # --- Control batch ---
    thrusters = Control("thrusters", shape=(4,))
    thrusters._slice = slice(0, 4)

    vmap_ctrl = Vmap(lambda t: t * 2.0, batch=thrusters)
    fn_ctrl = lower_to_jax(vmap_ctrl)

    u_val = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = fn_ctrl(None, u_val, None, {})
    assert jnp.allclose(result, jnp.array([2.0, 4.0, 6.0, 8.0]), atol=1e-12)

    # --- Mixed State + Parameter ---
    max_dists = Parameter("max_dists", shape=(3,), value=np.array([2.0, 2.0, 10.0]))
    vmap_mixed = Vmap(
        lambda pos, max_d: max_d - Norm(pos, ord=2),
        batch=[positions, max_dists],
    )
    assert vmap_mixed.is_state == (True, False)
    assert vmap_mixed.is_parameter == (False, True)

    fn_mixed = lower_to_jax(vmap_mixed)
    params = {"max_dists": jnp.array([2.0, 2.0, 10.0])}
    result = fn_mixed(x_val, None, None, params)
    assert jnp.allclose(result, jnp.array([1.0, 1.0, 5.0]), atol=1e-12)


def test_vmap_mixed_state_control_batch():
    """Test Vmap with mixed State and Control batches."""
    positions = State("positions", shape=(4, 3))
    thrusts = Control("thrusts", shape=(4, 3))

    vmap_expr = Vmap(
        lambda pos, thrust: Norm(pos) + Norm(thrust),
        batch=[positions, thrusts],
    )

    assert vmap_expr.num_batches == 2
    assert vmap_expr.is_state == (True, False)
    assert vmap_expr.is_control == (False, True)
    assert vmap_expr.check_shape() == (4,)


def test_vmap_state_no_slice_raises():
    """Test that lowering State batch without slice assigned raises error."""
    from openscvx.symbolic.lower import lower_to_jax

    positions = State("positions", shape=(3, 2))
    # Intentionally not setting _slice

    vmap_expr = Vmap(lambda pos: Norm(pos), batch=positions)

    with pytest.raises(ValueError, match="has no slice assigned"):
        lower_to_jax(vmap_expr)
