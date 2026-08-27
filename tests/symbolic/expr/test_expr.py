"""Tests for core AST functionality.

This module tests the fundamental AST infrastructure of the symbolic expression
system, including:
- to_expr() conversion function
- traverse() and walk() tree traversal functions
- Tree structure and pretty printing
- Base Expr/Leaf behavior

Note: Tests for specific node types (Add, Mul, Constraint, etc.) are in their
respective test_*_nodes.py files.
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Add,
    Constant,
    Control,
    Mul,
    State,
    Variable,
    to_expr,
    traverse,
    walk,
)
from openscvx.symbolic.preprocessing import validate_shapes

# =============================================================================
# to_expr() Conversion Tests
# =============================================================================


def test_to_expr_wraps_numbers_and_arrays():
    # scalars
    c1 = to_expr(5)
    assert isinstance(c1, Constant)
    assert c1.value.shape == () and c1.value == 5

    # 1-D arrays become Constant
    arr = [1, 2, 3]
    c2 = to_expr(arr)
    assert isinstance(c2, Constant)
    assert np.array_equal(c2.value, np.array(arr))

    # passing through an Expr unchanged
    a = Constant(np.array([1.0, 2.0]))
    assert to_expr(a) is a


def test_to_expr_passes_variables_through():
    v = Variable("v", (1,))
    x = State("x", (1,))
    u = Control("u", (1,))
    assert to_expr(v) is v
    assert to_expr(x) is x
    assert to_expr(u) is u


# =============================================================================
# Tree Structure and Pretty Printing
# =============================================================================


def test_pretty_print_tree_structure():
    # build a nested tree: -( (a + b) * c )
    a, b, c = Constant(1), Constant(2), Constant(3)
    tree = -((a + b) * c)
    p = tree.pretty()
    # Should indent like:
    # Neg
    #   Mul
    #     Add
    #       Const
    #       Const
    #     Const
    lines = p.splitlines()
    assert lines[0].strip() == "Neg"
    assert lines[1].strip() == "Mul"
    # deeper indent for Add's children:
    assert "Add" in lines[2]
    assert "Const" in lines[3]  # one of the leaves


# =============================================================================
# traverse() Function Tests
# =============================================================================


def test_traverse_visits_all_nodes_in_preorder():
    # build a small graph: (a + (b * c))
    a, b, c = Constant(1), Constant(2), Constant(3)
    expr = Add(a, Mul(b, c))
    visited = []

    def visit(node):
        visited.append(type(node).__name__)

    traverse(expr, visit)

    # preorder: Add → a → Mul → b → c
    assert visited == ["Add", "Constant", "Mul", "Constant", "Constant"]


# =============================================================================
# walk() Function Tests
# =============================================================================


def test_walk_yields_every_node_parents_first():
    a, b, c = Constant(1), Constant(2), Constant(3)
    expr = Add(a, Mul(b, c))

    visited = list(walk(expr))

    assert visited[0] is expr
    assert {id(n) for n in visited} == {id(n) for n in (expr, a, b, c, expr.children()[1])}


def test_walk_visits_shared_subexpressions_once():
    # (x + 1) * (x + 1) with a genuinely shared child, as canonicalization produces
    x = State("x", shape=(2,))
    shared = Add(x, Constant(1))
    expr = Mul(shared, shared)

    visited = list(walk(expr))

    assert sum(1 for n in visited if n is shared) == 1
    assert len(visited) == 4  # Mul, Add, State, Constant


# =============================================================================
# Expr.citation() Default Tests
# =============================================================================


def test_citation_defaults_to_empty():
    x = State("x", shape=(2,))
    assert x.citation() == []
    assert (x + 1).citation() == []


# =============================================================================
# Expr.canonicalize() Method Tests
# =============================================================================


def test_constants_are_unchanged_by_canonicalization():
    """Test that constants are already normalized and unchanged by canonicalization.

    Since canonicalization is now a method on Expr, this tests the base behavior
    that Constants return themselves unchanged.
    """
    # Constants are now normalized at construction time, so canonicalization should be a no-op
    const_scalar = Constant(5.0)
    const_vector = Constant([1.0, 2.0, 3.0])
    const_matrix = Constant([[1.0, 2.0], [3.0, 4.0]])

    # Canonicalization should return the same object (no changes needed)
    canon_scalar = const_scalar.canonicalize()
    canon_vector = const_vector.canonicalize()
    canon_matrix = const_matrix.canonicalize()

    assert canon_scalar is const_scalar  # Should be same object
    assert canon_vector is const_vector
    assert canon_matrix is const_matrix

    # Values should be already normalized
    assert const_scalar.value.shape == ()
    assert const_vector.value.shape == (3,)
    assert const_matrix.value.shape == (2, 2)


def test_vector_constraint_equivalence_after_canonicalization():
    """Test that different ways of creating vector constraints become equivalent after
    canonicalization.

    This tests that canonicalize() properly normalizes expressions across the tree.
    """
    x = State("x", shape=(3,))
    bounds = np.array([1.0, 2.0, 3.0])

    # Two ways to create the same constraint - constants are normalized at construction now
    constraint1 = x <= Constant(bounds)
    constraint2 = x <= Constant(np.array([bounds]))  # Extra dimension gets squeezed at construction

    # Constants should already be equivalent at construction time
    assert np.array_equal(constraint1.rhs.value, constraint2.rhs.value)
    assert constraint1.rhs.value.shape == constraint2.rhs.value.shape

    canon1 = constraint1.canonicalize()
    canon2 = constraint2.canonicalize()

    # After canonicalization, they should remain equivalent (and in canonical form)
    assert isinstance(canon1.rhs, Constant)
    assert isinstance(canon2.rhs, Constant)
    assert np.array_equal(canon1.rhs.value, canon2.rhs.value)
    assert canon1.rhs.value.shape == canon2.rhs.value.shape


# =============================================================================
# Constant
# =============================================================================


def test_constant_normalization_invariant():
    """Test that different ways of creating constants are normalized consistently"""
    import numpy as np

    # Test scalar normalization
    scalar = Constant(5.0)
    array_1d = Constant(np.array([5.0]))
    array_2d = Constant(np.array([[5.0]]))

    # All should have same shape and value after normalization
    assert scalar.value.shape == array_1d.value.shape == array_2d.value.shape
    assert np.allclose(scalar.value, array_1d.value)
    assert np.allclose(scalar.value, array_2d.value)

    # Test vector normalization
    vector = Constant(np.array([1.0, 2.0, 3.0]))
    wrapped_vector = Constant(np.array([[1.0, 2.0, 3.0]]))  # (1, 3) shape

    assert vector.value.shape == wrapped_vector.value.shape == (3,)
    assert np.array_equal(vector.value, wrapped_vector.value)

    # Test that meaningful dimensions are preserved
    matrix = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert matrix.value.shape == (2, 2)

    # Test multiple singleton dimensions
    multi_singleton = Constant(np.array([[[[1.0], [2.0]]]]))  # (1, 1, 2, 1)
    assert multi_singleton.value.shape == (2,)
    assert np.array_equal(multi_singleton.value, [1.0, 2.0])

    # Test the exact case from the old canonicalizer tests: (1, 1, 3, 1) -> (3,)
    array_multi_singleton = np.array([[[[1.0], [2.0], [3.0]]]])
    assert array_multi_singleton.shape == (1, 1, 3, 1)
    const = Constant(array_multi_singleton)
    assert const.value.shape == (3,)
    assert np.array_equal(const.value, [1.0, 2.0, 3.0])


def test_constant_normalization_validation_invariant():
    """Test that preprocessing validation catches improperly normalized constants"""
    # This test verifies our validation works, but we shouldn't be able to create
    # improperly normalized constants anymore due to the new __init__ logic

    # Create a properly normalized constant
    c = Constant(np.array([1.0, 2.0]))
    c.check_shape()  # Should not raise

    # Test that validation would catch violation if it occurred
    # (Though this shouldn't happen with new __init__ logic)
    assert c.value.shape == np.squeeze(c.value).shape


def test_to_expr_normalization_consistency():
    """Test that to_expr creates properly normalized constants"""
    from openscvx.symbolic.expr import to_expr

    # Different ways of creating same value through to_expr
    expr1 = to_expr(5.0)
    expr2 = to_expr([5.0])
    expr3 = to_expr([[5.0]])

    # All should be identical after normalization
    assert isinstance(expr1, Constant)
    assert isinstance(expr2, Constant)
    assert isinstance(expr3, Constant)

    assert expr1.value.shape == expr2.value.shape == expr3.value.shape
    assert np.allclose(expr1.value, expr2.value)
    assert np.allclose(expr1.value, expr3.value)


def test_constant_repr_format():
    """Test that constant repr shows clean Python values, not numpy arrays"""

    # Scalar should show as plain number
    scalar = Constant(1.5)
    assert repr(scalar) == "Const(1.5)"

    # Vector should show as Python list
    vector = Constant([1.0, 2.0, 3.0])
    assert repr(vector) == "Const([1.0, 2.0, 3.0])"

    # Matrix should show as nested Python list
    matrix = Constant([[1.0, 2.0], [3.0, 4.0]])
    assert repr(matrix) == "Const([[1.0, 2.0], [3.0, 4.0]])"

    # Verify that constants created with different input types have same repr
    scalar_from_array = Constant(np.array([1.5]))  # Gets squeezed to scalar
    assert repr(scalar_from_array) == "Const(1.5)"

    vector_from_nested = Constant(np.array([[1.0, 2.0, 3.0]]))  # Gets squeezed to vector
    assert repr(vector_from_nested) == "Const([1.0, 2.0, 3.0])"


def test_constant_normalization_preserves_broadcasting():
    """Test that normalized constants still broadcast correctly with other expressions"""

    # These should broadcast correctly after normalization
    scalar = Constant([[5.0]])  # (1,1) -> () after squeeze
    vector = Constant([1.0, 2.0, 3.0])  # (3,) stays (3,)
    matrix = Constant([[1.0, 2.0], [3.0, 4.0]])  # (2,2) stays (2,2)

    # Verify normalization happened
    assert scalar.value.shape == ()
    assert vector.value.shape == (3,)
    assert matrix.value.shape == (2, 2)

    # Broadcasting should still work with normalized constants
    scalar_plus_vector = scalar + vector  # () + (3,) should broadcast to (3,)
    scalar_plus_vector.check_shape()  # Should not raise

    # Test broadcasting between normalized constants
    scalar_times_matrix = scalar * matrix  # () * (2,2) should broadcast to (2,2)
    scalar_times_matrix.check_shape()  # Should not raise

    # Vector with matrix should fail (non-broadcastable)
    with pytest.raises(ValueError):
        (vector + matrix).check_shape()  # (3,) + (2,2) should fail


def test_vector_constraints_with_normalized_constants():
    """Test that vector constraints work correctly with normalized constants"""

    x = State("x", (3,))

    # Different ways of creating same constraint bounds - all should normalize to same thing
    bounds1 = Constant(np.array([1.0, 2.0, 3.0]))  # Already (3,)
    bounds2 = Constant(np.array([[1.0, 2.0, 3.0]]))  # (1,3) -> (3,) after squeeze
    bounds3 = Constant(np.array([[[1.0]], [[2.0]], [[3.0]]]))  # (3,1,1) -> (3,) after squeeze

    # All should have same normalized shape
    assert bounds1.value.shape == (3,)
    assert bounds2.value.shape == (3,)
    assert bounds3.value.shape == (3,)
    assert np.array_equal(bounds1.value, bounds2.value)
    assert np.array_equal(bounds1.value, bounds3.value)

    # All constraints should validate successfully
    constraint1 = x <= bounds1
    constraint2 = x <= bounds2
    constraint3 = x <= bounds3

    validate_shapes([constraint1, constraint2, constraint3])  # Should not raise

    # Broadcasting constraint: scalar bound with vector state
    scalar_bound = Constant([[2.0]])  # (1,1) -> () after squeeze
    assert scalar_bound.value.shape == ()

    scalar_constraint = x <= scalar_bound  # (3,) <= () should broadcast
    scalar_constraint.check_shape()  # Should not raise


def test_constant_normalization_preserves_dtype():
    """Test that normalization preserves numpy dtypes correctly"""

    # Test different dtypes with singleton dimensions
    int32_array = Constant(np.array([[1, 2, 3]], dtype=np.int32))  # (1,3) -> (3,)
    int64_array = Constant(np.array([[1, 2, 3]], dtype=np.int64))  # (1,3) -> (3,)
    float32_array = Constant(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))  # (1,3) -> (3,)
    float64_array = Constant(np.array([[1.0, 2.0, 3.0]], dtype=np.float64))  # (1,3) -> (3,)
    bool_array = Constant(np.array([[True, False, True]], dtype=np.bool_))  # (1,3) -> (3,)

    # Verify shapes were squeezed
    assert int32_array.value.shape == (3,)
    assert int64_array.value.shape == (3,)
    assert float32_array.value.shape == (3,)
    assert float64_array.value.shape == (3,)
    assert bool_array.value.shape == (3,)

    # Verify dtypes were preserved
    assert int32_array.value.dtype == np.int32
    assert int64_array.value.dtype == np.int64
    assert float32_array.value.dtype == np.float32
    assert float64_array.value.dtype == np.float64
    assert bool_array.value.dtype == np.bool_

    # Test scalar dtypes
    scalar_int = Constant(np.array([[42]], dtype=np.int32))  # (1,1) -> ()
    scalar_float = Constant(np.array([[3.14]], dtype=np.float64))  # (1,1) -> ()

    assert scalar_int.value.shape == ()
    assert scalar_float.value.shape == ()
    assert scalar_int.value.dtype == np.int32
    assert scalar_float.value.dtype == np.float64
    assert scalar_int.value == 42
    assert scalar_float.value == 3.14
