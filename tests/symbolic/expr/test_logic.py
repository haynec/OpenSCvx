"""Tests for logical and control flow operations.

This module tests logical/control flow nodes:

- All: Logical AND reduction over predicates
- Any: Logical OR reduction over predicates
- Cond: Conditional expression using jax.lax.cond

Tests are organized by node type, with each section covering:

1. Node creation and properties
2. Shape Checking
3. Canonicalization
4. JAX lowering
5. CVXPy lowering
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import All, Any, Cond, Constant, Inequality, State, Variable
from openscvx.symbolic.expr.linalg import Norm

# =============================================================================
# All
# =============================================================================


# --- All: Creation & Properties ---


def test_all_creation_single_predicate():
    """Test All creation with a single predicate."""
    x = Variable("x", shape=(3,))
    pred = Norm(x) <= 1.0

    all_expr = All(pred)

    assert len(all_expr.predicates) == 1
    assert all_expr.predicates[0] is pred


def test_all_creation_multiple_predicates():
    """Test All creation with multiple predicates."""
    x = Variable("x", shape=())
    pred1 = x >= 0.0
    pred2 = x <= 10.0

    all_expr = All([pred1, pred2])

    assert len(all_expr.predicates) == 2
    assert all_expr.predicates[0] is pred1
    assert all_expr.predicates[1] is pred2


def test_all_children():
    """Test that children() returns all predicates."""
    x = Variable("x", shape=())
    pred1 = x >= 0.0
    pred2 = x <= 10.0

    all_expr = All([pred1, pred2])
    children = all_expr.children()

    assert len(children) == 2
    assert children[0] is pred1
    assert children[1] is pred2


def test_all_repr():
    """Test string representation of All."""
    x = Variable("x", shape=())
    pred = x <= 1.0

    all_single = All(pred)
    assert "All(" in repr(all_single)

    all_multiple = All([pred, x >= 0.0])
    assert "All([" in repr(all_multiple)


# --- All: Validation ---


def test_all_requires_inequality():
    """Test that All raises TypeError for non-Inequality."""
    x = Variable("x", shape=())

    with pytest.raises(TypeError, match="must be an Inequality"):
        All(x)


def test_all_empty_list_raises():
    """Test that All raises ValueError for empty list."""
    with pytest.raises(ValueError, match="cannot be empty"):
        All([])


def test_all_list_with_non_inequality_raises():
    """Test that All raises TypeError for list containing non-Inequality."""
    x = Variable("x", shape=())
    pred = x <= 1.0

    with pytest.raises(TypeError, match="predicate\\[1\\] must be an Inequality"):
        All([pred, x])


# --- All: Shape Checking ---


def test_all_shape_always_scalar():
    """Test that All always returns scalar shape."""
    x = Variable("x", shape=())
    all_scalar = All(x <= 1.0)
    assert all_scalar.check_shape() == ()

    y = Variable("y", shape=(3,))
    all_vector = All(y <= np.array([1.0, 2.0, 3.0]))
    assert all_vector.check_shape() == ()


# --- All: Canonicalization ---


def test_all_canonicalize():
    """Test that canonicalization recurses into predicates."""
    from openscvx.symbolic.expr import Add, Sub

    x = Variable("x", shape=())
    pred = Add(x, Constant(0.0)) <= 1.0

    all_expr = All(pred)
    canonical = all_expr.canonicalize()

    assert isinstance(canonical, All)
    assert isinstance(canonical.predicates[0], Inequality)
    assert isinstance(canonical.predicates[0].lhs, Sub)


# --- All: JAX Lowering ---


def test_all_jax_scalar_predicates_all_satisfied():
    """Test All JAX lowering when all predicates are satisfied."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(1,))
    x._slice = slice(0, 1)

    all_expr = All([x >= 0.0, x <= 10.0])
    fn = lower_to_jax(all_expr)

    # x = 5.0: both satisfied -> True
    result = fn(jnp.array([5.0]), None, 0, {})
    assert result == True  # noqa: E712


def test_all_jax_scalar_predicates_one_violated():
    """Test All JAX lowering when one predicate is violated."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(1,))
    x._slice = slice(0, 1)

    all_expr = All([x >= 0.0, x <= 10.0])
    fn = lower_to_jax(all_expr)

    # x = -1.0: first violated -> False
    result = fn(jnp.array([-1.0]), None, 0, {})
    assert result == False  # noqa: E712

    # x = 15.0: second violated -> False
    result = fn(jnp.array([15.0]), None, 0, {})
    assert result == False  # noqa: E712


def test_all_jax_vector_predicate():
    """Test All JAX lowering with vector predicate."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(3,))
    x._slice = slice(0, 3)

    # All elements must be >= 0
    all_expr = All(x >= 0.0)
    fn = lower_to_jax(all_expr)

    # All positive -> True
    result = fn(jnp.array([1.0, 2.0, 3.0]), None, 0, {})
    assert result == True  # noqa: E712

    # One negative -> False
    result = fn(jnp.array([1.0, -1.0, 3.0]), None, 0, {})
    assert result == False  # noqa: E712


# --- All: CVXPy Lowering ---


def test_all_cvxpy_raises_not_implemented():
    """Test that All raises NotImplementedError in CVXPy lowering."""
    import cvxpy as cp

    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 1), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(1,))
    all_expr = All(x <= Constant(1.0))

    with pytest.raises(NotImplementedError, match="not DCP-compliant"):
        lowerer.lower(all_expr)


# =============================================================================
# Any
# =============================================================================


# --- Any: Creation & Properties ---


def test_any_creation_single_predicate():
    """Test Any creation with a single predicate."""
    x = Variable("x", shape=(3,))
    pred = Norm(x) <= 1.0

    any_expr = Any(pred)

    assert len(any_expr.predicates) == 1
    assert any_expr.predicates[0] is pred


def test_any_creation_multiple_predicates():
    """Test Any creation with multiple predicates."""
    x = Variable("x", shape=())
    pred1 = x <= 0.0
    pred2 = x >= 10.0

    any_expr = Any([pred1, pred2])

    assert len(any_expr.predicates) == 2


def test_any_children():
    """Test that children() returns all predicates."""
    x = Variable("x", shape=())
    pred1 = x <= 0.0
    pred2 = x >= 10.0

    any_expr = Any([pred1, pred2])
    children = any_expr.children()

    assert len(children) == 2


def test_any_repr():
    """Test string representation of Any."""
    x = Variable("x", shape=())
    pred = x <= 1.0

    any_single = Any(pred)
    assert "Any(" in repr(any_single)


# --- Any: Validation ---


def test_any_requires_inequality():
    """Test that Any raises TypeError for non-Inequality."""
    x = Variable("x", shape=())

    with pytest.raises(TypeError, match="must be an Inequality"):
        Any(x)


def test_any_empty_list_raises():
    """Test that Any raises ValueError for empty list."""
    with pytest.raises(ValueError, match="cannot be empty"):
        Any([])


# --- Any: Shape Checking ---


def test_any_shape_always_scalar():
    """Test that Any always returns scalar shape."""
    x = Variable("x", shape=())
    any_scalar = Any(x <= 1.0)
    assert any_scalar.check_shape() == ()

    y = Variable("y", shape=(3,))
    any_vector = Any(y <= np.array([1.0, 2.0, 3.0]))
    assert any_vector.check_shape() == ()


# --- Any: Canonicalization ---


def test_any_canonicalize():
    """Test that canonicalization recurses into predicates."""
    from openscvx.symbolic.expr import Add, Sub

    x = Variable("x", shape=())
    pred = Add(x, Constant(0.0)) <= 1.0

    any_expr = Any(pred)
    canonical = any_expr.canonicalize()

    assert isinstance(canonical, Any)
    assert isinstance(canonical.predicates[0], Inequality)
    assert isinstance(canonical.predicates[0].lhs, Sub)


# --- Any: JAX Lowering ---


def test_any_jax_scalar_predicates_one_satisfied():
    """Test Any JAX lowering when one predicate is satisfied."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(1,))
    x._slice = slice(0, 1)

    # x <= 0 OR x >= 10
    any_expr = Any([x <= 0.0, x >= 10.0])
    fn = lower_to_jax(any_expr)

    # x = -1.0: first satisfied -> True
    result = fn(jnp.array([-1.0]), None, 0, {})
    assert result == True  # noqa: E712

    # x = 15.0: second satisfied -> True
    result = fn(jnp.array([15.0]), None, 0, {})
    assert result == True  # noqa: E712


def test_any_jax_scalar_predicates_none_satisfied():
    """Test Any JAX lowering when no predicates are satisfied."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(1,))
    x._slice = slice(0, 1)

    # x <= 0 OR x >= 10
    any_expr = Any([x <= 0.0, x >= 10.0])
    fn = lower_to_jax(any_expr)

    # x = 5.0: neither satisfied -> False
    result = fn(jnp.array([5.0]), None, 0, {})
    assert result == False  # noqa: E712


def test_any_jax_vector_predicate():
    """Test Any JAX lowering with vector predicate.

    For a vector predicate, Any returns True if ANY element satisfies the
    predicate (element-wise OR semantics).
    """
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(3,))
    x._slice = slice(0, 3)

    # Any element >= 0
    any_expr = Any(x >= 0.0)
    fn = lower_to_jax(any_expr)

    # All positive -> True
    result = fn(jnp.array([1.0, 2.0, 3.0]), None, 0, {})
    assert result == True  # noqa: E712

    # Some positive, some negative -> True (any element satisfies)
    result = fn(jnp.array([1.0, -1.0, -2.0]), None, 0, {})
    assert result == True  # noqa: E712

    # All negative -> False (no element satisfies)
    result = fn(jnp.array([-1.0, -2.0, -3.0]), None, 0, {})
    assert result == False  # noqa: E712


# --- Any: CVXPy Lowering ---


def test_any_cvxpy_raises_not_implemented():
    """Test that Any raises NotImplementedError in CVXPy lowering."""
    import cvxpy as cp

    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 1), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(1,))
    any_expr = Any(x <= Constant(1.0))

    with pytest.raises(NotImplementedError, match="not DCP-compliant"):
        lowerer.lower(any_expr)


# =============================================================================
# Cond
# =============================================================================


# --- Cond: Creation & Properties ---


def test_cond_creation_basic():
    """Test basic Cond node creation and properties."""
    x = Variable("x", shape=(3,))
    pred = Norm(x) <= 1.0
    true_branch = Constant(5.0)
    false_branch = Constant(10.0)

    cond = Cond(pred, true_branch, false_branch)

    assert isinstance(cond.predicate, Inequality)
    assert cond.predicate is pred
    assert cond.true_branch is true_branch
    assert cond.false_branch is false_branch
    assert cond.node_ranges is None


def test_cond_creation_with_node_ranges():
    """Test Cond creation with node_ranges parameter."""
    x = Variable("x", shape=(3,))
    pred = Norm(x) <= 1.0

    cond = Cond(pred, 5.0, 10.0, node_ranges=[(0, 3), (5, 8)])

    assert cond.node_ranges == [(0, 3), (5, 8)]


def test_cond_creation_with_all():
    """Test Cond creation with All predicate."""
    x = Variable("x", shape=())
    all_pred = All([x >= 0.0, x <= 10.0])

    cond = Cond(all_pred, 5.0, 10.0)

    assert isinstance(cond.predicate, All)
    assert cond.predicate is all_pred


def test_cond_creation_with_any():
    """Test Cond creation with Any predicate."""
    x = Variable("x", shape=())
    any_pred = Any([x <= 0.0, x >= 10.0])

    cond = Cond(any_pred, 5.0, 10.0)

    assert isinstance(cond.predicate, Any)
    assert cond.predicate is any_pred


def test_cond_creation_with_list_wraps_in_all():
    """Test that Cond with list predicate wraps it in All."""
    x = Variable("x", shape=())
    pred1 = x >= 0.0
    pred2 = x <= 10.0

    cond = Cond([pred1, pred2], 5.0, 10.0)

    assert isinstance(cond.predicate, All)
    assert len(cond.predicate.predicates) == 2


def test_cond_children():
    """Test that children() returns predicate and both branches."""
    x = Variable("x", shape=(3,))
    pred = Norm(x) <= 1.0
    true_branch = Constant(5.0)
    false_branch = Constant(10.0)

    cond = Cond(pred, true_branch, false_branch)
    children = cond.children()

    assert len(children) == 3
    assert children[0] is pred
    assert children[1] is true_branch
    assert children[2] is false_branch


def test_cond_repr():
    """Test string representation of Cond."""
    x = Variable("x", shape=())
    pred = x <= 1.0

    cond = Cond(pred, 5.0, 10.0)
    assert "Cond(" in repr(cond)

    cond_with_ranges = Cond(pred, 5.0, 10.0, node_ranges=[(0, 2)])
    assert "node_ranges=" in repr(cond_with_ranges)


def test_cond_auto_converts_branches_to_expr():
    """Test that scalar branches are auto-converted to Constant."""
    x = Variable("x", shape=())
    pred = x <= 1.0

    cond = Cond(pred, 5.0, 10.0)

    assert isinstance(cond.true_branch, Constant)
    assert isinstance(cond.false_branch, Constant)


# --- Cond: Validation ---


def test_cond_requires_valid_predicate():
    """Test that Cond raises TypeError for invalid predicate type."""
    x = Variable("x", shape=())

    with pytest.raises(TypeError, match="must be an Inequality"):
        Cond(x, 5.0, 10.0)


def test_cond_invalid_node_ranges_not_list():
    """Test that node_ranges must be a list."""
    x = Variable("x", shape=())
    pred = x <= 1.0

    with pytest.raises(TypeError, match="must be a list"):
        Cond(pred, 5.0, 10.0, node_ranges=(0, 2))


def test_cond_invalid_node_ranges_not_tuple():
    """Test that node_ranges items must be tuples."""
    x = Variable("x", shape=())
    pred = x <= 1.0

    with pytest.raises(ValueError, match="must be a.*tuple"):
        Cond(pred, 5.0, 10.0, node_ranges=[[0, 2]])


def test_cond_invalid_node_ranges_start_ge_end():
    """Test that node_ranges must have start < end."""
    x = Variable("x", shape=())
    pred = x <= 1.0

    with pytest.raises(ValueError, match="start < end"):
        Cond(pred, 5.0, 10.0, node_ranges=[(3, 2)])


# --- Cond: Shape Checking ---


def test_cond_shape_scalar_branches():
    """Test Cond shape with scalar branches."""
    x = Variable("x", shape=())
    pred = x <= 1.0

    cond = Cond(pred, 5.0, 10.0)
    assert cond.check_shape() == ()


def test_cond_shape_vector_branches():
    """Test Cond shape with vector branches."""
    x = Variable("x", shape=())
    pred = x <= 1.0
    true_branch = Constant(np.array([1.0, 2.0, 3.0]))
    false_branch = Constant(np.array([4.0, 5.0, 6.0]))

    cond = Cond(pred, true_branch, false_branch)
    assert cond.check_shape() == (3,)


def test_cond_shape_broadcasts_branches():
    """Test that Cond broadcasts branch shapes."""
    x = Variable("x", shape=())
    pred = x <= 1.0
    true_branch = Constant(np.array([1.0, 2.0, 3.0]))
    false_branch = Constant(5.0)  # scalar broadcasts to (3,)

    cond = Cond(pred, true_branch, false_branch)
    assert cond.check_shape() == (3,)


def test_cond_shape_incompatible_branches_raises():
    """Test that incompatible branch shapes raise ValueError."""
    x = Variable("x", shape=())
    pred = x <= 1.0
    true_branch = Constant(np.array([1.0, 2.0, 3.0]))
    false_branch = Constant(np.array([1.0, 2.0]))

    cond = Cond(pred, true_branch, false_branch)
    with pytest.raises(ValueError, match="incompatible shapes"):
        cond.check_shape()


def test_cond_shape_non_scalar_predicate_raises():
    """Test that non-scalar predicate raises ValueError."""
    x = Variable("x", shape=(3,))
    pred = x <= np.array([1.0, 2.0, 3.0])  # vector predicate

    cond = Cond(pred, 5.0, 10.0)
    with pytest.raises(ValueError, match="must be scalar"):
        cond.check_shape()


def test_cond_shape_with_all_predicate():
    """Test Cond shape with All predicate (reduces vector to scalar)."""
    x = Variable("x", shape=(3,))
    # Vector predicate wrapped in All -> scalar
    all_pred = All(x <= np.array([1.0, 2.0, 3.0]))

    cond = Cond(all_pred, 5.0, 10.0)
    assert cond.check_shape() == ()


# --- Cond: Canonicalization ---


def test_cond_canonicalize_preserves_structure():
    """Test that canonicalization preserves Cond structure."""
    x = Variable("x", shape=())
    pred = x <= 1.0

    cond = Cond(pred, 5.0, 10.0)
    canonical = cond.canonicalize()

    assert isinstance(canonical, Cond)
    assert isinstance(canonical.predicate, Inequality)


def test_cond_canonicalize_preserves_node_ranges():
    """Test that canonicalization preserves node_ranges."""
    x = Variable("x", shape=())
    pred = x <= 1.0

    cond = Cond(pred, 5.0, 10.0, node_ranges=[(0, 3)])
    canonical = cond.canonicalize()

    assert canonical.node_ranges == [(0, 3)]


def test_cond_canonicalize_recurses_into_children():
    """Test that canonicalization recurses into predicate and branches."""
    from openscvx.symbolic.expr import Add, Sub

    x = Variable("x", shape=())
    # pred: (x + 0) <= 1 should canonicalize to (x - 1) <= 0
    pred = Add(x, Constant(0.0)) <= 1.0
    # true branch: 5 + 0 should canonicalize to 5
    true_branch = Add(Constant(5.0), Constant(0.0))

    cond = Cond(pred, true_branch, 10.0)
    canonical = cond.canonicalize()

    # Constraint canonicalizes to standard form: (lhs - rhs) <= 0
    # So (x + 0) <= 1 becomes (x - 1) <= 0, where lhs is Sub(x, 1)
    assert isinstance(canonical.predicate, Inequality)
    assert isinstance(canonical.predicate.lhs, Sub)
    assert isinstance(canonical.predicate.rhs, Constant)
    assert canonical.predicate.rhs.value == 0
    # Check true branch was canonicalized (Add(5, 0) -> 5)
    assert isinstance(canonical.true_branch, Constant)


# --- Cond: JAX Lowering ---


def test_cond_jax_constant_predicate_true():
    """Test Cond JAX lowering with constant predicate that evaluates to true."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    # -1 <= 0 is satisfied (true), so should return true_branch
    pred = Constant(-1.0) <= Constant(0.0)
    cond = Cond(pred, Constant(5.0), Constant(10.0))

    fn = lower_to_jax(cond)
    result = fn(None, None, 0, {})

    assert jnp.isclose(result, 5.0)


def test_cond_jax_constant_predicate_false():
    """Test Cond JAX lowering with constant predicate that evaluates to false."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    # 1 <= 0 is violated (false), so should return false_branch
    pred = Constant(1.0) <= Constant(0.0)
    cond = Cond(pred, Constant(5.0), Constant(10.0))

    fn = lower_to_jax(cond)
    result = fn(None, None, 0, {})

    assert jnp.isclose(result, 10.0)


def test_cond_jax_with_state():
    """Test Cond JAX lowering with state-dependent predicate."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(1,))
    x._slice = slice(0, 1)

    # x <= 0.5: true when x <= 0.5
    pred = x <= Constant(0.5)
    cond = Cond(pred, Constant(5.0), Constant(10.0))

    fn = lower_to_jax(cond)

    # x = 0.3 <= 0.5, predicate satisfied -> true branch
    result_true = fn(jnp.array([0.3]), None, 0, {})
    assert jnp.isclose(result_true, 5.0)

    # x = 0.7 > 0.5, predicate violated -> false branch
    result_false = fn(jnp.array([0.7]), None, 0, {})
    assert jnp.isclose(result_false, 10.0)


def test_cond_jax_with_expression_branches():
    """Test Cond JAX lowering with expression branches."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Mul
    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(1,))
    x._slice = slice(0, 1)

    # If x <= 1, return 2*x, else return 3*x
    pred = x <= Constant(1.0)
    true_branch = Mul(Constant(2.0), x)
    false_branch = Mul(Constant(3.0), x)
    cond = Cond(pred, true_branch, false_branch)

    fn = lower_to_jax(cond)

    # x = 0.5 <= 1, return 2*0.5 = 1.0
    result = fn(jnp.array([0.5]), None, 0, {})
    assert jnp.isclose(result, 1.0)

    # x = 2.0 > 1, return 3*2 = 6.0
    result = fn(jnp.array([2.0]), None, 0, {})
    assert jnp.isclose(result, 6.0)


def test_cond_jax_with_node_ranges():
    """Test Cond JAX lowering with node_ranges restriction."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    # Predicate always satisfied (true)
    pred = Constant(-1.0) <= Constant(0.0)
    # Only active at nodes [0, 2)
    cond = Cond(pred, Constant(5.0), Constant(10.0), node_ranges=[(0, 2)])

    fn = lower_to_jax(cond)

    # Node 0: in range, predicate true -> true branch
    assert jnp.isclose(fn(None, None, 0, {}), 5.0)

    # Node 1: in range, predicate true -> true branch
    assert jnp.isclose(fn(None, None, 1, {}), 5.0)

    # Node 2: out of range -> false branch (regardless of predicate)
    assert jnp.isclose(fn(None, None, 2, {}), 10.0)

    # Node 5: out of range -> false branch
    assert jnp.isclose(fn(None, None, 5, {}), 10.0)


def test_cond_jax_with_multiple_node_ranges():
    """Test Cond JAX lowering with multiple node_ranges."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    pred = Constant(-1.0) <= Constant(0.0)
    cond = Cond(pred, Constant(5.0), Constant(10.0), node_ranges=[(0, 2), (5, 7)])

    fn = lower_to_jax(cond)

    # In first range
    assert jnp.isclose(fn(None, None, 0, {}), 5.0)
    assert jnp.isclose(fn(None, None, 1, {}), 5.0)

    # Between ranges
    assert jnp.isclose(fn(None, None, 3, {}), 10.0)

    # In second range
    assert jnp.isclose(fn(None, None, 5, {}), 5.0)
    assert jnp.isclose(fn(None, None, 6, {}), 5.0)

    # After all ranges
    assert jnp.isclose(fn(None, None, 10, {}), 10.0)


def test_cond_jax_with_all_predicate():
    """Test Cond JAX lowering with All predicate."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(1,))
    x._slice = slice(0, 1)

    # Explicit All: x >= 0 AND x <= 10
    all_pred = All([x >= 0.0, x <= 10.0])
    cond = Cond(all_pred, Constant(1.0), Constant(0.0))

    fn = lower_to_jax(cond)

    # x = 5.0: both satisfied -> true branch
    assert jnp.isclose(fn(jnp.array([5.0]), None, 0, {}), 1.0)

    # x = -1.0: first violated -> false branch
    assert jnp.isclose(fn(jnp.array([-1.0]), None, 0, {}), 0.0)

    # x = 15.0: second violated -> false branch
    assert jnp.isclose(fn(jnp.array([15.0]), None, 0, {}), 0.0)


def test_cond_jax_with_any_predicate():
    """Test Cond JAX lowering with Any predicate."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(1,))
    x._slice = slice(0, 1)

    # Any: x <= 0 OR x >= 10
    any_pred = Any([x <= 0.0, x >= 10.0])
    cond = Cond(any_pred, Constant(1.0), Constant(0.0))

    fn = lower_to_jax(cond)

    # x = -1.0: first satisfied -> true branch
    assert jnp.isclose(fn(jnp.array([-1.0]), None, 0, {}), 1.0)

    # x = 15.0: second satisfied -> true branch
    assert jnp.isclose(fn(jnp.array([15.0]), None, 0, {}), 1.0)

    # x = 5.0: neither satisfied -> false branch
    assert jnp.isclose(fn(jnp.array([5.0]), None, 0, {}), 0.0)


# --- Cond: CVXPy Lowering ---


def test_cond_cvxpy_raises_not_implemented():
    """Test that Cond raises NotImplementedError in CVXPy lowering."""
    import cvxpy as cp

    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 1), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(1,))
    pred = x <= Constant(1.0)
    cond = Cond(pred, Constant(5.0), Constant(10.0))

    with pytest.raises(NotImplementedError, match="Conditional expressions.*not DCP-compliant"):
        lowerer.lower(cond)


# --- Cond: Multiple Predicates (backwards compatibility) ---


def test_cond_list_predicates_and_semantics():
    """Test Cond with list predicates uses AND semantics (backwards compat)."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(1,))
    x._slice = slice(0, 1)

    # List syntax wraps in All internally
    cond = Cond([x >= 0.0, x <= 10.0], Constant(1.0), Constant(0.0))

    # Verify it was wrapped in All
    assert isinstance(cond.predicate, All)

    fn = lower_to_jax(cond)

    # x = 5.0: both satisfied -> true branch
    assert jnp.isclose(fn(jnp.array([5.0]), None, 0, {}), 1.0)

    # x = -1.0: first violated -> false branch
    assert jnp.isclose(fn(jnp.array([-1.0]), None, 0, {}), 0.0)

    # x = 15.0: second violated -> false branch
    assert jnp.isclose(fn(jnp.array([15.0]), None, 0, {}), 0.0)


# =============================================================================
# Cond: pred=None (purely node-based switching)
# =============================================================================


def test_cond_pred_none_requires_node_ranges():
    """Test that Cond with pred=None requires node_ranges."""
    with pytest.raises(ValueError, match="pred=None requires node_ranges"):
        Cond(None, 5.0, 10.0)


def test_cond_pred_none_creation_and_properties():
    """Test Cond creation with pred=None and its properties."""
    cond = Cond(None, Constant(5.0), Constant(10.0), node_ranges=[(0, 5)])

    assert cond.predicate is None
    assert cond.node_ranges == [(0, 5)]
    assert len(cond.children()) == 2  # excludes None predicate
    assert "Cond(None," in repr(cond)
    assert cond.check_shape() == ()

    # Canonicalization preserves pred=None
    canonical = cond.canonicalize()
    assert canonical.predicate is None
    assert canonical.node_ranges == [(0, 5)]


def test_cond_pred_none_jax_lowering():
    """Test Cond JAX lowering with pred=None for node-based switching."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    cond = Cond(None, Constant(5.0), Constant(10.0), node_ranges=[(0, 3), (7, 10)])
    fn = lower_to_jax(cond)

    # In first range -> true branch
    assert jnp.isclose(fn(None, None, 0, {}), 5.0)
    assert jnp.isclose(fn(None, None, 2, {}), 5.0)

    # Between ranges -> false branch
    assert jnp.isclose(fn(None, None, 3, {}), 10.0)
    assert jnp.isclose(fn(None, None, 5, {}), 10.0)

    # In second range -> true branch
    assert jnp.isclose(fn(None, None, 7, {}), 5.0)

    # After all ranges -> false branch
    assert jnp.isclose(fn(None, None, 10, {}), 10.0)
