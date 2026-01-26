"""Tests for constraint nodes.

This module tests constraint node types:

- Constraint: Base constraint class
- Equality: Equality constraints (==)
- Inequality: Inequality constraints (<=, >=)
- NodalConstraint: Constraints applied at specific nodes
- CrossNodeConstraint: Constraints coupling specific trajectory nodes via NodeReference
- CTCS: Continuous-Time Constraint Satisfaction

Tests are organized by node type, with each section covering:

- Node creation and tree structure
- Shape Checking
- Canonicalization
- JAX lowering
- CVXPY lowering
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    CTCS,
    Constant,
    CrossNodeConstraint,
    Equality,
    Huber,
    Inequality,
    NodalConstraint,
    PositivePart,
    SmoothReLU,
    Square,
    State,
    Sum,
    Variable,
    ctcs,
    traverse,
)

# =============================================================================
# Equality & Inequality Constraints
# =============================================================================


# --- Equality & Inequality: Creation & Tree Structure ---


def test_equality_creation_and_children():
    x = Variable("x", shape=(3,))
    # build lhs ≤ rhs
    c = x == np.array([0.0, 1.0, 2.0])

    assert isinstance(c, Equality)
    # children are exactly the Expr on each side
    lhs, rhs = c.children()
    assert lhs is x
    assert isinstance(rhs, Constant)
    assert repr(c) == "Var('x') == Const([0.0, 1.0, 2.0])"


def test_inequality_creation_and_children():
    x = Variable("x", shape=(3,))
    # build lhs ≤ rhs
    c = x <= np.array([0.0, 1.0, 2.0])

    assert isinstance(c, Inequality)
    # children are exactly the Expr on each side
    lhs, rhs = c.children()
    assert lhs is x
    assert isinstance(rhs, Constant)
    assert repr(c) == "Var('x') <= Const([0.0, 1.0, 2.0])"


def test_inequality_reverse_creation_and_children():
    x = Variable("x", shape=(3,))
    # build lhs ≤ rhs
    c = x >= np.array([0.0, 1.0, 2.0])

    assert isinstance(c, Inequality)
    # children are exactly the Expr on each side
    lhs, rhs = c.children()
    assert rhs is x
    assert isinstance(lhs, Constant)
    assert repr(c) == "Const([0.0, 1.0, 2.0]) <= Var('x')"


# --- Equality & Inequality: Shape Checking ---


def test_constraint_zero_dim_scalar_passes():
    # a true scalar (shape=()) on both sides
    a = Constant(np.array(2.5))
    c = a == 1.0
    c.check_shape()


def test_constraint_length1_array_passes():
    # 1-element arrays count as "scalar"
    b = Constant(np.array([7.0]))
    c = b <= np.ones((1,))
    c.check_shape()


def test_constraint_vector_passes():
    """Vector constraints should now pass validation (interpreted element-wise)"""
    a = Constant(np.zeros((2,)))
    c = a <= np.ones((2,))
    c.check_shape()  # Should NOT raise


def test_constraint_shape_mismatch_raises():
    """Shape mismatches should still error out"""
    a = Constant(np.zeros((2,)))
    c = a == np.zeros((3,))
    with pytest.raises(ValueError):
        c.check_shape()


def test_constraint_broadcasting_passes():
    """Test constraint broadcasting: scalar op vector"""
    x = State("x", (3,))
    c = Constant(np.array(0.0)) <= x  # broadcasts to vector constraint
    c.check_shape()


# --- Equality & Inequality: Canonicalization ---


def test_constraint_recursion_and_type():
    """Test that constraints canonicalize their subexpressions recursively."""
    from openscvx.symbolic.expr import Add

    # test an inequality and equality on two equal constants 3+3 == 6
    lhs = Add(Constant(3), Constant(3))  # will fold to Constant(6)
    rhs = Constant(5)
    ineq = lhs <= rhs
    eq = lhs == rhs

    ineq_c = ineq.canonicalize()
    eq_c = eq.canonicalize()

    assert isinstance(ineq_c, Inequality)
    assert isinstance(ineq_c.lhs, Constant) and ineq_c.lhs.value == 1
    assert isinstance(ineq_c.rhs, Constant) and ineq_c.rhs.value == 0

    assert isinstance(eq_c, Equality)
    assert isinstance(eq_c.lhs, Constant) and eq_c.lhs.value == 1
    assert isinstance(eq_c.rhs, Constant) and eq_c.rhs.value == 0


def test_inequality_preserves_convex_flag():
    """Test that canonicalization preserves the is_convex flag for Inequality constraints."""
    from openscvx.symbolic.expr import State

    x = State("x", shape=(3,))

    # Create a regular (non-convex) inequality constraint
    constraint_nonconvex = x <= Constant([1, 2, 3])
    assert constraint_nonconvex.is_convex is False

    # Create a convex inequality constraint
    constraint_convex = (x <= Constant([1, 2, 3])).convex()
    assert constraint_convex.is_convex is True

    # Canonicalize both
    canon_nonconvex = constraint_nonconvex.canonicalize()
    canon_convex = constraint_convex.canonicalize()

    # Check that convex flags are preserved
    assert canon_nonconvex.is_convex is False
    assert canon_convex.is_convex is True

    # Verify they're still Inequality objects
    assert isinstance(canon_nonconvex, Inequality)
    assert isinstance(canon_convex, Inequality)


def test_equality_preserves_convex_flag():
    """Test that canonicalization preserves the is_convex flag for Equality constraints."""
    from openscvx.symbolic.expr import State

    x = State("x", shape=(3,))

    # Create a regular (non-convex) equality constraint
    constraint_nonconvex = x == Constant([1, 2, 3])
    assert constraint_nonconvex.is_convex is False

    # Create a convex equality constraint
    constraint_convex = (x == Constant([1, 2, 3])).convex()
    assert constraint_convex.is_convex is True

    # Canonicalize both
    canon_nonconvex = constraint_nonconvex.canonicalize()
    canon_convex = constraint_convex.canonicalize()

    # Check that convex flags are preserved
    assert canon_nonconvex.is_convex is False
    assert canon_convex.is_convex is True

    # Verify they're still Equality objects
    assert isinstance(canon_nonconvex, Equality)
    assert isinstance(canon_convex, Equality)


def test_mixed_convex_and_nonconvex_constraints():
    """Test canonicalization with a mix of convex and non-convex constraints."""
    from openscvx.symbolic.expr import State

    x = State("x", shape=(2,))

    # Create various constraints
    constraint1 = x <= Constant([1, 2])  # non-convex
    constraint2 = (x >= Constant([0, 0])).convex()  # convex inequality
    constraint3 = (x == Constant([5, 6])).convex()  # convex equality
    constraint4 = x == Constant([3, 4])  # non-convex equality

    constraints = [constraint1, constraint2, constraint3, constraint4]
    expected_convex = [False, True, True, False]

    # Canonicalize all constraints
    canonical_constraints = [c.canonicalize() for c in constraints]

    # Verify convex flags are preserved
    for canon_c, expected in zip(canonical_constraints, expected_convex):
        assert canon_c.is_convex == expected

    # Verify types are preserved
    assert isinstance(canonical_constraints[0], Inequality)
    assert isinstance(canonical_constraints[1], Inequality)
    assert isinstance(canonical_constraints[2], Equality)
    assert isinstance(canonical_constraints[3], Equality)


# --- Equality & Inequality: JAX Lowering ---


def test_equality_constraint_lowering():
    """Test that equality constraints are lowered to residual form (lhs - rhs)."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Control, Mul, State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.5, 1.0, 1.5])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (3,))
    control._slice = slice(0, 3)

    # Constraint: x == 2*u (should become x - 2*u == 0)
    lhs = state
    rhs = Mul(Constant(2.0), control)
    constraint = Equality(lhs, rhs)

    jl = JaxLowerer()
    fn = jl.lower(constraint)
    residual = fn(x, u, None, None)

    # Residual should be lhs - rhs = x - 2*u
    expected = x - 2.0 * u
    assert jnp.allclose(residual, expected)
    assert residual.shape == (3,)


def test_inequality_constraint_lowering():
    """Test that inequality constraints are lowered to residual form (lhs - rhs)."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.array([0.5, 1.5, 2.5])

    state = State("x", (3,))
    state._slice = slice(0, 3)

    # Constraint: x <= 2.0 (should become x - 2.0 <= 0)
    lhs = state
    rhs = Constant(np.array([2.0, 2.0, 2.0]))
    constraint = Inequality(lhs, rhs)

    jl = JaxLowerer()
    fn = jl.lower(constraint)
    residual = fn(x, None, None, None)

    # Residual should be lhs - rhs = x - 2.0
    expected = x - 2.0
    assert jnp.allclose(residual, expected)
    assert residual.shape == (3,)

    # Check that residual is negative when constraint is satisfied
    # and positive when violated
    assert residual[0] < 0  # 0.5 - 2.0 = -1.5 (satisfied)
    assert residual[1] < 0  # 1.5 - 2.0 = -0.5 (satisfied)
    assert residual[2] > 0  # 2.5 - 2.0 = 0.5 (violated)


def test_constraint_lowering_with_lower_to_jax():
    """Test constraint lowering through the top-level lower_to_jax function."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Add, Control, Mul, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([1.0, 3.0])
    u = jnp.array([0.5])

    pos = State("pos", (2,))
    pos._slice = slice(0, 2)
    vel = Control("vel", (1,))
    vel._slice = slice(0, 1)

    # Mixed constraint: pos[0] + 2*vel <= pos[1]
    # Rearranged: pos[0] + 2*vel - pos[1] <= 0
    lhs = Add(pos[0], Mul(Constant(2.0), vel))
    rhs = pos[1]
    constraint = Inequality(lhs, rhs)

    # Lower using the top-level function
    fn = lower_to_jax(constraint)
    residual = fn(x, u, None, None)

    # Expected: pos[0] + 2*vel - pos[1] = 1.0 + 2*0.5 - 3.0 = -1.0
    expected = 1.0 + 2.0 * 0.5 - 3.0
    assert jnp.allclose(residual, expected)
    assert residual < 0  # Constraint is satisfied


# --- Equality & Inequality: CVXPy Lowering ---


def test_cvxpy_equality_constraint():
    """Test equality constraints"""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    const = Constant(np.array(0.0))
    expr = Equality(x, const)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Constraint)


def test_cvxpy_inequality_constraint():
    """Test inequality constraints"""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    const = Constant(np.array(1.0))
    expr = Inequality(x, const)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Constraint)


# =============================================================================
# NodalConstraint
# =============================================================================


# --- NodalConstraint: Creation & Tree Structure ---


def test_nodal_constraint_creation_basic():
    """Test basic NodalConstraint creation with .at() method."""
    x = State("x", shape=(3,))
    constraint = x <= 1.0

    # Create NodalConstraint using .at() method
    nodal = constraint.at([0, 5, 10])

    assert isinstance(nodal, NodalConstraint)
    assert nodal.constraint is constraint
    assert nodal.nodes == [0, 5, 10]


def test_nodal_constraint_creation_direct():
    """Test direct NodalConstraint instantiation."""
    x = Variable("x", shape=(2,))
    constraint = x == np.array([1.0, 2.0])

    # Create NodalConstraint directly
    nodal = NodalConstraint(constraint, nodes=[1, 3, 5])

    assert isinstance(nodal, NodalConstraint)
    assert nodal.constraint is constraint
    assert nodal.nodes == [1, 3, 5]


def test_nodal_constraint_requires_constraint():
    """Test that NodalConstraint only accepts Constraint objects."""
    x = Variable("x", shape=(3,))
    not_a_constraint = x + 1.0

    with pytest.raises(TypeError, match="NodalConstraint must wrap a Constraint"):
        NodalConstraint(not_a_constraint, nodes=[0, 1, 2])


def test_nodal_constraint_requires_list():
    """Test that NodalConstraint requires nodes to be a list."""
    x = Variable("x", shape=(3,))
    constraint = x <= 1.0

    # Should reject tuples and other iterables
    with pytest.raises(TypeError, match="nodes must be a list"):
        NodalConstraint(constraint, nodes=(0, 1, 2))


def test_nodal_constraint_converts_numpy_integers():
    """Test that NodalConstraint converts numpy integers to Python ints."""
    x = Variable("x", shape=(2,))
    constraint = x >= 0.0

    # Use numpy integers
    nodes_numpy = [np.int32(0), np.int64(5), np.int16(10)]
    nodal = NodalConstraint(constraint, nodes=nodes_numpy)

    # Should convert to Python ints
    assert nodal.nodes == [0, 5, 10]
    assert all(isinstance(n, int) and not isinstance(n, np.integer) for n in nodal.nodes)


def test_nodal_constraint_rejects_non_integer_nodes():
    """Test that NodalConstraint rejects non-integer node indices."""
    x = Variable("x", shape=(3,))
    constraint = x <= 5.0

    with pytest.raises(TypeError, match="all node indices must be integers"):
        NodalConstraint(constraint, nodes=[0, 1.5, 2])

    with pytest.raises(TypeError, match="all node indices must be integers"):
        NodalConstraint(constraint, nodes=[0, "1", 2])


def test_nodal_constraint_children():
    """Test that NodalConstraint.children() returns only the wrapped constraint."""
    x = Variable("x", shape=(3,))
    constraint = x == 0.0
    nodal = constraint.at([0, 10, 20])

    children = nodal.children()
    assert len(children) == 1
    assert children[0] is constraint


def test_nodal_constraint_repr():
    """Test NodalConstraint string representation."""
    x = Variable("x", shape=(2,))
    constraint = x <= np.array([1.0, 2.0])
    nodal = constraint.at([0, 5])

    repr_str = repr(nodal)
    assert "NodalConstraint" in repr_str
    assert "nodes=[0, 5]" in repr_str
    assert "<=" in repr_str


def test_nodal_constraint_convex_method_chaining():
    """Test that NodalConstraint.convex() works in both chaining orders."""
    x = Variable("x", shape=(3,))

    # Test .at().convex() chaining
    nodal1 = (x <= [1, 2, 3]).at([0, 5, 10]).convex()
    assert isinstance(nodal1, NodalConstraint)
    assert nodal1.constraint.is_convex is True
    assert nodal1.nodes == [0, 5, 10]

    # Test .convex().at() chaining
    nodal2 = (x <= [1, 2, 3]).convex().at([0, 5, 10])
    assert isinstance(nodal2, NodalConstraint)
    assert nodal2.constraint.is_convex is True
    assert nodal2.nodes == [0, 5, 10]


# --- NodalConstraint: Shape Checking ---


def test_nodal_constraint_shape_validation_scalar():
    """Test NodalConstraint shape validation with scalar constraints."""
    x = State("x", shape=())
    constraint = x <= 1.0
    nodal = constraint.at([0, 5, 10])

    # Should validate successfully
    shape = nodal.check_shape()
    assert shape == ()  # NodalConstraint produces scalar like all constraints


def test_nodal_constraint_shape_validation_vector():
    """Test NodalConstraint shape validation with vector constraints."""
    x = State("x", shape=(3,))
    constraint = x <= np.ones(3)
    nodal = constraint.at([0, 5, 10])

    # Should validate successfully (vector constraints are element-wise)
    shape = nodal.check_shape()
    assert shape == ()  # Always returns scalar shape


def test_nodal_constraint_shape_validation_broadcasts():
    """Test NodalConstraint shape validation with broadcasting."""
    x = State("x", shape=(3,))
    constraint = x <= 1.0  # Scalar broadcasts to vector
    nodal = constraint.at([2, 4, 6])

    # Should validate successfully
    shape = nodal.check_shape()
    assert shape == ()


def test_nodal_constraint_shape_mismatch_raises():
    """Test that NodalConstraint detects shape mismatches in wrapped constraint."""
    x = State("x", shape=(2,))
    constraint = x == np.zeros(3)  # 2 vs 3 mismatch
    nodal = constraint.at([0, 1])

    # Should raise due to wrapped constraint shape mismatch
    with pytest.raises(ValueError):
        nodal.check_shape()


# --- NodalConstraint: Canonicalization ---


def test_nodal_constraint_preserves_inner_convex_flag():
    """Test that canonicalization preserves the is_convex flag for constraints wrapped in
    NodalConstraint"""
    from openscvx.symbolic.expr import State

    x = State("x", shape=(3,))

    # Create a convex constraint and wrap it in NodalConstraint
    base_constraint = (x <= Constant([1, 2, 3])).convex()
    assert base_constraint.is_convex is True

    nodal_constraint = base_constraint.at([0, 5, 10])
    assert isinstance(nodal_constraint, NodalConstraint)
    assert nodal_constraint.constraint.is_convex is True

    # Canonicalize the nodal constraint
    canon_nodal = nodal_constraint.canonicalize()

    # Check that the inner constraint's convex flag is preserved
    assert isinstance(canon_nodal, NodalConstraint)
    assert canon_nodal.constraint.is_convex is True
    assert canon_nodal.nodes == [0, 5, 10]

    # The inner constraint should still be an Inequality
    assert isinstance(canon_nodal.constraint, Inequality)


# --- NodalConstraint: JAX Lowering ---
# Note: NodalConstraint is a wrapper that gets processed during preprocessing.
# It doesn't have direct JAX lowering - the wrapped constraint is extracted
# and applied at specific nodes during problem compilation.


# --- NodalConstraint: CVXPy Lowering ---
# Note: NodalConstraint is a wrapper that gets processed during preprocessing.
# It doesn't have direct CVXPy lowering - the wrapped constraint is extracted
# and applied at specific nodes during problem compilation.


# =============================================================================
# CrossNodeConstraint
# =============================================================================


# --- CrossNodeConstraint: Creation & Tree Structure ---


def test_cross_node_constraint_creation_basic():
    """Test basic CrossNodeConstraint creation."""
    x = State("x", shape=(3,))

    # Create a constraint with NodeReferences
    constraint = x.at(5) - x.at(4) <= 0.1

    # Wrap in CrossNodeConstraint
    cross_node = CrossNodeConstraint(constraint)

    assert isinstance(cross_node, CrossNodeConstraint)
    assert cross_node.constraint is constraint
    assert cross_node.is_convex is False


def test_cross_node_constraint_requires_constraint():
    """Test that CrossNodeConstraint only accepts Constraint objects."""
    x = State("x", shape=(3,))
    not_a_constraint = x.at(5) - x.at(4)  # Expression, not constraint

    with pytest.raises(TypeError, match="CrossNodeConstraint must wrap a Constraint"):
        CrossNodeConstraint(not_a_constraint)


def test_cross_node_constraint_children():
    """Test that CrossNodeConstraint.children() returns only the wrapped constraint."""
    x = State("x", shape=(2,))
    constraint = x.at(10) == x.at(0)
    cross_node = CrossNodeConstraint(constraint)

    children = cross_node.children()
    assert len(children) == 1
    assert children[0] is constraint


def test_cross_node_constraint_repr():
    """Test CrossNodeConstraint string representation."""
    x = State("x", shape=(2,))
    constraint = x.at(5) - x.at(4) <= 0.1
    cross_node = CrossNodeConstraint(constraint)

    repr_str = repr(cross_node)
    assert "CrossNodeConstraint" in repr_str
    assert "<=" in repr_str


def test_cross_node_constraint_convex_method():
    """Test that CrossNodeConstraint.convex() marks the constraint as convex."""
    x = State("x", shape=(3,))
    constraint = x.at(5) - x.at(4) <= 1.0

    # Non-convex by default
    cross_node = CrossNodeConstraint(constraint)
    assert cross_node.is_convex is False

    # Mark as convex
    cross_node_convex = cross_node.convex()
    assert cross_node_convex.is_convex is True
    assert isinstance(cross_node_convex, CrossNodeConstraint)


def test_cross_node_constraint_convex_at_creation():
    """Test creating convex CrossNodeConstraint from already-convex constraint."""
    x = State("x", shape=(3,))

    # Create convex constraint first, then wrap
    convex_constraint = (x.at(5) - x.at(4) <= 1.0).convex()
    cross_node = CrossNodeConstraint(convex_constraint)

    assert cross_node.is_convex is True


def test_cross_node_constraint_with_equality():
    """Test CrossNodeConstraint with equality constraints."""
    x = State("x", shape=(2,))

    # Periodic boundary condition
    constraint = x.at(0) == x.at(100)
    cross_node = CrossNodeConstraint(constraint)

    assert isinstance(cross_node.constraint, Equality)
    assert cross_node.is_convex is False


def test_cross_node_constraint_with_inequality():
    """Test CrossNodeConstraint with inequality constraints."""
    x = State("x", shape=(3,))

    # Rate limit
    constraint = x.at(10) - x.at(9) <= 0.5
    cross_node = CrossNodeConstraint(constraint)

    assert isinstance(cross_node.constraint, Inequality)


# --- CrossNodeConstraint: Shape Checking ---


def test_cross_node_constraint_shape_validation_scalar():
    """Test CrossNodeConstraint shape validation with scalar result."""
    x = State("x", shape=(1,))
    constraint = x.at(5) - x.at(4) <= 0.1
    cross_node = CrossNodeConstraint(constraint)

    # Should validate successfully
    shape = cross_node.check_shape()
    assert shape == ()  # CrossNodeConstraint produces scalar


def test_cross_node_constraint_shape_validation_vector():
    """Test CrossNodeConstraint shape validation with vector constraint."""
    x = State("x", shape=(3,))
    constraint = x.at(5) - x.at(4) <= np.ones(3) * 0.1
    cross_node = CrossNodeConstraint(constraint)

    # Should validate successfully (vector constraints are element-wise)
    shape = cross_node.check_shape()
    assert shape == ()


def test_cross_node_constraint_shape_mismatch_raises():
    """Test that CrossNodeConstraint detects shape mismatches in wrapped constraint."""
    x = State("x", shape=(2,))
    # 2-dim state compared with 3-dim constant
    constraint = x.at(5) == np.zeros(3)
    cross_node = CrossNodeConstraint(constraint)

    # Should raise due to wrapped constraint shape mismatch
    with pytest.raises(ValueError):
        cross_node.check_shape()


# --- CrossNodeConstraint: Canonicalization ---


def test_cross_node_constraint_canonicalization():
    """Test that CrossNodeConstraint canonicalizes its inner constraint."""
    from openscvx.symbolic.expr import Add

    x = State("x", shape=(2,))

    # Create constraint with non-canonical expression
    lhs = Add(x.at(5), Constant(np.array([1.0, 2.0])))
    rhs = Add(Constant(np.array([3.0, 4.0])), Constant(np.array([5.0, 6.0])))
    constraint = lhs <= rhs

    cross_node = CrossNodeConstraint(constraint)
    canon = cross_node.canonicalize()

    # Should still be CrossNodeConstraint
    assert isinstance(canon, CrossNodeConstraint)
    # Inner constraint should be canonicalized
    assert isinstance(canon.constraint, Inequality)


def test_cross_node_constraint_preserves_convex_flag():
    """Test that canonicalization preserves the convex flag."""
    x = State("x", shape=(3,))

    # Create convex constraint
    constraint = (x.at(5) - x.at(4) <= 1.0).convex()
    cross_node = CrossNodeConstraint(constraint)

    assert cross_node.is_convex is True

    # Canonicalize
    canon = cross_node.canonicalize()

    # Convex flag should be preserved
    assert canon.is_convex is True


# --- CrossNodeConstraint: JAX Lowering ---


def test_cross_node_constraint_jax_lowering():
    """Test that CrossNodeConstraint lowers to trajectory-level function."""
    import jax.numpy as jnp

    from openscvx.symbolic.lowerers.jax import JaxLowerer

    position = State("pos", shape=(2,))
    position._slice = slice(0, 2)

    # Constraint: position[5] - position[4] <= 0
    constraint = position.at(5) - position.at(4) <= 0
    cross_node = CrossNodeConstraint(constraint)

    jl = JaxLowerer()
    fn = jl.lower(cross_node)

    # Create fake trajectory
    X = jnp.arange(20).reshape(10, 2).astype(float)  # 10 nodes, 2-dim state
    U = jnp.zeros((10, 0))
    params = {}

    # CrossNodeConstraint visitor provides (X, U, params) signature
    result = fn(X, U, params)

    # Expected: X[5] - X[4] - 0 = [10, 11] - [8, 9] = [2, 2]
    expected = jnp.array([2.0, 2.0])

    assert result.shape == (2,)
    assert jnp.allclose(result, expected)


def test_cross_node_constraint_jax_lowering_scalar():
    """Test CrossNodeConstraint JAX lowering with scalar constraint."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Norm
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    position = State("pos", shape=(2,))
    position._slice = slice(0, 2)

    # Constraint: norm(position[5] - position[4]) <= 1.0
    constraint = Norm(position.at(5) - position.at(4)) <= 1.0
    cross_node = CrossNodeConstraint(constraint)

    jl = JaxLowerer()
    fn = jl.lower(cross_node)

    # Create fake trajectory
    X = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [4.5, 0.0],
            [5.0, 0.0],
            [5.5, 0.0],
            [6.0, 0.0],
            [6.5, 0.0],
        ]
    )
    U = jnp.zeros((10, 0))
    params = {}

    result = fn(X, U, params)

    # Expected: norm([4.5, 0] - [4, 0]) - 1.0 = norm([0.5, 0]) - 1.0 = 0.5 - 1.0 = -0.5
    expected = -0.5

    assert result.shape == ()
    assert jnp.allclose(result, expected)


def test_cross_node_constraint_jax_jacobians():
    """Test that Jacobians can be computed for CrossNodeConstraint."""
    import jax.numpy as jnp
    from jax import jacfwd

    from openscvx.symbolic.lowerers.jax import JaxLowerer

    position = State("pos", shape=(2,))
    position._slice = slice(0, 2)

    # Constraint: position[3] - position[2] <= 0
    constraint = position.at(3) - position.at(2) <= 0
    cross_node = CrossNodeConstraint(constraint)

    jl = JaxLowerer()
    fn = jl.lower(cross_node)

    # Create trajectory
    X = jnp.ones((5, 2))  # 5 nodes, 2-dim state
    U = jnp.zeros((5, 0))
    params = {}

    # Compute Jacobian wrt X
    grad_g_X = jacfwd(fn, argnums=0)(X, U, params)

    # Shape should be (constraint_dim, N, n_x) = (2, 5, 2)
    assert grad_g_X.shape == (2, 5, 2)

    # Jacobian should only be non-zero at nodes 2 and 3
    # At node 3: derivative is +1 (for each component)
    # At node 2: derivative is -1 (for each component)
    assert jnp.allclose(grad_g_X[0, 3, 0], 1.0)  # d(result[0])/d(X[3, 0])
    assert jnp.allclose(grad_g_X[0, 2, 0], -1.0)  # d(result[0])/d(X[2, 0])
    assert jnp.allclose(grad_g_X[1, 3, 1], 1.0)  # d(result[1])/d(X[3, 1])
    assert jnp.allclose(grad_g_X[1, 2, 1], -1.0)  # d(result[1])/d(X[2, 1])

    # All other nodes should have zero gradient
    assert jnp.allclose(grad_g_X[:, 0, :], 0.0)
    assert jnp.allclose(grad_g_X[:, 1, :], 0.0)
    assert jnp.allclose(grad_g_X[:, 4, :], 0.0)


# --- CrossNodeConstraint: CVXPy Lowering ---


def test_cross_node_constraint_cvxpy_lowering():
    """Test that CrossNodeConstraint lowers to CVXPy constraint."""
    import cvxpy as cp

    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    position = State("pos", shape=(2,))
    position._slice = slice(0, 2)

    # Constraint: position[5] - position[4] <= 0.1
    constraint = position.at(5) - position.at(4) <= 0.1
    cross_node = CrossNodeConstraint(constraint)

    # Create CVXPy variables - full trajectory
    x_cvx = cp.Variable((10, 2), name="x")
    variable_map = {"x": x_cvx}

    lowerer = CvxpyLowerer(variable_map)
    result = lowerer.lower(cross_node)

    assert isinstance(result, cp.Constraint)


def test_cross_node_constraint_cvxpy_equality():
    """Test CVXPy lowering of CrossNodeConstraint with equality."""
    import cvxpy as cp

    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    state = State("x", shape=(3,))
    state._slice = slice(0, 3)

    # Periodic boundary: state[0] == state[N-1]
    constraint = state.at(0) == state.at(9)
    cross_node = CrossNodeConstraint(constraint)

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}

    lowerer = CvxpyLowerer(variable_map)
    result = lowerer.lower(cross_node)

    assert isinstance(result, cp.Constraint)


# =============================================================================
# CTCS (Continuous-Time Constraint Satisfaction)
# =============================================================================


# --- CTCS: Creation & Tree Structure ---


def test_ctcs_wraps_constraint():
    """CTCS should wrap a Constraint object."""
    x = Variable("x", shape=(3,))
    constraint = x <= 1.0

    ctcs_constraint = CTCS(constraint)

    assert isinstance(ctcs_constraint, CTCS)
    assert ctcs_constraint.constraint is constraint
    assert ctcs_constraint.penalty == "squared_relu"  # default


def test_ctcs_requires_constraint():
    """CTCS should only accept Constraint objects."""
    x = Variable("x", shape=(3,))
    not_a_constraint = x + 1.0

    with pytest.raises(TypeError, match="CTCS must wrap a Constraint"):
        CTCS(not_a_constraint)


def test_ctcs_with_different_penalties():
    """CTCS should accept different penalty types."""
    x = Variable("x", shape=(3,))
    constraint = x >= 0.0

    ctcs_squared = CTCS(constraint, penalty="squared_relu")
    ctcs_huber = CTCS(constraint, penalty="huber")
    ctcs_smooth = CTCS(constraint, penalty="smooth_relu")

    assert ctcs_squared.penalty == "squared_relu"
    assert ctcs_huber.penalty == "huber"
    assert ctcs_smooth.penalty == "smooth_relu"


def test_ctcs_helper_function():
    """The ctcs() helper should create CTCS objects."""
    x = Variable("x", shape=(2,))
    constraint = x == np.array([1.0, 2.0])

    # Default penalty
    ctcs1 = ctcs(constraint)
    assert isinstance(ctcs1, CTCS)
    assert ctcs1.constraint is constraint
    assert ctcs1.penalty == "squared_relu"

    # Custom penalty
    ctcs2 = ctcs(constraint, penalty="huber")
    assert ctcs2.penalty == "huber"


def test_ctcs_children():
    """CTCS should return its constraint as its only child."""
    x = Variable("x", shape=(3,))
    constraint = x <= 5.0
    ctcs_constraint = CTCS(constraint)

    children = ctcs_constraint.children()
    assert len(children) == 1
    assert children[0] is constraint


def test_ctcs_repr():
    """CTCS should have a readable representation."""
    x = Variable("x", shape=(3,))
    constraint = x <= 1.5

    ctcs_default = CTCS(constraint)
    assert repr(ctcs_default) == "CTCS(Var('x') <= Const(1.5), penalty='squared_relu')"

    ctcs_huber = CTCS(constraint, penalty="huber")
    assert repr(ctcs_huber) == "CTCS(Var('x') <= Const(1.5), penalty='huber')"


def test_ctcs_traversal():
    """CTCS should be traversable like other expressions."""
    x = Variable("x", shape=(2,))
    y = Variable("y", shape=(2,))

    # Create a CTCS constraint with some arithmetic
    constraint = (x + y) <= 10.0
    ctcs_constraint = CTCS(constraint)

    visited = []

    def visit(node):
        visited.append(type(node).__name__)

    traverse(ctcs_constraint, visit)

    # Should visit CTCS -> Inequality -> Add -> Variable -> Variable -> Constant
    assert visited[0] == "CTCS"
    assert visited[1] == "Inequality"
    assert visited[2] == "Add"
    assert "Variable" in visited
    assert "Constant" in visited


def test_ctcs_with_equality_constraint():
    """CTCS should work with Equality constraints."""
    x = Variable("x", shape=(3,))
    constraint = x == np.zeros(3)

    ctcs_constraint = ctcs(constraint, penalty="smooth_relu")

    assert isinstance(ctcs_constraint.constraint, Equality)
    assert ctcs_constraint.penalty == "smooth_relu"


def test_multiple_ctcs_constraints():
    """Should be able to create multiple CTCS constraints."""
    x = Variable("x", shape=(2,))
    u = Variable("u", shape=(1,))

    # Different constraints with different penalties
    c1 = ctcs(x <= 1.0, penalty="squared_relu")
    c2 = ctcs(x >= -1.0, penalty="huber")
    c3 = ctcs(u == 0.0, penalty="smooth_relu")

    assert c1.penalty == "squared_relu"
    assert c2.penalty == "huber"
    assert c3.penalty == "smooth_relu"

    # Verify they wrap different constraints
    assert isinstance(c1.constraint, Inequality)
    assert isinstance(c2.constraint, Inequality)
    assert isinstance(c3.constraint, Equality)


def test_ctcs_pretty_print():
    """CTCS should integrate with pretty printing."""
    x = Variable("x", shape=(2,))
    constraint = x <= 5.0
    ctcs_constraint = CTCS(constraint)

    pretty = ctcs_constraint.pretty()
    lines = pretty.splitlines()

    assert lines[0].strip() == "CTCS"
    assert "Inequality" in lines[1]
    # Should show the tree structure
    assert "Variable" in pretty
    assert "Constant" in pretty


# --- CTCS: Penalty Expression ---


def test_ctcs_penalty_expr_method():
    """Test building penalty expressions from CTCS constraints."""
    x = Variable("x", shape=(2,))
    constraint = x <= 1.0

    # squared_relu penalty
    ctcs1 = CTCS(constraint, penalty="squared_relu")
    penalty1 = ctcs1.penalty_expr()
    assert isinstance(penalty1, Sum)
    assert isinstance(penalty1.operand, Square)
    assert isinstance(penalty1.operand.x, PositivePart)
    assert penalty1.operand.x.x is constraint.lhs

    # huber penalty
    ctcs2 = CTCS(constraint, penalty="huber")
    penalty2 = ctcs2.penalty_expr()
    assert isinstance(penalty2.operand, Huber)
    assert isinstance(penalty2.operand.x, PositivePart)
    assert penalty2.operand.x.x is constraint.lhs

    # smooth_relu penalty
    ctcs3 = CTCS(constraint, penalty="smooth_relu")
    penalty3 = ctcs3.penalty_expr()
    assert isinstance(penalty3.operand, SmoothReLU)
    assert penalty3.operand.x is constraint.lhs


def test_ctcs_unknown_penalty():
    """CTCS should raise error for unknown penalty types."""
    x = Variable("x", shape=(1,))
    constraint = x <= 0.0

    ctcs_constraint = CTCS(constraint, penalty="unknown")

    with pytest.raises(ValueError, match="Unknown penalty"):
        ctcs_constraint.penalty_expr()


# --- CTCS: Shape Checking ---


def test_ctcs_basic_shape_validation():
    """Test basic CTCS shape validation with penalty expression checking"""
    from openscvx.symbolic.expr import ctcs

    x = State("x", (3,))
    constraint = x <= np.ones((3,))
    wrapped = ctcs(constraint, penalty="squared_relu")

    # Should validate both constraint and penalty expression shapes
    wrapped.check_shape()


def test_ctcs_penalty_shape_consistency():
    """Test that penalty expressions have same shape as constraint LHS"""
    from openscvx.symbolic.expr import ctcs

    x = State("x", (2, 2))  # matrix state
    constraint = x >= np.zeros((2, 2))
    wrapped = ctcs(constraint, penalty="huber")

    wrapped.check_shape()

    # Penalty should have same shape as constraint LHS
    penalty_expr = wrapped.penalty_expr()
    penalty_shape = penalty_expr.check_shape()
    assert penalty_shape == ()


def test_ctcs_constraint_shape_mismatch_raises():
    """Test that CTCS catches underlying constraint shape mismatches"""
    from openscvx.symbolic.expr import ctcs

    x = State("x", (2,))
    # Create constraint with mismatched shapes
    constraint = x <= np.ones((3,))  # 2 vs 3 mismatch
    wrapped = ctcs(constraint)

    # Should raise due to underlying constraint shape mismatch
    with pytest.raises(ValueError):
        wrapped.check_shape()


# --- CTCS: Canonicalization ---


def test_ctcs_canonicalization_preserves_parameters():
    """Test that CTCS canonicalization preserves penalty, nodes, idx, and check_nodally."""
    from openscvx.symbolic.expr import Add

    x = Variable("x", shape=(2,))

    # Create constraint with non-canonical expression (will fold constants)
    lhs = Add(x, Constant(np.array([1.0, 2.0])))
    rhs = Add(Constant(np.array([3.0, 4.0])), Constant(np.array([5.0, 6.0])))
    constraint = lhs <= rhs

    # Create CTCS with all parameters
    ctcs_constraint = CTCS(constraint, penalty="huber", nodes=(5, 10), idx=3, check_nodally=True)

    # Canonicalize
    canon = ctcs_constraint.canonicalize()

    # Check that CTCS parameters are preserved
    assert isinstance(canon, CTCS)
    assert canon.penalty == "huber"
    assert canon.nodes == (5, 10)
    assert canon.idx == 3
    assert canon.check_nodally is True

    # Check that inner constraint was canonicalized
    assert isinstance(canon.constraint, Inequality)
    # The rhs constants should have been folded: [3,4] + [5,6] = [8,10]
    # Then moved to canonical form: (lhs - rhs) <= 0
    inner_rhs = canon.constraint.rhs
    assert isinstance(inner_rhs, Constant)
    assert np.array_equal(inner_rhs.value, 0)


def test_ctcs_canonicalization_recursive():
    """Test that CTCS canonicalization recursively canonicalizes the constraint."""
    from openscvx.symbolic.expr import Add, Mul

    x = Variable("x", shape=(3,))

    # Create expression with redundant operations that will be canonicalized
    # 2*x + 3*x should fold to 5*x
    lhs = Add(Mul(Constant(2.0), x), Mul(Constant(3.0), x))
    rhs = Constant(10.0)
    constraint = lhs <= rhs

    ctcs_constraint = CTCS(constraint, penalty="squared_relu")

    # Canonicalize
    canon = ctcs_constraint.canonicalize()

    # The inner constraint should be canonicalized
    assert isinstance(canon, CTCS)
    assert isinstance(canon.constraint, Inequality)

    # After canonicalization, the constraint should be in form: (canonical_lhs - rhs) <= 0
    # The Add(2*x, 3*x) should be flattened and constants combined


def test_ctcs_canonicalization_preserves_convex_flag():
    """Test that CTCS canonicalization preserves the convex flag on the inner constraint."""
    x = State("x", shape=(2,))

    # Create a convex constraint
    constraint = (x <= Constant(np.array([5.0, 10.0]))).convex()
    assert constraint.is_convex is True

    ctcs_constraint = CTCS(constraint, penalty="smooth_relu", nodes=(0, 20))

    # Canonicalize
    canon = ctcs_constraint.canonicalize()

    # The inner constraint's convex flag should be preserved
    assert canon.constraint.is_convex is True


# --- CTCS: JAX Lowering ---


def test_ctcs_constraint_can_be_lowered_directly():
    """Test that CTCS constraints can now be lowered directly with node context."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.array([1.0, 2.0, 3.0])

    # Create state variable
    state = State("x", (3,))
    state._slice = slice(0, 3)

    # Create constraint: x <= 2.0
    lhs = state
    rhs = Constant(np.array([2.0, 2.0, 2.0]))
    constraint = Inequality(lhs - rhs, Constant(np.array([0.0, 0.0, 0.0])))

    # Wrap in CTCS
    ctcs_constraint = CTCS(constraint, penalty="squared_relu")

    jl = JaxLowerer()
    fn = jl.lower(ctcs_constraint)

    # Should work without node context (always active)
    result = fn(x, None, None, None)

    # Expected: sum(max(x - 2, 0)^2) = sum([0, 0, 1]) = 1.0
    assert jnp.allclose(result, 1.0)
    assert result.shape == ()  # Should be scalar


def test_ctcs_penalty_expression_can_be_lowered():
    """Test that the penalty expression from CTCS can be lowered successfully."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.array([0.5, 1.5, 2.5])

    # Create state variable
    state = State("x", (3,))
    state._slice = slice(0, 3)

    # Create constraint: x <= 2.0
    lhs = state
    rhs = Constant(np.array([2.0, 2.0, 2.0]))
    constraint = Inequality(lhs - rhs, 0)

    # Wrap in CTCS
    ctcs_constraint = CTCS(constraint, penalty="squared_relu")

    # Extract the penalty expression (this is what would happen during augmentation)
    penalty_expr = ctcs_constraint.penalty_expr()

    # The penalty expression should be lowerable
    jl = JaxLowerer()
    fn = jl.lower(penalty_expr)

    # Execute the penalty function
    result = fn(x, None, None, None)

    # Expected: Sum(Square(PositivePart(x - 2.0))) = sum([0, 0, 0.25]) = 0.25
    # Only x[2] = 2.5 violates the constraint x <= 2.0
    expected = 0.25  # Scalar result from Sum
    assert jnp.allclose(result, expected)
    assert result.shape == ()  # Should be scalar


def test_ctcs_penalty_lowering_with_different_penalties():
    """Test that CTCS penalty expressions can be lowered with different penalty types."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.array([1.0, 2.0, 3.0])

    state = State("x", (3,))
    state._slice = slice(0, 3)

    # Constraint: x <= 1.5 (violations at x[1] and x[2])
    constraint = Inequality(state - Constant(np.array([1.5, 1.5, 1.5])), np.array([0, 0, 0]))

    # Test different penalty types
    penalties = ["squared_relu", "huber", "smooth_relu"]

    jl = JaxLowerer()

    for penalty_type in penalties:
        ctcs_constraint = CTCS(constraint, penalty=penalty_type)
        penalty_expr = ctcs_constraint.penalty_expr()

        # Should be able to lower without error
        fn = jl.lower(penalty_expr)
        result = fn(x, None, None, None)
        # Result should be a scalar (sum of all penalties)
        assert result.shape == ()  # Should be scalar
        # Total penalty should be positive since there are violations
        assert result > 0
        if penalty_type == "squared_relu":
            # Expected: 0^2 + 0.5^2 + 1.5^2 = 0 + 0.25 + 2.25 = 2.5
            expected = 0.25 + 2.25
            assert jnp.allclose(result, expected, rtol=1e-5)


def test_ctcs_with_node_range():
    """Test that CTCS constraints respect node ranges."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.array([3.0])  # Violates constraint x <= 2.0

    state = State("x", (1,))
    state._slice = slice(0, 1)

    # Constraint: x <= 2.0
    constraint = Inequality(state - Constant(np.array([2.0])), Constant(np.array([0.0])))

    # CTCS active only between nodes 5-10
    ctcs_constraint = CTCS(constraint, penalty="squared_relu", nodes=(5, 10))

    jl = JaxLowerer()
    fn = jl.lower(ctcs_constraint)

    # Test at different nodes
    result_node_3 = fn(x, None, 3, None)  # Before active range
    result_node_7 = fn(x, None, 7, None)  # Within active range
    result_node_12 = fn(x, None, 12, None)  # After active range

    # Should be zero outside active range
    assert jnp.allclose(result_node_3, 0.0)
    assert jnp.allclose(result_node_12, 0.0)

    # Should have penalty within active range
    # Expected: sum(max(3 - 2, 0)^2) = 1.0
    assert jnp.allclose(result_node_7, 1.0)


def test_ctcs_without_node_range_always_active():
    """Test that CTCS constraints without node range are always active."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.array([2.5])  # Violates constraint x <= 2.0

    state = State("x", (1,))
    state._slice = slice(0, 1)

    # Constraint: x <= 2.0
    constraint = Inequality(state - Constant(np.array([2.0])), Constant(np.array([0.0])))

    # CTCS without node range (always active)
    ctcs_constraint = CTCS(constraint, penalty="squared_relu")

    jl = JaxLowerer()
    fn = jl.lower(ctcs_constraint)

    # Test at different nodes - should always be active
    result_node_0 = fn(x, None, 0, None)
    result_node_50 = fn(x, None, 50, None)
    result_node_100 = fn(x, None, 100, None)

    # Should have same penalty at all nodes
    # Expected: sum(max(2.5 - 2, 0)^2) = 0.25
    expected = 0.25
    assert jnp.allclose(result_node_0, expected)
    assert jnp.allclose(result_node_50, expected)
    assert jnp.allclose(result_node_100, expected)


def test_ctcs_with_extra_kwargs():
    """Test that kwargs flow through all expression types."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Add, Control, Div, Mul, State, Sub
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.5, 1.0])

    # Create a complex expression involving multiple nodes
    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (2,))
    control._slice = slice(0, 2)

    # Complex expression: (x[0] + x[1]) * u[0] + x[2] / u[1] - 5.0
    expr = Sub(
        Add(Mul(Add(state[0], state[1]), control[0]), Div(state[2], control[1])), Constant(5.0)
    )

    jl = JaxLowerer()
    fn = jl.lower(expr)

    result = fn(x, u, node=10, params=None)

    # Expected: (1 + 2) * 0.5 + 3 / 1.0 - 5.0 = 1.5 + 3.0 - 5.0 = -0.5
    expected = -0.5
    assert jnp.allclose(result, expected)


# --- CTCS: CVXPy Lowering ---


def test_cvxpy_ctcs_not_implemented():
    """Test that CTCS raises NotImplementedError"""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    constraint = Inequality(x, Constant(np.array(1.0)))
    expr = ctcs(constraint)

    with pytest.raises(NotImplementedError, match="CTCS constraints are for continuous-time"):
        lowerer.lower(expr)
