import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.symbolic.augmentation import (
    augment_dynamics_with_ctcs,
    decompose_vector_nodal_constraints,
    separate_constraints,
    sort_ctcs_constraints,
)
from openscvx.symbolic.constraint_set import ConstraintSet
from openscvx.symbolic.expr import (
    CTCS,
    Add,
    Concat,
    Constant,
    Control,
    Huber,
    Index,
    Inequality,
    NodalConstraint,
    PositivePart,
    SmoothReLU,
    Square,
    State,
    Sub,
    Sum,
    ctcs,
)
from openscvx.symbolic.lower import lower_to_jax


def test_separate_constraints_empty():
    """Test separate_constraints with no constraints."""
    n_nodes = 10
    constraint_set = ConstraintSet()
    result = separate_constraints(constraint_set, n_nodes)

    assert result.ctcs == []
    assert result.nodal == []
    assert result.nodal_convex == []
    assert result.cross_node == []
    assert result.cross_node_convex == []
    assert result.is_categorized


def test_separate_constraints_only_ctcs():
    """Test separate_constraints with only CTCS constraints."""
    n_nodes = 10
    x = State("x", (1,))
    c1 = ctcs(x <= 1.0, penalty="squared_relu")
    c2 = ctcs(x >= 0.0, penalty="huber", check_nodally=True)

    constraint_set = ConstraintSet(unsorted=[c1, c2])
    result = separate_constraints(constraint_set, n_nodes)

    assert result.ctcs == [c1, c2]
    assert len(result.nodal) == 1  # Only c2 should be in nodal (check_nodally=True)
    assert len(result.nodal_convex) == 0  # No convex constraints
    # Should be converted to NodalConstraint
    assert hasattr(result.nodal[0], "constraint")
    assert result.nodal[0].constraint == c2.constraint
    assert result.is_categorized


def test_separate_constraints_only_nodal():
    """Test separate_constraints with only regular constraints."""
    n_nodes = 10
    x = State("x", (1,))
    c1 = x <= 1.0
    c2 = x >= 0.0

    constraint_set = ConstraintSet(unsorted=[c1, c2])
    result = separate_constraints(constraint_set, n_nodes)

    assert result.ctcs == []
    assert len(result.nodal) == 2
    assert len(result.nodal_convex) == 0  # No convex constraints
    # Should be converted to NodalConstraint objects
    assert hasattr(result.nodal[0], "constraint")
    assert hasattr(result.nodal[1], "constraint")
    assert result.nodal[0].constraint == c1
    assert result.nodal[1].constraint == c2
    assert result.is_categorized


def test_separate_constraints_mixed():
    """Test separate_constraints with mixed constraint types."""
    n_nodes = 10
    x = State("x", (1,))
    c1 = ctcs(x <= 1.0, penalty="squared_relu")
    c2 = x >= 0.0  # Regular constraint
    c3 = ctcs(x <= 2.0, penalty="huber", check_nodally=True)

    constraint_set = ConstraintSet(unsorted=[c1, c2, c3])
    result = separate_constraints(constraint_set, n_nodes)

    assert result.ctcs == [c1, c3]
    assert len(result.nodal) == 2  # c2 + c3's underlying constraint
    assert len(result.nodal_convex) == 0  # No convex constraints
    # Should be converted to NodalConstraint objects
    assert result.nodal[0].constraint == c2
    assert result.nodal[1].constraint == c3.constraint
    assert result.is_categorized


def test_separate_constraints_invalid_type():
    """Test separate_constraints with invalid constraint type."""
    n_nodes = 10
    x = State("x", (1,))
    invalid = Add(x, Constant(1.0))  # Not a constraint

    constraint_set = ConstraintSet(unsorted=[invalid])
    with pytest.raises(ValueError) as exc:
        separate_constraints(constraint_set, n_nodes)

    assert "Constraints must be `Constraint`, `NodalConstraint`, or `CTCS`" in str(exc.value)


def test_separate_constraints_convex_constraints():
    """Test separate_constraints with convex constraints."""
    n_nodes = 10
    x = State("x", (1,))

    # Create convex and non-convex constraints
    c1 = (x <= 1.0).convex()  # Convex constraint
    c2 = x >= 0.0  # Non-convex constraint
    c3 = (x <= 2.0).convex()  # Another convex constraint

    constraint_set = ConstraintSet(unsorted=[c1, c2, c3])
    result = separate_constraints(constraint_set, n_nodes)

    assert result.ctcs == []
    assert len(result.nodal) == 1  # Only c2 (non-convex)
    assert len(result.nodal_convex) == 2  # c1 and c3 (convex)

    # Verify non-convex constraint
    assert result.nodal[0].constraint == c2
    assert not result.nodal[0].constraint.is_convex

    # Verify convex constraints
    assert result.nodal_convex[0].constraint == c1
    assert result.nodal_convex[0].constraint.is_convex
    assert result.nodal_convex[1].constraint == c3
    assert result.nodal_convex[1].constraint.is_convex
    assert result.is_categorized


def test_separate_constraints_convex_nodal_constraints():
    """Test separate_constraints with convex NodalConstraint objects."""
    n_nodes = 10
    x = State("x", (1,))

    # Create convex and non-convex constraints, then wrap in NodalConstraint
    c1 = (x <= 1.0).convex()
    c2 = x >= 0.0
    nodal1 = NodalConstraint(c1, [0, 1, 2])  # Convex
    nodal2 = NodalConstraint(c2, [1, 2, 3])  # Non-convex

    constraint_set = ConstraintSet(unsorted=[nodal1, nodal2])
    result = separate_constraints(constraint_set, n_nodes)

    assert result.ctcs == []
    assert len(result.nodal) == 1  # nodal2 (non-convex)
    assert len(result.nodal_convex) == 1  # nodal1 (convex)

    # Verify the NodalConstraint objects are preserved
    assert result.nodal[0] is nodal2
    assert result.nodal_convex[0] is nodal1
    assert result.is_categorized


def test_separate_constraints_convex_ctcs_check_nodally():
    """Test separate_constraints with convex CTCS constraints that have check_nodally=True."""
    n_nodes = 10
    x = State("x", (1,))

    # Create CTCS constraints with convex underlying constraints
    c1 = ctcs((x <= 1.0).convex(), penalty="squared_relu", check_nodally=True)
    c2 = ctcs(x >= 0.0, penalty="huber", check_nodally=True)  # Non-convex

    constraint_set = ConstraintSet(unsorted=[c1, c2])
    result = separate_constraints(constraint_set, n_nodes)

    assert result.ctcs == [c1, c2]
    assert len(result.nodal) == 1  # c2's underlying constraint (non-convex)
    assert len(result.nodal_convex) == 1  # c1's underlying constraint (convex)

    # Verify the underlying constraints from CTCS are correctly separated
    assert result.nodal[0].constraint == c2.constraint
    assert not result.nodal[0].constraint.is_convex
    assert result.nodal_convex[0].constraint == c1.constraint
    assert result.nodal_convex[0].constraint.is_convex
    assert result.is_categorized


def test_ctcs_with_node_reference_raises_error():
    """Test that CTCS constraints cannot contain NodeReferences."""
    n_nodes = 10
    x = State("x", (3,))

    # Try to create a CTCS constraint with NodeReference - should fail
    cross_node_constraint = x.at(5) - x.at(4) <= 1.0
    ctcs_constraint = ctcs(cross_node_constraint, penalty="squared_relu")

    constraint_set = ConstraintSet(unsorted=[ctcs_constraint])
    with pytest.raises(ValueError, match="CTCS constraints cannot contain NodeReferences"):
        separate_constraints(constraint_set, n_nodes)


def test_augment_no_ctcs_constraints():
    """Test augmentation with no constraints."""
    x = State("x", (2,))
    x.final = np.array([0.0, 0.0])
    time = State("time", (1,))
    time.final = np.array([10.0])
    xdot = Add(x, Constant(np.ones(2)))
    states = [x, time]
    controls = []
    N = 1
    time.guess = np.linspace(0, time.final[0], N).reshape(-1, 1)

    xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        xdot,
        states,
        controls,
        [],  # No CTCS constraints
        N,
    )

    # No augmentation should occur for dynamics, but time dilation should be added
    assert xdot_aug is xdot
    assert len(states_aug) == 2
    assert states_aug[0] is x
    assert states_aug[1] is time
    assert len(controls_aug) == 1
    assert controls_aug[0].name == "_time_dilation"
    assert controls_aug[0].shape == (1,)


def test_augment_single_ctcs_constraint():
    """Test augmentation with a single CTCS constraint."""
    x = State("x", (2,))
    x.final = np.array([0.0, 0.0])
    time = State("time", (1,))
    time.final = np.array([10.0])
    u = Control("u", (1,))
    xdot = Add(x, u)
    states = [x, time]
    controls = [u]
    N = 1
    time.guess = np.linspace(0, time.final[0], N).reshape(-1, 1)

    # CTCS constraint
    constraint = ctcs(x[0] <= 1.0, penalty="squared_relu")

    xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        xdot,
        states,
        controls,
        [constraint],  # Pass CTCS constraint directly
        N,
    )

    # Should have augmented dynamics and new State
    assert isinstance(xdot_aug, Concat)
    assert len(states_aug) == 3  # original + time + 1 augmented
    assert states_aug[0] is x
    assert states_aug[1] is time
    assert isinstance(states_aug[2], State)
    assert states_aug[2].name == "_ctcs_aug_0"
    assert states_aug[2].shape == (1,)

    # Should have original control + time dilation
    assert len(controls_aug) == 2
    assert controls_aug[0] is u
    assert controls_aug[1].name == "_time_dilation"


def test_augment_multiple_ctcs_constraints():
    """Test augmentation with multiple CTCS constraints."""
    x = State("x", (3,))
    x.final = np.array([0.0, 0.0, 0.0])
    time = State("time", (1,))
    time.final = np.array([10.0])
    xdot = x * 2.0
    states = [x, time]
    controls = []
    N = 1
    time.guess = np.linspace(0, time.final[0], N).reshape(-1, 1)

    # Multiple CTCS constraints with different penalties
    c1 = ctcs(x[0] <= 1.0, penalty="squared_relu")
    c2 = ctcs(x[1] >= -1.0, penalty="huber")
    c3 = ctcs(x[2] == 0.0, penalty="smooth_relu")

    xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        xdot,
        states,
        controls,
        [c1, c2, c3],
        N,
    )

    # Should have 1 augmented State (all penalties summed together)
    assert isinstance(xdot_aug, Concat)
    assert len(states_aug) == 3  # original + time + 1 augmented State
    assert states_aug[0] is x
    assert states_aug[1] is time

    # Check augmented State
    assert isinstance(states_aug[2], State)
    assert states_aug[2].name == "_ctcs_aug_0"
    assert states_aug[2].shape == (1,)

    # Should have time dilation control
    assert len(controls_aug) == 1
    assert controls_aug[0].name == "_time_dilation"

    # Check that the augmented dynamics contains the Add expression
    penalty_expr = xdot_aug.exprs[1]
    assert isinstance(penalty_expr, Add)


def test_augment_penalty_expression_structure():
    """Test that the penalty expressions are correctly structured."""
    x = State("x", (1,))
    x.final = np.array([0.0])
    time = State("time", (1,))
    time.final = np.array([10.0])
    xdot = x
    states = [time, x]  # time first since it was at index 0
    controls = []
    N = 1
    time.guess = np.linspace(0, time.final[0], N).reshape(-1, 1)

    # Create CTCS with squared_relu penalty
    constraint = ctcs(x <= 1.0, penalty="squared_relu")

    xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        xdot,
        states,
        controls,
        [constraint],
        N,
    )

    # Check structure of augmented dynamics
    assert isinstance(xdot_aug, Concat)
    assert len(xdot_aug.exprs) == 2

    # The second expression should be the CTCS constraint (not unwrapped)
    ctcs_expr = xdot_aug.exprs[1]
    assert isinstance(ctcs_expr, CTCS)

    # Check the penalty expression inside the CTCS wrapper
    penalty_expr = ctcs_expr.penalty_expr()
    assert isinstance(penalty_expr, Sum)
    assert isinstance(penalty_expr.operand, Square)
    assert isinstance(penalty_expr.operand.x, PositivePart)

    # Should have time dilation control
    assert len(controls_aug) == 1
    assert controls_aug[0].name == "_time_dilation"


def test_augment_single_penalty_no_add():
    """Test that single penalty doesn't create unnecessary Add expression."""
    x = State("x", (1,))
    x.final = np.array([0.0])
    time = State("time", (1,))
    time.final = np.array([10.0])
    xdot = x
    states = [time, x]  # time first since it was at index 0
    controls = []
    N = 1
    time.guess = np.linspace(0, time.final[0], N).reshape(-1, 1)

    constraint = ctcs(x <= 1.0, penalty="squared_relu")

    xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        xdot,
        states,
        controls,
        [constraint],
        N,
    )

    # Single penalty should not be wrapped in Add
    # But it should be wrapped in CTCS
    ctcs_expr = xdot_aug.exprs[1]
    assert isinstance(ctcs_expr, CTCS)  # CTCS wrapper preserved
    penalty_expr = ctcs_expr.penalty_expr()
    assert isinstance(penalty_expr, Sum)  # Direct penalty, not Add


def test_augment_multiple_penalties_create_add():
    """Test that multiple penalties create an Add expression."""
    x = State("x", (2,))
    x.final = np.array([0.0, 0.0])
    time = State("time", (1,))
    time.final = np.array([10.0])
    xdot = x
    states = [x, time]
    controls = []
    N = 1
    time.guess = np.linspace(0, time.final[0], N).reshape(-1, 1)

    c1 = ctcs(x[0] <= 1.0, penalty="squared_relu")
    c2 = ctcs(x[1] <= 2.0, penalty="huber")

    xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        xdot,
        states,
        controls,
        [c1, c2],
        N,
    )

    # Multiple penalties should be wrapped in Add
    penalty_expr = xdot_aug.exprs[1]
    assert isinstance(penalty_expr, Add)
    assert len(penalty_expr.terms) == 2


def test_augment_empty_states_list():
    """Test augmentation with just time state."""
    # Create a time state
    time = State("time", (1,))
    time.final = np.array([10.0])
    xdot = Constant(np.array([1.0, 2.0]))
    states = [time]
    controls = []
    N = 1
    time.guess = np.linspace(0, time.final[0], N).reshape(-1, 1)

    # CTCS constraint on a constant (unusual but valid)
    constraint = ctcs(Constant(1.0) <= 2.0)

    xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        xdot,
        states,
        controls,
        [constraint],
        N,
    )

    # Should create one augmented State (plus the time state)
    assert len(states_aug) == 2
    assert states_aug[0] is time
    assert isinstance(states_aug[1], State)
    assert states_aug[1].name == "_ctcs_aug_0"

    # Should have time dilation control
    assert len(controls_aug) == 1
    assert controls_aug[0].name == "_time_dilation"


def test_augment_with_different_penalties():
    """Test that different penalty types are correctly applied and summed."""
    x = State("x", (1,))
    x.final = np.array([0.0])
    time = State("time", (1,))
    time.final = np.array([10.0])
    xdot = x
    states = [time, x]  # time first since it was at index 0
    controls = []
    N = 1
    time.guess = np.linspace(0, time.final[0], N).reshape(-1, 1)

    penalties = ["squared_relu", "huber", "smooth_relu"]
    constraints = [ctcs(x <= float(i), penalty=p) for i, p in enumerate(penalties)]

    xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        xdot,
        states,
        controls,
        constraints,
        N,
    )

    # Should have 1 augmented State with summed penalties (time + x + augmented)
    assert len(states_aug) == 3
    assert isinstance(xdot_aug, Concat)
    assert len(xdot_aug.exprs) == 2  # original + summed penalties

    # The penalty expression should be an Add combining all penalties
    penalty_expr = xdot_aug.exprs[1]
    assert isinstance(penalty_expr, Add)
    assert len(penalty_expr.terms) == 3  # Three penalty terms

    # Should have time dilation control
    assert len(controls_aug) == 1
    assert controls_aug[0].name == "_time_dilation"


def test_augment_preserves_original_controls():
    """Test that original controls are preserved and time dilation is added."""
    x = State("x", (1,))
    x.final = np.array([0.0])
    time = State("time", (1,))
    time.final = np.array([10.0])
    u1 = Control("u1", (2,))
    u2 = Control("u2", (3,))
    xdot = x
    states = [time, x]  # time first since it was at index 0
    controls = [u1, u2]
    N = 1
    time.guess = np.linspace(0, time.final[0], N).reshape(-1, 1)

    constraint = ctcs(x <= 1.0, penalty="squared_relu")

    xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        xdot,
        states,
        controls,
        [constraint],
        N,
    )

    # Should preserve original controls and add time dilation
    assert len(controls_aug) == 3
    assert controls_aug[0] is u1
    assert controls_aug[1] is u2
    assert controls_aug[2].name == "_time_dilation"
    assert controls_aug[2].shape == (1,)


# Keep all the penalty behavior tests unchanged since they don't depend on the function signature
def test_inequality_constraint_penalty_activation():
    """Test that penalties are zero when x <= limit and positive when x > limit."""

    limit = 1.0
    test_cases = [
        (-2.0, False),  # Well below limit, no violation
        (0.0, False),  # Below limit, no violation
        (0.999, False),  # Just below limit, no violation
        (1.0, False),  # At limit, no violation
        (1.001, True),  # Just above limit, violation
        (2.0, True),  # Above limit, violation
        (5.0, True),  # Well above limit, violation
    ]

    for penalty_name in ["squared_relu", "huber", "smooth_relu"]:
        prev_violation_penalty = None

        for val, should_violate in test_cases:
            x = jnp.array([val])
            state = State("x", (1,))
            state._slice = slice(0, 1)

            # Scalar constraint: x <= limit
            constraint = state - limit <= 0

            ctcs = CTCS(constraint, penalty=penalty_name)
            penalty_expr = ctcs.penalty_expr()

            fn = lower_to_jax(penalty_expr)
            result = fn(x, None, None, None)

            if should_violate:
                assert result > 0, (
                    f"{penalty_name}: x={val} violates constraint, should have positive penalty"
                )

                # Check monotonic increase for violations
                if prev_violation_penalty is not None:
                    assert result > prev_violation_penalty, (
                        f"{penalty_name}: Penalty should increase with violation magnitude"
                    )
                prev_violation_penalty = result
            else:
                assert jnp.abs(result) < 1e-6, (
                    f"{penalty_name}: x={val} satisfies constraint, should have zero penalty"
                )


def test_reverse_inequality_constraint_penalty():
    """Test penalties for x >= limit (satisfied when x >= limit)."""
    # Constraint: x >= 2.0, equivalent to 2.0 <= x
    limit = 2.0

    x_vals = jnp.array([0.0, 1.0, 1.999, 2.0, 2.001, 3.0, 5.0])

    state = State("x", (7,))
    state._slice = slice(0, 7)

    # Build constraint violation: limit - x (positive when x < limit)
    violation = Sub(Constant(limit), state)

    # Test with squared ReLU penalty
    penalty = Square(PositivePart(violation))
    fn = lower_to_jax(penalty)
    result = fn(x_vals, None, None, None)

    # Violated constraints (x < 2.0) should have positive penalty
    assert result[0] > 0, "x=0 violates x>=2, should have positive penalty"
    assert result[1] > 0, "x=1 violates x>=2, should have positive penalty"
    assert result[2] > 0, "x=1.999 violates x>=2, should have positive penalty"

    # Satisfied constraints (x >= 2.0) should have zero penalty
    assert jnp.allclose(result[3], 0.0, atol=1e-10), (
        "x=2.0 satisfies x>=2, should have zero penalty"
    )
    assert jnp.allclose(result[4], 0.0), "x=2.001 satisfies x>=2, should have zero penalty"
    assert jnp.allclose(result[5], 0.0), "x=3 satisfies x>=2, should have zero penalty"
    assert jnp.allclose(result[6], 0.0), "x=5 satisfies x>=2, should have zero penalty"


def test_box_constraint_penalties():
    """Test penalties for box constraints: lower <= x <= upper."""
    lower = 1.0
    upper = 3.0

    # Test values spanning the range
    x_vals = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0])

    state = State("x", (7,))
    state._slice = slice(0, 7)

    # Lower bound violation: lower - x (positive when x < lower)
    lower_violation = Sub(Constant(lower), state)
    lower_penalty = Square(PositivePart(lower_violation))

    # Upper bound violation: x - upper (positive when x > upper)
    upper_violation = Sub(state, Constant(upper))
    upper_penalty = Square(PositivePart(upper_violation))

    fn_lower = lower_to_jax(lower_penalty)
    fn_upper = lower_to_jax(upper_penalty)

    result_upper = fn_upper(x_vals, None, None, None)
    result_lower = fn_lower(x_vals, None, None, None)

    # Test lower bound
    assert result_lower[0] > 0, "x=0 violates lower bound"
    assert result_lower[1] > 0, "x=0.5 violates lower bound"
    assert jnp.allclose(result_lower[2], 0.0, atol=1e-10), "x=1.0 at lower bound"
    assert jnp.allclose(result_lower[3], 0.0), "x=2.0 within bounds"
    assert jnp.allclose(result_lower[4], 0.0), "x=3.0 at upper bound"

    # Test upper bound
    assert jnp.allclose(result_upper[2], 0.0), "x=1.0 at lower bound"
    assert jnp.allclose(result_upper[3], 0.0), "x=2.0 within bounds"
    assert jnp.allclose(result_upper[4], 0.0, atol=1e-10), "x=3.0 at upper bound"
    assert result_upper[5] > 0, "x=3.5 violates upper bound"
    assert result_upper[6] > 0, "x=4.0 violates upper bound"

    # Total penalty (sum of both)
    total_penalty = result_lower + result_upper

    # Only values within [1, 3] should have zero total penalty
    assert total_penalty[0] > 0, "x=0 outside bounds"
    assert total_penalty[1] > 0, "x=0.5 outside bounds"
    assert jnp.allclose(total_penalty[2], 0.0, atol=1e-10), "x=1.0 on boundary"
    assert jnp.allclose(total_penalty[3], 0.0), "x=2.0 inside bounds"
    assert jnp.allclose(total_penalty[4], 0.0, atol=1e-10), "x=3.0 on boundary"
    assert total_penalty[5] > 0, "x=3.5 outside bounds"
    assert total_penalty[6] > 0, "x=4.0 outside bounds"


def test_norm_constraint_penalty():
    """Test penalty for ||x|| <= r constraint."""
    radius = 2.0

    # Test 2D points at various distances from origin
    points = np.array(
        [
            [0.0, 0.0],  # Inside, distance = 0
            [1.0, 0.0],  # Inside, distance = 1
            [1.4, 1.4],  # Inside, distance ≈ 1.98
            [2.0, 0.0],  # On boundary, distance = 2
            [1.5, 1.5],  # Outside, distance ≈ 2.12
            [3.0, 0.0],  # Outside, distance = 3
            [3.0, 4.0],  # Outside, distance = 5
        ]
    )

    state = State("x", (2,))
    state._slice = slice(0, 2)

    # For each point, compute ||x||^2 - r^2 (positive when violated)
    results = []
    for point in points:
        x_vals = jnp.array(point)

        # Build constraint violation: ||x||^2 - r^2
        x_squared = Square(state)  # Element-wise square
        norm_squared = x_squared[0] + x_squared[1]  # Sum to get ||x||^2
        violation = Sub(norm_squared, Constant(radius**2))

        penalty = Square(PositivePart(violation))
        fn = lower_to_jax(penalty)
        result = fn(x_vals, None, None, None)
        results.append(result)

    # Points inside or on the circle should have zero penalty
    assert jnp.allclose(results[0], 0.0), "Origin should have zero penalty"
    assert jnp.allclose(results[1], 0.0), "Point at distance 1 should have zero penalty"
    assert jnp.allclose(results[2], 0.0, atol=1e-6), (
        "Point at distance ~1.98 should have zero penalty"
    )
    assert jnp.allclose(results[3], 0.0, atol=1e-10), "Point on boundary should have zero penalty"

    # Points outside the circle should have positive penalty
    assert results[4] > 0, "Point at distance ~2.12 should have positive penalty"
    assert results[5] > 0, "Point at distance 3 should have positive penalty"
    assert results[6] > 0, "Point at distance 5 should have positive penalty"

    # Penalty should increase with distance from boundary
    assert results[6] > results[5] > results[4], "Penalty should increase with violation"


def test_penalty_gradients_at_boundary():
    """Test that penalties have correct gradient behavior at constraint boundaries."""
    # Test points very close to constraint boundary x <= 1.0
    x_vals = jnp.array([0.9999, 0.99999, 1.0, 1.00001, 1.0001])

    state = State("x", (5,))
    state._slice = slice(0, 5)

    violation = Sub(state, Constant(1.0))

    # Squared ReLU has a sharp transition at the boundary
    squared_penalty = Square(PositivePart(violation))
    fn_squared = lower_to_jax(squared_penalty)
    result_squared = fn_squared(x_vals, None, None, None)

    # Should be exactly zero below boundary, positive above
    assert jnp.allclose(result_squared[0], 0.0)
    assert jnp.allclose(result_squared[1], 0.0)
    assert jnp.allclose(result_squared[2], 0.0, atol=1e-12)
    assert result_squared[3] > 0
    assert result_squared[4] > 0

    # SmoothReLU should have smooth transition
    smooth_penalty = SmoothReLU(violation, c=1e-4)
    fn_smooth = lower_to_jax(smooth_penalty)
    result_smooth = fn_smooth(x_vals, None, None, None)

    # Should be approximately zero below, smoothly increasing above
    assert jnp.allclose(result_smooth[0], 0.0, atol=1e-8)
    assert jnp.allclose(result_smooth[1], 0.0, atol=1e-8)
    assert jnp.allclose(result_smooth[2], 0.0, atol=1e-8)
    assert result_smooth[3] > 0
    assert result_smooth[4] > result_smooth[3]


def test_equality_constraint_penalty():
    """Test penalties for equality constraints x == target."""
    target = 2.0

    # Test values around the target
    x_vals = jnp.array([0.0, 1.0, 1.9, 2.0, 2.1, 3.0, 4.0])

    state = State("x", (7,))
    state._slice = slice(0, 7)

    # For equality, we need to penalize |x - target|
    # This requires penalties on both positive and negative deviations
    deviation = Sub(state, Constant(target))

    # Test with Huber penalty (naturally handles both sides)
    penalty = Huber(deviation, delta=0.5)
    fn = lower_to_jax(penalty)
    result = fn(x_vals, None, None, None)

    # Only x == target should have zero penalty
    assert result[0] > 0, "x=0 != 2 should have positive penalty"
    assert result[1] > 0, "x=1 != 2 should have positive penalty"
    assert result[2] > 0, "x=1.9 != 2 should have positive penalty"
    assert jnp.allclose(result[3], 0.0, atol=1e-10), "x=2 == 2 should have zero penalty"
    assert result[4] > 0, "x=2.1 != 2 should have positive penalty"
    assert result[5] > 0, "x=3 != 2 should have positive penalty"
    assert result[6] > 0, "x=4 != 2 should have positive penalty"

    # Penalty should increase with distance from target
    assert result[0] > result[1], "Farther from target should have higher penalty"
    assert result[1] > result[2], "Closer to target should have lower penalty"
    assert result[4] < result[5], "Farther from target should have higher penalty"
    assert result[5] < result[6], "Even farther should have even higher penalty"


def test_augmented_state_bounds():
    """Test that augmented state has correct min/max bounds."""
    x = State("x", (1,))
    x.final = np.array([0.0])
    time = State("time", (1,))
    time.final = np.array([10.0])
    xdot = x
    states = [time, x]
    controls = []
    N = 5
    time.guess = np.linspace(0, time.final[0], N).reshape(-1, 1)

    constraint = ctcs(x <= 1.0, penalty="squared_relu")

    xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        xdot,
        states,
        controls,
        [constraint],
        N,
        licq_min=0.001,
        licq_max=0.01,
    )

    # Check augmented state bounds (third state after time and x)
    aug_state = states_aug[2]  # third state is the augmented one
    assert aug_state.min[0] == 0.001, "Augmented state min should match licq_min"
    assert aug_state.max[0] == 0.01, "Augmented state max should match licq_max"
    assert aug_state.initial[0] == 0.001, "Augmented state initial should be licq_min"
    assert aug_state.guess.shape == (N, 1), "Augmented state guess should have correct shape"
    assert np.allclose(aug_state.guess, 0.001), "Augmented state guess should be licq_min"


def test_time_dilation_control_bounds():
    """Test that time dilation control has correct min/max bounds based on time."""
    x = State("x", (2,))
    x.final = np.array([0.0, 0.0])
    time = State("time", (1,))
    time.final = np.array([15.0])
    xdot = x
    states = [x, time]
    controls = []
    N = 3
    time.guess = np.linspace(0, time.final[0], N).reshape(-1, 1)

    constraint = ctcs(x[0] <= 1.0, penalty="squared_relu")

    xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        xdot,
        states,
        controls,
        [constraint],
        N,
        time_dilation_factor_min=0.5,
        time_dilation_factor_max=2.5,
    )

    # Check time dilation control bounds
    time_dilation = controls_aug[0]  # should be the only control
    expected_min = 0.5 * 15.0  # min_factor * time_final
    expected_max = 2.5 * 15.0  # max_factor * time_final
    expected_guess = 15.0  # time_final

    assert time_dilation.min[0] == expected_min, (
        f"Expected min {expected_min}, got {time_dilation.min[0]}"
    )
    assert time_dilation.max[0] == expected_max, (
        f"Expected max {expected_max}, got {time_dilation.max[0]}"
    )
    assert time_dilation.guess.shape == (N, 1), "Time dilation guess should have correct shape"
    assert np.allclose(time_dilation.guess, expected_guess), (
        f"Expected guess {expected_guess}, got {time_dilation.guess}"
    )


def test_ctcs_idx_grouping_auto_assignment():
    """Test automatic idx assignment for constraints without explicit idx."""
    x = State("x", (2,))

    c1 = ctcs(x[0] <= 1.0, nodes=(0, 5))  # No idx specified
    c2 = ctcs(x[1] <= 2.0, nodes=(0, 5))  # Same nodes, should get same idx
    c3 = ctcs(x[0] <= 3.0, nodes=(3, 8))  # Different nodes, should get different idx

    constraints = [c1, c2, c3]
    sorted_constraints, node_intervals, num_states = sort_ctcs_constraints(constraints)

    # Should have 2 groups: (0,5) and (3,8)
    assert num_states == 2
    assert node_intervals == [(0, 5), (3, 8)]

    # c1 and c2 should have same idx (0), c3 should have idx 1
    assert c1.idx == c2.idx == 0
    assert c3.idx == 1


def test_ctcs_idx_grouping_explicit_assignment():
    """Test explicit idx assignment and validation."""
    x = State("x", (2,))

    c1 = ctcs(x[0] <= 1.0, nodes=(0, 5), idx=1)  # Explicit idx
    c2 = ctcs(x[1] <= 2.0, nodes=(0, 5), idx=1)  # Same nodes, same idx - OK
    c3 = ctcs(x[0] <= 3.0, nodes=(3, 8), idx=0)  # Different nodes, different idx - OK

    constraints = [c1, c2, c3]
    sorted_constraints, node_intervals, num_states = sort_ctcs_constraints(constraints)

    # Should have 2 groups with correct ordering
    assert num_states == 2
    assert node_intervals == [(3, 8), (0, 5)]  # idx 0, then idx 1

    assert c1.idx == c2.idx == 1
    assert c3.idx == 0


def test_ctcs_idx_grouping_mixed_assignment():
    """Test mixed explicit and auto idx assignment."""
    x = State("x", (3,))

    c1 = ctcs(x[0] <= 1.0, nodes=(0, 5))  # Auto - should get idx 0
    c2 = ctcs(x[1] <= 2.0, nodes=(0, 5), idx=2)  # Explicit idx 2
    c3 = ctcs(x[2] <= 3.0, nodes=(3, 8))  # Auto - should get idx 1

    constraints = [c1, c2, c3]
    sorted_constraints, node_intervals, num_states = sort_ctcs_constraints(constraints)

    # Should have 3 groups: auto-assigned 0, auto-assigned 1, explicit 2
    assert num_states == 3
    assert node_intervals == [(0, 5), (3, 8), (0, 5)]  # idx 0, 1, 2

    assert c1.idx == 0  # Auto-assigned to same interval as c2
    assert c2.idx == 2  # Explicit
    assert c3.idx == 1  # Auto-assigned


def test_ctcs_idx_validation_errors():
    """Test validation errors for invalid idx usage."""
    x = State("x", (2,))

    # Test: same idx with different node intervals
    c1 = ctcs(x[0] <= 1.0, nodes=(0, 5), idx=0)
    c2 = ctcs(x[1] <= 2.0, nodes=(3, 8), idx=0)  # Different nodes, same idx - ERROR

    with pytest.raises(
        ValueError, match="idx=0 was first used with interval.*but now you gave it interval"
    ):
        sort_ctcs_constraints([c1, c2])

    # Test: non-contiguous idx values
    c3 = ctcs(x[0] <= 1.0, nodes=(0, 5), idx=0)
    c4 = ctcs(x[1] <= 2.0, nodes=(3, 8), idx=2)  # Gap: missing idx=1

    with pytest.raises(ValueError, match="must form a contiguous block starting from 0"):
        sort_ctcs_constraints([c3, c4])


def test_ctcs_multiple_augmented_states():
    """Test augmentation creates multiple augmented states for different idx groups."""
    x = State("x", (2,))
    x.final = np.array([0.0, 0.0])
    time = State("time", (1,))
    time.final = np.array([10.0])
    xdot = x
    states = [x, time]
    controls = []
    N = 1
    time.guess = np.linspace(0, time.final[0], N).reshape(-1, 1)

    # Create constraints with different node intervals (different idx groups)
    c1 = ctcs(x[0] <= 1.0, nodes=(0, 5), idx=0)
    c2 = ctcs(x[1] <= 2.0, nodes=(0, 5), idx=0)  # Same group as c1
    c3 = ctcs(x[0] <= 3.0, nodes=(3, 8), idx=1)  # Different group

    xdot_aug, states_aug, controls_aug = augment_dynamics_with_ctcs(
        xdot,
        states,
        controls,
        [c1, c2, c3],
        N,
    )

    # Should have 2 augmented states (for 2 idx groups) plus original states
    assert len(states_aug) == 4  # original (x, time) + 2 augmented
    assert states_aug[0] is x
    assert states_aug[1] is time
    assert states_aug[2].name == "_ctcs_aug_0"
    assert states_aug[3].name == "_ctcs_aug_1"

    # Augmented dynamics should have 3 parts: original + 2 penalty expressions
    assert isinstance(xdot_aug, Concat)
    assert len(xdot_aug.exprs) == 3  # original + 2 penalty groups

    # Each penalty group should be wrapped in Add to sum multiple CTCS nodes
    penalty1 = xdot_aug.exprs[1]  # idx 0 group
    penalty2 = xdot_aug.exprs[2]  # idx 1 group

    # penalty1 should be Add wrapping two CTCS constraints (c1 + c2)
    assert isinstance(penalty1, Add)  # Multiple penalties summed
    assert len(penalty1.terms) == 2
    assert all(isinstance(term, CTCS) for term in penalty1.terms)

    # penalty2 should be a single CTCS constraint (c3 only), not wrapped in Add
    assert isinstance(penalty2, CTCS)  # Single penalty


# Tests for vector constraint decomposition


def test_decompose_scalar_constraints_unchanged():
    """Test that scalar constraints remain unchanged during decomposition."""
    x = State("x", (3,))

    # Create scalar constraints in canonical form: residual <= 0
    constraint1 = Inequality(Sub(Index(x, 0), Constant(1.0)), Constant(0))
    constraint2 = Inequality(Sub(Constant(0.0), Index(x, 1)), Constant(0))

    nodal1 = NodalConstraint(constraint1, [0, 1])
    nodal2 = NodalConstraint(constraint2, [0, 1, 2])

    result = decompose_vector_nodal_constraints([nodal1, nodal2])

    # Should return the same constraints unchanged
    assert len(result) == 2
    assert result[0] is nodal1
    assert result[1] is nodal2


def test_decompose_vector_constraint_basic():
    """Test basic vector constraint decomposition."""
    x = State("x", (3,))
    bounds = np.array([1.0, 2.0, 3.0])
    nodes = [0, 1, 2]

    # Create vector constraint in canonical form: (x - bounds) <= 0
    vector_constraint = Inequality(Sub(x, Constant(bounds)), Constant(0))
    nodal = NodalConstraint(vector_constraint, nodes)

    result = decompose_vector_nodal_constraints([nodal])

    # Should decompose into 3 scalar constraints
    assert len(result) == 3

    # Check each decomposed constraint
    for i, decomposed in enumerate(result):
        assert isinstance(decomposed, NodalConstraint)
        assert decomposed.nodes == [0, 1, 2]  # Same nodes as original

        constraint = decomposed.constraint
        assert isinstance(constraint, Inequality)

        # LHS should be indexed version of original LHS
        assert isinstance(constraint.lhs, Index)
        assert constraint.lhs.index == i

        # RHS should be same as original (Constant(0))
        assert isinstance(constraint.rhs, Constant)
        assert constraint.rhs.value == 0


def test_decompose_vector_constraint_preserves_nodes():
    """Test that node specifications are preserved during decomposition."""
    x = State("x", (2,))
    bounds = np.array([5.0, 10.0])

    # Create vector constraint in canonical form
    vector_constraint = Inequality(Sub(x, Constant(bounds)), Constant(0))
    specific_nodes = [1, 3, 5]
    nodal = NodalConstraint(vector_constraint, specific_nodes)

    result = decompose_vector_nodal_constraints([nodal])

    # Should decompose into 2 scalar constraints, each with same nodes
    assert len(result) == 2
    for decomposed in result:
        assert decomposed.nodes == specific_nodes


def test_decompose_empty_list():
    """Test decomposition with empty constraint list."""
    result = decompose_vector_nodal_constraints([])
    assert result == []


def test_ctcs_check_nodally_node_interval_preserved():
    """Test that CTCS constraints with check_nodally=True preserve their node interval."""
    n_nodes = 50
    x = State("x", (1,))

    # Create a CTCS constraint with check_nodally over a specific interval
    c1 = ctcs(x <= 1.0, nodes=(10, 30), penalty="squared_relu", check_nodally=True)

    constraint_set = ConstraintSet(unsorted=[c1])
    result = separate_constraints(constraint_set, n_nodes=n_nodes)

    # Should have the CTCS constraint
    assert len(result.ctcs) == 1
    assert result.ctcs[0] is c1

    # Should also have a nodal constraint extracted from CTCS
    assert len(result.nodal) == 1

    # The nodal constraint should only apply to nodes [10, 11, ..., 29]
    nodal_from_ctcs = result.nodal[0]
    expected_nodes = list(range(10, 30))
    assert nodal_from_ctcs.nodes == expected_nodes, (
        f"Expected nodes {expected_nodes}, got {nodal_from_ctcs.nodes}"
    )


# Tests for cross-node constraint validation


def test_convex_nodal_constraint_with_node_reference_accepted():
    """Test that convex constraint with NodeReference is now supported."""
    n_nodes = 10
    position = State("pos", shape=(3,))

    # Create a convex cross-node constraint (linear inequality marked as convex)
    # No outer .at([...]) needed - auto-detected as cross-node
    cross_node_constraint = (position.at(5) - position.at(4) <= 0.1).convex()

    # The constraint itself is convex (explicitly marked)
    assert cross_node_constraint.is_convex

    # Should successfully separate constraints without raising
    constraint_set = ConstraintSet(unsorted=[cross_node_constraint])
    result = separate_constraints(constraint_set, n_nodes=n_nodes)

    # Should be classified as convex cross-node constraint
    assert len(result.cross_node_convex) == 1
    assert len(result.cross_node) == 0
    assert len(result.nodal_convex) == 0
    assert len(result.nodal) == 0
    assert len(result.ctcs) == 0


def test_convex_bare_constraint_with_node_reference_accepted():
    """Test that bare convex constraint with NodeReference is now supported."""
    n_nodes = 10
    position = State("pos", shape=(3,))

    # Create a bare convex cross-node constraint (linear inequality marked as convex)
    # This will be auto-converted to CrossNodeConstraint
    cross_node_constraint = (position.at(5) - position.at(4) <= 0.1).convex()

    # The constraint itself is convex (explicitly marked)
    assert cross_node_constraint.is_convex

    # Should successfully separate constraints without raising
    constraint_set = ConstraintSet(unsorted=[cross_node_constraint])
    result = separate_constraints(constraint_set, n_nodes=n_nodes)

    # Should be classified as convex cross-node constraint
    assert len(result.cross_node_convex) == 1
    assert len(result.cross_node) == 0
    assert len(result.nodal_convex) == 0
    assert len(result.nodal) == 0
    assert len(result.ctcs) == 0


def test_nonconvex_cross_node_constraint_accepted():
    """Test that non-convex cross-node constraints are accepted."""
    n_nodes = 10
    position = State("pos", shape=(3,))

    # Import linalg for Norm
    from openscvx.symbolic.expr import linalg

    # Create a non-convex cross-node constraint (norm inequality)
    # No outer .at([...]) needed - auto-detected as cross-node
    cross_node_constraint = linalg.Norm(position.at(5) - position.at(4), ord=2) <= 0.1

    # The constraint itself is non-convex (norm)
    assert not cross_node_constraint.is_convex

    # Should NOT raise error
    constraint_set = ConstraintSet(unsorted=[cross_node_constraint])
    result = separate_constraints(constraint_set, n_nodes=n_nodes)

    # Should be in the non-convex cross-node constraints
    assert len(result.cross_node) == 1
    assert len(result.cross_node_convex) == 0
    assert len(result.nodal) == 0
    assert len(result.nodal_convex) == 0
    assert len(result.ctcs) == 0


def test_cross_node_constraint_with_at_wrapper_rejected():
    """Test that cross-node constraints with .at([...]) wrapper are rejected."""
    n_nodes = 10
    position = State("pos", shape=(3,))

    # Import linalg for Norm
    from openscvx.symbolic.expr import linalg

    # Create a cross-node constraint WITH .at([...]) wrapper - should be rejected
    cross_node_constraint = (linalg.Norm(position.at(5) - position.at(4), ord=2) <= 0.1).at([5])

    # Should raise error because .at([...]) is not allowed on cross-node constraints
    constraint_set = ConstraintSet(unsorted=[cross_node_constraint])
    with pytest.raises(
        ValueError,
        match=r"Cross-node constraints should not use \.at\(\[\.\.\.\]\) wrapper",
    ):
        separate_constraints(constraint_set, n_nodes=n_nodes)


def test_regular_convex_constraint_without_node_reference_accepted():
    """Test that regular convex constraints (without NodeReference) are still accepted."""
    n_nodes = 10
    position = State("pos", shape=(3,))

    # Create a regular convex constraint (no cross-node reference)
    regular_constraint = (position <= 10.0).convex().at([0, 5, 9])

    # The constraint itself is convex
    assert regular_constraint.constraint.is_convex

    # Should NOT raise error
    constraint_set = ConstraintSet(unsorted=[regular_constraint])
    result = separate_constraints(constraint_set, n_nodes=n_nodes)

    # Should be in convex nodal constraints
    assert len(result.nodal_convex) == 1
    assert len(result.nodal) == 0
    assert len(result.cross_node) == 0
    assert len(result.cross_node_convex) == 0
    assert len(result.ctcs) == 0
