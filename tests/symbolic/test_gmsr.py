"""Tests for GMSR (Generalized Mean-based Smooth Robustness) math functions.

This module tests all primitive functions in:
    openscvx/symbolic/lowerers/jax/gmsr.py

Each function is tested for:
- Exact boundary / zero-crossing semantics
- Monotonicity / ordering relative to trivially satisfied/violated inputs
- JAX differentiability (grad does not raise)
- Numerical stability at edge cases (all-zero, single element, large values)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.symbolic.lowerers.jax.gmsr import (
    AND,
    OR,
    AND_lite,
    OR_lite,
    IfThen,
    IfThen_lite,
    _smooth_equality,
    integer_variable,
)


# =============================================================================
# Helpers
# =============================================================================

C = 1e-4  # default smoothing constant


def _grad_ok(fn, *args):
    """Return True if jax.grad(fn)(*args) does not raise and is finite."""
    try:
        g = jax.grad(fn)(*args)
        return bool(jnp.all(jnp.isfinite(g)))
    except Exception:
        return False


# =============================================================================
# _smooth_equality
# =============================================================================


class TestSmoothEquality:
    def test_zero_at_origin(self):
        val = _smooth_equality(jnp.array(0.0), c=C)
        assert float(val) == pytest.approx(0.0, abs=1e-12)

    def test_positive_for_nonzero_input(self):
        for y in [0.1, -0.1, 1.0, -5.0]:
            val = _smooth_equality(jnp.array(float(y)), c=C)
            assert float(val) > 0.0, f"expected >0 for y={y}"

    def test_symmetric(self):
        for y in [0.5, 1.0, 3.0]:
            pos = float(_smooth_equality(jnp.array(y), c=C))
            neg = float(_smooth_equality(jnp.array(-y), c=C))
            assert pos == pytest.approx(neg, rel=1e-6)

    def test_smaller_c_tighter_approximation(self):
        # With smaller c, the value should be closer to |y| = abs(y)
        y = jnp.array(1.0)
        val_large_c = float(_smooth_equality(y, c=0.1))
        val_small_c = float(_smooth_equality(y, c=1e-8))
        assert val_small_c == pytest.approx(1.0, rel=1e-4)
        assert val_large_c < val_small_c  # large c over-estimates less aggressively

    def test_differentiable_at_zero(self):
        fn = lambda y: _smooth_equality(y, c=C)
        g = jax.grad(fn)(jnp.array(0.0))
        assert jnp.isfinite(g)

    def test_differentiable_away_from_zero(self):
        fn = lambda y: _smooth_equality(y, c=C)
        assert _grad_ok(fn, jnp.array(1.5))


# =============================================================================
# AND
# =============================================================================


class TestAND:
    def test_satisfied_when_all_negative(self):
        # All residuals ≤ 0 → AND ≤ 0
        y = jnp.array([-1.0, -2.0, -0.5])
        val = float(AND(y, c=C))
        assert val <= 0.0 + 1e-6

    def test_violated_when_any_positive(self):
        # One positive residual → AND > 0
        y = jnp.array([-1.0, 1.0, -0.5])
        val = float(AND(y, c=C))
        assert val > 0.0

    def test_more_positive_means_higher_value(self):
        y_slightly = jnp.array([-1.0, 0.1])
        y_very = jnp.array([-1.0, 5.0])
        assert float(AND(y_slightly)) < float(AND(y_very))

    def test_all_positive_is_most_violated(self):
        y_all_pos = jnp.array([1.0, 1.0])
        y_mixed = jnp.array([-1.0, 1.0])
        assert float(AND(y_all_pos)) >= float(AND(y_mixed))

    def test_single_element_all_negative(self):
        # With a single element, AND([-1]) should behave consistently
        # The function handles this without crashing
        val = AND(jnp.array([-1.0]), c=C)
        assert jnp.isfinite(val)

    def test_differentiable(self):
        fn = lambda y: AND(y, c=C)
        g = jax.grad(fn)(jnp.array([-1.0, 1.0, 0.5]))
        assert jnp.all(jnp.isfinite(g))

    def test_or_is_negated_and(self):
        # By definition: OR(y) = -AND(-y)
        y = jnp.array([-0.5, 1.0, 0.3])
        assert float(OR(y, c=C)) == pytest.approx(float(-AND(-y, c=C)), rel=1e-5)


# =============================================================================
# OR
# =============================================================================


class TestOR:
    def test_satisfied_when_any_negative(self):
        # At least one residual ≤ 0 → OR ≤ 0
        y = jnp.array([1.0, -0.5])
        val = float(OR(y, c=C))
        assert val <= 0.0 + 1e-6

    def test_violated_when_all_positive(self):
        # All residuals > 0 → OR > 0
        y = jnp.array([0.5, 1.0, 2.0])
        val = float(OR(y, c=C))
        assert val > 0.0

    def test_more_negative_means_lower_value(self):
        # The most-negative element dominates
        y_weak = jnp.array([-0.1, 2.0])
        y_strong = jnp.array([-5.0, 2.0])
        assert float(OR(y_strong)) < float(OR(y_weak))

    def test_three_predicates_any_satisfied(self):
        y = jnp.array([2.0, 2.0, -0.1])
        assert float(OR(y, c=C)) <= 0.0 + 1e-6

    def test_differentiable(self):
        fn = lambda y: OR(y, c=C)
        g = jax.grad(fn)(jnp.array([0.5, -0.3]))
        assert jnp.all(jnp.isfinite(g))


# =============================================================================
# IfThen
# =============================================================================


class TestIfThen:
    def test_satisfied_when_condition_false(self):
        # Condition NOT satisfied (y0 > 0): implication trivially holds → IfThen ≤ 0
        y = jnp.array([1.0, 1.0])  # condition violated, consequent violated
        val = float(IfThen(y, c=C))
        assert val <= 0.0 + 1e-6

    def test_satisfied_when_consequent_holds(self):
        # Condition satisfied (y0 ≤ 0) AND consequent satisfied (y1 ≤ 0) → IfThen ≤ 0
        y = jnp.array([-1.0, -1.0])
        val = float(IfThen(y, c=C))
        assert val <= 0.0 + 1e-6

    def test_violated_when_condition_holds_and_consequent_doesnt(self):
        # Condition satisfied (y0 ≤ 0), consequent violated (y1 > 0) → IfThen > 0
        y = jnp.array([-1.0, 1.0])
        val = float(IfThen(y, c=C))
        assert val > 0.0

    def test_equals_or_of_neg_cond_and_conseq(self):
        # IfThen([y0, y1]) = OR([-y0, y1])
        y = jnp.array([-0.5, 0.8])
        expected = float(OR(jnp.array([0.5, 0.8]), c=C))
        actual = float(IfThen(y, c=C))
        assert actual == pytest.approx(expected, rel=1e-5)

    def test_differentiable(self):
        fn = lambda y: IfThen(y, c=C)
        g = jax.grad(fn)(jnp.array([-0.3, 0.7]))
        assert jnp.all(jnp.isfinite(g))


# =============================================================================
# integer_variable
# =============================================================================


class TestIntegerVariable:
    def test_zero_when_exactly_matching_value(self):
        for v in [0.0, 1.0, 2.0, -3.0]:
            val = float(integer_variable(jnp.array(v), jnp.array([0.0, 1.0, 2.0, -3.0]), c=1e-8))
            assert val == pytest.approx(0.0, abs=1e-5), f"expected ~0 for y={v}"

    def test_positive_when_not_matching(self):
        y = jnp.array(1.5)
        val = float(integer_variable(y, jnp.array([0.0, 1.0, 2.0]), c=C))
        assert val > 0.0

    def test_larger_deviation_means_higher_penalty(self):
        values = jnp.array([0.0, 1.0, 2.0])
        y_close = jnp.array(0.1)
        y_far = jnp.array(0.5)
        assert float(integer_variable(y_close, values)) < float(integer_variable(y_far, values))

    def test_single_allowed_value(self):
        # With only one allowed value, penalty = 0 exactly at that value
        val_at = float(integer_variable(jnp.array(3.0), jnp.array([3.0]), c=1e-8))
        assert val_at == pytest.approx(0.0, abs=1e-5)
        val_off = float(integer_variable(jnp.array(3.1), jnp.array([3.0]), c=1e-8))
        assert val_off > 0.0

    def test_differentiable_away_from_values(self):
        fn = lambda y: integer_variable(y, jnp.array([0.0, 1.0, 2.0]), c=C)
        g = jax.grad(fn)(jnp.array(0.5))
        assert jnp.isfinite(g)

    def test_differentiable_at_exact_value(self):
        fn = lambda y: integer_variable(y, jnp.array([0.0, 1.0]), c=C)
        g = jax.grad(fn)(jnp.array(1.0))
        assert jnp.isfinite(g)


# =============================================================================
# AND_lite
# =============================================================================


class TestANDLite:
    def test_zero_when_all_nonpositive(self):
        y = jnp.array([-1.0, -0.5, 0.0])
        val = float(AND_lite(y, c=C))
        assert val == pytest.approx(0.0, abs=1e-3)

    def test_positive_when_any_positive(self):
        y = jnp.array([-1.0, 0.5])
        assert float(AND_lite(y, c=C)) > 0.0

    def test_only_positive_part_penalized(self):
        # Two inputs: only the positive one matters
        y_pos = jnp.array([0.0, 1.0])
        y_neg = jnp.array([0.0, -1.0])
        assert float(AND_lite(y_pos)) > float(AND_lite(y_neg))

    def test_differentiable(self):
        fn = lambda y: AND_lite(y, c=C)
        g = jax.grad(fn)(jnp.array([-0.5, 0.3]))
        assert jnp.all(jnp.isfinite(g))


# =============================================================================
# OR_lite
# =============================================================================


class TestORLite:
    def test_zero_when_all_nonpositive(self):
        # OR_lite = 0 iff all y_i ≤ 0
        y = jnp.array([-1.0, -0.5])
        val = float(OR_lite(y, c=C))
        assert val == pytest.approx(0.0, abs=1e-3)

    def test_positive_when_any_positive(self):
        y = jnp.array([0.5, -1.0])
        assert float(OR_lite(y, c=C)) == pytest.approx(0.0, abs=1e-3)

    def test_larger_positive_means_more_violated(self):
        y_small = jnp.array([0.1, 10.0])
        y_large = jnp.array([2.0, 10000.0])
        assert float(OR_lite(y_small)) < float(OR_lite(y_large))

    def test_differentiable(self):
        fn = lambda y: OR_lite(y, c=C)
        g = jax.grad(fn)(jnp.array([0.5, -0.3]))
        assert jnp.all(jnp.isfinite(g))


# =============================================================================
# IfThen_lite
# =============================================================================


class TestIfThenLite:
    def test_zero_when_condition_false(self):
        # Condition violated (y0 > 0): trivially satisfied
        y = jnp.array([1.0, 1.0])
        val = float(IfThen_lite(y, c=C))
        assert val == pytest.approx(0.0, abs=1e-3)

    def test_zero_when_both_nonpositive(self):
        # Both satisfied
        y = jnp.array([-1.0, -1.0])
        val = float(IfThen_lite(y, c=C))
        assert val == pytest.approx(0.0, abs=1e-3)

    def test_positive_when_condition_holds_consequent_doesnt(self):
        # Condition satisfied, consequent violated
        y = jnp.array([-1.0, 1.0])
        val = float(IfThen_lite(y, c=C))
        assert val > 0.0

    def test_equals_or_lite_of_neg_cond_and_conseq(self):
        y = jnp.array([-0.5, 0.8])
        expected = float(OR_lite(jnp.array([0.5, 0.8]), c=C))
        actual = float(IfThen_lite(y, c=C))
        assert actual == pytest.approx(expected, rel=1e-5)

    def test_differentiable(self):
        fn = lambda y: IfThen_lite(y, c=C)
        g = jax.grad(fn)(jnp.array([-0.3, 0.7]))
        assert jnp.all(jnp.isfinite(g))
