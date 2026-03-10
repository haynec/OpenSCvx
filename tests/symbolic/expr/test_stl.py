"""Tests for GMSR-based STL expression nodes.

This module tests the expression nodes defined in:
    openscvx/symbolic/expr/stl.py

And their JAX lowering via:
    openscvx/symbolic/lowerers/jax/stl.py

Coverage:
- Or: construction, children, repr, check_shape, canonicalization (flattening), validation
- And: construction, children, repr, check_shape, canonicalization (flattening), validation
- IfThen: construction, children, repr, check_shape, canonicalization
- IntegerVariable: construction, type-check, values, repr, check_shape
- STLExpr helpers: .over() produces CTCS, .at() produces NodalConstraint
- JAX lowering: all four operators produce correct-sign robustness values
"""

import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.symbolic.expr import Constant, State
from openscvx.symbolic.expr.constraint import CTCS, Inequality, NodalConstraint
from openscvx.symbolic.expr.stl import And, IfThen, IntegerVariable, Or, STLExpr
from openscvx.symbolic.lowerers.jax import JaxLowerer

# =============================================================================
# Helpers
# =============================================================================


def _make_predicates():
    """Return two simple inequality predicates using a 2-element state."""
    x = State("x", shape=(2,))
    x._slice = slice(0, 2)
    p1 = x[0] <= Constant(np.array(1.0))
    p2 = x[1] <= Constant(np.array(2.0))
    return x, p1, p2


def _lower_stl(expr):
    """Lower an STL expr to a JAX function via JaxLowerer."""
    return JaxLowerer().lower(expr)


# =============================================================================
# Or – Construction & Tree Structure
# =============================================================================


class TestOrConstruction:
    def test_two_predicates(self):
        _, p1, p2 = _make_predicates()
        node = Or(p1, p2)
        assert isinstance(node, STLExpr)
        assert len(node.predicates) == 2

    def test_three_predicates(self):
        x = State("x", shape=(3,))
        x._slice = slice(0, 3)
        p1 = x[0] <= Constant(np.array(1.0))
        p2 = x[1] <= Constant(np.array(2.0))
        p3 = x[2] <= Constant(np.array(3.0))
        node = Or(p1, p2, p3)
        assert len(node.predicates) == 3

    def test_requires_at_least_two_predicates(self):
        _, p1, _ = _make_predicates()
        with pytest.raises(ValueError, match="at least 2"):
            Or(p1)

    def test_rejects_non_constraint_predicate(self):
        _, p1, _ = _make_predicates()
        with pytest.raises(TypeError):
            Or(p1, 42)

    def test_children_returns_predicates(self):
        _, p1, p2 = _make_predicates()
        node = Or(p1, p2)
        assert node.children() == [p1, p2]

    def test_default_smoothing_param(self):
        _, p1, p2 = _make_predicates()
        assert Or(p1, p2).c == 1e-4

    def test_custom_smoothing_param(self):
        _, p1, p2 = _make_predicates()
        assert Or(p1, p2, c=1e-6).c == 1e-6

    def test_lite_flag(self):
        _, p1, p2 = _make_predicates()
        assert Or(p1, p2, lite=True).lite is True
        assert Or(p1, p2).lite is False

    def test_repr(self):
        _, p1, p2 = _make_predicates()
        r = repr(Or(p1, p2))
        assert r.startswith("Or(")

    def test_repr_lite_suffix(self):
        _, p1, p2 = _make_predicates()
        assert "lite=True" in repr(Or(p1, p2, lite=True))


# =============================================================================
# Or – check_shape & canonicalize
# =============================================================================


class TestOrShapeAndCanonicalize:
    def test_check_shape_returns_empty_tuple(self):
        _, p1, p2 = _make_predicates()
        assert Or(p1, p2).check_shape() == ()

    def test_canonicalize_flattens_nested_or(self):
        _, p1, p2 = _make_predicates()
        x = State("y", shape=(1,))
        x._slice = slice(0, 1)
        p3 = x[0] <= Constant(np.array(0.0))
        inner = Or(p1, p2)
        outer = Or(inner, p3)
        flat = outer.canonicalize()
        assert isinstance(flat, Or)
        assert len(flat.predicates) == 3

    def test_canonicalize_does_not_flatten_different_c(self):
        _, p1, p2 = _make_predicates()
        x = State("y", shape=(1,))
        x._slice = slice(0, 1)
        p3 = x[0] <= Constant(np.array(0.0))
        inner = Or(p1, p2, c=1e-6)
        outer = Or(inner, p3, c=1e-4)
        flat = outer.canonicalize()
        # Inner Or has different c — should NOT be flattened
        assert len(flat.predicates) == 2

    def test_canonicalize_does_not_flatten_lite_mismatch(self):
        _, p1, p2 = _make_predicates()
        x = State("y", shape=(1,))
        x._slice = slice(0, 1)
        p3 = x[0] <= Constant(np.array(0.0))
        inner = Or(p1, p2, lite=True)
        outer = Or(inner, p3, lite=False)
        flat = outer.canonicalize()
        assert len(flat.predicates) == 2


# =============================================================================
# And – Construction & Tree Structure
# =============================================================================


class TestAndConstruction:
    def test_two_predicates(self):
        _, p1, p2 = _make_predicates()
        node = And(p1, p2)
        assert isinstance(node, STLExpr)
        assert len(node.predicates) == 2

    def test_requires_at_least_two_predicates(self):
        _, p1, _ = _make_predicates()
        with pytest.raises(ValueError, match="at least 2"):
            And(p1)

    def test_rejects_non_constraint_predicate(self):
        _, p1, _ = _make_predicates()
        with pytest.raises(TypeError):
            And(p1, "bad")

    def test_children_returns_predicates(self):
        _, p1, p2 = _make_predicates()
        node = And(p1, p2)
        assert node.children() == [p1, p2]

    def test_repr(self):
        _, p1, p2 = _make_predicates()
        r = repr(And(p1, p2))
        assert r.startswith("And(")

    def test_lite_flag(self):
        _, p1, p2 = _make_predicates()
        assert And(p1, p2, lite=True).lite is True

    def test_check_shape_returns_empty_tuple(self):
        _, p1, p2 = _make_predicates()
        assert And(p1, p2).check_shape() == ()

    def test_canonicalize_flattens_nested_and(self):
        _, p1, p2 = _make_predicates()
        x = State("z", shape=(1,))
        x._slice = slice(0, 1)
        p3 = x[0] <= Constant(np.array(0.0))
        flat = And(And(p1, p2), p3).canonicalize()
        assert isinstance(flat, And)
        assert len(flat.predicates) == 3


# =============================================================================
# IfThen – Construction & Tree Structure
# =============================================================================


class TestIfThenConstruction:
    def test_basic_construction(self):
        _, p1, p2 = _make_predicates()
        node = IfThen(p1, p2)
        assert isinstance(node, STLExpr)
        assert node.condition is p1
        assert node.consequent is p2

    def test_children_order(self):
        _, p1, p2 = _make_predicates()
        node = IfThen(p1, p2)
        children = node.children()
        assert children[0] is p1
        assert children[1] is p2

    def test_repr(self):
        _, p1, p2 = _make_predicates()
        r = repr(IfThen(p1, p2))
        assert "IfThen(" in r
        assert "=>" in r

    def test_lite_flag(self):
        _, p1, p2 = _make_predicates()
        assert IfThen(p1, p2, lite=True).lite is True

    def test_check_shape_returns_empty_tuple(self):
        _, p1, p2 = _make_predicates()
        assert IfThen(p1, p2).check_shape() == ()

    def test_requires_exactly_two_args(self):
        _, p1, _ = _make_predicates()
        with pytest.raises((TypeError, ValueError)):
            IfThen(p1)

    def test_rejects_non_constraint_args(self):
        _, p1, _ = _make_predicates()
        with pytest.raises(TypeError):
            IfThen(p1, "not_a_predicate")

    def test_canonicalize_preserves_structure(self):
        _, p1, p2 = _make_predicates()
        canon = IfThen(p1, p2).canonicalize()
        assert isinstance(canon, IfThen)

    def test_accepts_nested_stl_as_consequent(self):
        _, p1, p2 = _make_predicates()
        x = State("z", shape=(1,))
        x._slice = slice(0, 1)
        p3 = x[0] <= Constant(np.array(0.0))
        nested_or = Or(p2, p3)
        node = IfThen(p1, nested_or)
        assert isinstance(node, IfThen)


# =============================================================================
# IntegerVariable – Construction & Tree Structure
# =============================================================================


class TestIntegerVariableConstruction:
    def test_basic_construction(self):
        x = State("g", shape=(1,))
        iv = IntegerVariable(x, [0.0, 1.0, 2.0])
        assert isinstance(iv, STLExpr)
        assert np.allclose(iv.values, [0.0, 1.0, 2.0])
        assert iv.expr is x

    def test_values_stored_as_ndarray(self):
        x = State("g", shape=(1,))
        iv = IntegerVariable(x, [1, 2, 3])
        assert isinstance(iv.values, np.ndarray)

    def test_custom_c(self):
        x = State("g", shape=(1,))
        iv = IntegerVariable(x, [0.0, 1.0], c=1e-6)
        assert iv.c == 1e-6

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError, match="Expr"):
            IntegerVariable(42, [0.0, 1.0])

    def test_children_returns_expr(self):
        x = State("g", shape=(1,))
        iv = IntegerVariable(x, [0.0, 1.0])
        assert iv.children() == [x]

    def test_check_shape_returns_empty_tuple(self):
        x = State("g", shape=(1,))
        x._slice = slice(0, 1)
        iv = IntegerVariable(x, [0.0, 1.0])
        assert iv.check_shape() == ()

    def test_repr(self):
        x = State("g", shape=(1,))
        r = repr(IntegerVariable(x, [1.0, 2.0]))
        assert "IntegerVariable(" in r
        assert "values=" in r

    def test_canonicalize_preserves_values(self):
        x = State("g", shape=(1,))
        iv = IntegerVariable(x, [0.0, 1.0, 2.0])
        canon = iv.canonicalize()
        assert isinstance(canon, IntegerVariable)
        assert np.allclose(canon.values, iv.values)


# =============================================================================
# STLExpr helpers: .over() and .at()
# =============================================================================


class TestSTLExprHelpers:
    def test_over_returns_ctcs(self):
        _, p1, p2 = _make_predicates()
        result = Or(p1, p2).over((0, 5))
        assert isinstance(result, CTCS)

    def test_over_interval_stored(self):
        _, p1, p2 = _make_predicates()
        ctcs = Or(p1, p2).over((3, 7))
        assert ctcs.nodes == (3, 7)

    def test_at_returns_nodal_constraint(self):
        _, p1, p2 = _make_predicates()
        result = Or(p1, p2).at([0, 5, 10])
        assert isinstance(result, NodalConstraint)

    def test_at_with_integer_node(self):
        _, p1, p2 = _make_predicates()
        result = Or(p1, p2).at(5)
        assert isinstance(result, NodalConstraint)

    def test_over_wraps_negated_inequality(self):
        _, p1, p2 = _make_predicates()
        ctcs = And(p1, p2).over((0, 10))
        # The inner constraint should be Inequality(-stl_expr <= 0)
        assert isinstance(ctcs.constraint, Inequality)

    def test_and_over_returns_ctcs(self):
        _, p1, p2 = _make_predicates()
        assert isinstance(And(p1, p2).over((0, 5)), CTCS)

    def test_ifthen_over_returns_ctcs(self):
        _, p1, p2 = _make_predicates()
        assert isinstance(IfThen(p1, p2).over((0, 5)), CTCS)

    def test_integer_variable_over_returns_ctcs(self):
        x = State("g", shape=(1,))
        x._slice = slice(0, 1)
        iv = IntegerVariable(x, [0.0, 1.0, 2.0])
        assert isinstance(iv.over((0, 5)), CTCS)


# =============================================================================
# JAX Lowering – Or
# =============================================================================


class TestOrJaxLowering:
    def _setup(self):
        x_sym = State("x", shape=(2,))
        x_sym._slice = slice(0, 2)
        p1 = x_sym[0] <= Constant(np.array(1.0))  # x[0] <= 1.0
        p2 = x_sym[1] <= Constant(np.array(2.0))  # x[1] <= 2.0
        expr = Or(p1, p2)
        fn = _lower_stl(expr)
        return fn

    def test_positive_when_one_predicate_satisfied(self):
        fn = self._setup()
        # x[0]=0.5 satisfies p1; x[1]=3.0 violates p2
        x = jnp.array([0.5, 3.0])
        assert float(fn(x, None, None, None)) > 0.0

    def test_positive_when_both_satisfied(self):
        fn = self._setup()
        x = jnp.array([0.5, 1.5])
        assert float(fn(x, None, None, None)) > 0.0

    def test_negative_when_all_violated(self):
        fn = self._setup()
        x = jnp.array([2.0, 3.0])
        assert float(fn(x, None, None, None)) < 0.0

    def test_output_is_scalar(self):
        fn = self._setup()
        out = fn(jnp.array([0.5, 1.5]), None, None, None)
        assert jnp.shape(out) == ()

    def test_or_lite_variant(self):
        x_sym = State("x", shape=(2,))
        x_sym._slice = slice(0, 2)
        p1 = x_sym[0] <= Constant(np.array(1.0))
        p2 = x_sym[1] <= Constant(np.array(2.0))
        fn = _lower_stl(Or(p1, p2, lite=True))
        # lite variant: satisfied when at least one predicate satisfied
        x_ok = jnp.array([0.5, 3.0])
        assert float(fn(x_ok, None, None, None)) >= 0.0


# =============================================================================
# JAX Lowering – And
# =============================================================================


class TestAndJaxLowering:
    def _setup(self):
        x_sym = State("x", shape=(2,))
        x_sym._slice = slice(0, 2)
        p1 = x_sym[0] <= Constant(np.array(1.0))
        p2 = x_sym[1] <= Constant(np.array(2.0))
        fn = _lower_stl(And(p1, p2))
        return fn

    def test_positive_when_both_satisfied(self):
        fn = self._setup()
        x = jnp.array([0.5, 1.5])
        assert float(fn(x, None, None, None)) > 0.0

    def test_negative_when_one_violated(self):
        fn = self._setup()
        # x[0] ok, x[1] violated
        x = jnp.array([0.5, 3.0])
        assert float(fn(x, None, None, None)) < 0.0

    def test_negative_when_both_violated(self):
        fn = self._setup()
        x = jnp.array([2.0, 3.0])
        assert float(fn(x, None, None, None)) < 0.0

    def test_output_is_scalar(self):
        fn = self._setup()
        out = fn(jnp.array([0.5, 1.5]), None, None, None)
        assert jnp.shape(out) == ()

    def test_and_lite_variant(self):
        x_sym = State("x", shape=(2,))
        x_sym._slice = slice(0, 2)
        p1 = x_sym[0] <= Constant(np.array(1.0))
        p2 = x_sym[1] <= Constant(np.array(2.0))
        fn = _lower_stl(And(p1, p2, lite=True))
        x_ok = jnp.array([0.5, 1.5])
        assert float(fn(x_ok, None, None, None)) >= 0.0


# =============================================================================
# JAX Lowering – IfThen
# =============================================================================


class TestIfThenJaxLowering:
    def _setup(self):
        x_sym = State("x", shape=(2,))
        x_sym._slice = slice(0, 2)
        cond = x_sym[0] <= Constant(np.array(1.0))  # condition: x[0] <= 1.0
        conseq = x_sym[1] <= Constant(np.array(2.0))  # consequent: x[1] <= 2.0
        fn = _lower_stl(IfThen(cond, conseq))
        return fn

    def test_positive_when_condition_and_consequent_both_satisfied(self):
        fn = self._setup()
        x = jnp.array([0.5, 1.5])  # cond OK, conseq OK → implication holds
        assert float(fn(x, None, None, None)) > 0.0

    def test_positive_when_condition_not_satisfied(self):
        fn = self._setup()
        # condition violated → implication trivially holds
        x = jnp.array([2.0, 3.0])
        assert float(fn(x, None, None, None)) > 0.0

    def test_negative_when_condition_holds_consequent_violated(self):
        fn = self._setup()
        # condition satisfied but consequent violated → implication fails
        x = jnp.array([0.5, 3.0])
        assert float(fn(x, None, None, None)) < 0.0

    def test_output_is_scalar(self):
        fn = self._setup()
        out = fn(jnp.array([0.5, 1.5]), None, None, None)
        assert jnp.shape(out) == ()

    def test_lite_variant_condition_satisfied_consequent_violated(self):
        x_sym = State("x", shape=(2,))
        x_sym._slice = slice(0, 2)
        cond = x_sym[0] <= Constant(np.array(1.0))
        conseq = x_sym[1] <= Constant(np.array(2.0))
        fn = _lower_stl(IfThen(cond, conseq, lite=True))
        x = jnp.array([0.5, 3.0])
        # lite variant: same semantic sign
        assert float(fn(x, None, None, None)) < 0.0


# =============================================================================
# JAX Lowering – IntegerVariable
# =============================================================================


class TestIntegerVariableJaxLowering:
    def _setup(self, values=(0.0, 1.0, 2.0)):
        x_sym = State("g", shape=(1,))
        x_sym._slice = slice(0, 1)
        iv = IntegerVariable(x_sym, list(values))
        fn = _lower_stl(iv)
        return fn

    def test_zero_when_exactly_at_allowed_value(self):
        fn = self._setup()
        for v in [0.0, 1.0, 2.0]:
            out = float(fn(jnp.array([v]), None, None, None))
            assert out == pytest.approx(0.0, abs=1e-4), f"expected ~0 for g={v}"

    def test_negative_when_not_at_allowed_value(self):
        fn = self._setup()
        out = float(fn(jnp.array([0.5]), None, None, None))
        assert out < 0.0

    def test_output_is_scalar(self):
        fn = self._setup()
        out = fn(jnp.array([1.0]), None, None, None)
        assert jnp.shape(out) == ()

    def test_larger_deviation_more_negative(self):
        fn = self._setup()
        close = float(fn(jnp.array([0.1]), None, None, None))
        far = float(fn(jnp.array([0.4]), None, None, None))
        assert close > far  # closer to 0.0 → less negative = better

    def test_single_allowed_value(self):
        fn = self._setup(values=[3.0])
        assert float(fn(jnp.array([3.0]), None, None, None)) == pytest.approx(0.0, abs=1e-4)
        assert float(fn(jnp.array([3.5]), None, None, None)) < 0.0
