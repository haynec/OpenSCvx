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
from openscvx.symbolic.expr.stl import (
    Always,
    And,
    Eventually,
    IfThen,
    IntegerVariable,
    Interval,
    NodeInterval,
    Not,
    Or,
    STLExpr,
    TimeInterval,
    Until,
)
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


def _setup_or_jax():
    x_sym = State("x", shape=(2,))
    x_sym._slice = slice(0, 2)
    p1 = x_sym[0] <= Constant(np.array(1.0))  # x[0] <= 1.0
    p2 = x_sym[1] <= Constant(np.array(2.0))  # x[1] <= 2.0
    return _lower_stl(Or(p1, p2))


def _setup_and_jax():
    x_sym = State("x", shape=(2,))
    x_sym._slice = slice(0, 2)
    p1 = x_sym[0] <= Constant(np.array(1.0))
    p2 = x_sym[1] <= Constant(np.array(2.0))
    return _lower_stl(And(p1, p2))


def _setup_ifthen_jax():
    x_sym = State("x", shape=(2,))
    x_sym._slice = slice(0, 2)
    cond = x_sym[0] <= Constant(np.array(1.0))  # condition: x[0] <= 1.0
    conseq = x_sym[1] <= Constant(np.array(2.0))  # consequent: x[1] <= 2.0
    return _lower_stl(IfThen(cond, conseq))


def _setup_integer_variable_jax(values=(0.0, 1.0, 2.0)):
    x_sym = State("g", shape=(1,))
    x_sym._slice = slice(0, 1)
    iv = IntegerVariable(x_sym, list(values))
    return _lower_stl(iv)


# =============================================================================
# Or – Construction & Tree Structure
# =============================================================================


def test_or_two_predicates():
    _, p1, p2 = _make_predicates()
    node = Or(p1, p2)
    assert isinstance(node, STLExpr)
    assert len(node.predicates) == 2


def test_or_three_predicates():
    x = State("x", shape=(3,))
    x._slice = slice(0, 3)
    p1 = x[0] <= Constant(np.array(1.0))
    p2 = x[1] <= Constant(np.array(2.0))
    p3 = x[2] <= Constant(np.array(3.0))
    node = Or(p1, p2, p3)
    assert len(node.predicates) == 3


def test_or_requires_at_least_two_predicates():
    _, p1, _ = _make_predicates()
    with pytest.raises(ValueError, match="at least 2"):
        Or(p1)


def test_or_rejects_non_constraint_predicate():
    _, p1, _ = _make_predicates()
    with pytest.raises(TypeError):
        Or(p1, 42)


def test_or_children_returns_predicates():
    _, p1, p2 = _make_predicates()
    node = Or(p1, p2)
    assert node.children() == [p1, p2]


def test_or_default_smoothing_param():
    _, p1, p2 = _make_predicates()
    assert Or(p1, p2).c == 1e-4


def test_or_custom_smoothing_param():
    _, p1, p2 = _make_predicates()
    assert Or(p1, p2, c=1e-6).c == 1e-6


def test_or_lite_flag():
    _, p1, p2 = _make_predicates()
    assert Or(p1, p2, lite=True).lite is True
    assert Or(p1, p2).lite is False


def test_or_repr():
    _, p1, p2 = _make_predicates()
    r = repr(Or(p1, p2))
    assert r.startswith("Or(")


def test_or_repr_lite_suffix():
    _, p1, p2 = _make_predicates()
    assert "lite=True" in repr(Or(p1, p2, lite=True))


# =============================================================================
# Or – check_shape & canonicalize
# =============================================================================


def test_or_check_shape_returns_empty_tuple():
    _, p1, p2 = _make_predicates()
    assert Or(p1, p2).check_shape() == ()


def test_or_canonicalize_flattens_nested_or():
    _, p1, p2 = _make_predicates()
    x = State("y", shape=(1,))
    x._slice = slice(0, 1)
    p3 = x[0] <= Constant(np.array(0.0))
    inner = Or(p1, p2)
    outer = Or(inner, p3)
    flat = outer.canonicalize()
    assert isinstance(flat, Or)
    assert len(flat.predicates) == 3


def test_or_canonicalize_does_not_flatten_different_c():
    _, p1, p2 = _make_predicates()
    x = State("y", shape=(1,))
    x._slice = slice(0, 1)
    p3 = x[0] <= Constant(np.array(0.0))
    inner = Or(p1, p2, c=1e-6)
    outer = Or(inner, p3, c=1e-4)
    flat = outer.canonicalize()
    # Inner Or has different c — should NOT be flattened
    assert len(flat.predicates) == 2


def test_or_canonicalize_does_not_flatten_lite_mismatch():
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


def test_and_two_predicates():
    _, p1, p2 = _make_predicates()
    node = And(p1, p2)
    assert isinstance(node, STLExpr)
    assert len(node.predicates) == 2


def test_and_requires_at_least_two_predicates():
    _, p1, _ = _make_predicates()
    with pytest.raises(ValueError, match="at least 2"):
        And(p1)


def test_and_rejects_non_constraint_predicate():
    _, p1, _ = _make_predicates()
    with pytest.raises(TypeError):
        And(p1, "bad")


def test_and_children_returns_predicates():
    _, p1, p2 = _make_predicates()
    node = And(p1, p2)
    assert node.children() == [p1, p2]


def test_and_repr():
    _, p1, p2 = _make_predicates()
    r = repr(And(p1, p2))
    assert r.startswith("And(")


def test_and_lite_flag():
    _, p1, p2 = _make_predicates()
    assert And(p1, p2, lite=True).lite is True


def test_and_check_shape_returns_empty_tuple():
    _, p1, p2 = _make_predicates()
    assert And(p1, p2).check_shape() == ()


def test_and_canonicalize_flattens_nested_and():
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


def test_ifthen_basic_construction():
    _, p1, p2 = _make_predicates()
    node = IfThen(p1, p2)
    assert isinstance(node, STLExpr)
    assert node.condition is p1
    assert node.consequent is p2


def test_ifthen_children_order():
    _, p1, p2 = _make_predicates()
    node = IfThen(p1, p2)
    children = node.children()
    assert children[0] is p1
    assert children[1] is p2


def test_ifthen_repr():
    _, p1, p2 = _make_predicates()
    r = repr(IfThen(p1, p2))
    assert "IfThen(" in r
    assert "=>" in r


def test_ifthen_lite_flag():
    _, p1, p2 = _make_predicates()
    assert IfThen(p1, p2, lite=True).lite is True


def test_ifthen_check_shape_returns_empty_tuple():
    _, p1, p2 = _make_predicates()
    assert IfThen(p1, p2).check_shape() == ()


def test_ifthen_requires_exactly_two_args():
    _, p1, _ = _make_predicates()
    with pytest.raises((TypeError, ValueError)):
        IfThen(p1)


def test_ifthen_rejects_non_constraint_args():
    _, p1, _ = _make_predicates()
    with pytest.raises(TypeError):
        IfThen(p1, "not_a_predicate")


def test_ifthen_canonicalize_preserves_structure():
    _, p1, p2 = _make_predicates()
    canon = IfThen(p1, p2).canonicalize()
    assert isinstance(canon, IfThen)


def test_ifthen_accepts_nested_stl_as_consequent():
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


def test_integer_variable_basic_construction():
    x = State("g", shape=(1,))
    iv = IntegerVariable(x, [0.0, 1.0, 2.0])
    assert isinstance(iv, STLExpr)
    assert np.allclose(iv.values, [0.0, 1.0, 2.0])
    assert iv.expr is x


def test_integer_variable_values_stored_as_ndarray():
    x = State("g", shape=(1,))
    iv = IntegerVariable(x, [1, 2, 3])
    assert isinstance(iv.values, np.ndarray)


def test_integer_variable_custom_c():
    x = State("g", shape=(1,))
    iv = IntegerVariable(x, [0.0, 1.0], c=1e-6)
    assert iv.c == 1e-6


def test_integer_variable_rejects_non_expr():
    with pytest.raises(TypeError, match="Expr"):
        IntegerVariable(42, [0.0, 1.0])


def test_integer_variable_children_returns_expr():
    x = State("g", shape=(1,))
    iv = IntegerVariable(x, [0.0, 1.0])
    assert iv.children() == [x]


def test_integer_variable_check_shape_returns_empty_tuple():
    x = State("g", shape=(1,))
    x._slice = slice(0, 1)
    iv = IntegerVariable(x, [0.0, 1.0])
    assert iv.check_shape() == ()


def test_integer_variable_repr():
    x = State("g", shape=(1,))
    r = repr(IntegerVariable(x, [1.0, 2.0]))
    assert "IntegerVariable(" in r
    assert "values=" in r


def test_integer_variable_canonicalize_preserves_values():
    x = State("g", shape=(1,))
    iv = IntegerVariable(x, [0.0, 1.0, 2.0])
    canon = iv.canonicalize()
    assert isinstance(canon, IntegerVariable)
    assert np.allclose(canon.values, iv.values)


# =============================================================================
# STLExpr helpers: .over() and .at()
# =============================================================================


def test_stl_over_returns_ctcs():
    _, p1, p2 = _make_predicates()
    result = Or(p1, p2).over((0, 5))
    assert isinstance(result, CTCS)


def test_stl_over_interval_stored():
    _, p1, p2 = _make_predicates()
    ctcs = Or(p1, p2).over((3, 7))
    assert ctcs.nodes == (3, 7)


def test_stl_at_returns_nodal_constraint():
    _, p1, p2 = _make_predicates()
    result = Or(p1, p2).at([0, 5, 10])
    assert isinstance(result, NodalConstraint)


def test_stl_at_with_integer_node():
    _, p1, p2 = _make_predicates()
    result = Or(p1, p2).at(5)
    assert isinstance(result, NodalConstraint)


def test_stl_over_wraps_negated_inequality():
    _, p1, p2 = _make_predicates()
    ctcs = And(p1, p2).over((0, 10))
    # The inner constraint should be Inequality(-stl_expr <= 0)
    assert isinstance(ctcs.constraint, Inequality)


def test_stl_and_over_returns_ctcs():
    _, p1, p2 = _make_predicates()
    assert isinstance(And(p1, p2).over((0, 5)), CTCS)


def test_stl_ifthen_over_returns_ctcs():
    _, p1, p2 = _make_predicates()
    assert isinstance(IfThen(p1, p2).over((0, 5)), CTCS)


def test_stl_integer_variable_over_returns_ctcs():
    x = State("g", shape=(1,))
    x._slice = slice(0, 1)
    iv = IntegerVariable(x, [0.0, 1.0, 2.0])
    assert isinstance(iv.over((0, 5)), CTCS)


# =============================================================================
# JAX Lowering – Or
# =============================================================================


def test_or_jax_positive_when_one_predicate_satisfied():
    fn = _setup_or_jax()
    # x[0]=0.5 satisfies p1; x[1]=3.0 violates p2
    x = jnp.array([0.5, 3.0])
    assert float(fn(x, None, None, None)) > 0.0


def test_or_jax_positive_when_both_satisfied():
    fn = _setup_or_jax()
    x = jnp.array([0.5, 1.5])
    assert float(fn(x, None, None, None)) > 0.0


def test_or_jax_negative_when_all_violated():
    fn = _setup_or_jax()
    x = jnp.array([2.0, 3.0])
    assert float(fn(x, None, None, None)) < 0.0


def test_or_jax_output_is_scalar():
    fn = _setup_or_jax()
    out = fn(jnp.array([0.5, 1.5]), None, None, None)
    assert jnp.shape(out) == ()


def test_or_jax_lite_variant():
    x_sym = State("x", shape=(2,))
    x_sym._slice = slice(0, 2)
    p1 = x_sym[0] <= Constant(np.array(1.0))
    p2 = x_sym[1] <= Constant(np.array(2.0))
    fn = _lower_stl(Or(p1, p2, lite=True))
    # lite variant: satisfied when at least one predicate satisfied
    x_ok = jnp.array([0.5, 3.0])
    assert float(fn(x_ok, None, None, None)) >= -1e-17


# =============================================================================
# JAX Lowering – And
# =============================================================================


def test_and_jax_positive_when_both_satisfied():
    fn = _setup_and_jax()
    x = jnp.array([0.5, 1.5])
    assert float(fn(x, None, None, None)) > 0.0


def test_and_jax_negative_when_one_violated():
    fn = _setup_and_jax()
    # x[0] ok, x[1] violated
    x = jnp.array([0.5, 3.0])
    assert float(fn(x, None, None, None)) < 0.0


def test_and_jax_negative_when_both_violated():
    fn = _setup_and_jax()
    x = jnp.array([2.0, 3.0])
    assert float(fn(x, None, None, None)) < 0.0


def test_and_jax_output_is_scalar():
    fn = _setup_and_jax()
    out = fn(jnp.array([0.5, 1.5]), None, None, None)
    assert jnp.shape(out) == ()


def test_and_jax_lite_variant():
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


def test_ifthen_jax_positive_when_condition_and_consequent_both_satisfied():
    fn = _setup_ifthen_jax()
    x = jnp.array([0.5, 1.5])  # cond OK, conseq OK → implication holds
    assert float(fn(x, None, None, None)) > 0.0


def test_ifthen_jax_positive_when_condition_not_satisfied():
    fn = _setup_ifthen_jax()
    # condition violated → implication trivially holds
    x = jnp.array([2.0, 3.0])
    assert float(fn(x, None, None, None)) > 0.0


def test_ifthen_jax_negative_when_condition_holds_consequent_violated():
    fn = _setup_ifthen_jax()
    # condition satisfied but consequent violated → implication fails
    x = jnp.array([0.5, 3.0])
    assert float(fn(x, None, None, None)) < 0.0


def test_ifthen_jax_output_is_scalar():
    fn = _setup_ifthen_jax()
    out = fn(jnp.array([0.5, 1.5]), None, None, None)
    assert jnp.shape(out) == ()


def test_ifthen_jax_lite_variant_condition_satisfied_consequent_violated():
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


def test_integer_variable_jax_zero_when_exactly_at_allowed_value():
    fn = _setup_integer_variable_jax()
    for v in [0.0, 1.0, 2.0]:
        out = float(fn(jnp.array([v]), None, None, None))
        assert out == pytest.approx(0.0, abs=1e-4), f"expected ~0 for g={v}"


def test_integer_variable_jax_negative_when_not_at_allowed_value():
    fn = _setup_integer_variable_jax()
    out = float(fn(jnp.array([0.5]), None, None, None))
    assert out < 0.0


def test_integer_variable_jax_output_is_scalar():
    fn = _setup_integer_variable_jax()
    out = fn(jnp.array([1.0]), None, None, None)
    assert jnp.shape(out) == ()


def test_integer_variable_jax_larger_deviation_more_negative():
    fn = _setup_integer_variable_jax()
    close = float(fn(jnp.array([0.1]), None, None, None))
    far = float(fn(jnp.array([0.4]), None, None, None))
    assert close > far  # closer to 0.0 → less negative = better


def test_integer_variable_jax_single_allowed_value():
    fn = _setup_integer_variable_jax(values=[3.0])
    assert float(fn(jnp.array([3.0]), None, None, None)) == pytest.approx(0.0, abs=1e-4)
    assert float(fn(jnp.array([3.5]), None, None, None)) < 0.0


# =============================================================================
# Not – Construction & Tree Structure
# =============================================================================


def test_not_basic_construction():
    _, p1, _ = _make_predicates()
    node = Not(p1)
    assert isinstance(node, STLExpr)
    assert node.predicate is p1


def test_not_children_returns_predicate():
    _, p1, _ = _make_predicates()
    assert Not(p1).children() == [p1]


def test_not_repr():
    _, p1, _ = _make_predicates()
    assert repr(Not(p1)).startswith("Not(")


def test_not_check_shape_returns_empty_tuple():
    _, p1, _ = _make_predicates()
    assert Not(p1).check_shape() == ()


def test_not_rejects_non_constraint_arg():
    with pytest.raises(TypeError):
        Not("bad")


def test_not_canonicalize_collapses_double_negation():
    _, p1, _ = _make_predicates()
    canon = Not(Not(p1)).canonicalize()
    # ~~p should collapse to (canonicalized) p, not a Not node
    assert not isinstance(canon, Not)


def test_not_canonicalize_preserves_single_negation():
    _, p1, _ = _make_predicates()
    canon = Not(p1).canonicalize()
    assert isinstance(canon, Not)


def test_not_accepts_nested_stl():
    _, p1, p2 = _make_predicates()
    inner = Or(p1, p2)
    node = Not(inner)
    assert isinstance(node, Not)
    assert node.predicate is inner


# =============================================================================
# Not – JAX Lowering
# =============================================================================


def _setup_not_jax():
    x_sym = State("x", shape=(2,))
    x_sym._slice = slice(0, 2)
    p = x_sym[0] <= Constant(np.array(1.0))  # x[0] <= 1
    return _lower_stl(Not(p))


def test_not_jax_negative_when_inner_satisfied():
    fn = _setup_not_jax()
    # x[0]=0.5 satisfies inner predicate → Not should be unsatisfied (negative)
    assert float(fn(jnp.array([0.5, 0.0]), None, None, None)) < 0.0


def test_not_jax_positive_when_inner_violated():
    fn = _setup_not_jax()
    # x[0]=2.0 violates inner predicate → Not should be satisfied (positive)
    assert float(fn(jnp.array([2.0, 0.0]), None, None, None)) > 0.0


def test_not_jax_double_negation_matches_inner():
    x_sym = State("x", shape=(2,))
    x_sym._slice = slice(0, 2)
    p = x_sym[0] <= Constant(np.array(1.0))
    inner_fn = _lower_stl(Or(p, x_sym[1] <= Constant(np.array(2.0))))
    double_fn = _lower_stl(Not(Not(Or(p, x_sym[1] <= Constant(np.array(2.0))))))
    x = jnp.array([0.5, 3.0])
    assert float(double_fn(x, None, None, None)) == pytest.approx(
        float(inner_fn(x, None, None, None)), abs=1e-6
    )


# =============================================================================
# Always – Construction & Tree Structure
# =============================================================================


def test_always_basic_construction():
    _, p1, _ = _make_predicates()
    node = Always(p1, (0, 5))
    assert isinstance(node, STLExpr)
    assert node.predicate is p1
    assert node.interval == NodeInterval(0, 5)


def test_always_interval_optional():
    _, p1, _ = _make_predicates()
    node = Always(p1)
    assert node.interval is None


def test_always_children_returns_predicate():
    _, p1, _ = _make_predicates()
    assert Always(p1, (0, 5)).children() == [p1]


def test_always_repr_with_interval():
    _, p1, _ = _make_predicates()
    r = repr(Always(p1, (0, 5)))
    assert r.startswith("Always(")
    assert "(0, 5)" in r


def test_always_repr_without_interval():
    _, p1, _ = _make_predicates()
    assert repr(Always(p1)).startswith("Always(")


def test_always_check_shape_returns_empty_tuple():
    _, p1, _ = _make_predicates()
    assert Always(p1, (0, 5)).check_shape() == ()


def test_always_rejects_non_constraint_predicate():
    with pytest.raises(TypeError):
        Always("bad", (0, 5))


def test_always_canonicalize_preserves_interval():
    _, p1, _ = _make_predicates()
    canon = Always(p1, (3, 7)).canonicalize()
    assert isinstance(canon, Always)
    assert canon.interval == NodeInterval(3, 7)


def test_always_accepts_nested_stl_predicate():
    _, p1, p2 = _make_predicates()
    inner = And(p1, p2)
    node = Always(inner, (0, 5))
    assert isinstance(node, Always)
    assert node.predicate is inner


# =============================================================================
# Always – .over() seals to CTCS
# =============================================================================


def test_always_over_uses_stored_interval():
    _, p1, _ = _make_predicates()
    ctcs = Always(p1, (3, 7)).over()
    assert isinstance(ctcs, CTCS)
    assert ctcs.nodes == (3, 7)


def test_always_over_rejects_interval_specified_twice():
    _, p1, _ = _make_predicates()
    with pytest.raises(ValueError, match="already carries an interval"):
        Always(p1, (3, 7)).over((0, 10))


def test_always_over_requires_interval_somewhere():
    _, p1, _ = _make_predicates()
    with pytest.raises(ValueError, match="requires an interval"):
        Always(p1).over()


def test_always_at_lowers_to_nodal_when_no_interval():
    _, p1, _ = _make_predicates()
    nodal = Always(p1).at([2, 3, 5])
    assert isinstance(nodal, NodalConstraint)
    assert nodal.nodes == [2, 3, 5]


def test_always_at_rejects_when_interval_set():
    _, p1, _ = _make_predicates()
    with pytest.raises(ValueError, match="already carries an interval"):
        Always(p1, (0, 5)).at([2, 3])


def test_always_over_with_stl_predicate_returns_ctcs():
    _, p1, p2 = _make_predicates()
    ctcs = Always(And(p1, p2), (0, 5)).over()
    assert isinstance(ctcs, CTCS)


# =============================================================================
# Always – JAX lowering (nested usage)
# =============================================================================


def _setup_always_nested_jax():
    """Always nested inside an And — children must be interval-free and
    inherit the ambient window from the enclosing .over()."""
    x_sym = State("x", shape=(2,))
    x_sym._slice = slice(0, 2)
    p1 = x_sym[0] <= Constant(np.array(1.0))
    p2 = x_sym[1] <= Constant(np.array(2.0))
    return _lower_stl(And(Always(p1), Always(p2)))


def test_always_nested_jax_positive_when_both_satisfied():
    fn = _setup_always_nested_jax()
    assert float(fn(jnp.array([0.5, 1.5]), None, None, None)) > 0.0


def test_always_nested_jax_negative_when_one_violated():
    fn = _setup_always_nested_jax()
    assert float(fn(jnp.array([0.5, 3.0]), None, None, None)) < 0.0


# =============================================================================
# Operator overloads on STLExpr (&, |, ~)
# =============================================================================


def test_stlexpr_or_operator_builds_or_node():
    _, p1, p2 = _make_predicates()
    node = Or(p1, p2) | Or(p2, p1)
    assert isinstance(node, Or)
    assert len(node.predicates) == 2


def test_stlexpr_and_operator_builds_and_node():
    _, p1, p2 = _make_predicates()
    node = Or(p1, p2) & Or(p2, p1)
    assert isinstance(node, And)
    assert len(node.predicates) == 2


def test_stlexpr_invert_operator_builds_not_node():
    _, p1, p2 = _make_predicates()
    node = ~Or(p1, p2)
    assert isinstance(node, Not)


def test_stlexpr_operator_with_constraint_on_left():
    """Constraint & STLExpr → STLExpr.__rand__ kicks in."""
    _, p1, p2 = _make_predicates()
    node = p1 & Or(p1, p2)
    assert isinstance(node, And)


def test_stlexpr_operator_with_constraint_on_right():
    _, p1, p2 = _make_predicates()
    node = Or(p1, p2) | p1
    assert isinstance(node, Or)


def test_constraint_does_not_have_and_overload():
    """Bare Constraint & Constraint should NOT build an STL node."""
    _, p1, p2 = _make_predicates()
    with pytest.raises(TypeError):
        _ = p1 & p2  # type: ignore[operator]


def test_stlexpr_operator_composition_lowers_correctly():
    """(Or & Or) | ~Or should produce a valid lowered function."""
    x_sym = State("x", shape=(2,))
    x_sym._slice = slice(0, 2)
    p1 = x_sym[0] <= Constant(np.array(1.0))
    p2 = x_sym[1] <= Constant(np.array(2.0))
    spec = (Or(p1, p2) & Or(p1, p2)) | ~Or(p1, p2)
    fn = _lower_stl(spec)
    out = fn(jnp.array([0.5, 1.5]), None, None, None)
    assert jnp.shape(out) == ()


# =============================================================================
# Eventually & Until – placeholders
# =============================================================================


def test_eventually_construction_succeeds():
    _, p1, _ = _make_predicates()
    node = Eventually(p1, (0, 5))
    assert isinstance(node, STLExpr)
    assert node.predicate is p1
    assert node.interval == NodeInterval(0, 5)


def test_eventually_canonicalize_raises():
    _, p1, _ = _make_predicates()
    with pytest.raises(NotImplementedError, match="Eventually"):
        Eventually(p1, (0, 5)).canonicalize()


def test_eventually_over_raises():
    _, p1, _ = _make_predicates()
    with pytest.raises(NotImplementedError, match="Eventually"):
        Eventually(p1, (0, 5)).over((0, 5))


def test_eventually_at_raises():
    _, p1, _ = _make_predicates()
    with pytest.raises(NotImplementedError, match="Eventually"):
        Eventually(p1, (0, 5)).at([0])


def test_eventually_repr():
    _, p1, _ = _make_predicates()
    r = repr(Eventually(p1, (0, 5)))
    assert "Eventually" in r


def test_until_construction_succeeds():
    _, p1, p2 = _make_predicates()
    node = Until(p1, p2, (0, 5))
    assert isinstance(node, STLExpr)
    assert node.left is p1
    assert node.right is p2
    assert node.interval == NodeInterval(0, 5)


def test_until_children_returns_both_sides():
    _, p1, p2 = _make_predicates()
    assert Until(p1, p2, (0, 5)).children() == [p1, p2]


def test_until_canonicalize_raises():
    _, p1, p2 = _make_predicates()
    with pytest.raises(NotImplementedError, match="Until"):
        Until(p1, p2, (0, 5)).canonicalize()


def test_until_over_raises():
    _, p1, p2 = _make_predicates()
    with pytest.raises(NotImplementedError, match="Until"):
        Until(p1, p2, (0, 5)).over((0, 5))


# =============================================================================
# Interval.coerce – explicit interval_type dispatch
# =============================================================================


def test_coerce_default_kind_is_nodes():
    assert Interval.coerce((0, 5)) == NodeInterval(0, 5)


def test_coerce_seconds_kind_builds_time_interval():
    iv = Interval.coerce((0.0, 5.0), "seconds")
    assert isinstance(iv, TimeInterval)
    assert iv.start == 0.0 and iv.end == 5.0


def test_coerce_seconds_accepts_int_bounds():
    # "seconds" is permissive: any numeric is fine
    iv = Interval.coerce((0, 5), "seconds")
    assert isinstance(iv, TimeInterval)


def test_coerce_nodes_rejects_non_integral_float():
    with pytest.raises(TypeError, match="integral"):
        Interval.coerce((0.0, 5.5), "nodes")


def test_coerce_nodes_accepts_integral_float():
    # 5.0 is exactly an integer — fine under "nodes"
    assert Interval.coerce((0.0, 5.0), "nodes") == NodeInterval(0, 5)


def test_coerce_invalid_kind_raises():
    with pytest.raises(ValueError, match="interval_type"):
        Interval.coerce((0, 5), "frames")


def test_coerce_passthrough_typed_interval_matching_kind():
    iv = NodeInterval(0, 5)
    assert Interval.coerce(iv, "nodes") is iv


def test_coerce_passthrough_typed_interval_mismatched_kind_raises():
    iv = TimeInterval(0.0, 5.0)
    with pytest.raises(TypeError, match="nodes"):
        Interval.coerce(iv, "nodes")


def test_always_interval_type_seconds_builds_time_interval():
    _, p1, _ = _make_predicates()
    node = Always(p1, (0.0, 5.0), interval_type="seconds")
    assert isinstance(node.interval, TimeInterval)


def test_always_over_interval_type_seconds_raises_not_implemented():
    _, p1, _ = _make_predicates()
    with pytest.raises(NotImplementedError, match="TimeInterval"):
        Always(p1).over((0.0, 5.0), interval_type="seconds")


def test_stl_over_interval_type_seconds_raises_not_implemented():
    _, p1, p2 = _make_predicates()
    with pytest.raises(NotImplementedError, match="TimeInterval"):
        Or(p1, p2).over((0.0, 5.0), interval_type="seconds")


def test_eventually_interval_optional():
    _, p1, _ = _make_predicates()
    node = Eventually(p1)
    assert node.interval is None


def test_until_interval_optional():
    _, p1, p2 = _make_predicates()
    node = Until(p1, p2)
    assert node.interval is None
    assert node.left is p1 and node.right is p2


def test_until_interval_type_seconds():
    _, p1, p2 = _make_predicates()
    node = Until(p1, p2, (0.0, 5.0), interval_type="seconds")
    assert isinstance(node.interval, TimeInterval)


def test_until_repr():
    _, p1, p2 = _make_predicates()
    r = repr(Until(p1, p2, (0, 5)))
    assert "Until" in r


# =============================================================================
# Citation Tests
# =============================================================================


def test_stl_operators_cite_gmsr():
    _, p1, p2 = _make_predicates()
    for node in (Or(p1, p2), And(p1, p2), Not(p1), Always(p1)):
        entries = node.citation()
        assert len(entries) == 1
        assert "uzun2024gmsr" in entries[0]


def test_stljax_operators_cite_stljax():
    from openscvx.symbolic.expr import stljax

    _, p1, p2 = _make_predicates()
    entries = stljax.Or(p1, p2).citation()
    assert len(entries) == 1
    assert "kapoor2025stlcg" in entries[0]
