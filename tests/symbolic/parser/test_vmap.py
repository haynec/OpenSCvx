"""Tests for parsing Vmap expressions.

Syntax: ``Vmap(name: source, ... [, axis=N] -> body_expr)``
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import Control, Parameter, State
from openscvx.symbolic.expr.linalg import Norm
from openscvx.symbolic.expr.vmap import Vmap, _Placeholder
from openscvx.symbolic.parser import ExprParser
from openscvx.symbolic.parser.parser import ParseError

# =============================================================================
# Helpers
# =============================================================================


def _parser(**extra):
    pos = State("pos", shape=(3,))
    vel = State("vel", shape=(3,))
    thrust = Control("thrust", shape=(3,))
    symbols = {"pos": pos, "vel": vel, "thrust": thrust}
    symbols.update(extra)
    return ExprParser(symbols), symbols


# =============================================================================
# Single-batch bindings
# =============================================================================


def test_vmap_single_parameter_batch():
    refs = Parameter("refs", shape=(10, 3), value=np.zeros((10, 3)))
    p, syms = _parser(refs=refs)
    expr = p.parse("Vmap(r: refs -> Norm(pos - r))")
    assert isinstance(expr, Vmap)
    assert expr.num_batches == 1
    assert expr._is_parameter == (True,)
    assert expr.batch is refs
    assert expr._axis == 0


def test_vmap_single_state_batch():
    agents = State("agents", shape=(5, 3))
    p, _ = _parser(agents=agents)
    expr = p.parse("Vmap(a: agents -> Norm(a))")
    assert isinstance(expr, Vmap)
    assert expr._is_state == (True,)
    assert expr.batch is agents


def test_vmap_single_control_batch():
    thrusters = Control("thrusters", shape=(4,))
    p, _ = _parser(thrusters=thrusters)
    expr = p.parse("Vmap(t: thrusters -> t * 2.0)")
    assert isinstance(expr, Vmap)
    assert expr._is_control == (True,)
    assert expr.batch is thrusters


# =============================================================================
# Multiple-batch bindings
# =============================================================================


def test_vmap_multi_batch():
    centers = Parameter("centers", shape=(100, 3), value=np.zeros((100, 3)))
    radii = Parameter("radii", shape=(100,), value=np.ones(100))
    p, _ = _parser(centers=centers, radii=radii)
    expr = p.parse("Vmap(c: centers, r: radii -> Norm(pos - c) - r)")
    assert isinstance(expr, Vmap)
    assert expr.num_batches == 2
    assert expr._is_parameter == (True, True)


# =============================================================================
# Axis keyword
# =============================================================================


def test_vmap_axis_kwarg():
    data = Parameter("data", shape=(3, 10), value=np.zeros((3, 10)))
    p, _ = _parser(data=data)
    expr = p.parse("Vmap(x: data, axis=1 -> Norm(x))")
    assert isinstance(expr, Vmap)
    assert expr._axis == 1
    # Per-element shape should be (3,) — kept dim 0, removed axis 1
    assert expr.placeholder.shape == (3,)


def test_vmap_axis_before_binding():
    """axis= can appear before bindings."""
    data = Parameter("data", shape=(3, 10), value=np.zeros((3, 10)))
    p, _ = _parser(data=data)
    expr = p.parse("Vmap(axis=1, x: data -> Norm(x))")
    assert isinstance(expr, Vmap)
    assert expr._axis == 1


# =============================================================================
# Placeholder shapes
# =============================================================================


def test_placeholder_shape_axis0():
    refs = Parameter("refs", shape=(10, 3), value=np.zeros((10, 3)))
    p, _ = _parser(refs=refs)
    expr = p.parse("Vmap(r: refs -> Norm(r))")
    assert expr.placeholder.shape == (3,)


def test_placeholder_shape_scalar():
    vals = Parameter("vals", shape=(10,), value=np.zeros(10))
    p, _ = _parser(vals=vals)
    expr = p.parse("Vmap(v: vals -> v * 2.0)")
    assert expr.placeholder.shape == ()


# =============================================================================
# Output shape
# =============================================================================


def test_vmap_check_shape():
    refs = Parameter("refs", shape=(10, 3), value=np.zeros((10, 3)))
    p, _ = _parser(refs=refs)
    expr = p.parse("Vmap(r: refs -> Norm(pos - r))")
    # Norm produces scalar, vmapped over 10 → (10,)
    assert expr.check_shape() == (10,)


# =============================================================================
# Body expression uses full grammar
# =============================================================================


def test_vmap_body_with_constraint():
    refs = Parameter("refs", shape=(10, 3), value=np.zeros((10, 3)))
    p, _ = _parser(refs=refs)
    expr = p.parse("Vmap(r: refs -> Norm(pos - r) >= 2.0)")
    assert isinstance(expr, Vmap)


def test_vmap_body_with_indexing():
    refs = Parameter("refs", shape=(10, 3), value=np.zeros((10, 3)))
    p, _ = _parser(refs=refs)
    expr = p.parse("Vmap(r: refs -> Norm(pos[:2] - r[:2]))")
    assert isinstance(expr, Vmap)


# =============================================================================
# Symbol table restoration
# =============================================================================


def test_placeholder_does_not_leak():
    """Placeholder names should not persist in the symbol table after parsing."""
    refs = Parameter("refs", shape=(10, 3), value=np.zeros((10, 3)))
    p, syms = _parser(refs=refs)
    p.parse("Vmap(r: refs -> Norm(r))")
    assert "r" not in syms


def test_placeholder_shadows_and_restores():
    """A placeholder that shadows an existing symbol should restore it."""
    refs = Parameter("refs", shape=(10, 3), value=np.zeros((10, 3)))
    r_state = State("r", shape=(3,))
    p, syms = _parser(refs=refs, r=r_state)
    p.parse("Vmap(r: refs -> Norm(r))")
    assert syms["r"] is r_state


# =============================================================================
# Error cases
# =============================================================================


def test_vmap_unknown_source():
    p, _ = _parser()
    with pytest.raises(ParseError, match="Unknown batch source"):
        p.parse("Vmap(x: nonexistent -> x)")


def test_vmap_no_bindings():
    p, _ = _parser()
    with pytest.raises(ParseError, match="at least one binding"):
        p.parse("Vmap(-> pos)")


def test_vmap_axis_out_of_bounds():
    refs = Parameter("refs", shape=(10, 3), value=np.zeros((10, 3)))
    p, _ = _parser(refs=refs)
    with pytest.raises(ParseError, match="out of bounds"):
        p.parse("Vmap(r: refs, axis=5 -> Norm(r))")


def test_vmap_batch_size_mismatch():
    a = Parameter("a", shape=(10, 3), value=np.zeros((10, 3)))
    b = Parameter("b", shape=(7,), value=np.zeros(7))
    p, _ = _parser(a=a, b=b)
    with pytest.raises(ParseError, match="Batch size mismatch"):
        p.parse("Vmap(x: a, y: b -> Norm(x) + y)")
