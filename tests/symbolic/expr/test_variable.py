"""Tests for variable nodes.

This module tests variable node types: Variable, State, Control.

Tests are organized by node type, with each section containing:
1. Node creation and properties
2. Shape checking
3. Canonicalization
4. JAX lowering tests
5. CVXPY lowering tests
"""

import pytest

# =============================================================================
# Variable (Base Class)
# =============================================================================

# --- Variable: Creation ---


def test_variable_creation():
    """Test basic Variable creation and properties."""
    from openscvx.symbolic.expr.variable import Variable

    v = Variable("x", shape=(3,))
    assert v.name == "x"
    assert v.shape == (3,)
    assert repr(v) == "Var('x')"
    assert v._min is None
    assert v._max is None
    assert v._guess is None


def test_variable_min_max_bounds():
    """Test setting min/max bounds on Variable."""
    import numpy as np

    from openscvx.symbolic.expr.variable import Variable

    v = Variable("x", shape=(2,))
    v.min = [-5.0, -10.0]
    v.max = [5.0, 10.0]
    assert np.allclose(v.min, [-5.0, -10.0])
    assert np.allclose(v.max, [5.0, 10.0])


def test_variable_guess():
    """Test setting initial guess trajectory."""
    import numpy as np

    from openscvx.symbolic.expr.variable import Variable

    v = Variable("x", shape=(2,))
    guess = np.linspace([0, 0], [10, 10], 50)
    v.guess = guess
    assert v.guess.shape == (50, 2)
    assert np.allclose(v.guess, guess)


# --- Variable: Shape Checking ---


def test_variable_min_shape_validation():
    """Test that min bounds must match variable shape."""
    from openscvx.symbolic.expr.variable import Variable

    v = Variable("x", shape=(3,))
    with pytest.raises(ValueError, match="min must be 1D with shape"):
        v.min = [1.0, 2.0]  # Wrong shape


def test_variable_max_shape_validation():
    """Test that max bounds must match variable shape."""
    from openscvx.symbolic.expr.variable import Variable

    v = Variable("x", shape=(3,))
    with pytest.raises(ValueError, match="max must be 1D with shape"):
        v.max = [1.0, 2.0, 3.0, 4.0]  # Wrong shape


def test_variable_guess_shape_validation():
    """Test that guess must be 2D with correct second dimension."""
    import numpy as np

    from openscvx.symbolic.expr.variable import Variable

    v = Variable("x", shape=(3,))
    with pytest.raises(ValueError, match="Guess must be a 2D array"):
        v.guess = np.array([1.0, 2.0, 3.0])  # 1D instead of 2D

    with pytest.raises(ValueError, match="Guess must have second dimension"):
        v.guess = np.zeros((10, 2))  # Wrong second dimension


# --- Variable: Canonicalization ---


def test_variable_canonicalize():
    """Test that Variable canonicalize returns itself unchanged."""
    from openscvx.symbolic.expr.variable import Variable

    v = Variable("x", shape=(3,))
    v_canon = v.canonicalize()
    assert v_canon is v  # Should return same object


# =============================================================================
# State
# =============================================================================

# --- State: Creation ---


def test_state_creation():
    """Test basic State creation and properties."""
    from openscvx.symbolic.expr import State

    s = State("pos", shape=(3,))
    assert s.name == "pos"
    assert s.shape == (3,)
    assert repr(s) == "State('pos', shape=(3,))"
    assert s._initial is None
    assert s._final is None


def test_state_boundary_conditions_fixed():
    """Test setting fixed boundary conditions on State."""
    import numpy as np

    from openscvx.symbolic.expr import State

    s = State("x", shape=(2,))
    s.min = [-10.0, -10.0]
    s.max = [10.0, 10.0]
    s.initial = [0.0, 1.0]  # Fixed by default
    s.final = [5.0, 6.0]

    assert np.allclose(s.initial, [0.0, 1.0])
    assert np.allclose(s.final, [5.0, 6.0])
    assert all(s.initial_type == "Fix")
    assert all(s.final_type == "Fix")


def test_state_boundary_conditions_mixed():
    """Test mixed boundary condition types."""
    import numpy as np

    from openscvx.symbolic.expr import State

    s = State("x", shape=(3,))
    s.min = [0.0, 0.0, 0.0]
    s.max = [10.0, 10.0, 10.0]
    s.initial = [0, ("free", 1.0), ("minimize", 2.0)]
    s.final = [10, ("maximize", 8.0), ("free", 5.0)]

    assert np.allclose(s.initial, [0.0, 1.0, 2.0])
    assert np.allclose(s.final, [10.0, 8.0, 5.0])
    assert s.initial_type[0] == "Fix"
    assert s.initial_type[1] == "Free"
    assert s.initial_type[2] == "Minimize"
    assert s.final_type[0] == "Fix"
    assert s.final_type[1] == "Maximize"
    assert s.final_type[2] == "Free"


def test_boundary_condition_helpers():
    """Test the Free, Fixed, Minimize, Maximize helper functions."""
    import numpy as np

    import openscvx as ox
    from openscvx.symbolic.expr import Fixed, Free, Maximize, Minimize, State

    # Test that helpers return correct tuples
    assert Free(5.0) == ("free", 5.0)
    assert Fixed(10.0) == ("fixed", 10.0)
    assert Minimize(3.0) == ("minimize", 3.0)
    assert Maximize(7.0) == ("maximize", 7.0)

    # Test using helpers with State
    s = State("x", shape=(3,))
    s.min = [0.0, 0.0, 0.0]
    s.max = [10.0, 10.0, 10.0]
    s.initial = [Fixed(0), Free(1.0), Minimize(2.0)]
    s.final = [10, Maximize(8.0), Free(5.0)]

    assert np.allclose(s.initial, [0.0, 1.0, 2.0])
    assert np.allclose(s.final, [10.0, 8.0, 5.0])
    # Note: Fixed() returns "Fixed" (capitalized), plain numbers return "Fix"
    assert s.initial_type[0] == "Fixed"
    assert s.initial_type[1] == "Free"
    assert s.initial_type[2] == "Minimize"
    assert s.final_type[0] == "Fix"  # Plain number
    assert s.final_type[1] == "Maximize"
    assert s.final_type[2] == "Free"

    # Test using helpers via ox namespace
    s2 = ox.State("y", shape=(2,))
    s2.min = [0.0, 0.0]
    s2.max = [5.0, 5.0]
    s2.initial = [ox.Free(1.0), ox.Fixed(2.0)]
    s2.final = [ox.Minimize(3.0), ox.Maximize(4.0)]

    assert np.allclose(s2.initial, [1.0, 2.0])
    assert np.allclose(s2.final, [3.0, 4.0])
    assert s2.initial_type[0] == "Free"
    assert s2.initial_type[1] == "Fixed"  # Fixed() returns "Fixed"
    assert s2.final_type[0] == "Minimize"
    assert s2.final_type[1] == "Maximize"

    # Test using helpers with Time
    # Time is now a State subclass, so initial/final return arrays
    from openscvx import Time

    time1 = Time(
        initial=0.0,  # Plain number for fixed
        final=ox.Minimize(10.0),
        min=0.0,
        max=20.0,
    )
    # Time.initial/final return numpy arrays (State behavior)
    assert np.allclose(time1.initial, [0.0])
    assert np.allclose(time1.final, [10.0])
    assert time1.final_type[0] == "Minimize"

    time2 = Time(
        initial=0.0,  # Plain number still works
        final=ox.Free(5.0),
        min=0.0,
        max=20.0,
    )
    assert np.allclose(time2.initial, [0.0])
    assert np.allclose(time2.final, [5.0])
    assert time2.final_type[0] == "Free"

    time3 = Time(
        initial=ox.Maximize(0.0),
        final=10.0,  # Plain number for fixed
        min=0.0,
        max=20.0,
    )
    assert np.allclose(time3.initial, [0.0])
    assert time3.initial_type[0] == "Maximize"
    assert np.allclose(time3.final, [10.0])


# --- State: Shape Checking ---


def test_state_min_max_shape_validation():
    """Test that State min/max must match state shape exactly."""
    from openscvx.symbolic.expr import State

    s = State("x", shape=(3,))
    with pytest.raises(ValueError, match="Min shape .* does not match State shape"):
        s.min = [1.0, 2.0]  # Wrong shape

    with pytest.raises(ValueError, match="Max shape .* does not match State shape"):
        s.max = [1.0, 2.0, 3.0, 4.0]  # Wrong shape


def test_state_initial_final_shape_validation():
    """Test that initial/final conditions must match state shape."""
    from openscvx.symbolic.expr import State

    s = State("x", shape=(3,))
    with pytest.raises(ValueError, match="Length mismatch"):
        s.initial = [0.0, 1.0]  # Wrong length

    with pytest.raises(ValueError, match="Length mismatch"):
        s.final = [0.0, 1.0, 2.0, 3.0]  # Wrong length


def test_state_bounds_validation():
    """Test that fixed boundary conditions must respect min/max bounds."""
    from openscvx.symbolic.expr import State

    # Test initial bounds violation
    s1 = State("x", shape=(2,))
    s1.min = [0.0, 0.0]
    s1.max = [10.0, 10.0]
    with pytest.raises(ValueError, match="Initial Fixed value .* is lower then the min"):
        s1.initial = [-1.0, 5.0]  # -1 < 0

    # Test final bounds violation
    s2 = State("x", shape=(2,))
    s2.min = [0.0, 0.0]
    s2.max = [10.0, 10.0]
    with pytest.raises(ValueError, match="Final Fixed value .* is greater then the max"):
        s2.final = [5.0, 15.0]  # 15 > 10


# --- State: Canonicalization ---


def test_state_canonicalize():
    """Test that State canonicalize returns itself unchanged."""
    from openscvx.symbolic.expr import State

    s = State("x", shape=(3,))
    s_canon = s.canonicalize()
    assert s_canon is s  # Should return same object


# =============================================================================
# Control
# =============================================================================

# --- Control: Creation ---


def test_control_creation():
    """Test basic Control creation and properties."""
    from openscvx.symbolic.expr import Control

    c = Control("thrust", shape=(3,))
    assert c.name == "thrust"
    assert c.shape == (3,)
    assert repr(c) == "Control('thrust', shape=(3,))"
    assert c._min is None
    assert c._max is None
    assert c._guess is None


def test_control_bounds():
    """Test setting min/max bounds on Control."""
    import numpy as np

    from openscvx.symbolic.expr import Control

    c = Control("u", shape=(2,))
    c.min = [-1.0, 0.0]
    c.max = [1.0, 10.0]
    assert np.allclose(c.min, [-1.0, 0.0])
    assert np.allclose(c.max, [1.0, 10.0])


# --- Control: Shape Checking ---


def test_control_min_max_shape_validation():
    """Test that Control min/max must match control shape."""
    from openscvx.symbolic.expr import Control

    c = Control("u", shape=(3,))
    with pytest.raises(ValueError, match="min must be 1D with shape"):
        c.min = [1.0, 2.0]  # Wrong shape

    with pytest.raises(ValueError, match="max must be 1D with shape"):
        c.max = [1.0, 2.0, 3.0, 4.0]  # Wrong shape


# --- Control: Canonicalization ---


def test_control_canonicalize():
    """Test that Control canonicalize returns itself unchanged."""
    from openscvx.symbolic.expr import Control

    c = Control("u", shape=(3,))
    c_canon = c.canonicalize()
    assert c_canon is c  # Should return same object


# --- State & Control: JAX Lowering ---


def test_jax_lower_state_without_slice_raises():
    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    s = State("s", (3,))
    jl = JaxLowerer()
    with pytest.raises(ValueError):
        jl.lower(s)


def test_jax_lower_control_without_slice_raises():
    from openscvx.symbolic.expr import Control
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    c = Control("c", (2,))
    jl = JaxLowerer()
    with pytest.raises(ValueError):
        jl.lower(c)


def test_jax_lower_state_with_slice():
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.arange(10.0)
    s = State("s", (4,))
    s._slice = slice(2, 6)
    jl = JaxLowerer()
    f = jl.lower(s)
    out = f(x, None, None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (4,)
    assert jnp.allclose(out, x[2:6])


def test_jax_lower_control_with_slice():
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Control
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    u = jnp.arange(8.0)
    c = Control("c", (3,))
    c._slice = slice(5, 8)
    jl = JaxLowerer()
    f = jl.lower(c)
    out = f(None, u, None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (3,)
    assert jnp.allclose(out, u[5:8])


# --- State & Control: CVXPY Lowering ---


def test_cvxpy_state_variable():
    """Test lowering state variables"""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    # Create CVXPy variables
    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    # Create symbolic state
    x = State("x", shape=(3,))

    # Lower to CVXPy
    result = lowerer.lower(x)
    assert result is x_cvx  # Should return the mapped variable


def test_cvxpy_state_variable_with_slice():
    """Test state variables with slices"""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 6), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    # State with slice
    x = State("x", shape=(3,))
    x._slice = slice(0, 3)

    result = lowerer.lower(x)
    # Should return x_cvx with slice applied
    assert isinstance(result, cp.Expression)


def test_cvxpy_control_variable():
    """Test lowering control variables"""
    import cvxpy as cp

    from openscvx.symbolic.expr import Control
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    u_cvx = cp.Variable((10, 2), name="u")
    variable_map = {"u": u_cvx}
    lowerer = CvxpyLowerer(variable_map)

    u = Control("u", shape=(2,))
    result = lowerer.lower(u)
    assert result is u_cvx


def test_cvxpy_missing_state_variable_error():
    """Test error when state vector not in map"""

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    lowerer = CvxpyLowerer({})
    x = State("missing", shape=(3,))

    with pytest.raises(ValueError, match="State vector 'x' not found"):
        lowerer.lower(x)


def test_cvxpy_missing_control_variable_error():
    """Test error when control vector not in map"""

    from openscvx.symbolic.expr import Control
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    lowerer = CvxpyLowerer({})
    u = Control("thrust", shape=(2,))

    with pytest.raises(ValueError, match="Control vector 'u' not found"):
        lowerer.lower(u)
