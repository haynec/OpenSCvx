"""Tests for variable nodes.

This module tests variable node types: Variable, State, Control.

Tests are organized by node type, with each section containing:
1. Node creation and properties
2. Shape checking
3. Canonicalization
4. JAX lowering tests
5. CVXPY lowering tests
"""

import numpy as np
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

    from openscvx.symbolic.expr.variable import Variable

    v = Variable("x", shape=(2,))
    v.min = [-5.0, -10.0]
    v.max = [5.0, 10.0]
    assert np.allclose(v.min, [-5.0, -10.0])
    assert np.allclose(v.max, [5.0, 10.0])


def test_variable_guess():
    """Test setting initial guess trajectory."""

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
    with pytest.raises(ValueError, match="min expected shape"):
        v.min = [1.0, 2.0]  # Wrong shape


def test_variable_max_shape_validation():
    """Test that max bounds must match variable shape."""
    from openscvx.symbolic.expr.variable import Variable

    v = Variable("x", shape=(3,))
    with pytest.raises(ValueError, match="max expected shape"):
        v.max = [1.0, 2.0, 3.0, 4.0]  # Wrong shape


def test_variable_guess_shape_validation():
    """Test that guess must be 2D with correct second dimension."""

    from openscvx.symbolic.expr.variable import Variable

    v = Variable("x", shape=(3,))
    with pytest.raises(ValueError, match="guess expected 2D array"):
        v.guess = np.array([1.0, 2.0, 3.0])  # 1D instead of 2D

    with pytest.raises(ValueError, match="guess expected second dimension"):
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


def test_state_creation_with_kwargs():
    """Test State creation with constructor kwargs matches setter style."""

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.expr.state import Free, Minimize

    # Constructor style
    s1 = State(
        "pos",
        shape=(3,),
        min=[0.0, 0.0, 0.0],
        max=[10.0, 10.0, 10.0],
        initial=[0.0, 1.0, 2.0],
        final=[10.0, ("free", 5.0), Minimize(8.0)],
    )

    # Setter style
    s2 = State("pos", shape=(3,))
    s2.min = [0.0, 0.0, 0.0]
    s2.max = [10.0, 10.0, 10.0]
    s2.initial = [0.0, 1.0, 2.0]
    s2.final = [10.0, Free(5.0), Minimize(8.0)]

    assert np.allclose(s1.min, s2.min)
    assert np.allclose(s1.max, s2.max)
    assert np.allclose(s1.initial, s2.initial)
    assert np.allclose(s1.final, s2.final)
    assert list(s1.initial_type) == list(s2.initial_type)
    assert list(s1.final_type) == list(s2.final_type)


def test_state_creation_partial_kwargs():
    """Test State creation with only some kwargs."""

    from openscvx.symbolic.expr import State

    s = State("vel", shape=(2,), min=[-5.0, -5.0], max=[5.0, 5.0])
    assert np.allclose(s.min, [-5.0, -5.0])
    assert np.allclose(s.max, [5.0, 5.0])
    assert s._initial is None
    assert s._final is None


def test_state_boundary_conditions_fixed():
    """Test setting fixed boundary conditions on State."""

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
    with pytest.raises(ValueError, match="State 'x': min expected shape"):
        s.min = [1.0, 2.0]  # Wrong shape

    with pytest.raises(ValueError, match="State 'x': max expected shape"):
        s.max = [1.0, 2.0, 3.0, 4.0]  # Wrong shape


def test_state_initial_final_shape_validation():
    """Test that initial/final conditions must match state shape."""
    from openscvx.symbolic.expr import State

    s = State("x", shape=(3,))
    with pytest.raises(ValueError, match="State 'x': initial expected 3 elements"):
        s.initial = [0.0, 1.0]  # Wrong length

    with pytest.raises(ValueError, match="State 'x': final expected 3 elements"):
        s.final = [0.0, 1.0, 2.0, 3.0]  # Wrong length


def test_state_bounds_validation():
    """Test that fixed boundary conditions must respect min/max bounds."""
    from openscvx.symbolic.expr import State

    # Test initial bounds violation
    s1 = State("x", shape=(2,))
    s1.min = [0.0, 0.0]
    s1.max = [10.0, 10.0]
    with pytest.raises(ValueError, match="State 'x': initial fixed value .* violates min bound"):
        s1.initial = [-1.0, 5.0]  # -1 < 0

    # Test final bounds violation
    s2 = State("x", shape=(2,))
    s2.min = [0.0, 0.0]
    s2.max = [10.0, 10.0]
    with pytest.raises(ValueError, match="State 'x': final fixed value .* violates max bound"):
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

    c = Control("thrust", shape=(2,))
    assert c.name == "thrust"
    assert c.shape == (2,)
    expected = "Control('thrust', shape=(2,), impulsive=[False False], nodes=None)"
    assert repr(c) == expected
    assert c._min is None
    assert c._max is None
    assert c._guess is None


def test_control_creation_with_kwargs():
    """Test Control creation with constructor kwargs matches setter style."""

    from openscvx.symbolic.expr import Control

    # Constructor style
    c1 = Control("thrust", shape=(3,), min=[-10, -10, 0], max=[10, 10, 50])

    # Setter style
    c2 = Control("thrust", shape=(3,))
    c2.min = [-10, -10, 0]
    c2.max = [10, 10, 50]

    assert np.allclose(c1.min, c2.min)
    assert np.allclose(c1.max, c2.max)


def test_control_bounds():
    """Test setting min/max bounds on Control."""

    from openscvx.symbolic.expr import Control

    c = Control("u", shape=(2,))
    c.min = [-1.0, 0.0]
    c.max = [1.0, 10.0]
    assert np.allclose(c.min, [-1.0, 0.0])
    assert np.allclose(c.max, [1.0, 10.0])


def test_control_hold_kwarg_and_property():
    """``hold`` accepts FOH/ZOH/None and appears in ``repr`` when set."""
    from openscvx.symbolic.expr import Control

    c0 = Control("u", shape=(1,))
    assert c0.hold is None

    c1 = Control("u", shape=(1,), hold="ZOH")
    assert c1.hold == "ZOH"
    assert "hold='ZOH'" in repr(c1)

    c1.hold = "FOH"
    assert c1.hold == "FOH"

    with pytest.raises(ValueError, match="hold must be"):
        Control("bad", shape=(1,), hold="BOH")

    with pytest.raises(ValueError, match="hold must be"):
        c1.hold = "X"


# --- Control: Shape Checking ---


def test_control_min_max_shape_validation():
    """Test that Control min/max must match control shape."""
    from openscvx.symbolic.expr import Control

    c = Control("u", shape=(3,))
    with pytest.raises(ValueError, match="min expected shape"):
        c.min = [1.0, 2.0]  # Wrong shape

    with pytest.raises(ValueError, match="max expected shape"):
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


# =============================================================================
# Time
# =============================================================================

# --- Time: Creation ---


def test_time_constructor_style():
    """Test Time creation with all constructor args (existing API)."""

    from openscvx import Time
    from openscvx.symbolic.expr.state import Minimize

    t = Time(initial=0.0, final=Minimize(10.0), min=0.0, max=20.0)
    assert np.allclose(t.initial, [0.0])
    assert np.allclose(t.final, [10.0])
    assert np.allclose(t.min, [0.0])
    assert np.allclose(t.max, [20.0])
    assert t.initial_type[0] == "Fix"
    assert t.final_type[0] == "Minimize"


def test_time_setter_style():
    """Test Time creation with setter-based API."""

    from openscvx import Time
    from openscvx.symbolic.expr.state import Minimize

    t = Time()
    t.min = 0.0
    t.max = 20.0
    t.initial = 0.0
    t.final = Minimize(10.0)

    assert np.allclose(t.initial, [0.0])
    assert np.allclose(t.final, [10.0])
    assert np.allclose(t.min, [0.0])
    assert np.allclose(t.max, [20.0])
    assert t.initial_type[0] == "Fix"
    assert t.final_type[0] == "Minimize"


def test_time_setter_and_constructor_equivalent():
    """Test that constructor and setter styles produce identical results."""

    from openscvx import Time
    from openscvx.symbolic.expr.state import Free

    # Constructor style
    t1 = Time(initial=0.0, final=Free(5.0), min=0.0, max=10.0)

    # Setter style
    t2 = Time()
    t2.min = 0.0
    t2.max = 10.0
    t2.initial = 0.0
    t2.final = Free(5.0)

    assert np.allclose(t1.min, t2.min)
    assert np.allclose(t1.max, t2.max)
    assert np.allclose(t1.initial, t2.initial)
    assert np.allclose(t1.final, t2.final)
    assert list(t1.initial_type) == list(t2.initial_type)
    assert list(t1.final_type) == list(t2.final_type)


def test_time_partial_construction():
    """Test Time with partial constructor args then setters for the rest."""

    from openscvx import Time

    t = Time(min=0.0, max=20.0)
    assert np.allclose(t.min, [0.0])
    assert np.allclose(t.max, [20.0])
    assert t._initial is None
    assert t._final is None

    t.initial = 0.0
    t.final = 10.0
    assert np.allclose(t.initial, [0.0])
    assert np.allclose(t.final, [10.0])


def test_time_setter_accepts_arrays():
    """Test that Time setters also accept array-form values."""

    from openscvx import Time

    t = Time()
    t.min = [0.0]
    t.max = [20.0]
    t.initial = [0.0]
    t.final = [10.0]
    assert np.allclose(t.min, [0.0])
    assert np.allclose(t.max, [20.0])
    assert np.allclose(t.initial, [0.0])
    assert np.allclose(t.final, [10.0])


def test_time_repr():
    """Test Time repr for constructed, partial, and empty."""
    from openscvx import Time

    t1 = Time(initial=0.0, final=10.0, min=0.0, max=20.0)
    assert repr(t1) == "Time(initial=0.0, final=10.0, min=0.0, max=20.0)"

    t2 = Time()
    assert repr(t2) == "Time()"

    t3 = Time(min=0.0, max=20.0)
    assert repr(t3) == "Time(min=0.0, max=20.0)"


# --- Time: Guess ---


def test_time_guess_constructor_1d():
    """Test Time guess via constructor with 1D array (auto-reshaped)."""

    from openscvx import Time

    guess = np.linspace(0, 10, 50)
    t = Time(initial=0.0, final=10.0, min=0.0, max=20.0, guess=guess)
    assert t.guess.shape == (50, 1)
    assert np.allclose(t.guess.flatten(), guess)


def test_time_guess_constructor_2d():
    """Test Time guess via constructor with 2D array."""

    from openscvx import Time

    guess = np.linspace(0, 10, 50).reshape(-1, 1)
    t = Time(initial=0.0, final=10.0, min=0.0, max=20.0, guess=guess)
    assert t.guess.shape == (50, 1)
    assert np.allclose(t.guess, guess)


def test_time_guess_setter():
    """Test Time guess via setter (1D auto-reshaped)."""

    from openscvx import Time

    t = Time(initial=0.0, final=10.0, min=0.0, max=20.0)
    t.guess = np.linspace(0, 10, 50)
    assert t.guess.shape == (50, 1)


def test_time_guess_overrides_default():
    """Test that user-provided guess prevents _generate_default_guess."""

    from openscvx import Time

    custom_guess = np.array([0, 2, 5, 8, 10]).reshape(-1, 1)
    t = Time(initial=0.0, final=10.0, min=0.0, max=20.0, guess=custom_guess)
    # guess is already set, so _generate_default_guess should not be needed
    assert np.allclose(t.guess, custom_guess)


# --- Time: Time Dilation ---


def test_time_dilation_min_max_constructor():
    """Test time_dilation_min/max via constructor."""
    from openscvx import Time

    t = Time(
        initial=0.0,
        final=10.0,
        min=0.0,
        max=20.0,
        time_dilation_min=1.0,
        time_dilation_max=30.0,
    )
    assert t.time_dilation_min == 1.0
    assert t.time_dilation_max == 30.0


def test_time_dilation_min_max_setter():
    """Test time_dilation_min/max via setters."""
    from openscvx import Time

    t = Time(initial=0.0, final=10.0, min=0.0, max=20.0)
    assert t.time_dilation_min is None
    assert t.time_dilation_max is None

    t.time_dilation_min = 2.0
    t.time_dilation_max = 25.0
    assert t.time_dilation_min == 2.0
    assert t.time_dilation_max == 25.0


def test_time_dilation_guess_constructor_1d():
    """Test time_dilation_guess via constructor with 1D array."""

    from openscvx import Time

    td_guess = np.full(50, 10.0)
    t = Time(
        initial=0.0,
        final=10.0,
        min=0.0,
        max=20.0,
        time_dilation_guess=td_guess,
    )
    assert t.time_dilation_guess.shape == (50, 1)
    assert np.allclose(t.time_dilation_guess.flatten(), 10.0)


def test_time_dilation_guess_setter():
    """Test time_dilation_guess via setter."""

    from openscvx import Time

    t = Time(initial=0.0, final=10.0, min=0.0, max=20.0)
    assert t.time_dilation_guess is None

    t.time_dilation_guess = np.full((50, 1), 10.0)
    assert t.time_dilation_guess.shape == (50, 1)


def test_time_dilation_guess_bad_shape():
    """Test time_dilation_guess rejects bad shapes."""
    import pytest

    from openscvx import Time

    t = Time()
    with pytest.raises(ValueError, match="time_dilation_guess expected shape"):
        t.time_dilation_guess = np.ones((10, 2))


def test_time_dilation_constructor_setter_equivalent():
    """Test constructor and setter styles produce identical time_dilation results."""

    from openscvx import Time

    td_guess = np.linspace(8, 12, 50)

    t1 = Time(
        initial=0.0,
        final=10.0,
        min=0.0,
        max=20.0,
        time_dilation_min=1.0,
        time_dilation_max=30.0,
        time_dilation_guess=td_guess,
    )

    t2 = Time(initial=0.0, final=10.0, min=0.0, max=20.0)
    t2.time_dilation_min = 1.0
    t2.time_dilation_max = 30.0
    t2.time_dilation_guess = td_guess

    assert t1.time_dilation_min == t2.time_dilation_min
    assert t1.time_dilation_max == t2.time_dilation_max
    assert np.allclose(t1.time_dilation_guess, t2.time_dilation_guess)


# --- Time: Uniform Time Grid ---


def test_time_uniform_time_grid_default():
    """Test that uniform_time_grid defaults to False."""
    from openscvx import Time

    t = Time()
    assert t.uniform_time_grid is False


def test_time_uniform_time_grid_constructor():
    """Test setting uniform_time_grid via constructor."""
    from openscvx import Time

    t = Time(initial=0.0, final=10.0, min=0.0, max=20.0, uniform_time_grid=True)
    assert t.uniform_time_grid is True
