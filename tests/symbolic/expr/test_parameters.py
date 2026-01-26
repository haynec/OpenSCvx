"""Tests for Parameter nodes.

This module tests the Parameter node type and its behavior throughout the symbolic
expression system, including:

- Parameter creation and properties
- Parameter usage in arithmetic expressions
- Parameter usage in constraints
- Parameter shape checking
- Parameter canonicalization
- Parameter lowering to JAX
- Parameter lowering to CVXPY
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Add,
    Inequality,
    Mul,
    Parameter,
    Variable,
)

# =============================================================================
# Parameter Creation and Properties
# =============================================================================


def test_parameter_creation():
    """Test basic Parameter node creation."""
    p1 = Parameter("mass", value=1.0)
    assert p1.name == "mass"
    assert p1.shape == ()
    assert isinstance(p1, Parameter)

    p2 = Parameter("position", shape=(3,), value=np.array([0.0, 0.0, 0.0]))
    assert p2.name == "position"
    assert p2.shape == (3,)


def test_parameter_arithmetic_operations():
    """Test Parameter in arithmetic operations."""
    p = Parameter("param", value=1.0)
    x = Variable("x", shape=())

    add_expr = p + x
    assert isinstance(add_expr, Add)
    assert p in add_expr.children()
    assert x in add_expr.children()

    mul_expr = p * 2
    assert isinstance(mul_expr, Mul)
    assert p in mul_expr.children()


def test_parameter_in_constraints():
    """Test Parameter in constraint creation."""
    p = Parameter("threshold", value=1.0)
    x = Variable("x", shape=())

    ineq = x <= p
    assert isinstance(ineq, Inequality)
    assert ineq.lhs is x
    assert ineq.rhs is p


# --- Parameter: Shape Checking ---


def test_parameter_shape_scalar():
    """Test Parameter with scalar shape."""
    p = Parameter("alpha", shape=(), value=5.0)
    assert p.shape == ()
    assert p.value.shape == ()


def test_parameter_shape_vector():
    """Test Parameter with vector shape."""
    p = Parameter("weights", shape=(3,), value=np.array([1.0, 2.0, 3.0]))
    assert p.shape == (3,)
    assert p.value.shape == (3,)


def test_parameter_shape_matrix():
    """Test Parameter with matrix shape."""
    p = Parameter("matrix", shape=(2, 3), value=np.ones((2, 3)))
    assert p.shape == (2, 3)
    assert p.value.shape == (2, 3)


def test_parameter_requires_value():
    """Test that Parameter requires a value."""
    with pytest.raises(ValueError, match="requires an initial value"):
        Parameter("param", shape=())


# --- Parameter: Canonicalization ---


def test_parameter_canonicalization_unchanged():
    """Test that Parameter canonicalization returns self."""
    p = Parameter("alpha", value=5.0)
    canonical = p.canonicalize()
    assert canonical is p


def test_parameter_canonicalization_with_array():
    """Test that Parameter with array value canonicalizes to self."""
    p = Parameter("weights", shape=(3,), value=np.array([1.0, 2.0, 3.0]))
    canonical = p.canonicalize()
    assert canonical is p


# --- Parameter: JAX Lowering ---


def test_jax_lower_parameter_scalar():
    """Test Parameter node with scalar value."""
    import jax.numpy as jnp

    from openscvx.symbolic.lowerers.jax import JaxLowerer

    param = Parameter("alpha", (), value=5.0)
    jl = JaxLowerer()
    f = jl.lower(param)
    parameters = dict(alpha=5.0)

    # Test with scalar parameter
    out = f(None, None, None, parameters)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == ()
    assert jnp.allclose(out, 5.0)

    parameters["alpha"] = -2.5

    # Test with different scalar value
    out = f(None, None, None, parameters)
    assert jnp.allclose(out, -2.5)


def test_jax_lower_parameter_vector():
    """Test Parameter node with vector value."""
    import jax.numpy as jnp
    import numpy as np

    from openscvx.symbolic.lowerers.jax import JaxLowerer

    param = Parameter("weights", (3,), value=np.array([1.0, 2.0, 3.0]))
    jl = JaxLowerer()
    f = jl.lower(param)

    # Test with vector parameter
    weights_val = np.array([1.0, 2.0, 3.0])
    out = f(None, None, None, dict(weights=weights_val))
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (3,)
    assert jnp.allclose(out, weights_val)

    # Test with different vector value
    weights_val2 = np.array([-1.0, 0.5, 2.5])
    out = f(None, None, None, dict(weights=weights_val2))
    assert jnp.allclose(out, weights_val2)


def test_jax_lower_parameter_matrix():
    """Test Parameter node with matrix value."""
    import jax.numpy as jnp
    import numpy as np

    from openscvx.symbolic.lowerers.jax import JaxLowerer

    param = Parameter("transform", (2, 3), value=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    jl = JaxLowerer()
    f = jl.lower(param)

    # Test with matrix parameter
    matrix_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = f(None, None, None, dict(transform=matrix_val))
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2, 3)
    assert jnp.allclose(out, matrix_val)


def test_parameter_in_arithmetic_expression():
    """Test Parameter nodes in arithmetic expressions with states."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Mul, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([1.0, 2.0, 3.0])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    gain = Parameter("gain", (), value=2.5)

    # Expression: gain * x
    expr = Mul(gain, state)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, dict(gain=2.5))

    expected = 2.5 * x
    assert jnp.allclose(result, expected)


def test_parameter_with_lower_to_jax():
    """Test Parameter nodes with the top-level lower_to_jax function."""
    import jax.numpy as jnp
    import numpy as np

    from openscvx.symbolic.lower import lower_to_jax

    param = Parameter("threshold", (2,), value=np.array([1.5, 2.5]))

    fn = lower_to_jax(param)
    param_val = np.array([1.5, 2.5])
    result = fn(None, None, None, dict(threshold=param_val))

    assert isinstance(result, jnp.ndarray)
    assert result.shape == (2,)
    assert jnp.allclose(result, param_val)


def test_parameter_in_double_integrator_dynamics():
    """Test Parameter nodes in a realistic double integrator dynamics expression."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Concat, Constant, Control, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([1.0, 2.0, 0.5, -0.2])  # [pos_x, pos_y, vel_x, vel_y]
    u = jnp.array([0.8, 1.2])  # [acc_x, acc_y]

    # State and control
    state = State("x", (4,))
    state._slice = slice(0, 4)
    control = Control("u", (2,))
    control._slice = slice(0, 2)

    # Parameters for physical system
    mass = Parameter("m", (), value=2.0)
    gravity = Parameter("g", (), value=9.81)

    # Extract state components
    # pos = state[0:2]  # [pos_x, pos_y]
    vel = state[2:4]  # [vel_x, vel_y]

    # Dynamics: pos_dot = vel, vel_dot = u/m + [0, -g]
    pos_dot = vel
    gravity_vec = Concat(Constant(0.0), -gravity)
    vel_dot = control / mass + gravity_vec

    dynamics = Concat(pos_dot, vel_dot)

    fn = lower_to_jax(dynamics)

    # Test with realistic parameter values
    m_val = 2.0  # kg
    g_val = 9.81  # m/s^2
    parameter = dict(m=m_val, g=g_val)
    result = fn(x, u, None, parameter)

    # Expected: [vel_x, vel_y, acc_x/m, acc_y/m - g]
    expected = jnp.array(
        [
            0.5,  # vel_x
            -0.2,  # vel_y
            0.8 / 2.0,  # acc_x / m = 0.4
            1.2 / 2.0 - 9.81,  # acc_y / m - g = 0.6 - 9.81 = -9.21
        ]
    )

    assert jnp.allclose(result, expected)
    assert result.shape == (4,)


def test_parameter_dynamics_with_jit_and_vmap():
    """Test Parameter nodes in dynamics function with JAX JIT compilation and vmap."""
    import jax
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Concat, Constant, Control, State
    from openscvx.symbolic.lower import lower_to_jax

    # Create double integrator dynamics with parameters
    state = State("x", (4,))
    state._slice = slice(0, 4)
    control = Control("u", (2,))
    control._slice = slice(0, 2)

    mass = Parameter("m", (), value=2.0)
    gravity = Parameter("g", (), value=9.81)

    # Dynamics: pos_dot = vel, vel_dot = u/m + [0, -g]
    pos_dot = state[2:4]  # velocity
    gravity_vec = Concat(Constant(0.0), -gravity)
    vel_dot = control / mass + gravity_vec
    dynamics = Concat(pos_dot, vel_dot)

    # Lower to JAX function
    dynamics_fn = lower_to_jax(dynamics)

    # Create function compatible with problem.py calling convention
    def dynamics_with_node(x, u, node, m, g):
        """Dynamics function with node parameter (similar to problem.py structure)."""
        parameter = dict(m=m, g=g)
        return dynamics_fn(x, u, node, parameter)

    # JIT compile the function
    dynamics_jit = jax.jit(dynamics_with_node)

    # Test single evaluation
    x = jnp.array([1.0, 2.0, 0.5, -0.2])
    u = jnp.array([0.8, 1.2])
    node = 0
    m_val = 2.0
    g_val = 9.81

    result_single = dynamics_jit(x, u, node, m_val, g_val)
    expected = jnp.array([0.5, -0.2, 0.4, -9.21])
    assert jnp.allclose(result_single, expected)

    # Test with vmap for multiple time steps (similar to problem.py)
    N = 5
    x_batch = jnp.tile(x[None, :], (N, 1))  # (N, 4)
    u_batch = jnp.tile(u[None, :], (N, 1))  # (N, 2)
    node_batch = jnp.arange(N)  # (N,)

    # Create vmapped function with parameters as None (not vectorized)
    dynamics_vmap = jax.vmap(
        dynamics_with_node,
        in_axes=(0, 0, 0, None, None),  # vmap over x, u, node; keep m, g scalar
    )

    # JIT compile the vmapped function
    dynamics_vmap_jit = jax.jit(dynamics_vmap)

    # Test batch evaluation
    result_batch = dynamics_vmap_jit(x_batch, u_batch, node_batch, m_val, g_val)
    expected_batch = jnp.tile(expected[None, :], (N, 1))  # (N, 4)

    assert result_batch.shape == (N, 4)
    assert jnp.allclose(result_batch, expected_batch)

    # Test parameter updates work correctly after compilation
    m_val_new = 3.0
    result_new_mass = dynamics_vmap_jit(x_batch, u_batch, node_batch, m_val_new, g_val)

    # Expected with new mass: [0.5, -0.2, 0.8/3.0, 1.2/3.0 - 9.81]
    expected_new = jnp.array([0.5, -0.2, 0.8 / 3.0, 1.2 / 3.0 - 9.81])
    expected_new_batch = jnp.tile(expected_new[None, :], (N, 1))

    assert jnp.allclose(result_new_mass, expected_new_batch)


# --- Parameter: CVXPy Lowering ---


def test_cvxpy_parameter_scalar():
    """Test Parameter node with scalar value."""
    import cvxpy as cp

    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    param_value = cp.Parameter(name="alpha", value=5.0)
    variable_map = {"alpha": param_value}
    lowerer = CvxpyLowerer(variable_map)

    # Create symbolic parameter
    param = Parameter("alpha", (), value=5.0)
    result = lowerer.lower(param)

    # Should return the CVXPy parameter
    assert result is param_value
    assert isinstance(result, cp.Parameter)


def test_cvxpy_parameter_vector():
    """Test Parameter node with vector value."""
    import cvxpy as cp

    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    param_value = cp.Parameter((3,), name="weights", value=[1.0, 2.0, 3.0])
    variable_map = {"weights": param_value}
    lowerer = CvxpyLowerer(variable_map)

    # Create symbolic parameter
    param = Parameter("weights", (3,), value=np.array([1.0, 2.0, 3.0]))
    result = lowerer.lower(param)

    # Should return the CVXPy parameter
    assert result is param_value
    assert isinstance(result, cp.Parameter)


def test_cvxpy_parameter_matrix():
    """Test Parameter node with matrix value."""
    import cvxpy as cp

    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    param_value = cp.Parameter((2, 3), name="transform")
    param_value.value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    variable_map = {"transform": param_value}
    lowerer = CvxpyLowerer(variable_map)

    # Create symbolic parameter
    param = Parameter("transform", (2, 3), value=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    result = lowerer.lower(param)

    # Should return the CVXPy parameter
    assert result is param_value
    assert isinstance(result, cp.Parameter)


def test_cvxpy_parameter_missing_from_map_raises():
    """Test that missing parameter from variable_map raises ValueError."""

    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    lowerer = CvxpyLowerer({})

    # Create symbolic parameter
    param = Parameter("missing_param", (), value=1.0)

    # Should raise ValueError when parameter is missing
    with pytest.raises(ValueError, match="Parameter 'missing_param' not found"):
        lowerer.lower(param)


def test_cvxpy_parameter_in_arithmetic_expression():
    """Test Parameter nodes in arithmetic expressions with states."""
    import cvxpy as cp

    from openscvx.symbolic.expr import Mul, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    # CVXPy variables
    x_var = cp.Variable(3, name="x")
    gain_param = cp.Parameter(name="gain", value=2.5)
    variable_map = {"x": x_var, "gain": gain_param}
    lowerer = CvxpyLowerer(variable_map)

    # Symbolic expressions
    state = State("x", (3,))
    gain = Parameter("gain", (), value=2.5)

    # Expression: gain * x
    expr = Mul(gain, state)
    result = lowerer.lower(expr)

    # Should be a CVXPy expression
    assert isinstance(result, cp.Expression)


def test_cvxpy_parameter_with_lower_to_cvxpy():
    """Test Parameter nodes with the top-level lower_to_cvxpy function."""
    import cvxpy as cp

    from openscvx.symbolic.lowerers.cvxpy import lower_to_cvxpy

    threshold_param = cp.Parameter((2,), name="threshold", value=[1.5, 2.5])
    variable_map = {"threshold": threshold_param}

    # Create symbolic parameter
    param = Parameter("threshold", (2,), value=np.array([1.5, 2.5]))

    result = lower_to_cvxpy(param, variable_map)
    assert result is threshold_param
    assert isinstance(result, cp.Parameter)


def test_cvxpy_parameter_in_constraint():
    """Test Parameter nodes in constraint expressions."""
    import cvxpy as cp

    from openscvx.symbolic.expr import Inequality, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    # CVXPy variables and parameters
    x_var = cp.Variable(3, name="x")
    limit_param = cp.Parameter((3,), name="limit", value=[1.0, 2.0, 3.0])
    variable_map = {"x": x_var, "limit": limit_param}
    lowerer = CvxpyLowerer(variable_map)

    # Symbolic expressions
    state = State("x", (3,))
    limit = Parameter("limit", (3,), value=np.array([1.0, 2.0, 3.0]))

    # Constraint: x <= limit
    constraint = Inequality(state, limit)
    result = lowerer.lower(constraint)

    # Should be a CVXPy constraint
    assert isinstance(result, cp.Constraint)


def test_cvxpy_parameter_in_complex_expression():
    """Test Parameter nodes in complex expressions similar to dynamics."""
    import cvxpy as cp

    from openscvx.symbolic.expr import Add, Concat, Constant, Control, Div, Index, Neg, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    # CVXPy variables and parameters
    x_var = cp.Variable(4, name="x")  # [pos_x, pos_y, vel_x, vel_y]
    u_var = cp.Variable(2, name="u")  # [acc_x, acc_y]
    mass_param = cp.Parameter(name="mass", value=2.0)
    gravity_param = cp.Parameter(name="gravity", value=9.81)
    variable_map = {"x": x_var, "u": u_var, "mass": mass_param, "gravity": gravity_param}
    lowerer = CvxpyLowerer(variable_map)

    # Symbolic expressions - double integrator dynamics
    state = State("x", (4,))
    state._slice = slice(0, 4)
    control = Control("u", (2,))
    control._slice = slice(0, 2)
    mass = Parameter("mass", (), value=2.0)
    gravity = Parameter("gravity", (), value=9.81)

    # Extract state components: pos = x[0:2], vel = x[2:4]
    # pos = Index(state, slice(0, 2))
    vel = Index(state, slice(2, 4))

    # Dynamics: pos_dot = vel, vel_dot = u/mass + [0, -gravity]
    pos_dot = vel
    gravity_vec = Concat(Constant(0.0), Neg(gravity))
    vel_dot = Add(Div(control, mass), gravity_vec)

    dynamics = Concat(pos_dot, vel_dot)
    result = lowerer.lower(dynamics)

    # Should be a CVXPy expression
    assert isinstance(result, cp.Expression)
