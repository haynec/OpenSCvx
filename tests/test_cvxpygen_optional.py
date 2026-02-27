"""
Test that cvxpygen is properly handled as an optional dependency.
"""

import numpy as np
import pytest

from openscvx import Time, ctcs
from openscvx.problem import Problem
from openscvx.symbolic.expr import Concat, Constant, Control, State

# Conditionally import cvxpygen to see if it's installed
try:
    import cvxpygen  # noqa: F401

    CVXPYGEN_INSTALLED = True
except ImportError:
    CVXPYGEN_INSTALLED = False


def test_cvxpygen_optional_import():
    """Test that cvxpygen import is optional."""
    # This should not raise an ImportError
    try:
        from openscvx.algorithms import PenalizedTrustRegion  # noqa: F401
        from openscvx.solvers import PTRSolver  # noqa: F401

        assert True
    except ImportError as e:
        if "cvxpygen" in str(e):
            pytest.fail("cvxpygen import should be optional")
        else:
            raise


def test_cvxpygen_disabled_by_default():
    """Test that cvxpygen is disabled by default."""
    # Create a simple problem
    n = 5
    x = State("x", shape=(2,))
    u = Control("u", shape=(1,))

    x.min = np.array([-5.0, -5.0])
    x.max = np.array([5.0, 5.0])
    x.initial = np.array([0, -1])
    x.final = np.array([0, 1])
    x.guess = np.linspace([0, -1], [0, 1], n)

    u.min = np.array([-2])
    u.max = np.array([2])
    u.guess = np.zeros((n, 1))

    # Define dynamics as dictionary - x_dot = [u[0], 0] for 2D state
    dynamics = {"x": Concat(u[0], Constant(0.0))}

    # Define constraints using symbolic expressions
    constraints = [ctcs(x <= x.max), ctcs(x.min <= x)]

    time = Time(initial=0.0, final=("minimize", 1.0), min=0.0, max=10.0)

    problem = Problem(
        dynamics=dynamics,
        states=[x],
        controls=[u],
        time=time,
        constraints=constraints,
        N=n,
    )

    # Check that cvxpygen is disabled by default
    assert not problem.solver.cvxpygen


def test_cvxpygen_enabled_raises_error_without_install():
    """Test that enabling cvxpygen without installation raises appropriate error."""

    # This test should only run if cvxpygen is NOT installed
    if CVXPYGEN_INSTALLED:
        pytest.skip("cvxpygen is installed, skipping test for error raising")

    # Create a simple problem
    n = 5
    x = State("x", shape=(2,))
    u = Control("u", shape=(1,))

    x.min = np.array([-5.0, -5.0])
    x.max = np.array([5.0, 5.0])
    x.initial = np.array([0, -1])
    x.final = np.array([0, 1])
    x.guess = np.linspace([0, -1], [0, 1], n)

    u.min = np.array([-2])
    u.max = np.array([2])
    u.guess = np.zeros((n, 1))

    # Define dynamics as dictionary - x_dot = [u[0], 0] for 2D state
    dynamics = {"x": Concat(u[0], Constant(0.0))}

    # Define constraints using symbolic expressions
    constraints = [ctcs(x <= x.max), ctcs(x.min <= x)]

    time = Time(initial=0.0, final=("minimize", 1.0), min=0.0, max=10.0)

    problem = Problem(
        dynamics=dynamics,
        states=[x],
        controls=[u],
        time=time,
        constraints=constraints,
        N=n,
    )

    # Enable cvxpygen
    problem.solver.cvxpygen = True
    problem.solver.cvxpygen_override = True

    # This should raise an ImportError with a helpful message
    with pytest.raises(ImportError) as exc_info:
        problem.initialize()

    assert "cvxpygen" in str(exc_info.value)
    assert "pip install openscvx[cvxpygen]" in str(exc_info.value)
