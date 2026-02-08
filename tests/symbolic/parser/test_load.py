"""Tests for the YAML / JSON problem loader.

This module tests the load_dict function that converts a parsed
dictionary into Problem constructor keyword arguments.
"""

import numpy as np

from openscvx.symbolic.expr import (
    Equality,
    Inequality,
    Norm,
    State,
)
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.parser.load import load_dict

# =============================================================================
# Minimal Valid Problem
# =============================================================================


def _minimal_data():
    """Return the smallest valid problem dict."""
    return {
        "N": 20,
        "time": {"initial": 0.0, "final": 10.0, "min": 0.0, "max": 20.0},
        "states": [
            {
                "name": "pos",
                "shape": [3],
                "min": [-10, -10, 0],
                "max": [10, 10, 100],
                "initial": [0, 0, 50],
                "final": [10, 5, 0],
            }
        ],
        "controls": [
            {
                "name": "thrust",
                "shape": [3],
                "min": [-10, -10, 0],
                "max": [10, 10, 50],
            }
        ],
        "dynamics": {"pos": "thrust"},
        "constraints": [],
    }


# =============================================================================
# Basic Structure
# =============================================================================


def test_load_dict_returns_expected_keys():
    result = load_dict(_minimal_data())
    assert "dynamics" in result
    assert "constraints" in result
    assert "states" in result
    assert "controls" in result
    assert "N" in result
    assert "time" in result


def test_load_dict_N():
    result = load_dict(_minimal_data())
    assert result["N"] == 20


# =============================================================================
# States
# =============================================================================


def test_states_are_state_objects():
    result = load_dict(_minimal_data())
    assert len(result["states"]) == 1
    s = result["states"][0]
    assert isinstance(s, State)
    assert s.name == "pos"
    assert s.shape == (3,)


def test_state_bounds():
    result = load_dict(_minimal_data())
    s = result["states"][0]
    assert np.array_equal(s.min, np.array([-10, -10, 0], dtype=float))
    assert np.array_equal(s.max, np.array([10, 10, 100], dtype=float))


def test_state_initial_fixed():
    result = load_dict(_minimal_data())
    s = result["states"][0]
    assert np.array_equal(s.initial, np.array([0, 0, 50], dtype=float))


def test_state_with_free_boundary():
    data = _minimal_data()
    data["states"][0]["final"] = [10, ["free", 5], 0]
    result = load_dict(data)
    s = result["states"][0]
    # The State setter handles tuples for free boundaries
    # Just check no error is raised
    assert s is not None


def test_state_guess():
    data = _minimal_data()
    data["states"][0]["guess"] = [[1, 2, 3]]
    result = load_dict(data)
    s = result["states"][0]
    assert s.guess is not None


def test_state_scaling():
    data = _minimal_data()
    data["states"][0]["scaling_min"] = [-1, -1, -1]
    data["states"][0]["scaling_max"] = [1, 1, 1]
    result = load_dict(data)
    s = result["states"][0]
    assert np.array_equal(s.scaling_min, np.array([-1, -1, -1], dtype=float))
    assert np.array_equal(s.scaling_max, np.array([1, 1, 1], dtype=float))


# =============================================================================
# Controls
# =============================================================================


def test_controls_are_control_objects():
    result = load_dict(_minimal_data())
    assert len(result["controls"]) == 1
    c = result["controls"][0]
    assert isinstance(c, Control)
    assert c.name == "thrust"
    assert c.shape == (3,)


def test_control_bounds():
    result = load_dict(_minimal_data())
    c = result["controls"][0]
    assert np.array_equal(c.min, np.array([-10, -10, 0], dtype=float))
    assert np.array_equal(c.max, np.array([10, 10, 50], dtype=float))


# =============================================================================
# Time
# =============================================================================


def test_time_object():
    result = load_dict(_minimal_data())
    time = result["time"]
    assert time.initial == 0.0
    assert time.final == 10.0
    assert time.min == 0.0
    assert time.max == 20.0


def test_time_with_minimize():
    data = _minimal_data()
    data["time"]["final"] = ["minimize", 10.0]
    result = load_dict(data)
    time = result["time"]
    # Time.final is set via State setter which stores a numpy array;
    # check the initial value was passed as a tuple to the constructor
    assert float(time.final[0]) == 10.0


# =============================================================================
# Parameters
# =============================================================================


def test_parameters_in_symbol_table():
    data = _minimal_data()
    data["parameters"] = [{"name": "gravity", "shape": [3], "value": [0, 0, -9.81]}]
    data["dynamics"]["pos"] = "thrust + gravity"
    result = load_dict(data)
    # gravity should be resolved in the dynamics expression
    assert "pos" in result["dynamics"]


# =============================================================================
# Dynamics
# =============================================================================


def test_dynamics_are_expr_objects():
    result = load_dict(_minimal_data())
    assert "pos" in result["dynamics"]
    # "thrust" resolves to the Control symbol
    expr = result["dynamics"]["pos"]
    assert isinstance(expr, Control)
    assert expr.name == "thrust"


def test_dynamics_with_arithmetic():
    data = _minimal_data()
    data["parameters"] = [{"name": "g", "shape": [3], "value": [0, 0, -9.81]}]
    data["dynamics"]["pos"] = "thrust + g"
    result = load_dict(data)
    from openscvx.symbolic.expr import Add

    assert isinstance(result["dynamics"]["pos"], Add)


# =============================================================================
# Constraints
# =============================================================================


def test_empty_constraints():
    result = load_dict(_minimal_data())
    assert result["constraints"] == []


def test_inequality_constraint():
    data = _minimal_data()
    data["constraints"] = ["pos[0] <= 5.0"]
    result = load_dict(data)
    assert len(result["constraints"]) == 1
    assert isinstance(result["constraints"][0], Inequality)


def test_equality_constraint():
    data = _minimal_data()
    data["constraints"] = ["pos[0] == 0.0"]
    result = load_dict(data)
    assert isinstance(result["constraints"][0], Equality)


def test_norm_constraint():
    data = _minimal_data()
    data["parameters"] = [{"name": "obs", "shape": [3], "value": [1, 2, 3]}]
    data["constraints"] = ["Norm(pos - obs) >= 2.0"]
    result = load_dict(data)
    c = result["constraints"][0]
    assert isinstance(c, Inequality)


# =============================================================================
# Optional: Propagation
# =============================================================================


def test_states_prop():
    data = _minimal_data()
    data["states_prop"] = [{"name": "distance", "shape": [1]}]
    data["dynamics_prop"] = {"distance": "Norm(thrust)"}
    result = load_dict(data)
    assert "states_prop" in result
    assert len(result["states_prop"]) == 1
    assert result["states_prop"][0].name == "distance"
    assert "dynamics_prop" in result


def test_algebraic_prop():
    data = _minimal_data()
    data["algebraic_prop"] = {"speed": "Norm(thrust)"}
    result = load_dict(data)
    assert "algebraic_prop" in result
    assert isinstance(result["algebraic_prop"]["speed"], Norm)


# =============================================================================
# Multiple States & Controls
# =============================================================================


def test_multiple_states():
    data = _minimal_data()
    data["states"].append(
        {
            "name": "vel",
            "shape": [3],
            "min": [-5, -5, -5],
            "max": [5, 5, 5],
            "initial": [0, 0, 0],
            "final": [0, 0, 0],
        }
    )
    data["dynamics"]["vel"] = "thrust"
    result = load_dict(data)
    assert len(result["states"]) == 2
    names = [s.name for s in result["states"]]
    assert "pos" in names
    assert "vel" in names
