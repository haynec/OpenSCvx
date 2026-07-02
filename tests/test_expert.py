"""
Unit tests for expert-mode validation logic.

Tests the validation of bring-your-own-functions (byof) to ensure proper
error handling and clear error messages for invalid user inputs.
"""

import numpy as np
import pytest


@pytest.fixture
def simple_states():
    """Create a simple state setup for validation testing."""
    import openscvx as ox

    position = ox.State("position", shape=(2,))
    velocity = ox.State("velocity", shape=(1,))
    return [position, velocity]


# ===== Valid Cases =====


def test_valid_byof_specifications(simple_states):
    """Test that all valid byof specifications are accepted."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    # Empty byof
    validate_byof({}, simple_states, n_x=3, n_u=1, N=50)

    # All valid keys
    validate_byof(
        {
            "dynamics": {},
            "nodal_constraints": [],
            "cross_nodal_constraints": [],
            "ctcs_constraints": [],
        },
        simple_states,
        n_x=3,
        n_u=1,
        N=50,
    )

    # Valid dynamics
    validate_byof(
        {"dynamics": {"velocity": lambda x, u, node, params: jnp.array([1.0])}},
        simple_states,
        n_x=3,
        n_u=1,
        N=50,
    )

    # Valid nodal constraint (scalar, all nodes)
    validate_byof(
        {"nodal_constraints": [{"constraint_fn": lambda x, u, node, params: x[0] - 10.0}]},
        simple_states,
        n_x=3,
        n_u=1,
        N=50,
    )

    # Valid nodal constraint (vector, all nodes)
    validate_byof(
        {
            "nodal_constraints": [
                {"constraint_fn": lambda x, u, node, params: jnp.array([x[0] - 10.0, x[1] - 5.0])}
            ]
        },
        simple_states,
        n_x=3,
        n_u=1,
        N=50,
    )

    # Valid nodal constraint with specific nodes
    validate_byof(
        {
            "nodal_constraints": [
                {"constraint_fn": lambda x, u, node, params: x[0] - 10.0, "nodes": [0, 5, 10]}
            ]
        },
        simple_states,
        n_x=3,
        n_u=1,
        N=50,
    )

    # Valid nodal constraint with negative node indices
    validate_byof(
        {
            "nodal_constraints": [
                {"constraint_fn": lambda x, u, node, params: x[0], "nodes": [0, -1]}
            ]
        },
        simple_states,
        n_x=3,
        n_u=1,
        N=50,
    )

    # Valid cross-nodal constraint
    validate_byof(
        {"cross_nodal_constraints": [lambda X, U, params: jnp.sum(X[:, 0])]},
        simple_states,
        n_x=3,
        n_u=1,
        N=50,
    )

    # Valid CTCS constraint with different penalties
    for penalty in ["square", "l1", "huber"]:
        validate_byof(
            {
                "ctcs_constraints": [
                    {"constraint_fn": lambda x, u, node, params: x[0] - 10.0, "penalty": penalty}
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )

    # Valid CTCS with custom penalty
    validate_byof(
        {
            "ctcs_constraints": [
                {
                    "constraint_fn": lambda x, u, node, params: x[0] - 10.0,
                    "penalty": lambda r: jnp.maximum(r, 0.0) ** 3,
                }
            ]
        },
        simple_states,
        n_x=3,
        n_u=1,
        N=50,
    )

    # Valid CTCS with bounds and over interval
    validate_byof(
        {
            "ctcs_constraints": [
                {
                    "constraint_fn": lambda x, u, node, params: x[0] - 10.0,
                    "bounds": (0.0, 1e-4),
                    "over": (10, 40),
                }
            ]
        },
        simple_states,
        n_x=3,
        n_u=1,
        N=50,
    )


# ===== Invalid Keys =====


def test_invalid_byof_keys(simple_states):
    """Unknown byof keys should raise ValueError (via pydantic extra='forbid')."""
    from openscvx.expert import validate_byof

    with pytest.raises(ValueError, match="invalid_key"):
        validate_byof({"invalid_key": []}, simple_states, n_x=3, n_u=1, N=50)

    with pytest.raises(ValueError, match="bad_key1"):
        validate_byof({"bad_key1": [], "bad_key2": {}}, simple_states, n_x=3, n_u=1, N=50)


# ===== Dynamics Validation =====


@pytest.mark.parametrize(
    "bad_dynamics,error_type,error_match",
    [
        # Wrong state name
        (
            {"nonexistent_state": lambda x, u, node, params: np.array([1.0])},
            ValueError,
            "does not match any state name",
        ),
        # Not callable (pydantic catches this during ByofSpec validation)
        ({"velocity": "not a function"}, ValueError, "callable"),
        # Wrong signature - too few params
        ({"velocity": lambda x, u: np.array([1.0])}, ValueError, "must have signature"),
        # Wrong signature - too many params
        (
            {"velocity": lambda x, u, node, params, extra: np.array([1.0])},
            ValueError,
            "must have signature",
        ),
    ],
)
def test_dynamics_validation_errors(simple_states, bad_dynamics, error_type, error_match):
    """Test various dynamics validation errors."""
    from openscvx.expert import validate_byof

    with pytest.raises(error_type, match=error_match):
        validate_byof({"dynamics": bad_dynamics}, simple_states, n_x=3, n_u=1, N=50)


def test_dynamics_runtime_errors(simple_states):
    """Test dynamics that fail at runtime or return wrong shapes."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    # Fails on call
    with pytest.raises(ValueError, match="failed on test call"):
        validate_byof(
            {
                "dynamics": {
                    "velocity": lambda x, u, node, params: (_ for _ in ()).throw(
                        RuntimeError("fail")
                    )
                }
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )

    # Wrong output shape
    with pytest.raises(ValueError, match="returned shape.*expected"):
        validate_byof(
            {"dynamics": {"velocity": lambda x, u, node, params: jnp.array([1.0, 2.0])}},
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )

    # Not differentiable (using numpy.linalg)
    with pytest.raises(ValueError, match="not differentiable with JAX"):
        validate_byof(
            {
                "dynamics": {
                    "velocity": lambda x, u, node, params: np.array([float(np.linalg.norm(x))])
                }
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )


# ===== Nodal Constraint Validation =====


@pytest.mark.parametrize(
    "bad_spec,error_type,error_match",
    [
        # Not a dict (pydantic catches during ByofSpec validation)
        ("not a dict", ValueError, "nodal_constraints"),
        # Missing 'constraint_fn' key (pydantic catches as required field)
        ({"nodes": [0, 1]}, ValueError, "constraint_fn"),
        # constraint_fn not callable (pydantic catches)
        ({"constraint_fn": "not a function"}, ValueError, "callable"),
        # Wrong signature
        ({"constraint_fn": lambda x, u: x[0]}, ValueError, "must have signature"),
        # nodes not a list (pydantic catches type mismatch)
        (
            {"constraint_fn": lambda x, u, node, params: x[0], "nodes": "invalid"},
            ValueError,
            "nodes",
        ),
        # Empty nodes list
        (
            {"constraint_fn": lambda x, u, node, params: x[0], "nodes": []},
            ValueError,
            "cannot be empty",
        ),
    ],
)
def test_nodal_constraint_validation_errors(simple_states, bad_spec, error_type, error_match):
    """Test various nodal constraint validation errors."""
    from openscvx.expert import validate_byof

    with pytest.raises(error_type, match=error_match):
        validate_byof({"nodal_constraints": [bad_spec]}, simple_states, n_x=3, n_u=1, N=50)


def test_nodal_constraint_runtime_errors(simple_states):
    """Test nodal constraints that fail at runtime."""
    from openscvx.expert import validate_byof

    # Fails on call
    with pytest.raises(ValueError, match="failed on test call"):
        validate_byof(
            {
                "nodal_constraints": [
                    {
                        "constraint_fn": lambda x, u, node, params: (_ for _ in ()).throw(
                            RuntimeError("fail")
                        )
                    }
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )

    # Not differentiable
    with pytest.raises(ValueError, match="not differentiable with JAX"):
        validate_byof(
            {
                "nodal_constraints": [
                    {"constraint_fn": lambda x, u, node, params: np.array([x[0] - 10.0])}
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )


def test_nodal_constraint_node_index_validation(simple_states):
    """Test that node indices are validated when N is provided."""
    from openscvx.expert import validate_byof

    # Out of range positive index
    with pytest.raises(ValueError, match="invalid index 50.*Valid range is \\[0, 50\\)"):
        validate_byof(
            {
                "nodal_constraints": [
                    {"constraint_fn": lambda x, u, node, params: x[0], "nodes": [0, 50]}
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )

    # Out of range negative index
    with pytest.raises(ValueError, match="invalid index -51.*Valid range"):
        validate_byof(
            {
                "nodal_constraints": [
                    {"constraint_fn": lambda x, u, node, params: x[0], "nodes": [0, -51]}
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )

    # Valid negative index should work
    validate_byof(
        {
            "nodal_constraints": [
                {"constraint_fn": lambda x, u, node, params: x[0], "nodes": [0, -1, -50]}
            ]
        },
        simple_states,
        n_x=3,
        n_u=1,
        N=50,
    )


# ===== Cross-Nodal Constraint Validation =====


@pytest.mark.parametrize(
    "bad_constraint,error_type,error_match",
    [
        # Not callable (pydantic catches)
        ("not a function", ValueError, "callable"),
        # Wrong signature
        (lambda X, U: np.sum(X[:, 0]), ValueError, "must have signature f\\(X, U, params\\)"),
    ],
)
def test_cross_nodal_constraint_validation_errors(
    simple_states, bad_constraint, error_type, error_match
):
    """Test various cross-nodal constraint validation errors."""
    from openscvx.expert import validate_byof

    with pytest.raises(error_type, match=error_match):
        validate_byof(
            {"cross_nodal_constraints": [bad_constraint]}, simple_states, n_x=3, n_u=1, N=50
        )


def test_cross_nodal_constraint_runtime_errors(simple_states):
    """Test cross-nodal constraints that fail at runtime."""
    from openscvx.expert import validate_byof

    # Fails on call
    with pytest.raises(ValueError, match="failed on test call"):
        validate_byof(
            {
                "cross_nodal_constraints": [
                    lambda X, U, params: (_ for _ in ()).throw(RuntimeError("fail"))
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )

    # Not differentiable
    with pytest.raises(ValueError, match="not differentiable with JAX"):
        validate_byof(
            {"cross_nodal_constraints": [lambda X, U, params: float(np.linalg.norm(X[:, 0]))]},
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )


# ===== CTCS Constraint Validation =====


@pytest.mark.parametrize(
    "bad_spec,error_type,error_match",
    [
        # Not a dict (pydantic catches during ByofSpec validation)
        ("not a dict", ValueError, "ctcs_constraints"),
        # Missing constraint_fn (pydantic catches as required field)
        ({"penalty": "square"}, ValueError, "constraint_fn"),
        # constraint_fn not callable (pydantic catches)
        ({"constraint_fn": "not a function"}, ValueError, "callable"),
        # Wrong signature
        ({"constraint_fn": lambda x, u: x[0]}, ValueError, "must have signature"),
        # Invalid penalty string (pydantic catches via Literal type)
        (
            {"constraint_fn": lambda x, u, node, params: x[0], "penalty": "invalid"},
            ValueError,
            "penalty",
        ),
    ],
)
def test_ctcs_constraint_validation_errors(simple_states, bad_spec, error_type, error_match):
    """Test various CTCS constraint validation errors."""
    from openscvx.expert import validate_byof

    with pytest.raises(error_type, match=error_match):
        validate_byof({"ctcs_constraints": [bad_spec]}, simple_states, n_x=3, n_u=1, N=50)


def test_ctcs_constraint_runtime_errors(simple_states):
    """Test CTCS constraints that fail at runtime."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    # constraint_fn fails on call
    with pytest.raises(ValueError, match="failed on test call"):
        validate_byof(
            {
                "ctcs_constraints": [
                    {
                        "constraint_fn": lambda x, u, node, params: (_ for _ in ()).throw(
                            RuntimeError("fail")
                        )
                    }
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )

    # constraint_fn not scalar
    with pytest.raises(ValueError, match="must return a scalar"):
        validate_byof(
            {
                "ctcs_constraints": [
                    {"constraint_fn": lambda x, u, node, params: jnp.array([x[0], x[1]])}
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )

    # constraint_fn not differentiable
    with pytest.raises(ValueError, match="not differentiable with JAX"):
        validate_byof(
            {"ctcs_constraints": [{"constraint_fn": lambda x, u, node, params: float(np.sum(x))}]},
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )

    # Custom penalty fails
    with pytest.raises(ValueError, match="penalty.*custom function failed"):
        validate_byof(
            {
                "ctcs_constraints": [
                    {
                        "constraint_fn": lambda x, u, node, params: x[0],
                        "penalty": lambda r: (_ for _ in ()).throw(RuntimeError("fail")),
                    }
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )


@pytest.mark.parametrize(
    "bad_bounds,error_match",
    [
        # Not tuple/list (pydantic catches type mismatch)
        ("not a tuple", "bounds"),
        # Wrong length (pydantic catches tuple length mismatch)
        ((0.0, 1e-4, 1.0), "bounds"),
        # Min > max
        ((1.0, 0.0), "min.*must be <= max"),
    ],
)
def test_ctcs_bounds_validation_errors(simple_states, bad_bounds, error_match):
    """Test CTCS bounds validation errors."""
    from openscvx.expert import validate_byof

    with pytest.raises(ValueError, match=error_match):
        validate_byof(
            {
                "ctcs_constraints": [
                    {
                        "constraint_fn": lambda x, u, node, params: x[0],
                        "bounds": bad_bounds,
                    }
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )


def test_ctcs_initial_must_be_within_bounds(simple_states):
    """Test CTCS initial value must be within bounds."""
    from openscvx.expert import validate_byof

    # Initial below bounds
    with pytest.raises(ValueError, match="initial.*must be within bounds"):
        validate_byof(
            {
                "ctcs_constraints": [
                    {
                        "constraint_fn": lambda x, u, node, params: x[0],
                        "bounds": (0.0, 1.0),
                        "initial": -0.5,
                    }
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )

    # Initial above bounds
    with pytest.raises(ValueError, match="initial.*must be within bounds"):
        validate_byof(
            {
                "ctcs_constraints": [
                    {
                        "constraint_fn": lambda x, u, node, params: x[0],
                        "bounds": (0.0, 1.0),
                        "initial": 1.5,
                    }
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )


# ===== Error Message Indexing =====


def test_error_messages_index_correctly(simple_states):
    """Error messages should have correct indices for multiple constraints."""

    from openscvx.expert import validate_byof

    # Multiple nodal constraints - error in second one
    with pytest.raises(ValueError, match="nodal_constraints\\[1\\]"):
        validate_byof(
            {
                "nodal_constraints": [
                    {"constraint_fn": lambda x, u, node, params: x[0] - 10.0},  # Good
                    {"constraint_fn": lambda x, u: x[0]},  # Bad - wrong signature
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )

    # Multiple CTCS constraints - error in second one (pydantic includes field path)
    with pytest.raises(ValueError, match="ctcs_constraints"):
        validate_byof(
            {
                "ctcs_constraints": [
                    {"constraint_fn": lambda x, u, node, params: x[0]},  # Good
                    {"constraint_fn": "not a function"},  # Bad
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )


# ===== Integration with Problem =====


def test_problem_validates_byof_during_construction():
    """Problem should validate byof during construction."""
    import jax.numpy as jnp

    import openscvx as ox

    position = ox.State("position", shape=(2,))
    position.min = np.array([0.0, 0.0])
    position.max = np.array([10.0, 10.0])
    position.initial = np.array([0.0, 10.0])
    position.final = np.array([10.0, 5.0])

    velocity = ox.State("velocity", shape=(1,))
    velocity.min = np.array([0.0])
    velocity.max = np.array([10.0])
    velocity.initial = np.array([0.0])
    velocity.final = [("free", 5.0)]

    theta = ox.Control("theta", shape=(1,))
    theta.min = np.array([0.0])
    theta.max = np.array([np.pi / 2])
    theta.guess = np.zeros((2, 1))

    dynamics = {"position": ox.Concat(velocity[0], velocity[0])}
    time = ox.Time(initial=0.0, final=1.0, min=0.0, max=2.0)

    # Invalid byof - dynamics with wrong signature (missing node, params)
    byof = {"dynamics": {"velocity": lambda x, u: jnp.array([1.0])}}

    # Validation happens during Problem construction (when lowering)
    with pytest.raises(ValueError, match="must have signature f\\(x, u, node, params\\)"):
        ox.Problem(
            dynamics=dynamics,
            constraints=[],
            states=[position, velocity],
            controls=[theta],
            time=time,
            N=2,
            byof=byof,
        )


def test_problem_accepts_valid_byof():
    """Problem should accept valid byof without errors during validation."""
    import jax.numpy as jnp

    import openscvx as ox
    from openscvx.expert import validate_byof

    # Create simple states
    position = ox.State("position", shape=(2,))
    velocity = ox.State("velocity", shape=(1,))
    states = [position, velocity]

    # Valid byof with proper dynamics signature
    byof = {
        "dynamics": {"velocity": lambda x, u, node, params: jnp.array([1.0])},
        "nodal_constraints": [{"constraint_fn": lambda x, u, node, params: x[0] - 10.0}],
        "ctcs_constraints": [
            {"constraint_fn": lambda x, u, node, params: x[1] - 5.0, "penalty": "square"}
        ],
    }

    # Direct validation should pass
    # n_x = 2 (position) + 1 (velocity) = 3, n_u = 1 (theta)
    validate_byof(byof, states, n_x=3, n_u=1, N=50)

    # Note: Full end-to-end integration testing is covered by test_brachistochrone.py::test_byof
    # This test just verifies the validation layer accepts valid byof specifications


def test_valid_convex_cost_signatures(simple_states):
    """Trajectory and nodal convex cost signatures are accepted."""
    import cvxpy as cp

    from openscvx.expert import validate_byof

    validate_byof(
        {
            "convex_costs": [
                {"cost_fn": lambda ocp_vars: cp.sum_squares(ocp_vars.x_nonscaled[0])},
                {
                    "cost_fn": lambda x, u, node, params, ocp_vars: cp.sum_squares(x),
                    "nodes": [0, -1],
                },
            ]
        },
        simple_states,
        n_x=3,
        n_u=1,
        N=50,
    )


def test_convex_cost_rejects_four_arg_nodal(simple_states):
    from openscvx.expert import validate_byof

    with pytest.raises(ValueError, match="4 parameters"):
        validate_byof(
            {
                "convex_costs": [
                    {"cost_fn": lambda x, u, node, params: x[0]},
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )


def test_convex_cost_rejects_nodes_on_trajectory_cost(simple_states):
    import cvxpy as cp

    from openscvx.expert import validate_byof

    with pytest.raises(ValueError, match="nodes.*trajectory-wide"):
        validate_byof(
            {
                "convex_costs": [
                    {
                        "cost_fn": lambda ocp_vars: cp.sum_squares(ocp_vars.x_nonscaled[0]),
                        "nodes": [0],
                    }
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )


def test_convex_cost_rejects_invalid_node_index(simple_states):
    import cvxpy as cp

    from openscvx.expert import validate_byof

    with pytest.raises(ValueError, match="invalid index"):
        validate_byof(
            {
                "convex_costs": [
                    {
                        "cost_fn": lambda x, u, node, params, ocp_vars: cp.sum_squares(x),
                        "nodes": [99],
                    }
                ]
            },
            simple_states,
            n_x=3,
            n_u=1,
            N=50,
        )
