"""
Integration tests for YAML/JSON problem loading.

These tests mirror a subset of test_brachistochrone.py but define the
problem via YAML/JSON files and ``load_dict`` instead of the Python API.
This validates the full pipeline: config file → parser → Problem → solve.

Tests that cannot be expressed in config files (byof, runtime parameter
modification, idempotency) are intentionally excluded.
"""

from pathlib import Path

import jax
import numpy as np
import pytest

from openscvx import Problem
from openscvx.symbolic.parser import load_dict, load_json, load_yaml
from tests.brachistochrone_analytical import compare_trajectory_to_analytical
from tests.test_brachistochrone import (
    _assert_brachistochrone_accuracy,
    _print_comparison_metrics,
)

# =============================================================================
# Constants
# =============================================================================

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Boundary conditions (must match the YAML file)
X0, Y0 = 0.0, 10.0
X1, Y1 = 10.0, 5.0
G = 9.81


# =============================================================================
# Helpers
# =============================================================================


def _configure_and_solve(problem):
    """Apply standard solver settings, solve, and post-process."""
    problem.settings.prp.dt = 0.01
    problem.settings.cvx.solver_args = {"abstol": 1e-6, "reltol": 1e-9}
    problem.settings.scp.lam_prox = 1e1
    problem.settings.scp.lam_cost = 1e0
    problem.settings.scp.lam_vc = 1e1
    problem.settings.scp.uniform_time_grid = True
    problem.settings.sim.save_compiled = False
    problem.settings.dev.printing = False

    problem.initialize()
    result = problem.solve()
    result = problem.post_process()
    return result


def _validate_result(result, problem, label="YAML"):
    """Assert convergence and accuracy against analytical solution."""
    assert result["converged"], f"{label} problem failed to converge"

    comparison = compare_trajectory_to_analytical(
        result.t_full,
        result.trajectory["position"],
        result.trajectory["velocity"],
        X0,
        Y0,
        X1,
        Y1,
        G,
    )
    _print_comparison_metrics(comparison, label)
    _assert_brachistochrone_accuracy(comparison, problem, result)
    return comparison


def _base_dict():
    """Return the base brachistochrone problem as a Python dict.

    Equivalent to the YAML fixture but as a dict for ``load_dict``.
    """
    return {
        "N": 2,
        "time": {"initial": 0.0, "final": ["minimize", 2.0], "min": 0.0, "max": 2.0},
        "states": [
            {
                "name": "position",
                "shape": [2],
                "min": [0.0, 0.0],
                "max": [10.0, 10.0],
                "initial": [0.0, 10.0],
                "final": [10.0, 5.0],
            },
            {
                "name": "velocity",
                "shape": [1],
                "min": [0.0],
                "max": [10.0],
                "initial": [0.0],
                "final": [["free", 10.0]],
            },
        ],
        "controls": [
            {
                "name": "theta",
                "shape": [1],
                "min": [0.0],
                "max": [1.755],
                "guess": [[0.0873], [1.7541]],
            },
        ],
        "parameters": [
            {"name": "g", "shape": [], "value": 9.81},
        ],
        "dynamics": {
            "position": "Concat(velocity[0] * Sin(theta[0]), -velocity[0] * Cos(theta[0]))",
            "velocity": "g * Cos(theta[0])",
        },
        "constraints": [
            "ctcs(position <= [10.0, 10.0])",
            "ctcs([0.0, 0.0] <= position)",
            "ctcs(velocity <= [10.0])",
            "ctcs([0.0] <= velocity)",
        ],
    }


# =============================================================================
# Core file loading test — proves the full load → Problem → solve pipeline
# =============================================================================

_LOADERS = {
    "yaml": (load_yaml, "brachistochrone.yaml"),
    "json": (load_json, "brachistochrone.json"),
}


@pytest.mark.parametrize("fmt", ["yaml", "json"])
def test_load_file(fmt):
    """Load brachistochrone from a config file, solve, and validate against analytical."""
    loader, filename = _LOADERS[fmt]
    kwargs = loader(FIXTURE_DIR / filename)
    problem = Problem(**kwargs)
    result = _configure_and_solve(problem)
    _validate_result(result, problem, f"Brachistochrone {fmt.upper()}")
    jax.clear_caches()


# =============================================================================
# Constraint type variants via load_dict
# =============================================================================


@pytest.mark.parametrize("constraint_type", ["ctcs", "nodal", "convex", "at", "over"])
def test_constraint_types(constraint_type):
    """Test YAML-style constraint strings with different constraint types."""
    data = _base_dict()

    if constraint_type == "ctcs":
        data["constraints"] = [
            "ctcs(position <= [10.0, 10.0])",
            "ctcs([0.0, 0.0] <= position)",
            "ctcs(velocity <= [10.0])",
            "ctcs([0.0] <= velocity)",
        ]
    elif constraint_type == "nodal":
        data["constraints"] = [
            "position <= [10.0, 10.0]",
            "[0.0, 0.0] <= position",
            "velocity <= [10.0]",
            "[0.0] <= velocity",
        ]
    elif constraint_type == "convex":
        data["constraints"] = [
            "(position <= [10.0, 10.0]).convex()",
            "([0.0, 0.0] <= position).convex()",
            "(velocity <= [10.0]).convex()",
            "([0.0] <= velocity).convex()",
        ]
    elif constraint_type == "at":
        # Explicit node enforcement for N=2
        data["constraints"] = []
        for k in range(2):
            data["constraints"].extend(
                [
                    f"(position <= [10.0, 10.0]).at({k})",
                    f"([0.0, 0.0] <= position).at({k})",
                    f"(velocity <= [10.0]).at({k})",
                    f"([0.0] <= velocity).at({k})",
                ]
            )
    elif constraint_type == "over":
        data["constraints"] = [
            "(position <= [10.0, 10.0]).over(0, 1)",
            "([0.0, 0.0] <= position).over(0, 1)",
            "(velocity <= [10.0]).over(0, 1)",
            "([0.0] <= velocity).over(0, 1)",
        ]

    kwargs = load_dict(data)
    problem = Problem(**kwargs)
    result = _configure_and_solve(problem)
    _validate_result(result, problem, f"YAML {constraint_type}")
    jax.clear_caches()


# =============================================================================
# Propagation (dynamics_prop, states_prop, algebraic_prop)
# =============================================================================


def test_propagation():
    """Test YAML-style problem with propagation states and algebraic outputs."""
    data = _base_dict()

    data["states_prop"] = [
        {
            "name": "distance",
            "shape": [1],
            "initial": [0.0],
            "min": [0.0],
            "max": [100.0],
            "guess": [[0.0], [0.0]],
        },
    ]

    data["dynamics_prop"] = {
        "distance": "velocity[0]",
    }

    data["algebraic_prop"] = {
        "kinetic_energy": "0.5 * velocity[0] ** 2",
        "potential_energy": "g * position[1]",
        "total_energy": "0.5 * velocity[0] ** 2 + g * position[1]",
        "distance_squared": "distance[0] ** 2",
    }

    kwargs = load_dict(data)
    problem = Problem(**kwargs)
    result = _configure_and_solve(problem)

    # --- Trajectory validation ---
    comparison = _validate_result(result, problem, "YAML Propagation")

    # --- Distance state ---
    assert "distance" in result.trajectory
    distance_values = result.trajectory["distance"].flatten()
    assert distance_values.shape[0] == len(result.t_full)
    assert np.all(np.diff(distance_values) >= -1e-6), "Distance should be monotonically increasing"
    assert abs(distance_values[0]) < 1e-6

    final_distance = distance_values[-1]
    assert final_distance > 0
    analytical_arc_length = comparison["arc_length"]
    distance_error_pct = 100 * abs(final_distance - analytical_arc_length) / analytical_arc_length
    assert distance_error_pct < 2.0, (
        f"Distance error {distance_error_pct:.2f}% exceeds 2% "
        f"(analytical: {analytical_arc_length:.4f}, numerical: {final_distance:.4f})"
    )

    # --- Energy conservation ---
    ke = result.trajectory["kinetic_energy"].flatten()
    pe = result.trajectory["potential_energy"].flatten()
    te = result.trajectory["total_energy"].flatten()

    te_variation_pct = 100 * np.std(te) / np.mean(te)
    assert te_variation_pct < 1.0, f"Total energy variation {te_variation_pct:.2f}% exceeds 1%"

    energy_sum_error = np.max(np.abs((ke + pe) - te))
    assert energy_sum_error < 1e-10, f"KE + PE != Total Energy, max error: {energy_sum_error:.2e}"
    assert ke[0] < 1e-6, "Initial kinetic energy should be ~0"
    assert ke[-1] > ke[0]
    assert pe[-1] < pe[0]

    # --- distance_squared depends on states_prop ---
    dist_sq = result.trajectory["distance_squared"].flatten()
    computed_dist_sq = distance_values**2
    dist_sq_error = np.max(np.abs(dist_sq - computed_dist_sq))
    assert dist_sq_error < 1e-5, f"distance_squared != distance^2, max error: {dist_sq_error:.2e}"

    jax.clear_caches()


# =============================================================================
# Cross-node constraints
# =============================================================================


@pytest.mark.parametrize("feasible", [True, False])
def test_cross_nodal(feasible):
    """Test YAML-style cross-node rate limit constraint."""
    # For N=2, distance between (0,10) and (10,5) is sqrt(125)
    max_step = float(np.sqrt(125)) if feasible else float(np.sqrt(124.9))

    data = _base_dict()
    data["constraints"].append(f"Norm(position.at(1) - position.at(0), ord=2) <= {max_step}")
    data["N"] = 2

    kwargs = load_dict(data)
    problem = Problem(**kwargs)

    problem.settings.prp.dt = 0.01
    problem.settings.cvx.solver_args = {"abstol": 1e-6, "reltol": 1e-9}
    problem.settings.scp.lam_prox = 1e1
    problem.settings.scp.lam_cost = 1e0
    problem.settings.scp.lam_vc = 1e1
    problem.settings.scp.uniform_time_grid = True
    problem.settings.sim.save_compiled = False
    problem.settings.scp.k_max = 50
    problem.settings.dev.printing = False

    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    assert result["converged"] == feasible, (
        f"Expected converged={feasible} with max_step={max_step:.4f}, "
        f"got converged={result['converged']}"
    )

    if feasible:
        _validate_result(result, problem, "YAML Cross-Nodal")

    jax.clear_caches()
