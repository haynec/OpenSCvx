"""
Automatically discover and test all examples in the examples/ directory.

Discovery is a cheap glob over ``examples/`` that returns file paths only —
no example is imported until its own test runs. Each test imports the example
(which constructs its ``problem`` and applies its own JAX float configuration,
exactly as running the file standalone would) and validates that it converges.
"""

import importlib.util
import sys
from fnmatch import fnmatch
from pathlib import Path

import jax
import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# Shared helper modules, not examples: package markers, the private per-family
# helpers (``examples/_maze.py``, ``examples/drone/_plotting.py``, ...), helper
# packages (``examples/car/racing/_tracks/``, ...), and the shared plotting tier
# (``plotting.py``, ``plotting_viser.py``). Matched by naming convention — an
# underscore prefix on any path component — rather than by hand.
IGNORED_FILES = ["__init__.py", "_*.py", "plotting*.py"]

# Examples excluded from CI (e.g. exceeds runner memory)
EXCLUDED_EXAMPLES = {
    "animations/*.py",
    "arm/7_dof_arm_collision.py",
    "drone/logo.py",
    "drone/openscvx_logo.py",
    "double_integrator/obstacle_avoidance_vmap.py",
    "mjx/triple_cartpole_game.py",
    "rocket/ascent_launch_vehicle.py",
}

# Examples that require an optional dependency; their params carry the matching
# marker so they skip cleanly when the package is not installed.
_MJX_EXAMPLES = frozenset(
    {
        "mjx/cartpole_mjx.py",
        "mjx/skydio_x2_mjx.py",
        "mjx/triple_cartpole_mjx.py",
        "mjx/triple_cartpole_3d_mjx.py",
        "mjx/double_cartpole_mjx.py",
    }
)
_QPAX_EXAMPLES = frozenset({"abstract/brachistochrone_batched.py"})

# Timing bounds for specific examples (in seconds)
# Format: "relative/path/to/example.py": {"init": max_init, "solve": max_solve, "post": max_post}
TIMING_BOUNDS = {
    "abstract/brachistochrone.py": {
        "init": 10.0,
        "solve": 1.0,
        "post": 5.0,
    },
    "car/dubins_car.py": {
        "init": 15.0,
        "solve": 2.0,
        "post": 5.0,
    },
    "drone/obstacle_avoidance.py": {
        "init": 20.0,
        "solve": 0.5,
        "post": 5.0,
    },
    "drone/dr_vp.py": {
        "init": 75.0,
        "solve": 4.0,
        "post": 6.0,
    },
    "drone/cinema_vp.py": {
        "init": 25.0,
        "solve": 2.0,
        "post": 5.0,
    },
}


def _excluded(rel_path: Path) -> bool:
    """Return True if ``rel_path`` (relative to examples/) should not be tested."""
    if any(fnmatch(rel_path.name, pat) for pat in IGNORED_FILES):
        return True
    # Underscore-prefixed directories hold support code, not examples
    if any(part.startswith("_") for part in rel_path.parts[:-1]):
        return True
    # Realtime examples require special event loop handling
    if "realtime" in rel_path.parts:
        return True
    return any(rel_path.match(pat) for pat in EXCLUDED_EXAMPLES)


def discover_example_paths() -> list:
    """Glob examples/, applying IGNORED_FILES / EXCLUDED_EXAMPLES; no imports."""
    params = []
    for py_file in sorted(EXAMPLES_DIR.rglob("*.py")):
        rel = py_file.relative_to(EXAMPLES_DIR)
        if _excluded(rel):
            continue
        marks = []
        if rel.as_posix() in _MJX_EXAMPLES:
            marks.append(pytest.mark.mjx)
        if rel.as_posix() in _QPAX_EXAMPLES:
            marks.append(pytest.mark.qpax)
        params.append(
            pytest.param(py_file, id=str(rel.with_suffix("")).replace("/", "_"), marks=marks)
        )
    return params


def _import_example(path: Path):
    """Import the example file at ``path`` as a module and return it."""
    rel = path.relative_to(EXAMPLES_DIR)
    module_name = "examples." + str(rel.with_suffix("")).replace("/", ".")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.examples
@pytest.mark.parametrize("path", discover_example_paths())
def test_example(path):
    """
    Test that an example converges successfully.

    Each example is run through:
    1. import (constructs the `problem` and applies the example's JAX config)
    2. problem.initialize()
    3. problem.solve()
    4. problem.post_process()
    5. Assert convergence
    6. Check timing bounds (if specified for this example)
    """
    rel_path = str(path.relative_to(EXAMPLES_DIR))
    module = _import_example(path)
    if not hasattr(module, "problem"):
        pytest.skip(f"{rel_path} defines no `problem`")
    problem = module.problem

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Run the optimization pipeline
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    # Check convergence
    assert result["converged"], f"Example {rel_path} failed to converge"

    # Check timing bounds if specified for this example
    if rel_path in TIMING_BOUNDS:
        bounds = TIMING_BOUNDS[rel_path]
        timing_issues = []

        if "init" in bounds and hasattr(problem, "timing_init"):
            if problem.timing_init > bounds["init"]:
                timing_issues.append(f"init: {problem.timing_init:.2f}s > {bounds['init']:.2f}s")

        if "solve" in bounds and hasattr(problem, "timing_solve"):
            if problem.timing_solve > bounds["solve"]:
                timing_issues.append(f"solve: {problem.timing_solve:.2f}s > {bounds['solve']:.2f}s")

        if "post" in bounds and hasattr(problem, "timing_post"):
            if problem.timing_post > bounds["post"]:
                timing_issues.append(f"post: {problem.timing_post:.2f}s > {bounds['post']:.2f}s")

        if timing_issues:
            actual = ""
            if hasattr(problem, "timing_init"):
                actual += f"init={problem.timing_init:.2f}s, "
            if hasattr(problem, "timing_solve"):
                actual += f"solve={problem.timing_solve:.2f}s, "
            if hasattr(problem, "timing_post"):
                actual += f"post={problem.timing_post:.2f}s"

            assert False, (
                f"Example {rel_path} exceeded timing bounds:\n"
                f"  Violations: {', '.join(timing_issues)}\n"
                f"  Actual: {actual}"
            )

    # Clean up JAX caches
    jax.clear_caches()


def test_discovery_report():
    """Report discovered example paths."""
    params = discover_example_paths()
    print(f"\nDiscovered {len(params)} examples for the examples sweep:")
    for param in params:
        print(f"  - {param.id}")
    assert len(params) > 0, "No examples were discovered!"
