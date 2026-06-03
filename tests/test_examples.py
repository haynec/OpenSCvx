"""
Automatically discover and test all examples in the examples/ directory.

This test discovers all Python files in examples/ that define a 'problem' variable
and validates that they converge successfully.

Before each run, JAX floating-point settings are synced to each problem's
``Problem._float_dtype`` so behavior matches executing that example file on its own
(see :func:`sync_jax_float_config_for_problem`).
"""

import importlib.util
import sys
from pathlib import Path

import jax
import pytest

IGNORED_FILES = ["__init__.py", "plotting.py", "_viser_embed_export.py"]

# Examples excluded from CI (e.g. exceeds runner memory)
EXCLUDED_EXAMPLES = {
    "animations/*.py",
    "arm/7_dof_arm_collision.py",
    "drone/logo.py",
    "drone/openscvx_logo.py",
    # Heavy MJX examples (keep cartpole + skydio below)
    "mjx/double_cartpole_mjx.py",
    "mjx/triple_cartpole_mjx.py",
    "mjx/triple_cartpole_game.py",
    "mjx/triple_cartpole_3d_mjx.py",
}

# MJX examples included in discovery only when mujoco.mjx is installed (others
# call sys.exit on ImportError, which would abort pytest if imported blindly).
_MJX_EXAMPLES_REQUIRE_MUJOCO = frozenset(
    {
        "mjx/cartpole_mjx.py",
        "mjx/skydio_x2_mjx.py",
    }
)

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


def sync_jax_float_config_for_problem(problem) -> None:
    """Align global JAX + lowerer dtype with ``problem`` (same as running the example script alone).

    Discovery imports every example in one process; each :class:`~openscvx.problem.Problem`
    mutates ``jax_enable_x64`` and ``set_default_float_dtype`` at construction time. Before
    ``initialize()`` we restore the settings for *this* problem so tests match a manual run.
    """
    from openscvx.symbolic.lowerers.jax.logic import set_default_float_dtype

    float_dtype = getattr(problem, "_float_dtype", "float32")
    enable_x64 = float_dtype.lower() in ("float64", "f64", "double")
    jax.config.update("jax_enable_x64", enable_x64)
    set_default_float_dtype(float_dtype)


def discover_examples():
    """Discover all runnable examples in the examples/ directory."""
    examples_dir = Path(__file__).parent.parent / "examples"
    discovered = {}

    # Find all .py files in examples/
    for py_file in examples_dir.rglob("*.py"):
        # Skip non-example files
        if py_file.name in IGNORED_FILES:
            continue

        # Skip realtime examples (require special event loop handling)
        if "realtime" in py_file.parts:
            continue

        # Get relative path for naming
        rel_path = py_file.relative_to(examples_dir)

        # Skip explicitly excluded examples (glob patterns, e.g. "animations/*.py")
        if any(rel_path.match(pat) for pat in EXCLUDED_EXAMPLES):
            continue

        rel_str = rel_path.as_posix()
        if rel_str in _MJX_EXAMPLES_REQUIRE_MUJOCO:
            try:
                import mujoco.mjx  # noqa: F401
            except ImportError:
                continue

        module_name = str(rel_path.with_suffix("")).replace("/", ".")

        try:
            # Import the example module
            spec = importlib.util.spec_from_file_location(f"examples.{module_name}", py_file)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"examples.{module_name}"] = module
            spec.loader.exec_module(module)

            # Only include if it has a 'problem' attribute
            if hasattr(module, "problem"):
                test_name = module_name.replace(".", "_")
                discovered[test_name] = {
                    "problem": module.problem,
                    "path": str(rel_path),
                }

        except Exception as e:
            # Skip files that can't be imported
            print(f"Warning: Could not import {rel_path}: {e}")
            continue

    return discovered


# Discover examples at module load time
DISCOVERED_EXAMPLES = discover_examples()


@pytest.mark.integration
@pytest.mark.parametrize(
    "name,metadata", DISCOVERED_EXAMPLES.items(), ids=list(DISCOVERED_EXAMPLES.keys())
)
def test_example(name, metadata):
    """
    Test that a discovered example converges successfully.

    Each example is run through:
    1. problem.initialize()
    2. problem.solve()
    3. problem.post_process()
    4. Assert convergence
    5. Check timing bounds (if specified for this example)
    """
    problem = metadata["problem"]
    sync_jax_float_config_for_problem(problem)

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Run the optimization pipeline
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    # Check convergence
    assert result["converged"], f"Example {name} ({metadata['path']}) failed to converge"

    # Check timing bounds if specified for this example
    example_path = metadata["path"]
    if example_path in TIMING_BOUNDS:
        bounds = TIMING_BOUNDS[example_path]
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
                f"Example {name} ({example_path}) exceeded timing bounds:\n"
                f"  Violations: {', '.join(timing_issues)}\n"
                f"  Actual: {actual}"
            )

    # Clean up JAX caches
    jax.clear_caches()


def test_discovery_report():
    """Report discovered examples."""
    print(f"\nDiscovered {len(DISCOVERED_EXAMPLES)} examples for integration testing:")
    for name, metadata in sorted(DISCOVERED_EXAMPLES.items()):
        print(f"  - {name:40s} ({metadata['path']})")
    assert len(DISCOVERED_EXAMPLES) > 0, "No examples were discovered!"
