"""
Unit test for brachistochrone problem.

The brachistochrone problem asks for the curve of fastest descent under gravity
between two points. This has a known analytical solution: a cycloid curve.

For the problem setup in examples/abstract/brachistochrone.py:
- Start: (0, 10)
- End: (10, 5)
- g = 9.81 m/s²

The analytical solution gives an optimal time of approximately 1.808 seconds.
"""

import jax
import numpy as np
import pytest

from tests.brachistochrone_analytical import compare_trajectory_to_analytical


def _print_comparison_metrics(comparison, test_name="Brachistochrone"):
    """Print comparison metrics for brachistochrone validation."""
    print(f"\n{test_name} Validation Metrics:")
    print(f"  Analytical time:     {comparison['analytical_time']:.4f} s")
    print(f"  Numerical time:      {comparison['numerical_time']:.4f} s")
    print(f"  Time error:          {comparison['time_error_pct']:.2f}%")
    print(f"  Position RMSE:       {comparison['position_rmse']:.4f}")
    print(f"  Max position error:  {comparison['position_max_error']:.4f}")
    if comparison["velocity_rmse"] is not None:
        print(f"  Velocity RMSE:       {comparison['velocity_rmse']:.4f} m/s")
    if "R" in comparison and "phi_final" in comparison:
        print(
            f"  Cycloid parameters:  R={comparison['R']:.4f}, φ_final={comparison['phi_final']:.4f}"
        )
    if "arc_length" in comparison:
        print(f"  Arc length:          {comparison['arc_length']:.4f} m")


def _assert_brachistochrone_accuracy(comparison, problem, result):
    """Common assertions for brachistochrone test validation."""
    # Check time accuracy: numerical should be within 1% of analytical
    time_error_pct = comparison["time_error_pct"]
    assert time_error_pct < 1.0, (
        f"Time error {time_error_pct:.2f}% exceeds 1% threshold "
        f"(analytical: {comparison['analytical_time']:.4f}s, "
        f"numerical: {comparison['numerical_time']:.4f}s)"
    )

    # Check that numerical time is close to but not significantly better than analytical
    # (since analytical is theoretically optimal)
    assert comparison["numerical_time"] >= comparison["analytical_time"] * 0.95, (
        f"Numerical time {comparison['numerical_time']:.4f}s is suspiciously "
        f"better than analytical {comparison['analytical_time']:.4f}s"
    )

    # Check trajectory shape: position RMSE should be small
    # Current performance: ~0.01, so enforce < 0.05 with margin
    position_rmse = comparison["position_rmse"]
    assert position_rmse < 0.05, f"Position RMSE {position_rmse:.4f} exceeds threshold of 0.05"

    # Check maximum position error
    max_pos_error = comparison["position_max_error"]
    assert max_pos_error < 0.1, (
        f"Maximum position error {max_pos_error:.4f} exceeds threshold of 0.1"
    )

    # Check velocity accuracy
    # Current performance: ~0.01 m/s, so enforce < 0.05 m/s with margin
    velocity_rmse = comparison["velocity_rmse"]
    assert velocity_rmse < 0.05, f"Velocity RMSE {velocity_rmse:.4f} exceeds threshold of 0.05 m/s"

    # Check that we didn't take too many iterations
    if "discretization_history" in result:
        num_iters = len(result["discretization_history"])
        assert num_iters < 27, f"Took {num_iters} SCP iterations (expected < 15)"

    # Check timing - these are generous limits for a simple problem like brachistochrone
    assert problem.timing_init < 10.0, (
        f"Initialization took {problem.timing_init:.2f}s (expected < 10s)"
    )
    assert problem.timing_solve < 1.2, f"Solve took {problem.timing_solve:.2f}s (expected < 1.2s)"
    assert problem.timing_post < 5.0, (
        f"Post-processing took {problem.timing_post:.2f}s (expected < 5s)"
    )


def test_example():
    """
    Test the brachistochrone example from examples/abstract/brachistochrone.py.

    This validates against the known analytical solution and checks timing.
    """
    from examples.abstract.brachistochrone import problem

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Run optimization
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    # Check convergence
    assert result["converged"], "Brachistochrone failed to converge"

    # Compare numerical solution to analytical brachistochrone solution
    # Extract boundary conditions from problem definition
    x0, y0 = 0.0, 10.0
    x1, y1 = 10.0, 5.0
    g = 9.81

    # Compare to analytical solution
    comparison = compare_trajectory_to_analytical(
        result.t_full,
        result.trajectory["position"],
        result.trajectory["velocity"],
        x0,
        y0,
        x1,
        y1,
        g,
    )

    _print_comparison_metrics(comparison, "Brachistochrone")
    _assert_brachistochrone_accuracy(comparison, problem, result)

    # Clean up JAX caches
    jax.clear_caches()


def test_monolithic():
    """
    Test brachistochrone with monolithic state representation.

    This tests the alternative problem formulation style where all state
    components (x, y, v) are packed into a single state vector instead of
    being defined as separate named states.
    """
    import jax.numpy as jnp

    import openscvx as ox
    from openscvx import Problem

    # Problem parameters
    n = 2
    total_time = 2.0
    g = 9.81

    # Boundary conditions
    x0, y0 = 0.0, 10.0
    x1, y1 = 10.0, 5.0

    # Define monolithic state vector (x, y, v)
    x = ox.State("x", shape=(3,))
    x.max = np.array([10.0, 10.0, 10.0])  # Upper bounds
    x.min = np.array([0.0, 0.0, 0.0])  # Lower bounds
    x.initial = np.array([x0, y0, 0.0])  # [x0, y0, v0]
    x.final = [x1, y1, ("free", 10.0)]  # [x1, y1, v_free]

    # Define control
    u = ox.Control("u", shape=(1,))  # Angle from vertical
    u.max = np.array([100.5 * jnp.pi / 180])
    u.min = np.array([0.0])
    u.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    # Define dynamics for monolithic state
    x_dot = x[2] * ox.Sin(u[0])  # dx/dt = v * sin(theta)
    y_dot = -x[2] * ox.Cos(u[0])  # dy/dt = -v * cos(theta)
    v_dot = g * ox.Cos(u[0])  # dv/dt = g * cos(theta)
    dyn_expr = ox.Concat(x_dot, y_dot, v_dot)

    # Box constraints
    constraint_exprs = [
        ox.ctcs(x <= x.max),
        ox.ctcs(x.min <= x),
    ]

    time = ox.Time(
        initial=0.0,
        final=("minimize", total_time),
        min=0.0,
        max=total_time,
    )

    problem = Problem(
        dynamics={"x": dyn_expr},
        states=[x],
        controls=[u],
        time=time,
        constraints=constraint_exprs,
        N=n,
        licq_max=1e-8,
    )

    problem.settings.prp.dt = 0.01
    problem.settings.cvx.solver_args = {"abstol": 1e-6, "reltol": 1e-9}
    problem.settings.scp.lam_prox = 1e1  # Weight on the Trust Region
    problem.settings.scp.lam_cost = 1e0  # Weight on the Minimal Time Objective
    problem.settings.scp.lam_vc = 1e1  # Weight on the Virtual Control Objective
    problem.settings.scp.uniform_time_grid = True
    problem.settings.sim.save_compiled = False

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Run optimization
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    # Check convergence
    assert result["converged"], "Problem failed to converge"

    # Extract position and velocity from monolithic state
    # x.trajectory is shape (N, 3) with columns [x, y, v]
    position = result.trajectory["x"][:, :2]  # First 2 columns: [x, y]
    velocity = result.trajectory["x"][:, 2:3]  # Third column: [v]

    # Compare to analytical solution
    comparison = compare_trajectory_to_analytical(
        result.t_full, position, velocity, x0, y0, x1, y1, g
    )

    _print_comparison_metrics(comparison, "Brachistochrone Monolithic")
    _assert_brachistochrone_accuracy(comparison, problem, result)

    # Clean up JAX caches
    jax.clear_caches()


@pytest.mark.parametrize("constraint_type", ["ctcs", "nodal", "convex", "over", "at"])
def test_constraint_types(constraint_type):
    """
    Test brachistochrone with different constraint types.

    Args:
        constraint_type: One of "nodal", "convex", or "over" to specify how
            constraints are defined.
    """
    import jax.numpy as jnp

    import openscvx as ox
    from openscvx import Problem

    # Problem parameters
    n = 2
    total_time = 2.0
    g = 9.81

    # Boundary conditions
    x0, y0 = 0.0, 10.0
    x1, y1 = 10.0, 5.0

    # Define state components
    position = ox.State("position", shape=(2,))  # 2D position [x, y]
    position.max = np.array([10.0, 10.0])
    position.min = np.array([0.0, 0.0])
    position.initial = np.array([x0, y0])
    position.final = [x1, y1]

    velocity = ox.State("velocity", shape=(1,))  # Scalar speed
    velocity.max = np.array([10.0])
    velocity.min = np.array([0.0])
    velocity.initial = np.array([0.0])
    velocity.final = [("free", 10.0)]

    # Define control
    theta = ox.Control("theta", shape=(1,))  # Angle from vertical
    theta.max = np.array([100.5 * jnp.pi / 180])
    theta.min = np.array([0.0])
    theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    # Define list of all states (needed for Problem and constraints)
    states = [position, velocity]
    controls = [theta]

    # Define dynamics as dictionary mapping state names to their derivatives
    dynamics = {
        "position": ox.Concat(
            velocity[0] * ox.Sin(theta[0]),  # x_dot
            -velocity[0] * ox.Cos(theta[0]),  # y_dot
        ),
        "velocity": g * ox.Cos(theta[0]),
    }

    # Generate box constraints for all states based on constraint_type
    constraint_exprs = []
    for state in states:
        if constraint_type == "ctcs":
            constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
        elif constraint_type == "nodal":
            constraint_exprs.extend([state <= state.max, state.min <= state])
        elif constraint_type == "convex":
            constraint_exprs.extend([(state <= state.max).convex(), (state.min <= state).convex()])
        elif constraint_type == "over":
            constraint_exprs.extend(
                [
                    ox.ctcs(state <= state.max).over((0, 1)),
                    ox.ctcs(state.min <= state).over((0, 1)),
                ]
            )
        elif constraint_type == "at":
            for k in range(n):
                constraint_exprs.extend([(state <= state.max).at(k), (state.min <= state).at(k)])

    time = ox.Time(
        initial=0.0,
        final=("minimize", total_time),
        min=0.0,
        max=total_time,
    )

    problem = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraint_exprs,
        N=n,
        licq_max=1e-8,
    )

    problem.settings.prp.dt = 0.01
    problem.settings.cvx.solver_args = {"abstol": 1e-6, "reltol": 1e-9}
    problem.settings.scp.lam_prox = 1e1  # Weight on the Trust Region
    problem.settings.scp.lam_cost = 1e0  # Weight on the Minimal Time Objective
    problem.settings.scp.lam_vc = 1e1  # Weight on the Virtual Control Objective
    problem.settings.scp.uniform_time_grid = True
    problem.settings.sim.save_compiled = False

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Run optimization
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    # Check convergence
    assert result["converged"], "Problem failed to converge"

    # Compare to analytical solution
    comparison = compare_trajectory_to_analytical(
        result.t_full,
        result.trajectory["position"],
        result.trajectory["velocity"],
        x0,
        y0,
        x1,
        y1,
        g,
    )

    _print_comparison_metrics(comparison, f"Brachistochrone {constraint_type.capitalize()}")
    _assert_brachistochrone_accuracy(comparison, problem, result)

    # Clean up JAX caches
    jax.clear_caches()


@pytest.mark.parametrize("algorithm_type", ["augmented_lagrangian", "constant_proximal"])
def test_algorithm_types(algorithm_type):
    """
    Test brachistochrone with different algorithm types.

    Args:
        constraint_type: Specifies which algorithm is used.
    """
    import jax.numpy as jnp

    import openscvx as ox
    from openscvx import Problem

    # Problem parameters
    n = 2
    total_time = 2.0
    g = 9.81

    # Boundary conditions
    x0, y0 = 0.0, 10.0
    x1, y1 = 10.0, 5.0

    # Define state components
    position = ox.State("position", shape=(2,))  # 2D position [x, y]
    position.max = np.array([10.0, 10.0])
    position.min = np.array([0.0, 0.0])
    position.initial = np.array([x0, y0])
    position.final = [x1, y1]

    velocity = ox.State("velocity", shape=(1,))  # Scalar speed
    velocity.max = np.array([10.0])
    velocity.min = np.array([0.0])
    velocity.initial = np.array([0.0])
    velocity.final = [("free", 10.0)]

    # Define control
    theta = ox.Control("theta", shape=(1,))  # Angle from vertical
    theta.max = np.array([100.5 * jnp.pi / 180])
    theta.min = np.array([0.0])
    theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    # Define list of all states (needed for Problem and constraints)
    states = [position, velocity]
    controls = [theta]

    # Define dynamics as dictionary mapping state names to their derivatives
    dynamics = {
        "position": ox.Concat(
            velocity[0] * ox.Sin(theta[0]),  # x_dot
            -velocity[0] * ox.Cos(theta[0]),  # y_dot
        ),
        "velocity": g * ox.Cos(theta[0]),
    }

    # Generate box constraints for all states based on constraint_type
    constraint_exprs = []
    for state in states:
        constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    time = ox.Time(
        initial=0.0,
        final=("minimize", total_time),
        min=0.0,
        max=total_time,
    )

    if algorithm_type == "augmented_lagrangian":
        autotuner = ox.AugmentedLagrangian()
    elif algorithm_type == "constant_proximal":
        autotuner = ox.ConstantProximalWeight()

    problem = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraint_exprs,
        N=n,
        licq_max=1e-8,
        autotuner=autotuner,
    )

    problem.settings.prp.dt = 0.01
    problem.settings.cvx.solver_args = {"abstol": 1e-6, "reltol": 1e-9}
    problem.settings.scp.lam_prox = 1e1  # Weight on the Trust Region
    problem.settings.scp.lam_cost = 1e0  # Weight on the Minimal Time Objective
    problem.settings.scp.lam_vc = 1e1  # Weight on the Virtual Control Objective
    problem.settings.scp.uniform_time_grid = True
    problem.settings.sim.save_compiled = False

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Run optimization
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    # Check convergence
    assert result["converged"], "Problem failed to converge"

    # Compare to analytical solution
    comparison = compare_trajectory_to_analytical(
        result.t_full,
        result.trajectory["position"],
        result.trajectory["velocity"],
        x0,
        y0,
        x1,
        y1,
        g,
    )

    _print_comparison_metrics(comparison, f"Brachistochrone {algorithm_type.capitalize()}")
    _assert_brachistochrone_accuracy(comparison, problem, result)

    # Clean up JAX caches
    jax.clear_caches()


@pytest.mark.parametrize(
    "test_case",
    [
        "feasible-nonconvex",
        "infeasible-nonconvex",
        "feasible-convex",
        "infeasible-convex",
    ],
)
def test_cross_nodal(test_case):
    """
    Test brachistochrone with a cross-nodal rate limit constraint.
    """
    import jax.numpy as jnp

    import openscvx as ox
    from openscvx import Problem

    # Parse test case
    is_feasible = test_case.startswith("feasible")
    is_convex = test_case.endswith("convex")
    should_converge = is_feasible

    # Set max_step based on feasibility
    # For n=2 nodes, the distance between (0,10) and (10,5) is sqrt(125)
    max_step = np.sqrt(125) if is_feasible else np.sqrt(124.9)

    # Problem parameters
    n = 2
    total_time = 2.0
    g = 9.81

    # Boundary conditions
    x0, y0 = 0.0, 10.0
    x1, y1 = 10.0, 5.0

    # Define state components
    position = ox.State("position", shape=(2,))  # 2D position [x, y]
    position.max = np.array([10.0, 10.0])
    position.min = np.array([0.0, 0.0])
    position.initial = np.array([x0, y0])
    position.final = [x1, y1]

    velocity = ox.State("velocity", shape=(1,))  # Scalar speed
    velocity.max = np.array([10.0])
    velocity.min = np.array([0.0])
    velocity.initial = np.array([0.0])
    velocity.final = [("free", 10.0)]

    # Define control
    theta = ox.Control("theta", shape=(1,))  # Angle from vertical
    theta.max = np.array([100.5 * jnp.pi / 180])
    theta.min = np.array([0.0])
    theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    # Define list of all states (needed for Problem and constraints)
    states = [position, velocity]
    controls = [theta]

    # Define dynamics as dictionary mapping state names to their derivatives
    dynamics = {
        "position": ox.Concat(
            velocity[0] * ox.Sin(theta[0]),  # x_dot
            -velocity[0] * ox.Cos(theta[0]),  # y_dot
        ),
        "velocity": g * ox.Cos(theta[0]),
    }

    # Generate box constraints for all states
    constraint_exprs = []
    for state in states:
        constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    # Rate limit constraint with parameterized max_step
    # Create constraint for each node using absolute indexing
    for k in range(1, n):
        # Cross-node constraints don't need outer .at(k) - they're auto-detected
        # and converted to single constraints (not replicated to all nodes)
        rate_limit_constraint = (
            ox.linalg.Norm(position.at(k) - position.at(k - 1), ord=2) <= max_step
        )

        # Mark as convex if requested
        if is_convex:
            rate_limit_constraint = rate_limit_constraint.convex()

        constraint_exprs.append(rate_limit_constraint)

    time = ox.Time(
        initial=0.0,
        final=("minimize", total_time),
        min=0.0,
        max=total_time,
    )

    problem = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraint_exprs,
        N=n,
        licq_max=1e-8,
    )

    problem.settings.prp.dt = 0.01
    problem.settings.cvx.solver_args = {"abstol": 1e-6, "reltol": 1e-9}
    problem.settings.scp.lam_prox = 1e1  # Weight on the Trust Region
    problem.settings.scp.lam_cost = 1e0  # Weight on the Minimal Time Objective
    problem.settings.scp.lam_vc = 1e1  # Weight on the Virtual Control Objective
    problem.settings.scp.uniform_time_grid = True
    problem.settings.sim.save_compiled = False
    problem.settings.scp.k_max = 50  # Set lower max iterations for non-convergence case

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Run optimization
    # For infeasible convex problems, the solver will raise an error during initialization
    # For infeasible non-convex problems, SCP will fail to converge
    if is_convex and not should_converge:
        # Convex infeasible case: expect SolverError from CVXPy during initialization
        import cvxpy as cp

        with pytest.raises(cp.error.SolverError):
            problem.initialize()
            result = problem.solve()
    else:
        # Solvable or non-convex infeasible case
        problem.initialize()
        result = problem.solve()
        result = problem.post_process()

        # Check convergence based on parameter
        assert result["converged"] == should_converge, (
            f"Expected converged={should_converge} with max_step={max_step:.4f}, "
            f"is_convex={is_convex}, got converged={result['converged']}"
        )

        # Compare to analytical solution if converged
        if should_converge:
            comparison = compare_trajectory_to_analytical(
                result.t_full,
                result.trajectory["position"],
                result.trajectory["velocity"],
                x0,
                y0,
                x1,
                y1,
                g,
            )
            _print_comparison_metrics(comparison, "Brachistochrone Cross-Nodal")
            _assert_brachistochrone_accuracy(comparison, problem, result)

    # Clean up JAX caches
    jax.clear_caches()


def test_parameters():
    """
    Test brachistochrone with Parameter objects.

    This tests the use of ox.Parameter for problem constants (like gravity).
    Parameters allow symbolic representation of constants in the problem formulation,
    and can be modified between solves without re-initialization.
    """
    import jax.numpy as jnp

    import openscvx as ox
    from openscvx import Problem

    # Problem parameters
    n = 2
    total_time = 2.0

    # Boundary conditions
    x0, y0 = 0.0, 10.0
    x1, y1 = 10.0, 5.0

    # Create parameter for gravity (instead of hardcoding)
    g_param = ox.Parameter("g", value=9.81)

    # Define state components
    position = ox.State("position", shape=(2,))  # 2D position [x, y]
    position.max = np.array([10.0, 10.0])
    position.min = np.array([0.0, 0.0])
    position.initial = np.array([x0, y0])
    position.final = [x1, y1]

    velocity = ox.State("velocity", shape=(1,))  # Scalar speed
    velocity.max = np.array([10.0])
    velocity.min = np.array([0.0])
    velocity.initial = np.array([0.0])
    velocity.final = [("free", 10.0)]

    # Define control
    theta = ox.Control("theta", shape=(1,))  # Angle from vertical
    theta.max = np.array([100.5 * jnp.pi / 180])
    theta.min = np.array([0.0])
    theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    # Define list of all states (needed for Problem and constraints)
    states = [position, velocity]
    controls = [theta]

    # Define dynamics as dictionary mapping state names to their derivatives
    dynamics = {
        "position": ox.Concat(
            velocity[0] * ox.Sin(theta[0]),  # x_dot
            -velocity[0] * ox.Cos(theta[0]),  # y_dot
        ),
        "velocity": g_param * ox.Cos(theta[0]),  # Use parameter instead of hardcoded value
    }

    # Generate box constraints for all states
    constraint_exprs = []
    for state in states:
        constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    time = ox.Time(
        initial=0.0,
        final=("minimize", total_time),
        min=0.0,
        max=10.0,
    )
    # Apply custom scaling for time (Time is a State with shape=(1,))
    time.scaling_min = [0.0]
    time.scaling_max = [2.0]

    problem = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraint_exprs,
        N=n,
        licq_max=1e-8,
    )

    problem.settings.cvx.solver_args = {"abstol": 1e-6, "reltol": 1e-9}
    problem.settings.scp.lam_prox = 1e0
    problem.settings.scp.lam_cost = 1e-1
    problem.settings.scp.lam_vc = 1e1
    problem.settings.scp.uniform_time_grid = True
    problem.settings.sim.save_compiled = False

    # Save original weight values for second problem setup
    original_lam_prox = problem.settings.scp.lam_prox
    original_lam_cost = problem.settings.scp.lam_cost
    original_lam_vc = problem.settings.scp.lam_vc

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Run optimization with initial gravity parameter
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    # Check convergence
    assert result["converged"], "Problem failed to converge (first run)"

    # Extract position and velocity
    position_traj = result.trajectory["position"]
    velocity_traj = result.trajectory["velocity"]

    # Compare to analytical solution (using g_param.value for comparison)
    comparison = compare_trajectory_to_analytical(
        result.t_full, position_traj, velocity_traj, x0, y0, x1, y1, g_param.value
    )

    _print_comparison_metrics(comparison, "Brachistochrone Parameters (g=9.81)")
    print(f"  Using g parameter:   {g_param.value} m/s^2")
    _assert_brachistochrone_accuracy(comparison, problem, result)

    # Second run with different gravity parameter (e.g., Moon gravity)
    # Similar to how dubins_car.py modifies obstacle parameters
    g_moon = 1.62  # Moon's gravity in m/s^2
    problem.parameters["g"] = g_moon

    # Update time bounds for Moon gravity (weaker gravity means longer optimal time)
    # Expected time scales as 1/sqrt(g), so Moon time ≈ 2.46x Earth time
    total_time_moon = 10.0  # Generous upper bound for Moon gravity
    time_slice = problem.settings.sim.x.time_slice
    problem.settings.sim.x.max[time_slice] = total_time_moon
    problem.settings.sim.x.final[time_slice] = total_time_moon

    # Reset guesses for second run
    position.guess = np.linspace(position.initial, position.final, n)
    velocity.guess = np.linspace(0.0, 10.0, n).reshape(-1, 1)
    theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    # Restore original weight values for second problem setup
    problem.settings.scp.lam_prox = original_lam_prox
    problem.settings.scp.lam_cost = original_lam_cost
    problem.settings.scp.lam_vc = original_lam_vc

    # Reset solver state for second solve (parameters are updated)
    problem.reset()

    # Solve again without re-initialization (parameters are updated)
    problem.solve()
    result2 = problem.post_process()

    # Check convergence
    assert result2["converged"], "Problem failed to converge (second run with Moon gravity)"

    # Extract position and velocity
    position_traj2 = result2.trajectory["position"]
    velocity_traj2 = result2.trajectory["velocity"]

    # Compare to analytical solution with Moon gravity
    comparison2 = compare_trajectory_to_analytical(
        result2.t_full, position_traj2, velocity_traj2, x0, y0, x1, y1, g_moon
    )

    _print_comparison_metrics(comparison2, "Brachistochrone Parameters (g=1.62, Moon)")
    print(f"  Using g parameter:   {g_moon} m/s^2")
    _assert_brachistochrone_accuracy(comparison2, problem, result2)

    # Verify that the optimal time is longer with Moon gravity (weaker gravity)
    # Time should be proportional to 1/sqrt(g), so Moon time should be ~2.45x longer
    time_ratio = comparison2["analytical_time"] / comparison["analytical_time"]
    expected_ratio = np.sqrt(g_param.value / g_moon)  # sqrt(9.81/1.62) ≈ 2.46
    assert abs(time_ratio - expected_ratio) < 0.1, (
        f"Time ratio {time_ratio:.2f} doesn't match expected {expected_ratio:.2f} "
        f"for gravity scaling"
    )
    print(f"  Time ratio (Moon/Earth): {time_ratio:.2f} (expected: {expected_ratio:.2f})")

    # Clean up JAX caches
    jax.clear_caches()


def test_propagation():
    """
    Test brachistochrone with propagation dynamics and algebraic outputs.

    This test demonstrates using dynamics_prop and states_prop to add an
    extra state (distance) that is only propagated forward and not included
    in the optimization problem. The distance state integrates velocity to
    track the total arc length travelled along the brachistochrone curve.
    It also uses algebraic_prop to compute algebraic outputs (kinetic energy,
    potential energy, total energy) that are evaluated at each timestep
    without integration.
    """
    import jax.numpy as jnp

    import openscvx as ox
    from openscvx import Problem

    # Problem parameters
    n = 2
    total_time = 2.0
    g = 9.81

    # Boundary conditions
    x0, y0 = 0.0, 10.0
    x1, y1 = 10.0, 5.0

    # Define state components (optimization states)
    position = ox.State("position", shape=(2,))  # 2D position [x, y]
    position.max = np.array([10.0, 10.0])
    position.min = np.array([0.0, 0.0])
    position.initial = np.array([x0, y0])
    position.final = [x1, y1]

    velocity = ox.State("velocity", shape=(1,))  # Scalar speed
    velocity.max = np.array([10.0])
    velocity.min = np.array([0.0])
    velocity.initial = np.array([0.0])
    velocity.final = [("free", 10.0)]

    # Define control
    theta = ox.Control("theta", shape=(1,))  # Angle from vertical
    theta.max = np.array([100.5 * jnp.pi / 180])
    theta.min = np.array([0.0])
    theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    # Optimization states and controls
    states = [position, velocity]
    controls = [theta]

    # Define propagation-only state for tracking total distance traveled
    # Note: propagation states need explicit guesses since fill_default_guesses
    # only runs on main optimization states
    distance = ox.State("distance", shape=(1,))
    distance.initial = np.array([0.0])
    distance.min = np.array([0.0])
    distance.max = np.array([100.0])
    distance.guess = np.zeros((n, 1))

    # Extra propagation states: only the NEW states, not optimization states
    states_prop_extra = [distance]

    # Define dynamics for optimization states
    dynamics = {
        "position": ox.Concat(
            velocity[0] * ox.Sin(theta[0]),  # x_dot
            -velocity[0] * ox.Cos(theta[0]),  # y_dot
        ),
        "velocity": g * ox.Cos(theta[0]),
    }

    # Define EXTRA propagation dynamics (only for new states)
    # distance_dot = velocity (total arc length traveled)
    dynamics_prop_extra = {
        "distance": velocity[0],
    }

    # Define algebraic outputs (computed, not integrated)
    # These are evaluated at each propagation timestep via vmap
    mass = 1.0  # kg
    algebraic_prop = {
        "kinetic_energy": 0.5 * mass * velocity[0] ** 2,
        "potential_energy": mass * g * position[1],  # mgh (using y as height)
        "total_energy": 0.5 * mass * velocity[0] ** 2 + mass * g * position[1],
        # Test that algebraic_prop can depend on states_prop (propagation-only states)
        "distance_squared": distance[0] ** 2,
    }

    # Generate box constraints for optimization states
    constraint_exprs = []
    for state in states:
        constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    time = ox.Time(
        initial=0.0,
        final=("minimize", total_time),
        min=0.0,
        max=total_time,
    )

    problem = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraint_exprs,
        N=n,
        licq_max=1e-8,
        dynamics_prop=dynamics_prop_extra,  # Only extra states
        states_prop=states_prop_extra,  # Only extra states
        algebraic_prop=algebraic_prop,  # Algebraic outputs
    )

    problem.settings.prp.dt = 0.01
    problem.settings.cvx.solver_args = {"abstol": 1e-6, "reltol": 1e-9}
    problem.settings.scp.lam_prox = 1e1  # Weight on the Trust Region
    problem.settings.scp.lam_cost = 1e0  # Weight on the Minimal Time Objective
    problem.settings.scp.lam_vc = 1e1  # Weight on the Virtual Control Objective
    problem.settings.scp.uniform_time_grid = True
    problem.settings.sim.save_compiled = False

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Run optimization
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    # Check convergence
    assert result["converged"], "Problem failed to converge"

    # Compare to analytical solution
    comparison = compare_trajectory_to_analytical(
        result.t_full,
        result.trajectory["position"],
        result.trajectory["velocity"],
        x0,
        y0,
        x1,
        y1,
        g,
    )

    _print_comparison_metrics(comparison, "Brachistochrone Propagation")
    _assert_brachistochrone_accuracy(comparison, problem, result)

    # Verify that the distance state was propagated
    assert "distance" in result.trajectory, "Distance state not found in trajectory"
    distance_traj = result.trajectory["distance"]
    assert distance_traj.shape[0] == len(result.t_full), "Distance trajectory length mismatch"

    # Check that distance is monotonically increasing (since velocity >= 0)
    distance_values = distance_traj.flatten()
    assert np.all(np.diff(distance_values) >= -1e-6), "Distance should be monotonically increasing"

    # Check that initial distance is 0
    assert abs(distance_values[0]) < 1e-6, f"Initial distance should be 0, got {distance_values[0]}"

    # Check that final distance is positive and reasonable
    final_distance = distance_values[-1]
    assert final_distance > 0, "Final distance should be positive"

    # Compare numerical distance to analytical arc length
    analytical_arc_length = comparison["arc_length"]
    distance_error_pct = 100 * abs(final_distance - analytical_arc_length) / analytical_arc_length

    # Distance should match analytical arc length within 2%
    assert distance_error_pct < 2.0, (
        f"Distance error {distance_error_pct:.2f}% exceeds 2% threshold "
        f"(analytical: {analytical_arc_length:.4f}m, "
        f"numerical: {final_distance:.4f}m)"
    )

    # Sanity checks: arc length should be longer than straight line
    straight_line_distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    assert analytical_arc_length > straight_line_distance * 0.99, (
        f"Analytical arc length {analytical_arc_length:.4f}m should be greater than "
        f"straight-line distance {straight_line_distance:.4f}m"
    )

    print(f"\n  Distance travelled:    {final_distance:.4f} m")
    print(f"  Analytical arc length: {analytical_arc_length:.4f} m")
    print(f"  Distance error:        {distance_error_pct:.2f}%")
    print(f"  Straight-line dist:    {straight_line_distance:.4f} m")
    print(f"  Ratio (arc/line):      {analytical_arc_length / straight_line_distance:.4f}")

    # ==================== Verify Algebraic Outputs ====================

    # Verify that algebraic outputs were computed
    assert "kinetic_energy" in result.trajectory, "kinetic_energy not found in trajectory"
    assert "potential_energy" in result.trajectory, "potential_energy not found in trajectory"
    assert "total_energy" in result.trajectory, "total_energy not found in trajectory"

    # Get output trajectories
    ke = result.trajectory["kinetic_energy"]
    pe = result.trajectory["potential_energy"]
    te = result.trajectory["total_energy"]

    # Check shapes match the time trajectory
    assert ke.shape[0] == len(result.t_full), "kinetic_energy trajectory length mismatch"
    assert pe.shape[0] == len(result.t_full), "potential_energy trajectory length mismatch"
    assert te.shape[0] == len(result.t_full), "total_energy trajectory length mismatch"

    # Verify energy conservation (total energy should be approximately constant)
    # In brachistochrone without friction, mechanical energy is conserved
    te_values = te.flatten()
    te_mean = np.mean(te_values)
    te_std = np.std(te_values)
    te_variation_pct = 100 * te_std / te_mean

    # Energy should be conserved within 1% (allowing for numerical integration error)
    assert te_variation_pct < 1.0, (
        f"Total energy variation {te_variation_pct:.2f}% exceeds 1% threshold "
        f"(mean: {te_mean:.4f}, std: {te_std:.4f})"
    )

    # Verify kinetic + potential = total
    computed_total = ke.flatten() + pe.flatten()
    energy_sum_error = np.max(np.abs(computed_total - te_values))
    assert energy_sum_error < 1e-10, f"KE + PE != Total Energy, max error: {energy_sum_error:.2e}"

    # Verify initial kinetic energy is ~0 (starts from rest)
    assert ke[0] < 1e-6, f"Initial kinetic energy should be ~0, got {ke[0]}"

    # Verify kinetic energy increases (object accelerates down)
    assert ke[-1] > ke[0], "Final kinetic energy should be greater than initial"

    # Verify potential energy decreases (object moves down)
    assert pe[-1] < pe[0], "Final potential energy should be less than initial"

    # ==================== Verify algebraic_prop can use states_prop ====================

    # Verify that distance_squared (which depends on the propagation-only state) was computed
    assert "distance_squared" in result.trajectory, "distance_squared not found in trajectory"
    dist_sq = result.trajectory["distance_squared"]
    assert dist_sq.shape[0] == len(result.t_full), "distance_squared trajectory length mismatch"

    # Verify distance_squared = distance^2 (allowing for numerical precision)
    distance_traj = result.trajectory["distance"]
    computed_dist_sq = distance_traj.flatten() ** 2
    dist_sq_error = np.max(np.abs(dist_sq.flatten() - computed_dist_sq))
    assert dist_sq_error < 1e-5, f"distance_squared != distance^2, max error: {dist_sq_error:.2e}"

    # Clean up JAX caches
    jax.clear_caches()


@pytest.mark.parametrize(
    "byof_mode",
    ["ctcs", "nodal", "cross_nodal", "dynamics", "mixed"],
)
def test_byof(byof_mode):
    """
    Test brachistochrone using byof (bring-your-own-functions) for expert users.

    This test demonstrates using raw JAX functions instead of the symbolic layer.
    The byof parameter allows expert users to bypass the symbolic layer and directly
    specify JAX functions for different purposes.

    Args:
        byof_mode: One of:
            - "ctcs": CTCS constraints via byof
            - "nodal": Nodal constraints via byof
            - "cross_nodal": Cross-nodal constraints via byof
            - "dynamics": Replace all dynamics with byof
            - "mixed": Mix symbolic and byof (position dynamics + velocity dynamics/constraints via
                byof)

    The unified state vector after augmentation is:
    - x[0], x[1]: position (x, y)
    - x[2]: velocity
    - x[3]: time
    - x[4+]: augmented states (for CTCS)

    The unified control vector after augmentation is:
    - u[0]: theta (angle)
    - u[1]: time_dilation
    """
    import jax.numpy as jnp

    import openscvx as ox
    from openscvx import Problem
    from openscvx.expert import ByofSpec

    # Problem parameters
    n = 2
    total_time = 2.0
    g = 9.81

    # Boundary conditions
    x0, y0 = 0.0, 10.0
    x1, y1 = 10.0, 5.0

    # Define state components
    position = ox.State("position", shape=(2,))
    position.max = np.array([10.0, 10.0])
    position.min = np.array([0.0, 0.0])
    position.initial = np.array([x0, y0])
    position.final = [x1, y1]

    velocity = ox.State("velocity", shape=(1,))
    velocity.max = np.array([10.0])
    velocity.min = np.array([0.0])
    velocity.initial = np.array([0.0])
    velocity.final = [("free", 10.0)]

    # Define control
    theta = ox.Control("theta", shape=(1,))
    theta.max = np.array([100.5 * jnp.pi / 180])
    theta.min = np.array([0.0])
    theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    states = [position, velocity]
    controls = [theta]

    # Setup dynamics and constraints based on mode
    if byof_mode == "dynamics":
        # Pure byof dynamics: replace all dynamics with raw JAX functions
        # Note: dynamics dict must still contain time if time_dilation is enabled
        dynamics = {"time": 1.0}  # Only time dynamics in symbolic

        # Define position and velocity dynamics via byof
        # Using .slice property for clean state/control access
        byof: ByofSpec = {
            "dynamics": {
                "position": lambda x, u, node, params: jnp.array(
                    [
                        x[velocity.slice][0] * jnp.sin(u[theta.slice][0]),
                        -x[velocity.slice][0] * jnp.cos(u[theta.slice][0]),
                    ]
                ),
                "velocity": lambda x, u, node, params: jnp.array([g * jnp.cos(u[theta.slice][0])]),
            }
        }

        # Use symbolic constraints for box bounds
        constraints = []
        for state in states:
            constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    elif byof_mode == "mixed":
        # Mix symbolic and byof for both dynamics and constraints
        # Position dynamics via symbolic, velocity dynamics via byof
        # Position constraints via symbolic, velocity constraints via byof
        dynamics = {
            "position": ox.Concat(
                velocity[0] * ox.Sin(theta[0]),
                -velocity[0] * ox.Cos(theta[0]),
            ),
            "time": 1.0,
        }

        # Velocity dynamics via byof, velocity constraints via byof
        byof: ByofSpec = {
            "dynamics": {
                "velocity": lambda x, u, node, params: jnp.array([g * jnp.cos(u[theta.slice][0])]),
            },
            "ctcs_constraints": [
                {
                    "constraint_fn": lambda x, u, node, params: x[velocity.slice][0] - 10.0,
                    "penalty": "square",
                },
                {
                    "constraint_fn": lambda x, u, node, params: 0.0 - x[velocity.slice][0],
                    "penalty": "square",
                },
            ],
        }

        # Use symbolic constraints for position bounds only
        constraints = [
            ox.ctcs(position <= position.max),
            ox.ctcs(position.min <= position),
        ]

    elif byof_mode == "ctcs":
        # CTCS constraints via byof
        dynamics = {
            "position": ox.Concat(
                velocity[0] * ox.Sin(theta[0]),
                -velocity[0] * ox.Cos(theta[0]),
            ),
            "velocity": g * ox.Cos(theta[0]),
        }

        # Define box constraints using byof instead of symbolic layer
        byof: ByofSpec = {
            "ctcs_constraints": [
                {
                    "constraint_fn": lambda x, u, node, params: x[position.slice][0] - 10.0,
                    "penalty": "square",
                },
                {
                    "constraint_fn": lambda x, u, node, params: 0.0 - x[position.slice][0],
                    "penalty": "square",
                },
                {
                    "constraint_fn": lambda x, u, node, params: x[position.slice][1] - 10.0,
                    "penalty": "square",
                },
                {
                    "constraint_fn": lambda x, u, node, params: 0.0 - x[position.slice][1],
                    "penalty": "square",
                },
                {
                    "constraint_fn": lambda x, u, node, params: x[velocity.slice][0] - 10.0,
                    "penalty": "square",
                    "idx": 1,
                    "over": (0, 1),
                },
                {
                    "constraint_fn": lambda x, u, node, params: 0.0 - x[velocity.slice][0],
                    "penalty": "square",
                    "idx": 1,
                    "over": (0, 1),
                },
            ],
        }
        constraints = []

    elif byof_mode == "nodal":
        # Nodal constraints via byof
        dynamics = {
            "position": ox.Concat(
                velocity[0] * ox.Sin(theta[0]),
                -velocity[0] * ox.Cos(theta[0]),
            ),
            "velocity": g * ox.Cos(theta[0]),
        }

        # Define box constraints using byof nodal constraints
        # Constraints follow g(x, u) <= 0 convention
        byof: ByofSpec = {
            "nodal_constraints": [
                # position bounds (applied to all nodes)
                {
                    "constraint_fn": lambda x, u, node, params: x[position.slice][0] - 10.0
                },  # position[0] <= 10.0
                {
                    "constraint_fn": lambda x, u, node, params: 0.0 - x[position.slice][0]
                },  # position[0] >= 0.0
                {
                    "constraint_fn": lambda x, u, node, params: x[position.slice][1] - 10.0
                },  # position[1] <= 10.0
                {
                    "constraint_fn": lambda x, u, node, params: 0.0 - x[position.slice][1]
                },  # position[1] >= 0.0
                # velocity bounds (applied to all nodes)
                {
                    "constraint_fn": lambda x, u, node, params: x[velocity.slice][0] - 10.0
                },  # velocity[0] <= 10.0
                {
                    "constraint_fn": lambda x, u, node, params: 0.0 - x[velocity.slice][0]
                },  # velocity[0] >= 0.0
                # Demonstrate selective node enforcement: velocity must be exactly 0 at start
                {
                    "constraint_fn": lambda x, u, node, params: x[velocity.slice][
                        0
                    ],  # velocity == 0
                    "nodes": [0],  # Only enforce at first node
                },
            ],
        }
        constraints = []

    elif byof_mode == "cross_nodal":
        # Cross-nodal constraints via byof
        dynamics = {
            "position": ox.Concat(
                velocity[0] * ox.Sin(theta[0]),
                -velocity[0] * ox.Cos(theta[0]),
            ),
            "velocity": g * ox.Cos(theta[0]),
        }

        # Use symbolic for most constraints, byof for a cross-nodal constraint
        # For this test, we'll add a constraint that the total distance traveled
        # doesn't exceed a threshold
        byof: ByofSpec = {
            "cross_nodal_constraints": [
                # Sum of velocities across all nodes should be positive
                # This is a simple cross-nodal constraint for demonstration
                # X is (N, n_x), U is (N, n_u)
                lambda X, U, params: -jnp.sum(X[:, velocity.slice]),  # -sum(velocities) <= 0
            ],
        }

        # Use symbolic constraints for box bounds
        constraints = []
        for state in states:
            constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    time = ox.Time(
        initial=0.0,
        final=("minimize", total_time),
        min=0.0,
        max=total_time,
    )

    problem = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraints,
        N=n,
        licq_max=1e-8,
        byof=byof,
    )

    problem.settings.prp.dt = 0.01
    problem.settings.cvx.solver_args = {"abstol": 1e-6, "reltol": 1e-9}
    problem.settings.scp.lam_prox = 1e1
    problem.settings.scp.lam_cost = 1e0
    problem.settings.scp.lam_vc = 1e1
    problem.settings.scp.uniform_time_grid = True
    problem.settings.sim.save_compiled = False

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Run optimization
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    # Check convergence
    assert result["converged"], f"BYOF {byof_mode} problem failed to converge"

    # Compare to analytical solution
    comparison = compare_trajectory_to_analytical(
        result.t_full,
        result.trajectory["position"],
        result.trajectory["velocity"],
        x0,
        y0,
        x1,
        y1,
        g,
    )

    _print_comparison_metrics(comparison, f"Brachistochrone BYOF ({byof_mode})")
    _assert_brachistochrone_accuracy(comparison, problem, result)

    # Clean up JAX caches
    jax.clear_caches()


def test_idempotency():
    """
    Test that each step of the pipeline is idempotent.

    Calling initialize(), solve(), and post_process() twice in a row should
    not break anything and should produce identical results.

    This validates that:
    1. initialize() can be called multiple times without double-wrapping vmap/jit
    2. solve() can be called multiple times without corrupting state
    3. post_process() can be called multiple times without mutating solution/settings
    """
    from examples.abstract.brachistochrone import problem

    # Disable printing for cleaner test output
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False

    # Problem parameters for analytical comparison
    x0, y0 = 0.0, 10.0
    x1, y1 = 10.0, 5.0
    g = 9.81

    # Call each step twice to test idempotency
    problem.initialize()
    problem.initialize()  # Should not double-wrap vmap/jit

    problem.solve()
    problem.solve()  # Should not corrupt state (likely converges immediately)

    result1 = problem.post_process()
    result2 = problem.post_process()  # Should return identical results

    # Check convergence
    assert result1["converged"], "First post_process result should show convergence"
    assert result2["converged"], "Second post_process result should show convergence"

    # Compare both results to analytical solution
    comparison1 = compare_trajectory_to_analytical(
        result1.t_full,
        result1.trajectory["position"],
        result1.trajectory["velocity"],
        x0,
        y0,
        x1,
        y1,
        g,
    )
    comparison2 = compare_trajectory_to_analytical(
        result2.t_full,
        result2.trajectory["position"],
        result2.trajectory["velocity"],
        x0,
        y0,
        x1,
        y1,
        g,
    )

    # Comparisons should be identical
    assert comparison1["analytical_time"] == comparison2["analytical_time"]
    assert comparison1["numerical_time"] == comparison2["numerical_time"]
    assert comparison1["time_error_pct"] == comparison2["time_error_pct"]
    assert comparison1["position_rmse"] == comparison2["position_rmse"]
    assert comparison1["position_max_error"] == comparison2["position_max_error"]
    assert comparison1["velocity_rmse"] == comparison2["velocity_rmse"]

    print("\nIdempotency Test Results:")
    _print_comparison_metrics(comparison1, "Idempotency")
    print("  initialize() x2, solve() x2, post_process() x2 all succeeded")

    # Clean up JAX caches
    jax.clear_caches()
