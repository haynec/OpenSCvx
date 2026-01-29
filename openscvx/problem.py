"""Core optimization problem interface for trajectory optimization.

This module provides the Problem class, the main entry point for defining
and solving trajectory optimization problems using Sequential Convex Programming (SCP).

Example:
    The prototypical flow is to define a problem, then initialize, solve, and post-process the
    results

        problem = Problem(dynamics, constraints, states, controls, N, time)
        problem.initialize()
        result = problem.solve()
        result = problem.post_process()
"""

import copy
import os
import queue
import threading
import time
from typing import List, Optional, Union

import jax

os.environ["EQX_ON_ERROR"] = "nan"

from openscvx.algorithms import (
    AlgorithmState,
    AugmentedLagrangian,
    AutotuningBase,
    OptimizationResults,
    PenalizedTrustRegion,
)
from openscvx.config import (
    Config,
    ConvexSolverConfig,
    DevConfig,
    DiscretizationConfig,
    PropagationConfig,
    ScpConfig,
    SimConfig,
)
from openscvx.discretization import get_discretization_solver
from openscvx.expert import ByofSpec
from openscvx.lowered import LoweredProblem, ParameterDict
from openscvx.lowered.dynamics import Dynamics
from openscvx.lowered.jax_constraints import (
    LoweredCrossNodeConstraint,
    LoweredJaxConstraints,
    LoweredNodalConstraint,
)
from openscvx.propagation import get_propagation_solver, propagate_trajectory_results
from openscvx.solvers import PTRSolver
from openscvx.symbolic.builder import preprocess_symbolic_problem
from openscvx.symbolic.expr import CTCS, Constraint
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State
from openscvx.symbolic.lower import lower_symbolic_problem
from openscvx.symbolic.problem import SymbolicProblem
from openscvx.symbolic.time import Time
from openscvx.utils import printing, profiling
from openscvx.utils.caching import (
    get_solver_cache_paths,
    load_or_compile_discretization_solver,
    load_or_compile_propagation_solver,
    prime_propagation_solver,
)


class Problem:
    def __init__(
        self,
        dynamics: dict,
        constraints: List[Union[Constraint, CTCS]],
        states: List[State],
        controls: List[Control],
        N: int,
        time: Time,
        *,
        dynamics_prop: Optional[dict] = None,
        states_prop: Optional[List[State]] = None,
        algebraic_prop: Optional[dict] = None,
        licq_min: float = 0.0,
        licq_max: float = 1e-4,
        time_dilation_factor_min: float = 0.3,
        time_dilation_factor_max: float = 3.0,
        autotuner: Optional[AutotuningBase] = AugmentedLagrangian(),
        byof: Optional[ByofSpec] = None,
    ):
        """The primary class in charge of compiling and exporting the solvers.

        Args:
            dynamics (dict): Dictionary mapping state names to their dynamics expressions.
                Each key should be a state name, and each value should be an Expr
                representing the derivative of that state.
            constraints (List[Union[CTCSConstraint, NodalConstraint]]):
                List of constraints decorated with @ctcs or @nodal
            states (List[State]): List of State objects representing the state variables.
                May optionally include a State named "time" (see time parameter below).
            controls (List[Control]): List of Control objects representing the control variables
            N (int): Number of segments in the trajectory
            time (Time): Time configuration object with initial, final, min, max.
                Required. If including a "time" state in states, the Time object will be ignored
                and time properties should be set on the time State object instead.
            dynamics_prop (dict, optional): Dictionary mapping EXTRA state names to their
                dynamics expressions for propagation. Only specify additional states beyond
                optimization states (e.g., {"distance": speed}). Do NOT duplicate optimization
                state dynamics here.
            states_prop (List[State], optional): List of EXTRA State objects for propagation only.
                Only specify additional states beyond optimization states. Used with dynamics_prop.
            algebraic_prop (dict, optional): Dictionary mapping names to symbolic expressions
                for outputs evaluated (not integrated) during propagation.
            licq_min (float): Minimum LICQ constraint value. Defaults to 0.0.
            licq_max (float): Maximum LICQ constraint value. Defaults to 1e-4.
            time_dilation_factor_min (float): Minimum time dilation factor.
                Defaults to 0.3.
            time_dilation_factor_max (float): Maximum time dilation factor.
                Defaults to 3.0.
            byof (ByofSpec, optional): Expert mode only. Raw JAX functions to
                bypass symbolic layer. See :class:`openscvx.expert.ByofSpec` for
                detailed documentation.

        Note:
            There are two approaches for handling time:
            1. Auto-create (simple): Don't include "time" in states, provide Time object
            2. User-provided (for time-dependent constraints): Include "time" State in states and
               in dynamics dict, don't provide Time object
        """

        # Symbolic Preprocessing & Augmentation
        self.symbolic: SymbolicProblem = preprocess_symbolic_problem(
            dynamics=dynamics,
            constraints=constraints,
            states=states,
            controls=controls,
            N=N,
            time=time,
            licq_min=licq_min,
            licq_max=licq_max,
            time_dilation_factor_min=time_dilation_factor_min,
            time_dilation_factor_max=time_dilation_factor_max,
            dynamics_prop_extra=dynamics_prop,
            states_prop_extra=states_prop,
            algebraic_prop=algebraic_prop,
            byof=byof,
        )

        # Validate byof early (after preprocessing, before lowering) to fail fast
        if byof is not None:
            from openscvx.expert.validation import validate_byof

            # Calculate unified state and control dimensions from preprocessed states/controls
            # These dimensions include symbolic augmentation (time, CTCS) but not byof CTCS
            # augmentation, which is exactly what user byof functions will see
            n_x = sum(
                state.shape[0] if len(state.shape) > 0 else 1 for state in self.symbolic.states
            )
            n_u = sum(
                control.shape[0] if len(control.shape) > 0 else 1
                for control in self.symbolic.controls
            )

            validate_byof(byof, self.symbolic.states, n_x, n_u, N)

        # Store byof for cache hashing
        self._byof = byof

        # Create solver before lowering (solver owns its variables)
        self._solver: PTRSolver = PTRSolver()

        # Lower to JAX and CVXPy (byof handling happens inside lower_symbolic_problem)
        self._lowered: LoweredProblem = lower_symbolic_problem(
            self.symbolic, self._solver, byof=byof
        )

        # Store parameters in two forms:
        self._parameters = self.symbolic.parameters  # Plain dict for JAX functions
        # Wrapper dict for user access that auto-syncs
        self._parameter_wrapper = ParameterDict(self, self._parameters, self.symbolic.parameters)

        # Setup SCP Configuration
        self.settings = Config(
            sim=SimConfig(
                x=self._lowered.x_unified,
                x_prop=self._lowered.x_prop_unified,
                u=self._lowered.u_unified,
                total_time=self._lowered.x_unified.initial[self._lowered.x_unified.time_slice][0],
                n_states=self._lowered.x_unified.initial.shape[0],
                n_states_prop=self._lowered.x_prop_unified.initial.shape[0],
                ctcs_node_intervals=self.symbolic.node_intervals,
            ),
            scp=ScpConfig(
                n=N,
                n_states=self._lowered.x_unified.shape[0],
                autotuner=autotuner,
            ),
            dis=DiscretizationConfig(),
            dev=DevConfig(),
            cvx=ConvexSolverConfig(),
            prp=PropagationConfig(),
        )

        # OCP construction happens in initialize() so users can modify
        # settings (like uniform_time_grid) between __init__ and initialize()
        self._discretization_solver: callable = None

        # Set up emitter & queue (thread started in initialize() after columns are known)
        if self.settings.dev.printing:
            self.print_queue = queue.Queue()
            self.emitter_function = lambda data: self.print_queue.put(data)
            self.print_thread = None  # Started in initialize()
        else:
            # no-op emitter; nothing ever gets queued or printed
            self.print_queue = None
            self.emitter_function = lambda data: None
            self.print_thread = None

        # Columns for printing (set in initialize() based on algorithm + autotuner)
        self._columns = None

        self.timing_init = None
        self.timing_solve = None
        self.timing_post = None

        # Compiled dynamics (vmapped versions, set in initialize())
        self._compiled_dynamics: Optional[Dynamics] = None
        self._compiled_dynamics_prop: Optional[Dynamics] = None

        # Compiled constraints (JIT-compiled versions, set in initialize())
        self._compiled_constraints: Optional[LoweredJaxConstraints] = None

        # Solver state (created fresh for each solve)
        self._state: Optional[AlgorithmState] = None

        # Final solution state (saved after successful solve)
        self._solution: Optional[AlgorithmState] = None

        # SCP algorithm (currently hardcoded to PTR)
        self._algorithm = PenalizedTrustRegion()

    @property
    def parameters(self) -> ParameterDict:
        """Get the parameters dictionary.

        The returned dictionary automatically syncs to CVXPy when modified:
            problem.parameters["obs_radius"] = 2.0  # Auto-syncs to CVXPy
            problem.parameters.update({"gate_0_center": center})  # Also syncs

        Returns:
            ParameterDict: Special dict that syncs to CVXPy on assignment.
        """
        return self._parameter_wrapper

    @parameters.setter
    def parameters(self, new_params: dict):
        """Replace the entire parameters dictionary and sync to CVXPy.

        Args:
            new_params: New parameters dictionary
        """
        self._parameters = dict(new_params)  # Create new plain dict
        self._parameter_wrapper = ParameterDict(self, self._parameters, new_params)
        self._sync_parameters()

    def _sync_parameters(self):
        """Sync all parameter values to CVXPy parameters."""
        if self._lowered.cvxpy_params is not None:
            for name, value in self._parameter_wrapper.items():
                if name in self._lowered.cvxpy_params:
                    self._lowered.cvxpy_params[name].value = value

    @property
    def state(self) -> Optional[AlgorithmState]:
        """Access the current solver state.

        The solver state contains all mutable state from the SCP iterations,
        including current guesses, costs, weights, and history.

        Returns:
            AlgorithmState if initialized, None otherwise

        Example:
            When using `Problem.step()` can use the state to check convergence _etc._

                problem.initialize()
                problem.step()
                print(f"Iteration {problem.state.k}, J_tr={problem.state.J_tr}")
        """
        return self._state

    @property
    def lowered(self) -> LoweredProblem:
        """Access the lowered problem containing JAX/CVXPy objects.

        Returns:
            LoweredProblem with dynamics, constraints, unified interfaces, and CVXPy vars
        """
        return self._lowered

    @property
    def x_unified(self):
        """Unified state interface (delegates to lowered.x_unified)."""
        return self._lowered.x_unified

    @property
    def u_unified(self):
        """Unified control interface (delegates to lowered.u_unified)."""
        return self._lowered.u_unified

    @property
    def slices(self) -> dict[str, slice]:
        """Get mapping of state and control names to their slices in unified vectors.

        This property returns a dictionary mapping each state and control variable name
        to its slice in the respective unified vector. This is particularly useful for
        expert users working with byof (bring-your-own functions) who need to manually
        index into the unified x and u vectors.

        Returns:
            Dictionary mapping variable names to slice objects.
                State variables map to slices in the x vector.
                Control variables map to slices in the u vector.

        Example:
            Usage with byof::

                problem = ox.Problem(dynamics, states, controls, ...)
                print(problem.slices)
                # {'position': slice(0, 3), 'velocity': slice(3, 6), 'theta': slice(0, 1)}

                # Use in byof functions
                byof = {
                    "nodal_constraints": [
                        lambda x, u, node, params: x[problem.slices["velocity"][0]] - 10.0,
                        lambda x, u, node, params: u[problem.slices["theta"][0]] - 1.57,
                    ]
                }
        """
        slices = {}
        slices.update({state.name: state.slice for state in self.symbolic.states})
        slices.update({control.name: control.slice for control in self.symbolic.controls})
        return slices

    def _format_result(self, state: AlgorithmState, converged: bool) -> OptimizationResults:
        """Format solver state as an OptimizationResults object.

        Converts the internal solver state into a user-facing results object,
        mapping state/control arrays to named fields based on symbolic metadata.

        Args:
            state: The AlgorithmState to extract results from.
            converged: Whether the optimization converged.

        Returns:
            OptimizationResults containing the solution data.
        """
        # Build nodes dictionary with all states and controls
        nodes_dict = {}

        # Add all states (user-defined and augmented)
        for sym_state in self.symbolic.states:
            nodes_dict[sym_state.name] = state.x[:, sym_state._slice]

        # Add all controls (user-defined and augmented)
        for control in self.symbolic.controls:
            nodes_dict[control.name] = state.u[:, control._slice]

        return OptimizationResults(
            converged=converged,
            t_final=state.x[:, self.settings.sim.time_slice][-1],
            nodes=nodes_dict,
            trajectory={},  # Populated by post_process
            _states=self.symbolic.states_prop,  # Use propagation states for trajectory dict
            _controls=self.symbolic.controls,
            X=state.X,  # Single source of truth - x and u are properties
            U=state.U,
            discretization_history=state.V_history,
            J_tr_history=state.J_tr,
            J_vb_history=state.J_vb,
            J_vc_history=state.J_vc,
            TR_history=state.TR_history,
            VC_history=state.VC_history,
            lam_prox_history=state.lam_prox_history.copy(),
            actual_reduction_history=state.actual_reduction_history.copy(),
            pred_reduction_history=state.pred_reduction_history.copy(),
            acceptance_ratio_history=state.acceptance_ratio_history.copy(),
        )

    def initialize(self):
        """Compile dynamics, constraints, and solvers; prepare for optimization.

        This method vmaps dynamics, JIT-compiles constraints, builds the convex
        subproblem, and initializes the solver state. Must be called before solve().

        Example:
            Prior to calling the `.solve()` method it is necessary to initialize the problem

                problem = Problem(dynamics, constraints, states, controls, N, time)
                problem.initialize()  # Compile and prepare
                problem.solve()       # Run optimization
        """
        printing.intro()

        # Enable the profiler
        pr = profiling.profiling_start(self.settings.dev.profiling)

        t_0_while = time.time()
        # Ensure parameter sizes and normalization are correct
        self.settings.scp.__post_init__()
        self.settings.sim.__post_init__()

        # Create compiled (vmapped) dynamics as new instances
        # This preserves the original un-vmapped versions in _lowered
        self._compiled_dynamics = Dynamics(
            f=jax.vmap(self._lowered.dynamics.f, in_axes=(0, 0, 0, None)),
            A=jax.vmap(self._lowered.dynamics.A, in_axes=(0, 0, 0, None)),
            B=jax.vmap(self._lowered.dynamics.B, in_axes=(0, 0, 0, None)),
        )

        self._compiled_dynamics_prop = Dynamics(
            f=jax.vmap(self._lowered.dynamics_prop.f, in_axes=(0, 0, 0, None)),
        )

        # Create compiled (JIT-compiled) constraints as new instances
        # This preserves the original un-JIT'd versions in _lowered
        # TODO: (haynec) switch to AOT instead of JIT
        compiled_nodal = [
            LoweredNodalConstraint(
                func=jax.jit(c.func),
                grad_g_x=jax.jit(c.grad_g_x),
                grad_g_u=jax.jit(c.grad_g_u),
                nodes=c.nodes,
            )
            for c in self._lowered.jax_constraints.nodal
        ]

        compiled_cross_node = [
            LoweredCrossNodeConstraint(
                func=jax.jit(c.func),
                grad_g_X=jax.jit(c.grad_g_X),
                grad_g_U=jax.jit(c.grad_g_U),
            )
            for c in self._lowered.jax_constraints.cross_node
        ]

        self._compiled_constraints = LoweredJaxConstraints(
            nodal=compiled_nodal,
            cross_node=compiled_cross_node,
            ctcs=self._lowered.jax_constraints.ctcs,  # CTCS aren't JIT-compiled here
        )

        # Generate solvers using compiled (vmapped) dynamics
        self._discretization_solver = get_discretization_solver(
            self._compiled_dynamics, self.settings
        )
        self._propagation_solver = get_propagation_solver(
            self._compiled_dynamics_prop.f, self.settings
        )

        # Build convex subproblem (solver was created in __init__, variables in lower)
        self._solver.initialize(self._lowered, self.settings)

        # Print problem summary (after solver is initialized so we can access problem stats)
        printing.print_problem_summary(self.settings, self._lowered, self._solver)

        # Get cache file paths using symbolic AST hashing
        # This is more stable than hashing lowered JAX code
        dis_solver_file, prop_solver_file = get_solver_cache_paths(
            self.symbolic,
            dt=self.settings.prp.dt,
            total_time=self.settings.sim.total_time,
            byof=self._byof,
        )

        # Compile the discretization solver
        self._discretization_solver = load_or_compile_discretization_solver(
            self._discretization_solver,
            dis_solver_file,
            self._parameters,  # Plain dict for JAX
            self.settings.scp.n,
            self.settings.sim.n_states,
            self.settings.sim.n_controls,
            save_compiled=self.settings.sim.save_compiled,
            debug=self.settings.dev.debug,
        )

        # Setup propagation solver parameters
        dtau = 1.0 / (self.settings.scp.n - 1)
        dt_max = self.settings.sim.u.max[self.settings.sim.time_dilation_slice][0] * dtau
        self.settings.prp.max_tau_len = int(dt_max / self.settings.prp.dt) + 2

        # Compile the propagation solver
        self._propagation_solver = load_or_compile_propagation_solver(
            self._propagation_solver,
            prop_solver_file,
            self._parameters,  # Plain dict for JAX
            self.settings.sim.n_states_prop,
            self.settings.sim.n_controls,
            self.settings.prp.max_tau_len,
            save_compiled=self.settings.sim.save_compiled,
        )

        # Initialize the SCP algorithm
        print("Initializing the SCvx Subproblem Solver...")
        self._algorithm.initialize(
            self._solver,
            self._discretization_solver,
            self._compiled_constraints,
            self.emitter_function,
            self._parameters,  # For warm-start only
            self.settings,  # For warm-start only
        )
        print("✓ SCvx Subproblem Solver initialized")

        # Get columns from algorithm (now that autotuner is set) and start print thread
        if self.settings.dev.printing:
            self._columns = self._algorithm.get_columns(self.settings.dev.verbosity)
            self.print_thread = threading.Thread(
                target=printing.intermediate,
                args=(self.print_queue, self.settings, self._columns),
                daemon=True,
            )
            self.print_thread.start()
        else:
            # Printing was disabled after __init__, disable emitter to avoid queue buildup
            self.emitter_function = lambda data: None

        # Create fresh solver state
        self._state = AlgorithmState.from_settings(self.settings)

        t_f_while = time.time()
        self.timing_init = t_f_while - t_0_while
        print("Total Initialization Time: ", self.timing_init)

        # Prime the propagation solver
        prime_propagation_solver(self._propagation_solver, self._parameters, self.settings)

        profiling.profiling_end(pr, "initialize")

    def reset(self):
        """Reset solver state to re-run optimization from initial conditions.

        Creates fresh AlgorithmState while preserving compiled dynamics and solvers.
        Use this to run multiple optimizations without re-initializing.

        Raises:
            ValueError: If initialize() has not been called yet.

        Example:
            After calling `.step()` it may be necessary to reset the problem back to the initial
            conditions

                problem.initialize()
                result1 = problem.step()
                problem.reset()
                result2 = problem.solve()  # Fresh run with same setup
        """
        if self._compiled_dynamics is None:
            raise ValueError("Problem has not been initialized. Call initialize() first")

        # Create fresh solver state from settings
        self._state = AlgorithmState.from_settings(self.settings)

        # Reset solution
        self._solution = None

        # Reset timing
        self.timing_solve = None
        self.timing_post = None

    def step(self) -> dict:
        """Perform a single SCP iteration.

        Designed for real-time plotting and interactive optimization. Performs one
        iteration including subproblem solve, state update, and progress emission.

        Note:
            This method is NOT idempotent - it mutates internal state and advances
            the iteration counter. Use reset() to return to initial conditions.

        Returns:
            dict: Contains "converged" (bool) and current iteration state

        Example:
            Call `.step()` manually in a loop to control the algorithm directly

                problem.initialize()
                while not problem.step()["converged"]:
                    plot_trajectory(problem.state.trajs[-1])
        """
        if self._state is None:
            raise ValueError("Problem has not been initialized. Call initialize() first")

        converged = self._algorithm.step(
            self._state,
            self._parameters,  # May change between steps
            self.settings,  # May change between steps
        )

        # Return dict matching original API
        return {
            "converged": converged,
            "scp_k": self._state.k,
            "scp_J_tr": self._state.J_tr,
            "scp_J_vb": self._state.J_vb,
            "scp_J_vc": self._state.J_vc,
        }

    def solve(
        self, max_iters: Optional[int] = None, continuous: bool = False
    ) -> OptimizationResults:
        """Run the SCP algorithm until convergence or iteration limit.

        Args:
            max_iters: Maximum iterations (default: settings.scp.k_max)
            continuous: If True, run all iterations regardless of convergence

        Returns:
            OptimizationResults with trajectory and convergence info
                (call post_process() for full propagation)
        """
        # Sync parameters before solving
        self._sync_parameters()

        required = [
            self._compiled_dynamics,
            self._compiled_constraints,
            self._solver,
            self._discretization_solver,
            self._state,
        ]
        if any(r is None for r in required):
            raise ValueError("Problem has not been initialized. Call initialize() before solve()")

        # Enable the profiler
        pr = profiling.profiling_start(self.settings.dev.profiling)

        t_0_while = time.time()
        # Print top header for solver results
        if self.settings.dev.printing:
            printing.header(self._columns)

        k_max = max_iters if max_iters is not None else self.settings.scp.k_max

        while self._state.k <= k_max:
            result = self.step()
            if result["converged"] and not continuous:
                break

        t_f_while = time.time()
        self.timing_solve = t_f_while - t_0_while

        # Wait for print queue to drain (only if thread is running)
        if self.print_thread is not None and self.print_thread.is_alive():
            while self.print_queue.qsize() > 0:
                time.sleep(0.1)

        # Print bottom footer for solver results
        if self.settings.dev.printing:
            printing.footer(self._columns)

        profiling.profiling_end(pr, "solve")

        # Store solution state
        self._solution = copy.deepcopy(self._state)

        return self._format_result(self._state, self._state.k <= k_max)

    def post_process(self) -> OptimizationResults:
        """Propagate solution through full nonlinear dynamics for high-fidelity trajectory.

        Integrates the converged SCP solution through the nonlinear dynamics to
        produce x_full, u_full, and t_full. Call after solve() for final results.

        Returns:
            OptimizationResults with propagated trajectory fields

        Raises:
            ValueError: If solve() has not been called yet.
        """
        if self._solution is None:
            raise ValueError("No solution available. Call solve() first.")

        # Enable the profiler
        pr = profiling.profiling_start(self.settings.dev.profiling)

        # Create result from stored solution state
        result = self._format_result(self._solution, self._solution.k <= self.settings.scp.k_max)

        t_0_post = time.time()
        result = propagate_trajectory_results(
            self._parameters,
            self.settings,
            result,
            self._propagation_solver,
            algebraic_prop=self._lowered.algebraic_prop,
        )
        t_f_post = time.time()

        self.timing_post = t_f_post - t_0_post

        # Store the propagated result back into _solution for plotting
        # Store as a cached attribute on the _solution object
        self._solution._propagated_result = result

        # Print results summary
        printing.print_results_summary(
            result, self.timing_post, self.timing_init, self.timing_solve
        )

        profiling.profiling_end(pr, "postprocess")
        return result

    def citation(self) -> str:
        """Return BibTeX citations for all components used in this problem.

        Aggregates citations from the algorithm and other components (discretization,
        convex solver, etc.) Each section is prefixed with a comment indicating which component the
        citation is for.

        Returns:
            Formatted string containing all BibTeX citations with comments.

        Example:
            Print all citations for a problem::

                problem = Problem(dynamics, constraints, states, controls, N, time)
                print(problem.citation())
        """
        sections = []

        sections.append(r"% --- AUTO-GENERATED CITATIONS FOR OPENSCVX CONFIGURATION ---")

        # Algorithm citations
        algo_citations = self._algorithm.citation()
        if algo_citations:
            algo_name = type(self._algorithm).__name__
            header = f"% Algorithm: {algo_name}"
            citations = "\n".join(algo_citations)
            sections.append(f"{header}\n\n{citations}")

        # Solver citations
        solver_citations = self._solver.citation()
        if solver_citations:
            solver_name = type(self._solver).__name__
            header = f"% Convex Solver: {solver_name}"
            citations = "\n".join(solver_citations)
            sections.append(f"{header}\n\n{citations}")

        # Future: add citations from discretization, constraint formulations, etc.

        sections.append(r"% --- END AUTO-GENERATED CITATIONS")

        return "\n\n".join(sections)
