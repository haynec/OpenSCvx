"""Core optimization problem interface for trajectory optimization.

This module provides the Problem class, the main entry point for defining
and solving trajectory optimization problems using Sequential Convex Programming (SCP).

Example:
    The prototypical flow is to define a problem, then initialize, solve, and post-process the
    results::

        problem = Problem(dynamics, constraints, states, controls, N, time)
        problem.initialize()
        result = problem.solve()
        result = problem.post_process()

    For batched solves over many initial conditions::

        results = problem.solve_batched(parameters={"initial_position": ic_batch})
        results = problem.post_process_batched(results)
"""

import copy
import os
import queue
import threading
import time
from dataclasses import fields as dc_fields
from dataclasses import replace as dc_replace
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import numpy as np

os.environ["EQX_ON_ERROR"] = "nan"

import jax.numpy as jnp

from openscvx.algorithms import (
    Algorithm,
    AlgorithmHistory,
    AlgorithmState,
    OptimizationResults,
    PenalizedTrustRegionConfig,
)
from openscvx.algorithms.loop import make_solve_loop
from openscvx.config import (
    Config,
    DevConfig,
    PropagationConfig,
    SimConfig,
)
from openscvx.discretization import (
    Discretizer,
    get_impulsive_discretization_solver,
    resolve_discretizer_config,
)
from openscvx.expert import ByofSpec
from openscvx.integrations.base import DynamicsAdapter, _merge_byof
from openscvx.lowered import LoweredProblem, ParameterDict
from openscvx.lowered.dynamics import Dynamics
from openscvx.lowered.jax_constraints import (
    LoweredCrossNodeConstraint,
    LoweredJaxConstraints,
    LoweredNodalConstraint,
)
from openscvx.propagation import get_propagation_solver, propagate_trajectory_results
from openscvx.solvers import ConvexSolver, resolve_solver_config
from openscvx.symbolic.builder import preprocess_symbolic_problem
from openscvx.symbolic.expr import CTCS, Constraint
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State
from openscvx.symbolic.expr.time import Time
from openscvx.symbolic.lower import lower_symbolic_problem
from openscvx.symbolic.problem import SymbolicProblem
from openscvx.utils import printing, profiling
from openscvx.utils.caching import (
    get_solve_batched_cache_path,
    get_solver_cache_paths,
    load_or_compile_discretization_solver,
    load_or_compile_propagation_solver,
    load_or_export_solve_batched,
    prime_propagation_solver,
)


def _resolve_batch_spec(
    entries: Dict[str, Tuple[Any, Tuple[int, ...]]],
    in_axes: Optional[Dict[str, Optional[int]]] = None,
    fill: Optional[set] = None,
) -> Tuple[int, Dict[str, Optional[int]], set]:
    """Decide which entries of a batched solve carry a batch axis.

    The one rule behind :meth:`Problem.solve_batched`: a value whose shape
    equals its declared (unbatched) shape is shared across the batch; a value
    with exactly one extra leading axis is batched along it. Rank against the
    declared shape decides — never the leading-axis value — so a shared entry
    whose leading axis happens to equal ``B`` is never misread as batched.

    Entries named in *fill* additionally accept two **fill** forms for array
    declared shapes: a scalar ``()`` (one value broadcast to the declared
    shape, shared) or a ``(B,)`` vector (one scalar per batch element, each
    broadcast to the declared shape). Exact shapes win when both parse — a
    rank-1 declared shape of length exactly ``B`` reads as shared-exact, and
    ``in_axes={name: 0}`` forces the per-element fill reading.

    Args:
        entries: ``{name: (value, declared_shape)}`` over every batchable
            input. ``None`` values are skipped (the caller substitutes
            defaults for them).
        in_axes: Optional ``{name: 0 | None}`` overriding inference for the
            named entries (``0`` = batched along the leading axis, ``None`` =
            shared); unlisted entries fall back to the rank rule.
        fill: Optional set of entry names that accept the scalar / ``(B,)``
            fill forms (the ``algorithm`` knobs).

    Returns:
        ``(B, axes, fills)`` — the common batch size, a ``{name: 0 | None}``
        dict over the non-``None`` entries ready to use as a ``jax.vmap``
        ``in_axes`` pytree, and the subset of names parsed as fills (their
        values still need broadcasting to the declared shape).

    Raises:
        ValueError: If a value matches none of its accepted shapes, if
            batched entries disagree on the leading axis, if no entry is
            batched, or if *in_axes* names an unknown or absent entry or uses
            an axis other than ``0`` / ``None``.
    """
    in_axes = in_axes or {}
    fill = fill or set()
    unknown = set(in_axes) - set(entries)
    if unknown:
        raise ValueError(
            f"solve_batched: in_axes names unknown entr{'y' if len(unknown) == 1 else 'ies'} "
            f"{sorted(unknown)}; batchable entries are {sorted(entries)}."
        )
    for name, ax in in_axes.items():
        if ax not in (0, None):
            raise ValueError(
                f"solve_batched: in_axes['{name}'] must be 0 (batched along the "
                f"leading axis) or None (shared); got {ax!r}."
            )
        if ax == 0 and entries[name][0] is None:
            raise ValueError(
                f"solve_batched: in_axes marks '{name}' as batched, but no value was passed for it."
            )

    def _shape_error(name, shape, declared):
        forms = f"pass {declared} to share it across the batch, or (B,) + {declared} to batch it"
        if name in fill:
            forms += (
                ", a scalar () to fill the shared value, or (B,) to fill one "
                "scalar per batch element"
            )
        return ValueError(
            f"solve_batched: '{name}' has shape {shape}, but its unbatched "
            f"shape is {declared}; {forms}."
        )

    axes: Dict[str, Optional[int]] = {}
    fills: set = set()
    batched: List[Tuple[str, int]] = []  # (name, leading axis)
    for name, (value, declared) in entries.items():
        if value is None:
            continue
        shape = tuple(jnp.shape(value))
        declared = tuple(declared)
        exact_batched = len(shape) == len(declared) + 1 and shape[1:] == declared
        if name in in_axes:
            axes[name] = in_axes[name]
            if in_axes[name] == 0:
                if not shape:
                    raise ValueError(
                        f"solve_batched: in_axes marks '{name}' as batched, but it "
                        f"is a scalar with no leading axis to batch over."
                    )
                if name in fill and not exact_batched:
                    if len(shape) != 1:
                        raise _shape_error(name, shape, declared)
                    fills.add(name)
                batched.append((name, shape[0]))
            elif name in fill and shape != declared:
                if shape != ():
                    raise _shape_error(name, shape, declared)
                fills.add(name)
        elif shape == declared:
            axes[name] = None
        elif exact_batched:
            axes[name] = 0
            batched.append((name, shape[0]))
        elif name in fill and shape == ():
            axes[name] = None
            fills.add(name)
        elif name in fill and len(shape) == 1:
            axes[name] = 0
            fills.add(name)
            batched.append((name, shape[0]))
        else:
            raise _shape_error(name, shape, declared)

    if not batched:
        raise ValueError(
            "solve_batched: every argument matches its unbatched shape, so there "
            "is no batch axis. Use solve_jax() for a single solve, or mark an "
            "entry as batched explicitly via in_axes (e.g. in_axes={'x_initial': 0})."
        )
    ref_name, B = batched[0]
    for name, b in batched[1:]:
        if b != B:
            raise ValueError(
                f"solve_batched: batch sizes disagree: '{ref_name}' has leading "
                f"axis {B} but '{name}' has leading axis {b}."
            )
    return B, axes, fills


def _flatten_in_axes(in_axes, entries: Dict[str, Tuple[Any, Tuple[int, ...]]]) -> dict:
    """Flatten the user-facing ``in_axes`` to the resolver's per-entry names.

    Mirrors ``jax.vmap``'s prefix semantics: a bare ``0`` batches every
    passed input, and ``0`` / ``None`` at ``"parameters"`` / ``"algorithm"``
    applies to every key passed in that dict; per-key dicts pick out
    individual entries. One deliberate divergence from ``vmap``: a bare
    ``None`` keeps its default meaning — infer by rank — because vmap's
    all-broadcast reading would leave a batched solve with no batch axis.

    Args:
        in_axes: ``None`` (infer), ``0`` (batch everything passed), or a dict
            keyed by argument name with leaves ``0`` / ``None`` — where
            ``"parameters"`` and ``"algorithm"`` take either a prefix
            (``0`` / ``None``) or a per-key dict.
        entries: The resolver's ``{name: (value, declared_shape)}`` map;
            supplies the passed-entry names a prefix expands over.

    Returns:
        ``{entry_name: 0 | None}`` for :func:`_resolve_batch_spec`.
    """
    if in_axes is None:
        return {}
    if in_axes == 0:
        return {name: 0 for name, (value, _) in entries.items() if value is not None}
    if not isinstance(in_axes, dict):
        raise ValueError(
            f"solve_batched: in_axes must be 0 (batch every input), a dict "
            f"keyed by argument name, or None (infer by rank); got {in_axes!r}."
        )
    flat = {}
    for name, ax in in_axes.items():
        if name in ("parameters", "algorithm"):
            if isinstance(ax, dict):
                flat.update({f"{name}.{key}": v for key, v in ax.items()})
            elif ax in (0, None):
                flat.update({entry: ax for entry in entries if entry.startswith(f"{name}.")})
            else:
                raise ValueError(
                    f"solve_batched: in_axes['{name}'] must be 0 or None (a "
                    f"prefix applied to every passed key) or a per-key dict, "
                    f"e.g. {{'gate_center': 0}}; got {ax!r}."
                )
        else:
            flat[name] = ax
    return flat


# Named solve kwargs and the AlgorithmState field each one targets. The
# ``algorithm`` dict reaches the same fields by name, so passing both for one
# field is ambiguous — the solve methods raise, pointing at the kwarg.
_OVERRIDE_KWARG_FIELDS = {
    "x_initial": "x_init_pin",
    "x_final": "x_term_pin",
    "x_guess": "x",
    "u_guess": "u",
    "max_iters": "k_max",
}

# Constructor-only keys of the ``algorithm`` config that select program
# structure or live outside the traced SCP loop: ``autotuner`` picks the
# weight-update rule, and ``t_max`` is a wall-clock budget only the
# interactive ``solve()`` honors. Neither can be a per-solve input, so
# ``algorithm={...}`` at solve time names them with a construction-time
# error rather than the generic unknown-name one.
_ALGORITHM_CONSTRUCTION_KEYS = ("autotuner", "t_max")


def _override_target(state: "AlgorithmState", name: str) -> jnp.ndarray:
    """The state leaf a (validated) override name targets.

    Override names are :class:`AlgorithmState` field names or fields of its
    autotuner-declared ``hyper`` container (the two never collide —
    ``AlgorithmState.from_settings`` rejects shadowing fields); the leaf
    supplies the declared shape and dtype the override value is checked and
    cast against.
    """
    if name in {fld.name for fld in dc_fields(state.hyper)}:
        return getattr(state.hyper, name)
    return getattr(state, name)


def _make_single_result(
    results: "OptimizationResults",
    b: int,
) -> "OptimizationResults":
    """Slice a single element ``b`` from a batched ``OptimizationResults``.

    Returns a minimal :class:`~openscvx.algorithms.OptimizationResults` with
    the trajectory arrays for element ``b`` at normal (non-batched) shape — ready
    to be passed to :func:`~openscvx.propagation.propagate_trajectory_results`.

    Only the fields consumed by propagation are populated: ``X``, ``U``,
    ``converged``, ``t_final``, ``_states``, ``_controls``.  History fields
    are left empty (they are not needed for post-processing).
    """
    x_b = np.asarray(results.X[-1])[b]  # (N, n_x)
    u_b = np.asarray(results.U[-1])[b]  # (N, n_u)
    t_final_b = float(np.asarray(results.t_final[b]).reshape(-1)[0])
    return OptimizationResults(
        converged=bool(np.asarray(results.converged).reshape(-1)[b]),
        t_final=t_final_b,
        X=[x_b],
        U=[u_b],
        _states=results._states,
        _controls=results._controls,
    )


class Problem:
    def __init__(
        self,
        dynamics: Union[dict, DynamicsAdapter],
        constraints: List[Union[Constraint, CTCS]],
        states: List[State],
        controls: List[Control],
        N: int,
        time: Time,
        *,
        dynamics_discrete: Optional[dict] = None,
        dynamics_prop: Optional[dict] = None,
        states_prop: Optional[List[State]] = None,
        algebraic_prop: Optional[dict] = None,
        licq_min: Union[float, Dict[int, float]] = 0.0,
        licq_max: Union[float, Dict[int, float]] = 1e-4,
        algorithm: Optional[Union[Algorithm, dict]] = None,
        discretizer: Optional[Union[Discretizer, dict]] = None,
        solver: Optional[Union[ConvexSolver, dict]] = None,
        byof: Optional[Union[ByofSpec, dict]] = None,
        float_dtype: str = "float32",
    ):
        """The primary class in charge of compiling and exporting the solvers.

        Args:
            dynamics: Dictionary mapping state names to their dynamics
                expressions. Each key should be a state name, and each value
                should be an ``Expr`` representing the derivative of that
                state. A ``DynamicsAdapter`` may also be passed here in
                place of the dict.
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
            licq_min: Minimum LICQ constraint value. Defaults to 0.0.
                Either a scalar (applied to all CTCS groups) or a dict
                mapping CTCS group ``idx`` to per-group bounds.
            licq_max: Maximum LICQ constraint value. Defaults to 1e-4.
                Either a scalar (applied to all CTCS groups) or a dict
                mapping CTCS group ``idx`` to per-group bounds.
            algorithm: SCP algorithm configuration. Accepts:

                - ``None`` — uses ``PenalizedTrustRegion()`` with defaults.
                - An ``Algorithm`` instance — used directly.
                - A ``dict`` — passed as kwargs to ``PenalizedTrustRegion()``.
                  Supports a nested ``autotuner`` key in any of these forms:

                  - **string** — class name with default parameters, e.g.
                    ``"RampProximalWeight"``.
                  - **dict** — class name via ``"type"`` key plus parameter
                    overrides, e.g.
                    ``{"type": "RampProximalWeight", "ramp_factor": 1.04}``.
                  - **instance** — an already-constructed autotuner object,
                    e.g. ``ox.RampProximalWeight(ramp_factor=1.04)``.

                The ``lam_cost`` key accepts either a float (applied
                uniformly to all minimize/maximize states) or a dict
                mapping state names to per-state weights::

                    # Uniform cost weight
                    algorithm={"lam_cost": 5e-1}

                    # Per-state cost weights
                    algorithm={"lam_cost": {"velocity": 1e-1, "time": 1e0}}

                    # Per-component weights for vector states
                    algorithm={"lam_cost": {"position": [0, 0, 1e-6], "fuel": 1e0}}

                When a dict is provided, every state that has a
                minimize/maximize objective must have an entry.  Dict
                values may be scalars (broadcast to all components) or
                arrays matching the state's shape.  States without
                objectives are automatically assigned weight 0.  The
                dict is expanded to an array of shape ``(n_states,)``
                during ``Problem`` construction.

                Examples::

                    # Just tweak weights (default algorithm & autotuner)
                    algorithm={"lam_cost": 5e-1, "k_max": 50}

                    # Autotuner by name (default parameters)
                    algorithm={"autotuner": "RampProximalWeight"}

                    # Autotuner as dict with overrides
                    algorithm={
                        "lam_cost": 5e-1,
                        "autotuner": {"type": "RampProximalWeight", "ramp_factor": 1.04},
                    }

                    # Autotuner as instance
                    algorithm={"autotuner": ox.RampProximalWeight(ramp_factor=1.04)}
            discretizer: Discretization method configuration. Accepts:

                - ``None`` — uses ``LinearizeDiscretize()`` with defaults
                  (FOH, Tsit5). Uses sparse Jacobians and compact variational
                  integration when sparsity patterns exist on dynamics; otherwise
                  falls back to dense ``jax.jacfwd`` (same numerics).
                - A ``Discretizer`` instance — used directly.
                - A ``dict`` — resolved via :func:`~openscvx.discretization._resolve_discretizer`
                  (default class is ``LinearizeDiscretize`` unless ``"type"`` is set).

                Examples::

                    # Per-control hold + ODE solver on the discretizer
                    #   thrust = ox.Control("thrust", shape=(3,), parameterization="ZOH")
                    discretizer={"ode_solver": "Dopri8"}

                    # Pass integrator kwargs (forwarded to Diffrax / diffeqsolve)
                    discretizer={
                        "diffrax_kwargs": {
                            "max_steps": 20_000,
                            "num_substeps": 100,
                        }
                    }

                    # Instance
                    discretizer=ox.LinearizeDiscretize(dis_type="ZOH", ode_solver="Dopri8")
            solver: Convex subproblem solver configuration. Accepts:

                - ``None`` — uses ``CVXPyPTRSolver()`` with defaults (QOCO backend).
                - A ``ConvexSolver`` instance — used directly.
                - A ``dict`` — validated as ``PTRSolverSpec``; the ``backend``
                  field (``"cvxpy"``, ``"qpax"``, or ``"moreau"``) selects the
                  concrete backend.

                Examples::

                    # Change CVXPY backend solver and tolerances
                    solver={"cvx_solver": "CLARABEL", "solver_args": {"tol_gap_abs": 1e-7}}

                    # Just change solver_args
                    solver={"solver_args": {"abstol": 1e-6, "reltol": 1e-9}}

                    # Enable cvxpygen code generation
                    solver={"cvxpygen": True}

                    # JAX-native QPAX backend (no cvx_solver / cvxpygen fields)
                    solver={"backend": "qpax"}

                    # JAX-native Moreau conic backend (SOC epigraphs, warm-start)
                    solver={"backend": "moreau"}

                    # Instance
                    solver=ox.CVXPyPTRSolver(cvx_solver="CLARABEL")
                    solver=ox.QPAXPTRSolver()
                    solver=ox.MoreauPTRSolver()
            byof (ByofSpec, optional): Expert mode only. Raw JAX functions to
                bypass symbolic layer. See :class:`openscvx.expert.ByofSpec` for
                detailed documentation.
            float_dtype (str): Default floating-point dtype for JAX lowering.
                Must be ``\"float32\"`` or ``\"float64\"``. This sets JAX's
                ``jax_enable_x64`` config flag (True for float64, False for float32),
                which controls the dtype used in all lowered JAX functions (including
                both branches of ``jax.lax.cond``) to avoid dtype-mismatch errors
                during integration.

        Note:
            There are two approaches for handling time:
            1. Auto-create (simple): Don't include "time" in states, provide Time object
            2. User-provided (for time-dependent constraints): Include "time" State in states and
               in dynamics dict, don't provide Time object
        """

        # Configure JAX's default dtype (float32 vs float64) via jax_enable_x64.
        # This must happen before lowering so that all JAX-based lowerers
        # (including conditionals) produce tensors with a consistent dtype.
        # jax_enable_x64=True means float64, jax_enable_x64=False means float32.
        enable_x64 = float_dtype.lower() in ("float64", "f64", "double")
        jax.config.update("jax_enable_x64", enable_x64)

        # Also set the dtype in the JAX lowerer module so it's available during lowering
        # This ensures conditionals use the correct dtype even if JAX config doesn't take effect
        from openscvx.symbolic.lowerers.jax.logic import set_default_float_dtype

        set_default_float_dtype(float_dtype)

        # Persist so integration tests (and callers) can re-sync process-wide JAX config
        # before initialize()/solve() after other examples have been imported.
        self._float_dtype: str = float_dtype

        # Symbolic Preprocessing & Augmentation
        # If `dynamics` is a DynamicsAdapter, expand it into the standard
        # (dynamics_dict, byof_dict) representation and merge into user byof.
        if isinstance(dynamics, DynamicsAdapter):
            dynamics, adapter_byof = dynamics.expand()
            byof = _merge_byof(byof, adapter_byof)

        # Resolve byof: dict → ByofSpec (validates keys and nested specs)
        if byof is not None:
            byof = ByofSpec.model_validate(byof)

        self.symbolic: SymbolicProblem = preprocess_symbolic_problem(
            dynamics=dynamics,
            dynamics_discrete=dynamics_discrete,
            constraints=constraints,
            states=states,
            controls=controls,
            N=N,
            time=time,
            licq_min=licq_min,
            licq_max=licq_max,
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

            validate_byof(byof, self.symbolic.states, n_x, n_u, N, self.symbolic.parameters)

        # Store byof for cache hashing
        self._byof = byof

        # Resolve algorithm: instance → use directly, dict/None → validate & build
        if isinstance(algorithm, Algorithm):
            self._algorithm = algorithm
        else:
            config = PenalizedTrustRegionConfig.model_validate(algorithm or {})
            self._algorithm = config.to_algorithm(
                states=self.symbolic.states, controls=self.symbolic.controls
            )

        # Resolve discretizer: instance → use directly, dict/None → validate & build
        if isinstance(discretizer, Discretizer):
            self._discretizer = discretizer
        else:
            spec = resolve_discretizer_config(discretizer or {})
            self._discretizer = spec.build()

        # Resolve solver: instance → use directly, dict/None → validate & build
        if isinstance(solver, ConvexSolver):
            self._solver = solver
        else:
            spec = resolve_solver_config(solver or {})
            self._solver = spec.build()

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
                n=N,
                n_states=self._lowered.x_unified.initial.shape[0],
                n_states_prop=self._lowered.x_prop_unified.initial.shape[0],
                ctcs_node_intervals=self.symbolic.node_intervals,
            ),
            dev=DevConfig(),
            prp=PropagationConfig(),
        )

        # Copy time grid setting from Time to sim config so the solver can
        # read it during constraint assembly.
        if isinstance(time, Time):
            self.settings.sim._uniform_time_grid = time.uniform_time_grid

        # Fused JAX-pure SCP iteration body (built in initialize()).
        self._iteration_fn: callable = None

        # Sibling iteration body built over plain-``jax.jit`` inner solvers
        # (never the ``jax.export`` wrappers ``save_compiled`` may produce),
        # so it stays ``vmap``-safe and can be folded into the single exported
        # ``solve_batched`` artifact. Equals ``_iteration_fn`` when
        # ``save_compiled=False``. Built in initialize().
        self._iteration_fn_jit_inner: callable = None

        # Closure cache for the ``lax.while_loop`` wrapper that backs
        # :meth:`solve_jax`. Built once per ``initialize()`` — convergence
        # thresholds and the iteration cap ride the state pytree, so they
        # never force a rebuild.
        self._solve_loop_fn: Optional[callable] = None

        # Closure cache for the batched ``vmap``'d loop that backs
        # :meth:`solve_batched`. Keyed on ``(B, param_axes)`` — batch size and
        # per-parameter batching are baked into the artifact, so changing
        # either rebuilds.
        self._solve_batched_fn: Optional[callable] = None
        self._solve_batched_key: Optional[tuple] = None

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
        self.timing_compile = None
        self.timing_solve = None
        self.timing_post = None
        self._profiling_session = None

        # Compiled dynamics (vmapped versions, set in initialize())
        self._compiled_dynamics_prop: Optional[Dynamics] = None

        # Compiled constraints (JIT-compiled versions, set in initialize())
        self._compiled_constraints: Optional[LoweredJaxConstraints] = None

        # Solver state (created fresh for each solve)
        self._state: Optional[AlgorithmState] = None
        self._history: Optional[AlgorithmHistory] = None

        # Final solution state (saved after successful solve)
        self._solution: Optional[AlgorithmState] = None
        self._solution_history: Optional[AlgorithmHistory] = None
        self._solution_parameters: Optional[Dict[str, np.ndarray]] = None

        # SCP algorithm (resolved from `algorithm` parameter above)

    @property
    def solver(self) -> ConvexSolver:
        """Access the convex subproblem solver instance.

        Backend-specific attributes (e.g. ``cvx_solver``, ``solver_args``,
        ``cvxpygen``, ``cvxpygen_override`` on :class:`CVXPyPTRSolver`) can
        be modified freely before ``initialize`` is called::

            problem.solver.solver_args = {"abstol": 1e-6, "reltol": 1e-9}
            problem.solver.cvxpygen = True
            problem.initialize()

        !!! warning
            Solver settings are compiled into the solve function during
            ``initialize()``.  Changes made **after** ``initialize()``
            will have no effect on subsequent solves.

        Returns:
            The solver instance — a concrete :class:`PTRSolver` subclass.
        """
        return self._solver

    @property
    def algorithm(self) -> Algorithm:
        """Access the SCP algorithm instance.

        Returns:
            The algorithm instance (e.g., PenalizedTrustRegion).
        """
        return self._algorithm

    @property
    def discretizer(self) -> Discretizer:
        """Access the discretizer instance.

        Attributes such as `dis_type`, `ode_solver`, and `diffrax_kwargs`
        can be modified freely before `initialize` is called:

            problem.discretizer.dis_type = "ZOH"
            problem.discretizer.ode_solver = "Dopri8"
            problem.discretizer.diffrax_kwargs = {"num_substeps": 100}
            problem.initialize()

        !!! warning
            Discretizer settings are compiled into the JIT-cached solver
            during `initialize`.  Changes made **after**
            `initialize()` will have no effect on subsequent solves.

        Returns:
            The discretizer instance (e.g., ``LinearizeDiscretize``).
        """
        return self._discretizer

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
        if self._lowered is None:
            return

        if self._lowered.cvxpy_params is not None:
            for name, value in self._parameter_wrapper.items():
                if name in self._lowered.cvxpy_params:
                    self._lowered.cvxpy_params[name].value = value

    def _sync_boundary_conditions(self):
        """Sync boundary conditions from State objects to lowered representation.

        This method reads the current `.initial` and `.final` values and types
        from the original State objects and updates both the unified state
        representation and the CVXPy solver parameters. This enables workflows
        where initial conditions change between solves.

        Note:
            Safe to call before initialize() - it will simply do nothing.
        """
        if self._lowered is None:
            return

        # Sync initial/final values and types from State objects to optimization unified
        # representation
        for state in self.symbolic.states:
            if state.initial is not None:
                self._lowered.x_unified.initial[state._slice] = state.initial
                self._lowered.x_unified.initial_type[state._slice] = state.initial_type
            if state.final is not None:
                self._lowered.x_unified.final[state._slice] = state.final
                self._lowered.x_unified.final_type[state._slice] = state.final_type

        # Sync initial/final values and types to propagation unified representation
        # (states_prop includes optimization states, so we sync those too)
        for state in self.symbolic.states_prop:
            if state.initial is not None:
                self._lowered.x_prop_unified.initial[state._slice] = state.initial
                self._lowered.x_prop_unified.initial_type[state._slice] = state.initial_type
            if state.final is not None:
                self._lowered.x_prop_unified.final[state._slice] = state.final
                self._lowered.x_prop_unified.final_type[state._slice] = state.final_type

        # Push to the solver — both backends short-circuit on a pre-initialize
        # call, so this is safe to invoke from any lifecycle point.
        self._solver.update_boundary_conditions(
            x_init=self._lowered.x_unified.initial,
            x_term=self._lowered.x_unified.final,
        )

    def _sync_scp_constants(self):
        """Sync the SCP loop constants from the algorithm config onto the state.

        ``ep_tr`` / ``ep_vb`` / ``ep_vc`` / ``k_max`` and the autotuner's
        declared hyperparameters (``hyper``) are snapshotted onto
        :class:`AlgorithmState` when it is created, so a mutation like
        ``problem.algorithm.ep_tr = 1e-6`` between ``solve()`` calls would
        otherwise go stale. Re-read them here, alongside
        :meth:`_sync_parameters` / :meth:`_sync_boundary_conditions`.

        Note:
            Safe to call before initialize() - it will simply do nothing.
        """
        if self._state is None:
            return

        algo = self._algorithm
        state = self._state
        # Commit each leaf to the device exactly as ``from_settings`` does:
        # ``iteration_fn``'s jit cache keys on committed sharding, so an
        # uncommitted leaf here would trigger a recompile on the next step.
        device = jax.devices()[0]

        def put(value, like):
            return jax.device_put(jnp.asarray(value, dtype=like.dtype), device)

        hyper = algo.autotuner.hyper
        self._state = state.replace(
            ep_tr=put(algo.ep_tr, state.ep_tr),
            ep_vb=put(algo.ep_vb, state.ep_vb),
            ep_vc=put(algo.ep_vc, state.ep_vc),
            k_max=put(algo.k_max, state.k_max),
            hyper=dc_replace(
                state.hyper,
                **{
                    fld.name: put(getattr(hyper, fld.name), getattr(state.hyper, fld.name))
                    for fld in dc_fields(state.hyper)
                },
            ),
        )

    def _sync_guesses(self):
        """Sync trajectory guesses from State/Control objects to lowered representation.

        This method reads the current `.guess` values from the original State and
        Control objects and updates the unified representations. This enables warm-starting
        workflows where the initial trajectory guess is updated between solves (e.g., shifting
        the previous solution).

        Note:
            This only updates the unified representation. The AlgorithmState is
            created from these values in reset() or initialize(), so this must
            be called before those methods to take effect.
        """
        if self._lowered is None:
            return

        # Sync optimization state guesses
        for state in self.symbolic.states:
            if state.guess is not None:
                self._lowered.x_unified.guess[:, state._slice] = state.guess

        # Sync propagation state guesses (includes optimization states)
        for state in self.symbolic.states_prop:
            if state.guess is not None:
                self._lowered.x_prop_unified.guess[:, state._slice] = state.guess

        # Sync control guesses
        for control in self.symbolic.controls:
            if control.guess is not None:
                self._lowered.u_unified.guess[:, control._slice] = control.guess

    def sync(self):
        """Sync parameters and boundary conditions to the solver.

        Call this after modifying State.initial/final or parameters when using
        step() without reset(). This allows warm-starting from the previous
        solution while updating problem data.

        Note:
            This is automatically called by solve() and reset(). Only needed
            when using step() directly with modified parameters or boundary
            conditions between iterations.

        Example:
            MPC with warm-starting::

                problem.initialize()
                while running:
                    # Update initial condition from measurement
                    pos.initial = measured_state
                    problem.sync()  # Sync without resetting algorithm state

                    # Continue from previous solution (warm-start)
                    for _ in range(max_iters):
                        if problem.step()["converged"]:
                            break
        """
        self._sync_parameters()
        self._sync_boundary_conditions()

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
    def history(self) -> Optional[AlgorithmHistory]:
        """Access the CPU-side iteration history (trajectories, weights, costs).

        Companion to :py:attr:`state` — :class:`AlgorithmState` carries the
        current-iterate pytree, while :class:`AlgorithmHistory` carries the
        append-only per-iteration log.
        """
        return self._history

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

    def _format_result(
        self,
        state: AlgorithmState,
        history: AlgorithmHistory,
        converged: bool,
    ) -> OptimizationResults:
        """Format the current iterate + history as :class:`OptimizationResults`.

        Thin wrapper around :meth:`OptimizationResults.from_history` — kept
        as a method so ``solve()`` / ``post_process()`` call sites read
        naturally.
        """
        return OptimizationResults.from_history(history, state, problem=self, converged=converged)

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

        # Create a new profiling session (shared across initialize/solve/post_process)
        self._profiling_session = (
            profiling._create_session() if self.settings.dev.profiling else None
        )
        pr = profiling.profiling_start(self.settings.dev.profiling, self._profiling_session)

        t_0_while = time.time()
        # Ensure scaling matrices are correct
        self.settings.sim.__post_init__()

        # Create compiled (vmapped) propagation dynamics
        # This preserves the original un-vmapped versions in _lowered
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

        # Generate discretization solvers via the discretizer (handles Jacobians
        # + vmapping). These are locals: they're captured by the fused
        # ``iteration_fn`` closure below, not stored on the problem.
        discretization_solver = self._discretizer.get_solver(self._lowered.dynamics, self.settings)
        discretization_solver_impulsive = get_impulsive_discretization_solver(
            self._lowered.dynamics_discrete
        )
        # Keep the raw (un-jit, un-exported) solvers: under ``save_compiled``
        # the versions below become ``jax.export`` wrappers, which have no
        # ``vmap`` rule and so cannot back :meth:`solve_batched`'s internal
        # ``vmap``. The jit-inner iteration body (built after ``_iteration_fn``)
        # wraps these in plain ``jax.jit`` instead.
        discretization_solver_raw = discretization_solver
        discretization_solver_impulsive_raw = discretization_solver_impulsive
        self._propagation_solver = get_propagation_solver(
            self._compiled_dynamics_prop.f, self.settings, self._discretizer
        )

        # Build convex subproblem (solver was created in __init__, variables in lower)
        self._solver.initialize(self._lowered, self.settings)

        # Print problem summary (after solver is initialized so we can access problem stats)
        printing.print_problem_summary(
            self.settings,
            self._lowered,
            self._solver,
            self._algorithm,
            self._discretizer,
        )

        # Get cache file paths using symbolic AST hashing
        # This is more stable than hashing lowered JAX code
        dis_solver_file, prop_solver_file = get_solver_cache_paths(
            self.symbolic,
            dt=self.settings.prp.dt,
            total_time=self.settings.sim.total_time,
            byof=self._byof,
        )

        # Compile the discretization solver (``jax.jit`` for the no-disk path,
        # ``jax.export`` when ``save_compiled``). Local — captured by iteration_fn.
        discretization_solver = load_or_compile_discretization_solver(
            discretization_solver,
            dis_solver_file,
            self._parameters,  # Plain dict for JAX
            self.settings.sim.n,
            self.settings.sim.n_states,
            self.settings.sim.n_controls,
            save_compiled=self.settings.sim.save_compiled,
            debug=self.settings.dev.debug,
        )

        # Compile the impulsive/discrete discretization solver with the same pipeline.
        # This solver is evaluated on node-wise inputs (x_nodes, u_nodes), shape (N, ...).
        dis_imp_solver_file = dis_solver_file.with_name(
            f"{dis_solver_file.stem}_impulsive{dis_solver_file.suffix}"
        )
        discretization_solver_impulsive = load_or_compile_discretization_solver(
            discretization_solver_impulsive,
            dis_imp_solver_file,
            self._parameters,  # Plain dict for JAX
            self.settings.sim.n,
            self.settings.sim.n_states,
            self.settings.sim.n_controls,
            save_compiled=self.settings.sim.save_compiled,
            debug=self.settings.dev.debug,
            name="discrete",
        )

        # Setup propagation solver parameters
        dtau = 1.0 / (self.settings.sim.n - 1)
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
            debug=self.settings.dev.debug,
        )

        # Build per-constraint lam_vb arrays from symbolic constraints.
        # Deferred to initialize() so that user-set lam_vb values
        # (assigned after Problem construction) are picked up.
        n_byof_nodal = len(self._byof.nodal_constraints) if self._byof else 0
        n_byof_cross = len(self._byof.cross_nodal_constraints) if self._byof else 0
        self._algorithm.weights.build_vb_arrays(
            N=self.symbolic.N,
            nodal_constraints=self.symbolic.constraints.nodal,
            cross_node_constraints=self.symbolic.constraints.cross_node,
            n_byof_nodal=n_byof_nodal,
            n_byof_cross=n_byof_cross,
        )

        # Build the fused JAX-pure SCP iteration body: it discretizes, linearizes
        # the constraints, solves the convex subproblem (via the backend's
        # iteration_callback), and folds the result through the autotuner — one
        # JIT'd call per SCP step in place of the old NumPy↔JAX stitching.
        print("Initializing the SCvx Subproblem Solver...")
        # Drop any prior ``solve_jax`` / ``solve_batched`` closures — they
        # capture the previous ``iteration_fn`` and would re-use stale traces
        # if ``initialize()`` is invoked twice.
        self._solve_loop_fn = None
        self._solve_batched_fn = None
        self._solve_batched_key = None
        self._solve_batched_is_exported = False
        self._iteration_fn = jax.jit(
            self._algorithm.build_iteration(
                dis_continuous=discretization_solver,
                dis_impulsive=discretization_solver_impulsive,
                jax_constraints=self._compiled_constraints,
                solver_callback=self._solver.iteration_callback(),
                settings=self.settings,
            )
        )
        # Sibling body for :meth:`solve_batched`'s internal ``vmap``. Under
        # ``save_compiled`` the inner solvers above are ``jax.export`` wrappers
        # that can't be vmapped, so rebuild over plain-``jax.jit`` solvers; the
        # whole vmapped loop is exported as one artifact instead. When
        # ``save_compiled=False`` the inner solvers are already jit, so reuse
        # ``_iteration_fn`` rather than recompile the dynamics (§3, Q3).
        if self.settings.sim.save_compiled and not self.settings.dev.debug:
            self._iteration_fn_jit_inner = jax.jit(
                self._algorithm.build_iteration(
                    dis_continuous=jax.jit(discretization_solver_raw),
                    dis_impulsive=jax.jit(discretization_solver_impulsive_raw),
                    jax_constraints=self._compiled_constraints,
                    solver_callback=self._solver.iteration_callback(),
                    settings=self.settings,
                )
            )
        else:
            self._iteration_fn_jit_inner = self._iteration_fn
        self._algorithm.initialize(self._iteration_fn, self.emitter_function)

        # Warm the JIT cache so the first ``.solve()`` doesn't pay the compile
        # cost. A single concrete call populates ``jax.jit``'s shape-keyed cache;
        # subsequent bare calls hit it (and a future vmap retraces to batched
        # shapes the first time, same as if no warmup had happened).
        # The ``_solve_loop_fn`` that backs :meth:`solve_jax` warms lazily on
        # first call (see :meth:`_get_or_build_solve_loop`) — pre-warming it
        # here would tax ``.solve()``-only users with ~1-3s of XLA compile work
        # they never benefit from.
        warmup_state = self._default_state()
        jax.block_until_ready(self._iteration_fn(warmup_state, self._parameters))
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

        # Create fresh solver state + history pair.
        self._state = self._default_state()
        self._history = AlgorithmHistory.from_settings(self.settings)

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

        This method automatically syncs:
            - Trajectory guesses from State/Control `.guess` attributes
            - Boundary conditions from State `.initial` and `.final` attributes

        Raises:
            ValueError: If initialize() has not been called yet.

        Example:
            After calling `.step()` it may be necessary to reset the problem back to the initial
            conditions

                problem.initialize()
                result1 = problem.step()
                problem.reset()
                result2 = problem.solve()  # Fresh run with same setup

            MPC with warm-starting from previous solution::

                for measured_state in measurements:
                    # Update initial condition
                    pos.initial = measured_state[:3]

                    # Warm-start: shift previous solution as new guess
                    pos.guess = np.roll(prev_result.nodes["pos"], -1, axis=0)

                    problem.reset()  # Syncs guesses and boundary conditions
                    result = problem.solve()
        """
        if self._compiled_dynamics_prop is None:
            raise ValueError("Problem has not been initialized. Call initialize() first")

        # Sync guesses from State/Control objects (must happen before AlgorithmState creation)
        self._sync_guesses()

        # Sync boundary conditions from State objects
        self._sync_boundary_conditions()

        # Create fresh solver state + history from settings
        self._state = self._default_state()
        self._history = AlgorithmHistory.from_settings(self.settings)

        # Reset solution
        self._solution = None
        self._solution_history = None
        self._solution_parameters = None

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

        self._state, converged = self._algorithm.step(
            self._state,
            self._history,
            self._parameters,  # May change between steps
            self.settings,  # May change between steps
        )

        # ``state.k`` is bumped to ``k + 1`` inside ``step()``; the J scalars on
        # the returned state are the metrics for the iteration just completed.
        return {
            "converged": converged,
            "scp_k": int(self._state.k),
            "scp_J_tr": float(self._state.J_tr),
            "scp_J_vb": float(self._state.J_vb),
            "scp_J_vc": float(self._state.J_vc),
        }

    def solve(
        self,
        max_iters: Optional[int] = None,
        time_limit: Optional[float] = None,
        continuous: bool = False,
    ) -> OptimizationResults:
        """Run the SCP algorithm until convergence or iteration limit.

        Drives the fused ``iteration_fn`` body from a Python ``while``
        loop — real-time prints, wall-clock ``time_limit``, ``continuous``
        mode, and the populated per-iteration history live here. For
        batched solves, ``jax.jit`` compilation across calls, or
        ``jax.grad`` through the solver, use :meth:`solve_jax`.

        Args:
            max_iters: Maximum iterations (default: algorithm.k_max)
            time_limit: Wall-clock time limit in seconds. Overrides
                ``algorithm.t_max`` when provided. ``None`` (default) falls
                back to ``algorithm.t_max``.
            continuous: If True, run all iterations regardless of convergence

        Returns:
            OptimizationResults with trajectory and convergence info
                (call post_process() for full propagation). The parameter
                values used for the solve are recorded on
                ``results.parameters``; :meth:`post_process` propagates
                with them.
        """
        # Sync parameters, boundary conditions, and SCP constants before solving
        self._sync_parameters()
        self._sync_boundary_conditions()
        self._sync_scp_constants()

        required = [
            self._compiled_dynamics_prop,
            self._compiled_constraints,
            self._solver,
            self._iteration_fn,
            self._state,
        ]
        if any(r is None for r in required):
            raise ValueError("Problem has not been initialized. Call initialize() before solve()")

        # Enable the profiler (reuse session from initialize)
        pr = profiling.profiling_start(self.settings.dev.profiling, self._profiling_session)

        t_0_while = time.time()
        # Print top header for solver results
        if self.settings.dev.printing:
            printing.header(self._columns)

        k_max = max_iters if max_iters is not None else self._algorithm.k_max
        t_max = time_limit if time_limit is not None else self._algorithm.t_max

        # ``state.k`` is the iteration counter (bumped inside each ``step()``).
        # Plan 2 replaces this Python ``while`` with a single ``lax.while_loop``
        # over ``make_solve_loop``; here it still drives ``iteration_fn`` step by
        # step so the interactive / printing path is unchanged.
        while int(self._state.k) <= k_max:
            result = self.step()
            if result["converged"] and not continuous:
                break
            if t_max is not None and (time.time() - t_0_while) >= t_max:
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

        # Snapshot state, history, and parameters for post_process() / plotting
        # reuse.
        self._solution = self._state
        self._solution_history = copy.deepcopy(self._history)
        self._solution_parameters = {
            name: np.array(value) for name, value in self._parameters.items()
        }

        timed_out = t_max is not None and self.timing_solve >= t_max
        result = self._format_result(
            self._state,
            self._history,
            int(self._state.k) <= k_max and not timed_out,
        )
        result.parameters = self._solution_parameters
        return result

    def _default_state(self) -> AlgorithmState:
        """Fresh :class:`AlgorithmState` from settings and the algorithm's SCP constants.

        The single assembly point for :meth:`AlgorithmState.from_settings`'
        keyword constants: convergence thresholds and the iteration cap come
        off the algorithm, the declared hyperparameters off its autotuner's
        ``hyper`` container.
        """
        algo = self._algorithm
        return AlgorithmState.from_settings(
            self.settings,
            algo.weights,
            ep_tr=algo.ep_tr,
            ep_vb=algo.ep_vb,
            ep_vc=algo.ep_vc,
            k_max=algo.k_max,
            hyper=algo.autotuner.hyper,
        )

    def _validate_overrides(self, where: str, algorithm: dict, kwargs: Dict[str, Any]) -> None:
        """Reject unknown ``algorithm`` keys and entries that shadow a named kwarg.

        The set of valid names is *derived*, never listed: every
        :class:`AlgorithmState` field (``_FIELDS``) plus every field of the
        autotuner's declared ``hyper`` container — the same names the user
        wrote in the ``Problem`` constructor's ``algorithm`` dict. Adding a
        state field — or declaring an autotuner knob — is thereby the
        registration. Construction-only keys of that dict
        (:data:`_ALGORITHM_CONSTRUCTION_KEYS`) get a teaching error of their
        own. *kwargs* maps each named solve kwarg to the value the caller
        received; an entry targeting the same field as a kwarg that was
        actually passed is ambiguous and raises.
        """
        fields = [name for name in AlgorithmState._FIELDS if name != "hyper"]
        hyper_keys = sorted(fld.name for fld in dc_fields(self._algorithm.autotuner.hyper))
        for name in algorithm:
            if name in _ALGORITHM_CONSTRUCTION_KEYS:
                raise ValueError(
                    f"{where}: algorithm['{name}'] is a construction-time setting, "
                    f"not a per-solve input; rebuild the Problem with "
                    f"Problem(..., algorithm={{'{name}': ...}}) to change it."
                )
        unknown = sorted(set(algorithm) - set(fields) - set(hyper_keys))
        if unknown:
            raise ValueError(
                f"{where}: unknown algorithm key(s) {unknown}; overridable names "
                f"are the AlgorithmState fields {sorted(fields)} and the "
                f"autotuner's declared hyperparameters {hyper_keys}."
            )
        for kwarg, field in _OVERRIDE_KWARG_FIELDS.items():
            if field in algorithm and kwargs.get(kwarg) is not None:
                raise ValueError(
                    f"{where}: algorithm['{field}'] and the {kwarg} kwarg target the "
                    f"same state field; pass the {kwarg} kwarg only."
                )

    def _resolve_initial_state(
        self,
        x_initial: Optional[jnp.ndarray],
        x_final: Optional[jnp.ndarray],
        x_guess: Optional[jnp.ndarray] = None,
        u_guess: Optional[jnp.ndarray] = None,
        max_iters: Optional[jnp.ndarray] = None,
        algorithm: Optional[dict] = None,
    ) -> AlgorithmState:
        """Build a fresh :class:`AlgorithmState` with per-solve overrides applied.

        This is the single place per-solve inputs land on the state pytree:
        boundary pins (``x_init_pin`` / ``x_term_pin``), trajectory
        warm-starts (``x`` / ``u``), ``max_iters`` → ``k_max``, and the
        validated *algorithm* knob dict (field names; values broadcast to
        the field shape, so a scalar fills an array-shaped field). Because
        every override is a pytree field, ``jax.vmap`` over any subset gives
        each batch slot its own boundary condition, seed iterate, and
        hyperparameters without recompilation.

        ``None`` falls back to the defaults from :meth:`_default_state`. A
        user-supplied pin replaces the corresponding field in full; pass
        ``jnp.nan`` at non-pinned entries to match the convention
        :meth:`AlgorithmState.from_settings` uses (the subproblem only
        consumes the pin where the boundary type is ``"Fix"``).
        """
        state = self._default_state()
        # Named kwargs first, algorithm knobs on top: ``solve_batched`` always
        # materializes the kwarg defaults (broadcast stacks), and an override
        # must beat a default. A genuine user-passed duplicate never reaches
        # here — ``_validate_overrides`` rejects it.
        updates = {}
        for kwarg, value in (
            ("x_initial", x_initial),
            ("x_final", x_final),
            ("x_guess", x_guess),
            ("u_guess", u_guess),
            ("max_iters", max_iters),
        ):
            if value is not None:
                updates[_OVERRIDE_KWARG_FIELDS[kwarg]] = value
        updates.update(algorithm or {})

        def cast(name, value):
            target = _override_target(state, name)
            return jnp.broadcast_to(jnp.asarray(value, dtype=target.dtype), target.shape)

        updates = {name: cast(name, value) for name, value in updates.items()}
        hyper_names = {fld.name for fld in dc_fields(state.hyper)}
        hyper = {name: updates.pop(name) for name in list(updates) if name in hyper_names}
        if hyper:
            updates["hyper"] = dc_replace(state.hyper, **hyper)
        return state.replace(**updates)

    def _get_or_build_solve_loop(self) -> callable:
        """Return the cached ``jax.jit``'d ``lax.while_loop`` wrapper.

        Built once per ``initialize()`` — the convergence thresholds and the
        iteration cap are :class:`AlgorithmState` fields, i.e. runtime inputs,
        so no per-solve setting forces a rebuild. The closure is wrapped
        in :func:`jax.jit`, so the first :meth:`solve_jax` call pays the XLA
        compile cost (~1-3s for brachistochrone-sized problems) and
        subsequent calls hit ``jax.jit``'s shape-keyed cache. Users with a
        latency-sensitive first call (MPC inner loops) can prime the cache
        by running a throwaway ``problem.solve_jax()`` before their timed
        window. AOT via ``.lower().compile()`` is not used: it returns a
        ``Compiled`` XLA executable that isn't ``vmap``-traceable, whereas a
        real-call warmup populates ``jax.jit``'s standard cache which vmap
        retraces from on first use.
        """
        if self._solve_loop_fn is None:
            self._solve_loop_fn = jax.jit(
                make_solve_loop(self._iteration_fn, self._algorithm.converged)
            )
        return self._solve_loop_fn

    def _get_or_build_solve_batched(
        self,
        B: int,
        params: dict,
        param_axes: Dict[str, Optional[int]],
    ) -> callable:
        """Return the cached ``jax.jit``'d batched ``lax.while_loop`` wrapper.

        Mirrors :meth:`_get_or_build_solve_loop` but the loop body is
        ``jax.vmap``'d over the leading batch axis of the ``AlgorithmState``
        and over each parameter whose entry in *param_axes* is ``0``;
        parameters at ``None`` are shared across the batch. A single XLA
        program runs all ``B`` SCP solves.
        It is built over :attr:`_iteration_fn_jit_inner` — the body whose inner
        discretization solvers are plain ``jax.jit``, hence ``vmap``-safe — not
        :attr:`_iteration_fn`, which under ``save_compiled`` wraps
        ``call_exported`` solvers that have no ``vmap`` rule (§3). The closure
        is cached on ``(B, param_axes)``: batch size and per-parameter
        batching are baked into the program, so changing either retraces.
        Tolerances and the iteration cap ride the state pytree, so they never
        retrace or re-export.

        Under ``save_compiled`` the whole vmapped loop is exported once to the
        solver cache and deserialized on later processes (the single artifact
        that makes ``solve_batched`` worth having); the returned object is then
        a ``jax.export`` wrapper called via ``.call``. Export requires an
        exportable backend — CVXPy raises a teaching error here rather than
        silently degrading to an in-process solve.
        """
        key = (B, tuple(sorted(param_axes.items())))
        if self._solve_batched_fn is None or self._solve_batched_key != key:
            loop = make_solve_loop(self._iteration_fn_jit_inner, self._algorithm.converged)
            batched = jax.vmap(loop, in_axes=(0, param_axes))
            if self.settings.sim.save_compiled and not self.settings.dev.debug:
                if not self._solver.exportable:
                    raise ValueError(
                        f"solve_batched(save_compiled=True) cannot export the "
                        f"{type(self._solver).__name__} backend: its solve runs through a "
                        f"jax.pure_callback, which jax.export cannot serialize. Use an "
                        f"exportable backend (QPAXPTRSolver or MoreauPTRSolver), or set "
                        f"save_compiled=False to run B sequential in-process solves."
                    )
                cache_file = get_solve_batched_cache_path(
                    self.symbolic,
                    self.settings,
                    self._algorithm,
                    self._solver,
                    self._discretizer,
                    B,
                    param_axes,
                )
                sample_state = jax.tree_util.tree_map(
                    lambda a: jnp.broadcast_to(a, (B,) + jnp.shape(a)),
                    self._default_state(),
                )
                # Trace the export against the same ``params`` the call will use,
                # not ``self._parameters`` — under an export the input avals are
                # frozen at trace time, so the two must agree.
                batched = load_or_export_solve_batched(batched, cache_file, sample_state, params)
                self._solve_batched_is_exported = True
            else:
                # AOT-compile now so solve_batched() execution is compile-free.
                # sample_state provides concrete shapes/dtypes; the compiled
                # artifact accepts any inputs with those same shapes/dtypes.
                sample_state = jax.tree_util.tree_map(
                    lambda a: jnp.broadcast_to(a, (B,) + jnp.shape(a)),
                    self._default_state(),
                )
                t_0_compile = time.time()
                batched = jax.jit(batched).lower(sample_state, params).compile()
                self.timing_compile = time.time() - t_0_compile
                self._solve_batched_is_exported = False
            self._solve_batched_fn = batched
            self._solve_batched_key = key
        return self._solve_batched_fn

    def solve_jax(
        self,
        x_initial: Optional[jnp.ndarray] = None,
        x_final: Optional[jnp.ndarray] = None,
        parameters: Optional[dict] = None,
        *,
        x_guess: Optional[jnp.ndarray] = None,
        u_guess: Optional[jnp.ndarray] = None,
        max_iters: Optional[int] = None,
        algorithm: Optional[dict] = None,
    ) -> OptimizationResults:
        """Run the SCP algorithm — JAX-pure.

        Composes with ``jax.vmap`` and ``jax.jit``. Drives the fused
        ``iteration_fn`` inside a ``lax.while_loop`` rather than the Python
        ``while`` loop that :meth:`solve` uses. Returns an
        :class:`OptimizationResults` pytree built by
        :meth:`OptimizationResults.from_final_state` — per-iteration history
        (``X``/``U`` lists past the final iterate, ``*_history`` fields,
        ``discretization_history``) is empty under this path, because list
        growth doesn't fit inside ``lax.while_loop``. For prints, wall-clock
        ``time_limit``, ``continuous`` mode, or populated history, use
        :meth:`solve`.

        With the CVXPy backend, ``jax.vmap(problem.solve_jax)`` runs ``B``
        sequential CVXPy solves (host CVXPy is not thread-safe); the QPAX
        and Moreau backends run in parallel under vmap.

        Args:
            x_initial: Initial-state boundary pin, shape ``(n_states,)``.
                Replaces ``state.x_init_pin``; pass ``jnp.nan`` at non-Fix
                entries to match :meth:`AlgorithmState.from_settings`'
                convention. ``None`` uses the default from settings.
            x_final: Terminal-state boundary pin, shape ``(n_states,)``.
                Same conventions as ``x_initial``.
            parameters: Problem parameters dict for the current solve. Use
                this rather than mutating ``self.parameters`` if the
                parameters are traced. ``None`` reuses ``self._parameters``.
                Unlike :meth:`solve` / :meth:`solve_batched`, the merged
                values are *not* recorded on the returned results — under a
                caller's ``jit`` / ``vmap`` they are tracers, which must not
                leak into the pytree's aux — so callers that post-process
                manage parameter bookkeeping themselves.
            x_guess: Initial state trajectory guess, shape ``(N, n_states)``.
                Replaces ``state.x``; ``None`` uses the default from
                settings (``State.guess`` at ``initialize()``).
            u_guess: Initial control trajectory guess, shape ``(N, n_controls)``.
                Replaces ``state.u``; ``None`` uses the default from settings.
            max_iters: SCP iteration cap (scalar). Sugar for
                ``algorithm={"k_max": ...}``; passing both raises. ``None``
                uses the configured ``k_max``. A runtime input on the state
                pytree — no value forces a retrace.
            algorithm: Per-solve algorithm knobs, by the same names as the
                ``Problem`` constructor's ``algorithm`` dict — convergence
                thresholds (``ep_tr`` / ``ep_vb`` / ``ep_vc``), weights
                (``lam_prox`` / ``lam_vc`` / ``lam_cost``), autotuner
                hyperparameters, and any other :class:`AlgorithmState`
                field; the ``AlgorithmState`` attribute docs are the
                reference. Each value either matches the field's shape or
                is a scalar (broadcast to it). All are runtime inputs: no
                value forces a retrace. Fields with a dedicated kwarg
                (``x_guess`` → ``x``, ``x_initial`` → ``x_init_pin``, ...)
                raise here when both are passed; construction-only keys
                (``autotuner``, ``t_max``) raise pointing at the
                constructor.

        Returns:
            :class:`OptimizationResults` pytree with the final iterate and
            empty histories.
        """
        if self._iteration_fn is None:
            raise ValueError(
                "Problem has not been initialized. Call initialize() before solve_jax()"
            )

        params = parameters if parameters is not None else self._parameters

        algorithm = dict(algorithm or {})
        self._validate_overrides(
            "solve_jax",
            algorithm,
            {
                "x_initial": x_initial,
                "x_final": x_final,
                "x_guess": x_guess,
                "u_guess": u_guess,
                "max_iters": max_iters,
            },
        )
        default_state = self._default_state()
        for name, value in algorithm.items():
            declared = tuple(jnp.shape(_override_target(default_state, name)))
            shape = tuple(jnp.shape(value))
            if shape not in (declared, ()):
                raise ValueError(
                    f"solve_jax: algorithm['{name}'] has shape {shape}, but the "
                    f"field has shape {declared}; pass {declared}, or a scalar to "
                    f"fill it. For a per-element sweep, use solve_batched()."
                )

        initial_state = self._resolve_initial_state(
            x_initial,
            x_final,
            x_guess=x_guess,
            u_guess=u_guess,
            max_iters=max_iters,
            algorithm=algorithm,
        )
        solve_fn = self._get_or_build_solve_loop()
        final_state = solve_fn(initial_state, params)
        return OptimizationResults.from_final_state(final_state, problem=self)

    def solve_batched(
        self,
        x_initial: Optional[jnp.ndarray] = None,
        x_final: Optional[jnp.ndarray] = None,
        parameters: Optional[dict] = None,
        *,
        x_guess: Optional[jnp.ndarray] = None,
        u_guess: Optional[jnp.ndarray] = None,
        max_iters: Optional[jnp.ndarray] = None,
        algorithm: Optional[dict] = None,
        in_axes: Optional[dict] = None,
    ) -> OptimizationResults:
        """Run ``B`` SCP solves in parallel as one batched XLA program.

        Takes the same arguments as :meth:`solve_jax`, with one rule on top:
        **add a leading batch axis to whatever varies across solves**. An
        argument whose shape matches its unbatched shape is shared by every
        batch element; an argument with exactly one extra leading axis is
        batched along it. The batch size ``B`` is the common leading axis of
        the batched entries, and every array leaf of the returned
        :class:`OptimizationResults` carries it (e.g. ``result.x.shape ==
        (B, N, n_x)``)::

            results = problem.solve_batched(
                x_initial=x0_batch,                       # (B, n_x): batched pins
                parameters={
                    "gate_center": centers,               # (B, 3) vs declared (3,) -> batched
                    "gate_radius": 0.5,                   # declared shape -> shared
                },
                algorithm={
                    "ep_tr": jnp.logspace(-5, -3, B),     # (B,) vs field () -> tolerance sweep
                    "lam_prox": jnp.linspace(1.0, 4.0, B),  # (B,) fill: one scalar per element
                },
            )

        Rank against the declared shape decides — never the leading-axis
        value — so a shared parameter whose leading axis happens to equal
        ``B`` stays shared. For the rare entry the rank rule cannot express
        (e.g. batching a declared rank-1 parameter element-wise), pass an
        explicit *in_axes*.

        Why this exists alongside ``jax.vmap(solve_jax)``: because the batch
        axis is owned here the whole vmapped loop is a single function that
        ``jax.export`` can serialize.  Under ``save_compiled=True`` the artifact
        is written to the solver cache on the first call and deserialized on
        later processes — skipping the XLA compile that ``jax.vmap(solve_jax)``
        pays on every fresh launch.  ``jax.vmap(solve_jax)`` is the right tool
        for in-program batching that composes with ``grad``/``scan``;
        ``solve_batched`` is for when cross-process cold-start dominates.

        Export requires a pure-JAX backend: under ``save_compiled=True`` the
        CVXPy backend raises.  The cache key invalidates on any change that
        alters the exported loop — backend, algorithm, discretizer, scaling,
        ``B``, the shared/batched parameter split, and the JAX version — so a
        stale artifact is never silently reused.  Tolerances and ``max_iters``
        are runtime inputs on the state pytree, so one artifact serves every
        setting of them.

        **What cannot be batched**: ``N``, the solver backend, the
        discretizer/integrator, the autotuner type, and ``float_dtype``
        change array shapes or program structure, so ``B`` such solves can
        never share one XLA program. There is deliberately no slot for them
        here — sweep them with a Python loop over separately built
        ``Problem`` instances.

        Args:
            x_initial: Initial-state boundary pin — shape ``(n_x,)`` shared
                across the batch or ``(B, n_x)`` batched. Same
                ``jnp.nan``-at-non-Fix convention as :meth:`solve_jax`.
                ``None`` uses the default from settings.
            x_final: Terminal-state boundary pin — ``(n_x,)`` or
                ``(B, n_x)``. Same conventions as ``x_initial``.
            parameters: Problem parameters for this solve, merged over
                ``self.parameters``. Each value either matches the
                parameter's declared shape (shared) or carries one extra
                leading axis, ``(B,) + declared`` (batched).
            x_guess: State trajectory guess — ``(N, n_x)`` shared or
                ``(B, N, n_x)`` batched. ``None`` uses the default from
                settings.
            u_guess: Control trajectory guess — ``(N, n_u)`` or
                ``(B, N, n_u)``. ``None`` uses the default from settings.
            max_iters: SCP iteration cap — ``()`` shared or ``(B,)``
                per-element budgets. Sugar for ``algorithm={"k_max": ...}``;
                passing both raises. ``None`` uses the configured ``k_max``.
                A runtime input: changing it reuses the cached (or exported)
                batched closure. The batch runs until its slowest element
                converges or exhausts its own cap; finished elements are
                frozen, not re-iterated.
            algorithm: Per-solve algorithm knobs, by the same names as the
                ``Problem`` constructor's ``algorithm`` dict (see
                :meth:`solve_jax`); all runtime inputs against one cached
                artifact. Each value takes the rank rule's two shapes — the
                field's shape (shared) or ``(B,) +`` it (batched) — plus two
                **fill** forms for array-shaped fields: a scalar (shared) or
                ``(B,)`` (one scalar per element, each broadcast to the
                field shape). Exact shapes win when both parse;
                ``in_axes={"algorithm": {name: 0}}`` forces the per-element
                fill reading.
            in_axes: Explicit batching override in ``jax.vmap``'s vocabulary,
                including its prefix semantics: a bare ``0`` batches every
                passed input; a dict keyed by argument name takes leaves
                ``0`` (batched) or ``None`` (shared), where ``"parameters"``
                and ``"algorithm"`` accept either a prefix (``0`` / ``None``,
                applied to every passed key) or a per-key dict, e.g.
                ``in_axes={"parameters": {"gate_center": 0}}``. Partial
                specs are fine — unlisted entries use the rank rule. Unlike
                ``vmap``, the default (``None``) means *infer by rank*, not
                broadcast-everything — an all-shared batched solve has no
                batch axis to find.

        Returns:
            :class:`OptimizationResults` pytree with a leading ``B`` axis on
            every leaf and empty per-iteration histories. The merged
            parameter values used for the solve are recorded on
            ``results.parameters``, post-broadcast — every value carries the
            leading ``(B,)`` axis, shared values replicated — and
            :meth:`post_process_batched` propagates each element with its
            own values.
        """
        if self._iteration_fn_jit_inner is None:
            raise ValueError(
                "Problem has not been initialized. Call initialize() before solve_batched()"
            )

        user_params = parameters or {}
        unknown = set(user_params) - set(self._parameters)
        if unknown:
            raise ValueError(
                f"solve_batched: unknown parameter(s) {sorted(unknown)}; "
                f"declared parameters are {sorted(self._parameters)}."
            )

        algorithm = dict(algorithm or {})
        self._validate_overrides(
            "solve_batched",
            algorithm,
            {
                "x_initial": x_initial,
                "x_final": x_final,
                "x_guess": x_guess,
                "u_guess": u_guess,
                "max_iters": max_iters,
            },
        )
        _default = self._default_state()

        N = self.settings.sim.n
        n_x = self.settings.sim.n_states
        n_u = self.settings.sim.n_controls
        entries = {
            "x_initial": (x_initial, (n_x,)),
            "x_final": (x_final, (n_x,)),
            "x_guess": (x_guess, (N, n_x)),
            "u_guess": (u_guess, (N, n_u)),
            "max_iters": (max_iters, ()),
        }
        for key, value in user_params.items():
            entries[f"parameters.{key}"] = (value, tuple(np.shape(self._parameters[key])))
        for name, value in algorithm.items():
            declared = tuple(jnp.shape(_override_target(_default, name)))
            entries[f"algorithm.{name}"] = (value, declared)

        flat_in_axes = _flatten_in_axes(in_axes, entries)

        B, axes, fills = _resolve_batch_spec(
            entries,
            flat_in_axes,
            fill={f"algorithm.{name}" for name in algorithm},
        )

        # Shared state-side inputs — and the defaults for omitted ones —
        # broadcast to a leading B axis; batched ones pass through.
        defaults = {
            "x_initial": _default.x_init_pin,
            "x_final": _default.x_term_pin,
            "x_guess": _default.x,
            "u_guess": _default.u,
            "max_iters": _default.k_max,
        }
        stacks = {}
        for name, default in defaults.items():
            value = entries[name][0]
            arr = jnp.asarray(default if value is None else value)
            if axes.get(name) is None:
                arr = jnp.broadcast_to(arr, (B,) + arr.shape)
            stacks[name] = arr

        # Algorithm knobs normalize to (B,) + field shape: fills first
        # broadcast their scalars up to the field shape, shared entries then
        # gain the leading batch axis.
        algorithm_stacks = {}
        for name, value in algorithm.items():
            entry = f"algorithm.{name}"
            field_shape = entries[entry][1]
            arr = jnp.asarray(value)
            if entry in fills and axes[entry] == 0:
                arr = jnp.broadcast_to(
                    arr.reshape(arr.shape + (1,) * len(field_shape)), (B,) + field_shape
                )
            elif entry in fills:
                arr = jnp.broadcast_to(arr, field_shape)
            if axes[entry] is None:
                arr = jnp.broadcast_to(arr, (B,) + arr.shape)
            algorithm_stacks[name] = arr

        states = jax.vmap(self._resolve_initial_state)(
            stacks["x_initial"],
            stacks["x_final"],
            stacks["x_guess"],
            stacks["u_guess"],
            stacks["max_iters"],
            algorithm_stacks,
        )

        params_for_solve = dict(self._parameters, **user_params)
        param_axes = {key: axes.get(f"parameters.{key}") for key in params_for_solve}
        batched_solve = self._get_or_build_solve_batched(B, params_for_solve, param_axes)
        t_0_solve = time.time()
        if getattr(self, "_solve_batched_is_exported", False):
            final_states = batched_solve.call(states, params_for_solve)
        else:
            final_states = batched_solve(states, params_for_solve)
        jax.block_until_ready(final_states)
        self.timing_solve = time.time() - t_0_solve
        results = OptimizationResults.from_final_state(final_states, problem=self)
        # Record the as-used parameter values with a leading (B,) axis on
        # every leaf, so post_process_batched() can index element b directly.
        results.parameters = {
            name: np.array(np.moveaxis(np.asarray(value), param_axes[name], 0))
            if param_axes[name] is not None
            else np.broadcast_to(np.asarray(value), (B,) + np.shape(value)).copy()
            for name, value in params_for_solve.items()
        }
        return results

    def post_process(self) -> OptimizationResults:
        """Propagate solution through full nonlinear dynamics for high-fidelity trajectory.

        Integrates the converged SCP solution through the nonlinear dynamics to
        produce x_full, u_full, and t_full. Call after solve() for final results.

        Propagation uses the parameter values recorded at solve time
        (``results.parameters``), so mutating ``self.parameters`` between
        :meth:`solve` and here only seeds the next solve.

        Returns:
            OptimizationResults with propagated trajectory fields

        Raises:
            ValueError: If solve() has not been called yet.
        """
        if self._solution is None:
            raise ValueError("No solution available. Call solve() first.")

        # Enable the profiler (reuse session from initialize)
        pr = profiling.profiling_start(self.settings.dev.profiling, self._profiling_session)

        # Create result from stored solution state
        result = self._format_result(
            self._solution,
            self._solution_history,
            int(self._solution.k) <= self._algorithm.k_max,
        )

        result.parameters = self._solution_parameters

        t_0_post = time.time()
        result = propagate_trajectory_results(
            self._solution_parameters,
            self.settings,
            result,
            self._propagation_solver,
            dynamics_discrete=self._lowered.dynamics_discrete.f,
            algebraic_prop=self._lowered.algebraic_prop,
            discretizer=self._discretizer,
        )
        t_f_post = time.time()

        self.timing_post = t_f_post - t_0_post

        # Print results summary
        printing.print_results_summary(
            result, self.timing_post, self.timing_init, self.timing_solve
        )

        profiling.profiling_end(pr, "postprocess")
        return result

    def post_process_batched(
        self,
        results: OptimizationResults,
        *,
        n_times: Optional[int] = None,
    ) -> OptimizationResults:
        """Propagate all batch elements through the nonlinear dynamics.

        Applies the same high-fidelity propagation as :meth:`post_process` to
        every element of a batched :class:`~openscvx.algorithms.OptimizationResults`
        returned by :meth:`solve_batched`.  Results are stacked so that the
        post-process fields carry a leading batch axis ``B``:

        - ``results.t_full``  — ``(B, n_times)``
        - ``results.x_full``  — ``(B, n_times, n_prop_states)``
        - ``results.u_full``  — ``(B, n_times, n_controls)``
        - ``results.trajectory`` — ``{name: (B, n_times, ...)}``
        - ``results.cost``    — ``(B,)``
        - ``results.ctcs_violation`` — ``(B, ...)`` or ``None``

        Propagation runs sequentially over the batch in a Python loop; each
        element propagates with its own parameter values from the snapshot
        :meth:`solve_batched` recorded on ``results.parameters``. Results
        without a snapshot (e.g. loaded from an older save) fall back to
        ``self.parameters``.

        Args:
            results: Batched :class:`~openscvx.algorithms.OptimizationResults`
                from :meth:`solve_batched`.  ``results.X[-1]`` must have shape
                ``(B, N, n_x)``.
            n_times: Number of interpolation points for the dense time grid.
                When ``None`` (default), uses
                ``ceil(T_max / settings.prp.dt) + 1`` where ``T_max`` is the
                longest terminal time across the batch.  All ``B`` elements are
                interpolated to the **same** ``n_times`` so the output arrays
                have uniform shape.

        Returns:
            The same ``results`` object with ``t_full``, ``x_full``, ``u_full``,
            ``trajectory``, ``cost``, and ``ctcs_violation`` populated.

        Raises:
            ValueError: If the problem has not been initialized yet.
        """
        if self._lowered is None:
            raise ValueError("Problem not initialized. Call initialize() first.")

        x_batch = np.asarray(results.X[-1])  # (B, N, n_x)
        B = x_batch.shape[0]

        if n_times is None:
            t_finals = np.asarray(results.t_final).reshape(-1)  # (B,)
            T_max = float(t_finals.max())
            n_times = max(int(np.ceil(T_max / self.settings.prp.dt)) + 1, 2)

        # The propagation solver is compiled with a per-segment tau capacity of
        # ``max_tau_len``.  A uniform ``linspace`` grid sized from the batch
        # ``T_max`` can place more output times into one normalized segment than
        # that capacity when shorter trajectories are padded to the same count.
        n_times = min(n_times, self.settings.prp.max_tau_len)

        t_fulls: List[np.ndarray] = []
        x_fulls: List[np.ndarray] = []
        u_fulls: List[np.ndarray] = []
        trajectories: List[Dict] = []
        costs: List[float] = []
        ctcs_violations = []

        # Empty for results saved before the snapshot existed — those fall
        # back to the Problem's current values, the pre-snapshot behaviour.
        snapshot = getattr(results, "parameters", None) or {}

        t_0_post = time.time()
        for b in range(B):
            single = _make_single_result(results, b)
            prop = propagate_trajectory_results(
                dict(self._parameters, **{k: np.asarray(v)[b] for k, v in snapshot.items()}),
                self.settings,
                single,
                self._propagation_solver,
                dynamics_discrete=self._lowered.dynamics_discrete.f,
                algebraic_prop=self._lowered.algebraic_prop,
                discretizer=self._discretizer,
                n_times=n_times,
            )
            t_fulls.append(prop.t_full)
            x_fulls.append(prop.x_full)
            u_fulls.append(prop.u_full)
            trajectories.append(prop.trajectory)
            costs.append(prop.cost)
            ctcs_violations.append(prop.ctcs_violation)

        self.timing_post = time.time() - t_0_post

        results.t_full = np.stack(t_fulls)  # (B, n_times)
        results.x_full = np.stack(x_fulls)  # (B, n_times, n_prop_states)
        results.u_full = np.stack(u_fulls)  # (B, n_times, n_controls)
        results.trajectory = {
            k: np.stack([traj[k] for traj in trajectories]) for k in trajectories[0]
        }
        results.cost = np.array(costs)  # (B,)
        if ctcs_violations[0] is not None:
            results.ctcs_violation = np.stack(ctcs_violations)

        printing.print_batch_results_summary(
            results,
            self.timing_post,
            self.timing_init,
            self.timing_solve,
            timing_compile=self.timing_compile,
        )
        return results

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

        # Discretization citations
        dis_citations = self._discretizer.citation()
        if dis_citations:
            dis_name = type(self._discretizer).__name__
            header = f"% Discretization: {dis_name}"
            citations = "\n".join(dis_citations)
            sections.append(f"{header}\n\n{citations}")

        sections.append(r"% --- END AUTO-GENERATED CITATIONS")

        return "\n\n".join(sections)
