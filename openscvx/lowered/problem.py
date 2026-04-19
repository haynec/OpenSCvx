"""LoweredProblem dataclass - container for all lowering outputs."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Optional

from openscvx.lowered.cvxpy_constraints import LoweredCvxpyConstraints
from openscvx.lowered.dynamics import Dynamics
from openscvx.lowered.jax_constraints import LoweredJaxConstraints
from openscvx.lowered.stm_meta import StmMeta
from openscvx.lowered.unified import UnifiedControl, UnifiedState

if TYPE_CHECKING:
    import cvxpy as cp


@dataclass
class LoweredProblem:
    """Container for all outputs from symbolic problem lowering.

    This dataclass holds all the results of lowering symbolic expressions
    to executable JAX and CVXPy code. It provides a clean, typed interface
    for accessing the various components needed for optimization.

    Note:
        CVXPy optimization variables (``ocp_vars``) are owned by the solver,
        not stored here. Access them via ``solver.ocp_vars``.

    Attributes:
        dynamics: Optimization dynamics with fields f, A, B (JAX functions)
        dynamics_discrete: Discrete optimization dynamics with fields f, A, B (JAX functions)
        dynamics_prop: Propagation dynamics with fields f, A, B
        jax_constraints: Non-convex constraints lowered to JAX with gradients
        cvxpy_constraints: Convex constraints lowered to CVXPy
        x_unified: Aggregated optimization state interface
        u_unified: Aggregated optimization control interface
        x_prop_unified: Aggregated propagation state interface
        cvxpy_params: Dict mapping user parameter names to CVXPy Parameter objects
        algebraic_prop: Dict mapping output names to vmapped JAX functions
            (evaluated, not integrated)

    Example:
        After lowering a symbolic problem::

            solver = CVXPySolver()
            lowered = lower_symbolic_problem(problem, solver, settings)

            # Access components
            dx_dt = lowered.dynamics.f(x, u, node, params)

            # Solver owns CVXPy variables
            ocp_vars = solver.ocp_vars
    """

    # JAX dynamics
    dynamics: Dynamics
    dynamics_discrete: Dynamics
    dynamics_prop: Dynamics

    # Lowered constraints (separate types for JAX vs CVXPy)
    jax_constraints: LoweredJaxConstraints
    cvxpy_constraints: LoweredCvxpyConstraints

    # Unified interfaces
    x_unified: UnifiedState
    u_unified: UnifiedControl
    x_prop_unified: UnifiedState

    # CVXPy constraint parameters (user-defined parameters lowered to CVXPy)
    cvxpy_params: Dict[str, "cp.Parameter"]

    # Algebraic outputs (vmapped JAX functions for propagation)
    algebraic_prop: Optional[Dict[str, Callable]] = field(default_factory=dict)

    # STM slot table (empty when problem declares no STM augmented states)
    stm_meta: StmMeta = field(default_factory=StmMeta)
