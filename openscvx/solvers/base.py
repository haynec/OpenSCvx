"""Base class for convex subproblem solvers.

This module defines the abstract interface that all convex solver implementations
must follow for use within successive convexification algorithms.

!!! note

    Solvers own both their optimization variables (``create_variables()``) and
    the lowering of any user ``.convex()`` constraints
    (``lower_convex_constraints()``). The default ``lower_convex_constraints``
    refuses user ``.convex()`` constraints with a clear error — backends that
    accept them override it. This keeps ``openscvx.symbolic.lower``
    backend-agnostic: it never branches on solver type, it just delegates.

    See :class:`openscvx.solvers.ptr_solver.PTRSolver` for the PTR-specific
    interface every PTR backend implements.
"""

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, model_validator

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered import LoweredProblem
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints
    from openscvx.lowered.unified import UnifiedControl, UnifiedState
    from openscvx.symbolic.constraint_set import ConstraintSet


class ConvexSolver(ABC):
    """Abstract base class for convex subproblem solvers.

    This class defines the interface for solvers that handle the convex
    subproblems generated at each iteration of a successive convexification
    algorithm.

    Subclasses must implement all abstract methods below.

    The solver lifecycle has two phases:

    **Setup (called once):**

    - create_variables: Create backend-specific variables
    - initialize: Build the problem structure using lowered constraints

    **Per-iteration (called each SCP iteration):**

    - update_dynamics_linearization: Set linearization point and dynamics matrices
    - update_constraint_linearizations: Set constraint values and gradients
    - update_penalties: Set penalty weights
    - solve: Solve and return results

    Example:
        Implementing a custom solver::

            class MySolver(ConvexSolver):
                def create_variables(self, N, x_unified, u_unified, jax_constraints):
                    self._vars = create_my_variables(N, x_unified, ...)

                def initialize(self, lowered, settings):
                    self._prob = build_my_problem(self._vars, lowered, settings)

                def update_dynamics_linearization(self, **kwargs):
                    # Set x_bar, u_bar, A_d, B_d, etc.
                    ...

                def update_constraint_linearizations(self, **kwargs):
                    # Set constraint function values and gradients
                    ...

                def update_penalties(self, **kwargs):
                    # Set lam_prox, lam_cost, lam_vc, lam_vb_nodal, lam_vb_cross
                    ...

                def solve(self):
                    self._prob.solve()
                    return MyResult(...)
    """

    @abstractmethod
    def create_variables(
        self,
        N: int,
        x_unified: "UnifiedState",
        u_unified: "UnifiedControl",
        jax_constraints: "LoweredJaxConstraints",
        dynamics_sparsity: Optional[tuple] = None,
        constraint_sparsity: Optional[list] = None,
    ) -> None:
        """Create backend-specific optimization variables.

        This method creates the optimization variables (decision variables and
        parameters) for this solver's backend. Called once during problem setup,
        before constraint lowering.

        The solver should store its variables on ``self`` for use in subsequent
        ``initialize()`` and ``solve()`` calls.

        Args:
            N: Number of discretization nodes
            x_unified: Unified state interface with dimensions and scaling bounds
            u_unified: Unified control interface with dimensions and scaling bounds
            jax_constraints: Lowered JAX constraints (for sizing linearization params)
            dynamics_sparsity: Optional tuple ``(A_d, B_d, C_d)`` of boolean
                ndarrays giving the discrete-time Jacobian sparsity patterns.
                ``A_d`` has shape ``(n_x, n_x)``; ``B_d`` and ``C_d`` have
                shape ``(n_x, n_u)``.
            constraint_sparsity: Optional list of ``(x_mask, u_mask)`` boolean
                1-D arrays, one per nodal constraint.
        """
        raise NotImplementedError

    def lower_convex_constraints(
        self,
        constraints: "ConstraintSet",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Lower user ``.convex()`` constraints into this backend's form.

        Called once by :func:`openscvx.symbolic.lower.lower_symbolic_problem`
        after ``create_variables()`` and before ``initialize()``.

        The default implementation refuses any user ``.convex()``
        constraints — appropriate for backends like
        :class:`openscvx.solvers.qpax_ptr_solver.QPAXPTRSolver` that don't
        accept second-order-cone constraints. Backends that do accept them
        (e.g. :class:`openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver`)
        override this to invoke their backend-specific lowerer.

        Args:
            constraints: Categorized symbolic constraints. Only the
                ``nodal_convex`` / ``cross_node_convex`` lists matter here;
                non-convex constraints go through the JAX lowering pipeline.
            parameters: Optional dict of symbolic ``Parameter`` objects
                referenced by the constraints. May be ``None``.

        Returns:
            ``(lowered_list, parameter_map)``. The first is a list of
            backend-specific constraint objects (e.g. ``cp.Constraint``);
            the second maps parameter names to backend-specific parameter
            objects. Both are empty for the default refusal path.

        Raises:
            NotImplementedError: if the user defined any ``.convex()``
                constraints and this backend doesn't override.
        """
        n = len(constraints.nodal_convex) + len(constraints.cross_node_convex)
        if n:
            raise NotImplementedError(
                f"{type(self).__name__} does not support user-defined "
                f".convex() constraints ({n} defined). Drop the .convex() "
                "constraint or switch to a backend that supports them "
                "(e.g. openscvx.CVXPyPTRSolver)."
            )
        return [], {}

    @abstractmethod
    def initialize(
        self,
        lowered: "LoweredProblem",
        settings: "Config",
    ) -> None:
        """Build the convex subproblem structure.

        This method constructs the optimization problem once, using CVXPy
        Parameters (or equivalent) for values that change each iteration.
        Called once during problem setup, not at each SCP iteration.

        The solver should store its problem representation on ``self`` for use
        in subsequent ``solve()`` calls.

        Args:
            lowered: Lowered problem containing:
                - ``cvxpy_constraints``: Lowered convex constraints
                - ``jax_constraints``: JAX constraint functions
                - ``x_unified``, ``u_unified``: State/control interfaces
            settings: Configuration object with solver settings
        """
        raise NotImplementedError

    @abstractmethod
    def update_dynamics_linearization(self, **kwargs) -> None:
        """Update dynamics linearization point and matrices.

        Called at each SCP iteration before ``solve()`` to set the current
        linearization point and discretized dynamics matrices.

        The specific parameters depend on the solver implementation.
        See concrete solver classes for expected arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def update_constraint_linearizations(self, **kwargs) -> None:
        """Update linearized constraint values and gradients.

        Called at each SCP iteration before ``solve()`` to set constraint
        function values and gradients at the current linearization point.

        The specific parameters depend on the solver implementation.
        See concrete solver classes for expected arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def update_penalties(self, **kwargs) -> None:
        """Update SCP penalty weights.

        Called at each SCP iteration before ``solve()`` to set the current
        penalty weights for trust region, virtual control, and virtual buffer.

        The specific parameters depend on the solver implementation.
        See concrete solver classes for expected arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def update_boundary_conditions(self, **kwargs) -> None:
        """Update boundary condition parameters.

        Called once during algorithm initialization to set initial and terminal
        state constraints.

        The specific parameters depend on the solver implementation.
        See concrete solver classes for expected arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> dict:
        """Get solver statistics for diagnostics and printing.

        Returns:
            Dict containing solver statistics. Expected keys:
                - ``n_variables``: Total number of optimization variables
                - ``n_parameters``: Total number of parameters
                - ``n_constraints``: Total number of constraints
        """
        raise NotImplementedError

    @abstractmethod
    def solve(self) -> Any:
        """Solve the convex subproblem and return results.

        Called at each SCP iteration after updating linearization and penalties.
        Returns a solver-specific result object containing the solution.

        Returns:
            Solver-specific result object (e.g., ``PTRSolveResult`` for PTR).
        """
        raise NotImplementedError

    @abstractmethod
    def citation(self) -> List[str]:
        """Return BibTeX citations for this solver.

        Implementations should return a list of BibTeX entry strings for the
        papers that should be cited when using this solver.

        Returns:
            List of BibTeX citation strings.
        """
        raise NotImplementedError


# =============================================================================
# Pydantic spec for dict / YAML validation
# =============================================================================


class PTRSolverSpec(BaseModel):
    """Validates PTR solver configuration from dict/YAML input.

    The ``backend`` discriminator selects which concrete PTR backend to build:
    ``"cvxpy"`` (the default,
    :class:`openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver`) or ``"qpax"``
    (:class:`openscvx.solvers.qpax_ptr_solver.QPAXPTRSolver`).

    ``cvx_solver``, ``cvxpygen``, and ``cvxpygen_override`` are CVXPy-only;
    setting them under ``backend="qpax"`` is a configuration error.

    !!! warning
        Enabling ``cvxpygen`` currently disables sparse parameter declarations.
        cvxpygen does not yet support the N-D sparsity indices used by
        OpenSCvx's tiled parameters, so all parameters are created as dense
        when code generation is active.
    """

    type: Literal["PTRSolver"] = "PTRSolver"
    backend: Literal["cvxpy", "qpax"] = "cvxpy"
    cvx_solver: Optional[str] = None
    solver_args: Optional[Dict[str, Any]] = None
    cvxpygen: bool = False
    cvxpygen_override: bool = False

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_backend_fields(self):
        if self.backend == "qpax":
            offenders = [
                name
                for name, value in (
                    ("cvx_solver", self.cvx_solver),
                    ("cvxpygen", self.cvxpygen),
                    ("cvxpygen_override", self.cvxpygen_override),
                )
                if value
            ]
            if offenders:
                raise ValueError(
                    f"{offenders} only valid for backend='cvxpy'; "
                    "remove these fields or set backend='cvxpy'."
                )
        return self

    def build(self) -> ConvexSolver:
        # Local imports keep CVXPy / qpax out of the import path until the
        # corresponding backend is actually requested.
        if self.backend == "cvxpy":
            from .cvxpy_ptr_solver import CVXPyPTRSolver

            return CVXPyPTRSolver(
                cvx_solver=self.cvx_solver or "QOCO",
                solver_args=self.solver_args,
                cvxpygen=self.cvxpygen,
                cvxpygen_override=self.cvxpygen_override,
            )
        from .qpax_ptr_solver import QPAXPTRSolver

        return QPAXPTRSolver(solver_args=self.solver_args)


def __getattr__(name: str):
    """Deprecated alias: ``SolverSpec`` → :class:`PTRSolverSpec`.

    Kept for one release so existing dict/YAML configs and tests that import
    ``SolverSpec`` continue to work. Emit a ``DeprecationWarning`` on access.
    """
    if name == "SolverSpec":
        warnings.warn(
            "openscvx.solvers.base.SolverSpec is deprecated; use PTRSolverSpec.",
            DeprecationWarning,
            stacklevel=2,
        )
        return PTRSolverSpec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
