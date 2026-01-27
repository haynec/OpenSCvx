"""Base class for convex subproblem solvers.

This module defines the abstract interface that all convex solver implementations
must follow for use within successive convexification algorithms.

!!! note

    Solvers own their optimization variables via ``create_variables()``.
    Convex constraint lowering remains in ``lower.py`` but uses the solver's
    variables.

    When adding non-CVXPy backends, there are two approaches:

    1. **Solver owns the lowerer**: The solver implements a
       ``lower_convex_constraints()`` method containing the lowering logic.

    2. **Solver determines the lowerer**: The solver references which lowerer
       to use, but the lowering logic stays in ``lower.py``. Example:

       ```python
       # In solver
       @property
       def lowerer(self):
           from openscvx.symbolic.lower import lower_cvxpy_constraints
           return lower_cvxpy_constraints

       # In lower_symbolic_problem()
       lowered_constraints = solver.lowerer(constraints, solver.variables, parameters)
       ```
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered import LoweredProblem
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints
    from openscvx.lowered.unified import UnifiedControl, UnifiedState


class ConvexSolver(ABC):
    """Abstract base class for convex subproblem solvers.

    This class defines the interface for solvers that handle the convex
    subproblems generated at each iteration of a successive convexification
    algorithm.

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
                    # Set lam_prox, lam_cost, lam_vc, lam_vb
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
        """
        ...

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
        ...

    @abstractmethod
    def update_dynamics_linearization(self, **kwargs) -> None:
        """Update dynamics linearization point and matrices.

        Called at each SCP iteration before ``solve()`` to set the current
        linearization point and discretized dynamics matrices.

        The specific parameters depend on the solver implementation.
        See concrete solver classes for expected arguments.
        """
        ...

    @abstractmethod
    def update_constraint_linearizations(self, **kwargs) -> None:
        """Update linearized constraint values and gradients.

        Called at each SCP iteration before ``solve()`` to set constraint
        function values and gradients at the current linearization point.

        The specific parameters depend on the solver implementation.
        See concrete solver classes for expected arguments.
        """
        ...

    @abstractmethod
    def update_penalties(self, **kwargs) -> None:
        """Update SCP penalty weights.

        Called at each SCP iteration before ``solve()`` to set the current
        penalty weights for trust region, virtual control, and virtual buffer.

        The specific parameters depend on the solver implementation.
        See concrete solver classes for expected arguments.
        """
        ...

    @abstractmethod
    def update_boundary_conditions(self, **kwargs) -> None:
        """Update boundary condition parameters.

        Called once during algorithm initialization to set initial and terminal
        state constraints.

        The specific parameters depend on the solver implementation.
        See concrete solver classes for expected arguments.
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        """Get solver statistics for diagnostics and printing.

        Returns:
            Dict containing solver statistics. Expected keys:
                - ``n_variables``: Total number of optimization variables
                - ``n_parameters``: Total number of parameters
                - ``n_constraints``: Total number of constraints
        """
        ...

    @abstractmethod
    def solve(self) -> Any:
        """Solve the convex subproblem and return results.

        Called at each SCP iteration after updating linearization and penalties.
        Returns a solver-specific result object containing the solution.

        Returns:
            Solver-specific result object (e.g., ``PTRSolveResult`` for PTR).
        """
        ...

    @abstractmethod
    def citation(self) -> List[str]:
        """Return BibTeX citations for this solver.

        Implementations should return a list of BibTeX entry strings for the
        papers that should be cited when using this solver.

        Returns:
            List of BibTeX citation strings.
        """
        ...
