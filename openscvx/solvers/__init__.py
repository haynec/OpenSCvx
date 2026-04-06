"""Convex subproblem solvers for trajectory optimization.

This module provides implementations of convex subproblem solvers used within
SCvx algorithms. At each iteration of a successive convexification algorithm,
the non-convex problem is approximated by a convex subproblem, which is then
solved using one of these solver backends.

All solvers inherit from :class:`ConvexSolver`, enabling pluggable solver
implementations and custom backends:

```python
class ConvexSolver(ABC):
    @abstractmethod
    def create_variables(self, N, x_unified, u_unified, jax_constraints) -> None:
        '''Create backend-specific optimization variables (called once).'''
        ...

    @abstractmethod
    def initialize(self, lowered, settings) -> None:
        '''Build the convex subproblem structure (called once).'''
        ...

    @abstractmethod
    def solve(self, state, params, settings) -> Any:
        '''Update parameters and solve (called each iteration).'''
        ...
```

This architecture enables users to implement custom solver backends such as:

- Direct Clarabel solver (Rust-based, GPU-capable)
- QPAX (JAX-based QP solver for end-to-end differentiability)
- OSQP direct interface (specialized for QP structure)
- Custom embedded solvers for real-time applications
- Research solvers with specialized structure exploitation

Note:
    Solvers own their optimization variables (e.g., ``CVXPySolver.ocp_vars``).
    The lowering process calls ``solver.create_variables()`` before constraint
    lowering, then ``solver.initialize()`` after. See :mod:`openscvx.solvers.base`
    for the interface details.
"""

from typing import Any

from .base import _SOLVER_MAP, ConvexSolver, SolverSpec
from .ptr_solver import PTRSolver, PTRSolveResult

# ---------------------------------------------------------------------------
# Populate the solver class map now that all classes are imported
# ---------------------------------------------------------------------------

_SOLVER_MAP.update(
    {
        "PTRSolver": PTRSolver,
    }
)


def resolve_solver_config(val: Any) -> SolverSpec:
    """Validate a dict/Spec into a :class:`SolverSpec` instance."""
    if isinstance(val, SolverSpec):
        return val
    return SolverSpec.model_validate(val)


__all__ = [
    # Base class
    "ConvexSolver",
    # PTR solver
    "PTRSolver",
    "PTRSolveResult",
    # Config
    "SolverSpec",
    "resolve_solver_config",
]
