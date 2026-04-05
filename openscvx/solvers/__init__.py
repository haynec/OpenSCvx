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

from typing import Annotated, Any, Union

from pydantic import Field, TypeAdapter

from .base import ConvexSolver
from .ptr_solver import PTRSolver, PTRSolveResult

# ---------------------------------------------------------------------------
# Solver config — discriminated union of each solver's inner Spec
# ---------------------------------------------------------------------------

DEFAULT_SOLVER_TYPE = "PTRSolver"

SolverConfig = Annotated[
    Union[PTRSolver.Spec,],
    Field(discriminator="type"),
]

solver_config_adapter = TypeAdapter(SolverConfig)


def resolve_solver_config(val: Any) -> ConvexSolver.Spec:
    """Validate a dict/Spec into a :class:`ConvexSolver.Spec` instance.

    Injects the default ``type`` (``PTRSolver``) when the input dict
    omits it, preserving backwards compatibility.
    """
    if isinstance(val, ConvexSolver.Spec):
        return val
    if isinstance(val, dict) and "type" not in val:
        val = {**val, "type": DEFAULT_SOLVER_TYPE}
    return solver_config_adapter.validate_python(val)


__all__ = [
    # Base class
    "ConvexSolver",
    # PTR solver
    "PTRSolver",
    "PTRSolveResult",
    # Config
    "SolverConfig",
    "resolve_solver_config",
]
