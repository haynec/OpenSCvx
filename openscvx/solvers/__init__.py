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

The Penalized Trust-Region (PTR) subproblem ships with two concrete backends:

- :class:`CVXPyPTRSolver` — DCP graph via CVXPy, dispatched to any of its
  supported conic solvers (QOCO, CLARABEL, ...). Optional code generation
  via cvxpygen for improved per-iteration performance.
- :class:`QPAXPTRSolver` — flat ``(Q, q, A, b, G, h)`` assembled as JAX
  arrays and solved with ``qpax.solve_qp``. Aimed at end-to-end JAX
  differentiability of the SCP loop (follow-up work).

Both share the abstract :class:`PTRSolver` contract.

Note:
    Solvers own their optimization variables (e.g., ``CVXPySolver.ocp_vars``).
    The lowering process calls ``solver.create_variables()`` before constraint
    lowering, then ``solver.initialize()`` after. See :mod:`openscvx.solvers.base`
    for the interface details.
"""

import warnings
from typing import Any

from .base import ConvexSolver, PTRSolverSpec
from .cvxpy_ptr_solver import CVXPyPTRSolver
from .ptr_solver import PTRSolveResult, PTRSolver


def resolve_solver_config(val: Any) -> PTRSolverSpec:
    """Validate a dict / Spec into a :class:`PTRSolverSpec` instance."""
    if isinstance(val, PTRSolverSpec):
        return val
    return PTRSolverSpec.model_validate(val)


def __getattr__(name: str):
    """Deprecated alias: ``SolverSpec`` → :class:`PTRSolverSpec`."""
    if name == "SolverSpec":
        warnings.warn(
            "openscvx.solvers.SolverSpec is deprecated; use PTRSolverSpec.",
            DeprecationWarning,
            stacklevel=2,
        )
        return PTRSolverSpec
    if name == "QPAXPTRSolver":
        # Lazy import so users without the qpax extra don't pay a hard
        # ImportError just for `from openscvx.solvers import QPAXPTRSolver`
        # — the import error gets deferred to instantiation time, where the
        # error message points at the install command.
        from .qpax_ptr_solver import QPAXPTRSolver

        return QPAXPTRSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "ConvexSolver",
    "PTRSolver",
    "PTRSolveResult",
    # PTR backends
    "CVXPyPTRSolver",
    "QPAXPTRSolver",
    # Config
    "PTRSolverSpec",
    "resolve_solver_config",
]
