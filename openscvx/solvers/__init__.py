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

import inspect
from typing import Any

from .base import ConvexSolver
from .ptr_solver import PTRSolver, PTRSolveResult

# ---------------------------------------------------------------------------
# Spec resolver — turn a dict into a ConvexSolver instance
# ---------------------------------------------------------------------------

_SOLVER_MAP = {
    "PTRSolver": PTRSolver,
}


def _resolve_solver(val: Any) -> ConvexSolver:
    """Resolve a solver specification into an instance.

    Accepted forms:

    * **instance** — already-constructed :class:`ConvexSolver` (pass-through).
    * **dict** — keyword arguments passed to :class:`PTRSolver`.
      An optional ``"type"`` key selects the class (currently only
      ``"PTRSolver"``).

    Examples::

        # Dict with keyword overrides (default class)
        _resolve_solver({"cvx_solver": "CLARABEL", "solver_args": {"tol_gap_abs": 1e-7}})

        # Dict with explicit type
        _resolve_solver({"type": "PTRSolver", "cvx_solver": "CLARABEL"})

        # Instance pass-through
        _resolve_solver(PTRSolver(cvx_solver="CLARABEL"))
    """
    if isinstance(val, ConvexSolver):
        return val

    if not isinstance(val, dict):
        raise TypeError(f"Expected a ConvexSolver instance or dict, got {type(val).__name__}")

    kwargs = dict(val)  # copy to avoid mutating caller's dict
    name = kwargs.pop("type", "PTRSolver")

    cls = _SOLVER_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown solver {name!r}; expected one of {sorted(_SOLVER_MAP)}")

    try:
        return cls(**kwargs)
    except TypeError as e:
        valid = list(inspect.signature(cls.__init__).parameters.keys())
        valid.remove("self")
        raise TypeError(f"Invalid solver keyword argument: {e}. Valid keys: {valid}") from None


__all__ = [
    # Base class
    "ConvexSolver",
    # PTR solver
    "PTRSolver",
    "PTRSolveResult",
]
