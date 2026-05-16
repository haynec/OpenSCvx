"""CVXPy-lowered constraint dataclass."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cvxpy as cp


@dataclass
class LoweredCvxpyConstraints:
    """CVXPy-lowered convex constraints.

    Contains constraints that have been lowered to CVXPy constraint objects.
    These are added directly to the optimal control problem without
    linearization.

    Attributes:
        constraints: List of CVXPy constraint objects (cp.Constraint).
            Includes both nodal and cross-node convex constraints.
        n_skipped: Number of user ``.convex()`` constraints that were
            *not* lowered because the chosen solver opted out of CVXPy
            lowering (e.g.,
            :class:`openscvx.solvers.qpax_ptr_solver.QPAXPTRSolver`). The
            solver's ``initialize()`` checks this so it can raise a clear
            "use CVXPyPTRSolver" error when the user has convex constraints
            that the QP backend can't accept.
    """

    constraints: list["cp.Constraint"] = field(default_factory=list)
    n_skipped: int = 0
