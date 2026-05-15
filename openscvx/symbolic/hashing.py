"""Structural hashing for symbolic problems.

This module provides name-invariant hashing for symbolic optimization problems.
Two problems with the same mathematical structure will produce the same hash,
regardless of the variable names used.

This enables efficient caching: if a problem has already been compiled with
the same structure, the cached compiled artifacts can be reused.
"""

import hashlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openscvx.symbolic.problem import SymbolicProblem


def _openscvx_version() -> str:
    """Package version for cache invalidation; works without generated _version.py."""
    try:
        return version("openscvx")
    except PackageNotFoundError:
        pass
    try:
        from openscvx._version import __version__ as v

        return v
    except ImportError:
        return "0.0.0"


def hash_symbolic_problem(problem: "SymbolicProblem") -> str:
    """Compute a structural hash of a symbolic optimization problem.

    This function computes a hash that depends only on the mathematical structure
    of the problem, not on variable names or runtime values. Two problems with the same:
    - Dynamics expressions (using _slice for canonical variable positions)
    - Constraints
    - State/control shapes and boundary condition types
    - Parameter shapes
    - Configuration (N, etc.)

    will produce the same hash, regardless of what names are used for variables.

    Notably, the following are NOT included in the hash (allowing solver reuse):
    - Boundary condition values (initial/final state values)
    - Bound values (min/max for states and controls)
    - Parameter values (only shapes are hashed)

    Args:
        problem: A SymbolicProblem (should be preprocessed for best results,
                 so that _slice attributes are set on states/controls)

    Returns:
        A hex string representing the SHA-256 hash of the problem structure
    """
    hasher = hashlib.sha256()

    # Include library version to invalidate cache on version changes
    hasher.update(f"openscvx:{_openscvx_version()}:".encode())

    # Hash the dynamics
    hasher.update(b"dynamics:")
    problem.dynamics._hash_into(hasher)

    # Hash propagation dynamics if present
    if problem.dynamics_prop is not None:
        hasher.update(b"dynamics_prop:")
        problem.dynamics_prop._hash_into(hasher)

    # Hash all constraints (order-invariant within each category)
    # We compute individual hashes and sort them so that the same set of
    # constraints produces the same hash regardless of definition order.
    hasher.update(b"constraints:")
    for constraint_list in [
        problem.constraints.ctcs,
        problem.constraints.nodal,
        problem.constraints.nodal_convex,
        problem.constraints.cross_node,
        problem.constraints.cross_node_convex,
    ]:
        # Compute individual hashes for each constraint
        constraint_hashes = sorted(c.structural_hash() for c in constraint_list)
        # Hash the count and sorted hashes
        hasher.update(len(constraint_hashes).to_bytes(4, "big"))
        for h in constraint_hashes:
            hasher.update(h)

    # Hash all states and controls explicitly to capture metadata (boundary
    # condition types) that may not appear in expressions. For example, a state
    # with dynamics dx/dt = 1.0 doesn't appear in the expression tree, but its
    # boundary condition types still affect the compiled problem structure.
    hasher.update(b"states:")
    for state in problem.states:
        state._hash_into(hasher)

    hasher.update(b"controls:")
    for control in problem.controls:
        control._hash_into(hasher)

    # Hash parameter shapes (not values) from the problem's parameter dict.
    # This allows the same compiled solver to be reused across parameter sweeps -
    # only the structure matters for compilation, not the actual values.
    hasher.update(b"parameters:")
    hasher.update(str(len(problem.parameters)).encode())  # Hash count for structure
    for name in sorted(problem.parameters.keys()):
        value = problem.parameters[name]
        # Only hash shape, not name - maintains name-invariance
        if isinstance(value, np.ndarray):
            hasher.update(str(value.shape).encode())
        else:
            hasher.update(b"scalar")

    # Hash configuration
    hasher.update(f"N:{problem.N}".encode())

    # Hash node intervals for CTCS
    hasher.update(b"node_intervals:")
    for interval in problem.node_intervals:
        hasher.update(f"{interval}".encode())

    return hasher.hexdigest()
