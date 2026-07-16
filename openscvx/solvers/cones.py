"""Cone-constraint data types for the principled convex constraint pipeline.

Each :class:`ConeConstraint` subclass represents a family of convex constraints
that can be assembled directly into a JAX-native solver's constraint matrix
without going through CVXPy.  The hierarchy mirrors the three fundamental
cones used by most interior-point solvers:

* :class:`ZeroConeConstraint` — affine equality ``f(x,u) = 0``.
* :class:`NonnegConeConstraint` — affine inequality ``f(x,u) ≤ 0``.
* :class:`SOCConstraint` — second-order-cone ``‖arg(x,u)‖₂ ≤ bound(x,u)``.

All three are produced by :func:`~openscvx.symbolic.canonicalize.canonicalize_nodal_constraint`
and consumed by the solver assembly methods of
:class:`~openscvx.solvers.qpax_ptr_solver.QPAXPTRSolver` and
:class:`~openscvx.solvers.moreau_ptr_solver.MoreauPTRSolver`.

Design notes
------------
* Every constraint stores its **output dimension** (``m``) so that
  :meth:`~openscvx.solvers.moreau_ptr_solver.MoreauPTRSolver._structural_pass`
  can compute the static CSR sparsity pattern at ``initialize()`` time.
* The :attr:`nodes` tuple records which discrete nodes the constraint is
  active at; the solver assembly loops over ``nodes`` and adds the
  appropriate rows for each.
* JAX callables have the uniform signature
  ``(x, u, node, params) -> array``, matching the rest of the lowering
  pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class ConeConstraint:
    """Abstract base for a single canonicalised convex constraint block.

    Attributes:
        nodes: Discrete-node indices at which this constraint is enforced.
    """

    nodes: Tuple[int, ...]


@dataclass
class ZeroConeConstraint(ConeConstraint):
    """Affine equality constraint: ``f(x, u, node, params) = 0``.

    The JAX callable :attr:`jax_fn` returns the residual vector of shape
    ``(m,)`` (or scalar when ``m == 1``).  At the solution the residual
    must be zero; no virtual-buffer slack is introduced.

    Attributes:
        jax_fn: ``(x, u, node, params) -> array[m]`` — the constraint
            residual ``lhs − rhs`` in standard form.
        m: Output dimension of *jax_fn*.  Must be ≥ 1.
    """

    jax_fn: Callable
    m: int


@dataclass
class NonnegConeConstraint(ConeConstraint):
    """Affine inequality constraint: ``f(x, u, node, params) ≤ 0``.

    The JAX callable :attr:`jax_fn` returns a residual vector of shape
    ``(m,)`` (or scalar when ``m == 1``) that must be component-wise
    non-positive at the solution.

    Attributes:
        jax_fn: ``(x, u, node, params) -> array[m]`` — the constraint
            residual ``lhs − rhs`` in standard form (≤ 0).
        m: Output dimension of *jax_fn*.  Must be ≥ 1.
    """

    jax_fn: Callable
    m: int
    slack_weight: Optional[float] = None


@dataclass
class SOCConstraint(ConeConstraint):
    """Second-order-cone constraint: ``‖arg(x, u, node, params)‖₂ ≤ bound(…)``.

    Represents a Lorentz cone constraint of dimension ``m_arg + 1``.

    Attributes:
        arg_fn: ``(x, u, node, params) -> array[m_arg]`` — the vector whose
            L2 norm must not exceed *bound_fn*.
        bound_fn: ``(x, u, node, params) -> scalar`` — the upper bound on
            the norm.
        m_arg: Dimension of the argument vector (``m_arg ≥ 1``).  The full
            SOC cone dimension is ``m_arg + 1``.
    """

    arg_fn: Callable
    bound_fn: Callable
    m_arg: int


@dataclass
class PSDConeConstraint(ConeConstraint):
    """Generic PSD / LMI constraint: matrix_fn(x, u) ≽ 0.
    CVXPY-only. JAX-native backends raise NotImplementedError.
    """

    matrix_fn: Callable  # (x_cvxpy_at_node, u_cvxpy_at_node) -> (n,n) cp.Expression
    n: int
    slack_weight: Optional[float] = None
