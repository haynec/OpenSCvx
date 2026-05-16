"""Abstract base class for Penalized Trust-Region (PTR) convex subproblem solvers.

The PTR formulation — its variables, slack structure, cost terms, and
linearization contract — is shared across backends. This module defines that
contract; concrete backends (CVXPy, QPAX) live in sibling modules and
implement the assembly/dispatch that each backend's modeling layer requires.

Backends:
    :class:`openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver`
        DCP graph assembled via CVXPy, dispatched to a conic solver
        (QOCO, CLARABEL, ...).
    :class:`openscvx.solvers.qpax_ptr_solver.QPAXPTRSolver`
        Flat ``(Q, q, A, b, G, h)`` assembled as JAX arrays and solved with
        ``qpax.solve_qp``. Enables an end-to-end JAX-differentiable SCP loop
        in follow-up work.
    :class:`openscvx.solvers.moreau_ptr_solver.MoreauPTRSolver`
        Sparse conic program assembled as CSR JAX arrays and solved with
        ``moreau.jax.Solver``.  Uses SOC epigraphs for the L1 / pos PTR
        penalties instead of QPAX-style slack expansion; warm-starts between
        SCP iterations.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np

from .base import ConvexSolver

if TYPE_CHECKING:
    from openscvx.lowered.unified import UnifiedControl, UnifiedState


@dataclass
class PTRSolveResult:
    """Result from solving a PTR convex subproblem.

    Contains the solution trajectories and slack variables from a single
    SCP iteration. All trajectories are unscaled (physical units).

    Attributes:
        x: State trajectory, shape (N, n_states). Unscaled.
        u: Control trajectory, shape (N, n_controls). Unscaled.
        nu: Virtual control slack for dynamics defects, shape (N-1, n_states).
        nu_vb: Nonconvex nodal constraint violation slacks. List of arrays,
            one per nodal constraint.
        nu_vb_cross: Cross-node constraint violation slacks. List of scalars,
            one per cross-node constraint.
        cost: Optimal objective value.
        status: Solver status string (e.g., "optimal", "infeasible").
    """

    x: np.ndarray
    u: np.ndarray
    nu: np.ndarray
    nu_vb: List[np.ndarray]
    nu_vb_cross: List[float]
    cost: float
    status: str


class PTRSolver(ConvexSolver):
    """Abstract base class for Penalized Trust-Region convex subproblem solvers.

    Defines the contract every PTR backend must satisfy: per-iteration entry
    points for updating the linearization point, constraint gradients, penalty
    weights, and boundary conditions, plus a ``solve()`` returning a
    :class:`PTRSolveResult`. The set of variables (state, control, virtual
    control ``nu``, per-constraint virtual buffer ``nu_vb``, cross-node slack
    ``nu_vb_cross``) is fixed by the PTR formulation; how each backend
    realizes those variables is an implementation detail.

    Subclasses must additionally implement :meth:`create_variables` and
    :meth:`initialize` from :class:`ConvexSolver`.
    """

    @abstractmethod
    def update_dynamics_linearization(
        self,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        A_d: np.ndarray,
        B_d: np.ndarray,
        C_d: np.ndarray,
        x_prop: np.ndarray,
        x_prop_plus: np.ndarray | None = None,
        D_d: np.ndarray | None = None,
        E_d: np.ndarray | None = None,
    ) -> None:
        """Update dynamics linearization point and discrete-time matrices.

        Args:
            x_bar: Previous state trajectory, shape (N, n_states).
            u_bar: Previous control trajectory, shape (N, n_controls).
            A_d: Discretized state Jacobian, shape (N-1, n_states, n_states).
            B_d: Discretized control Jacobian (current node),
                shape (N-1, n_states, n_controls).
            C_d: Discretized control Jacobian (next node),
                shape (N-1, n_states, n_controls).
            x_prop: Propagated state from continuous dynamics,
                shape (N-1, n_states).
            x_prop_plus: Optional impulsive/discrete propagated state,
                shape (N, n_states).
            D_d: Optional impulsive/discrete Jacobian wrt state,
                shape (N, n_states, n_states).
            E_d: Optional impulsive/discrete Jacobian wrt control,
                shape (N, n_states, n_controls).
        """
        raise NotImplementedError

    @abstractmethod
    def update_constraint_linearizations(
        self,
        nodal: List[dict] = None,
        cross_node: List[dict] = None,
    ) -> None:
        """Update linearized constraint values and gradients.

        Args:
            nodal: List of dicts, one per nodal constraint, each containing:
                ``g`` (value at linearization point),
                ``grad_g_x`` (gradient w.r.t. state),
                ``grad_g_u`` (gradient w.r.t. control).
            cross_node: List of dicts, one per cross-node constraint, each
                containing: ``g``, ``grad_g_X`` (gradient w.r.t. full state
                trajectory), ``grad_g_U`` (gradient w.r.t. full control
                trajectory).
        """
        raise NotImplementedError

    @abstractmethod
    def update_penalties(
        self,
        lam_prox: np.ndarray,
        lam_cost: Union[float, np.ndarray],
        lam_vc: np.ndarray,
        lam_vb_nodal: np.ndarray,
        lam_vb_cross: np.ndarray,
    ) -> None:
        """Update SCP penalty weights.

        Args:
            lam_prox: Trust region weights, shape (N, n_states + n_controls).
            lam_cost: Cost function weight. Scalar or shape (n_states,).
            lam_vc: Virtual control penalty weights, shape (N-1, n_states).
            lam_vb_nodal: Virtual buffer penalty weights for nodal constraints,
                shape (N, n_nodal_constraints).
            lam_vb_cross: Virtual buffer penalty weights for cross-node
                constraints, shape (n_cross_node_constraints,).
        """
        raise NotImplementedError

    @abstractmethod
    def update_boundary_conditions(
        self,
        x_init: np.ndarray = None,
        x_term: np.ndarray = None,
    ) -> None:
        """Update initial and/or terminal state parameters.

        Args:
            x_init: Initial state vector, shape (n_states,). Optional.
            x_term: Terminal state vector, shape (n_states,). Optional.
        """
        raise NotImplementedError

    @abstractmethod
    def solve(self) -> PTRSolveResult:
        """Solve the convex subproblem and return a :class:`PTRSolveResult`.

        Call the four ``update_*`` methods first to set the linearization
        point, constraint gradients, penalties, and boundary conditions.
        """
        raise NotImplementedError

    @staticmethod
    def _scaling(unified: Union["UnifiedState", "UnifiedControl"]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the affine scaling matrices ``(S, c)`` for a unified
        state or control interface.

        The PTR formulation works on scaled variables ``z = S⁻¹ (x - c)``;
        ``S`` and ``c`` are chosen so the scaled variables sit roughly on
        ``[-1, 1]``. Bounds come from ``scaling_min``/``scaling_max`` when
        provided, otherwise from ``min``/``max``. Shared by every backend.
        """
        from openscvx.config import get_affine_scaling_matrices

        n = len(unified.max)
        lower = np.array(
            unified.scaling_min if unified.scaling_min is not None else unified.min,
            dtype=float,
        )
        upper = np.array(
            unified.scaling_max if unified.scaling_max is not None else unified.max,
            dtype=float,
        )
        return get_affine_scaling_matrices(n, lower, upper)
