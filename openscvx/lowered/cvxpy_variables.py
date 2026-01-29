"""CVXPy variables and parameters dataclass for the optimal control problem."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    import cvxpy as cp


@dataclass
class CVXPyVariables:
    """CVXPy variables and parameters for the optimal control problem.

    This dataclass holds all CVXPy Variable and Parameter objects needed to
    construct and solve the optimal control problem. It replaces the previous
    untyped dictionary approach with a typed, self-documenting structure.

    The variables are organized into logical groups:
        - SCP weights: Parameters controlling trust region and penalty weights
        - State: Variables and parameters for the state trajectory
        - Control: Variables and parameters for the control trajectory
        - Dynamics: Parameters for the discretized dynamics constraints
        - Nodal constraints: Parameters for linearized non-convex nodal constraints
        - Cross-node constraints: Parameters for linearized cross-node constraints
        - Scaling: Affine scaling matrices and offset vectors
        - Scaled expressions: CVXPy expressions for scaled state/control at each node

    Attributes:
        lam_prox: Trust region weight parameter (scalar, nonneg)
        lam_cost: Cost function weight parameter (scalar, nonneg)
        lam_vc: Virtual control penalty weights (N-1 x n_states, nonneg)
        lam_vb: Virtual buffer penalty weight (scalar, nonneg)

        x: State variable (N x n_states)
        dx: State error variable (N x n_states)
        x_bar: Previous SCP state parameter (N x n_states)
        x_init: Initial state parameter (n_states,)
        x_term: Terminal state parameter (n_states,)

        u: Control variable (N x n_controls)
        du: Control error variable (N x n_controls)
        u_bar: Previous SCP control parameter (N x n_controls)

        A_d: Discretized state Jacobian parameter (N-1 x n_states*n_states)
        B_d: Discretized control Jacobian parameter (N-1 x n_states*n_controls)
        C_d: Discretized control Jacobian (next node) parameter
        x_prop: Propagated state parameter (N-1 x n_states)
        nu: Virtual control variable (N-1 x n_states)

        g: List of constraint value parameters (one per nodal constraint)
        grad_g_x: List of state gradient parameters (one per nodal constraint)
        grad_g_u: List of control gradient parameters (one per nodal constraint)
        nu_vb: List of virtual buffer variables (one per nodal constraint)

        g_cross: List of cross-node constraint value parameters
        grad_g_X_cross: List of trajectory state gradient parameters
        grad_g_U_cross: List of trajectory control gradient parameters
        nu_vb_cross: List of cross-node virtual buffer variables

        S_x: State scaling matrix (n_states x n_states)
        inv_S_x: Inverse state scaling matrix
        c_x: State offset vector (n_states,)
        S_u: Control scaling matrix (n_controls x n_controls)
        inv_S_u: Inverse control scaling matrix
        c_u: Control offset vector (n_controls,)

        x_nonscaled: List of scaled state expressions at each node
        u_nonscaled: List of scaled control expressions at each node
        dx_nonscaled: List of scaled state error expressions at each node
        du_nonscaled: List of scaled control error expressions at each node
    """

    # SCP weight parameters
    lam_prox: "cp.Parameter"
    lam_cost: "cp.Parameter"
    lam_vc: "cp.Parameter"
    lam_vb: "cp.Parameter"

    # State variables and parameters
    x: "cp.Variable"
    dx: "cp.Variable"
    x_bar: "cp.Parameter"
    x_init: "cp.Parameter"
    x_term: "cp.Parameter"

    # Control variables and parameters
    u: "cp.Variable"
    du: "cp.Variable"
    u_bar: "cp.Parameter"

    # Dynamics discretization parameters
    A_d: "cp.Parameter"
    B_d: "cp.Parameter"
    C_d: "cp.Parameter"
    x_prop: "cp.Parameter"
    nu: "cp.Variable"

    # Nodal constraint linearization (lists, one per constraint)
    g: List["cp.Parameter"] = field(default_factory=list)
    grad_g_x: List["cp.Parameter"] = field(default_factory=list)
    grad_g_u: List["cp.Parameter"] = field(default_factory=list)
    nu_vb: List["cp.Variable"] = field(default_factory=list)

    # Cross-node constraint linearization (lists, one per constraint)
    g_cross: List["cp.Parameter"] = field(default_factory=list)
    grad_g_X_cross: List["cp.Parameter"] = field(default_factory=list)
    grad_g_U_cross: List["cp.Parameter"] = field(default_factory=list)
    nu_vb_cross: List["cp.Variable"] = field(default_factory=list)

    # Scaling matrices and offsets (numpy arrays)
    S_x: np.ndarray = field(default_factory=lambda: np.array([]))
    inv_S_x: np.ndarray = field(default_factory=lambda: np.array([]))
    c_x: np.ndarray = field(default_factory=lambda: np.array([]))
    S_u: np.ndarray = field(default_factory=lambda: np.array([]))
    inv_S_u: np.ndarray = field(default_factory=lambda: np.array([]))
    c_u: np.ndarray = field(default_factory=lambda: np.array([]))

    # Scaled CVXPy expressions at each node (lists of length N)
    x_nonscaled: List = field(default_factory=list)
    u_nonscaled: List = field(default_factory=list)
    dx_nonscaled: List = field(default_factory=list)
    du_nonscaled: List = field(default_factory=list)
