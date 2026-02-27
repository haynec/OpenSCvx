"""Symbolic expression lowering to executable code.

This module provides the main entry point for converting symbolic expressions
(AST nodes) into executable code for different backends (JAX, CVXPy, etc.).
The lowering process translates the symbolic expression tree into functions
that can be executed during optimization.

Architecture:
    The lowering process follows a visitor pattern where each backend implements
    a lowerer class (e.g., JaxLowerer, CVXPyLowerer) with visitor methods for
    each expression type. The `lower()` function dispatches expression nodes
    to the appropriate backend.

    Lowering Flow:

    1. Symbolic expressions are built during problem specification
    2. lower_symbolic_expressions() coordinates the full lowering process
    3. Backend-specific lowerers convert each expression node to executable code
    4. Automatic differentiation creates Jacobians for dynamics and constraints
    5. Result is a set of executable functions ready for numerical optimization

Backends:
    - JAX: For dynamics and non-convex constraints (with automatic differentiation)
    - CVXPy: For convex constraints (with disciplined convex programming)

Example:
    Basic lowering to JAX::

        import openscvx as ox
        from openscvx.symbolic.lower import lower_to_jax

        # Define symbolic expression
        x = ox.State("x", shape=(3,))
        u = ox.Control("u", shape=(2,))
        expr = ox.Norm(x)**2 + 0.1 * ox.Norm(u)**2

        # Lower to JAX function
        f = lower_to_jax(expr)
        # f is now a callable: f(x_val, u_val, node, params) -> scalar

    Full problem lowering::

        # After building symbolic problem...
        lowered = lower_symbolic_problem(
            dynamics_aug, states_aug, controls_aug,
            constraints, parameters, N,
            dynamics_prop, states_prop, controls_prop
        )
        # Access via LoweredProblem dataclass
        dynamics = lowered.dynamics
        jax_constraints = lowered.jax_constraints
        # Now have executable JAX functions with Jacobians
"""

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import cvxpy as cp
import jax
import numpy as np
from jax import jacfwd

from openscvx.expert import apply_byof
from openscvx.lowered import (
    CVXPyVariables,
    Dynamics,
    LoweredCrossNodeConstraint,
    LoweredCvxpyConstraints,
    LoweredJaxConstraints,
    LoweredNodalConstraint,
    LoweredProblem,
)
from openscvx.symbolic.constraint_set import ConstraintSet
from openscvx.symbolic.expr import Expr, NodeReference, traverse
from openscvx.symbolic.expr.control import Control

if TYPE_CHECKING:
    from openscvx.solvers import ConvexSolver
    from openscvx.symbolic.problem import SymbolicProblem

__all__ = [
    "lower",
    "lower_to_jax",
    "lower_cvxpy_constraints",
    "create_cvxpy_variables",
    "lower_symbolic_problem",
]
from openscvx.symbolic.unified import unify_controls, unify_states


def _sync_control_slices_in_expr(expr: Expr, controls: Sequence[Any]) -> None:
    """Sync Control node slices in an expression from canonical control objects.

    This is needed when an expression tree contains copied Control nodes
    (e.g., deep-copied propagation dynamics). After unification reassigns slices,
    copied nodes may still carry stale `_slice` values.
    """
    from openscvx.symbolic.expr import traverse
    from openscvx.symbolic.expr.control import Control

    control_slices = {control.name: control._slice for control in controls}

    def _sync(node: Expr) -> None:
        if not isinstance(node, Control):
            return
        canonical_slice = control_slices.get(node.name)
        if canonical_slice is not None:
            node._slice = canonical_slice

    traverse(expr, _sync)


def lower(expr: Expr, lowerer: Any) -> Any:
    """Dispatch an expression node to the appropriate lowerer backend.

    This is the main entry point for lowering a single symbolic expression to
    executable code. It delegates to the lowerer's `lower()` method, which
    uses the visitor pattern to dispatch based on expression type.

    Args:
        expr: Symbolic expression to lower (any Expr subclass)
        lowerer: Backend lowerer instance (e.g., JaxLowerer, CVXPyLowerer)

    Returns:
        Backend-specific representation of the expression. For JaxLowerer,
        returns a callable with signature (x, u, node, params) -> result.
        For CVXPyLowerer, returns a CVXPy expression object.

    Raises:
        NotImplementedError: If the lowerer doesn't support the expression type

    Example:
        Lower an expression to the appropriate backend (here JAX):

            from openscvx.symbolic.lowerers.jax import JaxLowerer
            x = ox.State("x", shape=(3,))
            expr = ox.Norm(x)
            lowerer = JaxLowerer()
            f = lower(expr, lowerer)

        f is now callable: f(x_val, u_val, node, params) -> scalar
    """
    return lowerer.lower(expr)


# --- Convenience wrappers for common backends ---


def lower_to_jax(exprs: Union[Expr, Sequence[Expr]]) -> Union[callable, list[callable]]:
    """Lower symbolic expression(s) to JAX callable(s).

    Convenience wrapper that creates a JaxLowerer and lowers one or more
    symbolic expressions to JAX functions. The resulting functions can be
    JIT-compiled and automatically differentiated.

    Args:
        exprs: Single expression or sequence of expressions to lower

    Returns:
        - If exprs is a single Expr: Returns a single callable with signature
            (x, u, node, params) -> array
        - If exprs is a sequence: Returns a list of callables with the same signature

    Example:
        Single expression::

            x = ox.State("x", shape=(3,))
            expr = ox.Norm(x)**2
            f = lower_to_jax(expr)
            # f(x_val, u_val, node_idx, params_dict) -> scalar

        Multiple expressions::

            exprs = [ox.Norm(x), ox.Norm(u), x @ A @ x]
            fns = lower_to_jax(exprs)
            # fns is [f1, f2, f3], each with same signature

    Note:
        All returned JAX functions have a uniform signature
        (x, u, node, params) regardless of whether they use all arguments.
        This standardization simplifies vectorization and differentiation.
    """
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    jl = JaxLowerer()
    if isinstance(exprs, Expr):
        return lower(exprs, jl)
    fns = [lower(e, jl) for e in exprs]
    return fns


def _tile_sparsity(pattern_2d: np.ndarray, K: int):
    """Convert a 2-D boolean pattern to CVXPY sparsity indices tiled over *K* slices.

    Returns ``np.where``-style index tuple ``(dim0, dim1, dim2)`` suitable for
    ``cp.Parameter((K, m, n), sparsity=...)``, or ``None`` when the pattern is
    fully dense (no benefit from declaring sparsity).
    """
    if pattern_2d.all():
        return None
    rows, cols = np.where(pattern_2d)
    n_nz = len(rows)
    return (
        np.repeat(np.arange(K), n_nz),
        np.tile(rows, K),
        np.tile(cols, K),
    )


def _tile_sparsity_2d(mask_1d: np.ndarray, K: int):
    """Convert a 1-D boolean mask to CVXPY sparsity indices tiled over *K* rows.

    Returns ``np.where``-style index tuple ``(rows, cols)`` suitable for
    ``cp.Parameter((K, len(mask)), sparsity=...)``, or ``None`` when the
    mask is fully dense.
    """
    if mask_1d.all():
        return None
    (cols,) = np.where(mask_1d)
    n_nz = len(cols)
    return (
        np.repeat(np.arange(K), n_nz),
        np.tile(cols, K),
    )


def _augment_impulsive_constraints(
    constraints: ConstraintSet,
    controls: list,
    n_nodes: int,
) -> None:
    if not controls:
        return
    added = []
    for control in controls:
        is_imp = getattr(control, "is_impulsive", False)
        is_impulsive = bool(is_imp.any()) if hasattr(is_imp, "any") else bool(is_imp)
        if not is_impulsive:
            continue
        control_nodes = getattr(control, "nodes", None)
        if control_nodes is None:
            allowed = set()
        else:
            allowed = {int(idx) for idx in control_nodes}
        for idx in allowed:
            if idx < 0 or idx >= n_nodes:
                raise ValueError(
                    f"Impulsive node index {idx} for control '{control.name}' "
                    f"is out of range [0, {n_nodes})"
                )
        nodes = [i for i in range(n_nodes) if i not in allowed]
        if not nodes:
            continue
        added.append((control == 0.0).convex().at(nodes))
    if added:
        constraints.nodal_convex.extend(added)


def create_cvxpy_variables(
    N: int,
    n_states: int,
    n_controls: int,
    S_x: np.ndarray,
    c_x: np.ndarray,
    S_u: np.ndarray,
    c_u: np.ndarray,
    n_nodal_constraints: int,
    n_cross_node_constraints: int,
    A_d_sparsity: Optional[tuple] = None,
    B_d_sparsity: Optional[tuple] = None,
    C_d_sparsity: Optional[tuple] = None,
    constraint_sparsity: Optional[list] = None,
) -> CVXPyVariables:
    """Create CVXPy variables and parameters for the optimal control problem.

    Args:
        N: Number of discretization nodes
        n_states: Number of state variables
        n_controls: Number of control variables
        S_x: State scaling matrix
        c_x: State offset vector
        S_u: Control scaling matrix
        c_u: Control offset vector
        n_nodal_constraints: Number of non-convex nodal constraints (for linearization params)
        n_cross_node_constraints: Number of non-convex cross-node constraints
        A_d_sparsity: Optional CVXPY sparsity indices for A_d parameter (from _tile_sparsity)
        B_d_sparsity: Optional CVXPY sparsity indices for B_d parameter
        C_d_sparsity: Optional CVXPY sparsity indices for C_d parameter
        constraint_sparsity: Optional list of ``(x_mask, u_mask)`` boolean 1-D
            arrays, one per nodal constraint (from ``_tile_sparsity_2d``).

    Returns:
        CVXPyVariables dataclass containing all CVXPy variables and parameters for the OCP
    """
    ########################
    # VARIABLES & PARAMETERS
    ########################

    inv_S_x = np.linalg.inv(S_x)
    inv_S_u = np.linalg.inv(S_u)

    # Parameters
    lam_prox = cp.Parameter(nonneg=True, name="lam_prox")
    lam_cost = cp.Parameter(n_states, nonneg=True, name="lam_cost")
    lam_vc = cp.Parameter((N - 1, n_states), nonneg=True, name="lam_vc")
    lam_vb = cp.Parameter(nonneg=True, name="lam_vb")

    # State
    x = cp.Variable((N, n_states), name="x")  # Current State
    dx = cp.Variable((N, n_states), name="dx")  # State Error
    x_bar = cp.Parameter((N, n_states), name="x_bar")  # Previous SCP State
    x_init = cp.Parameter(n_states, name="x_init")  # Initial State
    x_term = cp.Parameter(n_states, name="x_term")  # Final State

    # Control
    u = cp.Variable((N, n_controls), name="u")  # Current Control
    du = cp.Variable((N, n_controls), name="du")  # Control Error
    u_bar = cp.Parameter((N, n_controls), name="u_bar")  # Previous SCP Control

    # Discretized Augmented Dynamics Constraints
    A_d = cp.Parameter((N - 1, n_states, n_states), name="A_d", sparsity=A_d_sparsity)
    B_d = cp.Parameter((N - 1, n_states, n_controls), name="B_d", sparsity=B_d_sparsity)
    C_d = cp.Parameter((N - 1, n_states, n_controls), name="C_d", sparsity=C_d_sparsity)
    x_prop_pp = cp.Parameter((N, n_states), name="x_prop_pp")
    D_d = cp.Parameter((N, n_states, n_states), name="D_d")
    E_d = cp.Parameter((N, n_states, n_controls), name="E_d")
    x_prop = cp.Parameter((N - 1, n_states), name="x_prop")
    nu = cp.Variable((N - 1, n_states), name="nu")  # Virtual Control

    # Linearized Nonconvex Nodal Constraints
    g = []
    grad_g_x = []
    grad_g_u = []
    nu_vb = []
    for idx_ncvx in range(n_nodal_constraints):
        g_x_sp = None
        g_u_sp = None
        if constraint_sparsity is not None:
            x_mask, u_mask = constraint_sparsity[idx_ncvx]
            g_x_sp = _tile_sparsity_2d(x_mask, N)
            g_u_sp = _tile_sparsity_2d(u_mask, N)
        g.append(cp.Parameter(N, name="g_" + str(idx_ncvx)))
        grad_g_x.append(
            cp.Parameter((N, n_states), name="grad_g_x_" + str(idx_ncvx), sparsity=g_x_sp)
        )
        grad_g_u.append(
            cp.Parameter((N, n_controls), name="grad_g_u_" + str(idx_ncvx), sparsity=g_u_sp)
        )
        nu_vb.append(cp.Variable(N, name="nu_vb_" + str(idx_ncvx)))  # Virtual Control for VB

    # Linearized Cross-Node Constraints
    g_cross = []
    grad_g_X_cross = []
    grad_g_U_cross = []
    nu_vb_cross = []
    for idx_cross in range(n_cross_node_constraints):
        # Cross-node constraints are single constraints with fixed node references
        g_cross.append(cp.Parameter(name="g_cross_" + str(idx_cross)))
        grad_g_X_cross.append(cp.Parameter((N, n_states), name="grad_g_X_cross_" + str(idx_cross)))
        grad_g_U_cross.append(
            cp.Parameter((N, n_controls), name="grad_g_U_cross_" + str(idx_cross))
        )
        nu_vb_cross.append(
            cp.Variable(name="nu_vb_cross_" + str(idx_cross))
        )  # Virtual Control for VB

    # Applying the affine scaling to state and control
    x_nonscaled = []
    u_nonscaled = []
    dx_nonscaled = []
    du_nonscaled = []
    for k in range(N):
        x_nonscaled.append(S_x @ x[k] + c_x)
        u_nonscaled.append(S_u @ u[k] + c_u)
        dx_nonscaled.append(S_x @ dx[k])
        du_nonscaled.append(S_u @ du[k])

    return CVXPyVariables(
        lam_prox=lam_prox,
        lam_cost=lam_cost,
        lam_vc=lam_vc,
        lam_vb=lam_vb,
        x=x,
        dx=dx,
        x_bar=x_bar,
        x_init=x_init,
        x_term=x_term,
        u=u,
        du=du,
        u_bar=u_bar,
        A_d=A_d,
        B_d=B_d,
        C_d=C_d,
        x_prop_pp=x_prop_pp,
        D_d=D_d,
        E_d=E_d,
        x_prop=x_prop,
        nu=nu,
        g=g,
        grad_g_x=grad_g_x,
        grad_g_u=grad_g_u,
        nu_vb=nu_vb,
        g_cross=g_cross,
        grad_g_X_cross=grad_g_X_cross,
        grad_g_U_cross=grad_g_U_cross,
        nu_vb_cross=nu_vb_cross,
        S_x=S_x,
        inv_S_x=inv_S_x,
        c_x=c_x,
        S_u=S_u,
        inv_S_u=inv_S_u,
        c_u=c_u,
        x_nonscaled=x_nonscaled,
        u_nonscaled=u_nonscaled,
        dx_nonscaled=dx_nonscaled,
        du_nonscaled=du_nonscaled,
    )


def lower_cvxpy_constraints(
    constraints: ConstraintSet,
    x_cvxpy: List,
    u_cvxpy: List,
    parameters: dict = None,
) -> Tuple[List, dict]:
    """Lower symbolic convex constraints to CVXPy constraints.

    Converts symbolic convex constraint expressions to CVXPy constraint objects
    that can be used in the optimal control problem. This function handles both
    nodal constraints (applied at specific trajectory nodes) and cross-node
    constraints (relating multiple nodes).

    Args:
        constraints: ConstraintSet containing nodal_convex and cross_node_convex
        x_cvxpy: List of CVXPy expressions for state at each node (length N).
            Typically the x_nonscaled list from create_cvxpy_variables().
        u_cvxpy: List of CVXPy expressions for controls at each node (length N).
            Typically the u_nonscaled list from create_cvxpy_variables().
        parameters: Optional dict of parameter values to use for any Parameter
            expressions in the constraints. If None, uses Parameter default values.

    Returns:
        Tuple of:
        - List of CVXPy constraint objects ready for the OCP
        - Dict mapping parameter names to their CVXPy Parameter objects

    Example:
        After creating CVXPy variables::

            ocp_vars = create_cvxpy_variables(settings)
            cvxpy_constraints, cvxpy_params = lower_cvxpy_constraints(
                constraint_set,
                ocp_vars.x_nonscaled,
                ocp_vars.u_nonscaled,
                parameters,
            )

    Note:
        This function only processes convex constraints (nodal_convex and
        cross_node_convex). Non-convex constraints are lowered to JAX in
        lower_symbolic_expressions() and handled via linearization in the SCP.
    """
    import cvxpy as cp

    from openscvx.symbolic.expr import Parameter, traverse
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State
    from openscvx.symbolic.lowerers.cvxpy import lower_to_cvxpy

    all_constraints = list(constraints.nodal_convex) + list(constraints.cross_node_convex)

    if not all_constraints:
        return [], {}

    # Collect all unique Parameters across all constraints and create cp.Parameter objects
    all_params = {}

    def collect_params(expr):
        if isinstance(expr, Parameter):
            if expr.name not in all_params:
                # Use value from params dict if provided, otherwise use Parameter's initial value
                if parameters and expr.name in parameters:
                    param_value = parameters[expr.name]
                else:
                    param_value = expr.value

                cvx_param = cp.Parameter(expr.shape, value=param_value, name=expr.name)
                all_params[expr.name] = cvx_param

    # Collect all parameters from all constraints
    for constraint in all_constraints:
        traverse(constraint.constraint, collect_params)

    cvxpy_constraints = []

    # Process nodal constraints
    for constraint in constraints.nodal_convex:
        # nodes should already be validated and normalized in preprocessing
        nodes = constraint.nodes

        # Collect all State and Control variables referenced in the constraint
        state_vars = {}
        control_vars = {}

        def collect_vars(expr):
            if isinstance(expr, State):
                state_vars[expr.name] = expr
            elif isinstance(expr, Control):
                control_vars[expr.name] = expr

        traverse(constraint.constraint, collect_vars)

        # Regular nodal constraint: apply at each specified node
        for node in nodes:
            # Create variable map for this specific node
            variable_map = {}

            if state_vars:
                variable_map["x"] = x_cvxpy[node]

            if control_vars:
                variable_map["u"] = u_cvxpy[node]

            # Add all CVXPy Parameter objects to the variable map
            variable_map.update(all_params)

            # Verify all variables have slices (should be guaranteed by preprocessing)
            for state_name, state_var in state_vars.items():
                if state_var._slice is None:
                    raise ValueError(
                        f"State variable '{state_name}' has no slice assigned. "
                        f"This indicates a bug in the preprocessing pipeline."
                    )

            for control_name, control_var in control_vars.items():
                if control_var._slice is None:
                    raise ValueError(
                        f"Control variable '{control_name}' has no slice assigned. "
                        f"This indicates a bug in the preprocessing pipeline."
                    )

            # Lower the constraint to CVXPy
            cvxpy_constraint = lower_to_cvxpy(constraint.constraint, variable_map)
            cvxpy_constraints.append(cvxpy_constraint)

    # Process cross-node constraints
    for constraint in constraints.cross_node_convex:
        # Collect all State and Control variables referenced in the constraint
        state_vars = {}
        control_vars = {}

        def collect_vars(expr):
            if isinstance(expr, State):
                state_vars[expr.name] = expr
            elif isinstance(expr, Control):
                control_vars[expr.name] = expr

        traverse(constraint.constraint, collect_vars)

        # Cross-node constraint: provide full trajectory
        variable_map = {}

        # Stack all nodes into (N, n_x) and (N, n_u) matrices
        if state_vars:
            variable_map["x"] = cp.vstack(x_cvxpy)

        if control_vars:
            variable_map["u"] = cp.vstack(u_cvxpy)

        # Add all CVXPy Parameter objects to the variable map
        variable_map.update(all_params)

        # Verify all variables have slices
        for state_name, state_var in state_vars.items():
            if state_var._slice is None:
                raise ValueError(
                    f"State variable '{state_name}' has no slice assigned. "
                    f"This indicates a bug in the preprocessing pipeline."
                )

        for control_name, control_var in control_vars.items():
            if control_var._slice is None:
                raise ValueError(
                    f"Control variable '{control_name}' has no slice assigned. "
                    f"This indicates a bug in the preprocessing pipeline."
                )

        # Lower the constraint once - NodeReference handles node indexing internally
        cvxpy_constraint = lower_to_cvxpy(constraint.constraint, variable_map)
        cvxpy_constraints.append(cvxpy_constraint)

    return cvxpy_constraints, all_params


def _lower_dynamics(dynamics_expr) -> Dynamics:
    """Lower symbolic dynamics to a JAX function.

    Converts a symbolic dynamics expression to a JAX function. Jacobians
    (df/dx, df/du) are computed by the discretizer, not here.

    Args:
        dynamics_expr: Symbolic dynamics expression (dx/dt = f(x, u))

    Returns:
        Dynamics object with f.
    """
    dyn_fn = lower_to_jax(dynamics_expr)
    return Dynamics(f=dyn_fn)


def _sync_control_slices(expr: Expr, controls: list[Control]) -> None:
    """Synchronize control slices in an expression to unified-control ordering.

    Some expressions (notably propagation dynamics) can hold deep-copied Control
    nodes. After unified control re-ordering, those copied nodes may carry stale
    slices. This pass rebinds slices by control name before lowering.
    """
    slice_by_name = {control.name: control._slice for control in controls}

    def _visit(node: Expr) -> None:
        if isinstance(node, Control):
            sl = slice_by_name.get(node.name)
            if sl is not None:
                node._slice = sl

    traverse(expr, _visit)


def _lower_jax_constraints(
    constraints: ConstraintSet,
) -> LoweredJaxConstraints:
    """Lower non-convex constraints to JAX functions with gradients.

    Converts symbolic non-convex constraints to JAX callable functions with
    automatically computed gradients for use in SCP linearization.

    Args:
        constraints: ConstraintSet containing nodal and cross_node constraints

    Returns:
        LoweredJaxConstraints with nodal, cross_node, and ctcs lists
    """
    lowered_nodal: List[LoweredNodalConstraint] = []
    lowered_cross_node: List[LoweredCrossNodeConstraint] = []

    # Lower regular nodal constraints
    if len(constraints.nodal) > 0:
        # Convert symbolic constraint expressions to JAX functions
        constraints_nodal_fns = lower_to_jax(constraints.nodal)

        # Create LoweredConstraint objects with Jacobians
        for i, fn in enumerate(constraints_nodal_fns):
            # Apply vectorization to handle (N, n_x) and (N, n_u) inputs
            constraint = LoweredNodalConstraint(
                func=jax.vmap(fn, in_axes=(0, 0, None, None)),
                grad_g_x=jax.vmap(jacfwd(fn, argnums=0), in_axes=(0, 0, None, None)),
                grad_g_u=jax.vmap(jacfwd(fn, argnums=1), in_axes=(0, 0, None, None)),
                nodes=constraints.nodal[i].nodes,
            )
            lowered_nodal.append(constraint)

    # Lower cross-node constraints (trajectory-level)
    for cross_node_constraint in constraints.cross_node:
        # Lower the CrossNodeConstraint - visitor handles wrapping
        constraint_fn = lower_to_jax(cross_node_constraint)

        # Compute Jacobians for trajectory-level function
        grad_g_X = jacfwd(constraint_fn, argnums=0)  # dg/dX - shape (N, n_x)
        grad_g_U = jacfwd(constraint_fn, argnums=1)  # dg/dU - shape (N, n_u)

        cross_node_lowered = LoweredCrossNodeConstraint(
            func=constraint_fn,
            grad_g_X=grad_g_X,
            grad_g_U=grad_g_U,
        )
        lowered_cross_node.append(cross_node_lowered)

    return LoweredJaxConstraints(
        nodal=lowered_nodal,
        cross_node=lowered_cross_node,
        ctcs=list(constraints.ctcs),  # Copy the list
    )


def _contains_node_reference(expr: Expr) -> bool:
    """Check if an expression contains any NodeReference nodes.

    Internal helper for routing constraints during lowering.

    Recursively traverses the expression tree to detect the presence of
    NodeReference nodes, which indicate cross-node constraints.

    Args:
        expr: Expression to check for NodeReference nodes

    Returns:
        True if the expression contains at least one NodeReference, False otherwise

    Example:
        position = State("pos", shape=(3,))

        # Regular expression - no NodeReference
        _contains_node_reference(position)  # False

        # Cross-node expression - has NodeReference
        _contains_node_reference(position.at(10) - position.at(9))  # True
    """
    if isinstance(expr, NodeReference):
        return True

    # Recursively check all children
    for child in expr.children():
        if _contains_node_reference(child):
            return True

    return False


def lower_symbolic_problem(
    problem: "SymbolicProblem",
    solver: "ConvexSolver",
    byof: Optional[dict] = None,
) -> LoweredProblem:
    """Lower symbolic problem specification to executable JAX and CVXPy code.

    This is the main orchestrator for converting a preprocessed SymbolicProblem
    into executable numerical code. It coordinates the lowering of dynamics,
    constraints, and state/control interfaces from symbolic AST representations
    to JAX functions (with automatic differentiation) and CVXPy constraints.

    This is pure translation - no validation, shape checking, or augmentation occurs
    here. The input problem must be preprocessed (problem.is_preprocessed == True).

    Args:
        problem: Preprocessed SymbolicProblem from preprocess_symbolic_problem().
            Must have is_preprocessed == True.
        solver: ConvexSolver instance to create backend-specific variables.
            The solver's ``create_variables()`` method will be called to create
            optimization variables before constraint lowering.
        byof: Optional dict of raw JAX functions for expert users. Supported keys:
            - "nodal_constraints": List of f(x, u, node, params) -> residual
            - "cross_nodal_constraints": List of f(X, U, params) -> residual
            - "ctcs_constraints": List of dicts with "constraint_fn", "penalty", "bounds"

    Returns:
        LoweredProblem dataclass containing lowered problem

    Example:
        After preprocessing::

            solver = CVXPySolver()
            problem = preprocess_symbolic_problem(...)
            lowered = lower_symbolic_problem(problem, solver)

            # Access dynamics
            dx = lowered.dynamics.f(x_val, u_val, node=0, params={...})

            # Solver now owns the CVXPy variables
            ocp_vars = solver.ocp_vars

    Raises:
        AssertionError: If problem.is_preprocessed is False
    """
    assert problem.is_preprocessed, "Problem must be preprocessed before lowering"

    # Create unified state/control interfaces
    x_unified = unify_states(problem.states, name="x")
    u_unified = unify_controls(problem.controls)
    x_prop_unified = unify_states(problem.states_prop, name="x_prop")

    # Keep control slices in copied expressions aligned with unified ordering.
    _sync_control_slices(problem.dynamics, problem.controls)
    _sync_control_slices(problem.dynamics_discrete, problem.controls)
    _sync_control_slices(problem.dynamics_prop, problem.controls)

    # Lower dynamics to JAX
    dynamics = _lower_dynamics(problem.dynamics)
    dynamics_discrete = _lower_dynamics(problem.dynamics_discrete)
    dynamics_prop = _lower_dynamics(problem.dynamics_prop)

    # Lower non-convex constraints to JAX
    jax_constraints = _lower_jax_constraints(problem.constraints)

    # Handle byof (bring-your-own-functions) for expert users
    # This must happen BEFORE CVXPy variable creation since CTCS constraints
    # augment the state dimension
    if byof is not None:
        (
            dynamics,
            dynamics_prop,
            dynamics_discrete,
            jax_constraints,
            x_unified,
            x_prop_unified,
        ) = apply_byof(
            byof,
            dynamics,
            dynamics_prop,
            dynamics_discrete,
            jax_constraints,
            x_unified,
            x_prop_unified,
            u_unified,
            problem.states,
            problem.states_prop,
            problem.N,
        )

    # Compute dynamics Jacobian sparsity from the symbolic AST.
    # Only available for symbolic dynamics (not byof).  Always use FOH
    # patterns (the superset) since dis_type isn't known yet.
    dynamics_sparsity = None
    if byof is None and problem.dynamics is not None:
        from openscvx.symbolic.sparsity import discrete_sparsity

        n_x = sum(s.shape[0] for s in problem.states)
        n_u = sum(c.shape[0] for c in problem.controls)
        A_c, B_c = problem.dynamics.sparsity(n_x, n_u)

        dynamics_sparsity = discrete_sparsity(A_c, B_c, dis_type="FOH")

    # Compute per-constraint Jacobian sparsity from the symbolic AST.
    # Each scalar nodal constraint produces a 1-D mask over x and u.
    # No time-dilation patch needed: constraints take the full u vector.
    constraint_sparsity = None
    if byof is None and problem.constraints.nodal:
        n_x = sum(s.shape[0] for s in problem.states)
        n_u = sum(c.shape[0] for c in problem.controls)
        constraint_sparsity = []
        for nc in problem.constraints.nodal:
            sx, su = nc.constraint.lhs.sparsity(n_x, n_u)
            constraint_sparsity.append((sx.flatten(), su.flatten()))

    # Solver creates its own backend-specific variables
    solver.create_variables(
        N=problem.N,
        x_unified=x_unified,
        u_unified=u_unified,
        jax_constraints=jax_constraints,
        dynamics_sparsity=dynamics_sparsity,
        constraint_sparsity=constraint_sparsity,
    )

    _augment_impulsive_constraints(problem.constraints, problem.controls, problem.N)

    # Lower convex constraints using solver's variables
    lowered_cvxpy_constraint_list, cvxpy_params = lower_cvxpy_constraints(
        problem.constraints,
        solver.ocp_vars.x_nonscaled,
        solver.ocp_vars.u_nonscaled,
        problem.parameters,
    )
    cvxpy_constraints = LoweredCvxpyConstraints(
        constraints=lowered_cvxpy_constraint_list,
    )

    # Lower algebraic outputs to vmapped JAX functions
    algebraic_prop_lowered = {}
    if problem.algebraic_prop:
        for name, expr in problem.algebraic_prop.items():
            # Lower expression to JAX function: f(x, u, node, params) -> output
            output_fn = lower_to_jax(expr)
            # Vmap over time axis: (T, n_x), (T, n_u) -> (T, output_dim)
            output_fn_vmapped = jax.vmap(output_fn, in_axes=(0, 0, None, None))
            algebraic_prop_lowered[name] = output_fn_vmapped

    return LoweredProblem(
        dynamics=dynamics,
        dynamics_discrete=dynamics_discrete,
        dynamics_prop=dynamics_prop,
        jax_constraints=jax_constraints,
        cvxpy_constraints=cvxpy_constraints,
        x_unified=x_unified,
        u_unified=u_unified,
        x_prop_unified=x_prop_unified,
        cvxpy_params=cvxpy_params,
        algebraic_prop=algebraic_prop_lowered,
    )
