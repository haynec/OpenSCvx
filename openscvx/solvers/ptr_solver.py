"""CVXPy-based convex subproblem solver for the penalized trust-region (PTR) SCP algorithm.

This module provides the default solver backend using CVXPy's modeling language
with support for multiple backend solvers (CLARABEL, etc.). Includes optional
code generation via cvxpygen for improved performance.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import cvxpy as cp
import numpy as np

from openscvx.config import Config

from .base import ConvexSolver


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


if TYPE_CHECKING:
    from openscvx.lowered import LoweredProblem
    from openscvx.lowered.cvxpy_variables import CVXPyVariables
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints
    from openscvx.lowered.unified import UnifiedControl, UnifiedState

# Optional cvxpygen import
try:
    from cvxpygen import cpg

    CVXPYGEN_AVAILABLE = True
except ImportError:
    CVXPYGEN_AVAILABLE = False
    cpg = None


class PTRSolver(ConvexSolver):
    """CVXPy-based convex subproblem solver for the PTR algorithm.

    This solver uses CVXPy's modeling language to construct and solve the convex
    subproblems generated at each SCP iteration. It supports multiple backend
    solvers (CLARABEL, ECOS, MOSEK, etc.) and optional code generation via
    cvxpygen for improved performance.

    The solver builds the problem structure once during ``initialize()``, using
    CVXPy Parameters for values that change each iteration. The ``solve()``
    method then solves and returns a structured ``PTRSolveResult``.

    The cost and constraint formulations are defined in the ``cost()`` and
    ``constraints()`` methods, which can be overridden in subclasses to
    customize the convex subproblem. For example::

        class MyPTRSolver(PTRSolver):
            def cost(self, settings, lowered):
                c = super().cost(settings, lowered)
                c += my_extra_term(self._ocp_vars)
                return c

    !!! note "Future Backend Support"

        When adding a new backend (QPAX, COCO, etc.), this class should be
        refactored:

        1. Rename ``PTRSolver`` to ``CVXPyPTRSolver``
        2. Extract ``PTRSolver`` as an abstract base class defining the PTR
           interface (``update_dynamics_linearization``, ``update_constraint_linearizations``,
           ``update_penalties``, ``solve`` returning ``PTRSolveResult``)
        3. Have ``CVXPyPTRSolver`` and the new backend (e.g., ``QPAXPTRSolver``)
           inherit from the abstract ``PTRSolver``

        This keeps the algorithm backend-agnostic while allowing multiple
        solver implementations for the PTR formulation.

    Example:
        Using PTRSolver with the SCP framework::

            solver = PTRSolver()
            solver.create_variables(N, x_unified, u_unified, jax_constraints)
            solver.initialize(lowered, settings)

            # Each iteration (parameter updates done by algorithm):
            result = solver.solve()
            x_sol = result.x  # Unscaled state trajectory

    Attributes:
        ocp_vars: The CVXPy variables and parameters (available after create_variables())
    """

    def __init__(self):
        """Initialize PTRSolver with unset problem.

        Call create_variables() then initialize() to build the problem structure.
        """
        self._ocp_vars: "CVXPyVariables" = None
        self._problem: cp.Problem = None
        self._solve_fn: callable = None

    @property
    def ocp_vars(self) -> "CVXPyVariables":
        """The CVXPy variables and parameters.

        Returns:
            The CVXPyVariables dataclass, or None if create_variables() not called.
        """
        return self._ocp_vars

    def create_variables(
        self,
        N: int,
        x_unified: "UnifiedState",
        u_unified: "UnifiedControl",
        jax_constraints: "LoweredJaxConstraints",
    ) -> None:
        """Create CVXPy optimization variables.

        Creates all CVXPy Variable and Parameter objects needed for the optimal
        control problem. This includes state/control variables, dynamics parameters,
        constraint linearization parameters, and scaling matrices.

        Args:
            N: Number of discretization nodes
            x_unified: Unified state interface with dimensions and scaling bounds
            u_unified: Unified control interface with dimensions and scaling bounds
            jax_constraints: Lowered JAX constraints (for sizing linearization params)
        """
        from openscvx.config import get_affine_scaling_matrices
        from openscvx.symbolic.lower import create_cvxpy_variables

        n_states = len(x_unified.max)
        n_controls = len(u_unified.max)

        # Compute scaling matrices from unified object bounds
        if x_unified.scaling_min is not None:
            lower_x = np.array(x_unified.scaling_min, dtype=float)
        else:
            lower_x = np.array(x_unified.min, dtype=float)

        if x_unified.scaling_max is not None:
            upper_x = np.array(x_unified.scaling_max, dtype=float)
        else:
            upper_x = np.array(x_unified.max, dtype=float)

        S_x, c_x = get_affine_scaling_matrices(n_states, lower_x, upper_x)

        if u_unified.scaling_min is not None:
            lower_u = np.array(u_unified.scaling_min, dtype=float)
        else:
            lower_u = np.array(u_unified.min, dtype=float)

        if u_unified.scaling_max is not None:
            upper_u = np.array(u_unified.scaling_max, dtype=float)
        else:
            upper_u = np.array(u_unified.max, dtype=float)

        S_u, c_u = get_affine_scaling_matrices(n_controls, lower_u, upper_u)

        # Create all CVXPy variables for the OCP
        self._ocp_vars = create_cvxpy_variables(
            N=N,
            n_states=n_states,
            n_controls=n_controls,
            S_x=S_x,
            c_x=c_x,
            S_u=S_u,
            c_u=c_u,
            n_nodal_constraints=len(jax_constraints.nodal),
            n_cross_node_constraints=len(jax_constraints.cross_node),
        )

    def initialize(
        self,
        lowered: "LoweredProblem",
        settings: "Config",
    ) -> None:
        """Build the CVXPy optimal control problem.

        Constructs the complete optimization problem by calling ``cost()`` and
        ``constraints()`` to build the objective and constraint formulations,
        then assembles them into a CVXPy Problem.

        If cvxpygen is enabled in settings, generates compiled solver code
        for improved performance.

        Note:
            ``create_variables()`` must be called before this method.

        Args:
            lowered: Lowered problem containing:
                - ``cvxpy_constraints``: Lowered convex constraints
                - ``jax_constraints``: JAX constraint functions (for structure)
            settings: Configuration object with solver settings

        Raises:
            RuntimeError: If create_variables() has not been called.
        """
        if self._ocp_vars is None:
            raise RuntimeError(
                "PTRSolver.initialize() called before create_variables(). "
                "Call create_variables() first to create optimization variables."
            )

        objective = self.cost(settings, lowered)
        constr = self.constraints(settings, lowered)
        prob = cp.Problem(cp.Minimize(objective), constr)

        if settings.cvx.cvxpygen:
            if not CVXPYGEN_AVAILABLE:
                raise ImportError(
                    "cvxpygen is required for code generation but not installed. "
                    "Install it with: pip install openscvx[cvxpygen] or pip install cvxpygen"
                )
            # Check to see if solver directory exists
            if not os.path.exists("solver"):
                cpg.generate_code(prob, solver=settings.cvx.solver, code_dir="solver", wrapper=True)
            else:
                # Prompt the use to indicate if they wish to overwrite the solver
                # directory or use the existing compiled solver
                if settings.cvx.cvxpygen_override:
                    cpg.generate_code(
                        prob,
                        solver=settings.cvx.solver,
                        code_dir="solver",
                        wrapper=True,
                    )
                else:
                    overwrite = input("Solver directory already exists. Overwrite? (y/n): ")
                    if overwrite.lower() == "y":
                        cpg.generate_code(
                            prob,
                            solver=settings.cvx.solver,
                            code_dir="solver",
                            wrapper=True,
                        )

        self._problem = prob
        self._setup_solve_function(settings)

    def cost(
        self,
        settings: "Config",
        lowered: "LoweredProblem",
    ) -> cp.Expression:
        """Build the cost expression for the convex subproblem.

        Constructs the PTR objective function including:

        - Boundary condition costs (Minimize/Maximize state components)
        - Trust region penalty (deviation from linearization point)
        - Virtual control penalty (dynamics defect relaxation)
        - Virtual buffer penalty (nonconvex constraint violation relaxation)

        Override this method in subclasses to customize the cost formulation.
        Use ``super().cost(settings, lowered)`` to include the standard PTR
        cost terms and add to them.

        Args:
            settings: Configuration object with solver settings
            lowered: Lowered problem containing constraint structure

        Returns:
            CVXPy expression representing the total cost to minimize.
        """
        ocp_vars = self._ocp_vars
        jax_constraints = lowered.jax_constraints

        lam_prox = ocp_vars.lam_prox
        lam_cost = ocp_vars.lam_cost
        lam_vc = ocp_vars.lam_vc
        lam_vb = ocp_vars.lam_vb
        _ = ocp_vars.x_nonscaled
        dx = ocp_vars.dx
        du = ocp_vars.du
        nu = ocp_vars.nu
        nu_vb = ocp_vars.nu_vb
        nu_vb_cross = ocp_vars.nu_vb_cross

        cost = lam_cost * 0
        cost += lam_vb * 0

        # Boundary condition cost terms (use scaled x for numerical conditioning)
        x = ocp_vars.x
        for i in range(settings.sim.true_state_slice.start, settings.sim.true_state_slice.stop):
            if settings.sim.x.initial_type[i] == "Minimize":
                cost += lam_cost * x[0][i]
            if settings.sim.x.final_type[i] == "Minimize":
                cost += lam_cost * x[-1][i]
            if settings.sim.x.initial_type[i] == "Maximize":
                cost -= lam_cost * x[0][i]
            if settings.sim.x.final_type[i] == "Maximize":
                cost -= lam_cost * x[-1][i]

        # Trust Region Cost
        cost += sum(
            lam_prox * cp.sum_squares(cp.hstack((dx[i], du[i]))) for i in range(settings.scp.n)
        )

        # Virtual Control Slack
        cost += sum(cp.sum(lam_vc[i - 1] * cp.abs(nu[i - 1])) for i in range(1, settings.scp.n))

        # Virtual buffer penalty for nodal constraints
        idx_ncvx = 0
        if jax_constraints.nodal:
            for constraint in jax_constraints.nodal:
                cost += lam_vb * cp.sum(cp.pos(nu_vb[idx_ncvx]))
                idx_ncvx += 1

        # Virtual slack penalty for cross-node constraints
        idx_cross = 0
        if jax_constraints.cross_node:
            for constraint in jax_constraints.cross_node:
                cost += lam_vb * cp.pos(nu_vb_cross[idx_cross])
                idx_cross += 1

        return cost

    def constraints(
        self,
        settings: "Config",
        lowered: "LoweredProblem",
    ) -> list:
        """Build the constraint list for the convex subproblem.

        Constructs all PTR constraints including:

        - Linearized nodal constraints (from JAX-lowered nonconvex constraints)
        - Linearized cross-node constraints
        - Convex constraints (already lowered to CVXPy)
        - Boundary conditions (fixed initial/terminal states)
        - Uniform time grid constraints
        - State and control deviation definitions
        - Linearized dynamics
        - State and control box constraints
        - CTCS constraints

        Override this method in subclasses to customize the constraint
        formulation. Use ``super().constraints(settings, lowered)`` to include
        the standard PTR constraints and extend them.

        Args:
            settings: Configuration object with solver settings
            lowered: Lowered problem containing lowered constraints

        Returns:
            List of CVXPy constraints.
        """
        ocp_vars = self._ocp_vars
        jax_constraints = lowered.jax_constraints
        cvxpy_constraints = lowered.cvxpy_constraints

        x = ocp_vars.x
        dx = ocp_vars.dx
        x_bar = ocp_vars.x_bar
        x_init = ocp_vars.x_init
        x_term = ocp_vars.x_term
        u = ocp_vars.u
        du = ocp_vars.du
        u_bar = ocp_vars.u_bar
        A_d = ocp_vars.A_d
        B_d = ocp_vars.B_d
        C_d = ocp_vars.C_d
        x_prop = ocp_vars.x_prop
        nu = ocp_vars.nu
        g = ocp_vars.g
        grad_g_x = ocp_vars.grad_g_x
        grad_g_u = ocp_vars.grad_g_u
        nu_vb = ocp_vars.nu_vb
        g_cross = ocp_vars.g_cross
        grad_g_X_cross = ocp_vars.grad_g_X_cross
        grad_g_U_cross = ocp_vars.grad_g_U_cross
        nu_vb_cross = ocp_vars.nu_vb_cross
        inv_S_x = ocp_vars.inv_S_x
        c_x = ocp_vars.c_x
        inv_S_u = ocp_vars.inv_S_u
        c_u = ocp_vars.c_u
        x_nonscaled = ocp_vars.x_nonscaled
        u_nonscaled = ocp_vars.u_nonscaled
        dx_nonscaled = ocp_vars.dx_nonscaled
        du_nonscaled = ocp_vars.du_nonscaled

        constr = []

        # Linearized nodal constraints (from JAX-lowered non-convex)
        idx_ncvx = 0
        if jax_constraints.nodal:
            for constraint in jax_constraints.nodal:
                # nodes should already be validated and normalized in preprocessing
                nodes = constraint.nodes
                constr += [
                    (
                        g[idx_ncvx][node]
                        + grad_g_x[idx_ncvx][node] @ dx[node]
                        + grad_g_u[idx_ncvx][node] @ du[node]
                    )
                    == nu_vb[idx_ncvx][node]
                    for node in nodes
                ]
                idx_ncvx += 1

        # Linearized cross-node constraints (from JAX-lowered non-convex)
        idx_cross = 0
        if jax_constraints.cross_node:
            for constraint in jax_constraints.cross_node:
                # Linearization: g(X_bar, U_bar) + ∇g_X @ dX + ∇g_U @ dU == nu_vb
                # Sum over all trajectory nodes to couple multiple nodes
                residual = g_cross[idx_cross]
                for k in range(settings.scp.n):
                    # Contribution from state at node k
                    residual += grad_g_X_cross[idx_cross][k, :] @ dx[k]
                    # Contribution from control at node k
                    residual += grad_g_U_cross[idx_cross][k, :] @ du[k]
                # Add constraint: residual == slack variable
                constr += [residual == nu_vb_cross[idx_cross]]
                idx_cross += 1

        # Convex constraints (already lowered to CVXPy)
        if cvxpy_constraints.constraints:
            constr += cvxpy_constraints.constraints

        # Boundary conditions (Fix)
        for i in range(settings.sim.true_state_slice.start, settings.sim.true_state_slice.stop):
            if settings.sim.x.initial_type[i] == "Fix":
                constr += [x_nonscaled[0][i] == x_init[i]]  # Initial Boundary Conditions
            if settings.sim.x.final_type[i] == "Fix":
                constr += [x_nonscaled[-1][i] == x_term[i]]  # Final Boundary Conditions

        if settings.scp.uniform_time_grid:
            S_u_inv_td = inv_S_u[settings.sim.time_dilation_slice, settings.sim.time_dilation_slice]
            c_u_td = c_u[settings.sim.time_dilation_slice]
            constr += [
                S_u_inv_td @ (u_nonscaled[i][settings.sim.time_dilation_slice] - c_u_td)
                == S_u_inv_td @ (u_nonscaled[i - 1][settings.sim.time_dilation_slice] - c_u_td)
                for i in range(1, settings.scp.n)
            ]

        constr += [
            (x[i] - inv_S_x @ (x_bar[i] - c_x) - dx[i]) == 0 for i in range(settings.scp.n)
        ]  # State Error
        constr += [
            (u[i] - inv_S_u @ (u_bar[i] - c_u) - du[i]) == 0 for i in range(settings.scp.n)
        ]  # Control Error

        constr += [
            inv_S_x @ (x_nonscaled[i] - c_x)
            == inv_S_x
            @ (
                A_d[i - 1] @ dx_nonscaled[i - 1]
                + B_d[i - 1] @ du_nonscaled[i - 1]
                + C_d[i - 1] @ du_nonscaled[i]
                + x_prop[i - 1]
                - c_x
            )
            + nu[i - 1]
            for i in range(1, settings.scp.n)
        ]  # Dynamics Constraint

        constr += [
            inv_S_u @ (u_nonscaled[i] - c_u) <= inv_S_u @ (settings.sim.u.max - c_u)
            for i in range(settings.scp.n)
        ]
        constr += [
            inv_S_u @ (u_nonscaled[i] - c_u) >= inv_S_u @ (settings.sim.u.min - c_u)
            for i in range(settings.scp.n)
        ]  # Control Constraints

        # TODO: (norrisg) formalize this
        constr += [
            inv_S_x @ (x_nonscaled[i][:] - c_x) <= inv_S_x @ (settings.sim.x.max - c_x)
            for i in range(settings.scp.n)
        ]
        constr += [
            inv_S_x @ (x_nonscaled[i][:] - c_x) >= inv_S_x @ (settings.sim.x.min - c_x)
            for i in range(settings.scp.n)
        ]  # State Constraints (Also implemented in CTCS but included for numerical stability)

        for idx, nodes in zip(
            np.arange(settings.sim.ctcs_slice.start, settings.sim.ctcs_slice.stop),
            settings.sim.ctcs_node_intervals,
        ):
            start_idx = 1 if nodes[0] == 0 else nodes[0]
            constr += [
                cp.abs(x_nonscaled[i][idx] - x_nonscaled[i - 1][idx]) <= settings.sim.x.max[idx]
                for i in range(start_idx, nodes[1])
            ]
            constr += [x_nonscaled[0][idx] == 0]

        return constr

    def _setup_solve_function(self, settings: "Config") -> None:
        """Configure the solve function based on settings.

        Sets up either cvxpygen-based solving or standard CVXPy solving
        based on the configuration.

        Args:
            settings: Configuration object with solver settings
        """
        if settings.cvx.cvxpygen:
            try:
                import pickle

                from solver.cpg_solver import cpg_solve

                with open("solver/problem.pickle", "rb") as f:
                    pickle.load(f)
                self._problem.register_solve("CPG", cpg_solve)
                solver_args = settings.cvx.solver_args
                self._solve_fn = lambda: self._problem.solve(method="CPG", **solver_args)
            except ImportError:
                raise ImportError(
                    "cvxpygen solver not found. Make sure cvxpygen is installed and code "
                    "generation has been run. Install with: pip install openscvx[cvxpygen]"
                )
        else:
            solver = settings.cvx.solver
            solver_args = settings.cvx.solver_args
            self._solve_fn = lambda: self._problem.solve(solver=solver, **solver_args)

    def update_dynamics_linearization(
        self,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        A_d: np.ndarray,
        B_d: np.ndarray,
        C_d: np.ndarray,
        x_prop: np.ndarray,
    ) -> None:
        """Update dynamics linearization point and matrices.

        Sets the current linearization point (previous iterate) and the
        discretized dynamics matrices for the convex subproblem.

        Args:
            x_bar: Previous state trajectory, shape (N, n_states)
            u_bar: Previous control trajectory, shape (N, n_controls)
            A_d: Discretized state Jacobian, shape (N-1, n_states, n_states)
            B_d: Discretized control Jacobian (current node), shape (N-1, n_states, n_controls)
            C_d: Discretized control Jacobian (next node), shape (N-1, n_states, n_controls)
            x_prop: Propagated state from dynamics, shape (N-1, n_states)
        """
        self._set_param("x_bar", x_bar)
        self._set_param("u_bar", u_bar)
        self._set_param("A_d", A_d)
        self._set_param("B_d", B_d)
        self._set_param("C_d", C_d)
        self._set_param("x_prop", x_prop)

    def update_constraint_linearizations(
        self,
        nodal: List[dict] = None,
        cross_node: List[dict] = None,
    ) -> None:
        """Update linearized constraint values and gradients.

        Sets constraint function values and gradients at the current
        linearization point for both nodal and cross-node constraints.

        Args:
            nodal: List of dicts for nodal constraints, each containing:
                - ``g``: Constraint value at linearization point
                - ``grad_g_x``: Gradient w.r.t. state
                - ``grad_g_u``: Gradient w.r.t. control
            cross_node: List of dicts for cross-node constraints, each containing:
                - ``g``: Constraint value at linearization point
                - ``grad_g_X``: Gradient w.r.t. full state trajectory
                - ``grad_g_U``: Gradient w.r.t. full control trajectory
        """
        if nodal:
            for g_id, constraint_data in enumerate(nodal):
                self._set_param(f"g_{g_id}", constraint_data["g"])
                self._set_param(f"grad_g_x_{g_id}", constraint_data["grad_g_x"])
                self._set_param(f"grad_g_u_{g_id}", constraint_data["grad_g_u"])

        if cross_node:
            for g_id, constraint_data in enumerate(cross_node):
                self._set_param(f"g_cross_{g_id}", constraint_data["g"])
                self._set_param(f"grad_g_X_cross_{g_id}", constraint_data["grad_g_X"])
                self._set_param(f"grad_g_U_cross_{g_id}", constraint_data["grad_g_U"])

    def update_penalties(
        self,
        lam_prox: float,
        lam_cost: float,
        lam_vc: np.ndarray,
        lam_vb: float,
    ) -> None:
        """Update SCP penalty weights.

        Sets the penalty weights that balance competing objectives in the
        PTR convex subproblem.

        Args:
            lam_prox: Trust region weight (penalizes deviation from linearization point)
            lam_cost: Cost function weight
            lam_vc: Virtual control penalty weights, shape (N-1, n_states)
            lam_vb: Virtual buffer penalty weight (for constraint violations)
        """
        self._set_param("lam_prox", lam_prox)
        self._set_param("lam_cost", lam_cost)
        self._set_param("lam_vc", lam_vc)
        self._set_param("lam_vb", lam_vb)

    def update_boundary_conditions(
        self,
        x_init: np.ndarray = None,
        x_term: np.ndarray = None,
    ) -> None:
        """Update boundary condition parameters.

        Sets initial and/or terminal state constraints. Only sets parameters
        that exist in the problem (some problems may not have both).

        Args:
            x_init: Initial state vector, shape (n_states,). Optional.
            x_term: Terminal state vector, shape (n_states,). Optional.
        """
        if x_init is not None and "x_init" in self._problem.param_dict:
            self._set_param("x_init", x_init)
        if x_term is not None and "x_term" in self._problem.param_dict:
            self._set_param("x_term", x_term)

    def get_stats(self) -> dict:
        """Get solver statistics for diagnostics and printing.

        Returns:
            Dict containing:
                - ``n_variables``: Total number of optimization variables
                - ``n_parameters``: Total number of parameters
                - ``n_constraints``: Total number of constraints
        """
        if self._problem is None:
            return {"n_variables": 0, "n_parameters": 0, "n_constraints": 0}

        return {
            "n_variables": sum(var.size for var in self._problem.variables()),
            "n_parameters": sum(param.size for param in self._problem.parameters()),
            "n_constraints": sum(constraint.size for constraint in self._problem.constraints),
        }

    def _set_param(self, name: str, value: np.ndarray) -> None:
        """Set a CVXPy parameter with helpful error messages on failure.

        Args:
            name: The parameter name in problem.param_dict
            value: The value to assign

        Raises:
            ValueError: If the value is not real, with diagnostic information.
        """
        try:
            param = self._problem.param_dict[name]
            value_arr = np.asarray(value)

            # Ensure the value shape matches the parameter shape exactly
            # This is critical for Python 3.11+ where NumPy/CVXPy are stricter about shapes
            if hasattr(param, "shape") and param.shape is not None:
                expected_shape = param.shape
                if value_arr.shape != expected_shape:
                    # Try to reshape if sizes match
                    if value_arr.size == np.prod(expected_shape):
                        value_arr = value_arr.reshape(expected_shape)
                    else:
                        # If sizes don't match, try squeezing extra dimensions first
                        value_arr = np.squeeze(value_arr)
                        if value_arr.shape != expected_shape and value_arr.size == np.prod(
                            expected_shape
                        ):
                            value_arr = value_arr.reshape(expected_shape)
                        elif value_arr.shape != expected_shape:
                            raise ValueError(
                                f"Parameter '{name}' shape mismatch: expected {expected_shape}, "
                                f"got {value.shape} (after squeezing: {value_arr.shape})"
                            )

            param.value = value_arr
        except ValueError as e:
            if "must be real" in str(e):
                arr = np.asarray(value)
                nan_mask = ~np.isfinite(arr)
                nan_indices = np.argwhere(nan_mask)

                index_value_strs = [
                    f"  {tuple(int(i) for i in idx)} -> {arr[tuple(idx)]}"
                    for idx in nan_indices[:20]
                ]
                if len(nan_indices) > 20:
                    index_value_strs.append(f"  ... and {len(nan_indices) - 20} more")

                arr_str = np.array2string(arr, threshold=200, edgeitems=3, max_line_width=120)
                msg = (
                    f"Parameter '{name}' with shape {arr.shape} contains "
                    f"{len(nan_indices)} non-real value(s):\n"
                    + "\n".join(index_value_strs)
                    + f"\n\n{name} = {arr_str}"
                )
                raise ValueError(msg) from e
            raise

    def solve(self) -> PTRSolveResult:
        """Solve the convex subproblem and return structured results.

        Call ``update_dynamics_linearization()``, ``update_constraint_linearizations()``,
        and ``update_penalties()`` before calling this method.

        Returns:
            PTRSolveResult containing unscaled trajectories, slack variables,
            cost, and solver status.

        Raises:
            RuntimeError: If initialize() has not been called.
        """
        if self._problem is None:
            raise RuntimeError(
                "PTRSolver.solve() called before initialize(). "
                "Call initialize() first to build the problem structure."
            )

        self._solve_fn()

        # Get scaling matrices
        S_x = self._ocp_vars.S_x
        c_x = self._ocp_vars.c_x
        S_u = self._ocp_vars.S_u
        c_u = self._ocp_vars.c_u

        # Unscale state and control trajectories
        x_scaled = self._problem.var_dict["x"].value  # (N, n_states)
        u_scaled = self._problem.var_dict["u"].value  # (N, n_controls)
        x = (S_x @ x_scaled.T + np.expand_dims(c_x, axis=1)).T
        u = (S_u @ u_scaled.T + np.expand_dims(c_u, axis=1)).T

        # Get virtual control slack
        nu = self._problem.var_dict["nu"].value

        # Get nodal constraint violation slacks
        nu_vb = [var.value for var in self._ocp_vars.nu_vb]

        # Get cross-node constraint violation slacks
        nu_vb_cross = [var.value for var in self._ocp_vars.nu_vb_cross]

        return PTRSolveResult(
            x=x,
            u=u,
            nu=nu,
            nu_vb=nu_vb,
            nu_vb_cross=nu_vb_cross,
            cost=self._problem.value,
            status=self._problem.status,
        )

    def citation(self) -> List[str]:
        """Return BibTeX citations for CVXPy.

        Returns:
            List containing BibTeX entries for CVXPy and DCCP papers.
        """
        return [
            r"""@article{diamond2016cvxpy,
  title={CVXPY: A Python-embedded modeling language for convex optimization},
  author={Diamond, Steven and Boyd, Stephen},
  journal={Journal of Machine Learning Research},
  volume={17},
  number={83},
  pages={1--5},
  year={2016}
}""",
            r"""@article{agrawal2018rewriting,
  title={A rewriting system for convex optimization problems},
  author={Agrawal, Akshay and Verschueren, Robin and Diamond, Steven and Boyd, Stephen},
  journal={Journal of Control and Decision},
  volume={5},
  number={1},
  pages={42--60},
  year={2018},
  publisher={Taylor \& Francis}
}""",
        ]
