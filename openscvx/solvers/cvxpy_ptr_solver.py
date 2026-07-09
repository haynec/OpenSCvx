"""CVXPy-based convex subproblem solver for the penalized trust-region (PTR) SCP algorithm.

This module provides the default backend for :class:`PTRSolver`, using CVXPy's
modeling language and dispatching to any of its supported conic solvers
(QOCO, CLARABEL, ...). Optional code generation via cvxpygen is available
for improved per-iteration performance.

Companion backend: :class:`openscvx.solvers.qpax_ptr_solver.QPAXPTRSolver`,
which targets pure-JAX execution.
"""

import os
import warnings
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np

from openscvx.config import Config

from .ptr_solver import (
    ProxConvexSubproblemData,
    PTRSolver,
    PTRSolveResult,
    StatusCode,
    SubproblemData,
    SubproblemSolution,
    status_str_to_code,
)

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

# CVXPY noise emitted by this backend's subproblem assembly and solve, accepted
# knowingly and silenced by message so every other warning stays visible. The
# ``*``-matmul deprecation is our own usage (fixing it means auditing the
# assembly expressions — tracked separately); the sparse-access warnings fire
# inside CVXPY when we read multi-dim parameter values.
warnings.filterwarnings(
    "ignore", message=r"\s*This use of ``\*`` has resulted in matrix multiplication"
)
warnings.filterwarnings("ignore", message=r"Reading from a sparse CVXPY expression via `\.value`")
warnings.filterwarnings(
    "ignore", message=r"Accessing a sparse CVXPY expression via a dense representation"
)
warnings.filterwarnings(
    "ignore", message=r"The problem has an expression with dimension greater than 2"
)


## TODO: (fabio) add support for impulsive controls


def _unstack_nodal(
    data: SubproblemData,
    jax_constraints: "LoweredJaxConstraints",
) -> List[dict]:
    """Unstack ``SubproblemData``'s ``nodal_*`` arrays into per-constraint dicts.

    The JAX-pure side stacks every nodal constraint's linearization into
    ``(N, n_nodal, ...)`` arrays with zero-fill at nodes outside each
    constraint's static ``nodes`` tuple. :meth:`update_constraint_linearizations`
    expects the list-of-dicts layout the SCP loop has historically built —
    this helper inverts the stack. The full-N arrays are returned (the zero
    rows are harmless: CVXPy's constraint set only references the
    constraint's own ``nodes``).
    """
    if not jax_constraints.nodal:
        return []
    nodal_g = np.asarray(data.nodal_g)
    nodal_grad_x = np.asarray(data.nodal_grad_x)
    nodal_grad_u = np.asarray(data.nodal_grad_u)
    out: List[dict] = []
    for c_idx, _constraint in enumerate(jax_constraints.nodal):
        out.append(
            {
                "g": nodal_g[:, c_idx],
                "grad_g_x": nodal_grad_x[:, c_idx, :],
                "grad_g_u": nodal_grad_u[:, c_idx, :],
            }
        )
    return out


def _unstack_cross(
    data: SubproblemData,
    jax_constraints: "LoweredJaxConstraints",
) -> List[dict]:
    """Unstack ``SubproblemData``'s cross-node arrays into per-constraint dicts.

    Cross-node constraints stack ``g`` to ``(n_cross,)`` and the gradients to
    ``(n_cross, N, n_x | n_u)``. :meth:`update_constraint_linearizations`
    expects per-constraint dicts with the natural orientation.
    """
    if not jax_constraints.cross_node:
        return []
    cross_g = np.asarray(data.cross_g)
    cross_grad_X = np.asarray(data.cross_grad_X)
    cross_grad_U = np.asarray(data.cross_grad_U)
    out: List[dict] = []
    for c_idx, _constraint in enumerate(jax_constraints.cross_node):
        out.append(
            {
                "g": cross_g[c_idx],
                "grad_g_X": cross_grad_X[c_idx],
                "grad_g_U": cross_grad_U[c_idx],
            }
        )
    return out


def _unmask_bc(bc: "jnp.ndarray") -> np.ndarray:
    """Replace NaN-sentinel free entries with zeros for CVXPy parameter assignment.

    ``SubproblemData.x_init`` / ``x_term`` carry ``jnp.nan`` at indices where
    the boundary condition is free; CVXPy's ``Parameter.value`` setter rejects
    non-real arrays via :meth:`CVXPyPTRSolver._set_param`. The CVXPy
    constraint set only references the indices flagged ``"Fix"`` in
    ``initial_type`` / ``final_type``, so zeroing the free entries is safe —
    they're written but never read.
    """
    arr = np.asarray(bc, dtype=float)
    return np.nan_to_num(arr, nan=0.0)


class CVXPyPTRSolver(PTRSolver):
    """CVXPy-backed implementation of the PTR convex subproblem.

    Builds the subproblem as a DCP program through CVXPy and dispatches it to
    one of CVXPy's supported conic solvers (QOCO by default, CLARABEL, etc.).
    Optional code generation via cvxpygen is available for improved per-iteration
    performance.

    The solver builds the problem structure once during ``initialize()``, using
    CVXPy Parameters for values that change each iteration. The ``solve()``
    method then solves and returns a structured ``PTRSolveResult``. The
    JAX-pure entry point :meth:`iteration_callback` wraps that same solve in
    :func:`jax.pure_callback` (``vmap_method="sequential"`` — CVXPy cannot
    batch) so the backend composes with ``jax.jit`` / ``jax.vmap`` alongside
    the JAX-native QPAX and Moreau backends.

    The cost and constraint formulations are defined in the ``cost()`` and
    ``constraints()`` methods, which can be overridden in subclasses to
    customize the convex subproblem. For example::

        class MyPTRSolver(CVXPyPTRSolver):
            def cost(self, settings, lowered):
                c = super().cost(settings, lowered)
                c += my_extra_term(self._ocp_vars)
                return c

    Example:
        Using CVXPyPTRSolver with the SCP framework::

            solver = CVXPyPTRSolver()
            solver.create_variables(N, x_unified, u_unified, jax_constraints)
            solver.initialize(lowered, settings)

            # Each iteration (parameter updates done by algorithm):
            result = solver.solve()
            x_sol = result.x  # Unscaled state trajectory

    Args:
        cvx_solver: CVXPY solver backend name. Defaults to ``"QOCO"``.
        solver_args: Keyword arguments forwarded to the CVXPY solver
            (e.g. tolerances). Defaults to
            ``{"abstol": 1e-6, "reltol": 1e-9, "enforce_dpp": True}``.
        cvxpygen: Enable CVXPy code generation for faster solves.
            Defaults to ``False``.

            !!! warning
                Enabling cvxpygen currently disables sparse parameter
                declarations. cvxpygen does not yet support the N-D sparsity
                indices used by OpenSCvx's tiled parameters, so all parameters
                are created as dense when code generation is active. This may
                increase the generated solver's memory footprint and compile
                time but does not affect solution correctness.
        cvxpygen_override: Overwrite existing generated solver directory
            without prompting. Defaults to ``False``.

    Attributes:
        ocp_vars: The CVXPy variables and parameters (available after create_variables())
    """

    def __init__(
        self,
        cvx_solver: str = "QOCO",
        solver_args: Optional[dict] = None,
        cvxpygen: bool = False,
        cvxpygen_override: bool = False,
    ):
        """Initialize CVXPyPTRSolver with solver configuration.

        Call create_variables() then initialize() to build the problem structure.
        """
        self.cvx_solver = cvx_solver
        self.solver_args = (
            solver_args
            if solver_args is not None
            else {"abstol": 1e-06, "reltol": 1e-09, "enforce_dpp": True}
        )
        self.cvxpygen = cvxpygen
        self.cvxpygen_override = cvxpygen_override

        self._ocp_vars: "CVXPyVariables" = None
        self._problem: cp.Problem = None
        self._solve_fn: callable = None
        # Stashed at create_variables / initialize so iteration_callback can
        # close over them without retracing them off the ``lowered`` /
        # ``settings`` arguments each iteration.
        self._jax_constraints: Optional["LoweredJaxConstraints"] = None
        self._settings: Optional[Config] = None
        self._slice_cont = slice(0, 0)
        self._slice_imp = slice(0, 0)

    @property
    def exportable(self) -> bool:
        """CVXPy is not ``jax.export``-serializable — see :attr:`PTRSolver.exportable`.

        The JAX-pure :meth:`iteration_callback` wraps the host CVXPy solve in a
        :func:`jax.pure_callback`, and ``jax.export`` cannot serialize host
        callbacks. ``solve_batched(save_compiled=True)`` therefore refuses this
        backend with a teaching error pointing at QPAX / Moreau rather than
        silently falling back to an uncached in-process solve.
        """
        return False

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
        dynamics_sparsity: Optional[tuple] = None,
        constraint_sparsity: Optional[list] = None,
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
            dynamics_sparsity: Optional tuple ``(A_d, B_d, C_d)`` of boolean
                ndarrays giving the discrete-time Jacobian sparsity patterns.
                ``A_d`` has shape ``(n_x, n_x)``; ``B_d`` and ``C_d`` have
                shape ``(n_x, n_u)``.
            constraint_sparsity: Optional list of ``(x_mask, u_mask)`` boolean
                1-D arrays, one per nodal constraint.
        """
        from openscvx.symbolic.lower import _tile_sparsity, create_cvxpy_variables

        n_states = len(x_unified.max)
        n_controls = len(u_unified.max)
        slice_cont = u_unified.slice_continuous
        slice_imp = u_unified.slice_impulsive
        n_controls_cont = int(slice_cont.stop - slice_cont.start)
        n_controls_imp = int(slice_imp.stop - slice_imp.start)
        if n_controls_cont + n_controls_imp != n_controls:
            raise ValueError(
                "Unified control slices are inconsistent with control dimension. "
                f"continuous={n_controls_cont}, impulsive={n_controls_imp}, total={n_controls}."
            )
        self._slice_cont = slice_cont
        self._slice_imp = slice_imp

        S_x, c_x = self._scaling(x_unified)
        S_u, c_u = self._scaling(u_unified)

        # Convert boolean sparsity patterns to CVXPY index format
        A_d_sp = B_d_sp = C_d_sp = None
        if dynamics_sparsity is not None:
            A_d_pat, B_d_pat, C_d_pat = dynamics_sparsity
            A_d_sp = _tile_sparsity(A_d_pat, N - 1)
            B_d_sp = _tile_sparsity(B_d_pat, N - 1)
            C_d_sp = _tile_sparsity(C_d_pat, N - 1)

        # TODO: (griffin-norris) Remove once cvxpygen supports N-D sparsity
        # indices. cvxpygen's handle_sparsity() assumes 2-D (rows, cols) but
        # our tiled parameters produce 3-D indices (slices, rows, cols).
        # Dropping sparsity here is safe — it only affects codegen performance.
        if self.cvxpygen:
            A_d_sp = B_d_sp = C_d_sp = None
            constraint_sparsity = None

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
            A_d_sparsity=A_d_sp,
            B_d_sparsity=B_d_sp,
            C_d_sparsity=C_d_sp,
            constraint_sparsity=constraint_sparsity,
        )

        # Stash the constraint catalog so iteration_callback can unstack the
        # JAX-pure ``SubproblemData`` back into the per-constraint dict layout
        # ``update_constraint_linearizations`` expects.
        self._jax_constraints = jax_constraints

    def lower_convex_constraints(self, constraints, parameters=None):
        """Lower user ``.convex()`` constraints into CVXPy constraint objects.

        Delegates to :func:`openscvx.symbolic.lower.lower_cvxpy_constraints`,
        feeding it the unscaled-state and unscaled-control CVXPy expressions
        built by :meth:`create_variables`.
        """
        from openscvx.symbolic.lower import lower_cvxpy_constraints

        if self._ocp_vars is None:
            raise RuntimeError(
                "CVXPyPTRSolver.lower_convex_constraints() called before "
                "create_variables(); the CVXPy variables it needs don't "
                "exist yet."
            )
        return lower_cvxpy_constraints(
            constraints,
            self._ocp_vars.x_nonscaled,
            self._ocp_vars.u_nonscaled,
            parameters,
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

        If cvxpygen is enabled, generates compiled solver code for improved
        performance.

        Note:
            ``create_variables()`` must be called before this method.

        Args:
            lowered: Lowered problem containing:
                - ``cvxpy_constraints``: Lowered convex constraints
                - ``jax_constraints``: JAX constraint functions (for structure)
            settings: Problem configuration (node count, scaling, etc.)

        Raises:
            RuntimeError: If create_variables() has not been called.
        """
        if self._ocp_vars is None:
            raise RuntimeError(
                "CVXPyPTRSolver.initialize() called before create_variables(). "
                "Call create_variables() first to create optimization variables."
            )

        objective = self.cost(settings, lowered)
        constr = self.constraints(settings, lowered)
        prob = cp.Problem(cp.Minimize(objective), constr)

        if self.cvxpygen:
            if not CVXPYGEN_AVAILABLE:
                raise ImportError(
                    "cvxpygen is required for code generation but not installed. "
                    "Install it with: pip install openscvx[cvxpygen] or pip install cvxpygen"
                )
            # Check to see if solver directory exists
            if not os.path.exists("solver"):
                cpg.generate_code(prob, solver=self.cvx_solver, code_dir="solver", wrapper=True)
            else:
                # Prompt the use to indicate if they wish to overwrite the solver
                # directory or use the existing compiled solver
                if self.cvxpygen_override:
                    cpg.generate_code(
                        prob,
                        solver=self.cvx_solver,
                        code_dir="solver",
                        wrapper=True,
                    )
                else:
                    overwrite = input("Solver directory already exists. Overwrite? (y/n): ")
                    if overwrite.lower() == "y":
                        cpg.generate_code(
                            prob,
                            solver=self.cvx_solver,
                            code_dir="solver",
                            wrapper=True,
                        )

        self._problem = prob
        self._settings = settings
        self._setup_solve_function()

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
        prox_c = ocp_vars.prox_c
        prox_cc = ocp_vars.prox_cc
        lam_cost = ocp_vars.lam_cost
        lam_vc = ocp_vars.lam_vc
        lam_vb_nodal = ocp_vars.lam_vb_nodal
        lam_vb_cross = ocp_vars.lam_vb_cross
        x = ocp_vars.x
        u = ocp_vars.u
        nu = ocp_vars.nu
        nu_vb = ocp_vars.nu_vb
        nu_vb_cross = ocp_vars.nu_vb_cross

        cost = cp.sum(lam_cost) * 0
        cost += cp.sum(lam_vb_nodal) * 0
        cost += cp.sum(lam_vb_cross) * 0

        # Boundary condition cost terms (use scaled x for numerical conditioning)
        for i in range(settings.sim.true_state_slice.start, settings.sim.true_state_slice.stop):
            if settings.sim.x.initial_type[i] == "Minimize":
                cost += lam_cost[i] * x[0][i]
            if settings.sim.x.final_type[i] == "Minimize":
                cost += lam_cost[i] * x[-1][i]
            if settings.sim.x.initial_type[i] == "Maximize":
                cost -= lam_cost[i] * x[0][i]
            if settings.sim.x.final_type[i] == "Maximize":
                cost -= lam_cost[i] * x[-1][i]

        # Trust-region cost in expanded form:
        #   sum_i [lam_i * z_i^2 + prox_c_i * z_i + prox_cc_i], z_i = [x_i, u_i].
        cost += sum(
            cp.sum(cp.multiply(lam_prox[i], cp.square(cp.hstack((x[i], u[i])))))
            + cp.sum(cp.multiply(prox_c[i], cp.hstack((x[i], u[i]))))
            + prox_cc[i]
            for i in range(settings.sim.n)
        )

        # Virtual Control Slack
        cost += sum(cp.sum(lam_vc[i - 1] * cp.abs(nu[i - 1])) for i in range(1, settings.sim.n))

        # Virtual buffer penalty for nodal constraints (per-node weighting)
        idx_ncvx = 0
        if jax_constraints.nodal:
            for constraint in jax_constraints.nodal:
                cost += lam_vb_nodal[:, idx_ncvx] @ cp.pos(nu_vb[idx_ncvx])
                idx_ncvx += 1

        # Virtual slack penalty for cross-node constraints
        idx_cross = 0
        if jax_constraints.cross_node:
            for constraint in jax_constraints.cross_node:
                cost += lam_vb_cross[idx_cross] * cp.pos(nu_vb_cross[idx_cross])
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

        x_init = ocp_vars.x_init
        x_term = ocp_vars.x_term
        A_d = ocp_vars.A_d
        B_d = ocp_vars.B_d
        C_d = ocp_vars.C_d
        E_d = ocp_vars.E_d
        dyn_bias = ocp_vars.dyn_bias
        x0_imp_bias = ocp_vars.x0_imp_bias
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
        slice_cont = settings.sim.u.slice_continuous
        slice_imp = settings.sim.u.slice_impulsive
        has_impulsive = bool(slice_imp.stop > slice_imp.start)

        constr = []

        # Linearized nodal constraints (from JAX-lowered non-convex)
        idx_ncvx = 0
        if jax_constraints.nodal:
            for constraint in jax_constraints.nodal:
                # nodes should already be validated and normalized in preprocessing
                nodes = constraint.nodes
                for node in nodes:
                    residual = (
                        g[idx_ncvx][node]
                        + grad_g_x[idx_ncvx][node] @ x_nonscaled[node]
                        + grad_g_u[idx_ncvx][node] @ u_nonscaled[node]
                    )
                    constr += [residual == nu_vb[idx_ncvx][node]]
                idx_ncvx += 1

        # Linearized cross-node constraints (from JAX-lowered non-convex)
        idx_cross = 0
        if jax_constraints.cross_node:
            for constraint in jax_constraints.cross_node:
                # Linearization in affine form:
                # g_tilde + Σ_k(∇g_X[k]·X[k] + ∇g_U[k]·U[k]) == nu_vb
                # Sum over all trajectory nodes to couple multiple nodes
                residual = g_cross[idx_cross]
                for k in range(settings.sim.n):
                    # Contribution from state at node k
                    residual += grad_g_X_cross[idx_cross][k, :] @ x_nonscaled[k]
                    # Contribution from control at node k
                    residual += grad_g_U_cross[idx_cross][k, :] @ u_nonscaled[k]
                # Add constraint: residual == slack variable
                constr += [residual == nu_vb_cross[idx_cross]]
                idx_cross += 1

        # Convex constraints (already lowered to CVXPy)
        if cvxpy_constraints.constraints:
            constr += cvxpy_constraints.constraints

        # Boundary conditions (Fix)
        for i in range(settings.sim.true_state_slice.start, settings.sim.true_state_slice.stop):
            if settings.sim.x.initial_type[i] == "Fix":
                if has_impulsive:
                    constr += [
                        x_nonscaled[0][i]
                        == x0_imp_bias[i] + E_d[0][i, slice_imp] @ u_nonscaled[0][slice_imp]
                    ]
                else:
                    constr += [x_nonscaled[0][i] == x_init[i]]  # Initial Boundary Conditions
            if settings.sim.x.final_type[i] == "Fix":
                constr += [x_nonscaled[-1][i] == x_term[i]]  # Final Boundary Conditions

        if settings.sim._uniform_time_grid:
            S_u_inv_td = inv_S_u[settings.sim.time_dilation_slice, settings.sim.time_dilation_slice]
            c_u_td = c_u[settings.sim.time_dilation_slice]
            constr += [
                S_u_inv_td @ (u_nonscaled[i][settings.sim.time_dilation_slice] - c_u_td)
                == S_u_inv_td @ (u_nonscaled[i - 1][settings.sim.time_dilation_slice] - c_u_td)
                for i in range(1, settings.sim.n)
            ]

        constr += [
            inv_S_x @ (x_nonscaled[i] - c_x)
            == inv_S_x
            @ (
                A_d[i - 1] @ x_nonscaled[i - 1]
                + B_d[i - 1][:, slice_cont] @ u_nonscaled[i - 1][slice_cont]
                + C_d[i - 1][:, slice_cont] @ u_nonscaled[i][slice_cont]
                + (E_d[i][:, slice_imp] @ u_nonscaled[i][slice_imp] if has_impulsive else 0)
                + dyn_bias[i - 1]
                - c_x
            )
            + nu[i - 1]
            for i in range(1, settings.sim.n)
        ]  # Dynamics Constraint

        constr += [
            inv_S_u @ (u_nonscaled[i] - c_u) <= inv_S_u @ (settings.sim.u.max - c_u)
            for i in range(settings.sim.n)
        ]
        constr += [
            inv_S_u @ (u_nonscaled[i] - c_u) >= inv_S_u @ (settings.sim.u.min - c_u)
            for i in range(settings.sim.n)
        ]  # Control Constraints

        # TODO: (norrisg) formalize this
        constr += [
            inv_S_x @ (x_nonscaled[i][:] - c_x) <= inv_S_x @ (settings.sim.x.max - c_x)
            for i in range(settings.sim.n)
        ]
        constr += [
            inv_S_x @ (x_nonscaled[i][:] - c_x) >= inv_S_x @ (settings.sim.x.min - c_x)
            for i in range(settings.sim.n)
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

    def _setup_solve_function(self) -> None:
        """Configure the solve function based on solver settings.

        Sets up either cvxpygen-based solving or standard CVXPy solving
        based on the configuration.
        """
        if self.cvxpygen:
            cpg_solver_path = os.path.join("solver", "cpg_solver.py")
            if not os.path.isfile(cpg_solver_path):
                raise ImportError(
                    "cvxpygen solver not found. Make sure cvxpygen is installed and code "
                    "generation has been run. Install with: uv pip install openscvx[cvxpygen]"
                )
            try:
                from solver.cpg_solver import cpg_solve
            except ImportError as exc:
                raise ImportError(
                    "cvxpygen solver not found. Make sure cvxpygen is installed and code "
                    "generation has been run. Install with: uv pip install openscvx[cvxpygen]"
                ) from exc
            # cvxpygen v1.0+ registers CPG during generate_code(); re-register here so
            # reusing an existing solver/ directory (without regenerating) still works.
            self._problem.register_solve("CPG", cpg_solve)
            solver_args = self.solver_args
            self._solve_fn = lambda: self._problem.solve(method="CPG", **solver_args)
        else:
            solver = self.cvx_solver
            solver_args = dict(self.solver_args)

            def _solve_with_dpp_fallback():
                try:
                    return self._problem.solve(solver=solver, **solver_args)
                except cp.error.DPPError:
                    fallback_args = dict(solver_args)
                    fallback_args.pop("enforce_dpp", None)
                    fallback_args["ignore_dpp"] = True
                    return self._problem.solve(solver=solver, **fallback_args)

            self._solve_fn = _solve_with_dpp_fallback

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
        """Update dynamics linearization point and matrices.

        Sets the current linearization point (previous iterate) and the
        discretized dynamics matrices for the convex subproblem.

        Args:
            x_bar: Previous state trajectory, shape (N, n_states)
            u_bar: Previous control trajectory, shape (N, n_controls)
            A_d: Discretized state Jacobian, shape (N-1, n_states, n_states)
            B_d: Discretized control Jacobian (current node), shape (N-1, n_states, n_controls)
            C_d: Discretized control Jacobian (next node), shape (N-1, n_states, n_controls)
            x_prop: Propagated state from continuous dynamics, shape (N-1, n_states)
            x_prop_plus: Optional impulsive/discrete propagated state, shape (N, n_states)
            D_d: Optional impulsive/discrete Jacobian wrt state, shape (N, n_states, n_states)
            E_d: Optional impulsive/discrete Jacobian wrt control, shape (N, n_states, n_controls)
        """
        x_bar_arr = np.asarray(x_bar)
        u_bar_arr = np.asarray(u_bar)
        self._ocp_vars.x_bar.value = x_bar_arr
        self._ocp_vars.u_bar.value = u_bar_arr

        A_eff = np.asarray(A_d)
        B_eff = np.asarray(B_d)
        C_eff = np.asarray(C_d)

        if D_d is not None:
            D_arr = np.asarray(D_d)
            # Temporary DPP-safe workaround: absorb D_d into A/B/C numerically
            # so the CVXPY graph has only Parameter@Variable products.
            if D_arr.ndim == 3 and D_arr.shape[0] == A_eff.shape[0] + 1:
                D_steps = D_arr[1:]
            elif D_arr.ndim == 3 and D_arr.shape[0] == A_eff.shape[0]:
                D_steps = D_arr
            else:
                raise ValueError(
                    "Unexpected D_d shape for dynamics update: "
                    f"{D_arr.shape}, expected "
                    f"{(A_eff.shape[0] + 1, A_eff.shape[1], A_eff.shape[2])} "
                    f"or {(A_eff.shape[0], A_eff.shape[1], A_eff.shape[2])}."
                )

            A_eff = np.einsum("kij,kjl->kil", D_steps, A_eff)
            B_eff = np.einsum("kij,kjl->kil", D_steps, B_eff)
            C_eff = np.einsum("kij,kjl->kil", D_steps, C_eff)

        self._set_param("A_d", A_eff)
        self._set_param("B_d", B_eff)
        self._set_param("C_d", C_eff)
        x_prop_arr = np.asarray(x_prop)
        x_prop_plus_arr = np.asarray(x_prop_plus) if x_prop_plus is not None else None
        E_arr = None
        if E_d is not None:
            E_arr = np.asarray(E_d)
            self._ocp_vars.E_d.value = E_arr

        self.dynamics_biases(
            A_d=A_eff,
            B_d=B_eff,
            C_d=C_eff,
            x_bar=x_bar_arr,
            u_bar=u_bar_arr,
            x_prop=x_prop_arr,
            x_prop_plus=x_prop_plus_arr,
            E_d=E_arr,
        )

    def dynamics_biases(
        self,
        A_d: np.ndarray,
        B_d: np.ndarray,
        C_d: np.ndarray,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        x_prop: np.ndarray,
        x_prop_plus: np.ndarray | None = None,
        E_d: np.ndarray | None = None,
    ) -> None:
        """Update affine dynamics bias parameters for the current linearization.

        Computes and stores:
            - ``dyn_bias``: per-step affine bias in linearized dynamics
            - ``x0_imp_bias``: initial-node impulsive affine bias
        """
        has_impulsive = bool(self._slice_imp.stop > self._slice_imp.start)

        dyn_bias = np.zeros((A_d.shape[0], A_d.shape[1]), dtype=float)
        for k in range(A_d.shape[0]):
            i = k + 1
            base = x_prop_plus[i] if has_impulsive else x_prop[k]
            bias_k = (
                base
                - A_d[k] @ x_bar[k]
                - B_d[k][:, self._slice_cont] @ u_bar[k][self._slice_cont]
                - C_d[k][:, self._slice_cont] @ u_bar[i][self._slice_cont]
                - (E_d[i][:, self._slice_imp] @ u_bar[i][self._slice_imp] if has_impulsive else 0)
            )
            dyn_bias[k] = bias_k
        self._ocp_vars.dyn_bias.value = dyn_bias

        if has_impulsive:
            x0_imp_bias = x_prop_plus[0] - E_d[0][:, self._slice_imp] @ u_bar[0][self._slice_imp]
        else:
            x0_imp_bias = np.zeros(A_d.shape[1], dtype=float)
        self._ocp_vars.x0_imp_bias.value = x0_imp_bias

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
            x_bar = self._ocp_vars.x_bar.value
            u_bar = self._ocp_vars.u_bar.value
            x_bar_arr = np.asarray(x_bar)
            u_bar_arr = np.asarray(u_bar)
            for g_id, constraint_data in enumerate(nodal):
                g_arr = np.asarray(constraint_data["g"])
                grad_x_arr = np.asarray(constraint_data["grad_g_x"])
                grad_u_arr = np.asarray(constraint_data["grad_g_u"])

                # Convert to affine form in (X, U):
                # g_tilde + grad_x·X + grad_u·U, where
                # g_tilde = g(X_bar, U_bar) - grad_x·X_bar - grad_u·U_bar.
                g_tilde = (
                    g_arr
                    - np.sum(grad_x_arr * x_bar_arr, axis=1)
                    - np.sum(grad_u_arr * u_bar_arr, axis=1)
                )

                self._set_param(f"g_{g_id}", g_tilde)
                self._set_param(f"grad_g_x_{g_id}", grad_x_arr)
                self._set_param(f"grad_g_u_{g_id}", grad_u_arr)

        if cross_node:
            x_bar = self._ocp_vars.x_bar.value
            u_bar = self._ocp_vars.u_bar.value
            x_bar_arr = np.asarray(x_bar)
            u_bar_arr = np.asarray(u_bar)
            for g_id, constraint_data in enumerate(cross_node):
                g_val = np.asarray(constraint_data["g"])
                grad_X_arr = np.asarray(constraint_data["grad_g_X"])
                grad_U_arr = np.asarray(constraint_data["grad_g_U"])

                g_tilde = g_val - np.sum(grad_X_arr * x_bar_arr) - np.sum(grad_U_arr * u_bar_arr)

                self._set_param(f"g_cross_{g_id}", g_tilde)
                self._set_param(f"grad_g_X_cross_{g_id}", grad_X_arr)
                self._set_param(f"grad_g_U_cross_{g_id}", grad_U_arr)

    def update_penalties(
        self,
        lam_prox: np.ndarray,
        lam_cost: Union[float, np.ndarray],
        lam_vc: np.ndarray,
        lam_vb_nodal: np.ndarray,
        lam_vb_cross: np.ndarray,
    ) -> None:
        """Update SCP penalty weights.

        Sets the penalty weights that balance competing objectives in the
        PTR convex subproblem.

        Args:
            lam_prox: Trust region weights, shape ``(N, n_states + n_controls)``.
            lam_cost: Cost function weight. Scalar or array of shape
                ``(n_states,)`` for per-state weighting.
            lam_vc: Virtual control penalty weights, shape (N-1, n_states)
            lam_vb_nodal: Virtual buffer penalty weights for nodal constraints,
                shape ``(N, n_nodal_constraints)``.
            lam_vb_cross: Virtual buffer penalty weights for cross-node
                constraints, shape ``(n_cross_node_constraints,)``.
        """
        lam_prox_arr = np.asarray(lam_prox)
        self._set_param("lam_prox", lam_prox_arr)
        self._set_param("lam_cost", lam_cost)
        self._set_param("lam_vc", lam_vc)
        self._set_param("lam_vb_nodal", lam_vb_nodal)
        self._set_param("lam_vb_cross", lam_vb_cross)

    def proximal_cost_terms(
        self,
        lam_prox: np.ndarray,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute linear and constant proximal cost terms."""
        x_ref = (self._ocp_vars.inv_S_x @ (x_bar.T - self._ocp_vars.c_x[:, None])).T
        u_ref = (self._ocp_vars.inv_S_u @ (u_bar.T - self._ocp_vars.c_u[:, None])).T
        z_ref = np.hstack((x_ref, u_ref))
        prox_c = -2.0 * lam_prox * z_ref
        prox_cc = np.sum(lam_prox * np.square(z_ref), axis=1)
        return prox_c, prox_cc

    def update_proximal_terms(self) -> None:
        """Update proximal expansion parameters from current references and weights."""
        lam_prox_arr = np.asarray(self._ocp_vars.lam_prox.value)
        x_bar = np.asarray(self._ocp_vars.x_bar.value)
        u_bar = np.asarray(self._ocp_vars.u_bar.value)
        prox_c, prox_cc = self.proximal_cost_terms(lam_prox_arr, x_bar, u_bar)
        self._set_param("prox_c", prox_c)
        self._set_param("prox_cc", prox_cc)

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
        # No-op before initialize() — the CVXPy problem (and its param_dict)
        # isn't built yet. Callers like Problem._sync_boundary_conditions
        # may invoke this both before and after initialize().
        if self._problem is None:
            return
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
                "CVXPyPTRSolver.solve() called before initialize(). "
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

    def iteration_callback(self) -> Callable[..., SubproblemSolution]:
        """JAX-pure ``(state, SubproblemData) -> SubproblemSolution``.

        Wraps the existing NumPy ``solve()`` path in :func:`jax.pure_callback`
        with ``vmap_method="sequential"``. CVXPy can't trace — it builds a DCP
        graph and reaches into a backend solver through Python attribute
        mutation — so the only JAX-friendly surface it can present is a host
        callback. Under :func:`jax.jit` the callback fires once per call; under
        :func:`jax.vmap` it fires ``B`` times serially (CVXPy can't ingest a
        batched parameter set).

        The closure unstacks :class:`SubproblemData` back to the per-constraint
        dict layout :meth:`update_constraint_linearizations` expects, runs the
        four ``update_*`` methods + :meth:`solve`, and packs the result into a
        :class:`SubproblemSolution` whose ``nu_vb`` is collapsed to a stacked
        ``(N, n_nodal)`` array (the list-of-arrays form on
        :attr:`PTRSolveResult.nu_vb` is preserved for the NumPy path).

        ``status_code`` is mapped via :func:`status_str_to_code`; CVXPy's
        ``"optimal_inaccurate"`` / ``"solver_error"`` / ... labels collapse to
        :attr:`StatusCode.UNKNOWN` — the SCP trust-region check is the
        authoritative convergence gate.

        The returned callable takes ``(state, data)``: ``state`` is the
        :class:`AlgorithmState` pytree, accepted for cross-backend signature
        uniformity but unused (CVXPy exposes no warm-start through
        ``update_*``); ``data`` is the :class:`SubproblemData` pytree carrying
        the per-iteration linearization arrays, penalty weights, and boundary
        conditions.
        """
        if self._problem is None or self._settings is None or self._jax_constraints is None:
            raise RuntimeError(
                "CVXPyPTRSolver.iteration_callback() requires initialize() to "
                "have been called first."
            )

        settings = self._settings
        jax_constraints = self._jax_constraints
        n_x = settings.sim.n_states
        n_u = settings.sim.n_controls
        N = settings.sim.n
        n_nodal = len(jax_constraints.nodal)
        n_cross = len(jax_constraints.cross_node)
        slice_imp = settings.sim.u.slice_impulsive
        has_impulsive = bool(slice_imp.stop > slice_imp.start)
        f = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

        result_struct = SubproblemSolution(
            x=jax.ShapeDtypeStruct((N, n_x), f),
            u=jax.ShapeDtypeStruct((N, n_u), f),
            nu=jax.ShapeDtypeStruct((N - 1, n_x), f),
            nu_vb=jax.ShapeDtypeStruct((N, n_nodal), f),
            nu_vb_cross=jax.ShapeDtypeStruct((n_cross,), f),
            cost=jax.ShapeDtypeStruct((), f),
            status_code=jax.ShapeDtypeStruct((), jnp.int32),
        )

        def host_solve(data: SubproblemData) -> SubproblemSolution:
            # SubproblemData always carries (N, n_x, n_x) D_d / (N, n_x, n_u) E_d
            # / (N, n_x) x_prop_plus arrays — zero-filled when the problem has
            # no impulsive component. ``update_dynamics_linearization`` treats
            # a non-None D_d as "absorb into A/B/C", which would zero them out
            # on the no-impulsive path; pass None in that case to keep A/B/C
            # intact. Same logic for x_prop_plus / E_d.
            x_prop_plus = np.asarray(data.x_prop_plus) if has_impulsive else None
            D_d = np.asarray(data.D_d) if has_impulsive else None
            E_d = np.asarray(data.E_d) if has_impulsive else None
            self.update_dynamics_linearization(
                x_bar=np.asarray(data.x_bar),
                u_bar=np.asarray(data.u_bar),
                A_d=np.asarray(data.A_d),
                B_d=np.asarray(data.B_d),
                C_d=np.asarray(data.C_d),
                x_prop=np.asarray(data.x_prop),
                x_prop_plus=x_prop_plus,
                D_d=D_d,
                E_d=E_d,
            )
            nodal = _unstack_nodal(data, jax_constraints)
            cross_node = _unstack_cross(data, jax_constraints)
            self.update_constraint_linearizations(
                nodal=nodal or None,
                cross_node=cross_node or None,
            )
            self.update_penalties(
                lam_prox=np.asarray(data.lam_prox),
                lam_cost=np.asarray(data.lam_cost),
                lam_vc=np.asarray(data.lam_vc),
                lam_vb_nodal=np.asarray(data.lam_vb_nodal),
                lam_vb_cross=np.asarray(data.lam_vb_cross),
            )
            # Set the proximal-term parameter (``prox_c``) from the freshly
            # updated penalties; the legacy ``_subproblem`` called this between
            # ``update_penalties`` and ``solve``, and without it ``prox_c`` has
            # no value on the first solve of a process.
            self.update_proximal_terms()
            self.update_boundary_conditions(
                x_init=_unmask_bc(data.x_init),
                x_term=_unmask_bc(data.x_term),
            )

            try:
                result = self.solve()
            except cp.error.SolverError:
                # Infeasible / numerical failure. Surface it as a non-OPTIMAL
                # status on a finite (zero) solution so the SCP loop fails
                # loudly at the step() boundary, instead of the exception
                # propagating through pure_callback as an opaque CpuCallback
                # error.
                return SubproblemSolution(
                    x=jnp.zeros((N, n_x), dtype=f),
                    u=jnp.zeros((N, n_u), dtype=f),
                    nu=jnp.zeros((N - 1, n_x), dtype=f),
                    nu_vb=jnp.zeros((N, n_nodal), dtype=f),
                    nu_vb_cross=jnp.zeros((n_cross,), dtype=f),
                    cost=jnp.asarray(0.0, dtype=f),
                    status_code=jnp.asarray(int(StatusCode.INFEASIBLE), dtype=jnp.int32),
                )

            # ``optimal_inaccurate`` is a usable solution — the legacy path
            # consumed it without error — so treat it as OPTIMAL; the step()
            # status gate would otherwise reject the UNKNOWN it maps to.
            code = (
                StatusCode.OPTIMAL
                if (result.status == "optimal_inaccurate")
                or (result.status == "solved")
                or (result.status == "1 (for description visit https://qoco-org.github.io/qoco/)")
                else status_str_to_code(result.status)
            )
            return SubproblemSolution(
                x=jnp.asarray(result.x, dtype=f),
                u=jnp.asarray(result.u, dtype=f),
                nu=jnp.asarray(result.nu, dtype=f),
                # nu_vb collapses from PTRSolveResult's list-of-(N,) layout to
                # the (N, n_nodal) stacked array SubproblemSolution declares
                # for pure_callback's ShapeDtypeStruct contract.
                nu_vb=(
                    jnp.zeros((N, 0), dtype=f)
                    if n_nodal == 0
                    else jnp.asarray(
                        np.stack([np.asarray(a) for a in result.nu_vb], axis=-1),
                        dtype=f,
                    )
                ),
                nu_vb_cross=jnp.asarray(
                    np.asarray(result.nu_vb_cross, dtype=float).reshape(n_cross),
                    dtype=f,
                ),
                cost=jnp.asarray(result.cost, dtype=f),
                status_code=jnp.asarray(int(code), dtype=jnp.int32),
            )

        def step(state, data: SubproblemData) -> SubproblemSolution:
            del state  # CVXPy exposes no warm-start through ``update_*``.
            return jax.pure_callback(
                host_solve,
                result_struct,
                data,
                vmap_method="sequential",
            )

        return step

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


def _sqrt_factor_h_plus_numerical_guard(H_plus_np: np.ndarray) -> np.ndarray:
    """Factor an already-PSD H⁺ matrix, allowing only roundoff-level negatives."""
    eigvals, eigvecs = np.linalg.eigh(H_plus_np)
    tolerance = 1e-10
    min_eig = float(np.min(eigvals)) if eigvals.size else 0.0
    if min_eig < -tolerance:
        raise ValueError(
            "Expected PSD-projected H_plus, but found eigenvalue "
            f"{min_eig:.3e} below numerical tolerance {-tolerance:.3e}."
        )
    sqrt_eigvals = np.sqrt(np.maximum(0.0, eigvals))
    return eigvecs @ np.diag(sqrt_eigvals)


class CVXPyProxConvexSolver(CVXPyPTRSolver):
    """CVXPy solver for :class:`~openscvx.algorithms.scvx.prox_convex.ProxConvex`.

    Extends :class:`CVXPyPTRSolver` with sign-based branching on the SR
    composite: for each component ``r_i`` of the composite,

    * ``∇_i s ≥ 0`` (exact channel): embed ``r_i(x, u, params)`` directly
      as a CVXPy expression — the convex structure is preserved in the
      subproblem.
    * ``∇_i s < 0`` (linearized channel): replace ``r_i`` with its first-order
      Taylor approximation around ``x_k``.

    The problem structure changes only when the sign mask ``I_neg_mask``
    changes, so a ``dict[tuple[bool,...], cp.Problem]`` cache is maintained.
    With a monotone-increasing ``s`` the mask is all-False every iteration,
    so a single problem is built once and reused with only Parameter updates.

    Args:
        composite: :class:`~openscvx.algorithms.scvx.prox_convex.SRComposite`
            instance (the *same* object passed to
            :class:`~openscvx.algorithms.scvx.prox_convex.ProxConvex`).
        **kwargs: Forwarded to :class:`CVXPyPTRSolver` (``cvx_solver``,
            ``solver_args``, ``cvxpygen``, ``cvxpygen_override``).
    """

    def __init__(self, use_hessian_constraints: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._composite = None
        # Single source of truth for the h(C(x)) curvature block H_{C,k}
        # (dynamics defects + nonconvex constraints).  Read by
        # ``Problem.initialize`` to wire the dynamics Hessian solver into the
        # iteration body, and here to gate the subproblem's ``hess_cost`` term.
        self.use_hessian_constraints = use_hessian_constraints
        # mask_key -> {'problem', 'ds_params', 'lin_coef_params', 'lin_bias_params'}
        self._sr_problem_cache: dict = {}
        self._cached_lowered = None
        self._cached_settings: Optional[Config] = None

    def set_composite(self, composite) -> None:
        """Receive the SR composite forwarded by :meth:`Problem.initialize`."""
        self._composite = composite

    def initialize(self, lowered, settings: Config) -> None:
        """Build base PTR problem and validate the SR composite.

        Calls :meth:`CVXPyPTRSolver.initialize` for standard PTR
        infrastructure, then runs
        :func:`~openscvx.algorithms.scvx.prox_convex.check_sr_composite`
        against the freshly created CVXPy Variables to catch DCP / scalar
        violations early.  Per-mask SR problems are built lazily inside
        :meth:`iteration_callback` on first encounter of each sign-mask
        pattern.
        """
        from openscvx.algorithms.scvx.prox_convex import check_sr_composite

        super().initialize(lowered, settings)
        self._cached_lowered = lowered
        self._cached_settings = settings

        if self._composite is not None:
            check_sr_composite(
                self._composite,
                self._ocp_vars.x_nonscaled,
                self._ocp_vars.u_nonscaled,
                {},
            )

    def _build_sr_problem(self, mask_key: tuple, params: dict) -> dict:
        """Build and cache a ``cp.Problem`` for the given sign-mask pattern.

        For linearized channels (``mask_key[i] = True``):
            ``Cost += bias_param_i + Σ_k coef_param_i[k] @ x_nonscaled[k]``

        For exact channels (``mask_key[i] = False``):
            ``Cost += ds_param_i * r_i(x_nonscaled, u_nonscaled, params)``

        All cached problems share ``self._ocp_vars`` Parameters, so the
        standard PTR update methods propagate to every cached problem.

        Returns:
            dict with keys ``'problem'``, ``'ds_params'``,
            ``'lin_coef_params'``, ``'lin_bias_params'``.
        """
        lowered = self._cached_lowered
        settings = self._cached_settings
        N = settings.sim.n
        n_x = settings.sim.n_states
        n_r = len(self._composite.r)
        x_nonscaled = self._ocp_vars.x_nonscaled
        u_nonscaled = self._ocp_vars.u_nonscaled

        ds_params: List[Optional[cp.Parameter]] = [None] * n_r
        lin_coef_params: List[Optional[List]] = [None] * n_r
        lin_bias_params: List[Optional[cp.Parameter]] = [None] * n_r

        sr_cost = cp.Constant(0)

        for i in range(n_r):
            if mask_key[i]:
                # Linearized channel: affine approximation via Parameters.
                node_coefs = [cp.Parameter(n_x) for _ in range(N)]
                bias_p = cp.Parameter()
                lin_coef_params[i] = node_coefs
                lin_bias_params[i] = bias_p
                channel_cost = bias_p
                for k in range(N):
                    channel_cost = channel_cost + cp.sum(cp.multiply(node_coefs[k], x_nonscaled[k]))
                sr_cost = sr_cost + channel_cost
            else:
                # Exact channel: embed r_i directly as a CVXPy expression.
                from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

                ds_p = cp.Parameter(nonneg=True)
                ds_params[i] = ds_p
                node = self._composite._nodes[i]
                lowerer = CvxpyLowerer(
                    variable_map={"x": x_nonscaled[node], "u": u_nonscaled[node]}
                )
                ri_expr = lowerer.lower(self._composite.r[i])
                sr_cost = sr_cost + ds_p * ri_expr

        # Hessian curvature term: 0.5 * ||L^T (z - z_bar)||² where
        # L L^T = H⁺.  L = V diag(√λ⁺); offset = L^T z_bar is
        # precomputed on the CPU so the CVXPy expression reduces to
        # cp.sum_squares(param_matrix @ variable - param_vector).
        any_hessian = (self._composite.use_hessian is not False) or self.use_hessian_constraints
        if not any_hessian:
            L_hplus_p = None
            offset_hplus_p = None
            hess_cost = cp.Constant(0)
        else:
            # Curvature metric acts on the full stacked variable
            # z = [x.flatten(); u.flatten()], dimension D = N*(n_x + n_u),
            # matching the iteration's (D, D) H⁺_k assembly.
            n_u = settings.sim.n_controls
            D = N * (n_x + n_u)
            L_hplus_p = cp.Parameter((D, D))
            offset_hplus_p = cp.Parameter(D)
            z_flat = cp.hstack(
                [x_nonscaled[k] for k in range(N)] + [u_nonscaled[k] for k in range(N)]
            )
            hess_cost = 0.5 * cp.sum_squares(L_hplus_p.T @ z_flat - offset_hplus_p)
            # Zero initialisation → first iteration behaves as Q_k = µ_k I.
            L_hplus_p.value = np.zeros((D, D))
            offset_hplus_p.value = np.zeros(D)

        # Re-build the base PTR cost expression (no side effects; all
        # Parameter references are shared with _ocp_vars).
        total_cost = CVXPyPTRSolver.cost(self, settings, lowered) + sr_cost + hess_cost
        problem = cp.Problem(cp.Minimize(total_cost), self.constraints(settings, lowered))

        return {
            "problem": problem,
            "ds_params": ds_params,
            "lin_coef_params": lin_coef_params,
            "lin_bias_params": lin_bias_params,
            "L_hplus": L_hplus_p,
            "offset_hplus": offset_hplus_p,
        }

    def _update_sr_params(
        self,
        entry: dict,
        mask_key: tuple,
        ds_val_np: np.ndarray,
        R_val_np: np.ndarray,
        grad_R_np: np.ndarray,
        x_bar_np: np.ndarray,
        u_bar_np: np.ndarray,
        H_plus_np: np.ndarray,
    ) -> None:
        """Push current-iterate SR values into the cached problem's Parameters."""
        n_r = len(self._composite.r)
        for i in range(n_r):
            ds_i = float(ds_val_np[i])
            if mask_key[i]:
                for k, coef_p in enumerate(entry["lin_coef_params"][i]):
                    coef_p.value = ds_i * np.asarray(grad_R_np[i, k])
                bias = ds_i * float(R_val_np[i]) - ds_i * float(
                    np.sum(np.asarray(grad_R_np[i]) * x_bar_np)
                )
                entry["lin_bias_params"][i].value = bias
            else:
                entry["ds_params"][i].value = max(0.0, ds_i)

        # Update the square-root factor L = V diag(√λ⁺) for Q_k = µ_k I + H⁺_k.
        # H⁺ is already PSD-projected by the ProxConvex iteration; the helper
        # only guards against roundoff-level negative eigenvalues before sqrt.
        # offset = L^T x_bar is precomputed here so _build_sr_problem only needs a
        # single param_matrix @ variable product (DPP-compliant).
        if entry["L_hplus"] is None:
            return
        L = _sqrt_factor_h_plus_numerical_guard(H_plus_np)
        # offset = L^T z_bar with z_bar = [x_bar.flatten(); u_bar.flatten()],
        # matching the stacked z = [x; u] the curvature metric acts on.
        z_bar_flat = np.concatenate([x_bar_np.reshape(-1), u_bar_np.reshape(-1)])
        entry["L_hplus"].value = L
        entry["offset_hplus"].value = L.T @ z_bar_flat

    def _solve_sr_problem(self, problem: cp.Problem) -> None:
        """Solve a cached SR problem with the standard DPP fallback."""
        solver = self.cvx_solver
        solver_args = dict(self.solver_args)
        try:
            problem.solve(solver=solver, **solver_args)
        except cp.error.DPPError:
            fallback = dict(solver_args)
            fallback.pop("enforce_dpp", None)
            fallback["ignore_dpp"] = True
            problem.solve(solver=solver, **fallback)

    def iteration_callback(self):
        """JAX-pure ``(state, ProxConvexSubproblemData) -> SubproblemSolution``.

        Overrides :meth:`CVXPyPTRSolver.iteration_callback` to accept
        :class:`~openscvx.solvers.ptr_solver.ProxConvexSubproblemData`.

        Standard PTR update methods are called first (they update the shared
        Parameters all cached problems reference), then the SR Parameters are
        updated and the active cached problem is solved.
        """
        if self._problem is None or self._settings is None or self._jax_constraints is None:
            raise RuntimeError(
                "CVXPyProxConvexSolver.iteration_callback() requires initialize() "
                "to have been called first."
            )

        settings = self._settings
        jax_constraints = self._jax_constraints
        n_x = settings.sim.n_states
        n_u = settings.sim.n_controls
        N = settings.sim.n
        n_nodal = len(jax_constraints.nodal)
        n_cross = len(jax_constraints.cross_node)
        slice_imp = settings.sim.u.slice_impulsive
        has_impulsive = bool(slice_imp.stop > slice_imp.start)
        f = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

        result_struct = SubproblemSolution(
            x=jax.ShapeDtypeStruct((N, n_x), f),
            u=jax.ShapeDtypeStruct((N, n_u), f),
            nu=jax.ShapeDtypeStruct((N - 1, n_x), f),
            nu_vb=jax.ShapeDtypeStruct((N, n_nodal), f),
            nu_vb_cross=jax.ShapeDtypeStruct((n_cross,), f),
            cost=jax.ShapeDtypeStruct((), f),
            status_code=jax.ShapeDtypeStruct((), jnp.int32),
        )

        def _make_infeasible():
            return SubproblemSolution(
                x=jnp.zeros((N, n_x), dtype=f),
                u=jnp.zeros((N, n_u), dtype=f),
                nu=jnp.zeros((N - 1, n_x), dtype=f),
                nu_vb=jnp.zeros((N, n_nodal), dtype=f),
                nu_vb_cross=jnp.zeros((n_cross,), dtype=f),
                cost=jnp.asarray(0.0, dtype=f),
                status_code=jnp.asarray(int(StatusCode.INFEASIBLE), dtype=jnp.int32),
            )

        def _unpack(problem: cp.Problem) -> SubproblemSolution:
            S_x = self._ocp_vars.S_x
            c_x = self._ocp_vars.c_x
            S_u = self._ocp_vars.S_u
            c_u = self._ocp_vars.c_u
            x_scaled = problem.var_dict["x"].value
            u_scaled = problem.var_dict["u"].value
            x_sol = (S_x @ x_scaled.T + np.expand_dims(c_x, 1)).T
            u_sol = (S_u @ u_scaled.T + np.expand_dims(c_u, 1)).T
            nu_sol = problem.var_dict["nu"].value
            nu_vb_list = [v.value for v in self._ocp_vars.nu_vb]
            nu_vb_cross_list = [v.value for v in self._ocp_vars.nu_vb_cross]
            raw_status = problem.status or "unknown"
            code = (
                StatusCode.OPTIMAL
                if raw_status
                in (
                    "optimal_inaccurate",
                    "solved",
                    "1 (for description visit https://qoco-org.github.io/qoco/)",
                )
                else status_str_to_code(raw_status)
            )
            return SubproblemSolution(
                x=jnp.asarray(x_sol, dtype=f),
                u=jnp.asarray(u_sol, dtype=f),
                nu=jnp.asarray(nu_sol, dtype=f),
                nu_vb=(
                    jnp.zeros((N, 0), dtype=f)
                    if n_nodal == 0
                    else jnp.asarray(
                        np.stack([np.asarray(a) for a in nu_vb_list], axis=-1), dtype=f
                    )
                ),
                nu_vb_cross=jnp.asarray(
                    np.asarray(nu_vb_cross_list, dtype=float).reshape(n_cross), dtype=f
                ),
                cost=jnp.asarray(problem.value, dtype=f),
                status_code=jnp.asarray(int(code), dtype=jnp.int32),
            )

        def host_solve(data: ProxConvexSubproblemData) -> SubproblemSolution:
            # Standard PTR updates — propagate to all cached problems via
            # shared _ocp_vars Parameters.
            x_prop_plus = np.asarray(data.x_prop_plus) if has_impulsive else None
            D_d = np.asarray(data.D_d) if has_impulsive else None
            E_d = np.asarray(data.E_d) if has_impulsive else None
            self.update_dynamics_linearization(
                x_bar=np.asarray(data.x_bar),
                u_bar=np.asarray(data.u_bar),
                A_d=np.asarray(data.A_d),
                B_d=np.asarray(data.B_d),
                C_d=np.asarray(data.C_d),
                x_prop=np.asarray(data.x_prop),
                x_prop_plus=x_prop_plus,
                D_d=D_d,
                E_d=E_d,
            )
            self.update_constraint_linearizations(
                nodal=_unstack_nodal(data, jax_constraints) or None,
                cross_node=_unstack_cross(data, jax_constraints) or None,
            )
            self.update_penalties(
                lam_prox=np.asarray(data.lam_prox),
                lam_cost=np.asarray(data.lam_cost),
                lam_vc=np.asarray(data.lam_vc),
                lam_vb_nodal=np.asarray(data.lam_vb_nodal),
                lam_vb_cross=np.asarray(data.lam_vb_cross),
            )
            self.update_proximal_terms()
            self.update_boundary_conditions(
                x_init=_unmask_bc(data.x_init),
                x_term=_unmask_bc(data.x_term),
            )

            # SR composite: get or build cached problem, then update SR params.
            I_neg_mask_np = np.asarray(data.I_neg_mask)
            mask_key = tuple(bool(b) for b in I_neg_mask_np)
            if mask_key not in self._sr_problem_cache:
                self._sr_problem_cache[mask_key] = self._build_sr_problem(mask_key, data.params)
            entry = self._sr_problem_cache[mask_key]
            self._update_sr_params(
                entry,
                mask_key,
                ds_val_np=np.asarray(data.ds_val),
                R_val_np=np.asarray(data.R_val),
                grad_R_np=np.asarray(data.grad_R),
                x_bar_np=np.asarray(data.x_bar),
                u_bar_np=np.asarray(data.u_bar),
                H_plus_np=np.asarray(data.H_plus),
            )

            try:
                self._solve_sr_problem(entry["problem"])
            except cp.error.SolverError:
                return _make_infeasible()

            return _unpack(entry["problem"])

        def step(state, data: ProxConvexSubproblemData) -> SubproblemSolution:
            del state
            return jax.pure_callback(
                host_solve,
                result_struct,
                data,
                vmap_method="sequential",
            )

        return step

    def citation(self) -> List[str]:
        return super().citation() + [
            r"""@misc{uzun2025proxconvex,
  title={A Proximal Method for Composite Optimization with Smooth and Convex Components},
  author={Uzun, Samet and Luo, Dayou and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et and Aravkin, Aleksandr Y.},
  year={2025},
  eprint={2512.20602},
  archivePrefix={arXiv},
  primaryClass={math.OC}
}""",
        ]
