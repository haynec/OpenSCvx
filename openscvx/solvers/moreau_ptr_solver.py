"""JAX-native conic backend for the PTR convex subproblem.

Assembles each SCP subproblem as a sparse conic program (SOCP-capable) and
dispatches it to ``moreau.jax.Solver``, a JAX-native interior-point conic
solver.  This backend sits alongside
:class:`openscvx.solvers.qpax_ptr_solver.QPAXPTRSolver` (QP-only) and
:class:`openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver` (DCP via CVXPy).

Strategic motivation
---------------------
QPAX must slack-reformulate the PTR penalties — ``|nu|`` (L1 virtual control)
and ``pos(nu_vb)`` (positive-part virtual buffer) — by introducing extra slack
variables and doubled inequality rows.  Moreau accepts Second-Order Cone (SOC)
constraints natively, so ``|nu_i| <= t_i`` is expressed as a 2-D SOC on
``[t_i, nu_i]`` with linear cost ``lam_vc * t_i``: one epigraph variable
instead of one slack plus two inequality rows.  ``pos(nu_vb)`` uses a
nonnegative epigraph instead of two inequality rows.  The net result is a
smaller problem for the same PTR math.

Moreau's JAX path is differentiable via implicit differentiation on the KKT
conditions.  When the surrounding SCP pipeline is made ``jax.jit``-friendly
(future work), ``jax.grad`` / ``jax.vmap`` can reach through a full SCvx solve.

Scope
------
* No user ``.convex()`` constraints.  The inherited default
  :meth:`ConvexSolver.lower_convex_constraints` refuses them upstream (at
  ``Problem(...)`` construction time) and points the user at
  :class:`openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver`.
* No cross-node or impulsive controls.  Each raises
  :class:`NotImplementedError` at :meth:`initialize` with a pointer to
  ``CVXPyPTRSolver``.
* CTCS constraints are supported — their LICQ-style absolute-value
  inequalities are affine and fit neatly in the nonneg cone.

Warm-start
-----------
Unlike QPAX (which cold-starts each iteration), ``MoreauPTRSolver`` carries
the previous solution as a :class:`moreau._types.WarmStart` and passes it to
each ``solver.solve()`` call.  Successive SCP subproblems differ only in the
linearization point and penalty weights, so warm-starting cuts iteration counts
substantially.

Moreau / Conic problem formulation
------------------------------------
Moreau solves::

    min  0.5 zᵀ P z + qᵀ z
    s.t. A z + s = b,  s ∈ K

where K is a product of zero, nonneg, and SOC cones ordered first-to-last.
For a zero-cone row: ``s = 0`` → ``Az = b`` (equality).
For a nonneg-cone row: ``s ≥ 0`` → ``b − Az ≥ 0`` (``Az ≤ b``).
For a SOC(d) block: ``s ∈ 𝒦_soc^d`` → ``‖(b − Az)[1:]‖ ≤ (b − Az)[0]``.

float64
--------
Moreau performs internal arithmetic in float64.  Pass ``float_dtype="float64"``
to :class:`openscvx.problem.Problem` for tight inner-solver tolerances; the
default float32 caps the QP's conditioning.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse as sp

from .ptr_solver import PTRSolver, PTRSolveResult

try:
    import jax.numpy as jnp
    import moreau
    from moreau.jax import Solver as _MoreauJaxSolver

    _MOREAU_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised by the install-error test
    moreau = None  # type: ignore[assignment]
    _MoreauJaxSolver = None  # type: ignore[assignment,misc]
    jnp = None  # type: ignore[assignment]
    _MOREAU_AVAILABLE = False

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered import LoweredProblem
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints
    from openscvx.lowered.unified import UnifiedControl, UnifiedState

# Tiny diagonal regularisation added to P on dx/du slots.  Moreau's IPM
# Cholesky factors (P + Gᵀ diag(z/s) G); keeping the diagonal positive avoids
# near-singular factorisation when lam_prox is very small.
_P_DIAG_EPS = 1e-10


def _moreau_solve_ok(status: "moreau.SolverStatus") -> bool:
    """Return True when moreau reports a primal solution worth warm-starting from."""
    return status in (
        moreau.SolverStatus.Solved,
        moreau.SolverStatus.AlmostSolved,
    )


# ---------------------------------------------------------------------------
# Decision-vector layout
# ---------------------------------------------------------------------------


@dataclass
class _ConicLayout:
    """Static index layout for the flat decision vector z.

    Moreau's conic formulation drops the slack blocks used by QPAX
    (``s_abs``, ``s_pos``) and replaces them with epigraph variables
    ``t_vc`` / ``t_vb``, one per penalty term:

    * ``|nu[k,j]| ≤ t_vc[k,j]`` expressed as a SOC(2).
    * ``nu_vb[c,k] ≤ t_vb[c,k]`` and ``t_vb[c,k] ≥ 0`` as two nonneg rows.

    Slices are computed once at :meth:`MoreauPTRSolver.create_variables` and
    reused on every solve.
    """

    N: int
    n_x: int
    n_u: int
    n_nodal: int

    sl_x: slice = field(init=False)
    sl_u: slice = field(init=False)
    sl_dx: slice = field(init=False)
    sl_du: slice = field(init=False)
    sl_nu: slice = field(init=False)
    sl_nu_vb: List[slice] = field(init=False)
    sl_t_vc: slice = field(init=False)  # (N-1)*n_x epigraph vars for |nu|
    sl_t_vb: List[slice] = field(init=False)  # N epigraph vars per nodal constraint
    n_z: int = field(init=False)

    def __post_init__(self):
        N, n_x, n_u, C = self.N, self.n_x, self.n_u, self.n_nodal
        cursor = 0

        def take(width: int) -> slice:
            nonlocal cursor
            s = slice(cursor, cursor + width)
            cursor += width
            return s

        self.sl_x = take(N * n_x)
        self.sl_u = take(N * n_u)
        self.sl_dx = take(N * n_x)
        self.sl_du = take(N * n_u)
        self.sl_nu = take((N - 1) * n_x)
        self.sl_nu_vb = [take(N) for _ in range(C)]
        self.sl_t_vc = take((N - 1) * n_x)
        self.sl_t_vb = [take(N) for _ in range(C)]
        self.n_z = cursor

    # -- helpers for clarity at call sites --

    def x_idx(self, k: int, j: int) -> int:
        return self.sl_x.start + k * self.n_x + j

    def u_idx(self, k: int, j: int) -> int:
        return self.sl_u.start + k * self.n_u + j

    def dx_idx(self, k: int, j: int) -> int:
        return self.sl_dx.start + k * self.n_x + j

    def du_idx(self, k: int, j: int) -> int:
        return self.sl_du.start + k * self.n_u + j

    def nu_idx(self, k: int, j: int) -> int:
        """``k`` ∈ [0, N-2]."""
        return self.sl_nu.start + k * self.n_x + j

    def nu_vb_idx(self, c: int, k: int) -> int:
        return self.sl_nu_vb[c].start + k

    def t_vc_idx(self, k: int, j: int) -> int:
        """``k`` ∈ [0, N-2]."""
        return self.sl_t_vc.start + k * self.n_x + j

    def t_vb_idx(self, c: int, k: int) -> int:
        return self.sl_t_vb[c].start + k


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------


class MoreauPTRSolver(PTRSolver):
    """JAX-native conic backend for the PTR convex subproblem.

    Assembles each SCP subproblem as a sparse conic program and dispatches to
    ``moreau.jax.Solver``.  See the module docstring for the penalty encoding,
    warm-start, and float64 advice.

    Compared to :class:`QPAXPTRSolver`:

    * Fewer decision variables and constraint rows for the same PTR physics
      (SOC epigraphs for ``|nu|`` instead of two nonneg slack rows each).
    * Warm-starts automatically between SCP iterations (QPAX cold-starts).
    * Opens the path to user ``.convex()`` SOCP support in a follow-up.

    Differentiability hook for future work:
        ``moreau.jax.Solver`` differentiates through the solve via implicit
        differentiation on the KKT conditions.  Once the surrounding SCP
        pipeline stays in ``jit``, ``jax.grad`` / ``jax.vmap`` can reach
        through a full SCvx solve.

    Note:
        Supported — state/control box, dynamics linearization, boundary Fix,
        uniform time grid, linearized nodal nonconvex, CTCS LICQ rows.

        Not supported — user ``.convex()`` constraints (rejected at
        ``Problem(...)`` construction time by the inherited
        :meth:`ConvexSolver.lower_convex_constraints`), cross-node
        constraints, and impulsive controls.  Each raises
        :class:`NotImplementedError` with a "use :class:`CVXPyPTRSolver`"
        pointer.

    Args:
        solver_args: Keyword arguments forwarded to :class:`moreau.Settings`.
            Useful keys: ``max_iter`` (default 200), ``verbose`` (default
            False), ``device`` (``'auto'``, ``'cpu'``, or ``'cuda'``), and a
            nested ``ipm_settings`` dict for fine-grained IPM tolerances (e.g.
            ``{"tol_gap_abs": 1e-8, "tol_feas": 1e-8}``).

    Attributes:
        layout: :class:`_ConicLayout` describing the flat decision-vector slot
            ranges.  Populated by :meth:`create_variables`.
    """

    def __init__(self, solver_args: Optional[Dict] = None):
        if not _MOREAU_AVAILABLE:
            raise ImportError(
                "MoreauPTRSolver requires the `moreau` package. "
                "Install it with: pip install openscvx[moreau]"
            )
        self.solver_args = dict(solver_args) if solver_args else {}

        self.layout: Optional[_ConicLayout] = None
        self._S_x: Optional[np.ndarray] = None
        self._c_x: Optional[np.ndarray] = None
        self._S_u: Optional[np.ndarray] = None
        self._c_u: Optional[np.ndarray] = None
        self._S_x_diag: Optional[np.ndarray] = None
        self._S_u_diag: Optional[np.ndarray] = None
        self._inv_S_x_diag: Optional[np.ndarray] = None
        self._inv_S_u_diag: Optional[np.ndarray] = None
        self._settings: Optional["Config"] = None
        self._jax_constraints: Optional["LoweredJaxConstraints"] = None

        # moreau.jax.Solver constructed at initialize().
        self._moreau: Optional[_MoreauJaxSolver] = None

        # Fixed CSR structure stored after initialize().
        self._P_diag_slots: Optional[np.ndarray] = None
        self._P_n_dx: int = 0  # N*n_x, for indexing into P_data
        self._coo_rows: Optional[np.ndarray] = None
        self._coo_cols: Optional[np.ndarray] = None
        self._n_con: int = 0

        # Per-iteration data, set by update_* methods.
        self._dyn: dict = {}
        self._cons: dict = {}
        self._pen: dict = {}
        self._x_init: Optional[np.ndarray] = None
        self._x_term: Optional[np.ndarray] = None

        # Warm-start carry; reset to None when problem structure changes.
        self._warm_start = None

        # Last solve status from moreau (populated by solve()).
        self._last_status: moreau.SolverStatus = moreau.SolverStatus.Unsolved

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def create_variables(
        self,
        N: int,
        x_unified: "UnifiedState",
        u_unified: "UnifiedControl",
        jax_constraints: "LoweredJaxConstraints",
        dynamics_sparsity: Optional[tuple] = None,
        constraint_sparsity: Optional[list] = None,
    ) -> None:
        """Compute scaling matrices and the static conic decision-vector layout.

        Sparsity hints are accepted for interface symmetry with
        :class:`CVXPyPTRSolver` but ignored — Moreau uses dense A/P data.
        """
        del dynamics_sparsity, constraint_sparsity

        n_x = len(x_unified.max)
        n_u = len(u_unified.max)

        slice_imp = u_unified.slice_impulsive
        if slice_imp.stop > slice_imp.start:
            raise NotImplementedError(
                "MoreauPTRSolver does not support impulsive controls "
                f"(u.slice_impulsive = {slice_imp!r}). "
                "Use CVXPyPTRSolver for problems with impulsive dynamics."
            )

        S_x, c_x = self._scaling(x_unified)
        S_u, c_u = self._scaling(u_unified)

        self._S_x = S_x
        self._c_x = c_x
        self._S_u = S_u
        self._c_u = c_u
        self._S_x_diag = np.diag(S_x)
        self._S_u_diag = np.diag(S_u)
        self._inv_S_x_diag = 1.0 / self._S_x_diag
        self._inv_S_u_diag = 1.0 / self._S_u_diag

        self.layout = _ConicLayout(N=N, n_x=n_x, n_u=n_u, n_nodal=len(jax_constraints.nodal))
        self._jax_constraints = jax_constraints

    def initialize(self, lowered: "LoweredProblem", settings: "Config") -> None:
        """Build the static conic structure and construct the ``moreau.jax.Solver``.

        Validates the supported constraint subset, enumerates all A-matrix
        nonzero positions (the fixed CSR structure), builds a
        :class:`moreau.Cones` spec, and constructs the
        :class:`moreau.jax.Solver`.  Numeric values in A/b/P/q are filled at
        each :meth:`solve` call.
        """
        if self.layout is None:
            raise RuntimeError(
                "MoreauPTRSolver.initialize() called before create_variables(). "
                "Call create_variables() first."
            )

        if lowered.jax_constraints.cross_node:
            raise NotImplementedError(
                "MoreauPTRSolver does not yet support cross-node constraints "
                f"({len(lowered.jax_constraints.cross_node)} defined). "
                "Use CVXPyPTRSolver."
            )
        slice_imp = settings.sim.u.slice_impulsive
        if slice_imp.stop > slice_imp.start:
            raise NotImplementedError(
                "MoreauPTRSolver does not support impulsive controls. Use CVXPyPTRSolver."
            )

        self._settings = settings
        # Reset warm-start on new initialization (problem structure may change).
        self._warm_start = None

        # Enumerate all (row, col) pairs in the A matrix (structural pass).
        coo_rows, coo_cols, n_eq, n_nn, soc_dims = self._structural_pass(settings)
        n_soc = sum(soc_dims)
        self._n_con = n_eq + n_nn + n_soc
        self._coo_rows = np.asarray(coo_rows, dtype=np.int32)
        self._coo_cols = np.asarray(coo_cols, dtype=np.int32)

        # Build CSR for A from the structural COO.
        if len(coo_rows) > 0:
            A_struct = sp.csr_matrix(
                (np.ones(len(coo_rows)), (coo_rows, coo_cols)),
                shape=(self._n_con, self.layout.n_z),
            )
            A_struct.sort_indices()
        else:
            A_struct = sp.csr_matrix((self._n_con, self.layout.n_z))
        A_indptr = np.asarray(A_struct.indptr, dtype=np.int32)
        A_indices = np.asarray(A_struct.indices, dtype=np.int32)
        self._A_indptr = A_indptr
        self._A_indices = A_indices

        # Build CSR for P (diagonal on dx/du slots only).
        L = self.layout
        N, n_x = L.N, L.n_x
        dx_cols = np.arange(L.sl_dx.start, L.sl_dx.stop, dtype=np.int32)
        du_cols = np.arange(L.sl_du.start, L.sl_du.stop, dtype=np.int32)
        self._P_diag_slots = np.concatenate([dx_cols, du_cols])
        self._P_n_dx = N * n_x  # boundary between dx and du in P_data

        # P is diagonal: for each slot col, the single nonzero is P[col, col].
        P_indptr = np.zeros(L.n_z + 1, dtype=np.int32)
        for col in self._P_diag_slots:
            P_indptr[col + 1] = 1
        P_indptr = np.cumsum(P_indptr)
        P_indices = self._P_diag_slots  # diagonal → col index == row index

        # Construct moreau.Cones spec.
        cones = moreau.Cones(
            num_zero_cones=n_eq,
            num_nonneg_cones=n_nn,
            so_cone_dims=soc_dims,
        )

        # Build moreau.Settings from solver_args.
        moreau_settings = _build_moreau_settings(self.solver_args)

        self._moreau = _MoreauJaxSolver(
            n=L.n_z,
            m=self._n_con,
            P_row_offsets=jnp.array(P_indptr),
            P_col_indices=jnp.array(P_indices),
            A_row_offsets=jnp.array(A_indptr),
            A_col_indices=jnp.array(A_indices),
            cones=cones,
            settings=moreau_settings,
        )

    # ------------------------------------------------------------------
    # Per-iteration update hooks
    # ------------------------------------------------------------------

    def update_dynamics_linearization(
        self,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        A_d: np.ndarray,
        B_d: np.ndarray,
        C_d: np.ndarray,
        x_prop: np.ndarray,
        x_prop_plus: Optional[np.ndarray] = None,
        D_d: Optional[np.ndarray] = None,
        E_d: Optional[np.ndarray] = None,
    ) -> None:
        # Impulsive rejected at initialize(); D_d / E_d / x_prop_plus dropped.
        del x_prop_plus, D_d, E_d
        self._dyn = {
            "x_bar": np.asarray(x_bar, dtype=float),
            "u_bar": np.asarray(u_bar, dtype=float),
            "A_d": np.asarray(A_d, dtype=float),
            "B_d": np.asarray(B_d, dtype=float),
            "C_d": np.asarray(C_d, dtype=float),
            "x_prop": np.asarray(x_prop, dtype=float),
        }

    def update_constraint_linearizations(
        self,
        nodal: Optional[List[dict]] = None,
        cross_node: Optional[List[dict]] = None,
    ) -> None:
        if cross_node:
            raise NotImplementedError(
                "MoreauPTRSolver received cross-node linearization data; "
                "cross-node constraints are not supported. Use CVXPyPTRSolver."
            )
        self._cons = {
            "nodal": [
                {
                    "g": np.asarray(d["g"], dtype=float),
                    "grad_g_x": np.asarray(d["grad_g_x"], dtype=float),
                    "grad_g_u": np.asarray(d["grad_g_u"], dtype=float),
                }
                for d in (nodal or [])
            ],
        }

    def update_penalties(
        self,
        lam_prox: np.ndarray,
        lam_cost: Union[float, np.ndarray],
        lam_vc: np.ndarray,
        lam_vb_nodal: np.ndarray,
        lam_vb_cross: np.ndarray,
    ) -> None:
        del lam_vb_cross  # cross-node rejected at initialize()
        self._pen = {
            "lam_prox": np.asarray(lam_prox, dtype=float),
            "lam_cost": np.asarray(lam_cost, dtype=float),
            "lam_vc": np.asarray(lam_vc, dtype=float),
            "lam_vb_nodal": np.asarray(lam_vb_nodal, dtype=float),
        }

    def update_boundary_conditions(
        self,
        x_init: Optional[np.ndarray] = None,
        x_term: Optional[np.ndarray] = None,
    ) -> None:
        if x_init is not None:
            self._x_init = np.asarray(x_init, dtype=float)
        if x_term is not None:
            self._x_term = np.asarray(x_term, dtype=float)

    # ------------------------------------------------------------------
    # Structural pass (called once at initialize)
    # ------------------------------------------------------------------

    def _structural_pass(
        self,
        settings: "Config",
    ) -> Tuple[List[int], List[int], int, int, List[int]]:
        """Enumerate all A-matrix ``(row, col)`` nonzero positions.

        Iterates through every constraint row in the exact order used by
        :meth:`_assemble_conic`, recording which columns are structurally
        nonzero.  Only the positions (not the values) matter here; values come
        from per-iteration data at solve time.

        Returns:
            Tuple of (coo_rows, coo_cols, n_eq, n_nn, soc_dims):

                coo_rows: Row indices of every A-matrix nonzero, in
                    traversal order.
                coo_cols: Corresponding column indices.
                n_eq: Number of zero-cone (equality) rows.
                n_nn: Number of nonneg-cone rows.
                soc_dims: SOC cone dimensions; one entry per ``|nu[k,j]|``
                    term, all equal to 2.
        """
        L = self.layout
        N, n_x, n_u = L.N, L.n_x, L.n_u
        jax_constraints = self._jax_constraints

        coo_rows: List[int] = []
        coo_cols: List[int] = []
        row = 0

        def add(r: int, c: int) -> None:
            coo_rows.append(r)
            coo_cols.append(c)

        # ================================================================
        # ZERO CONE (equality, s = 0 → Az = b)
        # ================================================================

        # State error definitions: x[k,j] − dx[k,j] = rhs  (N·n_x rows)
        for k in range(N):
            for j in range(n_x):
                add(row, L.x_idx(k, j))
                add(row, L.dx_idx(k, j))
                row += 1

        # Control error definitions: u[k,j] − du[k,j] = rhs  (N·n_u rows)
        for k in range(N):
            for j in range(n_u):
                add(row, L.u_idx(k, j))
                add(row, L.du_idx(k, j))
                row += 1

        # Dynamics (FOH, continuous PTR)  ((N-1)·n_x rows)
        for k in range(1, N):
            kp = k - 1
            for i in range(n_x):
                add(row, L.x_idx(k, i))
                for j in range(n_x):
                    add(row, L.dx_idx(kp, j))
                for j in range(n_u):
                    add(row, L.du_idx(kp, j))
                    add(row, L.du_idx(k, j))
                add(row, L.nu_idx(kp, i))
                row += 1

        # Linearized nodal constraints: grad·dx + grad·du − nu_vb = −g
        for c_idx, constraint in enumerate(jax_constraints.nodal):
            for node in constraint.nodes:
                for j in range(n_x):
                    add(row, L.dx_idx(node, j))
                for j in range(n_u):
                    add(row, L.du_idx(node, j))
                add(row, L.nu_vb_idx(c_idx, node))
                row += 1

        # Fix boundary conditions (initial / terminal)
        for i in range(settings.sim.true_state_slice.start, settings.sim.true_state_slice.stop):
            if settings.sim.x.initial_type[i] == "Fix":
                add(row, L.x_idx(0, i))
                row += 1
            if settings.sim.x.final_type[i] == "Fix":
                add(row, L.x_idx(N - 1, i))
                row += 1

        # Uniform time grid: u[k,j] − u[k-1,j] = 0
        if settings.sim._uniform_time_grid:
            td = settings.sim.time_dilation_slice
            for k in range(1, N):
                for j in range(td.start, td.stop):
                    add(row, L.u_idx(k, j))
                    add(row, L.u_idx(k - 1, j))
                    row += 1

        # CTCS initial: S_x[idx]·x[0,idx] = −c_x[idx]  (one per CTCS group)
        for idx in range(settings.sim.ctcs_slice.start, settings.sim.ctcs_slice.stop):
            add(row, L.x_idx(0, idx))
            row += 1

        n_eq = row

        # ================================================================
        # NONNEG CONE (s = b − Az ≥ 0)
        # ================================================================

        # State box upper: x[k,j] ≤ x_max_scaled[j]
        for k in range(N):
            for j in range(n_x):
                add(row, L.x_idx(k, j))
                row += 1

        # State box lower: x[k,j] ≥ x_min_scaled[j]
        for k in range(N):
            for j in range(n_x):
                add(row, L.x_idx(k, j))
                row += 1

        # Control box upper
        for k in range(N):
            for j in range(n_u):
                add(row, L.u_idx(k, j))
                row += 1

        # Control box lower
        for k in range(N):
            for j in range(n_u):
                add(row, L.u_idx(k, j))
                row += 1

        # CTCS difference rows (two per node interval per group)
        for idx, nodes in zip(
            range(settings.sim.ctcs_slice.start, settings.sim.ctcs_slice.stop),
            settings.sim.ctcs_node_intervals,
        ):
            start_i = 1 if nodes[0] == 0 else nodes[0]
            for i in range(start_i, nodes[1]):
                # +direction
                add(row, L.x_idx(i, idx))
                add(row, L.x_idx(i - 1, idx))
                row += 1
                # −direction
                add(row, L.x_idx(i, idx))
                add(row, L.x_idx(i - 1, idx))
                row += 1

        # Pos epigraph: t_vb[c,k] ≥ nu_vb[c,k]
        for c_idx in range(L.n_nodal):
            for k in range(N):
                add(row, L.nu_vb_idx(c_idx, k))
                add(row, L.t_vb_idx(c_idx, k))
                row += 1

        # Pos epigraph non-negativity: t_vb[c,k] ≥ 0
        for c_idx in range(L.n_nodal):
            for k in range(N):
                add(row, L.t_vb_idx(c_idx, k))
                row += 1

        n_nn = row - n_eq

        # ================================================================
        # SOC rows: for each scalar |nu[k,j]| ≤ t_vc[k,j], SOC(2)
        # ================================================================
        soc_dims: List[int] = []
        for k in range(N - 1):
            for j in range(n_x):
                # s[0] = t_vc (row 0 of this SOC block)
                add(row, L.t_vc_idx(k, j))
                row += 1
                # s[1] = nu   (row 1 of this SOC block)
                add(row, L.nu_idx(k, j))
                row += 1
                soc_dims.append(2)

        return coo_rows, coo_cols, n_eq, n_nn, soc_dims

    # ------------------------------------------------------------------
    # Assembly (called each SCP iteration)
    # ------------------------------------------------------------------

    def _assemble_conic(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fill ``P_data``, COO A-values, ``q``, and ``b`` from per-iteration data.

        Iterates through every row in the same order as :meth:`_structural_pass`
        so that the emitted value list corresponds to the stored
        ``self._coo_rows`` / ``self._coo_cols`` arrays.  scipy builds the same
        sorted CSR from the COO at every call, so ``A_csr.data`` aligns with
        the fixed ``A_indices`` built at :meth:`initialize`.

        Returns:
            tuple: (P_data, coo_vals, q, b), each a 1-D NumPy float array.
        """
        if not (self._dyn and self._cons and self._pen):
            raise RuntimeError(
                "MoreauPTRSolver.solve() requires update_dynamics_linearization, "
                "update_constraint_linearizations, and update_penalties to all "
                "have been called this iteration."
            )

        L = self.layout
        settings = self._settings
        N, n_x, n_u = L.N, L.n_x, L.n_u
        inv_S_x = self._inv_S_x_diag
        inv_S_u = self._inv_S_u_diag
        S_x = self._S_x_diag
        c_x = self._c_x
        c_u = self._c_u

        lam_prox = self._pen["lam_prox"]  # (N, n_x + n_u)
        lam_cost = self._pen["lam_cost"]  # scalar or (n_x,)
        lam_vc = self._pen["lam_vc"]  # (N-1, n_x)
        lam_vb_nodal = self._pen["lam_vb_nodal"]  # (N, n_nodal or 1)

        x_bar = self._dyn["x_bar"]  # (N, n_x)
        u_bar = self._dyn["u_bar"]  # (N, n_u)
        A_d = self._dyn["A_d"]  # (N-1, n_x, n_x)
        B_d = self._dyn["B_d"]  # (N-1, n_x, n_u)
        C_d = self._dyn["C_d"]  # (N-1, n_x, n_u)
        x_prop = self._dyn["x_prop"]  # (N-1, n_x)

        lam_cost_arr = np.broadcast_to(lam_cost, (settings.sim.n_states,))

        # ---- P_data (diagonal trust-region weights on dx/du) ----
        # P_diag_slots = [dx_0..dx_{N·n_x-1}, du_0..du_{N·n_u-1}]
        P_data = np.empty(len(self._P_diag_slots))
        n_dx = self._P_n_dx  # N * n_x
        for k in range(N):
            for j in range(n_x):
                P_data[k * n_x + j] = 2.0 * lam_prox[k, j] + _P_DIAG_EPS
        for k in range(N):
            for j in range(n_u):
                P_data[n_dx + k * n_u + j] = 2.0 * lam_prox[k, n_x + j] + _P_DIAG_EPS

        # ---- q (linear cost vector) ----
        q = np.zeros(L.n_z)
        for i in range(settings.sim.true_state_slice.start, settings.sim.true_state_slice.stop):
            init_t = settings.sim.x.initial_type[i]
            final_t = settings.sim.x.final_type[i]
            if init_t == "Minimize":
                q[L.x_idx(0, i)] += lam_cost_arr[i]
            elif init_t == "Maximize":
                q[L.x_idx(0, i)] -= lam_cost_arr[i]
            if final_t == "Minimize":
                q[L.x_idx(N - 1, i)] += lam_cost_arr[i]
            elif final_t == "Maximize":
                q[L.x_idx(N - 1, i)] -= lam_cost_arr[i]
        for k in range(N - 1):
            for j in range(n_x):
                q[L.t_vc_idx(k, j)] += lam_vc[k, j]
        for c_idx in range(L.n_nodal):
            for k in range(N):
                q[L.t_vb_idx(c_idx, k)] += lam_vb_nodal[k, c_idx]

        # ---- A values and b (in same row order as _structural_pass) ----
        # Convention: s = b − Az.
        #   Zero cone:  s = 0  → A has same signs as "Az = b" formulation.
        #   Nonneg:     s ≥ 0  → b − Az ≥ 0.
        #   SOC:        s ∈ K  → (b − Az) ∈ K_soc.

        coo_vals: List[float] = []
        b_list: List[float] = []

        def emit(a_coeffs: List[float], rhs: float) -> None:
            """Append one row's A-coefficients (in col order) and its b entry."""
            coo_vals.extend(a_coeffs)
            b_list.append(rhs)

        # ================================================================
        # ZERO CONE rows
        # ================================================================

        # State error defs: x[k,j] − dx[k,j] = inv_S_x[j]·(x_bar[k,j] − c_x[j])
        for k in range(N):
            for j in range(n_x):
                emit([1.0, -1.0], inv_S_x[j] * (x_bar[k, j] - c_x[j]))

        # Control error defs: u[k,j] − du[k,j] = inv_S_u[j]·(u_bar[k,j] − c_u[j])
        for k in range(N):
            for j in range(n_u):
                emit([1.0, -1.0], inv_S_u[j] * (u_bar[k, j] - c_u[j]))

        # Dynamics (continuous FOH):
        #   x[k] − A_blk·dx[k-1] − B_blk·du[k-1] − C_blk·du[k] − nu[k-1]
        #     = inv_S_x·(x_prop[k-1] − c_x)
        for k in range(1, N):
            kp = k - 1
            A_blk = (inv_S_x[:, None] * A_d[kp]) * S_x[None, :]
            B_blk = (inv_S_x[:, None] * B_d[kp]) * self._S_u_diag[None, :]
            C_blk = (inv_S_x[:, None] * C_d[kp]) * self._S_u_diag[None, :]
            rhs_k = inv_S_x * (x_prop[kp] - c_x)
            for i in range(n_x):
                # Coefficients in the same col order as _structural_pass added them,
                # then sorted by scipy within the row — values line up correctly.
                coeffs: List[float] = [1.0]  # x[k, i]
                for j in range(n_x):
                    coeffs.append(-A_blk[i, j])  # dx[kp, j]
                for j in range(n_u):
                    coeffs.append(-B_blk[i, j])  # du[kp, j]
                    coeffs.append(-C_blk[i, j])  # du[k,  j]
                coeffs.append(-1.0)  # nu[kp, i]
                emit(coeffs, rhs_k[i])

        # Linearized nodal constraints:
        #   grad_x·dx[k] + grad_u·du[k] − nu_vb[c,k] = −g[c,k]
        jax_constraints = self._jax_constraints
        for c_idx, constraint in enumerate(jax_constraints.nodal):
            data = self._cons["nodal"][c_idx]
            g = data["g"]  # (N,)
            grad_x = data["grad_g_x"]  # (N, n_x)
            grad_u = data["grad_g_u"]  # (N, n_u)
            for node in constraint.nodes:
                coeffs = []
                for j in range(n_x):
                    coeffs.append(grad_x[node, j])  # dx[node, j]
                for j in range(n_u):
                    coeffs.append(grad_u[node, j])  # du[node, j]
                coeffs.append(-1.0)  # nu_vb[c, node]
                emit(coeffs, -g[node])

        # Fix boundary conditions
        for i in range(settings.sim.true_state_slice.start, settings.sim.true_state_slice.stop):
            if settings.sim.x.initial_type[i] == "Fix":
                if self._x_init is None:
                    raise RuntimeError(
                        f"Fix initial condition on state {i} requires x_init; "
                        "call update_boundary_conditions() before solve()."
                    )
                emit([S_x[i]], self._x_init[i] - c_x[i])
            if settings.sim.x.final_type[i] == "Fix":
                if self._x_term is None:
                    raise RuntimeError(
                        f"Fix final condition on state {i} requires x_term; "
                        "call update_boundary_conditions() before solve()."
                    )
                emit([S_x[i]], self._x_term[i] - c_x[i])

        # Uniform time grid: u[k,j] − u[k-1,j] = 0
        if settings.sim._uniform_time_grid:
            td = settings.sim.time_dilation_slice
            for k in range(1, N):
                for j in range(td.start, td.stop):
                    emit([1.0, -1.0], 0.0)

        # CTCS initial: S_x[idx]·x[0,idx] = −c_x[idx]
        for idx in range(settings.sim.ctcs_slice.start, settings.sim.ctcs_slice.stop):
            emit([S_x[idx]], -c_x[idx])

        # ================================================================
        # NONNEG CONE rows (s = b − Az ≥ 0)
        # ================================================================
        x_max_sc = inv_S_x * (np.asarray(settings.sim.x.max, dtype=float) - c_x)
        x_min_sc = inv_S_x * (np.asarray(settings.sim.x.min, dtype=float) - c_x)
        u_max_sc = inv_S_u * (np.asarray(settings.sim.u.max, dtype=float) - c_u)
        u_min_sc = inv_S_u * (np.asarray(settings.sim.u.min, dtype=float) - c_u)

        # State box upper: s = x_max_sc[j] − x[k,j] ≥ 0
        for k in range(N):
            for j in range(n_x):
                emit([1.0], x_max_sc[j])

        # State box lower: s = x[k,j] − x_min_sc[j] ≥ 0  →  b=−x_min_sc, A=−1
        for k in range(N):
            for j in range(n_x):
                emit([-1.0], -x_min_sc[j])

        # Control box upper
        for k in range(N):
            for j in range(n_u):
                emit([1.0], u_max_sc[j])

        # Control box lower
        for k in range(N):
            for j in range(n_u):
                emit([-1.0], -u_min_sc[j])

        # CTCS difference rows
        for idx, nodes in zip(
            range(settings.sim.ctcs_slice.start, settings.sim.ctcs_slice.stop),
            settings.sim.ctcs_node_intervals,
        ):
            start_i = 1 if nodes[0] == 0 else nodes[0]
            x_max_un = float(settings.sim.x.max[idx])
            scale = S_x[idx]
            for i in range(start_i, nodes[1]):
                # Row for +direction: s = x_max − scale·(x[i]−x[i-1]) ≥ 0
                # structural cols: x[i,idx] then x[i-1,idx]; scipy sorts by col.
                # x[i-1,idx] < x[i,idx] in column index, so after sort:
                #   data = [−scale (at x[i-1]), +scale (at x[i])]
                # → s = x_max − (−scale·x[i-1] + scale·x[i]) = x_max − scale·(x[i]−x[i-1]) ✓
                emit([scale, -scale], x_max_un)  # col order: x[i], x[i-1]
                # Row for −direction: s = x_max − scale·(x[i-1]−x[i]) ≥ 0
                emit([-scale, scale], x_max_un)  # col order: x[i], x[i-1]

        # Pos epigraph: t_vb[c,k] − nu_vb[c,k] ≥ 0
        # structural col order: nu_vb first, t_vb second (nu_vb < t_vb in layout)
        # scipy sort: nu_vb first (smaller idx) → coefficients below match that order
        for c_idx in range(L.n_nodal):
            for k in range(N):
                # [nu_vb coeff, t_vb coeff] after sorting by col
                emit([1.0, -1.0], 0.0)  # s = 0 − (nu_vb − t_vb) = t_vb − nu_vb ≥ 0

        # Pos epigraph non-negativity: t_vb[c,k] ≥ 0
        for c_idx in range(L.n_nodal):
            for k in range(N):
                emit([-1.0], 0.0)  # s = 0 − (−t_vb) = t_vb ≥ 0

        # ================================================================
        # SOC rows: SOC(2) for |nu[k,j]| ≤ t_vc[k,j]
        # s = b − Az: row 0 has only t_vc (coeff −1), row 1 has only nu (coeff −1)
        # s[0] = 0 − (−t_vc) = t_vc; s[1] = 0 − (−nu) = nu
        # ‖s[1:]‖ = |nu| ≤ s[0] = t_vc  ✓
        # ================================================================
        for k in range(N - 1):
            for j in range(n_x):
                emit([-1.0], 0.0)  # row for t_vc
                emit([-1.0], 0.0)  # row for nu

        return (
            P_data,
            np.asarray(coo_vals, dtype=float),
            q,
            np.asarray(b_list, dtype=float),
        )

    # ------------------------------------------------------------------
    # Solve / unpack
    # ------------------------------------------------------------------

    def solve(self) -> PTRSolveResult:
        """Assemble the conic subproblem and dispatch to ``moreau.jax.Solver``."""
        P_data, coo_vals, q, b = self._assemble_conic()

        # Rebuild CSR from the same COO col/row arrays (with real values).
        # scipy sorts col indices within each row identically to how the
        # structural CSR was built at initialize(), so A_csr.data aligns with
        # the fixed A_col_indices passed to moreau.
        A_csr = sp.csr_matrix(
            (coo_vals, (self._coo_rows, self._coo_cols)),
            shape=(self._n_con, self.layout.n_z),
        )
        A_csr.sort_indices()

        P_j = jnp.asarray(P_data)
        A_j = jnp.asarray(A_csr.data)
        q_j = jnp.asarray(q)
        b_j = jnp.asarray(b)

        solution = self._moreau.solve(P_j, A_j, q_j, b_j, warm_start=self._warm_start)

        self._last_status = moreau.SolverStatus(int(np.asarray(self._moreau.info.status)))
        if _moreau_solve_ok(self._last_status):
            self._warm_start = solution.to_warm_start()

        z = np.asarray(solution.x)
        return self._unpack(z)

    def _unpack(self, z: np.ndarray) -> PTRSolveResult:
        """Reverse the layout into the structured :class:`PTRSolveResult`."""
        L = self.layout
        N, n_x, n_u = L.N, L.n_x, L.n_u

        x_scaled = z[L.sl_x].reshape(N, n_x)
        u_scaled = z[L.sl_u].reshape(N, n_u)
        x = x_scaled * self._S_x_diag[None, :] + self._c_x[None, :]
        u = u_scaled * self._S_u_diag[None, :] + self._c_u[None, :]

        nu = z[L.sl_nu].reshape(N - 1, n_x)
        nu_vb = [np.asarray(z[sl]) for sl in L.sl_nu_vb]
        nu_vb_cross: List[float] = []

        is_ok = _moreau_solve_ok(self._last_status)
        status = "optimal" if is_ok else "infeasible"

        cost = self._reconstruct_cost(z)

        return PTRSolveResult(
            x=x,
            u=u,
            nu=nu,
            nu_vb=nu_vb,
            nu_vb_cross=nu_vb_cross,
            cost=cost,
            status=status,
        )

    def _reconstruct_cost(self, z: np.ndarray) -> float:
        """Recompute the PTR objective at ``z`` from the stored penalty weights.

        Uses the epigraph variables ``t_vc`` / ``t_vb`` (which equal
        ``|nu|`` / ``max(nu_vb, 0)`` at optimality) rather than recomputing
        absolute values or positive parts.
        """
        L = self.layout
        settings = self._settings
        N, n_x = L.N, L.n_x
        lam_prox = self._pen["lam_prox"]
        lam_cost_arr = np.broadcast_to(self._pen["lam_cost"], (settings.sim.n_states,))
        lam_vc = self._pen["lam_vc"]
        lam_vb_nodal = self._pen["lam_vb_nodal"]

        dx = z[L.sl_dx].reshape(N, n_x)
        du = z[L.sl_du].reshape(N, L.n_u)
        x = z[L.sl_x].reshape(N, n_x)
        t_vc = z[L.sl_t_vc].reshape(N - 1, n_x)

        cost = float(np.sum(lam_prox[:, :n_x] * dx**2) + np.sum(lam_prox[:, n_x:] * du**2))
        for i in range(settings.sim.true_state_slice.start, settings.sim.true_state_slice.stop):
            init_t = settings.sim.x.initial_type[i]
            final_t = settings.sim.x.final_type[i]
            if init_t == "Minimize":
                cost += float(lam_cost_arr[i] * x[0, i])
            elif init_t == "Maximize":
                cost -= float(lam_cost_arr[i] * x[0, i])
            if final_t == "Minimize":
                cost += float(lam_cost_arr[i] * x[-1, i])
            elif final_t == "Maximize":
                cost -= float(lam_cost_arr[i] * x[-1, i])
        cost += float(np.sum(lam_vc * t_vc))
        for c_idx in range(L.n_nodal):
            t_vb = z[L.sl_t_vb[c_idx]]
            cost += float(np.dot(lam_vb_nodal[:, c_idx], t_vb))
        return cost

    # ------------------------------------------------------------------
    # Misc API
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Conic problem dimensions for the diagnostics summary box."""
        if self.layout is None:
            return {"n_variables": 0, "n_parameters": 0, "n_constraints": 0}
        return {
            "n_variables": self.layout.n_z,
            "n_parameters": 0,
            "n_constraints": self._n_con if self._n_con else -1,
        }

    def citation(self) -> List[str]:
        """BibTeX entry for Moreau."""
        return [
            r"""@software{moreau2024,
  title={Moreau: GPU-Accelerated Conic Optimization Solver},
  author={{Moreau Contributors}},
  url={https://docs.moreau.so},
  year={2024}
}"""
        ]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_moreau_settings(solver_args: dict):
    """Convert a ``solver_args`` dict to a :class:`moreau.Settings` object."""
    args = dict(solver_args)
    raw_ipm = args.pop("ipm_settings", None)
    if isinstance(raw_ipm, dict):
        args["ipm_settings"] = moreau.IPMSettings(**raw_ipm)
    elif raw_ipm is not None:
        args["ipm_settings"] = raw_ipm
    return moreau.Settings(**args)
