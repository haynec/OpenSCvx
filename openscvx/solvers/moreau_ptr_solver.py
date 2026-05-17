"""JAX-native conic backend for the PTR convex subproblem.

Assembles each SCP subproblem as a sparse conic program (SOCP-capable) and
dispatches it to ``moreau.jax.Solver``, a JAX-native interior-point conic
solver. This backend sits alongside
:class:`openscvx.solvers.qpax_ptr_solver.QPAXPTRSolver` (QP-only) and
:class:`openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver` (DCP via CVXPy).

Compared to QPAX, Moreau accepts second-order cone (SOC) constraints natively.
QPAX slack-reformulates the PTR penalties — ``|nu|`` (L1 virtual control) and
``pos(nu_vb)`` (positive-part virtual buffer) — with extra slack variables and
doubled inequality rows. Here ``|nu_i| <= t_i`` is a 2-D SOC on
``[t_i, nu_i]`` with linear cost ``lam_vc * t_i``, and ``pos(nu_vb)`` uses a
nonnegative epigraph instead of two inequality rows, yielding a smaller
problem for the same PTR math.

``moreau.jax.Solver`` differentiates through the solve via implicit
differentiation on the KKT conditions. Once the surrounding SCP pipeline stays
in ``jit``, ``jax.grad`` / ``jax.vmap`` can reach through a full SCvx solve
(future work).

Scope:
    * No user ``.convex()`` constraints — rejected upstream by
      :meth:`ConvexSolver.lower_convex_constraints`.
    * No cross-node constraints — raises :class:`NotImplementedError` at
      :meth:`MoreauPTRSolver.initialize`.
    * Impulsive controls (``parameterization="impulsive"``) are supported.
      ``D_d`` is absorbed numerically into ``A_d / B_d / C_d`` at update
      time and ``E_d`` enters the dynamics row on the impulsive control
      slice; the initial Fix boundary condition picks up the linearized
      impulse at node 0.  The static CSR pattern reserves the
      ``du[0, slice_imp]`` columns in the initial-Fix rows so warm-start
      structure stays valid across iterations.
    * CTCS constraints are supported; LICQ-style absolute-value inequalities
      are affine and fit in the nonneg cone.

Warm-start:
    Unlike QPAX, ``MoreauPTRSolver`` passes a
    :class:`moreau._types.WarmStart` from the previous successful solve into
    each ``solver.solve()`` call when the status is
    :attr:`moreau.SolverStatus.Solved` or
    :attr:`moreau.SolverStatus.AlmostSolved`.

Conic formulation:
    Moreau solves::

        min  0.5 zᵀ P z + qᵀ z
        s.t. A z + s = b,  s ∈ K

    where ``K`` is a product of zero, nonneg, and SOC cones (first to last).
    Zero cone: ``s = 0`` → ``Az = b``. Nonneg cone: ``s ≥ 0`` → ``Az ≤ b``.
    SOC(d): ``‖(b − Az)[1:]‖ ≤ (b − Az)[0]``.

Note:
    Moreau performs internal arithmetic in float64. Pass
    ``float_dtype="float64"`` to :class:`openscvx.problem.Problem` for tight
    inner-solver tolerances; the default ``float32`` caps conditioning.
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
    from openscvx.symbolic.constraint_set import ConstraintSet

# Tiny diagonal regularisation added to P on dx/du slots.  Moreau's IPM
# Cholesky factors (P + Gᵀ diag(z/s) G); keeping the diagonal positive avoids
# near-singular factorisation when lam_prox is very small.
_P_DIAG_EPS = 1e-10


def _moreau_solve_ok(status: "moreau.SolverStatus") -> bool:
    """Check whether a Moreau solve status indicates a usable primal.

    Args:
        status: Status returned by ``moreau.jax.Solver`` after a solve.

    Returns:
        True when ``status`` is :attr:`moreau.SolverStatus.Solved` or
        :attr:`moreau.SolverStatus.AlmostSolved`.
    """
    return status in (
        moreau.SolverStatus.Solved,
        moreau.SolverStatus.AlmostSolved,
    )


# ---------------------------------------------------------------------------
# Decision-vector layout
# ---------------------------------------------------------------------------


@dataclass
class _ConicLayout:
    """Static index layout for the flat decision vector ``z``.

    Moreau's conic formulation drops the slack blocks used by QPAX
    (``s_abs``, ``s_pos``) and replaces them with epigraph variables
    ``t_vc`` / ``t_vb``:

    * ``|nu[k,j]| ≤ t_vc[k,j]`` as a SOC(2).
    * ``nu_vb[c,k] ≤ t_vb[c,k]`` and ``t_vb[c,k] ≥ 0`` as two nonneg rows.

    Slices are computed once at :meth:`MoreauPTRSolver.create_variables` and
    reused on every solve.

    Attributes:
        N: Number of discretization nodes.
        n_x: State dimension per node.
        n_u: Control dimension per node.
        n_nodal: Number of nodal nonconvex constraints.
        n_z: Total decision-vector length.
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
        """Flat index for virtual control ``nu[k, j]``.

        Args:
            k: Segment index in ``[0, N-2]``.
            j: State component index.
        """
        return self.sl_nu.start + k * self.n_x + j

    def nu_vb_idx(self, c: int, k: int) -> int:
        return self.sl_nu_vb[c].start + k

    def t_vc_idx(self, k: int, j: int) -> int:
        """Flat index for the ``|nu|`` epigraph variable ``t_vc[k, j]``.

        Args:
            k: Segment index in ``[0, N-2]``.
            j: State component index.
        """
        return self.sl_t_vc.start + k * self.n_x + j

    def t_vb_idx(self, c: int, k: int) -> int:
        return self.sl_t_vb[c].start + k


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------


class MoreauPTRSolver(PTRSolver):
    """JAX-native conic backend for the PTR convex subproblem.

    Assembles each SCP subproblem as a sparse conic program and dispatches to
    ``moreau.jax.Solver``. See the module docstring for the SOC epigraph
    encoding, warm-start policy, and float64 advice.

    Compared to :class:`QPAXPTRSolver`, this backend uses fewer decision
    variables and constraint rows for the same PTR physics (SOC epigraphs for
    ``|nu|`` instead of two nonneg slack rows each), warm-starts between SCP
    iterations on successful solves, and opens a path to user ``.convex()``
    SOCP support in a follow-up.

    Scope:
        Supported — state/control box, dynamics linearization (continuous
        and impulsive), boundary Fix, uniform time grid, linearized nodal
        nonconvex, CTCS LICQ rows.

        Not supported — user ``.convex()`` constraints and cross-node
        constraints. Each raises :class:`NotImplementedError` with a "use
        :class:`openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver`" pointer.

    Differentiability hook for future work:
        ``moreau.jax.Solver`` differentiates through the solve via implicit
        differentiation on the KKT conditions. Once the surrounding SCP
        pipeline stays in ``jit``, ``jax.grad`` / ``jax.vmap`` can reach
        through a full SCvx solve.

    Args:
        max_iter: Maximum number of IPM iterations forwarded to
            :class:`moreau.Settings`. Defaults to ``200``.
        verbose: Whether Moreau prints per-iteration diagnostics.
            Forwarded to :class:`moreau.Settings`. Defaults to ``False``.
        device: Compute device for Moreau's JAX kernels. One of
            ``"auto"``, ``"cpu"``, or ``"cuda"``. Forwarded to
            :class:`moreau.Settings`. Defaults to ``"auto"``.
        tol_gap_abs: Absolute duality-gap tolerance forwarded to
            :class:`moreau.IPMSettings`. ``None`` uses Moreau's default.
        tol_feas: Primal/dual feasibility tolerance forwarded to
            :class:`moreau.IPMSettings`. ``None`` uses Moreau's default.
        solver_args: Additional keyword arguments forwarded verbatim to
            :class:`moreau.Settings`. Use for settings not covered by the
            named params above. ``solver_args["ipm_settings"]`` may be a
            dict or a :class:`moreau.IPMSettings` object; if it is a dict,
            ``tol_gap_abs`` / ``tol_feas`` are merged into it. Raises
            ``ValueError`` at construction time if any top-level key or
            IPM tolerance overlaps with a named param.

    Attributes:
        layout: :class:`_ConicLayout` describing flat decision-vector slot
            ranges. Populated by :meth:`create_variables`.
    """

    def __init__(
        self,
        *,
        max_iter: int = 200,
        verbose: bool = False,
        device: str = "auto",
        tol_gap_abs: Optional[float] = None,
        tol_feas: Optional[float] = None,
        solver_args: Optional[Dict] = None,
    ):
        if not _MOREAU_AVAILABLE:
            raise ImportError(
                "MoreauPTRSolver requires the `moreau` package. "
                "Install it with: pip install openscvx[moreau]"
            )

        _named = {"max_iter": max_iter, "verbose": verbose, "device": device}
        _extra = dict(solver_args) if solver_args else {}
        _overlap = _named.keys() & _extra.keys()
        if _overlap:
            raise ValueError(
                f"Moreau settings {sorted(_overlap)} appear as both named arguments "
                "and inside solver_args; use one or the other."
            )
        merged = {**_named, **_extra}

        _ipm = {
            k: v for k, v in [("tol_gap_abs", tol_gap_abs), ("tol_feas", tol_feas)] if v is not None
        }
        if _ipm:
            existing_ipm = merged.get("ipm_settings", {})
            if not isinstance(existing_ipm, dict):
                raise ValueError(
                    "Cannot combine tol_gap_abs / tol_feas named arguments with an "
                    "ipm_settings object in solver_args; use one form or the other."
                )
            ipm_overlap = _ipm.keys() & existing_ipm.keys()
            if ipm_overlap:
                raise ValueError(
                    f"Moreau IPM settings {sorted(ipm_overlap)} appear as both named "
                    "arguments and inside solver_args['ipm_settings']; use one or the other."
                )
            merged["ipm_settings"] = {**_ipm, **existing_ipm}

        self.solver_args = merged

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

        # Populated by lower_convex_constraints. Auto-augmented impulsive
        # zero-pin constraints land here; any genuine user .convex() trips
        # the default refusal.
        self._impulsive_pins: List[Tuple[List[int], slice]] = []

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
        :class:`openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver` but ignored;
        numeric values are filled into a fixed CSR structure at solve time.

        Args:
            N: Number of discretization nodes.
            x_unified: Unified state bounds and scaling metadata.
            u_unified: Unified control bounds and scaling metadata.
            jax_constraints: Lowered JAX constraints (nodal structure only).
            dynamics_sparsity: Ignored.
            constraint_sparsity: Ignored.
        """
        del dynamics_sparsity, constraint_sparsity

        n_x = len(x_unified.max)
        n_u = len(u_unified.max)

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

    def lower_convex_constraints(
        self,
        constraints: "ConstraintSet",
        parameters: Optional[Dict] = None,
    ) -> Tuple[List, Dict]:
        """Absorb auto-generated impulsive zero-pin constraints; refuse the rest.

        :func:`openscvx.symbolic.lower._augment_impulsive_constraints` injects
        ``Control == 0`` equalities at non-impulse nodes for every impulsive
        control. Those constraints live in ``constraints.nodal_convex`` even
        though no user ``.convex()`` was written; we recognize their fixed
        shape and stash a pin list for the structural pass / assembler to
        emit as plain zero-cone rows. Anything that doesn't match the
        auto-augmentation shape (e.g. a genuine user ``.convex()`` SOC) falls
        through to the default refusal in :class:`ConvexSolver`.
        """
        pins = self._extract_impulsive_pins(constraints)
        if pins is None:
            return super().lower_convex_constraints(constraints, parameters)
        self._impulsive_pins = pins
        return [], {}

    def initialize(self, lowered: "LoweredProblem", settings: "Config") -> None:
        """Build the static conic structure and construct ``moreau.jax.Solver``.

        Validates the supported constraint subset, enumerates all A-matrix
        nonzero positions (fixed CSR structure), builds a :class:`moreau.Cones`
        spec, and constructs :class:`moreau.jax.Solver`. Numeric values in
        ``A``, ``b``, ``P``, and ``q`` are filled at each :meth:`solve` call.

        Args:
            lowered: Lowered problem with constraint structure.
            settings: Problem configuration.

        Raises:
            RuntimeError: If :meth:`create_variables` was not called first.
            NotImplementedError: If cross-node constraints are present.
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
        A_eff = np.asarray(A_d, dtype=float)
        B_eff = np.asarray(B_d, dtype=float)
        C_eff = np.asarray(C_d, dtype=float)

        # Absorb the impulsive state Jacobian into the continuous step
        # matrices so the assembly row keeps a single A_d·dx + B_d·du term,
        # matching the recipe in CVXPyPTRSolver.update_dynamics_linearization
        # at openscvx/solvers/cvxpy_ptr_solver.py:636-654.
        if D_d is not None:
            D_arr = np.asarray(D_d, dtype=float)
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

        self._dyn = {
            "x_bar": np.asarray(x_bar, dtype=float),
            "u_bar": np.asarray(u_bar, dtype=float),
            "A_d": A_eff,
            "B_d": B_eff,
            "C_d": C_eff,
            "x_prop": np.asarray(x_prop, dtype=float),
            "x_prop_plus": (
                np.asarray(x_prop_plus, dtype=float) if x_prop_plus is not None else None
            ),
            "E_d": np.asarray(E_d, dtype=float) if E_d is not None else None,
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

        Args:
            settings: Problem configuration (bounds, CTCS groups, time grid).

        Returns:
            tuple: ``(coo_rows, coo_cols, n_eq, n_nn, soc_dims)`` where
            ``coo_rows`` / ``coo_cols`` list every A-matrix nonzero in
            traversal order, ``n_eq`` / ``n_nn`` count zero- and nonneg-cone
            rows, and ``soc_dims`` lists SOC dimensions (2 per ``|nu[k,j]|``).
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

        # Fix boundary conditions (initial / terminal).  Under impulsive
        # control the initial Fix row couples the post-impulse state at
        # node 0 to du[0, slice_imp]; declare those columns structurally
        # nonzero here so the static CSR pattern accommodates the
        # numerical values emitted in _assemble_conic.
        slice_imp = settings.sim.u.slice_impulsive
        has_impulsive = slice_imp.stop > slice_imp.start
        for i in range(settings.sim.true_state_slice.start, settings.sim.true_state_slice.stop):
            if settings.sim.x.initial_type[i] == "Fix":
                add(row, L.x_idx(0, i))
                if has_impulsive:
                    for j in range(slice_imp.start, slice_imp.stop):
                        add(row, L.du_idx(0, j))
                row += 1
            if settings.sim.x.final_type[i] == "Fix":
                add(row, L.x_idx(N - 1, i))
                row += 1

        # Impulsive zero-pin equalities: u[node, j] = const, one row per
        # (node, j) absorbed by lower_convex_constraints.
        for nodes, ctrl_slice in self._impulsive_pins:
            for node in nodes:
                for j in range(ctrl_slice.start, ctrl_slice.stop):
                    add(row, L.u_idx(node, j))
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
        so the emitted value list matches ``self._coo_rows`` /
        ``self._coo_cols``. scipy builds the same sorted CSR each call, so
        ``A_csr.data`` aligns with the fixed ``A_indices`` from
        :meth:`initialize`.

        Returns:
            tuple: ``(P_data, coo_vals, q, b)`` as 1-D NumPy float arrays.

        Raises:
            RuntimeError: If per-iteration update hooks were not called.
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
        S_u = self._S_u_diag
        c_x = self._c_x
        c_u = self._c_u
        slice_imp = settings.sim.u.slice_impulsive
        has_impulsive = slice_imp.stop > slice_imp.start

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
        E_d_arr = self._dyn["E_d"]  # (N, n_x, n_u) or None
        x_prop_plus_arr = self._dyn["x_prop_plus"]  # (N, n_x) or None

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
            """Append one constraint row's A coefficients and RHS."""
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

        # Dynamics (continuous FOH, with optional impulsive coupling at node k):
        #   x[k] − A_blk·dx[k-1] − B_blk·du[k-1] − C_blk·du[k]
        #        − E_blk·du[k][slice_imp] − nu[k-1]
        #     = inv_S_x·(x_prop_plus[k] − c_x)   if has_impulsive
        #     = inv_S_x·(x_prop[k-1]    − c_x)   otherwise
        # Mirrors CVXPyPTRSolver.constraints at cvxpy_ptr_solver.py:506-530.
        for k in range(1, N):
            kp = k - 1
            A_blk = (inv_S_x[:, None] * A_d[kp]) * S_x[None, :]
            B_blk = (inv_S_x[:, None] * B_d[kp]) * S_u[None, :]
            C_blk = (inv_S_x[:, None] * C_d[kp]) * S_u[None, :]
            if has_impulsive:
                E_blk = (inv_S_x[:, None] * E_d_arr[k]) * S_u[None, :]
                rhs_k = inv_S_x * (x_prop_plus_arr[k] - c_x)
            else:
                E_blk = None
                rhs_k = inv_S_x * (x_prop[kp] - c_x)
            for i in range(n_x):
                # Coefficients in the same col order as _structural_pass added them,
                # then sorted by scipy within the row — values line up correctly.
                coeffs: List[float] = [1.0]  # x[k, i]
                for j in range(n_x):
                    coeffs.append(-A_blk[i, j])  # dx[kp, j]
                for j in range(n_u):
                    coeffs.append(-B_blk[i, j])  # du[kp, j]
                    c_kj = -C_blk[i, j]
                    if has_impulsive and slice_imp.start <= j < slice_imp.stop:
                        c_kj -= E_blk[i, j]
                    coeffs.append(c_kj)  # du[k, j]
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

        # Fix boundary conditions.  Under impulsive control the initial Fix
        # row couples x[0, i] to du[0, slice_imp] via the linearized impulse
        # Jacobian (CVXPy reference: cvxpy_ptr_solver.py:484-495).  Emit
        # coefficients in ascending column-index order so they match the
        # CSR sort applied to the structural pass.
        for i in range(settings.sim.true_state_slice.start, settings.sim.true_state_slice.stop):
            if settings.sim.x.initial_type[i] == "Fix":
                if has_impulsive:
                    coeffs = [S_x[i]]
                    for j in range(slice_imp.start, slice_imp.stop):
                        coeffs.append(-E_d_arr[0, i, j] * S_u[j])
                    emit(coeffs, x_prop_plus_arr[0, i] - c_x[i])
                else:
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

        # Impulsive zero-pin equalities: u[node, j] = −inv_S_u[j]·c_u[j].
        # Mirrors CVXPy's lowering of ``u_nonscaled[node][slice_imp] == 0``.
        for nodes, ctrl_slice in self._impulsive_pins:
            for node in nodes:
                for j in range(ctrl_slice.start, ctrl_slice.stop):
                    emit([1.0], -inv_S_u[j] * c_u[j])

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
        """Assemble the conic subproblem and dispatch to ``moreau.jax.Solver``.

        Updates the warm-start carry only when the solve status is
        :attr:`moreau.SolverStatus.Solved` or
        :attr:`moreau.SolverStatus.AlmostSolved`.

        Returns:
            PTRSolveResult: Unscaled trajectories and solver status for this
            SCP iteration.
        """
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
        """Map the flat solution vector into a :class:`PTRSolveResult`.

        Args:
            z: Primal solution from Moreau, length ``layout.n_z``.

        Returns:
            PTRSolveResult: Unscaled ``x`` / ``u`` trajectories, slacks, cost,
            and ``"optimal"`` or ``"infeasible"`` status.
        """
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
        """Recompute the PTR objective at ``z`` from stored penalty weights.

        Uses epigraph variables ``t_vc`` / ``t_vb`` (equal to ``|nu|`` /
        ``max(nu_vb, 0)`` at optimality) instead of absolute values or
        positive parts.

        Args:
            z: Primal solution vector.

        Returns:
            Scalar PTR cost at ``z``.
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
        """Return conic problem dimensions for the diagnostics summary box.

        Returns:
            dict: Keys ``n_variables``, ``n_parameters`` (always 0), and
            ``n_constraints``.
        """
        if self.layout is None:
            return {"n_variables": 0, "n_parameters": 0, "n_constraints": 0}
        return {
            "n_variables": self.layout.n_z,
            "n_parameters": 0,
            "n_constraints": self._n_con if self._n_con else -1,
        }

    def citation(self) -> List[str]:
        """Return BibTeX citation entries for Moreau.

        Returns:
            list[str]: BibTeX strings for inclusion in a bibliography.
        """
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
    """Convert a ``solver_args`` dict to a :class:`moreau.Settings` instance.

    Args:
        solver_args: User-provided solver options; ``ipm_settings`` may be a
            dict that is expanded into :class:`moreau.IPMSettings`.

    Returns:
        moreau.Settings: Configured settings for :class:`moreau.jax.Solver`.
    """
    args = dict(solver_args)
    raw_ipm = args.pop("ipm_settings", None)
    if isinstance(raw_ipm, dict):
        args["ipm_settings"] = moreau.IPMSettings(**raw_ipm)
    elif raw_ipm is not None:
        args["ipm_settings"] = raw_ipm
    return moreau.Settings(**args)
