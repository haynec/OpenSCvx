"""JAX-native QP backend for the PTR convex subproblem.

Assembles each SCP subproblem as a flat ``(Q, q, A, b, G, h)`` quadratic
program and dispatches it to ``qpax.solve_qp``. The JAX-pure
:meth:`iteration_callback` path composes with ``jax.jit`` and ``jax.vmap``,
which makes it the wedge toward batched, end-to-end-JAX SCvx. It uses
``qpax.solve_qp`` for its reliable convergence flag rather than the
``jax.custom_vjp``-differentiable ``qpax.solve_qp_primal``: qpax exposes one
or the other, not both, and the SCP loop needs to fail loudly on a diverged
solve (see :meth:`iteration_callback`). ``solve_qp_primal`` remains the seam
toward a ``jax.grad``-differentiable loop — future work, see
``plans/jax-pure-solve.md``.

Scope
-----
* No user ``.convex()`` constraints (would need SOCP).
* No cross-node constraints. Each raises :class:`NotImplementedError` at
  :meth:`initialize` time and points the user at
  :class:`openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver` as the
  alternative. Cross-node support is gated on full-trajectory gradient
  stacking that hasn't been built yet.
* Impulsive controls (``parameterization="impulsive"``) **are** supported.
  ``D_d`` is absorbed numerically into ``A_d / B_d / C_d`` at update time
  (matching ``CVXPyPTRSolver``) and ``E_d`` enters the dynamics row as an
  additional control coefficient on the impulsive slice; the initial Fix
  boundary condition picks up the linearized impulse at node 0.
* CTCS constraints **are** supported — their LICQ-style absolute-value
  inequalities reduce to two affine rows per node, which is plain QP form.
  The library also auto-adds ``CTCS(time ≤ time.max)`` /
  ``CTCS(time.min ≤ time)`` to every problem, so this support is what
  makes QPAX usable at all.
* No warm-start. ``qpax.solve_qp`` initializes its own primal-dual state on
  every call and exposes no init hook; only ``qpax.relax_qp`` accepts a
  warm-start tuple. SCvx warm-starting is a known performance gap vs
  CVXPy+QOCO; could be threaded through here once upstream qpax exposes
  the seam.
* Dense ``Q``, ``A``, ``G``. Trust-region terms are diagonal-dominant per
  node and slack terms are sparse, so a sparse assembly may help in the
  future; at ``N`` ≤ 100 the dense path is simpler and fast enough.

L1 / positive-part reformulation
--------------------------------
The PTR cost contains ``|nu|`` (virtual-control L1) and ``pos(nu_vb)``
(positive-part virtual buffer). Each ``|νᵢ|`` is replaced by a slack
``sᵢ`` plus ``±νᵢ - sᵢ ≤ 0`` (the implied ``sᵢ ≥ 0`` follows from
both inequalities); each ``pos(ν_vbᵢ)`` becomes ``sᵢ ≥ νᵢ`` *and*
``sᵢ ≥ 0`` (the latter is explicit because pos has no symmetric pair).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from openscvx.config import Config

from .cones import NonnegConeConstraint, SOCConstraint, ZeroConeConstraint
from .ptr_solver import (
    PTRSolver,
    PTRSolveResult,
    StatusCode,
    SubproblemData,
    SubproblemSolution,
)

try:
    import jax
    import jax.numpy as jnp
    import qpax

    _QPAX_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised by the install-error test
    qpax = None
    jax = None
    jnp = None
    _QPAX_AVAILABLE = False


if TYPE_CHECKING:
    from openscvx.lowered import LoweredProblem
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints
    from openscvx.lowered.unified import UnifiedControl, UnifiedState
    from openscvx.symbolic.constraint_set import ConstraintSet


# Tiny diagonal regularization added to Q on rows whose cost is purely
# linear (nu, nu_vb, slacks, x, u). qpax's PDIP path Choleskys ``Q +
# Gᵀ diag(z/s) G``; keeping every diagonal entry strictly positive avoids a
# rank-deficient factorization the very first iteration and costs nothing in
# the final solution.
_Q_DIAG_EPS = 1e-10


@dataclass
class _QPLayout:
    """Static index layout for the flat decision vector ``z``.

    The PTR variables and their slack reformulations sit in one concatenated
    1-D vector. Slices are computed once at :meth:`QPAXPTRSolver.create_variables`
    time and reused on every solve.
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
    sl_s_abs: slice = field(init=False)
    sl_s_pos: List[slice] = field(init=False)
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
        self.sl_s_abs = take((N - 1) * n_x)
        self.sl_s_pos = [take(N) for _ in range(C)]
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
        """``k`` ∈ [0, N-2], indexing into ν[1..N-1] in the CVXPy notation."""
        return self.sl_nu.start + k * self.n_x + j

    def nu_vb_idx(self, c: int, k: int) -> int:
        return self.sl_nu_vb[c].start + k

    def s_abs_idx(self, k: int, j: int) -> int:
        return self.sl_s_abs.start + k * self.n_x + j

    def s_pos_idx(self, c: int, k: int) -> int:
        return self.sl_s_pos[c].start + k


class QPAXPTRSolver(PTRSolver):
    """JAX-native QP backend for the PTR convex subproblem.

    Assembles each SCP subproblem as a flat ``(Q, q, A, b, G, h)`` and
    dispatches to ``qpax.solve_qp``. See the module docstring for the L1 /
    positive-part slack reformulation and the rationale behind the design.

    The JAX-pure entry point :meth:`iteration_callback` performs the same
    ``assemble → solve → unpack`` cycle entirely on the JAX boundary using
    ``qpax.solve_qp``, so the backend composes with ``jax.jit`` and
    ``jax.vmap``. It is not reverse-mode differentiable: ``solve_qp`` carries
    the convergence flag the SCP loop needs but no ``custom_vjp`` rule (see
    :meth:`iteration_callback`).

    Scope:
        Supported — state/control box, dynamics linearization (continuous
        and impulsive), boundary ``Fix``, uniform time grid, linearized
        nodal nonconvex, CTCS LICQ-style rows, user affine equality /
        inequality ``.convex()`` constraints
        (:class:`~openscvx.solvers.cones.ZeroConeConstraint` and
        :class:`~openscvx.solvers.cones.NonnegConeConstraint`).

        Not supported — SOC ``.convex()`` constraints and cross-node
        constraints.  Each raises :class:`NotImplementedError`.  SOC
        support requires a second-order-cone solver; use
        :class:`openscvx.solvers.moreau_ptr_solver.MoreauPTRSolver` or
        :class:`openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver` instead.

    Differentiability hook for future work:
        ``qpax.solve_qp_primal`` is differentiable via ``jax.custom_vjp``.
        Swapping it in here (once the surrounding pipeline stays in JIT) is
        what lets ``jax.grad`` / ``jax.vmap`` reach through a full SCvx solve.

    Args:
        solver_tol: Convergence tolerance forwarded to ``qpax.solve_qp``
            as ``solver_tol``. Defaults to ``1e-5``.
        max_iter: Maximum number of PDIP iterations forwarded to
            ``qpax.solve_qp`` as ``max_iter``. Defaults to ``30``.
        solver_args: Additional keyword arguments forwarded verbatim to
            ``qpax.solve_qp``. Use for settings not covered by the named
            params above — e.g. ``backend="e"`` for the explicit
            predictor-corrector path, or ``linear_solver``. Raises
            ``ValueError`` at construction time if any key overlaps with a
            named param.

    Attributes:
        layout: ``_QPLayout`` describing the flat decision-vector slot ranges.
            Populated by :meth:`create_variables`.
    """

    SUPPORTED_CONE_TYPES = frozenset({ZeroConeConstraint, NonnegConeConstraint})

    def __init__(
        self,
        *,
        solver_tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        solver_args: Optional[dict] = None,
    ):
        if not _QPAX_AVAILABLE:
            raise ImportError(
                "QPAXPTRSolver requires the `qpax` package. "
                "Install it with: pip install openscvx[qpax]"
            )

        # None sentinels at the signature surface let us distinguish
        # "user explicitly passed this" from "default applied" — only the
        # former should collide with the same key inside solver_args.
        _named = {
            k: v for k, v in (("solver_tol", solver_tol), ("max_iter", max_iter)) if v is not None
        }
        _extra = dict(solver_args) if solver_args else {}
        _overlap = _named.keys() & _extra.keys()
        if _overlap:
            raise ValueError(
                f"QPAX settings {sorted(_overlap)} appear as both named arguments "
                "and inside solver_args; use one or the other."
            )
        merged = {**_named, **_extra}
        merged.setdefault("solver_tol", 1e-5)
        merged.setdefault("max_iter", 30)
        self.solver_args = merged

        # Populated by create_variables / initialize.
        self.layout: Optional[_QPLayout] = None
        self._S_x: Optional[np.ndarray] = None
        self._c_x: Optional[np.ndarray] = None
        self._S_u: Optional[np.ndarray] = None
        self._c_u: Optional[np.ndarray] = None
        self._inv_S_x_diag: Optional[np.ndarray] = None
        self._inv_S_u_diag: Optional[np.ndarray] = None
        self._S_x_diag: Optional[np.ndarray] = None
        self._S_u_diag: Optional[np.ndarray] = None
        self._settings: Optional[Config] = None
        self._jax_constraints: Optional["LoweredJaxConstraints"] = None
        # Static QP row count, computed in initialize() once the constraint
        # set, impulsive pins, and settings are all available.
        self._n_constraints: int = 0

        # Populated by lower_convex_constraints. Auto-augmented impulsive
        # zero-pin constraints land here; genuine user .convex() constraints
        # land in _user_cone_constraints after canonicalization.
        self._impulsive_pins: List[Tuple[List[int], slice]] = []
        self._user_cone_constraints: List = []
        self._parameters_dict: dict = {}

        # Per-iteration data, set by update_* methods.
        self._dyn: dict = {}
        self._cons: dict = {}
        self._pen: dict = {}
        self._x_init: Optional[np.ndarray] = None
        self._x_term: Optional[np.ndarray] = None

        # Last solve diagnostics (populated by solve()).
        self._last_iters: Optional[int] = None
        self._last_converged: Optional[bool] = None

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
        """Compute scaling matrices and the static QP decision-vector layout.

        Sparsity hints are accepted for interface symmetry with
        :class:`openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver` but ignored
        — QPAX consumes dense arrays.
        """
        del dynamics_sparsity, constraint_sparsity  # QPAX consumes dense arrays

        n_x = len(x_unified.max)
        n_u = len(u_unified.max)

        S_x, c_x = self._scaling(x_unified)
        S_u, c_u = self._scaling(u_unified)

        self._S_x = S_x
        self._c_x = c_x
        self._S_u = S_u
        self._c_u = c_u
        # S_x / S_u are diagonal — keep the diagonals around for fast scalar
        # arithmetic in the assembly loops.
        self._S_x_diag = np.diag(S_x)
        self._S_u_diag = np.diag(S_u)
        self._inv_S_x_diag = 1.0 / self._S_x_diag
        self._inv_S_u_diag = 1.0 / self._S_u_diag

        # JAX-side scaling diagonals for ``iteration_callback``. The dtype
        # tracks the global ``jax_enable_x64`` setting so the assembled
        # ``(Q, q, A, b, G, h)`` lands in the dtype ``qpax.solve_qp``
        # expects.
        f = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
        self._S_x_diag_j = jnp.asarray(self._S_x_diag, dtype=f)
        self._S_u_diag_j = jnp.asarray(self._S_u_diag, dtype=f)
        self._inv_S_x_diag_j = jnp.asarray(self._inv_S_x_diag, dtype=f)
        self._inv_S_u_diag_j = jnp.asarray(self._inv_S_u_diag, dtype=f)
        self._c_x_j = jnp.asarray(self._c_x, dtype=f)
        self._c_u_j = jnp.asarray(self._c_u, dtype=f)
        self._jax_dtype = f

        self.layout = _QPLayout(N=N, n_x=n_x, n_u=n_u, n_nodal=len(jax_constraints.nodal))
        self._jax_constraints = jax_constraints
        self._x_unified = x_unified



    def initialize(self, lowered: "LoweredProblem", settings: "Config") -> None:
        """Validate the constraint subset QPAX supports and stash settings.

        Cross-node constraints raise :class:`NotImplementedError` with a
        pointer to :class:`CVXPyPTRSolver` and may gain QP-side support
        here in the future. User ``.convex()`` constraints are filtered
        upstream by :meth:`lower_convex_constraints`, which absorbs the
        auto-generated impulsive zero-pins and refuses anything else.
        """
        if self.layout is None:
            raise RuntimeError(
                "QPAXPTRSolver.initialize() called before create_variables(). "
                "Call create_variables() first."
            )

        if lowered.jax_constraints.cross_node:
            raise NotImplementedError(
                "QPAXPTRSolver does not yet support cross-node constraints "
                f"({len(lowered.jax_constraints.cross_node)} defined). "
                "Use CVXPyPTRSolver."
            )

        self._settings = settings
        self._n_constraints = self._count_rows(lowered.jax_constraints, settings)

    def _count_rows(
        self,
        jax_constraints: "LoweredJaxConstraints",
        settings: "Config",
    ) -> int:
        """Static row count for ``A z = b`` and ``G z ≤ h`` combined.

        Mirrors the row enumeration in :meth:`_assemble_qp`. Used only for the
        diagnostics box — the assembly code itself doesn't consult this.
        """
        L = self.layout
        N, n_x, n_u = L.N, L.n_x, L.n_u
        sim = settings.sim

        n_eq = 0
        n_eq += sum(len(c.nodes) for c in jax_constraints.nodal)
        for i in range(sim.true_state_slice.start, sim.true_state_slice.stop):
            if sim.x.initial_type[i] == "Fix":
                n_eq += 1
            if sim.x.final_type[i] == "Fix":
                n_eq += 1
        for nodes, ctrl_slice in self._impulsive_pins:
            n_eq += len(nodes) * (ctrl_slice.stop - ctrl_slice.start)
        if sim._uniform_time_grid:
            td = sim.time_dilation_slice
            n_eq += (N - 1) * (td.stop - td.start)
        n_eq += N * n_x  # state error definitions
        n_eq += N * n_u  # control error definitions
        n_eq += (N - 1) * n_x  # dynamics
        n_eq += sim.ctcs_slice.stop - sim.ctcs_slice.start  # CTCS x[0] = 0

        n_ineq = 2 * N * (n_x + n_u)  # box constraints
        for nodes in sim.ctcs_node_intervals:
            start_i = 1 if nodes[0] == 0 else nodes[0]
            n_ineq += 2 * (nodes[1] - start_i)
        n_ineq += 2 * (N - 1) * n_x  # |nu| L1 slack
        n_ineq += 2 * N * L.n_nodal  # pos(nu_vb) slack

        # User cone constraints (hard, no virtual buffer).
        for cc in self._user_cone_constraints:
            if isinstance(cc, ZeroConeConstraint):
                n_eq += len(cc.nodes) * cc.m
            elif isinstance(cc, NonnegConeConstraint):
                n_ineq += len(cc.nodes) * cc.m

        return n_eq + n_ineq

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
        x_prop_plus: np.ndarray | None = None,
        D_d: np.ndarray | None = None,
        E_d: np.ndarray | None = None,
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
        nodal: List[dict] = None,
        cross_node: List[dict] = None,
    ) -> None:
        if cross_node:
            # Defensive — initialize() already rejected, but the algorithm
            # may still pass an empty list.
            raise NotImplementedError(
                "QPAXPTRSolver received cross-node linearization data; "
                "cross-node constraints are not supported."
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
        del lam_vb_cross  # cross-node constraints rejected at initialize()
        self._pen = {
            "lam_prox": np.asarray(lam_prox, dtype=float),
            "lam_cost": np.asarray(lam_cost, dtype=float),
            "lam_vc": np.asarray(lam_vc, dtype=float),
            "lam_vb_nodal": np.asarray(lam_vb_nodal, dtype=float),
        }

    def update_boundary_conditions(
        self,
        x_init: np.ndarray = None,
        x_term: np.ndarray = None,
    ) -> None:
        if x_init is not None:
            self._x_init = np.asarray(x_init, dtype=float)
        if x_term is not None:
            self._x_term = np.asarray(x_term, dtype=float)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble_qp(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build ``(Q, q, A, b, G, h)`` from the stored linearization data.

        The decision vector layout, the per-row recipes, and the variable
        scaling all mirror :meth:`CVXPyPTRSolver.constraints` so that the two
        backends solve the same convex subproblem up to slack reformulation.
        """
        if not (self._dyn and self._cons and self._pen):
            raise RuntimeError(
                "QPAXPTRSolver.solve() requires update_dynamics_linearization, "
                "update_constraint_linearizations, and update_penalties to all "
                "have been called this iteration."
            )

        L = self.layout
        settings = self._settings
        N, n_x, n_u = L.N, L.n_x, L.n_u
        inv_S_x = self._inv_S_x_diag  # diagonals
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
        lam_vb_nodal = self._pen["lam_vb_nodal"]  # (N, max(C, 1))

        # ---------- cost ----------
        q_diag = np.full(L.n_z, _Q_DIAG_EPS, dtype=float)
        q_vec = np.zeros(L.n_z, dtype=float)

        # Trust-region: ½ zᵀ Q z with Q_diag = 2 · lam_prox on dx / du slots
        # reproduces lam_prox · (dx² + du²) from the CVXPy cost.
        for k in range(N):
            for j in range(n_x):
                q_diag[L.dx_idx(k, j)] = 2.0 * lam_prox[k, j] + _Q_DIAG_EPS
            for j in range(n_u):
                q_diag[L.du_idx(k, j)] = 2.0 * lam_prox[k, n_x + j] + _Q_DIAG_EPS

        # Boundary cost terms (Minimize / Maximize) — operate on scaled x for
        # the same numerical conditioning CVXPyPTRSolver gets.
        lam_cost_arr = np.broadcast_to(lam_cost, (settings.sim.n_states,))
        for i in range(settings.sim.true_state_slice.start, settings.sim.true_state_slice.stop):
            init_t = settings.sim.x.initial_type[i]
            final_t = settings.sim.x.final_type[i]
            if init_t == "Minimize":
                q_vec[L.x_idx(0, i)] += lam_cost_arr[i]
            elif init_t == "Maximize":
                q_vec[L.x_idx(0, i)] -= lam_cost_arr[i]
            if final_t == "Minimize":
                q_vec[L.x_idx(N - 1, i)] += lam_cost_arr[i]
            elif final_t == "Maximize":
                q_vec[L.x_idx(N - 1, i)] -= lam_cost_arr[i]

        # L1 penalty on |nu| → linear cost on s_abs
        for k in range(N - 1):
            for j in range(n_x):
                q_vec[L.s_abs_idx(k, j)] += lam_vc[k, j]

        # Positive-part penalty on nu_vb → linear cost on s_pos
        for c_idx in range(L.n_nodal):
            for k in range(N):
                q_vec[L.s_pos_idx(c_idx, k)] += lam_vb_nodal[k, c_idx]

        Q = np.diag(q_diag)

        # ---------- equality rows (A z = b) ----------
        A_rows: List[np.ndarray] = []
        b_rows: List[float] = []

        # Linearized nodal constraints: g + grad_x·dx + grad_u·du = nu_vb
        for c_idx, constraint in enumerate(self._jax_constraints.nodal):
            data = self._cons["nodal"][c_idx]
            g = data["g"]  # (N,)
            grad_x = data["grad_g_x"]  # (N, n_x)
            grad_u = data["grad_g_u"]  # (N, n_u)
            for node in constraint.nodes:
                row = np.zeros(L.n_z, dtype=float)
                for j in range(n_x):
                    row[L.dx_idx(node, j)] = grad_x[node, j]
                for j in range(n_u):
                    row[L.du_idx(node, j)] = grad_u[node, j]
                row[L.nu_vb_idx(c_idx, node)] = -1.0
                A_rows.append(row)
                b_rows.append(-g[node])

        # Boundary conditions (Fix). The initial branch couples the
        # post-impulse state at node 0 to the linearized impulse,
        # matching CVXPyPTRSolver.constraints at cvxpy_ptr_solver.py:484-495.
        E_d_arr = self._dyn["E_d"]
        x_prop_plus_arr = self._dyn["x_prop_plus"]
        for i in range(settings.sim.true_state_slice.start, settings.sim.true_state_slice.stop):
            if settings.sim.x.initial_type[i] == "Fix":
                row = np.zeros(L.n_z, dtype=float)
                row[L.x_idx(0, i)] = S_x[i]
                if has_impulsive:
                    # x_nonscaled[0, i] - E_d[0, i, slice_imp] @ du_nonscaled[0, slice_imp]
                    #     = x_prop_plus[0, i]
                    # With du_nonscaled[0, j] = S_u[j] * du[0, j].
                    for j in range(slice_imp.start, slice_imp.stop):
                        row[L.du_idx(0, j)] = -E_d_arr[0, i, j] * S_u[j]
                    rhs = x_prop_plus_arr[0, i] - c_x[i]
                else:
                    if self._x_init is None:
                        raise RuntimeError(
                            f"Fix initial condition on state {i} requires x_init; "
                            "call update_boundary_conditions() before solve()."
                        )
                    rhs = self._x_init[i] - c_x[i]
                A_rows.append(row)
                b_rows.append(rhs)
            if settings.sim.x.final_type[i] == "Fix":
                if self._x_term is None:
                    raise RuntimeError(
                        f"Fix final condition on state {i} requires x_term; "
                        "call update_boundary_conditions() before solve()."
                    )
                row = np.zeros(L.n_z, dtype=float)
                row[L.x_idx(N - 1, i)] = S_x[i]
                A_rows.append(row)
                b_rows.append(self._x_term[i] - c_x[i])

        # Impulsive zero-pin equalities. The auto-augmentation forces every
        # impulsive control DOF to zero at every non-impulse node, mirroring
        # the CVXPy lowering of ``u_nonscaled[node][slice_imp] == 0``. In
        # scaled coords that reduces to ``u[node, j] = -inv_S_u[j] · c_u[j]``.
        for nodes, ctrl_slice in self._impulsive_pins:
            for node in nodes:
                for j in range(ctrl_slice.start, ctrl_slice.stop):
                    row = np.zeros(L.n_z, dtype=float)
                    row[L.u_idx(node, j)] = 1.0
                    A_rows.append(row)
                    b_rows.append(-inv_S_u[j] * c_u[j])

        # Uniform time-grid: scaled u along the time-dilation slice is equal
        # at consecutive nodes. The CVXPy formulation premultiplies by
        # ``inv_S_u`` on both sides; that cancels, leaving u[k] = u[k-1].
        if settings.sim._uniform_time_grid:
            td = settings.sim.time_dilation_slice
            for k in range(1, N):
                for j in range(td.start, td.stop):
                    row = np.zeros(L.n_z, dtype=float)
                    row[L.u_idx(k, j)] = 1.0
                    row[L.u_idx(k - 1, j)] = -1.0
                    A_rows.append(row)
                    b_rows.append(0.0)

        # State / control error definitions: dx[k] = x[k] - inv_S_x (x_bar[k] - c_x).
        # Mirrors CVXPyPTRSolver.constraints — same scaling, same sign.
        x_bar = self._dyn["x_bar"]
        u_bar = self._dyn["u_bar"]
        for k in range(N):
            for j in range(n_x):
                row = np.zeros(L.n_z, dtype=float)
                row[L.x_idx(k, j)] = 1.0
                row[L.dx_idx(k, j)] = -1.0
                A_rows.append(row)
                b_rows.append(inv_S_x[j] * (x_bar[k, j] - c_x[j]))
            for j in range(n_u):
                row = np.zeros(L.n_z, dtype=float)
                row[L.u_idx(k, j)] = 1.0
                row[L.du_idx(k, j)] = -1.0
                A_rows.append(row)
                b_rows.append(inv_S_u[j] * (u_bar[k, j] - c_u[j]))

        # Dynamics (continuous PTR branch, FOH-style coupling, with optional
        # impulsive coupling at node k):
        #   x[k] - inv_S_x A_d S_x dx[k-1] - inv_S_x B_d S_u du[k-1]
        #        - inv_S_x C_d S_u du[k] - inv_S_x E_d S_u du[k][slice_imp]
        #        - nu[k-1] = inv_S_x (x_prop_plus[k] - c_x)   if has_impulsive
        #        - nu[k-1] = inv_S_x (x_prop[k-1]  - c_x)     otherwise
        # Mirrors CVXPyPTRSolver.constraints at cvxpy_ptr_solver.py:506-530.
        A_d = self._dyn["A_d"]  # (N-1, n_x, n_x)
        B_d = self._dyn["B_d"]  # (N-1, n_x, n_u)
        C_d = self._dyn["C_d"]  # (N-1, n_x, n_u)
        x_prop = self._dyn["x_prop"]  # (N-1, n_x) — propagated from k-1 → k
        for k in range(1, N):
            kp = k - 1  # previous-segment index
            # Pre-scaled blocks: inv_S_x[i, i] · A_d[kp, i, :] · S_x[:, j] · dx[kp, j]
            A_block = (inv_S_x[:, None] * A_d[kp]) * S_x[None, :]
            B_block = (inv_S_x[:, None] * B_d[kp]) * S_u[None, :]
            C_block = (inv_S_x[:, None] * C_d[kp]) * S_u[None, :]
            if has_impulsive:
                E_block = (inv_S_x[:, None] * E_d_arr[k]) * S_u[None, :]
                rhs = inv_S_x * (x_prop_plus_arr[k] - c_x)
            else:
                E_block = None
                rhs = inv_S_x * (x_prop[kp] - c_x)
            for i in range(n_x):
                row = np.zeros(L.n_z, dtype=float)
                row[L.x_idx(k, i)] = 1.0
                for j in range(n_x):
                    row[L.dx_idx(kp, j)] = -A_block[i, j]
                for j in range(n_u):
                    row[L.du_idx(kp, j)] = -B_block[i, j]
                    row[L.du_idx(k, j)] = -C_block[i, j]
                if has_impulsive:
                    for j in range(slice_imp.start, slice_imp.stop):
                        row[L.du_idx(k, j)] -= E_block[i, j]
                row[L.nu_idx(kp, i)] = -1.0
                A_rows.append(row)
                b_rows.append(rhs[i])

        # ---------- inequality rows (G z ≤ h) ----------
        G_rows: List[np.ndarray] = []
        h_rows: List[float] = []

        # Control / state box constraints — in scaled coords these reduce to
        # simple bounds on x[k, j] and u[k, j].
        u_max_scaled = inv_S_u * (np.asarray(settings.sim.u.max, dtype=float) - c_u)
        u_min_scaled = inv_S_u * (np.asarray(settings.sim.u.min, dtype=float) - c_u)
        x_max_scaled = inv_S_x * (np.asarray(settings.sim.x.max, dtype=float) - c_x)
        x_min_scaled = inv_S_x * (np.asarray(settings.sim.x.min, dtype=float) - c_x)
        for k in range(N):
            for j in range(n_u):
                row = np.zeros(L.n_z, dtype=float)
                row[L.u_idx(k, j)] = 1.0
                G_rows.append(row)
                h_rows.append(u_max_scaled[j])
                row = np.zeros(L.n_z, dtype=float)
                row[L.u_idx(k, j)] = -1.0
                G_rows.append(row)
                h_rows.append(-u_min_scaled[j])
            for j in range(n_x):
                row = np.zeros(L.n_z, dtype=float)
                row[L.x_idx(k, j)] = 1.0
                G_rows.append(row)
                h_rows.append(x_max_scaled[j])
                row = np.zeros(L.n_z, dtype=float)
                row[L.x_idx(k, j)] = -1.0
                G_rows.append(row)
                h_rows.append(-x_min_scaled[j])

        # CTCS LICQ-style rows: for each CTCS group (one augmented-state
        # index per group), enforce |x[i][idx] - x[i-1][idx]| ≤ x.max[idx]
        # over the group's node interval, plus x[0][idx] = 0 (in unscaled
        # coords). Mirrors CVXPyPTRSolver's CTCS block.
        for idx, nodes in zip(
            np.arange(settings.sim.ctcs_slice.start, settings.sim.ctcs_slice.stop),
            settings.sim.ctcs_node_intervals,
        ):
            start_idx = 1 if nodes[0] == 0 else nodes[0]
            x_max_unscaled = float(settings.sim.x.max[idx])
            # In scaled coords:  x_nonscaled[i][idx] - x_nonscaled[i-1][idx]
            #                  = S_x[idx] * (x[i][idx] - x[i-1][idx]).
            scale = S_x[idx]
            for i in range(start_idx, nodes[1]):
                row = np.zeros(L.n_z, dtype=float)
                row[L.x_idx(i, idx)] = scale
                row[L.x_idx(i - 1, idx)] = -scale
                G_rows.append(row)
                h_rows.append(x_max_unscaled)
                row = np.zeros(L.n_z, dtype=float)
                row[L.x_idx(i, idx)] = -scale
                row[L.x_idx(i - 1, idx)] = scale
                G_rows.append(row)
                h_rows.append(x_max_unscaled)
            # x_nonscaled[0][idx] = 0 -> S_x[idx] * x[0][idx] = -c_x[idx]
            row = np.zeros(L.n_z, dtype=float)
            row[L.x_idx(0, idx)] = scale
            A_rows.append(row)
            b_rows.append(-c_x[idx])

        A_mat = np.asarray(A_rows, dtype=float) if A_rows else np.zeros((0, L.n_z))
        b_vec = np.asarray(b_rows, dtype=float) if b_rows else np.zeros(0)

        # L1 reformulation of |nu|: nu - s_abs ≤ 0, -nu - s_abs ≤ 0.
        # Both together imply s_abs ≥ |nu| ≥ 0, so an explicit nonneg row
        # would be redundant.
        for k in range(N - 1):
            for j in range(n_x):
                row = np.zeros(L.n_z, dtype=float)
                row[L.nu_idx(k, j)] = 1.0
                row[L.s_abs_idx(k, j)] = -1.0
                G_rows.append(row)
                h_rows.append(0.0)
                row = np.zeros(L.n_z, dtype=float)
                row[L.nu_idx(k, j)] = -1.0
                row[L.s_abs_idx(k, j)] = -1.0
                G_rows.append(row)
                h_rows.append(0.0)

        # Positive-part reformulation: pos(nu_vb) needs s ≥ nu_vb AND s ≥ 0
        # — the second is *not* implied by the first (unlike L1).
        for c_idx in range(L.n_nodal):
            for k in range(N):
                row = np.zeros(L.n_z, dtype=float)
                row[L.nu_vb_idx(c_idx, k)] = 1.0
                row[L.s_pos_idx(c_idx, k)] = -1.0
                G_rows.append(row)
                h_rows.append(0.0)
                row = np.zeros(L.n_z, dtype=float)
                row[L.s_pos_idx(c_idx, k)] = -1.0
                G_rows.append(row)
                h_rows.append(0.0)

        # User cone constraints — hard (no virtual buffer), affine.
        # J_x/J_u are physical-space Jacobians; multiply by S_x/S_u to convert
        # to coefficients for scaled decision variables dx_scaled / du_scaled.
        x_bar_np = self._dyn["x_bar"]  # (N, n_x) unscaled
        u_bar_np = self._dyn["u_bar"]  # (N, n_u) unscaled
        params = self._parameters_dict

        for cc in self._user_cone_constraints:
            for k in cc.nodes:
                x_k = x_bar_np[k]
                u_k = u_bar_np[k]
                f0 = np.asarray(cc.jax_fn(x_k, u_k, k, params), dtype=float).reshape(cc.m)
                J_x = np.asarray(
                    jax.jacfwd(cc.jax_fn, argnums=0)(x_k, u_k, k, params), dtype=float
                ).reshape(cc.m, n_x) * S_x[None, :]
                J_u = np.asarray(
                    jax.jacfwd(cc.jax_fn, argnums=1)(x_k, u_k, k, params), dtype=float
                ).reshape(cc.m, n_u) * S_u[None, :]
                rhs = -f0
                if isinstance(cc, ZeroConeConstraint):
                    for i in range(cc.m):
                        row = np.zeros(L.n_z, dtype=float)
                        row[L.sl_dx.start + k * n_x : L.sl_dx.start + (k + 1) * n_x] = J_x[i]
                        row[L.sl_du.start + k * n_u : L.sl_du.start + (k + 1) * n_u] = J_u[i]
                        A_rows.append(row)
                        b_rows.append(float(rhs[i]))
                elif isinstance(cc, NonnegConeConstraint):
                    for i in range(cc.m):
                        row = np.zeros(L.n_z, dtype=float)
                        row[L.sl_dx.start + k * n_x : L.sl_dx.start + (k + 1) * n_x] = J_x[i]
                        row[L.sl_du.start + k * n_u : L.sl_du.start + (k + 1) * n_u] = J_u[i]
                        G_rows.append(row)
                        h_rows.append(float(rhs[i]))

        # Rebuild A_mat/b_vec now that user-cone equality rows may have been added.
        A_mat = np.asarray(A_rows, dtype=float) if A_rows else np.zeros((0, L.n_z))
        b_vec = np.asarray(b_rows, dtype=float) if b_rows else np.zeros(0)

        G_mat = np.asarray(G_rows, dtype=float) if G_rows else np.zeros((0, L.n_z))
        h_vec = np.asarray(h_rows, dtype=float) if h_rows else np.zeros(0)

        return Q, q_vec, A_mat, b_vec, G_mat, h_vec

    # ------------------------------------------------------------------
    # Solve / unpack
    # ------------------------------------------------------------------

    def solve(self) -> PTRSolveResult:
        Q, q, A, b, G, h = self._assemble_qp()

        Q_j = jnp.asarray(Q)
        q_j = jnp.asarray(q)
        A_j = jnp.asarray(A)
        b_j = jnp.asarray(b)
        G_j = jnp.asarray(G)
        h_j = jnp.asarray(h)

        z, _s, _z_dual, _y_dual, converged, iters = qpax.solve_qp(
            Q_j, q_j, A_j, b_j, G_j, h_j, **self.solver_args
        )

        z = np.asarray(z)
        self._last_iters = int(np.asarray(iters))
        self._last_converged = bool(np.asarray(converged))

        # Silent NaN unpacking poisons the next SCP linearization and the
        # outer loop has no status gate. Fail at the boundary instead.
        primal_finite = bool(np.all(np.isfinite(z)))
        if not self._last_converged or not primal_finite:
            dtype = str(Q_j.dtype)
            raise RuntimeError(
                f"qpax.solve_qp failed to converge (iters={self._last_iters}, "
                f"converged={self._last_converged}, primal finite={primal_finite}). "
                f"JAX dtype was {dtype}; if this is float32, pass "
                f"float_dtype='float64' to Problem(...). Otherwise tighten "
                f"solver_args (e.g. solver_tol=1e-8, max_iter=200) or switch "
                f"to CVXPyPTRSolver to triage."
            )

        return self._unpack(z)

    def iteration_callback(self):
        """JAX-pure ``(state, SubproblemData) -> SubproblemSolution``.

        Composes :meth:`_assemble_qp_jax` + ``qpax.solve_qp`` +
        :meth:`_build_solution_jax` into a single ``@jax.jit``-decorated closure
        built once at :meth:`initialize` time. The closure is reusable across
        SCP iterations — only the ``SubproblemData`` pytree changes between calls.

        Uses ``qpax.solve_qp``, whose ``converged`` flag is mapped onto the
        returned :class:`SubproblemSolution`'s ``status_code`` so the SCP loop
        can fail loudly on a diverged solve. (``qpax.solve_qp_primal`` is the
        reverse-mode-differentiable variant but exposes no convergence flag —
        qpax cannot provide both at once, and reliable non-convergence
        detection was chosen over differentiability here; see
        ``plans/iteration-fn-primitive.md``.)

        The returned callable takes ``(state, data)``: ``state`` is the
        :class:`AlgorithmState` pytree, accepted for cross-backend signature
        uniformity but unused (every input the QP needs already lives in
        ``data``); ``data`` is the :class:`SubproblemData` pytree carrying the
        per-iteration linearization arrays, penalty weights, and boundary
        conditions.
        """
        if self.layout is None or self._settings is None:
            raise RuntimeError(
                "QPAXPTRSolver.iteration_callback() requires initialize() to "
                "have been called first."
            )

        solver_args = self.solver_args

        def step(state, data: SubproblemData) -> SubproblemSolution:
            del state  # every input the QP needs already lives on ``data``.
            Q, q, A, b, G, h = self._assemble_qp_jax(data)
            z, _s, _z_dual, _y_dual, converged, _iters = qpax.solve_qp(
                Q, q, A, b, G, h, **solver_args
            )
            return self._build_solution_jax(z, data, converged)

        return jax.jit(step)

    def _unpack(self, z: np.ndarray) -> PTRSolveResult:
        """Reverse the layout into the structured :class:`PTRSolveResult`."""
        L = self.layout
        N, n_x, n_u = L.N, L.n_x, L.n_u

        # Scaled trajectories → physical units via the affine scaling.
        x_scaled = z[L.sl_x].reshape(N, n_x)
        u_scaled = z[L.sl_u].reshape(N, n_u)
        x = x_scaled * self._S_x_diag[None, :] + self._c_x[None, :]
        u = u_scaled * self._S_u_diag[None, :] + self._c_u[None, :]

        nu = z[L.sl_nu].reshape(N - 1, n_x)
        nu_vb = [np.asarray(z[sl]) for sl in L.sl_nu_vb]
        nu_vb_cross: List[float] = []

        cost = self._reconstruct_cost(z)

        # solve() raises on non-convergence before reaching _unpack.
        status = "optimal"

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
        """Recompute the PTR objective value at ``z``.

        Avoids holding onto the assembled ``Q`` / ``q`` (which can be
        sizeable for large ``N``); the cost-defining quantities live on
        ``self._pen`` already.
        """
        L = self.layout
        settings = self._settings
        N, n_x, n_u = L.N, L.n_x, L.n_u
        lam_prox = self._pen["lam_prox"]
        lam_cost_arr = np.broadcast_to(self._pen["lam_cost"], (settings.sim.n_states,))
        lam_vc = self._pen["lam_vc"]
        lam_vb_nodal = self._pen["lam_vb_nodal"]

        dx = z[L.sl_dx].reshape(N, n_x)
        du = z[L.sl_du].reshape(N, n_u)
        x = z[L.sl_x].reshape(N, n_x)
        s_abs = z[L.sl_s_abs].reshape(N - 1, n_x)

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
        cost += float(np.sum(lam_vc * s_abs))
        for c_idx in range(L.n_nodal):
            s_pos = z[L.sl_s_pos[c_idx]]
            cost += float(np.dot(lam_vb_nodal[:, c_idx], s_pos))
        return cost

    # ------------------------------------------------------------------
    # JAX-pure assembly / unpack (used by iteration_callback)
    # ------------------------------------------------------------------

    def _assemble_qp_jax(
        self,
        data: SubproblemData,
    ) -> Tuple[
        "jnp.ndarray", "jnp.ndarray", "jnp.ndarray", "jnp.ndarray", "jnp.ndarray", "jnp.ndarray"
    ]:
        """Build ``(Q, q, A, b, G, h)`` from a :class:`SubproblemData` pytree.

        JAX-pure counterpart of :meth:`_assemble_qp`. The decision-vector
        layout, the per-row recipes, and the variable scaling are identical
        to the NumPy path — only the array library differs. Structural
        metadata (layout slices, boundary-condition types, impulsive pin
        schedule, CTCS node intervals) is closed over Python-side; numerical
        inputs flow through ``data``.
        """
        L = self.layout
        sim = self._settings.sim
        N, n_x, n_u = L.N, L.n_x, L.n_u
        inv_S_x = self._inv_S_x_diag_j
        inv_S_u = self._inv_S_u_diag_j
        S_x = self._S_x_diag_j
        S_u = self._S_u_diag_j
        c_x = self._c_x_j
        c_u = self._c_u_j
        f = self._jax_dtype
        slice_imp = sim.u.slice_impulsive
        has_impulsive = slice_imp.stop > slice_imp.start
        eps = jnp.asarray(_Q_DIAG_EPS, dtype=f)

        lam_prox = data.lam_prox
        lam_cost = data.lam_cost
        lam_vc = data.lam_vc
        lam_vb_nodal = data.lam_vb_nodal

        # ---------- cost: diagonal Q + linear q ----------
        q_diag = jnp.full((L.n_z,), eps, dtype=f)
        # Trust-region quadratic: 2·lam_prox on dx / du slots (+ eps).
        q_diag = q_diag.at[L.sl_dx].set((2.0 * lam_prox[:, :n_x] + eps).reshape(-1))
        q_diag = q_diag.at[L.sl_du].set((2.0 * lam_prox[:, n_x:] + eps).reshape(-1))
        Q = jnp.diag(q_diag)

        # Boundary-cost signs computed Python-side from initial / final type.
        init_signs_np = np.zeros(sim.n_states, dtype=float)
        final_signs_np = np.zeros(sim.n_states, dtype=float)
        for i in range(sim.true_state_slice.start, sim.true_state_slice.stop):
            init_t = sim.x.initial_type[i]
            final_t = sim.x.final_type[i]
            if init_t == "Minimize":
                init_signs_np[i] = 1.0
            elif init_t == "Maximize":
                init_signs_np[i] = -1.0
            if final_t == "Minimize":
                final_signs_np[i] = 1.0
            elif final_t == "Maximize":
                final_signs_np[i] = -1.0
        init_signs = jnp.asarray(init_signs_np, dtype=f)
        final_signs = jnp.asarray(final_signs_np, dtype=f)
        lam_cost_arr = jnp.broadcast_to(jnp.asarray(lam_cost, dtype=f), (sim.n_states,))

        q_vec = jnp.zeros((L.n_z,), dtype=f)
        x0_start = L.sl_x.start
        xN_start = L.sl_x.start + (N - 1) * n_x
        q_vec = q_vec.at[x0_start : x0_start + n_x].add(init_signs * lam_cost_arr)
        q_vec = q_vec.at[xN_start : xN_start + n_x].add(final_signs * lam_cost_arr)
        # L1 penalty on |nu| → linear cost on s_abs
        q_vec = q_vec.at[L.sl_s_abs].add(lam_vc.reshape(-1))
        # Positive-part penalty on nu_vb → linear cost on s_pos (one block per constraint)
        for c_idx in range(L.n_nodal):
            q_vec = q_vec.at[L.sl_s_pos[c_idx]].add(lam_vb_nodal[:, c_idx])

        # ---------- equality rows (A z = b) ----------
        A_blocks: List[jnp.ndarray] = []
        b_blocks: List[jnp.ndarray] = []

        # Linearized nodal constraints. SubproblemData stacks nodal_g /
        # nodal_grad_x / nodal_grad_u across all constraints to (N, n_nodal*),
        # with zero-fill for nodes not in a given constraint's static
        # ``nodes`` tuple — we index those rows out here.
        for c_idx, constraint in enumerate(self._jax_constraints.nodal):
            nodes_tuple = tuple(constraint.nodes)
            if not nodes_tuple:
                continue
            n_nodes = len(nodes_tuple)
            nodes_arr = jnp.asarray(nodes_tuple)
            dx_cols = jnp.asarray(
                np.asarray([[L.dx_idx(node, j) for j in range(n_x)] for node in nodes_tuple])
            )  # (n_nodes, n_x)
            du_cols = jnp.asarray(
                np.asarray([[L.du_idx(node, j) for j in range(n_u)] for node in nodes_tuple])
            )  # (n_nodes, n_u)
            nuvb_cols = jnp.asarray(np.asarray([L.nu_vb_idx(c_idx, node) for node in nodes_tuple]))
            row_idx = jnp.arange(n_nodes)
            block = jnp.zeros((n_nodes, L.n_z), dtype=f)
            block = block.at[row_idx[:, None], dx_cols].set(data.nodal_grad_x[nodes_arr, c_idx])
            block = block.at[row_idx[:, None], du_cols].set(data.nodal_grad_u[nodes_arr, c_idx])
            block = block.at[row_idx, nuvb_cols].set(-1.0)
            A_blocks.append(block)
            b_blocks.append(-data.nodal_g[nodes_arr, c_idx])

        # Boundary Fix rows — emitted in the same interleaved per-state order
        # as the NumPy loop (init row, then final row, for each true-state
        # index). When impulsive is active, the initial-Fix row also couples
        # du at node 0 through the linearized impulse Jacobian E_d.
        imp_j_list = list(range(slice_imp.start, slice_imp.stop)) if has_impulsive else []
        for i in range(sim.true_state_slice.start, sim.true_state_slice.stop):
            init_t = sim.x.initial_type[i]
            final_t = sim.x.final_type[i]
            if init_t == "Fix":
                row = jnp.zeros((L.n_z,), dtype=f).at[L.x_idx(0, i)].set(S_x[i])
                if has_impulsive:
                    for j in imp_j_list:
                        row = row.at[L.du_idx(0, j)].set(-data.E_d[0, i, j] * S_u[j])
                    rhs = data.x_prop_plus[0, i] - c_x[i]
                else:
                    rhs = data.x_init[i] - c_x[i]
                A_blocks.append(row[None, :])
                b_blocks.append(rhs[None])
            if final_t == "Fix":
                row = jnp.zeros((L.n_z,), dtype=f).at[L.x_idx(N - 1, i)].set(S_x[i])
                A_blocks.append(row[None, :])
                b_blocks.append((data.x_term[i] - c_x[i])[None])

        # Impulsive zero-pins — every impulsive control DOF forced to zero
        # at each non-impulse node (in scaled coords).
        for nodes, ctrl_slice in self._impulsive_pins:
            for node in nodes:
                for j in range(ctrl_slice.start, ctrl_slice.stop):
                    row = jnp.zeros((L.n_z,), dtype=f)
                    row = row.at[L.u_idx(node, j)].set(1.0)
                    A_blocks.append(row[None, :])
                    b_blocks.append((-inv_S_u[j] * c_u[j])[None])

        # Uniform time-grid: scaled u along the time-dilation slice is
        # constant across nodes — encoded as u[k] - u[k-1] = 0.
        if sim._uniform_time_grid:
            td = sim.time_dilation_slice
            for k in range(1, N):
                for j in range(td.start, td.stop):
                    row = jnp.zeros((L.n_z,), dtype=f)
                    row = row.at[L.u_idx(k, j)].set(1.0)
                    row = row.at[L.u_idx(k - 1, j)].set(-1.0)
                    A_blocks.append(row[None, :])
                    b_blocks.append(jnp.zeros((1,), dtype=f))

        # State / control error definitions, interleaved per-node to match
        # the NumPy loop ordering (for each k: all x errors then all u errors).
        # Vectorize per k: build the (n_x + n_u, n_z) "node block" once with
        # static columns, replicate over k, then patch in the rhs values.
        x_cols_per_k = np.asarray([L.x_idx(0, j) for j in range(n_x)])  # offsets at k=0
        dx_cols_per_k = np.asarray([L.dx_idx(0, j) for j in range(n_x)])
        u_cols_per_k = np.asarray([L.u_idx(0, j) for j in range(n_u)])
        du_cols_per_k = np.asarray([L.du_idx(0, j) for j in range(n_u)])
        # Per-node column strides (state offsets advance by n_x per k, control
        # offsets by n_u).
        node_block_rows = n_x + n_u
        n_rows = N * node_block_rows
        block = jnp.zeros((n_rows, L.n_z), dtype=f)
        x_err_rhs = inv_S_x * (data.x_bar - c_x[None, :])  # (N, n_x)
        u_err_rhs = inv_S_u * (data.u_bar - c_u[None, :])  # (N, n_u)
        b_err = jnp.concatenate([x_err_rhs, u_err_rhs], axis=1).reshape(-1)  # (N*(n_x+n_u),)
        for k in range(N):
            row_base = k * node_block_rows
            x_rows = row_base + jnp.arange(n_x)
            u_rows = row_base + n_x + jnp.arange(n_u)
            block = block.at[x_rows, jnp.asarray(x_cols_per_k + k * n_x)].set(1.0)
            block = block.at[x_rows, jnp.asarray(dx_cols_per_k + k * n_x)].set(-1.0)
            block = block.at[u_rows, jnp.asarray(u_cols_per_k + k * n_u)].set(1.0)
            block = block.at[u_rows, jnp.asarray(du_cols_per_k + k * n_u)].set(-1.0)
        A_blocks.append(block)
        b_blocks.append(b_err)

        # Dynamics (continuous FOH branch, plus optional impulsive coupling).
        # SubproblemData carries the raw A_d / B_d / C_d / D_d / E_d; the
        # NumPy path's D-absorption (cvxpy_ptr_solver.py:636-654 recipe) is
        # done inline here when impulsive.
        if has_impulsive:
            D_steps = data.D_d[1:]
            A_eff = jnp.einsum("kij,kjl->kil", D_steps, data.A_d)
            B_eff = jnp.einsum("kij,kjl->kil", D_steps, data.B_d)
            C_eff = jnp.einsum("kij,kjl->kil", D_steps, data.C_d)
        else:
            A_eff = data.A_d
            B_eff = data.B_d
            C_eff = data.C_d
        # Pre-scaled blocks: shape (N-1, n_x, n_x | n_u).
        A_blk = (inv_S_x[None, :, None] * A_eff) * S_x[None, None, :]
        B_blk = (inv_S_x[None, :, None] * B_eff) * S_u[None, None, :]
        C_blk = (inv_S_x[None, :, None] * C_eff) * S_u[None, None, :]

        # dyn_block: (N-1, n_x, n_z). For each (k_prime, i):
        #   row[x_idx(k_prime+1, i)] = 1
        #   row[dx_idx(k_prime, j)]  = -A_blk[k_prime, i, j]
        #   row[du_idx(k_prime, j)]  = -B_blk[k_prime, i, j]
        #   row[du_idx(k_prime+1, j)] = -C_blk[k_prime, i, j] (- E_blk[i, j] for j in slice_imp)
        #   row[nu_idx(k_prime, i)]  = -1
        n_dyn = (N - 1) * n_x
        kp_idx = np.repeat(np.arange(N - 1), n_x)
        i_idx = np.tile(np.arange(n_x), N - 1)
        # Flat row indices (kp * n_x + i) ↔ index into the (N-1, n_x) row grid.
        flat_row = jnp.arange(n_dyn)
        # Column indices for the four sets of writes, computed Python-side.
        x_term_cols = jnp.asarray(np.asarray([L.x_idx(kp + 1, i) for kp, i in zip(kp_idx, i_idx)]))
        nu_cols = jnp.asarray(np.asarray([L.nu_idx(kp, i) for kp, i in zip(kp_idx, i_idx)]))
        dx_cols_2d = jnp.asarray(
            np.asarray([[L.dx_idx(kp, j) for j in range(n_x)] for kp in kp_idx])
        )  # (n_dyn, n_x)
        du_prev_cols_2d = jnp.asarray(
            np.asarray([[L.du_idx(kp, j) for j in range(n_u)] for kp in kp_idx])
        )  # (n_dyn, n_u)
        du_curr_cols_2d = jnp.asarray(
            np.asarray([[L.du_idx(kp + 1, j) for j in range(n_u)] for kp in kp_idx])
        )  # (n_dyn, n_u)

        block = jnp.zeros((n_dyn, L.n_z), dtype=f)
        block = block.at[flat_row, x_term_cols].set(1.0)
        block = block.at[flat_row, nu_cols].set(-1.0)
        # A: -A_blk[kp_idx, i_idx, :] over the (kp_idx, i_idx, j) grid
        A_vals = -A_blk.reshape(n_dyn, n_x)  # rows already in (kp, i) → (kp*n_x + i) order
        B_vals = -B_blk.reshape(n_dyn, n_u)
        C_vals = -C_blk.reshape(n_dyn, n_u)
        block = block.at[flat_row[:, None], dx_cols_2d].set(A_vals)
        block = block.at[flat_row[:, None], du_prev_cols_2d].set(B_vals)
        block = block.at[flat_row[:, None], du_curr_cols_2d].set(C_vals)
        if has_impulsive:
            # E_blk: (N-1, n_x, n_u). E_d_arr[k] for k in 1..N-1 contributes.
            E_blk = (inv_S_x[None, :, None] * data.E_d[1:]) * S_u[None, None, :]
            imp_j = np.arange(slice_imp.start, slice_imp.stop)
            imp_du_cols_2d = jnp.asarray(
                np.asarray([[L.du_idx(kp + 1, j) for j in imp_j] for kp in kp_idx])
            )  # (n_dyn, n_imp)
            E_vals = -E_blk[:, :, imp_j].reshape(n_dyn, len(imp_j))
            block = block.at[flat_row[:, None], imp_du_cols_2d].add(E_vals)
        # RHS
        if has_impulsive:
            rhs = inv_S_x[None, :] * (data.x_prop_plus[1:] - c_x[None, :])
        else:
            rhs = inv_S_x[None, :] * (data.x_prop - c_x[None, :])
        A_blocks.append(block)
        b_blocks.append(rhs.reshape(-1))

        # CTCS x[0] = 0 (in unscaled coords) → S_x[idx] · x[0, idx] = -c_x[idx]
        ctcs_start = sim.ctcs_slice.start
        ctcs_stop = sim.ctcs_slice.stop
        if ctcs_stop > ctcs_start:
            ctcs_idx_list = list(range(ctcs_start, ctcs_stop))
            ctcs_idx_arr = jnp.asarray(ctcs_idx_list)
            ctcs_cols = jnp.asarray([L.x_idx(0, idx) for idx in ctcs_idx_list])
            row_idx = jnp.arange(len(ctcs_idx_list))
            block = jnp.zeros((len(ctcs_idx_list), L.n_z), dtype=f)
            block = block.at[row_idx, ctcs_cols].set(S_x[ctcs_idx_arr])
            A_blocks.append(block)
            b_blocks.append(-c_x[ctcs_idx_arr])

        # ---------- inequality rows (G z ≤ h) ----------
        G_blocks: List[jnp.ndarray] = []
        h_blocks: List[jnp.ndarray] = []

        # State / control box constraints in scaled coords. Ordering matches
        # the NumPy loop: per node, all u DOFs (max then -min row), then all
        # x DOFs (max then -min row).
        box_rows: List[jnp.ndarray] = []
        box_h: List[float] = []
        for k in range(N):
            for j in range(n_u):
                box_rows.append(jnp.zeros((L.n_z,), dtype=f).at[L.u_idx(k, j)].set(1.0))
                box_h.append(float(self._inv_S_u_diag[j] * (sim.u.max[j] - self._c_u[j])))
                box_rows.append(jnp.zeros((L.n_z,), dtype=f).at[L.u_idx(k, j)].set(-1.0))
                box_h.append(-float(self._inv_S_u_diag[j] * (sim.u.min[j] - self._c_u[j])))
            for j in range(n_x):
                box_rows.append(jnp.zeros((L.n_z,), dtype=f).at[L.x_idx(k, j)].set(1.0))
                box_h.append(float(self._inv_S_x_diag[j] * (sim.x.max[j] - self._c_x[j])))
                box_rows.append(jnp.zeros((L.n_z,), dtype=f).at[L.x_idx(k, j)].set(-1.0))
                box_h.append(-float(self._inv_S_x_diag[j] * (sim.x.min[j] - self._c_x[j])))
        G_blocks.append(jnp.stack(box_rows))
        h_blocks.append(jnp.asarray(box_h, dtype=f))

        # CTCS LICQ-style |x[i][idx] - x[i-1][idx]| ≤ x.max[idx] over each
        # group's node interval, in scaled coords.
        for idx, nodes in zip(
            range(sim.ctcs_slice.start, sim.ctcs_slice.stop),
            sim.ctcs_node_intervals,
        ):
            start_i = 1 if nodes[0] == 0 else nodes[0]
            scale = float(self._S_x_diag[idx])
            x_max_unscaled = float(sim.x.max[idx])
            for i in range(start_i, nodes[1]):
                row_pos = (
                    jnp.zeros((L.n_z,), dtype=f)
                    .at[L.x_idx(i, idx)]
                    .set(scale)
                    .at[L.x_idx(i - 1, idx)]
                    .set(-scale)
                )
                row_neg = (
                    jnp.zeros((L.n_z,), dtype=f)
                    .at[L.x_idx(i, idx)]
                    .set(-scale)
                    .at[L.x_idx(i - 1, idx)]
                    .set(scale)
                )
                G_blocks.append(jnp.stack([row_pos, row_neg]))
                h_blocks.append(jnp.asarray([x_max_unscaled, x_max_unscaled], dtype=f))

        # L1 reformulation of |nu|: nu - s_abs ≤ 0, -nu - s_abs ≤ 0.
        n_nu_rows = 2 * (N - 1) * n_x
        block = jnp.zeros((n_nu_rows, L.n_z), dtype=f)
        # Build columns and values vectorized:
        # For each (k, j) ∈ [0..N-2] × [0..n_x-1], two rows.
        pair_idx = np.arange((N - 1) * n_x)
        nu_cols_flat = np.asarray([L.nu_idx(k, j) for k in range(N - 1) for j in range(n_x)])
        sabs_cols_flat = np.asarray([L.s_abs_idx(k, j) for k in range(N - 1) for j in range(n_x)])
        row_pos = jnp.asarray(2 * pair_idx)
        row_neg = jnp.asarray(2 * pair_idx + 1)
        block = block.at[row_pos, jnp.asarray(nu_cols_flat)].set(1.0)
        block = block.at[row_pos, jnp.asarray(sabs_cols_flat)].set(-1.0)
        block = block.at[row_neg, jnp.asarray(nu_cols_flat)].set(-1.0)
        block = block.at[row_neg, jnp.asarray(sabs_cols_flat)].set(-1.0)
        G_blocks.append(block)
        h_blocks.append(jnp.zeros((n_nu_rows,), dtype=f))

        # Positive-part reformulation: pos(nu_vb) needs s ≥ nu_vb AND s ≥ 0.
        for c_idx in range(L.n_nodal):
            nuvb_cols = jnp.asarray([L.nu_vb_idx(c_idx, k) for k in range(N)])
            spos_cols = jnp.asarray([L.s_pos_idx(c_idx, k) for k in range(N)])
            row_idx = jnp.arange(N)
            block = jnp.zeros((2 * N, L.n_z), dtype=f)
            # Row 2k: nu_vb - s_pos ≤ 0
            block = block.at[2 * row_idx, nuvb_cols].set(1.0)
            block = block.at[2 * row_idx, spos_cols].set(-1.0)
            # Row 2k+1: -s_pos ≤ 0
            block = block.at[2 * row_idx + 1, spos_cols].set(-1.0)
            G_blocks.append(block)
            h_blocks.append(jnp.zeros((2 * N,), dtype=f))

        # User cone constraints (hard, affine).
        # J_x/J_u are physical-space Jacobians; multiply by S_x/S_u to convert
        # to coefficients for scaled decision variables dx_scaled / du_scaled.
        params = data.params
        for cc in self._user_cone_constraints:
            for k in cc.nodes:
                x_k = data.x_bar[k]
                u_k = data.u_bar[k]
                f0 = jnp.reshape(cc.jax_fn(x_k, u_k, k, params), (cc.m,))
                J_x = jnp.reshape(
                    jax.jacfwd(cc.jax_fn, argnums=0)(x_k, u_k, k, params), (cc.m, n_x)
                ) * S_x[None, :]
                J_u = jnp.reshape(
                    jax.jacfwd(cc.jax_fn, argnums=1)(x_k, u_k, k, params), (cc.m, n_u)
                ) * S_u[None, :]
                # Build dense row block: (cc.m, n_z).
                block = jnp.zeros((cc.m, L.n_z), dtype=f)
                dx_start = L.sl_dx.start + k * n_x
                du_start = L.sl_du.start + k * n_u
                dx_idx_range = jnp.arange(dx_start, dx_start + n_x)
                du_idx_range = jnp.arange(du_start, du_start + n_u)
                block = block.at[:, dx_idx_range].set(J_x)
                block = block.at[:, du_idx_range].set(J_u)
                rhs = -f0
                if isinstance(cc, ZeroConeConstraint):
                    A_blocks.append(block)
                    b_blocks.append(rhs)
                elif isinstance(cc, NonnegConeConstraint):
                    G_blocks.append(block)
                    h_blocks.append(rhs)

        A_mat = jnp.concatenate(A_blocks, axis=0) if A_blocks else jnp.zeros((0, L.n_z), dtype=f)
        b_vec = jnp.concatenate(b_blocks, axis=0) if b_blocks else jnp.zeros((0,), dtype=f)

        G_mat = jnp.concatenate(G_blocks, axis=0) if G_blocks else jnp.zeros((0, L.n_z), dtype=f)
        h_vec = jnp.concatenate(h_blocks, axis=0) if h_blocks else jnp.zeros((0,), dtype=f)

        return Q, q_vec, A_mat, b_vec, G_mat, h_vec

    def _build_solution_jax(
        self,
        z: "jnp.ndarray",
        data: SubproblemData,
        converged: "jnp.ndarray",
    ) -> SubproblemSolution:
        """Unpack a primal ``z`` into a :class:`SubproblemSolution` pytree.

        Mirrors :meth:`_unpack` but stays JAX-pure: returns scaled-to-physical
        trajectories, the collapsed ``(N, n_nodal)`` ``nu_vb`` layout, and a
        ``status_code`` of ``OPTIMAL`` when ``qpax.solve_qp`` reported
        convergence and the primal is finite, else ``UNKNOWN`` (which the SCP
        loop turns into a hard failure).
        """
        L = self.layout
        N, n_x, n_u = L.N, L.n_x, L.n_u
        f = self._jax_dtype

        x_scaled = z[L.sl_x].reshape(N, n_x)
        u_scaled = z[L.sl_u].reshape(N, n_u)
        x = x_scaled * self._S_x_diag_j[None, :] + self._c_x_j[None, :]
        u = u_scaled * self._S_u_diag_j[None, :] + self._c_u_j[None, :]
        nu = z[L.sl_nu].reshape(N - 1, n_x)
        if L.n_nodal:
            nu_vb = jnp.stack([z[sl] for sl in L.sl_nu_vb], axis=-1)
        else:
            nu_vb = jnp.zeros((N, 0), dtype=f)

        cost = self._reconstruct_cost_jax(z, data)

        ok = jnp.logical_and(jnp.asarray(converged, dtype=bool), jnp.all(jnp.isfinite(z)))
        status_code = jnp.where(ok, jnp.int32(StatusCode.OPTIMAL), jnp.int32(StatusCode.UNKNOWN))

        return SubproblemSolution(
            x=x,
            u=u,
            nu=nu,
            nu_vb=nu_vb,
            nu_vb_cross=jnp.zeros((0,), dtype=f),
            cost=cost,
            status_code=status_code,
        )

    def _reconstruct_cost_jax(
        self,
        z: "jnp.ndarray",
        data: SubproblemData,
    ) -> "jnp.ndarray":
        """JAX-pure mirror of :meth:`_reconstruct_cost`."""
        L = self.layout
        sim = self._settings.sim
        N, n_x, n_u = L.N, L.n_x, L.n_u
        f = self._jax_dtype
        lam_prox = data.lam_prox
        lam_cost_arr = jnp.broadcast_to(jnp.asarray(data.lam_cost, dtype=f), (sim.n_states,))
        lam_vc = data.lam_vc
        lam_vb_nodal = data.lam_vb_nodal

        dx = z[L.sl_dx].reshape(N, n_x)
        du = z[L.sl_du].reshape(N, n_u)
        x_scaled = z[L.sl_x].reshape(N, n_x)
        s_abs = z[L.sl_s_abs].reshape(N - 1, n_x)

        cost = jnp.sum(lam_prox[:, :n_x] * dx**2) + jnp.sum(lam_prox[:, n_x:] * du**2)
        # Boundary cost: same signed sum as _reconstruct_cost.
        for i in range(sim.true_state_slice.start, sim.true_state_slice.stop):
            init_t = sim.x.initial_type[i]
            final_t = sim.x.final_type[i]
            if init_t == "Minimize":
                cost = cost + lam_cost_arr[i] * x_scaled[0, i]
            elif init_t == "Maximize":
                cost = cost - lam_cost_arr[i] * x_scaled[0, i]
            if final_t == "Minimize":
                cost = cost + lam_cost_arr[i] * x_scaled[-1, i]
            elif final_t == "Maximize":
                cost = cost - lam_cost_arr[i] * x_scaled[-1, i]
        cost = cost + jnp.sum(lam_vc * s_abs)
        for c_idx in range(L.n_nodal):
            s_pos = z[L.sl_s_pos[c_idx]]
            cost = cost + jnp.dot(lam_vb_nodal[:, c_idx], s_pos)
        return cost.astype(f)

    # ------------------------------------------------------------------
    # Misc API
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """QP dimensions for the diagnostics box.

        ``n_parameters`` is reported as zero — QPAX consumes raw arrays
        rebuilt every solve, so there's no analogue to CVXPy's parameter
        cache. ``n_constraints`` is the total number of ``A z = b`` plus
        ``G z ≤ h`` rows, cached in :meth:`initialize`.
        """
        if self.layout is None:
            return {"n_variables": 0, "n_parameters": 0, "n_constraints": 0}
        return {
            "n_variables": self.layout.n_z,
            "n_parameters": 0,
            "n_constraints": self._n_constraints,
        }

    def citation(self) -> List[str]:
        """BibTeX entry for QPAX (Tracy & Howell, 2024)."""
        return [
            r"""@article{tracy2024qpax,
  title={QPAX: differentiable QP solver in JAX},
  author={Tracy, Kevin and Howell, Taylor},
  journal={arXiv preprint arXiv:2406.11749},
  year={2024}
}"""
        ]
