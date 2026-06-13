"""Abstract base class for successive convexification algorithms.

:class:`Algorithm` is the contract every SCP algorithm implements: it builds the
JAX-pure iteration body, stores it, drives it one step at a time, and owns its
convergence policy. The iterate carry it threads lives in sibling modules —
:class:`~openscvx.algorithms.state.AlgorithmState` (the JAX-traceable current
iterate) and :class:`~openscvx.algorithms.history.AlgorithmHistory` (the
CPU-side append-only log) — matching the package's ``base.py``-is-the-ABC idiom
(``solvers/base.py``, ``discretization/base.py``, ``integrations/base.py``).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import jax.numpy as jnp

from openscvx.utils.printing import Column

if TYPE_CHECKING:
    import hashlib

    from openscvx.config import Config
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints

    from .autotuner.base import AutotuningBase
    from .history import AlgorithmHistory
    from .state import AlgorithmState
    from .weights import Weights


class Algorithm(ABC):
    """Abstract base class for successive convexification algorithms.

    An ``Algorithm`` owns the SCP iteration end to end: it *builds* the
    JAX-pure iteration body (:meth:`build_iteration`), *stores* it
    (:meth:`initialize`), and *drives* it one step at a time (:meth:`step`).
    ``step`` takes the current :class:`AlgorithmState` and a CPU-side
    :class:`AlgorithmHistory`, returns the next state plus a convergence flag,
    and appends to ``history`` for diagnostics.

    !!! tip "Statefullness"
        Avoid storing mutable iteration state on ``self``. All iteration
        state lives on :class:`AlgorithmState` (JAX-traceable) and
        :class:`AlgorithmHistory` (Python-side), threaded explicitly through
        ``step()``.

    The surface :class:`~openscvx.problem.Problem` relies on, beyond the
    abstract methods below:

    - ``autotuner`` — the :class:`AutotuningBase` whose declared ``hyper``
      fields :class:`~openscvx.problem.Problem` enumerates to build the
      per-solve / batched override channel.
    - ``ep_tr`` / ``ep_vb`` / ``ep_vc`` — convergence thresholds
      :meth:`Problem._sync_scp_constants` snapshots onto
      :class:`AlgorithmState` before each solve.
    - :meth:`get_columns` — the iteration-table columns to print.
    - ``weights.build_vb_arrays`` — called once in
      :meth:`Problem.initialize` to size the virtual-buffer weight arrays.
    - :meth:`_hash_into` — contributes the algorithm's identity to the
      ``solve_batched`` export cache key.

    ``t_max`` is honored only on the Python ``solve()`` path; the JAX
    ``lax.while_loop`` behind ``solve_jax`` / ``solve_batched`` terminates on
    ``k_max`` and the convergence predicate alone (no wall-clock probe inside a
    trace).
    """

    def __init__(
        self,
        weights: "Weights",
        autotuner: "AutotuningBase",
        k_max: int,
        t_max: Optional[float],
        ep_tr: float,
        ep_vb: float,
        ep_vc: float,
    ):
        """Record the SCP weights, autotuner, and convergence parameters.

        Subclasses build the user-facing defaults (PTR owns them) and end their
        own ``__init__`` with ``super().__init__(...)``; every parameter here is
        required so the ABC is not a fourth place those defaults are declared.

        Args:
            weights: SCP weights used by the algorithm and autotuner.
            autotuner: The penalty-weight update rule (:class:`AutotuningBase`).
            k_max: Maximum number of SCP iterations.
            t_max: Optional wall-clock time limit in seconds (``solve()`` only;
                ``None`` means no limit).
            ep_tr: Trust-region convergence threshold.
            ep_vb: Virtual-buffer convergence threshold.
            ep_vc: Virtual-control convergence threshold.
        """
        self.weights = weights
        self.autotuner = autotuner
        self.k_max = k_max
        self.t_max = t_max
        self.ep_tr = ep_tr
        self.ep_vb = ep_vb
        self.ep_vc = ep_vc

    @abstractmethod
    def build_iteration(
        self,
        dis_continuous: Callable,
        dis_impulsive: Callable,
        jax_constraints: "LoweredJaxConstraints",
        solver_callback: Callable,
        settings: "Config",
    ) -> Callable:
        """Build the JAX-pure SCP iteration body for this algorithm.

        :class:`~openscvx.problem.Problem` assembles the discretization
        solvers, lowered constraints, and convex-solver callback, then asks
        the algorithm to fuse them into one step. The algorithm owns this
        because the fusion is algorithm-specific (which autotuner runs, which
        penalty terms are assembled); ``Problem`` stays algorithm-agnostic.

        Args:
            dis_continuous: Continuous-dynamics discretization solver,
                ``(x, u, params) -> (A_d, B_d, C_d, x_prop, V)``.
            dis_impulsive: Impulsive/discrete-dynamics discretization solver,
                ``(x_nodes, u, params) -> (x_prop_plus, D_d, E_d, W)``.
            jax_constraints: Lowered JAX constraints the body operates over.
            solver_callback: The convex backend's ``iteration_callback``,
                ``(state, SubproblemData) -> SubproblemSolution`` (see
                :class:`~openscvx.solvers.ptr_solver.PTRSolver`).
            settings: Problem configuration.

        Returns:
            The JAX-pure ``(state, params) -> (next_state, diagnostics)``
            iteration body. It must advance ``state.k`` by one per call —
            both ``Problem.solve()``'s Python loop and the ``lax.while_loop``
            behind ``solve_jax`` terminate on ``k`` reaching ``k_max``.
            :class:`~openscvx.problem.Problem` wraps the body in ``jax.jit``
            and hands it back via :meth:`initialize`.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self, iteration_fn: Callable, emitter: Callable) -> None:
        """Store the fused SCP iteration body and per-iteration infrastructure.

        Args:
            iteration_fn: The JAX-pure ``(state, params) -> (next_state,
                diagnostics)`` body returned by :meth:`build_iteration`,
                already wrapped in ``jax.jit`` by
                :class:`~openscvx.problem.Problem`.
            emitter: Per-iteration diagnostics sink (printing queue / no-op).
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        state: "AlgorithmState",
        history: "AlgorithmHistory",
        params: dict,
        settings: "Config",
    ) -> Tuple["AlgorithmState", bool]:
        """Execute one iteration.

        Args:
            state: Current-iterate pytree.
            history: CPU-side append-only iteration log.
            params: Problem parameters (may change between steps).
            settings: Configuration object (may change between steps).

        Returns:
            ``(next_state, converged)`` — the post-iteration state pytree
            and a flag indicating whether the SCP convergence criteria are
            satisfied.
        """
        raise NotImplementedError

    @abstractmethod
    def get_columns(self, verbosity: int) -> List[Column]:
        """Return the iteration-table columns to print at ``verbosity``.

        :class:`~openscvx.problem.Problem` calls this once per solve to lay
        out the progress table. Implementations typically concatenate the
        algorithm's own columns with the autotuner's ``COLUMNS`` and filter by
        each column's ``min_verbosity``.
        """
        raise NotImplementedError

    def converged(self, state: "AlgorithmState") -> jnp.ndarray:
        """Boolean SCP convergence test from the metrics and thresholds on ``state``.

        The default — every metric below its threshold — is algorithm-agnostic:
        :class:`AlgorithmState` carries ``J_tr`` / ``J_vb`` / ``J_vc`` and the
        ``ep_*`` tolerances generically, so it serves any SCP algorithm. Override
        to change the convergence policy; the override is honored on all three
        solve paths (``solve`` / ``solve_jax`` / ``solve_batched``), since
        :meth:`step` and the ``lax.while_loop`` harness both route through it.

        Must be JAX-traceable: it runs inside the ``lax.while_loop`` cond of
        ``solve_jax`` / ``solve_batched`` (where it is ``jax.vmap``'d per batch
        element) as well as on the Python ``solve()`` path.

        Under ``save_compiled``, the predicate is baked into the exported
        ``solve_batched`` artifact, but the cache key sees only the algorithm's
        class *name* (:meth:`_hash_into`): introducing an override — a new
        subclass — is a clean cache miss, while *editing the body* of an
        existing override is invisible to the key and silently loads the stale
        artifact. Clear the solver cache (or rename the class) when iterating
        on a predicate with ``save_compiled`` enabled.
        """
        return (state.J_tr < state.ep_tr) & (state.J_vb < state.ep_vb) & (state.J_vc < state.ep_vc)

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Contribute this algorithm's identity to the ``solve_batched`` cache key.

        The exported batched loop bakes in the initial penalty weights and the
        autotuner's update rule — none of which the symbolic problem hash
        covers. The default folds in the concrete class name, then every
        :class:`~openscvx.algorithms.weights.Weights` field (derived from the
        dataclass, so a new field is hashed automatically), then the autotuner
        via its own ``_hash_into`` — mirroring the symbolic ``_hash_into``
        protocol. The :class:`~openscvx.algorithms.state.AlgorithmState` field
        schema is folded in too: the state pytree is the artifact's input, its
        leaves bind positionally to the exported avals, and many are
        identically-shaped scalars — so a field reorder must be a clean cache
        miss, never a silent permutation (the same guard the autotuner applies
        to its ``hyper`` schema).

        The convergence thresholds (``ep_tr`` / ``ep_vb`` / ``ep_vc``) and the
        iteration cap ``k_max`` / time limit ``t_max`` are deliberately *not*
        folded in here: they ride the :class:`AlgorithmState` pytree as runtime
        inputs, so one exported artifact serves every tolerance and
        ``max_iters`` setting. Override only if an algorithm's identity needs
        something beyond ``weights`` + ``autotuner``.
        """
        from dataclasses import fields as dc_fields

        from openscvx.utils.caching import hash_value_into

        from .state import AlgorithmState

        hasher.update(type(self).__name__.encode())
        for fld in dc_fields(self.weights):
            hash_value_into(hasher, getattr(self.weights, fld.name))
        # Schema, not values: the AlgorithmState field order fixes the exported
        # artifact's input-leaf layout, so a reorder must invalidate cached
        # artifacts rather than silently permute runtime values onto the wrong
        # avals (mirrors the hyper-schema fold in AutotuningBase._hash_into).
        for name in AlgorithmState._FIELDS:
            hasher.update(name.encode())
        self.autotuner._hash_into(hasher)

    @abstractmethod
    def citation(self) -> List[str]:
        """Return BibTeX citations for this algorithm."""
        raise NotImplementedError
