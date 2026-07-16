"""JAX-traceable iterate state for SCP algorithms.

This module holds the pytrees that move *through* the SCP iteration:

* :class:`AlgorithmState` — the frozen, JAX-registered current iterate
  (state/control/weights/diagnostics). Every autotuner's
  :py:meth:`~openscvx.algorithms.autotuner.base.AutotuningBase.update_weights`
  is a pure functional update on this pytree, so the SCP body composes with
  ``jax.jit``, ``jax.vmap``, and ``jax.grad``.
* :class:`CandidateIterate` — the just-solved subproblem result the fused
  iteration body hands to the autotuner.
* :class:`AdaptiveStateCode` — the int32 outcome the autotuner records on the
  state, plus :func:`adaptive_state_code_to_str` for the printing path.

It sits in the middle of the algorithms import order: it depends on
:mod:`openscvx.algorithms.hyperparams` and is depended on by
:mod:`openscvx.algorithms.history`.
"""

from dataclasses import dataclass
from dataclasses import fields as dc_fields
from dataclasses import replace as dc_replace
from enum import IntEnum
from typing import TYPE_CHECKING, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import export

from .hyperparams import HyperParams

if TYPE_CHECKING:
    from openscvx.config import Config

    from .weights import Weights


@dataclass(frozen=True)
class CandidateIterate:
    """Subproblem candidate the fused iteration hands to the autotuner.

    Built once, in-trace, by the SCP iteration body
    (``scvx/iteration.py``): it carries the just-solved subproblem trajectory
    (``x`` / ``u``), its propagation (``x_prop`` / ``x_prop_plus``), and the
    linearized cost (``J_lin``) into :py:meth:`AutotuningBase.update_weights`,
    which copies the accepted fields onto the next :class:`AlgorithmState`. It
    is the autotuner's input and nothing else — host-side history recording
    takes its arrays through :py:meth:`AlgorithmHistory.record_iteration`'s
    explicit keyword arguments rather than through this type.
    """

    x: jnp.ndarray
    u: jnp.ndarray
    x_prop: jnp.ndarray
    x_prop_plus: jnp.ndarray
    J_lin: jnp.ndarray


# ---------------------------------------------------------------------------
# Adaptive-state enum
# ---------------------------------------------------------------------------


class AdaptiveStateCode(IntEnum):
    """Outcome of one autotuner ``update_weights`` call.

    The autotuner stores its decision as an ``int32`` on
    :py:attr:`AlgorithmState.adaptive_state_code` so the pytree remains
    JAX-traceable. The SCP printing path maps the code back to a human-readable
    string via :func:`adaptive_state_code_to_str`.
    """

    REJECT = 0
    ACCEPT_HIGHER = 1
    ACCEPT_CONSTANT = 2
    ACCEPT_LOWER = 3
    INITIAL = 4


_ADAPTIVE_STATE_NAMES = {
    AdaptiveStateCode.REJECT: "Reject Higher",
    AdaptiveStateCode.ACCEPT_HIGHER: "Accept Higher",
    AdaptiveStateCode.ACCEPT_CONSTANT: "Accept Constant",
    AdaptiveStateCode.ACCEPT_LOWER: "Accept Lower",
    AdaptiveStateCode.INITIAL: "Initial",
}


def adaptive_state_code_to_str(code: Union[int, jnp.ndarray, np.ndarray]) -> str:
    """Map an :class:`AdaptiveStateCode` value (int or 0-d array) to its label."""
    return _ADAPTIVE_STATE_NAMES[AdaptiveStateCode(int(code))]


# ---------------------------------------------------------------------------
# AlgorithmState — JAX-traceable current iterate
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AlgorithmState:
    """Current-iterate pytree for SCP iterations.

    Every field is a JAX array (or scalar). The state moves between iterations
    by way of pure functional updates: :py:meth:`replace` returns a new state
    rather than mutating in place. Registered as a JAX pytree so the SCP body
    composes with ``jax.jit`` / ``jax.vmap`` / ``jax.grad``.

    Append-only iteration histories (X, U, discretizations, ...) live on
    :class:`AlgorithmHistory`, which is CPU-side and grown by the SCP loop.

    Attributes:
        x: Nodal state trajectory, shape ``(N, n_states)``.
        u: Nodal control trajectory, shape ``(N, n_controls)``.
        x_prop: Continuous-time propagation of ``x``, shape ``(N-1, n_states)``.
        x_prop_plus: Impulsive/discrete dynamics evaluated at ``x_prop``,
            shape ``(N, n_states)``. Zeros when no impulsive component is
            present.
        lam_prox: Trust-region weight, shape ``(N, n_states + n_controls)``.
        lam_vc: Virtual-control penalty weight, shape ``(N-1, n_states)``.
        lam_cost: Cost weight, shape ``(n_states,)``.
        lam_cost_init: Initial cost weight (``weights.lam_cost`` broadcast to
            ``(n_states,)``). Used by autotuners as the "reset" value during
            early iterations and on the ``state.k <= hyper.lam_cost_drop``
            branch. Lives on the pytree (not closure-captured) so weight
            mutations between solves propagate through the JIT'd
            update_weights.
        lam_vb_nodal: Nodal virtual-buffer weights, shape ``(N, n_nodal)``.
        lam_vb_cross: Cross-node virtual-buffer weights, shape ``(n_cross,)``.
        lam_vb_cvx: Virtual-buffer weights for soft convex constraints, shape ``(N, n_cvx)``.
        k: Iteration counter (starts at 1).
        J_tr: Current trust-region cost (scalar).
        J_vb: Current virtual-buffer cost (scalar).
        J_vc: Current virtual-control cost (scalar).
        J_nonlin: Nonlinear-objective value for the current accepted iterate.
        predicted_reduction: Predicted reduction in ``J_nonlin`` for this iter.
        actual_reduction: Actual reduction in ``J_nonlin`` for this iter.
        acceptance_ratio: ``actual_reduction / predicted_reduction``.
        adaptive_state_code: :class:`AdaptiveStateCode` value as ``int32``.
        x_init_pin: Initial-state boundary condition, shape ``(n_states,)``.
            ``jnp.nan`` where the state is not pinned at t0 (``initial_type``
            is not ``"Fix"``). Carried on the pytree — rather than read from
            ``settings`` — so the fused SCP iteration body assembles the
            subproblem's initial boundary rows as a pure function of state,
            and a future ``jax.vmap`` over problems can batch boundary
            conditions per element.
        x_term_pin: Terminal-state boundary condition, shape ``(n_states,)``.
            ``jnp.nan`` where the state is not pinned at tf (``final_type`` is
            not ``"Fix"``).
        ep_tr: Convergence threshold on ``J_tr`` (scalar). Carried on the
            pytree — like ``x_init_pin`` / ``lam_cost_init`` — so the SCP loop
            reads it as a runtime input: per-solve overrides and ``jax.vmap``
            sweeps need no retrace, and one exported ``solve_batched`` artifact
            serves every tolerance setting.
        ep_vb: Convergence threshold on ``J_vb`` (scalar).
        ep_vc: Convergence threshold on ``J_vc`` (scalar).
        k_max: SCP iteration cap (scalar, ``k``'s integer dtype). The loop
            runs while ``k <= k_max``; a traced bound is valid inside
            ``lax.while_loop``, so the cap is per-solve and batchable too.
        hyper: Autotuner-declared hyperparameters — an instance of the
            autotuner's :class:`HyperParams` subclass with scalar array
            leaves (the empty ``HyperParams()`` when it declares none).
            Seeded from :attr:`AutotuningBase.hyper`; ``update_weights``
            reads its knobs here (e.g. ``state.hyper.lam_cost_drop``) so
            each is a per-solve override and a batchable sweep target like
            any other field.
    """

    x: jnp.ndarray
    u: jnp.ndarray
    x_prop: jnp.ndarray
    x_prop_plus: jnp.ndarray
    lam_prox: jnp.ndarray
    lam_vc: jnp.ndarray
    lam_cost: jnp.ndarray
    lam_cost_init: jnp.ndarray
    lam_vb_nodal: jnp.ndarray
    lam_vb_cross: jnp.ndarray
    lam_vb_cvx: jnp.ndarray  # Added for soft convex constraints
    k: jnp.ndarray
    J_tr: jnp.ndarray
    J_vb: jnp.ndarray
    J_vc: jnp.ndarray
    J_nonlin: jnp.ndarray
    predicted_reduction: jnp.ndarray
    actual_reduction: jnp.ndarray
    acceptance_ratio: jnp.ndarray
    adaptive_state_code: jnp.ndarray
    x_init_pin: jnp.ndarray
    x_term_pin: jnp.ndarray
    ep_tr: jnp.ndarray
    ep_vb: jnp.ndarray
    ep_vc: jnp.ndarray
    k_max: jnp.ndarray
    hyper: HyperParams

    def replace(self, **changes) -> "AlgorithmState":
        """Return a new state with ``changes`` applied (functional update)."""
        return dc_replace(self, **changes)

    def tree_flatten(self):
        children = tuple(getattr(self, name) for name in self._FIELDS)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(**dict(zip(cls._FIELDS, children)))

    @classmethod
    def from_settings(
        cls,
        settings: "Config",
        weights: "Weights",
        *,
        ep_tr: float,
        ep_vb: float,
        ep_vc: float,
        k_max: int,
        hyper: HyperParams,
    ) -> "AlgorithmState":
        """Construct the initial iterate from configuration.

        Copies trajectory guesses, expands weights to dense arrays, and seeds
        the diagnostic scalars to zero / :py:attr:`AdaptiveStateCode.INITIAL`.
        ``x_prop`` and ``x_prop_plus`` are zero-initialized; the autotuner
        fills them when it carries the first accepted candidate onto the
        state (see :meth:`AutotuningBase.update_weights`).

        The SCP loop constants (``ep_tr`` / ``ep_vb`` / ``ep_vc`` / ``k_max``)
        and the autotuner's declared hyperparameters (*hyper*, from
        :attr:`AutotuningBase.hyper`) are snapshotted onto the pytree here —
        the caller passes them off the algorithm and its autotuner (see
        ``Problem._default_state``), and per-solve overrides land via
        ``state.replace``.
        """
        shadowed = sorted({fld.name for fld in dc_fields(hyper)} & set(cls._FIELDS))
        if shadowed:
            raise ValueError(
                f"Autotuner hyperparameter(s) {shadowed} shadow AlgorithmState "
                f"fields; rename the HyperParams fields so overrides stay "
                f"unambiguous."
            )
        n = settings.sim.n
        n_states = settings.sim.n_states
        n_controls = settings.sim.n_controls
        n_total = n_states + n_controls

        # Strong-typed dtypes so the initial state matches the autotuner's
        # outputs. JAX caches `jit` traces by argument weak/strong dtype AND
        # committed sharding; a mismatch between the seed state and the
        # post-iter-1 state would trigger an extra recompile inside the SCP
        # loop. We route every leaf through ``jax.device_put(..., device)`` to
        # produce *committed* arrays so the cache key matches what comes back
        # from the JIT'd ``update_weights``.
        #
        # Removable once the SCP loop body is a single JAX trace — at that
        # point the cache-key match is irrelevant and the strong-dtype /
        # committed-sharding routing below can collapse to plain ``jnp.asarray``.
        f = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
        i = jnp.int32
        device = jax.devices()[0]

        def put(arr):
            return jax.device_put(arr, device)

        def hyper_leaf(fld):
            # int knobs (e.g. lam_cost_drop) share k's dtype so comparisons
            # against the iteration counter don't promote; floats follow the
            # problem's float dtype. The annotation is the rule — validated
            # when the HyperParams subclass is defined.
            dtype = i if fld.type in (int, "int") else f
            return put(jnp.asarray(getattr(hyper, fld.name), dtype=dtype))

        lam_vc_array = np.ones((n - 1, n_states)) * weights.lam_vc
        lam_prox_array = np.ones((n, n_total)) * weights.lam_prox

        if isinstance(weights.lam_cost, np.ndarray):
            lam_cost_init = weights.lam_cost.copy()
        else:
            lam_cost_init = np.full(n_states, weights.lam_cost)

        # Boundary-condition pins: the physical value where the state is fixed
        # at t0 / tf, ``nan`` elsewhere. The subproblem only reads these at
        # ``"Fix"`` entries, so the sentinel marks "unpinned" without poisoning
        # any value the solver consumes.
        x_initial = np.asarray(settings.sim.x.initial, dtype=float).reshape(-1)
        x_final = np.asarray(settings.sim.x.final, dtype=float).reshape(-1)
        init_fixed = np.asarray(settings.sim.x.initial_type) == "Fix"
        final_fixed = np.asarray(settings.sim.x.final_type) == "Fix"
        x_init_pin = np.where(init_fixed, x_initial, np.nan)
        x_term_pin = np.where(final_fixed, x_final, np.nan)
        # lam_vb_cvx is populated by Weights.build_cvx_arrays, which
        # Problem.initialize calls once the symbolic constraints are known. A
        # Weights that never went through that step — constructed directly, or
        # a problem with no slacked convex constraints — leaves it None. Fall
        # back to the scalar lam_vb over one column, matching
        # build_cvx_arrays' own ``n_cvx = max(len(slacked), 1)`` convention.
        lam_vb_cvx_array = weights.lam_vb_cvx
        if lam_vb_cvx_array is None:
            lam_vb_cvx_array = np.full((n, 1), float(weights.lam_vb))

        return cls(
            x=put(jnp.asarray(settings.sim.x.guess, dtype=f)),
            u=put(jnp.asarray(settings.sim.u.guess, dtype=f)),
            x_prop=put(jnp.zeros((n - 1, n_states), dtype=f)),
            x_prop_plus=put(jnp.zeros((n, n_states), dtype=f)),
            lam_prox=put(jnp.asarray(lam_prox_array, dtype=f)),
            lam_vc=put(jnp.asarray(lam_vc_array, dtype=f)),
            lam_cost=put(jnp.asarray(lam_cost_init, dtype=f)),
            lam_cost_init=put(jnp.asarray(lam_cost_init, dtype=f)),
            lam_vb_nodal=put(jnp.asarray(weights.lam_vb_nodal, dtype=f)),
            lam_vb_cross=put(jnp.asarray(weights.lam_vb_cross, dtype=f)),
            lam_vb_cvx=put(jnp.asarray(lam_vb_cvx_array, dtype=f)),
            k=put(jnp.asarray(1, dtype=i)),
            J_tr=put(jnp.asarray(1e2, dtype=f)),
            J_vb=put(jnp.asarray(1e2, dtype=f)),
            J_vc=put(jnp.asarray(1e2, dtype=f)),
            J_nonlin=put(jnp.asarray(0.0, dtype=f)),
            predicted_reduction=put(jnp.asarray(0.0, dtype=f)),
            actual_reduction=put(jnp.asarray(0.0, dtype=f)),
            acceptance_ratio=put(jnp.asarray(0.0, dtype=f)),
            adaptive_state_code=put(jnp.asarray(int(AdaptiveStateCode.INITIAL), dtype=i)),
            x_init_pin=put(jnp.asarray(x_init_pin, dtype=f)),
            x_term_pin=put(jnp.asarray(x_term_pin, dtype=f)),
            ep_tr=put(jnp.asarray(ep_tr, dtype=f)),
            ep_vb=put(jnp.asarray(ep_vb, dtype=f)),
            ep_vc=put(jnp.asarray(ep_vc, dtype=f)),
            # k_max shares k's dtype so `k <= k_max` compares without promotion.
            k_max=put(jnp.asarray(k_max, dtype=i)),
            hyper=dc_replace(hyper, **{fld.name: hyper_leaf(fld) for fld in dc_fields(hyper)}),
        )


# Field order is the source of truth for tree_flatten / tree_unflatten; derive
# it from the dataclass so it can never drift from the field declarations.
AlgorithmState._FIELDS = tuple(f.name for f in dc_fields(AlgorithmState))


# ``AlgorithmState`` is the in/out pytree of the exported ``solve_batched``
# artifact, so ``jax.export`` must know how to (de)serialize its treedef on top
# of the runtime pytree registration above — a separate registry. The auxdata is
# ``None`` (see ``tree_flatten``), so serialization is empty bytes and rebuild
# falls back to the registered ``tree_unflatten``. Registered at import so any
# process that deserializes the artifact has it.
export.register_pytree_node_serialization(
    AlgorithmState,
    # The serialized name is a stable serialization ID, not a module path: it
    # must match the string baked into already-exported ``solve_batched``
    # artifacts. It deliberately keeps the historical ``...base.AlgorithmState``
    # even though the class now lives in this module — changing it would
    # strand cached artifacts (``export.deserialize`` hard-errors on an
    # unknown name; see ``openscvx/utils/caching.py``).
    serialized_name="openscvx.algorithms.base.AlgorithmState",
    serialize_auxdata=lambda aux: b"",
    deserialize_auxdata=lambda data: None,
)
