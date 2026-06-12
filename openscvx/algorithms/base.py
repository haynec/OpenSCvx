"""Base classes for successive convexification algorithms.

This module defines the abstract interface that all SCP algorithm implementations
follow, along with two state containers used during the SCP iteration:

* :class:`AlgorithmState` — a JAX-registered, frozen pytree holding the
  *current iterate* (state/control/weights/diagnostics). Every autotuner's
  :py:meth:`AutotuningBase.update_weights` is a pure functional update on this
  pytree, so the SCP body composes with ``jax.jit``, ``jax.vmap``, and
  ``jax.grad``.
* :class:`AlgorithmHistory` — a CPU-side, mutable container for the append-only
  iteration histories (per-iteration trajectories, discretizations, costs,
  weight snapshots). Lives outside the JAX boundary; populated by the SCP loop
  via :py:meth:`AlgorithmHistory.record_iteration`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from dataclasses import replace as dc_replace
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import export

from openscvx.utils.printing import Column

if TYPE_CHECKING:
    import hashlib

    from openscvx.config import Config
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints

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


@dataclass(frozen=True, slots=True)
class DiscretizationResult:
    """Unpacked discretization data from a multi-shot discretization matrix.

    The discretization solver returns a matrix ``V`` that stores multiple blocks
    (propagated state and linearization matrices) across nodes/time. Historically,
    we stored the raw ``V`` matrices and re-unpacked them repeatedly via slicing.
    This dataclass unpacks once and makes access trivial.
    """

    V: np.ndarray  # raw V matrix, shape: (flattened_size, n_timesteps)
    x_prop: np.ndarray  # (N-1, n_x)
    A_d: np.ndarray  # (N-1, n_x, n_x)
    B_d: np.ndarray  # (N-1, n_x, n_u)
    C_d: np.ndarray  # (N-1, n_x, n_u)
    x_prop_plus: Optional[np.ndarray] = None  # (N, n_x), discrete dynamics on node states
    D_d: Optional[np.ndarray] = None  # (N, n_x, n_x), d(x_prop_plus)/d(x_node)
    E_d: Optional[np.ndarray] = None  # (N, n_x, n_u), d(x_prop_plus)/d(u_node)

    @classmethod
    def from_V(
        cls,
        V: np.ndarray,
        n_x: int,
        n_u: int,
        N: int,
    ) -> "DiscretizationResult":
        """Unpack the final timestep of a raw discretization matrix ``V``."""
        i1, i2 = n_x, n_x + n_x * n_x
        i3, i4 = i2 + n_x * n_u, i2 + 2 * n_x * n_u
        V_final = V[:, -1].reshape(-1, i4)
        return cls(
            V=np.asarray(V),
            x_prop=V_final[:, :i1],
            A_d=V_final[:, i1:i2].reshape(N - 1, n_x, n_x),
            B_d=V_final[:, i2:i3].reshape(N - 1, n_x, n_u),
            C_d=V_final[:, i3:i4].reshape(N - 1, n_x, n_u),
        )

    @classmethod
    def from_VW(
        cls,
        V: np.ndarray,
        W: np.ndarray,
        n_x: int,
        n_u: int,
        N: int,
    ) -> "DiscretizationResult":
        """Unpack continuous and impulsive discretization blocks from ``V`` and ``W``."""
        base = cls.from_V(V=V, n_x=n_x, n_u=n_u, N=N)

        W_arr = np.asarray(W)
        i_w = n_x + n_x * n_x + n_x * n_u
        i1 = n_x
        i2 = i1 + n_x * n_x
        i3 = i2 + n_x * n_u

        W_final = W_arr[:, -1].reshape(-1, i_w)

        return cls(
            V=base.V,
            x_prop=base.x_prop,
            A_d=base.A_d,
            B_d=base.B_d,
            C_d=base.C_d,
            x_prop_plus=W_final[:, :i1],
            D_d=W_final[:, i1:i2].reshape(W_final.shape[0], n_x, n_x),
            E_d=W_final[:, i2:i3].reshape(W_final.shape[0], n_x, n_u),
        )


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
# HyperParams — typed autotuner hyperparameter container
# ---------------------------------------------------------------------------


class HyperParams:
    """Base for autotuner hyperparameter containers.

    Subclassing is the entire integration. Declare each tunable knob as an
    annotated field with its default — bare annotations, no ``@dataclass``
    decorator — and the base class does the rest:

    * applies the frozen-dataclass transform, so instances are immutable
      value objects updated with :func:`dataclasses.replace`;
    * registers the subclass as a JAX pytree (and for ``jax.export`` treedef
      serialization), so instances ride :attr:`AlgorithmState.hyper` through
      ``jit`` / ``vmap`` / the exported batched ``solve_batched`` artifact;
    * wires dtype handling off the field annotations when
      :meth:`AlgorithmState.from_settings` snapshots the instance onto the
      state — ``int`` fields get the iteration counter's dtype, ``float``
      fields the problem float dtype. Any other annotation is rejected at
      class definition.

    Example::

        class MyHyper(HyperParams):
            ramp: float = 2.0
            drop: int = -1

    The empty base is the "no declared hyperparameters" container: it
    flattens to zero pytree leaves.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # ``__init_subclass__`` runs before any decorator on the subclass
        # could, so the transform must happen here — which is exactly what
        # lets authors skip the dataclass and pytree mechanics entirely.
        dataclass(frozen=True)(cls)
        for fld in dc_fields(cls):
            # fld.type is a string under postponed annotation evaluation
            # (``from __future__ import annotations``); accept both forms.
            if fld.type not in (int, float, "int", "float"):
                raise TypeError(
                    f"{cls.__name__}.{fld.name}: HyperParams fields must be "
                    f"annotated int or float (got {fld.type!r}) — the "
                    f"annotation decides the dtype the knob gets on "
                    f"AlgorithmState.hyper."
                )
        jax.tree_util.register_dataclass(cls)
        export.register_pytree_node_serialization(
            cls,
            serialized_name=f"{cls.__module__}.{cls.__qualname__}",
            serialize_auxdata=lambda aux: b"",
            deserialize_auxdata=lambda data: (),
        )


# The base itself is a zero-field frozen dataclass and a zero-leaf pytree, so
# the shared ``HyperParams()`` default on ``AutotuningBase`` is a valid value
# for ``AlgorithmState.hyper``. Auxdata of a ``register_dataclass`` node is
# the (empty) tuple of meta-field values, hence ``()`` — not ``None`` — on
# deserialize; same for the subclasses registered above.
dataclass(frozen=True)(HyperParams)
jax.tree_util.register_dataclass(HyperParams)
export.register_pytree_node_serialization(
    HyperParams,
    serialized_name="openscvx.algorithms.base.HyperParams",
    serialize_auxdata=lambda aux: b"",
    deserialize_auxdata=lambda data: (),
)


class LamCostRelaxHyper(HyperParams):
    """``lam_cost`` relaxation knobs shared by every built-in autotuner.

    Every autotuner relaxes ``lam_cost`` by the same rule (see
    :meth:`AutotuningBase._relaxed_lam_cost`), so the two knobs that steer it
    live here once instead of being re-declared per class. ``lam_cost_drop`` is
    the iteration after which relaxation applies (``state.k > lam_cost_drop``):
    ``-1`` relaxes from the first iteration, and the default
    ``lam_cost_relax=1.0`` makes that a no-op.
    """

    lam_cost_drop: int = -1
    lam_cost_relax: float = 1.0


# ---------------------------------------------------------------------------
# Autotuning base class
# ---------------------------------------------------------------------------


class AutotuningBase(ABC):
    """Base class for autotuning strategies in SCP algorithms.

    Subclasses implement :py:meth:`update_weights` as a **pure functional
    update** on an :class:`AlgorithmState` pytree. Concretely, an autotuner
    must:

    * Not mutate ``state`` or ``candidate``.
    * Not return strings, raise on data-dependent conditions, or append to
      Python lists at trace time.
    * Express all branching on iterate values via ``jax.lax.cond`` /
      ``jnp.where``.

    The autotuner's per-iteration outcome is encoded as an
    :class:`AdaptiveStateCode` int32 on the returned state; the SCP loop
    converts that back to a human-readable label only on the Python-loop
    printing path.

    **The returned state IS the next iterate.** The SCP loop discards
    everything except what ``update_weights`` returns, so accepting the
    subproblem's candidate means carrying its trajectory onto the returned
    state — ``x`` / ``u`` / ``x_prop`` / ``x_prop_plus`` from ``candidate`` —
    while rejecting means keeping the previous fields and adjusting only the
    weights. An update that never copies the candidate produces a solver that
    runs to ``k_max`` without the iterate ever moving, silently.

    **Declaring tunable hyperparameters.** A numeric knob a user might sweep
    (a penalty ramp, a relaxation iteration) must not be read off ``self``
    inside ``update_weights`` — a Python attribute is baked into the trace
    and invisible to every override channel. Instead, declare it on a
    :class:`HyperParams` subclass, assign an instance to ``self.hyper``, and
    read it from ``state.hyper``::

        class MyHyper(HyperParams):
            ramp: float = 2.0

        class MyAutotuner(AutotuningBase):
            def __init__(self, ramp: float = 2.0):
                self.hyper = MyHyper(ramp=ramp)

            def update_weights(self, state, candidate, *args):
                return state.replace(
                    # accept the candidate: it becomes the next iterate
                    x=candidate.x,
                    u=candidate.u,
                    x_prop=candidate.x_prop,
                    x_prop_plus=candidate.x_prop_plus,
                    lam_prox=state.lam_prox * state.hyper.ramp,
                    adaptive_state_code=...,
                )

    The declaration is the registration: the field becomes a per-solve
    override (``solve_jax(algorithm={"ramp": ...})``), a batchable sweep
    target (``solve_batched(algorithm={"ramp": jnp.linspace(...)})``), and a
    runtime input of the exported batched artifact — with zero core edits.
    Every numeric knob of the built-in autotuners is declared this way, so the
    only configuration left outside ``hyper`` is structural choices that select
    code paths (class attributes like ``COMPUTES_ACCEPTANCE_METRICS``); they
    are part of the traced program, not data. For ergonomics, declared knobs
    are still readable and writable as bare attributes
    (``autotuner.ramp = 3.0``): the proxy below routes the access into
    ``hyper`` so a constructor — or a test — may keep assigning knobs by name.
    Assign ``self.hyper`` *first* in ``__init__``; a knob assigned before
    ``hyper`` exists lands as a plain instance attribute the proxy cannot see.

    Class Attributes:
        COLUMNS: List of Column specs for autotuner-specific metrics to display.
            Subclasses override this to add their own columns.
        COMPUTES_ACCEPTANCE_METRICS: Whether ``update_weights`` computes a
            predicted/actual reduction and the resulting acceptance ratio. The
            SCP loop records and prints those diagnostics only when this is
            ``True`` (the default); tuners that never reject an iterate — e.g.
            :class:`~openscvx.algorithms.autotuner.constant_proximal_weight.ConstantProximalWeight`
            and :class:`~openscvx.algorithms.autotuner.ramp_proximal_weight.RampProximalWeight`
            — set it ``False``.
        hyper: The autotuner's declared hyperparameters — a
            :class:`HyperParams` instance carrying plain-Python defaults
            (the empty base when it declares none). Snapshotted onto
            ``AlgorithmState.hyper`` with array leaves (see
            :meth:`AlgorithmState.from_settings`; ``int`` fields get ``k``'s
            integer dtype, ``float`` fields the float dtype), which is what
            ``update_weights`` reads at trace time.
    """

    COLUMNS: List[Column] = []
    COMPUTES_ACCEPTANCE_METRICS: bool = True
    hyper: HyperParams = HyperParams()

    def __getattr__(self, name: str):
        """Read a declared hyperparameter as a bare attribute.

        Promoted knobs live on the frozen ``hyper`` container, but the
        documented API and constructors still touch them by name
        (``autotuner.lam_prox_max``). ``__getattr__`` fires only on failed
        normal lookups, so real attributes (``COLUMNS``, ``hyper`` itself) are
        unaffected; a name matching a ``hyper`` field resolves there.
        """
        hyper = self.__dict__.get("hyper")
        if hyper is not None and name in {f.name for f in dc_fields(hyper)}:
            return getattr(hyper, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        """Route a write to a declared hyperparameter into the frozen ``hyper``.

        Without this, assigning a promoted knob as a bare attribute
        (``autotuner.lam_prox_max = 1e6``) would silently shadow the ``hyper``
        field and never reach the solve. A name matching a ``hyper`` field is
        applied via :func:`dataclasses.replace`; every other attribute
        (including ``hyper`` itself) is set normally, so constructors may keep
        assigning knobs by name — they route into ``hyper``, provided
        ``self.hyper`` was assigned first (see the class docstring).
        """
        hyper = self.__dict__.get("hyper")
        if hyper is not None and name != "hyper" and name in {f.name for f in dc_fields(hyper)}:
            # Clear any instance attribute shadowing the knob (left by an
            # assignment made before ``hyper`` existed) so reads resolve
            # through ``__getattr__`` to the value just written.
            self.__dict__.pop(name, None)
            super().__setattr__("hyper", dc_replace(hyper, **{name: value}))
        else:
            super().__setattr__(name, value)

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Contribute the autotuner's update rule to the ``solve_batched`` cache key.

        The exported batched loop bakes in ``update_weights`` and every numeric
        parameter that steers it (penalty ramps, acceptance thresholds, weight
        clips). The default hashes the concrete class plus all instance
        attributes — sufficient because autotuner parameters are plain
        scalars. The ``hyper`` *values* are excluded: declared hyperparameters
        ride ``AlgorithmState.hyper`` as runtime inputs, so one artifact serves
        every setting of them (the same reasoning that keeps the ``ep_*``
        thresholds out of the algorithm's hash). The ``hyper`` *schema* is
        folded in, though: the ``hyper`` leaves map positionally onto the
        exported artifact's avals, so a field reorder or addition silently
        permutes runtime values against the wrong leaves. Hashing the field
        names in declaration order turns such a change into a clean cache miss
        and rebuild — this matters because :class:`LamCostRelaxHyper` moves
        ``lam_cost_drop`` / ``lam_cost_relax`` to the *front* of the field
        order. Folded in by :meth:`Algorithm._hash_into`; mirrors the symbolic
        ``_hash_into`` protocol.
        """
        from openscvx.utils.caching import hash_value_into

        hasher.update(type(self).__name__.encode())
        for name in sorted(vars(self)):
            if name == "hyper":
                continue
            hasher.update(name.encode())
            hash_value_into(hasher, getattr(self, name))
        # Schema, not values: the hyper field order/names fix the exported
        # artifact's leaf layout, so a reorder must invalidate cached artifacts
        # rather than silently permute runtime values onto the wrong avals.
        for fld in dc_fields(self.hyper):
            hasher.update(fld.name.encode())

    @staticmethod
    def calculate_cost_from_state(
        x: jnp.ndarray,
        settings: "Config",
        lam_cost: Union[float, jnp.ndarray] = 1.0,
    ) -> jnp.ndarray:
        """Compute the boundary-condition-weighted cost contribution for ``x``.

        Args:
            x: State trajectory, shape ``(N, n_states)``.
            settings: Configuration object carrying scaling matrices and
                boundary-condition types.
            lam_cost: Per-state cost weight. Scalar or array of shape
                ``(n_states,)``.

        Returns:
            Scalar cost (jnp), weighted by ``lam_cost``.
        """
        scaled_x = (settings.sim.inv_S_x @ (x.T - settings.sim.c_x[:, None])).T
        lam = jnp.asarray(lam_cost)
        cost = jnp.asarray(0.0)
        for i in range(settings.sim.n_states):
            w = lam[i] if lam.ndim > 0 else lam
            if settings.sim.x.final_type[i] == "Minimize":
                cost = cost + w * scaled_x[-1, i]
            if settings.sim.x.final_type[i] == "Maximize":
                cost = cost - w * scaled_x[-1, i]
            if settings.sim.x.initial_type[i] == "Minimize":
                cost = cost + w * scaled_x[0, i]
            if settings.sim.x.initial_type[i] == "Maximize":
                cost = cost - w * scaled_x[0, i]
        return cost

    @staticmethod
    def calculate_nonlinear_penalty(
        x_prop: jnp.ndarray,
        x_bar: jnp.ndarray,
        u_bar: jnp.ndarray,
        lam_vc: jnp.ndarray,
        lam_vb_nodal: jnp.ndarray,
        lam_vb_cross: jnp.ndarray,
        lam_cost: Union[float, jnp.ndarray],
        nodal_constraints: "LoweredJaxConstraints",
        params: dict,
        settings: "Config",
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute the three components of the nonlinear penalty.

        This is JAX-traceable: the Python loops over
        ``nodal_constraints.nodal`` / ``cross_node`` unroll at trace time
        (the lists are static-length lists of compiled closures built by
        :class:`Problem`).

        Args:
            x_prop: Propagated state, shape ``(N-1, n_states)``.
            x_bar: Nodal state, shape ``(N, n_states)``.
            u_bar: Nodal control, shape ``(N, n_controls)``.
            lam_vc: Virtual control weight, scalar or matrix.
            lam_vb_nodal: Nodal virtual-buffer weights, shape ``(N, n_nodal)``.
            lam_vb_cross: Cross-node virtual-buffer weights, shape ``(n_cross,)``.
            lam_cost: Cost weight, scalar or shape ``(n_states,)``.
            nodal_constraints: Lowered JAX constraints.
            params: Problem parameter dictionary.
            settings: Configuration object.

        Returns:
            ``(nonlinear_cost, nonlinear_penalty, nodal_penalty)`` — all
            scalar jnp arrays.
        """
        nodal_penalty = jnp.asarray(0.0)

        for idx, constraint in enumerate(nodal_constraints.nodal):
            g = constraint.func(x_bar, u_bar, 0, params)
            if constraint.nodes is not None:
                nodes_array = jnp.asarray(constraint.nodes)
                g_filtered = g[nodes_array]
                w = lam_vb_nodal[nodes_array, idx]
            else:
                g_filtered = g
                w = lam_vb_nodal[:, idx]
            nodal_penalty = nodal_penalty + jnp.sum(w * jnp.maximum(0.0, g_filtered))

        for idx, constraint in enumerate(nodal_constraints.cross_node):
            w = lam_vb_cross[idx]
            g = constraint.func(x_bar, u_bar, params)
            nodal_penalty = nodal_penalty + w * jnp.sum(jnp.maximum(0.0, g))

        cost = AutotuningBase.calculate_cost_from_state(x_bar, settings, lam_cost)
        x_diff = settings.sim.inv_S_x @ (x_bar[1:, :] - x_prop).T

        return cost, jnp.sum(lam_vc * jnp.abs(x_diff.T)), nodal_penalty

    @staticmethod
    def _relaxed_lam_cost(state: "AlgorithmState") -> jnp.ndarray:
        """``lam_cost`` for the next iterate per the shared relaxation rule.

        When ``state.k > state.hyper.lam_cost_drop``, scale ``state.lam_cost``
        by ``state.hyper.lam_cost_relax``; otherwise reset to the algorithm's
        initial weight (carried on the pytree as ``state.lam_cost_init``,
        broadcast at :meth:`AlgorithmState.from_settings`). Both constants ride
        the pytree so per-solve overrides and ``vmap`` sweeps reach the traced
        body; the scalar ``lam_cost_relax`` preserves the user-specified
        per-state weight ratios. Shared verbatim by every built-in autotuner
        via the :class:`LamCostRelaxHyper` knobs.
        """
        return jnp.where(
            state.k > state.hyper.lam_cost_drop,
            state.lam_cost * state.hyper.lam_cost_relax,
            state.lam_cost_init,
        )

    @abstractmethod
    def update_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        settings: "Config",
        params: dict,
    ) -> "AlgorithmState":
        """Return the next-iterate :class:`AlgorithmState`.

        Must be JAX-traceable. See the class docstring for the contract.

        Args:
            state: Current-iterate pytree. Initial weights live on
                ``state.lam_cost_init``; autotuners read from there rather than
                from the algorithm's ``Weights`` object so the JIT'd closure
                stays cache-stable across weight mutations.
            candidate: Subproblem result (read-only here). On acceptance its
                trajectory fields must be carried onto the returned state —
                see the class docstring.
            nodal_constraints: Lowered JAX constraints.
            settings: Configuration object.
            params: Problem parameter dictionary.

        Returns:
            The next-iterate :class:`AlgorithmState` — the SCP loop uses
            nothing else, so an accepting update carries
            ``candidate.x / u / x_prop / x_prop_plus`` onto it, while a
            rejecting update keeps the previous fields and adjusts only the
            weights. Its :py:attr:`AlgorithmState.adaptive_state_code` encodes
            the autotuner's decision; the SCP loop records that into history
            and maps it to a printable label.
        """


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
    serialized_name="openscvx.algorithms.base.AlgorithmState",
    serialize_auxdata=lambda aux: b"",
    deserialize_auxdata=lambda data: None,
)


# ---------------------------------------------------------------------------
# AlgorithmHistory — CPU-side append-only iteration log
# ---------------------------------------------------------------------------


@dataclass
class AlgorithmHistory:
    """Append-only iteration log, populated by the SCP loop.

    Mirrors the lists that previously lived directly on ``AlgorithmState``.
    Never appears on the JAX boundary; the SCP loop appends to it after each
    iteration via :py:meth:`record_iteration`. Purely diagnostic — the
    subproblem solver consumes ``SubproblemData`` in-trace, not the history.
    """

    n_x: int
    n_u: int
    N: int
    X: List[np.ndarray] = field(default_factory=list)
    U: List[np.ndarray] = field(default_factory=list)
    discretizations: List[DiscretizationResult] = field(default_factory=list)
    VC: List[np.ndarray] = field(default_factory=list)
    TR: List[np.ndarray] = field(default_factory=list)
    lam_prox: List[np.ndarray] = field(default_factory=list)
    lam_vc: List[Union[float, np.ndarray]] = field(default_factory=list)
    lam_cost: List[Union[float, np.ndarray]] = field(default_factory=list)
    lam_vb_nodal: List[np.ndarray] = field(default_factory=list)
    lam_vb_cross: List[np.ndarray] = field(default_factory=list)
    J_nonlin: List[float] = field(default_factory=list)
    J_lin: List[float] = field(default_factory=list)
    pred_reduction: List[float] = field(default_factory=list)
    actual_reduction: List[float] = field(default_factory=list)
    acceptance_ratio: List[float] = field(default_factory=list)
    adaptive_state: List[str] = field(default_factory=list)

    @classmethod
    def from_settings(cls, settings: "Config") -> "AlgorithmHistory":
        """Construct an empty history sized to the problem."""
        return cls(
            n_x=settings.sim.n_states,
            n_u=settings.sim.n_controls,
            N=settings.sim.n,
        )

    @property
    def V_history(self) -> List[np.ndarray]:
        """View of the raw discretization matrices recorded so far."""
        return [d.V for d in self.discretizations]

    # -- Per-iteration recording -------------------------------------------

    def record_iteration(
        self,
        state: AlgorithmState,
        *,
        V: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
        VC: Optional[np.ndarray] = None,
        TR: Optional[np.ndarray] = None,
        J_lin: Optional[float] = None,
        record_diagnostics: bool = True,
    ) -> Tuple[dict, np.ndarray]:
        """Append per-iteration data based on ``state.adaptive_state_code``.

        The accepted iterate (``x`` / ``u`` / weights / convergence scalars) is
        read off ``state``; the raw host-side diagnostics the SCP loop already
        synced — discretization matrices (``V`` / ``W``), virtual control
        (``VC``), trust region (``TR``), and the linearized cost (``J_lin``) —
        come in as explicit keyword arrays so the generic history never has to
        know the algorithm's diagnostics type.

        Reproduces the old ``accept_solution`` / ``reject_solution`` behavior:

        * **REJECT**: append ``lam_prox`` (the bumped weight that drives the
          next subproblem), plus the predicted/actual/acceptance diagnostics
          when ``record_diagnostics`` is set.
        * **INITIAL**: append every trajectory / weight history entry, but
          skip the predicted/actual/acceptance diagnostics (the autotuner
          didn't compute them on iter 1).
        * **ACCEPT_***: append everything, including diagnostics when
          ``record_diagnostics`` is set (the SCP loop turns this off for
          autotuners that don't compute them — :class:`ConstantProximalWeight`,
          :class:`RampProximalWeight`).

        Every device-resident leaf this method needs is pulled with one
        ``jax.device_get`` call (a single CPU<->device round trip) and the
        host-side numpy arrays are then partitioned onto the history lists.
        The SCP loop also reads the same scalars for printing + convergence,
        so the bundle is returned alongside the lam_prox array to avoid a
        second sync.

        Returns:
            ``(scalars, lam_prox_np)`` — a dict of host-side scalar floats
            (``J_tr``, ``J_vb``, ``J_vc``, ``J_nonlin``,
            ``predicted_reduction``, ``actual_reduction``,
            ``acceptance_ratio``) plus the int ``adaptive_state_code``, and
            the numpy copy of ``state.lam_prox``. The SCP emitter consumes
            these directly without re-syncing.
        """
        # Coalesce every device read into one transfer. ``jax.device_get`` on
        # a tuple lets XLA dispatch the whole bundle as a single host copy,
        # which is the entire point on tiny problems where dozens of independent
        # ``float(state.scalar)`` calls would each issue a sync.
        leaves = jax.device_get(
            (
                state.adaptive_state_code,
                state.lam_prox,
                state.J_tr,
                state.J_vb,
                state.J_vc,
                state.J_nonlin,
                state.predicted_reduction,
                state.actual_reduction,
                state.acceptance_ratio,
                state.x,
                state.u,
                state.lam_vc,
                state.lam_cost,
                state.lam_vb_nodal,
                state.lam_vb_cross,
            )
        )
        (
            asc_np,
            lam_prox_np,
            J_tr_np,
            J_vb_np,
            J_vc_np,
            J_nonlin_np,
            pred_np,
            actual_np,
            ratio_np,
            x_np,
            u_np,
            lam_vc_np,
            lam_cost_np,
            lam_vb_nodal_np,
            lam_vb_cross_np,
        ) = leaves

        adaptive_code = int(asc_np)
        code = AdaptiveStateCode(adaptive_code)

        scalars = {
            "J_tr": float(J_tr_np),
            "J_vb": float(J_vb_np),
            "J_vc": float(J_vc_np),
            "J_nonlin": float(J_nonlin_np),
            "predicted_reduction": float(pred_np),
            "actual_reduction": float(actual_np),
            "acceptance_ratio": float(ratio_np),
            "adaptive_state_code": adaptive_code,
        }

        self.lam_prox.append(np.asarray(lam_prox_np))
        self.adaptive_state.append(adaptive_state_code_to_str(code))

        if code is AdaptiveStateCode.REJECT:
            if record_diagnostics:
                self.pred_reduction.append(scalars["predicted_reduction"])
                self.actual_reduction.append(scalars["actual_reduction"])
                self.acceptance_ratio.append(scalars["acceptance_ratio"])
            return scalars, lam_prox_np

        # INITIAL and any ACCEPT_*: full record of the accepted iterate.
        self.X.append(np.asarray(x_np))
        self.U.append(np.asarray(u_np))

        if V is not None:
            if W is not None:
                self.discretizations.append(
                    DiscretizationResult.from_VW(
                        V,
                        W,
                        n_x=self.n_x,
                        n_u=self.n_u,
                        N=self.N,
                    )
                )
            else:
                self.discretizations.append(
                    DiscretizationResult.from_V(
                        V,
                        n_x=self.n_x,
                        n_u=self.n_u,
                        N=self.N,
                    )
                )
        if VC is not None:
            self.VC.append(np.asarray(VC))
        if TR is not None:
            self.TR.append(np.asarray(TR))

        self.lam_vc.append(np.asarray(lam_vc_np))
        self.lam_cost.append(np.asarray(lam_cost_np))
        self.lam_vb_nodal.append(np.asarray(lam_vb_nodal_np))
        self.lam_vb_cross.append(np.asarray(lam_vb_cross_np))

        self.J_nonlin.append(scalars["J_nonlin"])
        if J_lin is not None:
            self.J_lin.append(float(J_lin))

        # Diagnostics: only meaningful for iterations after the initial one.
        if record_diagnostics and code is not AdaptiveStateCode.INITIAL:
            self.pred_reduction.append(scalars["predicted_reduction"])
            self.actual_reduction.append(scalars["actual_reduction"])
            self.acceptance_ratio.append(scalars["acceptance_ratio"])

        return scalars, lam_prox_np


# ---------------------------------------------------------------------------
# Algorithm base class
# ---------------------------------------------------------------------------


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
        state: AlgorithmState,
        history: AlgorithmHistory,
        params: dict,
        settings: "Config",
    ) -> Tuple[AlgorithmState, bool]:
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

    def converged(self, state: AlgorithmState) -> jnp.ndarray:
        """Boolean SCP convergence test from the metrics and thresholds on ``state``.

        The default — every metric below its threshold — is algorithm-agnostic:
        :class:`AlgorithmState` carries ``J_tr`` / ``J_vb`` / ``J_vc`` and the
        ``ep_*`` tolerances generically, so it serves any SCP algorithm. Override
        to change the convergence policy; the override is honored on all three
        solve paths (``solve`` / ``solve_jax`` / ``solve_batched``), since
        :meth:`step` and the ``lax.while_loop`` harness both route through it.

        Must be JAX-traceable: it runs inside the ``lax.while_loop`` cond of
        ``solve_jax`` / ``solve_batched`` (where it is ``jax.vmap``'d per batch
        element) as well as on the Python ``solve()`` path. An override implies a
        subclass, hence a distinct ``type(self).__name__`` that :meth:`_hash_into`
        already folds into the ``solve_batched`` export cache key — so a changed
        predicate invalidates stale batched artifacts without extra machinery.
        """
        return (state.J_tr < state.ep_tr) & (state.J_vb < state.ep_vb) & (state.J_vc < state.ep_vc)

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Contribute this algorithm's identity to the ``solve_batched`` cache key.

        The exported batched loop bakes in the initial penalty weights and the
        autotuner's update rule — none of which the symbolic problem hash
        covers. The default folds in the concrete class name, then the six
        :class:`~openscvx.algorithms.weights.Weights` fields enumerated
        below (a new ``Weights`` field must be added to that tuple), then the
        autotuner via its own ``_hash_into`` — mirroring the symbolic
        ``_hash_into`` protocol.

        The convergence thresholds (``ep_tr`` / ``ep_vb`` / ``ep_vc``) and the
        iteration cap ``k_max`` / time limit ``t_max`` are deliberately *not*
        folded in here: they ride the :class:`AlgorithmState` pytree as runtime
        inputs, so one exported artifact serves every tolerance and
        ``max_iters`` setting. Override only if an algorithm's identity needs
        something beyond ``weights`` + ``autotuner``.
        """
        from openscvx.utils.caching import hash_value_into

        hasher.update(type(self).__name__.encode())
        w = self.weights
        for value in (
            w.lam_prox,
            w.lam_vc,
            w.lam_cost,
            w.lam_vb,
            w.lam_vb_nodal,
            w.lam_vb_cross,
        ):
            hash_value_into(hasher, value)
        self.autotuner._hash_into(hasher)

    @abstractmethod
    def citation(self) -> List[str]:
        """Return BibTeX citations for this algorithm."""
        raise NotImplementedError
