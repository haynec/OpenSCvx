"""Abstract base for autotuning strategies.

:class:`AutotuningBase` is the contract every penalty-weight update rule
implements: :py:meth:`~AutotuningBase.update_weights` is a pure functional
update on an :class:`~openscvx.algorithms.state.AlgorithmState` pytree, so the
SCP body composes with ``jax.jit`` / ``jax.vmap`` / ``jax.grad``. The base also
houses :class:`LamCostRelaxHyper`, the ``lam_cost`` relaxation knobs every
built-in autotuner shares, and :meth:`AutotuningBase._relaxed_lam_cost`, the
single implementation of that rule.

This module mirrors the package's ``base.py``-is-the-ABC idiom inside the
``autotuner`` subpackage; the concrete autotuners live alongside it.
"""

from abc import ABC, abstractmethod
from dataclasses import fields as dc_fields
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, List

import jax.numpy as jnp

from openscvx.utils.printing import Column

from ..hyperparams import HyperParams

if TYPE_CHECKING:
    import hashlib

    from openscvx.config import Config
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints

    from ..state import AlgorithmState, CandidateIterate


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

            def update_weights(self, state, candidate, nodal_constraints, settings, params):
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
