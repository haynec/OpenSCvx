"""Acceptance-ratio autotuner family.

The acceptance-ratio autotuners share one decision rule: an acceptance ratio
:math:`\\rho` between the predicted and actual reduction of the nonlinear
objective drives a four-bucket update of the trust-region weight ``lam_prox``
(``REJECT`` / ``ACCEPT_HIGHER`` / ``ACCEPT_CONSTANT`` / ``ACCEPT_LOWER``). What
differs between members is the *multiplier* rule — how the virtual-control and
virtual-buffer weights (``lam_vc`` / ``lam_vb_*``) respond once an iterate is
accepted.

:class:`AcceptanceRatioAutotuner` owns the shared decision body and exposes a
single :meth:`AcceptanceRatioAutotuner._update_multipliers` hook for the
multiplier rule; subclasses supply only that hook.
:class:`AdaptiveProximalWeight` is the hook-default (multipliers held
constant); :class:`AugmentedLagrangian` overrides it with the
constraint-violation update.
"""

from typing import TYPE_CHECKING, Callable, List, Optional

import jax
import jax.numpy as jnp

from openscvx.config import Config
from openscvx.utils.printing import (
    Column,
    Verbosity,
    color_acceptance_ratio,
    color_adaptive_state,
)

from ..penalty import calculate_nonlinear_penalty
from ..state import AdaptiveStateCode
from .base import AutotuningBase, LamCostRelaxHyper

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

    from ..state import AlgorithmState, CandidateIterate


class AcceptanceRatioHyper(LamCostRelaxHyper):
    """Knobs shared by every acceptance-ratio autotuner.

    Extends the shared :class:`LamCostRelaxHyper` ``lam_cost`` relaxation knobs
    with the acceptance-ratio decision thresholds and the ``lam_prox`` clips.
    """

    gamma_1: float = 2.0
    gamma_2: float = 0.5
    eta_0: float = 1e-2
    eta_1: float = 1e-1
    eta_2: float = 0.8
    lam_prox_min: float = 1e-3
    lam_prox_max: float = 1e4


class AcceptanceRatioAutotuner(AutotuningBase):
    """Four-bucket acceptance-ratio update on ``lam_prox``.

    The acceptance ratio :math:`\\rho` (actual over predicted nonlinear
    reduction) selects one of four buckets — ``REJECT`` / ``ACCEPT_HIGHER`` /
    ``ACCEPT_CONSTANT`` / ``ACCEPT_LOWER`` — via a ``jnp.where`` cascade so the
    whole update traces under ``jax.jit``. Subclasses supply the multiplier
    rule (how ``lam_vc`` / ``lam_vb_*`` respond on acceptance) through
    :meth:`_update_multipliers`, and must assign an
    :class:`AcceptanceRatioHyper` (or subclass) instance to ``self.hyper`` in
    ``__init__`` — the shared body reads its knobs off ``state.hyper``.

    ``update_weights`` is a pure functional update on the
    :class:`AlgorithmState` pytree; see the base-class contract.
    """

    COLUMNS: List[Column] = [
        Column("J_nonlin", "J_nonlin", 8, "{: .1e}", None, Verbosity.STANDARD),
        Column("J_lin", "J_lin", 8, "{: .1e}", None, Verbosity.STANDARD),
        Column("pred_reduction", "pred_red", 9, "{: .1e}", min_verbosity=Verbosity.FULL),
        Column("actual_reduction", "act_red", 9, "{: .1e}", min_verbosity=Verbosity.FULL),
        Column(
            "acceptance_ratio",
            "acc_ratio",
            9,
            "{: .2e}",
            color_acceptance_ratio,
            Verbosity.STANDARD,
        ),
        Column("lam_prox", "lam_prox", 8, "{: .1e}", min_verbosity=Verbosity.FULL),
        Column("adaptive_state", "Adaptive", 16, "{}", color_adaptive_state, Verbosity.FULL),
    ]

    def _update_multipliers(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        params: dict,
        settings: Config,
        new_lam_prox: jnp.ndarray,
    ) -> dict:
        """Return the multiplier-weight updates to apply on an accepted iterate.

        Each returned entry maps an :class:`AlgorithmState` field name
        (``lam_vc`` / ``lam_vb_nodal`` / ``lam_vb_cross``) to its updated
        value, computed against the *candidate* trajectory; the shared body
        gates each by acceptance (a rejected iterate keeps the previous value).
        The default holds every multiplier constant — return ``{}``.
        """
        return {}

    def update_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        settings: Config,
        params: dict,
        extra_cost_fn: Optional[Callable] = None,
    ) -> "AlgorithmState":
        """Return the next-iterate state per the acceptance-ratio rules.

        Pure functional update — see class docstring.

        Args:
            extra_cost_fn: Optional ``(x, u, params) -> scalar`` JAX callable
                whose value is added to both ``J_nonlin`` (at the candidate
                point) and ``prev_J_nonlin`` (at the previous accepted iterate).
                Pass the SR composite ``s(R(x, u, params))`` here so the
                acceptance ratio accounts for the full composite cost.  Default
                ``None`` leaves the standard PTR behaviour unchanged.
        """
        candidate_x_prop = candidate.x_prop_plus[1:]
        nonlin_cost, nonlin_pen, nodal_pen = calculate_nonlinear_penalty(
            candidate_x_prop,
            candidate.x,
            candidate.u,
            state.lam_vc,
            state.lam_vb_nodal,
            state.lam_vb_cross,
            state.lam_cost,
            nodal_constraints,
            params,
            settings,
        )
        extra_cand = (
            extra_cost_fn(candidate.x, candidate.u, params)
            if extra_cost_fn is not None
            else jnp.asarray(0.0)
        )
        J_nonlin = nonlin_cost + nonlin_pen + nodal_pen + extra_cand

        lam_cost_next = self._relaxed_lam_cost(state)

        def first_iter(state):
            # Iter 1: accept unconditionally, leave weights at their init values,
            # only refresh trajectory + propagation fields.
            return state.replace(
                x=candidate.x,
                u=candidate.u,
                x_prop=candidate.x_prop,
                x_prop_plus=candidate.x_prop_plus,
                lam_cost=lam_cost_next,
                J_nonlin=J_nonlin,
                adaptive_state_code=jnp.asarray(int(AdaptiveStateCode.INITIAL), dtype=jnp.int32),
            )

        def later_iter(state):
            # Recompute the previous iterate's J_nonlin from the pytree fields
            # (state.x/state.u were the *previous* accepted iterate).
            prev_x_prop = state.x_prop_plus[1:]
            prev_cost, prev_pen, prev_nodal_pen = calculate_nonlinear_penalty(
                prev_x_prop,
                state.x,
                state.u,
                state.lam_vc,
                state.lam_vb_nodal,
                state.lam_vb_cross,
                state.lam_cost,
                nodal_constraints,
                params,
                settings,
            )
            extra_prev = (
                extra_cost_fn(state.x, state.u, params)
                if extra_cost_fn is not None
                else jnp.asarray(0.0)
            )
            prev_J_nonlin = prev_cost + prev_pen + prev_nodal_pen + extra_prev

            actual = prev_J_nonlin - J_nonlin
            predicted = prev_J_nonlin - candidate.J_lin
            # If predicted reduction is exactly zero, force the reject bucket
            # (rho = -inf) deterministically instead of raising.
            safe_pred = jnp.where(predicted == 0.0, 1.0, predicted)
            rho = jnp.where(predicted == 0.0, -jnp.inf, actual / safe_pred)

            is_reject = rho < state.hyper.eta_0
            is_accept_higher = (rho >= state.hyper.eta_0) & (rho < state.hyper.eta_1)
            is_accept_constant = (rho >= state.hyper.eta_1) & (rho < state.hyper.eta_2)
            # is_accept_lower implicit (else)
            accepted = ~is_reject

            # Compute both lam_prox candidates and gate.
            lp_higher = jnp.minimum(state.hyper.lam_prox_max, state.hyper.gamma_1 * state.lam_prox)
            lp_lower = jnp.maximum(state.hyper.lam_prox_min, state.hyper.gamma_2 * state.lam_prox)
            new_lam_prox = jnp.where(
                is_reject | is_accept_higher,
                lp_higher,
                jnp.where(is_accept_constant, state.lam_prox, lp_lower),
            )

            # Multiplier (virtual-control / virtual-buffer) updates: computed
            # against the *candidate* trajectory and gated by `accepted`. Reject
            # keeps the previous values (so the next subproblem doesn't see a
            # bumped penalty that wasn't earned). The default hook returns {} —
            # multipliers held constant.
            multiplier_updates = self._update_multipliers(
                state, candidate, nodal_constraints, params, settings, new_lam_prox
            )
            gated_multipliers = {
                name: jnp.where(accepted, value, getattr(state, name))
                for name, value in multiplier_updates.items()
            }

            code = jnp.where(
                is_reject,
                jnp.int32(AdaptiveStateCode.REJECT),
                jnp.where(
                    is_accept_higher,
                    jnp.int32(AdaptiveStateCode.ACCEPT_HIGHER),
                    jnp.where(
                        is_accept_constant,
                        jnp.int32(AdaptiveStateCode.ACCEPT_CONSTANT),
                        jnp.int32(AdaptiveStateCode.ACCEPT_LOWER),
                    ),
                ),
            )

            return state.replace(
                x=jnp.where(accepted, candidate.x, state.x),
                u=jnp.where(accepted, candidate.u, state.u),
                x_prop=jnp.where(accepted, candidate.x_prop, state.x_prop),
                x_prop_plus=jnp.where(accepted, candidate.x_prop_plus, state.x_prop_plus),
                lam_prox=new_lam_prox,
                lam_cost=lam_cost_next,
                J_nonlin=jnp.where(accepted, J_nonlin, state.J_nonlin),
                predicted_reduction=predicted,
                actual_reduction=actual,
                acceptance_ratio=rho,
                adaptive_state_code=code,
                **gated_multipliers,
            )

        return jax.lax.cond(state.k == 1, first_iter, later_iter, state)
