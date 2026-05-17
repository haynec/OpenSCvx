"""Adaptive proximal-weight autotuner.

Same acceptance-ratio decision logic as :class:`AugmentedLagrangian` for
``lam_prox``, but ``lam_vc`` and ``lam_vb_*`` are held constant at their
current values.
"""

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict

from openscvx.config import Config

from ..base import AdaptiveStateCode, AutotuningBase
from .augmented_lagrangian import AugmentedLagrangian

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

    from ..base import AlgorithmState, CandidateIterate


class AdaptiveProximalWeight(AutotuningBase):
    """PTR-style proximal adaptation with fixed virtual-penalty weights.

    Same four-bucket acceptance-ratio logic as
    :class:`AugmentedLagrangian` for ``lam_prox``, but the virtual-control
    and virtual-buffer weights are carried unchanged.

    ``update_weights`` is a pure functional update on the
    :class:`AlgorithmState` pytree; see the base-class contract.
    """

    COLUMNS = AugmentedLagrangian.COLUMNS

    def __init__(
        self,
        gamma_1: float = 2.0,
        gamma_2: float = 0.5,
        eta_0: float = 1e-2,
        eta_1: float = 1e-1,
        eta_2: float = 0.8,
        lam_prox_min: float = 1e-3,
        lam_prox_max: float = 1e4,
        lam_cost_drop: int = -1,
        lam_cost_relax: float = 1.0,
    ):
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.eta_0 = eta_0
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.lam_prox_min = lam_prox_min
        self.lam_prox_max = lam_prox_max
        self.lam_cost_drop = lam_cost_drop
        self.lam_cost_relax = lam_cost_relax

    def update_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        settings: Config,
        params: dict,
    ) -> "AlgorithmState":
        """Return the next-iterate state.

        Pure functional update — see class docstring.
        """
        candidate_x_prop = candidate.x_prop_plus[1:]
        nonlin_cost, nonlin_pen, nodal_pen = self.calculate_nonlinear_penalty(
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
        J_nonlin = nonlin_cost + nonlin_pen + nodal_pen

        lam_cost_next = jnp.where(
            state.k > self.lam_cost_drop,
            state.lam_cost * self.lam_cost_relax,
            state.lam_cost_init,
        )

        def first_iter(state):
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
            prev_x_prop = state.x_prop_plus[1:]
            prev_cost, prev_pen, prev_nodal_pen = self.calculate_nonlinear_penalty(
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
            prev_J_nonlin = prev_cost + prev_pen + prev_nodal_pen

            actual = prev_J_nonlin - J_nonlin
            predicted = prev_J_nonlin - candidate.J_lin
            safe_pred = jnp.where(predicted == 0.0, 1.0, predicted)
            rho = jnp.where(predicted == 0.0, -jnp.inf, actual / safe_pred)

            is_reject = rho < self.eta_0
            is_accept_higher = (rho >= self.eta_0) & (rho < self.eta_1)
            is_accept_constant = (rho >= self.eta_1) & (rho < self.eta_2)
            accepted = ~is_reject

            lp_higher = jnp.minimum(self.lam_prox_max, self.gamma_1 * state.lam_prox)
            lp_lower = jnp.maximum(self.lam_prox_min, self.gamma_2 * state.lam_prox)
            new_lam_prox = jnp.where(
                is_reject | is_accept_higher,
                lp_higher,
                jnp.where(is_accept_constant, state.lam_prox, lp_lower),
            )

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
                # Virtual-control / virtual-buffer weights are held constant.
                lam_cost=lam_cost_next,
                J_nonlin=jnp.where(accepted, J_nonlin, state.J_nonlin),
                predicted_reduction=predicted,
                actual_reduction=actual,
                acceptance_ratio=rho,
                adaptive_state_code=code,
            )

        return jax.lax.cond(state.k == 1, first_iter, later_iter, state)


# =============================================================================
# Pydantic spec for dict / YAML validation
# =============================================================================


class AdaptiveProximalWeightSpec(BaseModel):
    """Validates AdaptiveProximalWeight configuration from dict/YAML input."""

    type: Literal["AdaptiveProximalWeight"] = "AdaptiveProximalWeight"
    gamma_1: float = 2.0
    gamma_2: float = 0.5
    eta_0: float = 1e-2
    eta_1: float = 1e-1
    eta_2: float = 0.8
    lam_prox_min: float = 1e-3
    lam_prox_max: float = 1e4
    lam_cost_drop: int = -1
    lam_cost_relax: float = 1.0

    model_config = ConfigDict(extra="forbid")

    def build(self) -> AdaptiveProximalWeight:
        return AdaptiveProximalWeight(**self.model_dump(exclude={"type"}, exclude_unset=True))
