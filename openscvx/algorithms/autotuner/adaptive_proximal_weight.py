"""Autotuning functions for SCP (Successive Convex Programming) parameters."""

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from openscvx.config import Config

from .augmented_lagrangian import AugmentedLagrangian
from ..base import AutotuningBase

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

    from ..base import AlgorithmState, CandidateIterate
    from ..weights import Weights


class AdaptiveProximalWeight(AutotuningBase):
    """PTR-style proximal adaptation with fixed virtual penalty weights.

    Same acceptance-ratio logic as :class:`AugmentedLagrangian` for ``lam_prox``,
    but ``lam_vc`` and ``lam_vb_*`` are held constant at their current state values.
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

    @staticmethod
    def _copy_virtual_weights(
        candidate: "CandidateIterate",
        state: "AlgorithmState",
    ) -> None:
        candidate.lam_vc = state.lam_vc
        candidate.lam_vb_nodal = state.lam_vb_nodal
        candidate.lam_vb_cross = state.lam_vb_cross

    def update_weights(
        self,
        state: "AlgorithmState",
        candidate: "CandidateIterate",
        nodal_constraints: "LoweredJaxConstraints",
        settings: Config,
        params: dict,
        weights: "Weights",
    ) -> str:
        """Update SCP proximal weight based on acceptance ratio; keep VC/VB fixed."""
        candidate_x_prop = (
            candidate.x_prop_plus[1:] if candidate.x_prop_plus is not None else candidate.x_prop
        )
        (
            nonlinear_cost,
            nonlinear_penalty,
            nodal_penalty,
        ) = self.calculate_nonlinear_penalty(
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

        candidate.J_nonlin = nonlinear_cost + nonlinear_penalty + nodal_penalty

        if state.k > self.lam_cost_drop:
            candidate.lam_cost = state.lam_cost * self.lam_cost_relax
        else:
            candidate.lam_cost = weights.lam_cost

        lam_prox_k = deepcopy(state.lam_prox)

        if state.k > 1:
            state_x_prop_plus = state.x_prop_plus()
            state_x_prop = (
                state_x_prop_plus[1:] if state_x_prop_plus is not None else state.x_prop()
            )
            (
                prev_nonlinear_cost,
                prev_nonlinear_penalty,
                prev_nodal_penalty,
            ) = self.calculate_nonlinear_penalty(
                state_x_prop,
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

            J_nonlin_prev = prev_nonlinear_cost + prev_nonlinear_penalty + prev_nodal_penalty

            actual_reduction = J_nonlin_prev - candidate.J_nonlin
            predicted_reduction = J_nonlin_prev - candidate.J_lin

            if predicted_reduction == 0:
                raise ValueError("Predicted reduction is 0.")

            rho = actual_reduction / predicted_reduction

            state.pred_reduction_history.append(predicted_reduction)
            state.actual_reduction_history.append(actual_reduction)
            state.acceptance_ratio_history.append(rho)

            if rho < self.eta_0:
                lam_prox_k1 = np.minimum(self.lam_prox_max, self.gamma_1 * lam_prox_k)
                candidate.lam_prox = lam_prox_k1
                state.reject_solution(candidate)
                adaptive_state = "Reject Higher"
            elif rho >= self.eta_0 and rho < self.eta_1:
                lam_prox_k1 = np.minimum(self.lam_prox_max, self.gamma_1 * lam_prox_k)
                candidate.lam_prox = lam_prox_k1
                self._copy_virtual_weights(candidate, state)
                state.accept_solution(candidate)
                adaptive_state = "Accept Higher"
            elif rho >= self.eta_1 and rho < self.eta_2:
                candidate.lam_prox = lam_prox_k
                self._copy_virtual_weights(candidate, state)
                state.accept_solution(candidate)
                adaptive_state = "Accept Constant"
            else:
                lam_prox_k1 = np.maximum(self.lam_prox_min, self.gamma_2 * lam_prox_k)
                candidate.lam_prox = lam_prox_k1
                self._copy_virtual_weights(candidate, state)
                state.accept_solution(candidate)
                adaptive_state = "Accept Lower"

        else:
            candidate.lam_prox = lam_prox_k
            self._copy_virtual_weights(candidate, state)
            state.accept_solution(candidate)
            adaptive_state = "Initial"

        return adaptive_state


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
