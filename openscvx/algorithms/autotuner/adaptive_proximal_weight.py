"""Adaptive proximal-weight autotuner.

Same acceptance-ratio decision logic as the rest of the family for
``lam_prox``, but ``lam_vc`` and ``lam_vb_*`` are held constant at their
current values — the default multiplier hook.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .acceptance_ratio import AcceptanceRatioAutotuner, AcceptanceRatioHyper


class AdaptiveProximalWeight(AcceptanceRatioAutotuner):
    """PTR-style proximal adaptation with fixed virtual-penalty weights.

    Inherits the four-bucket acceptance-ratio update on ``lam_prox`` from
    :class:`~openscvx.algorithms.autotuner.acceptance_ratio.AcceptanceRatioAutotuner`
    and keeps the default multiplier hook, so the virtual-control and
    virtual-buffer weights are carried unchanged.

    ``update_weights`` is a pure functional update on the
    :class:`AlgorithmState` pytree; see the base-class contract.
    """

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
        self.hyper = AcceptanceRatioHyper(
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            eta_0=eta_0,
            eta_1=eta_1,
            eta_2=eta_2,
            lam_prox_min=lam_prox_min,
            lam_prox_max=lam_prox_max,
            lam_cost_drop=lam_cost_drop,
            lam_cost_relax=lam_cost_relax,
        )


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
