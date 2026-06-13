"""Ramp proximal-weight autotuner.

Ramps ``lam_prox`` up by a constant factor each iteration until it hits the
configured maximum, then holds it constant.
"""

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict

from openscvx.config import Config

from ..state import AdaptiveStateCode
from .base import AutotuningBase, LamCostRelaxHyper

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

    from ..state import AlgorithmState, CandidateIterate


class RampProximalWeightHyper(LamCostRelaxHyper):
    """Declared hyperparameters for :class:`RampProximalWeight`.

    Extends the shared :class:`LamCostRelaxHyper` ``lam_cost`` relaxation knobs
    with the ramp factor and the ``lam_prox`` ceiling.
    """

    ramp_factor: float = 1.0
    lam_prox_max: float = 1e3


class RampProximalWeight(AutotuningBase):
    """Ramp ``lam_prox`` toward ``lam_prox_max`` then hold.

    ``update_weights`` is a pure functional update on the
    :class:`AlgorithmState` pytree; see the base-class contract.
    """

    COMPUTES_ACCEPTANCE_METRICS = False

    def __init__(
        self,
        ramp_factor: float = 1.0,
        lam_prox_max: float = 1e3,
        lam_cost_drop: int = -1,
        lam_cost_relax: float = 1.0,
    ):
        self.hyper = RampProximalWeightHyper(
            ramp_factor=ramp_factor,
            lam_prox_max=lam_prox_max,
            lam_cost_drop=lam_cost_drop,
            lam_cost_relax=lam_cost_relax,
        )

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
        lam_cost_next = self._relaxed_lam_cost(state)

        was_at_max = jnp.all(state.lam_prox >= state.hyper.lam_prox_max)
        new_lam_prox = jnp.minimum(
            state.lam_prox * state.hyper.ramp_factor, state.hyper.lam_prox_max
        )

        code = jnp.where(
            was_at_max,
            jnp.int32(AdaptiveStateCode.ACCEPT_CONSTANT),
            jnp.int32(AdaptiveStateCode.ACCEPT_HIGHER),
        )

        return state.replace(
            x=candidate.x,
            u=candidate.u,
            x_prop=candidate.x_prop,
            x_prop_plus=candidate.x_prop_plus,
            lam_prox=new_lam_prox,
            lam_cost=lam_cost_next,
            adaptive_state_code=code,
        )


# =============================================================================
# Pydantic spec for dict / YAML validation
# =============================================================================


class RampProximalWeightSpec(BaseModel):
    """Validates RampProximalWeight configuration from dict/YAML input."""

    type: Literal["RampProximalWeight"] = "RampProximalWeight"
    ramp_factor: float = 1.0
    lam_prox_max: float = 1e3
    lam_cost_drop: int = -1
    lam_cost_relax: float = 1.0

    model_config = ConfigDict(extra="forbid")

    def build(self) -> RampProximalWeight:
        return RampProximalWeight(**self.model_dump(exclude={"type"}, exclude_unset=True))
