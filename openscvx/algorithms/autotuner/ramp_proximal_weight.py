"""Ramp proximal-weight autotuner.

Ramps ``lam_prox`` up by a constant factor each iteration until it hits the
configured maximum, then holds it constant.
"""

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict

from openscvx.config import Config

from ..base import AdaptiveStateCode, AutotuningBase, HyperParams

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

    from ..base import AlgorithmState, CandidateIterate


class RampProximalWeightHyper(HyperParams):
    """Declared hyperparameters for :class:`RampProximalWeight`.

    ``lam_cost_drop`` is the iteration after which ``lam_cost`` relaxation
    applies (``state.k > lam_cost_drop``): ``-1`` relaxes from the first
    iteration, and the default ``lam_cost_relax=1.0`` makes that a no-op.
    """

    lam_cost_drop: int = -1


class RampProximalWeight(AutotuningBase):
    """Ramp ``lam_prox`` toward ``lam_prox_max`` then hold.

    ``update_weights`` is a pure functional update on the
    :class:`AlgorithmState` pytree; see the base-class contract.
    """

    def __init__(
        self,
        ramp_factor: float = 1.0,
        lam_prox_max: float = 1e3,
        lam_cost_drop: int = -1,
        lam_cost_relax: float = 1.0,
    ):
        self.ramp_factor = ramp_factor
        self.lam_prox_max = lam_prox_max
        self.hyper = RampProximalWeightHyper(lam_cost_drop=lam_cost_drop)
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
        lam_cost_next = jnp.where(
            state.k > state.hyper.lam_cost_drop,
            state.lam_cost * self.lam_cost_relax,
            state.lam_cost_init,
        )

        was_at_max = jnp.all(state.lam_prox >= self.lam_prox_max)
        new_lam_prox = jnp.minimum(state.lam_prox * self.ramp_factor, self.lam_prox_max)

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
