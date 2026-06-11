"""Constant proximal-weight autotuner.

Keeps ``lam_prox`` fixed across iterations while still relaxing ``lam_cost``
after the configured ``lam_cost_drop`` iteration.
"""

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict

from openscvx.config import Config

from ..base import AdaptiveStateCode, AutotuningBase, HyperParams

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

    from ..base import AlgorithmState, CandidateIterate


class ConstantProximalWeightHyper(HyperParams):
    """Declared hyperparameters for :class:`ConstantProximalWeight`."""

    lam_cost_drop: int = -1


class ConstantProximalWeight(AutotuningBase):
    """Hold ``lam_prox`` constant; relax ``lam_cost`` after ``lam_cost_drop``.

    Useful when you want a fixed trust-region size without adaptation.

    ``update_weights`` is a pure functional update on the
    :class:`AlgorithmState` pytree; see the base-class contract.
    """

    # The body is three jnp ops — JAX's eager dispatch is cheaper than the
    # pytree-flatten overhead of a JIT'd closure. Opt out of the SCP loop's
    # JIT wrapping.
    JIT_UPDATE_WEIGHTS: bool = False

    def __init__(
        self,
        lam_cost_drop: int = -1,
        lam_cost_relax: float = 1.0,
    ):
        self.hyper = ConstantProximalWeightHyper(lam_cost_drop=lam_cost_drop)
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

        return state.replace(
            x=candidate.x,
            u=candidate.u,
            x_prop=candidate.x_prop,
            x_prop_plus=candidate.x_prop_plus,
            lam_cost=lam_cost_next,
            adaptive_state_code=jnp.asarray(
                int(AdaptiveStateCode.ACCEPT_CONSTANT), dtype=jnp.int32
            ),
        )


# =============================================================================
# Pydantic spec for dict / YAML validation
# =============================================================================


class ConstantProximalWeightSpec(BaseModel):
    """Validates ConstantProximalWeight configuration from dict/YAML input."""

    type: Literal["ConstantProximalWeight"] = "ConstantProximalWeight"
    lam_cost_drop: int = -1
    lam_cost_relax: float = 1.0

    model_config = ConfigDict(extra="forbid")

    def build(self) -> ConstantProximalWeight:
        return ConstantProximalWeight(**self.model_dump(exclude={"type"}, exclude_unset=True))
