"""Successive convexification algorithms for trajectory optimization.

This module provides implementations of SCvx (Successive Convexification) algorithms
for solving non-convex trajectory optimization problems through iterative convex
approximation.

All algorithms inherit from :class:`Algorithm`, enabling pluggable algorithm
implementations and custom SCvx variants. Immutable components (solver,
discretization_solver, jax_constraints, etc.) are stored during ``initialize()``;
mutable configuration (params, settings) is passed per-step.

The iterate carry is split into two objects:

* :class:`AlgorithmState` — a frozen, JAX-registered pytree holding only the
  current iterate (``x``, ``u``, weights, propagated states, diagnostic scalars).
  Every leaf is a ``jnp.ndarray`` so the state composes with ``jax.vmap`` /
  ``jax.jit`` / ``jax.grad``.

* :class:`AlgorithmHistory` — a CPU-side mutable record of every iteration's
  trajectories, discretizations, weights, and diagnostics. Grown by the SCP loop
  via ``record_iteration``; never crosses the JAX boundary.

Autotuners follow the same contract: :class:`AutotuningBase` declares
``update_weights`` as a pure functional update on the :class:`AlgorithmState`
pytree (no mutation, no string returns, no list appends, no Python-level
branching on iterate values). The next-iterate adaptive state is reported via
the :class:`AdaptiveStateCode` IntEnum and converted to a human-readable string
on the printing path with :func:`adaptive_state_code_to_str`.

Current Implementations:

- :class:`PenalizedTrustRegion`: Penalized Trust Region (PTR) algorithm
"""

from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .autotuner import (
    AcceptanceRatioAutotuner,
    AcceptanceRatioHyper,
    AdaptiveProximalWeight,
    AdaptiveProximalWeightSpec,
    AugmentedLagrangian,
    AugmentedLagrangianSpec,
    ConstantProximalWeight,
    ConstantProximalWeightSpec,
    RampProximalWeight,
    RampProximalWeightSpec,
)
from .base import (
    AdaptiveStateCode,
    Algorithm,
    AlgorithmHistory,
    AlgorithmState,
    AutotuningBase,
    DiscretizationResult,
    HyperParams,
    adaptive_state_code_to_str,
)
from .optimization_results import OptimizationResults
from .scvx import PenalizedTrustRegion
from .weights import Weights

# ---------------------------------------------------------------------------
# Autotuner config — discriminated union of each autotuner's Spec
# ---------------------------------------------------------------------------

AutotunerConfig = Annotated[
    Union[
        AugmentedLagrangianSpec,
        AdaptiveProximalWeightSpec,
        RampProximalWeightSpec,
        ConstantProximalWeightSpec,
    ],
    Field(discriminator="type"),
]

# ---------------------------------------------------------------------------
# Algorithm config model
# ---------------------------------------------------------------------------


class PenalizedTrustRegionConfig(BaseModel):
    """Validates PenalizedTrustRegion configuration from dict input.

    The ``autotuner`` field accepts:

    * ``None`` — defaults to :class:`AugmentedLagrangian`.
    * A **string** — class name only, default parameters.
    * A **dict** — class name via ``"type"`` key plus overrides.
    * An **instance** — already-constructed autotuner (pass-through).
    """

    autotuner: Optional[Union[AutotunerConfig, AutotuningBase]] = None

    @field_validator("autotuner", mode="before")
    @classmethod
    def _wrap_bare_string(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"type": v}
        return v

    k_max: int = 200
    t_max: Optional[float] = None
    lam_prox: Union[float, Dict[str, Any]] = 1e-1
    lam_vc: Union[float, Dict[str, Any]] = 1e0
    lam_cost: Union[float, Dict[str, Any]] = 1e-2
    lam_vb: float = 0.0
    ep_tr: float = 1e-4
    ep_vb: float = 1e-4
    ep_vc: float = 1e-8

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def to_algorithm(
        self,
        states: Optional[List[Any]] = None,
        controls: Optional[List[Any]] = None,
    ) -> PenalizedTrustRegion:
        at = self.autotuner
        if at is None:
            autotuner = None
        elif isinstance(at, AutotuningBase):
            autotuner = at
        else:
            autotuner = at.build()
        kwargs = self.model_dump(exclude={"autotuner"}, exclude_unset=True)
        return PenalizedTrustRegion(
            autotuner=autotuner,
            states=states,
            controls=controls,
            **kwargs,
        )


__all__ = [
    # Base class
    "Algorithm",
    "AlgorithmHistory",
    "AlgorithmState",
    "AdaptiveStateCode",
    "adaptive_state_code_to_str",
    "DiscretizationResult",
    "Weights",
    # Core results
    "OptimizationResults",
    # PTR algorithm
    "PenalizedTrustRegion",
    "AutotuningBase",
    "HyperParams",
    "AcceptanceRatioAutotuner",
    "AcceptanceRatioHyper",
    "AugmentedLagrangian",
    "AdaptiveProximalWeight",
    "ConstantProximalWeight",
    "RampProximalWeight",
    # Config models
    "PenalizedTrustRegionConfig",
    "AutotunerConfig",
]
