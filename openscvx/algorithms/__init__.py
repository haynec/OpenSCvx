"""Successive convexification algorithms for trajectory optimization.

This module provides implementations of SCvx (Successive Convexification) algorithms
for solving non-convex trajectory optimization problems through iterative convex
approximation.

All algorithms inherit from :class:`Algorithm`, enabling pluggable algorithm
implementations and custom SCvx variants:

```python
class Algorithm(ABC):
    @abstractmethod
    def initialize(self, solver, discretization_solver, jax_constraints,
                   emitter, params, settings) -> None:
        '''Store compiled infrastructure and warm-start solvers.'''
        ...

    @abstractmethod
    def step(self, state, params, settings) -> bool:
        '''Execute one iteration using stored infrastructure.'''
        ...
```

Immutable components (solver, discretization_solver, jax_constraints, etc.) are stored
during ``initialize()``. Mutable configuration (params, settings) is passed per-step
to support runtime parameter updates and tolerance tuning.

:class:`AlgorithmState` holds mutable state during SCP iterations. Algorithms
that require additional state can subclass it:

```python
@dataclass
class MyAlgorithmState(AlgorithmState):
    my_custom_field: float = 0.0
```

Note:
    ``AlgorithmState`` currently combines iteration metrics (costs, weights),
    trajectory history, and discretization data. A future refactor may separate
    these concerns into distinct classes for clearer data flow:

    ```python
    @dataclass
    class AlgorithmState:
        # Mutable iteration state
        k: int
        J_tr: float
        J_vb: float
        J_vc: float
        lam_prox: float
        lam_cost: float
        lam_vc: ...
        lam_vb_nodal: np.ndarray  # (N, n_nodal)
        lam_vb_cross: np.ndarray  # (n_cross,)

    @dataclass
    class TrajectoryHistory:
        # Accumulated trajectory solutions
        X: List[np.ndarray]
        U: List[np.ndarray]

        @property
        def x(self): return self.X[-1]

        @property
        def u(self): return self.U[-1]

    @dataclass
    class DebugHistory:
        # Optional diagnostic data (discretization matrices, etc.)
        V_history: List[np.ndarray]
        VC_history: List[np.ndarray]
        TR_history: List[np.ndarray]
    ```

Current Implementations:

- :class:`PenalizedTrustRegion`: Penalized Trust Region (PTR) algorithm
"""

from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .autotuner import (
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
    "AugmentedLagrangian",
    "AdaptiveProximalWeight",
    "ConstantProximalWeight",
    "RampProximalWeight",
    # Config models
    "PenalizedTrustRegionConfig",
    "AutotunerConfig",
]
