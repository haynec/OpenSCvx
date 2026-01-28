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
        lam_vb: float

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

from .autotuning import (
    AugmentedLagrangian,
    AutotuningBase,
    ConstantProximalWeight,
    RampProximalWeight,
)
from .base import Algorithm, AlgorithmState, DiscretizationResult
from .optimization_results import OptimizationResults
from .penalized_trust_region import PenalizedTrustRegion

__all__ = [
    # Base class
    "Algorithm",
    "AlgorithmState",
    "DiscretizationResult",
    # Core results
    "OptimizationResults",
    # PTR algorithm
    "PenalizedTrustRegion",
    "AutotuningBase",
    "AugmentedLagrangian",
    "ConstantProximalWeight",
    "RampProximalWeight",
]
