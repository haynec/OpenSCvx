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

import inspect
from typing import Any, Dict

from .AugmentedLagrangian import AugmentedLagrangian
from .base import Algorithm, AlgorithmState, AutotuningBase, DiscretizationResult, Weights
from .ConstantProximalWeight import ConstantProximalWeight
from .optimization_results import OptimizationResults
from .penalized_trust_region import PenalizedTrustRegion
from .RampProximalWeight import RampProximalWeight

# ---------------------------------------------------------------------------
# Spec resolvers — turn dicts/strings into algorithm/autotuner instances
# ---------------------------------------------------------------------------

_AUTOTUNER_MAP: Dict[str, type] = {}


def _resolve_autotuner(val: Any) -> Any:
    """Resolve an autotuner specification into an instance.

    Accepted forms:

    * **string** — class name only, default parameters::

          "RampProximalWeight"

    * **dict** — class name + parameter overrides::

          {"type": "RampProximalWeight", "ramp_factor": 1.04}

    * **instance** — already-constructed autotuner (pass-through).
    """
    if not isinstance(val, (str, dict)):
        return val

    if isinstance(val, str):
        name = val
        params: dict = {}
    else:
        params = dict(val)  # copy to avoid mutating the input
        name = params.pop("type", None)
        if name is None:
            raise ValueError(
                "autotuner dict must include a 'type' key (e.g. type: RampProximalWeight)"
            )

    if not _AUTOTUNER_MAP:
        for cls in (AugmentedLagrangian, ConstantProximalWeight, RampProximalWeight):
            _AUTOTUNER_MAP[cls.__name__] = cls

    cls = _AUTOTUNER_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown autotuner {name!r}; expected one of {sorted(_AUTOTUNER_MAP)}")

    instance = cls()
    for key, value in params.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown autotuner parameter {key!r} for {name}")
        setattr(instance, key, value)
    return instance


def _resolve_algorithm(kwargs: dict) -> "PenalizedTrustRegion":
    """Build a :class:`PenalizedTrustRegion` from a user-supplied dict.

    Supports a nested ``autotuner`` key that is resolved via
    :func:`_resolve_autotuner` (string, dict, or instance).
    """
    kwargs = dict(kwargs)  # copy to avoid mutating the caller's dict

    # Resolve nested autotuner spec if present
    if "autotuner" in kwargs:
        kwargs["autotuner"] = _resolve_autotuner(kwargs["autotuner"])

    try:
        return PenalizedTrustRegion(**kwargs)
    except TypeError as e:
        valid = list(inspect.signature(PenalizedTrustRegion.__init__).parameters.keys())
        valid.remove("self")
        raise TypeError(f"Invalid algorithm keyword argument: {e}. Valid keys: {valid}") from None


__all__ = [
    # Base class
    "Algorithm",
    "AlgorithmState",
    "DiscretizationResult",
    "Weights",
    # Core results
    "OptimizationResults",
    # PTR algorithm
    "PenalizedTrustRegion",
    "AutotuningBase",
    "AugmentedLagrangian",
    "ConstantProximalWeight",
    "RampProximalWeight",
]
