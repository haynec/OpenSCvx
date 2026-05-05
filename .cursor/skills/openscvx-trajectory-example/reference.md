# OpenSCvx problem setup — reference

## Default config quirks

- **`discretizer` omitted**: `resolve_discretizer_config({})` injects **`type: "VectorizeDiscretizeLinearize"`** (see `openscvx/discretization/__init__.py`). This may differ from older prose that mentions only `LinearizeDiscretize`; trust the resolver + `DiscretizerSpec`.
- **`algorithm` omitted**: `PenalizedTrustRegion` with **`AugmentedLagrangian`** autotuner by default (`PenalizedTrustRegionConfig.to_algorithm`).
- **`solver` omitted**: `PTRSolver` with **`cvx_solver="QOCO"`** unless overridden (`SolverSpec`).

## Discretizer `type` field (YAML / dict)

| `type` | Role (high level) |
|--------|---------------------|
| `VectorizeDiscretizeLinearize` | Default vectorized linearize-then-discretize path. |
| `DiscretizeLinearizeVectorize` | Alternate order tradeoff (`DiscretizerSpec` docstring). |
| `LinearizeDiscretize` | Dense linearize-then-discretize (`LinearizeDiscretize` class docstring: FOH/ZOH, Diffrax ODE). |
| `LinearizeDiscretizeSparse` | Sparse Jacobian path when sparsity patterns exist (`linearize_discretize_sparse.py`). |

Shared knobs: **`dis_type`** (`"FOH"` / `"ZOH"` or per-control list), **`ode_solver`**, **`diffrax_kwargs`**, plus **`custom_integrator`** / **`args`** for the vectorized pair (`DiscretizerSpec`).

## Autotuner classes (symbol names)

Import from **`openscvx`** / **`openscvx.algorithms`**:

- `AugmentedLagrangian` (default)
- `RampProximalWeight`
- `ConstantProximalWeight`

Pass as `algorithm={"autotuner": "RampProximalWeight"}` or nested `{"type": "...", ...}`.

## Minimal symbolic skeleton

```python
import openscvx as ox
from openscvx import Problem

# states, controls, constraints, dynamics, time, N = ...

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    constraints=constraints,
    N=N,
    time=time,
    algorithm={...},   # optional
    discretizer={...}, # optional
    solver={...},      # optional
    byof=None,         # optional
    float_dtype="float64",
)
```

## Source files for full docstrings

| Topic | Module |
|-------|--------|
| `Problem` | `openscvx/problem.py` |
| `State` / helpers | `openscvx/symbolic/expr/state.py` |
| `Control` | `openscvx/symbolic/expr/control.py` |
| `Time` | `openscvx/symbolic/expr/time.py` |
| `Parameter` | `openscvx/symbolic/expr/parameter.py` |
| Constraints / CTCS | `openscvx/symbolic/expr/constraint.py` |
| STL | `openscvx/symbolic/expr/stl.py` |
| Hard logic | `openscvx/symbolic/expr/logic.py` |
| BYOF | `openscvx/expert/byof.py` |
| PTR algorithm | `openscvx/algorithms/penalized_trust_region.py` |
| Algorithm config | `openscvx/algorithms/__init__.py` (`PenalizedTrustRegionConfig`) |
| Discretizer spec | `openscvx/discretization/base.py`, `__init__.py` |
| Convex solver | `openscvx/solvers/ptr_solver.py`, `solvers/base.py` |
| Loader | `openscvx/loader.py` |
| MJX | `openscvx/integrations/mjx.py`, `integrations/__init__.py` |
| Init guesses | `openscvx/init/__init__.py` |
| Alternate STL backend | `openscvx/symbolic/expr/stljax.py` (`ox.stljax`) |
