# Lowering Architecture

This document explains how OpenSCvx converts symbolic problem definitions into executable code for optimization.

## The Big Picture

OpenSCvx separates trajectory optimization into four phases:

1. **Preprocessing** — Validate inputs, augment dynamics, categorize constraints
2. **Lowering** — Convert symbolic expressions to JAX/CVXPy code
3. **Solving** — Run the SCP (Sequential Convex Programming) loop
4. **Post-processing** — Propagate results, compute metrics

This document focuses on **Phase 2: Lowering**.

## Why Lowering?

When you define a problem in OpenSCvx, you write symbolic expressions:

```python
position = ox.State("pos", shape=(3,))
velocity = ox.State("vel", shape=(3,))
thrust = ox.Control("thrust", shape=(3,))

dynamics = {"pos": velocity, "vel": thrust / mass - gravity}
constraints = [ox.Norm(thrust) <= max_thrust]
```

These symbolic expressions form an AST (Abstract Syntax Tree). The **lowering** phase walks this AST and generates:

- **JAX functions** for dynamics and non-convex constraints (with automatic differentiation for Jacobians)
- **CVXPy expressions** for convex constraints (used directly in the convex subproblem)

## Pipeline Overview

```
SymbolicProblem                         LoweredProblem
(AST representation)                    (executable code)
        │                                      │
        │     lower_symbolic_problem()         │
        ├──────────────────────────────────────┤
        │                                      │
   dynamics  ──────►  JAX: f(x,u) → dx/dt      │
   (symbolic)         JAX: A = df/dx           │
                      JAX: B = df/du           │
        │                                      │
   non-convex  ────►  JAX: g(x,u) → residual   │
   constraints        JAX: ∇g_x, ∇g_u          │
        │                                      │
   convex      ────►  CVXPy constraint objects │
   constraints        (added directly to OCP)  │
        │                                      │
   states,     ────►  UnifiedState/Control     │
   controls           (aggregated vectors)     │
        │                                      │
        └──────────────────────────────────────┘
```

## Key Data Structures

### Input: `SymbolicProblem`

After preprocessing, all symbolic definitions are collected into a `SymbolicProblem`:

```python
@dataclass
class SymbolicProblem:
    dynamics: Expr              # Symbolic dx/dt = f(x, u)
    states: List[State]         # All state variables (including augmented)
    controls: List[Control]     # All control variables (including virtual)
    constraints: ConstraintSet  # Categorized constraints
    parameters: dict            # User-defined parameters
    N: int                      # Discretization nodes
    # ... plus propagation variants
```

### Output: `LoweredProblem`

Lowering produces a `LoweredProblem` containing everything needed for optimization:

```python
@dataclass
class LoweredProblem:
    # JAX dynamics (callable functions with Jacobians)
    dynamics: Dynamics           # f, A=df/dx, B=df/du
    dynamics_prop: Dynamics      # For forward propagation

    # Lowered constraints (by backend)
    jax_constraints: LoweredJaxConstraints
    cvxpy_constraints: LoweredCvxpyConstraints

    # Unified state/control interfaces
    x_unified: UnifiedState      # Aggregates all states into one vector
    u_unified: UnifiedControl    # Aggregates all controls into one vector

    # CVXPy optimization variables
    ocp_vars: OCPVariables       # x, u, dx, du, nu, etc.
    cvxpy_params: dict           # User parameters as cp.Parameter
```

## Constraint Routing

Constraints take different paths based on convexity:

| Constraint Type | Backend | How It's Used in SCP |
|-----------------|---------|----------------------|
| Non-convex nodal | JAX | Linearized at each iteration: `g(x̄) + ∇g·δx ≤ 0` |
| Non-convex cross-node | JAX | Same, but references multiple trajectory nodes |
| Convex nodal | CVXPy | Added directly to QP subproblem |
| Convex cross-node | CVXPy | Same, with `NodeReference` indexing |
| CTCS | JAX | Continuous-time via augmented dynamics |

### JAX-Lowered Constraints

Non-convex constraints become JAX functions with gradients:

```python
@dataclass
class LoweredJaxConstraints:
    nodal: List[LoweredNodalConstraint]      # func, grad_g_x, grad_g_u
    cross_node: List[LoweredCrossNodeConstraint]  # func, grad_g_X, grad_g_U
    ctcs: List[CTCS]                         # Handled via dynamics augmentation
```

### CVXPy-Lowered Constraints

Convex constraints become CVXPy constraint objects:

```python
@dataclass
class LoweredCvxpyConstraints:
    constraints: List[cp.Constraint]  # Added directly to OCP
```

## Design Principles

### Immutability

Lowering never mutates inputs. The original `SymbolicProblem` remains unchanged; a new `LoweredProblem` is returned. This enables:

- Inspecting symbolic expressions after lowering
- Reusing the same symbolic problem for multiple configurations
- Easier debugging and testing

### Type Separation

Symbolic and lowered representations use distinct types:

- `NodalConstraint` (symbolic) vs `LoweredNodalConstraint` (JAX functions)
- `ConstraintSet` (symbolic) vs `LoweredJaxConstraints` / `LoweredCvxpyConstraints`

This prevents accidentally mixing AST nodes with executable code.

### Backend Independence

JAX lowering has no dependency on:
- `N` (number of nodes)
- Scaling matrices
- CVXPy

This means JAX-lowered dynamics and constraints could be used with alternative solvers.

## Convex Subproblem Backends

The convex subproblem at each SCP iteration is solved by a concrete subclass
of the abstract `PTRSolver` base. Three backends ship today:

| Backend | Class | Selector | Notes |
|---------|-------|----------|-------|
| CVXPy (default) | `CVXPyPTRSolver` | `solver={"backend": "cvxpy"}` (default) | DCP graph via CVXPy, dispatched to QOCO / CLARABEL / etc. Supports user `.convex()` constraints, cross-node constraints, CTCS, and impulsive controls. Optional cvxpygen code generation. |
| QPAX | `QPAXPTRSolver` | `solver={"backend": "qpax"}` | JAX-native QP via `qpax.solve_qp`. Flat `(Q, q, A, b, G, h)` assembly. Supports box / dynamics / CTCS / boundary-Fix; **rejects** user `.convex()`, cross-node, and impulsive at `initialize()` with a clear "use `CVXPyPTRSolver`" message. Enables a path toward an end-to-end JAX-differentiable SCP loop in follow-up work. |
| Moreau | `MoreauPTRSolver` | `solver={"backend": "moreau"}` | JAX-native conic solver (`moreau.jax.Solver`). Sparse CSR assembly; SOC epigraphs for the L1 / pos PTR penalties (fewer variables and rows than QPAX). Warm-starts between SCP iterations. Same supported subset as QPAX. Paves the way for user `.convex()` SOC support in a follow-up. |

Picking a backend at construction time:

```python
import openscvx as ox

# Default — same as today's CVXPyPTRSolver.
problem = ox.Problem(...)

# Explicit CVXPy with a different inner solver.
problem = ox.Problem(..., solver={"backend": "cvxpy", "cvx_solver": "CLARABEL"})

# JAX-native QPAX (requires the `qpax` extra: pip install openscvx[qpax]).
problem = ox.Problem(..., solver={"backend": "qpax"})

# JAX-native Moreau conic (requires the `moreau` extra: pip install openscvx[moreau]).
problem = ox.Problem(..., solver={"backend": "moreau"})
```

Both JAX backends consume the global JAX dtype — pass `float_dtype="float64"`
to `Problem` if you need tight inner-solver tolerances; the default `float32`
is enough for
many problems but caps the QP's conditioning.

## Further Reading

- `openscvx/symbolic/lower.py` — Main lowering implementation
- `openscvx/lowered/` — Dataclass definitions
- `openscvx/symbolic/lowerers/jax.py` — JAX visitor implementation
- `openscvx/symbolic/lowerers/cvxpy.py` — CVXPy visitor implementation
- `openscvx/solvers/ptr_solver.py` — Abstract `PTRSolver` contract
- `openscvx/solvers/cvxpy_ptr_solver.py` / `qpax_ptr_solver.py` — Concrete backends
