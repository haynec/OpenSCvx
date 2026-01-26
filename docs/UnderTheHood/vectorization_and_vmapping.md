# Vectorization and Vmapping Across Decision Nodes

This page explains how OpenSCvx internally processes symbolic problem definitions into vectorized JAX computations. After symbolic preprocessing and augmentation (which you've likely seen in basic usage), the library creates unified state/control vectors and applies JAX's `vmap` to evaluate dynamics and non-convex constraints across decision nodes in parallel.

## Processing Pipeline

The transformation from symbolic expressions to vectorized execution happens in several stages:

1. **Symbolic Preprocessing**: Augmentation with time state, CTCS states, and time dilation (covered in basic usage)
2. **Unification**: Individual State/Control objects combined into monolithic vectors
3. **JAX Lowering**: Symbolic expressions compiled to executable JAX functions (dynamics and non-convex constraints)
4. **Vectorization**: `vmap` applied to batch dynamics and constraint computations across decision nodes

Understanding this pipeline is useful for performance optimization, debugging shape mismatches, and extending the library.

## Stage 1: Symbolic Problem Definition

Starting from a typical problem definition with individual states and controls:

```python
import openscvx as ox
import numpy as np

# Individual state components
position = ox.State("position", shape=(2,))
velocity = ox.State("velocity", shape=(1,))

# Control
theta = ox.Control("theta", shape=(1,))

# Dynamics per state
dynamics = {
    "position": ox.Concat(velocity[0] * ox.Sin(theta[0]), -velocity[0] * ox.Cos(theta[0])),
    "velocity": 9.81 * ox.Cos(theta[0]),
}
```

At this stage, each state/control is independent with its own shape, and dynamics are symbolic expressions without any notion of batching or decision nodes.

## Stage 2: Symbolic Preprocessing and Augmentation

During `Problem` construction (in `preprocess_symbolic_problem`), the symbolic problem is augmented:

```python
problem = Problem(
    dynamics=dynamics,
    states=[position, velocity],
    controls=[theta],
    N=10,
    time=ox.Time(initial=0.0, final=2.0),
)
```

Internally, additional states and controls are added:
- Time state (if not user-provided)
- CTCS augmented states for path constraints
- Time dilation control for time-optimal problems

After augmentation: `states_aug = [position, velocity, time, ...]` and `controls_aug = [theta, _time_dilation]`, with corresponding dynamics for all augmented states.

## Stage 3: Unification

The augmented states and controls are combined into unified vectors (in `lower_symbolic_expressions`):

```python
x_unified: UnifiedState = unify_states(states_aug)
u_unified: UnifiedControl = unify_controls(controls_aug)
```

The unification process (in `openscvx/symbolic/unified.py`) sorts variables (user-defined first, then augmented), concatenates properties (bounds, guesses, etc.), and assigns each State/Control a slice for indexing into the unified vector.

### Unified Vector Shapes

For a problem with `N` decision nodes:

```python
x_unified.shape = (n_x,)          # Sum of all state dimensions
u_unified.shape = (n_u,)          # Sum of all control dimensions
x_unified.guess.shape = (N, n_x)  # State trajectory
u_unified.guess.shape = (N, n_u)  # Control trajectory
```

**Concrete example** (brachistochrone with N=10, no CTCS constraints):
```python
x_unified.shape = (4,)        # position(2) + velocity(1) + time(1)
u_unified.shape = (2,)        # theta(1) + _time_dilation(1)
x_unified.guess.shape = (10, 4)
u_unified.guess.shape = (10, 2)
```

Each original State/Control retains a slice for extraction:
```python
position._slice = slice(0, 2)
velocity._slice = slice(2, 3)
time._slice = slice(3, 4)

# Extract during evaluation:
position_value = x_unified[position._slice]  # (2,)
```

## Stage 4: JAX Lowering

Symbolic expressions for dynamics and non-convex constraints are converted to executable JAX functions (in `openscvx/symbolic/lower.py`). Convex constraints are lowered to CVXPy separately.

### Dynamics Lowering

```python
# Convert symbolic dynamics to JAX function
dyn_fn = lower_to_jax(dynamics_aug)

# Create Dynamics object with Jacobians
dynamics_augmented = Dynamics(
    f=dyn_fn,                      # State derivative function
    A=jacfwd(dyn_fn, argnums=0),   # Jacobian df/dx
    B=jacfwd(dyn_fn, argnums=1),   # Jacobian df/du
)
```

**Dynamics Function Signature (Before Vmap):**

```python
def f(x: Array, u: Array, node: int, params: dict) -> Array:
    """Compute state derivative at a single decision node.

    Args:
        x: State vector at this node, shape (n_x,)
        u: Control vector at this node, shape (n_u,)
        node: Node index (0 to N-1), used for time-varying behavior
        params: Dictionary of problem parameters

    Returns:
        State derivative dx/dt, shape (n_x,)
    """
    ...
```

Jacobians have similar signatures:

```python
A(x, u, node, params) -> Array[n_x, n_x]  # df/dx
B(x, u, node, params) -> Array[n_x, n_u]  # df/du
```

### Non-Convex Constraint Lowering

Non-convex nodal constraints that are to be lowered to JAX follow the same pattern:

```python
# Convert symbolic constraint expressions to JAX functions
constraints_nodal_fns = lower_to_jax(constraints_nodal)

# Create LoweredNodalConstraint objects with Jacobians
for i, fn in enumerate(constraints_nodal_fns):
    constraint = LoweredNodalConstraint(
        func=fn,                          # Constraint function
        grad_g_x=jacfwd(fn, argnums=0),  # Jacobian dg/dx
        grad_g_u=jacfwd(fn, argnums=1),  # Jacobian dg/du
        nodes=constraints_nodal[i].nodes, # Node indices where constraint applies
    )
```

**Constraint Function Signature (Before Vmap):**

```python
def g(x: Array, u: Array, node: int, params: dict) -> float:
    """Evaluate constraint at a single decision node.

    Args:
        x: State vector at this node, shape (n_x,)
        u: Control vector at this node, shape (n_u,)
        node: Node index, used for time-varying constraints
        params: Dictionary of problem parameters

    Returns:
        Constraint value (scalar)
    """
    ...
```

Constraint Jacobians:

```python
grad_g_x(x, u, node, params) -> Array[n_x]  # dg/dx
grad_g_u(x, u, node, params) -> Array[n_u]  # dg/du
```

### Cross-Node Constraint Lowering

Cross-node constraints couple variables across multiple trajectory nodes. Unlike regular nodal constraints that evaluate at single nodes, cross-node constraints require access to the full trajectory.

**Key Signature Difference:**

| Constraint Type | Signature | Vectorization |
|-----------------|-----------|---------------|
| Regular Nodal | `(x, u, node, params) → scalar` | vmapped across nodes |
| Cross-Node | `(X, U, params) → scalar` | operates on full trajectory |

Cross-node constraints are represented by the `CrossNodeConstraint` expression type and lowered via a dedicated visitor (`JaxLowerer._visit_cross_node_constraint` in `openscvx/symbolic/lowerers/jax.py`). The visitor wraps the inner constraint to provide the trajectory-level signature.

**Cross-Node Constraint Function Signature:**

```python
def g_cross(X: Array, U: Array, params: dict) -> scalar:
    """Evaluate single cross-node constraint.

    Args:
        X: Full state trajectory, shape (N, n_x)
        U: Full control trajectory, shape (N, n_u)
        params: Dictionary of problem parameters

    Returns:
        Scalar constraint residual
    """
    ...
```

**Cross-Node Constraint Jacobians:**

```python
grad_g_X(X, U, params) -> Array[N, n_x]  # dg/dX - Jacobian wrt all states
grad_g_U(X, U, params) -> Array[N, n_u]  # dg/dU - Jacobian wrt all controls
```

**Jacobian Sparsity:** These Jacobians are stored as dense `(N, n_x)` and `(N, n_u)` arrays but are typically very sparse. A constraint coupling nodes `k` and `k-1` only has non-zero derivatives at rows `k` and `k-1`; all other rows are zero.

## Stage 5: Vectorization with Vmap

Finally, both dynamics and constraints are vectorized to operate on decision nodes simultaneously. This enables efficient parallel evaluation on GPU/TPU hardware.

### Dynamics Vectorization

Dynamics functions are vmapped to process all intervals in parallel (in `Problem.initialize`):

```python
# Vectorize dynamics functions across decision nodes
self.dynamics_augmented.f = jax.vmap(
    self.dynamics_augmented.f,
    in_axes=(0, 0, 0, None)
)
self.dynamics_augmented.A = jax.vmap(
    self.dynamics_augmented.A,
    in_axes=(0, 0, 0, None)
)
self.dynamics_augmented.B = jax.vmap(
    self.dynamics_augmented.B,
    in_axes=(0, 0, 0, None)
)
```

**Dynamics Vmap Configuration: `in_axes=(0, 0, 0, None)`**

This means:
- **Axis 0 of x**: Batch over states at different intervals
- **Axis 0 of u**: Batch over controls at different intervals
- **Axis 0 of node**: Batch over node indices
- **None for params**: Shared parameters (not batched)

**Dynamics Signature (After Vmap):**

```python
def f_vmapped(x_batch: Array, u_batch: Array, nodes: Array, params: dict) -> Array:
    """Compute state derivatives at all intervals simultaneously.

    Args:
        x_batch: States at interval starts, shape (N-1, n_x)
        u_batch: Controls at interval starts, shape (N-1, n_u)
        nodes: Node indices, shape (N-1,) - typically jnp.arange(0, N-1)
        params: Dictionary of problem parameters (shared across all nodes)

    Returns:
        State derivatives at all intervals, shape (N-1, n_x)
    """
    ...
```

Jacobians after vmap:

```python
A_vmapped(x_batch, u_batch, nodes, params) -> Array[N-1, n_x, n_x]
B_vmapped(x_batch, u_batch, nodes, params) -> Array[N-1, n_x, n_u]
```

**Why N-1 instead of N?**

Trajectory discretization operates on **intervals** between consecutive decision nodes:
- **N decision nodes**: Including initial and final states (e.g., nodes 0, 1, 2, ..., 9 for N=10)
- **N-1 intervals**: Between consecutive nodes (e.g., intervals [0→1], [1→2], ..., [8→9] for N=10)
- **Dynamics evaluation**: At the start of each interval, giving N-1 evaluations

This is why vmapped dynamics process batches of size `(N-1, ...)` rather than `(N, ...)`.

### Constraint Vectorization

Non-convex nodal constraints are also vectorized, but with a key difference (in `lower_symbolic_expressions`):

```python
# Vectorize constraint functions (during JAX lowering)
constraint = LoweredNodalConstraint(
    func=jax.vmap(fn, in_axes=(0, 0, None, None)),
    grad_g_x=jax.vmap(jacfwd(fn, argnums=0), in_axes=(0, 0, None, None)),
    grad_g_u=jax.vmap(jacfwd(fn, argnums=1), in_axes=(0, 0, None, None)),
    nodes=constraint.nodes,  # List of specific node indices where constraint applies
)
```

**Constraint Vmap Configuration: `in_axes=(0, 0, None, None)`**

Note the key difference from dynamics:
- **Axis 0 of x**: Batch over states
- **Axis 0 of u**: Batch over controls
- **None for node**: Node index is **not batched** (same value for all evaluations in a batch)
- **None for params**: Shared parameters (not batched)

**Why the difference?** Constraints are only evaluated at specific nodes (e.g., a collision avoidance constraint might only apply at nodes [2, 5, 7]). The constraint is vmapped to handle multiple constraint evaluations in parallel, but each evaluation receives the same `node` value since it's evaluating the same logical constraint at potentially different states/controls.

**Constraint Signature (After Vmap):**

```python
def g_vmapped(x_batch: Array, u_batch: Array, node: int, params: dict) -> Array:
    """Evaluate constraint at multiple state/control pairs simultaneously.

    Args:
        x_batch: State vectors, shape (batch_size, n_x)
        u_batch: Control vectors, shape (batch_size, n_u)
        node: Single node index (broadcast to all evaluations)
        params: Dictionary of problem parameters (shared across all evaluations)

    Returns:
        Constraint values, shape (batch_size,)
    """
    ...
```

Constraint Jacobians after vmap:

```python
grad_g_x_vmapped(x_batch, u_batch, node, params) -> Array[batch_size, n_x]
grad_g_u_vmapped(x_batch, u_batch, node, params) -> Array[batch_size, n_u]
```

When constraints are evaluated in practice:

```python
# Extract states/controls at nodes where constraint applies
x_batch = x[constraint.nodes]  # Shape: (len(nodes), n_x)
u_batch = u[constraint.nodes]  # Shape: (len(nodes), n_u)

# Evaluate constraint at all specified nodes
g_values = constraint.func(x_batch, u_batch, node_idx, params)  # Shape: (len(nodes),)
```

### Cross-Node Constraint Vectorization

Cross-node constraints are **not vmapped** because they already operate on full trajectory arrays. Each `CrossNodeConstraint` is a single constraint with fixed node indices baked into the expression via `NodeReference` nodes.

**Key Difference from Regular Constraints:**

| Aspect | Regular Nodal Constraints | Cross-Node Constraints |
|--------|--------------------------|------------------------|
| **Input Shape** | Single-node vectors `(n_x,)`, `(n_u,)` | Full trajectories `(N, n_x)`, `(N, n_u)` |
| **Vectorization** | `jax.vmap` with `in_axes=(0, 0, None, None)` | No vmap (already trajectory-level) |
| **Output** | Scalar per evaluation | Scalar per constraint |
| **Jacobian Shape** | `(n_x,)`, `(n_u,)` per node | `(N, n_x)`, `(N, n_u)` per constraint |

**Evaluation:** During SCP iterations, each cross-node constraint receives the full trajectory arrays and returns a scalar residual:

```python
# Each LoweredCrossNodeConstraint operates on full trajectories
residual = constraint.func(X, U, params)      # scalar
grad_X = constraint.grad_g_X(X, U, params)    # (N, n_x) - sparse, mostly zeros
grad_U = constraint.grad_g_U(X, U, params)    # (N, n_u) - sparse, mostly zeros
```

The Jacobians are dense arrays but exhibit sparsity patterns determined by which nodes the constraint couples.

## Usage in Discretization

The vmapped dynamics functions are called during discretization (in `calculate_discretization`):

```python
# Setup batch inputs
x = V[:, :n_x]                          # Shape: (N-1, n_x) - States at interval starts
u = u[: x.shape[0]]                     # Shape: (N-1, n_u) - Controls (includes time dilation)
nodes = jnp.arange(0, N-1)              # Shape: (N-1,) - Node indices

# Extract time dilation (last control dimension)
s = u[:, -1]                            # Shape: (N-1,) - Time dilation values

# Call vmapped dynamics - evaluates all intervals in parallel
# Note: dynamics receive u[:, :-1] (vehicle controls only, excluding time dilation)
f = state_dot(x, u[:, :-1], nodes, params)  # Shape: (N-1, n_x)
dfdx = A(x, u[:, :-1], nodes, params)       # Shape: (N-1, n_x, n_x)
dfdu_veh = B(x, u[:, :-1], nodes, params)   # Shape: (N-1, n_x, n_u-1)

# Build full control Jacobian including time dilation
dfdu = jnp.zeros((x.shape[0], n_x, n_u))
dfdu = dfdu.at[:, :, :-1].set(s[:, None, None] * dfdu_veh)  # Vehicle control derivatives
dfdu = dfdu.at[:, :, -1].set(f)                              # Time dilation derivative = f
```

**Why exclude time dilation from dynamics?** Time dilation is a meta-control that scales the entire dynamics (used for time-optimal problems). The actual vehicle dynamics are defined without it, and time dilation is applied as a scaling factor during discretization. This is why `n_u-1` appears in the vehicle dynamics Jacobians.

**Example with N=10:** This single call evaluates dynamics at all 9 intervals simultaneously, leveraging JAX's efficient vectorization on GPU/TPU.

## Shape Summary Table

Here's a complete reference for shapes at each stage, shown with symbolic dimensions (`N`, `n_x`, `n_u`) and a concrete example:

| **Stage** | **Variable** | **Symbolic Shape** | **Concrete Example (N=10, n_x=4, n_u=2)** |
|-----------|--------------|-------------------|-------------------------------------------|
| **User Definition** | `position` | `(2,)` | `(2,)` - Single 2D position vector |
| | `velocity` | `(1,)` | `(1,)` - Single scalar velocity |
| | `theta` | `(1,)` | `(1,)` - Single scalar control |
| | | | |
| **After Augmentation** | `states_aug` | List of States | [position, velocity, time] (3 states) |
| | `controls_aug` | List of Controls | [theta, _time_dilation] (2 controls) |
| | | | |
| **After Unification** | `x_unified.shape` | `(n_x,)` | `(4,)` - position(2) + velocity(1) + time(1) |
| | `u_unified.shape` | `(n_u,)` | `(2,)` - theta(1) + _time_dilation(1) |
| | `x_unified.guess` | `(N, n_x)` | `(10, 4)` - States at 10 nodes |
| | `u_unified.guess` | `(N, n_u)` | `(10, 2)` - Controls at 10 nodes |
| | `position._slice` | `slice(0, 2)` | `slice(0, 2)` - Extract position |
| | `velocity._slice` | `slice(2, 3)` | `slice(2, 3)` - Extract velocity |
| | `time._slice` | `slice(3, 4)` | `slice(3, 4)` - Extract time |
| | | | |
| **JAX Functions (Pre-Vmap)** | **Dynamics:** | | |
| | `f(x, u, node, params)` | Input: `(n_x,), (n_u,), scalar, dict` | Input: `(4,), (2,), scalar, dict` |
| | | Output: `(n_x,)` | Output: `(4,)` - Single state derivative |
| | `A(x, u, node, params)` | Output: `(n_x, n_x)` | Output: `(4, 4)` - Jacobian df/dx |
| | `B(x, u, node, params)` | Output: `(n_x, n_u)` | Output: `(4, 2)` - Jacobian df/du |
| | **Constraints:** | | |
| | `g(x, u, node, params)` | Input: `(n_x,), (n_u,), scalar, dict` | Input: `(4,), (2,), scalar, dict` |
| | | Output: `scalar` | Output: `scalar` - Single constraint value |
| | `grad_g_x(x, u, node, params)` | Output: `(n_x,)` | Output: `(4,)` - Gradient dg/dx |
| | `grad_g_u(x, u, node, params)` | Output: `(n_u,)` | Output: `(2,)` - Gradient dg/du |
| | | | |
| **JAX Functions (Post-Vmap)** | **Dynamics:** | | |
| | `f(x, u, nodes, params)` | Input: `(N-1, n_x), (N-1, n_u), (N-1,), dict` | Input: `(9, 4), (9, 2), (9,), dict` |
| | | Output: `(N-1, n_x)` | Output: `(9, 4)` - Derivatives at 9 intervals |
| | `A(x, u, nodes, params)` | Output: `(N-1, n_x, n_x)` | Output: `(9, 4, 4)` - Jacobians at 9 intervals |
| | `B(x, u, nodes, params)` | Output: `(N-1, n_x, n_u)` | Output: `(9, 4, 2)` - Jacobians at 9 intervals |
| | **Constraints:** | | |
| | `g(x, u, node, params)` | Input: `(M, n_x), (M, n_u), scalar, dict` | Input: `(3, 4), (3, 2), scalar, dict` |
| | | Output: `(M,)` | Output: `(3,)` - M=3 constraint evaluations |
| | `grad_g_x(x, u, node, params)` | Output: `(M, n_x)` | Output: `(3, 4)` - Gradients at M nodes |
| | `grad_g_u(x, u, node, params)` | Output: `(M, n_u)` | Output: `(3, 2)` - Gradients at M nodes |
| | **Cross-Node Constraints:** | | |
| | `g_cross(X, U, params)` | Input: `(N, n_x), (N, n_u), dict` | Input: `(10, 4), (10, 2), dict` |
| | | Output: `scalar` | Output: `scalar` - Single constraint |
| | `grad_g_X(X, U, params)` | Output: `(N, n_x)` | Output: `(10, 4)` - Trajectory Jacobian |
| | `grad_g_U(X, U, params)` | Output: `(N, n_u)` | Output: `(10, 2)` - Trajectory Jacobian |
| | | **Note:** Jacobians are dense but sparse | **Sparsity:** Typically only 2-3 rows non-zero |

## Performance Implications

**Why This Architecture?**

1. **GPU/TPU Acceleration**: Vmapping enables SIMD parallelism across nodes for both dynamics and constraints
2. **JIT Compilation**: JAX compiles vmapped functions once, not per-node
3. **Automatic Differentiation**: Jacobians and gradients computed automatically via `jacfwd`
4. **Reduced Python Overhead**: Single JAX call instead of Python loops for evaluation

## Implementation Files Reference

| **File** | **Function/Class** | **Purpose** |
|----------|-------------------|-------------|
| `openscvx/problem.py` | `Problem.__init__` | Orchestrates preprocessing pipeline |
| `openscvx/symbolic/builder.py` | `preprocess_symbolic_problem` | Augments states/controls/dynamics |
| `openscvx/symbolic/lower.py` | `lower_symbolic_expressions` | Unification and JAX lowering for dynamics/constraints |
| `openscvx/symbolic/unified.py` | `unify_states`, `unify_controls` | Combines individual variables into unified vectors |
| `openscvx/problem.py` | `initialize` | Applies vmap to dynamics |
| `openscvx/discretization.py` | `dVdt`, `calculate_discretization` | Uses vmapped dynamics |
| `openscvx/constraints/lowered.py` | `LoweredNodalConstraint` | Container for vmapped nodal constraints |
| `openscvx/constraints/cross_node.py` | `LoweredCrossNodeConstraint` | Container for trajectory-level cross-node constraints |
| `openscvx/symbolic/expr/constraint.py` | `CrossNodeConstraint` | Expression type for cross-node constraints |
| `openscvx/symbolic/expr/expr.py` | `NodeReference` | Expression for referencing specific trajectory nodes |
| `openscvx/symbolic/lowerers/jax.py` | `JaxLowerer._visit_cross_node_constraint` | Lowers CrossNodeConstraint to trajectory-level function |
| `openscvx/symbolic/lowerers/jax.py` | `JaxLowerer._visit_node_reference` | Lowers NodeReference to JAX array indexing |
| `openscvx/ocp.py` | `create_cvxpy_variables` | Creates CVXPy variables including cross-node parameters |
| `openscvx/ptr.py` | `PTR_subproblem` | Updates constraint parameters during SCP iterations |

## Advanced: Accessing Unified Vectors

During problem setup, you can access the unified objects:

```python
problem = Problem(...)
problem.initialize()

# Access unified state/control objects
x_unified = problem.x_unified
u_unified = problem.u_unified

print(f"Total state dimension: {x_unified.shape[0]}")
print(f"Total control dimension: {u_unified.shape[0]}")

# Access individual state slices
for state in problem.states:
    print(f"{state.name}: slice {state._slice}")
```

## Common Developer Pitfalls

1. **Confusing nodes vs intervals**: Discretization operates on N-1 intervals between N nodes, so vmapped dynamics have batch size `(N-1, ...)`, while constraints evaluate at specific nodes (batch size M where M = number of nodes where constraint applies)
2. **Forgetting augmented dimensions**: `n_x` and `n_u` include auto-added states/controls (time, CTCS augmented states, time dilation)
3. **Parameter mutability**: The `params` dict is shared across all evaluations - don't modify it during dynamics or constraint evaluation
4. **Node index usage**: The `node` parameter enables time-varying behavior (e.g., time-dependent constraints), not for indexing into trajectory arrays
5. **Constraint vs dynamics vmap axes**: Constraints use `in_axes=(0, 0, None, None)` (node not batched), while dynamics use `in_axes=(0, 0, 0, None)` (node batched across intervals)
6. **Cross-node constraint signature**: Regular nodal constraints use `(x, u, node, params)` while cross-node constraints use `(X, U, params)` with full trajectory inputs
7. **Cross-node Jacobian memory**: Cross-node Jacobians have shape `(N, n_x)` stored densely but are typically very sparse (most rows are zero). Be aware of memory usage for large N

## See Also

- [Hello world tutorial](../UsersGuide/01_hello_world_brachistochrone.md) - How to define problems
- [Discretization](../Foundations/discretization.md) - How discretization works in OpenSCvx
