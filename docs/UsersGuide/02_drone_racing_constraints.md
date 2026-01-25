# 02 Drone Racing: Constraints and 3-DOF Dynamics

In this tutorial we build on the concepts from [Hello Brachistochrone](01_hello_world_brachistochrone.md) to tackle a more interesting problem: time-optimal drone racing through gates.
We will introduce 3D double-integrator dynamics and explore the various constraint types available in OpenSCvx.

This tutorial covers:

- 3DoF double-integrator (point mass) dynamics
- Nodal constraints with `.at()`
- Convex constraint marking with `.convex()`
- Keyframe-based initialization with `ox.init.linspace()`

## The Drone Racing Problem

We consider a simplified drone racing scenario where a point-mass drone must fly through a sequence of gates in minimum time, returning to its starting position (loop closure).

$$
\begin{align}
\min_{\mathbf{x}, \mathbf{u}, t_f}\ &t_f & \\
\mathrm{s.t.}\ &\dot{\mathbf{x}}(t) = f(\mathbf{x}(t), \mathbf{u}(t)) & \forall t\in[0, t_f], \quad &\textrm{dynamics} \\
&\mathbf{x}_{\min} \leq \mathbf{x}(t) \leq \mathbf{x}_{\max} & \forall t\in[0, t_f], \quad &\textrm{state bounds} \\
&\mathbf{u}_{\min} \leq \mathbf{u}(t) \leq \mathbf{u}_{\max} & \forall t\in[0, t_f], \quad &\textrm{control bounds} \\
&\lVert A_i \mathbf{p}(t_i) - \mathbf{c}_i \rVert_\infty \leq 1 & \forall i \in [1, N_{\mathrm{gates}}], \quad &\textrm{gate constraints} \\
&\mathbf{x}(0) = \mathbf{x}_{\mathrm{init}}, & & \textrm{initial}\\
&\mathbf{p}(t_f) = \mathbf{p}_{\mathrm{init}} & & \textrm{loop closure}
\end{align}
$$

where the state $\mathbf{x} = [\mathbf{p}^\top, \mathbf{v}^\top]^\top$ consists of 3D position and velocity, and the control $\mathbf{u} = \mathbf{f}$ is the force vector. The double-integrator dynamics are:

$$
f(\mathbf{x}, \mathbf{u}) = \begin{bmatrix} \mathbf{v} \\ \frac{1}{m}\mathbf{f} + \mathbf{g} \end{bmatrix}
$$

The gate constraints use an infinity-norm formulation: the drone must pass through an ellipsoidal region defined by the matrix $A_i$ centered at $\mathbf{c}_i$ at specific times $t_i$.

## Implementation

### Variables

The state and control definitions follow the same pattern as the Brachistochrone problem, now in 3D:

```python
import openscvx as ox

# 3D position and velocity states
position = ox.State("position", shape=(3,))
position.max = np.array([200.0, 100, 50])
position.min = np.array([-200.0, -100, 15])
position.initial = np.array([10.0, 0, 20])
position.final = [10.0, 0, 20]  # Loop closure: return to start

velocity = ox.State("velocity", shape=(3,))
velocity.max = np.array([100, 100, 100])
velocity.min = np.array([-100, -100, -100])
velocity.initial = np.array([0, 0, 0])
velocity.final = [("free", 0), ("free", 0), ("free", 0)]

# 3D force control
force = ox.Control("force", shape=(3,))
f_max = 4.179 * 9.81
force.max = np.array([f_max, f_max, f_max])
force.min = np.array([-f_max, -f_max, -f_max])
force.guess = np.repeat(np.array([[0.0, 0, 10]]), n, axis=0)

states = [position, velocity]
controls = [force]
```

Note that `position.final` equals `position.initial`. This enforces loop closure, requiring the drone to return to its starting position.

### Dynamics

The double-integrator dynamics are straightforward: velocity is the derivative of position, and acceleration (force/mass plus gravity) is the derivative of velocity:

```python
m = 1.0  # Mass
g_const = -9.81  # Gravity (negative z)

dynamics = {
    "position": velocity,
    "velocity": (1 / m) * force + np.array([0, 0, g_const]),
}
```

This is simpler than the Brachistochrone dynamics because there's no angle—we directly control the force vector.

### Path Constraints

As before, we enforce box constraints on states continuously:

```python
constraints = []
for state in states:
    constraints.extend([
        ox.ctcs(state <= state.max),
        ox.ctcs(state.min <= state)
    ])
```

With our states now in 3D it becomes clear to us why such constraints are also referred to as _path constraints_; we are enforcing the entire path to follow the constraints, not just the discrete nodes.

### Discrete Constraints: Gate Passage

The key new concept is **nodal constraints**, constraints enforced at specific nodes rather than continuously.
Gate passage constraints are a perfect example: the drone must be within each gate at a specific node.

We use the `.at([node])` method to specify which nodes the constraint applies to:

```python
# Gate passage constraint at a specific node
gate_constraint = (
    ox.linalg.Norm(A_gate @ position - c_gate, ord="inf") <= 1.0
).at([node])
```

The infinity norm $\lVert \cdot \rVert_\infty$ defines a box-shaped region, which when combined with the scaling matrix `A_gate` creates an ellipsoidal gate region.

It should be noted that by default constraints are interpreted as nodal constraints, however they are applied to _all_ nodes when not otherwise noted. Writing

```python
gate_constraint = (
    ox.linalg.Norm(A_gate @ position - c_gate, ord="inf") <= 1.0
)
```

would result in all nodes being constrained to lie within the gate.

There is a similar syntax for defining CTCS constraints: we can use the `.over((k, j))` method to define a continuous interval between nodes where a constraint should be enforced:

```python
# Enforce altitude constraint continuously between nodes 0 and 5
altitude_constraint = (position[2] >= 15.0).over((0, 5))

# Enforce obstacle avoidance only during the approach phase
obstacle_center = np.array([50, -20, 20])
safe_distance = 5.0
diff = position - obstacle_center
obstacle_constraint = (diff.T @ diff >= safe_distance**2).over((5, 10))
```

The `.over()` method also accepts optional parameters for the penalty function type (`penalty`), grouping index (`idx`), and whether to also check nodally (`check_nodally`).

!!! note
    You can also directly instantiate `NodalConstraint` or `CTCS` objects if you prefer:

    ```python
    from openscvx.symbolic.expr.constraint import NodalConstraint, CTCS

    # Equivalent to (position[2] >= 15.0).at([5, 10])
    altitude_nodal = NodalConstraint(position[2] >= 15.0, nodes=[5, 10])

    # Equivalent to (position[2] >= 15.0).over((0, 15))
    altitude_ctcs = CTCS(position[2] >= 15.0, nodes=(0, 15))
    ```

#### Marking Convex Constraints

When a constraint is convex, we can mark it with `.convex()`:

```python
gate_constraint = (
    ox.linalg.Norm(A_gate @ position - c_gate, ord="inf") <= 1.0
).convex().at([node])
```

This tells OpenSCvx that no successive convexification is needed for this constraint, it is then lowered directly as a CVXPY constraint when the problem is lowered.
This helps improve numerical performance as the solver is operating directly on the true constraint, not a linearized version thereof.

#### Full Gate Setup

Here's the complete gate constraint setup for multiple gates:

```python
n_gates = 10
gate_centers = [
    np.array([59.436, 0.000, 20.0]),
    np.array([92.964, -23.750, 25.524]),
    # ... more gate centers
]

# Assign nodes to gates (evenly spaced)
nodes_per_gate = 2
gate_nodes = np.arange(nodes_per_gate, n, nodes_per_gate)

# Add gate constraints
for node, center in zip(gate_nodes, gate_centers):
    A_gate_scaled = A_gate @ center  # Pre-scaled center
    gate_constraint = (
        ox.linalg.Norm(A_gate @ position - A_gate_scaled, ord="inf") <= 1.0
    ).convex().at([node])
    constraints.append(gate_constraint)
```

### Keyframe Initialization

For problems with waypoints or gates, a linear interpolation between start and end isn't a good initial guess.
In fact, the start and end positions are identical to enforce loop closure. We need a more customized initial guess.

In this drone racing example the gate ordering is not free. There is a specific order in which the gates must be traversed.
We can use this _a priori_ knowledge to construct our initial guess; placing the required nodes directly in the center of the corresponding gate and linearly interpolating at the nodes in between gates.

To facilitate such initial guesses, OpenSCvx provides `ox.init.linspace()` for keyframe-based initialization:

```python
position.guess = ox.init.linspace(
    keyframes=[position.initial] + gate_centers + [position.final],
    nodes=[0] + list(gate_nodes) + [n - 1],
)
```

This lets us specify lists of key values, or `keyframes` to borrow an animation term and the corresponding nodes. At the nodes in between the keyframes the values will just be linearly interpolated.
This creates an initial trajectory that passes through each gate center at the appropriate node, giving the solver a much better starting point.

OpenSCvx also provides similar `ox.init.slerp(...)` and `ox.init.nlerp(...)` functions for _spherical linear interpolation_ (SLERP) and _normalized linear interpolation_ (NLERP) for quaternion initialization.

### Problem Definition and Solution

```python
time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
)

problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
)

problem.initialize()
results = problem.solve()
results = problem.post_process()
```

### Visualizing 3D Trajectories

With our drone now flying in 3D, simple time series plots don't capture the full picture. The `plot_projections_2d()` function shows XY, XZ, and YZ plane views:

```python
from openscvx.plotting import plot_states, plot_projections_2d

# Time series of states
plot_states(results, ["position", "velocity"]).show()

# 2D projections colored by velocity
plot_projections_2d(
    results,
    var_name="position",
    velocity_var_name="velocity"
).show()
```

For fully interactive 3D visualization with animated playback, gates, and thrust vectors, OpenSCvx integrates with [viser](https://viser.studio/). See [Tutorial 05: Visualization](05_visualization.md) for details on building rich 3D visualizations.

## Constraint Types Summary

| Type | Syntax | Use Case |
|------|--------|----------|
| Continuous (all nodes) | `ox.ctcs(expr)` | Path constraints enforced between all nodes |
| Continuous (interval) | `expr.over((start, end))` | Path constraints over a specific interval |
| Nodal (specific) | `expr.at([nodes])` | Waypoints, gate passage, events |
| Nodal (all) | `expr` | Constraint at every node |
| Convex | `expr.convex()` | Mark convex constraints for direct CVXPY lowering |
| Combined | `expr.convex().at([nodes])` | Convex nodal constraints |

!!! note "Cross-Node Constraints"
    OpenSCvx also supports **cross-node constraints** that couple values at different nodes within a single constraint expression. These are created by using `.at(k)` on variables (not constraints):

    ```python
    # Rate limit: position change between consecutive nodes
    rate_limit = position.at(5) - position.at(4) <= max_step
    ```

    Cross-node constraints are automatically detected and handled differently from nodal constraints—they operate on the full trajectory arrays rather than being evaluated node-by-node. We will cover these in a more advanced tutorial.

## Further Reading

- [Complete Drone Racing Example](../Examples/drone/dr_double_integrator.md)
- [Full 6-DOF Drone Racing](../Examples/drone/drone_racing.md) — adds attitude dynamics
- [API Reference: Constraints](../Reference/symbolic/expr/constraint.md)
- [Obstacle Avoidance: 6-DOF Dynamics, Parameters, and Vmap](03_obstacle_avoidance_vmap.md)
- [Viewpoint Constraints: Custom Functions and Perception](04_viewpoint_constraints.md)
- [Visualization: 2D Plots and 3D Interactive](05_visualization.md)
