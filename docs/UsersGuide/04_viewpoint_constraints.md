# 04 Looking Back: Viewpoint Constraints

In this tutorial we will combine what we have learned in the past several exercises while introducing a new viewplanning constraint formulation.
We'll combine 6-DOF dynamics, gate constraints, and Vmap to solve a drone racing problem where the drone must maintain line-of-sight to a set of targets throughout its entire trajectory.

This tutorial doesn't introduce any major new API features; instead, it serves as a capstone example showing how to combine the tools you've learned to solve a genuinely challenging problem.

!!! tip "Try it yourself"
    This tutorial is available as an interactive Colab notebook:

    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b3NEx288h4r4HuvCOj-fexmt90PPhKUw?usp=sharing)

This tutorial covers:

- Field-of-view (cone) constraint formulation
- Writing custom symbolic constraint functions
- Combining `ox.Vmap` with custom functions
- Attitude initialization for perception tasks

## The Viewplanning Drone Racing Problem

We consider a quadrotor racing through gates while keeping a rigidly mounted sensor pointed at multiple viewplanning targets.

$$
\begin{align}
\min_{\mathbf{x}, \mathbf{u}, t_f}\ &t_f & \\
\mathrm{s.t.}\ &\dot{\mathbf{x}}(t) = f(\mathbf{x}(t), \mathbf{u}(t)) & \forall t\in[0, t_f], \quad &\textrm{dynamics} \\
&g_{\text{fov}}(\mathbf{p}(t), \mathbf{q}(t), \mathbf{c}_i) \leq 0 & \forall t, \forall i, \quad &\textrm{viewplanning} \\
&\lVert A_{\text{gate}} (\mathbf{p}(t_k) - \mathbf{g}_k) \rVert_\infty \leq 1 & \forall k, \quad &\textrm{gates} \\
&\mathbf{x}_{\min} \leq \mathbf{x}(t) \leq \mathbf{x}_{\max} & \forall t\in[0, t_f], \quad &\textrm{state bounds} \\
&\mathbf{u}_{\min} \leq \mathbf{u}(t) \leq \mathbf{u}_{\max} & \forall t\in[0, t_f], \quad &\textrm{control bounds} \\
&\mathbf{x}(0) = \mathbf{x}_{\mathrm{init}}, & & \textrm{initial}\\
&\mathbf{p}(t_f) = \mathbf{p}_{\mathrm{init}} & & \textrm{terminal}
\end{align}
$$

The state and dynamics are identical to [Tutorial 03](03_obstacle_avoidance_vmap.md): position, velocity, quaternion attitude, and angular velocity with 6-DOF rigid body dynamics.
We won't repeat the derivation here; if you need a refresher, go read that tutorial.

What's new is the visibility constraint $g_{\text{fov}}$, which ensures that each target $\mathbf{c}_i$ remains within the camera's field of view throughout the entire trajectory.

## Field-of-View Constraints

A camera's field of view is typically modeled as a cone emanating from the sensor.
A point is "visible" if it lies inside this cone.
Mathematically, we can express this as a second-order cone constraint.

### The Geometry

Let $\mathbf{r}_S$ be the position of a target expressed in the _sensor frame_.
The target is visible if it lies within a cone defined by:

$$
\lVert A_{\text{cone}} \mathbf{r}_S \rVert_\rho \leq \mathbf{c}^\top \mathbf{r}_S
$$

where:

- $A_{\text{cone}} = \text{diag}(1/\tan\alpha_x, 1/\tan\alpha_y, 0)$ encodes the cone's opening angles
- $\mathbf{c} = [0, 0, 1]^\top$ is the camera boresight (pointing direction)
- $\rho$ is the norm type (we use $\rho = 2$ for a circular cone)

The right-hand side $\mathbf{c}^\top \mathbf{r}_S$ is positive when the target is in front of the camera, and the inequality ensures the target doesn't stray too far off-axis.

### Coordinate Transformations

There's a catch: our targets are defined in the _inertial frame_, but the constraint is expressed in the _sensor frame_.
We need to transform coordinates, which means chaining rotations:

$$
\mathbf{r}_S = C_{S \leftarrow B} \cdot C_{B \leftarrow I}(\mathbf{q}) \cdot (\mathbf{c}_I - \mathbf{p}_I)
$$

where:

- $\mathbf{c}_I$ is the target position in the inertial frame
- $\mathbf{p}_I$ is the drone position in the inertial frame
- $C_{B \leftarrow I}(\mathbf{q})$ rotates from inertial to body frame (depends on attitude)
- $C_{S \leftarrow B}$ rotates from body to sensor frame (fixed mounting)

## Implementation

### Setup

We'll use the same 6-DOF state and control definitions from Tutorial 03.
For brevity, here's the condensed version:

```python
import numpy as np
import jax.numpy as jnp
import openscvx as ox

n = 33
total_time = 40.0

# States: position, velocity, attitude (quaternion), angular velocity
position = ox.State("position", shape=(3,))
position.max = np.array([200.0, 100, 50])
position.min = np.array([-200.0, -100, 15])
position.initial = np.array([10.0, 0, 20])
position.final = [10.0, 0, 20]  # Loop closure

velocity = ox.State("velocity", shape=(3,))
velocity.max = np.array([100, 100, 100])
velocity.min = np.array([-100, -100, -100])
velocity.initial = np.array([0, 0, 0])
velocity.final = [("free", 0), ("free", 0), ("free", 0)]

attitude = ox.State("attitude", shape=(4,))
attitude.max = np.array([1, 1, 1, 1])
attitude.min = np.array([-1, -1, -1, -1])
attitude.initial = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]
attitude.final = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = np.array([10, 10, 10])
angular_velocity.min = np.array([-10, -10, -10])
angular_velocity.initial = [("free", 0), ("free", 0), ("free", 0)]
angular_velocity.final = [("free", 0), ("free", 0), ("free", 0)]

states = [position, velocity, attitude, angular_velocity]

# Controls: thrust force and torque
thrust_force = ox.Control("thrust_force", shape=(3,))
thrust_force.max = np.array([0, 0, 4.179 * 9.81])
thrust_force.min = np.array([0, 0, 0])
thrust_force.guess = np.repeat([[0.0, 0, 10]], n, axis=0)

torque = ox.Control("torque", shape=(3,))
torque.max = np.array([18.665, 18.665, 0.556])
torque.min = np.array([-18.665, -18.665, -0.556])
torque.guess = np.zeros((n, 3))

controls = [thrust_force, torque]
```

The dynamics are also identical to Tutorial 03:

```python
m = 1.0
g_const = -9.81
J_b = jnp.array([1.0, 1.0, 1.0])

q_norm = ox.linalg.Norm(attitude)
attitude_normalized = attitude / q_norm

dynamics = {
    "position": velocity,
    "velocity": (1.0 / m) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
                + np.array([0, 0, g_const]),
    "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
    "angular_velocity": ox.linalg.Diag(1.0 / J_b) @ (
        torque - ox.spatial.SSM(angular_velocity) @ ox.linalg.Diag(J_b) @ angular_velocity
    ),
}
```

### Sensor and Target Setup

Now for the new stuff.
First, we define our camera parameters and target locations:

```python
# Camera field-of-view parameters
alpha_x = 6.0  # Horizontal half-angle (pi/6 radians)
alpha_y = 6.0  # Vertical half-angle
A_cone = np.diag([
    1 / np.tan(np.pi / alpha_x),
    1 / np.tan(np.pi / alpha_y),
    0,
])
c_boresight = jnp.array([0, 0, 1])  # Camera points along +z in sensor frame
R_sensor_body = jnp.array([         # Sensor mounting rotation
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

# Generate random target positions
n_targets = 10
np.random.seed(0)
target_positions = np.array([
    [100.0 + np.random.random() * 20, -60.0 + np.random.random() * 20, 20.0]
    for _ in range(n_targets)
])
```

### Custom Symbolic Constraint Functions

We can, of course, define _custom functions_ that operate on symbolic expressions.
This lets us encapsulate complex constraint logic in a reusable, readable form:

```python
def visibility_constraint(target_inertial, drone_position, drone_attitude):
    """
    Compute the FoV constraint value for a single target.

    Returns a value <= 0 when the target is visible.
    """
    # Transform target position to sensor frame
    relative_pos = target_inertial - drone_position
    pos_body = ox.spatial.QDCM(drone_attitude).T @ relative_pos
    pos_sensor = R_sensor_body @ pos_body

    # Cone constraint: ||A @ r|| - c'r <= 0
    return ox.linalg.Norm(A_cone @ pos_sensor, ord=2) - (c_boresight.T @ pos_sensor)
```

This function takes symbolic expressions as inputs and returns a symbolic expression.
The magic of OpenSCvx's expression system means we can use this function just like any built-in constraint.

### Vectorizing with Vmap

We have 10 targets, and we need to keep _all_ of them visible _continuously_.
In Tutorial 03, we used `ox.Vmap` to vectorize obstacle avoidance over many obstacles.
We can do the same thing here with our custom function:

```python
# Single CTCS constraint covering all targets
visibility = ox.ctcs(
    ox.Vmap(
        lambda target: visibility_constraint(target, position, attitude),
        batch=target_positions,
    ) <= 0.0
)
```

This creates one constraint object that evaluates the visibility function for all 10 targets in parallel.
Much cleaner than a loop, and much faster too.

### Gate Constraints

The gate constraints are unchanged from [Tutorial 02](02_drone_racing_constraints.md) with the exception of `nodes_per_gate` jumping from 2 to 3:

```python
n_gates = 10
gate_centers = [
    np.array([59.436, 0.0, 20.0]),
    np.array([92.964, -23.75, 25.524]),
    # ... more gates
]

# Gate geometry (square gates using infinity norm)
radii = np.array([2.5, 1e-4, 2.5])
A_gate = np.diag(1 / radii)

# Assign nodes to gates
nodes_per_gate = 3
gate_nodes = np.arange(nodes_per_gate, n, nodes_per_gate)

# Build constraint list
constraints = []

# Box constraints on states
for state in states:
    constraints.extend([
        ox.ctcs(state <= state.max),
        ox.ctcs(state.min <= state)
    ])

# Visibility constraint (all targets, all time)
constraints.append(visibility)

# Gate passage constraints (specific nodes)
for node, center in zip(gate_nodes, gate_centers):
    gate_constraint = (
        ox.linalg.Norm(A_gate @ position - A_gate @ center, ord="inf") <= 1.0
    ).convex().at([node])
    constraints.append(gate_constraint)
```

### Attitude-Aware Initialization

In this case it is beneficial to leverage our _a priori_ knowledge about the problem for initialization.
We once again will linearly interpolate positions between the gates since we already know the order in which they must be traversed.
Additionally, we now compute the orientation of the drone at each node s.t. the mean target position is centered in the sensor boresight.

```python
import numpy.linalg as la

# Position guess through gates (from Tutorial 02)
position.guess = ox.init.linspace(
    keyframes=[position.initial] + gate_centers + [position.final],
    nodes=[0] + list(gate_nodes) + [n - 1],
)

# Attitude guess: point camera at mean target
mean_target = np.mean(target_positions, axis=0)
camera_axis = R_sensor_body @ np.array([0, 1, 0])  # Camera "up" in body frame

attitude_guess = np.zeros((n, 4))
for k in range(n):
    # Vector from drone to mean target
    to_target = mean_target - position.guess[k]

    # Quaternion that rotates camera_axis to point at target
    cross = np.cross(camera_axis, to_target)
    dot = np.dot(camera_axis, to_target)
    q = np.array([
        np.sqrt(la.norm(to_target)**2 + la.norm(camera_axis)**2) + dot,
        *cross
    ])
    attitude_guess[k] = q / la.norm(q)

attitude.guess = attitude_guess
```

This gives the solver a fighting chance.
Without it, the initial trajectory might have the camera pointing at the sky while the targets are on the ground, and the solver has to figure out how to rotate 90+ degrees while also flying through gates.

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

# Tuning for this problem
problem.settings.scp.lam_prox = 2e0
problem.settings.scp.lam_cost = 1e-1
problem.settings.scp.lam_vc = 1e2
# Configure cost relaxation via autotuner
problem.settings.scp.autotuner = ox.AugmentedLagrangian(
    lam_cost_drop=10,
    lam_cost_relax=0.8
)

problem.initialize()
results = problem.solve()
results = problem.post_process()
```

## Key Takeaways

This tutorial demonstrated several practical patterns:

1. **Custom symbolic functions**: You can write functions that take symbolic expressions and return symbolic expressions. This lets you encapsulate complex constraint logic cleanly.
2. **Vmap with custom functions**: `ox.Vmap` isn't limited to built-in operations. You can vectorize any function over a batch of inputs.
3. **Perception-aware initialization**: When attitude matters (and it usually does in perception problems), take the time to compute a sensible initial guess. Your solver will thank you.
4. **Combining constraint types**: This problem mixed continuous visibility constraints (CTCS), discrete gate constraints (nodal), and state bounds—all in one problem. OpenSCvx handles the bookkeeping.

## Further Reading

- [Complete Example Code](../Examples/drone/dr_vp.md)
- [RA-L Paper: Line-of-Sight Constrained Trajectory Optimization](https://haynec.github.io/papers/los/)
- [API Reference: Spatial Utilities](../Reference/symbolic/expr/spatial.md)
- [Visualization: 2D Plots and 3D Interactive](05_visualization.md)
