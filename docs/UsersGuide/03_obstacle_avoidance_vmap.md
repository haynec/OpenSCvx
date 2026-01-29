# 03 Obstacle Avoidance: 6-DOF Dynamics, Parameters, and Vmap

In this tutorial we tackle a more realistic drone control problem: navigating through obstacles with full 6-DOF rigid body dynamics.
We will introduce quaternion-based attitude representation, the spatial utility functions, and show how to define obstacles as Parameters for runtime updates.
In Part 2, we address scalability—what happens when you have dozens or hundreds of obstacles—and introduce `ox.Vmap` for efficient vectorized constraint evaluation.
Finally, we show how to use these tools for real-time model-predictive control (MPC) applications.

This tutorial covers:

- 6-DOF rigid body dynamics (position, velocity, attitude, angular velocity)
- Spatial utility functions (`ox.spatial.QDCM`, `ox.spatial.SSM`, `ox.spatial.SSMP`)
- Runtime-updatable parameters with `ox.Parameter`
- Ellipsoidal obstacle constraints
- Vectorized constraints with `ox.Vmap`
- Real-time control with `problem.step()`

## 6-DOF Obstacle Avoidance

!!! tip "Try it yourself"
    This tutorial is available as an interactive Colab notebook:

    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xLPC_UJWC35oPRIAY3vkxi8WEYnHCysQ?usp=sharing)

### The Problem

We consider a quadrotor navigating through a field of ellipsoidal obstacles in minimum time.
Unlike the double-integrator model from [Drone Racing: Constraints and 3-DOF Dynamics](02_drone_racing_constraints.md), we now model the full rotational dynamics:

$$
\begin{align}
\min_{\mathbf{x}, \mathbf{u}, t_f}\ &t_f & \\
\mathrm{s.t.}\ &\dot{\mathbf{x}}(t) = f(\mathbf{x}(t), \mathbf{u}(t)) & \forall t\in[0, t_f], \quad &\textrm{dynamics} \\
&\mathbf{x}_{\min} \leq \mathbf{x}(t) \leq \mathbf{x}_{\max} & \forall t\in[0, t_f], \quad &\textrm{state bounds} \\
&\mathbf{u}_{\min} \leq \mathbf{u}(t) \leq \mathbf{u}_{\max} & \forall t\in[0, t_f], \quad &\textrm{control bounds} \\
&(\mathbf{p}(t) - \mathbf{c}_i)^\top A_i (\mathbf{p}(t) - \mathbf{c}_i) \geq 1 & \forall t, \forall i, \quad &\textrm{obstacle avoidance} \\
&\mathbf{x}(0) = \mathbf{x}_{\mathrm{init}} & & \textrm{initial} \\
&\mathbf{p}(t_f) = \mathbf{p}_{\mathrm{final}} & & \textrm{terminal}
\end{align}
$$

The state now includes attitude and angular velocity:

$$
\mathbf{x} = \begin{bmatrix} \mathbf{p} \\ \mathbf{v} \\ \mathbf{q} \\ \boldsymbol{\omega} \end{bmatrix} \in \mathbb{R}^{13}
$$

where $\mathbf{p} \in \mathbb{R}^3$ is position, $\mathbf{v} \in \mathbb{R}^3$ is velocity, $\mathbf{q} \in \mathbb{R}^4$ is the attitude quaternion, and $\boldsymbol{\omega} \in \mathbb{R}^3$ is angular velocity in the body frame.

The control input consists of thrust force and torque:

$$
\mathbf{u} = \begin{bmatrix} \mathbf{f} \\ \boldsymbol{\tau} \end{bmatrix} \in \mathbb{R}^{6}
$$

### 6-DOF Rigid Body Dynamics

The equations of motion for a rigid body are:

$$
\begin{align}
\dot{\mathbf{p}} &= \mathbf{v} \\
\dot{\mathbf{v}} &= \frac{1}{m} C(\mathbf{q}) \mathbf{f} + \mathbf{g} \\
\dot{\mathbf{q}} &= \frac{1}{2} \Omega(\boldsymbol{\omega}) \mathbf{q} \\
\dot{\boldsymbol{\omega}} &= J^{-1} \left( \boldsymbol{\tau} - \boldsymbol{\omega} \times (J \boldsymbol{\omega}) \right)
\end{align}
$$

where $C(\mathbf{q})$ is the direction cosine matrix (DCM) that rotates vectors from body to inertial frame, $\Omega(\boldsymbol{\omega})$ is the skew-symmetric matrix for quaternion kinematics, $J$ is the inertia tensor, and $\mathbf{g} = [0, 0, g]^\top$ is gravity.

### Variables

The state definitions follow the familiar pattern, now with quaternion attitude:

```python
import numpy as np
import jax.numpy as jnp
import openscvx as ox

n = 6
total_time = 4.0

# Position and velocity (as before)
position = ox.State("position", shape=(3,))
position.max = np.array([200.0, 10, 20])
position.min = np.array([-200.0, -100, 0])
position.initial = np.array([10.0, 0, 2])
position.final = [-10.0, 0, 2]

velocity = ox.State("velocity", shape=(3,))
velocity.max = np.array([100, 100, 100])
velocity.min = np.array([-100, -100, -100])
velocity.initial = np.array([0, 0, 0])
velocity.final = [("free", 0), ("free", 0), ("free", 0)]

# Attitude quaternion [qw, qx, qy, qz]
attitude = ox.State("attitude", shape=(4,))
attitude.max = np.array([1, 1, 1, 1])
attitude.min = np.array([-1, -1, -1, -1])
attitude.initial = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]
attitude.final = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]

# Angular velocity in body frame
angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = np.array([10, 10, 10])
angular_velocity.min = np.array([-10, -10, -10])
angular_velocity.initial = [("free", 0), ("free", 0), ("free", 0)]
angular_velocity.final = [("free", 0), ("free", 0), ("free", 0)]

states = [position, velocity, attitude, angular_velocity]
```

Note that the attitude quaternion uses `("free", value)` for both initial and final conditions—we don't constrain the drone's orientation, only its position.

The control includes both thrust force (in body frame) and torque:

```python
# Thrust force in body frame (z-up, so thrust is along +z)
thrust_force = ox.Control("thrust_force", shape=(3,))
thrust_force.max = np.array([0, 0, 4.179 * 9.81])  # Only +z thrust
thrust_force.min = np.array([0, 0, 0])
thrust_force.guess = np.repeat([[0.0, 0.0, thrust_force.max[2]]], n, axis=0)

# Control torques
torque = ox.Control("torque", shape=(3,))
torque.max = np.array([18.665, 18.665, 0.556])
torque.min = np.array([-18.665, -18.665, -0.556])
torque.guess = np.zeros((n, 3))

controls = [thrust_force, torque]
```

### Dynamics with Spatial Utilities

OpenSCvx provides spatial utility functions for common rotational dynamics operations:

| Function | Description |
|----------|-------------|
| `ox.spatial.QDCM(q)` | Quaternion to Direction Cosine Matrix |
| `ox.spatial.SSM(v)` | Skew-Symmetric Matrix from vector (for cross products) |
| `ox.spatial.SSMP(v)` | Skew-Symmetric Matrix Product (for quaternion kinematics) |

Using these, the 6-DOF dynamics become:

```python
m = 1.0  # Mass
g_const = -9.81
J_b = jnp.array([1.0, 1.0, 1.0])  # Principal moments of inertia

# Normalize quaternion for numerical stability
q_norm = ox.linalg.Norm(attitude)
attitude_normalized = attitude / q_norm

# Dynamics
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

The key lines:

- `ox.spatial.QDCM(q) @ thrust_force` rotates the body-frame thrust into the inertial frame
- `ox.spatial.SSMP(ω) @ q` implements the quaternion derivative $\dot{\mathbf{q}} = \frac{1}{2}\Omega(\boldsymbol{\omega})\mathbf{q}$
- `ox.spatial.SSM(ω) @ (J @ ω)` computes the gyroscopic term $\boldsymbol{\omega} \times (J\boldsymbol{\omega})$

### Parameters

So far, all values in our problems have been fixed at compile time.
But what if you want to update obstacle positions at runtime, for example, in a receding-horizon controller where obstacle estimates are updated each solve?

OpenSCvx provides `ox.Parameter` for values that can be changed between solves without recompiling:

```python
# Define obstacle centers as parameters
obstacle_centers = [
    ox.Parameter("obstacle_center_1", shape=(3,), value=np.array([-5.1, 0.1, 2])),
    ox.Parameter("obstacle_center_2", shape=(3,), value=np.array([0.1, 0.1, 2])),
    ox.Parameter("obstacle_center_3", shape=(3,), value=np.array([5.1, 0.1, 2])),
]
```

Parameters are used in expressions exactly like constants, but their values can be updated later:

```python
# After problem.initialize(), update parameter values
problem.parameters["obstacle_center_1"] = np.array([-6.0, 1.0, 2.5])

# Solve with new values (no recompilation)
results = problem.solve()
```

!!! tip
    Parameters can be combined with `problem.step()` to update values in between SCP iteration.

### Ellipsoidal Obstacle Constraints

Ellipsoidal obstacles are defined by a center $\mathbf{c}$ and a shape matrix $A$. The constraint "stay outside the ellipsoid" is:

$$
(\mathbf{p} - \mathbf{c})^\top A (\mathbf{p} - \mathbf{c}) \geq 1
$$

In OpenSCvx:

```python
# Generate random ellipsoid shape matrices
A_obs = []
for _ in obstacle_centers:
    axes = generate_orthogonal_unit_vectors()  # Random orientation
    radii = np.random.rand(3) + 0.1
    A_obs.append(axes @ np.diag(radii**2) @ axes.T)

# Add obstacle avoidance constraints
constraints = []

# Box constraints on states (as before)
for state in states:
    constraints.extend([
        ox.ctcs(state <= state.max),
        ox.ctcs(state.min <= state)
    ])

# Obstacle constraints
for center, A in zip(obstacle_centers, A_obs):
    diff = position - center
    constraints.append(ox.ctcs(1.0 <= diff.T @ A @ diff))
```

These are continuous-time constraints (CTCS), so the drone avoids obstacles not just at discrete nodes but throughout the entire trajectory.

### Problem Setup

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
```

### Solver Settings

The SCP algorithm has several tuning parameters that affect convergence. For this obstacle avoidance problem:

```python
problem.settings.scp.lam_prox = 1e1        # Trust region weight
problem.settings.scp.lam_cost = 1e1    # Cost objective weight
problem.settings.scp.lam_vc = 1e2      # Virtual control weight
# Configure cost relaxation via autotuner
problem.settings.scp.autotuner = ox.AugmentedLagrangian(
    lam_cost_drop=4,     # Iteration to start relaxing cost
    lam_cost_relax=0.5  # Cost relaxation factor
)
```

### Solution

```python
problem.initialize()
results = problem.solve()
results = problem.post_process()
```

---

## Data Parallelism: Vmap

The approach above works well for a handful of obstacles. But what if you have 64 obstacles? Or 2000?

Creating individual CTCS constraints in a loop means each constraint object is processed separately by the constraint system, this doesn't scale.

### The Scaling Problem

Consider a 3D grid of obstacles:

```python
n_obstacles = 64  # 4 x 4 x 4 grid
obstacle_centers = np.array([...])  # Shape: (64, 3)
obstacle_radii = np.array([...])    # Shape: (64,)
```

The naive approach creates 64 separate constraints:

```python
# Approach 1: Loop (doesn't scale well)
for i in range(n_obstacles):
    center = obstacle_centers[i]
    radius = obstacle_radii[i]
    constraints.append(
        ox.ctcs(radius <= ox.linalg.Norm(position - center))
    )
```

This creates N separate constraint objects, each processed independently. For 64 obstacles this adds overhead; for 2000+ it becomes impractical.

### Vectorized Constraints with Vmap

`ox.Vmap` enables data-parallel evaluation by applying a function across a batch of inputs, returning a vector-valued result that is enforced element-wise:

```python
# Approach 2: Vmap (vectorized, efficient)
obstacle_avoidance = ox.ctcs(
    obstacle_radii <= ox.Vmap(
        lambda center: ox.linalg.Norm(position - center),
        batch=obstacle_centers,
    )
)
constraints.append(obstacle_avoidance)
```

This creates a **single** CTCS constraint that computes distances to all obstacles in parallel. The constraint is:

$$
r_i \leq \lVert \mathbf{p} - \mathbf{c}_i \rVert \quad \forall i \in [1, N_{\mathrm{obs}}]
$$

where the right-hand side is computed as one vectorized operation.

!!! warning
    `ox.Vmap` is specifically for **data parallelism**: applying the same operation across a batch of inputs. It is unfortunately not a _panacea_ for magically parallelizing every operation.

---

## Real-Time Control with `step()`

So far we've used `problem.solve()` which runs the full SCP algorithm to convergence. But for receding-horizon or model-predictive control (MPC) applications, you often want finer control: run a single SCP iteration, update parameters based on new sensor data, and repeat.

### Single-Step Iteration

Use `problem.step()` instead of `problem.solve()` to run one SCP iteration:

```python
problem.initialize()

# Real-time control loop
while running:
    # Run one SCP iteration
    step_result = problem.step()

    # step_result contains iteration metrics:
    # - step_result.J_tr: trust region cost
    # - step_result.J_vc: virtual control cost
    # - step_result.converged: whether convergence criteria met
```

Each call to `step()` performs one linearization and convex solve, warm-starting from the previous solution.

### Updating Parameters Between Steps

Parameters (introduced [earlier in this tutorial](#parameters)) are designed for exactly this use case. Update obstacle positions, target locations, or physical constants between iterations:

```python
while running:
    # Get new state estimate from sensors
    current_position = get_state_estimate()

    # Update obstacle positions from perception system
    for i, obs_pos in enumerate(detected_obstacles):
        problem.parameters[f"obstacle_center_{i}"] = obs_pos

    # Run one iteration with updated parameters
    step_result = problem.step()

    # Extract control to apply
    control = problem.state.V_history[-1]  # Access current solution
```

Parameter updates take effect immediately on the next `step()` call without any recompilation.

### Adjusting Solver Settings

You can also adjust SCP weights at runtime to change solver behavior:

```python
# Increase cost weight as solution converges
if step_result.J_vc < 1e-4:
    problem.settings.scp.lam_cost *= 1.1

# Tighten trust region if solution is oscillating
problem.settings.scp.lam_prox = 2.0
```

### Resetting the Problem

To reset the solver state back to the initial guess (e.g., when starting a new trajectory or after a large disturbance):

```python
problem.reset()  # Clears V_history, resets to initial guess
```

### Example: Interactive Drone Racing

For a complete real-time example with interactive 3D visualization, see [`examples/realtime/drone_racing_realtime.py`](https://github.com/OpenSCvx/OpenSCvx/blob/main/examples/realtime/drone_racing_realtime.py). This example demonstrates:

- Continuous SCP iteration in a background thread
- Draggable gates that update parameters in real-time
- Live trajectory visualization as the solution evolves
- Runtime adjustment of solver weights

The visualization uses [viser](https://viser.studio/) for the interactive 3D interface. The OpenSCvx integration is straightforward, most of the complexity is in the GUI, not the optimization loop.
For further information about visualization with viser see [Tutorial 05 Visualizing Results](05_visualization.md).

## Further Reading

- [Complete 6-DOF Obstacle Avoidance Example](../Examples/drone/obstacle_avoidance.md)
- [Complete Vmap Obstacle Avoidance Example](../Examples/drone/obstacle_avoidance_vmap.md)
- [Real-Time Obstacle Avoidance Example](https://github.com/OpenSCvx/OpenSCvx/blob/main/examples/realtime/obstacle_avoidance_realtime.py)
- [Real-Time Drone Racing Example](https://github.com/OpenSCvx/OpenSCvx/blob/main/examples/realtime/drone_racing_realtime.py)
- [API Reference: Spatial Utilities](../Reference/symbolic/expr/spatial.md)
- [API Reference: Parameters](../Reference/symbolic/expr/expr.md)
- [Viewpoint Constraints: Custom Functions and Perception](04_viewpoint_constraints.md)
- [Visualization: 2D Plots and 3D Interactive](05_visualization.md)
