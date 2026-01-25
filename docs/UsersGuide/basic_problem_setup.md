# Basic Problem Setup
Here we will cover all the necessary elements to setup your problem along with some tips and best practices to get the most out of the package.

## Imports
First, import OpenSCvx:

```python
import numpy as np
import jax.numpy as jnp
import openscvx as ox
from openscvx import Problem
```

## State Specification
States are defined as individual symbolic variables. Each state component gets its own `ox.State` object:

```python
# Create state variables
position = ox.State("position", shape=(3,))
velocity = ox.State("velocity", shape=(3,))

# Set bounds for each state
position.min = np.array([-10, -10, 0])
position.max = np.array([10, 10, 20])

velocity.min = np.array([-5, -5, -5])
velocity.max = np.array([5, 5, 5])

# Set initial conditions
position.initial = np.array([0, 0, 1])
velocity.initial = np.array([0, 0, 0])

# Set final conditions (can use tuples for free/minimize/maximize)
position.final = np.array([5, 5, 1])
velocity.final = [("free", 0), ("free", 0), ("free", 0)]

# Set initial guess for SCP (shape: (n_nodes, state_shape))
position.guess = np.linspace(position.initial, position.final, n_nodes)
velocity.guess = np.zeros((n_nodes, 3))

# Collect all states into a list
states = [position, velocity]
```

The boundary condition options use tuple syntax:

- Fixed value: `value` or `("fixed", value)`
- Free variable: `("free", guess)` - Can be optimized within bounds
- Minimize: `("minimize", guess)` - Variable to be minimized
- Maximize: `("maximize", guess)` - Variable to be maximized

## Control Specification
Controls are also defined as individual symbolic variables:

```python
# Create control variables
thrust = ox.Control("thrust", shape=(3,))

# Set bounds
thrust.min = np.array([0, 0, 0])
thrust.max = np.array([10, 10, 10])

# Set initial guess for SCP (shape: (n_nodes, control_shape))
thrust.guess = np.repeat(
    np.expand_dims(np.array([0, 0, 5]), axis=0),
    n_nodes, axis=0
)

# Collect all controls into a list
controls = [thrust]
```

## Dynamics
Dynamics are defined as a dictionary mapping state names to their time derivatives using symbolic expressions:

```python
# Physical parameters
m = 1.0  # Mass
g = -9.81  # Gravity

# Define dynamics using symbolic expressions
dynamics = {
    "position": velocity,
    "velocity": thrust / m + np.array([0, 0, g]),
}
```

The symbolic expressions support standard Python operators:
- Arithmetic: `+`, `-`, `*`, `/`, `**`
- Matrix multiplication: `@`
- Comparisons: `<=`, `>=`, `==`
- Indexing: `[...]`
- Transpose: `.T`

Common symbolic functions include:
- `ox.linalg.Norm()`: Vector/matrix norms
- `ox.linalg.Diag()`: Diagonal matrices
- `ox.spatial.QDCM()`: Quaternion to DCM
- `ox.spatial.SSM()`: Skew-symmetric matrix
- `ox.spatial.SSMP()`: Skew-symmetric matrix product

!!! Note
    Under the hood, symbolic expressions are compiled using JAX, so use `jax.numpy` for numerical constants and functions when needed.

## Time Definition
Define a `Time` object:

```python
# Fixed time horizon
time = ox.Time(
    initial=0.0,
    final=10.0,
    min=0.0,
    max=10.0,
)

# Minimum time problem
time = ox.Time(
    initial=0.0,
    final=("minimize", 10.0),  # Minimize final time with initial guess of 10.0
    min=0.0,
    max=20.0,
)
```

## Costs
You can choose states to minimize or maximize using the tuple syntax in boundary conditions:

```python
# Minimize a state component at the final time
energy = ox.State("energy", shape=())
energy.final = ("minimize", 0.0)

# Maximize a state component
reward = ox.State("reward", shape=())
reward.final = ("maximize", 100.0)
```

## Constraints
Constraints are created using symbolic expressions with comparison operators.

### Continuous Constraints
Continuous constraints are enforced over time intervals using `ox.ctcs()`:

```python
# Box constraints on states
constraints = []
for state in states:
    constraints.extend([
        ox.ctcs(state <= state.max),
        ox.ctcs(state.min <= state)
    ])

# Custom path constraints
max_speed = 10.0
constraints.append(ox.ctcs(ox.linalg.Norm(velocity) <= max_speed))

# Obstacle avoidance (distance >= safe_distance)
obstacle_center = ox.Parameter("obs_center", shape=(3,), value=np.array([5, 5, 5]))
safe_distance = 2.0
diff = position - obstacle_center
constraints.append(ox.ctcs(diff.T @ diff >= safe_distance**2))
```

### Discrete Constraints
Discrete constraints are enforced at specific nodes using the `.at()` method:

```python
# Waypoint constraint at node 10
target = np.array([5, 5, 5])
constraints.append(
    (position == target).at([10])
)

# Constraint at multiple nodes
constraints.append(
    (ox.linalg.Norm(velocity) <= 1.0).at([0, 5, 10])
)

# Convex constraints can be marked for better performance
constraints.append(
    (position[2] >= 0).convex().at([15])
)
```

## Parameters
Parameters allow values to be updated at runtime without recompiling:

```python
# Create parameters
obs_center = ox.Parameter("obs_center", shape=(3,), value=np.array([1.0, 2.0, 3.0]))
obs_radius = ox.Parameter("obs_radius", shape=(), value=0.5)

# Use in constraints
diff = position - obs_center
constraints.append(
    ox.ctcs(diff.T @ diff >= obs_radius**2)
)

# Update parameter values later
# problem.parameters["obs_center"] = new_center_value
```

## Initial Guess
While not strictly necessary for the initial guess to be dynamically feasible or satisfy constraints, a good guess helps the solver converge faster and avoid local minima.

For state trajectories, linear interpolation is a good starting point:

```python
position.guess = np.linspace(position.initial, position.final, n_nodes)
velocity.guess = np.zeros((n_nodes, 3))
```

For control trajectories, use constant values:

```python
thrust.guess = np.repeat(
    np.expand_dims(np.array([0, 0, 5]), axis=0),
    n_nodes, axis=0
)
```

## Problem Instantiation
Instantiate the problem with all components:

```python
problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n_nodes,
)
```

## Configure SCP Weights
The weights are used to scale the cost, trust region, and dynamic feasibility. A good place to start is to set `lam_cost = 0`, `lam_vc = 1E1` and `w_tr = 1E0`. Then you can slowly increase the cost weight and decrease the trust region weight until you find a good balance.

```python
problem.settings.scp.w_tr = 1E0      # Weight on the Trust Region
problem.settings.scp.lam_cost = 0E0  # Weight on the Cost
problem.settings.scp.lam_vc = 1E1    # Weight on the Virtual Control Objective
```

If you have nonconvex nodal constraints then you will also need to include `problem.settings.scp.lam_vb = 1E0`.

## Running the Problem
To solve the trajectory optimization problem:

1. Initialize the problem:
   ```python
   problem.initialize()
   ```

2. Solve the problem:
   ```python
   results = problem.solve()
   ```

3. Post-process the solution:
   ```python
   results = problem.post_process(results)
   ```

4. Access the solution:
   ```python
   # Extract state and control trajectories
   position_trajectory = results["position"]
   velocity_trajectory = results["velocity"]
   thrust_trajectory = results["thrust"]
   time_vector = results["time"]
   ```