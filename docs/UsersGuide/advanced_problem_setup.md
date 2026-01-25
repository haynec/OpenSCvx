# Advanced Problem Setup

## Using Parameters in Dynamics and Constraints

OpenSCvx allows you to define symbolic parameters that can be used in both dynamics and constraints. Parameters enable flexible, reusable problem definitions and can be updated at runtime without recompiling.

### Example: 3DoF Rocket Landing with Parameters

```python
import numpy as np
import openscvx as ox

# Define parameters for physical constants
g_e = 9.807  # Gravitational acceleration on Earth (m/s^2)

# Create symbolic parameters
I_sp = ox.Parameter("I_sp", value=225.0)
g = ox.Parameter("g", value=3.7114)
theta = ox.Parameter("theta", value=27 * np.pi / 180)

# Define states
position = ox.State("position", shape=(3,))
velocity = ox.State("velocity", shape=(3,))
mass = ox.State("mass", shape=(1,))

# Define control
thrust = ox.Control("thrust", shape=(3,))

# Use parameters in dynamics
g_vec = np.array([0, 0, 1], dtype=np.float64) * g

dynamics = {
    "position": velocity,
    "velocity": thrust / mass[0] - g_vec,
    "mass": -ox.linalg.Norm(thrust) / (I_sp * g_e * ox.Cos(theta)),
}
```

Parameters are automatically detected and handled by the problem - no need to manually collect or pass them.

### Using Parameters in Constraints

Parameters can be used in constraints just like in dynamics:

```python
# Define obstacle parameters
obs_center = ox.Parameter("obs_center", shape=(3,), value=np.array([100, 100, 50]))
obs_radius = ox.Parameter("obs_radius", value=50.0)

# Use in continuous constraint
diff = position - obs_center
constraints.append(
    ox.ctcs(diff.T @ diff >= obs_radius**2)
)

# Use in discrete constraint
constraints.append(
    (ox.linalg.Norm(position - obs_center) >= obs_radius).at([10])
)
```

### Updating Parameters at Runtime

Parameters can be updated between solves without recompiling:

```python
# Initial solve
problem.initialize()
results = problem.solve()

# Update parameter values
problem.parameters["obs_center"] = np.array([150, 150, 60])
problem.parameters["obs_radius"] = 60.0

# Resolve with new parameter values (no recompilation needed)
results = problem.solve()
```

## CTCS Constraints: Advanced Options

CTCS (Continuous-Time Constraint Satisfaction) constraints are enforced over continuous intervals using penalty functions.

### Penalty Function

You can specify a penalty function using the `penalty` argument. Built-in options include:

- `squared_relu` (default) - $\max(0, g)^2$
- `huber` - $\begin{cases} \frac{1}{2} g^2 & \text{if } |g| \leq \delta \\ \delta (|g| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$
- `smooth_relu` - $\|\max(0, g)+c\|-c$

Example:

```python
constraints.append(
    ox.ctcs(ox.linalg.Norm(position[:2]) <= 10.0, penalty="huber")
)
```

### Node-Specific Regions

To enforce a constraint only over a portion of the trajectory, use the `.over()` method:

```python
# Enforce constraint between nodes 3 and 8
constraint = (ox.linalg.Norm(position[:2]) <= 10.0).over(
    interval=(3, 8),
    penalty="squared_relu"
)
constraints.append(constraint)
```

### Multiple Augmented States

To associate different constraints with different augmented states, use the `idx` argument:

```python
# Box constraints use augmented state 0
constraints.extend([
    ox.ctcs(position <= position.max, idx=0),
    ox.ctcs(position.min <= position, idx=0),
])

# Thrust magnitude constraints use augmented state 1
constraints.extend([
    ox.ctcs(rho_min <= ox.linalg.Norm(thrust), idx=1),
    ox.ctcs(ox.linalg.Norm(thrust) <= rho_max, idx=1),
])

# Thrust pointing constraint uses augmented state 2
constraints.append(
    ox.ctcs(np.cos(theta_max) <= thrust[2] / ox.linalg.Norm(thrust), idx=2)
)
```

This allows different constraints to use separate virtual control and augmented state variables, which can improve convergence.

## Nodal Constraints: Advanced Options

Nodal constraints are enforced at specific discrete nodes in the trajectory.

### Convex and Nonconvex Constraints

For convex constraints, use the `.convex()` method:

```python
# Convex constraint - waypoint at node 10
target = np.array([100, 100, 50])
constraints.append(
    (ox.linalg.Norm(position - target, ord="inf") <= 1.0).convex().at([10])
)

# Convex constraint - altitude limit at node 15
constraints.append(
    (position[2] >= 10.0).convex().at([15])
)
```

For nonconvex constraints, simply use the constraint without `.convex()`:

```python
# Nonconvex constraint
constraints.append(
    (velocity.T @ velocity <= v_max**2).at([5, 10, 15])
)
```

### Node Specification

Use the `.at()` method to specify which nodes enforce the constraint:

```python
# Single node
constraints.append((position[2] >= 0).at([0]))

# Multiple nodes
constraints.append((ox.linalg.Norm(velocity) <= v_max).at([5, 10, 15, 20]))

# Final node
constraints.append((velocity == np.array([0, 0, 0])).at([n-1]))
```

### Combining with CTCS

You can enforce a constraint both continuously and at specific nodes:

```python
# Enforce continuously with CTCS
constraint = ox.linalg.Norm(position - target) >= safe_distance

# Also check nodally (optional, for verification)
constraints.append(
    constraint.over(interval=(0, n-1), check_nodally=True)
)
```

## Solver Settings

OpenSCvx provides extensive configuration options for tuning solver behavior.

### SCP Algorithm Settings

```python
# Trust region settings
problem.settings.scp.w_tr = 2e0                  # Trust region weight
problem.settings.scp.w_tr_adapt = 1.04           # Trust region adaptation factor
problem.settings.scp.w_tr_max_scaling_factor = 1e2  # Maximum trust region scaling

# Cost and virtual control weights
problem.settings.scp.lam_cost = 2.5e-1           # Weight on the cost objective
problem.settings.scp.lam_vc = 1.2e0              # Weight on virtual control
problem.settings.scp.lam_vb = 1e0                # Virtual buffer weight (for nonconvex nodal constraints)

# Convergence tolerances
problem.settings.scp.ep_tr = 1e-3                # Trust region tolerance
problem.settings.scp.ep_vc = 1e-8                # Virtual control tolerance

# Cost relaxation
problem.settings.scp.cost_drop = 10              # Iteration to start relaxing cost
problem.settings.scp.cost_relax = 0.8            # Cost relaxation factor
```

### Convex Solver Settings

```python
# Choose convex solver
problem.settings.cvx.solver = "CLARABEL"  # Options: "CLARABEL", "ECOS", "SCS", "MOSEK"

# Solver-specific arguments
problem.settings.cvx.solver_args = {
    "enforce_dpp": True,  # Data Parallel Processing for CLARABEL
}
```

### Integration Settings

```python
# Propagation time step
problem.settings.prp.dt = 0.01

# Integration method
problem.settings.dis.solver = "Dopri8"  # Options: "Dopri5", "Dopri8", "Tsit5", etc.
```

### Compilation and Caching

```python
# Save compiled JAX functions for faster subsequent runs
problem.settings.sim.save_compiled = True
```