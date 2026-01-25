# 6DoF Line-of-Sight Guidance

<a href="https://colab.research.google.com/drive/1b3NEx288h4r4HuvCOj-fexmt90PPhKUw?usp=sharing" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

Awesome! Now that we have a basic understanding of how to use OpenSCvx, let's solve a more complex problem. In this example we will be solving a 6DoF line-of-sight guidance problem. This example comes from a RA-L paper of mine which you can find [here](https://openscvx.github.io/papers/los/). The problem is more complex than the previous example, but demonstrates the power of OpenSCvx's symbolic expression layer.

In this problem, it is still a minimum time problem, but now there are 10 gates in which the drone must pass through in a predefined sequence while maintaining a line-of-sight to several key points throughout the entire trajectory. The problem can be expressed as follows:

$$
\begin{align}
\min_{x,u, t}\ &t_f, \\
\mathrm{s.t.}\ &\dot{x}(t) = f(t, x(t),u(t)) & \forall t\in[t_i, t_f], \\
& \lVert A_{\mathrm{cone}} C(q_{\mathcal{S}\to\mathcal{B}})C(q_{\mathcal{B}\to\mathcal{I}}(t))(r^{\mathrm{kp},i}_{\mathcal{I}} - r_{\mathcal{I}}(t))\rVert_\rho - c^\top C(q_{\mathcal{S}\to\mathcal{B}})C(q_{\mathcal{B}\to\mathcal{I}}(t))(r^{\mathrm{kp}, i}_{\mathcal{I}} - r_{\mathcal{I}}(t)) \leq 0 & \forall i \in [0, N_\mathrm{kp}], \forall t\in[t_i, t_f],\\
& \lVert A_{\mathrm{gate}} (r(t_i) - r^{i}_{\mathrm{gate}})\rVert_\infty \leq 1 & \forall i\in[0, N_\mathrm{gates}],\\
& x(t) \leq x_{\mathrm{max}}, x(t) \geq x_{\mathrm{min}} & \forall t\in[t_i, t_f],\\
& u(t) \leq u_{\mathrm{max}}, u(t) \geq u_{\mathrm{min}} & \forall t\in[t_i, t_f],\\
& x(0) = x_\mathrm{init}, \\
& p(t_f) = p_\mathrm{terminal}, \\
\end{align}
$$

where the state vector is the same as before, $x = \begin{bmatrix} p^\top & v^\top & q^\top & w^\top \end{bmatrix}^\top$. The control vector is also quite famaliar, $u = \begin{bmatrix}f^\top & \tau^\top \end{bmatrix}^\top$. The function $f(t, x(t),u(t))$ describes the dynamics of the drone. 

### LoS Contraint Formulation
The constraints are where things get a little more interesting. First we have the line of sight (LoS) constraint. I find it easiest to internally break it down into the following two components,

1. 
    A transformation component which take the location of a keypoint in the inertial frame, $r^{\mathrm{kp},i}_{\mathcal{I}}$, and expresses it in the sensor frame, $r^{\mathrm{kp},i}_{\mathcal{S}}$, as follows,

    $$ r^{\mathrm{kp},i}_{\mathcal{S}} = C(q_{\mathcal{S}\to\mathcal{B}})C(q_{\mathcal{B}\to\mathcal{I}}(t))(r^{\mathrm{kp},i}_{\mathcal{I}} - r_{\mathcal{I}}(t))$$

2.  A norm cone component expressed as follows,

    $$\lVert A_{\mathrm{C}} r^{\mathrm{kp},i}_{\mathcal{S}}\rVert_\rho \leq c^\top r^{\mathrm{kp},i}_{\mathcal{S}}$$

The long expression for the LoS constraint is obtained by simply plugging the first expression into the second. 

### Gate Constraint Formulation
The gate constraints are a little more straightforward and are notably convex.

$$\lVert A_{\mathrm{gate}} (r(t_i) - r^{i}_{\mathrm{gate}})\rVert_\infty \leq 1$$

The gate itself is assumed to be square, hence the $\infty$-norm but the user could certinaly choose a different norm. The only complication is that they are not path constraints, meaning I only want to enforce them at one single time instant as opposed to the entire trajecory and to make matters worse, the time instant is not known a priori. One could fix this but that would very likely lead to non-optimal solutions with respect to minimum time. 

## Imports
You'll need to import a few libraries to get started. The following code will import the necessary libraries for the example:

```python
import numpy as np
import numpy.linalg as la
import jax.numpy as jnp

import openscvx as ox
from openscvx import Problem
from openscvx.utils import rot, gen_vertices
```

Note how we import `openscvx as ox` - this gives us access to the symbolic expression system including `ox.State`, `ox.Control`, `ox.Parameter`, and symbolic operations.

## Problem Definition
Lets first define the number of discretization nodes and an initial guess for ToF.

```python
n = 33            # Number of discretization nodes
total_time = 40.0 # Initial ToF Guess for the simulation
```

## State Definition
With the new symbolic expression layer, we define each state component separately as a symbolic variable. Each state can have bounds, initial conditions, and final conditions specified as attributes.

```python
# Define state components
position = ox.State("position", shape=(3,))  # 3D position [x, y, z]
position.max = np.array([200.0, 100, 50])
position.min = np.array([-200.0, -100, 15])
position.initial = np.array([10.0, 0, 20])
position.final = [10.0, 0, 20]

velocity = ox.State("velocity", shape=(3,))  # 3D velocity [vx, vy, vz]
velocity.max = np.array([100, 100, 100])
velocity.min = np.array([-100, -100, -100])
velocity.initial = np.array([0, 0, 0])
velocity.final = [("free", 0), ("free", 0), ("free", 0)]

attitude = ox.State("attitude", shape=(4,))  # Quaternion [qw, qx, qy, qz]
attitude.max = np.array([1, 1, 1, 1])
attitude.min = np.array([-1, -1, -1, -1])
attitude.initial = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]
attitude.final = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]

angular_velocity = ox.State("angular_velocity", shape=(3,))  # Angular velocity [wx, wy, wz]
angular_velocity.max = np.array([10, 10, 10])
angular_velocity.min = np.array([-10, -10, -10])
angular_velocity.initial = [("free", 0), ("free", 0), ("free", 0)]
angular_velocity.final = [("free", 0), ("free", 0), ("free", 0)]

# Collect all states into a list
states = [position, velocity, attitude, angular_velocity]
```

The `("free", value)` tuple syntax indicates that a boundary condition is not fixed - the optimizer is free to choose the value, with `value` serving as an initial guess.

## Control Definition
Similar to states, we define control components as symbolic variables:

```python
# Define control components
thrust_force = ox.Control("thrust_force", shape=(3,))  # Thrust forces [fx, fy, fz]
thrust_force.max = np.array([0, 0, 4.179446268 * 9.81])
thrust_force.min = np.array([0, 0, 0])
thrust_force.guess = np.repeat(np.array([[0.0, 0, 10]]), n, axis=0)

torque = ox.Control("torque", shape=(3,))  # Control torques [tau_x, tau_y, tau_z]
torque.max = np.array([18.665, 18.665, 0.55562])
torque.min = np.array([-18.665, -18.665, -0.55562])
torque.guess = np.zeros((n, 3))

# Collect all controls into a list
controls = [thrust_force, torque]
```

## Problem Parameters
We will need to define a few parameters to describe the gates, sensor and keypoints for the problem.

### Sensor Paramters
Here we define the parameters we'll use to model the sensor with as follows,

```python
alpha_x = 6.0                                        # Angle for the x-axis of Sensor Cone
alpha_y = 6.0                                        # Angle for the y-axis of Sensor Cone
A_cone = np.diag([1 / np.tan(np.pi / alpha_x),
                  1 / np.tan(np.pi / alpha_y),
                  0,])                               # Conic Matrix in Sensor Frame
c = jnp.array([0, 0, 1])                             # Boresight Vector in Sensor Frame
norm_type = 2                                        # Norm Type
R_sb = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # Rotation Matrix from Sensor to Body Frame
```

### Gate Parameters
Here we define the parameters we'll use to model the gates with as follows,

```python
def gen_vertices(center, radii):
    """
    Obtains the vertices of the gate.
    """
    vertices = []
    vertices.append(center + rot @ [radii[0], 0, radii[2]])
    vertices.append(center + rot @ [-radii[0], 0, radii[2]])
    vertices.append(center + rot @ [-radii[0], 0, -radii[2]])
    vertices.append(center + rot @ [radii[0], 0, -radii[2]])
    return vertices


n_gates = 10                              # Number of gates
gate_centers = [                          # Center of the gates
    np.array([ 59.436,  0.0000, 20.0000]),
    np.array([ 92.964, -23.750, 25.5240]),
    np.array([ 92.964, -29.274, 20.0000]),
    np.array([ 92.964, -23.750, 20.0000]),
    np.array([130.150, -23.750, 20.0000]),
    np.array([152.400, -73.152, 20.0000]),
    np.array([ 92.964, -75.080, 20.0000]),
    np.array([ 92.964, -68.556, 20.0000]),
    np.array([ 59.436, -81.358, 20.0000]),
    np.array([ 22.250, -42.672, 20.0000]),
]

radii = np.array([2.5, 1e-4, 2.5])                 # Radii of the gates
A_gate = rot @ np.diag(1 / radii) @ rot.T
A_gate_cen = []
for center in gate_centers:
    center[0] = center[0] + 2.5
    center[2] = center[2] + 2.5
    A_gate_cen.append(A_gate @ center)             # Premultiplying A_gate @ center to ensure OCP is DPP compliant
n_per_gate = 3                                     # Number of nodes between each gate
gate_nodes = np.arange(n_per_gate, n, n_per_gate)  # Which node to enforce the gate constraint
vertices = []
for center in gate_centers:
    vertices.append(gen_vertices(center, radii))
```

### Keypoint Parameters
We can randomly generate some keypoints for the drone to observe. The keypoints are assumed to be in the inertial frame and can be generated as follows,

```python
n_subs = 10                          # Number of keypoints
init_poses = []
np.random.seed(0)
for i in range(n_subs):
    init_pose = np.array([100.0, -60.0, 20.0])
    init_pose[:2] = init_pose[:2] + np.random.random(2) * 20.0
    init_poses.append(init_pose)

init_poses = init_poses

```

## Dynamics
With the symbolic expression layer, we can define dynamics using natural mathematical notation. The dynamics are expressed as a dictionary mapping state names to their time derivatives as symbolic expressions:

```python
# Physical parameters
m = 1.0  # Mass of the drone
g_const = -9.18  # Gravity constant
J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of inertia

# Normalize quaternion for dynamics
q_norm = ox.linalg.Norm(attitude)
attitude_normalized = attitude / q_norm

# Define dynamics as dictionary mapping state names to their derivatives
J_b_inv = 1.0 / J_b
J_b_diag = ox.linalg.Diag(J_b)

dynamics = {
    "position": velocity,
    "velocity": (1.0 / m) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
                + np.array([0, 0, g_const], dtype=np.float64),
    "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
    "angular_velocity": ox.linalg.Diag(J_b_inv) @ (
        torque - ox.spatial.SSM(angular_velocity) @ J_b_diag @ angular_velocity
    ),
}
```

The symbolic expressions use:
- `ox.linalg.Norm()`: Compute vector norms
- `ox.linalg.Diag()`: Create diagonal matrices
- `ox.spatial.QDCM()`: Quaternion to direction cosine matrix
- `ox.spatial.SSMP()`: Skew-symmetric matrix product
- `ox.spatial.SSM()`: Skew-symmetric matrix
- Standard operators: `+`, `-`, `*`, `/`, `@` (matrix multiplication) 

## Constraints
In this problem, we have both continuous constraints (enforced over intervals) and discrete constraints (enforced at specific nodes). The symbolic expression layer makes constraint definition intuitive and readable.

### Continuous Constraints
First, we define a symbolic function for the line-of-sight (LoS) constraint:

```python
def g_vp(p_s_I, x_pos, x_quat):
    """Symbolic sensor visibility constraint function."""
    p_s_s = R_sb @ ox.spatial.QDCM(x_quat).T @ (p_s_I - x_pos)
    return ox.linalg.Norm(A_cone @ p_s_s, ord=norm_type) - (c.T @ p_s_s)
```

Now we can create constraints using symbolic expressions and the convenient `ox.ctcs()` wrapper:

```python
constraints = []

# Add box constraints for all states
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Add visibility constraints using symbolic expressions
for pose in init_poses:
    constraints.append(ox.ctcs(g_vp(pose, position, attitude) <= 0.0))
```

The `ox.ctcs()` function wraps a symbolic constraint expression and applies Continuous-Time Constraint Satisfaction (CTCS), which handles path constraints that must be satisfied over continuous intervals.

### Discrete Constraints
Gate constraints are only enforced at specific nodes in the trajectory. We use the `.at()` method to specify which nodes:

```python
# Add gate constraints using symbolic expressions
for node, cen in zip(gate_nodes, A_gate_cen):
    A_gate_const = A_gate
    cen_const = cen

    # Gate constraint: ||A @ pos - c||_inf <= 1
    gate_constraint = (
        (ox.linalg.Norm(A_gate_const @ position - cen_const, ord="inf") <= 1.0)
        .convex()
        .at([node])
    )
    constraints.append(gate_constraint)
```

The `.convex()` method marks the constraint as convex for the solver, and `.at([node])` specifies that this constraint is only enforced at a specific discrete node.

## Initial Guess
For complex problems, we need a sophisticated initial guess. We interpolate positions through each gate and compute attitudes to point the sensor at the keypoints:

```python
# Initialize position guess - linear interpolation through gates
position_bar = np.linspace(position.initial, position.final, n)
i = 0
origins = [position.initial]
ends = []
for center in gate_centers:
    origins.append(center)
    ends.append(center)
ends.append(position.final)
gate_idx = 0
for _ in range(n_gates + 1):
    for k in range(n // (n_gates + 1)):
        position_bar[i] = origins[gate_idx] + (k / (n // (n_gates + 1))) * (
            ends[gate_idx] - origins[gate_idx]
        )
        i += 1
    gate_idx += 1

# Modify attitude to point sensor at targets
R_sb = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
b = R_sb @ np.array([0, 1, 0])
attitude_bar = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))

for k in range(n):
    # Average keypoint positions
    kp = np.mean(init_poses, axis=0)
    a = kp - position_bar[k]

    # Compute quaternion to align sensor with relative position vector
    q_xyz = np.cross(b, a)
    q_w = np.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2) + np.dot(a, b)
    q_no_norm = np.hstack((q_w, q_xyz))
    q = q_no_norm / la.norm(q_no_norm)
    attitude_bar[k] = q

# Set all guesses
position.guess = position_bar
velocity.guess = np.zeros((n, 3))
attitude.guess = attitude_bar
angular_velocity.guess = np.zeros((n, 3))
```

## Time Definition
For minimum-time problems, we define a `Time` object:

```python
time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),  # Minimize final time with initial guess
    min=0.0,
    max=total_time,
)
```

The `("minimize", total_time)` tuple indicates that final time is a decision variable to be minimized, with `total_time` as the initial guess.

## Problem Instantiation
Now we instantiate the `Problem` with our symbolic expressions:

```python
problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
)
```

## Additional Parameters
We can define the PTR weights and other parameters as follows.

!!! tip
    Tuning is probably one of the hardest things to do when working with these type of algorithms. There are some approaches to automate this process (which will soon be included in OpenSCvx once they are published). A good place to start is to set ```lam_cost = 0```, ```lam_vc = 1E1``` and ```w_tr = 1E0```. Then you can slowly increase the cost weight and decrease the trust region weight until you find a good balance.

```python
problem.settings.scp.w_tr = 2e0                     # Weight on the Trust Reigon
problem.settings.scp.lam_cost = 1e-1                # Weight on the Cost
problem.settings.scp.lam_vc = 1e1                   # Weight on the Virtual Control
problem.settings.scp.ep_tr = 1e-3                   # Trust Region Tolerance
problem.settings.scp.ep_vc = 1e-8                   # Virtual Control Tolerance
problem.settings.scp.cost_drop = 10                 # SCP iteration to relax cost weight
problem.settings.scp.cost_relax = 0.8               # Minimal Time Relaxation Factor
problem.settings.scp.w_tr_adapt = 1.4               # Trust Region Adaptation Factor
problem.settings.scp.w_tr_max_scaling_factor = 1e2  # Maximum Trust Region Weight
```

Let's also set some propagation parameters:

```python
problem.settings.prp.dt = 0.1  # Time step of the nonlinear propagation
```

## Plotting
We generally leave plotting up to users as it's application-specific. Here we package relevant information into a dictionary for later visualization:

```python
plotting_dict = {
    "vertices": vertices,
    "n_subs": n_subs,
    "alpha_x": alpha_x,
    "alpha_y": alpha_y,
    "R_sb": R_sb,
    "init_poses": init_poses,
    "norm_type": norm_type,
}
```

## Running the Simulation
To run the simulation, follow these steps:

1. Initialize the problem:
   ```python
   problem.initialize()
   ```

2. Solve the problem:
   ```python
   results = problem.solve()
   ```

3. Post-process the solution for verification and plotting:
   ```python
   results = problem.post_process(results)
   results.update(plotting_dict)
   ```

4. Visualize the results:
   ```python
   from examples.plotting import plot_animation
   plot_animation(results, problem.settings).show()
   ```

## Key Takeaways

This example demonstrates the power of OpenSCvx's symbolic expression layer:

1. **Declarative Problem Definition**: States, controls, and dynamics are defined using natural mathematical notation with symbolic expressions.

2. **Operator Overloading**: Standard Python operators (`+`, `-`, `*`, `/`, `@`, `<=`, `==`) work directly on symbolic expressions, making code readable and intuitive.

3. **Automatic Differentiation**: The symbolic layer automatically handles differentiation during compilation to solver-specific formats.

4. **Flexible Constraint Specification**: Continuous constraints use `ox.ctcs()`, while discrete constraints use `.at([nodes])` - both work seamlessly with symbolic expressions.

5. **Shape Safety**: The expression system validates tensor dimensions before optimization, catching errors early.