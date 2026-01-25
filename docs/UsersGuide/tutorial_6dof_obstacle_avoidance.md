# 6DoF Obstacle Avoidance

<a href="https://colab.research.google.com/drive/1xLPC_UJWC35oPRIAY3vkxi8WEYnHCysQ?usp=sharing" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

This example demonstrates how to use OpenSCvx to solve a trajectory optimization problem in which a drone will navigate around obstacles to fly from point A to point B in minimum time. We will solve this problem in 6DoF, meaning there is 6 degrees of freedom in the problem, mainly 3 translational and 3 rotational degrees. Mathematically we can express this problem as the following,

$$
\begin{align}
\min_{x,u, t}\ &t_f, \\
\mathrm{s.t.}\ &\dot{x}(t) = f(t, x(t),u(t)) & \forall t\in[t_i, t_f], \\
& 1- (p(t) - p^i_{\mathrm{obs}})^\top A^i_\mathrm{obs} (r(t) - r^i_{\mathrm{obs}}) \leq 0  & \forall t\in[t_i, t_f], \forall i\in[0, N_\mathrm{obs}],\\
& x(t) \leq x_{\mathrm{max}}, x(t) \geq x_{\mathrm{min}} & \forall t\in[t_i, t_f],\\
& u(t) \leq u_{\mathrm{max}}, u(t) \geq u_{\mathrm{min}} & \forall t\in[t_i, t_f],\\
& x(0) = x_\mathrm{init}, \\
& p(t_f) = p_\mathrm{terminal}, \\
\end{align}
$$

where the state vector $x$ is expressed as  $x = \begin{bmatrix} r^\top & v^\top & q^\top & w^\top \end{bmatrix}^\top$. $p$ denotes the position of the drone, $v$ is the velocity, $q$ is the quaternion, $w$ is the angular velocity. The control vector $u$ is expressed as $u = \begin{bmatrix}f^\top & \tau^\top \end{bmatrix}^\top$. Here $f$ is the force in the body frame and $\tau$ is the torque of the body frame relative to the inertial frame. The function $f(t, x(t),u(t))$ describes the dynamics of the drone. The term $1- (r(t) - r^i_{\mathrm{obs}})^\top A^i_\mathrm{obs} (r(t) - r^i_{\mathrm{obs}})$ describes the obstacle avoidance constraints for $N_\mathrm{obs}$ number of obstacles, where $A_\mathrm{obs}$ is a positive definite matrix that describes the shape of the obstacle.

## Imports
You'll need to import a few libraries to get started:

```python
import numpy as np
import jax.numpy as jnp

import openscvx as ox
from openscvx import Problem
from openscvx.utils import generate_orthogonal_unit_vectors
```

## Problem Definition
Lets first define the number of discretization nodes and an initial guess for ToF.

```python
n = 6             # Number of discretization nodes
total_time = 4.0  # Initial ToF Guess for the simulation
```

## State Definition
We define each state component separately as a symbolic variable:

```python
# Define state components
position = ox.State("position", shape=(3,))  # 3D position [x, y, z]
position.max = np.array([200.0, 10, 20])
position.min = np.array([-200.0, -100, 0])
position.initial = np.array([10.0, 0, 2])
position.final = [-10.0, 0, 2]

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

## Control Definition
Similarly, we define control components as symbolic variables:

```python
# Define control components
thrust_force = ox.Control("thrust_force", shape=(3,))  # Thrust forces [fx, fy, fz]
thrust_force.max = np.array([0, 0, 4.179446268 * 9.81])
thrust_force.min = np.array([0, 0, 0])
initial_control = np.array([0.0, 0.0, thrust_force.max[2]])
thrust_force.guess = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)

torque = ox.Control("torque", shape=(3,))  # Control torques [tau_x, tau_y, tau_z]
torque.max = np.array([18.665, 18.665, 0.55562])
torque.min = np.array([-18.665, -18.665, -0.55562])
torque.guess = np.zeros((n, 3))

# Collect all controls into a list
controls = [thrust_force, torque]
```

## Dynamics
To describe the dynamics of the drone, lets first introduce some notation to describe in what frame quantities are being represented in. A quantity expressed in the frame $\mathcal{A}$ is denoted by the subscript $\Box_{\mathcal{A}}$. To parameterize the attitude of frame $\mathcal{B}$ with respect to frame $\mathcal{A}$, the unit quaternion, $q_{\mathcal{A} \to \mathcal{B}} \in \mathcal{S}^3$ where $\mathcal{S}^3\subset\mathbb{R}^4$ is the unit 3-sphere, is used. Here the inertial and body frames are denoted by $\mathcal{I}$ and $\mathcal{B}$ respectively. The dynamics of the drone can be expressed as follows:

$$
\begin{align*}
    % \label{eq:6dof_def}
    & \dot{r}_\mathcal{I}(t) = v_\mathcal{I}(t),\\
    & \dot{v}_\mathcal{I}(t) = \frac{1}{m}\left(C(q_{\mathcal{B \to I}}(t)) f_{ \mathcal{B}}(t)\right) + g_{\mathcal{I}},\\
    & \dot{q}_{\mathcal{I}\to \mathcal{B}} = \frac{1}{2} \Omega(\omega_\mathcal{B}(t))  q_{\mathcal{I \to B}}(t),\\
    & \dot{\omega}_\mathcal{B}(t) =  J_{\mathcal{B}}^{-1} \left(M_{\mathcal{B}}(t) - \left[\omega_\mathcal{B}(t)\times\right]J_{\mathcal{B}} \omega_\mathcal{B}(t) \right),
\end{align*} 
$$

where the operator $C:\mathcal{S}^3\mapsto SO(3)$ represents the direction cosine matrix (DCM), where $SO(3)$ denotes the special orthogonal group.

For a vector $\xi \in \mathbb{R}^3$, the skew-symmetric operators $\Omega(\xi)$ and $[\xi \times]$ are defined as follows:

$$
\begin{align}
[\xi \times] = \begin{bmatrix} 0 & -\xi_3 & \xi_2 \\ \xi_2 & 0 & -\xi_1 \\ -\xi_2 & \xi_1 & 0 \end{bmatrix}, \
\Omega(\xi) = \begin{bmatrix} 0 & -\xi_1 & \xi_2 & \xi_3 \\ \xi_1 & 0 & \xi_3 & -\xi_2 \\ \xi_2 & -\xi_3 & 0 & \xi_1 \\ \xi_3 & \xi_2 & -\xi_1 & 0 \end{bmatrix}
\end{align}
$$

OpenSCvx provides symbolic functions for these operations. We can express the dynamics as a dictionary mapping state names to their derivatives:

```python
# Physical parameters
m = 1.0  # Mass of the drone
g_const = -9.18
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

The symbolic functions used here are:
- `ox.linalg.Norm()`: Compute vector norms
- `ox.linalg.Diag()`: Create diagonal matrices
- `ox.spatial.QDCM()`: Quaternion to direction cosine matrix (DCM)
- `ox.spatial.SSMP()`: Skew-symmetric matrix product $\Omega(\xi)$
- `ox.spatial.SSM()`: Skew-symmetric matrix $[\xi \times]$

## Constraints
First, let's define the obstacle parameters. We'll use ellipsoidal obstacles parameterized by positive definite matrices:

```python
A_obs = []
radius = []
axes = []

# Default values for the obstacle centers
obstacle_center_positions = [
    np.array([-5.1, 0.1, 2]),
    np.array([0.1, 0.1, 2]),
    np.array([5.1, 0.1, 2]),
]

# Define obstacle centers as parameters for runtime updates
obstacle_centers = [
    ox.Parameter("obstacle_center_1", shape=(3,), value=obstacle_center_positions[0]),
    ox.Parameter("obstacle_center_2", shape=(3,), value=obstacle_center_positions[1]),
    ox.Parameter("obstacle_center_3", shape=(3,), value=obstacle_center_positions[2]),
]

# Randomly generate obstacle shapes
np.random.seed(0)
for _ in obstacle_center_positions:
    ax = generate_orthogonal_unit_vectors()
    axes.append(generate_orthogonal_unit_vectors())
    rad = np.random.rand(3) + 0.1 * np.ones(3)
    radius.append(rad)
    A_obs.append(ax @ np.diag(rad**2) @ ax.T)
```

Now we can create constraints using symbolic expressions:

```python
# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Add obstacle constraints using symbolic expressions
for center, A in zip(obstacle_centers, A_obs):
    A_const = A

    # Obstacle constraint: (pos - center)^T @ A @ (pos - center) >= 1
    diff = position - center
    obstacle_constraint = ox.ctcs(1.0 <= diff.T @ A_const @ diff)
    constraints.append(obstacle_constraint)
```

The `ox.ctcs()` function applies Continuous-Time Constraint Satisfaction, ensuring the constraints are satisfied over continuous intervals, not just at discrete nodes. Note that `ox.Parameter()` allows obstacle centers to be updated at runtime without recompiling the problem.

## Initial Guess
We set initial guesses for the state and control trajectories:

```python
# Set initial guesses
position.guess = np.linspace(position.initial, position.final, n)
velocity.guess = np.linspace(velocity.initial, [0, 0, 0], n)
attitude.guess = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
angular_velocity.guess = np.zeros((n, 3))
```

!!! tip
    The Penalized Trust Region method does not require the initial guess to be dynamically feasible or satisfy constraints. However, a guess close to the solution reduces iterations and improves numerical stability. Linear interpolation for states and constant values for controls are good starting points.

## Time Definition
For minimum-time problems, we define a `Time` object:

```python
time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
)
```

## Problem Instantiation
Now we instantiate the `Problem`:

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
We can configure solver parameters for better performance:

```python
problem.settings.prp.dt = 0.01                   # Time step of the nonlinear propagation
problem.settings.scp.lam_vb = 1e0                # Virtual buffer weight
problem.settings.scp.w_tr_adapt = 1.8            # Trust region adaptation factor
problem.settings.scp.w_tr = 1e1                  # Trust region weight
problem.settings.scp.lam_cost = 1e1              # Weight on the cost
problem.settings.scp.lam_vc = 1e2                # Weight on virtual control
problem.settings.scp.cost_drop = 4               # SCP iteration to relax cost
problem.settings.scp.cost_relax = 0.5            # Cost relaxation factor
```

## Plotting
We package relevant information for visualization:

```python
plotting_dict = {
    "obstacles_centers": obstacle_center_positions,
    "obstacles_axes": axes,
    "obstacles_radii": radius,
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