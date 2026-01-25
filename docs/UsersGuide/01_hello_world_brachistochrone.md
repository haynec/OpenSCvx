# 01 Hello Brachistochrone

In this _hello world_ tutorial we will introduce the reader to the fundamental concepts and API needed to define and solve a problem using OpenSCvx as well as how the results can be accessed.
Our example of choice in this endeavor is the _Brachistochrone problem_, which we briefly describe below before delving into the implementation.
Finally, we will lay out some further reading and tease the next tutorial for the interested reader.

This tutorial covers:

- Problem creation
- Variable instantiation
- Dynamics and constraint definition
- Solving the problem
- Accessing results

## The Brachistochrone Problem

The [Brachistochrone problem](https://en.wikipedia.org/wiki/Brachistochrone_curve) is concerned with finding the fastest path between 2 points under the influence of gravity and without friction.
Bernouli originally posed the problem as:

> _Given two points A and B in a vertical plane, what is the curve traced out by a point acted on only by gravity, which starts at A and reaches B in the shortest time._

As with most good things in life, this can be written as an optimal control problem in the Mayer form as:

$$
\begin{align}
\min_{\mathbf{x}, \mathbf{u}, t_f}\ &t_f & \\
\mathrm{s.t.}\ &\dot{\mathbf{x}}(t) = f(\mathbf{x}(t), \mathbf{u}(t)) & \forall t\in[0, t_f], \quad &\textrm{dynamics} \\
&\mathbf{x}_{\min} \leq \mathbf{x}(t) \leq \mathbf{x}_{\max} & \forall t\in[0, t_f], \quad &\textrm{state bounds} \\
&\mathbf{u}_{\min} \leq \mathbf{u}(t) \leq \mathbf{u}_{\max} & \forall t\in[0, t_f], \quad &\textrm{control bounds} \\
&\mathbf{x}(0) = \mathbf{x}_{\mathrm{init}}, & & \textrm{initial}\\
&\mathbf{p}(t_f) = \mathbf{p}_{\mathrm{final}} & & \textrm{terminal}
\end{align}
$$

where the state $\mathbf{x} = [x, y, v]^\top$ consists of 2D position $\mathbf{p} = [x, y]^\top$ and speed $v$, the control $\mathbf{u} = \theta$ is the angle from vertical, and the dynamics are given by:

$$
f(\mathbf{x}, \mathbf{u}) = \begin{bmatrix} v \sin(\theta) \\ -v \cos(\theta) \\ g \cos(\theta) \end{bmatrix}
$$

Note that the terminal velocity is unconstrained.

This problem is particularly interesting to us because of two main reasons: 

1. It is a very nice, small toy problem we can use to introduce the core concepts of OpenSCvx. It is quick to formulate and quick to solve.
2. _It has an analytical solution._ The Brachistochrone problem was solved by none other than Isaac Newton in 1697, who showed that the optimal path is given by a cycloid, the curve traced by a point on the rim of a rolling wheel:

$$
\begin{align}
x(\phi) &= r(\phi - \sin\phi) \\
y(\phi) &= r(1 - \cos\phi)
\end{align}
$$

where $\phi \in [0, \phi_f]$ is the curve parameter and $r$ is the radius of the generating circle, determined by the boundary conditions. The minimum time of descent is:

$$
t_f^* = \sqrt{\frac{r}{g}} \phi_f
$$

Because of these advantageous properties the Brachistochrone problem is extensively leveraged as a [unit test](https://github.com/OpenSCvx/OpenSCvx/blob/main/tests/test_brachistochrone.py).
We would highly recommend that anyone setting out to develop some kind of optimization software do the same.
It may not be necessary _nor_ sufficient, but a lot of things have to be working properly to solve Brachistochrone problem.

## Creating an OpenSCvx Problem

During the development of OpenSCvx we spent a great deal of time trying to make the library as easy to work with as possible.
If it isn't easy to use, people will just write their own.
For that reason, we took inspiration from fantastic libraries such as [NumPy](https://github.com/numpy/numpy), [JAX](https://github.com/jax-ml/jax), and [CVXPY](https://github.com/cvxpy/cvxpy) and developed a symbolic expression system.

This not only allows us to keep the syntax close to the mathematical notation but also enables us to do lots of preprocessing, validation, canonicalization, and augmentation under the hood without the user ever having to know about it if they choose not to.
It also means that we can keep the syntax similar to popular libraries such as Numpy and JAX to reduce the learning curve for new users.

Before we begin defining our problem we can take care of our imports as well as some high-level parameters.
Typically, OpenSCvx is imported as `ox` for brevity. This will be the standard throughout these tutorials

```python
import openscvx as ox
```

Next, we can define our number of discrete decision nodes `n`, our initial guess for the completion time `total_time`, and define the gravitational acceleration `g`

```python
n = 2
total_time = 2.0
g = 9.81
```

Note that `total_time` is an _initial guess_ for the time. We will solve this as a free final-time problem and find the optimal solution.

### Variable Definition

Now, we can start implementing the Brachistochrone problem, starting by creating the necessary state and control variables.
Each `ox.State` can be instantiated with a name and shape such as

```python
position = ox.State("position", shape=(2,))  # 2D position [x, y]
```

Then, we need to define the bounds. In this example we arbitrarily choose a 10 by 10 box for our position

```python
position.max = [10.0, 10.0]
position.min = [0.0, 0.0]
```

Finally, we must define the initial and final conditions. In this case we will find the Brachistochrone curve from $[0, 10]$ to $[10, 5]$ and set the values accordingly:

```python
position.initial = np.array([0.0, 10.0])
position.final = [10.0, 5.0]
```

All of these values can be either raw Python lists, NumPy or JAX arrays.
We can then similarly define the velocity state and it's bounds, resulting in the following code defining both states:

```python
# Define state components
position = ox.State("position", shape=(2,))  # 2D position [x, y]
position.max = np.array([10.0, 10.0])
position.min = np.array([0.0, 0.0])
position.initial = np.array([0.0, 10.0])
position.final = [10.0, 5.0]

velocity = ox.State("velocity", shape=(1,))  # Scalar speed
velocity.max = np.array([10.0])
velocity.min = np.array([0.0])
velocity.initial = np.array([0.0])
velocity.final = [("free", 10.0)]

# Define list of all states (needed for Problem and constraints)
states = [position, velocity]
```

The `.final` value of Velocity is defined as a tuple `("free", 10.0)`. This sets the corresponding value as "free" for the optimizer to choose.
This is how we can encode that the final velocity is unconstrained. The numerical value is necessary as an initial guess.
When only a value is specified as in the case of `position`, the values are assumed to be fixed.

To summarize, we can use the following syntax to define the initial and final conditions:

- Fixed value: `value` or `("fixed", value)`
- Free variable: `("free", guess)` - Can be optimized within bounds
- Minimize: `("minimize", guess)` - Variable to be minimized
- Maximize: `("maximize", guess)` - Variable to be maximized

For our control `theta` we can follow a similar syntax to define the symbolic `ox.Control` object:

```python
# Define control
theta = ox.Control("theta", shape=(1,))  # Angle from vertical
theta.max = np.array([100.5 * jnp.pi / 180])
theta.min = np.array([0.0])
theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

controls = [theta]
```

Here, we do _not_ specify an initial or final value for the control but we _do_ need to specify an initial guess.
The initial guess must be of shape $(n_u \times n_{\mathrm{nodes}})$, providing an initial guess for each control at every node.
Technically, we can also provide a `.guess` for states. However, these default to a linear interpolation between initial and final conditions which is sufficient in most cases.

### Dynamics

Dynamics are defined as a dictionary mapping state names to their time derivatives using symbolic expressions:

```python
# Define dynamics as dictionary mapping state names to their derivatives
dynamics = {
    "position": ox.Concat(
        velocity[0] * ox.Sin(theta[0]),  # x_dot
        -velocity[0] * ox.Cos(theta[0]),  # y_dot
    ),
    "velocity": g * ox.Cos(theta[0]),
}
```

!!! Note
    Every state passed to the problem must have a matching element in the dynamics dictionary under the same name.

The symbolic expressions support standard Python operators:

- Arithmetic: `+`, `-`, `*`, `/`, `**`
- Matrix multiplication: `@`
- Comparisons: `<=`, `>=`, `==` (for constraint definitions, see below)
- Indexing: `[...]`
- Transpose: `.T`

Common symbolic functions include:

- `ox.Concat()`: Concatenation
- `ox.Sin()`/`ox.Cos()`: Trigonometric functions
- `ox.linalg.Norm()`: Vector/matrix norms

!!! Note
    Under the hood, symbolic expressions are compiled using JAX, so use `jax.numpy` for numerical constants and functions when needed.

### Constraints (Continuous)

We already defined the box constraints when we instantiated the variables.
These are enforced at the discrete decision nodes automatically.
However, OpenSCvx offers another way to enforce constraints: we can also enforce the constraints _between_ the discrete nodes.
We call this continuous-time constraint-satisfaction (CTCS).
While the full description of CTCS is beyond the scope of this tutorial, what's important is that this is handled internally without user interaction.

We can define the boundary constraints as inequalities _i.e._ `state.min <= state` and `state <= state.max`.
To mark these as CTCS constraints we simply wrap them in `ox.ctcs(...)`

```python
# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max),
            ox.ctcs(state.min <= state)
        ]
    )
```

This style of constraint definition as a list should feel familiar to CVXPY users.
Note that we do not need to specify continuous box constraints for the controls.
This is because, by default, the controls are interpolated using first-order hold.
Therefore, by constraining the discrete nodes to lie within the bounds we can trivially see that the continuous case is guaranteed as well.

We will explore the various forms of constraints supported by OpenSCvx more in the [next tutorial](02_drone_racing_constraints.md)

### Time

The last remaining piece we need is `Time`. OpenSCvx allows for free final-time problems with non-constant spacing.
We can define our time object similar to a `State` object, including the initial and final values as well as the min. and max. bounds.
Similar to the states the `final` value is defined as a tuple indicating that the final time is to be minimized as well as providing an initial guess for the total time.

```python
time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
)
```

OpenSCvx treats the `Time` object like any other state; it can be used to formulate time-dependent dynamics or constraints.
This is a more advanced subject for later tutorials.

### Defining the Problem

Now we have everything we need to instantiate the `Problem` with our `dynamics`, `states`, `controls`, `time`, `constraints`, as well as the number of nodes `N`

```python
problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
)
```

### Solving the Problem

Solving the problem is split into three steps:

1. `problem.initialize()` initializes the problem, validating, preprocessing, canonicalizing, and augmenting the symbolic expressions before lowering them to JAX and CVXPY code.
2. `problem.solve()` iteratively solves the OCP and generates the discrete state and control solution at the decision nodes.
3. `problem.post_process()` propagates the nodal solution at high temporal fidelity. This lets us generate high resolution trajectories decoupled from the number of optimization nodes.

```python
problem.initialize()
results = problem.solve()
results = problem.post_process()
```

Both `.solve()` and `.post_process()` return an `OptimizationResults` object with the former containing only the nodal solution while the latter also includes the high-fidelity trajectories.

### Accessing the results

But how can we easily access the results? Could we _somehow_ leverage the symbolic variables to make this easy?

Fear not; once a problem has been solved and post-processed we can access the results using the exact same variable names we defined earlier for our states and controls.
We can do so both for the discrete decision nodes in `results.nodes` and the high-resolution `results.trajectory` _e.g._

```python
pos_nodes = results.nodes["position"]
theta_traj = results.trajectory["theta"]
```

### Visualizing the Results

OpenSCvx provides built-in plotting utilities for quick visualization of your results. The simplest way to see your solution is with `plot_states()` and `plot_controls()`:

```python
from openscvx.plotting import plot_states, plot_controls

# Plot all state trajectories in a subplot grid
plot_states(results, ["position", "velocity"]).show()

# Plot control trajectories
plot_controls(results, ["theta"]).show()
```

These functions create interactive Plotly figures showing:

- **Green lines**: High-fidelity propagated trajectory
- **Cyan markers**: Discrete optimization nodes
- **Red dashed lines**: Variable bounds

For the Brachistochrone problem, you'll see the position trace out the characteristic cycloid curve, while the control angle `theta` smoothly varies from near-vertical at the start to nearly horizontal at the end.

For more advanced visualization options including 3D interactive plots, see [Tutorial 05: Visualization](05_visualization.md).

## Further Reading

- [Complete Brachistochrone Example](../Examples/abstract/brachistochrone.md)
- [Drone Racing: Constraints and 3DoF Dynamics](02_drone_racing_constraints.md)
- [Visualization: 2D Plots and 3D Interactive](05_visualization.md)

At this point you are well-equipped to go out and start constructing trajectory optimization problems.
If you are so-inclined you can dive into the [API reference documentation](../Reference/problem.md) or the [examples](../Examples/drone/drone_racing.md) and figure the rest out yourself.
For the interested reader, we will continue our guided tour of the various features in OpenSCvx by examining a slightly more interesting example which shows off different kinds of constraints we can define.
