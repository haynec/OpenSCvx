# 08 Receding Horizon Drone Racing: Model Predictive Contouring Control

In this tutorial we step away from single-shot trajectory optimization and into the world of receding-horizon control.
We will implement _model predictive contouring control_ (MPCC), a technique for tracking a reference path as fast as possible, and in doing so introduce several new concepts: multi-objective Mayer costs using integrator states with `ox.Minimize` and `ox.Maximize`, cubic spline interpolation within the symbolic graph via `ox.Cinterp`, as well as the receding-horizon MPC loop pattern.

We will start from a simple 2D example with an analytical reference path to introduce the core concepts before generalizing to discrete reference paths. Finally, we will build a full drone racing MPCC problem that tracks a pre-solved time-optimal trajectory through gates and around dynamic obstacles.

This tutorial covers:

- Using OpenSCvx in a receding-horizon setting
- Multi-objective Mayer costs with `ox.Minimize` and `ox.Maximize` and per-state cost weighting with `lam_cost`
- Cubic spline lookup in the symbolic graph with `ox.Cinterp`
- Two-phase planning: offline trajectory optimization + online MPCC tracking

## Model Predictive Contouring Control

### The Problem

Before jumping into the MPCC implementation, let us first introduce the key concepts of the formulation. Readers in a hurry can jump straight to the [Dubins Car](#a-simple-example-dubins-car-on-a-circle) and [drone racing](#drone-racing-mpcc) sections and get to coding.

In the previous tutorials we solved trajectory optimization problems offline and obtained high-performance solutions.
In practice, however, we may need a closed-loop controller to _execute_ these trajectories in the presence of unmodeled dynamics, unseen obstacles, _etc._
A natural approach is to track the pre-solved trajectory using a receding-horizon controller to allow for replanning to account for changes in the environment in a closed-loop fashion.
In this tutorial we will implement _model predictive contouring control_ (MPCC), a formulation originally introduced by [Lam _et al._ 2010](https://doi.org/10.1016/j.automatica.2009.10.027) and applied to drone racing by [Romero _et al._ 2022](https://arxiv.org/abs/2108.13205) (see also [Krinner _et al._ 2024](https://arxiv.org/abs/2403.17551v2)). We will follow along with the implementation of Romero.

Standard MPC formulations track a trajectory sampled at fixed times; the controller iteratively tries to be at a specific position at a specific moment.
MPCC instead works with a spatial reference _path_ and lets the optimizer decide how fast to traverse it.
This decoupling of _where_ from _when_ gives the controller freedom to choose the best speed profile rather than being locked to a pre-determined time schedule or heuristic.

### The Ideal Cost

Consider a desired reference path $\mathbf{p}^d(\theta)$ parametrized by arc length $\theta$, and denote the system's position at time step $k$ as $\mathbf{p}_k$.
The _contour error_, the shortest distance from the current position to the reference path, is defined as:

$$
e_k^c = \min_{\theta} \| \mathbf{p}_k - \mathbf{p}^d(\theta) \|_2
$$

The ideal MPCC cost minimizes this contour error while maximizing progress $\theta_N$ along the path:

$$
J = \sum_{k=0}^{N} q_c \, (e_k^c)^2 - q_\theta \, \theta_N
$$

where $q_c > 0$ is the contour weight and $q_\theta > 0$ weights progress maximization (we will minimize our cost term so a negative sign incentivizes progress).
However, this formulation is not computationally tractable. Computing $e_k^c$ requires solving a nested optimization at every time step, making it unsuitable for use inside an online controller.

### Approximating the Contour Error

To avoid the nested minimization, Romero _et al._ introduce an _approximate progress_ variable $\hat{\theta}$ with its own dynamics:

$$
\hat{\theta}_{k+1} = \hat{\theta}_k + v_{\hat{\theta}} \, \Delta t
$$

where $v_{\hat{\theta}}$ is a new virtual control input determined by the optimizer.
Since $\hat{\theta}_N = \hat{\theta}_0 + \sum v_{\hat{\theta},k} \Delta t$, maximizing final progress is equivalent to maximizing the sum of progress rates, giving us the progress cost:

$$
J_\theta = \sum_{k=0}^{N} \left( -q_\theta \cdot v_{\hat{\theta},k} \right)
$$

Instead of searching for the closest point on the path, we simply evaluate the reference at $\hat{\theta}_k$ and work with the resulting position error:

$$
\mathbf{e}(\hat{\theta}_k) = \mathbf{p}_k - \mathbf{p}^d(\hat{\theta}_k)
$$

This error vector can be decomposed into two orthogonal components using the unit tangent $\mathbf{t}(\hat{\theta}_k)$ of the reference path (which has unit norm because the path is arc-length parametrized):

- **Lag error** $\hat{\mathbf{e}}^l$: the projection of $\mathbf{e}$ onto the tangent direction $\mathbf{t}$, measuring how far the approximate progress $\hat{\theta}$ is from the true closest point
- **Contour error** $\hat{\mathbf{e}}^c$: the component of $\mathbf{e}$ in the normal plane, approximating the true contour error $e_k^c$

The scalar lag error is the dot product:

$$
\hat{e}^l = \mathbf{e} \cdot \mathbf{t}(\hat{\theta}_k)
$$

and the contour error magnitude follows from the Pythagorean relationship $\mathbf{e} = \hat{\mathbf{e}}^l + \hat{\mathbf{e}}^c$ with the two components orthogonal:

$$
\| \hat{\mathbf{e}}^c \|^2 = \| \mathbf{e} \|^2 - (\hat{e}^l)^2
$$

Then the contour cost can be written as

$$
J_c = \sum_{k=0}^{N} \left( q_c \cdot \| \hat{\mathbf{e}}^c(\hat{\theta}_k) \|^2 \right)
$$

### Why the Lag Error Must Be Small

These are approximations — $\hat{\mathbf{e}}^c$ is not the same as the true contour error $e_k^c$, and $\hat{\theta}_k$ is not the same as the true optimal progress $\theta_k^* = \arg\min_\theta \| \mathbf{p}_k - \mathbf{p}^d(\theta) \|$.
How good are they?

Intuitively, the approximation quality is controlled entirely by the lag error.
When $\hat{\mathbf{e}}^l = \mathbf{0}$, the position $\mathbf{p}_k$ lies in the normal plane at $\hat{\theta}_k$, which means $\hat{\theta}_k$ is exactly the closest point on the path  $\hat{\theta}_k = \theta_k^*$ and $\| \hat{\mathbf{e}}^c \| = e_k^c$ (see the proof in [Romero _et al._ 2022, Proposition 1](https://arxiv.org/abs/2108.13205)).

This gives us a practical recipe: by minimizing the lag error term, we keep the optimizer "honest", preventing it from simply choosing a high $v_\theta$. By keeping the lag error small, the contour error approximation stays accurate.
We enforce this by adding the lag error to the cost function with a high weight $q_l$:

$$
J_l = \sum_{k=0}^{N} \left( q_l \cdot (\hat{e}^l(\hat{\theta}_k))^2 \right)
$$

The lag term is not just another tracking objective — it is what makes the entire approximation scheme valid.

### The MPCC Cost

Combining the contour error, lag error, and progress maximization gives us the full MPCC cost.

$$
J_{\textrm{MPCC}} = J_c + J_l + J_\theta = \sum_{k=0}^{N} \left( q_c \| \hat{\mathbf{e}}^c(\hat{\theta}_k) \|^2 + q_l (\hat{e}^l(\hat{\theta}_k))^2 - q_\theta \, v_{\hat{\theta},k} \right)
$$

### Encoding as a Multi-Objective Mayer Cost in OpenSCvx

As we saw in [Tutorial 01](01_hello_world_brachistochrone.md), OpenSCvx uses the Mayer form, expressing the cost purely as a function of the final state rather than as running Lagrange costs.
The transformation from one form to the other is trivial; we will simply include the cost terms as _integrator states_ which are summed continuously over the time horizon rather than just at the discrete nodes.
We define these new states as the integrated lag error $s_l$ and integrated contour error $s_c$

$$
\dot{s}_l = \left(\hat{e}^l\right)^2, \qquad \dot{s}_c = \| \hat{\mathbf{e}}^c \|^2
$$

with $s_l(0) = s_c(0) = 0$ and add these to the state vector.
Similarly, we track the progress $\hat{\theta}$ as a continuous state defined as the integral of the progress rate $v_{\hat\theta}$, which is appended to the control vector:

$$
\dot{\hat\theta} = v_{\hat\theta}
$$

with $\hat\theta(0)$ initialized to match the current arc-length position along the reference path at $t=0$.

So far our problems have had a single cost term, typically minimizing $t_f$.
MPCC requires balancing _three_ competing objectives: minimize contour error, minimize lag error, and maximize progress.
Minimizing $s_l(t_f)$ and $s_c(t_f)$ in the Mayer cost is then equivalent to minimizing the running integrals.
Similarly, we maximize the final progress $\hat{\theta}(t_f)$ to encourage forward motion.
We can write the full MPCC cost as follows.

$$
\begin{align*}
J_{\textrm{MPCC, OpenSCvx}} &=  q_c \cdot s_c + q_l \cdot s_l- q_\theta \cdot \hat{\theta} \\
&= \int_0^{t_f} \left( q_c \cdot \| \hat{\mathbf{e}}^c(t) \|^2 + q_l \cdot \left(\hat{e}^l(t)\right)^2  - q_\theta \cdot v_{\hat\theta}(t) \right) dt
\end{align*}
$$

!!! tip
    The cost could mathematically equivalently be constructed as a single integrated state appended to the state vector.
    We could define a single state $s$ defined as:
    
    $$
    \dot{s} = q_c \cdot \| \hat{\mathbf{e}}^c \|^2 + q_l \cdot \left(\hat{e}^l\right)^2  - q_\theta \cdot v_{\hat\theta}
    $$

    and append _that_ to the state vector instead.
    However, for convenience and for ease of tuning we will separate out the individual terms.

## A Simple Example: Dubins Car on a Circle

Let's build up the MPCC step by step, starting from a simple case: a 2D Dubins car tracking a circular reference path.
In this case, the reference is analytical (no discrete points needed), so we can focus entirely on getting the MPCC structure correct before expanding to the more general case.

### States

The physical states are position and heading, as you'd expect for a Dubins car.
The MPCC-specific states are progress, and the two cost integrators:

```python
import numpy as np
import openscvx as ox

n_mpc = 8
horizon_duration = 1.5

R_circle = 3.0
total_arc_length = 2 * np.pi * R_circle

# Physical states
position = ox.State("position", shape=(2,))
position.min = [-10.0, -10.0]
position.max = [10.0, 10.0]
position.initial = [R_circle, 0.0]
position.final = [ox.Free(0.0), ox.Free(0.0)]

heading = ox.State("heading", shape=(1,))
heading.min = [-4 * np.pi]
heading.max = [4 * np.pi]
heading.initial = [0.0]
heading.final = [ox.Free(0.0)]

# MPCC states
progress = ox.State("progress", shape=(1,))
progress.min = [-0.5 * total_arc_length]
progress.max = [1.5 * total_arc_length]
progress.initial = [0.0]
progress.final = [ox.Maximize(0.0)]

lag_sum = ox.State("lag_sum", shape=(1,))
lag_sum.min = [0.0]
lag_sum.max = [1e1]
lag_sum.initial = [0.0]
lag_sum.final = [ox.Minimize(0.0)]

contour_sum = ox.State("contour_sum", shape=(1,))
contour_sum.min = [0.0]
contour_sum.max = [1e1]
contour_sum.initial = [0.0]
contour_sum.final = [ox.Minimize(0.0)]
```

A few things to note here.

1. All the physical states have `ox.Free` finals — in a receding horizon setting, there is no fixed terminal condition.
2. `progress.final` is set to `ox.Maximize(0.0)`.
This tells OpenSCvx to maximize the final value of this state, which is how we encode the $-q_\theta \cdot v_{\hat{\theta}}$ term: maximizing the integral of the progress rate is the same as maximizing the final progress.
3. `lag_sum.final` and `contour_sum.final` are set to `ox.Minimize(0.0)`.
Since these states integrate the squared errors over the horizon, minimizing their final values is equivalent to the running cost we defined above.
1. The bounds on `progress` and `heading` are padded beyond a single lap: progress ranges from $-0.5L$ to $1.5L$ and heading allows multiple full rotations. This extra room is needed for the warm-starting and guess-shifting strategy in the MPC loop, which we will explain in detail in a [later section](#the-mpc-loop).

### Controls

```python
speed = ox.Control("speed", shape=(1,))
speed.min = [0.0]
speed.max = [10.0]
speed.guess = np.full((n_mpc, 1), 5.0)

angular_rate = ox.Control("angular_rate", shape=(1,))
angular_rate.min = [-5.0]
angular_rate.max = [5.0]

progress_rate = ox.Control("progress_rate", shape=(1,))
progress_rate.min = [0.0]
progress_rate.max = [10.0]
progress_rate.guess = np.full((n_mpc, 1), 5.0)
```

The `progress_rate` control is the virtual input $v_{\hat{\theta}}$ from the formulation.
It is constrained to be non-negative so the car can only move forward along the path.

### Reference Path and Error Decomposition

For a circle of radius $R$ centered at the origin, the reference path parametrized by arc length $s$ is:

$$
\mathbf{p}^d(s) = R \begin{bmatrix} \cos(s/R) \\ \sin(s/R) \end{bmatrix}, \qquad \mathbf{t}(s) = \begin{bmatrix} -\sin(s/R) \\ \cos(s/R) \end{bmatrix}
$$

In code, the symbolic expressions for the reference position and tangent are constructed directly from the `progress` state:

```python
angle = progress[0] / R_circle
p_ref = ox.Concat(
    R_circle * ox.Cos(angle),
    R_circle * ox.Sin(angle),
)
tangent = ox.Concat(
    -ox.Sin(angle),
    ox.Cos(angle),
)
```

The error decomposition follows the formulation exactly:

```python
e = position - p_ref

# Lag: projection onto tangent
lag_scalar = ox.Sum(e * tangent)
lag_cost = lag_scalar**2

# Contour: Pythagorean decomposition
contour_cost = ox.Max(ox.Sum(e * e) - lag_scalar**2, 0.0)
```

!!! note "Numerical Considerations"
    - We use `ox.Sum(e * e)` rather than `ox.linalg.Norm(e)**2` to avoid the derivative singularity $\partial \| \mathbf{e} \| / \partial \mathbf{e} = \mathbf{e} / \| \mathbf{e} \|$ at $\mathbf{e} = 0$
    - The `ox.Max(..., 0.0)` clamp handles floating-point cases where the subtraction goes slightly negative.

### Dynamics

```python
dynamics = {
    "position": ox.Concat(
        speed[0] * ox.Sin(heading[0]),
        speed[0] * ox.Cos(heading[0]),
    ),
    "heading": angular_rate[0],
    "progress": progress_rate,
    "lag_sum": lag_cost,
    "contour_sum": contour_cost,
}
```

The first three lines are the physical dynamics and the progress kinematics.
The last two lines are the cost integrators: `lag_sum` accumulates the lag cost, and `contour_sum` accumulates the contour cost.
Because these states have `ox.Minimize` finals, the optimizer will drive these integrals down while simultaneously maximizing `progress`.
The relative weighting between these objectives is handled separately via `lam_cost` in the problem setup, keeping the dynamics clean.

### Problem Setup

```python
states = [position, heading, progress, lag_sum, contour_sum]
controls = [speed, angular_rate, progress_rate]

constraints = []
for state in [position, heading]:
    constraints.extend([
        ox.ctcs(state <= state.max),
        ox.ctcs(state.min <= state),
    ])

t = ox.Time(
    initial=0.0,
    final=horizon_duration,
    min=0.0,
    max=horizon_duration,
    uniform_time_grid=True,
)

Q_LAG = 1e0
Q_CONTOUR = 1e-1
Q_PROGRESS = 1e-1

problem_mpc = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=t,
    constraints=constraints,
    N=n_mpc,
    algorithm={
        "autotuner": ox.ConstantProximalWeight(),
        "lam_cost": {
            "lag_sum": Q_LAG,
            "contour_sum": Q_CONTOUR,
            "progress": Q_PROGRESS,
        },
    },
)
```

Two differences from the single-shot problems in earlier tutorials.

1. The time is _fixed_ (`final=horizon_duration` with no `"minimize"`), in this case the horizon length is a design parameter, not an optimization variable.
2. The `lam_cost` dictionary assigns a weight to each cost state's Mayer contribution.
Following Romero _et al._ the lag weight `Q_LAG` is highest to keep the progress approximation accurate, while `Q_CONTOUR` and `Q_PROGRESS` are tuned lower.
This separates the cost tuning from the dynamics formulation — you can adjust these weights without touching the dynamics dictionary.

!!! note
    We use `ox.ConstantProximalWeight` as the autotuner here to keep the weights constant. This simplifies tuning and reliability during development. You may see better performance with other autotuning methods depending on your problem

### The MPC Loop

Now we get to the receding-horizon loop itself.
The pattern is: solve the problem, extract the solution, advance the initial conditions by one step, and shift the guess forward for warm-starting.

#### Initialization

Before the first solve we need a feasible initial guess.
For the analytical circle we assume a constant speed along the reference and compute position, heading, and controls directly:

```python
def set_initial_guess(theta_start: float = 0.0):
    ref_speed = 5.0
    arc_guess = np.linspace(
        theta_start, theta_start + ref_speed * horizon_duration, n_mpc
    )
    angle_guess = arc_guess / R_circle

    position.guess = np.column_stack([
        R_circle * np.cos(angle_guess),
        R_circle * np.sin(angle_guess),
    ])
    heading.guess = (-angle_guess + np.pi / 2).reshape(-1, 1)
    progress.guess = arc_guess.reshape(-1, 1)
    lag_sum.guess = np.zeros((n_mpc, 1))
    contour_sum.guess = np.zeros((n_mpc, 1))

    speed.guess = np.full((n_mpc, 1), ref_speed)
    angular_rate.guess = np.full((n_mpc, 1), -ref_speed / R_circle)
    progress_rate.guess = np.full((n_mpc, 1), ref_speed)
```

!!! note
    While this demonstrates the fundamentals of creating the initial guess, this will change once we swap to a discrete reference trajectory.

#### The Core Loop

Solving the MPC problem follows the familiar pattern of initialize -> solve -> post process, except now the solving is done in a loop.
For this initial example we will simply repeat this `max_steps` times:

```python
problem_mpc.initialize()

for step in range(max_steps):
    problem_mpc.reset() # Clear SCP history
    results = problem_mpc.solve()
    results = problem_mpc.post_process()
    nodes = results.nodes

    # Update initial conditions and warm-start for next iteration
    update_initial_conditions(nodes)
    shift_guess(nodes)
```

The `update_initial_conditions(...)` and `shift_guess(...)` functions handle the all-important tasks of preparing the previous solution as the next solve's warm-start and are explained below.

#### Advancing Initial Conditions and Wrapping

To update the initial condition, we take node 1 of the previous solution as the new initial state.
Because we are building up towards a racing example, which may run many laps, we will need to "wrap" the periodic values (`progress`, `heading`).
Otherwise, these would accumulate lap over lap and eventually reach their maximum values.
While we _could_ simply set the bounds high enough that we do not reach them within our expected lap count, this is not good engineering and could lead to numerical issues.

To handle this, we subtract complete laps via floor division to keep progress in range.

```python
wrap_offset = (nodes["progress"][1, 0] // total_arc_length) * total_arc_length
progress.initial = np.array([nodes["progress"][1, 0] - wrap_offset])
```

The state bounds cover $[-0.5L, 1.5L]$, allowing the tail of the MPC horizon to exceed a single lap before the system crosses the lap threshold and is reset.

Heading is wrapped similarly (to the nearest $2\pi$) to avoid numerical issues in trig functions.

```python
hdg_wrap = np.round(nodes["heading"][1, 0] / (2 * np.pi)) * (2 * np.pi)
heading.initial = nodes["heading"][1] - hdg_wrap
```

!!! note
    We use `np.round` for heading (which can be negative) and `//` (floor division) for progress (monotonically increasing). The same wrapping is applied in `shift_guess` below to keep the entire horizon consistent.

The cost integrators are reset to zero each horizon as they measure error _within_ the current horizon, not cumulatively.

Combining all of this results in:

```python
def update_initial_conditions(nodes: dict):
    position.initial = nodes["position"][1]

    hdg_wrap = np.round(nodes["heading"][1, 0] / (2 * np.pi)) * (2 * np.pi)
    heading.initial = nodes["heading"][1] - hdg_wrap

    wrap_offset = (nodes["progress"][1, 0] // total_arc_length) * total_arc_length
    progress.initial = np.array([nodes["progress"][1, 0] - wrap_offset])

    lag_sum.initial = np.array([0.0])
    contour_sum.initial = np.array([0.0])
```

#### Warm-Starting: Shifting the Guess

Since we advanced by one time step, the solution at nodes $[1, \ldots, N-1]$ is a good guess for nodes $[0, \ldots, N-2]$ of the next problem.
For the final node we will need a heuristic approach to append a new guess.
While this doesn't need to be perfect, the SCP algorithm will do its thing, it can have a large effect on convergence performance.
It can be worthwhile to think of a good strategy for _your_ specific problem.
For this analytical example, we fill in the new final node with an Euler extrapolation from the last node's state and control.

```python
def shift_guess(nodes: dict):
    dt = horizon_duration / (n_mpc - 1)

    # Extrapolate a new final node
    pos_last = nodes["position"][-1]
    hdg_last = nodes["heading"][-1, 0]
    spd_last = nodes["speed"][-1, 0]
    ar_last = nodes["angular_rate"][-1, 0]
    pr_last = nodes["progress_rate"][-1, 0]

    ext_pos = pos_last + dt * spd_last * np.array([
        np.sin(hdg_last), np.cos(hdg_last)
    ])
    ext_hdg = hdg_last + dt * ar_last
    ext_prog = nodes["progress"][-1, 0] + dt * pr_last

    # Shift states forward and apply wrapping
    shifted_progress = np.vstack([nodes["progress"][1:], [[ext_prog]]])
    wrap_offset = (nodes["progress"][1, 0] // total_arc_length) * total_arc_length
    shifted_progress -= wrap_offset

    shifted_heading = np.vstack([nodes["heading"][1:], [[ext_hdg]]])
    hdg_wrap = np.round(nodes["heading"][1, 0] / (2 * np.pi)) * (2 * np.pi)
    shifted_heading -= hdg_wrap

    position.guess = np.vstack([nodes["position"][1:], [ext_pos]])
    heading.guess = shifted_heading
    progress.guess = shifted_progress
    lag_sum.guess = np.zeros((n_mpc, 1))
    contour_sum.guess = np.zeros((n_mpc, 1))

    # Controls: shift forward, repeat last value for the new final node
    speed.guess = np.vstack([nodes["speed"][1:], nodes["speed"][-1:]])
    angular_rate.guess = np.vstack([
        nodes["angular_rate"][1:], nodes["angular_rate"][-1:]
    ])
    progress_rate.guess = np.vstack([
        nodes["progress_rate"][1:], nodes["progress_rate"][-1:]
    ])
```

The wrap offsets are computed from node 1 of the previous solution (which becomes node 0 of the new problem) so the whole horizon shifts consistently.
Controls are shifted forward with the last value repeated; cost integrator guesses are reset to zero.

## Discrete Reference Paths

We've now set up the core elements of the MPC(C) problem: initializing, solving, and shifting the initial condition and guess.
However, in the general case we will not have an analytical reference path.
Instead, we may have a set of discrete points from a pre-solved trajectory, a motion capture recording, or hand-placed waypoints.
Let us examine an extension of the analytical Dubins car MPCC problem, sampling $M$ discrete points from the circle as our discrete path.
We need a way to evaluate the reference position and tangent at arbitrary progress values within the symbolic expression graph.

Linear interpolation between sparse points creates kinks in the tangent field, which cause oscillations in the contour error.
We need something smoother: cubic spline interpolation.

### Periodicity via Tiling

Before building the splines, we need to do our chores and handle periodicity.
Just as in the analytical case, we must accommodate the horizon looking ahead of the current position for progress values in $[-0.5L, 1.5L]$ which necessitates us "tiling" the data to cover this range.
We tile the single-lap data by replicating the position samples at shifted arc-length offsets to create an `s_data` that covers the full range of the progress state:

```python
M = 30  # Samples per lap
# s_lap, px_lap, py_lap: single-lap arc-length and position arrays

s_min, s_max = -0.5 * total_arc_length, 1.5 * total_arc_length
n_before = int(np.ceil(-s_min / total_arc_length))
n_after = int(np.ceil(s_max / total_arc_length))
tile_laps = range(-n_before, n_after + 1)

s_data = np.concatenate([s_lap + k * total_arc_length for k in tile_laps])
px_data = np.tile(px_lap, len(tile_laps))
py_data = np.tile(py_lap, len(tile_laps))
```

### Cubic Splines

OpenSCvx supports cubic splines via the `ox.Cinterp(x, xp, fp)` operator.
Given breakpoints `xp` and values `fp` (both NumPy arrays, known at problem construction time), it evaluates the natural cubic spline at the symbolic query point `x`.
This is the symbolic analog of `scipy.interpolate.CubicSpline` — in fact, it uses the same coefficients under the hood.

With the tiled data in hand, building the reference path is straightforward:

```python
p_ref = ox.Concat(
    ox.Cinterp(progress[0], s_data, px_data),
    ox.Cinterp(progress[0], s_data, py_data),
)
```

The tangent field requires a bit more care for this example.
We compute the derivative of the cubic spline at the breakpoints using SciPy, normalize to get unit tangents, and then interpolate _those_ with a second `ox.Cinterp`:

```python
from scipy.interpolate import CubicSpline as _CS

_dpx = _CS(s_data, px_data)(s_data, 1)  # Derivative at breakpoints
_dpy = _CS(s_data, py_data)(s_data, 1)
_tnorm = np.sqrt(_dpx**2 + _dpy**2)
tx_data = _dpx / _tnorm
ty_data = _dpy / _tnorm

tangent = ox.Concat(
    ox.Cinterp(progress[0], s_data, tx_data),
    ox.Cinterp(progress[0], s_data, ty_data),
)
```

!!! tip
    If our reference trajectory also contains velocity information (which it will soon), this allows us to define the tangent direction without resorting to numerical differentiation

The rest of the MPCC formulation, error decomposition, cost integrators, dynamics, MPC loop, is identical to the analytical case with only minor adjustments to the heuristic components.
This is the beauty of the approach: swapping from an analytical reference to a discrete one is purely a matter of changing how `p_ref` and `tangent` are constructed.

### Initialization with Discrete References

With an analytical reference we could compute position and heading guesses from closed-form expressions.
With a discrete reference we instead interpolate the reference data arrays directly:

```python
def set_initial_guess(theta_start: float = 0.0, ref_speed: float = 5.0):
    arc_guess = np.linspace(
        theta_start, theta_start + ref_speed * horizon_duration, n_mpc
    )

    # Position: interpolate from reference sample arrays
    position.guess = np.column_stack([
        np.interp(arc_guess, s_data, px_data),
        np.interp(arc_guess, s_data, py_data),
    ])

    # Heading: infer from reference segment directions
    seg_idx = np.searchsorted(s_data, arc_guess, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(s_data) - 2)
    seg_dp = np.column_stack([
        px_data[seg_idx + 1] - px_data[seg_idx],
        py_data[seg_idx + 1] - py_data[seg_idx],
    ])
    hdg_guess = np.arctan2(seg_dp[:, 0], seg_dp[:, 1])
    heading.guess = hdg_guess.reshape(-1, 1)
    heading.initial = np.array([hdg_guess[0]])

    progress.guess = arc_guess.reshape(-1, 1)
    progress_rate.guess = np.full((n_mpc, 1), ref_speed)
    lag_sum.guess = np.zeros((n_mpc, 1))
    contour_sum.guess = np.zeros((n_mpc, 1))
```

Position is looked up via `np.interp` (linear is fine for the _guess_).
For heading, we need the direction of travel at each guess point — but with a discrete reference we don't have an analytical tangent.
Instead, `np.searchsorted` finds which reference segment each `arc_guess` value falls on, and we compute the heading from the direction of that segment (`seg_dp`).
The `shift_guess` and `update_initial_conditions` functions are structurally identical to the analytical case — wrapping and extrapolation depend on the dynamics, not the reference representation.

## Drone Racing MPCC

Now we can put everything together for the real application: a drone racing through gates and around obstacles, tracked by MPCC.
This is a two-phase problem:

1. **Offline**: Solve a time-optimal trajectory through the gates (single-shot, as in [Tutorial 02](02_drone_racing_constraints.md))
2. **Online**: Use MPCC to track the solved trajectory in a receding-horizon loop

### Time-Optimal Reference Trajectory

The offline problem is a standard time-optimal drone racing problem identical to what we built in [Tutorial 02](02_drone_racing_constraints.md): a 3D double integrator flying through a drone race course with gate constraints.
We have slightly modified the problem to enforce loop closure: initial and terminal positions and velocities are constrained to be identical.
We won't repeat the full setup here; the important output is the solved trajectory:

```python
problem_traj.initialize()
results_traj = problem_traj.solve()
results_traj = problem_traj.post_process()

ref_pos = results_traj.trajectory["position"]   # (N_dense, 3)
ref_vel = results_traj.trajectory["velocity"]   # (N_dense, 3)
ref_time = results_traj.trajectory["time"].flatten()
```

The post-processed trajectory gives us a dense set of position, velocity, and time samples.
We need to convert this to an arc-length parametrization for the MPCC reference.

Arc length is computed by integrating the speed:

```python
ref_speeds = np.linalg.norm(ref_vel, axis=1)
ds = ref_speeds[:-1] * np.diff(ref_time)
s_lap = np.concatenate([[0.0], np.cumsum(ds)])
total_arc_length = s_lap[-1]
```

Since the trajectory has loop closure (initial position equals final position), we drop the last point before tiling to avoid duplicate arc-length values at the boundary:

```python
s_lap = s_lap[:-1]
ref_pos_lap = ref_pos[:-1]
```

The tiling and `ox.Cinterp` setup for position follow the same pattern as before, now in 3D:

```python
p_ref = ox.Concat(
    ox.Cinterp(progress[0], s_data, px_data),
    ox.Cinterp(progress[0], s_data, py_data),
    ox.Cinterp(progress[0], s_data, pz_data),
)
```

For the tangent field, we no longer need to differentiate the position spline — the reference trajectory already contains velocity data.
We normalize it to get unit tangents, tile it alongside the position data, and interpolate:

```python
tx_data = ref_vel[:-1, 0] / ref_speeds[:-1]
ty_data = ref_vel[:-1, 1] / ref_speeds[:-1]
tz_data = ref_vel[:-1, 2] / ref_speeds[:-1]
# (tile tx_data, ty_data, tz_data the same way as position)

tangent = ox.Concat(
    ox.Cinterp(progress[0], s_data, tx_data),
    ox.Cinterp(progress[0], s_data, ty_data),
    ox.Cinterp(progress[0], s_data, tz_data),
)
```

### Warm-Starting from the Reference

A nice property of the two-phase approach is that we have the full reference trajectory available for initialization.
Rather than guessing a constant speed or linear interpolation, we can look up the reference position, velocity, and force at the arc-length values corresponding to our horizon:

```python
def set_initial_guess(theta_start: float = 0.0):
    t_start = np.interp(theta_start, s_data, t_data)
    t_guess = np.linspace(t_start, t_start + horizon_duration, n_mpc)
    arc_guess = np.interp(t_guess, t_data, s_data)

    position.guess = np.column_stack([
        np.interp(arc_guess, s_data, px_data),
        np.interp(arc_guess, s_data, py_data),
        np.interp(arc_guess, s_data, pz_data),
    ])
    velocity.guess = np.column_stack([
        np.interp(arc_guess, s_data, vx_data),
        np.interp(arc_guess, s_data, vy_data),
        np.interp(arc_guess, s_data, vz_data),
    ])
    # ... similarly for force, progress_rate
```

This gives the MPCC solver an excellent starting point: the initial guess is already close to the optimal trajectory, so convergence is fast from the very first MPC step.

### Guess Shifting with the Reference Trajectory

In the Dubins car examples we extrapolated the appended final node with a crude Euler step from the last node's dynamics.
With the full reference trajectory available, we can do much better: look up the reference state at the extrapolated progress value.

```python
def shift_guess(nodes: dict):
    # Map the last node's progress to reference time, step forward, map back
    t_last = np.interp(nodes["progress"][-1, 0], s_data, t_data)
    ext_prog = np.interp(t_last + dt_mpc, t_data, s_data)

    # Look up reference state at the extrapolated progress
    ext_pos = np.array([
        np.interp(ext_prog, s_data, px_data),
        np.interp(ext_prog, s_data, py_data),
        np.interp(ext_prog, s_data, pz_data),
    ])
    ext_vel = np.array([
        np.interp(ext_prog, s_data, vx_data),
        np.interp(ext_prog, s_data, vy_data),
        np.interp(ext_prog, s_data, vz_data),
    ])
    # ... similarly for force, progress_rate

    # Shift and wrap as before
    shifted_progress = np.vstack([nodes["progress"][1:], [[ext_prog]]])
    wrap_offset = (nodes["progress"][1, 0] // total_arc_length) * total_arc_length
    shifted_progress -= wrap_offset

    position.guess = np.vstack([nodes["position"][1:], [ext_pos]])
    velocity.guess = np.vstack([nodes["velocity"][1:], [ext_vel]])
    progress.guess = shifted_progress
    # ...
```

The appended node now lies on the reference trajectory rather than being a rough dynamical extrapolation.
This is the same idea as `set_initial_guess` — query the reference at the right arc-length — applied at every MPC step.

### Additional Constraints and Extensions

#### Obstacle Avoidance

Because we are now running a closed-loop controller we can include additional constraints that were not present when calculating the time-optimal reference trajectory, for example, obstacles placed along the path

```python
for obs_center in obstacle_centers:
    constraints.append(
        ox.ctcs(obstacle_radius <= ox.linalg.Norm(position - obs_center))
    )
```

#### Progress-Dependent Gate Constraints

For the time-optimal reference trajectory we enforced the gates as convex nodal constraints, placing every k-th optimization node at a gate.
This is no longer possible in the receding-horizon setting; we cannot guarantee that any node lies at a particular gate.
While the contour error minimization encourages the closed-loop path to follow the reference trajectory through the gates, it does not guarantee it.

To generically constrain the MPCC optimizer to travel through the gates, we borrow the cone formulation from [tutorial 04](04_viewpoint_constraints.md) to create approach cone constraints for each gate.
These are offset such that the cone exactly touches the edges of the gate.
This problem would quickly become infeasible for multiple cones.
Therefore, we use a _progress-dependent_ condition to trigger a gates cone constraint when the drone is between the previous gate and the current gate.
We use an `ox.Cond`/`ox.All` statement as introduced in [tutorial 06](06_logic.md) to encode this logic statement.
Additionally, we add the constraint that $v \cdot n_{\textrm{gate}} <=0$ (the velocity projected onto the cone normal is negative) to ensure that the constraint is only active when the drone is flying towards a gate.
This allows the drone to "fly around the corner" before the cone constraint becomes active.

Finally, to speed up initialization of the MPCC problem we use `ox.Vmap` from [tutorial 03](03_obstacle_avoidance_vmap.md) to leverage data-parallelism for all of the gates.

Putting all this together results in our cone constraints:

```python
cone_constraints = ox.Vmap(
    lambda apex, R_gate, n_hat, s_gate, s_prev: ox.Cond(
        ox.All([
            progress[0] >= s_prev,
            progress[0] <= s_gate,
            ox.Sum(velocity * n_hat) <= 0.0,
        ]),
        g_gate_cone(apex, R_gate, position),
        -1.0,  # Inactive: return feasible value
    ),
    batch=[all_apexes, all_rotations, all_n_hats, all_s_gates, all_s_prevs],
)
constraints.append(ox.ctcs(cone_constraints <= 0.0))
```

#### Free Time Dilation

We can leverage the built-in time dilation of OpenSCvx to allow the optimizer to dilate time "as necessary" to improve performance while keeping a fixed time horizon, giving it one more knob to turn.
To ensure a consistent loop rate, we constrain that time at node 1 is always $t_{\textrm{MPCC}}$ in the future.

```python
t = ox.Time(
    initial=0.0,
    final=horizon_duration, # Still fixed!
    min=0.0,
    max=horizon_duration,
    # Note: uniform_time_grid is NOT set, allowing non-uniform spacing
)

# Pin node 1 at a fixed dt so the MPC loop rate is constant
constraints.append(
    (t == horizon_duration / (n_mpc - 1)).convex().at(1)
)
```

We also shift the time and time dilation guesses identically to the other states and controls:

```python
# Inside shift_guess(nodes):

# Time: shift forward and renormalize so the horizon starts at t=0
dtau = 1.0 / (n_mpc - 1)
ext_time = nodes["time"][-1, 0] + nodes["_time_dilation"][-1, 0] * dtau
shifted_time = np.vstack([nodes["time"][1:], [[ext_time]]])
shifted_time -= shifted_time[0]  # Re-zero the horizon
t.guess = shifted_time

# Time dilation: shift forward, repeat last value
t._time_dilation_control.guess = np.vstack(
    [nodes["_time_dilation"][1:], nodes["_time_dilation"][-1:]]
)
```

!!! note
    We have not done extensive comparisons on the benefits/costs of allowing for non-constant time-dilation in the MPCC problem.
    Depending on your problem, it may or may not be beneficial.
    At the very least it's a cool feature that can be enabled at minimal cost :)

## Key Takeaways

1. **Multi-objective Mayer costs**: `ox.Minimize` and `ox.Maximize` on state finals let you encode multiple competing running costs and rewards as integrator states. This is a general pattern that works for any Lagrange-to-Mayer conversion. The `lam_cost` dictionary provides per-state cost weighting, separating the tuning of relative objective importance from the dynamics formulation.
2. **`ox.Cinterp`**: Cubic spline interpolation within the symbolic graph enables smooth lookup of discrete reference data. Pre-computing the tangent field from the spline derivative and re-interpolating it avoids singularities and gives a clean tangent field.
3. **The MPC pattern**: `problem.reset()` → `problem.solve()` → advance initial conditions → shift guess. Warm-starting from the shifted previous solution is essential for fast convergence.
4. **Progress-dependent gate constraints**: Combining `ox.Cond`, `ox.All`, and `ox.Vmap` lets you encode constraints that activate only when the system is in a particular region of the path. This replaces the fixed node-to-gate assignment from single-shot problems with a formulation that works regardless of which nodes happen to be near a gate.
5. **Two-phase planning**: Solving the hard global problem offline (time-optimal trajectory through gates) and tracking it online with MPCC is a powerful decomposition. The MPCC handles local disturbances and model mismatch while the offline solution provides the global plan.

## Further Reading

- [Dubins Car MPCC Example (analytical circle)](../Examples/mpc/dubins_car_circle.md)
- [Dubins Car MPCC Example (discrete reference)](../Examples/mpc/dubins_car_circle_polytope.md)
- [3D Double Integrator MPCC Example](../Examples/mpc/double_integrator_polytope.md)
- [Drone Racing MPCC Example](../Examples/mpc/double_integrator_drone_racing.md)
- [Romero _et al._ (2022). "Model Predictive Contouring Control for Time-Optimal Quadrotor Flight." _IEEE Transactions on Robotics._](https://arxiv.org/abs/2108.13205)
- [Lam _et al._ (2010). "Model Predictive Contouring Control." _IEEE Conference on Decision and Control._](https://web.archive.org/web/20170811172607id_/http://people.eng.unimelb.edu.au/manziec/resources/Publications%20pdfs/10_Conf_Lam.pdf)
- [Krinner _et al._ (2024). "MPCC++: Model Predictive Contouring Control for Time-Optimal Flight with Safety Constraints." _Robotics: Science and Systems._](https://arxiv.org/abs/2403.17551v2))
- [Drone Racing: Constraints and 3-DOF Dynamics](02_drone_racing_constraints.md) — the single-shot trajectory used as the MPCC reference
- [Obstacle Avoidance: Vmap](03_obstacle_avoidance_vmap.md) — vectorized constraints used in the gate cone formulation
