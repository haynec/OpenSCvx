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
J_{\text{MPCC}} = \sum_{k=0}^{N} q_c \, (e_k^c)^2 - q_\theta \, \theta_N
$$

where $q_c > 0$ is the contour weight and $q_\theta > 0$ weights progress maximization (we will minimize our cost term so a negative sign incentivizes progress).
However, this formulation is not computationally tractable. Computing $e_k^c$ requires solving a nested optimization at every time step, making it unsuitable for use inside an online controller.

### Approximating the Contour Error

To avoid the nested minimization, Romero _et al._ introduce an _approximate progress_ variable $\hat{\theta}$ with its own dynamics:

$$
\hat{\theta}_{k+1} = \hat{\theta}_k + v_{\hat{\theta}} \, \Delta t
$$

where $v_{\hat{\theta}}$ is a new virtual control input determined by the optimizer.
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
\lVert \hat{\mathbf{e}}^c \rVert^2 = \lVert \mathbf{e} \rVert^2 - (\hat{e}^l)^2
$$

### Why the Lag Error Must Be Small

These are approximations — $\hat{\mathbf{e}}^c$ is not the same as the true contour error $e_k^c$, and $\hat{\theta}_k$ is not the same as the true optimal progress $\theta_k^* = \arg\min_\theta \lVert \mathbf{p}_k - \mathbf{p}^d(\theta) \rVert$.
How good are they?

It turns out the approximation quality is controlled entirely by the lag error.
When $\hat{\mathbf{e}}^l = \mathbf{0}$, the position $\mathbf{p}_k$ lies in the normal plane at $\hat{\theta}_k$, which means $\hat{\theta}_k$ is exactly the closest point on the path — so $\hat{\theta}_k = \theta_k^*$ and $\lVert \hat{\mathbf{e}}^c \rVert = e_k^c$ (see the proof in [Romero _et al._ 2022, Proposition 1](https://arxiv.org/abs/2108.13205)).

This gives us a practical recipe: if we keep the lag error small, the contour error approximation stays accurate.
We enforce this by adding the lag error to the cost function with a high weight $q_l$:

$$
J = \sum_{k=0}^{N} q_c \lVert \hat{\mathbf{e}}^c(\hat{\theta}_k) \rVert^2 + q_l (\hat{e}^l(\hat{\theta}_k))^2 - q_\theta \, v_{\hat{\theta},k}
$$

The lag term is not just another tracking objective — it is what makes the entire approximation scheme valid.
The progress rate $v_{\hat{\theta}}$ replaces the terminal progress $\theta_N$ from the ideal cost (since $\hat{\theta}_N = \hat{\theta}_0 + \sum v_{\hat{\theta},k} \Delta t$, maximizing the sum of progress rates is equivalent to maximizing final progress).

### Encoding as a Multi-Objective Mayer Cost

As we saw in [Tutorial 01](01_hello_world_brachistochrone.md), OpenSCvx always works in Mayer form: the cost is expressed purely as a function of the final state.
So far our problems have had a single cost term — typically minimizing $t_f$.
MPCC requires balancing _three_ competing objectives: minimize contour error, minimize lag error, and maximize progress.

To encode the running costs as terminal costs, we will simply include the cost terms as _integrator states_ that accumulate each component over the horizon:

$$
\dot{s}_l = (\hat{e}^l)^2, \qquad \dot{s}_c = \lVert \hat{\mathbf{e}}^c \rVert^2
$$

with $s_l(0) = s_c(0) = 0$.
Minimizing $s_l(t_f)$ and $s_c(t_f)$ in the Mayer cost is then equivalent to minimizing the running integrals.
Similarly, we maximize the final progress $\hat{\theta}(t_f)$ to encourage forward motion.

This technique of converting Lagrange (running) costs to Mayer (terminal) costs via integrator states is general, but MPCC is a particularly clean application of it because the cost components have distinct physical meanings and naturally competing objectives.

## A Simple Example: Dubins Car on a Circle

Let's start with the simplest possible MPCC: a 2D Dubins car tracking a circular reference path.
The reference is analytical (no discrete points needed), so we can focus entirely on the MPCC structure.

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

Three things to note here.
First, all the physical states have `ox.Free` finals — in a receding horizon setting, there is no fixed terminal condition.
Second, `progress.final` is set to `ox.Maximize(0.0)`.
This tells OpenSCvx to maximize the final value of this state, which is how we encode the $-\mu\, v_{\hat{\theta}}$ term: maximizing the integral of the progress rate is the same as maximizing the final progress.
Third, `lag_sum.final` and `contour_sum.final` are set to `ox.Minimize(0.0)`.
Since these states integrate the squared errors over the horizon, minimizing their final values is equivalent to the running cost we defined above.

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

We use `ox.Sum(e * e)` rather than `ox.linalg.Norm(e)**2` to avoid the derivative singularity $\partial \lVert \mathbf{e} \rVert / \partial \mathbf{e} = \mathbf{e} / \lVert \mathbf{e} \rVert$ at $\mathbf{e} = 0$, which would be a problem since the whole point is to drive the error to zero.
The `ox.Max(..., 0.0)` clamp handles floating-point cases where the subtraction goes slightly negative.

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

Three differences from the single-shot problems in earlier tutorials.
First, the time is _fixed_ — `final=horizon_duration` with no `"minimize"` — because the horizon length is a design parameter, not something to optimize.
Second, we use `ox.ConstantProximalWeight()` as the autotuner.
For MPC where we warm-start each solve from the previous solution, we don't need the aggressive weight scheduling that helps cold-start convergence.
A constant proximal weight keeps the solver predictable and fast.
Third, the `lam_cost` dictionary assigns a weight to each cost state's Mayer contribution.
The lag weight `Q_LAG` is highest to keep the progress approximation accurate, while `Q_CONTOUR` and `Q_PROGRESS` are tuned lower.
This separates the cost tuning from the dynamics formulation — you can adjust these weights without touching the dynamics dictionary.

### The MPC Loop

Now we get to the receding-horizon loop itself.
The pattern is: solve the problem, extract the solution, advance the initial conditions by one step, and shift the guess forward for warm-starting.

```python
problem_mpc.initialize()

for step in range(max_steps):
    problem_mpc.reset()
    results = problem_mpc.solve()
    results = problem_mpc.post_process()
    nodes = results.nodes

    # Advance: set initial conditions from node 1 of previous solution
    position.initial = nodes["position"][1]
    heading.initial = nodes["heading"][1]
    progress.initial = nodes["progress"][1]
    lag_sum.initial = [0.0]      # Reset integrators
    contour_sum.initial = [0.0]  # each horizon

    # Warm-start: shift guess by one node
    # (shift states forward, extrapolate a new final node)
    ...
```

A few things to call out.
`problem_mpc.reset()` clears the SCP iteration history so each MPC solve starts fresh from the current guess, rather than continuing from the previous SCP state.
The cost integrator states (`lag_sum`, `contour_sum`) are always reset to zero at the start of each horizon — they measure the cost _within_ this horizon, not cumulatively.
The warm-starting shift (elided above for brevity) takes the solution from nodes $[1, \ldots, N]$ as the guess for nodes $[0, \ldots, N-1]$ and extrapolates a new final node.
This is critical for MPC performance: without it, each solve starts from scratch and convergence is much slower.

## Discrete Reference Paths with `ox.Cinterp`

The analytical circle is nice for exposition, but in practice our reference path will be a set of discrete points — perhaps from a pre-solved trajectory, a motion capture recording, or hand-placed waypoints.
We need a way to evaluate the reference position and tangent at arbitrary progress values within the symbolic expression graph.

Linear interpolation between sparse points creates kinks in the tangent field, which cause oscillations in the contour error.
We need something smoother: cubic spline interpolation.

### `ox.Cinterp`

`ox.Cinterp(x, xp, fp)` is a symbolic cubic spline interpolation node.
Given breakpoints `xp` and values `fp` (both NumPy arrays, known at problem construction time), it evaluates the natural cubic spline at the symbolic query point `x`.
This is the symbolic analog of `scipy.interpolate.CubicSpline` — in fact, it uses the same coefficients under the hood.

To build a discrete reference path from $M$ sample points:

```python
from scipy.interpolate import CubicSpline as _CS

M = 30  # Samples per lap
# s_data, px_data, py_data: arc-length and position arrays (NumPy)

p_ref = ox.Concat(
    ox.Cinterp(progress[0], s_data, px_data),
    ox.Cinterp(progress[0], s_data, py_data),
)
```

The tangent field requires a bit more care.
We compute the derivative of the cubic spline at the breakpoints using SciPy, normalize to get unit tangents, and then interpolate _those_ with a second `ox.Cinterp`:

```python
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

Why not differentiate `ox.Cinterp` symbolically?
We could, but pre-computing the tangent at the breakpoints and re-interpolating gives us a smooth, well-behaved tangent field with explicit normalization, and it avoids propagating derivative computations through the spline evaluation during SCP linearization.

The rest of the MPCC formulation — error decomposition, cost integrators, dynamics, MPC loop — is identical to the analytical case.
This is the beauty of the approach: swapping from an analytical reference to a discrete one is purely a matter of changing how `p_ref` and `tangent` are constructed.

### Periodicity via Tiling

For closed-loop racing (multiple laps), we need the reference path to be periodic.
We handle this by _tiling_ the single-lap data: replicate the position samples at shifted arc-length offsets so that `s_data` covers the full range of the progress state.

```python
s_min, s_max = -0.5 * total_arc_length, 1.5 * total_arc_length
n_before = int(np.ceil(-s_min / total_arc_length))
n_after = int(np.ceil(s_max / total_arc_length))
tile_laps = range(-n_before, n_after + 1)

s_data = np.concatenate([s_lap + k * total_arc_length for k in tile_laps])
px_data = np.tile(px_lap, len(tile_laps))
py_data = np.tile(py_lap, len(tile_laps))
```

The progress state bounds are set wide enough (here $[-0.5L, 1.5L]$) to accommodate the horizon looking ahead of the current position, and a wrapping operation in the MPC loop keeps the progress from growing without bound.

### Per-State Cost Weighting

As we saw in the simple example, `lam_cost` keeps the cost weights separate from the dynamics formulation:

```python
problem_mpc = ox.Problem(
    ...
    algorithm={
        "autotuner": ox.ConstantProximalWeight(),
        "lam_cost": {
            "lag_sum": 1e0,
            "contour_sum": 1e-1,
            "progress": 1e-1,
        },
    },
)
```

This is especially useful when iterating on a discrete-reference MPCC: you can re-tune the trade-off between progress maximization and tracking accuracy without touching the dynamics or re-deriving the error decomposition.

## Drone Racing MPCC

Now we can put everything together for the real application: a drone racing through gates and around obstacles, tracked by MPCC.
This is a two-phase problem:

1. **Offline**: Solve a time-optimal trajectory through the gates (single-shot, as in [Tutorial 02](02_drone_racing_constraints.md))
2. **Online**: Use MPCC to track the solved trajectory in a receding-horizon loop

### Phase 1: Time-Optimal Trajectory

The offline problem is a standard time-optimal drone racing problem identical to what we built in [Tutorial 02](02_drone_racing_constraints.md): a 3D double integrator flying through 10 gates with loop closure.
We won't repeat the full setup here; the important output is the solved trajectory:

```python
problem_traj.initialize()
results_traj = problem_traj.solve()
results_traj = problem_traj.post_process()

ref_pos = results_traj.trajectory["position"]   # (N_dense, 3)
ref_vel = results_traj.trajectory["velocity"]   # (N_dense, 3)
ref_time = results_traj.trajectory["time"].flatten()
```

### Extracting the Reference Path

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

The tiling, tangent computation, and `ox.Cinterp` setup follow the same pattern as before, now in 3D:

```python
p_ref = ox.Concat(
    ox.Cinterp(progress[0], s_data, px_data),
    ox.Cinterp(progress[0], s_data, py_data),
    ox.Cinterp(progress[0], s_data, pz_data),
)
```

### Phase 2: MPCC Problem

The MPCC problem uses 3D double-integrator dynamics with gravity, matching the offline problem:

```python
dynamics = {
    "position": velocity,
    "velocity": (1 / m) * force + [0, 0, g_const],
    "progress": progress_rate,
    "lag_sum": lag_cost,
    "contour_sum": contour_cost,
}
```

The error decomposition is the same as in the 2D case, just with 3-component vectors.

### Obstacle and Gate Constraints

The MPCC problem inherits obstacle avoidance constraints from the racing environment:

```python
for obs_center in obstacle_centers:
    constraints.append(
        ox.ctcs(obstacle_radius <= ox.linalg.Norm(position - obs_center))
    )
```

For gate constraints, we use a cone formulation that is _progress-dependent_: a gate cone constraint is only active when the drone's progress is between the previous gate and the current one, and when the drone is approaching the gate (velocity pointing toward it).
This is where `ox.Cond` and `ox.Vmap` come in:

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

`ox.Cond` evaluates a condition and returns one of two expressions.
When the condition is true (drone is in the approach segment for this gate), we evaluate the cone constraint.
When false, we return $-1.0$, which trivially satisfies the $\leq 0$ inequality.
`ox.Vmap` vectorizes this over all gates, just as we vectorized obstacle constraints in [Tutorial 03](03_obstacle_avoidance_vmap.md).

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

### Running the MPC Loop

The MPC loop follows the same pattern as the simple example, now with the full 3D state:

```python
problem_mpc.initialize()

for step in range(max_steps):
    problem_mpc.reset()
    results = problem_mpc.solve()
    results = problem_mpc.post_process()
    nodes = results.nodes

    # Advance initial conditions
    position.initial = nodes["position"][1]
    velocity.initial = nodes["velocity"][1]
    progress.initial = np.array([
        nodes["progress"][1, 0] - wrap_offset
    ])
    lag_sum.initial = [0.0]
    contour_sum.initial = [0.0]

    # Shift guess forward for warm-starting
    shift_guess(nodes)
```

The progress wrapping deserves a mention: as the drone completes laps, the raw progress grows beyond the tiled range.
We subtract full-lap offsets to keep the progress within `[progress.min, progress.max]` where the `ox.Cinterp` data is defined.

## Key Takeaways

1. **Multi-objective Mayer costs**: `ox.Minimize` and `ox.Maximize` on state finals let you encode multiple competing running costs and rewards as integrator states. This is a general pattern that works for any Lagrange-to-Mayer conversion.
2. **`ox.Cinterp`**: Cubic spline interpolation within the symbolic graph enables smooth lookup of discrete reference data. Pre-computing the tangent field from the spline derivative and re-interpolating it avoids singularities and gives a clean tangent field.
3. **The MPC pattern**: `problem.reset()` → `problem.solve()` → advance initial conditions → shift guess. Warm-starting from the shifted previous solution is essential for fast convergence.
4. **Two-phase planning**: Solving the hard global problem offline (time-optimal trajectory through gates) and tracking it online with MPCC is a powerful decomposition. The MPCC handles local disturbances and model mismatch while the offline solution provides the global plan.
5. **`lam_cost` dictionary**: Per-state cost weighting gives you a convenient tuning knob for the trade-off between multiple competing objectives.

## Further Reading

- [Dubins Car MPCC Example (analytical circle)](../Examples/mpc/dubins_car_circle.md)
- [Dubins Car MPCC Example (discrete reference)](../Examples/mpc/dubins_car_circle_polytope.md)
- [3D Double Integrator MPCC Example](../Examples/mpc/double_integrator_polytope.md)
- [Drone Racing MPCC Example](../Examples/mpc/double_integrator_drone_racing.md)
- Romero, A. _et al._ (2022). "Model Predictive Contouring Control for Time-Optimal Quadrotor Flight." _IEEE Transactions on Robotics._
- [Drone Racing: Constraints and 3-DOF Dynamics](02_drone_racing_constraints.md) — the single-shot trajectory used as the MPCC reference
- [Obstacle Avoidance: Vmap](03_obstacle_avoidance_vmap.md) — vectorized constraints used in the gate cone formulation
