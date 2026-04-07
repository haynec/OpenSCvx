# 07 Multi-Link Arms: Lie Algebra and Propagated States

In this tutorial we leave the world of free-flying rigid bodies and step into manipulation.
We will set up a trajectory optimization problem for a 7-DOF redundant arm performing a pick-and-place motion, using the **Product of Exponentials (PoE)** formula for forward kinematics built directly inside the symbolic graph.

Along the way, this tutorial introduces:

- The minimal vs maximal representations of multi-body dynamics, and why we use the minimal one
- Forward kinematics via the matrix exponential on $SE(3)$ with `ox.lie.SE3Exp`
- Capturing intermediate kinematic frames as **algebraic propagated states** for constraints and visualization
- Initializing joint trajectories with `ox.init.ik_interpolation`

This tutorial does not introduce a new constraint formulation in the way that the viewpoint or MPCC tutorials did.
What is new is the *modeling layer*: how to express a kinematic chain inside the symbolic graph so the solver can differentiate through it and so we can constrain task-space quantities (end-effector position, orientation, viewcone, ...) directly.

## Minimal vs Maximal Representations

A rigid-body arm with $n$ links can be described in two fundamentally different coordinate systems.
The choice shapes how the equations of motion look, which quantities appear explicitly, and what must be solved jointly.
Before writing any code it is worth being precise about which one we are using and why.

### Minimal representation (generalized coordinates)

**State:** joint angles $\mathbf{q} \in \mathbb{R}^n$ and velocities $\dot{\mathbf{q}}$. Total state dimension: $2n$.

The equations of motion are a compact ODE — the **Euler–Lagrange equations**:

$$
M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}
$$

| Term | Meaning | Size |
|---|---|---|
| $M(\mathbf{q})$ | Joint-space mass (inertia) matrix | $n \times n$ |
| $C(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}}$ | Coriolis and centrifugal forces | $n \times 1$ |
| $\mathbf{g}(\mathbf{q})$ | Gravity vector | $n \times 1$ |
| $\boldsymbol{\tau}$ | Applied joint torques | $n \times 1$ |

Joint constraints are *implicit* — they are baked into the choice of coordinates and never appear explicitly. There is no redundancy and no algebraic side-condition; just an ODE in $2n$ variables.

### Maximal representation (Cartesian / body coordinates)

**State:** the full Cartesian pose (position + orientation) of *every* link, plus their velocities. Total state dimension is $12n$ in 3D, before constraints.

Joints are not baked in — they are imposed as separate algebraic constraints. The equations of motion become a **Differential-Algebraic Equation (DAE)**:

$$
M\ddot{\mathbf{x}} = \mathbf{f}_{\text{ext}} + J_c^\top \boldsymbol{\lambda} + J_u^\top \boldsymbol{\tau}, \qquad \mathbf{g}(\mathbf{x}) = \mathbf{0}
$$

The Lagrange multipliers $\boldsymbol{\lambda}$ are joint reaction forces; they are *not* free inputs but are determined jointly with $\ddot{\mathbf{x}}$ by the requirement that the constraints be satisfied. Differentiating $\mathbf{g}(\mathbf{x}) = \mathbf{0}$ twice in time and combining with the equations of motion gives a saddle-point KKT system

$$
\begin{bmatrix} M & J_c^\top \\ J_c & 0 \end{bmatrix}
\begin{bmatrix} \ddot{\mathbf{x}} \\ -\boldsymbol{\lambda} \end{bmatrix}
=
\begin{bmatrix} \mathbf{f}_{\text{ext}} + J_u^\top \boldsymbol{\tau} \\ -\dot{J}_c \dot{\mathbf{x}} \end{bmatrix}
$$

which must be solved at every integration step. Drift in the position- and velocity-level constraints accumulates over time and typically requires Baumgarte stabilization to keep $\mathbf{g}(\mathbf{x}) \approx \mathbf{0}$.

### Comparison

|  | Minimal | Maximal |
|---|---|---|
| State size | $2n$ | $12n$ (+ $\boldsymbol{\lambda}$) |
| Equations of motion | ODE | DAE |
| Mass matrix $M$ | $n \times n$, dense, configuration-dependent | $6n \times 6n$, block-diagonal, constant |
| Joint constraints | Implicit | Explicit ($5(n-1)$ for revolute joints) |
| Constraint forces $\boldsymbol{\lambda}$ | Implicit | Explicit |
| Coordinate singularities | Possible | None (with quaternions) |
| Closed kinematic loops | Hard | Easy |
| Constraint drift | N/A | Present; requires stabilization |
| Primary use | Control, planning, RL | Simulation, contact-rich problems |

Both representations are viable for trajectory optimization, and there is a long line of work showing that maximal coordinates can actually be *advantageous* for trajopt — the constant block-diagonal mass matrix, the absence of coordinate singularities, the natural handling of closed loops and contact, and the well-conditioned linearizations near singular configurations are all real benefits that practitioners rightly care about. We plan to revisit the maximal formulation in a more advanced tutorial down the line.

For this first pass we will start with the minimal representation because it is the simpler of the two: a pure ODE in $2n$ variables, no algebraic side-conditions, no multipliers, and a clean fit with the rest of the OpenSCvx modeling stack. That lets us focus on what is genuinely new here: expressing forward kinematics symbolically so the *task-space* quantities we want to constrain (end-effector position, orientation, camera frame) — which are no longer state variables — can be built as **functions of $\mathbf{q}$** inside the symbolic graph.
That is exactly what `ox.lie` is for.

!!! note "Looking ahead: sparsity and the maximal representation"
    When we eventually tackle the maximal representation, OpenSCvx's automatic sparsity exploitation will be critical for performance. The maximal-coordinates state vector grows as $12n$ (vs $2n$ for the minimal case), the mass matrix becomes $6n \times 6n$, and the constraint Jacobian $J_c$ adds another $5(n-1)$ rows — but both $M$ (block-diagonal) and $J_c$ (each row touches only two adjacent bodies) are extremely sparse. A dense linearization would scale catastrophically; the symbolic graph's ability to track and exploit that sparsity end-to-end is what will make the maximal formulation tractable at all.

## Forward Kinematics via Product of Exponentials

The standard way to write forward kinematics for a serial chain is the **Product of Exponentials** formula. Let $\boldsymbol{\xi}_i \in \mathbb{R}^6$ be the *screw axis* of joint $i$ expressed in the base frame at the home configuration, and let $T_{\text{home}} \in SE(3)$ be the end-effector pose at $\mathbf{q} = \mathbf{0}$. Then

$$
T_{\text{ee}}(\mathbf{q}) = \exp\!\bigl(\hat{\boldsymbol{\xi}}_1 q_1\bigr) \, \exp\!\bigl(\hat{\boldsymbol{\xi}}_2 q_2\bigr) \, \cdots \, \exp\!\bigl(\hat{\boldsymbol{\xi}}_n q_n\bigr) \, T_{\text{home}}
$$

where $\hat{\boldsymbol{\xi}}_i \in \mathfrak{se}(3)$ is the matrix form of the screw axis and $\exp : \mathfrak{se}(3) \to SE(3)$ is the matrix exponential. Each screw axis decomposes as $\boldsymbol{\xi} = (\mathbf{v}, \boldsymbol{\omega})$ where $\boldsymbol{\omega}$ is the rotation axis and $\mathbf{v} = -\boldsymbol{\omega} \times \mathbf{q}$ for a point $\mathbf{q}$ on the joint axis.

The PoE formula is appealing for symbolic differentiation: it is a composition of matrix exponentials, each of which is smooth in its scalar argument $q_i$, and the chain rule passes cleanly through every factor. There are no Euler-angle singularities and no quaternion normalizations to worry about.

In OpenSCvx, the building block is `ox.lie.SE3Exp`, which takes a length-6 screw expression and returns a $4 \times 4$ symbolic transform.

## Building the Kinematic Chain

We will walk through the [7-DOF arm example](../Examples/arm/seven_link_arm.md). The robot is a Panda/iiwa-like arm with seven revolute joints in an alternating $z$–$y$ pattern. The link lengths and joint axes give us a fixed table of screw axes:

```python
import numpy as np
import openscvx as ox

N_JOINTS = 7
d1 = 0.340  # base height
a2 = 0.300  # shoulder -> elbow
a3 = 0.250  # elbow -> wrist
a4 = 0.150  # wrist -> end-effector

# Screw axes [v; omega] for each joint, in the base frame at q = 0
screw_axes = np.array([
    [0.0,        0.0,        0.0, 0.0, 0.0, 1.0],  # J1 z at origin
    [-d1,        0.0,        0.0, 0.0, 1.0, 0.0],  # J2 y at [0, 0, d1]
    [0.0,       -a2,         0.0, 0.0, 0.0, 1.0],  # J3 z at [a2, 0, d1]
    [-d1,        0.0,         a2, 0.0, 1.0, 0.0],  # J4 y at [a2, 0, d1]
    [0.0, -(a2 + a3),         0.0, 0.0, 0.0, 1.0],  # J5 z at [a2+a3, ...]
    [-d1,        0.0,    a2 + a3, 0.0, 1.0, 0.0],  # J6 y
    [0.0, -(a2 + a3 + a4),    0.0, 0.0, 0.0, 1.0],  # J7 z
])

# End-effector pose at q = 0
T_home = np.eye(4)
T_home[:3, 3] = [a2 + a3 + a4, 0.0, d1]
```

The state is just the joint angles and velocities (the minimal representation in action):

```python
angle = ox.State("angle", shape=(N_JOINTS,))
angle.max = np.deg2rad([170, 120, 170, 120, 170, 120, 175])
angle.min = -angle.max
angle.initial = np.zeros(N_JOINTS)
angle.final = [("free", 0.0)] * N_JOINTS

velocity = ox.State("velocity", shape=(N_JOINTS,))
velocity.max = np.full(N_JOINTS, 3.0)
velocity.min = -velocity.max
velocity.initial = np.zeros(N_JOINTS)
velocity.final = np.zeros(N_JOINTS)

torque = ox.Control("torque", shape=(N_JOINTS,))
torque.max = np.array([8.0, 8.0, 4.0, 4.0, 2.0, 1.0, 0.5])
torque.min = -torque.max
```

Now the interesting part: we build $T_{\text{ee}}(\mathbf{q})$ symbolically by chaining `SE3Exp` factors. To get intermediate frames (shoulder, elbow, wrist) "for free" we accumulate the chain incrementally and snapshot the partial product after the right number of joints.

```python
joint_names = ["base", "shoulder", "elbow", "wrist", "ee"]
keypoint_after_joint = [0, 1, 2, 4, 7]  # joints preceding each keypoint

T_chain = ox.Constant(np.eye(4))
joint_idx = 0
joint_transforms = {}
for name, n_joints in zip(joint_names, keypoint_after_joint):
    while joint_idx < n_joints:
        xi = ox.Constant(screw_axes[joint_idx])
        T_chain = T_chain @ ox.lie.SE3Exp(xi * angle[joint_idx])
        joint_idx += 1
    joint_transforms[name] = T_chain

T_ee = T_chain @ ox.Constant(T_home)
p_ee = T_ee[:3, 3]
R_ee = T_ee[:3, :3]
```

`T_ee`, `p_ee`, and `R_ee` are symbolic expressions in `angle`. We can use them anywhere we would use a state — in constraints, in costs, or as inputs to other symbolic functions.

## Dynamics

For this example we use a deliberately simplified second-order model — diagonal inertia, no Coriolis or gravity:

$$
I \ddot{\mathbf{q}} = \boldsymbol{\tau}
$$

```python
inertia = np.array([0.08, 0.06, 0.05, 0.04, 0.02, 0.01, 0.005])
I_inv = ox.Constant(1.0 / inertia)

dynamics = {
    "angle": velocity,
    "velocity": I_inv * torque,
}
```

The point of this tutorial is to demonstrate the Lie-algebra modeling layer, which is independent of the dynamics model. Plugging in a richer $M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$ would change only the right-hand side of `dynamics["velocity"]`.

## Task-Space Constraints

Because `p_ee` and `R_ee` are symbolic, we can constrain them directly. The example performs a five-waypoint pick-and-place (`home → pre_grasp → grasp → pre_grasp → pre_place → place`), with a tight position tolerance at each waypoint:

```python
ee_tolerance = 0.01  # 1 cm
for i, (wp, node) in enumerate(zip(waypoint_positions, waypoint_nodes)):
    wp_param = ox.Parameter(f"waypoint_{waypoint_names[i]}", shape=(3,), value=wp)
    constraints.append(
        (ox.linalg.Norm(p_ee - wp_param, ord=2) <= ee_tolerance).at([node])
    )
```

For orientation we use the $SO(3)$ logarithm, `ox.lie.SO3Log`, which maps a rotation matrix to its $\mathbb{R}^3$ axis–angle representation. The norm of $\log(R_{\text{des}}^\top R_{\text{ee}})$ is the geodesic angle error:

```python
ori_error = ox.lie.SO3Log(R_des.T @ R_ee)
ori_norm = ox.linalg.Norm(ori_error, ord=2)

# loose tolerance over the whole task, tight near grasp/place
constraints.append((ori_norm <= np.deg2rad(5.0)).over((wp1, wpN)))
constraints.append((ori_norm <= np.deg2rad(0.1)).over((wp_pregrasp, wp_pregrasp_2)))
```

This is a *continuous-time* constraint: the geodesic error is bounded everywhere along the trajectory, not just at the discrete nodes.

## Algebraic Propagated States

The intermediate transforms `joint_transforms[name]` are not state variables — they are deterministic functions of `angle`. But we still want them in the post-processed trajectory so we can plot the arm or build a viser animation. The mechanism for this is `algebraic_prop`:

```python
problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    algebraic_prop={
        "ee_position": p_ee,
        **{f"T_{name}": T for name, T in joint_transforms.items()},
    },
)
```

After `problem.post_process()` the propagated trajectory will contain `ee_position` (a $T \times 3$ array) and `T_base`, `T_shoulder`, ..., `T_ee` (each a $T \times 4 \times 4$ array of transforms), evaluated at the same dense propagation grid as the dynamics. Anything you can write as a symbolic expression in the states can be requested this way; it costs nothing in the optimization itself.

## Initializing Joint Trajectories with IK

Trajectory optimization in joint space is sensitive to the initial guess. A linear interpolation between $\mathbf{q}_{\text{init}}$ and a hand-tuned $\mathbf{q}_{\text{terminal}}$ works for simple cases, but as soon as you have multiple waypoints — or want to specify the task in *Cartesian* space, which is usually how a user thinks — it breaks down.

`ox.init.ik_interpolation` solves this by interpolating end-effector poses in task space (linear in position, slerp in orientation) and solving damped least-squares IK at every node:

```python
angle.guess = ox.init.ik_interpolation(
    keyframes=list(zip(waypoint_positions, waypoint_orientations)),
    nodes=waypoint_nodes,
    screw_axes=screw_axes,
    T_home=T_home,
    angles_init=angle.initial,
    angles_min=angle.min,
    angles_max=angle.max,
    sequential=True,
)
```

The `sequential=True` flag seeds each node's IK solve from the previous node's solution, producing a coherent joint-space path. The default (`sequential=False`) solves all nodes in parallel from the same seed via `jax.vmap`, giving each node its own independent shot at finding a good local minimum.

!!! tip "Sequential vs parallel IK: which to use?"
    The two modes trade off **coherence** against **per-node quality**, and neither is universally better:

    - **Sequential** produces smooth joint-space paths because each solve is warm-started from its neighbor — but a poor early solution propagates downstream, and the chain can get *trapped* in a branch of the IK manifold that becomes increasingly suboptimal (or infeasible against joint limits) as the task moves on. This is most pronounced for redundant arms, where the IK manifold is large and "the solution near my last one" can be a long way from "the best solution for this pose."
    - **Parallel** gives each node a fresh, independent solve from the same seed, so every node lands in a locally good configuration for *that* pose — at the cost of joint-space discontinuities where adjacent nodes pick different IK branches.

    A reasonable rule of thumb: try parallel first. If the resulting guess looks discontinuous or the solver struggles to pick up the trajectory, switch to sequential.

## Putting It Together

The full problem definition looks just like the drone examples — the only thing that has changed is *how* `p_ee` and `R_ee` are computed:

```python
time = ox.Time(initial=0.0, final=ox.Minimize(4.0), min=0.0, max=4.0)

problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    algebraic_prop={
        "ee_position": p_ee,
        **{f"T_{name}": T for name, T in joint_transforms.items()},
    },
)

problem.initialize()
results = problem.solve()
results = problem.post_process()
```

After solving, `results.trajectory["T_ee"]` gives you the dense end-effector transform along the propagated trajectory, ready to feed straight into a viser animation or a downstream controller.

## Key Takeaways

1. **Start simple with the minimal representation.** Joint angles + velocities give a pure ODE that drops cleanly into OpenSCvx's modeling stack — the easiest place to begin. Maximal coordinates are also a strong (and in some respects superior) choice for trajopt and will get their own tutorial in the future. The consequence of working in joint space is that task-space quantities are no longer states — they are symbolic functions of the state.
2. **Build forward kinematics symbolically.** `ox.lie.SE3Exp` lets you write the Product of Exponentials formula directly in the symbolic graph. The solver gets exact derivatives through the entire kinematic chain for free.
3. **Snapshot intermediate frames.** Accumulating the FK chain incrementally and stashing partial products gives you every joint frame at no extra cost — handy for visualization, collision checks, or per-link constraints.
4. **Use `algebraic_prop` for derived quantities.** Anything that is a function of the state can be propagated alongside the dynamics without entering the optimization, ready for plotting and analysis.
5. **Initialize in task space.** `ox.init.ik_interpolation` lets you specify what you want the *end-effector* to do and figures out joint angles via IK. Use `sequential=True` for redundant arms to get coherent guesses.

## Further Reading

- [API Reference: Lie algebra](../Reference/symbolic/expr/lie/index.md)
- [Three-link arm Example](../Examples/arm/three_link_arm.md)
- [Seven-link arm Example](../Examples/arm/seven_link_arm.md)
- [Seven-link arm with viewcone Example](../Examples/arm/seven_link_arm_vp.md)
- Lynch & Park, *Modern Robotics* — Chapter 4 covers the Product of Exponentials formula in detail
