---
description: >-
  Drive OpenSCvx trajectory optimization with MuJoCo MJX dynamics for contact-rich robotic systems.
---

# 09 MuJoCo MJX Dynamics

In earlier tutorials we wrote dynamics by hand as symbolic expressions. Many real systems already have a MuJoCo model — articulated robots, cartpoles, quadrotors — and re-deriving those equations is tedious and error-prone.

This tutorial shows how to plug a **MuJoCo MJX** model directly into OpenSCvx via `ox.MjxDynamics`. We use the cartpole swing-up as the main walkthrough (the same problem as `examples/mjx/cartpole_mjx.py`), then point to richer examples: multi-link cartpoles, 3D triple pendulums, and Skydio X2 gate racing.

This tutorial covers:

- Installing the optional MJX extra (`openscvx[mjx]`)
- Loading a MuJoCo model and uploading it to MJX
- The `MjxDynamics` adapter: states, controls, and `Problem` wiring
- Contact-free models and why contacts are disabled
- Free-joint models (`nq > nv`) and automatic quaternion kinematics
- The full set of MJX examples in `examples/mjx/`

!!! tip "Prerequisites"
    You should be comfortable with the core workflow from [Hello Brachistochrone](01_hello_world_brachistochrone.md): `State`, `Control`, `Time`, CTCS constraints, `initialize` / `solve` / `post_process`, and accessing results by variable name.

## Installation

MJX is an optional dependency. Install OpenSCvx with the MJX extra:

```sh
pip install openscvx[mjx]
```

This pulls in `mujoco` with JAX-backed MJX support. All examples under `examples/mjx/` guard-import MuJoCo and print a clear message if the extra is missing.

## Why MJX?

[MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) runs the same physics as classic MuJoCo, but on JAX arrays so the forward dynamics are compatible with OpenSCvx's JAX compilation and autodiff pipeline.

The integration layer (`openscvx.integrations`) wraps an MJX model as a **dynamics adapter** you pass to `Problem` in the same slot as a symbolic dynamics dict. You still set boundary conditions, costs, and constraints the usual way — only the equations of motion come from MuJoCo.

## The Cartpole Swing-Up Problem

We optimize a horizontal force on a cart with a passive hinged pole. The goal is to swing the pole from hanging down ($\theta = \pi$) to upright ($\theta = 0$) in minimum time.

In MuJoCo coordinates:

- **Position** $\mathbf{q} = [x, \theta]^\top$ — cart slide and hinge angle
- **Velocity** $\dot{\mathbf{q}} = [\dot{x}, \dot{\theta}]^\top$
- **Control** $u$ — normalized motor command on the slider joint

The continuous dynamics $\ddot{\mathbf{q}} = f(\mathbf{q}, \dot{\mathbf{q}}, u)$ are computed by MJX's `forward` pass; OpenSCvx does not require you to write them symbolically.

## Loading a MuJoCo Model

Examples typically define MJCF inline (self-contained) or load from a file:

```python
import mujoco
import mujoco.mjx as mjx

CARTPOLE_XML = """
<mujoco model="cartpole">
  <option gravity="0 0 -9.81" timestep="0.01" integrator="Euler"/>
  ...
</mujoco>
"""

mj_model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
# For assets on disk: mujoco.MjModel.from_xml_path("cartpole.xml")
```

### Disable contacts for differentiability

MJX's contact solver uses `lax.while_loop`, which is **not** reverse-mode differentiable. For manipulation and locomotion models that do not need contact forces in the optimizer, disable contacts before uploading:

```python
mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
mjx_model = mjx.put_model(mj_model)
```

Cartpoles, serial arms without collision, and quadrotors in free flight all fit this pattern. If you need contact-rich dynamics, treat that as an advanced case and consult the lower-level `mjx_dynamics` API in the [reference](../Reference/integrations/mjx.md).

After `put_model`, read off dimensions:

```python
n_q = int(mjx_model.nq)  # generalized positions
n_v = int(mjx_model.nv)  # generalized velocities
n_u = int(mjx_model.nu)  # actuators
```

For a simple cartpole, `nq == nv == 2` and `nu == 1`.

## The `MjxDynamics` Adapter

`ox.MjxDynamics(mjx_model)` is the recommended entry point. It:

1. Creates default `qpos`, `qvel`, and `ctrl` variables matching `nq`, `nv`, and `nu`
2. Exposes them on `.states` and `.controls`
3. Routes MJX forward dynamics through the internal BYOF channel — you do **not** pass `byof=` yourself

```python
import openscvx as ox

dyn = ox.MjxDynamics(mjx_model)
qpos, qvel = dyn.states
(ctrl,) = dyn.controls
```

`MjxDynamics` also seeds `.min` / `.max` from joint and actuator limits declared in MJCF (`jnt_range`, `actuator_ctrlrange`). Override them for your OCP as needed.

### Boundary conditions

Set initial and final values on the adapter's states like any other `ox.State`:

```python
qpos.min = np.array([-3.0, -2.0 * np.pi])
qpos.max = np.array([3.0, 2.0 * np.pi])
qpos.initial = np.array([0.0, np.pi])   # cart at origin, pole hanging
qpos.final = np.array([0.0, 0.0])     # upright

qvel.initial = np.zeros(2)
qvel.final = np.zeros(2)

ctrl.min = np.array([-1.0])
ctrl.max = np.array([1.0])
ctrl.guess = np.zeros((n, 1))
```

Use the same boundary-condition syntax as earlier tutorials (`ox.Free`, `ox.Minimize`, tuples, etc.) — see [Hello Brachistochrone](01_hello_world_brachistochrone.md).

### Initial guess

A good state guess helps underactuated swing-up problems converge:

```python
theta_guess = np.linspace(np.pi, 0.0, n)
qpos.guess = np.column_stack([np.zeros(n), theta_guess])
qvel.guess = np.zeros((n, 2))
```

### Constraints and time

CTCS box constraints follow the same pattern as tutorial 01:

```python
constraints = []
for state in dyn.states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
for control in dyn.controls:
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

time = ox.Time(
    initial=0.0,
    final=ox.Minimize(total_time),
    min=0.0,
    max=2.0 * total_time,
)
```

### Defining and solving the problem

Pass the adapter as `dynamics=` and reuse its states and controls:

```python
problem = ox.Problem(
    dynamics=dyn,
    states=dyn.states,
    controls=dyn.controls,
    time=time,
    constraints=constraints,
    N=n,
)

problem.initialize()
problem.solve()
results = problem.post_process()
```

Results are accessed by the same names:

```python
theta_traj = results.trajectory["qpos"][:, 1]
u_traj = results.trajectory["ctrl"]
```

### Run the full example

The complete script adds a Viser animation of the cart and pole:

```sh
python examples/mjx/cartpole_mjx.py
```

## How It Works Internally

For models with `nq == nv` (no quaternion free joint), `MjxDynamics` sets symbolic kinematics `qpos_dot = qvel` and uses MJX to provide `qvel_dot` (generalized accelerations) from `mjx.forward`.

For models with `nq > nv` (a floating base with a quaternion), the adapter inserts quaternion kinematics for `qpos` and MJX accelerations for `qvel`. You do not need to write the $\dot{q}$ equation for the attitude quaternion — that is handled automatically.

Supported joint types: **free**, **slide**, and **hinge**. All free joints must appear before slide/hinge joints in the MJCF layout. Ball joints are explicitly rejected with a clear error; use the lower-level `mjx_dynamics` helper if you need them.

## Free-Joint Example: Skydio X2 Gate Racing

`examples/mjx/skydio_x2_mjx.py` mirrors [Drone Racing](02_drone_racing_constraints.md) but uses a Menagerie (or inline fallback) quadrotor model:

- `nq = 7` — position plus unit quaternion `[w, x, y, z]`
- `nv = 6` — linear and angular velocity
- `nu = 4` — per-rotor thrusts

The problem setup is identical in spirit to tutorial 02: nodal gate constraints with `.at()`, minimum-time objective, loop closure in position. The only change is `dynamics=dyn` instead of a hand-written double-integrator dict.

```python
dyn = ox.MjxDynamics(mjx_model)
qpos, qvel = dyn.states
(ctrl,) = dyn.controls

# qpos: [x, y, z, qw, qx, qy, qz]
qpos.initial = np.concatenate([start_pos, hover_quat])
qpos.final = [10.0, 0.0, 20.0, ("free", 1.0), ("free", 0.0), ("free", 0.0), ("free", 0.0)]
```

Optional: initialize the MuJoCo Menagerie submodule for the textured mesh used in Viser:

```sh
git submodule update --init third_party/mujoco_menagerie
```

## MJX Examples Overview

| Example | Description | Highlights |
|---------|-------------|------------|
| [`cartpole_mjx`](../Examples/mjx/cartpole_mjx.md) | Single-link swing-up | Minimal `MjxDynamics` workflow, Viser animation |
| [`double_cartpole_mjx`](../Examples/mjx/double_cartpole_mjx.md) | Two-link cartpole | Underactuated serial links, `ox.Free` terminal cart |
| [`triple_cartpole_mjx`](../Examples/mjx/triple_cartpole_mjx.md) | Three-link cartpole | Longer horizon, ZOH control parameterization |
| [`triple_cartpole_3d_mjx`](../Examples/mjx/triple_cartpole_3d_mjx.md) | 3D triple pendulum on a cart | Higher DOF, 3D Viser scene |
| [`triple_cartpole_game`](../Examples/mjx/triple_cartpole_game.md) | Game-style triple cartpole | Variant with different cost / setup |
| [`skydio_x2_mjx`](../Examples/mjx/skydio_x2_mjx.md) | Gate racing quadrotor | Free joint, Menagerie model, nodal constraints |

All live under `examples/mjx/` and require `pip install openscvx[mjx]`.

## Advanced: `mjx_dynamics` and Custom Names

If you need custom `State` / `Control` names, extra states alongside MJX, or ball joints, drop to `openscvx.integrations.mjx_dynamics` and assemble `byof` manually:

```python
from openscvx.integrations import mjx_dynamics

qpos = ox.State("cart_and_pole", shape=(mjx_model.nq,))
qvel = ox.State("rates", shape=(mjx_model.nv,))
ctrl = ox.Control("force", shape=(mjx_model.nu,))

qvel_dot = mjx_dynamics(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)

problem = ox.Problem(
    dynamics={"cart_and_pole": qvel},
    byof={"dynamics": {"rates": qvel_dot}},
    states=[qpos, qvel],
    controls=[ctrl],
    ...
)
```

See the [API reference](../Reference/integrations/mjx.md) for `return_component`, `extra_postprocess`, and full signatures.

## Further Reading

- [Complete Cartpole MJX Example](../Examples/mjx/cartpole_mjx.md)
- [Skydio X2 MJX Gate Racing](../Examples/mjx/skydio_x2_mjx.md)
- [Drone Racing: Constraints](02_drone_racing_constraints.md) — same gate constraints without MJX
- [Visualization](05_visualization.md) — Plotly and Viser patterns used in MJX examples
- [API Reference: `MjxDynamics`](../Reference/integrations/mjx.md)
