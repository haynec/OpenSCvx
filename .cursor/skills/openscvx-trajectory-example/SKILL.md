---
name: openscvx-trajectory-example
description: >-
  Exhaustive guide to constructing openscvx.Problem specs: State, Control, Time,
  Parameter, dynamics (continuous and impulsive dynamics_discrete), constraints
  (nodal, CTCS, cross-node, STL), algorithm/discretizer/solver dicts, BYOF/
  mjx_byof, propagation extras, LICQ, float_dtype, YAML loader. Use when
  authoring examples/, trajectory optimization, SCP/PTR, or explaining OpenSCvx
  problem setup.
---

# OpenSCvx trajectory example skill

Authoritative details live in **`openscvx/problem.py`** (`Problem.__init__` docstring) and the linked modules below. This file is a structured index so the agent does not miss an option.

## End-to-end flow

1. Define symbolic `State` / `Control` / optional explicit `time` `State`.
2. Build `dynamics` (`dict` name → \(\dot x\) expression). If any control is **`impulsive`**, also provide matching **`dynamics_discrete`** (see below).
3. Build `constraints` (nodal, CTCS / `.over`, STL, cross-node via `state.at(k)`).
4. Configure **`Time`** (or rely on auto time augmentation — two modes below).
5. `problem = Problem(...)`; then `initialize()` → `solve()` → `post_process()`.

## `Problem(...)` — all constructor arguments

| Argument | Meaning |
|----------|---------|
| **`dynamics`** | `dict[str, Expr]`: every **optimized** state name → continuous-time derivative. |
| **`constraints`** | List of `Constraint` / `NodalConstraint` / `CTCS` / `CrossNodeConstraint` / STL nodes; preprocessor sorts them. |
| **`states`** | List of `State`; may include a `State` named **`"time"`** for time-dependent models (then **`time=`** `Time` argument behavior changes — see `Problem` note). |
| **`controls`** | List of `Control`. |
| **`N`** | Number of trajectory **segments** (nodes are \(0 \ldots N\) in examples). |
| **`time`** | `Time` instance **unless** `states` contains `"time"` — then boundary/objective for physical time live on that `State`. |
| **`dynamics_discrete`** | Optional `dict[str, Expr]`: **required** when at least one control has `parameterization="impulsive"`; must be **absent** when there are no impulsive controls. Maps state → **discrete update** \(x^+\). |
| **`dynamics_prop`** | Extra states’ \(\dot x\) for propagation only (not duplicated in `dynamics`). |
| **`states_prop`** | Extra `State` objects for those propagation dynamics. |
| **`algebraic_prop`** | `dict` name → expression evaluated during propagation (not integrated). |
| **`licq_min`**, **`licq_max`** | CTCS LICQ tube bounds: scalar or `dict` mapping CTCS group **`idx`** → bound (`Problem` docstring). |
| **`algorithm`** | `None` → default PTR; or `PenalizedTrustRegion` / `dict` validated as `PenalizedTrustRegionConfig`; or nested **`autotuner`**. |
| **`discretizer`** | `None` / `dict` / `Discretizer` — resolved by `resolve_discretizer_config` (default type `VectorizeDiscretizeLinearize` when type omitted). |
| **`solver`** | `None` / `dict` / `ConvexSolver` — default `PTRSolver` kwargs. |
| **`byof`** | `ByofSpec` or `dict` → expert JAX hooks (dynamics, constraints, CTCS). |
| **`float_dtype`** | `"float32"` or `"float64"` — sets JAX x64 and lowerer dtype for the whole problem. |

After construction, **`problem.settings`** is a **`openscvx.config.Config`**: **`sim`** (grid / dimensions — internal), **`dev`** (`printing`, `verbosity`, `debug`, `profiling`), **`prp`** (nonlinear **propagation** after solve: `dt`, ODE solver, tolerances). Mutate fields directly (as in examples) or call **`problem.settings.apply_dict({...})`** with nested `dev` / `prp` only (`Config.apply_dict` in `openscvx/config.py`).

YAML **`settings:`** from **`load_dict`** is returned **separate** from `Problem` constructor kwargs — pop it and call **`apply_dict`** after `Problem(...)`.

## `State` — boundary conditions and scaling

From **`openscvx/symbolic/expr/state.py`**:

- **Boundary kinds**: `fixed`, `free`, `minimize`, `maximize` per component — set via plain numbers (fixed), tuples `("free", guess)`, `ox.Free` / `Fixed` / `Minimize` / `Maximize` helpers.
- **`min` / `max`**: per-component box bounds on the trajectory (1-D array length = state dimension).
- **`guess`**: array shape `(N+1, dim)` preferred for warm starts.
- **`scaling_min` / `scaling_max`**: optional scaling bounds used internally (same shape as state); set when you need different scaling from physical bounds.

Dynamics keys **must** match `State.name` for each integrated state.

## `Time` — dedicated horizon object

From **`openscvx/symbolic/expr/time.py`** (`Time` subclasses `State`):

- Fields: `initial`, `final`, `min`, `max`, optional `guess`, **`time_dilation_min` / `time_dilation_max` / `time_dilation_guess`** (absolute bounds on augmented dilation; defaults documented in class).
- **`uniform_time_grid`**: if `True`, enforce equal time steps via dilation.
- **`Time.derivative`**: always `1.0` in normalized form.

**Two time modes** (`Problem` note): (1) no `"time"` in `states` — pass `time=Time(...)`; (2) include `"time"` in `states` and dynamics — `Time` object is ignored; configure that state like any other `State`.

## `Control` — holds and impulsive actuation

From **`openscvx/symbolic/expr/control.py`**:

- **`parameterization`**: `"foh"`, `"zoh"`, `"impulsive"`, or `None` (defer FOH/ZOH default to discretizer for non-impulsive controls).
- **`nodes`**: only for **`impulsive`** — list of nodes where the impulse applies.
- **`min` / `max` / `guess`**: same spirit as `State` (guess shape `(N+1, dim)`).
- **`scaling_min` / `scaling_max`**: optional control scaling.

**Impulsive rule** (preprocessing): if any impulsive control exists, you **must** supply **`dynamics_discrete`** (symbolic dict and/or `byof["dynamics_discrete"]`) explaining discrete jumps. If **no** impulsive controls, **do not** pass `dynamics_discrete`.

## `Parameter`

From **`openscvx/symbolic/expr/parameter.py`**: `ox.Parameter(name, shape, value=...)`. Values can be updated at runtime via `problem.parameters[...]` without recompilation; hashing uses **shape only**, not numeric value.

## Initial guesses (`ox.init`)

**`openscvx/init/__init__.py`**: `ox.init.linspace`, `nlerp`, `slerp`, `ik_interpolation` — build **`state.guess` / `control.guess`** trajectories from keyframes and node indices (see module docstring).

## Lie / spatial / linear algebra (`ox.lie`, `ox.linalg`, `ox.spatial`)

Used in arm and pose examples (e.g. `SE3Exp`, norms). Public surface is re-exported on **`openscvx`**; see `openscvx/__init__.py` and `symbolic/expr/lie/`.

## Constraints — composition

Core module: **`openscvx/symbolic/expr/constraint.py`**.

| Mechanism | What it does |
|-----------|----------------|
| **`expr <= rhs` / `==`** | Builds `Inequality` / `Equality`; bare constraints become **nodal on all nodes** in preprocessing. |
| **`.at([nodes])`** | `NodalConstraint`: enforce only at listed indices. Optional **`.weight(lam_vb)`** overrides global virtual-buffer weight (scalar, per-element, or per-node matrix). |
| **`.over((k0, k1), ...)`** or **`ox.ctcs(...)`** | `CTCS`: continuous-time satisfaction via augmented state + penalty accumulation on node interval **`[k0, k1)`** semantics per implementation. |
| **CTCS kwargs** | **`penalty`**: `"squared_relu"`, `"huber"`, `"smooth_relu"`. **`idx`**: group index for shared augmented state. **`check_nodally`**: also enforce at nodes. |
| **`.convex()`** | Mark constraint convex for CVXPy branch of lowering. |
| **`state.at(k)` inside an expression** | Becomes **cross-node** coupling; do **not** wrap with `.at([...])`. Use **`.weight(...)`** on the wrapper. |

**Constraint canonical form**: residuals are **`lhs - rhs`** compared to zero; when writing BYOF, use **\(g \le 0\)**: *negative = satisfied*, *positive = violated* (`openscvx/expert/byof.py`).

## STL vs hard logic

- **`ox.stl`** (**`openscvx/symbolic/expr/stl.py`**): GMSR smooth robustness — **`And`**, **`Or`**, **`Not`**, **`IfThen`**, **`Always`**, etc., for **task-level** smooth composition. **Read the module warning**: **`NodeInterval` (“nodes”)** is implemented end-to-end; **`TimeInterval` (“seconds”)** is accepted at construction but **lowering is not implemented** (`NotImplementedError`). **`Eventually` / `Until`** are placeholders (not yet lowered). Mixed nested temporal windows may be **rejected at construction** — prefer separate top-level constraints for distinct windows.
- **`ox.All` / `ox.Any` / `ox.Cond`** (**`openscvx/symbolic/expr/logic.py`**): **hard boolean** branching inside expressions — **JAX-only**, not DCP for CVXPy; use for switched dynamics, not soft task specs.

## `algorithm=` — `PenalizedTrustRegion`

From **`openscvx/algorithms/penalized_trust_region.py`** + **`PenalizedTrustRegionConfig`**:

- **`autotuner`**: `None` → **`AugmentedLagrangian`**; or string name, or `{"type": "RampProximalWeight", ...}`, or an instance (`Problem` docstring examples).
- **Weights**: **`lam_prox`** (trust region), **`lam_vc`** (virtual control / dynamics slack), **`lam_cost`** (objective weights), **`lam_vb`** (virtual buffer). Each can be a **scalar** or **dict** keyed by state/control name with optional per-component or per-node arrays (`PenalizedTrustRegion.__init__` docstring).
- **Tolerances**: **`ep_tr`**, **`ep_vb`**, **`ep_vc`**.
- **Limits**: **`k_max`**, optional **`t_max`** wall time.

## `discretizer=`

Resolved via **`openscvx/discretization/__init__.py`** (`resolve_discretizer_config`).

- Default **`type`** when omitted: **`VectorizeDiscretizeLinearize`**.
- Allowed **`type`** values: **`LinearizeDiscretize`**, **`LinearizeDiscretizeSparse`**, **`VectorizeDiscretizeLinearize`**, **`DiscretizeLinearizeVectorize`** — see **`DiscretizerSpec`** in `openscvx/discretization/base.py` (`ode_solver`, `dis_type`, `diffrax_kwargs`, `custom_integrator`, `args`).

## `solver=` — `PTRSolver`

From **`openscvx/solvers/ptr_solver.py`** + **`SolverSpec`** (`openscvx/solvers/base.py`):

- **`cvx_solver`**: backend name (e.g. `CLARABEL`, `QOCO`, …).
- **`solver_args`**: passed through to CVXPy (tolerances, `enforce_dpp`, …).
- **`cvxpygen`**: optional codegen; note sparse-parameter caveat in `SolverSpec` docstring.

## `byof=` — `ByofSpec`

From **`openscvx/expert/byof.py`** (full signatures and examples in module docstring):

| Key | Role |
|-----|------|
| **`parameters`** | `Parameter` objects referenced only inside BYOF functions. |
| **`dynamics`** | `dict[state_name, (x,u,node,params) → xdot]` — **replaces** symbolic `dynamics` for those states. |
| **`dynamics_discrete`** | Same signature, returns **\(x^+\)** for impulsive/discrete updates; pairs with impulsive controls. |
| **`nodal_constraints`** | List of `{ "constraint_fn": ..., "nodes": optional }`. |
| **`cross_nodal_constraints`** | List of `(X, U, params) → residual` with `X` shape `(N+1, n_x)` (full trajectory). |
| **`ctcs_constraints`** | List matching `CtcsConstraintSpec`: scalar `constraint_fn`, `penalty`, `bounds`, `over`, **`idx`** (must align with symbolic CTCS idx grouping — no gaps). |

After `Problem` construction, use **`state.slice` / `control.slice`** into unified vectors (set during preprocessing).

## MuJoCo MJX

Use **`openscvx.integrations.mjx_byof`** (`integrations/__init__.py` / `mjx.py`):

- Returns `byof["dynamics"]` dict; for **`nq == nv`** models you still set **`dynamics={"qpos": qvel}`**; for **free joint** models (`nq > nv`), **`qpos`** dynamics may be included automatically.
- **`mjx_dynamics`** available for advanced composition.

## Declarative YAML / JSON

**`openscvx.loader`**: `load_yaml`, `load_json`, `load_dict` → kwargs for `Problem`. Schema from **`ProblemSpec`**; generate JSON Schema via **`openscvx schema`** (module docstring in `loader.py`). Constraints are **strings** parsed by `ExprParser`.

## Integration tests / examples layout

**`tests/test_examples.py`**: discovers `examples/**/*.py` with a top-level **`problem`** symbol; skips `realtime/`, ignores `plotting.py`, respects **`EXCLUDED_EXAMPLES`**. Prefer defining `problem` at import without heavy side effects.

## Where to copy patterns

| Topic | Example |
|--------|---------|
| Minimum time | `examples/abstract/brachistochrone.py` |
| Dubins + STL | `examples/car/dubins_car_waypoint_stl.py` |
| Drone racing / obstacles | `examples/drone/drone_racing.py`, `examples/drone/obstacle_avoidance.py` |
| View planning | `examples/drone/dr_vp.py` |
| Arm + Lie | `examples/arm/3_dof_arm.py` |
| Spacecraft / rocket | `examples/spacecraft/*`, `examples/rocket/*` |
| MJX | `examples/mjx/cartpole_mjx.py` |
| MPC / discrete | `examples/mpc/double_integrator_discrete.py` |

## Further reading (one level deep)

See **[reference.md](reference.md)** for discretizer class purposes, default config quirks, and a minimal import skeleton.
