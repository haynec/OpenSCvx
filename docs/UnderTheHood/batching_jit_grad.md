---
description: >-
  How OpenSCvx's `solve_jax()` entry point runs the SCP loop as a pure JAX
  `lax.while_loop` so solves compose with `jax.vmap`, `jax.jit`, and `jax.grad`.
---

# Batching, JIT, and Grad with `Problem.solve_jax()`

`Problem` exposes two solve entry points. `solve()` is the familiar
Python-loop driver — real-time prints, wall-clock `time_limit`, `continuous`
mode, populated per-iteration history. `solve_jax()` is its JAX-pure
sibling: it drives the same fused `iteration_fn` body inside a
`lax.while_loop` and returns an `OptimizationResults`
(`openscvx/algorithms/optimization_results.py`) pytree, so it composes with
`jax.vmap`, `jax.jit`, and `jax.grad`.

```python
problem = ox.Problem(..., solver={"backend": "qpax"})
problem.initialize()

# Familiar interactive solve — unchanged: prints, history, time limit.
result = problem.solve()

# JAX-pure single solve — silent, no per-iteration history, batchable.
result = problem.solve_jax()

# Batched — standard jax.vmap, no library-specific API.
batched = jax.vmap(problem.solve_jax, in_axes=(0, 0, None))(
    x_initial_stack, x_final_stack, params
)

# JIT-compile once, solve forever (e.g. MPC inner loop).
fast_solve = jax.jit(problem.solve_jax)
```

A worked example lives at `examples/abstract/brachistochrone_batched.py`
— four brachistochrone problems with shifted starting x-coordinates,
solved in parallel via `jax.vmap(problem.solve_jax)`.

## When to choose `solve()` vs. `solve_jax()`

| Use case | Entry point |
|---|---|
| Interactive use, real-time prints, plotting per iteration | `solve()` |
| Wall-clock `time_limit` or `continuous=True` | `solve()` |
| Per-iteration trajectories / weights / diagnostics | `solve()` |
| Batched solve over many initial conditions or parameters | `solve_jax()` |
| JIT-compile once, solve forever (MPC inner loop, scenario sweeps) | `solve_jax()` |
| `jax.grad` through the solver (best-effort, untested) | `solve_jax()` |

The split exists to make user intent explicit — a single dispatched
`solve()` would have to infer from arguments which path the caller wants,
and silent routing failures (like `continuous=True` quietly skipping the
Python loop) are exactly the failure class the two-method design exists to
prevent. See `plans/jax-pure-solve.md`'s Decision Log for the longer
discussion.

## Batched, disk-cached solves with `solve_batched()`

`jax.vmap(solve_jax)` and `solve_batched()` both run `B` SCP solves over
stacked boundary conditions; they differ in **who owns the batch axis**, and
that one difference is the whole point.

* `jax.vmap(solve_jax)` — *the caller* owns the axis. Maximally idiomatic,
  composes with `grad` / `scan`, but **in-process JIT only**: `jax.export` has
  no outer `vmap` rule (`call_exported` cannot be batched), so an exported
  artifact can never be re-vmapped, and every fresh process re-traces and
  re-compiles the entire loop.
* `solve_batched(x_initial=x0_batch)` — *the library* owns the axis. The vmap
  is applied **inside** the method, before any export, so the whole loop lowers
  to ordinary batched XLA ops and serializes cleanly. Under
  `save_compiled=True` it is written to the solver cache on the first run and
  deserialized on later processes — skipping the XLA compile that
  `jax.vmap(solve_jax)` pays on every launch.

```python
problem = ox.Problem(..., solver={"backend": "qpax"})
problem.settings.sim.save_compiled = True
problem.initialize()

# First process: traces, exports, writes <cache>/compiled_solve_batched_<hash>.jax
# Later processes: deserialize-and-.call, no compile.
batched = problem.solve_batched(x_initial=x0_batch, parameters=params)
# batched.x.shape == (B, N, n_states) — same pytree jax.vmap(solve_jax) returns.
```

### One rule: add a leading batch axis to whatever varies

`solve_batched` takes exactly `solve_jax`'s arguments. Every batchable input
has a *declared* (unbatched) shape the library knows — `(n_x,)` for the
boundary pins, `(N, n_x)` / `(N, n_u)` for the trajectory guesses, the
parameter's stored shape for each parameter, and `()` for the SCP
hyperparameters. An argument that matches its declared shape is **shared** by
every batch element; one with exactly one extra leading axis is **batched**
along it. Rank against the declared shape decides — never the leading-axis
value — so a shared `(B, n)` matrix parameter is never misread as batched.
(`B` in these shape patterns is plain notation for the batch size, not
syntax: it is never passed anywhere — like `jax.vmap`, the library reads it
off the leading axis of whatever you batched and cross-checks that all
batched inputs agree.)

```python
results = problem.solve_batched(
    x_initial=x0_batch,                  # (B, n_x): rank = declared+1 -> batched
    parameters={
        "gate_center": centers,          # (B, 3) vs declared (3,)  -> batched
        "gate_radius": 0.5,              # declared shape           -> shared
    },
)
```

For the rare entry the rank rule cannot express — e.g. batching a declared
rank-1 parameter element-wise — pass an explicit `in_axes` in `jax.vmap`'s
vocabulary, including its prefix semantics (`0` = batched along the leading
axis, `None` = shared; a prefix applies to everything below it; partial specs
fall back to the rank rule):

```python
results = problem.solve_batched(
    parameters={"weights": w},           # (B,) is also w's declared shape...
    in_axes={"parameters": {"weights": 0}},  # ...so say "batched" explicitly
)

results = problem.solve_batched(parameters=params, in_axes={"parameters": 0})
# prefix: every passed parameter is batched, exactly like jax.vmap's
# pytree-prefix in_axes; a bare in_axes=0 batches every passed input.
```

One deliberate divergence from `jax.vmap`: the *default* is "infer by rank",
not `0`-everything — an all-shared batched solve has no batch axis to find,
and inference is what makes mixed shared/batched dicts ergonomic.

### Hyperparameter sweeps: the `algorithm` dict

Solve-time mirrors construction-time: the `Problem` constructor configures
the SCP algorithm with a dict (`algorithm={"ep_tr": 1e-5, "lam_prox": 1.0}`),
and the solve methods take the same box with the same names as per-solve,
batchable inputs. Every `AlgorithmState` field is a runtime input riding the
state pytree, and the `algorithm` dict reaches any of them by name —
convergence thresholds (`ep_tr` / `ep_vb` / `ep_vc`), penalty weights
(`lam_prox` / `lam_vc` / `lam_cost`), the seed iterate, autotuner
hyperparameters. There is no whitelist: the field declaration is the
registration, and the `AlgorithmState` attribute docs are the reference.
(`max_iters` survives as a named kwarg — sugar for `algorithm={"k_max":
...}`; passing both raises, as does an entry that shadows any other named
kwarg's field. Construction-only keys of the constructor dict — `autotuner`,
`t_max` — raise pointing back at the constructor.)

* On `solve_jax`, each value matches the field's shape, or is a scalar
  broadcast to it: `solve_jax(algorithm={"ep_tr": 1e-6, "lam_prox": 2.0})`
  retraces nothing.
* On `solve_batched`, each value follows the rank rule against the field's
  shape, plus two **fill** forms for array-shaped fields: a scalar (shared)
  or a `(B,)` vector (one scalar per element, broadcast to the field shape).
  Exact shapes win when both parse; `in_axes={"algorithm": {name: 0}}`
  forces the per-element reading. All against one cached artifact:

```python
B = 8  # batch size. Just an int for building the arrays below — there is no
       # B argument; solve_batched infers it from the leading-axis lengths.
sweep = problem.solve_batched(algorithm={"ep_tr": jnp.logspace(-6, -3, B)})
weights = problem.solve_batched(algorithm={"lam_prox": jnp.array([0.5, 1.0, 4.0])})
budgets = problem.solve_batched(max_iters=jnp.array([5, 10, 20]))  # per-element caps
```

The batch runs until its slowest element converges or exhausts its own cap;
finished elements are frozen, not re-iterated, so each element matches the
corresponding single `solve_jax` run.

**Every built-in autotuner knob batches for free.** All numeric knobs of the
built-in autotuners are declared hyperparameters — the trust-region factors
(`gamma_1` / `gamma_2`), the acceptance thresholds (`eta_0` / `eta_1` /
`eta_2`), the weight clips, the cost-relaxation schedule — so any of them is a
`solve_jax` override and a `solve_batched` sweep target by name:

```python
# sweep the trust-region growth factor across a batch, one cached artifact
results = problem.solve_batched(algorithm={"gamma_1": jnp.linspace(1.5, 3.0, B)})
```

**New autotuners batch the same way.** A custom autotuner's tunable knobs
follow the same derivation: declare them as fields on a `HyperParams` subclass
— bare annotations, no decorator; subclassing applies the frozen-dataclass
transform and the JAX pytree registration, and the annotation picks the dtype
(`int` gets the iteration counter's, `float` the problem's) — assign an
instance to `self.hyper`, and read `state.hyper.<field>` inside
`update_weights` (see the `AutotuningBase` docstring). The declaration is the
registration — each field immediately works as a `solve_jax` override, a
`solve_batched` sweep target, and a runtime input of the exported artifact,
with zero edits anywhere else:

```python
from openscvx.algorithms import AutotuningBase, HyperParams

class MyHyper(HyperParams):
    ramp: float = 2.0

class MyAutotuner(AutotuningBase):
    def __init__(self, ramp: float = 2.0):
        self.hyper = MyHyper(ramp=ramp)

    def update_weights(self, state, ...):
        ...state.hyper.ramp...

results = problem.solve_batched(algorithm={"ramp": jnp.linspace(1.5, 3.0, B)})
```

Pick `solve_batched` when **cross-process cold-start dominates** — short-lived
processes (CI sweeps, notebook restarts, deployed workers) that otherwise pay
the compile every launch. Pick `jax.vmap(solve_jax)` for everything in-program,
where you own the transforms and want `grad` / `scan` composition. With
`save_compiled=False`, `solve_batched` is just an in-process batched wrapper and
the two produce identical results.

The exported artifact is closed and deployable, not composable: it is the
`call_exported`-has-no-outer-`vmap`-rule wall again. Use `solve_jax` for
user-driven `jax.vmap` / `jax.grad`; `solve_batched` for a cached batched solve.

### CVXPy cannot be exported

Export requires every op in the loop — the backend solve included — to lower to
serializable XLA. CVXPy's `iteration_callback` is a `jax.pure_callback`, and
`jax.export` cannot serialize host callbacks. So `PTRSolver.exportable` is
`False` for CVXPy (and `True` for QPAX / Moreau), and
`solve_batched(save_compiled=True)` on the CVXPy backend **raises a teaching
error** pointing at QPAX / Moreau rather than silently degrading to an uncached
in-process solve. With `save_compiled=False`, CVXPy still runs `B` sequential
in-process solves.

### Cache invalidation is correctness, not just freshness

The exported loop closes over far more than the symbolic problem does. Caching
on the symbolic hash alone would silently deserialize a **stale artifact** when
any of that extra state changes — a wrong-answer bug, not a cache miss. So
`get_solve_batched_cache_path` (`openscvx/utils/caching.py`) extends the
symbolic hash with everything additionally baked into the loop, each runtime
object contributing through its own `_hash_into` (the same protocol the symbolic
layer uses):

* the convex backend class + its `solver_args`,
* the algorithm + autotuner + initial weights,
* the discretizer scheme (class + hold type + ODE solver),
* the state/control scaling matrices `inv_S_x` / `inv_S_u`,
* the fixed batch size `B` (the artifact is exported at one `B`),
* the per-parameter shared/batched split (baked into the `jax.vmap` program),
  and
* the JAX version (exported artifacts are not guaranteed to survive an
  incompatible jax bump).

Change any of these — backend, a penalty weight, a state bound, which
parameters are batched — and you get a different cache path, so a stale
artifact is never reused. Everything the `algorithm` dict can reach — convergence
thresholds, `max_iters`, weights, autotuner hyperparameters — is deliberately
*not* in the key: those are runtime inputs on the state pytree, so one
artifact serves every setting of them.

`examples/abstract/brachistochrone_batched.py` shows both paths side by side;
re-running it is itself the cross-process demo.

## What `solve_jax()` returns

`solve_jax()` returns the same `OptimizationResults` type as `solve()`,
but built via
`OptimizationResults.from_final_state(state, problem=...)` instead of
`from_history(history, final_state, ...)`. The differences:

* **Per-iteration history is empty.** `X = [state.x]` and `U = [state.u]`
  are single-element lists so `result.x` / `result.u` continue to return
  the final iterate; every `*_history` field is `[]`. List growth doesn't
  fit inside `lax.while_loop`. If you need per-iteration trajectories, use
  `solve()` or wrap `solve_jax()` in `lax.scan` manually.
* **Post-process fields stay `None`.** `t_full`, `x_full`, `u_full`,
  `cost`, `ctcs_violation` are populated only by `post_process()`. They're
  outside the JAX pytree's children surface, so a batched `solve_jax`
  result doesn't force every consumer to handle `None` leaves. If you want
  to post-process, call `post_process()` per batch element after the
  batched solve.
* **`converged` is a `jnp.bool_`** under the JAX-pure path (a `(B,)` array
  under vmap), not a Python `bool` like `solve()` returns. Most uses
  (`if result.converged: ...`) work either way.

## `solve_jax()` arguments

```python
problem.solve_jax(
    x_initial=None,      # boundary-condition pin (full unified state vector,
                         #     ``jnp.nan`` at non-Fix entries — see
                         #     ``AlgorithmState.from_settings``); falls back
                         #     to the default from settings
    x_final=None,        # terminal pin, same conventions
    parameters=None,     # parameters dict for this solve; falls back to
                         #     ``self._parameters``
    *,
    x_guess=None,        # (N, n_x) state trajectory warm-start
    u_guess=None,        # (N, n_u) control trajectory warm-start
    max_iters=None,      # SCP iteration cap; sugar for algorithm={"k_max": ...};
                         #     falls back to the configured k_max
    algorithm=None,      # any AlgorithmState field by name (ep_tr, lam_prox,
                         #     ...) — the constructor's algorithm-dict names;
                         #     values match the field shape or fill it from a
                         #     scalar. Runtime inputs on the state pytree — no
                         #     value forces a retrace
)
```

Positional kwargs (rather than a single `inputs` pytree) keep
`jax.vmap(problem.solve_jax, in_axes=(0, 0, None))` ergonomic;
multi-argument gradient uses `jax.grad(..., argnums=(0, 1))`.

## Caveats

### CVXPy under `vmap` is sequential

The CVXPy backend's `iteration_callback`
(`openscvx/solvers/cvxpy_ptr_solver.py`) host-calls CVXPy through
`jax.pure_callback` with `vmap_method="sequential"`. Host
CVXPy is not thread-safe, so a `jax.vmap(problem.solve_jax)` over the
CVXPy backend runs `B` sequential CVXPy solves. The QPAX and Moreau
backends are pure JAX end-to-end and run in parallel under vmap.

### Moreau warm-start is bypassed

The Moreau backend's `_warm_start` carry is host-side mutable state that
`lax.while_loop` doesn't thread. Both `solve()` and `solve_jax()` cold-start
the inner Moreau solve every SCP iteration (see the Moreau module
docstring and `plans/jax-pure-solve.md`'s Decision Log 2026-05-27).
Restoring warm-start to the SCP loop requires threading `(x, z, s)`
through an `AlgorithmState.moreau_carry` field — a Future Extension.

### `jax.grad` is best-effort, untested

The QPAX backend's `solve_qp` is not differentiable (its convergence flag
costs reverse-mode autodiff). The CVXPy backend's `pure_callback` is
non-differentiable by default. End-to-end gradient validation through
`solve_jax` is a follow-up — `jax.grad(loss_fn)(x0)` will run without
erroring under QPAX (`solve_qp_primal` would be `custom_vjp`, but the
backend currently uses `solve_qp`), but the gradient's correctness is not
yet pinned down by a test against finite differences.

### Convergence under `vmap` with mixed-rate elements

`lax.while_loop` continues while *any* batch element still needs
iterations — converged elements would otherwise keep receiving body calls
while their peers iterate. `make_solve_loop`'s body selects the unchanged
state for converged elements (`jax.tree.map(jnp.where, ...)`), so a
batched solve agrees with the single-problem `solve_jax` on each element.
The cost is per-iteration work × slowest-element iteration count.
