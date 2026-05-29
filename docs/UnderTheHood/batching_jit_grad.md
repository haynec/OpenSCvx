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
    max_iters=None,      # SCP iteration cap; non-default rebuilds the
                         #     cached ``lax.while_loop`` closure (one extra
                         #     trace; subsequent calls at the same cap hit
                         #     the cache)
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
