# SCP Split Iteration Timing

**Status**: Implemented  
**Date**: 2026-06-16

---

## Motivation

The current Python-loop `solve()` path reports a single `Step (ms)` column for every iteration, which conflates two qualitatively different costs:

- **Discretization** — the JAX/XLA computation that integrates the linearized dynamics and packs `SubproblemData` (steps 1–4 of the fused body).
- **Convex solve** — the backend call that solves the QP/conic subproblem (step 5, via the solver callback).

These are the call sites:
- Single-block timing: `openscvx/algorithms/scvx/penalized_trust_region.py:220–223`
- Column declaration: `penalized_trust_region.py:71` (`"subprop_time"`, `"Step (ms)"`)
- Emission key: `penalized_trust_region.py:255`

Splitting these lets users immediately see whether iteration time is dominated by XLA dispatch (discretization) or backend solve time (CVXPy / QPAX), without running a separate profiler.

**Intended call site**: `PenalizedTrustRegion.step()` on the Python `solve()` path only. The `solve_jax` / `solve_batched` paths drive the fused `_iteration_fn` inside `lax.while_loop`, where wall-clock timing from Python is meaningless and these changes have no effect.

---

## Approach

### 1. Expose sub-factories from `iteration.py`

Refactor the internals of `make_scp_iteration` (`iteration.py:89`) into two reusable sub-factories, then expose them publicly. The public signature and semantics of `make_scp_iteration` are **unchanged**.

**`make_scp_prepare(dis_continuous, dis_impulsive, jax_constraints, settings)`**  
Returns `prepare_fn(state, params) -> SubproblemData`.  
Contains exactly the code at `iteration.py:251–288`:
- `_discretize(state.x, state.u, params)` (steps 1–2)
- `_linearize_constraints(state.x, state.u, params)` (step 3)
- Pack and return `SubproblemData` (step 4)

**`make_scp_finalize(dis_continuous, dis_impulsive, autotuner, jax_constraints, settings)`**  
Returns `finalize_fn(state, solution, params) -> (next_state, IterationDiagnostics)`.  
Contains exactly the code at `iteration.py:295–336`:
- Discretize the candidate: `_discretize(solution.x, solution.u, params)` (step 6a)
- Compute TR/VC/VB metrics and fold through autotuner (steps 6b–6c)
- Build and return `IterationDiagnostics`

**`make_scp_iteration`** becomes a thin combiner — no logic changes, no duplication:
```python
def iteration_fn(state, params):
    data = prepare_fn(state, params)
    solution = solver_callback(state, data)
    return finalize_fn(state, solution, params)
```

### 2. `PenalizedTrustRegion.build_iteration()` — minimal side-effect extension

`build_iteration` (`penalized_trust_region.py:159`) already closes over every needed component. Alongside building the fused body it will also build and JIT-wrap the two halves:

```python
self._prepare_fn = jax.jit(make_scp_prepare(
    dis_continuous, dis_impulsive, jax_constraints, settings))
self._finalize_fn = jax.jit(make_scp_finalize(
    dis_continuous, dis_impulsive, self.autotuner, jax_constraints, settings))
self._solver_callback = solver_callback
```

These three attributes serve only `step()`; `solve_jax` / `solve_batched` continue to use the fused `_iteration_fn` returned to `Problem.initialize()`.

### 3. `PenalizedTrustRegion.step()` — split timing

Replace `penalized_trust_region.py:220–223`:

```python
t0 = time.time()
next_state, diag = self._iteration_fn(state, params)
jax.block_until_ready((next_state, diag))
step_time = time.time() - t0
```

With the three-phase sequence:

```python
t0 = time.time()
data = self._prepare_fn(state, params)
jax.block_until_ready(data)
t_dis = (time.time() - t0) * 1000.0

t0 = time.time()
solution = self._solver_callback(state, data)
jax.block_until_ready(solution)
t_cvx = (time.time() - t0) * 1000.0

next_state, diag = self._finalize_fn(state, solution, params)
jax.block_until_ready((next_state, diag))
```

### 4. Column/emission changes

Replace the single `"subprop_time"` / `"Step (ms)"` column (`penalized_trust_region.py:71`) with two STANDARD-verbosity columns:

```python
Column("t_dis", "Dis (ms)", 9, "{:6.2f}", min_verbosity=Verbosity.STANDARD),
Column("t_cvx", "Cvx (ms)", 9, "{:6.2f}", min_verbosity=Verbosity.STANDARD),
```

Update `emission_data` at `penalized_trust_region.py:253–255` accordingly.

---

## Files changed

| File | Change |
|------|--------|
| `openscvx/algorithms/scvx/iteration.py` | Add `make_scp_prepare` and `make_scp_finalize`; refactor `make_scp_iteration` to delegate to them |
| `openscvx/algorithms/scvx/penalized_trust_region.py` | `build_iteration()` stores split halves; `step()` uses split timing; swap `subprop_time` column for `t_dis` + `t_cvx` |

No changes to `base.py`, `problem.py`, `history.py`, or any test files.

---

## Out of Scope

- `solve_jax` / `solve_batched` paths — the fused `_iteration_fn` inside `lax.while_loop` is unchanged.
- The `Algorithm` ABC — `build_iteration` return type and `initialize()` signature stay the same.
- `AlgorithmHistory` — no new timing fields are stored; split times are display-only.
- Warming the two split-half JIT caches in `Problem.initialize()` — the first `step()` call pays the trace cost (same behavior as before the fused-body refactor).
- Other algorithm implementations (none exist beyond PTR today).

---

## Open Questions

- **Warmup of split halves**: Under `save_compiled`, the fused `_iteration_fn` uses `jax.export` wrappers as inner solvers (see `problem.py:1188–1196`). The split halves built in `build_iteration()` will use the same wrappers passed in — are those safe to call outside the fused JIT context on the first `step()` call before the XLA artifact is loaded?

---

## Decision Log

**2026-06-16 — Drop `"Step (ms)"`, replace with `"Dis (ms)"` + `"Cvx (ms)"`**  
The finalize phase (candidate discretization + autotuner weight update) is a small, relatively-constant overhead compared to the two variable phases, and exposing it as a third column adds clutter without actionable signal. Rejected: keeping all three columns.

**2026-06-16 — Split halves use the same inner-solver wrappers as the fused body**  
`build_iteration()` receives whichever solver wrappers `Problem.initialize()` has constructed (either `jax.export` under `save_compiled` or plain `jax.jit`). The split halves capture those same wrappers, consistent with the fused body. If `jax.export` wrappers prove unsafe when called outside their originating JIT boundary, the fix is to have `build_iteration` receive a second pair of `jax.jit` inner solvers — mirroring how `Problem.initialize()` already builds `_iteration_fn_jit_inner` for `solve_batched` (see `problem.py:1189–1199`). That is deferred until there is evidence it is needed. Rejected: always using plain `jax.jit` solvers for the split halves unconditionally.
