# JAX-Traceable Autotuners

Rewrite every autotuner's `update_weights` as a pure functional update on a JAX-traceable `AlgorithmState` pytree, so any autotuner composes with `jax.vmap` / `jax.jit` / `jax.grad` and the SCP loop can later live inside `lax.while_loop`.

---

## Motivation

Today, autotuners update SCP weights by mutating an `AlgorithmState` dataclass — appending to Python lists (`augmented_lagrangian.py:301–303`), raising on edge cases (`augmented_lagrangian.py:296`), and branching on scalar `rho` via Python `if/elif` (`augmented_lagrangian.py:305–328`). They also return human-readable strings (`"Accept Higher"`, `"Reject"`) that the SCP loop's emitter prints (`penalized_trust_region.py:486`). None of that traces under JAX: tracers can't drive Python `if`, can't be raised, can't be appended to a list at trace time, and don't become strings.

This blocks every downstream JAX composition — `jax.jit(problem.solve)`, `jax.vmap(problem.solve)`, end-to-end `jax.grad` — all of them need the SCP body, autotuners included, to be a pure function of pytrees.

Current call site (`penalized_trust_region.py:464–467`):

```python
adaptive_state = self.autotuner.update_weights(
    state, candidate, self._jax_constraints, settings, params, self.weights
)
```

`update_weights` mutates `state` in place (via `state.accept_solution(candidate)` / `state.reject_solution(candidate)` and direct field assignment), mutates `candidate`, and returns a string. The caller then reads `state.pred_reduction_history[-1]` etc. for the emitter (`penalized_trust_region.py:495–506`).

Intended call site — `update_weights` returns the next state pytree, the SCP loop records history separately:

```python
state = self.autotuner.update_weights(
    state, candidate, self._jax_constraints, settings, params, self.weights,
)
history.record_iteration(state, candidate)            # CPU-side; only on the Python loop path
emission_data["adaptive_state"] = _adaptive_state_code_to_str(state.adaptive_state_code)
```

User-visible delta: none. `Problem.solve()`'s behavior is unchanged — same trajectories, same prints, same `.solve()` API. The change is structural: `update_weights` is now a pure function of pytrees, and the SCP loop body is one step closer to being JAX-traceable end-to-end.

---

## Approach

Four pieces change in sequence: pytree split, base-class contract, four autotuner rewrites, SCP loop caller update.

### 1. `AlgorithmState` splits into a JAX pytree + a CPU-side history

Today `AlgorithmState` (`openscvx/algorithms/base.py:267–328`) holds both hot iteration data (current `x`, `u`, weights) and append-only history lists (`X`, `U`, `discretizations`, `VC_history`, `lam_*_history`, `pred_reduction_history`, etc.). Lists can't be fixed-shape JAX pytree leaves and `.append`-based growth doesn't trace. Split them:

```python
# openscvx/algorithms/base.py

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AlgorithmState:
    """JAX pytree — current iterate only. All fields are jnp arrays / scalars."""
    x: jnp.ndarray             # (N, n_x)
    u: jnp.ndarray             # (N, n_u)
    x_prop: jnp.ndarray        # (N-1, n_x) — discretization output for the current iterate
    x_prop_plus: jnp.ndarray   # (N, n_x)   — impulsive output; zeros when not present
    lam_prox: jnp.ndarray      # (N, n_x + n_u)
    lam_vc: jnp.ndarray        # (N-1, n_x)
    lam_cost: jnp.ndarray      # scalar or (n_x,)
    lam_vb_nodal: jnp.ndarray  # (N, n_nodal)
    lam_vb_cross: jnp.ndarray  # (n_cross,)
    k: jnp.int32
    J_tr: jnp.float32
    J_vb: jnp.float32
    J_vc: jnp.float32
    J_nonlin: jnp.float32
    # Autotuner diagnostics — overwritten each iteration; lifted up here so
    # `update_weights` is a pure (state, candidate) -> state function.
    predicted_reduction: jnp.float32
    actual_reduction: jnp.float32
    acceptance_ratio: jnp.float32
    adaptive_state_code: jnp.int32     # enum: see §2 below

    def _replace(self, **changes) -> "AlgorithmState": ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, aux, children): ...


@dataclass
class AlgorithmHistory:
    """CPU-side. Grown per iteration by the SCP loop; never appears on the JAX boundary."""
    X: List[np.ndarray] = field(default_factory=list)
    U: List[np.ndarray] = field(default_factory=list)
    discretizations: List[DiscretizationResult] = field(default_factory=list)
    VC: List[np.ndarray] = field(default_factory=list)
    TR: List[np.ndarray] = field(default_factory=list)
    lam_prox: List[np.ndarray] = field(default_factory=list)
    lam_vc: List[Union[float, np.ndarray]] = field(default_factory=list)
    lam_cost: List[Union[float, np.ndarray]] = field(default_factory=list)
    lam_vb_nodal: List[np.ndarray] = field(default_factory=list)
    lam_vb_cross: List[np.ndarray] = field(default_factory=list)
    J_nonlin: List[float] = field(default_factory=list)
    J_lin: List[float] = field(default_factory=list)
    pred_reduction: List[float] = field(default_factory=list)
    actual_reduction: List[float] = field(default_factory=list)
    acceptance_ratio: List[float] = field(default_factory=list)
    adaptive_state: List[str] = field(default_factory=list)
    x_full: List[np.ndarray] = field(default_factory=list)
    x_prop_full: List[np.ndarray] = field(default_factory=list)

    def record_iteration(self, state: AlgorithmState, candidate: CandidateIterate) -> None:
        """Copy current-iterate fields off the pytree onto the CPU-side history lists."""
        ...
```

`state.accept_solution(candidate)` / `state.reject_solution(candidate)` go away. The replace-the-iterate semantics they encode become explicit functional updates inside `update_weights`:

```python
# Today:
state.accept_solution(candidate)         # appends cand.x, cand.u to state.X, state.U
# After:
return state._replace(x=candidate.x, u=candidate.u, ...)
```

The `state.A_d()` / `state.B_d()` / `state.x_prop()` / `state.x_prop_plus()` / etc. accessors (`base.py:468–544`) move onto `AlgorithmHistory` as methods that index into `history.discretizations[i]`. The autotuner no longer reads through them — the previous iterate's `x_prop` and `x_prop_plus` are direct fields on the pytree.

`from_settings` (`base.py:606`) produces both objects.

### 2. `AutotuningBase` contract

`update_weights` is now declared as a pure functional update:

```python
class AutotuningBase(ABC):
    @abstractmethod
    def update_weights(
        self,
        state: AlgorithmState,
        candidate: CandidateIterate,
        nodal_constraints: LoweredJaxConstraints,
        settings: Config,
        params: dict,
        weights: Weights,
    ) -> AlgorithmState:
        """Return the next-iterate state.

        Must be JAX-traceable: no `state` / `candidate` mutation, no string
        returns, no list appends, no raises on data-dependent conditions.
        Branching on iterate values goes through `jax.lax.cond` / `jnp.where`.
        """
```

Diagnostic returns (the per-iteration `"Accept Higher"` / `"Reject"` string + the float metrics the emitter reads) move into pytree-resident scalars on `AlgorithmState`. The adaptive-state string collapses to an int enum mirroring the `StatusCode` pattern that lives next to `PTRSolveResult`:

```python
# openscvx/algorithms/base.py
class AdaptiveStateCode(IntEnum):
    REJECT          = 0
    ACCEPT_HIGHER   = 1
    ACCEPT_CONSTANT = 2
    ACCEPT_LOWER    = 3
    INITIAL         = 4

_ADAPTIVE_STATE_NAMES = {
    AdaptiveStateCode.REJECT:          "Reject Higher",
    AdaptiveStateCode.ACCEPT_HIGHER:   "Accept Higher",
    AdaptiveStateCode.ACCEPT_CONSTANT: "Accept Constant",
    AdaptiveStateCode.ACCEPT_LOWER:    "Accept Lower",
    AdaptiveStateCode.INITIAL:         "Initial",
}

def adaptive_state_code_to_str(code: int) -> str:
    return _ADAPTIVE_STATE_NAMES[AdaptiveStateCode(int(code))]
```

The SCP loop (`penalized_trust_region.py:486`) converts the code back to a string for the printing path; under a future JAX trace, the int code stays on the pytree and the printing path is skipped.

### 3. `AugmentedLagrangian` rewrite

The if/elif over `rho` (`augmented_lagrangian.py:305–340`) becomes a `jnp.where` cascade inside a `jax.lax.cond` on `state.k == 1`:

```python
# openscvx/algorithms/autotuner/augmented_lagrangian.py — rewritten update_weights
def update_weights(self, state, candidate, nodal_constraints, settings, params, weights):
    candidate_x_prop = jnp.where(
        # x_prop_plus[1:] when impulsive, else x_prop — both fields live on the pytree
        ...,
        candidate.x_prop_plus[1:],
        candidate.x_prop,
    )
    nonlin_cost, nonlin_penalty, nodal_penalty = self.calculate_nonlinear_penalty(
        candidate_x_prop, candidate.x, candidate.u,
        state.lam_vc, state.lam_vb_nodal, state.lam_vb_cross, state.lam_cost,
        nodal_constraints, params, settings,
    )
    J_nonlin = nonlin_cost + nonlin_penalty + nodal_penalty

    lam_cost_next = jnp.where(
        state.k > self.lam_cost_drop,
        state.lam_cost * self.lam_cost_relax,
        weights.lam_cost,
    )

    def first_iter(state):
        return state._replace(
            x=candidate.x, u=candidate.u,
            x_prop=candidate.x_prop, x_prop_plus=candidate.x_prop_plus,
            lam_cost=lam_cost_next,
            J_nonlin=J_nonlin,
            adaptive_state_code=AdaptiveStateCode.INITIAL,
        )

    def later_iter(state):
        prev_J_nonlin = _prev_J_nonlin(state, ...)        # pure JAX — see calculate_nonlinear_penalty port
        actual    = prev_J_nonlin - J_nonlin
        predicted = prev_J_nonlin - candidate.J_lin

        # Force reject (rho = -inf) when predicted == 0; replaces today's raise.
        safe_pred = jnp.where(predicted == 0, 1.0, predicted)
        rho = jnp.where(predicted == 0, -jnp.inf, actual / safe_pred)

        is_reject          = rho < self.eta_0
        is_accept_higher   = (rho >= self.eta_0) & (rho < self.eta_1)
        is_accept_constant = (rho >= self.eta_1) & (rho < self.eta_2)
        # is_accept_lower implicit (else branch)
        accepted = ~is_reject

        # Per-element lam_prox update — compute both candidates, gate by bucket.
        lp_higher = jnp.minimum(self.lam_prox_max, self.gamma_1 * state.lam_prox)
        lp_lower  = jnp.maximum(self.lam_prox_min, self.gamma_2 * state.lam_prox)
        new_lam_prox = jnp.where(
            is_reject | is_accept_higher, lp_higher,
            jnp.where(is_accept_constant, state.lam_prox, lp_lower),
        )

        # Weights updates: compute once, gate by `accepted`.
        lam_vc_upd       = self._update_virtual_control_weights(candidate, candidate_x_prop, settings, state.lam_vc, new_lam_prox)
        lam_vb_nodal_upd = self._update_virtual_buffer_nodal_weights(candidate, nodal_constraints, params, state.lam_vb_nodal, new_lam_prox)
        lam_vb_cross_upd = self._update_virtual_buffer_cross_weights(candidate, nodal_constraints, params, state.lam_vb_cross, new_lam_prox)

        code = jnp.where(
            is_reject,          AdaptiveStateCode.REJECT,
            jnp.where(is_accept_higher,   AdaptiveStateCode.ACCEPT_HIGHER,
            jnp.where(is_accept_constant, AdaptiveStateCode.ACCEPT_CONSTANT,
                                          AdaptiveStateCode.ACCEPT_LOWER)),
        )

        return state._replace(
            x            = jnp.where(accepted, candidate.x, state.x),
            u            = jnp.where(accepted, candidate.u, state.u),
            x_prop       = jnp.where(accepted, candidate.x_prop, state.x_prop),
            x_prop_plus  = jnp.where(accepted, candidate.x_prop_plus, state.x_prop_plus),
            lam_prox     = new_lam_prox,
            lam_vc       = jnp.where(accepted, lam_vc_upd, state.lam_vc),
            lam_vb_nodal = jnp.where(accepted, lam_vb_nodal_upd, state.lam_vb_nodal),
            lam_vb_cross = jnp.where(accepted, lam_vb_cross_upd, state.lam_vb_cross),
            lam_cost     = lam_cost_next,
            J_nonlin     = J_nonlin,
            predicted_reduction = predicted,
            actual_reduction    = actual,
            acceptance_ratio    = rho,
            adaptive_state_code = code,
        )

    return jax.lax.cond(state.k == 1, first_iter, later_iter, state)
```

The four helpers `_update_virtual_control_weights`, `_update_virtual_buffer_nodal_weights`, `_update_virtual_buffer_cross_weights` (`augmented_lagrangian.py:118–215`) are line-by-line ports: `np.where` → `jnp.where`, `np.maximum` → `jnp.maximum`, `np.minimum` → `jnp.minimum`, `lam_vb_nodal.copy()` → just rebuild with `jnp.where`. The Python `for idx, constraint in enumerate(nodal_constraints.nodal)` loops stay — `nodal_constraints.nodal` is a static-length list of jit'd closures (built at `problem.py:766–768`), so the loop unrolls at trace time.

`calculate_nonlinear_penalty` (`base.py:163–235`) is also ported: `np.maximum` / `np.abs` / `np.sum` → `jnp.*`, Python `for` over `nodal_constraints.nodal` / `nodal_constraints.cross_node` stays (same static-list argument).

**The 4-branch `jnp.where` cascade is not a slowdown.** All three "accept" branches compute the same `lam_vc` / `lam_vb` updates today; the rewrite computes them once and gates with `accepted`. The only redundant work is the two `lam_prox` candidate updates — cheap scalar arithmetic.

### 4. `AdaptiveProximalWeight` rewrite

Same shape as AL (`adaptive_proximal_weight.py:61–165` has the identical 4-way if/elif over `rho`). Differs in that `lam_vc` / `lam_vb_nodal` / `lam_vb_cross` are carried unchanged from `state` instead of recomputed:

```python
return state._replace(
    x            = jnp.where(accepted, candidate.x, state.x),
    u            = jnp.where(accepted, candidate.u, state.u),
    lam_prox     = new_lam_prox,
    lam_vc       = state.lam_vc,
    lam_vb_nodal = state.lam_vb_nodal,
    lam_vb_cross = state.lam_vb_cross,
    ...
)
```

The `_copy_virtual_weights` helper (`adaptive_proximal_weight.py:52–59`) disappears — it was just `candidate.lam_vc = state.lam_vc; candidate.lam_vb_nodal = ...`, which the functional update covers directly.

### 5. `ConstantProximalWeight` / `RampProximalWeight` ports

Mechanical:

- **ConstantProximalWeight** (`constant_proximal_weight.py:34–65`): drop `state.accept_solution(candidate)`, return `state._replace(lam_cost=..., lam_prox=state.lam_prox, x=candidate.x, u=candidate.u, adaptive_state_code=AdaptiveStateCode.ACCEPT_CONSTANT)`. The `state.k > self.lam_cost_drop` branch becomes `jnp.where`.
- **RampProximalWeight** (`ramp_proximal_weight.py:38–80`): `np.minimum` → `jnp.minimum`, `np.all` → `jnp.all`. The `was_at_max` branch determines only the printed string — it becomes a `jnp.where` on `adaptive_state_code` (ACCEPT_CONSTANT vs ACCEPT_HIGHER).

### 6. SCP loop caller update

`penalized_trust_region.py:464–467` switches from mutation to assignment, and the history-recording moves out of `update_weights`:

```python
state = self.autotuner.update_weights(
    state, candidate, self._jax_constraints, settings, params, self.weights,
)
history.record_iteration(state, candidate)

emission_data["adaptive_state"] = adaptive_state_code_to_str(state.adaptive_state_code)
# Emitter reads scalars directly off the pytree — no more *_history[-1] lookups.
if use_full_metrics:
    emission_data.update({
        "J_nonlin":         float(state.J_nonlin),
        "J_lin":            float(candidate.J_lin),
        "pred_reduction":   float(state.predicted_reduction),
        "actual_reduction": float(state.actual_reduction),
        "acceptance_ratio": float(state.acceptance_ratio),
    })
```

`history` is threaded through `_subproblem` / `step` / `solve` alongside `state`. The plumbing change is mechanical but touches every method in `penalized_trust_region.py` that currently reads from `state.X` / `state.discretizations` / `state.*_history`.

---

## What's Out of Scope

- **The SCP loop becoming `lax.while_loop`.** This plan makes the *body* JAX-traceable; the outer loop stays Python. The while-loop migration is its own piece of work.
- **Per-iteration JAX function (`make_scp_iteration`).** Likewise — composing a JIT-friendly iteration body around the now-traceable autotuner is a downstream concern.
- **`OptimizationResults` pytree registration.** The output object is untouched; `AlgorithmHistory` is internal.
- **Removing Python-side history recording.** The split makes it *possible* to skip recording under trace, but this plan always records. Future plans gate `history.record_iteration` on the loop path.
- **`moreau_carry` or other backend-specific pytree fields on `AlgorithmState`.** The field list above covers what the autotuners need. Other parts of the pipeline can add fields by extending the pytree later — that's the whole point of registering it.
- **A non-traceable autotuner base class.** If a future autotuner genuinely can't trace (e.g., calls a non-differentiable external service), it can subclass a sibling `PythonOnlyAutotuningBase` and route through a separate dispatch path. Not designed here.
- **`CandidateIterate` becoming a full JAX pytree.** Today `CandidateIterate` (`base.py:24–41`) is a mutable dataclass with `Optional[np.ndarray]` fields written incrementally during a subproblem solve. `update_weights` reads it as-is; making it a registered pytree is only useful once the *whole* iteration body is JAX-pure, which is downstream. Treat it as a structured-but-numpy input for now and convert leaves on the boundary if needed.

---

## Open Questions

1. **Where do the diagnostic floats (`predicted_reduction`, `actual_reduction`, `acceptance_ratio`, `adaptive_state_code`) live — on `AlgorithmState` or on `CandidateIterate`?** **Recommended:** on `AlgorithmState`. They're outputs of `update_weights`; the candidate is its input. Putting them on the candidate would force `CandidateIterate` to also become a registered pytree, doubling the surface area for a marginal semantic benefit. The trade-off — these are technically per-candidate values appearing on a per-iterate type — is acceptable since the emitter consumes them once per iteration and they get overwritten each cycle.

2. **Frozen dataclass + custom `_replace`, or `flax.struct.dataclass`?** **Recommended:** stdlib frozen dataclass + ~15 lines of `tree_flatten` / `tree_unflatten` / `_replace`. Flax isn't currently a dependency; pulling it in for one decorator isn't worth the install footprint.

3. **`x_prop` / `x_prop_plus` on `AlgorithmState` — kept up-to-date by every `update_weights`, or computed lazily from `history.discretizations[-1]`?** **Recommended:** kept on the pytree. AL reads the *previous* iterate's `x_prop` (`augmented_lagrangian.py:270–273`) to compute the predicted reduction; if it lives only in `history`, the autotuner has to reach into a CPU-side list, which defeats the point. Cost: one extra pytree leaf per state.

4. **Does `calculate_nonlinear_penalty` (`base.py:163–235`) trace cleanly with the Python `for` loop?** **Recommended:** yes — `nodal_constraints.nodal` is built at `problem.py:766–768` as a static-length list of jit'd closures, and `constraint.func` / `constraint.nodes` are static Python objects. The loop unrolls at trace time. **Confirm during implementation** by running the regression test under `jax.jit`.

5. **Reference trajectory for the regression test: committed fixture, or generated on-the-fly from a tagged commit?** **Recommended:** generated on-the-fly. Add a `tests/algorithms/autotuner/conftest.py` that runs the pre-rewrite AL on brachistochrone via a tagged-commit checkout in CI, caches the result locally. Avoids committing a ~MB-scale fixture for what's structurally a one-time test.

6. **Should `update_weights`'s return type include `state` only, or `(state, candidate)`?** Bare question — today `candidate` is mutated (`candidate.lam_prox = ...`). After the rewrite, `candidate`'s only relevant downstream reader is the emitter (`candidate.J_lin`, `candidate.J_nonlin`). If we move `J_lin` onto state too, `candidate` becomes read-only and the contract is cleaner. But `J_lin` is set by the discretizer / subproblem code, not the autotuner, so it conceptually belongs on the candidate. Decide once the SCP loop caller change is in flight.

---

## Checklist

### Phase 1 — pytree split

- [x] In `openscvx/algorithms/base.py:267`, replace `AlgorithmState` with a frozen dataclass holding only JAX-traceable fields per §1.
- [x] Register `AlgorithmState` via `jax.tree_util.register_pytree_node_class`. Add `_replace`, `tree_flatten`, `tree_unflatten`.
- [x] Add `AlgorithmHistory` dataclass at `base.py`. Move every list field (`X`, `U`, `discretizations`, `VC_history` → `VC`, `TR_history` → `TR`, `lam_*_history`, `J_*_history`, `*_reduction_history`, `acceptance_ratio_history`, `x_full`, `x_prop_full`) onto it; add `adaptive_state: List[str]`.
- [x] Implement `AlgorithmHistory.record_iteration(state, candidate)` — copies the relevant `state.*` and `candidate.*` fields onto the history lists.
- [x] Delete `AlgorithmState.accept_solution` and `reject_solution` (`base.py:329–394`). Every caller is rewritten.
- [x] Move `state.A_d()` / `B_d()` / `C_d()` / `D_d()` / `E_d()` / `x_prop()` / `x_prop_plus()` / `V_history` accessors (`base.py:414–544`) onto `AlgorithmHistory` as methods that index into `history.discretizations[i]`.
- [x] Move the `state.lam_prox` / `state.lam_*` `@property` getters that read from history (`base.py:546+`) onto `AlgorithmHistory`; on `AlgorithmState`, these are plain fields.
- [x] Update `from_settings` (`base.py:606`) to produce both `AlgorithmState` and `AlgorithmHistory`.
- [x] Update every internal accessor caller — grep for `state.X[`, `state.U[`, `state.A_d(`, `state.x_prop(`, `state.discretizations`, `state.*_history` — and route through `history`. Hotspots: `penalized_trust_region.py:440–525`, propagation code, post-process.

### Phase 2 — `AutotuningBase` contract

- [x] Update `AutotuningBase.update_weights` signature at `base.py:238` — return `AlgorithmState`, drop the `str` return. Document the pure-functional contract in the docstring.
- [x] Add `AdaptiveStateCode` IntEnum and `_ADAPTIVE_STATE_NAMES` / `adaptive_state_code_to_str` helper at `base.py`, alongside `AutotuningBase`.

### Phase 3 — autotuner rewrites

- [x] Rewrite `AugmentedLagrangian.update_weights` (`augmented_lagrangian.py:217`) per §3.
- [x] Port `_update_virtual_control_weights` / `_update_virtual_buffer_nodal_weights` / `_update_virtual_buffer_cross_weights` (`augmented_lagrangian.py:118–215`) — `np.*` → `jnp.*`, keep Python `for` over `nodal_constraints.nodal`.
- [x] Replace `if predicted_reduction == 0: raise ValueError` (`augmented_lagrangian.py:296`) with `rho = jnp.where(predicted == 0, -jnp.inf, actual / safe_pred)`. The `-inf` falls into the reject bucket — same outcome, deterministic under trace.
- [x] Port `calculate_nonlinear_penalty` (`base.py:163`) — `np.*` → `jnp.*`, keep Python `for` loops over `nodal_constraints.nodal` and `nodal_constraints.cross_node`.
- [x] Rewrite `AdaptiveProximalWeight.update_weights` (`adaptive_proximal_weight.py:61`) per §4 — same `lax.cond` + `jnp.where` cascade as AL, with `lam_vc` / `lam_vb_*` carried from state. Delete `_copy_virtual_weights` (`:52`).
- [x] Port `ConstantProximalWeight.update_weights` (`constant_proximal_weight.py:34`) per §5 — drop `state.accept_solution`, return `state._replace(...)`, `state.k > self.lam_cost_drop` → `jnp.where`.
- [x] Port `RampProximalWeight.update_weights` (`ramp_proximal_weight.py:38`) per §5 — `np.minimum` / `np.all` → `jnp.*`, `was_at_max` branch → `jnp.where` on `adaptive_state_code`.

### Phase 4 — SCP loop caller

- [x] Update `penalized_trust_region.py:465` — replace `adaptive_state = self.autotuner.update_weights(state, ...)` with `state = self.autotuner.update_weights(state, ...)`.
- [x] Thread `history: AlgorithmHistory` through `_step` / `step` / `_subproblem` / any sibling methods that today read `state.*_history`.
- [x] After `update_weights`, call `history.record_iteration(state, candidate)`.
- [x] Replace emitter reads (`penalized_trust_region.py:495–506`) — `state.pred_reduction_history[-1]` → `float(state.predicted_reduction)`, same for `actual_reduction`, `acceptance_ratio`. Map `state.adaptive_state_code` to a string for the `"adaptive_state"` field via `adaptive_state_code_to_str`.
- [x] Audit `openscvx/problem.py` for `self._state.*_history` reads (the print-thread drain, the post-processing code path) — route through `history` instead.

### Phase 5 — tests

- [x] Add `tests/algorithms/test_algorithm_state_pytree.py` — assert `AlgorithmState` round-trips through `jax.tree_util.tree_flatten` / `tree_unflatten`; `_replace` returns a new frozen instance; `jax.tree_map` on a state produces a state with the same shape.
- [~] ~~Add `tests/algorithms/autotuner/test_augmented_lagrangian_jax.py` — brachistochrone regression vs pre-rewrite trajectory.~~ **Skipped:** `tests/test_brachistochrone.py::test_autotuning` already sweeps all four autotuners against the analytic cycloid solution; that's the real acceptance gate and it passes. Adding a tagged-commit regression fixture is redundant.
- [~] ~~Add `tests/algorithms/autotuner/test_adaptive_proximal_weight_jax.py`.~~ **Skipped:** same reason — `test_autotuning[adaptive_proximal]` covers it.
- [~] ~~Add `tests/algorithms/autotuner/test_constant_proximal_weight_jax.py` and `test_ramp_proximal_weight_jax.py`.~~ **Skipped:** same reason — `test_autotuning[constant_proximal]` / `[ramp_proximal]` cover both.
- [x] Add `tests/algorithms/autotuner/test_update_weights_jit.py` — `jax.jit(autotuner.update_weights)` returns the same state as the bare call, for each autotuner.
- [x] Confirm full suite stays green: `pytest -n auto -m 'not integration'`.
- [x] Port the existing `tests/test_autotuning.py` (1743 → 1098 lines) to the new contract. Six tests that asserted the old mutate-in-place API were deleted; the 36 semantic survivors all pass.

### Phase 6 — docs

- [x] Update `AutotuningBase` docstring (`base.py:115`) to spell out the JAX-traceable contract.
- [x] Update each autotuner subclass docstring with a one-line note that `update_weights` is a pure functional update.
- [x] No user-facing docs change.

---

## Future Extensions

This plan's direct downstream is **`plans/batchable-problem.md`**, which uses the now-traceable `update_weights` and `AlgorithmState` pytree to make the SCP loop itself a `lax.while_loop`, register `OptimizationResults` as a pytree, and turn `Problem.solve()` into a JAX-pure entry point. Not repeated here.

- **A non-traceable autotuner base class.** If a future autotuner genuinely needs Python control flow, route it through `PythonOnlyAutotuningBase` and a sibling dispatch path. Revisit if and when.
- **Drop `predicted_reduction` / `actual_reduction` / `acceptance_ratio` from `AlgorithmState`.** If the diagnostic float fields turn out to bloat the pytree carry across `lax.while_loop`, move them to a separate `AutotunerMetrics` pytree returned alongside the state. Profile first.
- **`CandidateIterate` as a registered pytree.** Today the candidate is a mutable dataclass with `Optional[np.ndarray]` fields; tracers autoconvert at the boundary so the autotuner traces fine, but a registered pytree would make the autotuner's interface fully typed. Deferred until the discretizer / subproblem path that produces the candidate is JAX-pure end-to-end.

---

## Decision Log

- 2026-05-16 — chose **JAX-traceability as a contract on every autotuner** rather than gating with an `is_jax_traceable` flag. **Why.** A flag would create a two-tier autotuner ecosystem ("works with batching" vs "doesn't") that exists only to accommodate one or two classes (AL, `AdaptiveProximalWeight`) that turn out to be straightforwardly rewriteable as `jnp.where` cascades. The default autotuner has to support every JAX transform for batched/jitted solves to be a default-on feature; a flag would still force users with the default to swap autotuners just to vmap. Better to fix them once and require future autotuners to clear the same bar. **Trade-off accepted.** A future autotuner that genuinely can't trace (e.g., calls a non-differentiable external solver) can't live as an `AutotuningBase` subclass; it needs a separate base class and dispatch path. Acceptable — that's a different problem with different ergonomics. **Revisit if** such an autotuner shows up as a real user request.

- 2026-05-16 — chose to **split `AlgorithmState` into pytree + history** rather than making the existing class a pytree. **Why.** The current `AlgorithmState` mixes hot iteration data (current `x`, `u`, weights) with append-only histories (lists of trajectories, lists of discretizations). Lists can't be pytree leaves at fixed shape; preallocating to `k_max` slots would balloon memory and break the printing path that iterates `state.X` in real time. Two objects — one JAX pytree (current iterate only) + one Python-side history (lists) — keeps each clean. **Trade-off accepted.** Two objects to thread through the algorithm; callers that read history (post-processing, printing) switch from `state.X` to `history.X`. Mechanical change. **Revisit if** the split makes the algorithm body harder to read — but I expect the opposite (the JAX-pure path is shorter when histories live elsewhere).

- 2026-05-16 — chose **diagnostic floats (`predicted_reduction`, `actual_reduction`, `acceptance_ratio`, `adaptive_state_code`) live on `AlgorithmState`** rather than on `CandidateIterate` or in a separate `AutotunerMetrics` pytree. **Why.** They're outputs of `update_weights`; the candidate is its input. Putting them on the candidate would force `CandidateIterate` to also be a registered pytree, doubling the surface area for a marginal semantic benefit. A separate metrics pytree is cleaner conceptually but trades one return value for two on the autotuner contract. **Trade-off accepted.** These are technically per-candidate values living on a per-iterate type; the emitter overwrites them each cycle, so there's no aliasing risk. **Revisit if** the pytree carry across `lax.while_loop` shows up as a memory cost — the natural fix is to split them off into `AutotunerMetrics`.

- 2026-05-16 — chose **`AdaptiveStateCode` IntEnum on the pytree** rather than stripping the adaptive-state string on the JAX path. **Why.** Mirrors the `StatusCode` int-enum pattern about to land in `PTRSolver`. Cost: one `jnp.int32` per iterate. Benefit: a vmapped solve still reports per-batch-element adaptive states, which is exactly the kind of debugging output users will want when batched solves diverge. **Trade-off accepted.** The Python-loop caller has to map the int back to a string for printing; the helper is ~5 lines. **Revisit if** the enum grows unwieldy — but the four-bucket bias of trust-region SCP is structural, not implementation-specific.
