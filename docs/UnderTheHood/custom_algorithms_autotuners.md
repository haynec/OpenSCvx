---
description: >-
  How to swap OpenSCvx's SCP iteration and penalty-weight schedule by subclassing
  the `Algorithm` and `AutotuningBase` bases without forking the solver.
---

# Custom Algorithms and Autotuners

OpenSCvx solves through a pluggable SCP loop: an **algorithm** owns the
iteration (how a subproblem is built, when the loop is converged) and an
**autotuner** owns the penalty-weight update applied each step. Both are
ordinary Python classes behind small abstract bases — `Algorithm` and
`AutotuningBase`, importable from `openscvx.algorithms` — so swapping either
out is a subclass, not a fork.

Most users never touch these: the default
`PenalizedTrustRegion` + `AugmentedLagrangian` pair covers the common case.
Reach for this page when you want a different penalty schedule, a different
convergence rule, or a whole new SCP variant — and you want it to keep
working under `solve()`, `solve_jax()`, and `solve_batched()` alike.

Both bases live under `openscvx.algorithms`:

```python
from openscvx.algorithms import Algorithm, AutotuningBase, HyperParams
```

## A custom autotuner

An autotuner is a **pure functional update on the iterate**. Its one required
method, `update_weights`, takes the current `AlgorithmState`, the just-solved
subproblem `candidate`, the constraints, settings, and parameters, and returns
the *next* `AlgorithmState`:

```python
import jax.numpy as jnp
from openscvx.algorithms import AdaptiveStateCode, AutotuningBase, HyperParams


class MyHyper(HyperParams):
    ramp: float = 2.0


class MyAutotuner(AutotuningBase):
    COMPUTES_ACCEPTANCE_METRICS = False  # this tuner never rejects an iterate

    def __init__(self, ramp: float = 2.0):
        self.hyper = MyHyper(ramp=ramp)  # assign self.hyper FIRST

    def update_weights(self, state, candidate, nodal_constraints, settings, params):
        return state.replace(
            # accept the candidate: its trajectory becomes the next iterate
            x=candidate.x,
            u=candidate.u,
            x_prop=candidate.x_prop,
            x_prop_plus=candidate.x_prop_plus,
            # ramp the proximal weight every iteration
            lam_prox=state.lam_prox * state.hyper.ramp,
            # jnp.int32, not bare asarray: the code must keep the state field's
            # strong int32 dtype or the lax.while_loop carry check fails.
            adaptive_state_code=jnp.int32(AdaptiveStateCode.ACCEPT_HIGHER),
        )
```

Three rules carry the whole contract.

### The returned state *is* the next iterate

The SCP loop discards everything except what `update_weights` returns.
Accepting the subproblem's candidate therefore means **carrying its
trajectory onto the returned state** — `x` / `u` / `x_prop` / `x_prop_plus`
from `candidate` — while rejecting means keeping the previous fields and
adjusting only the weights.

!!! warning "The silent no-op"
    An update that never copies the candidate produces a solver that runs to
    `k_max` without the iterate ever moving — no error, just a flat history.
    If a custom tuner seems to "not converge," check that the accept branch
    copies all four trajectory fields.

Express any accept/reject branching with `jax.lax.cond` or `jnp.where`, not a
Python `if` on iterate values: `update_weights` is traced (it runs inside the
`lax.while_loop` of `solve_jax` / `solve_batched`, vmapped per batch element),
so it must not mutate `state` or `candidate`, raise on data-dependent
conditions, or return strings.

### Declare tunable knobs as `HyperParams`, not `self` attributes

A numeric knob a user might sweep (a ramp factor, a relaxation iteration) must
**not** be read off `self` inside `update_weights` — a plain Python attribute
is baked into the trace and invisible to every override channel. Declare it on
a `HyperParams` subclass, assign an instance to `self.hyper`, and read it from
`state.hyper`.

The declaration *is* the registration. With nothing else, the field becomes:

* a **per-solve override** — `problem.solve_jax(algorithm={"ramp": 1.5})`;
* a **batchable sweep target** —
  `problem.solve_batched(algorithm={"ramp": jnp.linspace(1.1, 2.0, 8)})`;
* a **runtime input** of the exported `solve_batched` artifact, so changing it
  needs no recompile.

!!! danger "Assign `self.hyper` first"
    Declared knobs are also readable and writable as bare attributes
    (`autotuner.ramp = 3.0`) for ergonomics — a proxy on `AutotuningBase`
    routes the access into `hyper`. That proxy only works once `self.hyper`
    exists, so assign it as the **first** line of `__init__`. A knob assigned
    before `hyper` exists lands as a plain instance attribute the proxy cannot
    see, and your override channels silently break.

### Acceptance metrics are opt-in

`COMPUTES_ACCEPTANCE_METRICS` (default `True`) tells the loop whether
`update_weights` produces a predicted/actual reduction and an acceptance
ratio. The loop records and prints those diagnostics only when it is `True`.
A tuner that never rejects an iterate — like the built-in
`ConstantProximalWeight` and `RampProximalWeight` — sets it `False`.

To add columns to the iteration table, set the class attribute `COLUMNS` to a
list of `openscvx.utils.printing.Column` specs; the algorithm concatenates
them into its own table via `get_columns`.

## A custom algorithm

An `Algorithm` owns the SCP iteration end to end: it *builds* the JAX-pure
iteration body, *stores* it, and *drives* it one step at a time. All iteration
state lives on `AlgorithmState` (JAX-traceable) and `AlgorithmHistory`
(Python-side) threaded explicitly through `step()` — avoid storing mutable
iteration state on `self`.

The surface is six methods. `converged` has a working default; the other five
are abstract.

### `__init__` — end with `super().__init__`

The base records the weights, autotuner, and convergence parameters. Every
parameter is required, so the ABC is not yet another place defaults are
declared — your subclass owns the user-facing defaults and forwards them:

```python
from openscvx.algorithms import Algorithm
from openscvx.algorithms.weights import Weights
from openscvx.algorithms import AugmentedLagrangian


class MyAlgorithm(Algorithm):
    def __init__(self, autotuner=None, k_max=200, **weight_kwargs):
        super().__init__(
            weights=Weights.build(**weight_kwargs),
            autotuner=autotuner if autotuner is not None else AugmentedLagrangian(),
            k_max=k_max,
            t_max=None,
            ep_tr=1e-4,
            ep_vb=1e-4,
            ep_vc=1e-8,
        )
```

### `build_iteration` — fuse the step

`Problem` assembles the discretization solvers, lowered constraints, and
convex-solver callback, then asks the algorithm to fuse them into one JAX-pure
`(state, params) -> (next_state, diagnostics)` body. The algorithm owns this
because the fusion is algorithm-specific (which autotuner runs, which penalty
terms are assembled). PTR's implementation is a thin wrapper around
`make_scp_iteration` (`openscvx/algorithms/scvx/iteration.py`) that threads its
`autotuner` into the fused body. The returned body must advance `state.k` by
one per call — every solve path terminates on `k` reaching `k_max`.

### `initialize` and `step` — store, then drive

`Problem.initialize()` builds and JIT-warms the body, then hands it back via
`initialize(iteration_fn, emitter)` for the algorithm to store. `step()` is the
Python-loop driver: it calls the stored body, records the per-iteration
diagnostics into `history`, emits progress, and returns
`(next_state, converged)`.

### `converged` — the one-method convergence hook

`converged(state)` is the only piece of convergence policy that is
algorithm-specific; it has a concrete default, so override it only to change
the rule:

```python
def converged(self, state):
    return state.J_vc < state.ep_vc  # custom: virtual control alone
```

The default — every metric below its threshold — is algorithm-agnostic, since
`AlgorithmState` carries `J_tr` / `J_vb` / `J_vc` and the `ep_*` tolerances
generically.

!!! note "Honored on every solve path"
    The override is honored on **all three** solve paths. `step()` routes
    through it on the Python `solve()` path, and the `lax.while_loop` harness
    (`openscvx/algorithms/loop.py`) takes it as the loop predicate for
    `solve_jax` / `solve_batched`, where it is vmapped per batch element. It
    must therefore be JAX-traceable. An override implies a subclass, hence a
    distinct class name that the export cache key already folds in — so a
    changed predicate invalidates stale batched artifacts on its own.

By contrast, `t_max` (the wall-clock time limit) is honored **only** on the
Python `solve()` path; the JAX loop terminates on `k_max` and `converged`
alone, since there is no wall-clock probe inside a trace.

### `get_columns` and `citation`

`get_columns(verbosity)` returns the iteration-table columns to print —
implementations typically concatenate the algorithm's own columns with the
autotuner's `COLUMNS` and filter by each column's `min_verbosity`.
`citation()` returns a list of BibTeX strings for the algorithm.

## Using a custom class

Pass either to the `Problem` constructor — the algorithm directly, an
autotuner through the algorithm:

```python
import openscvx as ox

problem = ox.Problem(
    ...,
    algorithm=MyAlgorithm(autotuner=MyAutotuner(ramp=1.5)),
)
```

The default `PenalizedTrustRegion` lives at
`openscvx/algorithms/scvx/penalized_trust_region.py` and is the reference
implementation to read alongside this guide.
