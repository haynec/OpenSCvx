"""Batched brachistochrone — two ways to solve four problems at once.

Solves four brachistochrone problems in parallel — each with a different
starting x-coordinate — and contrasts the two batching entry points:

* ``jax.vmap(problem.solve_jax)`` — the caller owns the batch axis. Maximally
  idiomatic, composes with ``grad`` / ``scan``, but in-process JIT only: every
  fresh process re-traces and re-compiles the whole SCP loop.
* ``problem.solve_batched(x_initial=...)`` — the library owns the batch
  axis, so the whole vmapped loop is a single function that ``jax.export`` can
  serialize. Under ``save_compiled=True`` it is written to the solver cache on
  the first run and deserialized on later runs, skipping that compile. Reach
  for it when cross-process cold-start dominates (CI sweeps, short-lived
  workers); reach for ``jax.vmap(solve_jax)`` when you stay inside one program.

Run with ``python examples/abstract/brachistochrone_batched.py``. Re-running is
itself the cross-process demo: the second invocation deserializes the cached
``solve_batched`` artifact instead of compiling it.

Notes:

* Under the QPAX backend used here, vmap'd subproblem solves run in
  parallel (no host callback). The CVXPy backend is single-threaded under
  vmap because :func:`jax.pure_callback` uses ``vmap_method="sequential"``
  for thread safety — see :meth:`solve_jax`'s docstring. That same host
  callback is why ``solve_batched(save_compiled=True)`` refuses CVXPy: it
  cannot be exported.
* The Moreau warm-start is bypassed on the JAX-pure path (its carry is
  host-side state that ``lax.while_loop`` doesn't thread). Use QPAX or
  CVXPy when you want batched solves.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem


def build_problem(n: int = 8):
    g = 9.81

    position = ox.State("position", shape=(2,))
    position.max = np.array([10.0, 10.0])
    position.min = np.array([0.0, 0.0])
    position.initial = np.array([0.0, 10.0])
    position.final = [10.0, 5.0]

    velocity = ox.State("velocity", shape=(1,))
    velocity.max = np.array([10.0])
    velocity.min = np.array([0.0])
    velocity.initial = np.array([0.0])
    velocity.final = [("free", 10.0)]

    theta = ox.Control("theta", shape=(1,))
    theta.max = np.array([100.5 * jnp.pi / 180])
    theta.min = np.array([0.0])
    theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    states = [position, velocity]
    controls = [theta]

    dynamics = {
        "position": ox.Concat(
            velocity[0] * ox.Sin(theta[0]),
            -velocity[0] * ox.Cos(theta[0]),
        ),
        "velocity": g * ox.Cos(theta[0]),
    }

    constraint_exprs = []
    for state in states:
        constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    time = ox.Time(
        initial=0.0,
        final=("minimize", 2.0),
        min=0.0,
        max=2.0,
        uniform_time_grid=True,
    )

    return Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraint_exprs,
        N=n,
        float_dtype="float64",
        algorithm={
            "autotuner": "ConstantProximalWeight",
            "lam_prox": 1e0,
            "lam_cost": 6e-1,
            "ep_tr": 1e-5,
            "ep_vb": 1e-5,
            "ep_vc": 1e-9,
            "k_max": 30,
        },
        solver={"backend": "qpax"},
    )


if __name__ == "__main__":
    problem = build_problem()
    problem.initialize()

    # The default boundary-condition pin (the full unified state vector with
    # ``jnp.nan`` at non-Fix entries — see
    # :meth:`AlgorithmState.from_settings`).
    default_pin = problem.state.x_init_pin

    # Stack four ICs by varying the starting x-coordinate (component 0 of
    # the unified state vector is the x-position).
    shifts = jnp.array([0.0, 0.3, 0.6, 0.9])
    x_initial_stack = jnp.stack([default_pin.at[0].set(default_pin[0] + s) for s in shifts])

    # --- caller owns the batch axis: jax.vmap(solve_jax) ---------------------
    # Bare ``jax.vmap`` — composes like any JAX transform, in-process only.
    batched_solve = jax.vmap(problem.solve_jax, in_axes=(0, None, None))
    results = batched_solve(x_initial_stack, None, None)

    # ``results`` is a batched ``OptimizationResults`` pytree — every child
    # (``X``, ``U``, ``t_final``, ``converged``, ...) has a leading batch axis.
    print(f"jax.vmap(solve_jax) over {x_initial_stack.shape[0]} initial conditions:")
    print(f"  result.x.shape:   {results.x.shape}")
    print(f"  result.u.shape:   {results.u.shape}")
    print(f"  result.t_final:   {np.asarray(results.t_final).reshape(-1)}")
    print(f"  result.converged: {np.asarray(results.converged)}")

    # --- library owns the batch axis: solve_batched, disk-cached -------------
    # The same batched solve, but ``solve_batched`` owns the vmap so the whole
    # loop is one exportable artifact. Under ``save_compiled=True`` it lands in
    # the solver cache on the first run; re-run this script (a fresh process) to
    # see it deserialized instead of recompiled. One rule: ``x_initial`` is
    # ``(B, n_x)`` — one extra leading axis — so it is batched; the terminal
    # pin is omitted, so every element shares the default.
    export_problem = build_problem()
    export_problem.settings.sim.save_compiled = True
    export_problem.initialize()

    batched = export_problem.solve_batched(x_initial=x_initial_stack)

    print(f"\nsolve_batched (save_compiled=True) over {x_initial_stack.shape[0]} ICs:")
    print(f"  result.x.shape:   {batched.x.shape}")
    print(f"  result.t_final:   {np.asarray(batched.t_final).reshape(-1)}")
    print(f"  result.converged: {np.asarray(batched.converged)}")

    # Algorithm knobs batch by the same rule, through the `algorithm` dict —
    # the same names as the Problem constructor's algorithm config. ep_tr is
    # a scalar field, so a (B,) vector sweeps the convergence tolerance per
    # element — same artifact, no recompile, since the knobs are runtime
    # inputs on the state pytree. The batch size is never passed: it is read
    # off the array's leading axis (4 tolerances -> B = 4).
    sweep = export_problem.solve_batched(
        algorithm={"ep_tr": jnp.logspace(-4, -1, 4), "ep_vc": 1e-7}
    )
    print(f"  ep_tr sweep t_final:   {np.asarray(sweep.t_final).reshape(-1)}")
    print(f"  ep_tr sweep converged: {np.asarray(sweep.converged)}")
