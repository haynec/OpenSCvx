"""Adaptive-step ODE integration through the Diffrax backend.

Wraps :mod:`diffrax` to integrate continuous-time dynamics for the two roles the
pipeline needs, both over the normalized pseudo-time ``tau`` rather than physical
time:

- :func:`solve_ivp_diffrax` saves the solution on a fixed grid of substeps and is
  the workhorse behind discretization, where the augmented dynamics are
  propagated between trajectory nodes.
- :func:`solve_ivp_diffrax_prop` retains the dense interpolant and evaluates it at
  arbitrary ``save_time`` points (with masking); post-optimization propagation
  uses it to reconstruct a high-resolution trajectory from the optimized nodes.

The solver, tolerances, and step count are configurable; ``SOLVER_MAP`` names the
available explicit and implicit Diffrax schemes (Dopri5/8, Tsit5, KenCarp3/4/5,
...). At import time the module registers Diffrax's ``DenseInterpolation`` as a
JAX pytree node so dense solution objects survive ``jit``/``vmap`` and can be
returned from transformed functions.
"""

import os
from typing import Any, Callable

import diffrax as dfx
import jax
import jax.numpy as jnp
from diffrax._global_interpolation import DenseInterpolation
from jax import tree_util

os.environ["EQX_ON_ERROR"] = "nan"

# Safely check if DenseInterpolation is already registered
try:
    dummy_instance = DenseInterpolation(
        ts=jnp.array([]),
        ts_size=0,
        infos=None,
        interpolation_cls=None,
        direction=None,
        t0_if_trivial=0.0,
        y0_if_trivial=jnp.array([]),
    )
    tree_util.tree_flatten(dummy_instance)
except ValueError:

    def dense_interpolation_flatten(obj):
        return (obj._data,), None

    def dense_interpolation_unflatten(aux_data, children):
        return DenseInterpolation(*children)

    tree_util.register_pytree_node(
        DenseInterpolation,
        dense_interpolation_flatten,
        dense_interpolation_unflatten,
    )

SOLVER_MAP = {
    "Tsit5": dfx.Tsit5,
    "Euler": dfx.Euler,
    "Heun": dfx.Heun,
    "Midpoint": dfx.Midpoint,
    "Ralston": dfx.Ralston,
    "Dopri5": dfx.Dopri5,
    "Dopri8": dfx.Dopri8,
    "Bosh3": dfx.Bosh3,
    "ReversibleHeun": dfx.ReversibleHeun,
    "ImplicitEuler": dfx.ImplicitEuler,
    "KenCarp3": dfx.KenCarp3,
    "KenCarp4": dfx.KenCarp4,
    "KenCarp5": dfx.KenCarp5,
}

DEFAULT_DIFFRAX_RTOL = 1e-6
DEFAULT_DIFFRAX_ATOL = 1e-3


def solve_ivp_diffrax(
    f: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    tau_final: float,
    y_0: jnp.ndarray,
    args: tuple,
    tau_0: float = 0.0,
    num_substeps: int = 50,
    solver_name: str = "Dopri8",
    rtol: float = DEFAULT_DIFFRAX_RTOL,
    atol: float = DEFAULT_DIFFRAX_ATOL,
    extra_kwargs: dict = None,
) -> jnp.ndarray:
    substeps = jnp.linspace(tau_0, tau_final, num_substeps)

    solver_class = SOLVER_MAP.get(solver_name)
    if solver_class is None:
        raise ValueError(f"Unknown solver: {solver_name}")
    solver = solver_class()

    term = dfx.ODETerm(lambda t, y, args: f(t, y, *args))
    user_kwargs = dict(extra_kwargs or {})
    dt0 = user_kwargs.pop("dt0", (tau_final - tau_0) / (len(substeps) - 1))
    solve_kwargs = {
        "stepsize_controller": dfx.PIDController(rtol=rtol, atol=atol),
        "saveat": dfx.SaveAt(ts=substeps),
        "progress_meter": dfx.NoProgressMeter(),
    }
    solve_kwargs.update(user_kwargs)
    if isinstance(solve_kwargs["stepsize_controller"], dfx.StepTo):
        dt0 = None

    solution = dfx.diffeqsolve(
        term,
        solver=solver,
        t0=tau_0,
        t1=tau_final,
        dt0=dt0,
        y0=y_0,
        args=args,
        **solve_kwargs,
    )

    return solution.ys


def solve_ivp_diffrax_prop(
    f: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    tau_final: float,
    y_0: jnp.ndarray,
    args: tuple,
    tau_0: float = 0.0,
    num_substeps: int = 50,
    solver_name: str = "Dopri8",
    rtol: float = DEFAULT_DIFFRAX_RTOL,
    atol: float = DEFAULT_DIFFRAX_ATOL,
    extra_kwargs: dict = None,
    save_time: jnp.ndarray = None,
    mask: jnp.ndarray = None,
) -> jnp.ndarray:
    if save_time is None:
        raise ValueError("save_time must be provided for export compatibility.")
    if mask is None:
        mask = jnp.ones_like(save_time, dtype=bool)

    solver_class = SOLVER_MAP.get(solver_name)
    if solver_class is None:
        raise ValueError(f"Unknown solver: {solver_name}")
    solver = solver_class()

    term = dfx.ODETerm(lambda t, y, args: f(t, y, *args))
    user_kwargs = dict(extra_kwargs or {})
    dt0 = user_kwargs.pop("dt0", (tau_final - tau_0) / 1)
    solve_kwargs = {
        "stepsize_controller": dfx.PIDController(rtol=rtol, atol=atol),
        "saveat": dfx.SaveAt(dense=True, t1=True),
    }
    solve_kwargs.update(user_kwargs)
    if isinstance(solve_kwargs["stepsize_controller"], dfx.StepTo):
        dt0 = None

    solution = dfx.diffeqsolve(
        term,
        solver=solver,
        t0=tau_0,
        t1=tau_final,
        dt0=dt0,
        y0=y_0,
        args=args,
        **solve_kwargs,
    )

    all_evals = jax.vmap(solution.evaluate)(save_time)
    terminal_state = jnp.asarray(solution.ys)[-1]
    at_terminal = jnp.isclose(save_time, tau_final)
    all_evals = jnp.where(at_terminal[:, None], terminal_state, all_evals)
    masked_array = jnp.where(mask[:, None], all_evals, jnp.zeros_like(all_evals))
    return masked_array
