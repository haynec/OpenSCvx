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
    # Attempt to flatten a dummy DenseInterpolation instance
    # Provide dummy arguments to create a valid instance
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
    # Register DenseInterpolation as a PyTree node if not already registered
    def dense_interpolation_flatten(obj):
        # Flatten the internal data of DenseInterpolation
        return (obj._data,), None

    def dense_interpolation_unflatten(aux_data, children):
        # Reconstruct DenseInterpolation from its flattened data
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


# fmt: off
def rk45_step(
    f: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    t: jnp.ndarray,
    y: jnp.ndarray,
    h: float,
    *args: Any
) -> jnp.ndarray:
    """
    Perform a single RK45 (Runge-Kutta-Fehlberg) integration step.

    This implements the classic Dorman-Prince coefficients for an
    explicit 4(5) method, returning the fourth-order estimate.

    Args:
        f (Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray]):
            ODE right-hand side; signature f(t, y, *args) -> dy/dt.
        t (jnp.ndarray): Current time.
        y (jnp.ndarray): Current state vector.
        h (float): Step size.
        *args: Additional arguments passed to `f`.

    Returns:
        jnp.ndarray: Next state estimate at t + h.
    """
    k1 = f(t, y, *args)
    k2 = f(t + h/4, y + h*k1/4, *args)
    k3 = f(t + 3*h/8, y + 3*h*k1/32 + 9*h*k2/32, *args)
    k4 = f(t + 12*h/13, y + 1932*h*k1/2197 - 7200*h*k2/2197 + 7296*h*k3/2197, *args)
    k5 = f(t + h, y + 439*h*k1/216 - 8*h*k2 + 3680*h*k3/513 - 845*h*k4/4104, *args)
    y_next = y + h * (25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
    return y_next
# fmt: on


def solve_ivp_rk45(
    f: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    tau_final: float,
    y_0: jnp.ndarray,
    args: tuple,
    tau_0: float = 0.0,
    num_substeps: int = 50,
    is_not_compiled: bool = False,
) -> jnp.ndarray:
    """
    Solve an initial-value ODE problem using fixed-step RK45 integration.

    Args:
        f (Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray]):
            ODE right-hand side; signature f(t, y, *args) -> dy/dt.
        tau_final (float): Final integration time.
        y_0 (jnp.ndarray): Initial state at tau_0.
        args (tuple): Extra arguments to pass to `f`.
        tau_0 (float, optional): Initial time. Defaults to 0.0.
        num_substeps (int, optional): Number of output time points. Defaults to 50.
        is_not_compiled (bool, optional): If True, use Python loop instead of
            JAX `lax.fori_loop`. Defaults to False.

    Returns:
        jnp.ndarray: Array of shape (num_substeps, state_dim) with solution at each time.
    """
    substeps = jnp.linspace(tau_0, tau_final, num_substeps)

    h = (tau_final - tau_0) / (len(substeps) - 1)
    solution = jnp.zeros((len(substeps), len(y_0)))
    solution = solution.at[0].set(y_0)

    if is_not_compiled:
        for i in range(1, len(substeps)):
            t = tau_0 + i * h
            solution = solution.at[i].set(rk45_step(f, t, solution[i - 1], h, *args))
    else:

        def body_fun(i, val):
            t, y, V_result = val
            y_next = rk45_step(f, t, y, h, *args)
            V_result = V_result.at[i].set(y_next)
            return (t + h, y_next, V_result)

        _, _, solution = jax.lax.fori_loop(1, len(substeps), body_fun, (tau_0, y_0, solution))

    return solution


def solve_ivp_diffrax(
    f: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    tau_final: float,
    y_0: jnp.ndarray,
    args: tuple,
    tau_0: float = 0.0,
    num_substeps: int = 50,
    solver_name: str = "Dopri8",
    rtol: float = 1e-3,
    atol: float = 1e-6,
    extra_kwargs: dict = None,
) -> jnp.ndarray:
    """
    Solve an initial-value ODE problem using a Diffrax adaptive solver.

    Args:
        f (Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray]):
            ODE right-hand side; f(t, y, *args).
        tau_final (float): Final integration time.
        y_0 (jnp.ndarray): Initial state at tau_0.
        args (tuple): Extra arguments to pass to `f` in the solver term.
        tau_0 (float, optional): Initial time. Defaults to 0.0.
        num_substeps (int, optional): Number of save points between tau_0 and tau_final.
            Defaults to 50.
        solver_name (str, optional): Key into SOLVER_MAP for the Diffrax solver class.
            Defaults to "Dopri8".
        rtol (float, optional): Relative tolerance for adaptive stepping. Defaults to 1e-3.
        atol (float, optional): Absolute tolerance for adaptive stepping. Defaults to 1e-6.
        extra_kwargs (dict, optional): Additional keyword arguments forwarded to `diffeqsolve`.

    Returns:
        jnp.ndarray: Solution states at the requested save points, shape (num_substeps, state_dim).

    Raises:
        ValueError: If `solver_name` is not in SOLVER_MAP.
    """
    substeps = jnp.linspace(tau_0, tau_final, num_substeps)

    solver_class = SOLVER_MAP.get(solver_name)
    if solver_class is None:
        raise ValueError(f"Unknown solver: {solver_name}")
    solver = solver_class()

    term = dfx.ODETerm(lambda t, y, args: f(t, y, *args))
    stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)
    solution = dfx.diffeqsolve(
        term,
        solver=solver,
        t0=tau_0,
        t1=tau_final,
        dt0=(tau_final - tau_0) / (len(substeps) - 1),
        y0=y_0,
        args=args,
        stepsize_controller=stepsize_controller,
        saveat=dfx.SaveAt(ts=substeps),
        progress_meter=dfx.NoProgressMeter(),
        **(extra_kwargs or {}),
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
    rtol: float = 1e-3,
    atol: float = 1e-6,
    extra_kwargs: dict = None,
    save_time: jnp.ndarray = None,
    mask: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Solve an initial-value ODE problem using a Diffrax adaptive solver.
    This function is specifically designed for use in the context of
    trajectory optimization and handles the nonlinear single-shot propagation
    of state variables in undilated time.

    Args:
        f (Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray]): ODE right-hand side;
            signature f(t, y, *args) -> dy/dt.
        tau_final (float): Final integration time.
        y_0 (jnp.ndarray): Initial state at tau_0.
        args (tuple): Extra arguments to pass to `f` in the solver term.
        tau_0 (float, optional): Initial time. Defaults to 0.0.
        num_substeps (int, optional): Number of save points between tau_0 and tau_final.
            Defaults to 50.
        solver_name (str, optional): Key into SOLVER_MAP for the Diffrax solver class.
            Defaults to "Dopri8".
        rtol (float, optional): Relative tolerance for adaptive stepping. Defaults to 1e-3.
        atol (float, optional): Absolute tolerance for adaptive stepping. Defaults to 1e-6.
        extra_kwargs (dict, optional): Additional keyword arguments forwarded to `diffeqsolve`.
        save_time (jnp.ndarray, optional): Time points at which to evaluate the solution.
            Must be provided for export compatibility.
        mask (jnp.ndarray, optional): Boolean mask for the save_time points.

    Returns:
        jnp.ndarray: Solution states at the requested save points, shape (num_substeps, state_dim).
    Raises:
        ValueError: If `solver_name` is not in SOLVER_MAP or if save_time is not provided.
    """

    if save_time is None:
        raise ValueError("save_time must be provided for export compatibility.")
    if mask is None:
        mask = jnp.ones_like(save_time, dtype=bool)

    solver_class = SOLVER_MAP.get(solver_name)
    if solver_class is None:
        raise ValueError(f"Unknown solver: {solver_name}")
    solver = solver_class()

    term = dfx.ODETerm(lambda t, y, args: f(t, y, *args))
    stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)

    solution = dfx.diffeqsolve(
        term,
        solver=solver,
        t0=tau_0,
        t1=tau_final,
        dt0=(tau_final - tau_0) / 1,
        y0=y_0,
        args=args,
        stepsize_controller=stepsize_controller,
        saveat=dfx.SaveAt(dense=True),
        **(extra_kwargs or {}),
    )

    # Evaluate all save_time points (static size), then mask them
    all_evals = jax.vmap(solution.evaluate)(save_time)  # shape: (MAX_TAU_LEN, n_states)
    masked_array = jnp.where(mask[:, None], all_evals, jnp.zeros_like(all_evals))
    # shape: (variable_len, n_states)

    return masked_array
