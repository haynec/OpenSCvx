from typing import Any, Callable

import jax
import jax.numpy as jnp


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
