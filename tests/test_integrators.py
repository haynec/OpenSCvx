# test_integrators.py

import diffrax as dfx
import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.integrators import (
    solve_ivp_diffrax,
    solve_ivp_diffrax_prop,
    solve_ivp_rk45,
)


def decay(t, y):
    return -y


@pytest.mark.parametrize("num_steps", [11, 21])
def test_solve_ivp_rk45_decay(num_steps):
    """RK45 fixed-step should approximate exp(-t)."""
    t0, t1 = 0.0, 1.0
    times = jnp.linspace(t0, t1, num_steps)
    y0 = jnp.array([1.0])
    sol = solve_ivp_rk45(decay, t1, y0, args=(), is_not_compiled=False, num_substeps=num_steps)
    sol_np = np.array(sol[:, 0])
    expected = np.exp(-np.array(times))
    # allow ~1% relative error
    np.testing.assert_allclose(sol_np, expected, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("solver_name", ["Tsit5", "Dopri5", "Dopri8", "Heun"])
@pytest.mark.parametrize("num_steps", [11, 21])
def test_solve_ivp_diffrax_decay(solver_name, num_steps):
    """Diffrax adaptive solver should approximate exp(-t)."""
    t0, t1 = 0.0, 1.0
    times = jnp.linspace(t0, t1, num_steps)
    y0 = jnp.array([1.0])
    sol = solve_ivp_diffrax(
        decay,
        t1,
        y0,
        args=(),
        solver_name=solver_name,
        rtol=1e-3,
        atol=1e-6,
        extra_kwargs=None,
        num_substeps=num_steps,
    )
    sol_np = np.array(sol[:, 0])
    expected = np.exp(-np.array(times))
    # allow slightly larger error for cheap solvers
    tol = 2e-2 if solver_name == "Euler" else 5e-3
    np.testing.assert_allclose(sol_np, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("solver_name", ["Tsit5", "Dopri5", "Dopri8"])
def test_solve_ivp_diffrax_prop_decay(solver_name):
    # Setup integration bounds
    tau0, tau1 = 0.0, 1.0
    y0 = jnp.array([1.0])
    args = ()  # decay doesn't need extra args

    # Determine dt and MAX_TAU_LEN
    dt = 0.1
    dtau = tau1 - tau0
    MAX_TAU_LEN = int(jnp.ceil(dtau / dt)) + 1  # +1 to ensure coverage

    # Define save_time and mask
    save_time_raw = jnp.linspace(tau0, tau1, MAX_TAU_LEN)
    mask = jnp.ones(MAX_TAU_LEN, dtype=bool)

    # Run solver
    sol = solve_ivp_diffrax_prop(
        f=decay,
        tau_final=tau1,
        y_0=y0,
        args=args,
        tau_0=tau0,
        solver_name=solver_name,
        rtol=1e-6,
        atol=1e-9,
        extra_kwargs={},
        save_time=save_time_raw,
        mask=mask,
    )

    # Check the solution
    ys = np.array(sol[:, 0])
    expected = np.exp(-np.linspace(tau0, tau1, MAX_TAU_LEN))
    np.testing.assert_allclose(ys, expected, rtol=1e-3, atol=1e-6)


def test_solve_ivp_diffrax_with_stepto_controller():
    t0, t1 = 0.0, 1.0
    num_steps = 11
    times = jnp.linspace(t0, t1, num_steps)
    y0 = jnp.array([1.0])
    sol = solve_ivp_diffrax(
        decay,
        t1,
        y0,
        args=(),
        solver_name="Tsit5",
        rtol=1e-3,
        atol=1e-6,
        num_substeps=num_steps,
        extra_kwargs={"stepsize_controller": dfx.StepTo(ts=times)},
    )
    sol_np = np.array(sol[:, 0])
    expected = np.exp(-np.array(times))
    np.testing.assert_allclose(sol_np, expected, rtol=5e-3, atol=5e-3)
