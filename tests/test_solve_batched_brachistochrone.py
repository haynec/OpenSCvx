"""``Problem.solve_batched`` matches per-element ``solve_jax`` solves.

``solve_batched`` owns the batch axis internally (``jax.vmap`` applied inside
the method) where :func:`jax.vmap` over :meth:`Problem.solve_jax` leaves it to
the caller. With no export wired up (Phase 1) the two are just different
spellings of the same batched solve, so over a stack of boundary pins or
parameter values each batch element must agree with the corresponding
``solve_jax`` result. CVXPy runs the ``B`` solves sequentially (host CVXPy
isn't thread-safe); QPAX runs them in parallel under vmap. Parallels
``tests/test_solve_jax_vmap_brachistochrone.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.solvers._iteration_callback_helpers import build_brachistochrone

# === Boundary-pin batching ===


@pytest.mark.parametrize("backend", ["cvxpy", "qpax"])
def test_solve_batched_matches_solve_jax_over_x_initial(backend):
    if backend == "qpax":
        pytest.importorskip("qpax")

    prob = build_brachistochrone(backend, n=8, k_max=20)
    prob.initialize()

    # Stack four initial pins by varying the starting x-coordinate (component
    # 0 of the unified state vector).
    default_pin = prob.state.x_init_pin
    shifts = jnp.array([0.0, 0.3, -0.3, 0.6])
    x_initial_stack = jnp.stack([default_pin.at[0].set(default_pin[0] + s) for s in shifts])

    # Per-element reference.
    bare_xs = []
    bare_us = []
    for i in range(x_initial_stack.shape[0]):
        res = prob.solve_jax(x_initial=x_initial_stack[i])
        bare_xs.append(np.asarray(res.x))
        bare_us.append(np.asarray(res.u))
    bare_xs = np.stack(bare_xs)
    bare_us = np.stack(bare_us)

    # Internal-vmap batched solve: x_initial is (B, n_x) -> batched, the
    # terminal pin is omitted -> shared default, broadcast automatically.
    batched = prob.solve_batched(x_initial=x_initial_stack)

    assert batched.x.shape == bare_xs.shape == (x_initial_stack.shape[0], 8, bare_xs.shape[2])
    np.testing.assert_allclose(np.asarray(batched.x), bare_xs, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(batched.u), bare_us, atol=1e-5, rtol=1e-5)
    assert batched.converged.shape == (x_initial_stack.shape[0],)

    jax.clear_caches()


# === Mixed shared/batched parameters ===


def _build_brachistochrone_with_params(backend: str, n: int = 8, k_max: int = 20):
    """Brachistochrone variant whose dynamics read two ``ox.Parameter``s.

    ``gravity`` is the sweep target (batched in the test); ``gain`` scales the
    position kinematics and stays shared, exercising the mixed per-key vmap
    axes that ``solve_batched`` resolves from declared parameter shapes.
    """
    import openscvx as ox
    from openscvx import Problem

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

    gravity = ox.Parameter("gravity", shape=(1,), value=np.array([9.81]))
    gain = ox.Parameter("gain", shape=(1,), value=np.array([1.0]))

    dynamics = {
        "position": ox.Concat(
            gain[0] * velocity[0] * ox.Sin(theta[0]),
            -gain[0] * velocity[0] * ox.Cos(theta[0]),
        ),
        "velocity": gravity[0] * ox.Cos(theta[0]),
    }

    constraint_exprs = []
    for state in [position, velocity]:
        constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    time = ox.Time(initial=0.0, final=("minimize", 2.0), min=0.0, max=2.0, uniform_time_grid=True)

    prob = Problem(
        dynamics=dynamics,
        states=[position, velocity],
        controls=[theta],
        time=time,
        constraints=constraint_exprs,
        N=n,
        float_dtype="float64",
        algorithm={
            "autotuner": "ConstantProximalWeight",
            "lam_prox": 1e0,
            "lam_cost": 6e-1,
            "k_max": k_max,
        },
        solver={"backend": backend},
    )
    prob.settings.dev.printing = False
    return prob


def test_solve_batched_mixed_shared_and_batched_parameters():
    pytest.importorskip("qpax")

    prob = _build_brachistochrone_with_params("qpax", n=8, k_max=20)
    prob.initialize()

    gravity_batch = jnp.array([[9.0], [9.81], [10.5]])  # (B, 1) vs declared (1,) -> batched
    gain_shared = jnp.array([0.9])  # declared shape (1,) -> shared

    batched = prob.solve_batched(
        parameters={"gravity": gravity_batch, "gain": gain_shared}
    )
    assert batched.x.shape[0] == gravity_batch.shape[0]

    for i in range(gravity_batch.shape[0]):
        ref = prob.solve_jax(
            parameters=dict(prob.parameters, gravity=gravity_batch[i], gain=gain_shared)
        )
        np.testing.assert_allclose(np.asarray(batched.x[i]), np.asarray(ref.x), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(batched.u[i]), np.asarray(ref.u), atol=1e-5, rtol=1e-5)

    jax.clear_caches()


# === Teaching errors ===


def test_solve_batched_unknown_parameter_key_raises():
    prob = _build_brachistochrone_with_params("cvxpy", n=4, k_max=1)
    prob.initialize()
    with pytest.raises(ValueError, match=r"unknown parameter.*'gravty'.*declared parameters.*'gravity'"):
        prob.solve_batched(parameters={"gravty": jnp.zeros((2, 1))})


def test_solve_batched_before_initialize_raises():
    prob = build_brachistochrone("qpax" if _has_qpax() else "cvxpy", n=8, k_max=1)
    N = prob.settings.sim.n
    n_x = prob.settings.sim.n_states
    x_stack = jnp.zeros((2, N, n_x))
    with pytest.raises(ValueError, match="initialize"):
        prob.solve_batched(x_guess=x_stack)


def _has_qpax() -> bool:
    try:
        import qpax  # noqa: F401

        return True
    except ImportError:
        return False
