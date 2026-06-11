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

from openscvx.algorithms import (
    AdaptiveStateCode,
    AugmentedLagrangian,
    AutotuningBase,
    HyperParams,
)
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

    batched = prob.solve_batched(parameters={"gravity": gravity_batch, "gain": gain_shared})
    assert batched.x.shape[0] == gravity_batch.shape[0]

    for i in range(gravity_batch.shape[0]):
        ref = prob.solve_jax(
            parameters=dict(prob.parameters, gravity=gravity_batch[i], gain=gain_shared)
        )
        np.testing.assert_allclose(
            np.asarray(batched.x[i]), np.asarray(ref.x), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(batched.u[i]), np.asarray(ref.u), atol=1e-5, rtol=1e-5
        )

    jax.clear_caches()


def test_solve_batched_in_axes_prefix_matches_inference():
    pytest.importorskip("qpax")

    prob = _build_brachistochrone_with_params("qpax", n=8, k_max=8)
    prob.initialize()

    # jax.vmap prefix semantics: in_axes={"parameters": 0} batches every
    # passed parameter — same resolution the rank rule infers for these
    # shapes, so the results must agree (and the batched closure is reused).
    params = {
        "gravity": jnp.array([[9.0], [10.5]]),  # (B, 1) vs declared (1,)
        "gain": jnp.array([[0.9], [1.1]]),
    }
    inferred = prob.solve_batched(parameters=params)
    prefixed = prob.solve_batched(parameters=params, in_axes={"parameters": 0})

    np.testing.assert_allclose(np.asarray(prefixed.x), np.asarray(inferred.x), atol=1e-12, rtol=0)

    jax.clear_caches()


# === Hyperparameter sweeps ===


def test_solve_batched_max_iters_sweep_matches_solve_jax():
    pytest.importorskip("qpax")

    prob = build_brachistochrone("qpax", n=8, k_max=20)
    prob.initialize()

    # Per-element iteration budgets, all below the convergence point so the
    # cap binds and each element stops at a different iterate. (B,) vs
    # declared () -> batched; everything else is the shared default.
    budgets = jnp.array([2, 4, 8])
    batched = prob.solve_batched(max_iters=budgets)
    assert batched.x.shape[0] == budgets.shape[0]

    for i, budget in enumerate(np.asarray(budgets)):
        ref = prob.solve_jax(max_iters=int(budget))
        np.testing.assert_allclose(
            np.asarray(batched.x[i]), np.asarray(ref.x), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(batched.u[i]), np.asarray(ref.u), atol=1e-5, rtol=1e-5
        )

    # The budgets actually bind: different caps end at different iterates.
    assert not np.allclose(np.asarray(batched.x[0]), np.asarray(batched.x[2]), atol=1e-8)

    jax.clear_caches()


def test_solve_batched_ep_tr_sweep_matches_solve_jax():
    pytest.importorskip("qpax")

    prob = build_brachistochrone("qpax", n=8, k_max=20)
    prob.initialize()

    # Loose-to-tight trust-region tolerances: the loose element converges
    # iterations earlier, so the per-element final iterates differ. ep_tr is
    # a scalar state field, so (B,) vs () -> batched through `algorithm`.
    tolerances = jnp.array([1e-1, 1e-5])
    batched = prob.solve_batched(algorithm={"ep_tr": tolerances})

    for i, tol in enumerate(np.asarray(tolerances)):
        ref = prob.solve_jax(algorithm={"ep_tr": float(tol)})
        np.testing.assert_allclose(
            np.asarray(batched.x[i]), np.asarray(ref.x), atol=1e-5, rtol=1e-5
        )

    assert not np.allclose(np.asarray(batched.x[0]), np.asarray(batched.x[1]), atol=1e-8)

    jax.clear_caches()


def test_solve_batched_lam_prox_fill_sweep_matches_solve_jax():
    pytest.importorskip("qpax")

    prob = build_brachistochrone("qpax", n=8, k_max=20)
    prob.initialize()

    # lam_prox is an (N, n_x + n_u) state field; a (B,) vector is the
    # batched *fill* form — one scalar per element, each broadcast to the
    # field shape. Different trust-region weights walk different iterate
    # paths, so the elements genuinely diverge.
    weights = jnp.array([0.5, 1.0, 4.0])
    batched = prob.solve_batched(algorithm={"lam_prox": weights})
    assert batched.x.shape[0] == weights.shape[0]

    for i, w in enumerate(np.asarray(weights)):
        # solve_jax takes the scalar (shared-fill) form of the same override.
        ref = prob.solve_jax(algorithm={"lam_prox": float(w)})
        np.testing.assert_allclose(
            np.asarray(batched.x[i]), np.asarray(ref.x), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(batched.u[i]), np.asarray(ref.u), atol=1e-5, rtol=1e-5
        )

    assert not np.allclose(np.asarray(batched.x[0]), np.asarray(batched.x[2]), atol=1e-8)

    jax.clear_caches()


def test_solve_batched_algorithm_k_max_matches_max_iters_kwarg():
    pytest.importorskip("qpax")

    prob = build_brachistochrone("qpax", n=8, k_max=20)
    prob.initialize()

    # k_max has a dedicated kwarg (max_iters) whose default stack
    # solve_batched always materializes; the override spelling must beat
    # that default, not silently lose to it.
    budgets = jnp.array([2, 8])
    via_kwarg = prob.solve_batched(max_iters=budgets)
    via_override = prob.solve_batched(algorithm={"k_max": budgets})

    np.testing.assert_allclose(
        np.asarray(via_override.x), np.asarray(via_kwarg.x), atol=1e-9, rtol=0
    )
    # The budgets actually bind — this is not two default-k_max solves agreeing.
    assert not np.allclose(np.asarray(via_kwarg.x[0]), np.asarray(via_kwarg.x[1]), atol=1e-8)

    jax.clear_caches()


# === Autotuner-declared hyperparameters (state.hyper) ===


class _ProxRampHyper(HyperParams):
    """The toy autotuner's one declared knob."""

    prox_scale: float = 1.0


class _ProxRampAutotuner(AutotuningBase):
    """Toy user autotuner: one declared knob scaling ``lam_prox`` per iteration.

    Defined entirely inside the test — the point is that declaring
    ``prox_scale`` on a :class:`HyperParams` subclass and reading it from
    ``state.hyper`` is the *whole* integration: overrides and batched sweeps
    work with zero library edits.
    """

    JIT_UPDATE_WEIGHTS = False

    def __init__(self, prox_scale: float = 1.0):
        self.hyper = _ProxRampHyper(prox_scale=prox_scale)

    def update_weights(self, state, candidate, nodal_constraints, settings, params):
        return state.replace(
            x=candidate.x,
            u=candidate.u,
            x_prop=candidate.x_prop,
            x_prop_plus=candidate.x_prop_plus,
            lam_prox=state.lam_prox * state.hyper.prox_scale,
            adaptive_state_code=jnp.asarray(
                int(AdaptiveStateCode.ACCEPT_CONSTANT), dtype=jnp.int32
            ),
        )


def test_user_autotuner_knob_sweeps_through_solve_batched():
    pytest.importorskip("qpax")

    prob = build_brachistochrone("qpax", n=8, k_max=10, autotuner=_ProxRampAutotuner())
    prob.initialize()

    # The declared knob is a valid override name purely by declaration; a
    # (B,) vector sweeps it per element and each element matches the
    # corresponding single solve.
    scales = jnp.array([0.8, 1.0, 1.4])
    batched = prob.solve_batched(algorithm={"prox_scale": scales})
    assert batched.x.shape[0] == scales.shape[0]

    for i, s in enumerate(np.asarray(scales)):
        ref = prob.solve_jax(algorithm={"prox_scale": float(s)})
        np.testing.assert_allclose(
            np.asarray(batched.x[i]), np.asarray(ref.x), atol=1e-5, rtol=1e-5
        )

    # The knob actually steers the solve: different ramps, different iterates.
    assert not np.allclose(np.asarray(batched.x[0]), np.asarray(batched.x[2]), atol=1e-8)

    jax.clear_caches()


def test_augmented_lagrangian_declared_knobs_are_overridable():
    pytest.importorskip("qpax")

    # AugmentedLagrangian declares rho_init / rho_max / lam_cost_drop on its
    # HyperParams container, so the override channel accepts them by name. rho_init
    # and rho_max currently have no read sites in update_weights (vestigial
    # knobs), so the sweep exercises the plumbing — accepted name, batched
    # state, per-element agreement — with results identical to the baseline.
    prob = build_brachistochrone("qpax", n=8, k_max=6, autotuner=AugmentedLagrangian())
    prob.initialize()

    base = prob.solve_jax()
    single = prob.solve_jax(algorithm={"rho_init": 5.0, "rho_max": 1e3})
    np.testing.assert_allclose(np.asarray(single.x), np.asarray(base.x), atol=1e-12)

    batched = prob.solve_batched(algorithm={"rho_init": jnp.array([1.0, 5.0])})
    assert batched.x.shape[0] == 2
    for i in range(2):
        np.testing.assert_allclose(
            np.asarray(batched.x[i]), np.asarray(base.x), atol=1e-5, rtol=1e-5
        )

    jax.clear_caches()


# === Teaching errors ===


def test_solve_batched_unknown_parameter_key_raises():
    prob = _build_brachistochrone_with_params("cvxpy", n=4, k_max=1)
    prob.initialize()
    with pytest.raises(
        ValueError, match=r"unknown parameter.*'gravty'.*declared parameters.*'gravity'"
    ):
        prob.solve_batched(parameters={"gravty": jnp.zeros((2, 1))})


def test_algorithm_unknown_key_lists_valid_names():
    prob = build_brachistochrone("cvxpy", n=4, k_max=1)
    prob.initialize()
    with pytest.raises(ValueError, match=r"unknown algorithm key.*'ep_trr'.*'ep_tr'"):
        prob.solve_batched(algorithm={"ep_trr": jnp.zeros(2)})
    with pytest.raises(ValueError, match=r"solve_jax: unknown algorithm key.*'ep_trr'"):
        prob.solve_jax(algorithm={"ep_trr": 1e-4})


def test_algorithm_construction_key_teaches_rebuild():
    prob = build_brachistochrone("cvxpy", n=4, k_max=1)
    prob.initialize()
    # `autotuner` selects program structure and `t_max` is a wall-clock budget
    # outside the traced loop — both are construction-time settings, so they
    # get the rebuild-the-Problem teaching error, not the generic unknown-name
    # one.
    with pytest.raises(
        ValueError, match=r"algorithm\['autotuner'\].*construction-time.*rebuild the Problem"
    ):
        prob.solve_jax(algorithm={"autotuner": "RampProximalWeight"})
    with pytest.raises(
        ValueError, match=r"algorithm\['t_max'\].*construction-time.*rebuild the Problem"
    ):
        prob.solve_batched(algorithm={"t_max": jnp.ones(2)})


def test_algorithm_kwarg_collision_names_the_kwarg():
    prob = build_brachistochrone("cvxpy", n=4, k_max=1)
    prob.initialize()
    with pytest.raises(ValueError, match=r"algorithm\['k_max'\].*max_iters kwarg"):
        prob.solve_batched(max_iters=jnp.array([2, 3]), algorithm={"k_max": 5})
    with pytest.raises(ValueError, match=r"algorithm\['x'\].*x_guess kwarg"):
        prob.solve_jax(x_guess=prob.state.x, algorithm={"x": prob.state.x})


def test_solve_jax_algorithm_shape_mismatch_points_at_solve_batched():
    prob = build_brachistochrone("cvxpy", n=4, k_max=1)
    prob.initialize()
    with pytest.raises(ValueError, match=r"algorithm\['ep_tr'\] has shape \(2,\).*solve_batched"):
        prob.solve_jax(algorithm={"ep_tr": jnp.zeros(2)})


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
