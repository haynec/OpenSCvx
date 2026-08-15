"""Tests for symbolic initial guesses.

``Variable.guess`` accepts an array, a symbolic expression in other states and
controls, or a callable of the normalized node coordinate tau. The symbolic forms
are evaluated per node at build time by ``resolve_guess_exprs``, after which every
downstream consumer sees an ordinary array guess.

Sections:
1. Setter dispatch (array / Expr / callable)
2. The internal Tau leaf and its JAX lowering
3. Guess resolution (ordering, parameters, shapes, errors)
4. Builder integration
5. Problem integration (initialize / reset)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.symbolic.expr import Concat, Control, Norm, Parameter, State
from openscvx.symbolic.expr.time import TAU_PARAM, Tau, Time
from openscvx.symbolic.lower import lower_to_jax
from openscvx.symbolic.preprocessing import (
    collect_and_assign_slices,
    fill_default_guesses,
    resolve_guess_exprs,
    validate_guesses,
)

# =============================================================================
# Setter Dispatch
# =============================================================================


def test_array_guess_is_stored_eagerly():
    """An array guess is validated and stored immediately, with nothing pending."""
    pos = State("pos", shape=(2,))
    guess = np.linspace([0.0, 0.0], [1.0, 2.0], 5)
    pos.guess = guess

    np.testing.assert_allclose(pos.guess, guess)
    assert pos._guess_expr is None


def test_expr_guess_is_stored_pending():
    """An expression guess is held symbolically; `.guess` reads back None until built."""
    vel = State("vel", shape=(3,))
    speed = State("speed", shape=(1,))
    speed.guess = Norm(vel)

    assert speed.guess is None
    assert speed._guess_expr is not None
    assert speed._guess_expr.children()[0] is vel


def test_callable_guess_is_applied_once_at_assignment():
    """The callable runs at assignment, not at build time, and receives a Tau leaf."""
    calls = []

    def ramp(tau):
        calls.append(tau)
        return 2.0 * tau

    pos = State("pos", shape=(1,))
    pos.guess = ramp

    assert len(calls) == 1
    assert isinstance(calls[0], Tau)
    assert pos.guess is None
    assert pos._guess_expr is not None


def test_callable_guess_returning_non_expr_raises():
    """A callable computing numerically instead of symbolically gets a teaching error."""
    pos = State("pos", shape=(1,))

    with pytest.raises(ValueError) as excinfo:
        pos.guess = lambda tau: np.linspace(0.0, 1.0, 5).reshape(-1, 1)

    msg = str(excinfo.value)
    assert "guess callable returned" in msg
    assert "lambda tau: p0 + (pf - p0) * tau" in msg


def test_array_guess_clears_pending_expr():
    """Re-assigning an array drops the symbolic guess."""
    vel = State("vel", shape=(3,))
    speed = State("speed", shape=(1,))
    speed.guess = Norm(vel)
    speed.guess = np.zeros((5, 1))

    assert speed._guess_expr is None
    np.testing.assert_allclose(speed.guess, np.zeros((5, 1)))


def test_expr_guess_clears_array():
    """Re-assigning an expression drops the stale array."""
    vel = State("vel", shape=(3,))
    speed = State("speed", shape=(1,))
    speed.guess = np.zeros((5, 1))
    speed.guess = Norm(vel)

    assert speed.guess is None
    assert speed._guess_expr is not None


def test_append_with_pending_guess_expr_raises():
    """An unresolved symbolic guess cannot be concatenated with another variable's."""
    vel = State("vel", shape=(3,))
    speed = State("speed", shape=(1,))
    speed.guess = Norm(vel)

    other = State("other", shape=(1,))
    other.guess = np.zeros((5, 1))

    with pytest.raises(ValueError, match="symbolic initial guess"):
        speed.append(other)

    with pytest.raises(ValueError, match="symbolic initial guess"):
        other.append(speed)


# =============================================================================
# Tau Leaf
# =============================================================================


def test_tau_check_shape_raises():
    """A tau leaf escaping into dynamics or constraints is caught by shape checking."""
    with pytest.raises(ValueError, match="tau is only meaningful inside initial-guess"):
        Tau().check_shape()


def test_tau_repr():
    assert repr(Tau()) == "Tau()"


def test_tau_lowers_to_the_node_grid_value():
    """The lowered tau reads the node grid supplied under the reserved params key."""
    fn = lower_to_jax(Tau())
    params = {TAU_PARAM: jnp.linspace(0.0, 1.0, 5)}

    assert float(fn(None, None, 0, params)) == pytest.approx(0.0)
    assert float(fn(None, None, 2, params)) == pytest.approx(0.5)
    assert float(fn(None, None, 4, params)) == pytest.approx(1.0)


def test_tau_lowering_composes_with_states():
    """Tau lowers into a larger expression alongside ordinary state slices."""
    pos = State("pos", shape=(2,))
    pos._slice = slice(0, 2)

    fn = lower_to_jax(pos * Tau())
    params = {TAU_PARAM: jnp.linspace(0.0, 1.0, 3)}
    out = fn(jnp.array([2.0, 4.0]), None, 1, params)

    np.testing.assert_allclose(np.asarray(out), [1.0, 2.0], rtol=1e-6)


# =============================================================================
# Guess Resolution
# =============================================================================


def _sliced(states, controls):
    """Assign slices the way preprocessing does, so guesses can be lowered."""
    collect_and_assign_slices(states, controls)


def test_resolve_tau_guess_matches_closed_form():
    """A tau guess reproduces the linear interpolation it spells out."""
    N = 7
    p0, pf = np.array([0.0, 1.0]), np.array([10.0, -3.0])
    pos = State("pos", shape=(2,))
    pos.guess = lambda tau: p0 + (pf - p0) * tau
    _sliced([pos], [])

    resolve_guess_exprs([pos], [], N)

    np.testing.assert_allclose(pos.guess, np.linspace(p0, pf, N), rtol=1e-6)
    assert pos._guess_expr is None


def test_resolve_cross_variable_reduction_is_per_node():
    """A reduction over a referenced variable reduces within each node, not across."""
    N = 6
    rng = np.random.default_rng(0)
    vel = State("vel", shape=(3,))
    vel.guess = rng.normal(size=(N, 3))
    speed = State("speed", shape=(1,))
    speed.guess = Norm(vel)
    _sliced([vel, speed], [])

    resolve_guess_exprs([vel, speed], [], N)

    np.testing.assert_allclose(speed.guess.ravel(), np.linalg.norm(vel.guess, axis=1), rtol=1e-6)


def test_resolve_orders_chained_dependencies():
    """A guess that depends on a guess that depends on an array is resolved in order."""
    N = 5
    c = State("c", shape=(1,))
    c.guess = np.arange(N, dtype=float).reshape(-1, 1)
    b = State("b", shape=(1,))
    b.guess = 2.0 * c
    a = State("a", shape=(1,))
    a.guess = b + 1.0
    # Declaration order deliberately puts the dependents first.
    states = [a, b, c]
    _sliced(states, [])

    resolve_guess_exprs(states, [], N)

    np.testing.assert_allclose(b.guess.ravel(), 2.0 * np.arange(N), rtol=1e-6)
    np.testing.assert_allclose(a.guess.ravel(), 2.0 * np.arange(N) + 1.0, rtol=1e-6)


def test_resolve_reads_controls_and_parameters():
    """Guesses may reference controls and Parameters as freely as states."""
    N = 4
    gain = Parameter("gain", shape=(1,), value=np.array([3.0]))
    u = Control("u", shape=(1,))
    u.guess = np.ones((N, 1))
    work = State("work", shape=(1,))
    work.guess = gain * u
    _sliced([work], [u])

    resolve_guess_exprs([work], [u], N)

    np.testing.assert_allclose(work.guess, np.full((N, 1), 3.0), rtol=1e-6)


def test_resolve_reshapes_scalar_result_for_shape_one_variable():
    """A per-node scalar fills a shape-(1,) variable's (N, 1) guess."""
    N = 5
    speed = State("speed", shape=(1,))
    speed.guess = lambda tau: 4.0 * tau
    _sliced([speed], [])

    resolve_guess_exprs([speed], [], N)

    assert speed.guess.shape == (N, 1)
    np.testing.assert_allclose(speed.guess.ravel(), 4.0 * np.linspace(0, 1, N), rtol=1e-6)


def test_resolve_cycle_raises_naming_the_variables():
    """Mutually dependent guesses cannot be ordered and say so."""
    N = 4
    a = State("a", shape=(1,))
    b = State("b", shape=(1,))
    a.guess = b + 1.0
    b.guess = a * 2.0
    _sliced([a, b], [])

    with pytest.raises(ValueError) as excinfo:
        resolve_guess_exprs([a, b], [], N)

    msg = str(excinfo.value)
    assert "Circular initial-guess dependency" in msg
    assert "'a'" in msg and "'b'" in msg


def test_resolve_unknown_dependency_raises():
    """Referencing a variable outside the problem names both variables."""
    N = 4
    stranger = State("stranger", shape=(1,))
    stranger.guess = np.zeros((N, 1))
    x = State("x", shape=(1,))
    x.guess = stranger + 1.0
    _sliced([x], [])

    with pytest.raises(ValueError) as excinfo:
        resolve_guess_exprs([x], [], N)

    msg = str(excinfo.value)
    assert "'x'" in msg and "'stranger'" in msg
    assert "not one of this problem's states or controls" in msg


def test_resolve_wrong_shape_raises_naming_the_variable():
    """A per-node result of the wrong width reports both shapes."""
    N = 5
    vel = State("vel", shape=(3,))
    vel.guess = np.ones((N, 3))
    pos = State("pos", shape=(3,))
    pos.guess = Norm(vel)  # per-node scalar, but pos needs three values
    _sliced([vel, pos], [])

    with pytest.raises(ValueError) as excinfo:
        resolve_guess_exprs([vel, pos], [], N)

    msg = str(excinfo.value)
    assert "'pos'" in msg
    assert f"({N}, 3)" in msg


def test_resolve_is_a_noop_without_symbolic_guesses():
    """Arrays are left untouched when nothing symbolic is pending."""
    N = 4
    pos = State("pos", shape=(1,))
    pos.guess = np.ones((N, 1))
    _sliced([pos], [])

    resolve_guess_exprs([pos], [], N)

    np.testing.assert_array_equal(pos.guess, np.ones((N, 1)))


# =============================================================================
# Builder Integration
# =============================================================================


def _double_integrator(N):
    """Minimal 1D double integrator, with guesses left for the caller to set."""
    pos = State("pos", shape=(1,))
    pos.min, pos.max = np.array([-10.0]), np.array([10.0])
    pos.initial, pos.final = np.array([0.0]), np.array([1.0])

    vel = State("vel", shape=(1,))
    vel.min, vel.max = np.array([-5.0]), np.array([5.0])
    vel.initial, vel.final = np.array([0.0]), np.array([0.0])

    accel = Control("accel", shape=(1,))
    accel.min, accel.max = np.array([-2.0]), np.array([2.0])

    dynamics = {"pos": vel, "vel": accel}
    time = Time(initial=0.0, final=2.0, min=0.0, max=4.0)
    return dynamics, [pos, vel], [accel], time


def test_fill_default_guesses_skips_pending_expr():
    """The linspace default must not clobber a guess the user wrote symbolically."""
    N = 5
    _, (pos, vel), _, _ = _double_integrator(N)
    pos.guess = 3.0 * vel

    fill_default_guesses([pos, vel], N)

    assert pos.guess is None
    assert pos._guess_expr is not None
    assert vel.guess is not None  # untouched sibling still gets the default


def test_validate_guesses_accepts_pending_expr():
    """A symbolic guess counts as a guess for the control that requires one."""
    vel = State("vel", shape=(1,))
    vel.guess = np.zeros((5, 1))
    accel = Control("accel", shape=(1,))
    accel.guess = -0.5 * vel

    validate_guesses([vel, accel])  # no error


def test_preprocess_resolves_guesses_and_derives_time_dilation():
    """The full pipeline resolves symbolic guesses before augmentation reads them."""
    from openscvx.symbolic.builder import preprocess_symbolic_problem

    N = 6
    dynamics, states, controls, time = _double_integrator(N)
    pos, vel = states
    (accel,) = controls
    pos.guess = lambda tau: tau**2
    vel.guess = 2.0 * pos
    accel.guess = np.zeros((N, 1))
    time.guess = lambda tau: 2.0 * tau

    problem = preprocess_symbolic_problem(
        dynamics=dynamics,
        constraints=[],
        states=states,
        controls=controls,
        N=N,
        time=time,
    )

    tau = np.linspace(0.0, 1.0, N)
    np.testing.assert_allclose(pos.guess.ravel(), tau**2, rtol=1e-6)
    np.testing.assert_allclose(vel.guess.ravel(), 2.0 * tau**2, rtol=1e-6)
    np.testing.assert_allclose(time.guess.ravel(), 2.0 * tau, rtol=1e-6)

    # Augmentation derives the time-dilation guess by differencing the time guess,
    # which only works because resolution ran first. Here dt/dtau is a constant 2.
    dilation = next(c for c in problem.controls if c.name == "_time_dilation")
    np.testing.assert_allclose(dilation.guess, np.full((N, 1), 2.0), rtol=1e-5)


def test_propagation_state_with_guess_expr_raises():
    """Propagation-only states carry no guess trajectory, so a symbolic one is rejected."""
    from openscvx.symbolic.builder import preprocess_symbolic_problem

    N = 5
    dynamics, states, controls, time = _double_integrator(N)
    pos, vel = states
    (accel,) = controls
    accel.guess = np.zeros((N, 1))

    distance = State("distance", shape=(1,))
    distance.initial = np.array([0.0])
    distance.guess = 2.0 * vel

    with pytest.raises(ValueError, match="not supported for propagation-only states"):
        preprocess_symbolic_problem(
            dynamics=dynamics,
            constraints=[],
            states=states,
            controls=controls,
            N=N,
            time=time,
            dynamics_prop_extra={"distance": vel},
            states_prop_extra=[distance],
        )


# =============================================================================
# Problem Integration
# =============================================================================


def test_problem_resolves_and_re_resolves_guess_exprs():
    """A symbolic guess reaches the solver state, and a new one takes effect on reset()."""
    import openscvx as ox

    N = 8
    dynamics, states, controls, time = _double_integrator(N)
    pos, vel = states
    (accel,) = controls
    accel.guess = np.zeros((N, 1))
    vel.guess = lambda tau: tau

    problem = ox.Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        constraints=[ox.ctcs(pos <= pos.max), ox.ctcs(pos.min <= pos)],
        N=N,
        time=time,
        algorithm={"k_max": 1},
    )
    problem.settings.dev.printing = False
    problem.initialize()

    tau = np.linspace(0.0, 1.0, N)
    seeded = np.asarray(problem._lowered.x_unified.guess[:, vel._slice]).ravel()
    np.testing.assert_allclose(seeded, tau, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(problem.state.x[:, vel._slice]).ravel(), tau, rtol=1e-6)

    # MPC-style re-assignment: a new expression between solves takes effect on reset().
    vel.guess = lambda tau: 1.0 - tau
    problem.reset()

    reseeded = np.asarray(problem._lowered.x_unified.guess[:, vel._slice]).ravel()
    np.testing.assert_allclose(reseeded, 1.0 - tau, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(problem.state.x[:, vel._slice]).ravel(), 1.0 - tau, rtol=1e-6
    )

    jax.clear_caches()


def test_problem_accepts_vector_guess_expr_across_variables():
    """Concatenated cross-variable guesses survive the build end to end."""
    import openscvx as ox

    N = 6
    pos = State("pos", shape=(2,))
    pos.min, pos.max = np.array([-10.0, -10.0]), np.array([10.0, 10.0])
    pos.initial, pos.final = np.array([0.0, 0.0]), np.array([1.0, 2.0])

    vel = State("vel", shape=(2,))
    vel.min, vel.max = np.array([-5.0, -5.0]), np.array([5.0, 5.0])
    vel.initial, vel.final = np.array([0.0, 0.0]), np.array([0.0, 0.0])
    vel.guess = lambda tau: Concat(tau, 2.0 * tau)

    accel = Control("accel", shape=(2,))
    accel.min, accel.max = np.array([-2.0, -2.0]), np.array([2.0, 2.0])
    accel.guess = -0.5 * vel

    time = Time(initial=0.0, final=2.0, min=0.0, max=4.0)
    problem = ox.Problem(
        dynamics={"pos": vel, "vel": accel},
        states=[pos, vel],
        controls=[accel],
        constraints=[],
        N=N,
        time=time,
        algorithm={"k_max": 1},
    )
    problem.settings.dev.printing = False
    problem.initialize()

    tau = np.linspace(0.0, 1.0, N)
    np.testing.assert_allclose(vel.guess, np.stack([tau, 2.0 * tau], axis=1), rtol=1e-6)
    np.testing.assert_allclose(accel.guess, -0.5 * vel.guess, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(problem._lowered.u_unified.guess[:, accel._slice]),
        accel.guess,
        rtol=1e-6,
    )

    jax.clear_caches()
