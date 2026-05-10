"""Tests for callable (deferred) guess specification on Variable / State / Control / Time."""

import numpy as np
import pytest

from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State
from openscvx.symbolic.expr.time import Time
from openscvx.symbolic.preprocessing import resolve_guesses

# ===========================================================================
# Setter behavior
# ===========================================================================


def test_callable_setter_stores_callable_clears_array():
    x = State("x", shape=(2,))
    x.initial = [0.0, 0.0]
    x.final = [1.0, 1.0]
    x.guess = np.zeros((5, 2))  # explicit array first
    assert x.guess is not None

    x.guess = lambda tau: np.outer(tau, [1.0, 1.0])

    assert x._guess_callable is not None
    assert x._guess is None  # array path cleared


def test_array_setter_clears_prior_callable():
    """Bug guard: if a callable was assigned, then later an array, the array must stick."""
    x = State("x", shape=(1,))
    x.initial = [0.0]
    x.final = [1.0]

    x.guess = lambda tau: tau.reshape(-1, 1)
    explicit = np.linspace(0.0, 1.0, 7).reshape(-1, 1) * 99.0
    x.guess = explicit

    assert x._guess_callable is None
    np.testing.assert_array_equal(x.guess, explicit)

    # Re-running resolution must not overwrite the explicit array.
    resolve_guesses([x], [], 7)
    np.testing.assert_array_equal(x.guess, explicit)


# ===========================================================================
# Eager signature validation at assignment time
# ===========================================================================


def test_callable_with_var_args_rejected_eagerly():
    x = State("x", shape=(1,))
    with pytest.raises(ValueError, match=r"\*args|VAR_POSITIONAL|\*"):
        x.guess = lambda *args: np.zeros((5, 1))


def test_callable_with_var_kwargs_rejected_eagerly():
    x = State("x", shape=(1,))
    with pytest.raises(ValueError, match=r"\*\*"):
        x.guess = lambda **kwargs: np.zeros((5, 1))


def test_callable_with_positional_only_rejected_eagerly():
    x = State("x", shape=(1,))

    def bad(tau, /):  # positional-only
        return tau.reshape(-1, 1)

    with pytest.raises(ValueError, match="positional-only"):
        x.guess = bad


# ===========================================================================
# Reserved-name (tau) dispatch
# ===========================================================================


def test_resolve_uses_tau_grid():
    N = 11
    x = State("x", shape=(2,))
    x.initial = [0.0, 0.0]
    x.final = [1.0, 1.0]
    x.guess = lambda tau: np.outer(tau, [10.0, 20.0])

    resolve_guesses([x], [], N)

    assert x.guess.shape == (N, 2)
    np.testing.assert_array_almost_equal(x.guess[0], [0.0, 0.0])
    np.testing.assert_array_almost_equal(x.guess[-1], [10.0, 20.0])
    np.testing.assert_array_almost_equal(x.guess[N // 2], [5.0, 10.0])


def test_control_callable_resolves():
    N = 8
    u = Control("u", shape=(2,))
    u.guess = lambda tau: np.tile([0.5, -0.5], (len(tau), 1))

    resolve_guesses([], [u], N)

    assert u.guess.shape == (N, 2)
    np.testing.assert_array_equal(u.guess[0], [0.5, -0.5])


# ===========================================================================
# Cross-variable references
# ===========================================================================


def test_cross_var_dependency_resolves_in_order():
    """A guess that names another variable receives that variable's resolved array."""
    N = 6
    pos = State("pos", shape=(1,))
    pos.initial = [0.0]
    pos.final = [10.0]
    # No explicit guess on pos -> default callable installed by resolver.

    vel = State("vel", shape=(1,))
    vel.initial = [0.0]
    vel.final = [0.0]
    # vel is finite-difference of pos
    vel.guess = lambda pos: np.gradient(pos.flatten()).reshape(-1, 1)

    resolve_guesses([pos, vel], [], N)

    expected_pos = np.linspace(0.0, 10.0, N).reshape(-1, 1)
    np.testing.assert_array_almost_equal(pos.guess, expected_pos)
    np.testing.assert_array_almost_equal(
        vel.guess.flatten(),
        np.gradient(expected_pos.flatten()),
    )


def test_unknown_dependency_name_raises():
    N = 5
    x = State("x", shape=(1,))
    x.guess = lambda velocity: velocity  # no such state

    with pytest.raises(ValueError, match=r"velocity"):
        resolve_guesses([x], [], N)


def test_default_argument_skipped_permissively():
    """A param with a default that matches no reserved/var name is left at its default."""
    N = 4
    x = State("x", shape=(1,))
    x.guess = lambda tau, scale=3.0: scale * tau.reshape(-1, 1)

    resolve_guesses([x], [], N)

    assert x.guess.shape == (N, 1)
    np.testing.assert_array_almost_equal(x.guess.flatten(), 3.0 * np.linspace(0.0, 1.0, N))


# ===========================================================================
# Cycle detection
# ===========================================================================


def test_cycle_detected_with_named_vars():
    N = 5
    a = State("a", shape=(1,))
    b = State("b", shape=(1,))
    a.guess = lambda b: b
    b.guess = lambda a: a

    with pytest.raises(ValueError, match="Cycle detected"):
        resolve_guesses([a, b], [], N)


# ===========================================================================
# Live MPC re-evaluation
# ===========================================================================


def test_callable_reflects_mutated_state_on_re_resolve():
    """A closure capturing state.initial picks up new values on re-resolve."""
    N = 5
    pos = State("pos", shape=(1,))
    pos.initial = [0.0]
    pos.final = [10.0]
    pos.guess = lambda tau, _self=pos: np.linspace(
        _self.initial[0], _self.final[0], len(tau)
    ).reshape(-1, 1)

    resolve_guesses([pos], [], N)
    np.testing.assert_array_almost_equal(pos.guess[0], [0.0])

    pos.initial = [5.0]  # MPC-style mutation
    # Re-resolve must reflect new initial.
    pos._resolve_guess(N, np.linspace(0, 1, N), {})
    np.testing.assert_array_almost_equal(pos.guess[0], [5.0])


# ===========================================================================
# User-wrapped ox.init.linspace via lambda
# ===========================================================================


def test_user_wrapped_linspace():
    """The Phase-1 'wrap an ox.init helper in a lambda' pattern works under dispatch."""
    from openscvx.init import linspace

    N = 9
    x = State("x", shape=(2,))
    x.guess = lambda tau: linspace(
        keyframes=[np.zeros(2), np.array([5.0, -5.0])],
        nodes=[0, len(tau) - 1],
    )

    resolve_guesses([x], [], N)

    assert x.guess.shape == (N, 2)
    np.testing.assert_array_almost_equal(x.guess[0], [0.0, 0.0])
    np.testing.assert_array_almost_equal(x.guess[-1], [5.0, -5.0])


# ===========================================================================
# Reserved name (tau) cannot be a state/control name
# ===========================================================================


def test_state_named_tau_rejected_at_validation():
    from openscvx.symbolic.preprocessing import validate_no_reserved_guess_names

    bad = State("tau", shape=(1,))
    with pytest.raises(ValueError, match="reserved"):
        validate_no_reserved_guess_names([bad])


# ===========================================================================
# Time integration
# ===========================================================================


def test_time_callable_guess():
    N = 7
    t = Time(initial=0.0, final=20.0, min=0.0, max=30.0)
    t.guess = lambda tau: 20.0 * tau  # 1D, will be auto-reshaped

    resolve_guesses([t], [], N)

    assert t.guess.shape == (N, 1)
    np.testing.assert_array_almost_equal(t.guess[0], [0.0])
    np.testing.assert_array_almost_equal(t.guess[-1], [20.0])


def test_time_default_when_no_guess_set():
    """Time with no guess gets a tau-driven default linspace via resolve_guesses."""
    N = 5
    t = Time(initial=2.0, final=12.0, min=0.0, max=20.0)
    assert t.guess is None

    resolve_guesses([t], [], N)

    np.testing.assert_array_almost_equal(t.guess.flatten(), np.linspace(2.0, 12.0, N))


def test_time_dilation_callable_guess():
    N = 6
    t = Time(initial=0.0, final=10.0, min=0.0, max=20.0)
    t.time_dilation_guess = lambda tau: 8.0 * np.ones_like(tau)

    resolve_guesses([t], [], N)

    assert t.time_dilation_guess.shape == (N, 1)
    np.testing.assert_array_almost_equal(t.time_dilation_guess.flatten(), 8.0 * np.ones(N))


# ===========================================================================
# End-to-end: Problem build with callable guesses
# ===========================================================================


def test_problem_builds_with_callable_guesses():
    """Building a full Problem with callable guesses on state, control, and time
    must succeed and produce concrete arrays everywhere downstream code reads them."""
    import openscvx as ox

    N = 10

    pos = ox.State("pos", shape=(2,))
    pos.min = np.array([-10.0, -10.0])
    pos.max = np.array([10.0, 10.0])
    pos.initial = np.array([0.0, 0.0])
    pos.final = np.array([5.0, 5.0])
    # Lazy state guess via tau (no init+final default needed)
    pos.guess = lambda tau: np.outer(tau, [5.0, 5.0])

    vel = ox.Control("vel", shape=(2,))
    vel.min = np.array([-2.0, -2.0])
    vel.max = np.array([2.0, 2.0])
    # Cross-variable dispatch: depend on the resolved pos guess.
    vel.guess = lambda pos: np.gradient(pos, axis=0)

    time = ox.Time(
        initial=0.0,
        final=("minimize", 5.0),
        min=0.0,
        max=10.0,
        guess=lambda tau: 5.0 * tau,
    )

    problem = ox.Problem(
        dynamics={"pos": vel},
        states=[pos],
        controls=[vel],
        time=time,
        constraints=[],
        N=N,
    )

    assert pos.guess is not None and pos.guess.shape == (N, 2)
    assert vel.guess is not None and vel.guess.shape == (N, 2)
    time_state = next(s for s in problem.symbolic.states if s.name == "time")
    assert time_state.guess.shape == (N, 1)
    np.testing.assert_array_almost_equal(time_state.guess[0], [0.0])
    np.testing.assert_array_almost_equal(time_state.guess[-1], [5.0])
