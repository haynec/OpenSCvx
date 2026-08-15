"""Tests for user-supplied Jacobians (``Expr.with_jacobian``).

``WithJacobian`` wraps a subexpression and replaces its derivative with respect
to chosen variables, leaving the value and all other directions untouched. The
override is emitted as a ``jax.custom_jvp`` rule inside the lowered function, so
every downstream differentiation — constraint linearization, the discretizer's
Jacobians — sees it without further plumbing.

Sections:
1. Node mechanics (children, canonicalize, shapes, hashing, sparsity)
2. JAX lowering semantics (value, full and partial overrides, composition)
3. Problem integration (dynamics Jacobians and a solve)
4. Other backends (CVXPy, LaTeX)
"""

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.symbolic.expr import Constant, Control, Norm, State, Sum
from openscvx.symbolic.expr.autodiff import WithJacobian
from openscvx.symbolic.expr.time import Time
from openscvx.symbolic.lower import lower_to_jax, to_latex

# =============================================================================
# Helpers
# =============================================================================


def _sliced_state(name, n, start):
    x = State(name, shape=(n,))
    x._slice = slice(start, start + n)
    return x


def _sliced_control(name, n, start):
    u = Control(name, shape=(n,))
    u._slice = slice(start, start + n)
    return u


def _hash(expr) -> bytes:
    hasher = hashlib.sha256()
    expr._hash_into(hasher)
    return hasher.digest()


# =============================================================================
# Node Mechanics
# =============================================================================


def test_children_are_the_expression_then_each_jacobian():
    """Overridden variables are directions, not operands: only exprs are children."""
    x = _sliced_state("x", 2, 0)
    u = _sliced_control("u", 2, 0)
    jac_x, jac_u = Constant(np.eye(2)), Constant(np.zeros((2, 2)))

    node = (x * u).with_jacobian({x: jac_x, u: jac_u})

    assert isinstance(node, WithJacobian)
    assert node.children() == [node.expr, jac_x, jac_u]
    assert [var for var, _ in node.overrides] == [x, u]


def test_non_variable_key_raises_teaching_error():
    x = _sliced_state("x", 2, 0)

    with pytest.raises(TypeError) as excinfo:
        (x * x).with_jacobian({"x": np.eye(2)})

    assert "State or Control" in str(excinfo.value)


def test_empty_override_raises_rather_than_wrapping_silently():
    x = _sliced_state("x", 2, 0)

    with pytest.raises(ValueError) as excinfo:
        (x * x).with_jacobian({})

    assert "at least one" in str(excinfo.value)


def test_jacobians_are_coerced_to_expressions():
    """A constant matrix is a legal Jacobian and arrives as a Constant."""
    x = _sliced_state("x", 2, 0)

    node = Sum(x).with_jacobian({x: np.array([1.0, 0.0])})

    ((_, jac),) = node.overrides
    assert isinstance(jac, Constant)


def test_canonicalize_preserves_the_overrides():
    x = _sliced_state("x", 2, 0)
    node = (x * x).with_jacobian({x: 1.0 * Constant(np.eye(2))})

    canon = node.canonicalize()

    assert isinstance(canon, WithJacobian)
    assert [var for var, _ in canon.overrides] == [x]
    # Both the body and the Jacobian are canonicalized: the identity factor folds.
    assert isinstance(canon.overrides[0][1], Constant)
    assert canon.check_shape() == (2,)


def test_check_shape_is_the_wrapped_shape():
    x = _sliced_state("x", 3, 0)

    assert (x * x).with_jacobian({x: np.eye(3)}).check_shape() == (3,)
    assert Sum(x).with_jacobian({x: np.zeros(3)}).check_shape() == ()


def test_check_shape_rejects_a_scalar_jacobian_of_wrong_length():
    x = _sliced_state("x", 3, 0)
    node = Sum(x).with_jacobian({x: np.zeros(2)})

    with pytest.raises(ValueError) as excinfo:
        node.check_shape()

    msg = str(excinfo.value)
    assert "'x'" in msg and "(2,)" in msg and "(3,)" in msg


def test_check_shape_rejects_a_vector_jacobian_of_wrong_shape():
    x = _sliced_state("x", 3, 0)
    node = (x * x).with_jacobian({x: np.zeros((3, 2))})

    with pytest.raises(ValueError) as excinfo:
        node.check_shape()

    assert "(3, 2)" in str(excinfo.value) and "(3, 3)" in str(excinfo.value)


def test_hash_is_stable_across_equivalent_overrides():
    x = _sliced_state("x", 2, 0)
    y = _sliced_state("y", 2, 0)  # same slice: hashing is name-invariant

    assert _hash(Sum(x).with_jacobian({x: np.zeros(2)})) == _hash(
        Sum(y).with_jacobian({y: np.zeros(2)})
    )


def test_hash_distinguishes_different_jacobians_and_the_bare_expression():
    x = _sliced_state("x", 2, 0)
    body = Sum(x)

    zeros = _hash(body.with_jacobian({x: np.zeros(2)}))
    ones = _hash(body.with_jacobian({x: np.ones(2)}))

    assert zeros != ones
    assert zeros != _hash(body)


def test_hash_distinguishes_the_overridden_variable():
    """Same Jacobian, different direction, different problem."""
    x = _sliced_state("x", 2, 0)
    u = _sliced_control("u", 2, 0)
    body = Sum(x) + Sum(u)

    assert _hash(body.with_jacobian({x: np.zeros(2)})) != _hash(
        body.with_jacobian({u: np.zeros(2)})
    )


def test_repr_shows_the_overridden_variables():
    x = _sliced_state("x", 2, 0)

    assert "with_jacobian({x})" in repr(Sum(x).with_jacobian({x: np.zeros(2)}))


def test_sparsity_marks_the_overridden_columns_dense():
    """A constant Jacobian carries no dependence of its own but still couples."""
    x = _sliced_state("x", 3, 0)
    u = _sliced_control("u", 2, 0)
    body = Sum(x[0:1]) + Sum(u)

    S_x, S_u = body.with_jacobian({x: np.ones(3)}).sparsity(3, 2)

    assert S_x.all()  # widened from column 0 alone to the whole `x` slice
    assert S_u.all()  # inherited from the body


# =============================================================================
# JAX Lowering Semantics
# =============================================================================


def test_value_is_unchanged_by_the_wrapper():
    x = _sliced_state("x", 3, 0)
    u = _sliced_control("u", 2, 0)
    body = Sum(x * x) + Sum(u)

    plain, wrapped = lower_to_jax([body, body.with_jacobian({x: np.zeros(3)})])
    xv, uv = jnp.array([1.0, 2.0, 3.0]), jnp.array([0.5, -1.5])

    assert plain(xv, uv, 0, {}) == wrapped(xv, uv, 0, {})


def test_zero_jacobian_replaces_autodiff():
    x = _sliced_state("x", 3, 0)
    body = Sum(x * x)  # autodiff would give 2x

    (fn,) = lower_to_jax([body.with_jacobian({x: np.zeros(3)})])
    grad = jax.jacfwd(fn, argnums=0)(jnp.array([1.0, 2.0, 3.0]), jnp.zeros(0), 0, {})

    np.testing.assert_allclose(grad, np.zeros(3))


def test_wrong_jacobian_comes_back_exactly():
    """Nothing reconciles the override with the truth — the user's value is used."""
    x = _sliced_state("x", 3, 0)
    wrong = np.array([7.0, -1.0, 0.5])

    (fn,) = lower_to_jax([Sum(x * x).with_jacobian({x: wrong})])
    grad = jax.jacfwd(fn, argnums=0)(jnp.array([1.0, 2.0, 3.0]), jnp.zeros(0), 0, {})

    np.testing.assert_allclose(grad, wrong)


def test_partial_override_leaves_other_directions_to_autodiff():
    pos = _sliced_state("pos", 2, 0)
    vel = _sliced_state("vel", 2, 2)
    acc = _sliced_control("acc", 2, 0)
    body = Sum(pos * pos) + Sum(vel * vel) + Sum(acc * acc)

    (fn,) = lower_to_jax([body.with_jacobian({vel: np.zeros(2)})])
    xv = jnp.array([1.0, 2.0, 3.0, 4.0])
    uv = jnp.array([5.0, 6.0])

    grad_x = jax.jacfwd(fn, argnums=0)(xv, uv, 0, {})
    grad_u = jax.jacfwd(fn, argnums=1)(xv, uv, 0, {})

    np.testing.assert_allclose(grad_x, [2.0, 4.0, 0.0, 0.0])  # pos autodiff, vel overridden
    np.testing.assert_allclose(grad_u, [10.0, 12.0])


def test_control_override_leaves_the_state_direction_to_autodiff():
    x = _sliced_state("x", 2, 0)
    u = _sliced_control("u", 2, 0)
    body = Sum(x * x) + Sum(u * u)

    (fn,) = lower_to_jax([body.with_jacobian({u: np.array([1.0, 1.0])})])
    xv, uv = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])

    np.testing.assert_allclose(jax.jacfwd(fn, argnums=0)(xv, uv, 0, {}), [2.0, 4.0])
    np.testing.assert_allclose(jax.jacfwd(fn, argnums=1)(xv, uv, 0, {}), [1.0, 1.0])


def test_vector_valued_override_uses_the_full_matrix():
    x = _sliced_state("x", 2, 0)
    J = np.array([[1.0, 0.0], [0.0, 0.0]])  # true Jacobian would be diag(2x)

    (fn,) = lower_to_jax([(x * x).with_jacobian({x: J})])
    out = jax.jacfwd(fn, argnums=0)(jnp.array([1.0, 2.0]), jnp.zeros(0), 0, {})

    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, J)


def test_expression_valued_jacobian_is_evaluated_at_the_linearization_point():
    """The override is an expression, so it tracks the state it is evaluated at."""
    x = _sliced_state("x", 2, 0)
    # Half the true Jacobian of sum(x^2): a damped search direction.
    (fn,) = lower_to_jax([Sum(x * x).with_jacobian({x: x})])

    for xv in (jnp.array([1.0, 2.0]), jnp.array([-3.0, 0.5])):
        grad = jax.jacfwd(fn, argnums=0)(xv, jnp.zeros(0), 0, {})
        np.testing.assert_allclose(grad, np.asarray(xv))


def test_override_deep_inside_an_expression_chains_correctly():
    x = _sliced_state("x", 2, 0)
    u = _sliced_control("u", 1, 0)
    J = np.array([[1.0, 0.0], [0.0, 0.0]])
    body = Norm((x * x).with_jacobian({x: J})) + Sum(u)

    (fn,) = lower_to_jax([body])
    xv, uv = jnp.array([1.0, 2.0]), jnp.array([3.0])
    grad = jax.jacfwd(fn, argnums=0)(xv, uv, 0, {})

    # d/dx ||w||, w = x*x  ==>  (w / ||w||) @ J, with the true dw/dx replaced by J.
    w = np.array([1.0, 4.0])
    np.testing.assert_allclose(grad, (w / np.linalg.norm(w)) @ J, rtol=1e-6)
    np.testing.assert_allclose(jax.jacfwd(fn, argnums=1)(xv, uv, 0, {}), [1.0])


def test_shared_subexpression_between_body_and_jacobian_lowers_once():
    """Body and override may share nodes; the nested JVP trace must not leak."""
    x = _sliced_state("x", 2, 0)
    shared = Sum(x * x)
    body = shared + 1.0

    (fn,) = lower_to_jax([body.with_jacobian({x: shared * np.array([1.0, 0.0])})])
    xv = jnp.array([1.0, 2.0])

    np.testing.assert_allclose(fn(xv, jnp.zeros(0), 0, {}), 6.0)
    np.testing.assert_allclose(jax.jacfwd(fn, argnums=0)(xv, jnp.zeros(0), 0, {}), [5.0, 0.0])


def test_batched_jacobian_matches_the_constraint_lowering_pattern():
    """`_lower_jax_constraints` vmaps jacfwd over the node axis; overrides survive."""
    x = _sliced_state("x", 2, 0)
    u = _sliced_control("u", 1, 0)
    body = Sum(x * x) + Sum(u * u)

    (fn,) = lower_to_jax([body.with_jacobian({x: np.zeros(2)})])
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    U = jnp.array([[1.0], [2.0], [3.0]])

    grad_x = jax.vmap(jax.jacfwd(fn, argnums=0), in_axes=(0, 0, None, None))(X, U, 0, {})
    grad_u = jax.vmap(jax.jacfwd(fn, argnums=1), in_axes=(0, 0, None, None))(X, U, 0, {})

    np.testing.assert_allclose(grad_x, np.zeros((3, 2)))
    np.testing.assert_allclose(grad_u, 2.0 * np.asarray(U))


def test_jit_traces_the_override():
    x = _sliced_state("x", 2, 0)
    (fn,) = lower_to_jax([Sum(x * x).with_jacobian({x: np.zeros(2)})])

    jitted = jax.jit(jax.jacfwd(fn, argnums=0), static_argnums=(2,))
    np.testing.assert_allclose(jitted(jnp.array([1.0, 2.0]), jnp.zeros(0), 0, {}), [0.0, 0.0])


def test_time_override_behaves_like_a_state_override():
    t = Time(initial=0.0, final=1.0, min=0.0, max=2.0)
    t._slice = slice(0, 1)

    (fn,) = lower_to_jax([Sum(t * t).with_jacobian({t: np.array([3.0])})])
    grad = jax.jacfwd(fn, argnums=0)(jnp.array([2.0]), jnp.zeros(0), 0, {})

    np.testing.assert_allclose(grad, [3.0])


def test_lowering_without_a_slice_raises():
    x = State("x", shape=(2,))

    with pytest.raises(ValueError) as excinfo:
        lower_to_jax([Sum(x).with_jacobian({x: np.zeros(2)})])

    assert "slice" in str(excinfo.value)


# =============================================================================
# Problem Integration
# =============================================================================


def _drag_problem(N, override):
    """1-D double integrator with a quadratic drag term on the velocity dynamics.

    ``override`` selects whether the drag term carries a (zeroed) Jacobian
    override, so the two builds differ only in how they linearize.
    """
    import openscvx as ox

    pos = State("pos", shape=(1,))
    pos.min, pos.max = np.array([-10.0]), np.array([10.0])
    pos.initial, pos.final = np.array([0.0]), np.array([1.0])

    vel = State("vel", shape=(1,))
    vel.min, vel.max = np.array([-5.0]), np.array([5.0])
    vel.initial, vel.final = np.array([0.0]), np.array([0.0])

    accel = Control("accel", shape=(1,))
    accel.min, accel.max = np.array([-2.0]), np.array([2.0])
    accel.guess = np.zeros((N, 1))

    drag = -0.1 * vel * vel
    if override:
        drag = drag.with_jacobian({vel: np.zeros((1, 1))})

    problem = ox.Problem(
        dynamics={"pos": vel, "vel": accel + drag},
        states=[pos, vel],
        controls=[accel],
        constraints=[],
        N=N,
        time=Time(initial=0.0, final=2.0, min=0.0, max=4.0),
        algorithm={"k_max": 2},
    )
    problem.settings.dev.printing = False
    problem.initialize()
    return problem, vel


def test_dynamics_jacobian_reflects_the_override():
    """The state Jacobian the discretizer builds (`jacfwd(dynamics.f, 0)`) is overridden."""
    plain, vel = _drag_problem(6, override=False)
    wrapped, vel_w = _drag_problem(6, override=True)

    x = jnp.zeros(plain._lowered.x_unified.initial.shape[0]).at[vel._slice].set(3.0)
    # Unit controls, so the time-dilation factor multiplying f is 1 and the
    # Jacobian entries are the bare partials.
    u = jnp.ones(plain._lowered.u_unified.guess.shape[1])

    A_plain = jax.jacfwd(plain._lowered.dynamics.f, argnums=0)(x, u, 0, {})
    A_wrapped = jax.jacfwd(wrapped._lowered.dynamics.f, argnums=0)(x, u, 0, {})

    vel_row, vel_col = vel._slice.start, vel._slice.start
    # d(vel_dot)/d(vel) is -0.2 * vel = -0.6 exactly, and 0 once overridden.
    np.testing.assert_allclose(A_plain[vel_row, vel_col], -0.6, rtol=1e-6)
    np.testing.assert_allclose(A_wrapped[vel_row, vel_col], 0.0, atol=1e-12)
    # Nothing else moves: the two Jacobians agree away from the overridden entry.
    delta = np.asarray(A_wrapped) - np.asarray(A_plain)
    delta[vel_row, vel_col] = 0.0
    np.testing.assert_allclose(delta, 0.0, atol=1e-12)
    assert vel_w._slice == vel._slice

    jax.clear_caches()


def test_problem_with_an_overridden_jacobian_solves():
    problem, _ = _drag_problem(6, override=True)

    problem.solve()

    assert problem.state.x.shape[0] == 6
    jax.clear_caches()


# =============================================================================
# Other Backends
# =============================================================================


def test_cvxpy_lowering_raises_a_teaching_error():
    import cvxpy as cp

    from openscvx.symbolic.lowerers.cvxpy import lower_to_cvxpy

    x = _sliced_state("x", 2, 0)
    constraint = Sum(x).with_jacobian({x: np.zeros(2)}) <= 1.0

    with pytest.raises(NotImplementedError) as excinfo:
        lower_to_cvxpy(constraint, {"x": cp.Variable(2)})

    msg = str(excinfo.value)
    assert "with_jacobian" in msg and "CVXPy" in msg


def test_latex_renders_the_wrapped_expression_unchanged():
    x = _sliced_state("x", 2, 0)
    body = Sum(x * x)

    assert to_latex(body.with_jacobian({x: np.zeros(2)})) == to_latex(body)
