"""Pytree contract tests for ``SubproblemData`` / ``SubproblemSolution``.

These are the JAX-pure I/O containers exchanged between the SCP loop and
each backend's ``iteration_callback``. They must:

* Round-trip through ``jax.tree_util.tree_flatten`` / ``tree_unflatten``.
* Compose with ``jax.tree.map`` (e.g. ``jnp.zeros_like``) so downstream JAX
  transforms can build sentinel/zero values structurally.

The shapes used below are arbitrary; the contract being tested is that the
pytree registration preserves identity and field order. Backend-specific
shape conventions are exercised by the per-backend iteration_callback tests
landing in later phases.
"""

import jax
import jax.numpy as jnp
import pytest

from openscvx.solvers.ptr_solver import (
    StatusCode,
    SubproblemData,
    SubproblemSolution,
    status_code_to_str,
)

# ============================================================================
# Helpers
# ============================================================================


def _dummy_subproblem_data(N=4, n_x=3, n_u=2, n_nodal=2, n_cross=1):
    """Fill a SubproblemData with deterministic, shape-correct test values."""
    return SubproblemData(
        x_bar=jnp.ones((N, n_x)),
        u_bar=jnp.full((N, n_u), 2.0),
        A_d=jnp.zeros((N - 1, n_x, n_x)),
        B_d=jnp.zeros((N - 1, n_x, n_u)),
        C_d=jnp.zeros((N - 1, n_x, n_u)),
        x_prop=jnp.zeros((N - 1, n_x)),
        x_prop_plus=jnp.zeros((N, n_x)),
        D_d=jnp.zeros((N, n_x, n_x)),
        E_d=jnp.zeros((N, n_x, n_u)),
        nodal_g=jnp.zeros((N, n_nodal)),
        nodal_grad_x=jnp.zeros((N, n_nodal, n_x)),
        nodal_grad_u=jnp.zeros((N, n_nodal, n_u)),
        cross_g=jnp.zeros((n_cross,)),
        cross_grad_X=jnp.zeros((n_cross, N, n_x)),
        cross_grad_U=jnp.zeros((n_cross, N, n_u)),
        lam_prox=jnp.ones((N, n_x + n_u)),
        lam_cost=jnp.ones((n_x,)),
        lam_vc=jnp.ones((N - 1, n_x)),
        lam_vb_nodal=jnp.ones((N, n_nodal)),
        lam_vb_cross=jnp.ones((n_cross,)),
        x_init=jnp.array([0.0, jnp.nan, 1.0]),
        x_term=jnp.full((n_x,), jnp.nan),
    )


def _dummy_subproblem_solution(N=4, n_x=3, n_u=2, n_nodal=2, n_cross=1, n_z=10, n_cone=8):
    return SubproblemSolution(
        x=jnp.ones((N, n_x)),
        u=jnp.ones((N, n_u)),
        nu=jnp.zeros((N - 1, n_x)),
        nu_vb=jnp.zeros((N, n_nodal)),
        nu_vb_cross=jnp.zeros((n_cross,)),
        cost=jnp.asarray(1.23),
        status_code=jnp.asarray(int(StatusCode.OPTIMAL), dtype=jnp.int32),
        moreau_carry=(jnp.zeros((n_z,)), jnp.zeros((n_cone,)), jnp.zeros((n_cone,))),
    )


# ============================================================================
# SubproblemData
# ============================================================================


def test_subproblem_data_roundtrip():
    data = _dummy_subproblem_data()
    leaves, treedef = jax.tree_util.tree_flatten(data)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(rebuilt, SubproblemData)
    assert jnp.array_equal(rebuilt.x_bar, data.x_bar)
    assert jnp.array_equal(rebuilt.u_bar, data.u_bar)
    # NaN equality: x_init has sentinel NaNs, so compare via isnan masks.
    assert jnp.array_equal(jnp.isnan(rebuilt.x_init), jnp.isnan(data.x_init))
    assert jnp.array_equal(rebuilt.x_init[~jnp.isnan(rebuilt.x_init)], jnp.array([0.0, 1.0]))


def test_subproblem_data_zeros_like():
    data = _dummy_subproblem_data()
    zeros = jax.tree.map(jnp.zeros_like, data)

    assert isinstance(zeros, SubproblemData)
    assert zeros.x_bar.shape == data.x_bar.shape
    assert jnp.all(zeros.x_bar == 0)
    assert jnp.all(zeros.lam_prox == 0)
    assert zeros.x_init.shape == data.x_init.shape


# ============================================================================
# SubproblemSolution
# ============================================================================


def test_subproblem_solution_roundtrip():
    sol = _dummy_subproblem_solution()
    leaves, treedef = jax.tree_util.tree_flatten(sol)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(rebuilt, SubproblemSolution)
    assert jnp.array_equal(rebuilt.x, sol.x)
    assert int(rebuilt.status_code) == int(StatusCode.OPTIMAL)
    # ``moreau_carry`` is a nested tuple of arrays — must round-trip as one.
    assert isinstance(rebuilt.moreau_carry, tuple)
    assert len(rebuilt.moreau_carry) == 3
    for a, b in zip(rebuilt.moreau_carry, sol.moreau_carry):
        assert jnp.array_equal(a, b)


def test_subproblem_solution_zeros_like():
    sol = _dummy_subproblem_solution()
    zeros = jax.tree.map(jnp.zeros_like, sol)

    assert isinstance(zeros, SubproblemSolution)
    assert zeros.cost.shape == sol.cost.shape
    assert int(zeros.status_code) == 0  # int32 zero
    assert isinstance(zeros.moreau_carry, tuple)
    for leaf, original in zip(zeros.moreau_carry, sol.moreau_carry):
        assert leaf.shape == original.shape
        assert jnp.all(leaf == 0)


# ============================================================================
# StatusCode
# ============================================================================


@pytest.mark.parametrize(
    "code,label",
    [
        (StatusCode.OPTIMAL, "optimal"),
        (StatusCode.INFEASIBLE, "infeasible"),
        (StatusCode.UNBOUNDED, "unbounded"),
        (StatusCode.UNKNOWN, "unknown"),
    ],
)
def test_status_code_to_str(code, label):
    assert status_code_to_str(int(code)) == label
    # Accepts 0-d jax arrays as well — that's the JAX-pure call site.
    assert status_code_to_str(jnp.asarray(int(code), dtype=jnp.int32)) == label
