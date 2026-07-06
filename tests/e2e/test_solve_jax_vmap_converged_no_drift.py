"""Under vmap, an early-converging element's iterate doesn't drift.

``jax.lax.while_loop`` runs until every batch element converges — converged
elements keep receiving body calls while their peers iterate. Without a
freeze, those extra iterations would compound subproblem-solver round-off
into the converged element's state and the autotuner would keep advancing
``lam_prox``, so the batched result for that element would not match the
single-problem ``solve_jax``. ``make_solve_loop`` selects the unchanged
state for converged elements (see Open Question 2 in
``plans/jax-pure-solve.md``); this test pins that behavior.

The harness uses a loose ``ep_tr`` so the first element converges in a few
iterations on a near-default initial pin, while the second element starts
much further off-default and needs more iterations to converge. Under vmap,
the loop keeps running for both — the freeze pins the first element's
state at its convergence iteration so it matches the single-problem solve.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _build_problem(backend: str, n: int = 30, k_max: int = 40):
    """A converging brachistochrone with a loose-enough ``ep_tr`` for SCP
    to actually fall below the threshold within ``k_max`` iterations."""
    import openscvx as ox
    from openscvx import Problem

    g = 9.81

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

    states = [position, velocity]
    controls = [theta]

    dynamics = {
        "position": ox.Concat(
            velocity[0] * ox.Sin(theta[0]),
            -velocity[0] * ox.Cos(theta[0]),
        ),
        "velocity": g * ox.Cos(theta[0]),
    }
    constraint_exprs = [ox.ctcs(s <= s.max) for s in states] + [ox.ctcs(s.min <= s) for s in states]

    time = ox.Time(
        initial=0.0,
        final=("minimize", 2.0),
        min=0.0,
        max=2.0,
        uniform_time_grid=True,
    )

    prob = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraint_exprs,
        N=n,
        float_dtype="float64",
        algorithm={
            "autotuner": "ConstantProximalWeight",
            "lam_prox": 1e0,
            "lam_cost": 6e-1,
            "k_max": k_max,
            "ep_tr": 1e-2,  # loose enough that the loop actually converges
            "ep_vb": 1e-2,
            "ep_vc": 1e-2,
        },
        solver={"backend": backend},
    )
    prob.settings.dev.printing = False
    return prob


@pytest.mark.parametrize(
    "backend", ["cvxpy", pytest.param("qpax", marks=pytest.mark.qpax)]
)
def test_vmap_no_drift_after_convergence(backend):
    prob = _build_problem(backend)
    prob.initialize()

    # Two ICs — element 0 keeps the default x-start (problem converges fast),
    # element 1 starts off by 1.0 in x (more aggressive shift, needs more
    # SCP iterations to converge).
    default_pin = prob.state.x_init_pin
    shifted_pin = default_pin.at[0].set(default_pin[0] + 1.0)
    stacked = jnp.stack([default_pin, shifted_pin])

    # Single-problem references.
    bare_xs = []
    for i in range(stacked.shape[0]):
        res = prob.solve_jax(x_initial=stacked[i])
        bare_xs.append(np.asarray(res.x))
    bare_xs = np.stack(bare_xs)

    # Batched.
    batched = jax.vmap(prob.solve_jax, in_axes=(0, None, None))(stacked, None, None)
    batched_xs = np.asarray(batched.x)

    for i in range(stacked.shape[0]):
        np.testing.assert_allclose(
            batched_xs[i],
            bare_xs[i],
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"batch element {i} drifted after convergence under vmap",
        )

    jax.clear_caches()
