"""``Problem.initialize`` seeds the solver state from current guesses/boundaries.

Trajectory guesses and boundary conditions live on the ``State``/``Control``
objects the user builds. They are only copied into the lowered representation by
``_sync_guesses`` / ``_sync_boundary_conditions``. Before #561, those syncs ran
only inside ``reset()``, so a ``.guess`` edited between ``Problem(...)`` and
``initialize()`` was silently ignored — the fresh state was seeded from the
guess snapshotted at construction. ``initialize()`` now re-reads both, matching
the ``.value`` / ``.initial`` mental model where post-construction edits take
effect on the next solve.
"""

import jax
import jax.numpy as jnp
import numpy as np

from tests.e2e.test_solve_batched_brachistochrone import _build_brachistochrone_with_params

# === initialize() re-reads post-construction guesses ===


def test_initialize_reads_guess_edited_after_construction():
    prob = _build_brachistochrone_with_params("cvxpy", n=8, k_max=1)

    # The fixture seeds theta.guess at construction; overwrite it afterwards.
    # (symbolic.controls also carries the augmented ``_time_dilation`` control.)
    (theta,) = [c for c in prob.symbolic.controls if c.name == "theta"]
    new_guess = np.full((prob.symbolic.N, 1), 0.42)
    theta.guess = new_guess

    prob.initialize()

    # The solver state is seeded from the *edited* guess, not the construction-
    # time one. Pre-#561 this held the fixture's linspace instead.
    np.testing.assert_allclose(np.asarray(prob.state.u[:, theta._slice]), new_guess)

    jax.clear_caches()


def test_initialize_reads_boundary_condition_edited_after_construction():
    prob = _build_brachistochrone_with_params("cvxpy", n=8, k_max=1)

    position = prob.symbolic.states[0]
    new_initial = np.array([1.0, 9.0])
    position.initial = new_initial

    prob.initialize()

    np.testing.assert_allclose(
        np.asarray(prob.state.x_init_pin[position._slice]), new_initial
    )

    jax.clear_caches()


# === the reset() workaround and initialize() now agree ===


def test_initialize_matches_reset_for_guess_sync():
    prob = _build_brachistochrone_with_params("cvxpy", n=8, k_max=1)

    (theta,) = [c for c in prob.symbolic.controls if c.name == "theta"]
    prob.initialize()

    # The historical workaround: edit the guess, then reset() to sync it.
    reset_guess = np.full((prob.symbolic.N, 1), 0.17)
    theta.guess = reset_guess
    prob.reset()
    np.testing.assert_allclose(np.asarray(prob.state.u[:, theta._slice]), reset_guess)

    jax.clear_caches()
