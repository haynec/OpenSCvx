"""Solved parameter values are recorded on results and drive post-processing.

``Problem.solve`` and ``Problem.solve_batched`` snapshot the merged, as-used
parameter values onto ``results.parameters``; ``post_process`` /
``post_process_batched`` propagate with the recorded values, so mutating
``problem.parameters`` between solve and post-process — or batching a
parameter that enters the dynamics — can no longer corrupt the dense
trajectory. The fixture's ``gain`` parameter scales the position kinematics,
which makes a wrong propagation parameter directly visible: gain 0 freezes
the dense positions, gain 1 moves them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_solve_batched_brachistochrone import _build_brachistochrone_with_params

# === solve() -> post_process() ===


def test_post_process_propagates_with_solved_parameters():
    prob = _build_brachistochrone_with_params("cvxpy", n=8, k_max=20)
    prob.initialize()
    result = prob.solve()

    # The solve records the as-used values as plain numpy.
    np.testing.assert_array_equal(result.parameters["gain"], np.array([1.0]))

    # Mutating the Problem's parameters afterwards only seeds the next solve;
    # post-processing propagates with the recorded gain of 1.0. Under gain 0
    # the position kinematics would be identically zero.
    prob.parameters["gain"] = np.array([0.0])
    result = prob.post_process()

    np.testing.assert_array_equal(result.parameters["gain"], np.array([1.0]))
    position = np.asarray(result.trajectory["position"])  # (n_times, 2)
    assert np.ptp(position[:, 0]) > 5.0  # moved from [0, 10] toward [10, 5]

    jax.clear_caches()


# === solve_batched() -> post_process_batched() ===


def test_post_process_batched_propagates_each_element_with_its_own_parameters():
    pytest.importorskip("qpax")

    prob = _build_brachistochrone_with_params("qpax", n=8, k_max=5)
    prob.initialize()

    gains = jnp.array([[0.0], [1.0]])  # (B, 1) vs declared (1,) -> batched
    results = prob.solve_batched(parameters={"gain": gains})

    # Post-broadcast snapshot: every value carries the leading (B,) axis,
    # the shared gravity replicated.
    np.testing.assert_allclose(np.asarray(results.parameters["gain"]), np.asarray(gains))
    np.testing.assert_allclose(
        np.asarray(results.parameters["gravity"]), np.broadcast_to([9.81], (2, 1))
    )

    results = prob.post_process_batched(results)
    position = np.asarray(results.trajectory["position"])  # (B, n_times, 2)

    # Element 0 solved with gain 0 and must propagate with gain 0: its
    # position kinematics are identically zero, so the dense trajectory
    # stays at the initial point. Element 1 (gain 1) moves.
    assert np.abs(position[0] - position[0, 0]).max() < 1e-6
    assert np.ptp(position[1, :, 0]) > 1.0

    jax.clear_caches()


def test_post_process_batched_without_snapshot_falls_back_to_problem_parameters():
    pytest.importorskip("qpax")

    prob = _build_brachistochrone_with_params("qpax", n=8, k_max=5)
    prob.initialize()

    results = prob.solve_batched(parameters={"gain": jnp.array([[0.0], [1.0]])})
    del results.parameters  # a results object that predates the snapshot

    with pytest.warns(UserWarning, match="no record of the parameter values"):
        results = prob.post_process_batched(results)
    position = np.asarray(results.trajectory["position"])

    # Every element falls back to the Problem's own gain of 1.0: even the
    # element solved with gain 0 moves during propagation.
    for b in range(2):
        assert np.ptp(position[b, :, 0]) > 0.5

    jax.clear_caches()
