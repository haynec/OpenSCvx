"""``post_process_batched`` sizes its dense grid per-segment, not per-total.

The propagation solver is compiled with a *per-segment* tau buffer of width
``settings.prp.max_tau_len``.  ``post_process_batched`` used to clamp the
*total* number of dense output points to that per-segment width, collapsing
trajectories that would otherwise carry hundreds of samples down to a handful.
These tests pin the fixed behaviour: the dense grid may far exceed
``max_tau_len`` in total, yet no single segment of any batch element ever
overfills the per-segment buffer.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.propagation import s_to_t
from tests.solvers._iteration_callback_helpers import build_brachistochrone


def _x_initial_stack(prob, shifts=(0.0, 0.3, -0.3)):
    """Stack a few initial pins by shifting the starting x-coordinate."""
    default_pin = prob.state.x_init_pin
    return jnp.stack([default_pin.at[0].set(default_pin[0] + s) for s in shifts])


def _worst_segment_occupancy(prob, results):
    """Largest per-segment dense-sample count across the batch.

    Mirrors the binning ``simulate_nonlinear_time`` performs: each dense time
    falls into the node-time segment ``[t_nodal[k], t_nodal[k + 1])``.  Returns
    the maximum count over all segments and all batch elements.
    """
    x_batch = np.asarray(results.X[-1])
    u_batch = np.asarray(results.U[-1])
    worst = 0
    for b in range(x_batch.shape[0]):
        t_nodal = np.asarray(
            s_to_t(x_batch[b], u_batch[b], prob.settings, prob._discretizer)
        ).reshape(-1)
        seg = np.clip(
            np.searchsorted(t_nodal, results.t_full[b], side="right") - 1,
            0,
            len(t_nodal) - 2,
        )
        counts = np.bincount(seg, minlength=len(t_nodal) - 1)
        worst = max(worst, int(counts.max()))
    return worst


# === Dense grid is sized by the total, capped only per-segment ===


def test_batched_dense_grid_exceeds_per_segment_capacity():
    """The default grid carries far more total points than one segment holds.

    ``ceil(T_max / prp.dt) + 1`` comfortably exceeds ``max_tau_len`` here, and
    with near-uniform segments each segment stays well within the per-segment
    buffer — so the total-count clamp was pure loss.  Before the fix the dense
    grid came out at exactly ``max_tau_len`` samples.
    """
    pytest.importorskip("qpax")

    prob = build_brachistochrone("qpax", n=8, k_max=20)
    prob.initialize()
    max_tau_len = prob.settings.prp.max_tau_len

    batched = prob.solve_batched(x_initial=_x_initial_stack(prob))
    results = prob.post_process_batched(batched)

    # The dense grid dwarfs the per-segment buffer — the old clamp would have
    # pinned it to exactly ``max_tau_len``.
    assert results.t_full.shape[-1] > max_tau_len
    # ...and the per-segment safety still holds: nothing overfills the buffer.
    assert _worst_segment_occupancy(prob, results) + 1 <= max_tau_len


# === Per-segment capacity is still enforced ===


def test_batched_large_n_times_request_is_clamped_to_per_segment_capacity():
    """An absurd explicit ``n_times`` is capped, and no segment overfills.

    Requesting far more points than the per-segment buffer could ever absorb
    forces the bound to bind.  A naive ``linspace`` of this size would drop
    thousands of samples into each segment and blow past ``max_tau_len`` — the
    propagation solver would crash on the negative pad.  The per-segment-aware
    bound reduces ``n_times`` just enough that every segment fits, so the call
    both returns a smaller grid and completes without error.
    """
    pytest.importorskip("qpax")

    prob = build_brachistochrone("qpax", n=8, k_max=20)
    prob.initialize()
    max_tau_len = prob.settings.prp.max_tau_len

    batched = prob.solve_batched(x_initial=_x_initial_stack(prob))
    huge = 100_000
    results = prob.post_process_batched(batched, n_times=huge)

    # The request was clamped well below what was asked for...
    assert results.t_full.shape[-1] < huge
    # ...precisely so that the busiest segment plus its endpoint sample stays
    # within the compiled per-segment buffer.
    assert _worst_segment_occupancy(prob, results) + 1 <= max_tau_len
