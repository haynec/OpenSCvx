"""Unit tests for the rank-based batch resolver behind ``Problem.solve_batched``.

``_resolve_batch_spec`` implements the one rule of the batched-solve API: a
value whose shape equals its declared (unbatched) shape is shared across the
batch; a value with exactly one extra leading axis is batched along it. Rank
against the declared shape decides — never the leading-axis value — with an
explicit ``jax.vmap``-style ``in_axes`` dict as the override. Pure-function
tests: no problem build, no solver runs.
"""

import numpy as np
import pytest

from openscvx.problem import _flatten_in_axes, _resolve_batch_spec

# === Rank rule ===


def test_shared_when_shape_matches_declared():
    B, axes, _ = _resolve_batch_spec(
        {
            "x_initial": (np.zeros((4, 3)), (3,)),
            "parameters.center": (np.zeros(3), (3,)),
        }
    )
    assert B == 4
    assert axes == {"x_initial": 0, "parameters.center": None}


def test_batched_when_one_extra_leading_axis():
    B, axes, _ = _resolve_batch_spec({"x_guess": (np.zeros((5, 10, 3)), (10, 3))})
    assert B == 5
    assert axes == {"x_guess": 0}


def test_scalar_declared_batched_as_vector():
    B, axes, _ = _resolve_batch_spec(
        {
            "parameters.radius": (np.zeros(7), ()),
            "parameters.gain": (1.5, ()),
        }
    )
    assert B == 7
    assert axes == {"parameters.radius": 0, "parameters.gain": None}


def test_none_values_are_skipped():
    B, axes, _ = _resolve_batch_spec(
        {
            "x_initial": (np.zeros((2, 3)), (3,)),
            "x_final": (None, (3,)),
        }
    )
    assert B == 2
    assert axes == {"x_initial": 0}


def test_shared_matrix_with_leading_axis_equal_to_B_stays_shared():
    # Declared (4, 3) passed as (4, 3) alongside a B=4 batched pin: rank
    # matches declared, so the coincidental leading axis must not batch it.
    B, axes, _ = _resolve_batch_spec(
        {
            "x_initial": (np.zeros((4, 6)), (6,)),
            "parameters.waypoints": (np.zeros((4, 3)), (4, 3)),
        }
    )
    assert B == 4
    assert axes == {"x_initial": 0, "parameters.waypoints": None}


def test_matrix_batched_with_full_extra_axis():
    B, axes, _ = _resolve_batch_spec({"parameters.waypoints": (np.zeros((4, 4, 3)), (4, 3))})
    assert B == 4
    assert axes == {"parameters.waypoints": 0}


# === in_axes overrides ===


def test_in_axes_forces_batched_on_rank_matching_value():
    # A declared-(4,) parameter passed as (4,) reads as shared; in_axes says
    # it is one scalar per batch element.
    B, axes, _ = _resolve_batch_spec(
        {"parameters.weights": (np.zeros(4), (4,))},
        in_axes={"parameters.weights": 0},
    )
    assert B == 4
    assert axes == {"parameters.weights": 0}


def test_in_axes_forces_shared():
    B, axes, _ = _resolve_batch_spec(
        {
            "x_initial": (np.zeros((2, 3)), (3,)),
            "x_guess": (np.zeros((2, 10, 3)), (10, 3)),
        },
        in_axes={"x_guess": None},
    )
    assert B == 2
    assert axes == {"x_initial": 0, "x_guess": None}


def test_in_axes_partial_spec_merges_with_inference():
    B, axes, _ = _resolve_batch_spec(
        {
            "parameters.weights": (np.zeros(4), (4,)),
            "parameters.center": (np.zeros((4, 3)), (3,)),
        },
        in_axes={"parameters.weights": 0},
    )
    assert B == 4
    assert axes == {"parameters.weights": 0, "parameters.center": 0}


def test_in_axes_unknown_name_raises():
    with pytest.raises(ValueError, match=r"unknown entry.*'x_intial'"):
        _resolve_batch_spec(
            {"x_initial": (np.zeros((2, 3)), (3,))},
            in_axes={"x_intial": 0},
        )


def test_in_axes_invalid_axis_value_raises():
    with pytest.raises(ValueError, match=r"in_axes\['x_initial'\] must be 0.*or None"):
        _resolve_batch_spec(
            {"x_initial": (np.zeros((2, 3)), (3,))},
            in_axes={"x_initial": 1},
        )


def test_in_axes_on_absent_entry_raises():
    with pytest.raises(ValueError, match=r"'x_final' as batched, but no value"):
        _resolve_batch_spec(
            {
                "x_initial": (np.zeros((2, 3)), (3,)),
                "x_final": (None, (3,)),
            },
            in_axes={"x_final": 0},
        )


def test_in_axes_forced_zero_on_scalar_raises():
    with pytest.raises(ValueError, match=r"scalar with no leading axis"):
        _resolve_batch_spec(
            {"parameters.gain": (1.5, ())},
            in_axes={"parameters.gain": 0},
        )


def test_in_axes_forced_zero_with_disagreeing_batch_size_raises():
    with pytest.raises(
        ValueError, match=r"'x_initial' has leading axis 2.*'parameters.weights' has leading axis 4"
    ):
        _resolve_batch_spec(
            {
                "x_initial": (np.zeros((2, 3)), (3,)),
                "parameters.weights": (np.zeros(4), (4,)),
            },
            in_axes={"parameters.weights": 0},
        )


# === Fill forms (algorithm entries) ===


def test_fill_scalar_is_shared_fill():
    B, axes, fills = _resolve_batch_spec(
        {
            "x_initial": (np.zeros((3, 6)), (6,)),
            "algorithm.lam_prox": (2.0, (10, 4)),
        },
        fill={"algorithm.lam_prox"},
    )
    assert B == 3
    assert axes["algorithm.lam_prox"] is None
    assert fills == {"algorithm.lam_prox"}


def test_fill_vector_is_batched_fill():
    B, axes, fills = _resolve_batch_spec(
        {"algorithm.lam_prox": (np.zeros(5), (10, 4))},
        fill={"algorithm.lam_prox"},
    )
    assert B == 5
    assert axes == {"algorithm.lam_prox": 0}
    assert fills == {"algorithm.lam_prox"}


def test_fill_exact_shapes_win_over_fills():
    # The field's exact shape and (B,) + it parse as exact, never as fill.
    B, axes, fills = _resolve_batch_spec(
        {
            "algorithm.lam_vc": (np.zeros((7, 4)), (7, 4)),
            "algorithm.lam_cost": (np.zeros((3, 2)), (2,)),
        },
        fill={"algorithm.lam_vc", "algorithm.lam_cost"},
    )
    assert B == 3
    assert axes == {"algorithm.lam_vc": None, "algorithm.lam_cost": 0}
    assert fills == set()


def test_fill_rank1_field_of_length_B_reads_shared_exact():
    # The documented collision: a rank-1 field whose length equals B parses
    # as shared-exact by precedence...
    B, axes, fills = _resolve_batch_spec(
        {
            "x_initial": (np.zeros((4, 6)), (6,)),
            "algorithm.lam_cost": (np.zeros(4), (4,)),
        },
        fill={"algorithm.lam_cost"},
    )
    assert axes["algorithm.lam_cost"] is None
    assert fills == set()


def test_fill_in_axes_forces_batched_fill_on_rank1_field():
    # ...and in_axes={name: 0} forces the per-element fill reading.
    B, axes, fills = _resolve_batch_spec(
        {"algorithm.lam_cost": (np.zeros(4), (4,))},
        in_axes={"algorithm.lam_cost": 0},
        fill={"algorithm.lam_cost"},
    )
    assert B == 4
    assert axes == {"algorithm.lam_cost": 0}
    assert fills == {"algorithm.lam_cost"}


def test_fill_scalar_on_scalar_field_is_exact_not_fill():
    B, axes, fills = _resolve_batch_spec(
        {
            "algorithm.ep_tr": (1e-4, ()),
            "algorithm.k_max": (np.zeros(6), ()),
        },
        fill={"algorithm.ep_tr", "algorithm.k_max"},
    )
    assert B == 6
    assert axes == {"algorithm.ep_tr": None, "algorithm.k_max": 0}
    assert fills == set()


def test_fill_bad_rank_lists_all_four_forms():
    with pytest.raises(ValueError, match=r"scalar \(\) to fill.*\(B,\) to fill one scalar"):
        _resolve_batch_spec(
            {"algorithm.lam_prox": (np.zeros((2, 3, 4, 5)), (10, 4))},
            fill={"algorithm.lam_prox"},
        )


def test_non_fill_entry_rejects_fill_forms():
    # Without fill capability a (B,) value against a (10, 3) declared shape
    # is just a shape mismatch — and the error does not advertise fills.
    with pytest.raises(ValueError, match=r"'x_guess' has shape \(5,\)") as exc:
        _resolve_batch_spec({"x_guess": (np.zeros(5), (10, 3))})
    assert "fill" not in str(exc.value)


# === in_axes prefix flattening ===


_PREFIX_ENTRIES = {
    "x_initial": (np.zeros((4, 3)), (3,)),
    "x_guess": (None, (10, 3)),
    "parameters.center": (np.zeros((4, 3)), (3,)),
    "parameters.radius": (0.5, ()),
    "algorithm.ep_tr": (np.zeros(4), ()),
}


def test_flatten_none_means_infer():
    assert _flatten_in_axes(None, _PREFIX_ENTRIES) == {}


def test_flatten_bare_zero_batches_every_passed_input():
    # x_guess was not passed (None) and must not be forced batched.
    assert _flatten_in_axes(0, _PREFIX_ENTRIES) == {
        "x_initial": 0,
        "parameters.center": 0,
        "parameters.radius": 0,
        "algorithm.ep_tr": 0,
    }


def test_flatten_subtree_prefix_applies_to_every_passed_key():
    flat = _flatten_in_axes({"parameters": 0}, _PREFIX_ENTRIES)
    assert flat == {"parameters.center": 0, "parameters.radius": 0}
    flat = _flatten_in_axes({"parameters": None, "x_initial": 0}, _PREFIX_ENTRIES)
    assert flat == {"parameters.center": None, "parameters.radius": None, "x_initial": 0}
    flat = _flatten_in_axes({"algorithm": 0}, _PREFIX_ENTRIES)
    assert flat == {"algorithm.ep_tr": 0}


def test_flatten_per_key_dict_unchanged():
    assert _flatten_in_axes({"parameters": {"center": 0}}, _PREFIX_ENTRIES) == {
        "parameters.center": 0
    }


def test_flatten_rejects_non_dict_non_zero_top_level():
    with pytest.raises(ValueError, match=r"in_axes must be 0.*a dict.*or None"):
        _flatten_in_axes(1, _PREFIX_ENTRIES)


def test_flatten_rejects_invalid_subtree_spec():
    with pytest.raises(
        ValueError, match=r"in_axes\['parameters'\] must be 0 or None.*per-key dict"
    ):
        _flatten_in_axes({"parameters": 1}, _PREFIX_ENTRIES)


# === Teaching errors ===


def test_shape_mismatch_names_entry_and_shapes():
    with pytest.raises(
        ValueError, match=r"'parameters.center' has shape \(5, 2\).*unbatched.*\(3,\)"
    ):
        _resolve_batch_spec({"parameters.center": (np.zeros((5, 2)), (3,))})


def test_batch_size_disagreement_names_both_entries():
    with pytest.raises(
        ValueError, match=r"'x_initial' has leading axis 4 but 'x_final' has leading axis 5"
    ):
        _resolve_batch_spec(
            {
                "x_initial": (np.zeros((4, 3)), (3,)),
                "x_final": (np.zeros((5, 3)), (3,)),
            }
        )


def test_no_batched_entry_suggests_solve_jax():
    with pytest.raises(ValueError, match=r"solve_jax\(\).*in_axes"):
        _resolve_batch_spec(
            {
                "x_initial": (np.zeros(3), (3,)),
                "parameters.center": (np.zeros(3), (3,)),
            }
        )
