"""Tests for OptimizationResults save/load round-trip."""

import numpy as np
import pytest

from openscvx.algorithms.optimization_results import OptimizationResults

# === Helpers ================================================================


def _make_result(include_optional=False, include_plotting=False):
    """Build a minimal OptimizationResults for testing."""
    N, n_x, n_u, n_iters = 10, 4, 2, 3

    X = [np.random.randn(N, n_x) for _ in range(n_iters)]
    U = [np.random.randn(N, n_u) for _ in range(n_iters)]

    result = OptimizationResults(
        converged=True,
        t_final=12.5,
        nodes={"pos": np.random.randn(N, 3), "vel": np.random.randn(N, 3)},
        trajectory={},
        X=X,
        U=U,
        discretization_history=[np.random.randn(N) for _ in range(n_iters)],
        J_tr_history=[np.array(float(i)) for i in range(n_iters)],
        J_vb_history=[np.array(float(i) * 0.1) for i in range(n_iters)],
        J_vc_history=[np.array(float(i) * 0.01) for i in range(n_iters)],
        TR_history=[np.array(1.0 / (i + 1)) for i in range(n_iters)],
        VC_history=[np.array(10.0 / (i + 1)) for i in range(n_iters)],
        lam_prox_history=[0.1, 0.2, 0.3],
        actual_reduction_history=[1.0, 0.5, 0.1],
        pred_reduction_history=[1.2, 0.6, 0.15],
        acceptance_ratio_history=[0.83, 0.83, 0.67],
    )

    if include_optional:
        result.t_full = np.linspace(0, 12.5, 200)
        result.x_full = np.random.randn(200, n_x)
        result.u_full = np.random.randn(200, n_u)
        result.cost = 42.0
        result.ctcs_violation = np.random.randn(200)
        result.trajectory = {
            "pos": np.random.randn(200, 3),
            "vel": np.random.randn(200, 3),
        }

    if include_plotting:
        result.plotting_data = {
            "custom_array": np.array([1.0, 2.0, 3.0]),
            "scalar_val": np.array(99.0),
        }

    return result


# === Tests ==================================================================


def test_save_load_roundtrip_minimal(tmp_path):
    """Core fields survive a save/load round-trip."""
    original = _make_result()
    path = tmp_path / "result.npz"
    original.save(path)

    loaded = OptimizationResults.load(path)

    assert loaded.converged == original.converged
    assert loaded.t_final == pytest.approx(original.t_final)

    # Nodes
    for key in original.nodes:
        np.testing.assert_array_equal(loaded.nodes[key], original.nodes[key])

    # History lists (stacked arrays)
    assert len(loaded.X) == len(original.X)
    for a, b in zip(loaded.X, original.X):
        np.testing.assert_array_equal(a, b)

    assert len(loaded.U) == len(original.U)
    for a, b in zip(loaded.U, original.U):
        np.testing.assert_array_equal(a, b)

    assert len(loaded.discretization_history) == len(original.discretization_history)

    # Float lists
    assert loaded.lam_prox_history == pytest.approx(original.lam_prox_history)
    assert loaded.actual_reduction_history == pytest.approx(original.actual_reduction_history)
    assert loaded.pred_reduction_history == pytest.approx(original.pred_reduction_history)
    assert loaded.acceptance_ratio_history == pytest.approx(original.acceptance_ratio_history)

    # Optional fields should be None when not saved
    assert loaded.t_full is None
    assert loaded.x_full is None
    assert loaded.u_full is None
    assert loaded.cost is None
    assert loaded.ctcs_violation is None
    assert loaded.trajectory == {}


def test_save_load_roundtrip_full(tmp_path):
    """Optional and plotting fields survive a save/load round-trip."""
    original = _make_result(include_optional=True, include_plotting=True)
    path = tmp_path / "result_full.npz"
    original.save(path)

    loaded = OptimizationResults.load(path)

    np.testing.assert_array_almost_equal(loaded.t_full, original.t_full)
    np.testing.assert_array_almost_equal(loaded.x_full, original.x_full)
    np.testing.assert_array_almost_equal(loaded.u_full, original.u_full)
    assert loaded.cost == pytest.approx(original.cost)
    np.testing.assert_array_almost_equal(loaded.ctcs_violation, original.ctcs_violation)

    # Trajectory dict
    for key in original.trajectory:
        np.testing.assert_array_equal(loaded.trajectory[key], original.trajectory[key])

    # Plotting data
    for key in original.plotting_data:
        np.testing.assert_array_equal(loaded.plotting_data[key], original.plotting_data[key])


def test_save_load_auto_npz_suffix(tmp_path):
    """numpy appends .npz automatically; load handles missing suffix."""
    original = _make_result()
    path = tmp_path / "no_suffix"
    original.save(path)

    # numpy creates no_suffix.npz
    loaded = OptimizationResults.load(path)
    assert loaded.converged == original.converged


def test_save_load_empty_histories(tmp_path):
    """Empty history lists round-trip as empty lists."""
    result = OptimizationResults(
        converged=False,
        t_final=0.0,
        X=[],
        U=[],
        discretization_history=[],
        J_tr_history=[],
        J_vb_history=[],
        J_vc_history=[],
        TR_history=[],
        VC_history=[],
        lam_prox_history=[],
        actual_reduction_history=[],
        pred_reduction_history=[],
        acceptance_ratio_history=[],
    )
    path = tmp_path / "empty.npz"
    result.save(path)

    loaded = OptimizationResults.load(path)
    assert loaded.X == []
    assert loaded.U == []
    assert loaded.lam_prox_history == []
    assert loaded.converged is False


def test_x_u_properties_after_load(tmp_path):
    """The .x and .u properties work on loaded results."""
    original = _make_result()
    path = tmp_path / "props.npz"
    original.save(path)

    loaded = OptimizationResults.load(path)
    np.testing.assert_array_equal(loaded.x, original.x)
    np.testing.assert_array_equal(loaded.u, original.u)


def test_plotting_data_skips_non_array(tmp_path):
    """Non-array plotting_data values are silently skipped on save."""
    result = _make_result()
    result.plotting_data = {
        "good": np.array([1, 2, 3]),
        "bad_func": lambda: None,  # not serializable
    }
    path = tmp_path / "mixed_plotting.npz"
    result.save(path)

    loaded = OptimizationResults.load(path)
    np.testing.assert_array_equal(loaded.plotting_data["good"], np.array([1, 2, 3]))
    assert "bad_func" not in loaded.plotting_data
