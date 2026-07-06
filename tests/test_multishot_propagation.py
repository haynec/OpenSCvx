"""Tests for multishot propagation unpacking."""

import numpy as np
import pytest

import openscvx as ox
from openscvx.algorithms.history import MultishotPropagation, unpack_multishot_V
from openscvx.algorithms.optimization_results import OptimizationResults


def _synthetic_V(
    *,
    n_x: int = 4,
    n_u: int = 1,
    n_segments: int = 3,
    n_substeps: int = 5,
) -> np.ndarray:
    seg_size = n_x + n_x * n_x + 2 * n_x * n_u
    V = np.zeros((n_segments * seg_size, n_substeps), dtype=np.float64)
    for seg in range(n_segments):
        base = seg * 100.0 + seg
        for j in range(n_substeps):
            row0 = seg * seg_size
            V[row0 : row0 + n_x, j] = base + j
    return V


def test_unpack_and_chronological_shapes():
    n_x, n_u, n_seg, n_sub = 4, 1, 3, 5
    V = _synthetic_V(n_x=n_x, n_u=n_u, n_segments=n_seg, n_substeps=n_sub)
    t_nodes = np.linspace(0.0, 1.0, n_seg + 1)
    prop = unpack_multishot_V(V, n_x=n_x, n_u=n_u, t_nodes=t_nodes)

    assert isinstance(prop, MultishotPropagation)
    assert prop.n_segments == n_seg
    assert prop.n_substeps == n_sub
    assert len(prop.segments()) == n_seg
    assert prop.segments()[0].shape == (n_sub, n_x)

    states, t = prop.chronological()
    expected_samples = n_seg * (n_sub - 1) + 1
    assert states.shape == (expected_samples, n_x)
    assert t.shape == (expected_samples,)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(1.0)


def test_t_nodes_length_mismatch_raises():
    V = _synthetic_V(n_segments=3, n_substeps=4)
    with pytest.raises(ValueError, match="t_nodes length"):
        unpack_multishot_V(V, n_x=4, n_u=1, t_nodes=np.array([0.0, 0.5]))


def test_state_by_name_and_object():
    q = ox.State("qpos", shape=(2,))
    qd = ox.State("qvel", shape=(2,))
    states = (q, qd)
    q._slice = slice(0, 2)
    qd._slice = slice(2, 4)

    V = _synthetic_V(n_x=4, n_u=1, n_segments=2, n_substeps=3)
    t_nodes = np.array([0.0, 0.5, 1.0])
    prop = unpack_multishot_V(V, n_x=4, n_u=1, t_nodes=t_nodes, states=states)

    by_name, t_name = prop.state("qpos")
    by_obj, t_obj = prop.state(q)
    assert np.allclose(by_name, by_obj)
    assert np.allclose(t_name, t_obj)
    assert by_name.shape[1] == 2

    with pytest.raises(KeyError, match="Unknown state"):
        prop.state("missing")


def test_multishot_propagation_on_results_empty_history():
    result = OptimizationResults(
        converged=True,
        t_final=1.0,
        nodes={},
        trajectory={},
        X=[np.zeros((3, 2))],
        U=[np.zeros((3, 1))],
        discretization_history=[],
    )
    assert result.multishot_propagation() is None


def test_multishot_propagation_on_results_with_history():
    n_x, n_u, n_seg, n_sub = 4, 1, 2, 3
    V = _synthetic_V(n_x=n_x, n_u=n_u, n_segments=n_seg, n_substeps=n_sub)
    q = ox.State("qpos", shape=(2,))
    q._slice = slice(0, 2)

    result = OptimizationResults(
        converged=True,
        t_final=1.0,
        nodes={"time": np.array([0.0, 0.5, 1.0])},
        trajectory={},
        _states=[q],
        X=[np.zeros((3, n_x))],
        U=[np.zeros((3, n_u))],
        discretization_history=[V],
    )
    prop = result.multishot_propagation()
    assert prop is not None
    q_traj, t = prop.state("qpos")
    assert q_traj.shape[0] == t.shape[0]


@pytest.mark.qpax
def test_brachistochrone_multishot_roundtrip():
    from tests.test_brachistochrone import _make_brachistochrone_problem

    problem = _make_brachistochrone_problem({"backend": "qpax", "verbose": False})
    problem.settings.prp.inter_sample = 4
    problem.settings.algorithm.k_max = 2
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False
    result = problem.solve()
    result = problem.post_process(result)

    prop = result.multishot_propagation()
    assert prop is not None
    states_full, t = prop.chronological()
    assert states_full.ndim == 2
    assert t.ndim == 1
    assert len(t) == states_full.shape[0]
    assert prop.n_segments == len(result.x) - 1
