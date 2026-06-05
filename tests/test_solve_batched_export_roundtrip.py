"""``solve_batched`` exports once and deserializes across processes.

Under ``save_compiled=True`` the whole vmapped SCP loop is exported to the
solver cache on the first call; a later ``Problem`` with the same structure
deserializes that artifact instead of recompiling. This is the entire reason
``solve_batched`` exists over ``jax.vmap(solve_jax)`` (which re-traces every
launch). The second half asserts the correctness-critical cache key (§4):
anything that changes the exported loop — backend, ``solver_args``, ``k_max``,
discretizer, scaling — must produce a *different* path, so a stale artifact is
never silently reused.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.utils.caching import get_solve_batched_cache_path
from tests.solvers._iteration_callback_helpers import build_brachistochrone

pytestmark = pytest.mark.filterwarnings("ignore")


def _guess_stack(prob, shifts=(0.0, 0.3, -0.3, 0.6)):
    base_x = prob.state.x
    return jnp.stack([base_x.at[0, 0].set(base_x[0, 0] + s) for s in shifts])


def test_export_roundtrip_matches_and_skips_recompile(monkeypatch, tmp_path):
    pytest.importorskip("qpax")
    monkeypatch.setenv("OPENSCVX_CACHE_DIR", str(tmp_path))

    # Baseline: the in-process (``save_compiled=False``) batched solve, which
    # Phase 1 verified equals ``jax.vmap(solve_jax)``. ``jax.vmap(solve_jax)``
    # itself can't be the baseline once ``save_compiled=True`` — its inner
    # solvers are then ``call_exported``, which has no outer vmap rule (FACT2).
    ref_prob = build_brachistochrone("qpax", n=8, k_max=20)
    ref_prob.settings.sim.save_compiled = False
    ref_prob.initialize()
    reference = ref_prob.solve_batched(x_guess=_guess_stack(ref_prob))
    jax.clear_caches()

    # First process: traces, exports, writes the artifact.
    prob1 = build_brachistochrone("qpax", n=8, k_max=20)
    prob1.settings.sim.save_compiled = True
    prob1.initialize()
    first = prob1.solve_batched(x_guess=_guess_stack(prob1))

    artifacts = list(tmp_path.glob("compiled_solve_batched_*.jax"))
    assert len(artifacts) == 1, "first solve_batched should write exactly one artifact"
    artifact = artifacts[0]
    mtime_after_first = artifact.stat().st_mtime_ns

    assert first.x.shape == reference.x.shape == (4, 8, ref_prob.settings.sim.n_states)
    np.testing.assert_allclose(np.asarray(first.x), np.asarray(reference.x), atol=1e-5, rtol=1e-5)

    jax.clear_caches()

    # Second process: a fresh Problem deserializes the artifact (no re-export).
    prob2 = build_brachistochrone("qpax", n=8, k_max=20)
    prob2.settings.sim.save_compiled = True
    prob2.initialize()
    second = prob2.solve_batched(x_guess=_guess_stack(prob2))

    assert artifact.stat().st_mtime_ns == mtime_after_first, "second solve must not re-export"
    np.testing.assert_allclose(np.asarray(second.x), np.asarray(first.x), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(second.u), np.asarray(first.u), atol=1e-5, rtol=1e-5)

    jax.clear_caches()


def test_cache_key_invalidates_on_artifact_changing_state(tmp_path):
    pytest.importorskip("qpax")

    prob = build_brachistochrone("qpax", n=8, k_max=20)
    prob.initialize()

    def path(p, B=4, k_max=None):
        return get_solve_batched_cache_path(
            p.symbolic,
            p.settings,
            p._algorithm,
            p._solver,
            p._discretizer,
            B,
            p._algorithm.k_max if k_max is None else k_max,
            cache_dir=tmp_path,
        )

    base = path(prob)

    # Same problem, same call → identical path (a cache hit, not a stale miss).
    assert path(prob) == base

    # Batch size is baked into the artifact → part of the key.
    assert path(prob, B=2) != base

    # k_max is the resolved loop bound. Both routes that change it must re-key:
    # bumping the algorithm default, and a per-call ``max_iters`` override.
    prob._algorithm.k_max += 1
    assert path(prob) != base
    prob._algorithm.k_max -= 1
    assert path(prob) == base
    assert path(prob, k_max=prob._algorithm.k_max + 1) != base

    # solver_args (tolerances / iteration caps) are baked into the backend solve.
    saved_max_iter = prob._solver.solver_args.get("max_iter")
    prob._solver.solver_args["max_iter"] = (saved_max_iter or 30) + 7
    assert path(prob) != base
    prob._solver.solver_args["max_iter"] = saved_max_iter
    assert path(prob) == base

    # Scaling matrices (derived from state/control bounds, Q2) feed the metrics.
    saved = prob.settings.sim.inv_S_x
    prob.settings.sim.inv_S_x = saved * 2.0
    assert path(prob) != base
    prob.settings.sim.inv_S_x = saved
    assert path(prob) == base

    # A different convex backend is a different exported program.
    other = build_brachistochrone("cvxpy", n=8, k_max=20)
    other.initialize()
    assert path(other) != base
