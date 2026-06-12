"""The JAX-pure SCP solve loop shared by ``solve_jax`` and ``solve_batched``.

:func:`make_solve_loop` wraps any algorithm's fused iteration body in a
``lax.while_loop`` keyed on convergence. The loop structure is generic harness
logic — the iteration cap, the freeze-on-converged ``jax.vmap`` semantics, the
projecting-away of per-iteration diagnostics — identical for every algorithm.
The one algorithm-specific piece is the convergence predicate, which the caller
passes in (typically :meth:`Algorithm.converged`), so a custom algorithm's
policy is honored on the JAX paths exactly as it is on the Python ``solve()``
path.
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from .base import AlgorithmState


def make_solve_loop(
    iteration_fn: Callable[[AlgorithmState, dict], Tuple[AlgorithmState, object]],
    converged: Callable[[AlgorithmState], jnp.ndarray],
) -> Callable[[AlgorithmState, dict], AlgorithmState]:
    """Wrap ``iteration_fn`` in a ``lax.while_loop`` keyed on convergence.

    The loop runs ``iteration_fn`` until either ``converged(state)`` holds or
    the iteration counter ``state.k`` exceeds ``state.k_max`` — matching the
    Python ``while`` loop in ``Problem.solve()``. The cap and the thresholds the
    predicate reads are :class:`AlgorithmState` fields (runtime inputs, not
    closure constants), so one built loop serves every tolerance / ``max_iters``
    setting and ``jax.vmap`` batches them per element. The per-iteration
    diagnostics returned alongside the next state are projected away so the loop
    carry stays ``state -> state`` (XLA dead-code-eliminates their host-only
    pieces).

    This loop backs ``Problem.solve_jax`` and ``Problem.solve_batched``; the
    Python ``Problem.solve()`` drives ``iteration_fn`` directly so it can record
    the diagnostics into ``AlgorithmHistory`` per iteration.

    Args:
        iteration_fn: A fused SCP body, ``(state, params) -> (next_state,
            diagnostics)``, built by
            :func:`~openscvx.algorithms.scvx.iteration.make_scp_iteration`.
        converged: The algorithm's convergence predicate,
            ``state -> bool``, JAX-traceable (it runs inside the loop cond and
            is ``jax.vmap``'d per batch element). Pass
            :meth:`~openscvx.algorithms.base.Algorithm.converged`.

    Returns:
        ``solve_loop(state, params) -> final_state``.
    """

    def solve_loop(state: AlgorithmState, params: dict) -> AlgorithmState:
        def cond(state: AlgorithmState) -> jnp.ndarray:
            return (state.k <= state.k_max) & jnp.logical_not(converged(state))

        def body(state: AlgorithmState) -> AlgorithmState:
            # Under ``jax.vmap`` the ``lax.while_loop`` keeps running until
            # every batch element has converged; without a freeze, the body
            # would keep mutating already-converged elements (their iterates
            # drift through repeated subproblem solves, and the autotuner
            # would keep advancing ``lam_prox`` / ``lam_cost``). Selecting
            # ``state`` for converged elements pins them to their first
            # post-convergence iterate, so a batched solve agrees with the
            # single-problem ``solve_jax`` on each element.
            is_converged = converged(state)
            next_state, _ = iteration_fn(state, params)

            def freeze(nxt, prev):
                # ``is_converged`` is scalar (single-problem) or shape ``(B,)``
                # (vmap'd); reshape with trailing 1-axes so it broadcasts over
                # each leaf's remaining dims.
                mask_shape = is_converged.shape + (1,) * (nxt.ndim - is_converged.ndim)
                return jnp.where(is_converged.reshape(mask_shape), prev, nxt)

            return jax.tree.map(freeze, next_state, state)

        return jax.lax.while_loop(cond, body, state)

    return solve_loop
