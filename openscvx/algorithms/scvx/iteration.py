"""One fused, JAX-pure SCP iteration body.

Historically a single SCP step was stitched together in Python: ``Problem``
exported three separate JAX callables (continuous discretization, impulsive
discretization, propagation) and :meth:`PenalizedTrustRegion._subproblem`
invoked each one, copied NumPy out of every result, pushed the data through
the solver's ``update_*`` methods, and finally called ``solver.solve()``. Every
step crossed the NumPy↔JAX boundary several times, and there was no batch axis
anywhere.

This module collapses that stitching into one function. :func:`make_scp_iteration`
returns a ``(state, params) -> (state, IterationDiagnostics)`` callable that, on
the JAX side end to end, discretizes the current iterate, linearizes the
constraints, packs a :class:`~openscvx.solvers.ptr_solver.SubproblemData`, hands
it to the backend's :meth:`~openscvx.solvers.ptr_solver.PTRSolver.iteration_callback`,
and folds the returned :class:`~openscvx.solvers.ptr_solver.SubproblemSolution`
through the autotuner — producing the next :class:`AlgorithmState`. The
:class:`IterationDiagnostics` riding alongside carry the host-side bookkeeping
(raw discretization matrices, trust-region / virtual-control matrices, cost,
status) that the Python-loop ``step()`` records into ``AlgorithmHistory`` and
emits, exactly as the legacy path did. Because state is a registered pytree, the
body composes with ``jax.jit`` and ``jax.vmap``.

:func:`~openscvx.algorithms.loop.make_solve_loop` wraps that body in a
``lax.while_loop`` for the JAX-pure ``solve_jax`` / ``solve_batched`` paths;
``Problem.solve()`` instead drives the body from a Python ``while`` loop so it
can record the :class:`IterationDiagnostics` into ``AlgorithmHistory`` each
iteration.

For the Python ``solve()`` path, the two natural halves of each iteration are
also exposed as standalone factories:

- :func:`make_scp_prepare` — steps 1–4: discretize the current iterate,
  linearize constraints, and pack a :class:`~openscvx.solvers.ptr_solver.SubproblemData`.
- :func:`make_scp_finalize` — steps 6a–6c: discretize the candidate, compute
  SCP convergence metrics, and fold through the autotuner.

:func:`make_scp_iteration` is a thin combiner around them; its public signature
and semantics are unchanged. The split-half callables are used by
:meth:`PenalizedTrustRegion.step` to record separate discretization and convex-
solve wall-clock times per iteration.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from openscvx.solvers.ptr_solver import SubproblemData, SubproblemSolution

from ..state import AlgorithmState, CandidateIterate

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered import LoweredJaxConstraints

    from ..autotuner.base import AutotuningBase


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class IterationDiagnostics:
    """Host-side bookkeeping produced alongside the next :class:`AlgorithmState`.

    These ride beside the state out of :func:`make_scp_iteration` so the
    Python-loop ``step()`` can populate :class:`AlgorithmHistory` and the
    emitter exactly as the legacy ``_subproblem`` did — the data they carry is
    not on the JAX-traceable state pytree (raw discretization matrices are
    large and host-only; cost / status / ``J_lin`` come from the subproblem
    solve). :func:`make_solve_loop` projects them away.

    Attributes:
        cost: Boundary-weighted objective at the candidate (scalar) — the
            ``cost[-1]`` the emitter prints.
        status: Subproblem :class:`StatusCode` as ``int32``.
        J_lin: Subproblem optimal (linearized) cost (scalar).
        V: Raw continuous discretization matrix of the candidate.
        W: Raw impulsive discretization matrix of the candidate.
        TR: Scaled trust-region step matrix, shape ``(n_x + n_u, N)``.
        VC: Scaled virtual-control matrix, shape ``(N-1, n_x)``.
    """

    cost: jnp.ndarray
    status: jnp.ndarray
    J_lin: jnp.ndarray
    V: jnp.ndarray
    W: jnp.ndarray
    TR: jnp.ndarray
    VC: jnp.ndarray

    def tree_flatten(self):
        children = tuple(getattr(self, f.name) for f in fields(self))
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def make_scp_prepare(
    dis_continuous: Callable,
    dis_impulsive: Callable,
    jax_constraints: "LoweredJaxConstraints",
    settings: "Config",
) -> Callable[[AlgorithmState, dict], SubproblemData]:
    """Build the discretize-linearize-pack phase of one SCP iteration.

    Returns ``prepare_fn(state, params) -> SubproblemData`` covering steps 1–4
    of the full iteration:

    1. Discretize the current iterate through continuous dynamics →
       ``A_d, B_d, C_d, x_prop``.
    2. Discretize the impulsive dynamics about the propagated nodes →
       ``x_prop_plus, D_d, E_d``.
    3. Evaluate every nodal / cross-node constraint and its gradients.
    4. Pack the linearization, penalty weights, and boundary pins into a
       :class:`~openscvx.solvers.ptr_solver.SubproblemData`.

    The returned callable is JIT-able and forms the first of two pieces in the
    split-timing API used by :meth:`PenalizedTrustRegion.step`. The fused
    :func:`make_scp_iteration` delegates to this factory internally.

    Args:
        dis_continuous: Continuous discretization solver,
            ``(x, u, params) -> (A_d, B_d, C_d, x_prop, V)``.
        dis_impulsive: Impulsive discretization solver,
            ``(x_nodes, u, params) -> (x_prop_plus, D_d, E_d, W)``.
        jax_constraints: Lowered nodal / cross-node JAX constraints.
        settings: Problem configuration.

    Returns:
        ``prepare_fn(state, params) -> SubproblemData``.
    """
    N = settings.sim.n
    n_x = settings.sim.n_states
    n_u = settings.sim.n_controls
    n_nodal = len(jax_constraints.nodal)

    dis_continuous = dis_continuous.call if hasattr(dis_continuous, "call") else dis_continuous
    dis_impulsive = dis_impulsive.call if hasattr(dis_impulsive, "call") else dis_impulsive

    init_fixed = jnp.asarray(np.asarray(settings.sim.x.initial_type) == "Fix")
    x_initial = jnp.asarray(np.asarray(settings.sim.x.initial, dtype=float))

    def _discretize(x: jnp.ndarray, u: jnp.ndarray, params: dict):
        A_d, B_d, C_d, x_prop, V = dis_continuous(x, u, params)
        x0_prior = jnp.where(init_fixed, x_initial, x[0])
        x_nodes_prior = jnp.concatenate([x0_prior[None, :], x_prop], axis=0)
        x_prop_plus, D_d, E_d, W = dis_impulsive(x_nodes_prior, u, params)
        return A_d, B_d, C_d, x_prop, x_prop_plus, D_d, E_d, V, W

    def _linearize_constraints(x: jnp.ndarray, u: jnp.ndarray, params: dict):
        nodal_g = jnp.zeros((N, n_nodal))
        nodal_grad_x = jnp.zeros((N, n_nodal, n_x))
        nodal_grad_u = jnp.zeros((N, n_nodal, n_u))
        for c_idx, constraint in enumerate(jax_constraints.nodal):
            g = jnp.squeeze(jnp.asarray(constraint.func(x, u, 0, params)))
            if g.ndim == 0:
                g = jnp.broadcast_to(g, (N,))
            elif g.ndim > 1:
                g = g.reshape(g.shape[0], -1).sum(axis=1)

            grad_x = jnp.asarray(constraint.grad_g_x(x, u, 0, params))
            if grad_x.ndim == 1:
                grad_x = jnp.broadcast_to(grad_x, (N, grad_x.shape[0]))
            elif grad_x.ndim > 2:
                grad_x = grad_x.reshape(grad_x.shape[0], -1)[:, :n_x]

            grad_u = jnp.asarray(constraint.grad_g_u(x, u, 0, params))
            if grad_u.ndim == 1:
                grad_u = jnp.broadcast_to(grad_u, (N, grad_u.shape[0]))
            elif grad_u.ndim > 2:
                grad_u = grad_u.reshape(grad_u.shape[0], -1)[:, :n_u]

            nodes = jnp.asarray(constraint.nodes) if constraint.nodes is not None else jnp.arange(N)
            nodal_g = nodal_g.at[nodes, c_idx].set(g[nodes])
            nodal_grad_x = nodal_grad_x.at[nodes, c_idx].set(grad_x[nodes])
            nodal_grad_u = nodal_grad_u.at[nodes, c_idx].set(grad_u[nodes])

        if jax_constraints.cross_node:
            cross_g = jnp.stack(
                [jnp.asarray(c.func(x, u, params)) for c in jax_constraints.cross_node]
            )
            cross_grad_X = jnp.stack(
                [jnp.asarray(c.grad_g_X(x, u, params)) for c in jax_constraints.cross_node]
            )
            cross_grad_U = jnp.stack(
                [jnp.asarray(c.grad_g_U(x, u, params)) for c in jax_constraints.cross_node]
            )
        else:
            cross_g = jnp.zeros((0,))
            cross_grad_X = jnp.zeros((0, N, n_x))
            cross_grad_U = jnp.zeros((0, N, n_u))

        return nodal_g, nodal_grad_x, nodal_grad_u, cross_g, cross_grad_X, cross_grad_U

    def prepare_fn(state: AlgorithmState, params: dict) -> SubproblemData:
        A_d, B_d, C_d, x_prop, x_prop_plus, D_d, E_d, _, _ = _discretize(state.x, state.u, params)
        (
            nodal_g,
            nodal_grad_x,
            nodal_grad_u,
            cross_g,
            cross_grad_X,
            cross_grad_U,
        ) = _linearize_constraints(state.x, state.u, params)
        return SubproblemData(
            x_bar=state.x,
            u_bar=state.u,
            A_d=A_d,
            B_d=B_d,
            C_d=C_d,
            x_prop=x_prop,
            x_prop_plus=x_prop_plus,
            D_d=D_d,
            E_d=E_d,
            nodal_g=nodal_g,
            nodal_grad_x=nodal_grad_x,
            nodal_grad_u=nodal_grad_u,
            cross_g=cross_g,
            cross_grad_X=cross_grad_X,
            cross_grad_U=cross_grad_U,
            lam_prox=state.lam_prox,
            lam_cost=state.lam_cost,
            lam_vc=state.lam_vc,
            lam_vb_nodal=state.lam_vb_nodal,
            lam_vb_cross=state.lam_vb_cross,
            x_init=state.x_init_pin,
            x_term=state.x_term_pin,
            params=params,
        )

    return prepare_fn


def make_scp_finalize(
    dis_continuous: Callable,
    dis_impulsive: Callable,
    autotuner: "AutotuningBase",
    jax_constraints: "LoweredJaxConstraints",
    settings: "Config",
) -> Callable[
    [AlgorithmState, SubproblemSolution, dict], Tuple[AlgorithmState, IterationDiagnostics]
]:
    """Build the candidate-discretize / metrics / autotune phase of one SCP iteration.

    Returns ``finalize_fn(state, solution, params) -> (next_state, diagnostics)``
    covering steps 6a–6c of the full iteration:

    6a. Discretize the candidate ``(solution.x, solution.u)`` for the
        autotuner's propagation fields and the history's raw matrices.
    6b. Compute the scaled trust-region (``TR``), virtual-control (``VC``),
        and virtual-buffer (``VB``) convergence metrics.
    6c. Fold the candidate through the autotuner to produce the next
        :class:`AlgorithmState`, alongside an :class:`IterationDiagnostics`.

    The returned callable is JIT-able and forms the second of two pieces in the
    split-timing API used by :meth:`PenalizedTrustRegion.step`. The fused
    :func:`make_scp_iteration` delegates to this factory internally.

    Args:
        dis_continuous: Continuous discretization solver,
            ``(x, u, params) -> (A_d, B_d, C_d, x_prop, V)``.
        dis_impulsive: Impulsive discretization solver,
            ``(x_nodes, u, params) -> (x_prop_plus, D_d, E_d, W)``.
        autotuner: Weight-update strategy applied to the candidate.
        jax_constraints: Lowered nodal / cross-node JAX constraints (forwarded
            to the autotuner's ``update_weights``).
        settings: Problem configuration (scaling matrices, boundary types).

    Returns:
        ``finalize_fn(state, solution, params) -> (next_state, diagnostics)``.
    """
    dis_continuous = dis_continuous.call if hasattr(dis_continuous, "call") else dis_continuous
    dis_impulsive = dis_impulsive.call if hasattr(dis_impulsive, "call") else dis_impulsive

    inv_S_x = jnp.asarray(settings.sim.inv_S_x)
    inv_S_u = jnp.asarray(settings.sim.inv_S_u)

    final_type = list(settings.sim.x.final_type)

    init_fixed = jnp.asarray(np.asarray(settings.sim.x.initial_type) == "Fix")
    x_initial = jnp.asarray(np.asarray(settings.sim.x.initial, dtype=float))

    def _discretize(x: jnp.ndarray, u: jnp.ndarray, params: dict):
        A_d, B_d, C_d, x_prop, V = dis_continuous(x, u, params)
        x0_prior = jnp.where(init_fixed, x_initial, x[0])
        x_nodes_prior = jnp.concatenate([x0_prior[None, :], x_prop], axis=0)
        x_prop_plus, D_d, E_d, W = dis_impulsive(x_nodes_prior, u, params)
        return A_d, B_d, C_d, x_prop, x_prop_plus, D_d, E_d, V, W

    def _candidate_cost(x: jnp.ndarray) -> jnp.ndarray:
        cost = jnp.asarray(0.0)
        for i, bc_type in enumerate(final_type):
            if bc_type == "Minimize":
                cost = cost + x[-1, i]
            elif bc_type == "Maximize":
                cost = cost - x[-1, i]
        return cost

    def finalize_fn(
        state: AlgorithmState,
        solution: SubproblemSolution,
        params: dict,
    ) -> Tuple[AlgorithmState, IterationDiagnostics]:
        # 6a. Discretize the candidate for the autotuner's propagation fields
        # and the history's raw discretization matrices.
        _, _, _, cand_x_prop, cand_x_prop_plus, _, _, V_cand, W_cand = _discretize(
            solution.x, solution.u, params
        )
        candidate = CandidateIterate(
            x=solution.x,
            u=solution.u,
            x_prop=cand_x_prop,
            x_prop_plus=cand_x_prop_plus,
            J_lin=solution.cost,
        )

        # 6b. SCP convergence metrics (scaled trust region / virtual control /
        # virtual buffer), matching the legacy ``_subproblem`` reductions.
        tr_x = inv_S_x @ (solution.x - state.x).T
        tr_u = inv_S_u @ (solution.u - state.u).T
        TR = jnp.concatenate([tr_x, tr_u], axis=0)
        VC = jnp.abs(inv_S_x @ solution.nu.T).T
        J_tr = jnp.sum(TR**2)
        J_vc = jnp.sum(VC)
        J_vb = jnp.sum(jnp.maximum(0.0, solution.nu_vb)) + jnp.sum(
            jnp.maximum(0.0, solution.nu_vb_cross)
        )
        state = state.replace(
            J_tr=jnp.asarray(J_tr, dtype=state.J_tr.dtype),
            J_vb=jnp.asarray(J_vb, dtype=state.J_vb.dtype),
            J_vc=jnp.asarray(J_vc, dtype=state.J_vc.dtype),
        )

        # 6c. Autotuner: pure functional update producing the next iterate.
        next_state = autotuner.update_weights(state, candidate, jax_constraints, settings, params)
        next_state = next_state.replace(k=state.k + 1)

        diagnostics = IterationDiagnostics(
            cost=_candidate_cost(solution.x),
            status=solution.status_code,
            J_lin=solution.cost,
            V=V_cand,
            W=W_cand,
            TR=TR,
            VC=VC,
        )
        return next_state, diagnostics

    return finalize_fn


def make_scp_iteration(
    dis_continuous: Callable,
    dis_impulsive: Callable,
    jax_constraints: "LoweredJaxConstraints",
    solver_callback: Callable[[AlgorithmState, SubproblemData], SubproblemSolution],
    autotuner: "AutotuningBase",
    settings: "Config",
) -> Callable[[AlgorithmState, dict], Tuple[AlgorithmState, IterationDiagnostics]]:
    """Build one JAX-pure SCP iteration body.

    The returned ``iteration_fn(state, params)`` performs a single SCP step and
    returns ``(next_state, diagnostics)``. It is the fused mirror of the legacy
    discretize → linearize → solve → autotune pipeline:

    1. Discretize the continuous dynamics about the current iterate
       (``state.x`` / ``state.u``) → ``A_d, B_d, C_d, x_prop``.
    2. Discretize the impulsive/discrete dynamics about the propagated nodes →
       ``x_prop_plus, D_d, E_d``.
    3. Evaluate every nodal / cross-node constraint and its gradients.
    4. Pack the linearization, penalty weights, and boundary pins into a
       :class:`SubproblemData`.
    5. Hand it to ``solver_callback`` (the backend's ``iteration_callback``),
       which returns a :class:`SubproblemSolution`.
    6. Compute the ``J_tr / J_vb / J_vc`` metrics and fold the candidate through
       the autotuner to produce the next state, alongside an
       :class:`IterationDiagnostics`.

    This function is a thin combiner around :func:`make_scp_prepare` (steps 1–4)
    and :func:`make_scp_finalize` (steps 6a–6c). The two sub-factories are also
    exposed directly for the split-timing API in :meth:`PenalizedTrustRegion.step`.

    The current-iterate discretization (steps 1–2) is recomputed every call
    rather than read from a carried ``state.x_prop``: the candidate is
    re-discretized next iteration instead of carrying its discretization on
    :class:`AlgorithmState`. This trades roughly 2× discretization work per
    iteration for a smaller loop carry and a simpler accept/reject rule in the
    autotuner — the next iterate is a pure function of ``state.x`` / ``state.u``,
    so acceptance copies only the trajectory, never a discretization. The
    discretization solvers are built by the caller (``Problem``) and
    captured here so the caching policy stays in its own layer; the constraint
    linearizers, per-backend solver callback, autotuner, and settings are
    likewise closure constants.

    Args:
        dis_continuous: Continuous discretization solver,
            ``(x, u, params) -> (A_d, B_d, C_d, x_prop, V)``. A ``jax.jit``
            callable (vmappable) or a ``jax.export`` wrapper (``.call``).
        dis_impulsive: Impulsive discretization solver,
            ``(x_nodes, u, params) -> (x_prop_plus, D_d, E_d, W)``.
        jax_constraints: Lowered nodal / cross-node JAX constraints.
        solver_callback: ``(state, SubproblemData) -> SubproblemSolution`` from
            the convex backend's :meth:`iteration_callback`.
        autotuner: Weight-update strategy applied to the candidate.
        settings: Problem configuration (scaling matrices, boundary types).

    Returns:
        ``iteration_fn(state, params) -> (next_state, diagnostics)``.
    """
    prepare_fn = make_scp_prepare(dis_continuous, dis_impulsive, jax_constraints, settings)
    finalize_fn = make_scp_finalize(
        dis_continuous, dis_impulsive, autotuner, jax_constraints, settings
    )

    def iteration_fn(
        state: AlgorithmState, params: dict
    ) -> Tuple[AlgorithmState, IterationDiagnostics]:
        # Steps 1–4: discretize, linearize constraints, pack SubproblemData.
        data = prepare_fn(state, params)

        # Step 5: solve the convex subproblem.
        solution = solver_callback(state, data)

        # Steps 6a–6c: candidate discretization, metrics, autotune.
        return finalize_fn(state, solution, params)

    return iteration_fn
