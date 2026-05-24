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

:func:`make_solve_loop` wraps that body in a ``lax.while_loop`` keyed on the SCP
convergence metrics, projecting the diagnostics away so the loop carry stays
``state -> state``. It exists as a primitive for tests and for the JAX-pure
``.solve()`` path in follow-up work; today's ``Problem.solve()`` still drives the
body from a Python ``while`` loop, so behavior is unchanged.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from openscvx.solvers.ptr_solver import SubproblemData, SubproblemSolution

from ..base import AlgorithmState, CandidateIterate

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered import LoweredJaxConstraints

    from ..base import AutotuningBase


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

    The current-iterate discretization (steps 1–2) is recomputed every call
    rather than read from a carried ``state.x_prop``; see the plan's decision
    log. The discretization solvers are built by the caller (``Problem``) and
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
    N = settings.sim.n
    n_x = settings.sim.n_states
    n_u = settings.sim.n_controls
    n_nodal = len(jax_constraints.nodal)

    # Accept either a bare ``jax.jit`` callable or a ``jax.export`` wrapper.
    dis_continuous = dis_continuous.call if hasattr(dis_continuous, "call") else dis_continuous
    dis_impulsive = dis_impulsive.call if hasattr(dis_impulsive, "call") else dis_impulsive

    # Scaling matrices for the SCP convergence / diagnostic metrics (static).
    inv_S_x = jnp.asarray(settings.sim.inv_S_x)
    inv_S_u = jnp.asarray(settings.sim.inv_S_u)

    # Boundary types drive the candidate cost reduction (static, host-side).
    final_type = list(settings.sim.x.final_type)

    # Node-0 prior recovery: fixed initial entries come from the boundary
    # condition, free entries from the iterate. The impulsive discretization
    # linearizes about the propagated nodes prefixed with this prior.
    init_fixed = jnp.asarray(np.asarray(settings.sim.x.initial_type) == "Fix")
    x_initial = jnp.asarray(np.asarray(settings.sim.x.initial, dtype=float))

    def _discretize(x: jnp.ndarray, u: jnp.ndarray, params: dict):
        """Discretize ``(x, u)`` through both dynamics; return propagation pieces.

        Returns ``(A_d, B_d, C_d, x_prop, x_prop_plus, D_d, E_d, V, W)`` — ``V``
        / ``W`` are the raw multi-shot matrices the history recorder unpacks.
        """
        A_d, B_d, C_d, x_prop, V = dis_continuous(x, u, params)
        x0_prior = jnp.where(init_fixed, x_initial, x[0])
        x_nodes_prior = jnp.concatenate([x0_prior[None, :], x_prop], axis=0)
        x_prop_plus, D_d, E_d, W = dis_impulsive(x_nodes_prior, u, params)
        return A_d, B_d, C_d, x_prop, x_prop_plus, D_d, E_d, V, W

    def _linearize_constraints(x: jnp.ndarray, u: jnp.ndarray, params: dict):
        """Stack nodal / cross-node constraint values and gradients.

        Nodal data is laid out ``(N, n_nodal[, ·])`` with zero-fill at nodes
        outside each constraint's static ``nodes`` set — the backend assembly
        closes over those node sets and skips the zeros.
        """
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

    def _candidate_cost(x: jnp.ndarray) -> jnp.ndarray:
        """Boundary-weighted reduction objective at the candidate's terminal node."""
        cost = jnp.asarray(0.0)
        for i, bc_type in enumerate(final_type):
            if bc_type == "Minimize":
                cost = cost + x[-1, i]
            elif bc_type == "Maximize":
                cost = cost - x[-1, i]
        return cost

    def iteration_fn(
        state: AlgorithmState, params: dict
    ) -> Tuple[AlgorithmState, IterationDiagnostics]:
        # 1–2. Discretize the current iterate for the subproblem linearization.
        A_d, B_d, C_d, x_prop, x_prop_plus, D_d, E_d, _, _ = _discretize(state.x, state.u, params)

        # 3. Linearize the constraints about the current iterate.
        (
            nodal_g,
            nodal_grad_x,
            nodal_grad_u,
            cross_g,
            cross_grad_X,
            cross_grad_U,
        ) = _linearize_constraints(state.x, state.u, params)

        # 4. Pack the subproblem inputs.
        data = SubproblemData(
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
        )

        # 5. Solve the convex subproblem.
        solution = solver_callback(state, data)

        # 6a. Discretize the candidate for the autotuner's propagation fields
        # and the history's raw discretization matrices.
        _, _, _, cand_x_prop, cand_x_prop_plus, _, _, V_cand, W_cand = _discretize(
            solution.x, solution.u, params
        )
        candidate = CandidateIterate()
        candidate.x = solution.x
        candidate.u = solution.u
        candidate.x_prop = cand_x_prop
        candidate.x_prop_plus = cand_x_prop_plus
        candidate.J_lin = solution.cost

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

    return iteration_fn


def _converged(state: AlgorithmState, ep_tr: float, ep_vb: float, ep_vc: float) -> jnp.ndarray:
    """Boolean SCP convergence test from the metrics on ``state``."""
    return (state.J_tr < ep_tr) & (state.J_vb < ep_vb) & (state.J_vc < ep_vc)


def make_solve_loop(
    iteration_fn: Callable[[AlgorithmState, dict], Tuple[AlgorithmState, IterationDiagnostics]],
    ep_tr: float,
    ep_vb: float,
    ep_vc: float,
    k_max: int,
) -> Callable[[AlgorithmState, dict], AlgorithmState]:
    """Wrap ``iteration_fn`` in a ``lax.while_loop`` keyed on convergence.

    The loop runs ``iteration_fn`` until either the SCP metrics fall below the
    ``ep_*`` thresholds or the iteration counter ``state.k`` exceeds ``k_max`` —
    matching the Python ``while`` loop in ``Problem.solve()``. The per-iteration
    :class:`IterationDiagnostics` are projected away so the loop carry stays
    ``state -> state`` (XLA dead-code-eliminates their host-only pieces). It
    exists as a primitive for tests and the future JAX-pure ``.solve()`` path;
    the public ``Problem.solve()`` continues to drive ``iteration_fn`` from
    Python.

    Args:
        iteration_fn: A body built by :func:`make_scp_iteration`.
        ep_tr: Convergence threshold on ``J_tr`` (trust-region step).
        ep_vb: Convergence threshold on ``J_vb`` (virtual buffer).
        ep_vc: Convergence threshold on ``J_vc`` (virtual control).
        k_max: Maximum number of SCP iterations.

    Returns:
        ``solve_loop(state, params) -> final_state``.
    """

    def solve_loop(state: AlgorithmState, params: dict) -> AlgorithmState:
        def cond(state: AlgorithmState) -> jnp.ndarray:
            return (state.k <= k_max) & jnp.logical_not(_converged(state, ep_tr, ep_vb, ep_vc))

        def body(state: AlgorithmState) -> AlgorithmState:
            next_state, _ = iteration_fn(state, params)
            return next_state

        return jax.lax.while_loop(cond, body, state)

    return solve_loop
