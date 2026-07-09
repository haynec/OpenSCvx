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
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from openscvx.solvers.ptr_solver import ProxConvexSubproblemData, SubproblemData, SubproblemSolution

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
    large and host-only; cost / status / ``J_cvx`` come from the subproblem
    solve). :func:`make_solve_loop` projects them away.

    Attributes:
        cost: Boundary-weighted objective at the candidate (scalar) — the
            ``cost[-1]`` the emitter prints.
        status: Subproblem :class:`StatusCode` as ``int32``.
        J_cvx: Subproblem optimal (linearized) cost (scalar).
        V: Raw continuous discretization matrix of the candidate.
        W: Raw impulsive discretization matrix of the candidate.
        TR: Scaled trust-region step matrix, shape ``(n_x + n_u, N)``.
        VC: Scaled virtual-control matrix, shape ``(N-1, n_x)``.
    """

    cost: jnp.ndarray
    status: jnp.ndarray
    J_cvx: jnp.ndarray
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


def _discretize(
    x: jnp.ndarray,
    u: jnp.ndarray,
    params: dict,
    dis_continuous,
    dis_impulsive,
    init_fixed: jnp.ndarray,
    x_initial: jnp.ndarray,
):
    A_d, B_d, C_d, x_prop, V = dis_continuous(x, u, params)
    x0_prior = jnp.where(init_fixed, x_initial, x[0])
    x_nodes_prior = jnp.concatenate([x0_prior[None, :], x_prop], axis=0)
    x_prop_plus, D_d, E_d, W = dis_impulsive(x_nodes_prior, u, params)
    return A_d, B_d, C_d, x_prop, x_prop_plus, D_d, E_d, V, W


def _linearize_constraints(
    x: jnp.ndarray,
    u: jnp.ndarray,
    params: dict,
    jax_constraints,
    N: int,
    n_nodal: int,
    n_x: int,
    n_u: int,
):
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
        cross_g = jnp.stack([jnp.asarray(c.func(x, u, params)) for c in jax_constraints.cross_node])
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


def _constraint_curvature(
    x: jnp.ndarray,
    u: jnp.ndarray,
    params: dict,
    jax_constraints,
    lam_vb_nodal: jnp.ndarray,
    lam_vb_cross: jnp.ndarray,
    N: int,
    n_x: int,
    n_u: int,
) -> jnp.ndarray:
    """Curvature of the nonconvex-constraint penalties ``Σ_i y_i ∇²g_i``.

    The ``h(C(x,u))`` exact-penalty term includes ``Σ_i w_i max(0, g_i(x,u))``
    over the nonconvex constraints (penalty.py).  Its inner-only curvature weights
    each constraint Hessian by the hinge subgradient ``y_i = w_i · 𝟙[g_i > 0]`` —
    the minimum-norm selection (``0`` at the kink ``g_i = 0``).  The Hessian is
    taken over the **full** stacked ``[x, u]`` variable: nodal constraints add
    per-node ``∇²_xx / ∇²_uu / ∂²/∂x∂u`` blocks; cross-node constraints add their
    full trajectory blocks.  Returns the **unprojected** ``(D, D)`` matrix with
    ``D = N*n_x + N*n_u`` ordered ``[x.flatten(); u.flatten()]``.
    """
    Nx = N * n_x
    Nu = N * n_u
    D = Nx + Nu
    H = jnp.zeros((D, D))

    for c_idx, constraint in enumerate(jax_constraints.nodal):
        if constraint.hess_g_xx is None:
            continue
        g = jnp.asarray(constraint.func(x, u, 0, params)).reshape(N, -1)  # (N, m)
        m = g.shape[1]
        h_xx = jnp.asarray(constraint.hess_g_xx(x, u, 0, params)).reshape(N, m, n_x, n_x)
        h_uu = jnp.asarray(constraint.hess_g_uu(x, u, 0, params)).reshape(N, m, n_u, n_u)
        h_xu = jnp.asarray(constraint.hess_g_xu(x, u, 0, params)).reshape(N, m, n_x, n_u)
        # Hinge subgradient per (node, component), with the per-node penalty weight.
        y = lam_vb_nodal[:, c_idx][:, None] * (g > 0)  # (N, m)
        if constraint.nodes is not None:
            mask = jnp.zeros((N, 1)).at[jnp.asarray(constraint.nodes)].set(1.0)
            y = y * mask
        b_xx = jnp.einsum("nm,nmab->nab", y, h_xx)  # (N, n_x, n_x)
        b_uu = jnp.einsum("nm,nmab->nab", y, h_uu)  # (N, n_u, n_u)
        b_xu = jnp.einsum("nm,nmab->nab", y, h_xu)  # (N, n_x, n_u)
        for n in range(N):
            xs = n * n_x
            us = Nx + n * n_u
            H = H.at[xs : xs + n_x, xs : xs + n_x].add(b_xx[n])
            H = H.at[us : us + n_u, us : us + n_u].add(b_uu[n])
            H = H.at[xs : xs + n_x, us : us + n_u].add(b_xu[n])
            H = H.at[us : us + n_u, xs : xs + n_x].add(b_xu[n].T)

    for c_idx, constraint in enumerate(jax_constraints.cross_node):
        if getattr(constraint, "hess_g_XX", None) is None:
            continue
        g = jnp.asarray(constraint.func(x, u, params))  # scalar
        y = lam_vb_cross[c_idx] * (g > 0)
        h_XX = jnp.asarray(constraint.hess_g_XX(x, u, params)).reshape(Nx, Nx)
        h_UU = jnp.asarray(constraint.hess_g_UU(x, u, params)).reshape(Nu, Nu)
        h_XU = jnp.asarray(constraint.hess_g_XU(x, u, params)).reshape(Nx, Nu)
        H = H.at[:Nx, :Nx].add(y * h_XX)
        H = H.at[Nx:, Nx:].add(y * h_UU)
        H = H.at[:Nx, Nx:].add(y * h_XU)
        H = H.at[Nx:, :Nx].add(y * h_XU.T)

    return H


def _candidate_cost(x: jnp.ndarray, final_type: list) -> jnp.ndarray:
    cost = jnp.asarray(0.0)
    for i, bc_type in enumerate(final_type):
        if bc_type == "Minimize":
            cost = cost + x[-1, i]
        elif bc_type == "Maximize":
            cost = cost - x[-1, i]
    return cost


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
    N = settings.sim.n
    n_x = settings.sim.n_states
    n_u = settings.sim.n_controls
    n_nodal = len(jax_constraints.nodal)

    # Equality columns measure violation two-sided (|nu_vb|); inequalities use
    # the positive part. Built once at trace time as static boolean masks.
    nodal_eq_mask = jnp.asarray([c.is_equality for c in jax_constraints.nodal], dtype=bool)
    cross_eq_mask = jnp.asarray([c.is_equality for c in jax_constraints.cross_node], dtype=bool)

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

    def iteration_fn(
        state: AlgorithmState, params: dict
    ) -> Tuple[AlgorithmState, IterationDiagnostics]:
        # 1–2. Discretize the current iterate for the subproblem linearization.
        A_d, B_d, C_d, x_prop, x_prop_plus, D_d, E_d, _, _ = _discretize(
            state.x, state.u, params, dis_continuous, dis_impulsive, init_fixed, x_initial
        )

        # 3. Linearize the constraints about the current iterate.
        (
            nodal_g,
            nodal_grad_x,
            nodal_grad_u,
            cross_g,
            cross_grad_X,
            cross_grad_U,
        ) = _linearize_constraints(state.x, state.u, params, jax_constraints, N, n_nodal, n_x, n_u)

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
            params=params,
        )

        # 5. Solve the convex subproblem.
        solution = solver_callback(state, data)

        # 6a. Discretize the candidate for the autotuner's propagation fields
        # and the history's raw discretization matrices.
        _, _, _, cand_x_prop, cand_x_prop_plus, _, _, V_cand, W_cand = _discretize(
            solution.x, solution.u, params, dis_continuous, dis_impulsive, init_fixed, x_initial
        )
        candidate = CandidateIterate(
            x=solution.x,
            u=solution.u,
            x_prop=cand_x_prop,
            x_prop_plus=cand_x_prop_plus,
            J_cvx=solution.cost,
        )

        # 6b. SCP convergence metrics (scaled trust region / virtual control /
        # virtual buffer), matching the legacy ``_subproblem`` reductions.
        tr_x = inv_S_x @ (solution.x - state.x).T
        tr_u = inv_S_u @ (solution.u - state.u).T
        TR = jnp.concatenate([tr_x, tr_u], axis=0)
        VC = jnp.abs(inv_S_x @ solution.nu.T).T
        J_tr = jnp.sum(TR**2)
        J_vc = jnp.sum(VC)
        nodal_vb = jnp.where(
            nodal_eq_mask, jnp.abs(solution.nu_vb), jnp.maximum(0.0, solution.nu_vb)
        )
        cross_vb = jnp.where(
            cross_eq_mask, jnp.abs(solution.nu_vb_cross), jnp.maximum(0.0, solution.nu_vb_cross)
        )
        J_vb = jnp.sum(nodal_vb) + jnp.sum(cross_vb)
        state = state.replace(
            J_tr=jnp.asarray(J_tr, dtype=state.J_tr.dtype),
            J_vb=jnp.asarray(J_vb, dtype=state.J_vb.dtype),
            J_vc=jnp.asarray(J_vc, dtype=state.J_vc.dtype),
        )

        # 6c. Autotuner: pure functional update producing the next iterate.
        next_state = autotuner.update_weights(state, candidate, jax_constraints, settings, params)
        next_state = next_state.replace(k=state.k + 1)

        diagnostics = IterationDiagnostics(
            cost=_candidate_cost(solution.x, final_type),
            status=solution.status_code,
            J_cvx=solution.cost,
            V=V_cand,
            W=W_cand,
            TR=TR,
            VC=VC,
        )
        return next_state, diagnostics

    return iteration_fn


def make_proxconvex_iteration(
    composite,
    dis_continuous: Callable,
    dis_impulsive: Callable,
    jax_constraints: "LoweredJaxConstraints",
    solver_callback: Callable[[AlgorithmState, ProxConvexSubproblemData], "SubproblemSolution"],
    autotuner: "AutotuningBase",
    settings: "Config",
    dis_hessian: Callable = None,
    use_hessian_constraints: bool = False,
) -> Callable[[AlgorithmState, dict], Tuple[AlgorithmState, IterationDiagnostics]]:
    """Build one JAX-pure ProxConvex iteration body.

    Mirrors :func:`make_scp_iteration` exactly, except it also evaluates
    the SR composite (``composite.eval``) and packs a
    :class:`~openscvx.solvers.ptr_solver.ProxConvexSubproblemData` so the
    solver callback can branch on the sign of ``∇s(R(x_k))``.

    Args:
        composite: :class:`~openscvx.algorithms.scvx.prox_convex.SRComposite`
            instance.  Its ``eval`` method is called inside the JAX trace.
        dis_continuous, dis_impulsive, jax_constraints, solver_callback,
            autotuner, settings: same as :func:`make_scp_iteration`.

    Returns:
        ``iteration_fn(state, params) -> (next_state, diagnostics)``.
    """
    from .prox_convex import compute_h_plus

    N = settings.sim.n
    n_x = settings.sim.n_states
    n_u = settings.sim.n_controls
    n_nodal = len(jax_constraints.nodal)
    # Full [x, u] proximal-metric dimension, ordered [x.flatten(); u.flatten()].
    D = N * (n_x + n_u)

    dis_continuous = dis_continuous.call if hasattr(dis_continuous, "call") else dis_continuous
    dis_impulsive = dis_impulsive.call if hasattr(dis_impulsive, "call") else dis_impulsive
    if dis_hessian is not None and hasattr(dis_hessian, "call"):
        dis_hessian = dis_hessian.call

    # Static gate: the curvature block is built iff the s(R) block or the h(C)
    # (dynamics + nonconvex-constraint) block is active.  Mirrors the solver's
    # ``hess_cost`` gate so the iteration and subproblem agree on H⁺_k presence.
    any_hessian = (composite.use_hessian is not False) or use_hessian_constraints

    inv_S_x = jnp.asarray(settings.sim.inv_S_x)
    inv_S_u = jnp.asarray(settings.sim.inv_S_u)
    final_type = list(settings.sim.x.final_type)
    init_fixed = jnp.asarray(np.asarray(settings.sim.x.initial_type) == "Fix")
    x_initial = jnp.asarray(np.asarray(settings.sim.x.initial, dtype=float))

    def iteration_fn(
        state: AlgorithmState, params: dict
    ) -> Tuple[AlgorithmState, IterationDiagnostics]:
        A_d, B_d, C_d, x_prop, x_prop_plus, D_d, E_d, _, _ = _discretize(
            state.x, state.u, params, dis_continuous, dis_impulsive, init_fixed, x_initial
        )

        (
            nodal_g,
            nodal_grad_x,
            nodal_grad_u,
            cross_g,
            cross_grad_X,
            cross_grad_U,
        ) = _linearize_constraints(state.x, state.u, params, jax_constraints, N, n_nodal, n_x, n_u)

        # Evaluate the SR composite: R(x_k), ∇s(R), sign mask, ∇R.
        R_val, ds_val, I_neg_mask, grad_R = composite.eval(state.x, state.u, params)

        # Curvature augmentation: H⁺_k = Π_{S+}(H_{C,k} + H_{s,k}) for the
        # proximal metric Q_k = µ_k I + H⁺_k.  The s(R) block H_{s,k} and the
        # h(C) block H_{C,k} (dynamics defects + nonconvex constraints) are
        # summed and PSD-projected once here.
        H_s_raw = composite.compute_hessian_s_raw(
            state.x, state.u, params, R_val, ds_val, I_neg_mask, grad_R
        )

        H_c_raw = jnp.zeros((D, D))
        if use_hessian_constraints:
            # Dynamics-defect curvature Σ_j y_j ∇²C_j over [x, u], with the L1
            # virtual-control subgradient on the propagated state C = x_prop:
            #   y[k] = − inv_S_xᵀ (lam_vc[k] ⊙ sign(inv_S_x (x_bar[k+1] − x_prop[k]))).
            if dis_hessian is not None:
                d_scaled = (inv_S_x @ (state.x[1:] - x_prop).T).T  # (N-1, n_x)
                sub = state.lam_vc * jnp.sign(d_scaled)  # (N-1, n_x), broadcast lam_vc
                w_vc = -(inv_S_x.T @ sub.T).T  # (N-1, n_x)
                H_c_raw = H_c_raw + dis_hessian(state.x, state.u, w_vc, params)
            # Nonconvex-constraint curvature Σ_i y_i ∇²g_i over [x, u].
            H_c_raw = H_c_raw + _constraint_curvature(
                state.x,
                state.u,
                params,
                jax_constraints,
                state.lam_vb_nodal,
                state.lam_vb_cross,
                N,
                n_x,
                n_u,
            )

        if any_hessian:
            H_plus = compute_h_plus(H_s_raw + H_c_raw)
        else:
            # Sentinel: no curvature → solver keeps Q_k = µ_k I (no hess_cost).
            H_plus = jnp.zeros(())

        data = ProxConvexSubproblemData(
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
            R_val=R_val,
            ds_val=ds_val,
            I_neg_mask=I_neg_mask,
            grad_R=grad_R,
            H_plus=H_plus,
        )

        solution = solver_callback(state, data)

        _, _, _, cand_x_prop, cand_x_prop_plus, _, _, V_cand, W_cand = _discretize(
            solution.x, solution.u, params, dis_continuous, dis_impulsive, init_fixed, x_initial
        )
        candidate = CandidateIterate(
            x=solution.x,
            u=solution.u,
            x_prop=cand_x_prop,
            x_prop_plus=cand_x_prop_plus,
            J_cvx=solution.cost,
        )

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

        next_state = autotuner.update_weights(state, candidate, jax_constraints, settings, params)
        next_state = next_state.replace(k=state.k + 1)

        diagnostics = IterationDiagnostics(
            cost=_candidate_cost(solution.x, final_type),
            status=solution.status_code,
            J_cvx=solution.cost,
            V=V_cand,
            W=W_cand,
            TR=TR,
            VC=VC,
        )
        return next_state, diagnostics

    return iteration_fn
