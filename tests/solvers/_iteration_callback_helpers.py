"""Shared fixtures for ``iteration_callback`` tests across backends.

The brachistochrone problem and the ``SubproblemData`` reconstruction step
were previously copy-pasted across four test modules; this module owns the
canonical definitions. Per-backend test files import from here.

* :func:`build_brachistochrone` parametrizes over the solver backend and
  optional constraint style (CTCS vs. plain nodal), matching every variant
  the existing tests need.
* :func:`subproblem_data_from_numpy_stash` reads the JAX-pure
  :class:`SubproblemData` back from the QPAX/Moreau internal stash
  (``solver._dyn`` / ``solver._cons`` / ``solver._pen``). CVXPy reads the
  same iterate from its CVXPy ``Parameter`` values instead, so its variant
  lives in ``test_iteration_callback_cvxpy.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from openscvx import Problem
from openscvx.discretization import get_impulsive_discretization_solver
from openscvx.solvers.ptr_solver import SubproblemData


def build_brachistochrone(
    backend: str,
    n: int = 4,
    k_max: int = 1,
    constraint_style: str = "ctcs",
    autotuner="ConstantProximalWeight",
):
    """Build the brachistochrone problem at small ``N`` for callback tests.

    Mirrors ``examples/abstract/brachistochrone.py`` but rebuilds the symbolic
    state in-test so the cached problem from the example doesn't carry over.

    Args:
        backend: ``"qpax"`` / ``"moreau"`` / ``"cvxpy"`` — passed straight
            through to ``solver={"backend": ...}``.
        n: Number of discretization nodes. Small values keep the assembly
            matrices small enough to compare element-wise.
        k_max: SCP iteration cap. One iteration is enough to populate the
            solver's per-iteration stash for parity tests.
        constraint_style: ``"ctcs"`` (CTCS box rows; the default) or
            ``"nodal"`` (plain nodal inequalities; exercises the
            nodal-constraint assembly block that's dormant under CTCS-only).
        autotuner: Autotuner name or instance, passed straight through to the
            algorithm config (instances let tests exercise custom autotuners).
    """
    import openscvx as ox

    g = 9.81

    position = ox.State("position", shape=(2,))
    position.max = np.array([10.0, 10.0])
    position.min = np.array([0.0, 0.0])
    position.initial = np.array([0.0, 10.0])
    position.final = [10.0, 5.0]

    velocity = ox.State("velocity", shape=(1,))
    velocity.max = np.array([10.0])
    velocity.min = np.array([0.0])
    velocity.initial = np.array([0.0])
    velocity.final = [("free", 10.0)]

    theta = ox.Control("theta", shape=(1,))
    theta.max = np.array([100.5 * jnp.pi / 180])
    theta.min = np.array([0.0])
    theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    states = [position, velocity]
    controls = [theta]

    dynamics = {
        "position": ox.Concat(
            velocity[0] * ox.Sin(theta[0]),
            -velocity[0] * ox.Cos(theta[0]),
        ),
        "velocity": g * ox.Cos(theta[0]),
    }

    constraint_exprs = []
    if constraint_style == "ctcs":
        for state in states:
            constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
    elif constraint_style == "nodal":
        for state in states:
            constraint_exprs.extend([state <= state.max, state.min <= state])
    else:
        raise ValueError(f"unknown constraint_style {constraint_style!r}")

    time = ox.Time(
        initial=0.0,
        final=("minimize", 2.0),
        min=0.0,
        max=2.0,
        uniform_time_grid=True,
    )

    prob = Problem(
        dynamics=dynamics,
        states=states,
        controls=controls,
        time=time,
        constraints=constraint_exprs,
        N=n,
        float_dtype="float64",
        algorithm={
            "autotuner": autotuner,
            "lam_prox": 1e0,
            "lam_cost": 6e-1,
            "k_max": k_max,
        },
        solver={"backend": backend},
    )
    prob.settings.dev.printing = False
    return prob


def populate_numpy_stash(prob) -> None:
    """Run one discretize → linearize → ``update_*`` cycle on the initial iterate.

    The SCP loop no longer drives the solver's NumPy ``update_*`` path (it uses
    the JAX-pure ``iteration_callback``), so ``prob.solve()`` no longer leaves
    ``solver._dyn`` / ``_cons`` / ``_pen`` populated. Tests that read that stash
    (or call the legacy ``solver.solve()`` / ``_assemble_qp``) call this after
    ``initialize()`` to set it up, mirroring what the former
    ``PenalizedTrustRegion._subproblem`` did for the first iterate.
    """
    solver = prob.solver
    settings = prob.settings
    params = prob._parameters
    state = prob.state
    jc = prob._compiled_constraints

    dis = jax.jit(prob.discretizer.get_solver(prob.lowered.dynamics, settings))
    dis_imp = jax.jit(get_impulsive_discretization_solver(prob.lowered.dynamics_discrete))

    x_bar = np.asarray(state.x)
    u_bar = np.asarray(state.u, dtype=float)
    A_d, B_d, C_d, x_prop, _ = dis(x_bar, u_bar, params)

    init_fixed = np.asarray(settings.sim.x.initial_type) == "Fix"
    x0_init = np.asarray(settings.sim.x.initial, dtype=float)
    x0_prior = np.where(init_fixed, x0_init, x_bar[0]).reshape(1, -1)
    x_nodes_prior = np.vstack((x0_prior, np.asarray(x_prop)))
    x_prop_plus, D_d, E_d, _ = dis_imp(x_nodes_prior, u_bar, params)

    slice_imp = settings.sim.u.slice_impulsive
    has_impulsive = bool(slice_imp.stop > slice_imp.start)

    solver.update_boundary_conditions(x_init=settings.sim.x.initial, x_term=settings.sim.x.final)
    solver.update_dynamics_linearization(
        x_bar=x_bar,
        u_bar=u_bar,
        A_d=np.asarray(A_d),
        B_d=np.asarray(B_d),
        C_d=np.asarray(C_d),
        x_prop=np.asarray(x_prop),
        x_prop_plus=np.asarray(x_prop_plus) if has_impulsive else None,
        D_d=np.asarray(D_d) if has_impulsive else None,
        E_d=np.asarray(E_d) if has_impulsive else None,
    )

    nodal = []
    for c in jc.nodal:
        g = np.squeeze(np.asarray(c.func(x_bar, u_bar, 0, params)))
        gx = np.asarray(c.grad_g_x(x_bar, u_bar, 0, params))
        gu = np.asarray(c.grad_g_u(x_bar, u_bar, 0, params))
        if g.ndim == 0:
            g = np.broadcast_to(g, (x_bar.shape[0],))
        elif g.ndim > 1:
            g = g.reshape(g.shape[0], -1).sum(axis=1)
        if gx.ndim == 1:
            gx = np.broadcast_to(gx, (x_bar.shape[0], gx.shape[0]))
        elif gx.ndim > 2:
            gx = gx.reshape(gx.shape[0], -1)[:, : x_bar.shape[1]]
        if gu.ndim == 1:
            gu = np.broadcast_to(gu, (u_bar.shape[0], gu.shape[0]))
        elif gu.ndim > 2:
            gu = gu.reshape(gu.shape[0], -1)[:, : u_bar.shape[1]]
        nodal.append({"g": g, "grad_g_x": gx, "grad_g_u": gu})

    cross = []
    for c in jc.cross_node:
        cross.append(
            {
                "g": np.asarray(c.func(x_bar, u_bar, params)),
                "grad_g_X": np.asarray(c.grad_g_X(x_bar, u_bar, params)),
                "grad_g_U": np.asarray(c.grad_g_U(x_bar, u_bar, params)),
            }
        )
    solver.update_constraint_linearizations(nodal=nodal or None, cross_node=cross or None)
    solver.update_penalties(
        lam_prox=np.asarray(state.lam_prox),
        lam_cost=np.asarray(state.lam_cost),
        lam_vc=np.asarray(state.lam_vc),
        lam_vb_nodal=np.asarray(state.lam_vb_nodal),
        lam_vb_cross=np.asarray(state.lam_vb_cross),
    )
    solver.update_proximal_terms()


def subproblem_data_from_numpy_stash(solver) -> SubproblemData:
    """Reconstruct :class:`SubproblemData` from QPAX/Moreau's NumPy stash.

    After one ``update_*`` cycle the solver has every per-iteration array on
    hand (``solver._dyn`` / ``solver._cons`` / ``solver._pen``). This helper
    rebuilds the stacked-array layout the JAX callback expects so both
    assembly paths see the same iterate.

    The NumPy path absorbs ``D_d`` into ``A_d`` / ``B_d`` / ``C_d`` at update
    time, so we pass a zero ``D_d`` here — the JAX path's ``has_impulsive``
    branch is statically False for brachistochrone and the absorption einsum
    is skipped.

    CVXPy stores the same data on its CVXPy parameters and has its own reader
    that lives in ``test_iteration_callback_cvxpy.py``.
    """
    L = solver.layout
    N, n_x, n_u = L.N, L.n_x, L.n_u

    dyn = solver._dyn
    cons = solver._cons
    pen = solver._pen
    n_nodal = L.n_nodal

    # Stack nodal linearizations into (N, n_nodal*) layout with zero-fill for
    # nodes outside each constraint's static ``nodes`` tuple.
    nodal_g = np.zeros((N, max(n_nodal, 1)), dtype=float)
    nodal_grad_x = np.zeros((N, max(n_nodal, 1), n_x), dtype=float)
    nodal_grad_u = np.zeros((N, max(n_nodal, 1), n_u), dtype=float)
    for c_idx, (constraint, entry) in enumerate(
        zip(solver._jax_constraints.nodal, cons.get("nodal", []))
    ):
        for node in constraint.nodes:
            nodal_g[node, c_idx] = entry["g"][node]
            nodal_grad_x[node, c_idx] = entry["grad_g_x"][node]
            nodal_grad_u[node, c_idx] = entry["grad_g_u"][node]
    if n_nodal == 0:
        nodal_g = np.zeros((N, 0))
        nodal_grad_x = np.zeros((N, 0, n_x))
        nodal_grad_u = np.zeros((N, 0, n_u))

    x_prop_plus = dyn["x_prop_plus"] if dyn["x_prop_plus"] is not None else np.zeros((N, n_x))
    E_d = dyn["E_d"] if dyn["E_d"] is not None else np.zeros((N, n_x, n_u))
    D_d = np.zeros((N, n_x, n_x))

    x_init = solver._x_init if solver._x_init is not None else np.full(n_x, np.nan)
    x_term = solver._x_term if solver._x_term is not None else np.full(n_x, np.nan)

    return SubproblemData(
        x_bar=jnp.asarray(dyn["x_bar"]),
        u_bar=jnp.asarray(dyn["u_bar"]),
        A_d=jnp.asarray(dyn["A_d"]),
        B_d=jnp.asarray(dyn["B_d"]),
        C_d=jnp.asarray(dyn["C_d"]),
        x_prop=jnp.asarray(dyn["x_prop"]),
        x_prop_plus=jnp.asarray(x_prop_plus),
        D_d=jnp.asarray(D_d),
        E_d=jnp.asarray(E_d),
        nodal_g=jnp.asarray(nodal_g),
        nodal_grad_x=jnp.asarray(nodal_grad_x),
        nodal_grad_u=jnp.asarray(nodal_grad_u),
        cross_g=jnp.zeros((0,)),
        cross_grad_X=jnp.zeros((0, N, n_x)),
        cross_grad_U=jnp.zeros((0, N, n_u)),
        lam_prox=jnp.asarray(pen["lam_prox"]),
        lam_cost=jnp.asarray(pen["lam_cost"]),
        lam_vc=jnp.asarray(pen["lam_vc"]),
        lam_vb_nodal=jnp.asarray(pen["lam_vb_nodal"]),
        lam_vb_cross=jnp.zeros((0,)),
        x_init=jnp.asarray(x_init),
        x_term=jnp.asarray(x_term),
        params=solver._parameters_dict,
    )
