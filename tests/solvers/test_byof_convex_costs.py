"""Tests for BYOF convex cost terms (CVXPyPTRSolver only)."""

import cvxpy as cp
import jax.numpy as jnp
import numpy as np
import pytest

import openscvx as ox
from openscvx import ByofSpec, Problem


def _make_brachistochrone(n=6, byof=None, lam_prox=1e0, backend="cvxpy"):
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
    theta.max = np.array([179.0 * jnp.pi / 180])
    theta.min = np.array([0.0])
    theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

    states = [position, velocity]
    dynamics = {
        "position": ox.Concat(
            velocity[0] * ox.Sin(theta[0]),
            -velocity[0] * ox.Cos(theta[0]),
        ),
        "velocity": g * ox.Cos(theta[0]),
    }
    constraint_exprs = []
    for state in states:
        constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    time = ox.Time(
        initial=0.0,
        final=("minimize", 2.0),
        min=0.0,
        max=2.0,
        uniform_time_grid=True,
    )

    return Problem(
        dynamics=dynamics,
        states=states,
        controls=[theta],
        time=time,
        constraints=constraint_exprs,
        N=n,
        float_dtype="float64",
        byof=byof,
        algorithm={
            "autotuner": "ConstantProximalWeight",
            "lam_prox": lam_prox,
            "lam_cost": 6e-1,
            "k_max": 200,
            "ep_tr": 1e-5,
            "ep_vb": 1e-5,
            "ep_vc": 1e-9,
        },
        solver={
            "backend": backend,
            "solver_args": {"enforce_dpp": True, "abstol": 1e-8, "reltol": 1e-10},
        },
    )


PROX_WEIGHT = 1e0


def _byof_proximal_cost(ocp_vars):
    cost = 0
    for i in range(ocp_vars.x.shape[0]):
        x_dev = ocp_vars.inv_S_x @ (ocp_vars.x_nonscaled[i] - ocp_vars.x_bar[i])
        u_dev = ocp_vars.inv_S_u @ (ocp_vars.u_nonscaled[i] - ocp_vars.u_bar[i])
        z = cp.hstack([x_dev, u_dev])
        cost += PROX_WEIGHT * cp.sum_squares(z)
    return cost


def test_convex_costs_rejected_on_qpax_backend():
    byof: ByofSpec = {
        "convex_costs": [{"cost_fn": lambda ocp_vars: cp.sum_squares(ocp_vars.x_nonscaled[0])}]
    }
    with pytest.raises(NotImplementedError, match="convex_costs"):
        _make_brachistochrone(n=4, byof=byof, backend="qpax")


def test_brachistochrone_byof_proximal_converges():
    byof: ByofSpec = {"convex_costs": [{"cost_fn": _byof_proximal_cost}]}
    problem = _make_brachistochrone(n=6, byof=byof, lam_prox=0.0)
    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False
    problem.settings.sim.save_compiled = False

    problem.initialize()
    problem.solve()
    result = problem.post_process()

    assert result["converged"], "BYOF proximal brachistochrone failed to converge"


def test_nodal_convex_cost_nodes_field():
    position = ox.State("position", shape=(2,))
    position.min = np.array([-10.0, -10.0])
    position.max = np.array([10.0, 10.0])
    position.initial = np.array([0.0, 0.0])
    position.final = np.array([1.0, 1.0])
    velocity = ox.State("velocity", shape=(1,))
    velocity.min = np.array([-5.0])
    velocity.max = np.array([5.0])
    velocity.initial = np.array([0.0])
    velocity.final = np.array([0.0])
    theta = ox.Control("theta", shape=(1,))
    theta.min = np.array([-1.0])
    theta.max = np.array([1.0])
    theta.guess = np.zeros((3, 1))

    def terminal_cost(x, u, node, params, ocp_vars):
        return cp.sum_squares(x[0:2])

    byof: ByofSpec = {"convex_costs": [{"cost_fn": terminal_cost, "nodes": [-1]}]}

    problem = Problem(
        dynamics={
            "position": ox.Concat(velocity[0], velocity[0]),
            "velocity": theta[0],
        },
        states=[position, velocity],
        controls=[theta],
        time=ox.Time(initial=0.0, final=1.0, min=0.0, max=1.0),
        constraints=[],
        N=3,
        byof=byof,
        solver={"backend": "cvxpy", "solver_args": {"enforce_dpp": True}},
    )
    assert len(problem._lowered.byof_convex_costs) == 1
    assert problem._lowered.byof_convex_costs[0].nodes == [-1]
