"""Unit test for the Hohmann transfer impulsive example.

This validates the example in ``examples/spacecraft/hohmann_transfer.py``
against the analytical LEO → GEO Hohmann transfer from:

    https://orbital-mechanics.space/orbital-maneuvers/hohmann-transfer-example.html

The symbolic problem build is parametrized across all three PTR backends —
``cvxpy``, ``qpax``, and ``moreau`` — to lock in impulsive-control parity
between the JAX-native solvers and the CVXPy reference.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tests._marks import _MOREAU_OK
from tests.hohmann_analytical import compute_hohmann_delta_v_and_mass

# Problem configuration (match Weber's LEO → GEO example)
n = 15

mu = 3.986e5  # km^3 / s^2
R_E = 6378.0  # km
r_leo = 250.0 + R_E

sidereal_day = 86164.0905  # s
r_cubed = mu * sidereal_day**2 / (4 * np.pi**2)
r_geo = r_cubed ** (1.0 / 3.0)

r1 = r_leo
r2 = r_geo

# Hohmann transfer half-period for the transfer ellipse (fixed final time)
a_transfer = 0.5 * (r1 + r2)
T_transfer = np.pi * np.sqrt(a_transfer**3 / mu)


def _build_hohmann_problem(backend: str):
    """Build the Hohmann transfer problem with the requested PTR backend.

    Mirrors ``examples/spacecraft/hohmann_transfer.py`` but threads the
    backend selector and ``float_dtype="float64"`` through so the JAX
    backends meet their inner-solver conditioning requirements.
    """
    import openscvx as ox
    from openscvx import Problem

    position = ox.State("position", shape=(2,))
    position.initial = np.array([r1, 0.0])
    position.final = np.array([-r2, 0.0])
    position.min = np.array([-(1.5 * r2), -(1.5 * r2)])
    position.max = np.array([1.5 * r2, 1.5 * r2])

    theta_guess = np.linspace(0.0, np.pi, n)
    radius_guess = np.linspace(r1, r2, n)
    position.guess = np.stack(
        [radius_guess * np.cos(theta_guess), radius_guess * np.sin(theta_guess)],
        axis=1,
    )

    velocity = ox.State("velocity", shape=(2,))
    v_c1 = np.sqrt(mu / r1)
    v_c2 = np.sqrt(mu / r2)
    velocity.initial = np.array([0.0, v_c1])
    velocity.final = np.array([0.0, -v_c2])
    v_bound = 2.0 * max(v_c1, v_c2)
    velocity.min = np.array([-v_bound, -v_bound])
    velocity.max = np.array([v_bound, v_bound])
    velocity.guess = np.tile(np.array([0.0, (v_c1 + v_c2) / 2.0]), (n, 1))

    cost = ox.State("cost", shape=(1,))
    cost.initial = np.array([0.0])
    cost.final = [("minimize", 10.0)]
    cost.min = np.array([0.0])
    cost.max = np.array([10.0])
    cost.guess = np.zeros((n, 1))

    dv = ox.Control(
        "delta_v",
        shape=(2,),
        parameterization="impulsive",
        nodes=[0, n - 1],
    )
    dv_bound = 5.0
    dv.min = np.array([-dv_bound, -dv_bound])
    dv.max = np.array([dv_bound, dv_bound])
    dv.guess = np.zeros((n, 2))

    states = [position, velocity, cost]
    controls = [dv]

    r = ox.linalg.Norm(position)
    dynamics = {
        "position": velocity,
        "velocity": ox.Concat(
            -mu * position[0] / r**3,
            -mu * position[1] / r**3,
        ),
        "cost": 0.0,
    }

    eps_impulse = 1e-6
    d_impulse = ox.linalg.Norm(dv + eps_impulse)
    dynamics_discrete = {
        "position": position,
        "velocity": velocity + dv,
        "cost": cost + d_impulse,
    }

    constraints = []
    for state in states:
        constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    time = ox.Time(initial=0.0, final=T_transfer, min=0.0, max=T_transfer)

    problem = Problem(
        dynamics=dynamics,
        dynamics_discrete=dynamics_discrete,
        states=states,
        controls=controls,
        time=time,
        constraints=constraints,
        N=n,
        float_dtype="float64",
        solver={"backend": backend},
    )
    problem.discretizer.ode_solver = "Dopri8"
    return problem


@pytest.mark.parametrize("backend", ["cvxpy", "qpax", "moreau"])
def test_hohmann_transfer(backend):
    """Check that optimized impulsive Δv matches analytical Hohmann Δv,
    independently for each PTR backend."""
    if backend == "qpax":
        pytest.importorskip("qpax")
    if backend == "moreau" and not _MOREAU_OK:
        pytest.skip("moreau not installed or license key not found (pip install openscvx[moreau])")

    problem = _build_hohmann_problem(backend)

    if hasattr(problem.settings, "dev"):
        problem.settings.dev.printing = False
    problem.settings.prp.dt = 100.0

    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    assert result["converged"], f"Hohmann impulsive ({backend}) failed to converge"

    analytic = compute_hohmann_delta_v_and_mass()
    dv_nodes = np.asarray(result.nodes["delta_v"])
    dv_mags = np.linalg.norm(dv_nodes, axis=1)
    delta_v_num = dv_mags.sum()

    delta_v_ref = analytic["delta_v"]
    rel_err = abs(delta_v_num - delta_v_ref) / delta_v_ref
    assert rel_err < 0.01, (
        f"Hohmann Δv mismatch ({backend}): numerical={delta_v_num:.4f} km/s, "
        f"analytical={delta_v_ref:.4f} km/s, rel_err={rel_err:.3%}"
    )

    jax.clear_caches()


def test_hohmann_transfer_byof():
    """Check that optimized impulsive Δv matches analytical Hohmann Δv using the byof interface."""
    import jax.numpy as jnp

    import openscvx as ox
    from openscvx import Problem
    from openscvx.expert import ByofSpec

    # States: planar position, planar velocity, and scalar accumulated cost
    position = ox.State("position", shape=(2,))
    position.initial = np.array([r1, 0.0])
    position.final = np.array([-r2, 0.0])  # Opposite side for half-period transfer
    position.min = np.array([-(1.5 * r2), -(1.5 * r2)])
    position.max = np.array([1.5 * r2, 1.5 * r2])

    # Initial guess: follow an approximate Hohmann-like arc that never passes
    # through the origin (to avoid r ≈ 0 causing singular dynamics).
    theta_guess = np.linspace(0.0, np.pi, n)
    radius_guess = np.linspace(r1, r2, n)
    position_guess = np.stack(
        [radius_guess * np.cos(theta_guess), radius_guess * np.sin(theta_guess)],
        axis=1,
    )
    position.guess = position_guess

    velocity = ox.State("velocity", shape=(2,))
    v_c1 = np.sqrt(mu / r1)
    v_c2 = np.sqrt(mu / r2)
    velocity.initial = np.array([0.0, v_c1])
    velocity.final = np.array([0.0, -v_c2])
    velocity_min_mag = -2.0 * max(v_c1, v_c2)
    velocity_max_mag = 2.0 * max(v_c1, v_c2)
    velocity.min = np.array([velocity_min_mag, velocity_min_mag])
    velocity.max = np.array([velocity_max_mag, velocity_max_mag])
    velocity.guess = np.tile(np.array([0.0, (v_c1 + v_c2) / 2.0]), (n, 1))

    cost = ox.State("cost", shape=(1,))
    cost.initial = np.array([0.0])
    # Minimize final accumulated impulse norm with a loose upper bound
    cost.final = [("minimize", 10.0)]
    cost.min = np.array([0.0])
    cost.max = np.array([10.0])
    cost.guess = np.zeros((n, 1))

    # Impulsive delta-v control applied only at the first and last nodes
    dv = ox.Control(
        "delta_v",
        shape=(2,),
        parameterization="impulsive",
        nodes=[0, n - 1],
    )
    dv_bound = 5.0  # km/s
    dv.min = np.array([-dv_bound, -dv_bound])
    dv.max = np.array([dv_bound, dv_bound])
    dv.guess = np.zeros((n, 2))

    states = [position, velocity, cost]
    controls = [dv]

    # Continuous dynamics: planar two-body gravity with no continuous thrust.
    # We clamp the radius away from zero for robustness in the linearization.
    r = ox.linalg.Norm(position)

    dynamics = {
        "position": velocity,
        "velocity": ox.Concat(
            -mu * position[0] / r**3,
            -mu * position[1] / r**3,
        ),
        "cost": 0.0,
    }

    # Discrete dynamics at impulsive nodes: update velocity and accumulate cost.
    # Use a small epsilon inside the norm to avoid NaNs in derivatives at dv = 0.
    eps_impulse = 1e-6

    byof: ByofSpec = {
        "dynamics_discrete": {
            "position": lambda x, u, node, params: x[position.slice],
            "velocity": lambda x, u, node, params: x[velocity.slice] + u[dv.slice],
            "cost": lambda x, u, node, params: (
                x[cost.slice] + jnp.linalg.norm(u[dv.slice] + eps_impulse)
            ),
        }
    }

    constraints = []
    for state in states:
        constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    time = ox.Time(
        initial=0.0,
        final=T_transfer,
        min=0.0,
        max=T_transfer,
    )

    problem = Problem(
        dynamics=dynamics,
        byof=byof,
        states=states,
        controls=controls,
        time=time,
        constraints=constraints,
        N=n,
    )

    problem.discretizer.ode_solver = "Dopri8"

    # Increase propagation dt to speedup post-processing
    problem.settings.prp.dt = 100.0

    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    # Basic convergence check
    assert result["converged"], "Hohmann impulsive example failed to converge"

    # Analytical reference
    analytic = compute_hohmann_delta_v_and_mass()

    # Numerical Δv: sum of norms of impulsive delta-v at nodes
    dv_nodes = np.asarray(result.nodes["delta_v"])
    dv_mags = np.linalg.norm(dv_nodes, axis=1)
    delta_v_num = dv_mags.sum()

    # Compare Δv (expect < 1% relative error)
    delta_v_ref = analytic["delta_v"]
    rel_err = abs(delta_v_num - delta_v_ref) / delta_v_ref
    assert rel_err < 0.01, (
        f"Hohmann Δv mismatch: numerical={delta_v_num:.4f} km/s, "
        f"analytical={delta_v_ref:.4f} km/s, rel_err={rel_err:.3%}"
    )

    # Clean up JAX caches
    jax.clear_caches()
