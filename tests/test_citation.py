"""``Problem.citation()`` credits every component a problem actually uses.

The aggregate string carries one commented section per component — algorithm,
autotuner, convex solver, discretization — plus a section for cited symbolic
nodes found by walking the problem's expression graphs. Construction only; no
solves.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import openscvx as ox
from openscvx import Problem
from openscvx.symbolic.expr import stl


@pytest.fixture(scope="module")
def problem():
    """Brachistochrone with a GMSR STL constraint and the AL autotuner."""
    n = 8

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

    dynamics = {
        "position": ox.Concat(
            velocity[0] * ox.Sin(theta[0]),
            -velocity[0] * ox.Cos(theta[0]),
        ),
        "velocity": 9.81 * ox.Cos(theta[0]),
    }

    constraints = []
    for state in [position, velocity]:
        constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

    # Two GMSR operators so the aggregate proves per-entry deduplication.
    slow = velocity[0] <= 9.0
    high = 1.0 <= position[1]
    constraints.append(stl.Or(slow, high).at([n - 1]))
    constraints.append(stl.And(slow, high).at([0]))

    time = ox.Time(initial=0.0, final=("minimize", 2.0), min=0.0, max=2.0, uniform_time_grid=True)

    prob = Problem(
        dynamics=dynamics,
        states=[position, velocity],
        controls=[theta],
        time=time,
        constraints=constraints,
        N=n,
        algorithm={"autotuner": "AugmentedLagrangian", "lam_prox": 1e0, "lam_cost": 6e-1},
    )
    prob.settings.dev.printing = False
    return prob


def test_citation_has_a_section_per_component(problem):
    text = problem.citation()

    assert "% Algorithm: " in text
    assert "% Autotuner: AugmentedLagrangian" in text
    assert "% Convex Solver: " in text
    assert "% Symbolic Operators: " in text


def test_citation_names_the_cited_node_classes(problem):
    text = problem.citation()

    header = next(line for line in text.splitlines() if line.startswith("% Symbolic Operators: "))
    assert "And" in header and "Or" in header


def test_citation_dedupes_shared_node_entries(problem):
    # Or and And both cite the GMSR paper; the aggregate lists it once.
    assert problem.citation().count("uzun2024gmsr") == 1


def test_citation_includes_autotuner_references(problem):
    text = problem.citation()
    assert "oguri2023scvxstar" in text
    assert "mao2016scvx" in text
