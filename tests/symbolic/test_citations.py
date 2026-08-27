"""``SymbolicProblem.citations()`` gathers node citations from the expression graphs.

The collector walks dynamics, constraints, and propagation expressions and
returns a mapping from node class name to that class's BibTeX entries, so
``Problem.citation()`` can credit the published methods a problem actually
uses. These tests build raw (pre-preprocessing) problems by hand — no
lowering, no solves.
"""

import numpy as np
import pytest

from openscvx.symbolic.constraint_set import ConstraintSet
from openscvx.symbolic.expr import Constant, Control, State, stl, stljax
from openscvx.symbolic.problem import SymbolicProblem

# =============================================================================
# Helpers
# =============================================================================


def _make_problem(dynamics, constraints):
    x = State("x", shape=(2,))
    u = Control("u", shape=(1,))
    return SymbolicProblem(
        dynamics=dynamics,
        states=[x],
        controls=[u],
        constraints=ConstraintSet(unsorted=constraints),
        parameters={},
        N=10,
    )


def _predicates():
    x = State("x", shape=(2,))
    p1 = x[0] <= Constant(np.array(1.0))
    p2 = x[1] <= Constant(np.array(2.0))
    return x, p1, p2


# =============================================================================
# Collection
# =============================================================================


def test_uncited_problem_returns_empty():
    x, p1, _ = _predicates()
    problem = _make_problem(dynamics=x + 1, constraints=[p1.at([0])])
    assert problem.citations() == {}


def test_collects_cited_nodes_from_constraints():
    x, p1, p2 = _predicates()
    problem = _make_problem(dynamics=x + 1, constraints=[stl.Or(p1, p2).at([0])])

    cited = problem.citations()

    assert set(cited) == {"Or"}
    assert any("uzun2024gmsr" in entry for entry in cited["Or"])


@pytest.mark.lie
def test_collects_cited_nodes_from_dynamics():
    from openscvx.symbolic.expr import SO3Exp

    x, p1, _ = _predicates()
    problem = _make_problem(dynamics=SO3Exp(Constant(np.zeros(3))), constraints=[p1.at([0])])

    cited = problem.citations()

    assert set(cited) == {"SO3Exp"}
    assert any("yi2021iros" in entry for entry in cited["SO3Exp"])


def test_distinct_node_classes_are_keyed_separately():
    x, p1, p2 = _predicates()
    problem = _make_problem(
        dynamics=x + 1,
        constraints=[stl.Or(p1, p2).at([0]), stl.And(p1, p2).at([0]), stljax.Or(p1, p2).at([0])],
    )

    cited = problem.citations()

    # The GMSR and stljax Or share the "Or" key; their entries are merged so
    # neither reference is lost, and the repeated GMSR entry is deduplicated.
    assert set(cited) == {"Or", "And"}
    assert sum("uzun2024gmsr" in entry for entry in cited["Or"]) == 1
    assert any("kapoor2025stlcg" in entry for entry in cited["Or"])
    assert any("uzun2024gmsr" in entry for entry in cited["And"])
