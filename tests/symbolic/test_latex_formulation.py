"""Tests for ``Problem.to_latex`` and the Mayer-form formulation builder.

Builds one small double-integrator problem (module-scoped, since constructing a
``Problem`` triggers JAX/CVXPy lowering) and asserts the rendered rows across the
``dynamics`` / ``constraints`` mode combinations: that the expected objective,
dynamics, constraint, box-bound, and boundary rows appear; that ``g_i`` / ``h_j``
numbering lines up between the ``"symbolic"`` references and the ``"separate"``
definitions; that the solver-augmented ``_``-prefixed states/controls never leak
into any output; and that invalid mode arguments raise ``ValueError``.
"""

import re

import numpy as np
import pytest

import openscvx as ox
from openscvx import Problem
from openscvx.symbolic.expr import Norm
from openscvx.symbolic.lowerers.latex import LatexLowerer
from openscvx.symbolic.lowerers.latex.formulation import _objective

N = 5


@pytest.fixture(scope="module")
def problem() -> Problem:
    """A minimal double integrator: minimize final time, one CTCS + two nodal.

    - states ``position`` (2,), ``velocity`` (2,); control ``u`` (2,)
    - minimize final time (Mayer objective, weight 0.01)
    - CTCS path constraint ``|position| <= 5`` over the full horizon
    - nodal inequality ``|u| <= 10`` at the last node
    - nodal equality ``velocity[0] == 0`` at the last node
    - box bounds on every variable; fixed initial/final on ``position``
    """
    position = ox.State("position", shape=(2,))
    position.min = np.array([-10.0, -10.0])
    position.max = np.array([10.0, 10.0])
    position.initial = np.array([0.0, 0.0])
    position.final = np.array([5.0, 5.0])

    velocity = ox.State("velocity", shape=(2,))
    velocity.min = np.array([-5.0, -5.0])
    velocity.max = np.array([5.0, 5.0])
    velocity.initial = np.array([0.0, 0.0])
    velocity.final = [("free", 0.0), ("free", 0.0)]

    u = ox.Control("u", shape=(2,))
    u.min = np.array([-1.0, -1.0])
    u.max = np.array([1.0, 1.0])
    u.guess = np.zeros((N, 2))

    time = ox.Time(initial=0.0, final=("minimize", 2.0), min=0.0, max=2.0)

    dynamics = {"position": velocity, "velocity": u}
    constraints = [
        (Norm(position) <= 5.0).over((0, N)),
        (Norm(u) <= 10.0).at([N - 1]),
        (velocity[0] == 0.0).at([N - 1]),
    ]

    prob = Problem(
        dynamics=dynamics,
        states=[position, velocity],
        controls=[u],
        time=time,
        constraints=constraints,
        N=N,
        float_dtype="float64",
        algorithm={"lam_cost": 0.01, "k_max": 1},
    )
    prob.settings.dev.printing = False
    return prob


_MODES = ("inline", "symbolic", "separate")


def _all_renderings(prob):
    """Every ``(dynamics, constraints)`` mode combination as one joined string."""
    return "\n".join(
        prob.to_latex(dynamics=d, constraints=c) for d in _MODES for c in _MODES
    )


# === dynamics property ======================================================


def test_dynamics_property_returns_user_dict(problem):
    dyn = problem.dynamics
    assert set(dyn.keys()) == {"position", "velocity"}
    # Pre-augmentation: no injected time state, no CTCS penalty state.
    assert "time" not in dyn
    assert not any(name.startswith("_") for name in dyn)


def test_dynamics_property_decoupled_from_input_dict():
    # The stash is a shallow copy, so mutating the caller's original dynamics
    # dict after construction does not leak into ``problem.dynamics``.
    r = ox.State("r", shape=(1,), min=[-1.0], max=[1.0], initial=[0.0], final=[("free", 0.0)])
    a = ox.Control("a", shape=(1,), min=[-1.0], max=[1.0])
    a.guess = np.zeros((3, 1))
    dynamics = {"r": a[0]}
    prob = Problem(
        dynamics=dynamics,
        states=[r],
        controls=[a],
        time=ox.Time(initial=0.0, final=("minimize", 1.0), min=0.0, max=1.0),
        constraints=[],
        N=3,
        float_dtype="float64",
        algorithm={"lam_cost": 1.0, "k_max": 1},
    )
    dynamics["injected"] = 1.0
    assert "injected" not in prob.dynamics
    assert set(prob.dynamics) == {"r"}


# === objective ==============================================================


def test_objective_symbolic_weight_is_default(problem):
    # Default weights="symbolic": coefficient is a \lambda subscripted by the
    # state's inner symbol (t for time), always shown.
    assert r"\min_{x,\,u} \quad & \lambda_{t}\, t(t_f)" in problem.to_latex()


def test_objective_numeric_weight_substitutes_lam_cost(problem):
    # weights="numeric" substitutes the lam_cost value, formatted via %g.
    assert r"\min_{x,\,u} \quad & 0.01\, t(t_f)" in problem.to_latex(weights="numeric")


def test_objective_symbolic_weight_on_vector_state_element():
    # A minimized element of a vector state: both the \lambda subscript and the
    # element term comma-merge the element index into the role-prefixed group.
    v = ox.State("velocity", shape=(2,))
    v._slice = slice(0, 2)
    v.final = [ox.Minimize(0.0), ("free", 0.0)]
    got = _objective([v], lam_cost=1.0, weights="symbolic", lowerer=LatexLowerer())
    assert got == r"\lambda_{\mathrm{velocity},0}\, x_{\mathrm{velocity},0}(t_f)"


# === dynamics section =======================================================


def test_dynamics_symbolic_is_placeholder(problem):
    assert r"\dot{x} = f(x, u)" in problem.to_latex(dynamics="symbolic")


def test_dynamics_inline_expands_each_row(problem):
    out = problem.to_latex(dynamics="inline")
    # The accent sits on the role letter, subscript after: \dot{x}_{...}.
    assert r"\dot{x}_{\mathrm{position}} = x_{\mathrm{velocity}}" in out
    assert r"\dot{x}_{\mathrm{velocity}} = u" in out


def test_dynamics_separate_appends_where_block(problem):
    out = problem.to_latex(dynamics="separate")
    assert r"\dot{x} = f(x, u)" in out  # skeleton stays in the formulation
    assert r"\text{where}" in out
    assert r"\dot{x}_{\mathrm{position}} &= x_{\mathrm{velocity}}" in out
    assert r"\dot{x}_{\mathrm{velocity}} &= u" in out


# === constraint section =====================================================


def test_constraints_inline_shows_bodies_and_annotations(problem):
    out = problem.to_latex(constraints="inline")
    # CTCS over the whole horizon collapses to a bare \forall t.
    assert r"\left\| x_{\mathrm{position}} \right\| - 5 \le 0 \quad \forall t" in out
    # Nodal constraint at a single node keeps its k = ... annotation.
    assert r"\left\| u \right\| - 10 \le 0 \quad k = 4" in out


def test_constraints_symbolic_numbers_ineq_and_eq(problem):
    out = problem.to_latex(constraints="symbolic")
    # Inequalities are g_i, equalities h_j; annotations stay on the reference.
    assert r"g_{1}(x, u) \le 0 \quad \forall t" in out
    assert r"h_{1}(x, u) = 0 \quad k = 4" in out


def test_constraints_separate_defines_each_residual(problem):
    out = problem.to_latex(constraints="separate")
    assert r"\text{where}" in out
    assert r"g_{1}(x, u) &= \left\| x_{\mathrm{position}} \right\| - 5" in out
    assert r"h_{1}(x, u) &= x_{\mathrm{velocity},0} - 0" in out


def test_separate_numbering_matches_between_refs_and_defs(problem):
    out = problem.to_latex(dynamics="symbolic", constraints="separate")
    formulation, _, definitions = out.partition(r"\text{where}")

    ref_labels = set(re.findall(r"([gh]_\{\d+\})\(x, u\) (?:\\le|=) 0", formulation))
    def_labels = set(re.findall(r"([gh]_\{\d+\})\(x, u\) &=", definitions))

    assert ref_labels  # something got numbered
    assert ref_labels == def_labels


# === box bounds & boundary conditions =======================================


def test_box_bounds_render_finite_sides(problem):
    out = problem.to_latex()
    # Scalar (length-1) bounds collapse to bare scalars; vectors use bmatrix.
    assert r"0 \le t \le 2" in out
    assert (
        r"\begin{bmatrix} -1 \\ -1 \end{bmatrix} \le u "
        r"\le \begin{bmatrix} 1 \\ 1 \end{bmatrix}"
    ) in out


def test_boundary_conditions_render_fixed_only(problem):
    out = problem.to_latex()
    # Whole-vector fixed initial/final render as one bmatrix row.
    assert r"x_{\mathrm{position}}(t_0) = \begin{bmatrix} 0 \\ 0 \end{bmatrix}" in out
    assert r"x_{\mathrm{position}}(t_f) = \begin{bmatrix} 5 \\ 5 \end{bmatrix}" in out
    # Fixed initial time renders as a scalar; free final velocity is omitted.
    assert r"t(t_0) = 0" in out
    assert r"x_{\mathrm{velocity}}(t_f)" not in out


# === augmentation filtering =================================================


def test_augmented_names_never_appear(problem):
    out = _all_renderings(problem)
    assert "_ctcs_aug" not in out
    assert "_time_dilation" not in out
    # The augmented time-dilation control is the only underscore-name in play;
    # no raw underscore-prefixed identifier should survive to any rendering.
    assert "time_dilation" not in out


def test_time_bound_ctcs_render_as_path_constraints(problem):
    # Preprocessing appends CTCS(time <= max) / CTCS(min <= time); they are
    # real constraints on the problem and render alongside the time box row.
    out = problem.to_latex(constraints="inline")
    assert r"t - 2 \le 0" in out
    assert r"0 - t \le 0" in out
    assert r"0 \le t \le 2" in out


# === mode validation ========================================================


def test_invalid_dynamics_mode_raises(problem):
    with pytest.raises(ValueError, match="bogus"):
        problem.to_latex(dynamics="bogus")


def test_invalid_constraints_mode_raises(problem):
    with pytest.raises(ValueError, match="bogus"):
        problem.to_latex(constraints="bogus")


def test_invalid_weights_mode_raises(problem):
    with pytest.raises(ValueError, match="weights"):
        problem.to_latex(weights="bogus")


# === notebook hook ==========================================================


def test_repr_latex_wraps_in_display_math(problem):
    rendered = problem._repr_latex_()
    assert rendered.startswith("$$")
    assert rendered.rstrip().endswith("$$")
    assert r"\begin{aligned}" in rendered
