"""Tests for the LaTeX lowering backend.

Covers per-node string rendering across the Component-A coverage list,
precedence-driven parenthesization, the ``latex_symbol`` and
``format_constant`` helpers, and the ``NotImplementedError`` fallback for
nodes with no registered visitor.
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Abs,
    Acos,
    Add,
    Asin,
    Atan,
    Atan2,
    Concat,
    Constant,
    Control,
    Cos,
    CrossNodeConstraint,
    Diag,
    Div,
    Exp,
    Hstack,
    Index,
    Inv,
    Log,
    MatMul,
    Max,
    Min,
    Mul,
    Neg,
    Norm,
    Parameter,
    PositivePart,
    Power,
    Sin,
    Sqrt,
    Square,
    Stack,
    State,
    Sub,
    Sum,
    Tan,
    Time,
    Transpose,
    Variable,
    Vstack,
    stl,
)
from openscvx.symbolic.expr.constraint import CTCS, Equality, Inequality, NodalConstraint
from openscvx.symbolic.expr.expr import Expr
from openscvx.symbolic.expr.lie.adjoint import (
    Adjoint,
    AdjointDual,
    SE3Adjoint,
    SE3AdjointDual,
)
from openscvx.symbolic.expr.lie.se3 import SE3Exp, SE3Log
from openscvx.symbolic.expr.lie.so3 import SO3Exp, SO3Log
from openscvx.symbolic.expr.logic import All, Any, Cond
from openscvx.symbolic.expr.math import (
    Bilerp,
    Cinterp,
    Huber,
    Linterp,
    LogSumExp,
    SmoothReLU,
)
from openscvx.symbolic.expr.spatial import QDCM, SSM, SSMP
from openscvx.symbolic.expr.vmap import Vmap
from openscvx.symbolic.lower import to_latex
from openscvx.symbolic.lowerers.latex import LatexLowerer, format_constant, latex_symbol
from openscvx.symbolic.lowerers.latex._lowerer import merge_subscript


def _sliced_state(name, dim):
    s = State(name, (dim,))
    s._slice = slice(0, dim)
    return s


def _sliced_control(name, dim):
    c = Control(name, (dim,))
    c._slice = slice(0, dim)
    return c


def lower(expr) -> str:
    return LatexLowerer().lower(expr)


# =============================================================================
# Leaves
# =============================================================================


def test_state_named_x_renders_bare():
    # The bare-`x` exception: a state literally named "x" keeps the role letter.
    assert lower(_sliced_state("x", 3)) == "x"


def test_state_gets_x_role_prefix():
    # Any other state is grounded in the skeleton's f(x, u) via an x_ prefix.
    assert lower(_sliced_state("r", 3)) == r"x_{r}"
    assert lower(_sliced_state("pos", 3)) == r"x_{\mathrm{pos}}"
    assert lower(_sliced_state("velocity", 3)) == r"x_{\mathrm{velocity}}"


def test_control_named_u_renders_bare():
    # The bare-`u` exception mirrors the bare-`x` one.
    assert lower(_sliced_control("u", 2)) == "u"


def test_control_gets_u_role_prefix():
    assert lower(_sliced_control("theta", 2)) == r"u_{\theta}"
    assert lower(_sliced_control("thrust", 2)) == r"u_{\mathrm{thrust}}"


def test_time_renders_as_t():
    # Time carries the literal name "time" but always renders as t.
    assert lower(Time()) == "t"


def test_variable_renders_symbol():
    v = Variable("y", (2,))
    assert lower(v) == "y"


def test_parameter_renders_symbol():
    assert lower(Parameter("obs", shape=(2,), value=np.zeros(2))) == r"\mathrm{obs}"


def test_constant_scalar():
    assert lower(Constant(np.array(5.0))) == "5"


def test_node_reference_superscript():
    pos = _sliced_state("pos", 2)
    assert lower(pos.at(5)) == r"x_{\mathrm{pos}}^{(5)}"


# =============================================================================
# Arithmetic
# =============================================================================


def test_add():
    x = _sliced_state("x", 1)
    assert lower(Add(x, Constant(np.array(1.0)))) == "x + 1"


def test_sub():
    x = _sliced_state("x", 1)
    assert lower(Sub(x, Constant(np.array(1.0)))) == "x - 1"


def test_mul_uses_cdot():
    x = _sliced_state("x", 1)
    assert lower(Mul(Constant(np.array(2.0)), x)) == r"2 \cdot x"


def test_div_uses_frac():
    x = _sliced_state("x", 1)
    assert lower(Div(x, Constant(np.array(2.0)))) == r"\frac{x}{2}"


def test_matmul_juxtaposition():
    A = Parameter("A", shape=(2, 2), value=np.eye(2))
    x = _sliced_state("x", 2)
    assert lower(MatMul(A, x)) == "A x"


def test_neg():
    x = _sliced_state("x", 1)
    assert lower(Neg(x)) == "-x"


def test_power():
    x = _sliced_state("x", 1)
    assert lower(Power(x, Constant(np.array(3.0)))) == "x^{3}"


# =============================================================================
# Precedence / parenthesization
# =============================================================================


def test_mul_wraps_add_child():
    x = _sliced_state("x", 1)
    expr = Mul(Add(x, Constant(np.array(1.0))), Constant(np.array(2.0)))
    assert lower(expr) == r"\left( x + 1 \right) \cdot 2"


def test_add_child_of_mul_not_double_wrapped_at_top():
    # Top-level Add is not wrapped; only children below a tighter parent are.
    x = _sliced_state("x", 1)
    assert lower(Add(x, x)) == "x + x"


def test_sub_wraps_equal_precedence_right_operand():
    x = _sliced_state("x", 1)
    inner = Add(x, Constant(np.array(1.0)))
    assert lower(Sub(x, inner)) == r"x - \left( x + 1 \right)"


def test_power_wraps_add_base():
    x = _sliced_state("x", 1)
    base = Add(x, Constant(np.array(1.0)))
    assert lower(Power(base, Constant(np.array(2.0)))) == r"\left( x + 1 \right)^{2}"


def test_neg_wraps_add_operand():
    x = _sliced_state("x", 1)
    assert lower(Neg(Add(x, x))) == r"-\left( x + x \right)"


# =============================================================================
# Array
# =============================================================================


def test_index_int():
    x = _sliced_state("x", 3)
    assert lower(Index(x, 0)) == "x_{0}"


def test_index_slice():
    x = _sliced_state("x", 3)
    assert lower(Index(x, slice(0, 2))) == "x_{0:2}"


def test_index_merges_into_role_prefixed_subscript():
    # An index on a role-prefixed state comma-merges into the existing group
    # rather than emitting the invalid double subscript x_{...}_{0}.
    v = _sliced_state("velocity", 3)
    assert lower(Index(v, 0)) == r"x_{\mathrm{velocity},0}"


def test_concat_column_bmatrix():
    x = _sliced_state("x", 1)
    y = Constant(np.array(1.0))
    assert lower(Concat(x, y)) == r"\begin{bmatrix} x \\ 1 \end{bmatrix}"


def test_hstack_row_bmatrix():
    x = _sliced_state("x", 1)
    y = Variable("y", (1,))
    assert lower(Hstack([x, y])) == r"\begin{bmatrix} x & y \end{bmatrix}"


def test_stack_and_vstack_column_bmatrix():
    x = _sliced_state("x", 1)
    y = Variable("y", (1,))
    assert lower(Stack([x, y])) == r"\begin{bmatrix} x \\ y \end{bmatrix}"
    assert lower(Vstack([x, y])) == r"\begin{bmatrix} x \\ y \end{bmatrix}"


# =============================================================================
# Linalg
# =============================================================================


def test_transpose():
    A = Parameter("A", shape=(2, 2), value=np.eye(2))
    assert lower(Transpose(A)) == r"A^{\top}"


def test_inv():
    A = Parameter("A", shape=(2, 2), value=np.eye(2))
    assert lower(Inv(A)) == r"A^{-1}"


def test_sum():
    x = _sliced_state("x", 3)
    assert lower(Sum(x)) == r"\sum x"


def test_diag():
    x = _sliced_state("x", 3)
    assert lower(Diag(x)) == r"\operatorname{diag}\left( x \right)"


def test_norm_default_no_subscript():
    x = _sliced_state("x", 3)
    assert lower(Norm(x)) == r"\left\| x \right\|"


def test_norm_l1_subscript():
    x = _sliced_state("x", 3)
    assert lower(Norm(x, ord=1)) == r"\left\| x \right\|_{1}"


def test_norm_inf_subscript():
    x = _sliced_state("x", 3)
    assert lower(Norm(x, ord="inf")) == r"\left\| x \right\|_{\infty}"


# =============================================================================
# Math
# =============================================================================


def test_trig_functions():
    x = _sliced_state("x", 1)
    assert lower(Sin(x)) == r"\sin\left( x \right)"
    assert lower(Cos(x)) == r"\cos\left( x \right)"
    assert lower(Tan(x)) == r"\tan\left( x \right)"


def test_inverse_trig_functions():
    x = _sliced_state("x", 1)
    assert lower(Asin(x)) == r"\arcsin\left( x \right)"
    assert lower(Acos(x)) == r"\arccos\left( x \right)"
    assert lower(Atan(x)) == r"\arctan\left( x \right)"


def test_atan2():
    x = _sliced_state("x", 1)
    y = Variable("y", (1,))
    assert lower(Atan2(y, x)) == r"\operatorname{atan2}\left( y, x \right)"


def test_square():
    x = _sliced_state("x", 1)
    assert lower(Square(x)) == "x^{2}"


def test_sqrt():
    x = _sliced_state("x", 1)
    assert lower(Sqrt(x)) == r"\sqrt{x}"


def test_exp_and_log():
    x = _sliced_state("x", 1)
    assert lower(Exp(x)) == r"\exp\left( x \right)"
    assert lower(Log(x)) == r"\ln\left( x \right)"


def test_abs():
    x = _sliced_state("x", 1)
    assert lower(Abs(x)) == r"\left| x \right|"


def test_max_and_min():
    x = _sliced_state("x", 1)
    y = Variable("y", (1,))
    assert lower(Max(x, y)) == r"\max\left( x, y \right)"
    assert lower(Min(x, y)) == r"\min\left( x, y \right)"


def test_positive_part():
    x = _sliced_state("x", 1)
    assert lower(PositivePart(x)) == r"\left( x \right)_{+}"


# =============================================================================
# Constraints
# =============================================================================


def test_equality():
    x = _sliced_state("x", 1)
    assert lower(Equality(x, Constant(np.array(0.0)))) == "x = 0"


def test_inequality():
    x = _sliced_state("x", 1)
    assert lower(Inequality(x, Constant(np.array(5.0)))) == r"x \le 5"


def test_nodal_constraint_contiguous_range():
    x = _sliced_state("x", 1)
    nodal = NodalConstraint(Inequality(x, Constant(np.array(5.0))), nodes=[0, 1, 2, 3])
    assert lower(nodal) == r"x \le 5 \quad k \in \{0, \dots, 3\}"


def test_nodal_constraint_explicit_set():
    x = _sliced_state("x", 1)
    nodal = NodalConstraint(Inequality(x, Constant(np.array(5.0))), nodes=[0, 5, 10])
    assert lower(nodal) == r"x \le 5 \quad k \in \{0, 5, 10\}"


def test_nodal_constraint_single_node():
    x = _sliced_state("x", 1)
    nodal = NodalConstraint(Inequality(x, Constant(np.array(5.0))), nodes=[7])
    assert lower(nodal) == r"x \le 5 \quad k = 7"


def test_nodal_constraint_elides_large_set():
    x = _sliced_state("x", 1)
    nodes = [0, 2, 4, 6, 8, 10, 12, 14]  # non-contiguous, > 6 entries
    nodal = NodalConstraint(Inequality(x, Constant(np.array(5.0))), nodes=nodes)
    assert lower(nodal) == r"x \le 5 \quad k \in \{0, 2, 4, 6, 8, 10, \dots\}"


def test_ctcs_forall_t():
    x = _sliced_state("x", 1)
    assert lower(CTCS(Inequality(x, Constant(np.array(5.0))))) == r"x \le 5 \quad \forall t"


def test_ctcs_with_node_interval():
    x = _sliced_state("x", 1)
    ctcs = CTCS(Inequality(x, Constant(np.array(5.0))), nodes=(0, 10))
    assert lower(ctcs) == r"x \le 5 \quad \forall t \in [t_{0}, t_{10}]"


def test_cross_node_constraint_renders_inner():
    pos = _sliced_state("pos", 2)
    cnc = CrossNodeConstraint(Inequality(pos.at(5) - pos.at(4), Constant(np.array(0.1))))
    assert lower(cnc) == r"x_{\mathrm{pos}}^{(5)} - x_{\mathrm{pos}}^{(4)} \le 0.1"


# =============================================================================
# Math extras (Huber, SmoothReLU, LogSumExp, interpolation)
# =============================================================================


def test_huber_subscripts_delta():
    x = _sliced_state("x", 1)
    assert lower(Huber(x, delta=0.5)) == r"\operatorname{huber}_{0.5}\left( x \right)"


def test_smooth_relu():
    x = _sliced_state("x", 1)
    assert lower(SmoothReLU(x)) == r"\operatorname{smoothrelu}\left( x \right)"


def test_logsumexp():
    x = _sliced_state("x", 1)
    y = Variable("y", (1,))
    assert lower(LogSumExp(x, y)) == r"\operatorname{logsumexp}\left( x, y \right)"


def test_linterp_lowers_all_operands():
    x = _sliced_state("x", 1)
    xp = Constant(np.array([0.0, 1.0]))
    fp = Constant(np.array([0.0, 2.0]))
    assert lower(Linterp(x, xp, fp)) == (
        r"\operatorname{linterp}\left( x, "
        r"\begin{bmatrix} 0 \\ 1 \end{bmatrix}, "
        r"\begin{bmatrix} 0 \\ 2 \end{bmatrix} \right)"
    )


def test_cinterp_lowers_only_query_point():
    # Breakpoints/coeffs are baked in, so x is the only symbolic operand.
    x = _sliced_state("x", 1)
    assert lower(Cinterp(x, np.arange(5.0), np.arange(5.0))) == (
        r"\operatorname{cinterp}\left( x \right)"
    )


def test_bilerp_lowers_all_operands():
    x = _sliced_state("x", 1)
    y = Variable("y", ())
    xp = Constant(np.array([0.0, 1.0]))
    yp = Constant(np.array([0.0, 1.0]))
    fp = Constant(np.zeros((2, 2)))
    got = lower(Bilerp(x[0], y, xp, yp, fp))
    assert got.startswith(r"\operatorname{bilerp}\left( x_{0}, y, ")


# =============================================================================
# Logic (All, Any, Cond)
# =============================================================================


def test_all_bigwedge():
    x = _sliced_state("x", 1)
    p1 = Inequality(x, Constant(np.array(5.0)))
    p2 = Inequality(Neg(x), Constant(np.array(0.0)))
    assert lower(All([p1, p2])) == r"\bigwedge \left( x \le 5, -x \le 0 \right)"


def test_any_bigvee():
    x = _sliced_state("x", 1)
    p1 = Inequality(x, Constant(np.array(5.0)))
    p2 = Inequality(Neg(x), Constant(np.array(0.0)))
    assert lower(Any([p1, p2])) == r"\bigvee \left( x \le 5, -x \le 0 \right)"


def test_cond_cases_environment():
    x = _sliced_state("x", 1)
    pred = Inequality(x, Constant(np.array(5.0)))
    got = lower(Cond(pred, Constant(np.array(1.0)), Constant(np.array(0.0))))
    assert got == (
        r"\begin{cases} 1 & \text{if } x \le 5 \\ 0 & \text{otherwise} \end{cases}"
    )


def test_cond_node_ranges_without_predicate():
    got = lower(
        Cond(None, Constant(np.array(1.0)), Constant(np.array(0.0)), node_ranges=[(0, 2), (5, 7)])
    )
    assert got == (
        r"\begin{cases} 1 & \text{if } k \in [0, 2) \cup [5, 7) "
        r"\\ 0 & \text{otherwise} \end{cases}"
    )


# =============================================================================
# Spatial (SSM, SSMP, QDCM)
# =============================================================================


def test_ssm_cross_product_matrix():
    w = _sliced_state("w", 3)
    assert lower(SSM(w)) == r"\left[ x_{w} \right]_{\times}"


def test_ssmp_omega():
    w = _sliced_state("w", 3)
    assert lower(SSMP(w)) == r"\Omega\left( x_{w} \right)"


def test_qdcm():
    q = _sliced_state("q", 4)
    assert lower(QDCM(q)) == r"C\left( x_{q} \right)"


# =============================================================================
# Lie (SO3/SE3 exp & log, adjoints)
# =============================================================================


def test_so3_exp_and_log():
    w = _sliced_state("w", 3)
    R = Parameter("R", shape=(3, 3), value=np.eye(3))
    assert lower(SO3Exp(w)) == r"\operatorname{Exp}_{SO(3)}\left( x_{w} \right)"
    assert lower(SO3Log(R)) == r"\operatorname{Log}_{SO(3)}\left( R \right)"


def test_se3_exp_and_log():
    xi = _sliced_state("xi", 6)
    T = Parameter("T", shape=(4, 4), value=np.eye(4))
    assert lower(SE3Exp(xi)) == r"\operatorname{Exp}_{SE(3)}\left( x_{\xi} \right)"
    assert lower(SE3Log(T)) == r"\operatorname{Log}_{SE(3)}\left( T \right)"


def test_little_adjoint_and_coadjoint():
    a = _sliced_state("a", 6)
    b = _sliced_state("b", 6)
    assert lower(Adjoint(a, b)) == r"\operatorname{ad}_{x_{a}}\left( x_{b} \right)"
    assert lower(AdjointDual(a, b)) == r"\operatorname{ad}^{*}_{x_{a}}\left( x_{b} \right)"


def test_big_adjoint_and_coadjoint():
    T = Parameter("T", shape=(4, 4), value=np.eye(4))
    assert lower(SE3Adjoint(T)) == r"\operatorname{Ad}_{SE(3)}\left( T \right)"
    assert lower(SE3AdjointDual(T)) == r"\operatorname{Ad}^{*}_{SE(3)}\left( T \right)"


# =============================================================================
# STL (propositional and temporal)
# =============================================================================


def _ball(radius):
    pos = _sliced_state("pos", 2)
    return Norm(pos) <= Constant(np.array(float(radius)))


def test_stl_or_nary():
    got = lower(stl.Or(_ball(1), _ball(2), _ball(3)))
    assert got == (
        r"\left( \left\| x_{\mathrm{pos}} \right\| \le 1 \right) \vee "
        r"\left( \left\| x_{\mathrm{pos}} \right\| \le 2 \right) \vee "
        r"\left( \left\| x_{\mathrm{pos}} \right\| \le 3 \right)"
    )


def test_stl_and_nary():
    got = lower(stl.And(_ball(1), _ball(2)))
    assert got == (
        r"\left( \left\| x_{\mathrm{pos}} \right\| \le 1 \right) \wedge "
        r"\left( \left\| x_{\mathrm{pos}} \right\| \le 2 \right)"
    )


def test_stl_not():
    assert lower(stl.Not(_ball(1))) == (
        r"\neg \left( \left\| x_{\mathrm{pos}} \right\| \le 1 \right)"
    )


def test_stl_ifthen():
    got = lower(stl.IfThen(_ball(1), _ball(2)))
    assert got == (
        r"\left( \left\| x_{\mathrm{pos}} \right\| \le 1 \right) \implies "
        r"\left( \left\| x_{\mathrm{pos}} \right\| \le 2 \right)"
    )


def test_stl_integer_variable_membership():
    g = _sliced_state("g", 1)
    assert lower(stl.IntegerVariable(g, [1, 2, 3, 4])) == (
        r"x_{g} \in \left\{ 1, 2, 3, 4 \right\}"
    )


def test_stl_always_with_bounded_interval():
    assert lower(stl.Always(_ball(1), (0, 5))) == (
        r"\Box_{[0, 5]} \left( \left\| x_{\mathrm{pos}} \right\| \le 1 \right)"
    )


def test_stl_always_unbounded_omits_subscript():
    # A nested/interval-free Always renders no interval subscript.
    assert lower(stl.Always(_ball(1))) == (
        r"\Box \left( \left\| x_{\mathrm{pos}} \right\| \le 1 \right)"
    )


def test_stl_eventually_with_interval():
    assert lower(stl.Eventually(_ball(1), (2, 8))) == (
        r"\Diamond_{[2, 8]} \left( \left\| x_{\mathrm{pos}} \right\| \le 1 \right)"
    )


def test_stl_until_with_interval():
    got = lower(stl.Until(_ball(1), _ball(2), (1, 4)))
    assert got == (
        r"\left( \left\| x_{\mathrm{pos}} \right\| \le 1 \right) \, "
        r"\mathcal{U}_{[1, 4]} \, "
        r"\left( \left\| x_{\mathrm{pos}} \right\| \le 2 \right)"
    )


# =============================================================================
# Vmap
# =============================================================================


def test_vmap_renders_body_with_placeholder_box():
    position = _sliced_state("position", 3)
    vm = Vmap(lambda pt: Norm(position - pt), batch=np.eye(3))
    assert lower(vm) == (
        r"\operatorname{vmap}\left( \left\| x_{\mathrm{position}} - \square \right\| \right)"
    )


# =============================================================================
# latex_symbol
# =============================================================================


def test_latex_symbol_single_letter():
    assert latex_symbol("x") == "x"


def test_latex_symbol_greek_word():
    assert latex_symbol("alpha") == r"\alpha"
    assert latex_symbol("Omega") == r"\Omega"


def test_latex_symbol_base_subscript():
    assert latex_symbol("x_pos") == r"x_{\mathrm{pos}}"
    assert latex_symbol("v_max") == r"v_{\mathrm{max}}"


def test_latex_symbol_greek_base_subscript():
    assert latex_symbol("theta_dot") == r"\theta_{\mathrm{dot}}"


def test_latex_symbol_multi_letter():
    assert latex_symbol("position") == r"\mathrm{position}"


# =============================================================================
# merge_subscript
# =============================================================================


def test_merge_subscript_appends_when_no_subscript_group():
    # A base with no trailing _{...} group gets a fresh subscript.
    assert merge_subscript(r"\theta", 0) == r"\theta_{0}"
    assert merge_subscript("x", 3) == "x_{3}"


def test_merge_subscript_splices_into_existing_group():
    # A trailing _{...} group is extended with a comma-joined subscript.
    assert merge_subscript(r"x_{\mathrm{velocity}}", 0) == r"x_{\mathrm{velocity},0}"


def test_merge_subscript_handles_nested_brace_base():
    # The closing brace is matched by depth-scanning, so nested groups inside
    # the subscript don't confuse the splice point.
    assert (
        merge_subscript(r"x_{\lambda_{\mathrm{a}}}", 1) == r"x_{\lambda_{\mathrm{a}},1}"
    )


def test_merge_subscript_appends_after_non_subscript_group():
    # A trailing brace group that is not a subscript (an accent, \mathrm{...})
    # appends a new subscript instead of splicing into it.
    assert merge_subscript(r"\dot{x}", 0) == r"\dot{x}_{0}"
    assert merge_subscript(r"\mathrm{position}", 1) == r"\mathrm{position}_{1}"


# =============================================================================
# format_constant
# =============================================================================


def test_format_constant_scalar():
    assert format_constant(np.array(0.01)) == "0.01"
    assert format_constant(np.array(5.0)) == "5"


def test_format_constant_vector_column_bmatrix():
    assert format_constant(np.array([0.0, 0.0])) == r"\begin{bmatrix} 0 \\ 0 \end{bmatrix}"


def test_format_constant_matrix_bmatrix():
    got = format_constant(np.eye(2))
    assert got == r"\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}"


def test_format_constant_large_vector_placeholder():
    assert format_constant(np.arange(10.0)) == r"\mathrm{const} \in \mathbb{R}^{10}"


def test_format_constant_large_matrix_placeholder():
    assert format_constant(np.zeros((8, 3))) == r"\mathrm{const} \in \mathbb{R}^{8 \times 3}"


# =============================================================================
# Unregistered nodes
# =============================================================================


class _UnregisteredExpr(Expr):
    """A stand-in node type with no registered LaTeX visitor.

    Every shipped node type now has a visitor, so the fallback is exercised
    with a synthesized subclass — this is exactly the path a future,
    not-yet-supported node takes.
    """


def test_unregistered_node_raises_not_implemented():
    node = _UnregisteredExpr()
    with pytest.raises(NotImplementedError) as excinfo:
        lower(node)
    msg = str(excinfo.value)
    assert "LatexLowerer" in msg
    assert "_UnregisteredExpr" in msg


# =============================================================================
# to_latex entry point
# =============================================================================


def test_to_latex_single_expr_returns_str():
    x = _sliced_state("x", 3)
    result = to_latex(Norm(x) - 5.0)
    assert isinstance(result, str)
    assert result == r"\left\| x \right\| - 5"


def test_to_latex_sequence_returns_list_in_order():
    x = _sliced_state("x", 3)
    results = to_latex([Norm(x), Sum(x)])
    assert isinstance(results, list)
    assert results == [r"\left\| x \right\|", r"\sum x"]


def test_to_latex_exported_from_top_level_package():
    import openscvx as ox

    assert ox.to_latex is to_latex
    assert ox.to_latex(Time()) == "t"


# =============================================================================
# Expr._repr_latex_
# =============================================================================


def test_repr_latex_wraps_supported_node_in_dollars():
    x = _sliced_state("x", 3)
    assert (Norm(x) - 5.0)._repr_latex_() == r"$\left\| x \right\| - 5$"


def test_repr_latex_returns_none_for_unsupported_node():
    node = _UnregisteredExpr()
    assert node._repr_latex_() is None
