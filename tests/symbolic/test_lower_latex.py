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
)
from openscvx.symbolic.expr.constraint import CTCS, Equality, Inequality, NodalConstraint
from openscvx.symbolic.expr.lie.so3 import SO3Exp
from openscvx.symbolic.lower import to_latex
from openscvx.symbolic.lowerers.latex import LatexLowerer, format_constant, latex_symbol


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


def test_state_single_letter_renders_bare():
    assert lower(_sliced_state("x", 3)) == "x"


def test_state_multi_letter_uses_mathrm():
    assert lower(_sliced_state("pos", 3)) == r"\mathrm{pos}"


def test_control_renders_symbol():
    assert lower(_sliced_control("u", 2)) == "u"


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
    assert lower(pos.at(5)) == r"\mathrm{pos}^{(5)}"


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


def test_concat_column_bmatrix():
    x = _sliced_state("x", 1)
    y = Constant(np.array(1.0))
    assert lower(Concat(x, y)) == r"\begin{bmatrix} x \\ 1 \end{bmatrix}"


def test_hstack_row_bmatrix():
    x = _sliced_state("x", 1)
    y = _sliced_state("y", 1)
    assert lower(Hstack([x, y])) == r"\begin{bmatrix} x & y \end{bmatrix}"


def test_stack_and_vstack_column_bmatrix():
    x = _sliced_state("x", 1)
    y = _sliced_state("y", 1)
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
    y = _sliced_state("y", 1)
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
    y = _sliced_state("y", 1)
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
    assert lower(cnc) == r"\mathrm{pos}^{(5)} - \mathrm{pos}^{(4)} \le 0.1"


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


def test_unregistered_node_raises_not_implemented():
    node = SO3Exp(Constant(np.array([0.0, 0.0, 1.0])))
    with pytest.raises(NotImplementedError) as excinfo:
        lower(node)
    msg = str(excinfo.value)
    assert "LatexLowerer" in msg
    assert "SO3Exp" in msg


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
    node = SO3Exp(Constant(np.array([0.0, 0.0, 1.0])))
    assert node._repr_latex_() is None
