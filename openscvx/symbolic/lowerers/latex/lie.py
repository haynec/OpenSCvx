"""LaTeX visitors for Lie-group and Lie-algebra expressions.

Visitors: SO3Exp, SO3Log, SE3Exp, SE3Log, Adjoint, AdjointDual,
          SE3Adjoint, SE3AdjointDual

Notation follows the standard screw-theory / Lie-group conventions used in
robotics texts (Lynch & Park, *Modern Robotics*; Murray, Li & Sastry): the
group exp/log maps are ``\\operatorname{Exp}_{G}`` / ``\\operatorname{Log}_{G}``;
the big Adjoint of a group element is ``\\operatorname{Ad}`` and the little
adjoint (Lie bracket) of an algebra element is ``\\operatorname{ad}``; a dual
(coadjoint) operator carries a ``{}^{*}`` superscript.
"""

from openscvx.symbolic.expr.lie.adjoint import (
    Adjoint,
    AdjointDual,
    SE3Adjoint,
    SE3AdjointDual,
)
from openscvx.symbolic.expr.lie.se3 import SE3Exp, SE3Log
from openscvx.symbolic.expr.lie.so3 import SO3Exp, SO3Log
from openscvx.symbolic.lowerers.latex._registry import visitor


@visitor(SO3Exp)
def _visit_so3_exp(lowerer, node: SO3Exp):
    """Render the SO(3) exponential map as ``\\operatorname{Exp}_{SO(3)}(w)``."""
    return rf"\operatorname{{Exp}}_{{SO(3)}}\left( {lowerer.lower(node.omega)} \right)"


@visitor(SO3Log)
def _visit_so3_log(lowerer, node: SO3Log):
    """Render the SO(3) logarithm map as ``\\operatorname{Log}_{SO(3)}(R)``."""
    return rf"\operatorname{{Log}}_{{SO(3)}}\left( {lowerer.lower(node.rotation)} \right)"


@visitor(SE3Exp)
def _visit_se3_exp(lowerer, node: SE3Exp):
    """Render the SE(3) exponential map as ``\\operatorname{Exp}_{SE(3)}(\\xi)``."""
    return rf"\operatorname{{Exp}}_{{SE(3)}}\left( {lowerer.lower(node.twist)} \right)"


@visitor(SE3Log)
def _visit_se3_log(lowerer, node: SE3Log):
    """Render the SE(3) logarithm map as ``\\operatorname{Log}_{SE(3)}(T)``."""
    return rf"\operatorname{{Log}}_{{SE(3)}}\left( {lowerer.lower(node.transform)} \right)"


@visitor(Adjoint)
def _visit_adjoint(lowerer, node: Adjoint):
    """Render the little adjoint (Lie bracket) as ``\\operatorname{ad}_{\\xi_1}(\\xi_2)``.

    :class:`Adjoint` is the algebra-level ``\\operatorname{ad}_{\\xi_1}(\\xi_2) =
    [\\xi_1, \\xi_2]`` (see the class docstring), so it takes the lowercase
    ``\\operatorname{ad}`` with the acting twist as the subscript.
    """
    return (
        rf"\operatorname{{ad}}_{{{lowerer.lower(node.twist1)}}}"
        rf"\left( {lowerer.lower(node.twist2)} \right)"
    )


@visitor(AdjointDual)
def _visit_adjoint_dual(lowerer, node: AdjointDual):
    """Render the little coadjoint as ``\\operatorname{ad}^{*}_{\\xi}(\\mu)``.

    :class:`AdjointDual` is the algebra-level coadjoint ``\\operatorname{ad}^{*}``
    acting on a momentum covector, related to the adjoint by
    ``\\operatorname{ad}^{*}_{\\xi} = -(\\operatorname{ad}_{\\xi})^{\\top}`` (class
    docstring), so it carries the dual ``{}^{*}`` superscript.
    """
    return (
        rf"\operatorname{{ad}}^{{*}}_{{{lowerer.lower(node.twist)}}}"
        rf"\left( {lowerer.lower(node.momentum)} \right)"
    )


@visitor(SE3Adjoint)
def _visit_se3_adjoint(lowerer, node: SE3Adjoint):
    """Render the big SE(3) Adjoint matrix as ``\\operatorname{Ad}_{SE(3)}(T)``.

    :class:`SE3Adjoint` is the group-level ``\\operatorname{Ad}_T`` (6x6 matrix)
    that transforms twists between frames (class docstring), hence the capital
    ``\\operatorname{Ad}``.
    """
    return rf"\operatorname{{Ad}}_{{SE(3)}}\left( {lowerer.lower(node.transform)} \right)"


@visitor(SE3AdjointDual)
def _visit_se3_adjoint_dual(lowerer, node: SE3AdjointDual):
    """Render the big SE(3) coadjoint matrix as ``\\operatorname{Ad}^{*}_{SE(3)}(T)``.

    :class:`SE3AdjointDual` is the group-level coadjoint ``\\operatorname{Ad}^{*}_T
    = (\\operatorname{Ad}_T)^{-\\top}`` that transforms wrenches between frames
    (class docstring), so it carries the dual ``{}^{*}`` superscript.
    """
    return rf"\operatorname{{Ad}}^{{*}}_{{SE(3)}}\left( {lowerer.lower(node.transform)} \right)"
