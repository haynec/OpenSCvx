"""LaTeX visitors for spatial (6-DOF) expressions.

Visitors: QDCM, SSMP, SSM
"""

from openscvx.symbolic.expr.spatial import QDCM, SSM, SSMP
from openscvx.symbolic.lowerers.latex._registry import visitor


@visitor(SSM)
def _visit_ssm(lowerer, node: SSM):
    """Render the 3x3 skew-symmetric (cross-product) matrix as ``[w]_{\\times}``.

    ``[\\,\\cdot\\,]_{\\times}`` is the standard notation for the hat/cross-product
    map that sends a 3-vector ``w`` to the matrix with ``[w]_{\\times} v = w
    \\times v`` (Murray, Li & Sastry).
    """
    return rf"\left[ {lowerer.lower(node.w)} \right]_{{\times}}"


@visitor(SSMP)
def _visit_ssmp(lowerer, node: SSMP):
    """Render the 4x4 quaternion-kinematics matrix as ``\\Omega(w)``.

    ``\\Omega(\\omega)`` is the conventional symbol for the skew matrix in the
    quaternion kinematic equation ``\\dot q = \\tfrac{1}{2}\\,\\Omega(\\omega)\\,q``
    (see the class docstring), so the rendering matches the equation it appears
    in.
    """
    return rf"\Omega\left( {lowerer.lower(node.w)} \right)"


@visitor(QDCM)
def _visit_qdcm(lowerer, node: QDCM):
    """Render the quaternion-to-DCM conversion as ``C(q)``.

    ``C(q)`` (or ``R(q)``) is the standard notation for the direction cosine
    matrix / rotation matrix parameterized by a quaternion ``q``; ``C`` reads
    unambiguously as "the DCM of ``q``".
    """
    return rf"C\left( {lowerer.lower(node.q)} \right)"
