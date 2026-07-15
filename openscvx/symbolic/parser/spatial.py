"""Parser handlers for spatial / 6-DOF operations.

Handlers: QDCM, SSM, SSMP

Each handler is registered under its function name via ``@function`` and turns the
call-syntax form (e.g. ``QDCM(q)``) that the Pratt parser encounters in an
expression string into the corresponding 6-DOF ``Expr`` node — the
quaternion-to-DCM conversion and the skew-symmetric matrices.
"""

from openscvx.symbolic.expr.spatial import QDCM, SSM, SSMP
from openscvx.symbolic.parser._registry import function


@function("QDCM")
def _parse_qdcm(args, kwargs):
    if len(args) != 1:
        raise ValueError("QDCM() takes exactly 1 argument (quaternion)")
    return QDCM(args[0])


@function("SSM")
def _parse_ssm(args, kwargs):
    if len(args) != 1:
        raise ValueError("SSM() takes exactly 1 argument (3D vector)")
    return SSM(args[0])


@function("SSMP")
def _parse_ssmp(args, kwargs):
    if len(args) != 1:
        raise ValueError("SSMP() takes exactly 1 argument (angular velocity)")
    return SSMP(args[0])
