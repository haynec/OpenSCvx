"""Spatial and 6-DOF utility operations for trajectory optimization.

This module provides efficient symbolic expression nodes for common 6-DOF (six degree of
freedom) operations used in aerospace and robotics applications. These operations directly
map to optimized JAX implementations for high-performance evaluation.
"""

from typing import Tuple, Union

import numpy as np

from .expr import Expr, to_expr


class QDCM(Expr):
    """Quaternion to Direction Cosine Matrix (DCM) conversion.

    Converts a unit quaternion representation to a 3x3 direction cosine matrix
    (also known as a rotation matrix). This operation is commonly used in 6-DOF
    spacecraft dynamics, aircraft simulation, and robotics applications.

    The quaternion is expected in scalar-last format: [qx, qy, qz, qw] where
    qw is the scalar component. The resulting DCM can be used to transform vectors
    from one reference frame to another.

    Attributes:
        q: Quaternion expression with shape (4,)

    Example:
        Use the QDCM to rotate a vector:

            import openscvx as ox
            q = ox.State("q", shape=(4,))
            dcm = ox.QDCM(q)  # Creates rotation matrix, shape (3, 3)
            v_body = ox.Variable("v_body", shape=(3,))
            v_inertial = dcm @ v_body

    Note:
        The input quaternion does not need to be normalized; the implementation
        automatically handles normalization during evaluation.
    """

    def __init__(self, q: Union[Expr, float, int, np.ndarray]):
        """Initialize a quaternion to DCM conversion.

        Args:
            q: Quaternion expression with shape (4,) in [qx, qy, qz, qw] format
        """
        self.q = to_expr(q)

    def children(self):
        return [self.q]

    def canonicalize(self) -> "Expr":
        q = self.q.canonicalize()
        return QDCM(q)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that input is a quaternion and return DCM shape.

        Returns:
            tuple: Shape (3, 3) for the resulting direction cosine matrix

        Raises:
            ValueError: If quaternion does not have shape (4,)
        """
        q_shape = self.q.check_shape()
        if q_shape != (4,):
            raise ValueError(f"QDCM expects quaternion with shape (4,), got {q_shape}")
        return (3, 3)

    def __repr__(self) -> str:
        return f"qdcm({self.q!r})"


class SSMP(Expr):
    """Angular rate to 4x4 skew-symmetric matrix for quaternion dynamics.

    Constructs the 4x4 skew-symmetric matrix Ω(ω) used in quaternion kinematic
    differential equations. This matrix relates angular velocity to the time
    derivative of the quaternion:

        q̇ = (1/2) * Ω(ω) @ q

    The resulting matrix has the form:
        ⎡  0   ωz  -ωy   ωx ⎤
        ⎢-ωz    0   ωx   ωy ⎥
        ⎢ ωy  -ωx    0   ωz ⎥
        ⎣-ωx  -ωy  -ωz    0 ⎦

    This is particularly useful for formulating quaternion-based attitude
    dynamics in spacecraft and aircraft trajectory optimization problems.

    Attributes:
        w: Angular velocity vector expression with shape (3,)

    Example:
        Use the SSMP to compute the quaternion derivative:

            import openscvx as ox
            omega = ox.Control("omega", shape=(3,))
            q = ox.State("q", shape=(4,))
            # Quaternion kinematic equation
            q_dot = 0.5 * ox.SSMP(omega) @ q

    See Also:
        SSM: 3x3 skew-symmetric matrix for cross product operations
    """

    def __init__(self, w: Union[Expr, float, int, np.ndarray]):
        """Initialize an angular velocity to skew-symmetric matrix conversion.

        Args:
            w: Angular velocity vector expression with shape (3,) in [ωx, ωy, ωz] format
        """
        self.w = to_expr(w)

    def children(self):
        return [self.w]

    def canonicalize(self) -> "Expr":
        w = self.w.canonicalize()
        return SSMP(w)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that input is a 3D angular velocity and return matrix shape.

        Returns:
            tuple: Shape (4, 4) for the resulting skew-symmetric matrix

        Raises:
            ValueError: If angular velocity does not have shape (3,)
        """
        w_shape = self.w.check_shape()
        if w_shape != (3,):
            raise ValueError(f"SSMP expects angular velocity with shape (3,), got {w_shape}")
        return (4, 4)

    def __repr__(self) -> str:
        return f"ssmp({self.w!r})"


class SSM(Expr):
    """Angular rate vector to 3x3 skew-symmetric matrix (cross product matrix).

    Constructs the 3x3 skew-symmetric matrix [ω]x that represents the cross
    product operation. For any 3D vector v, the cross product ω x v can be
    computed as the matrix-vector product [ω]x @ v.

    The resulting matrix has the form:
        ⎡  0  -ωz   ωy ⎤
        ⎢ ωz    0  -ωx ⎥
        ⎣-ωy   ωx    0 ⎦

    This operation is widely used in:
    - Rigid body dynamics (angular momentum calculations)
    - DCM time derivatives: Ṙ = [ω]x @ R
    - Velocity kinematics in robotics
    - Coriolis and centrifugal acceleration terms

    Attributes:
        w: Angular velocity or 3D vector expression with shape (3,)

    Example:
        Use the SSM to compute the rotation matrix derivative:

            import openscvx as ox
            omega = ox.Control("omega", shape=(3,))
            R = ox.State("R", shape=(3, 3))  # Direction cosine matrix
            # DCM time derivative
            R_dot = ox.SSM(omega) @ R

    Note:
        The skew-symmetric property ensures that [ω]xᵀ = -[ω]x, which is
        important for preserving orthogonality in DCM propagation.

    See Also:
        SSMP: 4x4 skew-symmetric matrix for quaternion dynamics
    """

    def __init__(self, w: Union[Expr, float, int, np.ndarray]):
        """Initialize a vector to skew-symmetric matrix conversion.

        Args:
            w: 3D vector expression with shape (3,) in [ωx, ωy, ωz] format
        """
        self.w = to_expr(w)

    def children(self):
        return [self.w]

    def canonicalize(self) -> "Expr":
        w = self.w.canonicalize()
        return SSM(w)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that input is a 3D vector and return matrix shape.

        Returns:
            tuple: Shape (3, 3) for the resulting skew-symmetric matrix

        Raises:
            ValueError: If input vector does not have shape (3,)
        """
        w_shape = self.w.check_shape()
        if w_shape != (3,):
            raise ValueError(f"SSM expects angular velocity with shape (3,), got {w_shape}")
        return (3, 3)

    def __repr__(self) -> str:
        return f"ssm({self.w!r})"
