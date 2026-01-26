"""SO(3) Lie group operations for rotation matrices.

This module provides exponential and logarithm maps for the SO(3) rotation
group, enabling axis-angle to rotation matrix conversions and vice versa.

Requires jaxlie: pip install openscvx[lie]
"""

from typing import Tuple, Union

import jaxlie  # noqa: F401 - validates jaxlie is installed
import numpy as np

from ..expr import Expr, to_expr


class SO3Exp(Expr):
    """Exponential map from so(3) to SO(3) rotation matrix.

    Maps a 3D rotation vector (axis-angle representation) to a 3×3 rotation
    matrix using the Rodrigues formula. Uses jaxlie for numerically robust
    implementation with proper handling of small angles.

    The rotation vector ω has direction equal to the rotation axis and
    magnitude equal to the rotation angle in radians.

    Attributes:
        omega: 3D rotation vector with shape (3,)

    Example:
        Create a rotation about the z-axis::

            import openscvx as ox
            import numpy as np

            # 90 degree rotation about z
            omega = ox.Constant(np.array([0, 0, np.pi/2]))
            R = ox.lie.SO3Exp(omega)  # 3×3 rotation matrix

        Parameterized rotation for optimization::

            theta = ox.State("theta", shape=(1,))
            axis = ox.Constant(np.array([0, 0, 1]))  # z-axis
            R = ox.lie.SO3Exp(axis * theta)

    See Also:
        - SO3Log: Inverse operation (rotation matrix to rotation vector)
        - SE3Exp: Full rigid body transformation including translation
    """

    def __init__(self, omega: Union[Expr, float, int, np.ndarray]):
        """Initialize SO3 exponential map.

        Args:
            omega: 3D rotation vector (axis × angle) with shape (3,)
        """
        self.omega = to_expr(omega)

    def children(self):
        return [self.omega]

    def canonicalize(self) -> "Expr":
        omega = self.omega.canonicalize()
        return SO3Exp(omega)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that input is a 3D vector and return output shape.

        Returns:
            tuple: Shape (3, 3) for the rotation matrix

        Raises:
            ValueError: If omega does not have shape (3,)
        """
        omega_shape = self.omega.check_shape()
        if omega_shape != (3,):
            raise ValueError(f"SO3Exp expects omega with shape (3,), got {omega_shape}")
        return (3, 3)

    def __repr__(self) -> str:
        return f"SO3Exp({self.omega!r})"


class SO3Log(Expr):
    """Logarithm map from SO(3) rotation matrix to so(3) rotation vector.

    Maps a 3×3 rotation matrix to a 3D rotation vector (axis-angle
    representation). Uses jaxlie for numerically robust implementation.

    The output rotation vector ω has direction equal to the rotation axis
    and magnitude equal to the rotation angle in radians.

    Attributes:
        rotation: 3×3 rotation matrix with shape (3, 3)

    Example:
        Extract rotation vector from a rotation matrix::

            import openscvx as ox

            R = ox.State("R", shape=(3, 3))  # Rotation matrix state
            omega = ox.lie.SO3Log(R)  # 3D rotation vector

    See Also:
        - SO3Exp: Inverse operation (rotation vector to rotation matrix)
        - SE3Log: Full rigid body transformation logarithm
    """

    def __init__(self, rotation: Union[Expr, float, int, np.ndarray]):
        """Initialize SO3 logarithm map.

        Args:
            rotation: 3×3 rotation matrix with shape (3, 3)
        """
        self.rotation = to_expr(rotation)

    def children(self):
        return [self.rotation]

    def canonicalize(self) -> "Expr":
        rotation = self.rotation.canonicalize()
        return SO3Log(rotation)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that input is a 3×3 matrix and return output shape.

        Returns:
            tuple: Shape (3,) for the rotation vector

        Raises:
            ValueError: If rotation does not have shape (3, 3)
        """
        rotation_shape = self.rotation.check_shape()
        if rotation_shape != (3, 3):
            raise ValueError(f"SO3Log expects rotation with shape (3, 3), got {rotation_shape}")
        return (3,)

    def __repr__(self) -> str:
        return f"SO3Log({self.rotation!r})"
