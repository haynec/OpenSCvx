"""SE(3) Lie group operations for rigid body transformations.

This module provides exponential and logarithm maps for the SE(3) rigid
transformation group, enabling twist to transformation matrix conversions
and vice versa. These are essential for Product of Exponentials (PoE)
forward kinematics in robotic manipulators.

Requires jaxlie: pip install openscvx[lie]

Note:
    The twist convention [v; ω] (linear first, angular second) matches jaxlie's
    SE3 tangent parameterization, so no reordering is needed during lowering.
"""

from typing import Tuple, Union

import jaxlie  # noqa: F401 - validates jaxlie is installed
import numpy as np

from ..expr import Expr, to_expr


class SE3Exp(Expr):
    """Exponential map from se(3) twist to SE(3) transformation matrix.

    Maps a 6D twist vector to a 4×4 homogeneous transformation matrix.
    Uses jaxlie for numerically robust implementation with proper handling
    of small angles and translations.

    The twist ξ = [v; ω] follows the convention:

    - v: 3D linear velocity component
    - ω: 3D angular velocity component

    This is the key operation for Product of Exponentials (PoE) forward
    kinematics in robotic manipulators.

    Attributes:
        twist: 6D twist vector [v; ω] with shape (6,)

    Example:
        Product of Exponentials forward kinematics::

            import openscvx as ox
            import numpy as np

            # Screw axis for revolute joint about z-axis at origin
            screw_axis = np.array([0, 0, 0, 0, 0, 1])  # [v; ω]
            theta = ox.State("theta", shape=(1,))

            # Joint transformation
            T = ox.lie.SE3Exp(ox.Constant(screw_axis) * theta)  # 4×4 matrix

            # Chain multiple joints
            T_01 = ox.lie.SE3Exp(screw1 * q1)
            T_12 = ox.lie.SE3Exp(screw2 * q2)
            T_02 = T_01 @ T_12

        Extract position from transformation::

            T_ee = forward_kinematics(joint_angles)
            p_ee = T_ee[:3, 3]  # End-effector position

    Note:
        The twist convention [v; ω] matches jaxlie's SE3 tangent
        parameterization, so no reordering is performed.

    See Also:
        - SE3Log: Inverse operation (transformation matrix to twist)
        - SO3Exp: Rotation-only exponential map
        - AdjointDual: For dynamics computations with twists
    """

    def __init__(self, twist: Union[Expr, float, int, np.ndarray]):
        """Initialize SE3 exponential map.

        Args:
            twist: 6D twist vector [v; ω] with shape (6,)
        """
        self.twist = to_expr(twist)

    def children(self):
        return [self.twist]

    def canonicalize(self) -> "Expr":
        twist = self.twist.canonicalize()
        return SE3Exp(twist)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that input is a 6D vector and return output shape.

        Returns:
            tuple: Shape (4, 4) for the homogeneous transformation matrix

        Raises:
            ValueError: If twist does not have shape (6,)
        """
        twist_shape = self.twist.check_shape()
        if twist_shape != (6,):
            raise ValueError(f"SE3Exp expects twist with shape (6,), got {twist_shape}")
        return (4, 4)

    def __repr__(self) -> str:
        return f"SE3Exp({self.twist!r})"


class SE3Log(Expr):
    """Logarithm map from SE(3) transformation matrix to se(3) twist.

    Maps a 4×4 homogeneous transformation matrix to a 6D twist vector.
    Uses jaxlie for numerically robust implementation.

    The output twist ξ = [v; ω] follows the convention:

    - v: 3D linear component
    - ω: 3D angular component (rotation vector)

    This is useful for computing error metrics between poses in optimization.

    Attributes:
        transform: 4×4 homogeneous transformation matrix with shape (4, 4)

    Example:
        Compute pose error for trajectory optimization::

            import openscvx as ox

            T_current = forward_kinematics(q)
            T_target = ox.Parameter("T_target", shape=(4, 4), value=goal_pose)

            # Relative transformation
            T_error = ox.linalg.inv(T_target) @ T_current

            # Convert to twist for error metric
            twist_error = ox.lie.SE3Log(T_error)
            pose_cost = ox.linalg.Norm(twist_error) ** 2

    See Also:
        - SE3Exp: Inverse operation (twist to transformation matrix)
        - SO3Log: Rotation-only logarithm map
    """

    def __init__(self, transform: Union[Expr, float, int, np.ndarray]):
        """Initialize SE3 logarithm map.

        Args:
            transform: 4×4 homogeneous transformation matrix with shape (4, 4)
        """
        self.transform = to_expr(transform)

    def children(self):
        return [self.transform]

    def canonicalize(self) -> "Expr":
        transform = self.transform.canonicalize()
        return SE3Log(transform)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that input is a 4×4 matrix and return output shape.

        Returns:
            tuple: Shape (6,) for the twist vector

        Raises:
            ValueError: If transform does not have shape (4, 4)
        """
        transform_shape = self.transform.check_shape()
        if transform_shape != (4, 4):
            raise ValueError(f"SE3Log expects transform with shape (4, 4), got {transform_shape}")
        return (6,)

    def __repr__(self) -> str:
        return f"SE3Log({self.transform!r})"
