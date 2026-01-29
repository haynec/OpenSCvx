"""Adjoint and coadjoint operators for rigid body dynamics.

This module provides the core Lie algebra operators for 6-DOF rigid body
dynamics. These operators require no external dependencies and work with
the standard openscvx installation.

The module uses the following conventions:
    - Twist (spatial velocity): ξ = [v; ω] where v ∈ ℝ³ is linear velocity
      and ω ∈ ℝ³ is angular velocity (both in body frame)
    - Wrench (spatial force): F = [f; τ] where f ∈ ℝ³ is force and τ ∈ ℝ³
      is torque (both in body frame)
    - Momentum: μ = [p; L] where p ∈ ℝ³ is linear momentum and L ∈ ℝ³
      is angular momentum
"""

from typing import Tuple, Union

import numpy as np

from ..expr import Expr, to_expr


class AdjointDual(Expr):
    """Coadjoint operator ad* for computing Coriolis and centrifugal forces.

    Computes the coadjoint action ad*_ξ(μ) which represents the rate of change
    of momentum due to body rotation. This is the key term in Newton-Euler
    dynamics that captures Coriolis and centrifugal effects.

    For se(3), given twist ξ = [v; ω] and momentum μ = [f; τ]:

        ad*_ξ(μ) = [ ω × f + v × τ ]
                   [     ω × τ     ]

    This appears in the Newton-Euler equations as:

        M @ ξ_dot = F_ext - ad*_ξ(M @ ξ)

    where M is the spatial inertia matrix and F_ext is the external wrench.

    Attributes:
        twist: 6D twist vector [v; ω] (linear velocity, angular velocity)
        momentum: 6D momentum vector [p; L] or [f; τ] (linear, angular)

    Example:
        Compute the bias force (Coriolis + centrifugal) for rigid body dynamics::

            import openscvx as ox

            twist = ox.State("twist", shape=(6,))
            M = ox.Parameter("M", shape=(6, 6), value=inertia_matrix)

            momentum = M @ twist
            bias_force = ox.lie.AdjointDual(twist, momentum)

            # In dynamics: twist_dot = M_inv @ (wrench - bias_force)

    Note:
        The coadjoint is related to the adjoint by: ad*_ξ = -(ad_ξ)^T

        For the special case of pure rotation (v=0) with diagonal inertia,
        the angular part reduces to the familiar ω × (J @ ω) term.

    See Also:
        Adjoint: The adjoint operator for twist-on-twist action
        SSM: 3x3 skew-symmetric matrix for cross products
    """

    def __init__(
        self,
        twist: Union[Expr, float, int, np.ndarray],
        momentum: Union[Expr, float, int, np.ndarray],
    ):
        """Initialize a coadjoint operator.

        Args:
            twist: 6D twist vector [v; ω] with shape (6,)
            momentum: 6D momentum vector [p; L] with shape (6,)
        """
        self.twist = to_expr(twist)
        self.momentum = to_expr(momentum)

    def children(self):
        return [self.twist, self.momentum]

    def canonicalize(self) -> "Expr":
        twist = self.twist.canonicalize()
        momentum = self.momentum.canonicalize()
        return AdjointDual(twist, momentum)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that inputs are 6D vectors and return output shape.

        Returns:
            tuple: Shape (6,) for the resulting coadjoint vector

        Raises:
            ValueError: If twist or momentum do not have shape (6,)
        """
        twist_shape = self.twist.check_shape()
        momentum_shape = self.momentum.check_shape()

        if twist_shape != (6,):
            raise ValueError(f"AdjointDual expects twist with shape (6,), got {twist_shape}")
        if momentum_shape != (6,):
            raise ValueError(f"AdjointDual expects momentum with shape (6,), got {momentum_shape}")

        return (6,)

    def __repr__(self) -> str:
        return f"ad_dual({self.twist!r}, {self.momentum!r})"


class Adjoint(Expr):
    """Adjoint operator ad (Lie bracket) for twist-on-twist action.

    Computes the adjoint action ad_ξ₁(ξ₂) which represents the Lie bracket
    [ξ₁, ξ₂] of two twists. This is used for velocity propagation in
    kinematic chains and acceleration computations.

    For se(3), given twists ξ₁ = [v₁; ω₁] and ξ₂ = [v₂; ω₂]:

        ad_ξ₁(ξ₂) = [ξ₁, ξ₂] = [ ω₁ × v₂ - ω₂ × v₁ ]
                                [     ω₁ × ω₂       ]

    Equivalently using the adjoint matrix:

        ad_ξ = [ [ω]×    0   ]
               [ [v]×   [ω]× ]

    where [·]× denotes the 3x3 skew-symmetric (cross product) matrix.

    Attributes:
        twist1: First 6D twist vector [v₁; ω₁]
        twist2: Second 6D twist vector [v₂; ω₂]

    Example:
        Compute the Lie bracket of two twists::

            import openscvx as ox

            twist1 = ox.State("twist1", shape=(6,))
            twist2 = ox.State("twist2", shape=(6,))

            bracket = ox.lie.Adjoint(twist1, twist2)

        Velocity propagation in a kinematic chain::

            # Child link velocity includes parent velocity plus relative motion
            # V_child = Ad_T @ V_parent + joint_twist * q_dot

    Note:
        The adjoint satisfies the Jacobi identity and is antisymmetric:
        ad_ξ₁(ξ₂) = -ad_ξ₂(ξ₁)

    See Also:
        AdjointDual: The coadjoint operator for momentum dynamics
    """

    def __init__(
        self,
        twist1: Union[Expr, float, int, np.ndarray],
        twist2: Union[Expr, float, int, np.ndarray],
    ):
        """Initialize an adjoint operator.

        Args:
            twist1: First 6D twist vector [v; ω] with shape (6,)
            twist2: Second 6D twist vector [v; ω] with shape (6,)
        """
        self.twist1 = to_expr(twist1)
        self.twist2 = to_expr(twist2)

    def children(self):
        return [self.twist1, self.twist2]

    def canonicalize(self) -> "Expr":
        twist1 = self.twist1.canonicalize()
        twist2 = self.twist2.canonicalize()
        return Adjoint(twist1, twist2)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that inputs are 6D vectors and return output shape.

        Returns:
            tuple: Shape (6,) for the resulting Lie bracket

        Raises:
            ValueError: If either twist does not have shape (6,)
        """
        twist1_shape = self.twist1.check_shape()
        twist2_shape = self.twist2.check_shape()

        if twist1_shape != (6,):
            raise ValueError(f"Adjoint expects twist1 with shape (6,), got {twist1_shape}")
        if twist2_shape != (6,):
            raise ValueError(f"Adjoint expects twist2 with shape (6,), got {twist2_shape}")

        return (6,)

    def __repr__(self) -> str:
        return f"ad({self.twist1!r}, {self.twist2!r})"


class SE3Adjoint(Expr):
    """SE(3) Adjoint representation Ad_T for transforming twists between frames.

    Computes the 6×6 adjoint matrix Ad_T that transforms twists from one
    coordinate frame to another. Given a transformation T_ab from frame A to
    frame B, the adjoint transforms a twist expressed in frame A to frame B:

        ξ_b = Ad_{T_ab} @ ξ_a

    For SE(3), given T with rotation R and translation p:

        Ad_T = [ R      0   ]
               [ [p]×R  R   ]

    where [p]× is the 3×3 skew-symmetric matrix of p.

    This is essential for:

    - Velocity propagation through kinematic chains
    - Computing geometric Jacobians for manipulators
    - Recursive Newton-Euler dynamics algorithms

    Attributes:
        transform: 4×4 homogeneous transformation matrix

    Example:
        Transform a body twist to the world frame::

            import openscvx as ox

            T_world_body = forward_kinematics(q)  # 4×4 transform
            twist_body = ox.State("twist_body", shape=(6,))

            # Transform twist to world frame
            Ad_T = ox.lie.SE3Adjoint(T_world_body)  # 6×6 matrix
            twist_world = Ad_T @ twist_body

        Compute geometric Jacobian columns::

            # Each column of the geometric Jacobian is Ad_{T_0i} @ ξ_i
            J_col_i = ox.lie.SE3Adjoint(T_0_to_i) @ screw_axis_i

    Note:
        The adjoint satisfies: Ad_{T1 @ T2} = Ad_{T1} @ Ad_{T2}

    See Also:
        - SE3AdjointDual: For transforming wrenches between frames
        - Adjoint: The small adjoint (Lie bracket) for twist-on-twist action
    """

    def __init__(self, transform: Union[Expr, float, int, np.ndarray]):
        """Initialize SE3 Adjoint operator.

        Args:
            transform: 4×4 homogeneous transformation matrix with shape (4, 4)
        """
        self.transform = to_expr(transform)

    def children(self):
        return [self.transform]

    def canonicalize(self) -> "Expr":
        transform = self.transform.canonicalize()
        return SE3Adjoint(transform)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that input is a 4×4 matrix and return output shape.

        Returns:
            tuple: Shape (6, 6) for the adjoint matrix

        Raises:
            ValueError: If transform does not have shape (4, 4)
        """
        transform_shape = self.transform.check_shape()
        if transform_shape != (4, 4):
            raise ValueError(
                f"SE3Adjoint expects transform with shape (4, 4), got {transform_shape}"
            )
        return (6, 6)

    def __repr__(self) -> str:
        return f"Ad({self.transform!r})"


class SE3AdjointDual(Expr):
    """SE(3) coadjoint representation Ad*_T for transforming wrenches between frames.

    Computes the 6×6 coadjoint matrix Ad*_T that transforms wrenches from one
    coordinate frame to another. Given a transformation T_ab from frame A to
    frame B, the coadjoint transforms a wrench expressed in frame B to frame A:

        F_a = Ad*_{T_ab} @ F_b

    For SE(3), given T with rotation R and translation p:

        Ad*_T = [ R     [p]×R ]
                [ 0       R   ]

    This is the transpose-inverse of Ad_T: Ad*_T = (Ad_T)^{-T}

    This is essential for:

    - Force/torque propagation in dynamics
    - Transforming wrenches between end-effector and base frames
    - Recursive Newton-Euler dynamics algorithms

    Attributes:
        transform: 4×4 homogeneous transformation matrix

    Example:
        Transform a wrench from end-effector to base frame::

            import openscvx as ox

            T_base_ee = forward_kinematics(q)  # 4×4 transform
            wrench_ee = ox.Control("wrench_ee", shape=(6,))

            # Transform wrench to base frame
            Ad_star_T = ox.lie.SE3AdjointDual(T_base_ee)  # 6×6 matrix
            wrench_base = Ad_star_T @ wrench_ee

    Note:
        The coadjoint is related to the adjoint by: Ad*_T = (Ad_T)^{-T}

    See Also:
        - SE3Adjoint: For transforming twists between frames
        - AdjointDual: The small coadjoint for Coriolis/centrifugal forces
    """

    def __init__(self, transform: Union[Expr, float, int, np.ndarray]):
        """Initialize SE3 coadjoint operator.

        Args:
            transform: 4×4 homogeneous transformation matrix with shape (4, 4)
        """
        self.transform = to_expr(transform)

    def children(self):
        return [self.transform]

    def canonicalize(self) -> "Expr":
        transform = self.transform.canonicalize()
        return SE3AdjointDual(transform)

    def check_shape(self) -> Tuple[int, ...]:
        """Check that input is a 4×4 matrix and return output shape.

        Returns:
            tuple: Shape (6, 6) for the coadjoint matrix

        Raises:
            ValueError: If transform does not have shape (4, 4)
        """
        transform_shape = self.transform.check_shape()
        if transform_shape != (4, 4):
            raise ValueError(
                f"SE3AdjointDual expects transform with shape (4, 4), got {transform_shape}"
            )
        return (6, 6)

    def __repr__(self) -> str:
        return f"Ad_dual({self.transform!r})"
