"""JAX visitors for Lie group expressions.

Visitors: AdjointDual, Adjoint, SE3Adjoint, SE3AdjointDual,
          SO3Exp, SO3Log, SE3Exp, SE3Log
"""

import jax.numpy as jnp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.lie.adjoint import (
    Adjoint,
    AdjointDual,
    SE3Adjoint,
    SE3AdjointDual,
)
from openscvx.symbolic.expr.lie.se3 import SE3Exp, SE3Log
from openscvx.symbolic.expr.lie.so3 import SO3Exp, SO3Log
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(AdjointDual)
def _visit_adjoint_dual(lowerer, node: AdjointDual):
    """Lower coadjoint operator ad* for rigid body dynamics.

    Computes the coadjoint action ad*_ξ(μ) which represents Coriolis and
    centrifugal forces in rigid body dynamics. This is the key term in
    Newton-Euler equations.

    For se(3), given twist ξ = [v; ω] and momentum μ = [f; τ]:

        ad*_ξ(μ) = [ ω × f + v × τ ]
                    [     ω × τ     ]

    This appears in the equations of motion as:
        M @ ξ_dot = F_ext - ad*_ξ(M @ ξ)

    Args:
        node: AdjointDual expression node

    Returns:
        Function (x, u, node, params) -> 6D coadjoint result

    Note:
        Convention: twist = [v; ω] (linear velocity, angular velocity)
                    momentum = [f; τ] (force, torque)
    """
    f_twist = lowerer.lower(node.twist)
    f_momentum = lowerer.lower(node.momentum)

    def adjoint_dual_fn(x, u, node, params):
        twist = f_twist(x, u, node, params)
        momentum = f_momentum(x, u, node, params)

        # Extract components: twist = [v; ω], momentum = [f; τ]
        v = twist[:3]  # Linear velocity
        omega = twist[3:]  # Angular velocity
        f = momentum[:3]  # Force (or linear momentum)
        tau = momentum[3:]  # Torque (or angular momentum)

        # Coadjoint action: ad*_ξ(μ) = [ω × f + v × τ; ω × τ]
        linear_part = jnp.cross(omega, f) + jnp.cross(v, tau)
        angular_part = jnp.cross(omega, tau)

        return jnp.concatenate([linear_part, angular_part])

    return adjoint_dual_fn


@visitor(Adjoint)
def _visit_adjoint(lowerer, node: Adjoint):
    """Lower adjoint operator ad (Lie bracket) for twist-on-twist action.

    Computes the adjoint action ad_ξ₁(ξ₂) which represents the Lie bracket
    [ξ₁, ξ₂] of two twists. Used for velocity propagation and acceleration
    computation in kinematic chains.

    For se(3), given twists ξ₁ = [v₁; ω₁] and ξ₂ = [v₂; ω₂]:

        ad_ξ₁(ξ₂) = [ ω₁ × v₂ - ω₂ × v₁ ]
                    [     ω₁ × ω₂       ]

    Args:
        node: Adjoint expression node

    Returns:
        Function (x, u, node, params) -> 6D Lie bracket result

    Note:
        The Lie bracket is antisymmetric: [ξ₁, ξ₂] = -[ξ₂, ξ₁]
    """
    f_twist1 = lowerer.lower(node.twist1)
    f_twist2 = lowerer.lower(node.twist2)

    def adjoint_fn(x, u, node, params):
        twist1 = f_twist1(x, u, node, params)
        twist2 = f_twist2(x, u, node, params)

        # Extract components: twist = [v; ω]
        v1 = twist1[:3]
        omega1 = twist1[3:]
        v2 = twist2[:3]
        omega2 = twist2[3:]

        # Lie bracket: [ξ₁, ξ₂] = [ω₁ × v₂ - ω₂ × v₁; ω₁ × ω₂]
        linear_part = jnp.cross(omega1, v2) - jnp.cross(omega2, v1)
        angular_part = jnp.cross(omega1, omega2)

        return jnp.concatenate([linear_part, angular_part])

    return adjoint_fn


@visitor(SE3Adjoint)
def _visit_se3_adjoint(lowerer, node: SE3Adjoint):
    """Lower SE3 Adjoint (big Ad) for transforming twists between frames.

    Computes the 6×6 adjoint matrix Ad_T that transforms twists:
        ξ_b = Ad_{T_ab} @ ξ_a

    For SE(3) with rotation R and translation p:
        Ad_T = [ R      0   ]
                [ [p]×R  R   ]

    Args:
        node: SE3Adjoint expression node

    Returns:
        Function (x, u, node, params) -> 6×6 adjoint matrix
    """
    f_transform = lowerer.lower(node.transform)

    def se3_adjoint_fn(x, u, node, params):
        T = f_transform(x, u, node, params)

        # Extract rotation and translation from 4×4 homogeneous matrix
        R = T[:3, :3]
        p = T[:3, 3]

        # Build skew-symmetric matrix [p]×
        p_skew = jnp.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])

        # Build 6×6 adjoint matrix
        # Ad_T = [ R      0   ]
        #        [ [p]×R  R   ]
        top_row = jnp.hstack([R, jnp.zeros((3, 3))])
        bottom_row = jnp.hstack([p_skew @ R, R])

        return jnp.vstack([top_row, bottom_row])

    return se3_adjoint_fn


@visitor(SE3AdjointDual)
def _visit_se3_adjoint_dual(lowerer, node: SE3AdjointDual):
    """Lower SE3 coadjoint (big Ad*) for transforming wrenches between frames.

    Computes the 6×6 coadjoint matrix Ad*_T that transforms wrenches:
        F_a = Ad*_{T_ab} @ F_b

    For SE(3) with rotation R and translation p:
        Ad*_T = [ R     [p]×R ]
                [ 0       R   ]

    This is the transpose-inverse of Ad_T.

    Args:
        node: SE3AdjointDual expression node

    Returns:
        Function (x, u, node, params) -> 6×6 coadjoint matrix
    """
    f_transform = lowerer.lower(node.transform)

    def se3_adjoint_dual_fn(x, u, node, params):
        T = f_transform(x, u, node, params)

        # Extract rotation and translation from 4×4 homogeneous matrix
        R = T[:3, :3]
        p = T[:3, 3]

        # Build skew-symmetric matrix [p]×
        p_skew = jnp.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])

        # Build 6×6 coadjoint matrix
        # Ad*_T = [ R     [p]×R ]
        #         [ 0       R   ]
        top_row = jnp.hstack([R, p_skew @ R])
        bottom_row = jnp.hstack([jnp.zeros((3, 3)), R])

        return jnp.vstack([top_row, bottom_row])

    return se3_adjoint_dual_fn


@visitor(SO3Exp)
def _visit_so3_exp(lowerer, node: SO3Exp):
    """Lower SO3 exponential map using jaxlie.

    Maps a 3D rotation vector (axis-angle) to a 3×3 rotation matrix
    using jaxlie's numerically robust implementation.

    Args:
        node: SO3Exp expression node

    Returns:
        Function (x, u, node, params) -> 3×3 rotation matrix
    """
    import jaxlie

    f_omega = lowerer.lower(node.omega)

    def so3_exp_fn(x, u, node, params):
        omega = f_omega(x, u, node, params)
        return jaxlie.SO3.exp(omega).as_matrix()

    return so3_exp_fn


@visitor(SO3Log)
def _visit_so3_log(lowerer, node: SO3Log):
    """Lower SO3 logarithm map using jaxlie.

    Maps a 3×3 rotation matrix to a 3D rotation vector (axis-angle)
    using jaxlie's numerically robust implementation.

    Args:
        node: SO3Log expression node

    Returns:
        Function (x, u, node, params) -> 3D rotation vector
    """
    import jaxlie

    f_rotation = lowerer.lower(node.rotation)

    def so3_log_fn(x, u, node, params):
        rotation = f_rotation(x, u, node, params)
        return jaxlie.SO3.from_matrix(rotation).log()

    return so3_log_fn


@visitor(SE3Exp)
def _visit_se3_exp(lowerer, node: SE3Exp):
    """Lower SE3 exponential map using jaxlie.

    Maps a 6D twist vector [v; ω] to a 4×4 homogeneous transformation
    matrix using jaxlie's numerically robust implementation.

    The twist convention [v; ω] (linear first, angular second) matches
    jaxlie's SE3 tangent parameterization, so no reordering is needed.

    Args:
        node: SE3Exp expression node

    Returns:
        Function (x, u, node, params) -> 4×4 transformation matrix
    """
    import jaxlie

    f_twist = lowerer.lower(node.twist)

    def se3_exp_fn(x, u, node, params):
        twist = f_twist(x, u, node, params)
        return jaxlie.SE3.exp(twist).as_matrix()

    return se3_exp_fn


@visitor(SE3Log)
def _visit_se3_log(lowerer, node: SE3Log):
    """Lower SE3 logarithm map using jaxlie.

    Maps a 4×4 homogeneous transformation matrix to a 6D twist vector
    [v; ω] using jaxlie's numerically robust implementation.

    Args:
        node: SE3Log expression node

    Returns:
        Function (x, u, node, params) -> 6D twist vector
    """
    import jaxlie

    f_transform = lowerer.lower(node.transform)

    def se3_log_fn(x, u, node, params):
        transform = f_transform(x, u, node, params)
        return jaxlie.SE3.from_matrix(transform).log()

    return se3_log_fn
