"""JAX visitors for spatial expressions.

Visitors: QDCM, SSMP, SSM
"""

import jax.numpy as jnp

# Expression types to handle — uncomment as you paste visitors:
from openscvx.symbolic.expr.spatial import QDCM, SSM, SSMP
from openscvx.symbolic.lowerers.jax._registry import visitor  # noqa: F401


@visitor(QDCM)
def _visit_qdcm(self, node: QDCM):
    """Lower quaternion to direction cosine matrix (DCM) conversion.

    Converts a unit quaternion [q0, q1, q2, q3] to a 3x3 rotation matrix.
    Used in 6-DOF spacecraft and robotics applications.

    The quaternion is normalized before conversion to ensure a valid rotation
    matrix. The DCM is computed using the standard quaternion-to-DCM formula.

    Args:
        node: QDCM expression node

    Returns:
        Function (x, u, node, params) -> 3x3 rotation matrix

    Note:
        Quaternion convention: [w, x, y, z] where w is the scalar part
    """
    f = self.lower(node.q)

    def qdcm_fn(x, u, node, params):
        q = f(x, u, node, params)
        # Normalize the quaternion
        q_norm = jnp.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
        w, qx, qy, qz = q / q_norm
        # Convert to direction cosine matrix
        return jnp.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * w), 2 * (qx * qz + qy * w)],
                [2 * (qx * qy + qz * w), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * w)],
                [2 * (qx * qz - qy * w), 2 * (qy * qz + qx * w), 1 - 2 * (qx**2 + qy**2)],
            ]
        )

    return qdcm_fn


@visitor(SSMP)
def _visit_ssmp(self, node: SSMP):
    """Lower skew-symmetric matrix for quaternion dynamics (4x4).

    Creates a 4x4 skew-symmetric matrix from angular velocity vector for
    quaternion kinematic propagation: q_dot = 0.5 * SSMP(omega) @ q

    The SSMP matrix is used in quaternion kinematics to compute quaternion
    derivatives from angular velocity vectors.

    Args:
        node: SSMP expression node

    Returns:
        Function (x, u, node, params) -> 4x4 skew-symmetric matrix

    Note:
        For angular velocity w = [x, y, z], returns:
        [[0, -x, -y, -z],
            [x,  0,  z, -y],
            [y, -z,  0,  x],
            [z,  y, -x,  0]]
    """
    f = self.lower(node.w)

    def ssmp_fn(x, u, node, params):
        w = f(x, u, node, params)
        wx, wy, wz = w[0], w[1], w[2]
        return jnp.array(
            [
                [0, -wx, -wy, -wz],
                [wx, 0, wz, -wy],
                [wy, -wz, 0, wx],
                [wz, wy, -wx, 0],
            ]
        )

    return ssmp_fn


@visitor(SSM)
def _visit_ssm(self, node: SSM):
    """Lower skew-symmetric matrix for cross product (3x3).

    Creates a 3x3 skew-symmetric matrix from a vector such that
    SSM(a) @ b = a x b (cross product).

    The SSM is the matrix representation of the cross product operator,
    allowing cross products to be computed as matrix-vector multiplication.

    Args:
        node: SSM expression node

    Returns:
        Function (x, u, node, params) -> 3x3 skew-symmetric matrix

    Note:
        For vector w = [x, y, z], returns:
        [[ 0, -z,  y],
            [ z,  0, -x],
            [-y,  x,  0]]
    """
    f = self.lower(node.w)

    def ssm_fn(x, u, node, params):
        w = f(x, u, node, params)
        wx, wy, wz = w[0], w[1], w[2]
        return jnp.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])

    return ssm_fn
