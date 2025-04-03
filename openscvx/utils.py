import jax.numpy as jnp

def qdcm(q: jnp.ndarray) -> jnp.ndarray:
    # Convert a quaternion to a direction cosine matrix
    q_norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
    w, x, y, z = q / q_norm
    return jnp.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def SSMP(w: jnp.ndarray):
    # Convert an angular rate to a 4 x 4 skew symetric matrix
    x, y, z = w
    return jnp.array([[0, -x, -y, -z], [x, 0, z, -y], [y, -z, 0, x], [z, y, -x, 0]])

def SSM(w: jnp.ndarray):
    # Convert an angular rate to a 3 x 3 skew symetric matrix
    x, y, z = w
    return jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])