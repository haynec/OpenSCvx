"""Damped least-squares inverse kinematics using JAX autodiff.

A lightweight IK solver for generating initial guesses for SCP-based
trajectory optimization. Uses the geometric Jacobian from jax.jacfwd
applied to a Product of Exponentials FK function.
"""

import jax
import jax.numpy as jnp
import numpy as np


def poe_fk_position(screw_axes, T_home, q):
    """Product of Exponentials FK returning end-effector position.

    Args:
        screw_axes: (n_joints, 6) array of screw axes [v; omega].
        T_home: (4, 4) home configuration transform.
        q: (n_joints,) joint angles.

    Returns:
        (3,) end-effector position.
    """
    import jaxlie

    T = jnp.eye(4)
    for i in range(screw_axes.shape[0]):
        T = T @ jaxlie.SE3.exp(screw_axes[i] * q[i]).as_matrix()
    T = T @ T_home
    return T[:3, 3]


def ik_solve(
    screw_axes,
    T_home,
    p_target,
    q0=None,
    *,
    max_iter=200,
    tol=1e-6,
    damping=1e-3,
    q_min=None,
    q_max=None,
):
    """Damped least-squares IK solver.

    Args:
        screw_axes: (n_joints, 6) array of screw axes.
        T_home: (4, 4) home configuration.
        p_target: (3,) desired end-effector position.
        q0: (n_joints,) initial joint angle guess. Defaults to zeros.
        max_iter: Maximum iterations.
        tol: Position error tolerance (meters).
        damping: Damping factor for least-squares (higher = more stable, slower).
        q_min: (n_joints,) optional lower joint limits.
        q_max: (n_joints,) optional upper joint limits.

    Returns:
        q_sol: (n_joints,) joint angles that place the EE near p_target.
    """
    n_joints = screw_axes.shape[0]
    if q0 is None:
        q0 = np.zeros(n_joints)

    screw_axes_jnp = jnp.array(screw_axes)
    T_home_jnp = jnp.array(T_home)
    p_target_jnp = jnp.array(p_target)

    def fk(q):
        return poe_fk_position(screw_axes_jnp, T_home_jnp, q)

    J_fn = jax.jacfwd(fk)

    q = jnp.array(q0, dtype=float)
    for i in range(max_iter):
        p = fk(q)
        err = p_target_jnp - p
        err_norm = jnp.linalg.norm(err)
        if err_norm < tol:
            break

        J = J_fn(q)
        # Damped least-squares: dq = J^T (J J^T + lambda^2 I)^{-1} e
        dq = J.T @ jnp.linalg.solve(J @ J.T + damping * jnp.eye(3), err)
        q = q + dq

        # Clamp to joint limits if provided
        if q_min is not None:
            q = jnp.maximum(q, jnp.array(q_min))
        if q_max is not None:
            q = jnp.minimum(q, jnp.array(q_max))

    return np.array(q)
