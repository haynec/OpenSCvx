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


def poe_fk_pose(screw_axes, T_home, q):
    """Product of Exponentials FK returning full 4x4 transform.

    Args:
        screw_axes: (n_joints, 6) array of screw axes [v; omega].
        T_home: (4, 4) home configuration transform.
        q: (n_joints,) joint angles.

    Returns:
        (4, 4) end-effector homogeneous transform.
    """
    import jaxlie

    T = jnp.eye(4)
    for i in range(screw_axes.shape[0]):
        T = T @ jaxlie.SE3.exp(screw_axes[i] * q[i]).as_matrix()
    T = T @ T_home
    return T


def _so3_log(R):
    """SO(3) logarithm: rotation matrix -> rotation vector.

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        (3,) rotation vector (axis * angle).
    """
    cos_angle = jnp.clip((jnp.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = jnp.arccos(cos_angle)
    # Skew-symmetric part: (R - R^T) / 2
    skew = (R - R.T) / 2.0
    omega_hat = jnp.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    # Near zero angle: omega ≈ skew components (sinc(0) = 1)
    safe_angle = jnp.where(jnp.abs(angle) < 1e-10, 1.0, angle)
    safe_sin = jnp.where(jnp.abs(angle) < 1e-10, 1.0, jnp.sin(safe_angle))
    scale = jnp.where(jnp.abs(angle) < 1e-10, 1.0, safe_angle / safe_sin)
    return omega_hat * scale


def ik_solve(
    screw_axes,
    T_home,
    p_target,
    q0=None,
    *,
    R_target=None,
    max_iter=200,
    tol=1e-6,
    damping=1e-3,
    q_min=None,
    q_max=None,
):
    """Damped least-squares IK solver.

    Solves for joint angles that place the end-effector at a target position
    and optionally a target orientation. When R_target is provided, the solver
    minimizes the full 6D pose error (position + orientation via SO(3) log).

    Args:
        screw_axes: (n_joints, 6) array of screw axes.
        T_home: (4, 4) home configuration.
        p_target: (3,) desired end-effector position.
        q0: (n_joints,) initial joint angle guess. Defaults to zeros.
        R_target: (3, 3) desired end-effector rotation matrix, or None for
            position-only IK.
        max_iter: Maximum iterations.
        tol: Convergence tolerance (meters for position, combined for pose).
        damping: Damping factor for least-squares (higher = more stable, slower).
        q_min: (n_joints,) optional lower joint limits.
        q_max: (n_joints,) optional upper joint limits.

    Returns:
        q_sol: (n_joints,) joint angles that place the EE near the target.
    """
    n_joints = screw_axes.shape[0]
    if q0 is None:
        q0 = np.zeros(n_joints)

    screw_axes_jnp = jnp.array(screw_axes)
    T_home_jnp = jnp.array(T_home)
    p_target_jnp = jnp.array(p_target)

    if R_target is not None:
        R_target_jnp = jnp.array(R_target)

        def fk_vec(q):
            """Task-space vector: [position; orientation_error].

            Orientation component is so3_log(R_target^T @ R), which is zero
            when R = R_target. We differentiate this to get the task Jacobian.
            """
            T = poe_fk_pose(screw_axes_jnp, T_home_jnp, q)
            p = T[:3, 3]
            ori_err = _so3_log(R_target_jnp.T @ T[:3, :3])
            return jnp.concatenate([p, ori_err])

        # Target vector: [p_target; 0,0,0] since ori_err = 0 at goal
        target_vec = jnp.concatenate([p_target_jnp, jnp.zeros(3)])
        task_dim = 6
    else:

        def fk_vec(q):
            return poe_fk_position(screw_axes_jnp, T_home_jnp, q)

        target_vec = p_target_jnp
        task_dim = 3

    J_fn = jax.jacfwd(fk_vec)

    q = jnp.array(q0, dtype=float)
    for i in range(max_iter):
        current = fk_vec(q)
        err = target_vec - current
        err_norm = jnp.linalg.norm(err)
        if err_norm < tol:
            break

        J = J_fn(q)
        # Damped least-squares: dq = J^T (J J^T + λ² I)^{-1} e
        dq = J.T @ jnp.linalg.solve(J @ J.T + damping * jnp.eye(task_dim), err)
        q = q + dq

        # Clamp to joint limits if provided
        if q_min is not None:
            q = jnp.maximum(q, jnp.array(q_min))
        if q_max is not None:
            q = jnp.minimum(q, jnp.array(q_max))

    return np.array(q)
