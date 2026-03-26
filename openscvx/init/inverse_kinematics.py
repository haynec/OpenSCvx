"""IK-based trajectory initialization.

Generates joint-space trajectory guesses by interpolating task-space poses
(position + orientation) between keyframes and solving inverse kinematics
at each node. Combines linspace (position), slerp (orientation), and
damped least-squares IK into a single initialization call.

Requires jaxlie: pip install openscvx[lie]
"""

from typing import Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

from openscvx.init.interpolation import linspace, slerp

Pose = Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[Sequence[float], Sequence[float]],
]


# =============================================================================
# Quaternion / SO(3) helpers
# =============================================================================


def _quat_wxyz_to_rotmat(q_wxyz):
    """Convert [w, x, y, z] quaternion to 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@jax.jit
def _so3_log(R):
    """SO(3) logarithm: rotation matrix -> rotation vector (axis * angle)."""
    cos_angle = jnp.clip((jnp.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = jnp.arccos(cos_angle)
    skew = (R - R.T) / 2.0
    omega_hat = jnp.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    safe_angle = jnp.where(jnp.abs(angle) < 1e-10, 1.0, angle)
    safe_sin = jnp.where(jnp.abs(angle) < 1e-10, 1.0, jnp.sin(safe_angle))
    scale = jnp.where(jnp.abs(angle) < 1e-10, 1.0, safe_angle / safe_sin)
    return omega_hat * scale


# =============================================================================
# Product of Exponentials FK
# =============================================================================


@jax.jit
def _poe_fk_pose(screw_axes, T_home, q):
    """PoE FK returning (4, 4) end-effector transform."""

    T = jnp.eye(4)
    for i in range(screw_axes.shape[0]):
        T = T @ jaxlie.SE3.exp(screw_axes[i] * q[i]).as_matrix()
    return T @ T_home


@jax.jit
def _poe_fk_position(screw_axes, T_home, q):
    """PoE FK returning (3,) end-effector position."""
    return _poe_fk_pose(screw_axes, T_home, q)[:3, 3]


# =============================================================================
# IK solver (JIT-compiled via lax.while_loop)
# =============================================================================


@jax.jit
def _ik_loop_pose(screw_axes, T_home, p_target, R_target, q0, q_lo, q_hi, damping, tol, max_iter):
    """JIT'd 6D pose IK (position + orientation) via damped least-squares."""
    target_vec = jnp.concatenate([p_target, jnp.zeros(3)])

    def fk_vec(q):
        T = _poe_fk_pose(screw_axes, T_home, q)
        return jnp.concatenate([T[:3, 3], _so3_log(R_target.T @ T[:3, :3])])

    J_fn = jax.jacfwd(fk_vec)

    def cond(state):
        _, err_norm, i = state
        return (err_norm >= tol) & (i < max_iter)

    def body(state):
        q, _, i = state
        current = fk_vec(q)
        err = target_vec - current
        J = J_fn(q)
        dq = J.T @ jnp.linalg.solve(J @ J.T + damping * jnp.eye(6), err)
        q_new = jnp.clip(q + dq, q_lo, q_hi)
        return q_new, jnp.linalg.norm(target_vec - fk_vec(q_new)), i + 1

    init_err = jnp.linalg.norm(target_vec - fk_vec(q0))
    q_sol, _, _ = jax.lax.while_loop(cond, body, (q0, init_err, jnp.int32(0)))
    return q_sol


@jax.jit
def _ik_loop_position(screw_axes, T_home, p_target, q0, q_lo, q_hi, damping, tol, max_iter):
    """JIT'd position-only IK via damped least-squares."""

    def fk_pos(q):
        return _poe_fk_position(screw_axes, T_home, q)

    J_fn = jax.jacfwd(fk_pos)

    def cond(state):
        _, err_norm, i = state
        return (err_norm >= tol) & (i < max_iter)

    def body(state):
        q, _, i = state
        err = p_target - fk_pos(q)
        J = J_fn(q)
        dq = J.T @ jnp.linalg.solve(J @ J.T + damping * jnp.eye(3), err)
        q_new = jnp.clip(q + dq, q_lo, q_hi)
        return q_new, jnp.linalg.norm(p_target - fk_pos(q_new)), i + 1

    init_err = jnp.linalg.norm(p_target - fk_pos(q0))
    q_sol, _, _ = jax.lax.while_loop(cond, body, (q0, init_err, jnp.int32(0)))
    return q_sol


def _ik_solve(
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

    The inner loop is JIT-compiled via ``jax.lax.while_loop``.

    Args:
        screw_axes: (n_joints, 6) array of screw axes.
        T_home: (4, 4) home configuration.
        p_target: (3,) desired end-effector position.
        q0: (n_joints,) initial joint angle guess. Defaults to zeros.
        R_target: (3, 3) desired end-effector rotation matrix, or None for
            position-only IK.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        damping: Damping factor for least-squares.
        q_min: (n_joints,) optional lower joint limits.
        q_max: (n_joints,) optional upper joint limits.

    Returns:
        (n_joints,) joint angles that place the EE near the target.
    """
    n_joints = screw_axes.shape[0]
    if q0 is None:
        q0 = np.zeros(n_joints)

    screw_axes_j = jnp.array(screw_axes)
    T_home_j = jnp.array(T_home)
    p_target_j = jnp.array(p_target)
    q0_j = jnp.array(q0, dtype=float)
    q_lo = jnp.array(q_min) if q_min is not None else jnp.full(n_joints, -jnp.inf)
    q_hi = jnp.array(q_max) if q_max is not None else jnp.full(n_joints, jnp.inf)

    if R_target is not None:
        q_sol = _ik_loop_pose(
            screw_axes_j,
            T_home_j,
            p_target_j,
            jnp.array(R_target),
            q0_j,
            q_lo,
            q_hi,
            damping,
            tol,
            max_iter,
        )
    else:
        q_sol = _ik_loop_position(
            screw_axes_j,
            T_home_j,
            p_target_j,
            q0_j,
            q_lo,
            q_hi,
            damping,
            tol,
            max_iter,
        )

    return np.array(q_sol)


# =============================================================================
# Public API
# =============================================================================


def ik_interpolation(
    keyframes: Sequence[Pose],
    nodes: Sequence[int],
    screw_axes: np.ndarray,
    T_home: np.ndarray,
    *,
    q_init: np.ndarray = None,
    q_min: np.ndarray = None,
    q_max: np.ndarray = None,
    damping: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> np.ndarray:
    """Generate joint-angle trajectory guess via task-space interpolation and IK.

    Interpolates end-effector poses between keyframes (linspace for position,
    slerp for orientation) and solves inverse kinematics at each trajectory
    node. Each IK solve is warm-started with the previous node's solution for
    smooth joint trajectories.

    Args:
        keyframes: Sequence of (position, quaternion_wxyz) tuples. Each position
            is array-like with shape (3,) and each quaternion is array-like with
            shape (4,) in [w, x, y, z] order.
        nodes: Sequence of node indices where keyframes occur. Must be sorted in
            ascending order and have the same length as keyframes. The last node
            determines the output size (N = nodes[-1] + 1).
        screw_axes: (n_joints, 6) array of screw axes for Product of Exponentials.
        T_home: (4, 4) home configuration transform.
        q_init: (n_joints,) initial joint angle guess for the first node.
            Defaults to zeros.
        q_min: (n_joints,) optional lower joint limits.
        q_max: (n_joints,) optional upper joint limits.
        damping: Damping factor for least-squares IK.
        max_iter: Maximum IK iterations per node.
        tol: IK convergence tolerance.

    Returns:
        np.ndarray of shape (N, n_joints) containing joint angles at each node.

    Example:
        Initialize a 7-DOF arm trajectory reaching for a target::

            import openscvx as ox

            angle.guess = ox.init.ik_interpolation(
                keyframes=[
                    ([0.7, 0, 0.34], [1, 0, 0, 0]),  # home pose
                    ([0.3, 0.3, 0.5], [1, 0, 0, 0]),  # target pose
                ],
                nodes=[0, 49],
                screw_axes=screw_axes,
                T_home=T_home,
            )
    """
    positions = [np.asarray(kf[0], dtype=np.float64) for kf in keyframes]
    quaternions = [np.asarray(kf[1], dtype=np.float64) for kf in keyframes]

    for i, (p, q) in enumerate(zip(positions, quaternions)):
        if p.shape != (3,):
            raise ValueError(f"Keyframe {i} position has shape {p.shape}, expected (3,)")
        if q.shape != (4,):
            raise ValueError(f"Keyframe {i} quaternion has shape {q.shape}, expected (4,)")

    # Interpolate task-space trajectory
    p_traj = linspace(keyframes=positions, nodes=nodes)  # (N, 3)
    q_traj = slerp(keyframes=quaternions, nodes=nodes)  # (N, 4)

    N = nodes[-1] + 1
    n_joints = screw_axes.shape[0]
    result = np.zeros((N, n_joints), dtype=np.float64)

    # Solve IK at each node, warm-starting from previous solution
    q_prev = q_init if q_init is not None else np.zeros(n_joints)
    for k in range(N):
        R_target = _quat_wxyz_to_rotmat(q_traj[k])
        q_sol = _ik_solve(
            screw_axes,
            T_home,
            p_traj[k],
            q0=q_prev,
            R_target=R_target,
            max_iter=max_iter,
            tol=tol,
            damping=damping,
            q_min=q_min,
            q_max=q_max,
        )
        result[k] = q_sol
        q_prev = q_sol

    return result
