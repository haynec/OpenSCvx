"""IK-based trajectory initialization.

Generates joint-space trajectory guesses by interpolating task-space poses
(position + orientation) between keyframes and solving inverse kinematics
at each node via damped least-squares. Combines linspace (position),
slerp (orientation), and IK into a single initialization call.

Two solve modes are available:

- **Parallel** (default): Solves all nodes independently via ``jax.vmap``.
  Each node starts from the same ``angles_init`` seed. Avoids propagating
  bad local minima but may produce less coherent joint-space paths.

- **Sequential**: Solves nodes in order via ``jax.lax.scan``, seeding each
  node with the previous node's solution. Produces smoother trajectories
  when the seed is good, but a bad early solution can propagate.

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
    """Convert [w, x, y, z] quaternion to 3x3 rotation matrix (JAX-compatible)."""
    w, x, y, z = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
    return jnp.array(
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
def _poe_fk_pose(screw_axes, T_home, angles):
    """PoE FK returning (4, 4) end-effector transform."""

    def step(T, xi_angle):
        xi, angle = xi_angle[:6], xi_angle[6]
        return T @ jaxlie.SE3.exp(xi * angle).as_matrix(), None

    xi_angles = jnp.concatenate([screw_axes, angles[:, None]], axis=1)
    T, _ = jax.lax.scan(step, jnp.eye(4), xi_angles)
    return T @ T_home


# =============================================================================
# IK solver (JIT-compiled via lax.while_loop)
# =============================================================================


@jax.jit
def _ik_loop_pose(
    screw_axes, T_home, p_target, R_target, angles0, angles_lo, angles_hi, damping, tol, max_iter
):
    """JIT'd 6D pose IK (position + orientation) via damped least-squares."""
    target_vec = jnp.concatenate([p_target, jnp.zeros(3)])

    def fk_vec(angles):
        T = _poe_fk_pose(screw_axes, T_home, angles)
        return jnp.concatenate([T[:3, 3], _so3_log(R_target.T @ T[:3, :3])])

    J_fn = jax.jacfwd(fk_vec)

    def cond(state):
        _, err_norm, i = state
        return (err_norm >= tol) & (i < max_iter)

    def body(state):
        angles, _, i = state
        current = fk_vec(angles)
        err = target_vec - current
        J = J_fn(angles)
        dj = J.T @ jnp.linalg.solve(J @ J.T + damping * jnp.eye(6), err)
        angles_new = jnp.clip(angles + dj, angles_lo, angles_hi)
        return angles_new, jnp.linalg.norm(target_vec - fk_vec(angles_new)), i + 1

    init_err = jnp.linalg.norm(target_vec - fk_vec(angles0))
    angles_sol, _, _ = jax.lax.while_loop(cond, body, (angles0, init_err, jnp.int32(0)))
    return angles_sol


# =============================================================================
# Public API
# =============================================================================


def ik_interpolation(
    keyframes: Sequence[Pose],
    nodes: Sequence[int],
    screw_axes: np.ndarray,
    T_home: np.ndarray,
    *,
    angles_init: np.ndarray = None,
    angles_min: np.ndarray = None,
    angles_max: np.ndarray = None,
    sequential: bool = False,
    damping: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> np.ndarray:
    """Generate joint-angle trajectory guess via task-space interpolation and IK.

    Interpolates end-effector poses between keyframes (linspace for position,
    slerp for orientation) and solves inverse kinematics at each trajectory
    node.

    Args:
        keyframes: Sequence of (position, quaternion_wxyz) tuples. Each position
            is array-like with shape (3,) and each quaternion is array-like with
            shape (4,) in [w, x, y, z] order.
        nodes: Sequence of node indices where keyframes occur. Must be sorted in
            ascending order and have the same length as keyframes. The last node
            determines the output size (N = nodes[-1] + 1).
        screw_axes: (n_joints, 6) array of screw axes for Product of Exponentials.
        T_home: (4, 4) home configuration transform.
        angles_init: (n_joints,) initial joint angle guess. In parallel mode
            this seeds every node; in sequential mode it seeds only the first
            node. Defaults to zeros.
        angles_min: (n_joints,) optional lower joint limits.
        angles_max: (n_joints,) optional upper joint limits.
        sequential: If True, solve nodes sequentially (each node seeded by the
            previous solution). If False (default), solve all nodes in parallel
            from the same ``angles_init`` seed.
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

    for i, (p, quat) in enumerate(zip(positions, quaternions)):
        if p.shape != (3,):
            raise ValueError(f"Keyframe {i} position has shape {p.shape}, expected (3,)")
        if quat.shape != (4,):
            raise ValueError(f"Keyframe {i} quaternion has shape {quat.shape}, expected (4,)")

    # Interpolate task-space trajectory
    p_traj = jnp.array(linspace(keyframes=positions, nodes=nodes))  # (N, 3)
    quat_traj = jnp.array(slerp(keyframes=quaternions, nodes=nodes))  # (N, 4)
    R_traj = jax.vmap(_quat_wxyz_to_rotmat)(quat_traj)  # (N, 3, 3)

    n_joints = screw_axes.shape[0]
    screw_axes_j = jnp.array(screw_axes)
    T_home_j = jnp.array(T_home)
    angles_lo = jnp.array(angles_min) if angles_min is not None else jnp.full(n_joints, -jnp.inf)
    angles_hi = jnp.array(angles_max) if angles_max is not None else jnp.full(n_joints, jnp.inf)

    angles0 = jnp.array(angles_init) if angles_init is not None else jnp.zeros(n_joints)

    if sequential:
        # Solve nodes in order, seeding each from the previous solution
        def scan_step(prev_angles, node_data):
            p, R = node_data
            sol = _ik_loop_pose(
                screw_axes_j,
                T_home_j,
                p,
                R,
                prev_angles,
                angles_lo,
                angles_hi,
                damping,
                tol,
                max_iter,
            )
            return sol, sol

        _, result = jax.lax.scan(scan_step, angles0, (p_traj, R_traj))
    else:
        # Solve all nodes in parallel from the same seed
        N = nodes[-1] + 1
        angles0_all = jnp.broadcast_to(angles0, (N, n_joints))
        result = jax.vmap(
            lambda p, R, a0: _ik_loop_pose(
                screw_axes_j, T_home_j, p, R, a0, angles_lo, angles_hi, damping, tol, max_iter
            )
        )(p_traj, R_traj, angles0_all)

    return np.array(result)
