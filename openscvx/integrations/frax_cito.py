"""Contact-implicit trajectory optimization (CITO) helpers for frax-backed legged models.

All CITO-specific dynamics, contact kinematics, complementarity BYOF factories, and
control layout helpers live here. OpenSCvx symbolic/discretization layers are consumed
as-is from the example via ``Problem``, ``ox.ctcs``, and impulsive controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from openscvx.integrations._utils import _resolve_slice
from openscvx.integrations.base import DynamicsAdapter
from openscvx.integrations.frax import _bounds_from_robot, _desentinel

if False:  # TYPE_CHECKING
    from frax.core.robot import Robot
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State


# Frax floating-base URDF convention (x, y, z prismatic then roll, pitch, yaw).
DEFAULT_ROLL_IDX = 3
DEFAULT_PITCH_IDX = 4
DEFAULT_YAW_IDX = 5
DEFAULT_BASE_SLICE = slice(0, 6)
DEFAULT_LEG_SLICE = slice(6, 8)
# URDF rigid-body chain for drawing (yaw/torso, hip, knee); excludes virtual-base joints.
TORSO_JOINT_IDX = 5
HIP_JOINT_IDX = 6
KNEE_JOINT_IDX = 7
MONOPED_N_DISPLAY_LINKS = 3

# Foot point in knee frame: shank length 0.35 m along -z.
DEFAULT_FOOT_OFFSET = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.35],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def monoped_display_chain(
    robot: Monoped3D,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame polyline for the three URDF leg links plus the foot contact point.

    The model has eight *joints* (six virtual-base + hip + knee) but only three
    moving rigid links (torso, thigh, shank). Visualization should connect
    torso → hip → knee → foot, not every joint origin (which draws a spurious
    fixed-base arm through the prismatic/RPY chain).

    Returns:
        chain: ``(4, 3)`` positions ``[torso, hip, knee, foot]`` (3 segments).
        foot: ``(3,)`` foot / CITO contact position (same as ``chain[-1]``).
    """
    q = np.asarray(q, dtype=float)
    links = np.asarray(robot.link_to_world_transforms(q))
    foot = np.asarray(robot.foot_transform(q))[:3, 3]
    chain = np.stack(
        [
            links[TORSO_JOINT_IDX, :3, 3],
            links[HIP_JOINT_IDX, :3, 3],
            links[KNEE_JOINT_IDX, :3, 3],
            foot,
        ],
        axis=0,
    )
    return chain, foot


def monoped_display_chain_trajectory(
    robot: Monoped3D,
    q_traj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch version of :func:`monoped_display_chain` over a ``q`` trajectory."""
    q_traj = np.asarray(q_traj, dtype=float)
    n_frames = q_traj.shape[0]
    n_pts = MONOPED_N_DISPLAY_LINKS + 1
    chain = np.zeros((n_frames, n_pts, 3), dtype=float)
    foot = np.zeros((n_frames, 3), dtype=float)
    for t in range(n_frames):
        chain[t], foot[t] = monoped_display_chain(robot, q_traj[t])
    return chain, foot


def monoped_urdf_path() -> Path:
    """Path to the bundled 3D monoped URDF."""
    return Path(__file__).resolve().parents[2] / "assets/robots/monoped_3d/monoped.urdf"


@jax.tree_util.register_static
class Monoped3D:
    """frax ``Robot`` with foot frame kinematics for CITO."""

    def __init__(
        self,
        urdf_filename: str | Path,
        *,
        foot_offset: np.ndarray | None = None,
    ) -> None:
        import frax

        self._robot = frax.Robot(str(urdf_filename))
        self.foot_offset = np.asarray(foot_offset if foot_offset is not None else DEFAULT_FOOT_OFFSET)
        self.foot_parent_chain = np.arange(self._robot.num_joints, dtype=int)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot, name)

    def foot_transform(self, q: jnp.ndarray) -> jnp.ndarray:
        joint_transforms = self._robot.joint_to_world_transforms(q)
        parent_index = int(self.foot_parent_chain[-1])
        offset = jnp.asarray(self.foot_offset)
        return self._robot._frame_transform(joint_transforms, offset, parent_index)

    def foot_jacobian(self, q: jnp.ndarray) -> jnp.ndarray:
        joint_transforms = self._robot.joint_to_world_transforms(q)
        parent_chain = jnp.asarray(self.foot_parent_chain)
        offset = jnp.asarray(self.foot_offset)
        return self._robot._frame_jacobian(joint_transforms, offset, parent_chain)

    def foot_linear_jacobian(self, q: jnp.ndarray) -> jnp.ndarray:
        return self.foot_jacobian(q)[:3, :]


def load_monoped_3d(urdf_filename: str | Path | None = None) -> Monoped3D:
    """Load the bundled 3D monoped model."""
    path = Path(urdf_filename) if urdf_filename is not None else monoped_urdf_path()
    return Monoped3D(path)


def effective_actuated_joint_count(robot: Monoped3D) -> int:
    """Count joints with nonzero URDF effort (excludes virtual floating-base DOFs).

    frax sets ``includes_floating_dof=False`` for the prismatic+RPY base chain, so
    ``robot.num_actuated_joints`` can equal ``num_joints`` even when only the leg
    joints are physically actuated.
    """
    forces = np.asarray(robot.joint_max_forces, dtype=float)
    return int(np.sum(forces > 1e-6))


def actuated_torque_bounds(robot: Monoped3D) -> tuple[np.ndarray, np.ndarray]:
    """Torque limits for actuated joints only (hip, knee on the bundled monoped)."""
    forces = _desentinel(np.asarray(robot.joint_max_forces, dtype=float))
    tau_max = forces[forces > 1e-6]
    return -tau_max, tau_max


@dataclass
class ContactModelConfig:
    """CITO contact and complementarity parameters."""

    n_c: int = 1
    mu: float = 1.0
    delta: float = 0.05
    epsilon_c: float = 0.1
    z_ground: float = 0.0
    huber_eps: float = 1
    apply_attitude_path_limits: bool = False
    roll_limit_deg: float = 30.0
    pitch_limit_deg: float = 30.0
    roll_idx: int = DEFAULT_ROLL_IDX
    pitch_idx: int = DEFAULT_PITCH_IDX
    yaw_idx: int = DEFAULT_YAW_IDX
    max_normal_force: float = 500.0
    max_tangential_force: float = 500.0
    max_impulse_normal: float = 50.0
    max_impulse_tangential: float = 50.0
    enable_impulses: bool = True
    enable_cross_complementarity: bool = True
    enable_friction_complementarity: bool = True


@dataclass
class DfohControlSlices:
    """Resolved unified-vector slices (available after ``Problem`` preprocessing)."""

    phi_t_zoh: slice
    phi_t_foh: slice
    phi_n_zoh: slice
    phi_n_foh: slice
    gamma_zoh: slice
    gamma_foh: slice
    phi_t_imp: slice
    phi_n_imp: slice
    gamma_imp: slice
    tau: slice
    q: slice
    qd: slice
    y_phi_n: slice
    y_sd: slice
    y_lambda: slice
    y_gamma: slice
    y_rho: slice


@dataclass
class DfohControlLayout:
    """State/Control handles; slices are resolved lazily on first BYOF evaluation."""

    phi_t_zoh: Any
    phi_t_foh: Any
    phi_n_zoh: Any
    phi_n_foh: Any
    gamma_zoh: Any
    gamma_foh: Any
    phi_t_imp: Any
    phi_n_imp: Any
    gamma_imp: Any
    tau: Any
    q: Any
    qd: Any
    y_phi_n: Any | None = None
    y_sd: Any | None = None
    y_lambda: Any | None = None
    y_gamma: Any | None = None
    y_rho: Any | None = None
    _slices: Optional[DfohControlSlices] = field(default=None, repr=False, compare=False)

    def slices(self) -> DfohControlSlices:
        if self._slices is None:
            empty = slice(0, 0)

            def _sl(obj: Any, name: str) -> slice:
                return _resolve_slice(obj, name) if obj is not None else empty

            imp = (
                _resolve_slice(self.phi_t_imp, "Phi_t"),
                _resolve_slice(self.phi_n_imp, "Phi_n"),
                _resolve_slice(self.gamma_imp, "Gamma"),
            ) if self.phi_t_imp is not None else (empty, empty, empty)
            self._slices = DfohControlSlices(
                phi_t_zoh=_resolve_slice(self.phi_t_zoh, "phi_t_zoh"),
                phi_t_foh=_resolve_slice(self.phi_t_foh, "phi_t_foh"),
                phi_n_zoh=_resolve_slice(self.phi_n_zoh, "phi_n_zoh"),
                phi_n_foh=_resolve_slice(self.phi_n_foh, "phi_n_foh"),
                gamma_zoh=_resolve_slice(self.gamma_zoh, "gamma_zoh"),
                gamma_foh=_resolve_slice(self.gamma_foh, "gamma_foh"),
                phi_t_imp=imp[0],
                phi_n_imp=imp[1],
                gamma_imp=imp[2],
                tau=_resolve_slice(self.tau, "tau"),
                q=_resolve_slice(self.q, "q"),
                qd=_resolve_slice(self.qd, "qd"),
                y_phi_n=_sl(self.y_phi_n, "y_phi_n"),
                y_sd=_sl(self.y_sd, "y_sd"),
                y_lambda=_sl(self.y_lambda, "y_lambda"),
                y_gamma=_sl(self.y_gamma, "y_gamma"),
                y_rho=_sl(self.y_rho, "y_rho"),
            )
        return self._slices


def fischer_burmeister(a: jnp.ndarray, b: jnp.ndarray, delta: float | jnp.ndarray) -> jnp.ndarray:
    """Relaxed Fischer–Burmeister; feasible when ``<= 0`` (BYOF sign convention)."""
    d = jnp.asarray(delta, dtype=a.dtype)
    out = a + b - jnp.sqrt(a * a + b * b + d * d)
    return jnp.asarray(out).reshape(())


def smooth_abs(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    return jnp.sqrt(x * x + eps * eps)


def smooth_norm(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(x * x) + eps * eps)


def contact_wrench_world(phi_t: jnp.ndarray, phi_n: jnp.ndarray, R_c: jnp.ndarray) -> jnp.ndarray:
    """Map tangential + normal contact scalars to a world-frame wrench direction."""
    phi = jnp.concatenate([jnp.ravel(phi_t), jnp.ravel(jnp.atleast_1d(phi_n))])
    return R_c @ phi


def contact_kinematics(
    robot: Monoped3D,
    q: jnp.ndarray,
    qd: jnp.ndarray,
    *,
    z_ground: float = 0.0,
) -> Dict[str, jnp.ndarray]:
    """Flat-ground contact kinematics for a single foot (``n_c = 1``)."""
    p_c = robot.foot_transform(q)[:3, 3]
    J_c = robot.foot_linear_jacobian(q)
    n_c = jnp.array([0.0, 0.0, 1.0])
    t_c = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    R_c = jnp.column_stack([t_c, n_c])
    sd = p_c[2] - z_ground
    v_ct = t_c.T @ (J_c @ qd)
    return {
        "p_c": p_c,
        "J_c": J_c,
        "R_c": R_c,
        "n_c": n_c,
        "t_c": t_c,
        "sd": sd,
        "v_ct": v_ct,
    }


def stationarity_measure(
    phi_n: jnp.ndarray,
    v_ct: jnp.ndarray,
    *,
    huber_eps: float,
) -> jnp.ndarray:
    """Scalar stationarity residual ``phi_n * ||v_ct||~`` for complementarity with ``phi_n``."""
    return phi_n * smooth_norm(v_ct, huber_eps)


def rho_friction_cone(
    phi_t: jnp.ndarray,
    phi_n: jnp.ndarray,
    gamma: jnp.ndarray,
    *,
    mu: float,
    huber_eps: float,
) -> jnp.ndarray:
    """Friction cone slack ``|mu phi_n|~ - ||phi_t||~`` (scalar)."""
    return smooth_abs(mu * phi_n, huber_eps) - smooth_norm(phi_t, huber_eps)


def _phi_gamma_eff(u: jnp.ndarray, sl: DfohControlSlices) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    phi_t = u[sl.phi_t_zoh] + u[sl.phi_t_foh]
    phi_n = u[sl.phi_n_zoh] + u[sl.phi_n_foh]
    gamma = u[sl.gamma_zoh] + u[sl.gamma_foh]
    return phi_t, phi_n, gamma


def cito_qdd_byof(
    robot: Monoped3D,
    layout: DfohControlLayout,
    *,
    q_arg: Any,
    qd_arg: Any,
    tau_arg: Any,
) -> Callable:
    """BYOF continuous acceleration with contact forces mapped through ``J^T R``."""

    _resolved: list = []
    _layout_slices: list = []

    def f(x, u, node, params):
        del node, params
        if not _layout_slices:
            _layout_slices.append(layout.slices())
        sl = _layout_slices[0]
        if not _resolved:
            _resolved.append(_resolve_slice(q_arg, "q"))
            _resolved.append(_resolve_slice(qd_arg, "qd"))
            _resolved.append(_resolve_slice(tau_arg, "tau"))
        q_sl, qd_sl, tau_sl = _resolved

        q_val = x[q_sl]
        qd_val = x[qd_sl]
        tau_val = u[tau_sl]
        phi_t, phi_n, _gamma = _phi_gamma_eff(u, sl)

        kin = contact_kinematics(robot, q_val, qd_val)
        f_world = contact_wrench_world(phi_t, phi_n, kin["R_c"])
        tau_contact = kin["J_c"].T @ f_world

        n_pad = robot.num_joints - tau_val.shape[0]
        if n_pad > 0:
            tau_full = jnp.concatenate([jnp.zeros(n_pad, dtype=tau_val.dtype), tau_val])
        else:
            tau_full = tau_val
        tau_full = tau_full + tau_contact
        return robot.forward_dynamics(q_val, qd_val, tau_full, fext=None)

    return f


def cito_aux_integrand_byof(
    robot: Monoped3D,
    layout: DfohControlLayout,
    which: str,
    *,
    q_arg: Any,
    qd_arg: Any,
    config: ContactModelConfig,
) -> Callable:
    """Return integrand for one auxiliary cross-complementarity channel."""

    _resolved: list = []
    _layout_slices: list = []

    def f(x, u, node, params):
        del node, params
        if not _layout_slices:
            _layout_slices.append(layout.slices())
        sl = _layout_slices[0]
        if not _resolved:
            _resolved.append(_resolve_slice(q_arg, "q"))
            _resolved.append(_resolve_slice(qd_arg, "qd"))
        q_val = x[_resolved[0]]
        qd_val = x[_resolved[1]]
        phi_t, phi_n, gamma = _phi_gamma_eff(u, sl)
        kin = contact_kinematics(robot, q_val, qd_val, z_ground=config.z_ground)
        v_ct = kin["v_ct"]
        if which == "phi_n":
            return jnp.atleast_1d(phi_n)
        if which == "sd":
            return jnp.atleast_1d(kin["sd"])
        if which == "lambda":
            return stationarity_measure(phi_n, v_ct, huber_eps=config.huber_eps)
        if which == "gamma":
            return gamma
        if which == "rho":
            return rho_friction_cone(phi_t, phi_n, gamma, mu=config.mu, huber_eps=config.huber_eps)
        raise ValueError(f"unknown aux integrand {which!r}")

    return f


def cito_impact_qd_byof(
    robot: Monoped3D,
    layout: DfohControlLayout,
    config: ContactModelConfig,
    *,
    q_arg: Any,
    qd_arg: Any,
) -> Callable:
    """Impulsive update of ``qd`` from contact impulses ``Phi`` (paper eq. 6a)."""

    _resolved: list = []
    _layout_slices: list = []

    def f(x, u, node, params):
        del node, params
        if not _layout_slices:
            _layout_slices.append(layout.slices())
        sl = _layout_slices[0]
        if not _resolved:
            _resolved.append(_resolve_slice(q_arg, "q"))
            _resolved.append(_resolve_slice(qd_arg, "qd"))
        q_sl, qd_sl = _resolved

        q_val = x[q_sl]
        qd_val = x[qd_sl]
        phi_t_imp = u[sl.phi_t_imp]
        phi_n_imp = u[sl.phi_n_imp]
        kin = contact_kinematics(robot, q_val, qd_val, z_ground=config.z_ground)
        f_world = contact_wrench_world(phi_t_imp, phi_n_imp, kin["R_c"])
        M = robot.mass_matrix(q_val)
        delta_qd = jnp.linalg.solve(M, kin["J_c"].T @ f_world)
        return qd_val + delta_qd

    return f


def estimate_normal_force_guess(
    robot: Monoped3D,
    q: np.ndarray,
    qd: np.ndarray | None = None,
    *,
    z_ground: float = 0.0,
    contact_gap_tol: float = 0.02,
) -> float:
    """Normal force ``phi_n`` guess balancing gravity when the foot is on the ground.

    At a static configuration, ``phi_n ≈ |gravity_vector(q)[2]|`` drives
    ``qdd ≈ 0`` with zero actuation torques via the contact wrench map.
    """
    q = np.asarray(q, dtype=float)
    qd = np.zeros(robot.num_joints, dtype=float) if qd is None else np.asarray(qd, dtype=float)
    kin = contact_kinematics(robot, jnp.array(q), jnp.array(qd), z_ground=z_ground)
    if float(kin["sd"]) > contact_gap_tol:
        return 0.0
    g = np.asarray(robot.gravity_vector(q), dtype=float)
    return float(abs(g[2]))


def monoped_trajectory_guess(
    robot: Monoped3D,
    foot_xy_start: tuple[float, float],
    foot_xy_goal: tuple[float, float],
    n_nodes: int,
    *,
    z_ground: float = 0.0,
    hip: float = 0.0,
    knee: float = 0.0,
) -> np.ndarray:
    """``(n_nodes, num_joints)`` poses with foot on the ground along a straight foot path."""
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    poses = []
    for i in range(n_nodes):
        t = i / (n_nodes - 1)
        foot_xy = (
            (1.0 - t) * foot_xy_start[0] + t * foot_xy_goal[0],
            (1.0 - t) * foot_xy_start[1] + t * foot_xy_goal[1],
        )
        poses.append(
            monoped_standing_pose(
                robot,
                foot_xy=foot_xy,
                z_ground=z_ground,
                hip=hip,
                knee=knee,
            )
        )
    return np.stack(poses, axis=0)


def rollout_qd_guess(
    robot: Monoped3D,
    q_traj: np.ndarray,
    tau_traj: np.ndarray,
    phi_n_traj: np.ndarray,
    dt: float,
    *,
    z_ground: float = 0.0,
) -> np.ndarray:
    """Forward-Euler ``qd`` rollout using contact + actuation (matches BYOF acceleration)."""
    n, nj = q_traj.shape
    na = tau_traj.shape[1]
    qd_traj = np.zeros((n, nj), dtype=float)
    qd = np.zeros(nj, dtype=float)
    for k in range(n - 1):
        q = jnp.array(q_traj[k])
        phi_n = float(phi_n_traj[k])
        kin = contact_kinematics(robot, q, jnp.array(qd), z_ground=z_ground)
        f_world = contact_wrench_world(jnp.zeros(2), jnp.array(phi_n), kin["R_c"])
        tau_contact = np.asarray(kin["J_c"].T @ f_world, dtype=float)
        tau_full = np.zeros(nj, dtype=float)
        tau_full[-na:] = tau_traj[k]
        tau_full += tau_contact
        qdd = np.asarray(robot.forward_dynamics(q, jnp.array(qd), jnp.array(tau_full), fext=None))
        qd = qd + dt * qdd
        qd_traj[k + 1] = qd
    return qd_traj


def refine_discrete_trajectory_guess(
    problem: Any,
    *,
    n_passes: int = 10,
) -> None:
    """Align state trajectory guesses with multishot discrete propagation.

    Kinematic ``q`` paths (e.g. foot-on-ground poses) often disagree with
    integrated dynamics, leaving a large irreducible ``J_vc``. This iterates
    ``x <- x_prop_plus`` from the same discretizer SCvx uses, re-pinning fixed
    boundary entries each pass. Call after ``problem.initialize()``; updates
    ``State.guess`` and ``problem.reset()``.
    """
    import jax.numpy as jnp

    phases = getattr(problem._algorithm, "_timed_phases", None)
    if phases is None:
        raise RuntimeError("Call problem.initialize() before refine_discrete_trajectory_guess().")

    sim = problem.settings.sim
    init_fixed = np.asarray(sim.x.initial_type) == "Fix"
    final_fixed = np.asarray(sim.x.final_type) == "Fix"
    x_initial = np.asarray(sim.x.initial, dtype=float).reshape(-1)
    x_final = np.asarray(sim.x.final, dtype=float).reshape(-1)

    state = problem._state
    params = problem._parameters
    for _ in range(n_passes):
        disc = phases.discretize_current(state, params)
        x_prop_plus = np.asarray(disc[4], dtype=float)
        x_np = x_prop_plus.copy()
        x_np[0] = np.where(init_fixed, x_initial, x_np[0])
        x_np[-1] = np.where(final_fixed, x_final, x_np[-1])
        state = state.replace(x=jnp.asarray(x_np, dtype=state.x.dtype))

    x_np = np.asarray(state.x)
    for st in problem.symbolic.states:
        if st._slice is not None:
            st.guess = x_np[:, st._slice].copy()
    problem._sync_guesses()
    problem.reset()


def _sample_phi_t_in_friction_cone(
    phi_n: float,
    mu: float,
    rng: np.random.Generator,
    n_c: int = 1,
) -> np.ndarray:
    """Random tangential ``phi_t`` with ``||phi_t|| = ratio * mu * phi_n``, ``ratio ~ U(0,1)``."""
    phi_t = np.zeros(2 * n_c, dtype=float)
    if phi_n <= 1e-9 or mu <= 0.0:
        return phi_t
    ratio = float(rng.uniform(0.0, 1.0))
    mag = ratio * mu * phi_n
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    phi_t[:2] = mag * np.array([np.cos(ang), np.sin(ang)], dtype=float)
    return phi_t


def _assign_dfoh_contact_guess(
    contact_controls: Sequence["Control"],
    phi_t_traj: np.ndarray,
    phi_n_traj: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Split effective ``(phi_t, phi_n)`` across ZOH/FOH so sums match the sampled wrench."""
    by_name = {c.name: c for c in contact_controls}
    n_nodes = phi_n_traj.shape[0]
    phi_n_traj = np.asarray(phi_n_traj, dtype=float).reshape(n_nodes, -1)
    phi_t_traj = np.asarray(phi_t_traj, dtype=float).reshape(n_nodes, -1)

    phi_n_zoh = np.zeros_like(phi_n_traj)
    phi_n_foh = np.zeros_like(phi_n_traj)
    phi_t_zoh = np.zeros_like(phi_t_traj)
    phi_t_foh = np.zeros_like(phi_t_traj)

    for k in range(n_nodes):
        frac_n = float(rng.uniform(0.0, 1.0))
        frac_t = float(rng.uniform(0.0, 1.0))
        phi_n_zoh[k] = frac_n * phi_n_traj[k]
        phi_n_foh[k] = (1.0 - frac_n) * phi_n_traj[k]
        phi_t_zoh[k] = frac_t * phi_t_traj[k]
        phi_t_foh[k] = (1.0 - frac_t) * phi_t_traj[k]

    if "phi_n_zoh" in by_name:
        by_name["phi_n_zoh"].guess = phi_n_zoh
    if "phi_n_foh" in by_name:
        by_name["phi_n_foh"].guess = phi_n_foh
    if "phi_t_zoh" in by_name:
        by_name["phi_t_zoh"].guess = phi_t_zoh
    if "phi_t_foh" in by_name:
        by_name["phi_t_foh"].guess = phi_t_foh

    for name in ("gamma_zoh", "gamma_foh", "Gamma"):
        if name in by_name:
            c = by_name[name]
            c.guess = np.zeros((n_nodes, int(np.prod(c.shape))), dtype=float)


def _sample_impulse_in_friction_cone(
    mu: float,
    rng: np.random.Generator,
    *,
    max_impulse_normal: float,
    max_impulse_tangential: float,
    n_c: int = 1,
    magnitude_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample impulsive ``(phi_t, phi_n)`` inside the friction cone with magnitude in ``[0, frac·max]``."""
    phi_t = np.zeros(2 * n_c, dtype=float)
    phi_n = np.zeros(n_c, dtype=float)
    if magnitude_frac <= 0.0:
        return phi_t, phi_n
    phi_n[0] = float(rng.uniform(0.0, magnitude_frac * max_impulse_normal))
    if phi_n[0] > 1e-9 and mu > 0.0:
        phi_t[: 2 * n_c] = _sample_phi_t_in_friction_cone(phi_n[0], mu, rng, n_c=n_c)
        mag_t = float(np.linalg.norm(phi_t[:2]))
        cap_t = magnitude_frac * max_impulse_tangential
        if mag_t > cap_t > 1e-9:
            phi_t[:2] *= cap_t / mag_t
    return phi_t, phi_n


def seed_cito_initial_guess(
    robot: Monoped3D,
    config: ContactModelConfig,
    q_state: "State",
    qd_state: "State",
    aux_states: Sequence["State"],
    tau_control: "Control",
    contact_controls: Sequence["Control"],
    *,
    q_initial: np.ndarray,
    q_final: np.ndarray,
    n_nodes: int,
    rng: np.random.Generator | None = None,
    z_ground: float | None = None,
    time_state: Optional["State"] = None,
    total_time: float | None = None,
) -> np.ndarray:
    """Populate SCvx initial guesses for a CITO monoped problem.

    Strategy:
        1. ``q``: standing pose at each node with foot ``(x, y)`` linearly interpolated
           between the boundary foot positions (not joint-space lerp).
        2. ``qd``: finite-difference of ``q`` along the nodal time grid (kinematic).
        3. CTCS augmented states: left at OpenSCvx defaults (zero after preprocess).
        4. Cross-complementarity integrators ``y_*``: zero.
        5. ``tau``: zero.
        6. Contact ``phi``: normal balances gravity; random tangential sample inside
           friction cone; random ZOH/FOH split preserving ``phi_eff = phi_zoh + phi_foh``.
        7. Impulsive ``Phi_*``: if enabled, uniform in ``[0, 0.1·max]`` with ``phi_t`` in the friction cone.
        8. ``gamma`` / ``Gamma``: zero.

    Args:
        robot: Frax monoped adapter.
        config: CITO contact parameters (``mu``, ``z_ground``, impulse flags).
        q_state, qd_state: Position / velocity states.
        aux_states: ``y_phi_n``, ``y_sd``, … (empty if cross complementarity off).
        tau_control: Actuation control.
        contact_controls: dFOH + optional impulsive contact controls.
        q_initial, q_final: Resolved configuration vectors (length ``num_joints``).
        n_nodes: Discretization grid size ``N``.
        rng: RNG for tangential contact and dFOH splits (default ``numpy`` default).
        z_ground: Ground height for gravity balance (defaults to ``config.z_ground``).

    Returns:
        ``(n_nodes, num_joints)`` configuration guess.
    """
    z_ground = config.z_ground if z_ground is None else z_ground
    rng = np.random.default_rng() if rng is None else rng

    q_initial = np.asarray(q_initial, dtype=float).reshape(-1)
    q_final = np.asarray(q_final, dtype=float).reshape(-1)
    if q_initial.shape != q_final.shape:
        raise ValueError("q_initial and q_final must have the same shape")

    # Interpolate foot placement in task space, then resolve standing ``q`` per node.
    # Joint-space lerp between two standing poses does not keep the foot on the ground
    # or on the straight-line path between boundary configurations.
    foot0 = np.asarray(robot.foot_transform(q_initial), dtype=float)[:3, 3]
    foot1 = np.asarray(robot.foot_transform(q_final), dtype=float)[:3, 3]
    q_guess = np.zeros((n_nodes, q_initial.size), dtype=float)
    for k in range(n_nodes):
        alpha = k / max(n_nodes - 1, 1)
        foot_xy = (
            (1.0 - alpha) * foot0[0] + alpha * foot1[0],
            (1.0 - alpha) * foot0[1] + alpha * foot1[1],
        )
        q_guess[k] = monoped_standing_pose(robot, foot_xy=foot_xy, z_ground=z_ground)
    q_state.guess = q_guess

    nj = int(np.prod(qd_state.shape))
    if time_state is not None and getattr(time_state, "guess", None) is not None:
        t_nodes = np.asarray(time_state.guess, dtype=float).reshape(-1)
    elif total_time is not None:
        t_nodes = np.linspace(0.0, float(total_time), n_nodes)
    else:
        t_nodes = np.linspace(0.0, 1.0, n_nodes)
    if t_nodes.shape[0] != n_nodes:
        raise ValueError(f"time grid length {t_nodes.shape[0]} != n_nodes {n_nodes}")

    qd_guess = np.zeros((n_nodes, nj), dtype=float)
    for k in range(n_nodes):
        if n_nodes == 1:
            break
        if k == 0:
            dt = max(t_nodes[1] - t_nodes[0], 1e-9)
            qd_guess[0] = (q_guess[1] - q_guess[0]) / dt
        elif k == n_nodes - 1:
            dt = max(t_nodes[-1] - t_nodes[-2], 1e-9)
            qd_guess[-1] = (q_guess[-1] - q_guess[-2]) / dt
        else:
            dt = max(t_nodes[k + 1] - t_nodes[k - 1], 1e-9)
            qd_guess[k] = (q_guess[k + 1] - q_guess[k - 1]) / dt
    qd_state.guess = qd_guess

    for aux in aux_states:
        aux.guess = np.zeros((n_nodes, int(np.prod(aux.shape))), dtype=float)

    na = int(np.prod(tau_control.shape))
    tau_control.guess = np.zeros((n_nodes, na), dtype=float)

    phi_n_traj = np.zeros((n_nodes, config.n_c), dtype=float)
    phi_t_traj = np.zeros((n_nodes, 2 * config.n_c), dtype=float)
    for k in range(n_nodes):
        phi_n_traj[k, 0] = estimate_normal_force_guess(
            robot, q_guess[k], z_ground=z_ground
        )
        phi_t_traj[k] = _sample_phi_t_in_friction_cone(
            phi_n_traj[k, 0], config.mu, rng, n_c=config.n_c
        )

    _assign_dfoh_contact_guess(contact_controls, phi_t_traj, phi_n_traj, rng)

    if config.enable_impulses:
        by_name = {c.name: c for c in contact_controls}
        phi_t_imp_traj = np.zeros((n_nodes, 2 * config.n_c), dtype=float)
        phi_n_imp_traj = np.zeros((n_nodes, config.n_c), dtype=float)
        for k in range(n_nodes):
            phi_t_k, phi_n_k = _sample_impulse_in_friction_cone(
                config.mu,
                rng,
                max_impulse_normal=config.max_impulse_normal,
                max_impulse_tangential=config.max_impulse_tangential,
                n_c=config.n_c,
                magnitude_frac=0.1,
            )
            phi_t_imp_traj[k] = phi_t_k
            phi_n_imp_traj[k] = phi_n_k
        if "Phi_t" in by_name:
            by_name["Phi_t"].guess = phi_t_imp_traj
        if "Phi_n" in by_name:
            by_name["Phi_n"].guess = phi_n_imp_traj

    return q_guess


def sync_cito_kinematic_qd_guess(problem: Any) -> None:
    """Recompute ``qd`` nodal guesses from ``q`` and the current time grid in ``problem._state``."""
    import jax.numpy as jnp

    q_state = next(s for s in problem.symbolic.states if s.name == "q")
    qd_state = next(s for s in problem.symbolic.states if s.name == "qd")
    time_state = next(s for s in problem.symbolic.states if s.name == "time")

    x = np.asarray(problem._state.x, dtype=float)
    t_nodes = x[:, time_state._slice].reshape(-1)
    q_nodes = x[:, q_state._slice]
    n_nodes = q_nodes.shape[0]
    nj = q_nodes.shape[1]

    qd_guess = np.zeros((n_nodes, nj), dtype=float)
    for k in range(n_nodes):
        if n_nodes == 1:
            break
        if k == 0:
            dt = max(t_nodes[1] - t_nodes[0], 1e-9)
            qd_guess[0] = (q_nodes[1] - q_nodes[0]) / dt
        elif k == n_nodes - 1:
            dt = max(t_nodes[-1] - t_nodes[-2], 1e-9)
            qd_guess[-1] = (q_nodes[-1] - q_nodes[-2]) / dt
        else:
            dt = max(t_nodes[k + 1] - t_nodes[k - 1], 1e-9)
            qd_guess[k] = (q_nodes[k + 1] - q_nodes[k - 1]) / dt

    qd_state.guess = qd_guess
    x_new = x.copy()
    x_new[:, qd_state._slice] = qd_guess
    problem._state = problem._state.replace(x=jnp.asarray(x_new, dtype=problem._state.x.dtype))
    problem._sync_guesses()


def capture_initial_multishot(problem: Any) -> dict[str, np.ndarray]:
    """Snapshot ``(x, u, V)`` for the pre-solve iterate (true initial guess)."""
    phases = getattr(problem._algorithm, "_timed_phases", None)
    if phases is None:
        raise RuntimeError("Call problem.initialize() before capture_initial_multishot().")
    x = np.asarray(problem._state.x, dtype=float)
    u = np.asarray(problem._state.u, dtype=float)
    disc = phases.discretize_current(problem._state, problem._parameters)
    V = np.asarray(disc[7], dtype=float)
    return {"x": x, "u": u, "V": V}


def seed_dfoh_contact_guess(
    contact_controls: Sequence["Control"],
    q_traj: np.ndarray,
    robot: Monoped3D,
    *,
    z_ground: float = 0.0,
    mu: float = 1.0,
    rng: np.random.Generator | None = None,
) -> None:
    """Contact-force guesses only (legacy helper).

    Prefer :func:`seed_cito_initial_guess` for full-problem seeding.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = q_traj.shape[0]
    phi_n_traj = np.array(
        [estimate_normal_force_guess(robot, q_traj[k], z_ground=z_ground) for k in range(n)]
    ).reshape(n, -1)
    phi_t_traj = np.stack(
        [
            _sample_phi_t_in_friction_cone(float(phi_n_traj[k, 0]), mu, rng, n_c=1)
            for k in range(n)
        ],
        axis=0,
    )
    _assign_dfoh_contact_guess(contact_controls, phi_t_traj, phi_n_traj, rng)


def monoped_standing_pose(
    robot: Monoped3D,
    *,
    foot_xy: tuple[float, float] = (0.0, 0.0),
    z_ground: float = 0.0,
    hip: float = 0.0,
    knee: float = 0.0,
) -> np.ndarray:
    """Joint configuration with leg fully extended and foot on the ground.

    Uses the bundled URDF convention: hip/knee revolute about ``y``, knee lower
    limit ``0`` is full extension. Base ``q[0:3]`` is set so the foot rests at
    ``(foot_xy[0], foot_xy[1], z_ground)`` with base roll/pitch/yaw at zero.
    """
    q = np.zeros(robot.num_joints, dtype=float)
    q[6] = hip
    q[7] = knee
    foot_ref = np.asarray(robot.foot_transform(q), dtype=float)[:3, 3]
    q[0] = foot_xy[0] - foot_ref[0]
    q[1] = foot_xy[1] - foot_ref[1]
    q[2] = z_ground - foot_ref[2]
    return q


def apply_base_attitude_limits(q_state: "State", config: ContactModelConfig) -> None:
    """Optional roll/pitch box limits on ``q`` (off by default in ``CitoFraxDynamics``)."""
    lim_r = np.deg2rad(config.roll_limit_deg)
    lim_p = np.deg2rad(config.pitch_limit_deg)
    q_state.min[config.roll_idx] = -lim_r
    q_state.max[config.roll_idx] = lim_r
    q_state.min[config.pitch_idx] = -lim_p
    q_state.max[config.pitch_idx] = lim_p


def build_cito_controls(
    robot: Monoped3D,
    config: ContactModelConfig,
) -> List["Control"]:
    """Contact / impulse controls (dFOH + optional impulsive). Does not include ``tau``."""
    from openscvx.symbolic.expr.control import Control

    nc = config.n_c
    contact_controls = [
        Control("phi_t_zoh", shape=(2 * nc,), parameterization="zoh"),
        Control("phi_t_foh", shape=(2 * nc,), parameterization="foh"),
        Control("phi_n_zoh", shape=(nc,), parameterization="zoh"),
        Control("phi_n_foh", shape=(nc,), parameterization="foh"),
        Control("gamma_zoh", shape=(nc,), parameterization="zoh"),
        Control("gamma_foh", shape=(nc,), parameterization="foh"),
    ]
    if config.enable_impulses:
        contact_controls.extend(
            [
                Control("Phi_t", shape=(2 * nc,), parameterization="impulsive"),
                Control("Phi_n", shape=(nc,), parameterization="impulsive"),
                Control("Gamma", shape=(nc,), parameterization="impulsive"),
            ]
        )
    fmax = config.max_normal_force
    ftmax = config.max_tangential_force
    for c in contact_controls:
        if "Phi" in c.name:
            if "t" in c.name:
                c.min = -config.max_impulse_tangential * np.ones(c.shape)
                c.max = config.max_impulse_tangential * np.ones(c.shape)
            else:
                c.min = np.zeros(c.shape)
                c.max = config.max_impulse_normal * np.ones(c.shape)
        elif "gamma" in c.name:
            c.min = np.zeros(c.shape)
            c.max = fmax * np.ones(c.shape)
        elif "phi_t" in c.name:
            c.min = -ftmax * np.ones(c.shape)
            c.max = ftmax * np.ones(c.shape)
        else:
            c.min = np.zeros(c.shape)
            c.max = fmax * np.ones(c.shape)
    return list(contact_controls)


def build_auxiliary_states(n_c: int = 1) -> List["State"]:
    """Five integral states per contact for cross-complementarity (paper eq. 9–11).

    ``y_sd`` integrates signed distance ``sd`` continuously; cross constraints pair
    ``Δy_phi_n`` with ``Δy_sd`` (not instantaneous ``sd`` at nodes).
    """
    from openscvx.symbolic.expr.state import State

    shape = (n_c,)
    aux = [
        State("y_phi_n", shape=shape),
        State("y_sd", shape=shape),
        State("y_lambda", shape=shape),
        State("y_gamma", shape=shape),
        State("y_rho", shape=shape),
    ]
    for s in aux:
        s.min = np.array([0.0])
        s.max = np.array([1e3])
    return aux


def build_control_layout(
    q: "State",
    qd: "State",
    tau: "Control",
    aux_states: Sequence["State"],
    contact_controls: Sequence["Control"],
) -> DfohControlLayout:
    """Build layout handles (slices resolved lazily on first BYOF call)."""
    by_name_c = {c.name: c for c in contact_controls}
    by_name_s = {s.name: s for s in aux_states}
    return DfohControlLayout(
        phi_t_zoh=by_name_c["phi_t_zoh"],
        phi_t_foh=by_name_c["phi_t_foh"],
        phi_n_zoh=by_name_c["phi_n_zoh"],
        phi_n_foh=by_name_c["phi_n_foh"],
        gamma_zoh=by_name_c["gamma_zoh"],
        gamma_foh=by_name_c["gamma_foh"],
        phi_t_imp=by_name_c.get("Phi_t"),
        phi_n_imp=by_name_c.get("Phi_n"),
        gamma_imp=by_name_c.get("Gamma"),
        tau=tau,
        q=q,
        qd=qd,
        y_phi_n=by_name_s.get("y_phi_n"),
        y_sd=by_name_s.get("y_sd"),
        y_lambda=by_name_s.get("y_lambda"),
        y_gamma=by_name_s.get("y_gamma"),
        y_rho=by_name_s.get("y_rho"),
    )


def configure_impulsive_nodes(controls: Sequence["Control"], n_nodes: int) -> None:
    """Enable impulsive contact controls at every node (required when ``nodes`` is unset)."""
    nodes = list(range(n_nodes))
    for control in controls:
        if control.parameterization == "impulsive":
            control.nodes = nodes


def cito_build_byof(
    robot: Monoped3D,
    layout: DfohControlLayout,
    config: ContactModelConfig,
    *,
    q: "State",
    qd: "State",
) -> dict:
    """Assemble ``ByofSpec`` dict for CITO (dynamics + complementarity)."""
    delta = config.delta
    eps = config.huber_eps
    _layout_slices: list = []

    def sl() -> DfohControlSlices:
        if not _layout_slices:
            _layout_slices.append(layout.slices())
        return _layout_slices[0]

    def _kin(q_val, qd_val, u):
        phi_t, phi_n, gamma = _phi_gamma_eff(u, sl())
        kin = contact_kinematics(robot, q_val, qd_val, z_ground=config.z_ground)
        lam = stationarity_measure(phi_n, kin["v_ct"], huber_eps=eps)
        rho = rho_friction_cone(phi_t, phi_n, gamma, mu=config.mu, huber_eps=eps)
        return kin, phi_t, phi_n, gamma, lam, rho

    nodal: list = []

    def _nodal_fb(a_fn, b_fn):
        nodal.append(
            {
                "constraint_fn": lambda x, u, node, params: fischer_burmeister(
                    a_fn(x, u, node, params), b_fn(x, u, node, params), delta
                ),
            }
        )

    def _xkin(x, u):
        s = sl()
        return _kin(x[s.q], x[s.qd], u)

    def _fb_phi_sd(x, u, node, params):
        del node, params
        kin, _, phi_n, _, _, _ = _xkin(x, u)
        return fischer_burmeister(phi_n, jnp.maximum(kin["sd"], 0.0), delta)

    def _fb_phi_lam(x, u, node, params):
        del node, params
        _, _, phi_n, _, lam, _ = _xkin(x, u)
        return fischer_burmeister(phi_n, lam, delta)

    def _fb_gamma_rho(x, u, node, params):
        del node, params
        _, _, _, gamma, _, rho = _xkin(x, u)
        return fischer_burmeister(gamma, rho, delta)

    nodal.append({"constraint_fn": _fb_phi_sd})
    if config.enable_friction_complementarity:
        nodal.append({"constraint_fn": _fb_phi_lam})
        nodal.append({"constraint_fn": _fb_gamma_rho})

    if config.enable_impulses:

        def kin_imp(x, u):
            s = sl()
            kin = contact_kinematics(robot, x[s.q], x[s.qd], z_ground=config.z_ground)
            phi_t = u[s.phi_t_imp]
            phi_n = u[s.phi_n_imp]
            gam = u[s.gamma_imp]
            lam = stationarity_measure(phi_n, kin["v_ct"], huber_eps=eps)
            rho = rho_friction_cone(phi_t, phi_n, gam, mu=config.mu, huber_eps=eps)
            return phi_n, kin["sd"], lam, rho, gam

        def _fb_imp_phi_sd(x, u, node, params):
            del node, params
            phi_n, sd, _, _, _ = kin_imp(x, u)
            return fischer_burmeister(phi_n, jnp.maximum(sd, 0.0), delta)

        def _fb_imp_phi_lam(x, u, node, params):
            del node, params
            phi_n, _, lam, _, _ = kin_imp(x, u)
            return fischer_burmeister(phi_n, lam, delta)

        def _fb_imp_gamma_rho(x, u, node, params):
            del node, params
            _, _, _, rho, gam = kin_imp(x, u)
            return fischer_burmeister(gam, rho, delta)

        nodal.extend(
            [
                {"constraint_fn": _fb_imp_phi_sd},
                {"constraint_fn": _fb_imp_phi_lam},
                {"constraint_fn": _fb_imp_gamma_rho},
            ]
        )

    def _cross_pair_sum(y_a: slice, y_b: slice, X, U, params):
        del U, params
        vals = []
        for k in range(X.shape[0] - 1):
            a0, a1 = X[k, y_a], X[k + 1, y_a]
            b0, b1 = X[k, y_b], X[k + 1, y_b]
            da = jnp.maximum(a1 - a0, 0.0)
            db = jnp.maximum(b1 - b0, 0.0)
            fb = jax.vmap(
                lambda d_ai, d_bi: fischer_burmeister(d_ai, d_bi, delta),
                in_axes=0,
            )(da, db)
            vals.append(jnp.sum(fb))
        return jnp.sum(jnp.stack(vals)) if vals else jnp.asarray(0.0)

    cross_nodal: list = []
    if config.enable_cross_complementarity:
        cross_nodal = [
            lambda X, U, params: _cross_pair_sum(sl().y_phi_n, sl().y_sd, X, U, params),
            lambda X, U, params: _cross_pair_sum(sl().y_phi_n, sl().y_lambda, X, U, params),
            lambda X, U, params: _cross_pair_sum(sl().y_gamma, sl().y_rho, X, U, params),
        ]

    dynamics: dict = {
        "qd": cito_qdd_byof(robot, layout, q_arg=q, qd_arg=qd, tau_arg=layout.tau),
    }
    if config.enable_cross_complementarity:
        dynamics.update(
            {
                "y_phi_n": cito_aux_integrand_byof(
                    robot, layout, "phi_n", q_arg=q, qd_arg=qd, config=config
                ),
                "y_sd": cito_aux_integrand_byof(
                    robot, layout, "sd", q_arg=q, qd_arg=qd, config=config
                ),
                "y_lambda": cito_aux_integrand_byof(
                    robot, layout, "lambda", q_arg=q, qd_arg=qd, config=config
                ),
                "y_gamma": cito_aux_integrand_byof(
                    robot, layout, "gamma", q_arg=q, qd_arg=qd, config=config
                ),
                "y_rho": cito_aux_integrand_byof(
                    robot, layout, "rho", q_arg=q, qd_arg=qd, config=config
                ),
            }
        )
    dynamics_discrete = {}
    if config.enable_impulses:
        dynamics_discrete["qd"] = cito_impact_qd_byof(robot, layout, config, q_arg=q, qd_arg=qd)

    return {
        "dynamics": dynamics,
        "dynamics_discrete": dynamics_discrete,
        "nodal_constraints": nodal,
        "cross_nodal_constraints": cross_nodal,
    }


class CitoFraxDynamics(DynamicsAdapter):
    """Frax monoped with CITO contact controls and BYOF dynamics."""

    def __init__(
        self,
        robot: Monoped3D | None = None,
        *,
        config: ContactModelConfig | None = None,
        urdf_filename: str | Path | None = None,
    ) -> None:
        from openscvx.symbolic.expr.control import Control
        from openscvx.symbolic.expr.state import State

        self._robot = robot if robot is not None else load_monoped_3d(urdf_filename)
        self.config = config if config is not None else ContactModelConfig()
        nj = self._robot.num_joints
        na = effective_actuated_joint_count(self._robot)

        self._q = State("q", shape=(nj,))
        self._qd = State("qd", shape=(nj,))
        self._tau = Control("tau", shape=(na,), parameterization="foh")

        q_min, q_max, qd_min, qd_max, _, _ = _bounds_from_robot(self._robot)
        tau_min, tau_max = actuated_torque_bounds(self._robot)
        self._q.min = q_min
        self._q.max = q_max
        self._qd.min = qd_min
        self._qd.max = qd_max
        self._tau.min = tau_min
        self._tau.max = tau_max
        if self.config.apply_attitude_path_limits:
            apply_base_attitude_limits(self._q, self.config)

        self._aux = (
            build_auxiliary_states(self.config.n_c)
            if self.config.enable_cross_complementarity
            else []
        )
        self._contact_controls = build_cito_controls(self._robot, self.config)
        self.states = [self._q, self._qd, *self._aux]
        self.controls = [self._tau, *self._contact_controls]
        self._layout = build_control_layout(
            self._q,
            self._qd,
            self._tau,
            self._aux,
            self._contact_controls,
        )

    @property
    def robot(self) -> Monoped3D:
        return self._robot

    def layout(self) -> DfohControlLayout:
        return self._layout

    def expand(self) -> Tuple[dict, dict]:
        layout = self._layout
        byof_dict = cito_build_byof(
            self._robot,
            layout,
            self.config,
            q=self._q,
            qd=self._qd,
        )
        return {"q": self._qd}, byof_dict
