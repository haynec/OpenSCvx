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
    huber_eps: float = 1e-3
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
            return phi_n
        if which == "sd":
            return kin["sd"]
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


def build_auxiliary_states() -> List["State"]:
    from openscvx.symbolic.expr.state import State

    aux = [
        State("y_phi_n", shape=(1,)),
        State("y_lambda", shape=(1,)),
        State("y_gamma", shape=(1,)),
        State("y_rho", shape=(1,)),
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

    def _sd_at(Xk, s: DfohControlSlices):
        return jnp.maximum(
            contact_kinematics(robot, Xk[s.q], Xk[s.qd], z_ground=config.z_ground)["sd"],
            0.0,
        )

    def _cross_pair_sum(y_a, y_b, X, U, params):
        del U, params
        s = sl()
        vals = []
        for k in range(X.shape[0] - 1):
            a0, a1 = X[k, y_a], X[k + 1, y_a]
            if y_b == "sd":
                b0, b1 = _sd_at(X[k], s), _sd_at(X[k + 1], s)
            else:
                b0, b1 = X[k, y_b], X[k + 1, y_b]
            da = jnp.maximum(a1 - a0, 0.0)
            db = jnp.maximum(b1 - b0, 0.0)
            vals.append(fischer_burmeister(da, db, delta))
        return jnp.sum(jnp.stack(vals)) if vals else jnp.asarray(0.0)

    cross_nodal: list = []
    if config.enable_cross_complementarity:
        cross_nodal = [
            lambda X, U, params: _cross_pair_sum(sl().y_phi_n, "sd", X, U, params),
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
            build_auxiliary_states() if self.config.enable_cross_complementarity else []
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
