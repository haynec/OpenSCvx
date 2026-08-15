"""Franka FR3 v2 pick-and-place with Product of Exponentials FK.

Pick-and-place trajectory optimisation for the Franka Research 3 v2 arm
from MuJoCo Menagerie (``third_party/mujoco_menagerie/franka_fr3_v2``).

Kinematics:
    Space-frame Product of Exponentials (PoE) forward kinematics with screw
    axes derived analytically from the FR3 v2 MJCF kinematic chain:

        T_ee(q) = exp(ξ₁q₁) @ ··· @ exp(ξ₇q₇) @ T_home

    Contacts are **not** modelled; dynamics use a simple diagonal-inertia
    approximation (I q̈ = τ), sufficient to demonstrate Lie-algebra FK and
    IK-seeded trajectory planning.

Task:
    Five-segment trajectory with EE position and approach-orientation constraints:

        home → pre_grasp → grasp → pre_grasp → pre_place → place

    Self-collision and a spherical workspace obstacle are enforced with the same
    sampled-segment model as ``7_dof_arm_collision.py`` (non-adjacent link
    capsule pairs + obstacle clearance along each link).

Visualization:
    When ``mujoco`` and ``trimesh`` are installed, the Viser window renders the
    actual FR3 v2 CAD meshes (OBJ files from the menagerie ``assets/`` directory)
    with per-link colouring animated via MuJoCo FK. Each link mesh is drawn in its
    body frame and posed with ``position`` / ``wxyz`` each frame (not full vertex
    rewrites), which keeps Viser recordings much smaller. Falls back to animated line
    segments if those packages are unavailable. The EE path uses a growing trail
    plus a moving marker (no separate full-path point cloud).

    After optimisation, the converged torque sequence is replayed in MuJoCo's
    CPU simulator using ``data.qfrc_applied`` (generalized joint torques in Nm) on a
    patched MJCF with **no** position/motor actuators, matching OpenSCvx's τ in
    ``q̈ = diag(I)^{-1} τ`` (printed diagnostics when rollout runs).

    Embedded Viser recording (``--export-viser``) uses the **same CAD link meshes**
    as the interactive viewer when ``mujoco`` and ``trimesh`` are available; export
    fails fast with an error if CAD cannot be loaded (line fallback is not accepted
    for embedded docs). Prefer rigid link poses over vertex streaming so exports stay
    smaller without aggressive temporal subsampling::

        python examples/arm/franka_fr3v2_pick_place.py --export-viser

    Default output: ``docs/assets/viser-recordings/franka_fr3v2_pick_place.viser``
    (from repo root). The initial camera sits on **world +X** (motion mostly in **Y**)
    so the path is seen face-on from the opposite side as −X. See
    ``examples/_viser_embed_export.py``.

Requirements:
    pip install openscvx[lie]                # jaxlie for SE3Exp / SO3Log
    pip install mujoco trimesh               # CAD Viser (required for ``--export-viser``)
"""

import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting_viser import (
    build_cad_link_snapshot_builder,
    create_snapshot_plotting_server,
)
from openscvx.plotting import plot_controls, plot_scp_iterations

# =============================================================================
# FR3 v2 Kinematic Parameters
# =============================================================================
# Screw axes and T_home derived from mujoco_menagerie/franka_fr3_v2/fr3v2.xml.
#
# At zero configuration (q = 0) the kinematic chain accumulates rotations:
#   link2 ← Rx(−π/2),  link3 ← Rx(+π/2) → net I,  link4 ← Rx(+π/2),
#   link5 ← Rx(−π/2) → net I,  link6 ← Rx(+π/2),  link7 ← Rx(+π/2)
#   → cumulative link7 orientation = Rx(π) = diag(1,−1,−1).
#
# Flange (link8) is offset [0, 0, 0.107] in link7 local frame, which maps to
# [0, 0, −0.107] in world, so EE world pos = [d7, 0, z567 − d_ee].

N_JOINTS = 7

# Body-offset distances from fr3v2.xml <body pos="..."> attributes (metres)
d1 = 0.333  # world-z of joint 1 (link0 → link1)
d3 = 0.316  # z-offset link2 → link3 (in link2 local frame)
a4 = 0.0825  # x-offset link3 → link4 (in link3 local frame)
d5 = 0.384  # z-offset link4 → link5 (in link4 local frame)
d7 = 0.088  # x-offset link6 → link7 (in link6 local frame)
d_ee = 0.107  # z-offset link7 → flange (in link7 local; maps to −z world)

# World-frame z-heights of joint axes at zero configuration
z1 = d1
z35 = d1 + d3  # joints 3 and 4
z567 = d1 + d3 + d5  # joints 5, 6 and 7

# Joint limits (rad) — from fr3v2.xml <joint range="lo hi"/>
# Note: J4 is always negative (elbow stay bent), J6 always positive.
angle_min = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
angle_max = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])

# Torque limits (Nm) — from fr3v2.xml <joint actuatorfrcrange="−τ τ"/>
torque_max = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# Simplified diagonal inertias (kg·m²) — decreasing from base to tip
inertia = np.array([0.10, 0.08, 0.07, 0.06, 0.03, 0.02, 0.01])

# FR3 home keyframe from fr3v2.xml <key name="home" qpos="..."/>
q_home = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4])

# =============================================================================
# Screw Axes  ξᵢ = [v; ω]  (space-frame PoE, Modern Robotics convention)
# =============================================================================
# v = −ω × q₀,  where q₀ is any point on the joint axis at zero config.
#
# Zero-config joint positions and axes in world frame:
#   J1  [0,    0,   z1  ],  ω = [0,  0,  1]   base yaw
#   J2  [0,    0,   z1  ],  ω = [0,  1,  0]   shoulder pitch
#   J3  [0,    0,   z35 ],  ω = [0,  0,  1]   forearm roll
#   J4  [a4,   0,   z35 ],  ω = [0, −1,  0]   elbow pitch  (−y because Rx(π/2))
#   J5  [0,    0,   z567],  ω = [0,  0,  1]   wrist roll
#   J6  [0,    0,   z567],  ω = [0, −1,  0]   wrist pitch  (−y because Rx(π/2))
#   J7  [d7,   0,   z567],  ω = [0,  0, −1]   tool roll    (−z because Rx(π))

screw_axes = np.array(
    [
        # J1  z-axis through (x=0, y=0): v = −[0,0,1] × [0,0,z1] = 0
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        # J2  y-axis at [0,0,z1]: v = −[0,1,0] × [0,0,z1] = [−z1, 0, 0]
        [-z1, 0.0, 0.0, 0.0, 1.0, 0.0],
        # J3  z-axis through (x=0, y=0): v = 0
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        # J4  −y at [a4,0,z35]: v = −[0,−1,0] × [a4,0,z35] = [z35, 0, −a4]
        [z35, 0.0, -a4, 0.0, -1.0, 0.0],
        # J5  z-axis through (x=0, y=0): v = 0
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        # J6  −y at [0,0,z567]: v = −[0,−1,0] × [0,0,z567] = [z567, 0, 0]
        [z567, 0.0, 0.0, 0.0, -1.0, 0.0],
        # J7  −z at [d7,0,z567]: v = −[0,0,−1] × [d7,0,z567] = [0, d7, 0]
        [0.0, d7, 0.0, 0.0, 0.0, -1.0],
    ]
)

# EE (flange) transform at zero configuration
# Rotation accumulated = Rx(π) = diag(1, −1, −1)
# Position = [d7, 0, z567 − d_ee]  (d_ee flips sign under Rx(π))
T_home = np.array(
    [
        [1.0, 0.0, 0.0, d7],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, z567 - d_ee],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Joint-axis anchor points at q = 0 (world frame, m). Used for collision
# geometry keypoints and Viser fallback kinematics (matches PoE comments above).
_joint_zero_pos = np.array(
    [
        [0.0, 0.0, z1],
        [0.0, 0.0, z1],
        [0.0, 0.0, z35],
        [a4, 0.0, z35],
        [0.0, 0.0, z567],
        [0.0, 0.0, z567],
        [d7, 0.0, z567],
    ]
)

# =============================================================================
# Pick-and-Place Waypoints
# =============================================================================
# Desired EE orientation during grasping: z-axis pointing world −z ("down").
# This matches T_home, whose z-column is [0, 0, −1].
# Quaternion wxyz for Rx(π): [0, 1, 0, 0].

R_des = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
q_down = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, Rx(π), EE −z → world −z

pre_height = 0.15  # m above grasp / place positions

home_pos = np.array([0.45, 0.00, 0.55])
grasp_pos = np.array([0.45, -0.25, 0.06])
place_pos = np.array([0.45, 0.25, 0.06])
pre_grasp_pos = grasp_pos + np.array([0.0, 0.0, pre_height])
pre_place_pos = place_pos + np.array([0.0, 0.0, pre_height])

waypoint_names = ["home", "pre_grasp", "grasp", "pre_grasp", "pre_place", "place"]
waypoint_positions = [home_pos, pre_grasp_pos, grasp_pos, pre_grasp_pos, pre_place_pos, place_pos]
# Use EE-pointing-down orientation for all IK keyframes
waypoint_orientations = [q_down] * len(waypoint_names)

nodes_per_segment = 1
n_segments = len(waypoint_positions) - 1
n = nodes_per_segment * n_segments + 1
waypoint_nodes = [i * nodes_per_segment for i in range(len(waypoint_positions))]
total_time = 5.0

# =============================================================================
# Collision / obstacle (same structure as 7_dof_arm_collision.py)
# =============================================================================
# Spherical obstacle in the task workspace (between pedestal height and home).
obstacle_center = np.array([0.45, 0.0, 0.28])
obstacle_radius = 0.055
n_collision_samples = 4

# =============================================================================
# States and Controls
# =============================================================================

angle = ox.State("angle", shape=(N_JOINTS,))
angle.max = angle_max
angle.min = angle_min
angle.initial = np.clip(q_home, angle_min, angle_max)
angle.final = [("free", 0.0)] * N_JOINTS

velocity = ox.State("velocity", shape=(N_JOINTS,))
velocity.max = np.full(N_JOINTS, 2.5)
velocity.min = -velocity.max
velocity.initial = np.zeros(N_JOINTS)
velocity.final = np.zeros(N_JOINTS)

states = [angle, velocity]

torque = ox.Control("torque", shape=(N_JOINTS,))
torque.max = torque_max
torque.min = -torque_max

controls = [torque]

# =============================================================================
# Forward Kinematics (Product of Exponentials)
# =============================================================================
# Build the FK chain incrementally so intermediate transforms are available
# for algebraic_prop (used in post-processing for visualisation).

T_chain = ox.Constant(np.eye(4))
joint_transforms = {}
for j in range(N_JOINTS):
    xi = ox.Constant(screw_axes[j])
    T_chain = T_chain @ ox.lie.SE3Exp(xi * angle[j])
    joint_transforms[f"T_j{j + 1}"] = T_chain

T_ee = T_chain @ ox.Constant(T_home)
p_ee = ox.Concat(T_ee[0, 3], T_ee[1, 3], T_ee[2, 3])

# =============================================================================
# Dynamics  (simplified: I q̈ = τ)
# =============================================================================
# A full Lagrangian model would add Coriolis, gravity and coupled inertia.
# The simple model here is sufficient to demonstrate the PoE FK constraints.

I_inv = ox.Constant(1.0 / inertia)
dynamics = {
    "angle": velocity,
    "velocity": I_inv * torque,
}

# =============================================================================
# Constraints
# =============================================================================

constraints = []

# State box constraints (CTCS = continuous-time constraint satisfaction)
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# EE must stay above the floor
constraints.append(ox.ctcs(p_ee[2] >= 0.0))

# EE position constraint at each waypoint node (1 cm tolerance)
ee_tol = 0.01
for i, (wp, node) in enumerate(zip(waypoint_positions, waypoint_nodes)):
    wp_param = ox.Parameter(f"wp_{i}", shape=(3,), value=wp)
    constraints.append((ox.linalg.Norm(p_ee - wp_param, ord=2) <= ee_tol).at([node]))

# Orientation constraint: keep EE z-axis pointing down during approach / contact.
# ori_error = SO3Log(R_des.T @ R_ee); its norm is the geodesic rotation angle.
ori_error = ox.lie.SO3Log(R_des.T @ T_ee[:3, :3])
ori_norm = ox.linalg.Norm(ori_error, ord=2)

ori_tol_loose = np.deg2rad(60.0)  # relaxed during transit
ori_tol_tight = np.deg2rad(5.0)  # precise at grasp / place approach

# Loose over the full approach-to-place range
constraints.append((ori_norm <= ori_tol_loose).over((waypoint_nodes[1], waypoint_nodes[-1])))
# Tight from pre_grasp through the grasp-return waypoint
constraints.append((ori_norm <= ori_tol_tight).over((waypoint_nodes[1], waypoint_nodes[3])))
# Tight from pre_place to place
constraints.append((ori_norm <= ori_tol_tight).over((waypoint_nodes[4], waypoint_nodes[5])))

# Self-collision: sampled segment--segment distance for non-adjacent link pairs.
# Spherical obstacle: sample each link segment; enforce distance to obstacle
# center >= obstacle_radius + link radius.
joint_axis_home_h = [np.append(_joint_zero_pos[i], 1.0) for i in range(N_JOINTS)]

kp = {}
T_I = ox.Constant(np.eye(4))
p_base_h = T_I @ ox.Constant(np.array([0.0, 0.0, 0.0, 1.0]))
kp["base"] = ox.Concat(p_base_h[0], p_base_h[1], p_base_h[2])
for k in range(N_JOINTS):
    T_k = joint_transforms[f"T_j{k + 1}"]
    ph = T_k @ ox.Constant(joint_axis_home_h[k])
    kp[f"j{k + 1}"] = ox.Concat(ph[0], ph[1], ph[2])
kp["ee"] = p_ee

link_specs = [
    ("base", "j1", 0.065),
    ("j1", "j2", 0.055),
    ("j2", "j3", 0.052),
    ("j3", "j4", 0.048),
    ("j4", "j5", 0.042),
    ("j5", "j6", 0.038),
    ("j6", "j7", 0.035),
    ("j7", "ee", 0.032),
]

ts = np.linspace(0.0, 1.0, n_collision_samples)
tt = np.stack(np.meshgrid(ts, ts, indexing="ij"), axis=-1).reshape(-1, 2)

p_obstacle = ox.Constant(obstacle_center)
for a_start, a_end, r_link in link_specs:
    pa0, pa1 = kp[a_start], kp[a_end]
    r_clear = obstacle_radius + r_link
    obs_dists = ox.Vmap(
        lambda t, pa0=pa0, pa1=pa1: ox.linalg.Norm(((1 - t) * pa0 + t * pa1) - p_obstacle),
        batch=ts,
    )
    constraints.append(ox.ctcs(r_clear <= obs_dists))

# =============================================================================
# Initial Guess (IK interpolation)
# =============================================================================
# Seed with the FR3 home pose; sequential IK propagates solutions along the path.

angle.guess = ox.init.ik_interpolation(
    keyframes=list(zip(waypoint_positions, waypoint_orientations)),
    nodes=waypoint_nodes,
    screw_axes=screw_axes,
    T_home=T_home,
    angles_init=q_home,
    angles_min=angle_min,
    angles_max=angle_max,
    sequential=True,
)
angle.initial = np.clip(angle.guess[0], angle_min, angle_max)
velocity.guess = np.zeros((n, N_JOINTS))
torque.guess = np.zeros((n, N_JOINTS))

# =============================================================================
# Problem
# =============================================================================

time = ox.Time(
    initial=0.0,
    final=ox.Minimize(total_time),
    min=0.0,
    max=total_time,
)

problem = ox.Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    algorithm={
        "lam_vb": 1e1,
        "lam_vc": 4e2,
        "lam_cost": 4e-1,
        "autotuner": ox.AugmentedLagrangian(eta_lambda=1e0, ep=1e-1),
    },
    algebraic_prop={
        "ee_position": p_ee,
        **{name: T for name, T in joint_transforms.items()},
    },
)

problem.settings.prp.dt = 0.01

# MuJoCo rollout: ``results.t_full`` + ``results.u_full[:, true_control_slice]`` matches
# ``post_process`` timing and **true** controls only (excludes ``_time_dilation``).
# Torques are applied as ``data.qfrc_applied`` on an actuator-free model — OpenSCvx τ is a
# generalized joint torque (Nm), not Menagerie ``position`` / ``motor`` ``ctrl``.
MUJOCO_ROLLOUT_ZERO_GRAVITY = True


def _patch_fr3_xml_no_actuators(xml_text: str) -> str:
    """Remove Menagerie PD actuators; rollout uses ``qfrc_applied`` only."""
    m = re.search(r"<actuator>.*?</actuator>", xml_text, flags=re.DOTALL)
    if m is None:
        raise ValueError("Expected an <actuator>...</actuator> block in fr3v2.xml")
    return xml_text[: m.start()] + "  <actuator>\n  </actuator>\n" + xml_text[m.end() :]


def simulate_franka_torque_rollout(results, xml_path: Path, *, problem) -> dict:
    """Replay SCP generalized torques with MuJoCo ``qfrc_applied``.

    Torques are read from ``results.u_full[:, problem.settings.sim.true_control_slice]`` —
    exactly the **true** (non-``_``) control components of the unified vector produced in
    ``propagate_trajectory_results``, i.e. the same τ as in ``velocity_dot = I^{-1} τ``.

    Menagerie ``motor`` / ``position`` ``ctrl`` semantics do **not** match that model; we
    clear ``<actuator>`` and write τ into ``data.qfrc_applied`` at each hinge DOF address.

    Args:
        results: Post-processed OpenSCvx results containing ``t_full``, ``u_full``,
            and propagated state trajectories.
        xml_path: Path to the FR3 menagerie XML file.
        problem: OpenSCvx problem instance used to read ``true_control_slice``.

    Returns:
        dict: Rollout record with keys ``time``, ``qpos``, ``qvel``, ``ee_link8``.
    """
    import mujoco

    if getattr(results, "t_full", None) is None or getattr(results, "u_full", None) is None:
        raise RuntimeError(
            "Run ``problem.post_process()`` before MuJoCo rollout (needs t_full / u_full)."
        )

    t_series = np.asarray(results.t_full, dtype=np.float64).ravel()
    torque_sl = problem.settings.sim.true_control_slice
    traj_torque = np.asarray(results.u_full[:, torque_sl], dtype=np.float64)
    if traj_torque.ndim == 1:
        traj_torque = traj_torque.reshape(-1, N_JOINTS)
    if traj_torque.shape[1] != N_JOINTS:
        raise ValueError(
            f"Expected {N_JOINTS} torque columns from "
            f"true_control_slice {torque_sl}, got {traj_torque.shape}."
        )

    traj_angle = np.asarray(results.trajectory["angle"], dtype=np.float64)
    traj_vel = np.asarray(results.trajectory["velocity"], dtype=np.float64)

    n_t = len(t_series)
    if traj_torque.shape[0] != n_t:
        raise ValueError(
            f"Trajectory/time length mismatch: len(t_full)={n_t}, "
            f"torque rows={traj_torque.shape[0]}."
        )

    xml_path = Path(xml_path)
    patched_xml = _patch_fr3_xml_no_actuators(xml_path.read_text(encoding="utf-8"))

    fd, tmp_path = tempfile.mkstemp(
        suffix="_fr3_qfrc_rollout.xml",
        dir=str(xml_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(patched_xml)
        mj_roll = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if MUJOCO_ROLLOUT_ZERO_GRAVITY:
        mj_roll.opt.gravity[:] = 0.0
    mj_roll.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    if mj_roll.nu != 0:
        raise ValueError(
            f"Expected actuator-free rollout model (nu=0); got nu={mj_roll.nu}. "
            "XML patch should leave an empty <actuator> block."
        )
    if mj_roll.nv < N_JOINTS or mj_roll.nq < N_JOINTS:
        raise ValueError(f"Unexpected MuJoCo dims nq={mj_roll.nq}, nv={mj_roll.nv}.")

    dof_addrs: list[int] = []
    for i in range(1, N_JOINTS + 1):
        jid = mujoco.mj_name2id(mj_roll, mujoco.mjtObj.mjOBJ_JOINT, f"fr3v2_joint{i}")
        if jid < 0:
            raise ValueError(f"Joint fr3v2_joint{i} not found in MuJoCo model.")
        dof_addrs.append(int(mj_roll.jnt_dofadr[jid]))

    data = mujoco.MjData(mj_roll)
    data.qpos[:N_JOINTS] = traj_angle[0, :N_JOINTS]
    data.qvel[:N_JOINTS] = traj_vel[0, :N_JOINTS]

    t_lim = np.asarray(torque_max, dtype=np.float64)

    ee_bid = mujoco.mj_name2id(mj_roll, mujoco.mjtObj.mjOBJ_BODY, "fr3v2_link8")
    if ee_bid < 0:
        raise ValueError("Body fr3v2_link8 not found (menagerie flange link).")

    rec_t = np.zeros(n_t)
    rec_q = np.zeros((n_t, mj_roll.nq))
    rec_qd = np.zeros((n_t, mj_roll.nv))
    rec_ee = np.zeros((n_t, 3))

    rec_t[0] = t_series[0]
    rec_q[0] = data.qpos.copy()
    rec_qd[0] = data.qvel.copy()
    mujoco.mj_forward(mj_roll, data)
    rec_ee[0] = data.xpos[ee_bid].copy()

    for i in range(n_t - 1):
        dt_seg = float(t_series[i + 1] - t_series[i])
        if dt_seg <= 1e-12:
            rec_t[i + 1] = t_series[i + 1]
            rec_q[i + 1] = data.qpos.copy()
            rec_qd[i + 1] = data.qvel.copy()
            mujoco.mj_forward(mj_roll, data)
            rec_ee[i + 1] = data.xpos[ee_bid].copy()
            continue

        mj_roll.opt.timestep = dt_seg
        tau_mid = 0.5 * (traj_torque[i] + traj_torque[i + 1])
        tau_mid = np.clip(tau_mid, -t_lim, t_lim)

        data.qfrc_applied[:] = 0.0
        for k, adr in enumerate(dof_addrs):
            data.qfrc_applied[adr] = float(tau_mid[k])

        mujoco.mj_step(mj_roll, data)

        rec_t[i + 1] = t_series[i + 1]
        rec_q[i + 1] = data.qpos.copy()
        rec_qd[i + 1] = data.qvel.copy()
        rec_ee[i + 1] = data.xpos[ee_bid].copy()

    q_fin_prop = traj_angle[-1, :N_JOINTS]
    q_fin_sim = rec_q[-1, :N_JOINTS]
    print(
        f"MuJoCo τ rollout (qfrc_applied, u_full slice {torque_sl}): {n_t - 1} segments | "
        f"‖τ‖_max={float(np.max(np.abs(traj_torque))):.3f} Nm | "
        f"‖q_sim − q_prop‖ at t_end = {np.linalg.norm(q_fin_sim - q_fin_prop):.4f} rad"
    )
    return {"time": rec_t, "qpos": rec_q, "qvel": rec_qd, "ee_link8": rec_ee}


def _load_viser_embed_export_module():
    """Load ``examples/_viser_embed_export.py`` (no ``examples`` package required)."""
    embed_py = Path(__file__).resolve().parent.parent / "_viser_embed_export.py"
    spec = importlib.util.spec_from_file_location("_viser_embed_export_franka", embed_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Viser export helpers from {embed_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    _vis_embed_mod = None
    _export_viser_path: Path | None = None
    if "--export-viser" in sys.argv:
        _vis_embed_mod = _load_viser_embed_export_module()
        _export_viser_path = _vis_embed_mod.parse_export_viser_path(
            default_output=Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "assets"
            / "viser-recordings"
            / "franka_fr3v2_pick_place.viser"
        )

    print("Franka FR3 v2 Pick-and-Place — PoE FK + OpenSCvx")
    print("=" * 60)
    print(
        f"Obstacle: center={list(np.round(obstacle_center, 3))} m, "
        f"r={obstacle_radius} m (spherical; samples/link={n_collision_samples})"
    )
    print(f"Nodes: {n}  ({nodes_per_segment} per segment, {n_segments} segments)")
    for name, pos, node in zip(waypoint_names, waypoint_positions, waypoint_nodes):
        print(f"  {name:>12s} (node {node:2d}): {list(np.round(pos, 3))}")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    ee_pos_traj = results.trajectory["ee_position"]  # (T_frames, 3)
    final_q = results.trajectory["angle"][-1]

    print()
    print("Results:")
    print(f"  Final joint angles [deg]: {np.round(np.rad2deg(final_q), 1)}")
    for i, (name, wp, node) in enumerate(zip(waypoint_names, waypoint_positions, waypoint_nodes)):
        t_node = node / (n - 1)
        frame_idx = int(round(t_node * (len(ee_pos_traj) - 1)))
        ee_at_wp = ee_pos_traj[frame_idx]
        err = np.linalg.norm(ee_at_wp - wp)
        print(
            f"  {name:>12s}: "
            f"pos=[{ee_at_wp[0]:.3f}, {ee_at_wp[1]:.3f}, {ee_at_wp[2]:.3f}]  "
            f"err={err:.4f} m"
        )

    if _export_viser_path is None:
        plot_scp_iterations(results).show()
        plot_controls(results).show()

    sim_rollout = None
    try:
        from openscvx.integrations.menagerie import get_xml_path

        _xml_roll = Path(str(get_xml_path("franka_fr3_v2", prefer_mjx=False)))
        sim_rollout = simulate_franka_torque_rollout(results, _xml_roll, problem=problem)
    except Exception as exc:
        print(f"[mujoco rollout] skipped ({type(exc).__name__}: {exc})")

    # =========================================================================
    # Viser 3D Visualisation
    # =========================================================================

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        add_ellipsoid_obstacles,
        add_position_marker,
        add_target_markers,
        compute_velocity_colors,
        create_server,
    )

    angle_traj = np.asarray(results.trajectory["angle"])  # (T_frames, 7)
    n_frames = len(angle_traj)
    traj_time = np.asarray(results.trajectory["time"]).flatten()

    # -------------------------------------------------------------------------
    # Load FR3 v2 CAD meshes (trimesh) and per-frame link transforms (MuJoCo FK).
    # -------------------------------------------------------------------------
    _use_cad_mesh = False
    _link_meshes_local: dict = {}  # link_name → (vertices_local, faces)
    _link_body_ids: dict = {}  # link_name → MuJoCo body_id
    _link_world_T: dict = {}  # link_name → (n_frames, 4, 4)
    mj_model_vis = None
    mj_data_vis = None

    try:
        import mujoco
        import trimesh  # type: ignore

        from openscvx.integrations.menagerie import get_xml_path

        xml_path = str(get_xml_path("franka_fr3_v2", prefer_mjx=False))
        mj_model_vis = mujoco.MjModel.from_xml_path(xml_path)
        # Disable contacts — we only need kinematics for visualisation.
        mj_model_vis.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        mj_data_vis = mujoco.MjData(mj_model_vis)
        asset_dir = Path(xml_path).parent / "assets"

        # Visual OBJ files per body — taken from <geom class="visual" mesh="..."/>
        link_visual_files = {
            "fr3v2_link0": [
                "link0_0.obj",
                "link0_1.obj",
                "link0_2.obj",
                "link0_3.obj",
                "link0_4.obj",
                "link0_5.obj",
            ],
            "fr3v2_link1": ["link1_3.obj"],
            "fr3v2_link2": ["link2_3.obj"],
            "fr3v2_link3": ["link3_3.obj"],
            "fr3v2_link4": ["link4_3.obj"],
            "fr3v2_link5": ["link5_3.obj", "link5_4.obj"],
            "fr3v2_link6": [
                "link6_0.obj",
                "link6_1.obj",
                "link6_2.obj",
                "link6_3.obj",
                "link6_4.obj",
                "link6_5.obj",
                "link6_6.obj",
                "link6_7.obj",
                "link6_8.obj",
                "link6_9.obj",
            ],
            "fr3v2_link7": [
                "link7_0.obj",
                "link7_1.obj",
                "link7_2.obj",
                "link7_3.obj",
                "link7_4.obj",
                "link7_5.obj",
                "link7_6.obj",
            ],
        }

        for link_name, files in link_visual_files.items():
            all_verts, all_faces, offset = [], [], 0
            for fname in files:
                obj_path = asset_dir / fname
                if not obj_path.exists():
                    continue
                tm = trimesh.load(str(obj_path), force="mesh", process=False)
                all_verts.append(np.asarray(tm.vertices, dtype=np.float32))
                all_faces.append(np.asarray(tm.faces, dtype=np.uint32) + offset)
                offset += len(tm.vertices)
            if not all_verts:
                continue
            _link_meshes_local[link_name] = (
                np.vstack(all_verts),
                np.vstack(all_faces),
            )
            _link_body_ids[link_name] = mujoco.mj_name2id(
                mj_model_vis, mujoco.mjtObj.mjOBJ_BODY, link_name
            )

        # Pre-compute world-frame transforms at every trajectory frame
        for name in _link_meshes_local:
            _link_world_T[name] = np.zeros((n_frames, 4, 4))

        for t_idx in range(n_frames):
            mj_data_vis.qpos[:N_JOINTS] = angle_traj[t_idx]
            mujoco.mj_kinematics(mj_model_vis, mj_data_vis)
            for name, body_id in _link_body_ids.items():
                T = np.eye(4)
                T[:3, :3] = mj_data_vis.xmat[body_id].copy().reshape(3, 3)
                T[:3, 3] = mj_data_vis.xpos[body_id].copy()
                _link_world_T[name][t_idx] = T

        _use_cad_mesh = len(_link_meshes_local) > 0
        if _use_cad_mesh:
            print(
                f"[viser] Loaded {len(_link_meshes_local)} CAD link meshes from menagerie assets."
            )

    except Exception as exc:
        print(
            f"[viser] CAD mesh unavailable "
            f"({type(exc).__name__}: {exc}); falling back to line segments."
        )

    if _export_viser_path is not None and not _use_cad_mesh:
        raise SystemExit(
            "--export-viser requires FR3 v2 CAD meshes (mujoco + trimesh and menagerie "
            "``franka_fr3_v2`` assets). Interactive mode can still use line segments; "
            "embed export does not. Underlying error was printed above."
        )

    # -------------------------------------------------------------------------
    # PoE-based joint keypoints for fallback line-segment visualisation
    # and joint frame axes (always computed, independent of CAD meshes).
    # -------------------------------------------------------------------------
    # Each T_j{k} is the cumulative PoE product after k joints.
    # _joint_zero_pos is defined with kinematic parameters at module scope.

    # (n_frames, N_JOINTS+1, 3): joint world positions + EE
    keypoints = np.zeros((n_frames, N_JOINTS + 1, 3))
    for t_idx in range(n_frames):
        for k in range(N_JOINTS):
            T_k = np.asarray(results.trajectory[f"T_j{k + 1}"])[t_idx]  # (4, 4)
            q0 = np.append(_joint_zero_pos[k], 1.0)
            keypoints[t_idx, k] = (T_k @ q0)[:3]
        # EE = after all joints + T_home translation
        T_7 = np.asarray(results.trajectory["T_j7"])[t_idx]
        keypoints[t_idx, N_JOINTS] = (T_7 @ T_home)[:3, 3]

    ee_pos = np.asarray(ee_pos_traj)

    # -------------------------------------------------------------------------
    # Build Viser scene
    # -------------------------------------------------------------------------
    server = create_server(ee_pos, show_grid=False)
    server.scene.add_grid("/grid", width=1.5, height=1.5, cell_size=0.2)
    server.scene.add_frame("/origin", axes_length=0.08, axes_radius=0.003)

    add_ellipsoid_obstacles(
        server,
        centers=[obstacle_center],
        radii=[np.full(3, 1.0 / obstacle_radius)],
    )

    # Waypoint spheres
    _marker_colors = {
        "home": (100, 150, 255),
        "pre_grasp": (255, 180, 50),
        "grasp": (255, 50, 50),
        "pre_place": (255, 180, 50),
        "place": (255, 50, 50),
    }
    add_target_markers(
        server,
        waypoint_positions,
        radius=0.012,
        colors=[_marker_colors[name] for name in waypoint_names],
        show_trails=False,
    )

    ee_colors = compute_velocity_colors(np.asarray(results.trajectory.get("velocity")))
    _, update_trail = add_animated_trail(server, ee_pos, ee_colors, point_size=0.008)
    _, update_marker = add_position_marker(server, ee_pos, radius=0.012)

    # -------------------------------------------------------------------------
    # Robot body: actual CAD meshes OR line-segment fallback
    # -------------------------------------------------------------------------
    update_robot = None

    if _use_cad_mesh:
        from scipy.spatial.transform import Rotation as _Rotation

        def _pose_from_T(T: np.ndarray) -> tuple[Vec3, QuatWxyz]:
            """Body pose for Viser: position + wxyz quaternion (mesh vertices stay link-local)."""
            R = np.asarray(T, dtype=np.float64)[:3, :3]
            t = T[:3, 3]
            q_xyzw = _Rotation.from_matrix(R).as_quat()
            wxyz = (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))
            pos = (float(t[0]), float(t[1]), float(t[2]))
            return pos, wxyz

        # Per-link colour: darker base, lighter links, slight blue tint on wrist
        _link_color = {
            "fr3v2_link0": (120, 120, 120),
            "fr3v2_link1": (215, 215, 218),
            "fr3v2_link2": (215, 215, 218),
            "fr3v2_link3": (215, 215, 218),
            "fr3v2_link4": (215, 215, 218),
            "fr3v2_link5": (210, 210, 215),
            "fr3v2_link6": (200, 205, 215),
            "fr3v2_link7": (190, 195, 210),
        }
        _link_handles = {}
        for link_name, (verts_local, faces) in _link_meshes_local.items():
            T0 = _link_world_T[link_name][0]
            pos0, wxyz0 = _pose_from_T(T0)
            handle = server.scene.add_mesh_simple(
                f"/robot/{link_name}",
                vertices=np.asarray(verts_local, dtype=np.float32, order="C"),
                faces=faces,
                color=_link_color.get(link_name, (210, 210, 215)),
                opacity=1.0,
                position=pos0,
                wxyz=wxyz0,
            )
            _link_handles[link_name] = handle

        def update_robot(frame_idx: int) -> None:
            for link_name, handle in _link_handles.items():
                T = _link_world_T[link_name][frame_idx]
                pos, wxyz = _pose_from_T(T)
                handle.position = pos
                handle.wxyz = wxyz

    else:
        # Fallback: 8 line segments (world origin → J1 → ··· → J7 → EE)
        n_segs = N_JOINTS + 1  # 8 segments
        seg_col = np.full((n_segs, 2, 3), [200, 200, 200], dtype=np.uint8)

        def _build_segments(frame_idx: int) -> np.ndarray:
            pts = np.zeros((n_segs, 2, 3), dtype=np.float32)
            pts[0] = [np.zeros(3), keypoints[frame_idx, 0]]
            for k in range(N_JOINTS - 1):
                pts[k + 1] = [keypoints[frame_idx, k], keypoints[frame_idx, k + 1]]
            pts[N_JOINTS] = [keypoints[frame_idx, N_JOINTS - 1], keypoints[frame_idx, N_JOINTS]]
            return pts

        arm_handle = server.scene.add_line_segments(
            "/arm_links",
            points=_build_segments(0),
            colors=seg_col,
            line_width=5.0,
        )

        def update_robot(frame_idx: int) -> None:
            arm_handle.points = _build_segments(frame_idx)

    # -------------------------------------------------------------------------
    # Animation controls (interactive) or embedded ``.viser`` export
    # -------------------------------------------------------------------------
    _anim_cb = [update_trail, update_marker, update_robot]

    _vis_cam_mod = _vis_embed_mod or _load_viser_embed_export_module()
    # Motion is mostly along world Y; camera on +X (opposite of −X) for the other
    # face-on horizontal view of the pick→place sweep.
    _vis_cam_mod.set_initial_camera_look_at_trajectory(
        server, ee_pos, front_world_axis=0, front_from_positive=True
    )

    if _export_viser_path is not None:
        assert _vis_embed_mod is not None

        def step_frame(frame_idx: int) -> None:
            for cb in _anim_cb:
                if cb is not None:
                    cb(int(frame_idx))

        saved = _vis_embed_mod.export_animated_viser_recording(
            server,
            step_frame=step_frame,
            n_frames=n_frames,
            output_path=_export_viser_path,
            fps=24.0,
            max_keyframes=80,
        )
        _vis_embed_mod.print_viser_embed_followup(saved)
        raise SystemExit(0)

    add_animation_controls(
        server,
        traj_time,
        _anim_cb,
        loop=True,
    )

    _cad_snapshot_builder = None
    if _use_cad_mesh:
        _cad_snapshot_builder = build_cad_link_snapshot_builder(
            _link_meshes_local,
            _link_world_T,
            link_colors=_link_color,
        )
    create_snapshot_plotting_server(
        results,
        position_key="ee_position",
        velocity_key="velocity",
        show_body_frame=False,
        show_viewcone=False,
        waypoint_positions=[np.asarray(p) for p in waypoint_positions],
        waypoint_colors=[_marker_colors[name] for name in waypoint_names],
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
        arm_keypoints=None if _use_cad_mesh else keypoints,
        snapshot_builder=_cad_snapshot_builder,
        initial_n_snapshots=5,
        target_radius=0.012,
    )

    server.sleep_forever()
