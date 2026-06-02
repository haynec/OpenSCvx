"""Franka Panda pick-and-place via frax rigid-body dynamics.

Mirrors ``examples/arm/franka_fr3v2_pick_place.py`` but replaces the
simplified diagonal-inertia model (I q̈ = τ) and hand-coded PoE FK with:

  - ``FraxDynamics`` — full rigid-body forward dynamics from the frax URDF.
  - frax FK (``ee_transform`` / ``link_to_world_transforms``) for all task
    constraints via the BYOF channel.
  - frax collision spheres (``link_collision_positions``,
    ``self_collision_distances``) for obstacle and self-collision avoidance,
    also via BYOF CTCS.

Task
----
Six-waypoint pick-and-place::

    home → pre_grasp → grasp → pre_grasp → pre_place → place

Waypoints are derived from forward kinematics of known-reachable joint
configurations, so the gravity-compensated joint-space initial guess is
already consistent with the dynamics and the waypoint targets.

Constraints
-----------
- Box bounds: CTCS from ``FraxDynamics`` auto-populated limits.
- EE position at each of the six waypoint nodes (1 cm tolerance), nodal.
- EE stays above floor (z ≥ 0), CTCS.
- Self-collision avoidance via ``robot.self_collision_distances``, CTCS.
- Spherical obstacle clearance via ``robot.link_collision_positions``, CTCS.

Requires
--------
    pip install openscvx[frax]
"""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

try:
    import frax
except ImportError:
    print(
        "frax is not installed. Install with: pip install openscvx[frax]",
        file=sys.stderr,
    )
    sys.exit(1)

import openscvx as ox

# =============================================================================
# Robot and dynamics
# =============================================================================

robot = frax.load_panda()
dyn = ox.FraxDynamics(robot)
q, qd = dyn.states
(tau,) = dyn.controls
n_j = robot.num_joints  # 7

# =============================================================================
# Pick-and-place task geometry
# =============================================================================
# Waypoint joint configurations — chosen so that frax FK gives clearly
# reachable EE positions.  Using FK-derived targets means the joint-space
# linear-interpolation initial guess is already at the right EE locations.

q_home = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
q_pre_grasp = np.array([0.5, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
q_grasp = np.array([0.5, -0.3, 0.0, -2.5, 0.0, 2.2, 0.7854])
q_pre_place = np.array([-0.5, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
q_place = np.array([-0.5, -0.3, 0.0, -2.5, 0.0, 2.2, 0.7854])

# Reference configs ordered along the trajectory (pre_grasp appears twice)
_wp_configs = [q_home, q_pre_grasp, q_grasp, q_pre_grasp, q_pre_place, q_place]
_wp_names = ["home", "pre_grasp", "grasp", "pre_grasp", "pre_place", "place"]

# Compute EE targets from FK (exact, no IK error)
waypoint_ee_positions = [
    np.asarray(robot.ee_transform(q_i))[:3, 3] for q_i in _wp_configs
]

nodes_per_segment = 4
n_segments = len(_wp_configs) - 1  # 5
n = nodes_per_segment * n_segments + 1  # 21
waypoint_nodes = [i * nodes_per_segment for i in range(len(_wp_configs))]
total_time = 6.0
ee_tol = 0.01  # m — nodal EE position tolerance

# Spherical obstacle (sits in the sweep path between home and grasp)
obstacle_center = np.array([0.38, 0.08, 0.43])
obstacle_radius = 0.07

# =============================================================================
# States and controls (from FraxDynamics)
# =============================================================================

q.initial = q_home
q.final = [("free", 0.0)] * n_j
qd.initial = np.zeros(n_j)
qd.final = np.zeros(n_j)

# =============================================================================
# Initial guess
# =============================================================================
# Joint-space linear interpolation through the FK-derived waypoint configs.
# Torque seeded with gravity compensation (not zero) for dynamics consistency.

q_guess = ox.init.linspace(_wp_configs, waypoint_nodes)
q.guess = q_guess
qd.guess = np.zeros((n, n_j))
grav = np.array([np.asarray(robot.gravity_vector(qi)) for qi in q_guess])
tau.guess = grav[:, n_j - robot.num_actuated_joints :]

# =============================================================================
# Box constraints (symbolic CTCS from FraxDynamics bounds)
# =============================================================================

constraints = []
for state in dyn.states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
for control in dyn.controls:
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

# =============================================================================
# BYOF: frax-based task and safety constraints
# =============================================================================
# All BYOF CTCS use idx=0 (sharing the same augmented state as the symbolic
# box CTCS above).  Sign convention: g <= 0 means satisfied.

# --- Nodal EE position constraints (one per waypoint) ---------------------
# Use a factory to capture each waypoint array without adding extra parameters
# to the BYOF function signature (which must be exactly (x, u, node, params)).
def _make_ee_pos_residual(wp_jnp):
    def _fn(x, u, node, params):
        q_val = x[q.slice]
        T = robot.ee_transform(q_val)
        p = jnp.asarray(T)[:3, 3]
        return jnp.linalg.norm(p - wp_jnp) - ee_tol
    return _fn


_byof_nodal: list[dict] = [
    {"constraint_fn": _make_ee_pos_residual(jnp.array(wp_ee)), "nodes": [node]}
    for wp_ee, node in zip(waypoint_ee_positions, waypoint_nodes)
]


# --- CTCS: EE stays above floor -------------------------------------------
def _floor_ctcs(x, u, node, params):
    q_val = x[q.slice]
    T = robot.ee_transform(q_val)
    return -jnp.asarray(T)[2, 3]  # <= 0 when EE z >= 0


# --- CTCS: self-collision avoidance (frax sphere-pair distances) -----------
# robot.self_collision_distances returns gap distances for non-adjacent body
# sphere pairs; positive = no collision.  We want min(dists) >= 0.
def _self_collision_ctcs(x, u, node, params):
    q_val = x[q.slice]
    dists = robot.self_collision_distances(q_val)
    return -jnp.min(dists)  # <= 0 when all pairwise distances >= 0


# --- CTCS: obstacle clearance (frax link collision spheres) ---------------
_obs_center_jnp = jnp.array(obstacle_center)
_flat_radii_jnp = jnp.array(robot.flat_collision_radii)


def _obstacle_ctcs(x, u, node, params):
    q_val = x[q.slice]
    centers = robot.link_collision_positions(q_val)  # (n_spheres, 3)
    dists = jnp.linalg.norm(centers - _obs_center_jnp, axis=1)
    clearances = dists - _flat_radii_jnp - obstacle_radius
    return -jnp.min(clearances)  # <= 0 when all clearances >= 0


_byof_ctcs = [
    {"constraint_fn": _floor_ctcs},
    # {"constraint_fn": _self_collision_ctcs},
    {"constraint_fn": _obstacle_ctcs},
]

byof: dict = {
    "nodal_constraints": _byof_nodal,
    "ctcs_constraints": _byof_ctcs,
}

# =============================================================================
# Time and Problem
# =============================================================================

time = ox.Time(
    initial=0.0,
    final=ox.Minimize(total_time),
    min=0.0,
    max=2.0 * total_time,
)

problem = ox.Problem(
    dynamics=dyn,
    states=dyn.states,
    controls=dyn.controls,
    time=time,
    constraints=constraints,
    byof=byof,
    N=n,
    algorithm={
        "lam_vb": 1e1,
        "lam_vc": 1e2,
        "lam_cost": 1e-1,
    },
    float_dtype="float64",
)

# =============================================================================
# Visualization helpers
# =============================================================================


def _compute_panda_keypoints(
    q_traj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stick-model keypoints and EE positions using frax FK."""
    q_traj = np.asarray(q_traj)
    n_frames = q_traj.shape[0]
    n_links = robot.num_joints
    keypoints = np.zeros((n_frames, n_links + 2, 3))
    ee_pos = np.zeros((n_frames, 3))

    for t in range(n_frames):
        qi = q_traj[t]
        links = np.asarray(robot.link_to_world_transforms(qi))
        ee = np.asarray(robot.ee_transform(qi))
        keypoints[t, 0] = (0.0, 0.0, 0.0)
        keypoints[t, 1 : 1 + n_links] = links[:, :3, 3]
        keypoints[t, -1] = ee[:3, 3]
        ee_pos[t] = ee[:3, 3]

    return keypoints, ee_pos


def _q_from_V_multishot(
    V: np.ndarray,
    n_x: int,
    n_u: int,
    q_slice: slice,
    t_nodes: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Unpack joint angles from the SCP multi-shoot V matrix.

    V has shape ``((N-1) * i4, n_substeps)`` where
    ``i4 = n_x + n_x² + 2·n_x·n_u``.  The first ``n_x`` rows of each
    ``i4``-row block are the integrated state at that substep.

    Returns ``(q_traj, t_traj)`` or ``(None, None)`` if extraction fails.
    """
    i4 = n_x + n_x * n_x + 2 * n_x * n_u
    n_rows, n_sub = V.shape
    if i4 <= 0 or n_rows % i4 != 0 or n_sub < 1:
        return None, None
    n_seg = n_rows // i4
    if n_seg != len(t_nodes) - 1:
        return None, None
    q_rows: list[np.ndarray] = []
    t_rows: list[float] = []
    for seg in range(n_seg):
        t0, t1 = float(t_nodes[seg]), float(t_nodes[seg + 1])
        j0 = 0 if seg == 0 else 1  # skip duplicated segment-start sample
        for j in range(j0, n_sub):
            alpha = j / (n_sub - 1) if n_sub > 1 else 0.0
            x_vec = np.asarray(V[seg * i4 : seg * i4 + n_x, j], dtype=np.float64)
            q_rows.append(x_vec[q_slice])
            t_rows.append((1.0 - alpha) * t0 + alpha * t1)
    if not q_rows:
        return None, None
    return np.stack(q_rows), np.asarray(t_rows, dtype=np.float64)


def visualize(results, robot) -> None:
    """Animate the optimised trajectory in Viser (CAD mesh + markers).

    Uses the multi-shoot V matrix from the final SCP iteration when available
    (``results.discretization_history[-1]``), showing the per-segment
    integration substeps exactly as the solver used them.  Falls back to the
    post-process single propagation (``results.trajectory``) if V is absent.
    """
    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        add_ellipsoid_obstacles,
        add_position_marker,
        add_target_markers,
        compute_velocity_colors,
        create_server,
    )

    t_vec = np.asarray(results.trajectory["time"]).flatten()
    q_traj = np.asarray(results.trajectory["q"])

    # ── Prefer multishot V over post-process single propagation ──────────────
    _dh = getattr(results, "discretization_history", None) or []
    if _dh:
        _V = np.asarray(_dh[-1], dtype=np.float64)
        _n_x = results.x.shape[1]
        _n_u = results.u.shape[1]
        _t_nodes_raw = results.nodes.get("time", None)
        if _t_nodes_raw is None:
            _t_nodes = np.linspace(0.0, float(t_vec[-1]), len(results.nodes["q"]))
        else:
            _t_nodes = np.asarray(_t_nodes_raw).flatten()
        _q_ms, _t_ms = _q_from_V_multishot(_V, _n_x, _n_u, q.slice, _t_nodes)
        if _q_ms is not None:
            print(
                f"[viser] Multishot V: {len(_q_ms)} frames across "
                f"{len(_t_nodes) - 1} segments."
            )
            q_traj, t_vec = _q_ms, _t_ms
        else:
            print("[viser] Multishot V extraction failed; using post-process trajectory.")

    n_frames = len(q_traj)
    n_segs = robot.num_joints + 1

    keypoints, ee_pos = _compute_panda_keypoints(q_traj)
    ee_colors = compute_velocity_colors(np.gradient(ee_pos, t_vec, axis=0, edge_order=2))

    server = create_server(ee_pos, show_grid=False)
    server.scene.add_grid("/grid", width=1.5, height=1.5, cell_size=0.2)
    server.scene.add_frame("/origin", axes_length=0.08, axes_radius=0.003)

    # Obstacle sphere
    add_ellipsoid_obstacles(
        server,
        centers=[obstacle_center],
        radii=[np.full(3, 1.0 / obstacle_radius)],
    )

    # Waypoint markers
    _wp_colors = {
        "home": (100, 150, 255),
        "pre_grasp": (255, 180, 50),
        "grasp": (255, 50, 50),
        "pre_place": (255, 180, 50),
        "place": (255, 50, 50),
    }
    add_target_markers(
        server,
        waypoint_ee_positions,
        radius=0.012,
        colors=[_wp_colors.get(n, (200, 200, 200)) for n in _wp_names],
        show_trails=False,
    )

    _, update_trail = add_animated_trail(server, ee_pos, ee_colors, point_size=0.008)
    _, update_marker = add_position_marker(server, ee_pos, radius=0.012)

    # -----------------------------------------------------------------------
    # Load Panda CAD meshes and pre-compute per-frame link transforms.
    # Falls back to line-segment stick model if mujoco / trimesh / menagerie
    # assets are unavailable.
    # -----------------------------------------------------------------------
    _use_cad_mesh = False
    _link_meshes_local: dict = {}
    _link_body_ids: dict = {}
    _link_world_T: dict = {}
    try:
        import mujoco
        import trimesh  # type: ignore

        from openscvx.integrations.menagerie import get_model_dir

        _panda_dir = get_model_dir("franka_emika_panda")
        _mj_model_vis = mujoco.MjModel.from_xml_path(str(_panda_dir / "panda_nohand.xml"))
        _mj_model_vis.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        _mj_data_vis = mujoco.MjData(_mj_model_vis)
        _asset_dir = _panda_dir / "assets"

        _link_visual_files = {
            "link0": [
                "link0_0.obj", "link0_1.obj", "link0_2.obj", "link0_3.obj",
                "link0_4.obj", "link0_5.obj", "link0_7.obj", "link0_8.obj",
                "link0_9.obj", "link0_10.obj", "link0_11.obj",
            ],
            "link1": ["link1.obj"],
            "link2": ["link2.obj"],
            "link3": ["link3_0.obj", "link3_1.obj", "link3_2.obj", "link3_3.obj"],
            "link4": ["link4_0.obj", "link4_1.obj", "link4_2.obj", "link4_3.obj"],
            "link5": ["link5_0.obj", "link5_1.obj", "link5_2.obj"],
            "link6": [
                "link6_0.obj", "link6_1.obj", "link6_2.obj", "link6_3.obj",
                "link6_4.obj", "link6_5.obj", "link6_6.obj", "link6_7.obj",
                "link6_8.obj", "link6_9.obj", "link6_10.obj", "link6_11.obj",
                "link6_12.obj", "link6_13.obj", "link6_14.obj", "link6_15.obj",
                "link6_16.obj",
            ],
            "link7": [
                "link7_0.obj", "link7_1.obj", "link7_2.obj", "link7_3.obj",
                "link7_4.obj", "link7_5.obj", "link7_6.obj", "link7_7.obj",
            ],
        }

        for link_name, files in _link_visual_files.items():
            all_verts, all_faces, offset = [], [], 0
            for fname in files:
                obj_path = _asset_dir / fname
                if not obj_path.exists():
                    continue
                tm = trimesh.load(str(obj_path), force="mesh", process=False)
                all_verts.append(np.asarray(tm.vertices, dtype=np.float32))
                all_faces.append(np.asarray(tm.faces, dtype=np.uint32) + offset)
                offset += len(tm.vertices)
            if not all_verts:
                continue
            _link_meshes_local[link_name] = (np.vstack(all_verts), np.vstack(all_faces))
            _link_body_ids[link_name] = mujoco.mj_name2id(
                _mj_model_vis, mujoco.mjtObj.mjOBJ_BODY, link_name
            )

        for name in _link_meshes_local:
            _link_world_T[name] = np.zeros((n_frames, 4, 4))
        for t_idx in range(n_frames):
            _mj_data_vis.qpos[:7] = q_traj[t_idx]
            mujoco.mj_kinematics(_mj_model_vis, _mj_data_vis)
            for name, body_id in _link_body_ids.items():
                T = np.eye(4)
                T[:3, :3] = _mj_data_vis.xmat[body_id].copy().reshape(3, 3)
                T[:3, 3] = _mj_data_vis.xpos[body_id].copy()
                _link_world_T[name][t_idx] = T

        _use_cad_mesh = len(_link_meshes_local) > 0
        if _use_cad_mesh:
            print(f"[viser] Loaded {len(_link_meshes_local)} Panda CAD link meshes from menagerie.")
    except Exception as exc:
        print(
            f"[viser] CAD mesh unavailable "
            f"({type(exc).__name__}: {exc}); falling back to line segments."
        )

    # -----------------------------------------------------------------------
    # Robot body: CAD meshes when available, line segments otherwise.
    # -----------------------------------------------------------------------
    server.scene.add_box(
        "/panda_base",
        dimensions=(0.12, 0.12, 0.08),
        position=(0.0, 0.0, 0.04),
        color=(60, 60, 60),
    )
    update_robot = None
    if _use_cad_mesh:
        from scipy.spatial.transform import Rotation as _Rotation

        def _pose_from_T(T: np.ndarray):
            R = np.asarray(T, dtype=np.float64)[:3, :3]
            t = T[:3, 3]
            q_xyzw = _Rotation.from_matrix(R).as_quat()
            wxyz = (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))
            return (float(t[0]), float(t[1]), float(t[2])), wxyz

        _panda_link_color = {
            "link0": (120, 120, 120),
            "link1": (215, 215, 218),
            "link2": (215, 215, 218),
            "link3": (215, 215, 218),
            "link4": (215, 215, 218),
            "link5": (210, 210, 215),
            "link6": (200, 205, 215),
            "link7": (190, 195, 210),
        }
        _link_handles = {}
        for link_name, (verts_local, faces) in _link_meshes_local.items():
            T0 = _link_world_T[link_name][0]
            pos0, wxyz0 = _pose_from_T(T0)
            handle = server.scene.add_mesh_simple(
                f"/robot/{link_name}",
                vertices=np.asarray(verts_local, dtype=np.float32, order="C"),
                faces=faces,
                color=_panda_link_color.get(link_name, (210, 210, 215)),
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
        link_rgb = np.linspace([80, 100, 180], [255, 120, 80], n_segs).astype(np.uint8)
        link_colors = np.stack([link_rgb, link_rgb], axis=1)

        def _arm_segments(frame_idx: int) -> np.ndarray:
            pts = np.zeros((n_segs, 2, 3), dtype=np.float32)
            pts[0] = [np.zeros(3), keypoints[frame_idx, 0]]
            for k in range(robot.num_joints - 1):
                pts[k + 1] = [keypoints[frame_idx, k], keypoints[frame_idx, k + 1]]
            pts[-1] = [keypoints[frame_idx, -2], keypoints[frame_idx, -1]]
            return pts

        arm_handle = server.scene.add_line_segments(
            "/panda_links",
            points=_arm_segments(0),
            colors=link_colors,
            line_width=5.0,
        )

        def update_robot(frame_idx: int) -> None:
            arm_handle.points = _arm_segments(frame_idx)

    add_animation_controls(
        server,
        t_vec,
        [update_robot, update_trail, update_marker],
        loop=True,
    )
    server.sleep_forever()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Franka Panda Pick-and-Place via frax dynamics")
    print("=" * 60)
    print(f"Nodes: {n}  ({nodes_per_segment} per segment, {n_segments} segments)")
    print(
        f"Obstacle: center={list(np.round(obstacle_center, 3))} m, "
        f"r={obstacle_radius} m"
    )
    for name, pos, node in zip(_wp_names, waypoint_ee_positions, waypoint_nodes):
        print(f"  {name:>12s} (node {node:2d}): EE={list(np.round(pos, 3))}")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    q_traj = np.asarray(results.trajectory["q"])
    ee_traj = np.array([np.asarray(robot.ee_transform(qi))[:3, 3] for qi in q_traj])

    print("\nResults:")
    for name, wp_ee, node in zip(_wp_names, waypoint_ee_positions, waypoint_nodes):
        frame_idx = int(round(node / (n - 1) * (len(ee_traj) - 1)))
        err = np.linalg.norm(ee_traj[frame_idx] - wp_ee)
        print(
            f"  {name:>12s}: "
            f"EE=[{ee_traj[frame_idx, 0]:.3f}, "
            f"{ee_traj[frame_idx, 1]:.3f}, "
            f"{ee_traj[frame_idx, 2]:.3f}]  err={err:.4f} m"
        )
    print(f"\nCTCS violation: {results.ctcs_violation}")
    print("\nLaunching Viser visualization (Ctrl+C to exit)...")
    visualize(results, robot)
