"""Franka Panda point-to-point motion with an intermediate EE waypoint (frax dynamics).

Extends ``examples/frax/panda_frax.py``: same ``FraxDynamics`` setup, joint
start/goal boundary conditions, CTCS box limits, and Viser stick-model animation,
but adds a **nodal** end-effector position constraint at the midpoint of the
horizon (node ``N // 2``). Task kinematics use ``frax`` forward kinematics
(``ee_transform``) via a BYOF nodal constraint so the waypoint matches the
bundled Panda URDF exactly.

Nodal (non-CTCS) constraints are handled by the default ``CVXPyPTRSolver``; use
``lam_vb=1e3`` so the virtual-buffer penalty can drive the waypoint constraint
to feasibility alongside full frax rigid-body dynamics.

Requires:
    pip install openscvx[frax]
"""

import os
import sys

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

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

robot = frax.load_panda()
dyn = ox.FraxDynamics(robot)
q, qd = dyn.states
(tau,) = dyn.controls

n_j = robot.num_joints
n = 12
total_time = 5.0
waypoint_node = n // 2
ee_tol = 0.02  # m

q_start = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
q_goal = np.array([0.6, -0.3, 0.2, -1.8, 0.3, 1.2, 0.5])

ee_start = np.asarray(robot.ee_transform(q_start))[:3, 3]
ee_goal = np.asarray(robot.ee_transform(q_goal))[:3, 3]
# Intermediate target: modest offset from the start pose (reachable, visibly off the start→goal chord).
ee_waypoint = ee_start + np.array([0.08, 0.05, 0.04])

q_min = np.asarray(robot.joint_lower_limits, dtype=float)
q_max = np.asarray(robot.joint_upper_limits, dtype=float)


def _ik_position(target: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
    """Position-only IK for initial-guess seeding (not part of the optimization model)."""

    def objective(qv: np.ndarray) -> float:
        p = np.asarray(robot.ee_transform(qv))[:3, 3]
        return float(np.sum((p - target) ** 2))

    result = minimize(
        objective,
        q_seed,
        method="L-BFGS-B",
        bounds=list(zip(q_min, q_max)),
    )
    return result.x


def _build_initial_guess(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    ee_waypoint: np.ndarray,
    *,
    n_nodes: int,
    waypoint_node: int,
    total_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint / velocity guesses respecting BCs and the EE waypoint.

    * ``q_start`` / ``q_goal`` are pinned at nodes 0 and N-1 (match boundaries).
    * ``q_mid`` comes from position-only IK to ``ee_waypoint``, seeded from
      ``q_start`` so the leg does not jump toward the chord midpoint in joint space.
    * ``ox.init.linspace`` connects the three joint keyframes; interior nodes are
      filled consistently on both trajectory segments.
    * ``qd`` uses a nominal-time finite difference with zero velocity at endpoints.
    """
    keyframe_nodes = [0, waypoint_node, n_nodes - 1]
    q_mid = _ik_position(ee_waypoint, q_start)
    q_guess = ox.init.linspace([q_start, q_mid, q_goal], keyframe_nodes)
    q_guess[0] = q_start
    q_guess[waypoint_node] = q_mid
    q_guess[-1] = q_goal

    qd_guess = np.zeros((n_nodes, n_j))
    if n_nodes > 2:
        dt_nom = total_time / (n_nodes - 1)
        qd_guess[1:-1] = np.gradient(q_guess, dt_nom, axis=0)[1:-1]
    return q_guess, qd_guess


q.initial = q_start
q.final = q_goal
qd.initial = np.zeros(n_j)
qd.final = np.zeros(n_j)

q.guess, qd.guess = _build_initial_guess(
    q_start,
    q_goal,
    ee_waypoint,
    n_nodes=n,
    waypoint_node=waypoint_node,
    total_time=total_time,
)
# Gravity-compensating torque guess (not zero). frax integrates the full
# rigid-body dynamics, so a zero-torque guess implies free-fall — joint
# accelerations of tens of rad/s² — which produces large dynamics defects that
# can drive the convex subproblem infeasible at low node counts. Seeding tau with
# g(q) makes forward_dynamics(q, 0, g(q)) ≈ 0, so the qd channel of the guess is
# self-consistent and SCvx stays robust even on a coarse time grid.
grav = np.array([np.asarray(robot.gravity_vector(qi)) for qi in q.guess])
tau.guess = grav[:, n_j - robot.num_actuated_joints :]

ee_waypoint_param = ox.Parameter("ee_waypoint", shape=(3,), value=ee_waypoint)


def _ee_waypoint_residual(x, u, node, params):
    del u, node
    q_val = x[q.slice]
    p = jnp.asarray(robot.ee_transform(q_val))[:3, 3]
    return jnp.linalg.norm(p - params["ee_waypoint"]) - ee_tol


byof = {
    "parameters": [ee_waypoint_param],
    "nodal_constraints": [
        {
            "constraint_fn": _ee_waypoint_residual,
            "nodes": [waypoint_node],
        }
    ],
}

constraints = []
for state in dyn.states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

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
    float_dtype="float64",
    algorithm={
        "lam_vb": 1e1,
        "lam_vc": 1e1,
        "lam_cost": 1e-1,
    },
)


def _compute_panda_keypoints(q_traj: np.ndarray, robot) -> tuple[np.ndarray, np.ndarray]:
    q_traj = np.asarray(q_traj, dtype=float)
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


def visualize(
    results,
    robot,
    ee_targets: list[np.ndarray],
    target_colors: list[tuple[int, int, int]],
) -> None:
    """Animate the Panda trajectory in Viser with joint/torque plots.

    Uses the multi-shoot V matrix from the final SCP iteration when available
    (``results.discretization_history[-1]``), showing the per-segment
    integration substeps exactly as the solver used them.  Falls back to the
    post-process single propagation (``results.trajectory``) if V is absent.
    """
    import plotly.graph_objects as go

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        add_ghost_trajectory,
        add_position_marker,
        add_target_markers,
        compute_velocity_colors,
        create_server,
    )
    from openscvx.plotting.viser.plotly_integration import add_animated_plotly_vline

    t_vec = np.asarray(results.trajectory["time"]).flatten()
    q_traj = np.asarray(results.trajectory["q"])
    tau_traj = np.asarray(results.trajectory["tau"])

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
            # Torque is ZOH: replicate each node's tau value across its substeps.
            _tau_nodes = np.asarray(results.nodes["tau"])
            _n_sub = _V.shape[1]
            _tau_rows: list[np.ndarray] = []
            for _seg in range(len(_t_nodes) - 1):
                _j0 = 0 if _seg == 0 else 1
                for _j in range(_j0, _n_sub):
                    _tau_rows.append(_tau_nodes[_seg])
            tau_traj = np.stack(_tau_rows)
        else:
            print("[viser] Multishot V extraction failed; using post-process trajectory.")

    keypoints, ee_pos = _compute_panda_keypoints(q_traj, robot)
    n_segs = robot.num_joints + 1

    ee_vel = np.gradient(ee_pos, t_vec, axis=0, edge_order=2)
    ee_colors = compute_velocity_colors(ee_vel)

    server = create_server(ee_pos, show_grid=False)
    server.scene.add_grid("/grid", width=1.5, height=1.5, cell_size=0.25)
    server.scene.add_frame("/origin", axes_length=0.08, axes_radius=0.003)

    add_target_markers(server, ee_targets, radius=0.012, colors=target_colors)

    add_ghost_trajectory(server, ee_pos, ee_colors, point_size=0.005)
    _, update_trail = add_animated_trail(server, ee_pos, ee_colors, point_size=0.008)
    _, update_marker = add_position_marker(server, ee_pos, radius=0.012)

    # -----------------------------------------------------------------------
    # Load Panda CAD meshes and pre-compute per-frame link transforms.
    # Falls back to line-segment stick model if mujoco / trimesh / menagerie
    # assets are unavailable.
    # -----------------------------------------------------------------------
    n_frames = len(q_traj)
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
        init_points = np.stack(
            [np.stack([keypoints[0, k], keypoints[0, k + 1]]) for k in range(n_segs)]
        ).astype(np.float32)
        arm_handle = server.scene.add_line_segments(
            "/panda_links",
            points=init_points,
            colors=link_colors,
            line_width=5.0,
        )

        def update_robot(frame_idx: int) -> None:
            pts = np.stack(
                [np.stack([keypoints[frame_idx, k], keypoints[frame_idx, k + 1]]) for k in range(n_segs)]
            ).astype(np.float32)
            arm_handle.points = pts

    fig_joints = go.Figure()
    for j in range(robot.num_joints):
        fig_joints.add_trace(
            go.Scatter(
                x=t_vec.tolist(),
                y=np.rad2deg(q_traj[:, j]).tolist(),
                mode="lines",
                name=f"q{j + 1}",
            )
        )
    fig_joints.update_layout(
        title="Joint angles (deg)",
        xaxis_title="Time (s)",
        yaxis_title="θ (deg)",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    fig_tau = go.Figure()
    for j in range(robot.num_actuated_joints):
        fig_tau.add_trace(
            go.Scatter(
                x=t_vec.tolist(),
                y=tau_traj[:, j].tolist(),
                mode="lines",
                name=f"τ{j + 1}",
            )
        )
    fig_tau.update_layout(
        title="Joint torques (Nm)",
        xaxis_title="Time (s)",
        yaxis_title="τ (Nm)",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    with server.gui.add_folder("Plots"):
        _, update_joints = add_animated_plotly_vline(server, fig_joints, t_vec, folder_name=None)
        _, update_tau = add_animated_plotly_vline(server, fig_tau, t_vec, folder_name=None)

    add_animation_controls(
        server,
        t_vec,
        [update_robot, update_trail, update_marker, update_joints, update_tau],
    )
    server.sleep_forever()


if __name__ == "__main__":
    print("Franka Panda via frax dynamics — intermediate EE waypoint")
    print("=" * 60)
    print(f"num_joints = {n_j}, N = {n}, waypoint node = {waypoint_node}")
    print(f"EE start:     {np.round(ee_start, 3)}")
    print(f"EE waypoint:  {np.round(ee_waypoint, 3)}  (tol = {ee_tol} m)")
    print(f"EE goal:      {np.round(ee_goal, 3)}")
    ee_guess_mid = np.asarray(robot.ee_transform(q.guess[waypoint_node]))[:3, 3]
    print(
        f"Init guess @ node {waypoint_node}: "
        f"||EE - waypoint|| = {np.linalg.norm(ee_guess_mid - ee_waypoint):.4e} m"
    )
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    q_traj = np.asarray(results.trajectory["q"])
    ee_traj = np.array([np.asarray(robot.ee_transform(qi))[:3, 3] for qi in q_traj])

    err_joints = np.linalg.norm(q_traj[-1] - q_goal)
    err_waypoint = np.linalg.norm(ee_traj[waypoint_node] - ee_waypoint)

    print()
    print("Results:")
    print(f"  ||q_final - q_goal|| = {err_joints:.4e}")
    print(f"  ||EE(node {waypoint_node}) - waypoint|| = {err_waypoint:.4e} m")
    print(f"  CTCS violation = {results.ctcs_violation}")
    print()
    print("Launching Viser visualization (Ctrl+C to exit)...")
    visualize(
        results,
        robot,
        [ee_start, ee_waypoint, ee_goal],
        [(100, 150, 255), (255, 180, 50), (255, 80, 80)],
    )
