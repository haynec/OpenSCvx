"""Franka Panda joint-space point-to-point motion using frax dynamics.

This example demonstrates how to plug a `frax.Robot` directly into an OpenSCvx
problem via the `FraxDynamics` adapter. The 7-DOF Panda is driven from a home
configuration to a target joint configuration in minimum time, subject to the
robot's URDF joint position / velocity / torque limits (read automatically
from the model).

Requires:
    pip install openscvx[frax]

The Franka Panda model ships inside the ``frax`` package, so this example is
fully self-contained — no external URDF file is needed.
"""

import os
import sys

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

# ── frax dynamics as a first-class adapter ────────────────────────────────────
# `FraxDynamics` builds default q / qd / tau State and Control objects matching
# the robot's joint count and routes frax's forward dynamics into the BYOF
# channel internally — no separate `byof=` plumbing required. Joint limits and
# torque bounds are auto-populated from the robot's URDF.
robot = frax.load_panda()
dyn = ox.FraxDynamics(robot)
q, qd = dyn.states
(tau,) = dyn.controls

n_j = robot.num_joints  # 7
n = 4
total_time = 1.0

# Home configuration and a reachable target configuration.
q_start = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
q_goal = np.array([0.6, -0.3, 0.2, -1.8, 0.3, 1.2, 0.5])

q.initial = q_start
q.final = q_goal
qd.initial = np.zeros(n_j)
qd.final = np.zeros(n_j)

# Initial guesses: linear interpolation in joint space, zero velocity.
#
# The torque guess is *gravity-compensating*, not zero. frax integrates the
# full rigid-body dynamics (mass matrix + Coriolis + gravity), so a zero-torque
# guess implies the arm is in free-fall: forward_dynamics(q, 0, 0) returns joint
# accelerations of tens of rad/s² at this configuration. Over a coarse time grid
# (few nodes ⇒ large dt) the propagated guess diverges wildly from the qd≈0
# guess, producing huge dynamics defects that drive the convex subproblem
# infeasible. Seeding tau with g(q) makes forward_dynamics(q, 0, g(q)) ≈ 0, so
# the qd channel of the guess is self-consistent and SCvx converges even at very
# low node counts. (Linear ``I q̈ = τ`` models without gravity don't need this.)
q_guess = np.linspace(q_start, q_goal, n)
q.guess = q_guess
qd.guess = np.zeros((n, n_j))
# gravity_vector returns one entry per joint; tau covers only the actuated
# joints, so take the last num_actuated_joints entries (the FraxDynamics adapter
# slices torque bounds the same way for floating-base robots).
grav = np.array([np.asarray(robot.gravity_vector(qi)) for qi in q_guess])
tau.guess = grav[:, n_j - robot.num_actuated_joints :]

# Box constraints from the auto-populated bounds (continuous-time enforcement).
constraints = []
for state in dyn.states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
for control in dyn.controls:
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

time = ox.Time(
    initial=0.0,
    final=ox.Minimize(total_time),
    min=0.0,
    max=2.0 * total_time,
    uniform_time_grid=True,
)

problem = ox.Problem(
    dynamics=dyn,
    states=dyn.states,
    controls=dyn.controls,
    time=time,
    constraints=constraints,
    N=n,
    algorithm={
        "lam_vc": 1e1,
        "lam_cost": 4e-1,
        "autotuner": ox.AugmentedLagrangian(ep = 1E0),
    },
    float_dtype="float64",
)


def _compute_panda_keypoints(q_traj: np.ndarray, robot) -> tuple[np.ndarray, np.ndarray]:
    """Compute skeleton keypoints and EE positions for a joint trajectory.

    Returns:
        keypoints: ``(T, num_joints + 2, 3)`` — world base, link origins, EE tip.
        ee_pos: ``(T, 3)`` end-effector positions from ``robot.ee_transform``.
    """
    q_traj = np.asarray(q_traj, dtype=float)
    n_frames = q_traj.shape[0]
    n_links = robot.num_joints
    keypoints = np.zeros((n_frames, n_links + 2, 3))
    ee_pos = np.zeros((n_frames, 3))

    for t in range(n_frames):
        q = q_traj[t]
        links = np.asarray(robot.link_to_world_transforms(q))
        ee = np.asarray(robot.ee_transform(q))
        keypoints[t, 0] = (0.0, 0.0, 0.0)
        keypoints[t, 1 : 1 + n_links] = links[:, :3, 3]
        keypoints[t, -1] = ee[:3, 3]
        ee_pos[t] = ee[:3, 3]

    return keypoints, ee_pos


def visualize(results, robot, q_start: np.ndarray, q_goal: np.ndarray) -> None:
    """Animate the optimized Panda trajectory in a Viser 3D scene.

    Uses frax forward kinematics (``link_to_world_transforms`` / ``ee_transform``)
    to pose a stick-model arm, an EE trail, and sidebar joint/torque plots.
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

    keypoints, ee_pos = _compute_panda_keypoints(q_traj, robot)
    n_segs = robot.num_joints + 1

    ee_start = np.asarray(robot.ee_transform(q_start))[:3, 3]
    ee_goal = np.asarray(robot.ee_transform(q_goal))[:3, 3]

    ee_vel = np.gradient(ee_pos, t_vec, axis=0, edge_order=2)
    ee_colors = compute_velocity_colors(ee_vel)

    server = create_server(ee_pos, show_grid=False)
    server.scene.add_grid("/grid", width=1.5, height=1.5, cell_size=0.25)
    server.scene.add_frame("/origin", axes_length=0.08, axes_radius=0.003)

    add_target_markers(
        server,
        [ee_start, ee_goal],
        radius=0.012,
        colors=[(100, 150, 255), (255, 80, 80)],
    )

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

        # Visual OBJ files per body (group=2 geoms from panda_nohand.xml).
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
    print("Franka Panda joint-space point-to-point via frax dynamics")
    print("=" * 60)
    print(f"num_joints = {n_j}, N = {n}")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    q_final = np.asarray(results.nodes["q"][-1])
    err = np.linalg.norm(q_final - q_goal)
    print()
    print(f"Final joint config:  {np.array2string(q_final, precision=3)}")
    print(f"Target joint config: {np.array2string(q_goal, precision=3)}")
    print(f"||q_final - q_goal|| = {err:.4e}")
    print()
    print("Launching Viser visualization (Ctrl+C to exit)...")
    visualize(results, robot, q_start, q_goal)
