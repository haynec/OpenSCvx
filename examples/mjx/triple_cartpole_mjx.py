"""Triple-link cartpole swing-up using MuJoCo MJX dynamics.

Extends the single-link cartpole to three serial links (a triple pendulum on a
cart). The optimizer drives all three links from the hanging equilibrium
(θ₁=π, θ₂=0, θ₃=0) to the unstable upright equilibrium (θ₁=θ₂=θ₃=0) using
a single horizontal force applied to the cart — a classic underactuated
control benchmark.

State  : qpos = [cart_x, θ₁, θ₂, θ₃],  qvel = [ẋ, θ̇₁, θ̇₂, θ̇₃]
Control: ctrl = [F_cart]  (normalised; gear=60 → max ±60 N)

Link lengths: L₁=0.5 m, L₂=0.4 m, L₃=0.3 m.

Requires:
    pip install openscvx[mjx]
"""

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

try:
    import mujoco
    import mujoco.mjx as mjx
except ImportError:
    print(
        "MuJoCo MJX is not installed. Install with: pip install openscvx[mjx]",
        file=sys.stderr,
    )
    sys.exit(1)

import openscvx as ox
from examples.plotting_viser import (
    create_snapshot_plotting_server,
)
from openscvx import Problem

L1, L2, L3 = 0.5, 0.4, 0.3  # link lengths (m)

TRIPLE_CARTPOLE_XML = f"""
<mujoco model="triple_cartpole">
  <option gravity="0 0 -9.81" timestep="0.005" integrator="Euler"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-10 10"/>
      <geom name="cart_geom" type="box" size="0.25 0.15 0.1"
            mass="2.0" rgba="0.35 0.35 0.75 1"/>
      <!-- Link 1 — pivot at cart centre (z=0) -->
      <body name="link1" pos="0 0 0">
        <joint name="hinge1" type="hinge" axis="0 1 0" limited="false"/>
        <geom name="pole1" type="capsule" fromto="0 0 0 0 0 {L1}"
              size="0.06" mass="2.0" rgba="0.85 0.3 0.3 1"/>
        <!-- Link 2 — pivot at tip of link 1 -->
        <body name="link2" pos="0 0 {L1}">
          <joint name="hinge2" type="hinge" axis="0 1 0" limited="false"/>
          <geom name="pole2" type="capsule" fromto="0 0 0 0 0 {L2}"
                size="0.035" mass="1.25" rgba="0.3 0.8 0.3 1"/>
          <!-- Link 3 — pivot at tip of link 2 -->
          <body name="link3" pos="0 0 {L2}">
            <joint name="hinge3" type="hinge" axis="0 1 0" limited="false"/>
            <geom name="pole3" type="capsule" fromto="0 0 0 0 0 {L3}"
                  size="0.03" mass="0.75" rgba="0.3 0.3 0.85 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- Single actuator on the cart slider; gear scales ctrl ∈ [−1,1] to force in N -->
    <motor joint="slider" name="cart_force" gear="60"
           ctrlrange="-3 3" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

mj_model = mujoco.MjModel.from_xml_string(TRIPLE_CARTPOLE_XML)
# Contacts not needed; disabling them keeps MJX forward-dynamics JAX-differentiable.
mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
mjx_model = mjx.put_model(mj_model)

n_q = int(mjx_model.nq)  # 4: cart_x, θ1, θ2, θ3
n_v = int(mjx_model.nv)  # 4: ẋ, θ̇1, θ̇2, θ̇3  (nq == nv, no quaternion)
n_u = int(mjx_model.nu)  # 1: cart force

n = 60  # more nodes → finer resolution near the unstable upright equilibrium
total_time = 2.5

# ── MJX dynamics adapter ──────────────────────────────────────────────────────
dyn = ox.MjxDynamics(mjx_model)
qpos, qvel = dyn.states
(ctrl,) = dyn.controls

qpos.min = np.array([-100.0, -2 * np.pi, -2 * np.pi, -2 * np.pi])
qpos.max = np.array([100.0, 2 * np.pi, 2 * np.pi, 2 * np.pi])
qpos.initial = np.array([0.0, np.pi, 0.0, 0.0])  # all links hanging down
qpos.final = [ox.Free(0.0), 0.0, 0.0, 0.0]  # all links upright

qvel.min = np.array([-12.0, -12.0, -12.0, -12.0])
qvel.max = np.array([12.0, 12.0, 12.0, 12.0])
qvel.initial = np.zeros(n_v)
qvel.final = [0.0, 0.0, 0.0, 0.0]

ctrl.min = np.array([-3.0])
ctrl.max = np.array([3.0])
ctrl.guess = np.zeros((n, n_u))

# ── Constraints (CTCS on state / control bounds) ───────────────────────────────
constraints = []
for state in dyn.states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# ── Initial guess: linearly swing θ₁ from π → 0, others stay 0 ───────────────
th1_guess = np.linspace(np.pi, 0.0, n)
qpos.guess = np.column_stack([np.zeros(n), th1_guess, np.zeros(n), np.zeros(n)])
qvel.guess = np.zeros((n, n_v))

time = ox.Time(
    initial=0.0,
    final=total_time,
    min=0.0,
    max=total_time,
)

problem = Problem(
    dynamics=dyn,
    states=dyn.states,
    controls=dyn.controls,
    time=time,
    constraints=constraints,
    N=n,
    algorithm={
        "lam_prox": 1e0,
        "lam_cost": 0e0,
        "lam_vc": 4e1,
        "autotuner": ox.ConstantProximalWeight(),
    },
    float_dtype="float64",
)


# ── Forward kinematics helpers ─────────────────────────────────────────────────
def fk_joints(q: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return world-frame positions (in XZ plane) of cart, h1, h2, h3, tip."""
    cx = float(q[0])
    t1, t2, t3 = float(q[1]), float(q[2]), float(q[3])
    cart = np.array([cx, 0.0, 0.0])
    h1 = cart  # hinge1 is at cart centre
    h2 = h1 + np.array([L1 * np.sin(t1), 0.0, L1 * np.cos(t1)])
    h3 = h2 + np.array([L2 * np.sin(t1 + t2), 0.0, L2 * np.cos(t1 + t2)])
    tip = h3 + np.array([L3 * np.sin(t1 + t2 + t3), 0.0, L3 * np.cos(t1 + t2 + t3)])
    return cart, h1, h2, h3, tip


def _foh_controls_at_times(
    t_samples: np.ndarray,
    u_nodes: np.ndarray,
    t_nodes: np.ndarray,
) -> np.ndarray:
    """First-order hold on SCP node controls at multishot sample times."""
    t_nodes = np.asarray(t_nodes, dtype=np.float64).ravel()
    u_nodes = np.asarray(u_nodes, dtype=np.float64)
    t_samples = np.asarray(t_samples, dtype=np.float64).ravel()
    u_out = np.empty((len(t_samples), u_nodes.shape[1]), dtype=np.float64)
    for i, t in enumerate(t_samples):
        k = int(np.clip(np.searchsorted(t_nodes, t, side="right") - 1, 0, len(t_nodes) - 2))
        t0, t1 = float(t_nodes[k]), float(t_nodes[k + 1])
        alpha = float(np.clip((t - t0) / (t1 - t0) if t1 > t0 else 0.0, 0.0, 1.0))
        u_out[i] = (1.0 - alpha) * u_nodes[k] + alpha * u_nodes[k + 1]
    return u_out


def segment_tip_paths_from_V(
    V_multi_shoot: np.ndarray,
    *,
    n_x: int,
    n_u: int,
    n_q: int,
) -> list[np.ndarray]:
    """Per-segment tip paths from every column of ``V`` (for propagation line overlays)."""
    segment_size = n_x + n_x * n_x + 2 * n_x * n_u
    n_seg = V_multi_shoot.shape[0] // segment_size
    n_sub = V_multi_shoot.shape[1]
    paths: list[np.ndarray] = []
    for seg_idx in range(n_seg):
        seg_start = seg_idx * segment_size
        tips: list[np.ndarray] = []
        for t_idx in range(n_sub):
            state = np.asarray(
                V_multi_shoot[seg_start : seg_start + n_x, t_idx], dtype=np.float64
            ).ravel()
            tips.append(fk_joints(state[:n_q])[4])
        paths.append(np.asarray(tips, dtype=np.float32))
    return paths


def visualize(results) -> None:
    """Animate the triple-link cartpole using the SCP multi-shoot matrix ``V``.

    When ``results.discretization_history`` is present, every plot and the 3D rig
    use propagated states read directly from ``V`` (same unpacking as the realtime
    examples). Nodal ``np.interp`` is only used if ``V`` cannot be decoded.
    """
    import plotly.graph_objects as go
    import viser

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        compute_velocity_colors,
    )
    from openscvx.plotting.viser.plotly_integration import add_animated_plotly_vline

    n_x = int(results.X[0].shape[1])
    n_u = int(results.U[0].shape[1])

    t_post = results.trajectory["time"].flatten()
    q_nodes = results.nodes["qpos"]
    u_nodes = results.nodes["ctrl"]
    t_nodes = results.nodes.get("time", None)
    if t_nodes is None:
        t_nodes = np.linspace(float(t_post[0]), float(t_post[-1]), len(q_nodes))
    else:
        t_nodes = np.asarray(t_nodes).flatten()

    prop = results.multishot_propagation(t_nodes=t_nodes)
    using_multishot = prop is not None
    if using_multishot:
        q_angle, t_play = prop.state("qpos")
        t_nodes_ms = prop.t_nodes
        u_play = _foh_controls_at_times(t_play, u_nodes, t_nodes_ms)
        fk_multishot_anim = [fk_joints(q_angle[i]) for i in range(len(q_angle))]
        tip_pos = np.array([fk[4] for fk in fk_multishot_anim], dtype=np.float64)
        if len(t_play) > 1:
            tip_vel = np.gradient(tip_pos, t_play, axis=0)
        else:
            tip_vel = np.zeros_like(tip_pos)
        V_multishot = prop.V
        print(
            f"[viser] Multi-shoot V: {len(t_play)} propagated samples "
            f"({prop.n_substeps} cols × {prop.n_segments} segments)."
        )
    else:
        V_multishot = None
        print(
            "[viser] WARNING: could not decode discretization_history V; "
            "falling back to nodal linear interpolation."
        )
        t_play = t_post
        u_play = results.trajectory["ctrl"]
        q_angle = np.column_stack(
            [np.interp(t_play, t_nodes, q_nodes[:, j]) for j in range(q_nodes.shape[1])]
        )
        fk_multishot_anim = [fk_joints(q_angle[i]) for i in range(len(q_angle))]
        tip_pos = np.array([fk[4] for fk in fk_multishot_anim], dtype=np.float64)
        if len(t_play) > 1:
            tip_vel = np.gradient(tip_pos, t_play, axis=0)
        else:
            tip_vel = np.zeros_like(tip_pos)

    cart_pos = np.array(
        [[float(fk_multishot_anim[i][0][0]), 0.0, 0.0] for i in range(len(fk_multishot_anim))],
        dtype=np.float64,
    )
    if len(t_play) > 1:
        cart_vel = np.gradient(cart_pos, t_play, axis=0)
    else:
        cart_vel = np.zeros_like(cart_pos)

    results.trajectory["tip_position"] = tip_pos
    results.trajectory["tip_velocity"] = tip_vel
    results.trajectory["cart_position"] = cart_pos
    results.trajectory["cart_velocity"] = cart_vel

    n_nodes = len(q_nodes)
    upright_tip = np.array([0.0, 0.0, L1 + L2 + L3])

    # ── Viser server ───────────────────────────────────────────────────────────
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")

    rail = np.array([[[-4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]], dtype=np.float32)
    server.scene.add_line_segments(
        "/rail",
        points=rail,
        colors=np.array([80, 80, 80], dtype=np.uint8),
        line_width=4.0,
    )

    server.scene.add_icosphere(
        "/target",
        radius=0.05,
        color=(50, 220, 100),
        position=tuple(float(v) for v in upright_tip),
    )

    # ── Multishot propagation paths (dense samples from ``V``, not nodal chords) ─
    cart_multishot_segs = None
    tip_multishot_segs = None
    if using_multishot and V_multishot is not None:
        all_qpos, _ = prop.state("qpos")
        if len(all_qpos) > 0:
            tip_cloud = np.array(
                [fk_joints(all_qpos[i])[4] for i in range(len(all_qpos))],
                dtype=np.float32,
            )
            server.scene.add_point_cloud(
                "/multishot/prop_samples",
                points=tip_cloud,
                colors=np.tile(np.array([140, 140, 140], dtype=np.uint8), (len(tip_cloud), 1)),
                point_size=0.006,
            )

        seg_tip_paths = segment_tip_paths_from_V(V_multishot, n_x=n_x, n_u=n_u, n_q=n_q)
        tip_seg_list = [
            np.stack([seg[i], seg[i + 1]], axis=0)
            for seg in seg_tip_paths
            if len(seg) >= 2
            for i in range(len(seg) - 1)
        ]
        tip_multishot_segs = np.stack(tip_seg_list, axis=0) if tip_seg_list else None
        cart_multishot_segs = np.stack(
            [
                np.stack([cart_pos[i], cart_pos[i + 1]], axis=0, dtype=np.float32)
                for i in range(len(cart_pos) - 1)
            ],
            axis=0,
        )
    elif n_nodes >= 2:
        fk_nd = [fk_joints(q_nodes[i]) for i in range(n_nodes)]
        cart_node_pos = np.array(
            [[float(fk_nd[i][0][0]), 0.0, 0.0] for i in range(n_nodes)],
            dtype=np.float32,
        )
        tip_node_pos = np.array([fk_nd[i][4] for i in range(n_nodes)], dtype=np.float32)
        cart_multishot_segs = np.stack(
            [
                np.stack([cart_node_pos[i], cart_node_pos[i + 1]], axis=0)
                for i in range(n_nodes - 1)
            ],
            axis=0,
        )
        tip_multishot_segs = np.stack(
            [np.stack([tip_node_pos[i], tip_node_pos[i + 1]], axis=0) for i in range(n_nodes - 1)],
            axis=0,
        )

    if cart_multishot_segs is not None:
        server.scene.add_line_segments(
            "/multishot/cart_path",
            points=cart_multishot_segs,
            colors=np.array([90, 90, 230], dtype=np.uint8),
            line_width=3.5,
        )
    if tip_multishot_segs is not None:
        server.scene.add_line_segments(
            "/multishot/tip_path",
            points=tip_multishot_segs,
            colors=np.array([230, 170, 40], dtype=np.uint8),
            line_width=3.5,
        )

    # ── Animated multishot rig ─────────────────────────────────────────────────
    ms_cart_handle = server.scene.add_box(
        "/multishot/cart",
        dimensions=(0.42, 0.24, 0.16),
        position=tuple(float(v) for v in fk_multishot_anim[0][0]),
        color=(55, 150, 255),
    )

    def _multishot_link_segments(i: int) -> np.ndarray:
        _, h1, h2, h3, tip = fk_multishot_anim[i]
        return np.array([[h1, h2], [h2, h3], [h3, tip]], dtype=np.float32)

    ms_link_colors = np.array(
        [
            [[120, 190, 255], [120, 190, 255]],
            [[95, 175, 245], [95, 175, 245]],
            [[70, 160, 235], [70, 160, 235]],
        ],
        dtype=np.uint8,
    )
    ms_link_handle = server.scene.add_line_segments(
        "/multishot/links",
        points=_multishot_link_segments(0),
        colors=ms_link_colors,
        line_width=5.0,
    )

    ms_joint_handles = []
    for jname in ["/multishot/j1", "/multishot/j2", "/multishot/j3"]:
        ms_joint_handles.append(
            server.scene.add_icosphere(
                jname,
                radius=0.03,
                color=(80, 170, 245),
                position=tuple(float(v) for v in fk_multishot_anim[0][1]),
            )
        )

    tip_colors = compute_velocity_colors(tip_pos)
    _, update_trail = add_animated_trail(server, tip_pos, tip_colors, point_size=0.02)

    fig_angles = go.Figure()
    angle_names = ["θ₁ (link 1)", "θ₂ (link 2)", "θ₃ (link 3)"]
    angle_colors = ["royalblue", "darkorange", "green"]
    for k in range(3):
        fig_angles.add_trace(
            go.Scatter(
                x=t_play.tolist(),
                y=np.rad2deg(q_angle[:, k + 1]).tolist(),
                mode="lines",
                name=angle_names[k],
                line={"color": angle_colors[k], "width": 2},
            )
        )
    fig_angles.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Upright")
    fig_angles.update_layout(
        title="Joint angles",
        xaxis_title="Time (s)",
        yaxis_title="Angle (deg)",
        legend={"orientation": "h"},
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    fig_cart = go.Figure()
    fig_cart.add_trace(
        go.Scatter(
            x=t_play.tolist(),
            y=q_angle[:, 0].tolist(),
            mode="lines",
            name="Cart x (multi-shoot)",
            line={"color": "royalblue", "width": 2},
        )
    )
    fig_cart.update_layout(
        title="Cart position",
        xaxis_title="Time (s)",
        yaxis_title="x (m)",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    fig_ctrl = go.Figure()
    ctrl_label = "multi-shoot FOH" if using_multishot else "post-process"
    fig_ctrl.add_trace(
        go.Scatter(
            x=t_play.tolist(),
            y=u_play[:, 0].tolist(),
            mode="lines",
            name=f"Cart force ({ctrl_label})",
            line={"color": "crimson", "width": 2},
        )
    )
    fig_ctrl.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_ctrl.update_layout(
        title="Cart control (normalised)",
        xaxis_title="Time (s)",
        yaxis_title="u",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    with server.gui.add_folder("Plots"):
        _, update_angles = add_animated_plotly_vline(server, fig_angles, t_play, folder_name=None)
        _, update_ctrl = add_animated_plotly_vline(server, fig_ctrl, t_play, folder_name=None)
        server.gui.add_plotly(fig_cart)

    def update_multishot_scene(frame_idx: int) -> None:
        ms_i = int(np.clip(frame_idx, 0, len(fk_multishot_anim) - 1))
        _, h1, h2, h3, _ = fk_multishot_anim[ms_i]
        cart = fk_multishot_anim[ms_i][0]
        ms_cart_handle.position = (float(cart[0]), 0.0, 0.0)
        ms_link_handle.points = _multishot_link_segments(ms_i)
        for handle, pos in zip(ms_joint_handles, (h1, h2, h3)):
            handle.position = tuple(float(v) for v in pos)

    add_animation_controls(
        server,
        t_play,
        [
            update_multishot_scene,
            update_trail,
            update_angles,
            update_ctrl,
        ],
    )

    _snapshot_link_colors = np.array(
        [
            [[120, 190, 255], [120, 190, 255]],
            [[95, 175, 245], [95, 175, 245]],
            [[70, 160, 235], [70, 160, 235]],
        ],
        dtype=np.uint8,
    )

    def _cartpole_snapshot_builder(
        snap_server: viser.ViserServer, snapshot_i: int, frame_idx: int
    ) -> list:
        ms_i = int(np.clip(frame_idx, 0, len(fk_multishot_anim) - 1))
        cart, h1, h2, h3, tip = fk_multishot_anim[ms_i]
        handles: list = []
        handles.append(
            snap_server.scene.add_box(
                f"/snapshots/cart_{snapshot_i}",
                dimensions=(0.42, 0.24, 0.16),
                position=(float(cart[0]), 0.0, 0.0),
                color=(55, 150, 255),
            )
        )
        handles.append(
            snap_server.scene.add_line_segments(
                f"/snapshots/links_{snapshot_i}",
                points=np.array([[h1, h2], [h2, h3], [h3, tip]], dtype=np.float32),
                colors=_snapshot_link_colors,
                line_width=5.0,
            )
        )
        for j, pos in enumerate((h1, h2, h3)):
            handles.append(
                snap_server.scene.add_icosphere(
                    f"/snapshots/j{j}_{snapshot_i}",
                    radius=0.03,
                    color=(80, 170, 245),
                    position=tuple(float(v) for v in pos),
                )
            )
        return handles

    snapshot_server = create_snapshot_plotting_server(
        results,
        position_key="tip_position",
        velocity_key="tip_velocity",
        show_body_frame=False,
        show_viewcone=False,
        target_positions=[upright_tip],
        target_radius=0.05,
        snapshot_builder=_cartpole_snapshot_builder,
        initial_n_snapshots=5,
        ghost_point_size=0.02,
        show_grid=True,
    )
    snapshot_server.scene.set_up_direction("+z")

    snapshot_server.scene.add_line_segments(
        "/snapshots/rail",
        points=rail,
        colors=np.array([80, 80, 80], dtype=np.uint8),
        line_width=4.0,
    )
    if len(cart_pos) >= 2:
        cart_path_segs = np.stack(
            [
                np.stack([cart_pos[i], cart_pos[i + 1]], axis=0, dtype=np.float32)
                for i in range(len(cart_pos) - 1)
            ],
            axis=0,
        )
        snapshot_server.scene.add_line_segments(
            "/snapshots/cart_path",
            points=cart_path_segs,
            colors=np.array([90, 90, 230], dtype=np.uint8),
            line_width=3.5,
        )

    print("Viser running — open http://localhost:8080 in your browser.")
    server.sleep_forever()


if __name__ == "__main__":
    print("Triple-link cartpole swing-up — MuJoCo MJX + OpenSCvx")
    print("=" * 60)
    print(f"nq={n_q}, nv={n_v}, nu={n_u}, N={n}")
    print(f"Links: L1={L1} m, L2={L2} m, L3={L3} m  (total {L1 + L2 + L3} m)")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    final_q = results.nodes["qpos"][-1]
    final_qd = results.nodes["qvel"][-1]
    print()
    print(f"Final joint angles [deg]: {np.rad2deg(final_q[1:])}")
    print(f"Final cart position:      {final_q[0]:.4f} m")
    print(f"Final joint rates [rad/s]:{final_qd[1:]}")

    from openscvx.plotting import plot_controls, plot_states

    plot_states(results).show()
    plot_controls(results).show()

    visualize(results)
