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
from openscvx import ByofSpec, Problem
from openscvx.integrations import mjx_byof

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

# ── State / control definitions ───────────────────────────────────────────────
qpos = ox.State("qpos", shape=(n_q,))
qpos.min = np.array([-100.0, -2 * np.pi, -2 * np.pi, -2 * np.pi])
qpos.max = np.array([100.0, 2 * np.pi, 2 * np.pi, 2 * np.pi])
qpos.initial = np.array([0.0, np.pi, 0.0, 0.0])  # all links hanging down
qpos.final = [ox.Free(0.0), 0.0, 0.0, 0.0]  # all links upright

qvel = ox.State("qvel", shape=(n_v,))
qvel.min = np.array([-12.0, -12.0, -12.0, -12.0])
qvel.max = np.array([12.0, 12.0, 12.0, 12.0])
qvel.initial = np.zeros(n_v)
qvel.final = [0.0, 0.0, 0.0, 0.0]

ctrl = ox.Control("ctrl", shape=(n_u,))
ctrl.min = np.array([-3.0])
ctrl.max = np.array([3.0])
ctrl.guess = np.zeros((n, n_u))

states = [qpos, qvel]
controls = [ctrl]

# ── Dynamics: position kinematics symbolically, velocity via MJX ──────────────
dynamics: dict = {"qpos": qvel}  # nq==nv so this is always valid

byof: ByofSpec = {"dynamics": mjx_byof(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)}

# ── Constraints (CTCS on state / control bounds) ───────────────────────────────
constraints = []
# for state in states:
#     constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

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
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    byof=byof,
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


def qpos_from_V_multishot(
    V: np.ndarray,
    *,
    n_q: int,
    n_v: int,
    n_u: int,
    t_nodes: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Unpack generalized coordinates from the SCP multi-shoot matrix ``V``.

    ``V`` has shape ``((N - 1) * i4, n_substeps)`` where each segment occupies ``i4``
    consecutive rows (state + sensitivities). The first ``n_x = n_q + n_v`` rows of
    each segment block are the propagated state at that substep.

    Returns ``(qpos, t_sample)`` with one row per substep (duplicate boundary samples
    between segments are skipped), or ``(None, None)`` if ``V`` cannot be decoded.
    """
    if V.size == 0:
        return None, None
    n_x = n_q + n_v
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
        t0 = float(t_nodes[seg])
        t1 = float(t_nodes[seg + 1])
        j0 = 0 if seg == 0 else 1  # skip duplicated state at segment joints
        for j in range(j0, n_sub):
            alpha = j / (n_sub - 1) if n_sub > 1 else 0.0
            t_s = (1.0 - alpha) * t0 + alpha * t1
            row0 = seg * i4
            x_vec = np.asarray(V[row0 : row0 + n_x, j], dtype=np.float64).ravel()
            q_rows.append(x_vec[:n_q])
            t_rows.append(t_s)
    if not q_rows:
        return None, None
    return np.stack(q_rows, axis=0), np.asarray(t_rows, dtype=np.float64)


def visualize(results) -> None:
    """Animate the triple-link cartpole in a Viser 3D scene (multi-shoot integrated path only)."""
    import plotly.graph_objects as go
    import viser

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        compute_velocity_colors,
    )
    from openscvx.plotting.viser.plotly_integration import add_animated_plotly_vline

    # ── Extract trajectory data ────────────────────────────────────────────────
    t_vec = results.trajectory["time"].flatten()  # (N_fine,) — playback clock
    u_traj = results.trajectory["ctrl"]  # (N_fine, 1)

    q_nodes = results.nodes["qpos"]  # (n_nodes, 4)
    t_nodes = results.nodes.get("time", None)
    if t_nodes is None:
        t_nodes = np.linspace(float(t_vec[0]), float(t_vec[-1]), len(q_nodes))
    else:
        t_nodes = np.asarray(t_nodes).flatten()

    N = len(t_vec)

    # Multi-shoot integrated trajectory from ``V`` (last SCP discretization matrix).
    _dh = getattr(results, "discretization_history", None) or []
    V_multishot = _dh[-1] if len(_dh) > 0 else None
    q_ms_v, t_ms_v = (
        qpos_from_V_multishot(
            np.asarray(V_multishot, dtype=np.float64),
            n_q=n_q,
            n_v=n_v,
            n_u=n_u,
            t_nodes=t_nodes,
        )
        if V_multishot is not None
        else (None, None)
    )

    if q_ms_v is not None and t_ms_v is not None:
        fk_multishot_anim = [fk_joints(q_ms_v[i]) for i in range(len(q_ms_v))]
        t_multishot_lookup = t_ms_v
        t_angle = t_ms_v
        q_angle = q_ms_v
    else:
        # Fallback: interpolate nodal qpos onto the post-process time grid.
        q_aligned = np.column_stack(
            [np.interp(t_vec, t_nodes, q_nodes[:, j]) for j in range(q_nodes.shape[1])]
        )
        fk_multishot_anim = [fk_joints(q_aligned[i]) for i in range(N)]
        t_multishot_lookup = t_vec
        t_angle = t_vec
        q_angle = q_aligned

    # Tip trail samples (multi-shoot pose at each playback time)
    tip_pos = np.zeros((N, 3))
    for i in range(N):
        ms_i = int(np.argmin(np.abs(t_multishot_lookup - float(t_vec[i]))))
        ms_i = int(np.clip(ms_i, 0, len(fk_multishot_anim) - 1))
        tip_pos[i] = fk_multishot_anim[ms_i][4]

    # ── Viser server ───────────────────────────────────────────────────────────
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")

    # Static rail
    rail = np.array([[[-4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]], dtype=np.float32)
    server.scene.add_line_segments(
        "/rail",
        points=rail,
        colors=np.array([80, 80, 80], dtype=np.uint8),
        line_width=4.0,
    )

    # Upright target marker at top of triple stack
    upright_tip = np.array([0.0, 0.0, L1 + L2 + L3])
    server.scene.add_icosphere(
        "/target",
        radius=0.05,
        color=(50, 220, 100),
        position=tuple(float(v) for v in upright_tip),
    )

    n_nodes = len(q_nodes)

    # Multishot path: dense integrated samples from ``V_multishot``, else chords between nodes.
    if q_ms_v is not None and len(q_ms_v) >= 2:
        fk_ms_poly = [fk_joints(q_ms_v[i]) for i in range(len(q_ms_v))]
        cart_ms_xy = np.array(
            [[float(fk_ms_poly[i][0][0]), 0.0, 0.0] for i in range(len(fk_ms_poly))],
            dtype=np.float32,
        )
        tip_ms_xy = np.array([fk_ms_poly[i][4] for i in range(len(fk_ms_poly))], dtype=np.float32)
        cart_multishot_segs = np.stack(
            [
                np.stack([cart_ms_xy[i], cart_ms_xy[i + 1]], axis=0)
                for i in range(len(cart_ms_xy) - 1)
            ],
            axis=0,
        )
        tip_multishot_segs = np.stack(
            [np.stack([tip_ms_xy[i], tip_ms_xy[i + 1]], axis=0) for i in range(len(tip_ms_xy) - 1)],
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
    else:
        cart_multishot_segs = None

    if cart_multishot_segs is not None:
        server.scene.add_line_segments(
            "/multishot/cart_path",
            points=cart_multishot_segs,
            colors=np.array([90, 90, 230], dtype=np.uint8),
            line_width=3.5,
        )
        server.scene.add_line_segments(
            "/multishot/tip_path",
            points=tip_multishot_segs,
            colors=np.array([230, 170, 40], dtype=np.uint8),
            line_width=3.5,
        )

    # ── Animated multishot rig (aligned to same playback clock) ───────────────
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
        h = server.scene.add_icosphere(
            jname,
            radius=0.03,
            color=(80, 170, 245),
            position=tuple(float(v) for v in fk_multishot_anim[0][1]),
        )
        ms_joint_handles.append(h)

    # ── Animated tip trail ─────────────────────────────────────────────────────
    tip_colors = compute_velocity_colors(tip_pos)
    _, update_trail = add_animated_trail(server, tip_pos, tip_colors, point_size=0.02)

    # ── Sidebar: joint angles ──────────────────────────────────────────────────
    fig_angles = go.Figure()
    angle_names = ["θ₁ (link 1)", "θ₂ (link 2)", "θ₃ (link 3)"]
    angle_colors = ["royalblue", "darkorange", "green"]
    for k in range(3):
        fig_angles.add_trace(
            go.Scatter(
                x=t_angle.tolist(),
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

    # ── Sidebar: control force ─────────────────────────────────────────────────
    fig_ctrl = go.Figure()
    fig_ctrl.add_trace(
        go.Scatter(
            x=t_vec.tolist(),
            y=u_traj[:, 0].tolist(),
            mode="lines",
            name="Cart force",
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
        _, update_angles = add_animated_plotly_vline(server, fig_angles, t_vec, folder_name=None)
        _, update_ctrl = add_animated_plotly_vline(server, fig_ctrl, t_vec, folder_name=None)

    def update_multishot_scene(frame_idx: int) -> None:
        t_cur = float(t_vec[frame_idx])
        ms_i = int(np.argmin(np.abs(t_multishot_lookup - t_cur)))
        ms_i = int(np.clip(ms_i, 0, len(fk_multishot_anim) - 1))
        _, h1, h2, h3, _ = fk_multishot_anim[ms_i]
        ms_cart_x = float(fk_multishot_anim[ms_i][0][0])
        ms_cart_handle.position = (ms_cart_x, 0.0, 0.0)
        ms_link_handle.points = _multishot_link_segments(ms_i)
        for handle, pos in zip(ms_joint_handles, (h1, h2, h3)):
            handle.position = tuple(float(v) for v in pos)

    # ── Animation controls ─────────────────────────────────────────────────────
    add_animation_controls(
        server,
        t_vec,
        [
            update_multishot_scene,
            update_trail,
            update_angles,
            update_ctrl,
        ],
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
