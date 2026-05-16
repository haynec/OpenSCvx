"""3D triple-link cartpole swing-up using MuJoCo MJX dynamics.

Extends ``triple_cartpole_mjx.py`` to **3D**:

* The cart slides on a **horizontal plane** (two slide joints, X and Y).
* Each pendulum link has a **2-DOF universal joint** — two hinges with
  perpendicular axes (around the parent's X then Y) — giving spherical
  pendulum motion *without* the quaternion bookkeeping of a ball joint.

The resulting system has

    nq = nv = 8   (cart_x, cart_y, α₁, β₁, α₂, β₂, α₃, β₃)
    nu     = 2    (cart force in X and Y)

The optimizer drives all three links from the hanging equilibrium
(α₁=π, every other angle 0) to the unstable upright equilibrium
(all angles 0) using only horizontal forces on the cart.

Link convention
---------------
Each link's geom extends in its parent's local +Z direction.  With both
hinge angles equal to 0 the link points along the parent's +Z axis.  For
the bottom link this means upright at angle 0; the link hangs straight
down at α=π.

Both 2-DOF parameterisations (X-then-Y and Y-then-X intrinsic rotations)
have a coordinate singularity when the second hinge angle reaches ±π/2.
The straight swing-up trajectory keeps β=0 throughout, so the
parameterisation is well-conditioned for this problem.

State  : qpos = [cart_x, cart_y, α₁, β₁, α₂, β₂, α₃, β₃],  qvel = q̇
Control: ctrl = [F_x, F_y]  (normalised; gear=60 → ±60 N per axis)

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
from openscvx import Problem

L1, L2, L3 = 0.5, 0.4, 0.3  # link lengths (m)

TRIPLE_CARTPOLE_3D_XML = f"""
<mujoco model="triple_cartpole_3d">
  <option gravity="0 0 -9.81" timestep="0.005" integrator="Euler"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <!-- Cart slides freely on the horizontal plane -->
      <joint name="slide_x" type="slide" axis="1 0 0" limited="true" range="-4 4"/>
      <joint name="slide_y" type="slide" axis="0 1 0" limited="true" range="-4 4"/>
      <geom name="cart_geom" type="box" size="0.25 0.25 0.1"
            mass="2.0" rgba="0.35 0.35 0.75 1"/>
      <!-- Link 1 — 2-DOF universal joint at cart centre -->
      <body name="link1" pos="0 0 0">
        <joint name="hinge1_x" type="hinge" axis="1 0 0" limited="false"/>
        <joint name="hinge1_y" type="hinge" axis="0 1 0" limited="false"/>
        <geom name="pole1" type="capsule" fromto="0 0 0 0 0 {L1}"
              size="0.04" mass="0.5" rgba="0.85 0.3 0.3 1"/>
        <!-- Link 2 — 2-DOF universal joint at tip of link 1 -->
        <body name="link2" pos="0 0 {L1}">
          <joint name="hinge2_x" type="hinge" axis="1 0 0" limited="false"/>
          <joint name="hinge2_y" type="hinge" axis="0 1 0" limited="false"/>
          <geom name="pole2" type="capsule" fromto="0 0 0 0 0 {L2}"
                size="0.035" mass="0.4" rgba="0.3 0.8 0.3 1"/>
          <!-- Link 3 — 2-DOF universal joint at tip of link 2 -->
          <body name="link3" pos="0 0 {L2}">
            <joint name="hinge3_x" type="hinge" axis="1 0 0" limited="false"/>
            <joint name="hinge3_y" type="hinge" axis="0 1 0" limited="false"/>
            <geom name="pole3" type="capsule" fromto="0 0 0 0 0 {L3}"
                  size="0.03" mass="0.3" rgba="0.3 0.3 0.85 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- Two actuators on the cart slide joints; gear scales ctrl ∈ [−1,1] to N -->
    <motor joint="slide_x" name="cart_force_x" gear="60"
           ctrlrange="-1 1" ctrllimited="true"/>
    <motor joint="slide_y" name="cart_force_y" gear="60"
           ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

mj_model = mujoco.MjModel.from_xml_string(TRIPLE_CARTPOLE_3D_XML)
mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
mjx_model = mjx.put_model(mj_model)

n_q = int(mjx_model.nq)  # 8: cart_xy + 3 × (αᵢ, βᵢ)
n_v = int(mjx_model.nv)  # 8 (nq == nv)
n_u = int(mjx_model.nu)  # 2: F_x, F_y

n = 60
total_time = 4.0

# Convenience indices for clarity.
IDX_CART_X, IDX_CART_Y = 0, 1
IDX_A1, IDX_B1 = 2, 3
IDX_A2, IDX_B2 = 4, 5
IDX_A3, IDX_B3 = 6, 7

# ── MJX dynamics adapter ──────────────────────────────────────────────────────
dyn = ox.MjxDynamics(mjx_model)
qpos, qvel = dyn.states
(ctrl,) = dyn.controls

qpos.min = np.array(
    [-8.0, -8.0, -2 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi]
)
qpos.max = -qpos.min
# Hanging: cart at origin, link 1 hanging (α₁ = π), all others 0.
qpos.initial = np.array([-0.5, 1.0, np.pi, np.pi / 64, 0.0, 0.0, 0.0, 0.0])
# Upright: cart free, all angles 0.
qpos.final = [
    ox.Free(0.0),
    ox.Free(0.0),  # cart x, y free
    0.0,
    0.0,  # link 1 upright
    0.0,
    0.0,  # link 2 upright
    0.0,
    0.0,  # link 3 upright
]

qvel.min = np.full(n_v, -12.0)
qvel.max = np.full(n_v, 12.0)
qvel.initial = np.zeros(n_v)
qvel.final = [0.0] * n_v

ctrl.min = np.array([-2.0, -2.0])
ctrl.max = np.array([2.0, 2.0])
ctrl.guess = np.zeros((n, n_u))

# ── Constraints (CTCS on state / control bounds) ─────────────────────────────
constraints = []

# ── Initial guess: linearly swing α₁ from π → 0; everything else stays 0 ─────
a1_guess = np.linspace(np.pi, 0.0, n)
qpos_guess = np.zeros((n, n_q))
qpos_guess[:, IDX_A1] = a1_guess
qpos.guess = qpos_guess
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
        "lam_prox": 1e-1,
        "lam_cost": 0e0,
        "lam_vc": 4e0,
        "autotuner": ox.ConstantProximalWeight(),
    },
    float_dtype="float64",
)


# ── Forward kinematics helpers ────────────────────────────────────────────────
def _R_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _R_y(b: float) -> np.ndarray:
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def fk_joints(q: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return world-frame positions of cart, h1, h2, h3, tip (each 3-vectors).

    The pendulum joint frames compose intrinsically: each link's hinges
    rotate the body frame inherited from the parent.  In rotation-matrix
    form, the cumulative orientation of the i-th link is

        R_i = R_x(α₁) R_y(β₁) … R_x(αᵢ) R_y(βᵢ),

    and the link extends along its body +Z axis, so its tip is
    ``parent_pos + Lᵢ · (R_i @ ẑ)``.
    """
    cart = np.array([float(q[IDX_CART_X]), float(q[IDX_CART_Y]), 0.0])
    z_hat = np.array([0.0, 0.0, 1.0])

    R = np.eye(3)
    h1 = cart  # hinge 1 at cart centre

    R = R @ _R_x(float(q[IDX_A1])) @ _R_y(float(q[IDX_B1]))
    h2 = h1 + L1 * (R @ z_hat)

    R = R @ _R_x(float(q[IDX_A2])) @ _R_y(float(q[IDX_B2]))
    h3 = h2 + L2 * (R @ z_hat)

    R = R @ _R_x(float(q[IDX_A3])) @ _R_y(float(q[IDX_B3]))
    tip = h3 + L3 * (R @ z_hat)

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

    Identical structure to the 2D version — generic in (n_q, n_v, n_u).
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
        j0 = 0 if seg == 0 else 1
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
    """Animate the 3D triple-link cartpole in a Viser scene (multi-shoot integrated path only)."""
    import plotly.graph_objects as go
    import viser

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        compute_velocity_colors,
    )
    from openscvx.plotting.viser.plotly_integration import add_animated_plotly_vline

    # ── Extract trajectory data ────────────────────────────────────────────────
    t_vec = results.trajectory["time"].flatten()  # playback clock
    u_traj = results.trajectory["ctrl"]

    q_nodes = results.nodes["qpos"]
    t_nodes = results.nodes.get("time", None)
    if t_nodes is None:
        t_nodes = np.linspace(float(t_vec[0]), float(t_vec[-1]), len(q_nodes))
    else:
        t_nodes = np.asarray(t_nodes).flatten()

    N = len(t_vec)

    # Multishot path from V (last SCP discretization matrix)
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
        q_aligned = np.column_stack(
            [np.interp(t_vec, t_nodes, q_nodes[:, j]) for j in range(q_nodes.shape[1])]
        )
        fk_multishot_anim = [fk_joints(q_aligned[i]) for i in range(N)]
        t_multishot_lookup = t_vec
        t_angle = t_vec
        q_angle = q_aligned

    tip_pos = np.zeros((N, 3))
    for i in range(N):
        ms_i = int(np.argmin(np.abs(t_multishot_lookup - float(t_vec[i]))))
        ms_i = int(np.clip(ms_i, 0, len(fk_multishot_anim) - 1))
        tip_pos[i] = fk_multishot_anim[ms_i][4]

    n_nodes = len(q_nodes)

    # ── Viser server ───────────────────────────────────────────────────────────
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")

    # Cart-motion plane (a faint horizontal grid + outline box).
    server.scene.add_grid(
        "/plane",
        width=10.0,
        height=10.0,
        cell_size=0.5,
        position=(0.0, 0.0, -0.105),
    )
    plane_outline = np.array(
        [
            [
                [-4.0, -4.0, 0.0],
                [4.0, -4.0, 0.0],
            ],
            [
                [4.0, -4.0, 0.0],
                [4.0, 4.0, 0.0],
            ],
            [
                [4.0, 4.0, 0.0],
                [-4.0, 4.0, 0.0],
            ],
            [
                [-4.0, 4.0, 0.0],
                [-4.0, -4.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )
    server.scene.add_line_segments(
        "/plane/outline",
        points=plane_outline,
        colors=np.array([110, 110, 110], dtype=np.uint8),
        line_width=2.0,
    )

    # Upright target marker
    upright_tip = np.array([0.0, 0.0, L1 + L2 + L3])
    server.scene.add_icosphere(
        "/target",
        radius=0.05,
        color=(50, 220, 100),
        position=tuple(float(v) for v in upright_tip),
    )

    # ── Multishot polylines (cart path on plane + tip path in 3D) ────────────
    if q_ms_v is not None and len(q_ms_v) >= 2:
        fk_ms_poly = [fk_joints(q_ms_v[i]) for i in range(len(q_ms_v))]
        cart_ms = np.array(
            [
                [float(fk_ms_poly[i][0][0]), float(fk_ms_poly[i][0][1]), 0.0]
                for i in range(len(fk_ms_poly))
            ],
            dtype=np.float32,
        )
        tip_ms = np.array([fk_ms_poly[i][4] for i in range(len(fk_ms_poly))], dtype=np.float32)
        cart_multishot_segs = np.stack(
            [np.stack([cart_ms[i], cart_ms[i + 1]], axis=0) for i in range(len(cart_ms) - 1)],
            axis=0,
        )
        tip_multishot_segs = np.stack(
            [np.stack([tip_ms[i], tip_ms[i + 1]], axis=0) for i in range(len(tip_ms) - 1)], axis=0
        )
    elif n_nodes >= 2:
        fk_nd = [fk_joints(q_nodes[i]) for i in range(n_nodes)]
        cart_node_pos = np.array(
            [[float(fk_nd[i][0][0]), float(fk_nd[i][0][1]), 0.0] for i in range(n_nodes)],
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

    # ── Multishot animated rig ─────────────────────────────────────────────────
    ms_cart_handle = server.scene.add_box(
        "/multishot/cart",
        dimensions=(0.42, 0.42, 0.16),
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

    # ── Animated tip trail ────────────────────────────────────────────────────
    tip_colors = compute_velocity_colors(tip_pos)
    _, update_trail = add_animated_trail(server, tip_pos, tip_colors, point_size=0.02)

    # ── Sidebar: joint angles (all 6) ─────────────────────────────────────────
    fig_angles = go.Figure()
    angle_specs = [
        ("α₁ (link 1, X)", IDX_A1, "royalblue"),
        ("β₁ (link 1, Y)", IDX_B1, "deepskyblue"),
        ("α₂ (link 2, X)", IDX_A2, "darkorange"),
        ("β₂ (link 2, Y)", IDX_B2, "gold"),
        ("α₃ (link 3, X)", IDX_A3, "green"),
        ("β₃ (link 3, Y)", IDX_B3, "limegreen"),
    ]
    for name, idx, col in angle_specs:
        fig_angles.add_trace(
            go.Scatter(
                x=t_angle.tolist(),
                y=np.rad2deg(q_angle[:, idx]).tolist(),
                mode="lines",
                name=name,
                line={"color": col, "width": 2},
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

    # ── Sidebar: cart position (XY plot) and control forces (2 traces) ───────
    fig_cart = go.Figure()
    fig_cart.add_trace(
        go.Scatter(
            x=q_angle[:, IDX_CART_X].tolist(),
            y=q_angle[:, IDX_CART_Y].tolist(),
            mode="lines",
            name="Cart trajectory (multi-shoot)",
            line={"color": "royalblue", "width": 2},
        )
    )
    fig_cart.update_layout(
        title="Cart path on plane",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        yaxis={"scaleanchor": "x", "scaleratio": 1.0},
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    fig_ctrl = go.Figure()
    fig_ctrl.add_trace(
        go.Scatter(
            x=t_vec.tolist(),
            y=u_traj[:, 0].tolist(),
            mode="lines",
            name="F_x",
            line={"color": "crimson", "width": 2},
        )
    )
    fig_ctrl.add_trace(
        go.Scatter(
            x=t_vec.tolist(),
            y=u_traj[:, 1].tolist(),
            mode="lines",
            name="F_y",
            line={"color": "darkmagenta", "width": 2},
        )
    )
    fig_ctrl.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_ctrl.update_layout(
        title="Cart controls (normalised)",
        xaxis_title="Time (s)",
        yaxis_title="u",
        legend={"orientation": "h"},
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    with server.gui.add_folder("Plots"):
        _, update_angles = add_animated_plotly_vline(server, fig_angles, t_vec, folder_name=None)
        _, update_ctrl = add_animated_plotly_vline(server, fig_ctrl, t_vec, folder_name=None)
        # cart-XY plot is static (no time slider needed)
        server.gui.add_plotly(fig_cart)

    def update_multishot_scene(frame_idx: int) -> None:
        t_cur = float(t_vec[frame_idx])
        ms_i = int(np.argmin(np.abs(t_multishot_lookup - t_cur)))
        ms_i = int(np.clip(ms_i, 0, len(fk_multishot_anim) - 1))
        cart, h1, h2, h3, _ = fk_multishot_anim[ms_i]
        ms_cart_handle.position = (float(cart[0]), float(cart[1]), 0.0)
        ms_link_handle.points = _multishot_link_segments(ms_i)
        for handle, pos in zip(ms_joint_handles, (h1, h2, h3)):
            handle.position = tuple(float(v) for v in pos)

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
    print("3D Triple-link cartpole swing-up — MuJoCo MJX + OpenSCvx")
    print("=" * 60)
    print(f"nq={n_q}, nv={n_v}, nu={n_u}, N={n}")
    print(f"Links: L1={L1} m, L2={L2} m, L3={L3} m  (total {L1 + L2 + L3} m)")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    final_q = results.nodes["qpos"][-1]
    print()
    print(f"Final cart position:    ({final_q[IDX_CART_X]:.4f}, {final_q[IDX_CART_Y]:.4f}) m")
    a1, b1 = np.rad2deg(final_q[IDX_A1]), np.rad2deg(final_q[IDX_B1])
    a2, b2 = np.rad2deg(final_q[IDX_A2]), np.rad2deg(final_q[IDX_B2])
    a3, b3 = np.rad2deg(final_q[IDX_A3]), np.rad2deg(final_q[IDX_B3])
    print(f"Final α₁,β₁ [deg]:      ({a1:.2f}, {b1:.2f})")
    print(f"Final α₂,β₂ [deg]:      ({a2:.2f}, {b2:.2f})")
    print(f"Final α₃,β₃ [deg]:      ({a3:.2f}, {b3:.2f})")

    from openscvx.plotting import plot_controls, plot_states

    plot_states(results).show()
    plot_controls(results).show()

    visualize(results)
