"""Double-link cartpole swing-up using MuJoCo MJX dynamics.

The optimizer drives two serial links on a cart from the hanging equilibrium
(θ₁=π, θ₂=0) to the unstable upright equilibrium (θ₁=θ₂=0) using a single
horizontal force on the cart — a classic underactuated benchmark that is
harder than the single-link case but tractable in a few seconds.

State  : qpos = [cart_x, θ₁, θ₂],  qvel = [ẋ, θ̇₁, θ̇₂]
Control: ctrl = [F_cart]  (normalised; gear=60 → max ±60 N)

Link lengths: L₁=0.5 m, L₂=0.4 m.

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

L1, L2 = 0.5, 0.4  # link lengths (m)

DOUBLE_CARTPOLE_XML = f"""
<mujoco model="double_cartpole">
  <option gravity="0 0 -9.81" timestep="0.005" integrator="Euler"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-4 4"/>
      <geom name="cart_geom" type="box" size="0.25 0.15 0.1"
            mass="2.0" rgba="0.35 0.35 0.75 1"/>
      <!-- Link 1 — pivot at cart centre -->
      <body name="link1" pos="0 0 0">
        <joint name="hinge1" type="hinge" axis="0 1 0" limited="false"/>
        <geom name="pole1" type="capsule" fromto="0 0 0 0 0 {L1}"
              size="0.045" mass="0.5" rgba="0.85 0.3 0.3 1"/>
        <!-- Link 2 — pivot at tip of link 1 -->
        <body name="link2" pos="0 0 {L1}">
          <joint name="hinge2" type="hinge" axis="0 1 0" limited="false"/>
          <geom name="pole2" type="capsule" fromto="0 0 0 0 0 {L2}"
                size="0.038" mass="0.4" rgba="0.3 0.8 0.3 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- Single actuator on the cart slider; gear scales ctrl ∈ [−1,1] to force in N -->
    <motor joint="slider" name="cart_force" gear="60"
           ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

mj_model = mujoco.MjModel.from_xml_string(DOUBLE_CARTPOLE_XML)
# Contacts not needed; disabling them keeps MJX forward-dynamics JAX-differentiable.
mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
mjx_model = mjx.put_model(mj_model)

n_q = int(mjx_model.nq)  # 3: cart_x, θ₁, θ₂
n_v = int(mjx_model.nv)  # 3: ẋ, θ̇₁, θ̇₂  (nq == nv, no quaternion)
n_u = int(mjx_model.nu)  # 1: cart force

n = 400
total_time = 2.5

# ── MJX dynamics adapter ──────────────────────────────────────────────────────
dyn = ox.MjxDynamics(mjx_model)
qpos, qvel = dyn.states
(ctrl,) = dyn.controls
ctrl.parameterization = "ZOH"

qpos.min = np.array([-8.0, -2 * np.pi, -2 * np.pi])
qpos.max = np.array([8.0, 2 * np.pi, 2 * np.pi])
qpos.initial = np.array([0.0, np.pi, 0.0])  # cart at origin, link1 hanging down
qpos.final = [ox.Free(0.0), 0.0, 0.0]  # cart free, both links upright

qvel.min = np.array([-12.0, -12.0, -12.0])
qvel.max = np.array([12.0, 12.0, 12.0])
qvel.initial = np.zeros(n_v)
qvel.final = [0.0, 0.0, 0.0]

ctrl.min = np.array([-2.0])
ctrl.max = np.array([2.0])
ctrl.guess = np.zeros((n, n_u))

# ── Constraints (CTCS on state / control bounds) ───────────────────────────────
constraints = []

# ── Initial guess: linearly swing θ₁ from π → 0, θ₂ stays 0 ─────────────────
th1_guess = np.linspace(np.pi, 0.0, n)
qpos.guess = np.column_stack([np.zeros(n), th1_guess, np.zeros(n)])
qvel.guess = np.zeros((n, n_v))

time = ox.Time(
    initial=0.0,
    final=total_time,
    min=0.0,
    max=total_time,
)

problem = ox.Problem(
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
    discretizer={"diffrax_kwargs": {"atol": 1e-12, "rtol": 1e-12}},
    solver={
        "solver_args": {
            "enforce_dpp": True,
            "canon_backend": "COO",
            "abs_tol": 1e-12,
            "rel_tol": 1e-12,
        }
    },
    float_dtype="float64",
)


# ── Forward kinematics helpers ─────────────────────────────────────────────────
def fk_joints(q: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return world-frame positions (in XZ plane) of cart, h1, h2, tip."""
    cx = float(q[0])
    t1, t2 = float(q[1]), float(q[2])
    cart = np.array([cx, 0.0, 0.0])
    h1 = cart  # hinge1 is at cart centre
    h2 = h1 + np.array([L1 * np.sin(t1), 0.0, L1 * np.cos(t1)])
    tip = h2 + np.array([L2 * np.sin(t1 + t2), 0.0, L2 * np.cos(t1 + t2)])
    return cart, h1, h2, tip


def simulate_mujoco(results) -> dict:
    """Run the optimised control sequence through MuJoCo's CPU simulator.

    Matches the setup used by ``post_process``:

    * **dt = 0.01 s (100 Hz)** — same as ``PropagationConfig.dt`` default.
    * **FOH**: control linearly interpolated between consecutive SCP nodes.
    * **Actual node times**: read from ``results.nodes["time"]`` when
      available; falls back to linspace for fixed-time problems.
    * **Initial state**: taken from the converged SCP node 0.

    Returns a dict with keys ``"time"``, ``"qpos"``, ``"qvel"``.
    """
    dt = 0.01
    mj_model.opt.timestep = dt

    data = mujoco.MjData(mj_model)
    data.qpos[:] = results.nodes["qpos"][0]
    data.qvel[:] = results.nodes["qvel"][0]
    mujoco.mj_forward(mj_model, data)

    u_nodes = results.nodes["ctrl"]
    n_nodes = len(u_nodes)

    raw_t = results.trajectory["time"].flatten()
    t_start, t_end = float(raw_t[0]), float(raw_t[-1])

    t_nodes_raw = results.nodes.get("time", None)
    t_nodes_sim = (
        np.asarray(t_nodes_raw).flatten()
        if t_nodes_raw is not None
        else np.linspace(t_start, t_end, n_nodes)
    )

    n_steps = int(round((t_end - t_start) / dt)) + 1
    rec_t = np.empty(n_steps)
    rec_q = np.empty((n_steps, n_q))
    rec_qd = np.empty((n_steps, n_v))

    sim_t = t_start
    for step in range(n_steps):
        rec_t[step] = sim_t
        rec_q[step] = data.qpos.copy()
        rec_qd[step] = data.qvel.copy()

        k = int(np.clip(np.searchsorted(t_nodes_sim, sim_t, side="right") - 1, 0, n_nodes - 2))
        t0, t1 = float(t_nodes_sim[k]), float(t_nodes_sim[k + 1])
        alpha = float(np.clip((sim_t - t0) / (t1 - t0) if t1 > t0 else 0.0, 0.0, 1.0))
        data.ctrl[:] = (1.0 - alpha) * u_nodes[k] + alpha * u_nodes[k + 1]

        mujoco.mj_step(mj_model, data)
        sim_t += dt

    print(
        f"MuJoCo simulation: {n_steps} steps, dt={dt * 1e3:.1f} ms (FOH, 100 Hz)  "
        f"| final θ₁={np.rad2deg(rec_q[-1, 1]):.2f}°  θ₂={np.rad2deg(rec_q[-1, 2]):.2f}°"
    )
    return {"time": rec_t, "qpos": rec_q, "qvel": rec_qd}


def qpos_from_V_multishot(
    V: np.ndarray,
    *,
    n_q: int,
    n_v: int,
    n_u: int,
    t_nodes: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Unpack generalized coordinates from the SCP multi-shoot matrix ``V``."""
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


def visualize(results, sim: dict | None = None) -> None:
    """Animate the double-link cartpole in a Viser 3D scene.

    Args:
        results: OptimizationResults from ``problem.post_process()``.
        sim:     Optional dict from ``simulate_mujoco(results)`` — when provided,
                 a second (orange) pendulum overlays the MuJoCo simulation.
    """
    import plotly.graph_objects as go
    import viser

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        compute_velocity_colors,
    )
    from openscvx.plotting.viser.plotly_integration import add_animated_plotly_vline

    # ── Extract trajectory data ────────────────────────────────────────────────
    t_vec = results.trajectory["time"].flatten()  # (N_fine,)
    q_traj = results.trajectory["qpos"]  # (N_fine, 3)
    u_traj = results.trajectory["ctrl"]  # (N_fine, 1)

    q_nodes = results.nodes["qpos"]  # (n_nodes, 3)
    t_nodes = results.nodes.get("time", None)
    if t_nodes is None:
        t_nodes = np.linspace(float(t_vec[0]), float(t_vec[-1]), len(q_nodes))
    else:
        t_nodes = np.asarray(t_nodes).flatten()

    N = len(t_vec)

    fk_all = [fk_joints(q_traj[i]) for i in range(N)]
    tip_pos = np.array([j[3] for j in fk_all], dtype=np.float32)  # index 3 = tip

    fk_nodes = [fk_joints(q_nodes[i]) for i in range(len(q_nodes))]

    # Multi-shoot integrated trajectory from ``V``
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
    else:
        q_aligned = np.column_stack(
            [np.interp(t_vec, t_nodes, q_nodes[:, j]) for j in range(q_nodes.shape[1])]
        )
        fk_multishot_anim = [fk_joints(q_aligned[i]) for i in range(N)]
        t_multishot_lookup = t_vec

    # ── Viser server ───────────────────────────────────────────────────────────
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")

    server.scene.add_line_segments(
        "/rail",
        points=np.array([[[-4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]], dtype=np.float32),
        colors=np.array([80, 80, 80], dtype=np.uint8),
        line_width=4.0,
    )
    server.scene.add_icosphere(
        "/target",
        radius=0.05,
        color=(50, 220, 100),
        position=(0.0, 0.0, float(L1 + L2)),
    )

    # ── Static discretization node markers ────────────────────────────────────
    n_nodes = len(fk_nodes)
    ghost_segs = np.stack(
        [
            np.stack(
                [
                    np.array(fk_nodes[i][j], dtype=np.float32),
                    np.array(fk_nodes[i][j + 1], dtype=np.float32),
                ],
                axis=0,
            )
            for i in range(n_nodes)
            for j in range(1, 3)  # h1→h2, h2→tip
        ],
        axis=0,
    )  # (n_nodes*2, 2, 3)

    server.scene.add_line_segments(
        "/nodes/links",
        points=ghost_segs,
        colors=np.array([160, 160, 160], dtype=np.uint8),
        line_width=1.5,
    )
    cart_node_pos = np.array(
        [[float(fk_nodes[i][0][0]), 0.0, 0.0] for i in range(n_nodes)], dtype=np.float32
    )
    server.scene.add_point_cloud(
        "/nodes/cart",
        points=cart_node_pos,
        colors=np.tile(np.array([100, 100, 220], dtype=np.uint8), (n_nodes, 1)),
        point_size=0.04,
    )
    tip_node_pos = np.array([fk_nodes[i][3] for i in range(n_nodes)], dtype=np.float32)
    server.scene.add_point_cloud(
        "/nodes/tips",
        points=tip_node_pos,
        colors=np.tile(np.array([220, 180, 50], dtype=np.uint8), (n_nodes, 1)),
        point_size=0.05,
    )

    # ── Multishot path overlay ─────────────────────────────────────────────────
    if q_ms_v is not None and len(q_ms_v) >= 2:
        fk_ms_poly = [fk_joints(q_ms_v[i]) for i in range(len(q_ms_v))]
        cart_ms_xy = np.array(
            [[float(fk_ms_poly[i][0][0]), 0.0, 0.0] for i in range(len(fk_ms_poly))],
            dtype=np.float32,
        )
        tip_ms_xy = np.array([fk_ms_poly[i][3] for i in range(len(fk_ms_poly))], dtype=np.float32)
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
        cart_multishot_segs = np.stack(
            [
                np.stack([cart_node_pos[i], cart_node_pos[i + 1]], axis=0)
                for i in range(n_nodes - 1)
            ],
            axis=0,
        ).astype(np.float32)
        tip_multishot_segs = np.stack(
            [np.stack([tip_node_pos[i], tip_node_pos[i + 1]], axis=0) for i in range(n_nodes - 1)],
            axis=0,
        ).astype(np.float32)
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

    # ── Animated cart ──────────────────────────────────────────────────────────
    cart_handle = server.scene.add_box(
        "/cart",
        dimensions=(0.5, 0.3, 0.2),
        position=tuple(float(v) for v in fk_all[0][0]),
        color=(90, 90, 190),
    )

    # ── Animated links (2 segments: h1→h2, h2→tip) ────────────────────────────
    def _link_segments(i: int) -> np.ndarray:
        _, h1, h2, tip = fk_all[i]
        return np.array([[h1, h2], [h2, tip]], dtype=np.float32)

    link_colors = np.array(
        [
            [[220, 80, 80], [220, 80, 80]],  # link 1 — red
            [[80, 200, 80], [80, 200, 80]],  # link 2 — green
        ],
        dtype=np.uint8,
    )  # shape (2, 2, 3)
    link_handle = server.scene.add_line_segments(
        "/links",
        points=_link_segments(0),
        colors=link_colors,
        line_width=7.0,
    )

    joint_handles = []
    for jname, jcol in [("/j1", (200, 60, 60)), ("/j2", (60, 200, 60))]:
        h = server.scene.add_icosphere(
            jname,
            radius=0.045,
            color=jcol,
            position=tuple(float(v) for v in fk_all[0][1]),
        )
        joint_handles.append(h)

    # ── Animated multishot rig ─────────────────────────────────────────────────
    ms_cart_handle = server.scene.add_box(
        "/multishot/cart",
        dimensions=(0.42, 0.24, 0.16),
        position=tuple(float(v) for v in fk_multishot_anim[0][0]),
        color=(55, 150, 255),
    )

    def _multishot_link_segments(i: int) -> np.ndarray:
        _, h1, h2, tip = fk_multishot_anim[i]
        return np.array([[h1, h2], [h2, tip]], dtype=np.float32)

    ms_link_colors = np.array(
        [
            [[120, 190, 255], [120, 190, 255]],
            [[95, 175, 245], [95, 175, 245]],
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
    for jname in ["/multishot/j1", "/multishot/j2"]:
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
    for k, (name, col) in enumerate(
        zip(["θ₁ (link 1)", "θ₂ (link 2)"], ["royalblue", "darkorange"])
    ):
        fig_angles.add_trace(
            go.Scatter(
                x=t_vec.tolist(),
                y=np.rad2deg(q_traj[:, k + 1]).tolist(),
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

    # ── MuJoCo simulation overlay (orange chain) ──────────────────────────────
    sim_callbacks: list = []
    if sim is not None:
        sim_t = sim["time"]
        sim_q = sim["qpos"]
        fk_sim = [fk_joints(sim_q[i]) for i in range(len(sim_t))]

        sim_tip = np.array([j[3] for j in fk_sim], dtype=np.float32)
        server.scene.add_point_cloud(
            "/sim/tip_trail",
            points=sim_tip,
            colors=np.tile(np.array([230, 120, 30], dtype=np.uint8), (len(sim_t), 1)),
            point_size=0.01,
        )

        sim_link_colors = np.array(
            [
                [[230, 100, 20], [230, 100, 20]],
                [[230, 150, 40], [230, 150, 40]],
            ],
            dtype=np.uint8,
        )

        def _sim_link_segments(i: int) -> np.ndarray:
            _, h1, h2, tip = fk_sim[i]
            return np.array([[h1, h2], [h2, tip]], dtype=np.float32)

        sim_cart_handle = server.scene.add_box(
            "/sim/cart",
            dimensions=(0.5, 0.3, 0.2),
            position=tuple(float(v) for v in fk_sim[0][0]),
            color=(200, 80, 20),
        )
        sim_link_handle = server.scene.add_line_segments(
            "/sim/links",
            points=_sim_link_segments(0),
            colors=sim_link_colors,
            line_width=4.0,
        )
        sim_joint_handles = []
        for jname, jcol in [("/sim/j1", (200, 80, 20)), ("/sim/j2", (220, 140, 40))]:
            h = server.scene.add_icosphere(
                jname,
                radius=0.035,
                color=jcol,
                position=tuple(float(v) for v in fk_sim[0][1]),
            )
            sim_joint_handles.append(h)

        for k, (name, col) in enumerate(zip(["θ₁ sim", "θ₂ sim"], ["#e85", "#ea6"])):
            fig_angles.add_trace(
                go.Scatter(
                    x=sim_t.tolist(),
                    y=np.rad2deg(sim_q[:, k + 1]).tolist(),
                    mode="lines",
                    name=name,
                    line={"color": col, "width": 1.5, "dash": "dot"},
                )
            )
        update_angles(0)

        def update_sim(frame_idx: int) -> None:
            t_cur = float(t_vec[frame_idx])
            si = int(np.clip(np.searchsorted(sim_t, t_cur) - 1, 0, len(sim_t) - 1))
            _, h1, h2, _ = fk_sim[si]
            sim_cart_handle.position = (float(fk_sim[si][0][0]), 0.0, 0.0)
            sim_link_handle.points = _sim_link_segments(si)
            for handle, pos in zip(sim_joint_handles, (h1, h2)):
                handle.position = tuple(float(v) for v in pos)

        sim_callbacks.append(update_sim)

    # ── Per-frame update ───────────────────────────────────────────────────────
    def update_scene(frame_idx: int) -> None:
        _, h1, h2, _ = fk_all[frame_idx]
        cart_handle.position = (float(q_traj[frame_idx, 0]), 0.0, 0.0)
        link_handle.points = _link_segments(frame_idx)
        for handle, pos in zip(joint_handles, (h1, h2)):
            handle.position = tuple(float(v) for v in pos)

    def update_multishot_scene(frame_idx: int) -> None:
        t_cur = float(t_vec[frame_idx])
        ms_i = int(
            np.clip(np.argmin(np.abs(t_multishot_lookup - t_cur)), 0, len(fk_multishot_anim) - 1)
        )
        _, h1, h2, _ = fk_multishot_anim[ms_i]
        ms_cart_handle.position = (float(fk_multishot_anim[ms_i][0][0]), 0.0, 0.0)
        ms_link_handle.points = _multishot_link_segments(ms_i)
        for handle, pos in zip(ms_joint_handles, (h1, h2)):
            handle.position = tuple(float(v) for v in pos)

    add_animation_controls(
        server,
        t_vec,
        [
            update_scene,
            update_multishot_scene,
            update_trail,
            update_angles,
            update_ctrl,
            *sim_callbacks,
        ],
    )

    print("Viser running — open http://localhost:8080 in your browser.")
    server.sleep_forever()


if __name__ == "__main__":
    print("Double-link cartpole swing-up — MuJoCo MJX + OpenSCvx")
    print("=" * 60)
    print(f"nq={n_q}, nv={n_v}, nu={n_u}, N={n}")
    print(f"Links: L1={L1} m, L2={L2} m  (total {L1 + L2} m)")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    final_q = results.nodes["qpos"][-1]
    final_qd = results.nodes["qvel"][-1]
    print()
    print(f"Final joint angles [deg]: {np.rad2deg(final_q[1:])}")
    print(f"Final cart position:      {final_q[0]:.4f} m")
    print(f"Final joint rates [rad/s]: {final_qd[1:]}")

    print()
    print("Running MuJoCo CPU simulation with solved controls…")
    sim = simulate_mujoco(results)

    visualize(results, sim=sim)
