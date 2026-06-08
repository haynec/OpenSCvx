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
from examples.plotting_viser import (
    create_snapshot_plotting_server,
    extract_multishoot_trajectory,
)
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

problem.settings.prp.dt = 1e-3


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


def extract_multishoot_qpos_chronological(
    V_multi_shoot: np.ndarray,
    *,
    n_x: int,
    n_u: int,
    n_q: int,
    t_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Unpack propagated states from ``V`` in time order (SCP / realtime layout).

    ``V`` has shape ``((N-1) * segment_size, n_substeps)``.  Each segment block
    stores the integrated state in its first ``n_x`` rows at every substep column.
    This matches :func:`openscvx.plotting.viser.scp.extract_propagation_positions`
    and the realtime ``extract_multishoot_trajectory`` helpers — **not** nodal
    linear interpolation of ``results.nodes``.
    """
    V_multi_shoot = np.asarray(V_multi_shoot, dtype=np.float64)
    if V_multi_shoot.size == 0:
        return None

    segment_size = n_x + n_x * n_x + 2 * n_x * n_u
    n_rows, n_sub = V_multi_shoot.shape
    if segment_size <= 0 or n_rows % segment_size != 0 or n_sub < 1:
        return None
    n_seg = n_rows // segment_size

    t_nodes = np.asarray(t_nodes, dtype=np.float64).ravel()
    if t_nodes.size != n_seg + 1:
        if t_nodes.size < 2:
            return None
        t_nodes = np.linspace(float(t_nodes[0]), float(t_nodes[-1]), n_seg + 1)

    q_rows: list[np.ndarray] = []
    qd_rows: list[np.ndarray] = []
    t_rows: list[float] = []
    for seg_idx in range(n_seg):
        seg_start = seg_idx * segment_size
        t0, t1 = float(t_nodes[seg_idx]), float(t_nodes[seg_idx + 1])
        j0 = 0 if seg_idx == 0 else 1
        for t_idx in range(j0, n_sub):
            alpha = t_idx / (n_sub - 1) if n_sub > 1 else 0.0
            state = np.asarray(
                V_multi_shoot[seg_start : seg_start + n_x, t_idx], dtype=np.float64
            ).ravel()
            q_rows.append(state[:n_q])
            qd_rows.append(state[n_q:n_x])
            t_rows.append((1.0 - alpha) * t0 + alpha * t1)
    if not q_rows:
        return None

    t_ms = np.asarray(t_rows, dtype=np.float64)
    q_ms = np.stack(q_rows, axis=0)
    qd_ms = np.stack(qd_rows, axis=0)
    return q_ms, qd_ms, t_ms, t_nodes


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
    """Animate the 3D triple-link cartpole using the SCP multi-shoot matrix ``V``.

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

    # SCP state / control dimensions (must match the packed ``V`` layout).
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

    _dh = getattr(results, "discretization_history", None) or []
    V_multishot = np.asarray(_dh[-1], dtype=np.float64) if len(_dh) > 0 else None
    ms_traj = (
        extract_multishoot_qpos_chronological(
            V_multishot,
            n_x=n_x,
            n_u=n_u,
            n_q=n_q,
            t_nodes=t_nodes,
        )
        if V_multishot is not None
        else None
    )

    using_multishot = ms_traj is not None
    if using_multishot:
        q_angle, _, t_play, t_nodes_ms = ms_traj
        u_play = _foh_controls_at_times(t_play, u_nodes, t_nodes_ms)
        fk_multishot_anim = [fk_joints(q_angle[i]) for i in range(len(q_angle))]
        tip_pos = np.array([fk[4] for fk in fk_multishot_anim], dtype=np.float64)
        if len(t_play) > 1:
            tip_vel = np.gradient(tip_pos, t_play, axis=0)
        else:
            tip_vel = np.zeros_like(tip_pos)
        n_seg = V_multishot.shape[0] // (n_x + n_x * n_x + 2 * n_x * n_u)
        print(
            f"[viser] Multi-shoot V: {len(t_play)} propagated samples "
            f"({V_multishot.shape[1]} cols × {n_seg} segments)."
        )
    else:
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
        [
            [float(fk_multishot_anim[i][0][0]), float(fk_multishot_anim[i][0][1]), 0.0]
            for i in range(len(fk_multishot_anim))
        ],
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

    # ── Multishot propagation paths (dense samples from ``V``, not nodal chords) ─
    cart_multishot_segs = None
    tip_multishot_segs = None
    if using_multishot and V_multishot is not None:
        # Faint cloud of every integrated state (realtime-style unpack).
        all_qpos, _ = extract_multishoot_trajectory(
            V_multishot,
            n_x,
            n_u,
            position_slice=slice(0, n_q),
            velocity_slice=None,
        )
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

        # Per-segment nonlinear propagation (each segment = one integrate call).
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
                np.stack(
                    [
                        [float(fk_multishot_anim[i][0][0]), float(fk_multishot_anim[i][0][1]), 0.0],
                        [
                            float(fk_multishot_anim[i + 1][0][0]),
                            float(fk_multishot_anim[i + 1][0][1]),
                            0.0,
                        ],
                    ],
                    axis=0,
                )
                for i in range(len(fk_multishot_anim) - 1)
            ],
            axis=0,
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
                x=t_play.tolist(),
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
    ctrl_label = "multi-shoot FOH" if using_multishot else "post-process"
    fig_ctrl.add_trace(
        go.Scatter(
            x=t_play.tolist(),
            y=u_play[:, 0].tolist(),
            mode="lines",
            name=f"F_x ({ctrl_label})",
            line={"color": "crimson", "width": 2},
        )
    )
    fig_ctrl.add_trace(
        go.Scatter(
            x=t_play.tolist(),
            y=u_play[:, 1].tolist(),
            mode="lines",
            name=f"F_y ({ctrl_label})",
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
        _, update_angles = add_animated_plotly_vline(server, fig_angles, t_play, folder_name=None)
        _, update_ctrl = add_animated_plotly_vline(server, fig_ctrl, t_play, folder_name=None)
        # cart-XY plot is static (no time slider needed)
        server.gui.add_plotly(fig_cart)

    def update_multishot_scene(frame_idx: int) -> None:
        ms_i = int(np.clip(frame_idx, 0, len(fk_multishot_anim) - 1))
        cart, h1, h2, h3, _ = fk_multishot_anim[ms_i]
        ms_cart_handle.position = (float(cart[0]), float(cart[1]), 0.0)
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
                dimensions=(0.42, 0.42, 0.16),
                position=(float(cart[0]), float(cart[1]), 0.0),
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

    # Cart path on the motion plane (matches animated ``/multishot/cart_path``).
    snapshot_server.scene.add_grid(
        "/snapshots/plane",
        width=10.0,
        height=10.0,
        cell_size=0.5,
        position=(0.0, 0.0, -0.105),
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
