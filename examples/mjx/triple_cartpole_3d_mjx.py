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
from openscvx import ByofSpec, Problem
from openscvx.integrations import mjx_byof

L1, L2, L3 = 0.5, 0.4, 0.3   # link lengths (m)

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

n_q = int(mjx_model.nq)   # 8: cart_xy + 3 × (αᵢ, βᵢ)
n_v = int(mjx_model.nv)   # 8 (nq == nv)
n_u = int(mjx_model.nu)   # 2: F_x, F_y

n          = 600
total_time = 4.0

# Convenience indices for clarity.
IDX_CART_X, IDX_CART_Y = 0, 1
IDX_A1, IDX_B1 = 2, 3
IDX_A2, IDX_B2 = 4, 5
IDX_A3, IDX_B3 = 6, 7

# ── State / control definitions ───────────────────────────────────────────────
qpos = ox.State("qpos", shape=(n_q,))
qpos.min = np.array([-8.0, -8.0,
                     -2 * np.pi, -2 * np.pi,
                     -2 * np.pi, -2 * np.pi,
                     -2 * np.pi, -2 * np.pi])
qpos.max = -qpos.min
# Hanging: cart at origin, link 1 hanging (α₁ = π), all others 0.
qpos.initial = np.array([-0.5, 1.0,
                         np.pi, np.pi/64,
                         0.0,   0.0,
                         0.0,   0.0])
# Upright: cart free, all angles 0.
qpos.final = [
    ox.Free(0.0), ox.Free(0.0),                       # cart x, y free
    0.0, 0.0,                                         # link 1 upright
    0.0, 0.0,                                         # link 2 upright
    0.0, 0.0,                                         # link 3 upright
]

qvel = ox.State("qvel", shape=(n_v,))
qvel.min = np.full(n_v, -12.0)
qvel.max = np.full(n_v,  12.0)
qvel.initial = np.zeros(n_v)
qvel.final   = [ox.Free(0.0)] * n_v

ctrl = ox.Control("ctrl", shape=(n_u,), parameterization="ZOH")
ctrl.min   = np.array([-2.0, -2.0])
ctrl.max   = np.array([ 2.0,  2.0])
ctrl.guess = np.zeros((n, n_u))

states   = [qpos, qvel]
controls = [ctrl]

# ── Dynamics: position kinematics symbolically, velocity via MJX ──────────────
dynamics: dict = {"qpos": qvel}                          # nq == nv

byof: ByofSpec = {"dynamics": mjx_byof(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)}

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
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    byof=byof,
    algorithm={
        "lam_prox": 1e-1,
        "lam_cost": 0e0,
        "lam_vc":   4e0,
        "autotuner": ox.ConstantProximalWeight(),
    },
    discretizer={"diffrax_kwargs": {"atol": 1e-12, "rtol": 1e-12}, "ode_solver": "Dopri8"},
    solver={"solver_args": {"enforce_dpp": True, "canon_backend": "COO", "abs_tol": 1e-12, "rel_tol": 1e-12}},
    float_dtype="float64",
)


# ── Forward kinematics helpers ────────────────────────────────────────────────
def _R_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)


def _R_y(b: float) -> np.ndarray:
    c, s = np.cos(b), np.sin(b)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


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
    h1 = cart                                            # hinge 1 at cart centre

    R = R @ _R_x(float(q[IDX_A1])) @ _R_y(float(q[IDX_B1]))
    h2 = h1 + L1 * (R @ z_hat)

    R = R @ _R_x(float(q[IDX_A2])) @ _R_y(float(q[IDX_B2]))
    h3 = h2 + L2 * (R @ z_hat)

    R = R @ _R_x(float(q[IDX_A3])) @ _R_y(float(q[IDX_B3]))
    tip = h3 + L3 * (R @ z_hat)

    return cart, h1, h2, h3, tip


def simulate_mujoco(results) -> dict:
    """Run the optimised control sequence through MuJoCo's CPU simulator.

    Mirrors ``simulate_nonlinear_time``: dt=0.01 s (100 Hz), FOH on
    controls, actual node times read from ``results.nodes['time']`` when
    available.  Returns a dict with keys ``"time"``, ``"qpos"``, ``"qvel"``.
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
    if t_nodes_raw is not None:
        t_nodes_sim = np.asarray(t_nodes_raw).flatten()
    else:
        t_nodes_sim = np.linspace(t_start, t_end, n_nodes)

    n_steps = int(round((t_end - t_start) / dt)) + 1
    rec_t  = np.empty(n_steps)
    rec_q  = np.empty((n_steps, n_q))
    rec_qd = np.empty((n_steps, n_v))

    sim_t = t_start
    for step in range(n_steps):
        rec_t[step]  = sim_t
        rec_q[step]  = data.qpos.copy()
        rec_qd[step] = data.qvel.copy()

        k = int(np.clip(np.searchsorted(t_nodes_sim, sim_t, side="right") - 1,
                        0, n_nodes - 2))
        t0 = float(t_nodes_sim[k])
        t1 = float(t_nodes_sim[k + 1])
        alpha = float(np.clip((sim_t - t0) / (t1 - t0) if t1 > t0 else 0.0, 0.0, 1.0))
        data.ctrl[:] = (1.0 - alpha) * u_nodes[k] + alpha * u_nodes[k + 1]

        mujoco.mj_step(mj_model, data)
        sim_t += dt

    print(
        f"MuJoCo simulation: {n_steps} steps, dt={dt*1e3:.1f} ms (FOH, 100 Hz)"
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


def visualize(results, sim: dict | None = None) -> None:
    """Animate the 3D triple-link cartpole in a Viser scene."""
    import plotly.graph_objects as go
    import viser

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        compute_velocity_colors,
    )
    from openscvx.plotting.viser.plotly_integration import add_animated_plotly_vline

    # ── Extract trajectory data ────────────────────────────────────────────────
    t_vec  = results.trajectory["time"].flatten()
    q_traj = results.trajectory["qpos"]
    u_traj = results.trajectory["ctrl"]

    q_nodes = results.nodes["qpos"]
    t_nodes = results.nodes.get("time", None)
    if t_nodes is None:
        t_nodes = np.linspace(float(t_vec[0]), float(t_vec[-1]), len(q_nodes))
    else:
        t_nodes = np.asarray(t_nodes).flatten()

    N = len(t_vec)
    fk_all   = [fk_joints(q_traj[i]) for i in range(N)]
    tip_pos  = np.array([j[4] for j in fk_all])

    fk_nodes = [fk_joints(q_nodes[i]) for i in range(len(q_nodes))]

    # Multishot path from V (last SCP discretization matrix)
    _dh = getattr(results, "discretization_history", None) or []
    V_multishot = _dh[-1] if len(_dh) > 0 else None
    q_ms_v, t_ms_v = (
        qpos_from_V_multishot(
            np.asarray(V_multishot, dtype=np.float64),
            n_q=n_q, n_v=n_v, n_u=n_u, t_nodes=t_nodes,
        )
        if V_multishot is not None
        else (None, None)
    )

    if q_ms_v is not None and t_ms_v is not None:
        fk_multishot_anim  = [fk_joints(q_ms_v[i]) for i in range(len(q_ms_v))]
        t_multishot_lookup = t_ms_v
    else:
        q_aligned = np.column_stack(
            [np.interp(t_vec, t_nodes, q_nodes[:, j]) for j in range(q_nodes.shape[1])]
        )
        fk_multishot_anim  = [fk_joints(q_aligned[i]) for i in range(N)]
        t_multishot_lookup = t_vec

    # ── Viser server ───────────────────────────────────────────────────────────
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")

    # Cart-motion plane (a faint horizontal grid + outline box).
    server.scene.add_grid(
        "/plane", width=10.0, height=10.0, cell_size=0.5,
        position=(0.0, 0.0, -0.105),
    )
    plane_outline = np.array([[
        [-4.0, -4.0, 0.0], [ 4.0, -4.0, 0.0],
    ], [
        [ 4.0, -4.0, 0.0], [ 4.0,  4.0, 0.0],
    ], [
        [ 4.0,  4.0, 0.0], [-4.0,  4.0, 0.0],
    ], [
        [-4.0,  4.0, 0.0], [-4.0, -4.0, 0.0],
    ]], dtype=np.float32)
    server.scene.add_line_segments(
        "/plane/outline", points=plane_outline,
        colors=np.array([110, 110, 110], dtype=np.uint8),
        line_width=2.0,
    )

    # Upright target marker
    upright_tip = np.array([0.0, 0.0, L1 + L2 + L3])
    server.scene.add_icosphere(
        "/target", radius=0.05, color=(50, 220, 100),
        position=tuple(float(v) for v in upright_tip),
    )

    # ── Static discretization-node ghost rig ──────────────────────────────────
    n_nodes = len(fk_nodes)
    ghost_segs = np.stack([
        np.stack([np.array(fk_nodes[i][j],     dtype=np.float32),
                  np.array(fk_nodes[i][j + 1], dtype=np.float32)], axis=0)
        for i in range(n_nodes)
        for j in range(1, 4)
    ], axis=0)
    server.scene.add_line_segments(
        "/nodes/links", points=ghost_segs,
        colors=np.array([160, 160, 160], dtype=np.uint8),
        line_width=1.5,
    )
    cart_node_pos = np.array(
        [[float(fk_nodes[i][0][0]), float(fk_nodes[i][0][1]), 0.0]
         for i in range(n_nodes)], dtype=np.float32,
    )
    server.scene.add_point_cloud(
        "/nodes/cart", points=cart_node_pos,
        colors=np.tile(np.array([100, 100, 220], dtype=np.uint8), (n_nodes, 1)),
        point_size=0.04,
    )
    tip_node_pos = np.array(
        [fk_nodes[i][4] for i in range(n_nodes)], dtype=np.float32
    )
    server.scene.add_point_cloud(
        "/nodes/tips", points=tip_node_pos,
        colors=np.tile(np.array([220, 180, 50], dtype=np.uint8), (n_nodes, 1)),
        point_size=0.05,
    )

    # ── Multishot polylines (cart path on plane + tip path in 3D) ────────────
    if q_ms_v is not None and len(q_ms_v) >= 2:
        fk_ms_poly = [fk_joints(q_ms_v[i]) for i in range(len(q_ms_v))]
        cart_ms = np.array(
            [[float(fk_ms_poly[i][0][0]), float(fk_ms_poly[i][0][1]), 0.0]
             for i in range(len(fk_ms_poly))], dtype=np.float32)
        tip_ms  = np.array([fk_ms_poly[i][4] for i in range(len(fk_ms_poly))],
                            dtype=np.float32)
        cart_multishot_segs = np.stack(
            [np.stack([cart_ms[i], cart_ms[i + 1]], axis=0)
             for i in range(len(cart_ms) - 1)], axis=0)
        tip_multishot_segs = np.stack(
            [np.stack([tip_ms[i], tip_ms[i + 1]], axis=0)
             for i in range(len(tip_ms) - 1)], axis=0)
    elif n_nodes >= 2:
        cart_multishot_segs = np.stack(
            [np.stack([cart_node_pos[i], cart_node_pos[i + 1]], axis=0)
             for i in range(n_nodes - 1)], axis=0).astype(np.float32)
        tip_multishot_segs = np.stack(
            [np.stack([tip_node_pos[i], tip_node_pos[i + 1]], axis=0)
             for i in range(n_nodes - 1)], axis=0).astype(np.float32)
    else:
        cart_multishot_segs = None

    if cart_multishot_segs is not None:
        server.scene.add_line_segments(
            "/multishot/cart_path", points=cart_multishot_segs,
            colors=np.array([90, 90, 230], dtype=np.uint8), line_width=3.5)
        server.scene.add_line_segments(
            "/multishot/tip_path", points=tip_multishot_segs,
            colors=np.array([230, 170, 40], dtype=np.uint8), line_width=3.5)

    # ── Animated cart ──────────────────────────────────────────────────────────
    cart_handle = server.scene.add_box(
        "/cart", dimensions=(0.5, 0.5, 0.2),
        position=tuple(float(v) for v in fk_all[0][0]),
        color=(90, 90, 190),
    )

    # ── Animated links ────────────────────────────────────────────────────────
    def _link_segments(i: int) -> np.ndarray:
        _, h1, h2, h3, tip = fk_all[i]
        return np.array([[h1, h2], [h2, h3], [h3, tip]], dtype=np.float32)

    link_colors = np.array([
        [[220, 80,  80],  [220, 80,  80]],
        [[80,  200, 80],  [80,  200, 80]],
        [[80,  80,  220], [80,  80,  220]],
    ], dtype=np.uint8)
    link_handle = server.scene.add_line_segments(
        "/links", points=_link_segments(0),
        colors=link_colors, line_width=7.0,
    )

    joint_handles = []
    for jname, jcol in [("/j1", (200, 60, 60)),
                         ("/j2", (60, 200, 60)),
                         ("/j3", (60, 60, 200))]:
        joint_handles.append(
            server.scene.add_icosphere(
                jname, radius=0.04, color=jcol,
                position=tuple(float(v) for v in fk_all[0][1]),
            )
        )

    # ── Multishot animated rig ─────────────────────────────────────────────────
    ms_cart_handle = server.scene.add_box(
        "/multishot/cart", dimensions=(0.42, 0.42, 0.16),
        position=tuple(float(v) for v in fk_multishot_anim[0][0]),
        color=(55, 150, 255),
    )

    def _multishot_link_segments(i: int) -> np.ndarray:
        _, h1, h2, h3, tip = fk_multishot_anim[i]
        return np.array([[h1, h2], [h2, h3], [h3, tip]], dtype=np.float32)

    ms_link_colors = np.array([
        [[120, 190, 255], [120, 190, 255]],
        [[95,  175, 245], [95,  175, 245]],
        [[70,  160, 235], [70,  160, 235]],
    ], dtype=np.uint8)
    ms_link_handle = server.scene.add_line_segments(
        "/multishot/links", points=_multishot_link_segments(0),
        colors=ms_link_colors, line_width=5.0,
    )

    ms_joint_handles = []
    for jname in ["/multishot/j1", "/multishot/j2", "/multishot/j3"]:
        ms_joint_handles.append(
            server.scene.add_icosphere(
                jname, radius=0.03, color=(80, 170, 245),
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
        fig_angles.add_trace(go.Scatter(
            x=t_vec.tolist(),
            y=np.rad2deg(q_traj[:, idx]).tolist(),
            mode="lines", name=name,
            line={"color": col, "width": 2},
        ))
    fig_angles.add_hline(y=0, line_dash="dash", line_color="gray",
                          annotation_text="Upright")
    fig_angles.update_layout(
        title="Joint angles",
        xaxis_title="Time (s)", yaxis_title="Angle (deg)",
        legend={"orientation": "h"},
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    # ── Sidebar: cart position (XY plot) and control forces (2 traces) ───────
    fig_cart = go.Figure()
    fig_cart.add_trace(go.Scatter(
        x=q_traj[:, IDX_CART_X].tolist(),
        y=q_traj[:, IDX_CART_Y].tolist(),
        mode="lines+markers", name="Cart trajectory",
        line={"color": "royalblue", "width": 2},
        marker={"size": 3, "color": "royalblue"},
    ))
    fig_cart.update_layout(
        title="Cart path on plane",
        xaxis_title="x (m)", yaxis_title="y (m)",
        yaxis={"scaleanchor": "x", "scaleratio": 1.0},
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    fig_ctrl = go.Figure()
    fig_ctrl.add_trace(go.Scatter(
        x=t_vec.tolist(), y=u_traj[:, 0].tolist(),
        mode="lines", name="F_x", line={"color": "crimson", "width": 2},
    ))
    fig_ctrl.add_trace(go.Scatter(
        x=t_vec.tolist(), y=u_traj[:, 1].tolist(),
        mode="lines", name="F_y", line={"color": "darkmagenta", "width": 2},
    ))
    fig_ctrl.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_ctrl.update_layout(
        title="Cart controls (normalised)",
        xaxis_title="Time (s)", yaxis_title="u",
        legend={"orientation": "h"},
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    with server.gui.add_folder("Plots"):
        _, update_angles = add_animated_plotly_vline(server, fig_angles, t_vec, folder_name=None)
        _, update_ctrl   = add_animated_plotly_vline(server, fig_ctrl,   t_vec, folder_name=None)
        # cart-XY plot is static (no time slider needed)
        server.gui.add_plotly(fig_cart)

    # ── MuJoCo simulation overlay (orange chain) ──────────────────────────────
    sim_callbacks: list = []
    if sim is not None:
        sim_t  = sim["time"]
        sim_q  = sim["qpos"]
        fk_sim = [fk_joints(sim_q[i]) for i in range(len(sim_t))]

        sim_tip = np.array([j[4] for j in fk_sim], dtype=np.float32)
        sim_tip_colors = np.tile(
            np.array([230, 120, 30], dtype=np.uint8), (len(sim_t), 1)
        )
        server.scene.add_point_cloud(
            "/sim/tip_trail", points=sim_tip, colors=sim_tip_colors,
            point_size=0.01,
        )

        sim_link_colors = np.array([
            [[230, 100, 20], [230, 100, 20]],
            [[230, 140, 40], [230, 140, 40]],
            [[230, 180, 60], [230, 180, 60]],
        ], dtype=np.uint8)

        def _sim_link_segments(i: int) -> np.ndarray:
            _, h1, h2, h3, tip = fk_sim[i]
            return np.array([[h1, h2], [h2, h3], [h3, tip]], dtype=np.float32)

        sim_cart_handle = server.scene.add_box(
            "/sim/cart", dimensions=(0.5, 0.5, 0.2),
            position=tuple(float(v) for v in fk_sim[0][0]),
            color=(200, 80, 20),
        )
        sim_link_handle = server.scene.add_line_segments(
            "/sim/links", points=_sim_link_segments(0),
            colors=sim_link_colors, line_width=4.0,
        )
        sim_joint_handles = []
        for jname, jcol in [("/sim/j1", (200, 80, 20)),
                              ("/sim/j2", (210, 110, 30)),
                              ("/sim/j3", (220, 150, 50))]:
            sim_joint_handles.append(
                server.scene.add_icosphere(
                    jname, radius=0.035, color=jcol,
                    position=tuple(float(v) for v in fk_sim[0][1]),
                )
            )

        def update_sim(frame_idx: int) -> None:
            t_cur = float(t_vec[frame_idx])
            si = int(np.clip(np.searchsorted(sim_t, t_cur) - 1, 0, len(sim_t) - 1))
            cart, h1, h2, h3, _ = fk_sim[si]
            sim_cart_handle.position = (float(cart[0]), float(cart[1]), 0.0)
            sim_link_handle.points   = _sim_link_segments(si)
            for handle, pos in zip(sim_joint_handles, (h1, h2, h3)):
                handle.position = tuple(float(v) for v in pos)

        sim_callbacks.append(update_sim)

    # ── Per-frame update ───────────────────────────────────────────────────────
    def update_scene(frame_idx: int) -> None:
        cart, h1, h2, h3, _ = fk_all[frame_idx]
        cart_handle.position = (float(cart[0]), float(cart[1]), 0.0)
        link_handle.points   = _link_segments(frame_idx)
        for handle, pos in zip(joint_handles, (h1, h2, h3)):
            handle.position = tuple(float(v) for v in pos)

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
        server, t_vec,
        [update_scene, update_multishot_scene, update_trail,
         update_angles, update_ctrl, *sim_callbacks],
    )

    print("Viser running — open http://localhost:8080 in your browser.")
    server.sleep_forever()


if __name__ == "__main__":
    print("3D Triple-link cartpole swing-up — MuJoCo MJX + OpenSCvx")
    print("=" * 60)
    print(f"nq={n_q}, nv={n_v}, nu={n_u}, N={n}")
    print(f"Links: L1={L1} m, L2={L2} m, L3={L3} m  (total {L1+L2+L3} m)")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    final_q  = results.nodes["qpos"][-1]
    print()
    print(f"Final cart position:    ({final_q[IDX_CART_X]:.4f}, {final_q[IDX_CART_Y]:.4f}) m")
    print(f"Final α₁,β₁ [deg]:      ({np.rad2deg(final_q[IDX_A1]):.2f}, {np.rad2deg(final_q[IDX_B1]):.2f})")
    print(f"Final α₂,β₂ [deg]:      ({np.rad2deg(final_q[IDX_A2]):.2f}, {np.rad2deg(final_q[IDX_B2]):.2f})")
    print(f"Final α₃,β₃ [deg]:      ({np.rad2deg(final_q[IDX_A3]):.2f}, {np.rad2deg(final_q[IDX_B3]):.2f})")

    print()
    print("Running MuJoCo CPU simulation with solved controls…")
    sim = simulate_mujoco(results)

    from openscvx.plotting import plot_states, plot_controls
    plot_states(results).show()
    plot_controls(results).show()

    visualize(results, sim=sim)
