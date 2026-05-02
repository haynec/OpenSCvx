"""Skydio X2 quadrotor racing through sequential gates with MuJoCo MJX dynamics.

This mirrors the scenario structure in `examples/drone/drone_racing.py`:
- sequential gate traversal constraints at prescribed nodes
- minimum-time objective
- loop closure in position (start equals end)

Model loading (in priority order)
----------------------------------
1. **MuJoCo Menagerie submodule** — if ``third_party/mujoco_menagerie`` is
   present (``git submodule update --init third_party/mujoco_menagerie``),
   the official Skydio X2 model is loaded from there, complete with the
   low-poly mesh and texture used for Viser visualisation.
2. **Inline XML fallback** — if the submodule is absent the example uses a
   self-contained inline XML that matches the menagerie physics (same rotor
   positions, gear vectors, masses, hover thrust) but replaces mesh assets
   with primitive geoms so the example works with no extra files.

Visualisation
-------------
When the menagerie mesh is available and ``trimesh`` is installed the Viser
scene shows the actual Skydio X2 low-poly mesh.  Otherwise it falls back to
a box-and-disc primitive representation.

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
from openscvx.utils import rot

# ── Inline XML fallback ───────────────────────────────────────────────────────
# Used when the MuJoCo Menagerie submodule is not available.
# Physics matches the official model (same rotor positions, gear vectors,
# masses, and hover thrust) but replaces mesh assets with primitive geoms.
_X2_XML_FALLBACK = """
<mujoco model="skydio_x2">
  <option timestep="0.01" integrator="Euler" gravity="0 0 -9.81"/>
  <default>
    <motor ctrlrange="0 13"/>
  </default>
  <worldbody>
    <body name="x2" pos="0 0 0.3">
      <freejoint name="root"/>
      <geom name="body"   type="box"       size="0.08 0.08 0.04"      mass="0.325" rgba="0.25 0.25 0.25 1"/>
      <geom name="arm_lr" type="box"       size="0.18 0.015 0.008"    mass="0"     rgba="0.4 0.4 0.4 1"/>
      <geom name="arm_fb" type="box"       size="0.015 0.14 0.008"    mass="0"     rgba="0.4 0.4 0.4 1"/>
      <!-- Rotor discs — positions and masses from menagerie -->
      <geom name="rotor1" type="ellipsoid" size="0.13 0.13 0.01" pos="-.14 -.18 .05" mass=".25" rgba="0.15 0.15 0.15 1"/>
      <geom name="rotor2" type="ellipsoid" size="0.13 0.13 0.01" pos="-.14  .18 .05" mass=".25" rgba="0.15 0.15 0.15 1"/>
      <geom name="rotor3" type="ellipsoid" size="0.13 0.13 0.01" pos=" .14  .18 .08" mass=".25" rgba="0.85 0.35 0.1  1"/>
      <geom name="rotor4" type="ellipsoid" size="0.13 0.13 0.01" pos=" .14 -.18 .08" mass=".25" rgba="0.85 0.35 0.1  1"/>
      <!-- Thrust sites at rotor positions -->
      <site name="thrust1" pos="-.14 -.18 .05"/>
      <site name="thrust2" pos="-.14  .18 .05"/>
      <site name="thrust3" pos=" .14  .18 .08"/>
      <site name="thrust4" pos=" .14 -.18 .08"/>
    </body>
  </worldbody>
  <actuator>
    <!-- gear: [Fx,Fy,Fz, Tx,Ty,Tz] in site frame.
         Rotors 1 & 3 spin CW  → negative yaw torque (−0.0201).
         Rotors 2 & 4 spin CCW → positive yaw torque (+0.0201). -->
    <motor name="thrust1" site="thrust1" gear="0 0 1 0 0 -.0201"/>
    <motor name="thrust2" site="thrust2" gear="0 0 1 0 0  .0201"/>
    <motor name="thrust3" site="thrust3" gear="0 0 1 0 0 -.0201"/>
    <motor name="thrust4" site="thrust4" gear="0 0 1 0 0  .0201"/>
  </actuator>
</mujoco>
"""

HOVER_CTRL = 3.2495625   # N per motor for level hover (from menagerie keyframe)
START_POS  = np.array([10.0, 0.0, 20.0])
HOVER_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # w=1 → level attitude

# ── Load MuJoCo model — try menagerie first, fall back to inline XML ──────────
_menagerie_xml_path: "str | None" = None
try:
    from openscvx.integrations.menagerie import get_xml_path
    _menagerie_xml_path = str(get_xml_path("skydio_x2"))
    mj_model = mujoco.MjModel.from_xml_path(_menagerie_xml_path)
    print(f"[skydio_x2] loaded from MuJoCo Menagerie: {_menagerie_xml_path}")
except FileNotFoundError:
    mj_model = mujoco.MjModel.from_xml_string(_X2_XML_FALLBACK)
    print("[skydio_x2] MuJoCo Menagerie not found — using inline XML fallback.")
    print("  To enable mesh rendering, run:")
    print("    git submodule update --init third_party/mujoco_menagerie")

# Disable contact solver: MJX's contact pipeline uses lax.while_loop which is
# not forward-mode differentiable. Quadrotors don't rely on contact dynamics.
mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
mjx_model = mjx.put_model(mj_model)

n_q = int(mjx_model.nq)   # 7 — xyz + quaternion (free joint)
n_v = int(mjx_model.nv)   # 6 — linear + angular velocity
n_u = int(mjx_model.nu)   # 4 — rotor thrusts

n = 22
total_time = 24.0

# ── State / control definitions ───────────────────────────────────────────────
qpos = ox.State("qpos", shape=(n_q,))
qpos.min     = np.array([-200.0, -100.0, 15.0, -1.0, -1.0, -1.0, -1.0])
qpos.max     = np.array([200.0, 100.0, 200.0, 1.0, 1.0, 1.0, 1.0])
qpos.initial = np.concatenate([START_POS, HOVER_QUAT])
qpos.final   = [10.0, 0.0, 20.0, ("free", 1.0), ("free", 0.0), ("free", 0.0), ("free", 0.0)]

qvel = ox.State("qvel", shape=(n_v,))
qvel.min     = np.array([-100.0, -100.0, -100.0, -10.0, -10.0, -10.0])
qvel.max     = np.array([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
qvel.initial = np.zeros(n_v)
qvel.final   = [("free", 0.0), ("free", 0.0), ("free", 0.0), ("free", 0.0), ("free", 0.0), ("free", 0.0)]

ctrl = ox.Control("ctrl", shape=(n_u,))
ctrl.min   = np.zeros(n_u)
ctrl.max   = 13.0 * np.ones(n_u)
ctrl.guess = HOVER_CTRL * np.ones((n, n_u))

states   = [qpos, qvel]
controls = [ctrl]

# ── Dynamics via BYOF ─────────────────────────────────────────────────────────
# The free joint has nq=7 but nv=6 (quaternion adds one extra position DOF).
# nq=7, nv=6 (free joint): mjx_byof detects nq > nv and automatically
# includes quaternion kinematics for "qpos" alongside the MJX "qvel" dynamics.
byof: ByofSpec = {"dynamics": mjx_byof(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)}

# ── Gate parameters (matching examples/drone/drone_racing.py) ───────────────
n_gates = 10
initial_gate_centers = [
    np.array([59.436, 0.000, 20.0000]),
    np.array([92.964, -23.750, 25.5240]),
    np.array([92.964, -29.274, 20.0000]),
    np.array([92.964, -23.750, 20.0000]),
    np.array([130.150, -23.750, 20.0000]),
    np.array([152.400, -73.152, 20.0000]),
    np.array([92.964, -75.080, 20.0000]),
    np.array([92.964, -68.556, 20.0000]),
    np.array([59.436, -81.358, 20.0000]),
    np.array([22.250, -42.672, 20.0000]),
]
radii = np.array([2.5, 1e-4, 2.5])
A_gate_const = rot @ np.diag(1 / radii) @ rot.T

modified_centers = []
for center in initial_gate_centers:
    modified_center = center.copy()
    modified_center[0] = modified_center[0] + 2.5
    modified_center[2] = modified_center[2] + 2.5
    modified_centers.append(modified_center)

nodes_per_gate = 2
gate_nodes = np.arange(nodes_per_gate, n, nodes_per_gate)
gate_centers = np.array(modified_centers)

# ── Constraints ───────────────────────────────────────────────────────────────
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
for control in controls:
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

# Enforce sequential gate traversal using nodal constraints on qpos position.
position = ox.Concat(qpos[0], qpos[1], qpos[2])
for node, gate_center in zip(gate_nodes, gate_centers):
    constraints.append(
        (
            ox.linalg.Norm(
                A_gate_const @ position - A_gate_const @ ox.Constant(gate_center),
                ord="inf",
            )
            <= 1.0
        )
        .convex()
        .at([node])
    )

# ── Initial guess: piecewise-linear through all gates, level attitude ─────────
pos_guess = ox.init.linspace(
    keyframes=[START_POS] + modified_centers + [START_POS],
    nodes=[0] + list(gate_nodes) + [n - 1],
)
quat_guess = np.tile(HOVER_QUAT, (n, 1))
qpos.guess = np.column_stack([pos_guess, quat_guess])
qvel.guess = np.zeros((n, n_v))

time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
)

problem = Problem(
    dynamics={},           # all dynamics go through BYOF
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    byof=byof,
    algorithm={
        "lam_prox": 1e-1,
        "lam_cost": 1e-2,
        "lam_vc":   1e0,
        "autotuner": ox.ConstantProximalWeight(),
    },
    float_dtype="float64",
)

# ── Rotor positions in body frame (for visualization) ─────────────────────────
ROTOR_OFFSETS = np.array([
    [-.14, -.18, .05],
    [-.14,  .18, .05],
    [ .14,  .18, .08],
    [ .14, -.18, .08],
])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q = [qw, qx, qy, qz]."""
    qw, qx, qy, qz = q
    # Rodrigues formula via double cross product
    uv = np.cross([qx, qy, qz], v)
    uuv = np.cross([qx, qy, qz], uv)
    return v + 2.0 * (qw * uv + uuv)


def visualize(results) -> None:
    """Animate the Skydio X2 racing trajectory in a Viser 3D scene."""
    import plotly.graph_objects as go
    import viser

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        compute_velocity_colors,
    )
    from openscvx.plotting.viser.plotly_integration import add_animated_plotly_vline

    # ── Extract trajectory data ────────────────────────────────────────────────
    t_vec  = results.trajectory["time"].flatten()   # (N_fine,)
    q_traj = results.trajectory["qpos"]             # (N_fine, 7)
    u_traj = results.trajectory["ctrl"]             # (N_fine, 4)

    pos  = q_traj[:, :3]                            # (N_fine, 3)
    quat = q_traj[:, 3:]                            # (N_fine, 4) [qw,qx,qy,qz]

    N = len(t_vec)

    # ── Precompute world-frame rotor positions ─────────────────────────────────
    rotor_world = np.zeros((N, 4, 3))
    for i in range(N):
        for j, offset in enumerate(ROTOR_OFFSETS):
            rotor_world[i, j] = pos[i] + quat_rotate(quat[i], offset)

    # ── Viser server ───────────────────────────────────────────────────────────
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/ground",
        width=220.0,
        height=120.0,
        cell_size=5.0,
        position=(80.0, -40.0, 0.0),
    )

    # Gate center markers
    for idx, center in enumerate(gate_centers):
        center_tuple = tuple(float(v) for v in center)
        server.scene.add_icosphere(
            f"/gates/gate_{idx + 1}",
            radius=0.25,
            color=(40, 180, 255),
            position=center_tuple,
        )
        server.scene.add_label(
            f"/gates/gate_{idx + 1}/label",
            text=f"G{idx + 1}",
            position=tuple(float(v) for v in center + np.array([0.0, 0.0, 0.6])),
        )

    # ── Animated drone body ────────────────────────────────────────────────────
    # Try to load the Skydio X2 low-poly mesh from the menagerie for a richer
    # visualisation.  Fall back to primitive-geom representation otherwise.
    mesh_handle = None
    rotor_handles: list = []
    arm_handle = None

    _use_mesh = False
    if _menagerie_xml_path is not None:
        try:
            import trimesh  # type: ignore
            from pathlib import Path
            _asset_dir = Path(_menagerie_xml_path).parent / "assets"
            _obj_path  = _asset_dir / "X2_lowpoly.obj"
            _tm = trimesh.load(_obj_path, force="mesh", process=False)

            # Apply menagerie mesh defaults: scale="0.01 0.01 0.01"
            _tm.apply_scale(0.01)

            # Apply visual geom rotation quat="0 0 1 1" (MuJoCo [w,x,y,z]).
            # Normalised: [0, 0, 1/√2, 1/√2].  Rotation matrix:
            #   R = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
            _R_vis = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
            _tm.vertices = (_tm.vertices @ _R_vis.T).astype(np.float32)

            _verts = np.array(_tm.vertices, dtype=np.float32)
            _faces = np.array(_tm.faces,    dtype=np.uint32)

            mesh_handle = server.scene.add_mesh_simple(
                "/drone/mesh",
                vertices=_verts,
                faces=_faces,
                color=(200, 200, 210),
                position=tuple(float(v) for v in pos[0]),
                wxyz=tuple(float(v) for v in quat[0]),
            )
            _use_mesh = True
            print("[viser] rendering Skydio X2 low-poly mesh from menagerie.")
        except Exception as _mesh_err:
            print(f"[viser] mesh load failed ({_mesh_err}), using primitives.")

    if not _use_mesh:
        # Primitive-geom fallback: box body + rotor discs + arm wires
        server.scene.add_box(
            "/drone/body",
            dimensions=(0.16, 0.16, 0.08),
            position=tuple(float(v) for v in pos[0]),
            wxyz=tuple(float(v) for v in quat[0]),
            color=(60, 60, 60),
        )
        rotor_colors = [(40, 40, 40), (40, 40, 40), (220, 90, 30), (220, 90, 30)]
        for j in range(4):
            h = server.scene.add_icosphere(
                f"/drone/rotor{j+1}",
                radius=0.06,
                color=rotor_colors[j],
                position=tuple(float(v) for v in rotor_world[0, j]),
            )
            rotor_handles.append(h)

        arm_pts_0 = np.array([
            [[float(rotor_world[0, 0, k]) for k in range(3)],
             [float(rotor_world[0, 2, k]) for k in range(3)]],
            [[float(rotor_world[0, 1, k]) for k in range(3)],
             [float(rotor_world[0, 3, k]) for k in range(3)]],
        ], dtype=np.float32)
        arm_handle = server.scene.add_line_segments(
            "/drone/arms",
            points=arm_pts_0,
            colors=np.array([80, 80, 80], dtype=np.uint8),
            line_width=3.0,
        )

    # Attitude frame axes (always visible)
    frame_handle = server.scene.add_frame(
        "/drone/frame",
        axes_length=0.25,
        axes_radius=0.008,
        position=tuple(float(v) for v in pos[0]),
        wxyz=tuple(float(v) for v in quat[0]),
    )

    # ── Animated COM trail ─────────────────────────────────────────────────────
    trail_colors = compute_velocity_colors(pos)
    _, update_trail = add_animated_trail(server, pos, trail_colors, point_size=0.03)

    # ── Sidebar: altitude + rotor thrusts ─────────────────────────────────────
    fig_alt = go.Figure()
    fig_alt.add_trace(go.Scatter(
        x=t_vec.tolist(), y=pos[:, 2].tolist(),
        mode="lines", name="Altitude (m)",
        line={"color": "royalblue", "width": 2},
    ))
    fig_alt.add_hline(
        y=float(START_POS[2]),
        line_dash="dash",
        line_color="gray",
        annotation_text="Track altitude",
    )
    fig_alt.update_layout(
        title="Altitude",
        xaxis_title="Time (s)", yaxis_title="z (m)",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    fig_thrust = go.Figure()
    labels = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]
    colors_u = ["royalblue", "darkorange", "green", "red"]
    for k in range(4):
        fig_thrust.add_trace(go.Scatter(
            x=t_vec.tolist(), y=u_traj[:, k].tolist(),
            mode="lines", name=labels[k],
            line={"color": colors_u[k], "width": 1.5},
        ))
    fig_thrust.add_hline(y=HOVER_CTRL, line_dash="dash",
                         line_color="gray", annotation_text="Hover")
    fig_thrust.update_layout(
        title="Rotor thrusts (N)",
        xaxis_title="Time (s)", yaxis_title="Thrust (N)",
        legend={"orientation": "h"},
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    with server.gui.add_folder("Plots"):
        _, update_alt    = add_animated_plotly_vline(server, fig_alt,    t_vec, folder_name=None)
        _, update_thrust = add_animated_plotly_vline(server, fig_thrust, t_vec, folder_name=None)

    # ── Per-frame drone update ─────────────────────────────────────────────────
    def update_drone(frame_idx: int) -> None:
        p = tuple(float(v) for v in pos[frame_idx])
        q = tuple(float(v) for v in quat[frame_idx])
        frame_handle.position = p
        frame_handle.wxyz     = q
        if _use_mesh and mesh_handle is not None:
            mesh_handle.position = p
            mesh_handle.wxyz     = q
        else:
            for j, h in enumerate(rotor_handles):
                h.position = tuple(float(v) for v in rotor_world[frame_idx, j])
            if arm_handle is not None:
                arm_pts = np.array([
                    [[float(rotor_world[frame_idx, 0, k]) for k in range(3)],
                     [float(rotor_world[frame_idx, 2, k]) for k in range(3)]],
                    [[float(rotor_world[frame_idx, 1, k]) for k in range(3)],
                     [float(rotor_world[frame_idx, 3, k]) for k in range(3)]],
                ], dtype=np.float32)
                arm_handle.points = arm_pts

    # ── Animation controls ─────────────────────────────────────────────────────
    add_animation_controls(
        server,
        t_vec,
        [update_drone, update_trail, update_alt, update_thrust],
    )

    print("Viser running — open http://localhost:8080 in your browser.")
    server.sleep_forever()


if __name__ == "__main__":
    print("Skydio X2 drone racing — MuJoCo MJX + OpenSCvx")
    print("=" * 60)
    print(f"nq={n_q}, nv={n_v}, nu={n_u}, N={n}")
    print(f"Start/Finish: {START_POS}")
    print(f"Gates: {n_gates} sequential constraints")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    final_pos = results.nodes["qpos"][-1, :3]
    final_vel = results.nodes["qvel"][-1]
    pos_err   = np.linalg.norm(final_pos - START_POS)

    print()
    print(f"Final position: {final_pos}")
    print(f"Loop-closure position error: {pos_err:.4f} m")
    print(f"Final velocity:  {np.linalg.norm(final_vel):.4f} m/s")
    print()
    visualize(results)
