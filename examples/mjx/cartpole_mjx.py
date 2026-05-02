"""Cartpole swing-up using MuJoCo MJX dynamics.

This example demonstrates how to plug a MuJoCo MJX model directly into an
OpenSCvx problem via the BYOF dynamics interface. The cart slides along the
x-axis and a passive pole hangs from a hinge; the optimization computes a
control sequence that swings the pole from downward (theta = pi) to upright
(theta = 0).

Requires:
    pip install openscvx[mjx]

The MJX model is defined inline as an XML string so the example is fully
self-contained. For larger models, ``mujoco.MjModel.from_xml_path`` works
exactly the same way.
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
from openscvx import ByofSpec, Minimize, Problem
from openscvx.integrations import mjx_byof

CARTPOLE_XML = """
<mujoco model="cartpole">
  <option gravity="0 0 -9.81" timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-3 3"/>
      <geom name="cart_geom" type="box" size="0.2 0.15 0.1" mass="1.0" rgba="0.4 0.4 0.8 1"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" limited="false"/>
        <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 0.6"
              size="0.04" mass="0.1" rgba="0.8 0.4 0.4 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" gear="20" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

mj_model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
# MJX's contact solver uses lax.while_loop which is not reverse-mode
# differentiable. Disabling contact keeps the dynamics fully JAX-differentiable.
mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
mjx_model = mjx.put_model(mj_model)

n_q = int(mjx_model.nq)
n_v = int(mjx_model.nv)
n_u = int(mjx_model.nu)

n = 60
total_time = 3.0

qpos = ox.State("qpos", shape=(n_q,))
qpos.min = np.array([-3.0, -2.0 * np.pi])
qpos.max = np.array([3.0, 2.0 * np.pi])
qpos.initial = np.array([0.0, np.pi])
qpos.final = np.array([0.0, 0.0])

qvel = ox.State("qvel", shape=(n_v,))
qvel.min = np.array([-10.0, -15.0])
qvel.max = np.array([10.0, 15.0])
qvel.initial = np.array([0.0, 0.0])
qvel.final = np.array([0.0, 0.0])

ctrl = ox.Control("ctrl", shape=(n_u,))
ctrl.min = np.array([-1.0])
ctrl.max = np.array([1.0])
ctrl.guess = np.zeros((n, n_u))

states = [qpos, qvel]
controls = [ctrl]

dynamics = {
    "qpos": qvel,
}

byof: ByofSpec = {"dynamics": mjx_byof(mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)}

constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])
for control in controls:
    constraints.extend([ox.ctcs(control <= control.max), ox.ctcs(control.min <= control)])

theta_guess = np.linspace(np.pi, 0.0, n)
qpos.guess = np.column_stack([np.zeros(n), theta_guess])
qvel.guess = np.zeros((n, n_v))

time = ox.Time(
    initial=0.0,
    final=Minimize(total_time),
    min=0.0,
    max=2.0 * total_time,
    time_dilation_min=0.05 * total_time,
    time_dilation_max=2.0 * total_time,
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
        "lam_cost": 1e-2,
        "lam_vc": 1e0,
        "autotuner": ox.ConstantProximalWeight(),
    },
    float_dtype="float64",
)

POLE_LENGTH = 0.6  # matches <geom fromto="0 0 0 0 0 0.6"> in XML


def visualize(results) -> None:
    """Animate the optimized cartpole trajectory in a Viser 3D scene.

    The scene uses the MuJoCo XML convention:
      - Cart slides along the x-axis at z = 0.
      - Pole hangs from the cart pivot; hinge rotates around y.
      - theta = 0  → pole upright (tip at +z).
      - theta = pi → pole hanging (tip at -z).
    """
    import plotly.graph_objects as go
    import viser

    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        compute_velocity_colors,
    )
    from openscvx.plotting.viser.plotly_integration import add_animated_plotly_vline

    # ── Pull data from results.trajectory ─────────────────────────────────────
    t_vec   = results.trajectory["time"].flatten()          # (N_fine,)
    q_traj  = results.trajectory["qpos"]                   # (N_fine, 2)
    qd_traj = results.trajectory["qvel"]                   # (N_fine, 2)
    u_traj  = results.trajectory["ctrl"]                   # (N_fine, 1)

    cart_x = q_traj[:, 0]
    theta  = q_traj[:, 1]

    # Pole tip in world frame (hinge rotates around y-axis)
    tip_x = cart_x + POLE_LENGTH * np.sin(theta)
    tip_z = POLE_LENGTH * np.cos(theta)

    # 3D positions for trail (tip of the pole)
    tip_pos = np.column_stack([tip_x, np.zeros_like(tip_x), tip_z])

    # ── Viser server ───────────────────────────────────────────────────────────
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")

    # Static rail along x
    rail_pts = np.array([[-3.5, 0, 0], [3.5, 0, 0]], dtype=np.float32)
    server.scene.add_line_segments(
        "/rail",
        points=rail_pts[None],  # shape (1, 2, 3)
        colors=np.array([80, 80, 80], dtype=np.uint8),  # (3,) broadcast to all segments
        line_width=4.0,
    )

    # ── Animated cart (box) ────────────────────────────────────────────────────
    cart_handle = server.scene.add_box(
        "/cart",
        dimensions=(0.4, 0.3, 0.2),
        position=(float(cart_x[0]), 0.0, 0.0),
        color=(80, 100, 200),
    )

    # ── Animated pole (line segment from pivot to tip) ─────────────────────────
    pole_pts = np.array(
        [[[float(cart_x[0]), 0.0, 0.0], [float(tip_x[0]), 0.0, float(tip_z[0])]]],
        dtype=np.float32,
    )
    pole_handle = server.scene.add_line_segments(
        "/pole",
        points=pole_pts,
        colors=np.array([200, 80, 80], dtype=np.uint8),  # (3,) broadcast to all segments
        line_width=6.0,
    )

    # ── Animated trail of the pole tip ────────────────────────────────────────
    tip_colors = compute_velocity_colors(tip_pos)
    _, update_trail = add_animated_trail(server, tip_pos, tip_colors, point_size=0.02)

    # ── Sidebar plots (phase portrait + control) ───────────────────────────────
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(
        x=np.rad2deg(theta).tolist(),
        y=qd_traj[:, 1].tolist(),
        mode="lines",
        line={"color": "royalblue", "width": 2},
        name="Phase",
    ))
    fig_phase.update_layout(
        title="Phase portrait (θ vs θ̇)",
        xaxis_title="θ (deg)",
        yaxis_title="θ̇ (rad/s)",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    fig_ctrl = go.Figure()
    fig_ctrl.add_trace(go.Scatter(
        x=t_vec.tolist(),
        y=u_traj[:, 0].tolist(),
        mode="lines",
        line={"color": "darkorange", "width": 2},
        name="Control",
    ))
    fig_ctrl.update_layout(
        title="Control force (normalized)",
        xaxis_title="Time (s)",
        yaxis_title="u",
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
    )

    with server.gui.add_folder("Plots"):
        _, update_phase = add_animated_plotly_vline(
            server, fig_phase, t_vec, folder_name=None
        )
        _, update_ctrl = add_animated_plotly_vline(
            server, fig_ctrl, t_vec, folder_name=None
        )

    # ── Per-frame cart + pole update ───────────────────────────────────────────
    def update_cartpole(frame_idx: int) -> None:
        cx  = float(cart_x[frame_idx])
        tx  = float(tip_x[frame_idx])
        tz  = float(tip_z[frame_idx])
        cart_handle.position = (cx, 0.0, 0.0)
        pole_handle.points = np.array(
            [[[cx, 0.0, 0.0], [tx, 0.0, tz]]], dtype=np.float32
        )

    # ── Animation controls (play / pause / scrub) ──────────────────────────────
    add_animation_controls(
        server,
        t_vec,
        [update_cartpole, update_trail, update_phase, update_ctrl],
    )

    print("Viser running — open http://localhost:8080 in your browser.")
    server.sleep_forever()


if __name__ == "__main__":
    print("Cartpole swing-up via MuJoCo MJX dynamics")
    print("=" * 60)
    print(f"nq = {n_q}, nv = {n_v}, nu = {n_u}, N = {n}")
    print()

    problem.initialize()
    problem.solve()
    results = problem.post_process()

    final_theta = float(results.nodes["qpos"][-1, 1])
    final_thetadot = float(results.nodes["qvel"][-1, 1])
    print()
    print(f"Final pole angle:      {np.rad2deg(final_theta):.2f} deg  (target: 0.0)")
    print(f"Final pole angle rate: {final_thetadot:.4f} rad/s  (target: 0.0)")
    print()
    visualize(results)
