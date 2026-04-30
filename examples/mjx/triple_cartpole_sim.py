"""Interactive MuJoCo physics sandbox for the triple-link cartpole.

No trajectory optimisation — pure MuJoCo forward simulation running in
real-time with an interactive Viser 3D interface.

The cart is driven by a **PD position controller** that tracks a draggable
setpoint.  Drag the red X-axis arrow on top of the cart to move the setpoint;
the cart follows under full physics (inertia, gravity, pendulum coupling).

The setpoint itself is rate-limited (default 4 m/s) so it cannot teleport.

Interaction
-----------
- **Drag the red X-arrow** on the cart to steer it along the rail.
- **"Max setpoint speed" slider** — caps how fast the target can move.
- **Kp / Kd sliders** — tune the PD controller gains live.
- **"Reset — hanging"**: restarts with all links pointing straight down.
- **"Reset — upright"**: restarts near the unstable upright equilibrium.

Usage::

    python examples/mjx/triple_cartpole_sim.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

try:
    import mujoco
except ImportError:
    print("MuJoCo is not installed.  Install with: pip install mujoco", file=sys.stderr)
    sys.exit(1)

# ── Model ────────────────────────────────────────────────────────────────────
L1, L2, L3    = 0.5, 0.4, 0.3
RAIL_LIMIT    = 4.0                       # cart joint range ±4 m
RAIL_WIDTH    = 2.0 * RAIL_LIMIT          # 8 m
DEFAULT_RATE  = 0.5 * RAIL_WIDTH          # 4 m/s  (0.5 × rail width / s)
GEAR          = 60.0                      # actuator gear: ctrl ∈ [−1,1] → ±60 N
DEFAULT_KP    = 60.0                      # N / m
DEFAULT_KD    = 15.0                      # N·s / m

_XML = f"""
<mujoco model="triple_cartpole">
  <option gravity="0 0 -9.81" timestep="0.005" integrator="Euler"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0"
             limited="true" range="-{RAIL_LIMIT} {RAIL_LIMIT}"/>
      <geom name="cart_geom" type="box" size="0.25 0.15 0.1"
            mass="2.0" rgba="0.35 0.35 0.75 1"/>
      <body name="link1" pos="0 0 0">
        <joint name="hinge1" type="hinge" axis="0 1 0" limited="false"/>
        <geom name="pole1" type="capsule" fromto="0 0 0 0 0 {L1}"
              size="0.04" mass="0.5" rgba="0.85 0.3 0.3 1"/>
        <body name="link2" pos="0 0 {L1}">
          <joint name="hinge2" type="hinge" axis="0 1 0" limited="false"/>
          <geom name="pole2" type="capsule" fromto="0 0 0 0 0 {L2}"
                size="0.035" mass="0.4" rgba="0.3 0.8 0.3 1"/>
          <body name="link3" pos="0 0 {L2}">
            <joint name="hinge3" type="hinge" axis="0 1 0" limited="false"/>
            <geom name="pole3" type="capsule" fromto="0 0 0 0 0 {L3}"
                  size="0.03" mass="0.3" rgba="0.3 0.3 0.85 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" name="cart_force" gear="60"
           ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


# ── Forward kinematics ────────────────────────────────────────────────────────

def fk_joints(q: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return (cart, h1, h2, h3, tip) world-frame XZ-plane positions."""
    cx = float(q[0])
    t1, t2, t3 = float(q[1]), float(q[2]), float(q[3])
    cart = np.array([cx, 0.0, 0.0])
    h1   = cart
    h2   = h1 + np.array([L1 * np.sin(t1),           0.0, L1 * np.cos(t1)])
    h3   = h2 + np.array([L2 * np.sin(t1 + t2),      0.0, L2 * np.cos(t1 + t2)])
    tip  = h3 + np.array([L3 * np.sin(t1+t2+t3),     0.0, L3 * np.cos(t1+t2+t3)])
    return cart, h1, h2, h3, tip


# ── Simulator ─────────────────────────────────────────────────────────────────

class _CartpoleSim:
    """Real-time MuJoCo simulator with a PD-controlled cart.

    ``x_commanded`` is the raw setpoint from the drag gizmo.
    ``x_target`` is the rate-limited version (advances at most
    ``max_rate * dt`` per step) that the PD controller tracks.
    Full physics — including pendulum inertial coupling — are preserved.
    """

    def __init__(self) -> None:
        self._mj_model = mujoco.MjModel.from_xml_string(_XML)
        self._mj_data  = mujoco.MjData(self._mj_model)

        # x_commanded: raw from gizmo (may jump).
        # x_target:    rate-limited setpoint tracked by PD.
        self.x_commanded: float = 0.0
        self.x_target:    float = 0.0
        self.max_rate:    float = DEFAULT_RATE
        self.kp:          float = DEFAULT_KP
        self.kd:          float = DEFAULT_KD

        self._pending_reset: str | None = "hanging"

        self._snap_lock = threading.Lock()
        self._qpos_snap = np.array([0.0, np.pi, 0.0, 0.0])
        self._qvel_snap = np.zeros(4)

        self._running = False
        self._thread: threading.Thread | None = None

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        with self._snap_lock:
            return self._qpos_snap.copy(), self._qvel_snap.copy()

    def request_reset(self, kind: str = "hanging") -> None:
        self._pending_reset = kind

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _apply_reset(self, kind: str) -> None:
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        if kind == "hanging":
            self._mj_data.qpos[:] = [0.0, np.pi, 0.0, 0.0]
        elif kind == "upright":
            self._mj_data.qpos[:] = [0.0, 0.04, 0.02, 0.01]
        self._mj_data.qvel[:] = 0.0
        self.x_target    = float(self._mj_data.qpos[0])
        self.x_commanded = self.x_target
        mujoco.mj_forward(self._mj_model, self._mj_data)

    def _loop(self) -> None:
        dt = float(self._mj_model.opt.timestep)

        while self._running:
            t0 = time.perf_counter()

            pending = self._pending_reset
            if pending is not None:
                self._pending_reset = None
                self._apply_reset(pending)

            # Rate-limit setpoint: x_target ramps toward x_commanded.
            dx_max        = self.max_rate * dt
            x_cmd         = float(np.clip(self.x_commanded, -RAIL_LIMIT, RAIL_LIMIT))
            self.x_target = self.x_target + float(
                np.clip(x_cmd - self.x_target, -dx_max, dx_max)
            )

            # PD control — full physics, no kinematic override.
            x    = float(self._mj_data.qpos[0])
            xdot = float(self._mj_data.qvel[0])
            u_N  = self.kp * (self.x_target - x) - self.kd * xdot
            self._mj_data.ctrl[0] = float(np.clip(u_N / GEAR, -1.0, 1.0))

            mujoco.mj_step(self._mj_model, self._mj_data)

            with self._snap_lock:
                self._qpos_snap = self._mj_data.qpos.copy()
                self._qvel_snap = self._mj_data.qvel.copy()

            spare = dt - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)


# ── Visualisation ─────────────────────────────────────────────────────────────

def run() -> None:
    try:
        import viser
    except ImportError:
        print("viser is not installed.  Install with: pip install viser", file=sys.stderr)
        sys.exit(1)

    sim = _CartpoleSim()
    sim.start()

    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=10.0, height=4.0, cell_size=0.5,
                          position=(0.0, 0.0, -0.115))

    # ── Static rail ───────────────────────────────────────────────────────────
    server.scene.add_box("/rail", dimensions=(RAIL_WIDTH + 0.1, 0.04, 0.015),
                         position=(0.0, 0.0, 0.0), color=(130, 130, 130))
    for sign in (-1, 1):
        server.scene.add_box(
            f"/rail/stop{'L' if sign < 0 else 'R'}",
            dimensions=(0.04, 0.12, 0.12),
            position=(sign * RAIL_LIMIT, 0.0, 0.0), color=(180, 60, 60))

    # ── Initial geometry ──────────────────────────────────────────────────────
    qpos0 = np.array([0.0, np.pi, 0.0, 0.0])
    cart0, h1_0, h2_0, h3_0, tip0 = fk_joints(qpos0)

    cart_handle = server.scene.add_box(
        "/cart", dimensions=(0.5, 0.3, 0.2),
        position=tuple(float(v) for v in cart0), color=(90, 90, 190))

    def _link_segs(cart, h1, h2, h3, tip) -> np.ndarray:
        return np.array([[h1, h2], [h2, h3], [h3, tip]], dtype=np.float32)

    link_colors = np.array([
        [[220, 80,  80],  [220, 80,  80]],
        [[80,  200, 80],  [80,  200, 80]],
        [[80,  80,  220], [80,  80,  220]],
    ], dtype=np.uint8)
    link_handle = server.scene.add_line_segments(
        "/links", points=_link_segs(cart0, h1_0, h2_0, h3_0, tip0),
        colors=link_colors, line_width=7.0)

    joint_handles = []
    for name, col, p0 in [("/j1", (200, 60, 60), h1_0),
                           ("/j2", (60, 200, 60), h2_0),
                           ("/j3", (60, 60, 200), h3_0)]:
        joint_handles.append(
            server.scene.add_icosphere(name, radius=0.05, color=col,
                                       position=tuple(float(v) for v in p0)))

    # ── Tip trail ─────────────────────────────────────────────────────────────
    MAX_TRAIL = 600
    tip_trail: list[np.ndarray] = []
    tip_cloud = server.scene.add_point_cloud(
        "/tip_trail", points=np.array([tip0], dtype=np.float32),
        colors=np.array([[255, 200, 50]], dtype=np.uint8), point_size=0.025)

    # ── Drag gizmo — X-axis only, always on the cart ──────────────────────────
    # Only the X (red) arrow is active.  Translation limits clamp to the rail.
    # The gizmo stays at x_commanded; the cart box tracks x_target (rate-limited).
    gizmo = server.scene.add_transform_controls(
        "/cart_gizmo",
        scale=0.9,
        active_axes=(True, False, False),   # X only
        disable_rotations=True,
        translation_limits=(
            (-RAIL_LIMIT, RAIL_LIMIT),
            (0.0, 0.0),
            (0.0, 0.0),
        ),
        position=(0.0, 0.0, 0.0),
        visible=True,
    )

    @gizmo.on_update
    def _drag(_event) -> None:
        x_new = float(np.clip(gizmo.position[0], -RAIL_LIMIT, RAIL_LIMIT))
        # Snap Y/Z to zero in case of floating-point drift.
        gizmo.position    = (x_new, 0.0, 0.0)
        sim.x_commanded   = x_new

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with server.gui.add_folder("Simulation"):
        rate_slider = server.gui.add_slider(
            "Max setpoint speed (m/s)",
            min=0.1, max=RAIL_WIDTH, step=0.1,
            initial_value=DEFAULT_RATE,
        )
        btn_hanging = server.gui.add_button("Reset — hanging")
        btn_upright = server.gui.add_button("Reset — upright")

        @rate_slider.on_update
        def _(_event) -> None:
            sim.max_rate = float(rate_slider.value)

        @btn_hanging.on_click
        def _(_event) -> None:
            tip_trail.clear()
            sim.request_reset("hanging")
            gizmo.position = (0.0, 0.0, 0.0)

        @btn_upright.on_click
        def _(_event) -> None:
            tip_trail.clear()
            sim.request_reset("upright")
            gizmo.position = (0.0, 0.0, 0.0)

    with server.gui.add_folder("PD Gains"):
        kp_slider = server.gui.add_slider(
            "Kp  (N/m)",   min=0.0, max=300.0, step=1.0,  initial_value=DEFAULT_KP)
        kd_slider = server.gui.add_slider(
            "Kd  (N·s/m)", min=0.0, max=80.0,  step=0.5,  initial_value=DEFAULT_KD)

        @kp_slider.on_update
        def _(_event) -> None:
            sim.kp = float(kp_slider.value)

        @kd_slider.on_update
        def _(_event) -> None:
            sim.kd = float(kd_slider.value)

    with server.gui.add_folder("State"):
        state_md = server.gui.add_markdown("*Waiting...*")

    # ── Render loop ───────────────────────────────────────────────────────────
    RENDER_DT = 1.0 / 60.0

    def _render_loop() -> None:
        while True:
            t0 = time.perf_counter()

            qpos, qvel = sim.get_state()
            x = float(qpos[0])        # actual cart position from physics
            cart, h1, h2, h3, tip = fk_joints(qpos)

            cart_handle.position = (x, 0.0, 0.0)
            link_handle.points   = _link_segs(cart, h1, h2, h3, tip)
            for jh, jp in zip(joint_handles, [h1, h2, h3]):
                jh.position = tuple(float(v) for v in jp)

            # Tip trail.
            tip_trail.append(tip.copy())
            if len(tip_trail) > MAX_TRAIL:
                tip_trail.pop(0)
            if tip_trail:
                tip_arr = np.array(tip_trail, dtype=np.float32)
                n       = len(tip_arr)
                frac    = np.linspace(0.0, 1.0, n)
                colors  = np.zeros((n, 3), dtype=np.uint8)
                colors[:, 0] = (255 * frac).astype(np.uint8)
                colors[:, 1] = (200 * (1.0 - frac)).astype(np.uint8)
                colors[:, 2] = 50
                tip_cloud.points = tip_arr
                tip_cloud.colors = colors

            # State readout.
            deg = np.rad2deg(qpos[1:4])
            state_md.content = (
                f"**Cart x:** {x:.3f} m  ·  **ẋ:** {qvel[0]:.2f} m/s\n\n"
                f"**θ₁:** {deg[0]:.1f}°  ·  **θ̇₁:** {np.rad2deg(qvel[1]):.1f}°/s\n\n"
                f"**θ₂:** {deg[1]:.1f}°  ·  **θ̇₂:** {np.rad2deg(qvel[2]):.1f}°/s\n\n"
                f"**θ₃:** {deg[2]:.1f}°  ·  **θ̇₃:** {np.rad2deg(qvel[3]):.1f}°/s"
            )

            spare = RENDER_DT - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)

    threading.Thread(target=_render_loop, daemon=True).start()

    print("Triple cartpole physics sandbox — open http://localhost:8080")
    print("Drag the red X-arrow on the cart to move it.")
    print("Adjust 'Max cart speed' in the sidebar to change the rate limit.")
    server.sleep_forever()


if __name__ == "__main__":
    run()
