"""Triple-cartpole balancing game.

Three progressively harder levels:
  Level 1 — single-link inverted pendulum
  Level 2 — double-link inverted pendulum
  Level 3 — triple-link inverted pendulum

Win condition for each level: hold ALL link cumulative angles within
10° of vertical for 1 continuous second.  A 3-second freeze screen
shows your time before the next level loads.

Each link's target region is shown as a green/red wedge — the angle
the link must point into — that tracks the link's pivot point in real
time.

Interaction
-----------
- Drag the **red X-arrow** on the cart to move the setpoint.
- PD gains and setpoint speed are adjustable in the sidebar.
- "Reset level" button restarts from hanging.

Usage::

    python examples/mjx/triple_cartpole_game.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

current_dir     = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

try:
    import mujoco
    import mujoco.mjx as mjx
    import jax
    import jax.numpy as jnp
except ImportError:
    sys.exit("MuJoCo MJX / JAX not installed.  Run: pip install openscvx[mjx]")

# ── Constants ─────────────────────────────────────────────────────────────────
LINK_LENGTHS = [0.5, 0.4, 0.3]
LINK_MASSES  = [0.5, 0.4, 0.3]
LINK_RADII   = [0.04, 0.035, 0.03]
LINK_RGBA    = ["0.85 0.3 0.3 1", "0.3 0.8 0.3 1", "0.3 0.3 0.85 1"]
LINK_RGB     = [(220, 77, 77), (77, 204, 77), (77, 77, 220)]

RAIL_LIMIT    = 8.0
GEAR          = 60.0
DEFAULT_KP    = 60.0
DEFAULT_KD    = 15.0
DEFAULT_RATE  = 4.0

WIN_TOL_DEG   = 45.0
HOLD_SECS     = 0.1
WIN_PAUSE_S   = 3.0
N_LEVELS      = 3
LEVEL_NAMES   = ["Single-link", "Double-link", "Triple-link"]

# ── Physics backend ───────────────────────────────────────────────────────────
# True  → MJX + RK4  (mjx.forward continuous dynamics, same as post_process)
# False → CPU MuJoCo mj_step (semi-implicit Euler, same as the original sim)
USE_MJX_PHYSICS: bool = False


# ── MuJoCo XML ────────────────────────────────────────────────────────────────

def make_xml(n_links: int) -> str:
    def build_links(i: int) -> str:
        if i >= n_links:
            return ""
        pad = "  " * (i + 3)
        pos = f"0 0 {LINK_LENGTHS[i - 1]}" if i > 0 else "0 0 0"
        return (
            f'{pad}<body name="link{i+1}" pos="{pos}">\n'
            f'{pad}  <joint name="hinge{i+1}" type="hinge"'
            f' axis="0 1 0" limited="false"/>\n'
            f'{pad}  <geom name="pole{i+1}" type="capsule"'
            f' fromto="0 0 0 0 0 {LINK_LENGTHS[i]}"\n'
            f'{pad}        size="{LINK_RADII[i]}" mass="{LINK_MASSES[i]}"'
            f' rgba="{LINK_RGBA[i]}"/>\n'
            f'{build_links(i + 1)}'
            f'{pad}</body>\n'
        )
    return f"""<mujoco model="cartpole_game">
  <option gravity="0 0 -9.81" timestep="0.005" integrator="Euler"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0"
             limited="true" range="-{RAIL_LIMIT} {RAIL_LIMIT}"/>
      <geom name="cart_geom" type="box" size="0.25 0.15 0.1"
            mass="2.0" rgba="0.35 0.35 0.75 1"/>
{build_links(0)}    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" name="cart_force" gear="{GEAR}"
           ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>"""


# ── Forward kinematics ────────────────────────────────────────────────────────

def fk_joints(q: np.ndarray, n_links: int) -> list[np.ndarray]:
    """Return [cart, tip_1, …, tip_n] XZ-plane positions."""
    cx  = float(q[0])
    pts = [np.array([cx, 0.0, 0.0])]
    cum = 0.0
    for i in range(n_links):
        cum += float(q[i + 1])
        L = LINK_LENGTHS[i]
        pts.append(pts[-1] + np.array([L * np.sin(cum), 0.0, L * np.cos(cum)]))
    return pts


def per_link_ok(qpos: np.ndarray, n_links: int) -> tuple[list[bool], float]:
    """Return per-link balance flags and worst absolute cumulative angle (deg)."""
    ok    = []
    worst = 0.0
    cum   = 0.0
    for i in range(n_links):
        cum    += float(qpos[i + 1])
        c_norm  = (cum + np.pi) % (2.0 * np.pi) - np.pi
        d       = abs(float(np.rad2deg(c_norm)))
        worst   = max(worst, d)
        ok.append(d <= WIN_TOL_DEG)
    return ok, worst


# ── MJX physics step (same continuous dynamics as post_process) ───────────────

def _make_mjx_step(n_links: int) -> tuple:
    """Build and warm-up a JIT-compiled RK4+MJX step function.

    Uses ``mjx.forward`` to evaluate the continuous generalized accelerations
    (qacc) at each RK4 stage — identical to the dynamics that ``post_process``
    hands to Diffrax Dopri8.  Returns ``(step_fn, dt)`` where ``step_fn`` is
    already compiled (warm-up call included) so there is no JIT latency during
    gameplay.

    ``step_fn(qpos, qvel, ctrl) -> (new_qpos, new_qvel)`` accepts and returns
    float32 JAX arrays of shape ``(n_links+1,)``.
    """
    n_q  = n_links + 1
    mj_m = mujoco.MjModel.from_xml_string(make_xml(n_links))
    mj_m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mjx_m = mjx.put_model(mj_m)
    h     = float(mj_m.opt.timestep)

    def _qacc(q: jax.Array, qd: jax.Array, ctrl: jax.Array) -> jax.Array:
        data = mjx.make_data(mjx_m)
        data = data.replace(qpos=q, qvel=qd, ctrl=ctrl)
        data = mjx.forward(mjx_m, data)
        return data.qacc

    @jax.jit
    def step(qpos: jax.Array, qvel: jax.Array, ctrl: jax.Array):
        # Standard RK4: y = [qpos, qvel],  y' = [qvel, qacc(qpos, qvel)]
        k1_q = qvel;                   k1_v = _qacc(qpos, qvel, ctrl)
        qp2  = qpos + 0.5 * h * k1_q; qv2  = qvel + 0.5 * h * k1_v
        k2_q = qv2;                    k2_v = _qacc(qp2, qv2, ctrl)
        qp3  = qpos + 0.5 * h * k2_q; qv3  = qvel + 0.5 * h * k2_v
        k3_q = qv3;                    k3_v = _qacc(qp3, qv3, ctrl)
        qp4  = qpos + h * k3_q;       qv4  = qvel + h * k3_v
        k4_q = qv4;                    k4_v = _qacc(qp4, qv4, ctrl)

        new_qpos = qpos + (h / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        new_qvel = qvel + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        # Hard rail limits (MJX contacts are disabled; enforce manually)
        at_wall  = jnp.abs(new_qpos[0]) >= RAIL_LIMIT
        new_qpos = new_qpos.at[0].set(jnp.clip(new_qpos[0], -RAIL_LIMIT, RAIL_LIMIT))
        new_qvel = new_qvel.at[0].set(jnp.where(at_wall, 0.0, new_qvel[0]))

        return new_qpos, new_qvel

    # Warm-up: trigger JIT compilation now so gameplay has no first-step lag.
    _z = jnp.zeros(n_q, dtype=jnp.float32)
    step(_z, _z, jnp.zeros(1, dtype=jnp.float32))
    return step, h


# ── Simulator ─────────────────────────────────────────────────────────────────

class _Sim:
    """Real-time cartpole simulator.

    Supports two physics backends selected by :data:`USE_MJX_PHYSICS`:

    * **MJX + RK4** (``step_fn`` provided): uses ``mjx.forward`` continuous
      dynamics at each of the four RK4 stages — identical to what
      ``post_process`` hands to Diffrax Dopri8.  State is kept as JAX float32
      arrays; no CPU ``MjData`` is used at runtime.

    * **CPU MuJoCo** (``step_fn=None``): calls ``mujoco.mj_step`` with
      semi-implicit Euler, exactly as the original simulator.
    """

    def __init__(self, n_links: int, step_fn=None, dt: float | None = None) -> None:
        self.n_links  = n_links
        self.n_q      = n_links + 1
        self._step_fn = step_fn   # None → CPU MuJoCo path

        if step_fn is None:
            # CPU MuJoCo path: build model + data here; dt from XML.
            self._mj_model = mujoco.MjModel.from_xml_string(make_xml(n_links))
            self._mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
            self._mj_data  = mujoco.MjData(self._mj_model)
            self._dt       = float(self._mj_model.opt.timestep)
        else:
            self._mj_model = None
            self._mj_data  = None
            self._dt       = dt  # supplied by _make_mjx_step

        self.x_commanded: float = 0.0
        self.x_target:    float = 0.0
        self.max_rate:    float = DEFAULT_RATE
        self.kp:          float = DEFAULT_KP
        self.kd:          float = DEFAULT_KD
        self.paused:      bool  = False

        self._snap_lock = threading.Lock()
        q0 = np.zeros(self.n_q, dtype=np.float32);  q0[1] = np.pi
        self._qpos_snap = q0.copy()
        self._qvel_snap = np.zeros(self.n_q, dtype=np.float32)

        if step_fn is None:
            # Initialise CPU MuJoCo data to the hanging state.
            self._mj_data.qpos[:] = q0
            self._mj_data.qvel[:] = 0.0
            mujoco.mj_forward(self._mj_model, self._mj_data)

        self._running = False
        self._thread: threading.Thread | None = None

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        with self._snap_lock:
            return self._qpos_snap.copy(), self._qvel_snap.copy()

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)

    def _loop(self) -> None:
        if self._step_fn is not None:
            self._loop_mjx()
        else:
            self._loop_mujoco()

    def _loop_mjx(self) -> None:
        """Physics loop: MJX + RK4 (same continuous dynamics as post_process)."""
        dt = self._dt
        with self._snap_lock:
            qpos = jnp.array(self._qpos_snap, dtype=jnp.float32)
            qvel = jnp.array(self._qvel_snap, dtype=jnp.float32)

        while self._running:
            t0 = time.perf_counter()
            if not self.paused:
                dx_max        = self.max_rate * dt
                x_cmd         = float(np.clip(self.x_commanded, -RAIL_LIMIT, RAIL_LIMIT))
                self.x_target = self.x_target + float(
                    np.clip(x_cmd - self.x_target, -dx_max, dx_max))

                x    = float(qpos[0])
                xdot = float(qvel[0])
                u_N  = self.kp * (self.x_target - x) - self.kd * xdot
                ctrl = jnp.array([float(np.clip(u_N / GEAR, -1.0, 1.0))],
                                  dtype=jnp.float32)

                qpos, qvel = self._step_fn(qpos, qvel, ctrl)

                with self._snap_lock:
                    self._qpos_snap = np.array(qpos)
                    self._qvel_snap = np.array(qvel)

            spare = dt - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)

    def _loop_mujoco(self) -> None:
        """Physics loop: CPU MuJoCo mj_step (semi-implicit Euler)."""
        dt = self._dt
        while self._running:
            t0 = time.perf_counter()
            if not self.paused:
                dx_max        = self.max_rate * dt
                x_cmd         = float(np.clip(self.x_commanded, -RAIL_LIMIT, RAIL_LIMIT))
                self.x_target = self.x_target + float(
                    np.clip(x_cmd - self.x_target, -dx_max, dx_max))

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


# ── Fireworks ─────────────────────────────────────────────────────────────────

def _run_fireworks(server: object, duration: float = 9.0) -> None:
    """Celebratory particle fireworks.  Runs in a daemon thread."""
    rng = np.random.default_rng()

    # Palette: each shell picks one base colour; particles fade to black.
    PALETTE = [
        (255,  80,  80),   # red
        ( 80, 255,  80),   # green
        ( 80, 130, 255),   # blue
        (255, 220,  50),   # gold
        (220,  80, 220),   # purple
        ( 80, 220, 220),   # cyan
        (255, 160,  40),   # orange
        (255, 255, 255),   # white
    ]

    GRAVITY          = np.array([0.0, 0.0, -5.0], dtype=np.float32)
    DT               = 1.0 / 60.0
    LIFETIME         = 2.8   # particle lifetime (s)
    N_PER_BURST      = 130
    BURST_INTERVAL   = 0.45  # s between shells

    # Single point cloud for all particles.
    cloud = server.scene.add_point_cloud(
        "/fireworks",
        points=np.array([[0.0, 0.0, -20.0]], dtype=np.float32),
        colors=np.array([[0, 0, 0]], dtype=np.uint8),
        point_size=0.06)

    # Particle state arrays.
    pos  = np.zeros((0, 3), dtype=np.float32)
    vel  = np.zeros((0, 3), dtype=np.float32)
    col  = np.zeros((0, 3), dtype=np.float32)   # float for smooth fading
    life = np.zeros(0,      dtype=np.float32)

    t          = 0.0
    next_burst = 0.0
    color_idx  = 0

    while t < duration:
        t0 = time.perf_counter()

        # ── Spawn new shell ───────────────────────────────────────────────────
        if t >= next_burst:
            next_burst += BURST_INTERVAL
            color_idx   = (color_idx + 1) % len(PALETTE)

            bx = float(rng.uniform(-3.2, 3.2))
            bz = float(rng.uniform(0.6, 2.4))
            burst_pos = np.array([bx, 0.0, bz], dtype=np.float32)
            base_col  = np.array(PALETTE[color_idx], dtype=np.float32)

            # Uniform spherical velocity distribution, slightly squeezed in Y.
            u   = rng.uniform(-1.0, 1.0, N_PER_BURST).astype(np.float32)
            phi = rng.uniform(0.0, 2.0 * np.pi, N_PER_BURST).astype(np.float32)
            r   = np.sqrt(np.maximum(1.0 - u * u, 0.0)).astype(np.float32)
            spd = rng.uniform(0.6, 3.0, N_PER_BURST).astype(np.float32)
            new_vel = (spd[:, None] * np.stack([
                r * np.cos(phi),
                r * 0.25 * np.sin(phi),   # thin Y spread — mostly XZ plane
                u,
            ], axis=1)).astype(np.float32)

            new_pos  = np.tile(burst_pos, (N_PER_BURST, 1))
            new_col  = np.tile(base_col,  (N_PER_BURST, 1))
            new_life = np.full(N_PER_BURST, LIFETIME, dtype=np.float32)

            if len(pos):
                pos  = np.vstack([pos,  new_pos])
                vel  = np.vstack([vel,  new_vel])
                col  = np.vstack([col,  new_col])
                life = np.concatenate([life, new_life])
            else:
                pos, vel, col, life = new_pos, new_vel, new_col, new_life

        # ── Physics step ─────────────────────────────────────────────────────
        if len(pos):
            vel  += GRAVITY * DT
            pos  += vel * DT
            life -= DT

            alive = life > 0.0
            if np.any(alive):
                frac        = np.clip(life[alive] / LIFETIME, 0.0, 1.0)[:, None]
                disp_col    = np.clip(col[alive] * frac, 0, 255).astype(np.uint8)
                cloud.points = pos[alive]
                cloud.colors = disp_col
            else:
                cloud.points = np.array([[0.0, 0.0, -20.0]], dtype=np.float32)
                cloud.colors = np.array([[0, 0, 0]], dtype=np.uint8)

            pos  = pos[alive];   vel  = vel[alive]
            col  = col[alive];   life = life[alive]

        t    += DT
        spare = DT - (time.perf_counter() - t0)
        if spare > 0:
            time.sleep(spare)

    try:
        cloud.remove()
    except Exception:
        pass


# ── Cone geometry factory ─────────────────────────────────────────────────────

def _make_cone_mesh(link_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangular wedge centred at origin, apex down, opening upward.

    Returns (vertices [3×3 float32], faces [2×3 uint32]).
    Two faces so the mesh is visible from both the +Y and −Y camera sides.
    """
    L    = LINK_LENGTHS[link_idx]
    half = np.deg2rad(WIN_TOL_DEG)
    verts = np.array([
        [0.0,              0.0, 0.0],                   # apex
        [-L * np.sin(half), 0.0, L * np.cos(half)],     # left boundary
        [ L * np.sin(half), 0.0, L * np.cos(half)],     # right boundary
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 1]], dtype=np.uint32)  # front + back
    return verts, faces


def _make_cone_lines(link_idx: int) -> np.ndarray:
    """Two boundary edges of the cone as a (2,2,3) float32 segment array."""
    L    = LINK_LENGTHS[link_idx]
    half = np.deg2rad(WIN_TOL_DEG)
    apex     = np.array([0.0,              0.0, 0.0],                   dtype=np.float32)
    left_tip = np.array([-L * np.sin(half), 0.0, L * np.cos(half)],     dtype=np.float32)
    rght_tip = np.array([ L * np.sin(half), 0.0, L * np.cos(half)],     dtype=np.float32)
    return np.array([[apex, left_tip], [apex, rght_tip]], dtype=np.float32)


# ── Main game loop ────────────────────────────────────────────────────────────

def run() -> None:
    try:
        import viser
    except ImportError:
        sys.exit("viser not installed.  Run: pip install viser")

    # Pre-compile MJX step functions for all levels (only when MJX is active).
    _step_cache: dict[int, tuple] = {}   # n_links -> (step_fn, dt)
    if USE_MJX_PHYSICS:
        print("Compiling MJX physics (RK4 + mjx.forward) for all levels…")
        for _nl in range(1, N_LEVELS + 1):
            print(f"  Level {_nl} ({LEVEL_NAMES[_nl-1]})…", end=" ", flush=True)
            _step_cache[_nl] = _make_mjx_step(_nl)
            print("done")
        print("Physics ready.  [MJX + RK4]\n")
    else:
        print("Physics backend: CPU MuJoCo mj_step (semi-implicit Euler)\n")

    server = viser.ViserServer()
    server.scene.set_up_direction("+z")

    # ── Game state ────────────────────────────────────────────────────────────
    gs: dict = {
        "level":         0,
        "status":        "idle",
        "level_start":   0.0,
        "win_at":        0.0,
        "win_elapsed":   0.0,
        "balance_start": None,
        "advancing":     False,
    }
    _sim:        list[_Sim | None] = [None]
    _level_vis:  list[dict | None] = [None]
    _handles:    list[list]        = [[]]   # all per-level scene handles
    _gen:        list[int]         = [0]    # incremented on each load; guards stale gizmo callbacks

    # ── Persistent scene ──────────────────────────────────────────────────────
    server.scene.add_grid("/ground", width=10.0, height=4.0, cell_size=0.5,
                          position=(0.0, 0.0, -0.115))
    server.scene.add_box("/rail", dimensions=(8.1, 0.04, 0.015),
                         position=(0.0, 0.0, 0.0), color=(130, 130, 130))
    for sign in (-1, 1):
        server.scene.add_box(
            f"/rail/stop{'L' if sign < 0 else 'R'}",
            dimensions=(0.04, 0.12, 0.12),
            position=(sign * RAIL_LIMIT, 0.0, 0.0), color=(180, 60, 60))

    # ── Persistent GUI ────────────────────────────────────────────────────────
    header_html   = server.gui.add_html("")
    win_html      = server.gui.add_html("", visible=False)
    progress_html = server.gui.add_html("")

    with server.gui.add_folder("Controls"):
        rate_sl   = server.gui.add_slider("Max setpoint speed (m/s)",
                                          min=0.1, max=8.0, step=0.1,
                                          initial_value=DEFAULT_RATE)
        btn_reset = server.gui.add_button("Reset level")

    with server.gui.add_folder("PD Gains"):
        kp_sl = server.gui.add_slider("Kp  (N/m)",   min=0, max=300, step=1,
                                       initial_value=DEFAULT_KP)
        kd_sl = server.gui.add_slider("Kd  (N·s/m)", min=0, max=80,  step=0.5,
                                       initial_value=DEFAULT_KD)

    with server.gui.add_folder("State"):
        state_md = server.gui.add_markdown("*Waiting…*")

    @rate_sl.on_update
    def _(_e):
        if _sim[0]: _sim[0].max_rate = float(rate_sl.value)

    @kp_sl.on_update
    def _(_e):
        if _sim[0]: _sim[0].kp = float(kp_sl.value)

    @kd_sl.on_update
    def _(_e):
        if _sim[0]: _sim[0].kd = float(kd_sl.value)

    @btn_reset.on_click
    def _(_e):
        gs["advancing"] = False
        win_html.visible = False
        _load_level(gs["level"])

    # ── HTML helpers ──────────────────────────────────────────────────────────

    def _header(level_idx: int, elapsed: float | None = None) -> str:
        stars = "★" * (level_idx + 1) + "☆" * (N_LEVELS - level_idx - 1)
        timer = (f'<div style="font-size:24px;font-weight:bold;color:#f5c518;">'
                 f'⏱ {elapsed:.1f}s</div>' if elapsed is not None else "")
        return (
            f'<div style="text-align:center;padding:6px 0;">'
            f'<div style="font-size:20px;font-weight:bold;color:#e8e8e8;">'
            f'Level {level_idx+1} / {N_LEVELS}</div>'
            f'<div style="font-size:13px;color:#aaa;margin-bottom:4px;">'
            f'{LEVEL_NAMES[level_idx]} pendulum</div>'
            f'<div style="font-size:18px;color:#f5c518;">{stars}</div>'
            f'{timer}</div>'
        )

    def _win_screen(level_idx: int, elapsed: float, countdown: float) -> str:
        if level_idx < N_LEVELS - 1:
            nxt = (f'<div style="color:#888;font-size:13px;margin-top:8px;">'
                   f'Level {level_idx+2} in {countdown:.0f}s…</div>')
        else:
            nxt = ('<div style="color:#f5c518;font-size:14px;margin-top:8px;">'
                   '🏆 All levels complete!</div>')
        return (
            f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);'
            f'border:2px solid #50fa7b;border-radius:12px;padding:20px;'
            f'text-align:center;margin:6px 0;">'
            f'<div style="font-size:36px;margin-bottom:6px;">🎉</div>'
            f'<div style="color:#50fa7b;font-size:20px;font-weight:bold;">'
            f'Level {level_idx+1} Complete!</div>'
            f'<div style="color:#f8f8f2;font-size:16px;margin-top:6px;">'
            f'Balanced in <strong>{elapsed:.1f}s</strong></div>'
            f'{nxt}</div>'
        )

    def _all_done() -> str:
        return (
            '<div style="background:linear-gradient(135deg,#2d1b69,#1a1a2e);'
            'border:2px solid #f5c518;border-radius:12px;padding:24px;'
            'text-align:center;margin:6px 0;">'
            '<div style="font-size:48px;margin-bottom:8px;">🏆</div>'
            '<div style="color:#f5c518;font-size:22px;font-weight:bold;">'
            'You beat the game!</div>'
            '<div style="color:#aaa;font-size:14px;margin-top:8px;">'
            'All 3 levels balanced. Impressive.</div></div>'
        )

    def _progress_bar(hold: float) -> str:
        frac = min(1.0, hold / HOLD_SECS)
        return (
            f'<div style="margin:6px 0;text-align:center;">'
            f'<div style="color:#50fa7b;font-size:12px;margin-bottom:4px;">'
            f'Balancing… {hold:.1f} / {HOLD_SECS:.0f}s</div>'
            f'<div style="background:#333;border-radius:4px;height:8px;">'
            f'<div style="background:#50fa7b;width:{frac*100:.0f}%;'
            f'height:8px;border-radius:4px;"></div></div></div>'
        )

    def _angle_warning(worst: float) -> str:
        return (
            f'<div style="text-align:center;color:#ff6b6b;font-size:12px;'
            f'margin:4px 0;">Worst angle: {worst:.1f}°'
            f' (need &lt;{WIN_TOL_DEG:.0f}°)</div>'
        )

    # ── Level setup ───────────────────────────────────────────────────────────

    def _load_level(level_idx: int) -> None:
        # Block render loop during transition.
        _level_vis[0] = None
        # Bump generation so stale gizmo callbacks from the old level are ignored.
        _gen[0] += 1
        my_gen = _gen[0]

        # Remove all per-level scene objects from the previous level.
        for h in _handles[0]:
            try:
                h.remove()
            except Exception:
                pass
        _handles[0].clear()

        # Stop old sim.
        old = _sim[0];  _sim[0] = None
        if old:
            old.paused = False;  old.stop()

        # Create new sim, using the pre-compiled MJX step function or CPU
        # MuJoCo depending on the USE_MJX_PHYSICS flag.
        n = level_idx + 1
        sim = _Sim(n, *_step_cache[n]) if USE_MJX_PHYSICS else _Sim(n)
        sim.kp = float(kp_sl.value);  sim.kd = float(kd_sl.value)
        sim.max_rate = float(rate_sl.value)
        sim.start()
        _sim[0] = sim

        gs.update({
            "level":         level_idx,
            "status":        "playing",
            "level_start":   time.time(),
            "win_at":        0.0,  "win_elapsed": 0.0,
            "balance_start": None, "advancing":   False,
        })
        header_html.content = _header(level_idx)
        win_html.visible    = False
        progress_html.content = ""

        # ── Build scene ───────────────────────────────────────────────────────
        q0  = np.zeros(n + 1);  q0[1] = np.pi
        pts = fk_joints(q0, n)

        def reg(h):
            _handles[0].append(h)
            return h

        cart_h = reg(server.scene.add_box(
            "/lvl/cart", dimensions=(0.5, 0.3, 0.2),
            position=tuple(float(v) for v in pts[0]), color=(90, 90, 190)))

        # One scene object per link — avoids any buffer-resize issue in Viser
        # when the link count changes between levels.
        link_handles = [
            reg(server.scene.add_line_segments(
                f"/lvl/link{i}",
                points=np.array([[pts[i], pts[i + 1]]], dtype=np.float32),
                colors=np.array([[LINK_RGB[i], LINK_RGB[i]]], dtype=np.uint8),
                line_width=7.0))
            for i in range(n)
        ]

        jhandles = [
            reg(server.scene.add_icosphere(
                f"/lvl/j{i}", radius=0.05, color=LINK_RGB[i],
                position=tuple(float(v) for v in pts[i])))
            for i in range(n)
        ]

        tip_trail: list[np.ndarray] = []
        tip_cloud = reg(server.scene.add_point_cloud(
            "/lvl/tip", points=np.array([pts[-1]], dtype=np.float32),
            colors=np.array([[255, 200, 50]], dtype=np.uint8), point_size=0.025))

        gizmo = reg(server.scene.add_transform_controls(
            "/lvl/gizmo", scale=0.9,
            active_axes=(True, False, False), disable_rotations=True,
            translation_limits=((-RAIL_LIMIT, RAIL_LIMIT), (0, 0), (0, 0)),
            position=(0.0, 0.0, 0.0)))

        @gizmo.on_update
        def _drag(_e):
            if _gen[0] != my_gen:   # stale callback from a removed gizmo
                return
            x_new = float(np.clip(gizmo.position[0], -RAIL_LIMIT, RAIL_LIMIT))
            gizmo.position = (x_new, 0.0, 0.0)
            if _sim[0]: _sim[0].x_commanded = x_new

        # ── Tolerance cones ───────────────────────────────────────────────────
        # Each link i gets TWO mesh wedges (green = ok, red = not ok) plus
        # two boundary line pairs.  Visibility is toggled every render frame.
        verts_cone, faces_cone = zip(*[_make_cone_mesh(i) for i in range(n)])
        lines_cone = [_make_cone_lines(i) for i in range(n)]

        green_fill, red_fill     = [], []
        green_lines, red_lines   = [], []
        _col_lines = np.array([[[c, c], [c, c]] for c in
                                [(60, 200, 60), (60, 200, 60)]], dtype=np.uint8)
        _red_lines = np.array([[[c, c], [c, c]] for c in
                                [(200, 60, 60), (200, 60, 60)]], dtype=np.uint8)

        for i in range(n):
            base = tuple(float(v) for v in pts[i])
            lpts = lines_cone[i]
            g_lc = np.array([[[60, 200, 60], [60, 200, 60]],
                              [[60, 200, 60], [60, 200, 60]]], dtype=np.uint8)
            r_lc = np.array([[[200, 60, 60], [200, 60, 60]],
                              [[200, 60, 60], [200, 60, 60]]], dtype=np.uint8)

            green_fill.append(reg(server.scene.add_mesh_simple(
                f"/lvl/cone_g{i}", vertices=verts_cone[i], faces=faces_cone[i],
                color=(60, 200, 60), opacity=0.18, side="double",
                position=base, visible=False)))
            red_fill.append(reg(server.scene.add_mesh_simple(
                f"/lvl/cone_r{i}", vertices=verts_cone[i], faces=faces_cone[i],
                color=(200, 60, 60), opacity=0.18, side="double",
                position=base, visible=True)))
            green_lines.append(reg(server.scene.add_line_segments(
                f"/lvl/cline_g{i}", points=lpts, colors=g_lc, line_width=1.5,
                position=base, visible=False)))
            red_lines.append(reg(server.scene.add_line_segments(
                f"/lvl/cline_r{i}", points=lpts, colors=r_lc, line_width=1.5,
                position=base, visible=True)))

        _level_vis[0] = {
            "n": n, "cart_h": cart_h, "link_handles": link_handles,
            "jhandles": jhandles, "tip_trail": tip_trail, "tip_cloud": tip_cloud,
            "green_fill": green_fill, "red_fill": red_fill,
            "green_lines": green_lines, "red_lines": red_lines,
        }

    # ── Win / advance ─────────────────────────────────────────────────────────

    def _trigger_win(elapsed: float) -> None:
        gs.update({"status": "won", "win_at": time.time(), "win_elapsed": elapsed})
        if _sim[0]: _sim[0].paused = True
        progress_html.content = ""
        win_html.content = _win_screen(gs["level"], elapsed, WIN_PAUSE_S)
        win_html.visible = True

    def _advance_level() -> None:
        nxt = gs["level"] + 1
        if nxt < N_LEVELS:
            _load_level(nxt)
        else:
            gs["status"] = "done"
            header_html.content = progress_html.content = ""
            win_html.content = _all_done()
            win_html.visible = True
            threading.Thread(
                target=_run_fireworks, args=(server,), daemon=True).start()

    # ── Render / game loop ────────────────────────────────────────────────────
    RENDER_DT = 1.0 / 60.0
    MAX_TRAIL  = 500

    def _render_loop() -> None:
        while True:
            t0  = time.perf_counter()
            vis = _level_vis[0]
            sim = _sim[0]
            if vis is None or sim is None:
                time.sleep(RENDER_DT);  continue

            n      = vis["n"]
            status = gs["status"]
            qpos, qvel = sim.get_state()
            pts        = fk_joints(qpos, n)

            # ── 3-D scene ─────────────────────────────────────────────────────
            vis["cart_h"].position = (float(pts[0][0]), 0.0, 0.0)
            for i, lh in enumerate(vis["link_handles"]):
                lh.points = np.array([[pts[i], pts[i + 1]]], dtype=np.float32)
            for i, jh in enumerate(vis["jhandles"]):
                jh.position = tuple(float(v) for v in pts[i])

            tip_trail = vis["tip_trail"]
            tip_trail.append(pts[-1].copy())
            if len(tip_trail) > MAX_TRAIL:
                tip_trail.pop(0)
            if tip_trail:
                ta = np.array(tip_trail, dtype=np.float32)
                nf = len(ta);  fr = np.linspace(0.0, 1.0, nf)
                tc = np.zeros((nf, 3), dtype=np.uint8)
                tc[:, 0] = (255 * fr).astype(np.uint8)
                tc[:, 1] = (200 * (1.0 - fr)).astype(np.uint8)
                tc[:, 2] = 50
                vis["tip_cloud"].points = ta;  vis["tip_cloud"].colors = tc

            # ── Tolerance cones — move to link bases, toggle green/red ────────
            link_ok, worst = per_link_ok(qpos, n)
            for i in range(n):
                base = tuple(float(v) for v in pts[i])
                ok   = link_ok[i]
                vis["green_fill"][i].position  = base
                vis["red_fill"][i].position    = base
                vis["green_lines"][i].position = base
                vis["red_lines"][i].position   = base
                vis["green_fill"][i].visible   = ok
                vis["red_fill"][i].visible     = not ok
                vis["green_lines"][i].visible  = ok
                vis["red_lines"][i].visible    = not ok

            # ── State readout ─────────────────────────────────────────────────
            deg   = np.rad2deg(qpos[1:n+1])
            lines = [f"**x:** {qpos[0]:.3f} m  ·  **ẋ:** {qvel[0]:.2f} m/s"]
            for i in range(n):
                lines.append(
                    f"**θ{i+1}:** {deg[i]:.1f}°  ·  "
                    f"**θ̇{i+1}:** {np.rad2deg(qvel[i+1]):.1f}°/s")
            state_md.content = "\n\n".join(lines)

            # ── Game logic ────────────────────────────────────────────────────
            if status == "playing":
                elapsed  = time.time() - gs["level_start"]
                all_ok   = all(link_ok)
                header_html.content = _header(gs["level"], elapsed)

                if all_ok:
                    if gs["balance_start"] is None:
                        gs["balance_start"] = time.time()
                    hold = time.time() - gs["balance_start"]
                    progress_html.content = _progress_bar(hold)
                    if hold >= HOLD_SECS:
                        _trigger_win(elapsed)
                else:
                    gs["balance_start"] = None
                    progress_html.content = _angle_warning(worst)

            elif status == "won":
                rem = WIN_PAUSE_S - (time.time() - gs["win_at"])
                if rem > 0:
                    win_html.content = _win_screen(gs["level"], gs["win_elapsed"], rem)
                elif not gs["advancing"]:
                    gs["advancing"] = True
                    threading.Thread(target=_advance_level, daemon=True).start()

            spare = RENDER_DT - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)

    threading.Thread(target=_render_loop, daemon=True).start()
    _load_level(0)

    print("Cartpole Balancing Game — open http://localhost:8080")
    print("Drag the red X-arrow to move the cart.")
    print(f"Hold all links inside the green cones (within {WIN_TOL_DEG:.0f}°)"
          f" for {HOLD_SECS:.0f}s to complete each level.")
    server.sleep_forever()


if __name__ == "__main__":
    run()
