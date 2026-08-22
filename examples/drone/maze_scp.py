"""6-DOF quadrotor maze navigation: JAX wavefront guess + SCP refinement.

Same maze, planner and corner-preserving *position* guess as the planar
double-integrator example (``examples/double_integrator/maze_rrt_scp.py``) —
both call ``examples/_maze.py`` — but with full 6-DOF quadrotor dynamics.
Attitude and thrust are seeded from differential flatness (specific force →
body +z), because identity attitude is dynamically infeasible with
body-z-only thrust on a moving path.  Altitude is boxed inside the wall height
so the drone cannot fly over or under walls; wall footprints are enforced with
batched CTCS infinity-norm constraints on (x, y).

Pipeline
--------
1. DFS maze on a 40×40 grid.
2. JAX wavefront planner + analytical LoS shortcutting (xy).
3. Vertex-preserving resample → xy position/velocity guess, lifted to cruise
   altitude and given a flatness-consistent attitude and thrust.
4. SCP with 6-DOF dynamics and CTCS wall + state bounds.

Run::

    python examples/drone/maze_scp.py

Viser
-----
* ``:8081`` — wavefront / initial-guess animation (same phases as the planar maze).
* ``:8080`` — SCP trajectory with quadrotor mesh + follow camera (GUI sliders).
"""

from __future__ import annotations

import os
import sys
import time

import jax.numpy as jnp
import numpy as np
import viser

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
from examples._maze import make_maze_walls, path_to_guess, wavefront_solve
from examples._maze_viz import (
    MazeHeights,
    add_maze_scene,
    animate_wavefront,
    path_line_segments,
    uniform_segment_colors,
)
from examples.animations._camera import chase_pose, look_at_wxyz
from examples.drone.logo_utils.quadrotor_mesh import make_quadrotor_mesh
from openscvx import Problem
from openscvx.plotting.viser import compute_velocity_colors
from openscvx.plotting.viser.animated import (
    add_animated_trail,
    add_animation_controls,
)

# ── Grid / domain ──────────────────────────────────────────────────────────────
GRID_COLS = 40
GRID_ROWS = 40
CELL_W = 1.0
CELL_H = 1.0
WALL_T = 0.10
MAZE_SEED = 0

DOMAIN_LO = np.array([0.0, 0.0])
DOMAIN_HI = np.array([GRID_COLS * CELL_W, GRID_ROWS * CELL_H])
DOMAIN = (DOMAIN_LO, DOMAIN_HI)
START = np.array([0.5 * CELL_W, 0.5 * CELL_H])
GOAL = np.array([(GRID_COLS - 0.5) * CELL_W, (GRID_ROWS - 0.5) * CELL_H])

# ── Altitude band (keep COM inside wall height) ────────────────────────────────
WALL_H = 2.5
Z_CRUISE = 1.25
Z_MIN = 0.0  # coincides with the maze floor
Z_MAX = WALL_H - 0.15

# ── SCP parameters ─────────────────────────────────────────────────────────────
N = 800
T_MAX = 2000.0
# Seed final-time below T_MAX; also used for FD velocity / flatness guess.
# Gentler cruise → milder corner accel → attitudes stay near hover.
T_GUESS = 1000.0
V_MAX = 6.0
MASS = 1.0
G_CONST = -9.18
GRAVITY = np.array([0.0, 0.0, G_CONST], dtype=np.float64)
THRUST_MAX = 4.179446268 * 9.81
HOVER_THRUST = MASS * abs(G_CONST)

# ── Maze + plan ────────────────────────────────────────────────────────────────
print("Generating maze …")
MAZE_WALLS = make_maze_walls(
    GRID_COLS, GRID_ROWS, cell_w=CELL_W, cell_h=CELL_H, wall_t=WALL_T, seed=MAZE_SEED
)
print(
    f"  {len(MAZE_WALLS)} wall segments "
    f"({GRID_COLS}×{GRID_ROWS} grid, {GRID_COLS * GRID_ROWS} cells)"
)

print("Planning (JAX wavefront) …")
_t0 = time.time()
plan_path, wf_history = wavefront_solve(MAZE_WALLS, START, GOAL, DOMAIN, record_history=True)
_t_plan = time.time() - _t0
path_len = float(np.sum(np.linalg.norm(np.diff(plan_path, axis=0), axis=1)))
print(f"  Wavefront path: {len(plan_path)} waypoints, length ≈ {path_len:.1f} m  ({_t_plan:.2f} s)")

N = max(N, len(plan_path))
T_GUESS = max(T_GUESS, path_len / 1.5)  # ~1.5 m/s mean ground speed
print(f"  SCP nodes: N = {N}, T_GUESS ≈ {T_GUESS:.1f} s")

pos_xy, vel_xy = path_to_guess(plan_path, N, T_GUESS)
pos_guess = np.column_stack([pos_xy, np.full(N, Z_CRUISE)])
vel_guess = np.column_stack([vel_xy, np.zeros(N)])

# Node times for finite-difference acceleration (same nonuniform grid as path_to_guess).
_node_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos_guess, axis=0), axis=1))])
_t_nodes = T_GUESS * _node_arc / _node_arc[-1]
accel_guess = np.gradient(vel_guess, _t_nodes, axis=0)


def _orientation_from_accel(accel: np.ndarray) -> np.ndarray:
    """Unit quaternion aligning body +z with specific thrust (diff. flatness).

    Identity attitude cannot produce horizontal acceleration with body-z-only
    thrust — a level guess is dynamically infeasible for any non-hover path.
    """
    thrust_dir = accel - GRAVITY
    norm = float(np.linalg.norm(thrust_dir))
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    z_des = thrust_dir / norm
    z_body = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_body, z_des)
    dot = float(np.dot(z_body, z_des))
    if dot < -0.999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    q = np.array([1.0 + dot, cross[0], cross[1], cross[2]])
    return q / np.linalg.norm(q)


att_guess = np.array([_orientation_from_accel(accel_guess[k]) for k in range(N)])
att_guess /= np.linalg.norm(att_guess, axis=1, keepdims=True)

# Body thrust magnitude m‖a − g‖, clipped to the actuator box; direction is +z.
thrust_mag = MASS * np.linalg.norm(accel_guess - GRAVITY[None, :], axis=1)
thrust_mag = np.clip(thrust_mag, 0.0, THRUST_MAX)
thrust_guess = np.column_stack([np.zeros(N), np.zeros(N), thrust_mag])

max_tilt_deg = np.degrees(2.0 * np.arccos(np.clip(np.abs(att_guess[:, 0]), 0.0, 1.0))).max()
print(
    f"  Flatness guess: thrust ∈ [{thrust_mag.min():.1f}, {thrust_mag.max():.1f}] N "
    f"(hover {HOVER_THRUST:.1f}), "
    f"max tilt ≈ {max_tilt_deg:.1f}°"
)

# ── States / controls ──────────────────────────────────────────────────────────
position = ox.State("position", shape=(3,))
position.min = np.array([DOMAIN_LO[0], DOMAIN_LO[1], Z_MIN])
position.max = np.array([DOMAIN_HI[0], DOMAIN_HI[1], Z_MAX])
position.initial = np.array([START[0], START[1], Z_CRUISE])
position.final = np.array([GOAL[0], GOAL[1], Z_CRUISE])
position.guess = pos_guess

velocity = ox.State("velocity", shape=(3,))
velocity.min = np.array([-V_MAX, -V_MAX, -V_MAX])
velocity.max = np.array([V_MAX, V_MAX, V_MAX])
velocity.initial = np.array([0.0, 0.0, 0.0])
velocity.final = np.array([0.0, 0.0, 0.0])
velocity.guess = vel_guess

attitude = ox.State("attitude", shape=(4,))
attitude.max = np.array([1.0, 1.0, 1.0, 1.0])
attitude.min = np.array([-1.0, -1.0, -1.0, -1.0])
attitude.initial = [ox.Free(1.0), ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]
attitude.final = [ox.Free(1.0), ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]
attitude.guess = att_guess

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = np.array([10.0, 10.0, 10.0])
angular_velocity.min = np.array([-10.0, -10.0, -10.0])
angular_velocity.initial = np.array([0.0, 0.0, 0.0])
angular_velocity.final = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0)]
angular_velocity.guess = np.zeros((N, 3))

thrust_force = ox.Control("thrust_force", shape=(3,))
thrust_force.max = np.array([0.0, 0.0, THRUST_MAX])
thrust_force.min = np.array([0.0, 0.0, 0.0])
thrust_force.guess = thrust_guess

torque = ox.Control("torque", shape=(3,))
torque.max = np.array([18.665, 18.665, 0.55562])
torque.min = np.array([-18.665, -18.665, -0.55562])
torque.guess = np.zeros((N, 3))

# ── Dynamics ───────────────────────────────────────────────────────────────────
J_b = jnp.array([1.0, 1.0, 1.0])
J_b_inv = 1.0 / J_b
J_b_diag = ox.linalg.Diag(J_b)
q_norm = ox.linalg.Norm(attitude)
attitude_normalized = attitude / q_norm

dynamics = {
    "position": velocity,
    "velocity": (1.0 / MASS) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
    + ox.Constant(np.array([0.0, 0.0, G_CONST], dtype=np.float64)),
    "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
    "angular_velocity": ox.linalg.Diag(J_b_inv)
    @ (torque - ox.spatial.SSM(angular_velocity) @ J_b_diag @ angular_velocity),
}

# ── Constraints ────────────────────────────────────────────────────────────────
states = [position, velocity, attitude, angular_velocity]
controls = [thrust_force, torque]
constraints = []
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max, penalty="huber", idx=0),
            ox.ctcs(state.min <= state, penalty="huber", idx=0),
        ]
    )

wall_centers = np.array([((x0 + x1) / 2.0, (y0 + y1) / 2.0) for x0, y0, x1, y1 in MAZE_WALLS])
wall_inv_scales = np.array([(2.0 / (x1 - x0), 2.0 / (y1 - y0)) for x0, y0, x1, y1 in MAZE_WALLS])
# Tall prism walls: half-height covers the altitude band so the infinity-norm
# CTCS reduces to the planar (x, y) footprint (same pattern as the 2D maze).
_z_mid = 0.5 * (Z_MIN + Z_MAX)
_z_half = 0.5 * (Z_MAX - Z_MIN) + 1.0
wall_centers_3d = np.column_stack([wall_centers, np.full(len(wall_centers), _z_mid)])
wall_inv_scales_3d = np.column_stack(
    [wall_inv_scales, np.full(len(wall_inv_scales), 1.0 / _z_half)]
)
constraints.append(
    ox.ctcs(
        np.ones(len(MAZE_WALLS))
        <= ox.Vmap(
            lambda center, inv_scale: ox.linalg.Norm(inv_scale * (position - center), ord="inf"),
            batch=[wall_centers_3d, wall_inv_scales_3d],
        ),
        penalty="huber",
        idx=1,
    )
)

time_var = ox.Time(
    initial=0.0,
    final=ox.Minimize(T_GUESS),  # seed cruise time, not T_MAX
    min=0.0,
    max=T_MAX,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time_var,
    constraints=constraints,
    N=N,
    float_dtype="float64",
    licq_max=1e-10,
    algorithm={
        "lam_cost": 2e-1,
        "lam_vc": 1e1,
        "lam_prox": 1e0,
        "k_max": 200,
    },
    discretizer=ox.DiscretizeLinearizeVectorize(diffrax_kwargs={"atol": 1e-5, "rtol": 1e-5}),
)

# ── Visualisation ──────────────────────────────────────────────────────────────
# The guess and the planner guide are flat overlays at cruise altitude; the SCP
# trajectory is drawn in true 3-D from the solved positions.

HEIGHTS = MazeHeights(
    wall=WALL_H,
    field=0.08,
    wavefront=Z_CRUISE + 0.15,
    guess=Z_CRUISE,
    plan=Z_CRUISE + 0.05,
)
_MARKER_R = 0.25


def _add_scene(server: viser.ViserServer) -> None:
    """Add the maze floor, walls and markers at this example's altitudes."""
    add_maze_scene(
        server,
        MAZE_WALLS,
        DOMAIN,
        wall_height=HEIGHTS.wall,
        start=(START[0], START[1], Z_CRUISE),
        goal=(GOAL[0], GOAL[1], Z_CRUISE),
        marker_radius=_MARKER_R,
    )


def animate_wavefront_viser(
    wf_history: dict,
    plan_path: np.ndarray,
    guess_path_xy: np.ndarray,
    *,
    port: int = 8081,
) -> viser.ViserServer:
    """Open the shared four-phase planner animation on its own server.

    Runs on ``port`` (default 8081) so it can coexist with the results viewer.
    """
    server = viser.ViserServer(port=port)
    print(f"  Wavefront animation → http://localhost:{port}")
    _add_scene(server)
    animate_wavefront(server, wf_history, plan_path, guess_path_xy, domain=DOMAIN, heights=HEIGHTS)
    return server


def _follow_camera_pose(
    pos: np.ndarray,
    vel: np.ndarray,
    *,
    back: float,
    side: float,
    up: float,
    yaw_deg: float,
    pitch_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chase cam with body-aligned offsets + yaw/pitch orientation offsets."""
    forward = np.asarray(vel, dtype=np.float64).copy()
    fwd_norm = np.linalg.norm(forward[:2])
    if fwd_norm < 1e-3:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        forward = np.array([forward[0], forward[1], 0.0], dtype=np.float64)
        forward /= np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= max(np.linalg.norm(right), 1e-9)

    subject = np.asarray(pos, dtype=np.float64)
    focus = subject + 0.5 * forward
    # Base chase behind the subject, then apply lateral/vertical offsets.
    cam_pos, _, _ = chase_pose(
        subject,
        focus,
        chase_distance=back,
        vertical_offset=up,
        up=world_up,
    )
    cam_pos = cam_pos + side * right

    # Orientation offsets: rotate look direction about world up (yaw) and right (pitch).
    look_dir = focus - cam_pos
    look_dir /= max(np.linalg.norm(look_dir), 1e-9)
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    c_y, s_y = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[c_y, -s_y, 0.0], [s_y, c_y, 0.0], [0.0, 0.0, 1.0]])
    look_dir = R_yaw @ look_dir
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    # Pitch about camera-right after yaw.
    cam_right = np.cross(look_dir, world_up)
    if np.linalg.norm(cam_right) < 1e-6:
        cam_right = right
    else:
        cam_right /= np.linalg.norm(cam_right)
    R_pitch = (
        c_p * np.eye(3)
        + s_p
        * np.array(
            [
                [0.0, -cam_right[2], cam_right[1]],
                [cam_right[2], 0.0, -cam_right[0]],
                [-cam_right[1], cam_right[0], 0.0],
            ]
        )
        + (1.0 - c_p) * np.outer(cam_right, cam_right)
    )
    look_dir = R_pitch @ look_dir
    look_at = cam_pos + look_dir
    wxyz = look_at_wxyz(cam_pos, look_at, world_up)
    return cam_pos, wxyz, look_at


def plot_results(
    plan_path: np.ndarray,
    guess_path_xy: np.ndarray,
    results,
    *,
    port: int = 8080,
    loop_animation: bool = True,
) -> viser.ViserServer:
    """SCP trajectory animation with quadrotor mesh + follow camera."""
    server = viser.ViserServer(port=port)
    print(f"  Trajectory animation → http://localhost:{port}")
    print("  Press Play in the Animation folder.  Ctrl-C to exit.")

    _add_scene(server)

    guess_segs = path_line_segments(guess_path_xy, HEIGHTS.guess)
    server.scene.add_line_segments(
        "/initial_guess",
        guess_segs,
        uniform_segment_colors(len(guess_segs), (255, 180, 40)),
        line_width=2.5,
    )
    plan_segs = path_line_segments(plan_path, HEIGHTS.plan)
    server.scene.add_line_segments(
        "/plan_path",
        plan_segs,
        uniform_segment_colors(len(plan_segs), (30, 100, 220)),
        line_width=2.0,
    )

    traj = results.trajectory
    pos_traj = np.asarray(traj["position"], dtype=np.float64)
    vel_traj = np.asarray(traj["velocity"], dtype=np.float64)
    att_traj = np.asarray(traj["attitude"], dtype=np.float64)
    traj_time = np.asarray(traj["time"], dtype=np.float64).reshape(-1)
    if traj_time.size != len(pos_traj):
        traj_time = np.linspace(
            0.0,
            float(traj_time[-1]) if traj_time.size else 1.0,
            len(pos_traj),
        )

    # Full path ghost (true 3-D, unlike the flat overlays above).
    if len(pos_traj) >= 2:
        ghost_segs = np.stack([pos_traj[:-1], pos_traj[1:]], axis=1).astype(np.float32)
        server.scene.add_line_segments(
            "/scp_trajectory/full",
            ghost_segs,
            uniform_segment_colors(len(ghost_segs), (120, 40, 40)),
            line_width=2.0,
        )

    trail_colors = compute_velocity_colors(vel_traj, fallback_length=len(pos_traj))
    _, update_trail = add_animated_trail(server, pos_traj.astype(np.float32), trail_colors)

    mesh_verts, mesh_faces = make_quadrotor_mesh(scale=1.0)
    mesh_handle = server.scene.add_mesh_simple(
        "/vehicle_mesh",
        vertices=np.asarray(mesh_verts, dtype=np.float32),
        faces=np.asarray(mesh_faces, dtype=np.uint32),
        color=(200, 200, 210),
        position=tuple(float(x) for x in pos_traj[0]),
        wxyz=tuple(float(x) for x in att_traj[0]),
    )

    def update_vehicle(frame_idx: int) -> None:
        mesh_handle.position = tuple(float(x) for x in pos_traj[frame_idx])
        mesh_handle.wxyz = tuple(float(x) for x in att_traj[frame_idx])

    # Follow-camera GUI (mutable offsets read each frame).
    cam_state = {
        "back": 3.5,
        "side": 0.0,
        "up": 1.5,
        "yaw": 0.0,
        "pitch": -12.0,
        "enabled": True,
    }

    with server.gui.add_folder("Follow Camera"):
        enable_cb = server.gui.add_checkbox("Enabled", initial_value=True)
        back_sl = server.gui.add_slider("Back [m]", min=0.5, max=15.0, step=0.1, initial_value=3.5)
        side_sl = server.gui.add_slider("Side [m]", min=-8.0, max=8.0, step=0.1, initial_value=0.0)
        up_sl = server.gui.add_slider("Up [m]", min=0.0, max=10.0, step=0.1, initial_value=1.5)
        yaw_sl = server.gui.add_slider(
            "Yaw offset [deg]", min=-180.0, max=180.0, step=1.0, initial_value=0.0
        )
        pitch_sl = server.gui.add_slider(
            "Pitch offset [deg]", min=-89.0, max=89.0, step=1.0, initial_value=-12.0
        )

    @enable_cb.on_update
    def _(_e) -> None:
        cam_state["enabled"] = bool(enable_cb.value)

    @back_sl.on_update
    def _(_e) -> None:
        cam_state["back"] = float(back_sl.value)

    @side_sl.on_update
    def _(_e) -> None:
        cam_state["side"] = float(side_sl.value)

    @up_sl.on_update
    def _(_e) -> None:
        cam_state["up"] = float(up_sl.value)

    @yaw_sl.on_update
    def _(_e) -> None:
        cam_state["yaw"] = float(yaw_sl.value)

    @pitch_sl.on_update
    def _(_e) -> None:
        cam_state["pitch"] = float(pitch_sl.value)

    def update_follow_camera(frame_idx: int) -> None:
        if not cam_state["enabled"]:
            return
        cam_pos, cam_wxyz, look_at = _follow_camera_pose(
            pos_traj[frame_idx],
            vel_traj[frame_idx],
            back=cam_state["back"],
            side=cam_state["side"],
            up=cam_state["up"],
            yaw_deg=cam_state["yaw"],
            pitch_deg=cam_state["pitch"],
        )
        for client in server.get_clients().values():
            client.camera.position = tuple(float(x) for x in cam_pos)
            client.camera.wxyz = tuple(float(x) for x in cam_wxyz)
            client.camera.look_at = tuple(float(x) for x in look_at)

    # Seed initial / connect cameras.
    cam0, wxyz0, look0 = _follow_camera_pose(
        pos_traj[0],
        vel_traj[0],
        back=cam_state["back"],
        side=cam_state["side"],
        up=cam_state["up"],
        yaw_deg=cam_state["yaw"],
        pitch_deg=cam_state["pitch"],
    )
    server.initial_camera.position = tuple(float(x) for x in cam0)
    server.initial_camera.wxyz = tuple(float(x) for x in wxyz0)
    server.initial_camera.look_at = tuple(float(x) for x in look0)
    server.initial_camera.up = (0.0, 0.0, 1.0)

    @server.on_client_connect
    def _on_connect(client) -> None:
        cam_pos, cam_wxyz, look_at = _follow_camera_pose(
            pos_traj[0],
            vel_traj[0],
            back=cam_state["back"],
            side=cam_state["side"],
            up=cam_state["up"],
            yaw_deg=cam_state["yaw"],
            pitch_deg=cam_state["pitch"],
        )
        client.camera.position = tuple(float(x) for x in cam_pos)
        client.camera.wxyz = tuple(float(x) for x in cam_wxyz)
        client.camera.look_at = tuple(float(x) for x in look_at)

    add_animation_controls(
        server,
        traj_time,
        [update_trail, update_vehicle, update_follow_camera],
        loop=loop_animation,
    )
    update_trail(0)
    update_vehicle(0)
    update_follow_camera(0)

    with server.gui.add_folder("Legend"):
        server.gui.add_markdown(
            f"**Orange** — SCP initial guess ({len(guess_path_xy)} nodes)  \n"
            f"**Blue** — planner shortcut ({len(plan_path)} waypoints)  \n"
            f"**Dark red** — full SCP solution  \n"
            f"**Mesh + trail** — animated quadrotor (press Play)  \n"
            f"**Follow Camera** — chase with position/orientation sliders  \n"
            f"Wavefront animation: port 8081"
        )
    return server


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initializing problem …")
    problem.initialize()

    print("Solving …")
    results = problem.solve()
    results = problem.post_process()

    converged = getattr(results, "converged", "?")
    print(f"  Converged: {converged}")
    try:
        print(f"  Final time: {results.t_f:.3f} s")
    except AttributeError:
        pass

    print("Plotting …")
    animate_wavefront_viser(wf_history, plan_path, pos_xy, port=8081)
    traj_server = plot_results(plan_path, pos_xy, results, port=8080)
    traj_server.sleep_forever()
