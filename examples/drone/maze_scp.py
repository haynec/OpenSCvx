"""6-DOF quadrotor maze navigation: JAX wavefront guess + SCP refinement.

Same maze / wavefront / corner-preserving *position* guess as the planar
double-integrator example (``examples/double_integrator/maze_rrt_scp.py``), but
with full 6-DOF quadrotor dynamics.  Attitude and thrust are seeded from
differential flatness (specific force → body +z), because identity attitude is
dynamically infeasible with body-z-only thrust on a moving path.  Altitude is
boxed inside the wall height so the drone cannot fly over or under walls; wall
footprints are enforced with batched CTCS infinity-norm constraints on (x, y).

Pipeline
--------
1. DFS maze on a 20×20 grid.
2. JAX wavefront planner + analytical LoS shortcutting (xy).
3. Vertex-preserving resample → xy position/velocity guess, lifted to cruise
   altitude with identity attitude and hover thrust.
4. SCP with 6-DOF dynamics and CTCS wall + state bounds.

Run::

    python examples/drone/maze_rrt_scp.py

Viser
-----
* ``:8081`` — wavefront / initial-guess animation (same phases as the planar maze).
* ``:8080`` — SCP trajectory with quadrotor mesh + follow camera (GUI sliders).
"""

from __future__ import annotations

import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import viser
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import openscvx as ox
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
START = np.array([0.5 * CELL_W, 0.5 * CELL_H])
GOAL = np.array([(GRID_COLS - 0.5) * CELL_W, (GRID_ROWS - 0.5) * CELL_H])

# ── Altitude band (keep COM inside wall height) ────────────────────────────────
WALL_H = 2.5
Z_CRUISE = 1.25
Z_MIN = 0.0          # coincides with the maze floor
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

# ── Maze generation ────────────────────────────────────────────────────────────


def _make_maze_walls(
    n_cols: int = GRID_COLS,
    n_rows: int = GRID_ROWS,
    cell_w: float = CELL_W,
    cell_h: float = CELL_H,
    wall_t: float = WALL_T,
    seed: int = MAZE_SEED,
) -> tuple[list[tuple[float, float, float, float]], np.ndarray, np.ndarray]:
    """DFS-backtracking perfect maze → wall rectangles (x0, y0, x1, y1)."""
    rng = np.random.default_rng(seed)
    h_walls = np.ones((n_rows - 1, n_cols), dtype=bool)
    v_walls = np.ones((n_rows, n_cols - 1), dtype=bool)
    visited = np.zeros((n_rows, n_cols), dtype=bool)

    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        r, c = stack[-1]
        nbrs: list[tuple[str, int, int]] = []
        if r > 0 and not visited[r - 1, c]:
            nbrs.append(("D", r - 1, c))
        if r < n_rows - 1 and not visited[r + 1, c]:
            nbrs.append(("U", r + 1, c))
        if c > 0 and not visited[r, c - 1]:
            nbrs.append(("L", r, c - 1))
        if c < n_cols - 1 and not visited[r, c + 1]:
            nbrs.append(("R", r, c + 1))
        if nbrs:
            d, nr, nc = nbrs[rng.integers(len(nbrs))]
            if d in ("U", "D"):
                h_walls[min(r, nr), c] = False
            else:
                v_walls[r, min(c, nc)] = False
            visited[nr, nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    half_t = wall_t / 2.0
    walls: list[tuple[float, float, float, float]] = []
    for r in range(n_rows - 1):
        yc = (r + 1) * cell_h
        for c in range(n_cols):
            if h_walls[r, c]:
                walls.append((c * cell_w, yc - half_t, (c + 1) * cell_w, yc + half_t))
    for r in range(n_rows):
        for c in range(n_cols - 1):
            if v_walls[r, c]:
                xc = (c + 1) * cell_w
                walls.append((xc - half_t, r * cell_h, xc + half_t, (r + 1) * cell_h))
    return walls, h_walls, v_walls


print("Generating maze …")
MAZE_WALLS, _H_WALLS, _V_WALLS = _make_maze_walls()
print(
    f"  {len(MAZE_WALLS)} wall segments "
    f"({GRID_COLS}×{GRID_ROWS} grid, {GRID_COLS * GRID_ROWS} cells)"
)
_WALL_ARR = np.array(MAZE_WALLS, dtype=np.float64)

# ── Path planner ───────────────────────────────────────────────────────────────

_SMOOTH_CLEARANCE = 0.06
PLAN_RES = 8
PLAN_CLEARANCE = 0.08
SHORTCUT_CLEAR = 0.10
_BFS_INF = 1 << 30


def _segment_collides(
    p1: np.ndarray,
    p2: np.ndarray,
    clearance: float = _SMOOTH_CLEARANCE,
) -> bool:
    """Exact analytical segment–AABB collision (slab method)."""
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    wx0 = _WALL_ARR[:, 0] - clearance
    wy0 = _WALL_ARR[:, 1] - clearance
    wx1 = _WALL_ARR[:, 2] + clearance
    wy1 = _WALL_ARR[:, 3] + clearance
    n = len(_WALL_ARR)
    t_min = np.zeros(n)
    t_max = np.ones(n)

    if abs(dx) < 1e-12:
        t_max = np.where((p1[0] < wx0) | (p1[0] > wx1), -1.0, t_max)
    else:
        ta = (wx0 - p1[0]) / dx
        tb = (wx1 - p1[0]) / dx
        t_min = np.maximum(t_min, np.minimum(ta, tb))
        t_max = np.minimum(t_max, np.maximum(ta, tb))

    if abs(dy) < 1e-12:
        t_max = np.where((p1[1] < wy0) | (p1[1] > wy1), -1.0, t_max)
    else:
        ta = (wy0 - p1[1]) / dy
        tb = (wy1 - p1[1]) / dy
        t_min = np.maximum(t_min, np.minimum(ta, tb))
        t_max = np.minimum(t_max, np.maximum(ta, tb))

    return bool((t_min <= t_max).any())


def _build_occupancy(lo: np.ndarray, hi: np.ndarray, res: int, clearance: float) -> np.ndarray:
    nx = int(round((hi[0] - lo[0]) * res))
    ny = int(round((hi[1] - lo[1]) * res))
    occ = np.zeros((ny, nx), dtype=bool)
    for x0, y0, x1, y1 in _WALL_ARR:
        i0 = max(int(np.floor((x0 - clearance - lo[0]) * res)), 0)
        i1 = min(int(np.ceil((x1 + clearance - lo[0]) * res)), nx)
        j0 = max(int(np.floor((y0 - clearance - lo[1]) * res)), 0)
        j1 = min(int(np.ceil((y1 + clearance - lo[1]) * res)), ny)
        occ[j0:j1, i0:i1] = True
    return occ


def _relax(dist: jnp.ndarray, free: jnp.ndarray) -> jnp.ndarray:
    ny, nx = dist.shape
    pad = jnp.pad(dist, 1, constant_values=_BFS_INF)
    up = pad[0:ny, 1 : nx + 1]
    down = pad[2 : ny + 2, 1 : nx + 1]
    left = pad[1 : ny + 1, 0:nx]
    right = pad[1 : ny + 1, 2 : nx + 2]
    nmin = jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right))
    cand = jnp.minimum(dist, nmin + 1)
    return jnp.where(free, cand, _BFS_INF)


@partial(jax.jit, static_argnames=("start_rc", "max_iters"))
def _wavefront(
    free: jnp.ndarray,
    dist0: jnp.ndarray,
    start_rc: tuple[int, int],
    max_iters: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    si, sj = start_rc

    def cond(state):
        dist, it = state
        return jnp.logical_and(it < max_iters, dist[si, sj] >= _BFS_INF)

    def body(state):
        dist, it = state
        return _relax(dist, free), it + 1

    return jax.lax.while_loop(cond, body, (dist0, jnp.int32(0)))


def _descend_path(dist: np.ndarray, start_rc, goal_rc) -> list[tuple[int, int]]:
    ny, nx = dist.shape
    cur = tuple(int(v) for v in start_rc)
    goal = tuple(int(v) for v in goal_rc)
    cells = [cur]
    while cur != goal:
        r, c = cur
        target = dist[r, c] - 1
        nxt = None
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < ny and 0 <= cc < nx and dist[rr, cc] == target:
                nxt = (rr, cc)
                break
        if nxt is None:
            break
        cur = nxt
        cells.append(cur)
    return cells


def _shortcut_with_history(
    path: np.ndarray,
    clearance: float = SHORTCUT_CLEAR,
) -> tuple[np.ndarray, list[np.ndarray]]:
    smoothed = [path[0]]
    stages: list[np.ndarray] = [np.array([path[0]])]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and _segment_collides(path[i], path[j], clearance):
            j -= 1
        smoothed.append(path[j])
        stages.append(np.array(smoothed))
        i = j
    return np.array(smoothed), stages


def wavefront_solve(
    start: np.ndarray,
    goal: np.ndarray,
    *,
    record_history: bool = False,
    history_stride: int = 12,
) -> np.ndarray | tuple[np.ndarray, dict]:
    lo, hi = DOMAIN_LO, DOMAIN_HI
    occ = _build_occupancy(lo, hi, PLAN_RES, PLAN_CLEARANCE)
    ny, nx = occ.shape
    free = jnp.asarray(~occ)
    free_np = np.asarray(~occ)

    def _rc(p):
        col = min(max(int((p[0] - lo[0]) * PLAN_RES), 0), nx - 1)
        row = min(max(int((p[1] - lo[1]) * PLAN_RES), 0), ny - 1)
        return row, col

    start_rc = _rc(start)
    goal_rc = _rc(goal)
    dist0 = np.full((ny, nx), _BFS_INF, dtype=np.int32)
    dist0[goal_rc[0], goal_rc[1]] = 0

    dist_frames: list[np.ndarray] = []
    if record_history:
        dist_np = dist0.copy()
        dist_frames.append(dist_np.copy())
        sweep = 0
        max_iters = nx * ny
        while dist_np[start_rc[0], start_rc[1]] >= _BFS_INF and sweep < max_iters:
            dist_np = np.asarray(_relax(jnp.asarray(dist_np), free))
            sweep += 1
            if sweep % history_stride == 0 or dist_np[start_rc[0], start_rc[1]] < _BFS_INF:
                dist_frames.append(dist_np.copy())
        n_iter = sweep
        dist = dist_np
    else:
        dist_j, n_iter = _wavefront(free, jnp.asarray(dist0), start_rc, nx * ny)
        dist = np.asarray(dist_j)

    if dist[start_rc[0], start_rc[1]] >= _BFS_INF:
        raise RuntimeError("Wavefront planner failed to reach the start cell.")

    cells = _descend_path(dist, start_rc, goal_rc)
    pts = np.array(
        [[lo[0] + (c + 0.5) / PLAN_RES, lo[1] + (r + 0.5) / PLAN_RES] for r, c in cells]
    )
    raw_path = np.vstack([start.copy(), pts, goal.copy()])
    deduped = [raw_path[0]]
    for wp in raw_path[1:]:
        if np.linalg.norm(wp - deduped[-1]) > 1e-9:
            deduped.append(wp)
    raw_path = np.array(deduped)

    if record_history:
        plan_path, shortcut_stages = _shortcut_with_history(raw_path)
        descent_stages = [
            np.vstack(
                [
                    start,
                    np.array(
                        [
                            [lo[0] + (c + 0.5) / PLAN_RES, lo[1] + (r + 0.5) / PLAN_RES]
                            for r, c in cells[:k]
                        ]
                    ),
                ]
            )
            for k in range(1, len(cells) + 1)
        ]
        descent_stages[-1] = np.vstack([descent_stages[-1], goal.reshape(1, 2)])
        history = {
            "dist_frames": dist_frames,
            "free": free_np,
            "lo": lo.copy(),
            "res": PLAN_RES,
            "start_rc": start_rc,
            "goal_rc": goal_rc,
            "n_sweeps": int(n_iter),
            "descent_stages": descent_stages,
            "shortcut_stages": shortcut_stages,
            "raw_path": raw_path,
        }
        return plan_path, history

    return _shortcut_with_history(raw_path)[0]


def path_to_guess(
    path: np.ndarray,
    n_nodes: int,
    t_total: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-preserving resample → (position_xy, velocity_xy)."""
    K = len(path)
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)

    if n_nodes < K:
        arc = np.concatenate([[0.0], np.cumsum(seg_len)])
        arc /= arc[-1]
        s = np.linspace(0.0, 1.0, n_nodes)
        pos = np.column_stack([np.interp(s, arc, path[:, 0]), np.interp(s, arc, path[:, 1])])
    else:
        extra = n_nodes - K
        raw = seg_len / seg_len.sum() * extra
        interior = np.floor(raw).astype(int)
        deficit = extra - int(interior.sum())
        if deficit > 0:
            order = np.argsort(-(raw - interior))
            interior[order[:deficit]] += 1
        chunks: list[np.ndarray] = []
        for i in range(K - 1):
            chunks.append(path[i][None, :])
            if interior[i] > 0:
                ts = np.linspace(0.0, 1.0, interior[i] + 2)[1:-1, None]
                chunks.append(path[i] * (1.0 - ts) + path[i + 1] * ts)
        chunks.append(path[-1][None, :])
        pos = np.vstack(chunks)

    node_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    t_nodes = t_total * node_arc / node_arc[-1]
    vel = np.gradient(pos, t_nodes, axis=0)
    vel[0] = 0.0
    vel[-1] = 0.0
    return pos, vel


# ── Plan ───────────────────────────────────────────────────────────────────────
print("Planning (JAX wavefront) …")
_t0 = time.time()
plan_path, wf_history = wavefront_solve(START, GOAL, record_history=True)
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
_node_arc = np.concatenate(
    [[0.0], np.cumsum(np.linalg.norm(np.diff(pos_guess, axis=0), axis=1))]
)
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

print(
    f"  Flatness guess: thrust ∈ [{thrust_mag.min():.1f}, {thrust_mag.max():.1f}] N "
    f"(hover {HOVER_THRUST:.1f}), "
    f"max tilt ≈ {np.degrees(2.0 * np.arccos(np.clip(np.abs(att_guess[:, 0]), 0.0, 1.0))).max():.1f}°"
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
    constraints.extend([ox.ctcs(state <= state.max, penalty="huber", idx=0), ox.ctcs(state.min <= state, penalty="huber", idx=0)])

wall_centers = np.array(
    [((x0 + x1) / 2.0, (y0 + y1) / 2.0) for x0, y0, x1, y1 in MAZE_WALLS]
)
wall_inv_scales = np.array(
    [(2.0 / (x1 - x0), 2.0 / (y1 - y0)) for x0, y0, x1, y1 in MAZE_WALLS]
)
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
            lambda center, inv_scale: ox.linalg.Norm(
                inv_scale * (position - center), ord="inf"
            ),
            batch=[wall_centers_3d, wall_inv_scales_3d],
        ),
        penalty="huber", idx=1
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
        # "autotuner": ox.ConstantProximalWeight(),
    },
    discretizer=ox.DiscretizeLinearizeVectorize(diffrax_kwargs={"atol": 1e-5, "rtol": 1e-5}),
)

# ── Visualization helpers (Z-up: maze xy → world x,y ; altitude → z) ───────────

_GUESS_Z = Z_CRUISE
_PLAN_Z = Z_CRUISE + 0.05
_TRAJ_Z_FALLBACK = Z_CRUISE
_WF_FIELD_Z = 0.08
_WF_PATH_Z = Z_CRUISE + 0.15
_MARKER_R = 0.25


def _maze_xy_to_world(xy: np.ndarray, height: float) -> np.ndarray:
    """Maze (x, y) → world (x, y, z=height)."""
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim == 1:
        return np.array([xy[0], xy[1], height], dtype=np.float32)
    return np.column_stack(
        [xy[:, 0], xy[:, 1], np.full(len(xy), height, dtype=np.float32)]
    )


def _path_line_segments(xy: np.ndarray, height: float) -> np.ndarray:
    pts3 = _maze_xy_to_world(xy, height)
    if len(pts3) < 2:
        return np.zeros((1, 2, 3), dtype=np.float32)
    return np.stack([pts3[:-1], pts3[1:]], axis=1)


def _collapse_collinear(xy: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 3:
        return xy
    keep = [0]
    for i in range(1, len(xy) - 1):
        v1 = xy[i] - xy[keep[-1]]
        v2 = xy[i + 1] - xy[i]
        if abs(v1[0] * v2[1] - v1[1] * v2[0]) > tol:
            keep.append(i)
    keep.append(len(xy) - 1)
    return xy[keep]


def _walls_mesh(wall_h: float = WALL_H, z0: float = Z_MIN) -> tuple[np.ndarray, np.ndarray]:
    """Combined triangle mesh for maze walls (Z-up extrusion from floor)."""
    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    v_off = 0
    for x0, y0, x1, y1 in MAZE_WALLS:
        v = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y0, wall_h],
                [x0, y0, wall_h],
                [x0, y1, z0],
                [x1, y1, z0],
                [x1, y1, wall_h],
                [x0, y1, wall_h],
            ],
            dtype=np.float32,
        )
        f = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 3, 7],
                [0, 7, 4],
                [1, 5, 6],
                [1, 6, 2],
                [3, 2, 6],
                [3, 6, 7],
            ],
            dtype=np.uint32,
        ) + v_off
        all_v.append(v)
        all_f.append(f)
        v_off += 8
    return np.vstack(all_v), np.vstack(all_f)


def _add_maze_scene(server: viser.ViserServer) -> None:
    lo, hi = DOMAIN_LO, DOMAIN_HI
    z0 = float(Z_MIN)
    floor_v = np.array(
        [
            [lo[0], lo[1], z0],
            [hi[0], lo[1], z0],
            [hi[0], hi[1], z0],
            [lo[0], hi[1], z0],
        ],
        dtype=np.float32,
    )
    floor_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    server.scene.add_mesh_simple(
        "/floor", floor_v, floor_f, color=(230, 220, 200), side="double"
    )
    w_verts, w_faces = _walls_mesh()
    server.scene.add_mesh_simple(
        "/walls",
        w_verts,
        w_faces,
        color=(75, 75, 85),
        flat_shading=True,
        side="double",
    )
    server.scene.add_icosphere(
        "/start",
        radius=_MARKER_R,
        color=(30, 200, 80),
        position=(float(START[0]), float(START[1]), Z_CRUISE),
    )
    server.scene.add_icosphere(
        "/goal",
        radius=_MARKER_R,
        color=(220, 50, 50),
        position=(float(GOAL[0]), float(GOAL[1]), Z_CRUISE),
    )


def _dist_field_cloud(
    dist: np.ndarray,
    free: np.ndarray,
    lo: np.ndarray,
    res: int,
    height: float = _WF_FIELD_Z,
) -> tuple[np.ndarray, np.ndarray]:
    reached = free & (dist < _BFS_INF)
    rows, cols = np.where(reached)
    if len(rows) == 0:
        return np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.uint8)
    xs = lo[0] + (cols + 0.5) / res
    ys = lo[1] + (rows + 0.5) / res
    pts = np.column_stack(
        [
            xs.astype(np.float32),
            ys.astype(np.float32),
            np.full(len(xs), height, dtype=np.float32),
        ]
    )
    d = dist[reached].astype(np.float64)
    d_norm = 1.0 - (d / max(float(d.max()), 1.0))
    rgba = plt.cm.plasma(d_norm)
    colors = (rgba[:, :3] * 255.0).astype(np.uint8)
    return pts, colors


def animate_wavefront_viser(
    wf_history: dict,
    plan_path: np.ndarray,
    guess_path_xy: np.ndarray,
    *,
    port: int = 8081,
) -> viser.ViserServer:
    """Same wavefront animation phases as the planar maze, in Z-up."""
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    print(f"  Wavefront animation → http://localhost:{port}")

    _add_maze_scene(server)

    lo = wf_history["lo"]
    free = wf_history["free"]
    res = wf_history["res"]
    raw_path = np.asarray(wf_history["raw_path"], dtype=np.float64)
    dist_frames = wf_history["dist_frames"]
    shortcut_stages = [
        np.asarray(s, dtype=np.float64)
        for s in wf_history["shortcut_stages"]
        if len(s) >= 2
    ]
    if not shortcut_stages:
        shortcut_stages = [np.asarray(plan_path, dtype=np.float64)]

    n_wf = min(len(dist_frames), 60)
    n_des = min(max(len(raw_path) - 1, 2), 80)
    n_sc = min(len(shortcut_stages), 50)
    wf_idx = np.unique(np.linspace(0, len(dist_frames) - 1, n_wf).astype(int))
    des_count = np.unique(np.linspace(2, len(raw_path), n_des).astype(int))
    sc_idx = np.unique(np.linspace(0, len(shortcut_stages) - 1, n_sc).astype(int))

    frames: list[tuple[str, object]] = [("wavefront", dist_frames[i]) for i in wf_idx]
    frames.extend(("descent", int(k)) for k in des_count)
    frames.extend(("shortcut", shortcut_stages[i]) for i in sc_idx)
    frames.append(("guess", guess_path_xy))

    phase_duration = {"wavefront": 8.0, "descent": 7.0, "shortcut": 6.0, "guess": 2.0}
    t_arr = np.zeros(len(frames), dtype=np.float64)
    t = 0.0
    i = 0
    while i < len(frames):
        phase = frames[i][0]
        j = i
        while j < len(frames) and frames[j][0] == phase:
            j += 1
        n_phase = j - i
        dt = phase_duration[phase] / max(n_phase - 1, 1)
        for k in range(n_phase):
            t_arr[i + k] = t + k * dt
        t = t_arr[j - 1] + 0.25
        i = j

    dist_final = dist_frames[-1]
    field_final_pts, field_final_cols = _dist_field_cloud(dist_final, free, lo, res)
    field_dim_cols = (field_final_cols.astype(np.float32) * 0.22).astype(np.uint8)
    raw_disp = _collapse_collinear(raw_path)
    plan_segs = _path_line_segments(plan_path, _WF_PATH_Z)
    ghost_segs = _path_line_segments(raw_disp, _WF_PATH_Z * 0.55)

    field_cloud = server.scene.add_point_cloud(
        "/wavefront/reached",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        point_size=0.04,
    )
    path_line = server.scene.add_line_segments(
        "/wavefront/path",
        points=np.zeros((1, 2, 3), dtype=np.float32),
        colors=np.array([40, 140, 255], dtype=np.uint8),
        line_width=5.0,
    )
    path_line.visible = False
    path_cloud = server.scene.add_point_cloud(
        "/wavefront/path_pts",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.array([[40, 140, 255]], dtype=np.uint8),
        point_size=0.11,
    )
    path_cloud.visible = False
    ghost_line = server.scene.add_line_segments(
        "/wavefront/raw_ghost",
        points=ghost_segs,
        colors=np.array([90, 90, 100], dtype=np.uint8),
        line_width=1.5,
    )
    ghost_line.visible = False
    guess_line = server.scene.add_line_segments(
        "/wavefront/guess",
        points=_path_line_segments(guess_path_xy, _GUESS_Z),
        colors=np.array([255, 180, 40], dtype=np.uint8),
        line_width=2.5,
    )
    guess_line.visible = False

    phase_md = server.gui.add_markdown("Phase: (press Play)")

    def _set_path(xy: np.ndarray, rgb: tuple[int, int, int], *, segs=None) -> None:
        xy = np.asarray(xy, dtype=np.float64)
        if len(xy) < 2:
            path_line.visible = False
            path_cloud.visible = False
            return
        if segs is None:
            segs = _path_line_segments(xy, _WF_PATH_Z)
        pts3 = _maze_xy_to_world(xy, _WF_PATH_Z)
        path_line.points = segs
        path_line.colors = np.array(rgb, dtype=np.uint8)
        path_line.visible = True
        path_cloud.points = pts3
        path_cloud.colors = np.broadcast_to(np.array(rgb, dtype=np.uint8), (len(pts3), 3)).copy()
        path_cloud.visible = True

    def _update(frame_idx: int) -> None:
        phase, payload = frames[frame_idx]
        if phase == "wavefront":
            pts, cols = _dist_field_cloud(payload, free, lo, res)
            field_cloud.points = pts
            field_cloud.colors = cols
            field_cloud.visible = True
            path_line.visible = False
            path_cloud.visible = False
            ghost_line.visible = False
            guess_line.visible = False
            n_wf_frames = sum(1 for fr in frames if fr[0] == "wavefront")
            wf_frame = sum(1 for fr in frames[: frame_idx + 1] if fr[0] == "wavefront")
            phase_md.content = (
                f"**Phase 1 — wavefront flood**  \n"
                f"Keyframe {wf_frame}/{n_wf_frames}  "
                f"(planner sweeps: {wf_history['n_sweeps']})"
            )
        elif phase == "descent":
            k = int(payload)
            field_cloud.points = field_final_pts
            field_cloud.colors = field_dim_cols
            field_cloud.visible = True
            ghost_line.visible = False
            guess_line.visible = False
            _set_path(_collapse_collinear(raw_path[:k]), (40, 140, 255))
            phase_md.content = (
                f"**Phase 2 — steepest descent**  \n"
                f"{k} / {len(raw_path)} cells along cost-to-go gradient"
            )
        elif phase == "shortcut":
            field_cloud.points = field_final_pts
            field_cloud.colors = field_dim_cols
            field_cloud.visible = True
            ghost_line.visible = True
            guess_line.visible = False
            _set_path(payload, (60, 220, 120))
            phase_md.content = (
                f"**Phase 3 — line-of-sight shortcut**  \n"
                f"{len(payload)} waypoints  "
                f"(grey = raw descent, green = shortcut)"
            )
        else:
            field_cloud.points = field_final_pts
            field_cloud.colors = (field_final_cols.astype(np.float32) * 0.12).astype(np.uint8)
            field_cloud.visible = True
            ghost_line.visible = False
            guess_line.visible = True
            _set_path(plan_path, (40, 140, 255), segs=plan_segs)
            phase_md.content = (
                f"**Phase 4 — SCP initial guess**  \n"
                f"{len(guess_path_xy)} nodes "
                f"(vertex-preserving resample of {len(plan_path)}-pt plan)"
            )

    add_animation_controls(
        server, t_arr, [_update], loop=True, folder_name="Wavefront Animation"
    )

    scene_center = np.array(
        [
            0.5 * (DOMAIN_LO[0] + DOMAIN_HI[0]),
            0.5 * (DOMAIN_LO[1] + DOMAIN_HI[1]),
            WALL_H,
        ],
        dtype=np.float32,
    )
    server.initial_camera.position = tuple(scene_center + np.array([0.0, -25.0, 20.0]))
    server.initial_camera.look_at = tuple(scene_center)
    server.initial_camera.up = (0.0, 0.0, 1.0)
    _update(0)
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
        + s_p * np.array(
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
    server.scene.set_up_direction("+z")
    print(f"  Trajectory animation → http://localhost:{port}")
    print("  Press Play in the Animation folder.  Ctrl-C to exit.")

    _add_maze_scene(server)

    guess_segs = _path_line_segments(guess_path_xy, _GUESS_Z)
    server.scene.add_line_segments(
        "/initial_guess",
        guess_segs,
        np.tile(np.array([255, 180, 40], dtype=np.uint8), (len(guess_segs), 2, 1)),
        line_width=2.5,
    )
    plan_segs = _path_line_segments(plan_path, _PLAN_Z)
    server.scene.add_line_segments(
        "/plan_path",
        plan_segs,
        np.tile(np.array([30, 100, 220], dtype=np.uint8), (len(plan_segs), 2, 1)),
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

    # Full path ghost
    if len(pos_traj) >= 2:
        ghost_segs = np.stack([pos_traj[:-1], pos_traj[1:]], axis=1).astype(np.float32)
        server.scene.add_line_segments(
            "/scp_trajectory/full",
            ghost_segs,
            np.tile(np.array([120, 40, 40], dtype=np.uint8), (len(ghost_segs), 2, 1)),
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


def plot_results_mpl(
    plan_path: np.ndarray,
    guess_path_xy: np.ndarray,
    results,
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """Top-down matplotlib figure (xy) + speed / thrust."""
    if save_path is None:
        save_path = os.path.join(current_dir, "maze_rrt_scp_result.png")

    traj = results.trajectory
    pos = np.asarray(traj["position"], dtype=float)
    vel = np.asarray(traj["velocity"], dtype=float)
    thr = np.asarray(traj["thrust_force"], dtype=float)
    t = np.asarray(traj["time"], dtype=float).reshape(-1)
    speed = np.linalg.norm(vel, axis=1)

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.0, 1.0, 1.0], hspace=0.28, wspace=0.22)
    ax_maze = fig.add_subplot(gs[0, :])
    ax_spd = fig.add_subplot(gs[1, :])
    ax_thr = fig.add_subplot(gs[2, :])

    rects = [Rectangle((x0, y0), x1 - x0, y1 - y0) for x0, y0, x1, y1 in _WALL_ARR]
    ax_maze.add_collection(
        PatchCollection(rects, facecolor=(0.28, 0.28, 0.32), edgecolor="none")
    )
    ax_maze.plot(
        guess_path_xy[:, 0],
        guess_path_xy[:, 1],
        "-",
        color="#ffb428",
        lw=1.2,
        label=f"initial guess ({len(guess_path_xy)} nodes)",
        zorder=3,
    )
    ax_maze.plot(
        plan_path[:, 0],
        plan_path[:, 1],
        "--",
        color="#1e64dc",
        lw=1.6,
        label="planner guide",
        zorder=3,
    )
    pts = pos[:, :2].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="autumn_r", zorder=4, linewidths=2.6)
    lc.set_array(t[:-1])
    ax_maze.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax_maze, fraction=0.025, pad=0.01)
    cbar.set_label("time [s]")
    ax_maze.plot(*START, "o", color="#1ec850", ms=13, mec="k", label="start", zorder=5)
    ax_maze.plot(*GOAL, "*", color="#dc3232", ms=20, mec="k", label="goal", zorder=5)
    t_f = t[-1] if len(t) else float("nan")
    conv = getattr(results, "converged", "?")
    ax_maze.set_title(
        f"6DoF maze — wavefront guess + SCP  (t_f ≈ {t_f:.1f} s, converged={conv})"
    )
    ax_maze.set_xlabel("x [m]")
    ax_maze.set_ylabel("y [m]")
    ax_maze.set_xlim(DOMAIN_LO[0], DOMAIN_HI[0])
    ax_maze.set_ylim(DOMAIN_LO[1], DOMAIN_HI[1])
    ax_maze.set_aspect("equal")
    ax_maze.legend(loc="upper left", framealpha=0.9)

    ax_spd.plot(t, speed, color="#111111", lw=1.6)
    ax_spd.axhline(V_MAX, color="tab:red", ls="--", lw=1.0, label=f"v_max = {V_MAX}")
    ax_spd.set_ylabel("‖v‖ [m/s]")
    ax_spd.set_xlabel("time [s]")
    ax_spd.grid(alpha=0.3)
    ax_spd.legend(loc="upper right")

    ax_thr.plot(t, thr[:, 2], color="tab:blue", lw=1.4, label="f_z (body)")
    ax_thr.axhline(MASS * abs(G_CONST), color="tab:gray", ls=":", lw=1.0, label="hover")
    ax_thr.set_ylabel("thrust [N]")
    ax_thr.set_xlabel("time [s]")
    ax_thr.grid(alpha=0.3)
    ax_thr.legend(loc="upper right")

    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    print(f"  Saved matplotlib figure → {save_path}")
    if show:
        plt.show()
    return save_path


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
    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
