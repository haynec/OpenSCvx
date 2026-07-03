"""2D maze navigation: JAX wavefront planner + SCP refinement.

A planar double integrator navigates a complicated 2-D maze from start to
goal.  The key challenge: the maze has no straight-line path, so naive SCP
initialisation always starts infeasible.

Pipeline
--------
1. **Maze generator** — DFS-backtracking perfect maze on a 40×40 cell grid.
2. **JAX wavefront planner** — the maze wall geometry is rasterised into a
   binary occupancy grid (with obstacle inflation for clearance), and a
   fully-vectorised, JIT-compiled BFS wavefront (a parallel Dijkstra / A*-style
   grid search) floods cost-to-go outward from the goal over every free cell
   simultaneously.  Each relaxation step is a handful of shifted-array ``min``
   operations, so the whole planner runs on-device with no Python loop.  A
   collision-free path is then recovered by steepest descent on the cost-to-go
   field and cleaned up with an exact analytical line-of-sight shortcutter.
3. **Guess constructor** — arc-length resamples the planned path to N nodes
   and finite-differences to produce a topologically correct position +
   velocity initial iterate for SCP.
4. **SCP** — OpenSCvx double-integrator formulation with CTCS infinity-norm
   obstacle constraints refines the guess into a dynamically feasible,
   time-optimal trajectory.

Run::

    python examples/double_integrator/maze_rrt_scp.py
"""

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
from openscvx import Problem

# ── Grid / domain parameters ──────────────────────────────────────────────────
GRID_COLS = 40
GRID_ROWS = 40
CELL_W    = 1.0          # cell width  [m]
CELL_H    = 1.0          # cell height [m]
WALL_T    = 0.10         # wall thickness (centred on grid edge) [m]
MAZE_SEED = 0            # RNG seed for DFS maze generator

DOMAIN_LO = np.array([0.0, 0.0])
DOMAIN_HI = np.array([GRID_COLS * CELL_W, GRID_ROWS * CELL_H])   # [40, 40]
START     = np.array([0.5 * CELL_W, 0.5 * CELL_H])                # [0.5, 0.5]
GOAL      = np.array([(GRID_COLS - 0.5) * CELL_W,
                       (GRID_ROWS - 0.5) * CELL_H])               # [39.5, 39.5]

# ── SCP parameters ────────────────────────────────────────────────────────────
N       = 800     # SCP discretisation nodes
T_MAX   = 1000.0   # upper bound on flight time [s]
T_GUESS = 1000.0   # assumed time for velocity finite-differencing [s]
V_MAX   = 8.0     # speed limit   [m/s]
F_MAX   = 5.0     # force limit   [m/s²]
MASS    = 1.0

# ── Maze generation ───────────────────────────────────────────────────────────

def _make_maze_walls(
    n_cols: int  = GRID_COLS,
    n_rows: int  = GRID_ROWS,
    cell_w: float = CELL_W,
    cell_h: float = CELL_H,
    wall_t: float = WALL_T,
    seed: int    = MAZE_SEED,
) -> tuple[list[tuple[float, float, float, float]], np.ndarray, np.ndarray]:
    """DFS-backtracking perfect maze.

    Returns (walls, h_walls_bool, v_walls_bool) where:
      walls        — list of (x0, y0, x1, y1) rectangles (centred on grid edges)
      h_walls_bool — (n_rows-1, n_cols) bool array, True = wall intact
      v_walls_bool — (n_rows, n_cols-1) bool array, True = wall intact
    """
    rng = np.random.default_rng(seed)

    # h_walls[r, c]: horizontal wall between cell (c,r) and (c,r+1)
    h_walls = np.ones((n_rows - 1, n_cols), dtype=bool)
    # v_walls[r, c]: vertical wall between cell (c,r) and (c+1,r)
    v_walls = np.ones((n_rows, n_cols - 1), dtype=bool)
    visited = np.zeros((n_rows, n_cols), dtype=bool)

    # Iterative DFS
    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        r, c = stack[-1]
        nbrs: list[tuple[str, int, int]] = []
        if r > 0          and not visited[r - 1, c    ]: nbrs.append(('D', r - 1, c    ))
        if r < n_rows - 1 and not visited[r + 1, c    ]: nbrs.append(('U', r + 1, c    ))
        if c > 0          and not visited[r    , c - 1]: nbrs.append(('L', r    , c - 1))
        if c < n_cols - 1 and not visited[r    , c + 1]: nbrs.append(('R', r    , c + 1))
        if nbrs:
            d, nr, nc = nbrs[rng.integers(len(nbrs))]
            if d in ('U', 'D'):
                h_walls[min(r, nr), c ] = False   # remove horizontal wall
            else:
                v_walls[r , min(c, nc)] = False   # remove vertical wall
            visited[nr, nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    half_t = wall_t / 2.0
    walls: list[tuple[float, float, float, float]] = []

    # Horizontal walls (centred at y = (r+1)*cell_h)
    for r in range(n_rows - 1):
        yc = (r + 1) * cell_h
        for c in range(n_cols):
            if h_walls[r, c]:
                walls.append((c * cell_w, yc - half_t, (c + 1) * cell_w, yc + half_t))

    # Vertical walls (centred at x = (c+1)*cell_w)
    for r in range(n_rows):
        for c in range(n_cols - 1):
            if v_walls[r, c]:
                xc = (c + 1) * cell_w
                walls.append((xc - half_t, r * cell_h, xc + half_t, (r + 1) * cell_h))

    return walls, h_walls, v_walls


print("Generating maze …")
MAZE_WALLS, _H_WALLS, _V_WALLS = _make_maze_walls()
print(f"  {len(MAZE_WALLS)} wall segments "
      f"({GRID_COLS}×{GRID_ROWS} grid, {GRID_COLS*GRID_ROWS} cells)")

# Precompute as numpy array for vectorised collision checks
_WALL_ARR = np.array(MAZE_WALLS, dtype=np.float64)   # (n_walls, 4)

# ── Path planner (BFS on cell graph + line-of-sight smoothing) ────────────────

_SMOOTH_CLEARANCE = 0.06   # safety margin for smoothed-segment collision check


def _segment_collides(
    p1: np.ndarray,
    p2: np.ndarray,
    clearance: float = _SMOOTH_CLEARANCE,
) -> bool:
    """Exact analytical segment–AABB collision test, vectorised over all walls.

    Uses the slab method: decomposes the segment into 1-D parametric intervals
    along each axis and intersects them.  Correct for any wall thickness — no
    sampling artefacts.

    Args:
        p1, p2: Endpoints of the segment (2,).
        clearance: Expand each wall rectangle by this margin before testing.

    Returns:
        True if the segment [p1, p2] intersects any wall (with clearance).
    """
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])

    # Wall bounds expanded by clearance
    wx0 = _WALL_ARR[:, 0] - clearance
    wy0 = _WALL_ARR[:, 1] - clearance
    wx1 = _WALL_ARR[:, 2] + clearance
    wy1 = _WALL_ARR[:, 3] + clearance

    n = len(_WALL_ARR)
    t_min = np.zeros(n)
    t_max = np.ones(n)

    # X slab: find t-interval where segment is inside [wx0, wx1]
    if abs(dx) < 1e-12:
        # Segment is axis-parallel in x; outside slab → no hit
        t_max = np.where((p1[0] < wx0) | (p1[0] > wx1), -1.0, t_max)
    else:
        ta = (wx0 - p1[0]) / dx
        tb = (wx1 - p1[0]) / dx
        t_min = np.maximum(t_min, np.minimum(ta, tb))
        t_max = np.minimum(t_max, np.maximum(ta, tb))

    # Y slab
    if abs(dy) < 1e-12:
        t_max = np.where((p1[1] < wy0) | (p1[1] > wy1), -1.0, t_max)
    else:
        ta = (wy0 - p1[1]) / dy
        tb = (wy1 - p1[1]) / dy
        t_min = np.maximum(t_min, np.minimum(ta, tb))
        t_max = np.minimum(t_max, np.maximum(ta, tb))

    return bool((t_min <= t_max).any())


# ── JAX wavefront planner (vectorised parallel BFS / A*-style grid search) ────
#
# Rather than exploiting the known cell-connectivity graph, we treat the maze
# purely as collision geometry: rasterise the walls into a binary occupancy
# grid and flood cost-to-go outward from the goal over *every* free cell at
# once.  A single BFS relaxation ``dist ← min(dist, min(shift(dist)) + 1)`` is
# a handful of shifted-array minimums, which JAX fuses into one vectorised
# kernel; iterating it inside ``lax.while_loop`` keeps the entire search
# on-device with no Python-level loop.  This is the data-parallel analogue of
# A* / Dijkstra and scales to fine grids trivially.

PLAN_RES       = 8        # planning-grid resolution [cells / metre]
PLAN_CLEARANCE = 0.08     # obstacle inflation applied when rasterising [m]
SHORTCUT_CLEAR = 0.10     # clearance for line-of-sight shortcutting [m]
_BFS_INF       = 1 << 30  # "unreached" sentinel for the cost-to-go field


def _build_occupancy(
    lo: np.ndarray,
    hi: np.ndarray,
    res: int,
    clearance: float,
) -> np.ndarray:
    """Rasterise MAZE_WALLS into a boolean occupancy grid.

    Each wall rectangle is inflated by ``clearance`` and then every grid cell
    whose extent overlaps the inflated rectangle is marked occupied (full
    span coverage via floor/ceil).  This guarantees thin walls never "leak"
    between grid lines regardless of alignment.

    Returns:
        occ: (ny, nx) bool array, indexed ``occ[row=y, col=x]``; True = blocked.
    """
    nx = int(round((hi[0] - lo[0]) * res))
    ny = int(round((hi[1] - lo[1]) * res))
    occ = np.zeros((ny, nx), dtype=bool)

    for x0, y0, x1, y1 in _WALL_ARR:
        i0 = max(int(np.floor((x0 - clearance - lo[0]) * res)), 0)
        i1 = min(int(np.ceil ((x1 + clearance - lo[0]) * res)), nx)
        j0 = max(int(np.floor((y0 - clearance - lo[1]) * res)), 0)
        j1 = min(int(np.ceil ((y1 + clearance - lo[1]) * res)), ny)
        occ[j0:j1, i0:i1] = True

    return occ


def _relax(dist: jnp.ndarray, free: jnp.ndarray) -> jnp.ndarray:
    """One BFS relaxation sweep over the whole grid (4-connectivity).

    ``dist`` is the current cost-to-go field; ``free`` masks traversable cells.
    Occupied cells are pinned to _BFS_INF so paths cannot cut through walls.
    """
    ny, nx = dist.shape
    pad   = jnp.pad(dist, 1, constant_values=_BFS_INF)
    up    = pad[0:ny,     1:nx + 1]
    down  = pad[2:ny + 2, 1:nx + 1]
    left  = pad[1:ny + 1, 0:nx]
    right = pad[1:ny + 1, 2:nx + 2]
    nmin  = jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right))
    cand  = jnp.minimum(dist, nmin + 1)
    return jnp.where(free, cand, _BFS_INF)


@partial(jax.jit, static_argnames=("start_rc", "max_iters"))
def _wavefront(
    free: jnp.ndarray,
    dist0: jnp.ndarray,
    start_rc: tuple[int, int],
    max_iters: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled BFS wavefront: flood cost-to-go until the start is reached.

    Runs relaxation sweeps inside ``lax.while_loop`` (one BFS layer per sweep),
    terminating as soon as the start cell has a finite cost-to-go or the
    iteration cap is hit.

    Args:
        free:      (ny, nx) bool traversability mask.
        dist0:     (ny, nx) int32 cost-to-go initialised to _BFS_INF with the
                   goal cell set to 0.
        start_rc:  (row, col) of the start cell (static — used for early stop).
        max_iters: hard cap on relaxation sweeps.

    Returns:
        (dist, n_iter) — final cost-to-go field and sweeps performed.
    """
    si, sj = start_rc

    def cond(state):
        dist, it = state
        return jnp.logical_and(it < max_iters, dist[si, sj] >= _BFS_INF)

    def body(state):
        dist, it = state
        return _relax(dist, free), it + 1

    dist, n_iter = jax.lax.while_loop(cond, body, (dist0, jnp.int32(0)))
    return dist, n_iter


def _descend_path(dist: np.ndarray, start_rc, goal_rc) -> list[tuple[int, int]]:
    """Recover a cell path by steepest descent on the goal-sourced cost-to-go.

    From the start cell, repeatedly step to the 4-neighbour whose cost-to-go is
    exactly one lower.  Guaranteed to reach the goal (cost 0) for a BFS field.
    """
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
        if nxt is None:                       # dead-end (should not happen)
            break
        cur = nxt
        cells.append(cur)
    return cells


def _shortcut(path: np.ndarray, clearance: float = SHORTCUT_CLEAR) -> np.ndarray:
    """Greedy line-of-sight shortcutting via the exact analytical slab test.

    Repeatedly connects the current waypoint to the furthest later waypoint
    reachable by a collision-free straight segment, collapsing the staircase
    produced by 4-connected grid descent into a clean polyline.
    """
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and _segment_collides(path[i], path[j], clearance):
            j -= 1
        smoothed.append(path[j])
        i = j
    return np.array(smoothed)


def wavefront_plan(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Plan a collision-free path with the JAX wavefront + shortcutting.

    Args:
        start: physical start position (2,).
        goal:  physical goal position  (2,).

    Returns:
        (K, 2) ordered collision-free waypoints from ``start`` to ``goal``.
    """
    lo, hi = DOMAIN_LO, DOMAIN_HI
    occ = _build_occupancy(lo, hi, PLAN_RES, PLAN_CLEARANCE)
    ny, nx = occ.shape
    free = jnp.asarray(~occ)

    def _rc(p):
        col = min(max(int((p[0] - lo[0]) * PLAN_RES), 0), nx - 1)
        row = min(max(int((p[1] - lo[1]) * PLAN_RES), 0), ny - 1)
        return row, col

    start_rc = _rc(start)
    goal_rc  = _rc(goal)

    dist0 = np.full((ny, nx), _BFS_INF, dtype=np.int32)
    dist0[goal_rc[0], goal_rc[1]] = 0

    dist, n_iter = _wavefront(free, jnp.asarray(dist0), start_rc, nx * ny)
    dist = np.asarray(dist)
    if dist[start_rc[0], start_rc[1]] >= _BFS_INF:
        raise RuntimeError("Wavefront planner failed to reach the start cell.")

    cells = _descend_path(dist, start_rc, goal_rc)

    # Cell indices → physical cell centres, bookended by the exact start/goal.
    pts = np.array([
        [lo[0] + (c + 0.5) / PLAN_RES, lo[1] + (r + 0.5) / PLAN_RES]
        for r, c in cells
    ])
    path = np.vstack([start.copy(), pts, goal.copy()])

    # Drop consecutive duplicates, then line-of-sight shortcut.
    deduped = [path[0]]
    for wp in path[1:]:
        if np.linalg.norm(wp - deduped[-1]) > 1e-9:
            deduped.append(wp)
    return _shortcut(np.array(deduped))


def path_to_guess(
    path: np.ndarray,
    n_nodes: int,
    t_total: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a collision-free polyline to N SCP nodes, PRESERVING vertices.

    Naive uniform arc-length resampling places nodes at evenly-spaced arc
    positions that almost never coincide with the path's corner vertices.  In a
    maze the polyline turns ~90° at every corner, so a chord between two nodes
    that straddle a corner cuts diagonally across the wall — i.e. the resampled
    guess passes straight through obstacles even though the *polyline* is
    collision-free.

    This resampler instead keeps every original vertex as a node and only ever
    *inserts* extra nodes in segment interiors (distributed across segments by
    length).  Every consecutive node pair therefore lies on a single original
    (collision-free) segment, so the guess is collision-free by construction —
    provided ``n_nodes >= len(path)``.

    Args:
        path: (K, 2) ordered collision-free waypoints.
        n_nodes: Number of SCP discretisation nodes (should be >= K).
        t_total: Assumed total time for finite-differencing velocity [s].

    Returns:
        position_guess: (n_nodes, 2)
        velocity_guess: (n_nodes, 2) — finite differences, endpoints zeroed
        force_guess:    (n_nodes, 2) — zeros
    """
    K = len(path)
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)

    if n_nodes < K:
        # Not enough nodes to keep every corner — fall back to uniform
        # resampling (guess may clip corners; increase N to avoid this).
        arc = np.concatenate([[0.0], np.cumsum(seg_len)])
        arc /= arc[-1]
        s = np.linspace(0.0, 1.0, n_nodes)
        pos = np.column_stack(
            [np.interp(s, arc, path[:, 0]), np.interp(s, arc, path[:, 1])]
        )
    else:
        # Distribute the (n_nodes - K) interior points across segments in
        # proportion to length, using largest-remainder rounding to hit the
        # target count exactly.
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

    # Velocity from arc-length-consistent finite differences (non-uniform
    # spacing → pass the per-node time stamps to np.gradient).
    node_arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
    )
    t_nodes = t_total * node_arc / node_arc[-1]
    vel = np.gradient(pos, t_nodes, axis=0)
    vel[0]  = 0.0
    vel[-1] = 0.0

    return pos, vel, np.zeros((len(pos), 2))


# ── Plan (JAX wavefront + analytical shortcutting) ────────────────────────────
# The wavefront floods cost-to-go from the goal over the rasterised free space;
# steepest descent then recovers a collision-free corridor path, which the
# analytical slab-test shortcutter collapses into a clean polyline.  Each
# consecutive segment is verified collision-free (with clearance).
print("Planning (JAX wavefront) …")
_t0 = time.time()
plan_path = wavefront_plan(START, GOAL)
_t_plan = time.time() - _t0
path_len = float(np.sum(np.linalg.norm(np.diff(plan_path, axis=0), axis=1)))
print(f"  Wavefront path: {len(plan_path)} waypoints, "
      f"length ≈ {path_len:.1f} m  ({_t_plan:.2f} s)")

# The maze solution has one genuine turn per corridor bend.  A collision-free
# guess must keep *every* corner as a node — otherwise chords between nodes cut
# across walls.  So the node count must be at least the number of waypoints;
# grow N to guarantee this (path_to_guess then preserves all corners exactly).
N = max(N, len(plan_path))
print(f"  SCP nodes: N = {N}")

pos_guess, vel_guess, frc_guess = path_to_guess(plan_path, N, T_GUESS)

# ── States ─────────────────────────────────────────────────────────────────────
position = ox.State("position", shape=(2,))
position.min   = DOMAIN_LO
position.max   = DOMAIN_HI
position.initial = START
position.final   = GOAL
position.guess   = pos_guess

velocity = ox.State("velocity", shape=(2,))
velocity.min   = np.array([-V_MAX, -V_MAX])
velocity.max   = np.array([ V_MAX,  V_MAX])
velocity.initial = np.array([0.0, 0.0])
velocity.final   = np.array([0.0, 0.0])
velocity.guess   = vel_guess

# ── Controls ───────────────────────────────────────────────────────────────────
force = ox.Control("force", shape=(2,))
force.max  = np.array([ F_MAX,  F_MAX])
force.min  = np.array([-F_MAX, -F_MAX])
force.guess = frc_guess

# ── Dynamics ───────────────────────────────────────────────────────────────────
dynamics = {
    "position": velocity,
    "velocity": (1.0 / MASS) * force,
}

# ── Constraints ────────────────────────────────────────────────────────────────
constraints = []

# CTCS bounds on states
for state in [position, velocity]:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Wall obstacle constraints: batched infinity-norm CTCS via Vmap.
# For wall with centre c and half-widths (a, b):
#   ‖diag(1/a, 1/b) @ (position − c)‖_∞ ≥ 1  ↔  stay outside the wall
wall_centers    = np.array([
    ((x0 + x1) / 2.0, (y0 + y1) / 2.0) for x0, y0, x1, y1 in MAZE_WALLS
])   # (n_walls, 2)
wall_inv_scales = np.array([
    (2.0 / (x1 - x0), 2.0 / (y1 - y0)) for x0, y0, x1, y1 in MAZE_WALLS
])   # (n_walls, 2)

wall_avoidance = ox.ctcs(
    np.ones(len(MAZE_WALLS))
    <= ox.Vmap(
        lambda center, inv_scale: ox.linalg.Norm(
            inv_scale * (position - center), ord="inf"
        ),
        batch=[wall_centers, wall_inv_scales],
    )
)
constraints.append(wall_avoidance)

# ── Time ───────────────────────────────────────────────────────────────────────
time_var = ox.Time(
    initial=0.0,
    final=ox.Minimize(T_MAX),
    min=0.0,
    max=T_MAX,
)

# ── Problem ────────────────────────────────────────────────────────────────────
problem = Problem(
    dynamics=dynamics,
    states=[position, velocity],
    controls=[force],
    time=time_var,
    constraints=constraints,
    N=N,
    float_dtype="float64",
    licq_max=1e-8,
    algorithm={
        "lam_cost": 1e-1,
        "lam_vc": 1e1,
        "lam_prox": 1e0,
        # "autotuner": ox.RampProximalWeight(ramp_factor=1.04, lam_prox_max=1e3),
        # "autotuner": ox.ConstantProximalWeight(),
        # "k_max": 20,
    },
)


# ── Visualisation (viser) ─────────────────────────────────────────────────────

_WALL_H  = 0.5    # visual height of wall boxes
_PATH_Y  = 0.08   # height of BFS guide path above floor
_TRAJ_Y  = 0.15   # height of SCP trajectory above floor
_MARKER_R = 0.4   # radius of start/goal spheres


def _walls_mesh(wall_h: float = _WALL_H) -> tuple[np.ndarray, np.ndarray]:
    """Build a single combined triangle mesh for all MAZE_WALLS.

    Coordinate convention: maze (x, y) → viser (x, up=Y, z).
    Only the top and four side faces are included (open bottom saves triangles).
    """
    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    v_off = 0
    for x0, y0, x1, y1 in MAZE_WALLS:
        v = np.array([
            [x0, 0,      y0], [x1, 0,      y0],   # 0 1  bottom front/back
            [x1, wall_h, y0], [x0, wall_h, y0],   # 2 3  top front/back
            [x0, 0,      y1], [x1, 0,      y1],   # 4 5
            [x1, wall_h, y1], [x0, wall_h, y1],   # 6 7
        ], dtype=np.float32)
        f = np.array([
            [0, 3, 2], [0, 2, 1],   # front face  (z = y0)
            [4, 5, 6], [4, 6, 7],   # back face   (z = y1)
            [0, 4, 7], [0, 7, 3],   # left face   (x = x0)
            [1, 2, 6], [1, 6, 5],   # right face  (x = x1)
            [3, 7, 6], [3, 6, 2],   # top face
        ], dtype=np.uint32) + v_off
        all_v.append(v)
        all_f.append(f)
        v_off += 8
    return np.vstack(all_v), np.vstack(all_f)


def plot_results_mpl(
    plan_path: np.ndarray,
    results,
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """Static matplotlib figure: top-down maze + trajectory, plus speed/force.

    Panels
    ------
    * Top    — maze walls (grey), planner guide path (blue dashed), SCP
               trajectory (coloured by time), start/goal markers.
    * Bottom — speed ‖v‖ and force components vs time, with their limits.

    Args:
        plan_path: (K, 2) planner guide waypoints.
        results:   solved OptimizationResults (post-processed).
        save_path: PNG output path.  Defaults to ``maze_rrt_scp_result.png``
                   next to this script.
        show:      whether to call ``plt.show()`` (blocks until closed).

    Returns:
        The path the figure was written to.
    """
    if save_path is None:
        save_path = os.path.join(current_dir, "maze_rrt_scp_result.png")

    traj  = results.trajectory
    pos   = np.asarray(traj["position"], dtype=float)
    vel   = np.asarray(traj["velocity"], dtype=float)
    frc   = np.asarray(traj["force"],    dtype=float)
    t     = np.asarray(traj["time"],     dtype=float).reshape(-1)
    speed = np.linalg.norm(vel, axis=1)

    fig = plt.figure(figsize=(12, 14))
    gs  = fig.add_gridspec(3, 2, height_ratios=[3.0, 1.0, 1.0], hspace=0.28,
                           wspace=0.22)
    ax_maze = fig.add_subplot(gs[0, :])
    ax_spd  = fig.add_subplot(gs[1, :])
    ax_frc  = fig.add_subplot(gs[2, :])

    # ── Maze walls (single PatchCollection — 1500+ rects) ─────────────────────
    rects = [
        Rectangle((x0, y0), x1 - x0, y1 - y0)
        for x0, y0, x1, y1 in _WALL_ARR
    ]
    ax_maze.add_collection(
        PatchCollection(rects, facecolor=(0.28, 0.28, 0.32), edgecolor="none")
    )

    # ── Planner guide path (dashed blue) ──────────────────────────────────────
    ax_maze.plot(
        plan_path[:, 0], plan_path[:, 1], "--", color="#1e64dc",
        lw=1.6, label="planner guide", zorder=3,
    )

    # ── SCP trajectory, coloured by time ──────────────────────────────────────
    pts  = pos.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="autumn_r", zorder=4, linewidths=2.6)
    lc.set_array(t[:-1])
    ax_maze.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax_maze, fraction=0.025, pad=0.01)
    cbar.set_label("time [s]")

    # ── Start / goal ──────────────────────────────────────────────────────────
    ax_maze.plot(*START, "o", color="#1ec850", ms=13, mec="k",
                 label="start", zorder=5)
    ax_maze.plot(*GOAL, "*", color="#dc3232", ms=20, mec="k",
                 label="goal", zorder=5)

    t_f = t[-1] if len(t) else float("nan")
    conv = getattr(results, "converged", "?")
    ax_maze.set_title(
        f"Maze navigation — JAX wavefront guess + SCP  "
        f"(t_f ≈ {t_f:.1f} s, converged={conv})"
    )
    ax_maze.set_xlabel("x [m]"); ax_maze.set_ylabel("y [m]")
    ax_maze.set_xlim(DOMAIN_LO[0], DOMAIN_HI[0])
    ax_maze.set_ylim(DOMAIN_LO[1], DOMAIN_HI[1])
    ax_maze.set_aspect("equal")
    ax_maze.legend(loc="upper left", framealpha=0.9)

    # ── Speed vs time ─────────────────────────────────────────────────────────
    ax_spd.plot(t, speed, color="#111111", lw=1.6)
    ax_spd.axhline(V_MAX, color="tab:red", ls="--", lw=1.0, label=f"v_max = {V_MAX}")
    ax_spd.set_ylabel("‖v‖ [m/s]"); ax_spd.set_xlabel("time [s]")
    ax_spd.grid(alpha=0.3); ax_spd.legend(loc="upper right")

    # ── Force components vs time ──────────────────────────────────────────────
    ax_frc.plot(t, frc[:, 0], color="tab:blue",   lw=1.4, label="fₓ")
    ax_frc.plot(t, frc[:, 1], color="tab:orange", lw=1.4, label="f_y")
    for lim in (F_MAX, -F_MAX):
        ax_frc.axhline(lim, color="tab:red", ls="--", lw=1.0)
    ax_frc.set_ylabel("force [m/s²]"); ax_frc.set_xlabel("time [s]")
    ax_frc.grid(alpha=0.3); ax_frc.legend(loc="upper right")

    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    print(f"  Saved matplotlib figure → {save_path}")
    if show:
        plt.show()
    return save_path


def plot_results(plan_path: np.ndarray, results) -> None:
    """Open an interactive viser scene with the maze, planner guide path, and
    SCP trajectory.  Blocks until Ctrl-C.

    Coordinate convention: maze (x, y) maps to viser (x, Y=height, z).
    """
    server = viser.ViserServer()
    print(f"  Viser viewer → open http://localhost:8080 in your browser")
    print("  Press Ctrl-C to exit.")

    # ── Floor ─────────────────────────────────────────────────────────────────
    lo, hi = DOMAIN_LO, DOMAIN_HI
    floor_v = np.array([
        [lo[0], 0, lo[1]], [hi[0], 0, lo[1]],
        [hi[0], 0, hi[1]], [lo[0], 0, hi[1]],
    ], dtype=np.float32)
    floor_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    server.scene.add_mesh_simple(
        "floor", floor_v, floor_f, color=(230, 220, 200), side="double"
    )

    # ── Walls ─────────────────────────────────────────────────────────────────
    w_verts, w_faces = _walls_mesh()
    server.scene.add_mesh_simple(
        "walls", w_verts, w_faces, color=(75, 75, 85),
        flat_shading=True, side="double",
    )

    # ── Planner guide path ──────────────────────────────────────────────────
    plan_3d = np.column_stack([
        plan_path[:, 0],
        np.full(len(plan_path), _PATH_Y),
        plan_path[:, 1],
    ]).astype(np.float32)
    server.scene.add_spline_catmull_rom(
        "plan_path", plan_3d, color=(30, 100, 220), line_width=3,
    )

    # ── SCP trajectory ────────────────────────────────────────────────────────
    pos_traj = np.asarray(results.trajectory["position"], dtype=np.float32)
    traj_3d = np.column_stack([
        pos_traj[:, 0],
        np.full(len(pos_traj), _TRAJ_Y),
        pos_traj[:, 1],
    ]).astype(np.float32)
    # line_segments expects (N, 2, 3) pairs and (N, 3) uint8 colours
    segs   = np.stack([traj_3d[:-1], traj_3d[1:]], axis=1)
    # viser requires colors shape (N, 2, 3) — one colour per segment endpoint
    colors = np.full((len(segs), 2, 3), [210, 50, 50], dtype=np.uint8)
    server.scene.add_line_segments("scp_trajectory", segs, colors, line_width=4)

    # ── Start / goal markers ──────────────────────────────────────────────────
    server.scene.add_icosphere(
        "start", radius=_MARKER_R, color=(30, 200, 80),
        position=(float(START[0]), _PATH_Y + _MARKER_R, float(START[1])),
    )
    server.scene.add_icosphere(
        "goal", radius=_MARKER_R, color=(220, 50, 50),
        position=(float(GOAL[0]), _PATH_Y + _MARKER_R, float(GOAL[1])),
    )

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────
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
    plot_results_mpl(plan_path, results)   # static matplotlib figure (blocks until closed)
    plot_results(plan_path, results)       # interactive viser scene
