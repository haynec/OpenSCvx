"""Maze generation and JAX wavefront path planning, shared by the maze examples.

Why an examples tree ships a planner
------------------------------------
SCP refines an initial iterate; it does not search.  In a maze there is no
straight line from start to goal, so the usual "interpolate the boundary
conditions" guess is not merely suboptimal — it threads straight through walls,
and every subsequent linearisation is taken about a point deep inside the
infeasible set.  What SCP needs first is a *topologically* correct guess: a
path that goes around the walls the way the solution has to.  That is what this
module produces, and nothing more — the dynamics, the constraints and the
optimisation all stay in the example files.

The planner
-----------
The maze is treated purely as collision geometry.  Walls are rasterised into a
binary occupancy grid (inflated for clearance), and a cost-to-go field is
flooded outward from the goal over *every* free cell at once.  One BFS
relaxation is ``dist ← min(dist, min(shift(dist)) + 1)`` — a handful of
shifted-array minimums that JAX fuses into a single kernel — iterated inside
``lax.while_loop`` so the whole search stays on-device with no Python loop.
This is the data-parallel analogue of Dijkstra / A* and scales to fine grids
trivially.  A path is then recovered by steepest descent on the cost-to-go
field, cleaned up by an exact analytical line-of-sight shortcutter, and
resampled onto the SCP node grid in a way that preserves every corner.

Consumers: ``examples/double_integrator/maze_rrt_scp.py`` (planar) and
``examples/drone/maze_scp.py`` (6-DOF).  Rendering lives in
``examples/_maze_viz.py``; this module is pure NumPy/JAX geometry.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# ── Planner tuning ─────────────────────────────────────────────────────────────
PLAN_RES = 8  # planning-grid resolution [cells / metre]
PLAN_CLEARANCE = 0.08  # obstacle inflation applied when rasterising [m]
SHORTCUT_CLEARANCE = 0.10  # clearance for line-of-sight shortcutting [m]

BFS_INF = 1 << 30  # "unreached" sentinel: cost-to-go of a cell not yet flooded


# ── Maze generation ────────────────────────────────────────────────────────────


def make_maze_walls(
    n_cols: int,
    n_rows: int,
    *,
    cell_w: float = 1.0,
    cell_h: float = 1.0,
    wall_t: float = 0.10,
    seed: int = 0,
) -> np.ndarray:
    """Generate a perfect maze by DFS backtracking and return its walls.

    "Perfect" means exactly one path connects any two cells: the maze has no
    loops and no isolated regions, so a plan always exists and is unique up to
    shortcutting.

    Args:
        n_cols, n_rows: Maze size in cells.  The domain spans
            ``[0, n_cols * cell_w] × [0, n_rows * cell_h]``.
        cell_w, cell_h: Cell size [m].
        wall_t: Wall thickness [m], centred on the grid edge it occupies.
        seed: RNG seed for the DFS.

    Returns:
        ``(n_walls, 4)`` float64 array of axis-aligned rectangles
        ``(x0, y0, x1, y1)`` — the form both the collision tests and the
        infinity-norm CTCS constraints consume directly.
    """
    rng = np.random.default_rng(seed)

    # h_walls[r, c]: horizontal wall between cell (c, r) and (c, r+1)
    h_walls = np.ones((n_rows - 1, n_cols), dtype=bool)
    # v_walls[r, c]: vertical wall between cell (c, r) and (c+1, r)
    v_walls = np.ones((n_rows, n_cols - 1), dtype=bool)
    visited = np.zeros((n_rows, n_cols), dtype=bool)

    # Iterative DFS: carve a wall whenever we step into an unvisited cell.
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
    # Horizontal walls, centred at y = (r + 1) * cell_h.
    for r in range(n_rows - 1):
        yc = (r + 1) * cell_h
        for c in range(n_cols):
            if h_walls[r, c]:
                walls.append((c * cell_w, yc - half_t, (c + 1) * cell_w, yc + half_t))
    # Vertical walls, centred at x = (c + 1) * cell_w.
    for r in range(n_rows):
        for c in range(n_cols - 1):
            if v_walls[r, c]:
                xc = (c + 1) * cell_w
                walls.append((xc - half_t, r * cell_h, xc + half_t, (r + 1) * cell_h))

    return np.array(walls, dtype=np.float64)


# ── Collision geometry ─────────────────────────────────────────────────────────


def segment_collides(
    walls: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    clearance: float = SHORTCUT_CLEARANCE,
) -> bool:
    """Test a segment against every wall at once, exactly.

    Uses the slab method: decompose the segment into 1-D parametric intervals
    along each axis and intersect them.  Correct for any wall thickness — no
    sampling artefacts, which matters because the shortcutter trusts this test
    to certify long diagonal chords through a thin-walled maze.

    Args:
        walls: ``(n_walls, 4)`` rectangles from :func:`make_maze_walls`.
        p1, p2: Segment endpoints, shape ``(2,)``.
        clearance: Expand every wall by this margin before testing [m].

    Returns:
        True if ``[p1, p2]`` intersects any inflated wall.
    """
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])

    wx0 = walls[:, 0] - clearance
    wy0 = walls[:, 1] - clearance
    wx1 = walls[:, 2] + clearance
    wy1 = walls[:, 3] + clearance

    n = len(walls)
    t_min = np.zeros(n)
    t_max = np.ones(n)

    # X slab: the t-interval over which the segment lies inside [wx0, wx1].
    if abs(dx) < 1e-12:
        # Axis-parallel in x; outside the slab → no hit for that wall.
        t_max = np.where((p1[0] < wx0) | (p1[0] > wx1), -1.0, t_max)
    else:
        ta = (wx0 - p1[0]) / dx
        tb = (wx1 - p1[0]) / dx
        t_min = np.maximum(t_min, np.minimum(ta, tb))
        t_max = np.minimum(t_max, np.maximum(ta, tb))

    # Y slab.
    if abs(dy) < 1e-12:
        t_max = np.where((p1[1] < wy0) | (p1[1] > wy1), -1.0, t_max)
    else:
        ta = (wy0 - p1[1]) / dy
        tb = (wy1 - p1[1]) / dy
        t_min = np.maximum(t_min, np.minimum(ta, tb))
        t_max = np.minimum(t_max, np.maximum(ta, tb))

    return bool((t_min <= t_max).any())


def build_occupancy(
    walls: np.ndarray,
    domain: tuple[np.ndarray, np.ndarray],
    res: int,
    clearance: float,
) -> np.ndarray:
    """Rasterise walls into the boolean occupancy grid the wavefront floods.

    Each rectangle is inflated by ``clearance`` and every grid cell whose extent
    overlaps it is marked occupied (floor/ceil span coverage), so thin walls can
    never leak between grid lines regardless of alignment.

    Args:
        walls: ``(n_walls, 4)`` rectangles from :func:`make_maze_walls`.
        domain: ``(lo, hi)`` corners of the planning region, each ``(2,)``.
        res: Grid resolution [cells / metre].
        clearance: Obstacle inflation [m].

    Returns:
        ``(ny, nx)`` bool array indexed ``occ[row=y, col=x]``; True = blocked.
    """
    lo, hi = domain
    nx = int(round((hi[0] - lo[0]) * res))
    ny = int(round((hi[1] - lo[1]) * res))
    occ = np.zeros((ny, nx), dtype=bool)

    for x0, y0, x1, y1 in walls:
        i0 = max(int(np.floor((x0 - clearance - lo[0]) * res)), 0)
        i1 = min(int(np.ceil((x1 + clearance - lo[0]) * res)), nx)
        j0 = max(int(np.floor((y0 - clearance - lo[1]) * res)), 0)
        j1 = min(int(np.ceil((y1 + clearance - lo[1]) * res)), ny)
        occ[j0:j1, i0:i1] = True

    return occ


# ── Wavefront flood ────────────────────────────────────────────────────────────


def _relax(dist: jnp.ndarray, free: jnp.ndarray) -> jnp.ndarray:
    """One BFS relaxation sweep over the whole grid (4-connectivity).

    ``dist`` is the current cost-to-go field; ``free`` masks traversable cells.
    Occupied cells are pinned to ``BFS_INF`` so paths cannot cut through walls.
    """
    ny, nx = dist.shape
    pad = jnp.pad(dist, 1, constant_values=BFS_INF)
    up = pad[0:ny, 1 : nx + 1]
    down = pad[2 : ny + 2, 1 : nx + 1]
    left = pad[1 : ny + 1, 0:nx]
    right = pad[1 : ny + 1, 2 : nx + 2]
    nmin = jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right))
    cand = jnp.minimum(dist, nmin + 1)
    return jnp.where(free, cand, BFS_INF)


@partial(jax.jit, static_argnames=("start_rc", "max_iters"))
def _wavefront(
    free: jnp.ndarray,
    dist0: jnp.ndarray,
    start_rc: tuple[int, int],
    max_iters: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Flood cost-to-go until the start is reached, entirely on-device.

    One relaxation sweep per BFS layer inside ``lax.while_loop``, terminating as
    soon as the start cell has a finite cost-to-go or the cap is hit.

    Args:
        free: ``(ny, nx)`` bool traversability mask.
        dist0: ``(ny, nx)`` int32 field, ``BFS_INF`` everywhere but the goal.
        start_rc: ``(row, col)`` of the start cell — static, drives early exit.
        max_iters: Hard cap on relaxation sweeps.

    Returns:
        ``(dist, n_iter)`` — final cost-to-go field and sweeps performed.
    """
    si, sj = start_rc

    def cond(state):
        dist, it = state
        return jnp.logical_and(it < max_iters, dist[si, sj] >= BFS_INF)

    def body(state):
        dist, it = state
        return _relax(dist, free), it + 1

    return jax.lax.while_loop(cond, body, (dist0, jnp.int32(0)))


def _descend_path(dist: np.ndarray, start_rc, goal_rc) -> list[tuple[int, int]]:
    """Recover a cell path by steepest descent on the goal-sourced cost-to-go.

    From the start cell, repeatedly step to the 4-neighbour whose cost-to-go is
    exactly one lower.  For a BFS field this always reaches the goal (cost 0).
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
        if nxt is None:  # dead end (should not happen)
            break
        cur = nxt
        cells.append(cur)
    return cells


def _shortcut_with_history(
    walls: np.ndarray,
    path: np.ndarray,
    clearance: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Greedy line-of-sight shortcut, keeping each intermediate polyline.

    From each kept waypoint, jump as far ahead as an exact collision test
    allows.  The intermediate polylines are what the viser animation replays.
    """
    smoothed = [path[0]]
    stages: list[np.ndarray] = [np.array([path[0]])]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and segment_collides(walls, path[i], path[j], clearance=clearance):
            j -= 1
        smoothed.append(path[j])
        stages.append(np.array(smoothed))
        i = j
    return np.array(smoothed), stages


def wavefront_solve(
    walls: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    domain: tuple[np.ndarray, np.ndarray],
    *,
    res: int = PLAN_RES,
    clearance: float = PLAN_CLEARANCE,
    shortcut_clearance: float = SHORTCUT_CLEARANCE,
    record_history: bool = False,
    history_stride: int = 12,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Plan a collision-free polyline through the maze: flood, descend, shortcut.

    Args:
        walls: ``(n_walls, 4)`` rectangles from :func:`make_maze_walls`.
        start, goal: Physical positions, shape ``(2,)``.
        domain: ``(lo, hi)`` corners of the planning region, each ``(2,)``.
        res: Planning-grid resolution [cells / metre].
        clearance: Obstacle inflation when rasterising [m].
        shortcut_clearance: Clearance for the line-of-sight shortcutter [m].
        record_history: Also capture the intermediate planner states the viser
            animation replays.  This runs the flood sweep-by-sweep on the host
            instead of inside ``lax.while_loop``, so it is slower — pass False
            when only the path is wanted.
        history_stride: Record one cost-to-go snapshot every this many sweeps.

    Returns:
        ``plan_path`` — ``(K, 2)`` ordered collision-free waypoints.  When
        ``record_history``, returns ``(plan_path, history)`` where ``history``
        carries ``dist_frames``, ``free``, ``lo``, ``res``, ``n_sweeps``,
        ``shortcut_stages`` and ``raw_path`` for
        :func:`examples._maze_viz.animate_wavefront`.

    Raises:
        RuntimeError: If the flood never reaches the start cell — the start or
            goal is inside an inflated wall, or the maze is disconnected.
    """
    lo, _ = domain
    occ = build_occupancy(walls, domain, res, clearance)
    ny, nx = occ.shape
    free = jnp.asarray(~occ)

    def _rc(p):
        col = min(max(int((p[0] - lo[0]) * res), 0), nx - 1)
        row = min(max(int((p[1] - lo[1]) * res), 0), ny - 1)
        return row, col

    start_rc = _rc(start)
    goal_rc = _rc(goal)

    dist0 = np.full((ny, nx), BFS_INF, dtype=np.int32)
    dist0[goal_rc[0], goal_rc[1]] = 0

    dist_frames: list[np.ndarray] = []
    if record_history:
        dist_np = dist0.copy()
        dist_frames.append(dist_np.copy())
        sweep = 0
        max_iters = nx * ny
        while dist_np[start_rc[0], start_rc[1]] >= BFS_INF and sweep < max_iters:
            dist_np = np.asarray(_relax(jnp.asarray(dist_np), free))
            sweep += 1
            if sweep % history_stride == 0 or dist_np[start_rc[0], start_rc[1]] < BFS_INF:
                dist_frames.append(dist_np.copy())
        n_iter = sweep
        dist = dist_np
    else:
        dist_j, n_iter = _wavefront(free, jnp.asarray(dist0), start_rc, nx * ny)
        dist = np.asarray(dist_j)

    if dist[start_rc[0], start_rc[1]] >= BFS_INF:
        raise RuntimeError("Wavefront planner failed to reach the start cell.")

    cells = _descend_path(dist, start_rc, goal_rc)
    pts = np.array([[lo[0] + (c + 0.5) / res, lo[1] + (r + 0.5) / res] for r, c in cells])
    raw_path = np.vstack([start.copy(), pts, goal.copy()])
    deduped = [raw_path[0]]
    for wp in raw_path[1:]:
        if np.linalg.norm(wp - deduped[-1]) > 1e-9:
            deduped.append(wp)
    raw_path = np.array(deduped)

    plan_path, shortcut_stages = _shortcut_with_history(walls, raw_path, shortcut_clearance)
    if not record_history:
        return plan_path

    history = {
        "dist_frames": dist_frames,
        "free": np.asarray(~occ),
        "lo": np.asarray(lo).copy(),
        "res": res,
        "n_sweeps": int(n_iter),
        "shortcut_stages": shortcut_stages,
        "raw_path": raw_path,
    }
    return plan_path, history


# ── Guess construction ─────────────────────────────────────────────────────────


def path_to_guess(
    path: np.ndarray,
    n_nodes: int,
    t_total: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a collision-free polyline onto ``n_nodes`` SCP nodes, keeping corners.

    Uniform arc-length resampling places nodes at evenly spaced arc positions
    that almost never coincide with the polyline's corners.  In a maze the path
    turns ~90° at every corner, so a chord between two nodes straddling a corner
    cuts diagonally across the wall — the resampled guess passes through
    obstacles even though the polyline does not.

    This resampler keeps every original vertex as a node and only ever *inserts*
    nodes into segment interiors (distributed by segment length, largest
    remainder rounding to hit the count exactly).  Every consecutive node pair
    therefore lies on a single original segment, so the guess is collision-free
    by construction — provided ``n_nodes >= len(path)``.

    Args:
        path: ``(K, 2)`` ordered collision-free waypoints.
        n_nodes: Number of SCP nodes; should be ``>= K``.  Below that the
            corner-preserving property is impossible and this falls back to
            uniform resampling, which may clip corners.
        t_total: Assumed total traversal time, used to finite-difference the
            velocity guess [s].

    Returns:
        ``(position, velocity)``, each ``(n_nodes, 2)``.  Endpoint velocities
        are zeroed to match the usual rest-to-rest boundary conditions.
    """
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

    # Node spacing is non-uniform, so hand np.gradient the per-node time stamps.
    node_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    t_nodes = t_total * node_arc / node_arc[-1]
    vel = np.gradient(pos, t_nodes, axis=0)
    vel[0] = 0.0
    vel[-1] = 0.0
    return pos, vel
