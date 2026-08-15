"""Viser scene and wavefront animation shared by the maze examples.

The two maze examples — planar double integrator and 6-DOF quadrotor — draw the
same world: a floor, extruded walls, start/goal markers, and a four-phase replay
of the planner building the SCP initial guess.  Only the *results* view differs
between them (a marker on a trail versus a posed quadrotor and a chase camera),
so that half stays in each example file where it can be read next to the problem
it visualises.

Everything here is Z-up, matching the rest of the repo: maze ``(x, y)`` is world
``(x, y)`` and the layer heights in :class:`MazeHeights` are world ``z``.

Companion to ``examples/_maze.py``, which owns the planner itself.
"""

from __future__ import annotations

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import viser

from examples._maze import BFS_INF
from openscvx.plotting.viser.animated import add_animation_controls


class MazeHeights(NamedTuple):
    """Z-levels of the layered maze scene, with the floor at ``z = 0``.

    A maze view is a stack of flat things drawn at different altitudes so they
    do not z-fight.  Each example declares one of these and every drawing call
    — here and in the example's own ``plot_results`` — reads its heights from
    it, so the stacking order lives in exactly one place.

    Attributes:
        wall: Height of the extruded wall boxes [m].  Also the altitude the
            overview camera looks at.
        field: Wavefront cost-to-go point cloud.
        wavefront: Planner polyline during the wavefront animation.
        guess: SCP initial-guess polyline.
        plan: Planner shortcut polyline in the results view.
    """

    wall: float
    field: float
    wavefront: float
    guess: float
    plan: float


# ── Geometry ───────────────────────────────────────────────────────────────────


def maze_xy_to_world(xy: np.ndarray, z: float) -> np.ndarray:
    """Lift maze ``(x, y)`` to world ``(x, y, z)`` at a fixed altitude.

    Accepts a single point ``(2,)`` or a polyline ``(K, 2)`` and returns the
    matching ``(3,)`` / ``(K, 3)`` float32 array viser expects.
    """
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim == 1:
        return np.array([xy[0], xy[1], z], dtype=np.float32)
    return np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), z, dtype=np.float32)])


def path_line_segments(xy: np.ndarray, z: float) -> np.ndarray:
    """Maze polyline ``(K, 2)`` → viser ``add_line_segments`` array ``(K-1, 2, 3)``.

    Line segments rather than a spline: the guess is dense and every corner is
    load-bearing, and a Catmull-Rom curve through it rounds corners into the
    walls the planner just avoided.
    """
    pts3 = maze_xy_to_world(xy, z)
    if len(pts3) < 2:
        return np.zeros((1, 2, 3), dtype=np.float32)
    return np.stack([pts3[:-1], pts3[1:]], axis=1)


def uniform_segment_colors(n_segs: int, rgb: tuple[int, int, int]) -> np.ndarray:
    """One flat colour for every endpoint of a line-segment array: ``(N, 2, 3)`` uint8."""
    return np.tile(np.array(rgb, dtype=np.uint8), (n_segs, 2, 1))


# Triangulation of one extruded box, indexing the eight corners emitted by
# :func:`wall_boxes_mesh`.  The bottom face is omitted: it is never visible from
# above the floor, and a maze has thousands of these.
_BOX_FACES = np.array(
    [
        # y = y0 side
        [0, 1, 2],
        [0, 2, 3],
        # y = y1 side
        [4, 6, 5],
        [4, 7, 6],
        # x = x0 side
        [0, 3, 7],
        [0, 7, 4],
        # x = x1 side
        [1, 5, 6],
        [1, 6, 2],
        # top
        [3, 2, 6],
        [3, 6, 7],
    ],
    dtype=np.uint32,
)


def wall_boxes_mesh(walls: np.ndarray, *, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Extrude wall rectangles into one combined triangle mesh, from the floor up.

    Every wall goes into a single mesh: a maze has thousands of them, and one
    scene node draws far faster than one node per wall.

    Args:
        walls: ``(n_walls, 4)`` rectangles from ``examples._maze.make_maze_walls``.
        height: Extrusion height [m]; boxes span ``z ∈ [0, height]``.

    Returns:
        ``(vertices, faces)`` for ``server.scene.add_mesh_simple``.
    """
    x0, y0, x1, y1 = np.asarray(walls, dtype=np.float64).T
    z0 = np.zeros_like(x0)
    z1 = np.full_like(x0, height)
    # Corner order fixed by _BOX_FACES: bottom then top of the y0 edge, then of
    # the y1 edge.
    corners = np.stack(
        [
            np.column_stack([x0, y0, z0]),
            np.column_stack([x1, y0, z0]),
            np.column_stack([x1, y0, z1]),
            np.column_stack([x0, y0, z1]),
            np.column_stack([x0, y1, z0]),
            np.column_stack([x1, y1, z0]),
            np.column_stack([x1, y1, z1]),
            np.column_stack([x0, y1, z1]),
        ],
        axis=1,
    )
    faces = _BOX_FACES + 8 * np.arange(len(walls), dtype=np.uint32)[:, None, None]
    return corners.reshape(-1, 3).astype(np.float32), faces.reshape(-1, 3)


# ── Scene ──────────────────────────────────────────────────────────────────────


def add_maze_scene(
    server: viser.ViserServer,
    walls: np.ndarray,
    domain: tuple[np.ndarray, np.ndarray],
    *,
    wall_height: float,
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    marker_radius: float = 0.3,
) -> None:
    """Populate a server with the static maze: floor, walls, start and goal.

    Both viewers each example opens — the planner animation and the results
    playback — begin with this call, so they share one world.  It also sets the
    scene up-direction to ``+z``, the convention every height in this module
    assumes.

    Args:
        server: Server to add the scene to.
        walls: ``(n_walls, 4)`` rectangles from ``examples._maze.make_maze_walls``.
        domain: ``(lo, hi)`` corners of the floor, each ``(2,)``.
        wall_height: Extrusion height of the wall boxes [m].
        start, goal: World positions of the two markers.  Passed in full 3-D
            because each example parks them at its own working altitude.
        marker_radius: Radius of the start/goal spheres [m].
    """
    server.scene.set_up_direction("+z")

    lo, hi = domain
    floor_v = np.array(
        [
            [lo[0], lo[1], 0.0],
            [hi[0], lo[1], 0.0],
            [hi[0], hi[1], 0.0],
            [lo[0], hi[1], 0.0],
        ],
        dtype=np.float32,
    )
    floor_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    server.scene.add_mesh_simple("/floor", floor_v, floor_f, color=(230, 220, 200), side="double")

    w_verts, w_faces = wall_boxes_mesh(walls, height=wall_height)
    server.scene.add_mesh_simple(
        "/walls", w_verts, w_faces, color=(75, 75, 85), flat_shading=True, side="double"
    )

    server.scene.add_icosphere(
        "/start",
        radius=marker_radius,
        color=(30, 200, 80),
        position=tuple(float(v) for v in start),
    )
    server.scene.add_icosphere(
        "/goal",
        radius=marker_radius,
        color=(220, 50, 50),
        position=tuple(float(v) for v in goal),
    )


def set_overview_camera(
    server: viser.ViserServer,
    domain: tuple[np.ndarray, np.ndarray],
    focus_height: float,
) -> None:
    """Frame the whole maze from behind and above, scaled to the domain size."""
    lo, hi = domain
    center = np.array(
        [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), focus_height], dtype=np.float64
    )
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    server.initial_camera.position = tuple(center + np.array([0.0, -0.6 * span, 0.5 * span]))
    server.initial_camera.look_at = tuple(center)
    server.initial_camera.up = (0.0, 0.0, 1.0)


# ── Wavefront animation ────────────────────────────────────────────────────────


def _collapse_collinear(xy: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Drop interior vertices on a straight run — same polyline, far fewer segments."""
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


def _dist_field_cloud(
    dist: np.ndarray,
    free: np.ndarray,
    lo: np.ndarray,
    res: int,
    z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reached planning cells as a plasma-coloured point cloud (bright at the goal)."""
    reached = free & (dist < BFS_INF)
    rows, cols = np.where(reached)
    if len(rows) == 0:
        return np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.uint8)

    xs = lo[0] + (cols + 0.5) / res
    ys = lo[1] + (rows + 0.5) / res
    pts = np.column_stack(
        [xs.astype(np.float32), ys.astype(np.float32), np.full(len(xs), z, dtype=np.float32)]
    )
    d = dist[reached].astype(np.float64)
    d_norm = 1.0 - (d / max(float(d.max()), 1.0))  # 1 at the goal, 0 at the frontier
    return pts, (plt.cm.plasma(d_norm)[:, :3] * 255.0).astype(np.uint8)


def animate_wavefront(
    server: viser.ViserServer,
    wf_history: dict,
    plan_path: np.ndarray,
    guess_path_xy: np.ndarray,
    *,
    domain: tuple[np.ndarray, np.ndarray],
    heights: MazeHeights,
) -> None:
    """Replay the planner building the SCP initial guess, in four phases.

    Press Play in the Animation folder:

    1. Cost-to-go floods outward from the goal over every free cell.
    2. Steepest descent traces a corridor path from start to goal.
    3. Line-of-sight shortcutting collapses the staircase into a clean polyline.
    4. Vertex-preserving resampling reveals the dense SCP initial guess.

    Each phase gets its own wall-clock budget rather than a share proportional
    to its frame count — the flood has orders of magnitude more raw frames than
    the shortcutter, and pacing by frame count made the interesting phases flash
    past in under a second.

    Call :func:`add_maze_scene` on ``server`` first; this adds only the animated
    overlay and the animation controls.

    Args:
        server: Server hosting the animation.
        wf_history: The history dict from
            ``examples._maze.wavefront_solve(..., record_history=True)``.
        plan_path: ``(K, 2)`` shortcut polyline — the animation's end state.
        guess_path_xy: ``(N, 2)`` resampled SCP guess revealed in phase 4.
        domain: ``(lo, hi)`` corners of the maze, for framing the camera.
        heights: Z-levels for the field cloud, planner path and guess polyline.
    """
    lo = wf_history["lo"]
    free = wf_history["free"]
    res = wf_history["res"]
    raw_path = np.asarray(wf_history["raw_path"], dtype=np.float64)
    dist_frames = wf_history["dist_frames"]
    shortcut_stages = [
        np.asarray(s, dtype=np.float64) for s in wf_history["shortcut_stages"] if len(s) >= 2
    ]
    if not shortcut_stages:
        shortcut_stages = [np.asarray(plan_path, dtype=np.float64)]

    # Subsample each phase to a watchable keyframe count.
    n_wf = min(len(dist_frames), 60)
    n_des = min(max(len(raw_path) - 1, 2), 80)
    n_sc = min(len(shortcut_stages), 50)
    wf_idx = np.unique(np.linspace(0, len(dist_frames) - 1, n_wf).astype(int))
    des_count = np.unique(np.linspace(2, len(raw_path), n_des).astype(int))  # prefix lengths
    sc_idx = np.unique(np.linspace(0, len(shortcut_stages) - 1, n_sc).astype(int))

    # Frames are (phase, payload); the descent payload is a prefix length.
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
        t = t_arr[j - 1] + 0.25  # brief pause between phases
        i = j

    # Geometry reused on every descent / shortcut / guess frame.
    field_final_pts, field_final_cols = _dist_field_cloud(
        dist_frames[-1], free, lo, res, heights.field
    )
    field_dim_cols = (field_final_cols.astype(np.float32) * 0.22).astype(np.uint8)
    raw_disp = _collapse_collinear(raw_path)
    plan_segs = path_line_segments(plan_path, heights.wavefront)
    ghost_segs = path_line_segments(raw_disp, heights.wavefront * 0.55)

    field_cloud = server.scene.add_point_cloud(
        "/wavefront/reached",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        point_size=0.04,
    )
    # Scalar colours broadcast to any segment count, so the polyline can grow
    # and shrink across frames without a shape mismatch.
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
        points=path_line_segments(guess_path_xy, heights.guess),
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
            segs = path_line_segments(xy, heights.wavefront)
        pts3 = maze_xy_to_world(xy, heights.wavefront)
        path_line.points = segs
        path_line.colors = np.array(rgb, dtype=np.uint8)
        path_line.visible = True
        path_cloud.points = pts3
        path_cloud.colors = np.broadcast_to(np.array(rgb, dtype=np.uint8), (len(pts3), 3)).copy()
        path_cloud.visible = True

    def _update(frame_idx: int) -> None:
        phase, payload = frames[frame_idx]
        if phase == "wavefront":
            pts, cols = _dist_field_cloud(payload, free, lo, res, heights.field)
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
            # Collapse the growing prefix: exact corridor path, one segment per
            # straight hallway instead of thousands of cells.
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
        else:  # guess
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

    add_animation_controls(server, t_arr, [_update], loop=True, folder_name="Wavefront Animation")
    set_overview_camera(server, domain, heights.wall)
    _update(0)
