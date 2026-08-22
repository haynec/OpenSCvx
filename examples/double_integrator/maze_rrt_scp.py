"""2D maze navigation: JAX wavefront planner + SCP refinement.

A planar double integrator crosses a complicated 2-D maze from start to goal.
The key challenge: the maze has no straight-line path, so a naive SCP
initialisation starts deep inside the infeasible set.

Pipeline
--------
1. **Maze + planner** — ``examples/_maze.py`` generates a DFS-backtracking
   perfect maze on an 80×80 cell grid, floods a JIT-compiled BFS wavefront over
   the rasterised free space, recovers a path by steepest descent on the
   cost-to-go field, and shortcuts it with an exact analytical line-of-sight
   test.
2. **Guess constructor** — ``path_to_guess`` resamples the planned polyline onto
   the SCP node grid while preserving every corner, so the guess is
   collision-free by construction, and finite-differences it for velocity.
3. **SCP** — the double-integrator problem below refines that guess into a
   dynamically feasible, time-optimal trajectory, with CTCS infinity-norm
   constraints keeping the position outside every wall rectangle.

Run::

    python examples/double_integrator/maze_rrt_scp.py

Viser
-----
* ``:8081`` — wavefront / initial-guess animation (four phases, press Play).
* ``:8080`` — SCP trajectory with a velocity-coloured trail and an agent marker.
"""

from __future__ import annotations

import os
import sys
import time

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
    maze_xy_to_world,
    path_line_segments,
    set_overview_camera,
    uniform_segment_colors,
)
from openscvx import Problem
from openscvx.plotting.viser import (
    add_animated_trail,
    add_animation_controls,
    add_position_marker,
    compute_velocity_colors,
)

# ── Grid / domain parameters ──────────────────────────────────────────────────
GRID_COLS = 80
GRID_ROWS = 80
CELL_W = 1.0  # cell width  [m]
CELL_H = 1.0  # cell height [m]
WALL_T = 0.10  # wall thickness (centred on grid edge) [m]
MAZE_SEED = 0  # RNG seed for DFS maze generator

DOMAIN_LO = np.array([0.0, 0.0])
DOMAIN_HI = np.array([GRID_COLS * CELL_W, GRID_ROWS * CELL_H])  # [80, 80]
DOMAIN = (DOMAIN_LO, DOMAIN_HI)
START = np.array([0.5 * CELL_W, 0.5 * CELL_H])  # [0.5, 0.5]
GOAL = np.array([(GRID_COLS - 0.5) * CELL_W, (GRID_ROWS - 0.5) * CELL_H])  # [79.5, 79.5]

# ── SCP parameters ────────────────────────────────────────────────────────────
N = 1100  # SCP discretisation nodes
T_MAX = 12000.0  # upper bound on flight time [s]
T_GUESS = 12000.0  # assumed time for velocity finite-differencing [s]
V_MAX = 8.0  # speed limit   [m/s]
F_MAX = 5.0  # force limit   [m/s²]
MASS = 1.0

# ── Maze + plan ───────────────────────────────────────────────────────────────
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

# The maze solution has one genuine turn per corridor bend.  A collision-free
# guess must keep *every* corner as a node — otherwise chords between nodes cut
# across walls.  So the node count must be at least the number of waypoints;
# grow N to guarantee this (path_to_guess then preserves all corners exactly).
N = max(N, len(plan_path))
print(f"  SCP nodes: N = {N}")

pos_guess, vel_guess = path_to_guess(plan_path, N, T_GUESS)

# ── States ─────────────────────────────────────────────────────────────────────
position = ox.State("position", shape=(2,))
position.min = DOMAIN_LO
position.max = DOMAIN_HI
position.initial = START
position.final = GOAL
position.guess = pos_guess

velocity = ox.State("velocity", shape=(2,))
velocity.min = np.array([-V_MAX, -V_MAX])
velocity.max = np.array([V_MAX, V_MAX])
velocity.initial = np.array([0.0, 0.0])
velocity.final = np.array([0.0, 0.0])
velocity.guess = vel_guess

# ── Controls ───────────────────────────────────────────────────────────────────
force = ox.Control("force", shape=(2,))
force.max = np.array([F_MAX, F_MAX])
force.min = np.array([-F_MAX, -F_MAX])
force.guess = np.zeros((N, 2))

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
wall_centers = np.array(
    [((x0 + x1) / 2.0, (y0 + y1) / 2.0) for x0, y0, x1, y1 in MAZE_WALLS]
)  # (n_walls, 2)
wall_inv_scales = np.array(
    [(2.0 / (x1 - x0), 2.0 / (y1 - y0)) for x0, y0, x1, y1 in MAZE_WALLS]
)  # (n_walls, 2)

wall_avoidance = ox.ctcs(
    np.ones(len(MAZE_WALLS))
    <= ox.Vmap(
        lambda center, inv_scale: ox.linalg.Norm(inv_scale * (position - center), ord="inf"),
        batch=[wall_centers, wall_inv_scales],
    ),
    penalty="huber",
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
    licq_max=1e-10,
    algorithm={
        "lam_cost": 1e-2,
        "lam_vc": 1e1,
        "lam_prox": 1e0,
        "autotuner": ox.ConstantProximalWeight(),
    },
)


# ── Visualisation ──────────────────────────────────────────────────────────────
# The problem is planar, so every overlay is a flat layer stacked just above the
# floor at z = 0; the exact altitudes only have to keep the layers from z-fighting.

HEIGHTS = MazeHeights(wall=0.5, field=0.04, wavefront=0.28, guess=0.06, plan=0.10)
_TRAJ_Z = 0.15  # SCP trajectory polyline, above the guide paths
_MARKER_R = 0.4  # radius of the start/goal spheres
_MARKER_Z = HEIGHTS.guess + _MARKER_R


def _add_scene(server: viser.ViserServer) -> None:
    """Add the maze floor, walls and markers at this example's altitudes."""
    add_maze_scene(
        server,
        MAZE_WALLS,
        DOMAIN,
        wall_height=HEIGHTS.wall,
        start=(START[0], START[1], _MARKER_Z),
        goal=(GOAL[0], GOAL[1], _MARKER_Z),
        marker_radius=_MARKER_R,
    )


def animate_wavefront_viser(
    wf_history: dict,
    plan_path: np.ndarray,
    guess_path: np.ndarray,
    *,
    port: int = 8081,
) -> viser.ViserServer:
    """Open the shared four-phase planner animation on its own server.

    Runs on ``port`` (default 8081) so it can coexist with the results viewer.
    """
    server = viser.ViserServer(port=port)
    print(f"  Wavefront animation → http://localhost:{port}")
    _add_scene(server)
    animate_wavefront(server, wf_history, plan_path, guess_path, domain=DOMAIN, heights=HEIGHTS)
    return server


def plot_results(
    plan_path: np.ndarray,
    guess_path: np.ndarray,
    results,
    *,
    port: int = 8080,
    loop_animation: bool = True,
) -> viser.ViserServer:
    """Animated viser scene: maze, initial guess, planner guide, SCP trajectory.

    Static orange/blue polylines show the SCP initial guess and planner guide.
    The full SCP path is drawn faintly; press Play to grow a velocity-coloured
    trail and move an agent marker along the post-processed trajectory.
    """
    server = viser.ViserServer(port=port)
    print(f"  Results animation → http://localhost:{port}")
    print("  Press Play in the Animation folder.  Ctrl-C to exit.")

    _add_scene(server)

    # ── SCP initial guess (dense polyline, one segment per node pair) ────────
    guess_segs = path_line_segments(guess_path, HEIGHTS.guess)
    server.scene.add_line_segments(
        "/initial_guess",
        guess_segs,
        uniform_segment_colors(len(guess_segs), (255, 180, 40)),
        line_width=2.5,
    )

    # ── Planner guide path (shortcut polyline) ───────────────────────────────
    plan_segs = path_line_segments(plan_path, HEIGHTS.plan)
    server.scene.add_line_segments(
        "/plan_path",
        plan_segs,
        uniform_segment_colors(len(plan_segs), (30, 100, 220)),
        line_width=2.0,
    )

    # ── SCP trajectory (static full path + animated trail / agent) ───────────
    pos_traj = np.asarray(results.trajectory["position"], dtype=np.float64)
    traj_time = np.asarray(results.trajectory["time"], dtype=np.float64).reshape(-1)
    if traj_time.size != len(pos_traj):
        traj_time = np.linspace(0.0, float(traj_time[-1]) if traj_time.size else 1.0, len(pos_traj))
    vel_raw = results.trajectory.get("velocity")
    vel_traj = np.asarray(vel_raw, dtype=np.float64) if vel_raw is not None else None

    traj_segs = path_line_segments(pos_traj, _TRAJ_Z)
    server.scene.add_line_segments(
        "/scp_trajectory/full",
        traj_segs,
        uniform_segment_colors(len(traj_segs), (120, 40, 40)),
        line_width=2.0,
    )

    pos_vis = maze_xy_to_world(pos_traj, _TRAJ_Z)
    trail_colors = compute_velocity_colors(vel_traj, fallback_length=len(pos_vis))
    _, update_trail = add_animated_trail(
        server,
        pos_vis,
        trail_colors,
        point_size=0.12,
    )
    _, update_marker = add_position_marker(
        server,
        pos_vis,
        radius=0.35,
        color=(240, 240, 245),
    )
    add_animation_controls(
        server,
        traj_time,
        [update_trail, update_marker],
        loop=loop_animation,
    )
    update_trail(0)
    update_marker(0)

    set_overview_camera(server, DOMAIN, HEIGHTS.wall)

    with server.gui.add_folder("Legend"):
        server.gui.add_markdown(
            f"**Orange** — SCP initial guess ({len(guess_path)} nodes)  \n"
            f"**Blue** — planner shortcut ({len(plan_path)} waypoints)  \n"
            f"**Dark red** — full SCP solution  \n"
            f"**Trail + white sphere** — animated agent (press Play)  \n"
            f"Wavefront animation: port 8081"
        )
    return server


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
    # Two viser viewers on different ports (both stay alive until Ctrl-C):
    #   :8081 — wavefront planner animation (press Play)
    #   :8080 — SCP results animation (press Play)
    animate_wavefront_viser(wf_history, plan_path, pos_guess, port=8081)
    traj_server = plot_results(plan_path, pos_guess, results, port=8080)
    traj_server.sleep_forever()
