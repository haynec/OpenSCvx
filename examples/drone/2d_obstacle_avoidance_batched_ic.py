"""Batched 2D obstacle avoidance: solve over B random initial positions.

Same planar double-integrator + CTCS obstacle-avoidance problem as
``2d_obstacle_avoidance.py``, but ``solve_batched`` runs all B solves in one
compiled dispatch (Moreau backend).  The initial position is a ``Parameter``
pinned via ``(position == initial_position).convex().at([0])``.  Passing a ``(B, 2)`` array as ``parameters={"initial_position": ic_batch}``
pins each IC via the convex equality at node 0.  Per-batch ``x_guess`` stacks
(a position linspace from each IC to the shared target) seed the SCP iterate
independently.  ``post_process_batched`` then propagates
every converged solution through the full nonlinear dynamics.

All B trajectories are visualised together with Plotly — converged runs in
blue, diverged in orange — on top of the obstacle field.
"""

import os
import sys
import time

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem

# ── Batch configuration ────────────────────────────────────────────────────────
N_BATCH = 200    # number of initial conditions to solve simultaneously
N = 40           # SCP discretisation nodes
TOTAL_TIME = 120.0

# ── Obstacle field (identical to 2d_obstacle_avoidance.py) ────────────────────
obstacle_radius_min, obstacle_radius_max = 1.0, 2.5

np.random.seed(42)
obstacle_centers_list = []
n_rows, n_cols = 20, 20
for i in range(n_rows):
    for j in range(n_cols):
        x = -6.0 + i * 6.0
        y = -7.5 + j * 5.0
        x += np.random.uniform(-1.0, 1.0)
        y += np.random.uniform(-1.0, 1.0)
        obstacle_centers_list.append([x, y])

obstacle_centers = np.array(obstacle_centers_list)   # (n_obs, 2)
obstacle_radii = np.random.uniform(
    obstacle_radius_min, obstacle_radius_max, size=len(obstacle_centers_list)
)

# ── States ─────────────────────────────────────────────────────────────────────
_default_ic = np.array([-10.0, -10.0])   # used for guess; actual IC set by ZeroConeConstraint

position = ox.State("position", shape=(2,))
position.max = np.array([150.0, 150.0])
position.min = np.array([-15.0, -15.0])
position.initial = [ox.Free(-10.0), ox.Free(-10.0)]
position.final = np.array([100.0, 100.0])

velocity = ox.State("velocity", shape=(2,))
velocity.max = np.array([10.0, 10.0])
velocity.min = np.array([-10.0, -10.0])
velocity.initial = np.array([0.0, 0.0])
velocity.final = [("free", 0.0), ("free", 0.0)]

# ── Controls ───────────────────────────────────────────────────────────────────
force = ox.Control("force", shape=(2,))
a_max = 20.0
force.max = np.array([a_max, a_max])
force.min = np.array([-a_max, -a_max])

m = 1.0

# ── Batched parameter: initial position ───────────────────────────────────────
initial_position = ox.Parameter("initial_position", shape=(2,), value=_default_ic)

# ── Dynamics ───────────────────────────────────────────────────────────────────
dynamics = {
    "position": velocity,
    "velocity": (1.0 / m) * force,
}

# ── Constraints ────────────────────────────────────────────────────────────────
states = [position, velocity]
controls = [force]
constraints = []

for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

constraints.extend([force <= force.max, force.min <= force])

obstacle_avoidance = ox.ctcs(
    obstacle_radii
    <= ox.Vmap(
        lambda obs_center: ox.linalg.Norm(position - obs_center),
        batch=obstacle_centers,
    )
)
constraints.append(obstacle_avoidance)

# ── Constraints ── initial position pinned via convex equality at node 0 ───────
constraints.append((position == initial_position).convex().at([0]))

# ── Initial guesses ────────────────────────────────────────────────────────────
position.guess = np.linspace(_default_ic, position.final, N)
velocity.guess = np.zeros((N, 2))
force.guess = np.zeros((N, 2))

# ── Time ───────────────────────────────────────────────────────────────────────
time_var = ox.Time(
    initial=0.0,
    final=("minimize", TOTAL_TIME),
    min=0.0,
    max=TOTAL_TIME,
)

# ── Problem ────────────────────────────────────────────────────────────────────
# Moreau backend: JAX-native, exportable, required for solve_batched.
problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time_var,
    constraints=constraints,
    N=N,
    algorithm={
        "autotuner": ox.RampProximalWeight(ramp_factor=1.1, lam_prox_max=1e3),
        "lam_cost": 1e-2,
        "lam_vc": 1e1,
    },
    solver=ox.MoreauPTRSolver(),
    float_dtype="float64",
)


# ── Helper: sample valid initial positions ─────────────────────────────────────
def sample_initial_positions(
    B: int,
    seed: int = 0,
    x_range: tuple[float, float] = (-14.0, 20.0),
    y_range: tuple[float, float] = (-14.0, 20.0),
    safety_margin: float = 0.5,
) -> np.ndarray:
    """Sample B positions in [x_range × y_range] that lie outside all obstacles.

    The default range covers the lower-left quadrant of the obstacle field,
    giving varied starting positions from far-corner approaches to positions
    already adjacent to the first few rows of obstacles.

    Args:
        B: Number of positions to sample.
        seed: RNG seed.
        x_range: (min, max) for the x coordinate.
        y_range: (min, max) for the y coordinate.
        safety_margin: Extra clearance beyond the obstacle radius.

    Returns:
        Array of shape ``(B, 2)``.
    """
    rng = np.random.default_rng(seed)
    positions: list[np.ndarray] = []
    while len(positions) < B:
        p = rng.uniform(
            [x_range[0], y_range[0]],
            [x_range[1], y_range[1]],
        )
        dists = np.linalg.norm(obstacle_centers - p, axis=1)
        if np.all(dists > obstacle_radii + safety_margin):
            positions.append(p)
    return np.array(positions)


def build_batched_x_guess(problem, ic_batch: np.ndarray) -> np.ndarray:
    """Build per-batch state guesses with position linspace IC → target."""
    base_x = np.asarray(problem.state.x)  # (N, n_x)
    pos_sl = position._slice
    final_pos = np.asarray(position.final)
    B = ic_batch.shape[0]
    x_guess_batch = np.broadcast_to(base_x, (B,) + base_x.shape).copy()
    t = np.linspace(0.0, 1.0, N)
    x_guess_batch[:, :, pos_sl] = (
        ic_batch[:, None, :] * (1.0 - t[None, :, None])
        + final_pos[None, None, :] * t[None, :, None]
    )
    return x_guess_batch


# ── Visualisation ──────────────────────────────────────────────────────────────
def plot_batched_2d_trajectories(results, ic_batch: np.ndarray):
    """Plotly figure overlaying all B propagated trajectories on the obstacle field.

    Args:
        results: :class:`~openscvx.algorithms.OptimizationResults` returned by
            :meth:`~openscvx.problem.Problem.post_process_batched`.  Must have
            ``trajectory["position"]`` of shape ``(B, T, 2)``.
        ic_batch: ``(B, 2)`` array of initial positions used for the batch.
    """
    import plotly.graph_objects as go

    converged = np.asarray(results.converged, dtype=bool).reshape(-1)
    B = len(converged)

    # Extract propagated positions: (B, T, 2)
    pos_all = np.asarray(results.trajectory["position"], dtype=np.float64)
    if pos_all.ndim == 2:
        # Fallback: single-batch dim was squeezed
        pos_all = pos_all[np.newaxis]

    fig = go.Figure()

    # ── Background obstacles ──────────────────────────────────────────────────
    for center, radius in zip(obstacle_centers, obstacle_radii):
        xc, yc = center
        fig.add_shape(
            type="circle",
            x0=xc - radius, y0=yc - radius,
            x1=xc + radius, y1=yc + radius,
            fillcolor="rgba(220, 180, 180, 0.30)",
            line={"color": "rgba(200, 120, 120, 0.45)", "width": 1},
            layer="below",
        )

    # ── Trajectories ──────────────────────────────────────────────────────────
    # Draw diverged runs first so converged runs render on top.
    for b in range(B):
        ok = bool(converged[b])
        color = "rgba(30, 130, 220, 0.65)" if ok else "rgba(220, 100, 30, 0.50)"
        pos = pos_all[b]        # (T, 2)
        fig.add_trace(
            go.Scatter(
                x=pos[:, 0], y=pos[:, 1],
                mode="lines",
                line={"color": color, "width": 1.5 if ok else 1.0},
                showlegend=False,
                hovertemplate=f"run {b}  {'✓' if ok else '✗'}<extra></extra>",
            )
        )

    # ── Start markers (one per batch element) ─────────────────────────────────
    starts_ok  = ic_batch[converged]
    starts_bad = ic_batch[~converged]
    if len(starts_ok):
        fig.add_trace(
            go.Scatter(
                x=starts_ok[:, 0], y=starts_ok[:, 1],
                mode="markers", name="Start (converged)",
                marker={"color": "rgba(20, 100, 200, 0.9)", "size": 7, "symbol": "circle"},
            )
        )
    if len(starts_bad):
        fig.add_trace(
            go.Scatter(
                x=starts_bad[:, 0], y=starts_bad[:, 1],
                mode="markers", name="Start (diverged)",
                marker={"color": "rgba(200, 80, 20, 0.9)", "size": 7, "symbol": "circle-open"},
            )
        )

    # ── Target ────────────────────────────────────────────────────────────────
    tgt = position.final
    fig.add_trace(
        go.Scatter(
            x=[tgt[0]], y=[tgt[1]],
            mode="markers", name="Target",
            marker={"color": "black", "size": 12, "symbol": "star"},
        )
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    n_ok = int(converged.sum())
    fig.update_layout(
        title=f"Batched 2-D obstacle avoidance  —  {B} ICs  |  {n_ok}/{B} converged",
        xaxis_title="x  [m]",
        yaxis_title="y  [m]",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        autosize=False,
        width=720,
        height=720,
        margin={"l": 60, "r": 30, "t": 50, "b": 60},
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Initializing 2-D obstacle avoidance problem (Moreau, B={N_BATCH}) ...")
    problem.initialize()

    # Sample B valid initial positions
    ic_batch = sample_initial_positions(N_BATCH)
    print(f"Sampled {N_BATCH} initial positions in x∈[-14,20], y∈[-14,20]")
    print(f"  ic_batch.shape = {ic_batch.shape}")

    # ── Compile + solve ───────────────────────────────────────────────────────
    # B is inferred from ic_batch.shape[0].  solve_batched vmaps over
    # initial_position (shape (B, 2)) and the per-batch x_guess stack.
    x_guess_batch = build_batched_x_guess(problem, ic_batch)
    print("Compiling and running solve_batched …")
    results = problem.solve_batched(
        parameters={"initial_position": ic_batch},
        x_guess=x_guess_batch,
    )

    # ── Post-process ──────────────────────────────────────────────────────────
    print("Post-processing (nonlinear propagation) …")
    results = problem.post_process_batched(results)

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("Plotting …")
    fig = plot_batched_2d_trajectories(results, ic_batch)
    fig.show()
