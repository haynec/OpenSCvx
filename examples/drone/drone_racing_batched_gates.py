"""Batched quadrotor gate racing: solve over B perturbed gate layouts.

Same 6-DOF gate-racing problem as ``drone_racing.py``, but ``solve_batched``
runs all B solves in one compiled dispatch (Moreau backend).  Each gate
center is an ``ox.Parameter``; a batch draw adds a small random offset to
every gate before solving.  Per-batch ``x_guess`` stacks seed the SCP iterate
with a position linspace through the perturbed gate waypoints.
All ``B`` trajectories and perturbed gate layouts are visualised together in
Viser with simultaneous playback.

Run::

    python examples/drone/drone_racing_batched_gates.py
"""

import os
import sys

import jax.numpy as jnp
import numpy as np
import viser

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem
from openscvx.plotting.viser import (
    add_animation_controls,
    compute_velocity_colors,
    create_server,
)
from openscvx.utils import gen_vertices, rot

# ── Batch configuration ────────────────────────────────────────────────────────
N_BATCH = 16  # parallel solves (increase if you have GPU headroom)
N = 22  # SCP discretisation nodes
TOTAL_TIME = 24.0

# Gaussian perturbation applied to each gate center (metres).
# Lateral σ ≈ 2 m is ~40 % of the gate width; vertical σ is kept smaller.
GATE_PERTURB_STD = np.array([5.0, 5.0, 2.0])

# ── States ─────────────────────────────────────────────────────────────────────
position = ox.State("position", shape=(3,))
position.max = np.array([200.0, 100.0, 200.0])
position.min = np.array([-200.0, -100.0, 15.0])
position.initial = np.array([10.0, 0.0, 20.0])
position.final = [10.0, 0.0, 20.0]

velocity = ox.State("velocity", shape=(3,))
velocity.max = np.array([100.0, 100.0, 100.0])
velocity.min = np.array([-100.0, -100.0, -100.0])
velocity.initial = np.array([0.0, 0.0, 0.0])
velocity.final = [("free", 0.0), ("free", 0.0), ("free", 0.0)]

attitude = ox.State("attitude", shape=(4,))
attitude.max = np.array([1.0, 1.0, 1.0, 1.0])
attitude.min = np.array([-1.0, -1.0, -1.0, -1.0])
attitude.initial = [("free", 1.0), ("free", 0.0), ("free", 0.0), ("free", 0.0)]
attitude.final = [("free", 1.0), ("free", 0.0), ("free", 0.0), ("free", 0.0)]

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = np.array([10.0, 10.0, 10.0])
angular_velocity.min = np.array([-10.0, -10.0, -10.0])
angular_velocity.initial = [("free", 0.0), ("free", 0.0), ("free", 0.0)]
angular_velocity.final = [("free", 0.0), ("free", 0.0), ("free", 0.0)]

# ── Controls ───────────────────────────────────────────────────────────────────
thrust_force = ox.Control("thrust_force", shape=(3,))
thrust_force.max = np.array([0.0, 0.0, 4.179446268 * 9.81])
thrust_force.min = np.array([0.0, 0.0, 0.0])
thrust_force.guess = np.repeat(np.array([[0.0, 0.0, 10.0]]), N, axis=0)

torque = ox.Control("torque", shape=(3,))
torque.max = np.array([18.665, 18.665, 0.55562])
torque.min = np.array([-18.665, -18.665, -0.55562])
torque.guess = np.zeros((N, 3))

m = 1.0
g_const = -9.18
J_b = jnp.array([1.0, 1.0, 1.0])

# ── Gate geometry (nominal layout from drone_racing.py) ───────────────────────
N_GATES = 10
INITIAL_GATE_CENTERS = [
    np.array([59.436, 0.000, 20.0000]),
    np.array([92.964, -23.750, 25.5240]),
    np.array([92.964, -29.274, 20.0000]),
    np.array([92.964, -23.750, 20.0000]),
    np.array([130.150, -23.750, 20.0000]),
    np.array([152.400, -73.152, 20.0000]),
    np.array([92.964, -75.080, 20.0000]),
    np.array([92.964, -68.556, 20.0000]),
    np.array([59.436, -81.358, 20.0000]),
    np.array([22.250, -42.672, 20.0000]),
]

radii = np.array([2.5, 1e-4, 2.5])
A_gate_const = rot @ np.diag(1 / radii) @ rot.T

# Nominal centres used for parameters, guesses, and static plot overlays.
modified_centers = []
for center in INITIAL_GATE_CENTERS:
    modified_center = center.copy()
    modified_center[0] = modified_center[0] + 2.5
    modified_center[2] = modified_center[2] + 2.5
    modified_centers.append(modified_center)
modified_centers = np.asarray(modified_centers)

gate_center_params = []
for i, modified_center in enumerate(modified_centers):
    gate_center_params.append(
        ox.Parameter(f"gate_{i}_center", shape=(3,), value=modified_center)
    )

nodes_per_gate = 2
gate_nodes = np.arange(nodes_per_gate, N, nodes_per_gate)
nominal_vertices = [gen_vertices(c, radii) for c in modified_centers]

# ── Dynamics & constraints ─────────────────────────────────────────────────────
states = [position, velocity, attitude, angular_velocity]
controls = [thrust_force, torque]
constraints = []

for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

for node, gate_center_param in zip(gate_nodes, gate_center_params):
    gate_constraint = (
        (
            ox.linalg.Norm(A_gate_const @ position - A_gate_const @ gate_center_param, ord="inf")
            <= 1.0
        )
        .convex()
        .at([node])
    )
    constraints.append(gate_constraint)

q_norm = ox.linalg.Norm(attitude)
attitude_normalized = attitude / q_norm
J_b_inv = 1.0 / J_b
J_b_diag = ox.linalg.Diag(J_b)

dynamics = {
    "position": velocity,
    "velocity": (1.0 / m) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
    + ox.Constant(np.array([0.0, 0.0, g_const], dtype=np.float64)),
    "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
    "angular_velocity": ox.linalg.Diag(J_b_inv)
    @ (torque - ox.spatial.SSM(angular_velocity) @ J_b_diag @ angular_velocity),
}

position.guess = ox.init.linspace(
    keyframes=[position.initial] + list(modified_centers) + [position.final],
    nodes=[0] + list(gate_nodes) + [N - 1],
)

time_var = ox.Time(
    initial=0.0,
    final=("minimize", TOTAL_TIME),
    min=0.0,
    max=TOTAL_TIME,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time_var,
    constraints=constraints,
    N=N,
    algorithm={"ep_tr": 1e-3},
    solver=ox.MoreauPTRSolver(),
    float_dtype="float64",
)
problem.solver.solver_args = {"abstol": 1e-6, "reltol": 1e-9}


# ── Batching helpers ───────────────────────────────────────────────────────────
def sample_gate_centers_batch(
    base_centers: np.ndarray,
    B: int,
    *,
    std: np.ndarray = GATE_PERTURB_STD,
    seed: int = 0,
) -> np.ndarray:
    """Return shape ``(B, n_gates, 3)`` with small perturbations about *base_centers*."""
    base = np.asarray(base_centers)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=(B, base.shape[0], 3)) * np.asarray(std)
    return base[None, :, :] + noise


def gate_batch_to_parameters(gate_centers_batch: np.ndarray) -> dict:
    """Map a ``(B, n_gates, 3)`` stack to the solver parameter dict."""
    params = {}
    for i in range(gate_centers_batch.shape[1]):
        params[f"gate_{i}_center"] = gate_centers_batch[:, i, :]
    return params


def build_batched_x_guess(problem, gate_centers_batch: np.ndarray) -> np.ndarray:
    """Per-batch state guesses: position linspace through perturbed gate waypoints."""
    base_x = np.asarray(problem.state.x)
    pos_sl = position._slice
    B = gate_centers_batch.shape[0]
    x_guess_batch = np.broadcast_to(base_x, (B,) + base_x.shape).copy()
    keyframe_nodes = [0] + list(gate_nodes) + [N - 1]
    for b in range(B):
        centers_b = gate_centers_batch[b]
        keyframes = [position.initial] + list(centers_b) + [position.final]
        x_guess_batch[b, :, pos_sl] = ox.init.linspace(
            keyframes=keyframes,
            nodes=keyframe_nodes,
        )
    return x_guess_batch


# ── Visualisation ──────────────────────────────────────────────────────────────
def _batch_palette(B: int) -> list[tuple[int, int, int]]:
    """Distinct saturated RGB colours — one hue per batch index."""
    import colorsys

    if B <= 0:
        return []
    return [
        tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / B, 0.82, 0.95))
        for i in range(B)
    ]


def _run_color(
    palette: list[tuple[int, int, int]],
    b: int,
    *,
    converged: bool,
) -> tuple[int, int, int]:
    """Return the batch colour, dimmed when the run did not converge."""
    r, g, bl = palette[b]
    if converged:
        return (r, g, bl)
    return (int(0.45 * r), int(0.45 * g), int(0.45 * bl))


def _add_gate_wireframes(
    server: viser.ViserServer,
    vertices_list: list,
    prefix: str,
    color: tuple[int, int, int],
    *,
    line_width: float = 2.0,
) -> None:
    """Add planar gate wireframes under a unique scene-path *prefix*."""
    for i, verts in enumerate(vertices_list):
        verts = np.asarray(verts, dtype=np.float32)
        edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        points = np.array([[verts[e[0]], verts[e[1]]] for e in edges], dtype=np.float32)
        server.scene.add_line_segments(
            f"{prefix}/gate_{i}",
            points=points,
            colors=color,
            line_width=line_width,
        )


def create_racing_batched_viser_server(
    results,
    gate_centers_batch: np.ndarray,
) -> viser.ViserServer:
    """Build a Viser scene animating all batched gate-racing trajectories.

    All ``B`` propagated trajectories play back simultaneously.  Each batch
    element shares one hue across its gates, static trace, and position
    marker (dimmed when diverged).  A faint nominal gate overlay is included
    for reference.  Animated trails remain velocity-coloured.
    """
    pos_all = np.asarray(results.trajectory["position"], dtype=np.float64)
    vel_all = np.asarray(results.trajectory["velocity"], dtype=np.float64)
    t_full = np.asarray(results.t_full, dtype=np.float64)
    converged = np.asarray(results.converged, dtype=bool).reshape(-1)

    if pos_all.ndim == 2:
        pos_all = pos_all[np.newaxis]
        vel_all = vel_all[np.newaxis]
        t_full = t_full[np.newaxis]

    B, T = pos_all.shape[0], pos_all.shape[1]
    gate_centers_batch = np.asarray(gate_centers_batch, dtype=np.float64)
    palette = _batch_palette(B)

    server = create_server(pos_all.reshape(-1, 3))
    server.gui.configure_theme(dark_mode=True)

    # Nominal gate layout (reference).
    _add_gate_wireframes(
        server,
        nominal_vertices,
        "/gates/nominal",
        color=(90, 90, 90),
        line_width=1.5,
    )

    # Every batch element's perturbed gates (hue keyed to run index).
    for b in range(B):
        verts_b = [gen_vertices(c, radii) for c in gate_centers_batch[b]]
        ok = bool(converged[b])
        run_rgb = _run_color(palette, b, converged=ok)
        _add_gate_wireframes(
            server,
            verts_b,
            f"/gates/batch_{b}",
            run_rgb,
            line_width=2.5 if ok else 1.5,
        )

    start = np.asarray(position.initial, dtype=np.float32)
    server.scene.add_icosphere(
        "/markers/start_finish",
        radius=0.35,
        position=tuple(start),
        color=(240, 240, 240),
    )

    # Faint static traces (full trajectory extent), matched to run hue.
    for b in range(B):
        ok = bool(converged[b])
        trace_rgb = _run_color(palette, b, converged=ok)
        pts = pos_all[b].astype(np.float32)
        server.scene.add_line_segments(
            f"/traces/{b}",
            points=np.stack([pts[:-1], pts[1:]], axis=1),
            colors=trace_rgb,
            line_width=1.2 if ok else 0.8,
        )

    def _make_trail(handle, pts: np.ndarray, cols: np.ndarray):
        def update(frame_idx: int) -> None:
            idx = frame_idx + 1
            handle.points = pts[:idx]
            handle.colors = cols[:idx]

        return update

    def _make_marker(handle, pts: np.ndarray):
        def update(frame_idx: int) -> None:
            handle.position = pts[frame_idx]

        return update

    all_update_cbs = []
    for b in range(B):
        pts_b = pos_all[b].astype(np.float32)
        colors_b = compute_velocity_colors(vel_all[b]).astype(np.uint8)
        ok = bool(converged[b])

        trail_handle = server.scene.add_point_cloud(
            f"/animated/{b}/trail",
            points=pts_b[:1],
            colors=colors_b[:1],
            point_size=0.18,
        )
        marker_handle = server.scene.add_icosphere(
            f"/animated/{b}/marker",
            radius=0.25,
            color=_run_color(palette, b, converged=ok),
            position=tuple(pts_b[0]),
        )

        all_update_cbs.append(_make_trail(trail_handle, pts_b, colors_b))
        all_update_cbs.append(_make_marker(marker_handle, pts_b))

    n_ok = int(converged.sum())
    legend_lines = [
        f"**{B} trajectories animating simultaneously**  ",
        f"{n_ok}/{B} converged — each run has a unique hue on gates, "
        "trace, and marker (dimmed if diverged).",
        "",
        "**Run colours:**",
    ]
    for b in range(B):
        r, g, bl = palette[b]
        status = "✓" if converged[b] else "✗"
        legend_lines.append(
            f'<span style="color:rgb({r},{g},{bl})">■</span> run {b} {status}'
        )
    with server.gui.add_folder("Batch overview"):
        server.gui.add_markdown("\n".join(legend_lines))

    mean_t_final = float(t_full[:, -1].mean())
    traj_time = np.linspace(0.0, mean_t_final, T)
    add_animation_controls(server, traj_time, all_update_cbs, folder_name="Playback")

    return server


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Initializing gate-racing problem (Moreau, B={N_BATCH}) …")
    problem.initialize()

    gate_batch = sample_gate_centers_batch(modified_centers, N_BATCH, seed=42)
    print(f"Sampled {N_BATCH} gate layouts  (σ = {GATE_PERTURB_STD} m)")
    print(f"  gate_batch.shape = {gate_batch.shape}")

    params = gate_batch_to_parameters(gate_batch)
    x_guess_batch = build_batched_x_guess(problem, gate_batch)

    print("Compiling and running solve_batched …")
    results = problem.solve_batched(parameters=params, x_guess=x_guess_batch)

    print("Post-processing (nonlinear propagation) …")
    results = problem.post_process_batched(results)

    n_ok = int(np.asarray(results.converged, dtype=bool).sum())
    print(f"Done — {n_ok}/{N_BATCH} converged")

    print("\nLaunching Viser — all trajectories and gate layouts animating …")
    viser_server = create_racing_batched_viser_server(results, gate_batch)
    viser_server.sleep_forever()
