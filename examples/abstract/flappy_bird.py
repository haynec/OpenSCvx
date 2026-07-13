"""Flappy Bird-style 2D navigation with impulsive vertical flaps.

A point mass moves in the plane with constant horizontal speed and gravity.
Vertical velocity is changed only by impulsive delta-v in the y direction
(discrete dynamics at each flap node). The agent must pass through gaps between
pipe obstacles while drifting rightward from a non-zero initial x velocity.

Continuous dynamics (between impulses):

    x_dot = vx
    y_dot = vy
    vx_dot = 0
    vy_dot = -g

Discrete dynamics at impulsive nodes:

    position unchanged
    velocity += [0, flap_magnitude * flap_sign]  (fixed |Δv|, sign ∈ {−1, +1})
"""

import os
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import viser

# Add grandparent directory to path for optional plotting imports
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem
from openscvx.plotting import plot_controls, plot_states

# Discretization
n = 60
total_time = 18.0
g = 9.81
v_x0 = 1.2  # Constant rightward drift (Flappy Bird scroll speed)
flap_magnitude = 2.0  # Every impulsive flap has this |Δv_y|

# Pipe layout: (x_center, gap_center_y, gap_half_height)
pipe_width = 0.35
pipe_buffer = 0.02  # Extra clearance from pipe solids (m)
world_y = (0.0, 1.2)
pipes = [
    (3.5, 0.52, 0.2),
    (7.0, 0.36, 0.2),
    (10.5, 0.60, 0.2),
    (14.0, 0.38, 0.2),
    (17.5, 0.55, 0.2),
    (21.0, 0.34, 0.2),
    (24.5, 0.62, 0.2),
    (28.0, 0.42, 0.2),
]

x0, y0 = 0.0, 0.5
x_final = 32.0

# States
position = ox.State("position", shape=(2,))
position.initial = np.array([x0, y0])
position.final = [x_final, ox.Free(y0)]
position.min = np.array([-1.0, 0.0])
position.max = np.array([35.0, 2.0])

velocity = ox.State("velocity", shape=(2,))
velocity.initial = np.array([v_x0, 0.0])
velocity.final = [ox.Free(v_x0), ox.Free(0.0)]
velocity.min = np.array([0.5, -6.0])
velocity.max = np.array([2.0, 6.0])

# Impulsive flap: delta-v in y only, at every node
impulse_nodes = list(range(n))
delta_v_y = ox.Control(
    "delta_v_y",
    shape=(1,),
    parameterization="impulsive",
    nodes=impulse_nodes,
)
delta_v_y.min = np.array([-3.0])
delta_v_y.max = np.array([5.0])
delta_v_y.guess = np.zeros((n, 1))
delta_v_y.scaling_min = np.array([-3.0])
delta_v_y.scaling_max = np.array([5.0])

states = [position, velocity]
controls = [delta_v_y]

# Continuous single-integrator kinematics with constant vx and gravity on vy
dynamics = {
    "position": velocity,
    "velocity": ox.Concat(0.0, -g),
}

# Discrete map: horizontal velocity unchanged, vertical kick from flap
dynamics_discrete = {
    "position": position,
    "velocity": velocity + ox.Concat(0.0, delta_v_y[0]),
}

# Box constraints on states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Stay outside top/bottom pipe solids (CTCS enforced along the whole horizon).
# For each axis-aligned pipe box, require max signed exterior margin >= 0.
pipe_half_width = 0.5 * pipe_width
for x_c, gap_center, gap_half in pipes:
    y_lo = gap_center - gap_half + pipe_buffer
    y_hi = gap_center + gap_half - pipe_buffer
    x_lo = x_c - pipe_half_width - pipe_buffer
    x_hi = x_c + pipe_half_width + pipe_buffer

    # Bottom pipe: [x_lo, x_hi] x [world_y[0], y_lo] (buffered)
    constraints.append(
        ox.ctcs(
            ox.Max(
                x_lo - position[0],
                position[0] - x_hi,
                world_y[0] - position[1],
                position[1] - y_lo,
            )
            >= 0.0,
            penalty="smooth_relu",
        )
    )
    # Top pipe: [x_lo, x_hi] x [y_hi, world_y[1]]
    constraints.append(
        ox.ctcs(
            ox.Max(
                x_lo - position[0],
                position[0] - x_hi,
                y_hi - position[1],
                position[1] - world_y[1],
            )
            >= 0.0,
            penalty="smooth_relu",
        )
    )

# Initial guess: monotone x, y weaves through gap centers at pipe nodes
x_guess = np.linspace(x0, x_final, n)
y_guess = np.interp(
    x_guess,
    [p[0] for p in pipes],
    [p[1] for p in pipes],
    left=y0,
    right=pipes[-1][1],
)
position.guess = np.column_stack([x_guess, y_guess])
velocity.guess = np.tile(np.array([v_x0, 0.0]), (n, 1))

time = ox.Time(
    initial=0.0,
    final=ox.Free(total_time),
    min=0.0,
    max=30.0,
)

problem = Problem(
    dynamics=dynamics,
    dynamics_discrete=dynamics_discrete,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    algorithm={
        "lam_prox": 1e0,
        "lam_vc": 1e2,
        "ep_tr": 1e-6,
        "autotuner": ox.ConstantProximalWeight(),
    },
    licq_max=1e-8,
)

plotting_dict = {
    "pipes": pipes,
    "pipe_width": pipe_width,
    "pipe_buffer": pipe_buffer,
    "world_y": world_y,
    "g": g,
    "v_x0": v_x0,
}


def _extract_multishot_position_segments(results) -> list[np.ndarray]:
    """Extract per-segment propagated states from SCP discretization history."""
    prop = results.multishot_propagation()
    if prop is None:
        return []
    pos_slice = None
    for state in prop.states:
        if state.name == "position":
            pos_slice = state._slice
            break
    if pos_slice is None:
        return []
    return [seg[:, pos_slice] for seg in prop.segments()]


def _build_singleshot_propagated_trajectory(
    results,
    t_fine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample the final SCP single-shot integrated path onto ``t_fine``.

    Stitches per-segment states from the last ``discretization_history`` entry (the
    same single-shot propagation used inside SCP, not the dense ``post_process`` path).
    """
    segments = _extract_multishot_position_segments(results)
    if not segments:
        pos = np.asarray(results.trajectory["position"], dtype=np.float64)
        vel = np.asarray(results.trajectory.get("velocity"), dtype=np.float64)
        if vel.shape != pos.shape:
            vel = np.tile(np.array([v_x0, 0.0]), (len(pos), 1))
        return pos, vel

    n_nodes = len(results.nodes["position"])
    t_nodes = np.linspace(float(t_fine[0]), float(t_fine[-1]), n_nodes)

    path_rows: list[np.ndarray] = []
    t_path: list[float] = []
    for seg_idx, seg in enumerate(segments):
        t0 = float(t_nodes[seg_idx])
        t1 = float(t_nodes[seg_idx + 1])
        j0 = 0 if seg_idx == 0 else 1
        for j in range(j0, len(seg)):
            alpha = j / (len(seg) - 1) if len(seg) > 1 else 0.0
            t_path.append((1.0 - alpha) * t0 + alpha * t1)
            path_rows.append(np.asarray(seg[j], dtype=np.float64))
    path_xy = np.vstack(path_rows)
    t_path_arr = np.asarray(t_path, dtype=np.float64)
    pos_xy = np.column_stack(
        [
            np.interp(t_fine, t_path_arr, path_xy[:, 0]),
            np.interp(t_fine, t_path_arr, path_xy[:, 1]),
        ]
    )
    vel_xy = np.gradient(pos_xy, t_fine, axis=0)
    return pos_xy, vel_xy


def _plot_flappy(
    results,
    pipes,
    pipe_width=pipe_width,
    pipe_buffer=pipe_buffer,
    world_y=world_y,
):
    """Plot trajectory and pipe obstacles in the x-y plane.

    Overlays:
    - SCP optimization nodes (discrete knot points)
    - Multishot linearized propagation between nodes (from discretization history)
    - Dense nonlinear propagation from ``post_process`` (segment-wise integration)
    """
    import plotly.graph_objects as go

    from openscvx.plotting.publication import LM_PLOTLY_FONT as _LM_PLOTLY_FONT
    from openscvx.plotting.publication import LM_PLOTLY_TICK_FONT as _LM_PLOTLY_TICK_FONT

    pos_nodes = np.asarray(results.nodes["position"])
    fig = go.Figure()

    # Multishot propagation used inside SCP (one polyline per node segment)
    multishot_segments = _extract_multishot_position_segments(results)
    for seg_idx, seg in enumerate(multishot_segments):
        fig.add_trace(
            go.Scatter(
                x=seg[:, 0],
                y=seg[:, 1],
                mode="lines",
                name=r"$\text{SCP multishot propagation}$",
                legendgroup="multishot",
                showlegend=(seg_idx == 0),
                line={"color": "darkorange", "width": 2, "dash": "dot"},
            )
        )

    # Dense nonlinear propagation after solve (also integrated segment-wise)
    if results.trajectory and "position" in results.trajectory:
        pos_prop = np.asarray(results.trajectory["position"])
        fig.add_trace(
            go.Scatter(
                x=pos_prop[:, 0],
                y=pos_prop[:, 1],
                mode="lines",
                name=r"$\text{Nonlinear propagation (post-process)}$",
                line={"color": "limegreen", "width": 2.5},
            )
        )

    # Optimization nodes at discretization points
    fig.add_trace(
        go.Scatter(
            x=pos_nodes[:, 0],
            y=pos_nodes[:, 1],
            mode="markers",
            name=r"$\text{SCP nodes}$",
            marker={"color": "royalblue", "size": 8, "symbol": "circle"},
        )
    )
    half_w = 0.5 * pipe_width + pipe_buffer
    for x_c, gap_c, gap_h in pipes:
        y_top = gap_c + gap_h - pipe_buffer
        y_bot = gap_c - gap_h + pipe_buffer
        # Top pipe (above gap), including buffer
        fig.add_shape(
            type="rect",
            x0=x_c - half_w,
            x1=x_c + half_w,
            y0=y_top,
            y1=world_y[1],
            fillcolor="seagreen",
            opacity=0.45,
            line_width=0,
        )
        # Bottom pipe (below gap), including buffer
        fig.add_shape(
            type="rect",
            x0=x_c - half_w,
            x1=x_c + half_w,
            y0=world_y[0],
            y1=y_bot,
            fillcolor="seagreen",
            opacity=0.45,
            line_width=0,
        )
    fig.update_layout(
        title="Flappy Bird trajectory (impulsive y flaps)",
        xaxis_title=r"$x$",
        yaxis_title=r"$y$",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        showlegend=True,
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_LM_PLOTLY_FONT,
        legend={"font": _LM_PLOTLY_FONT},
    )
    fig.update_xaxes(title_font=_LM_PLOTLY_FONT, tickfont=_LM_PLOTLY_TICK_FONT)
    fig.update_yaxes(title_font=_LM_PLOTLY_FONT, tickfont=_LM_PLOTLY_TICK_FONT)
    return fig


def save_flappy_bird_pdf(
    results,
    pipes,
    path,
    pipe_width=pipe_width,
    pipe_buffer=pipe_buffer,
    world_y=world_y,
) -> None:
    """Save the Flappy Bird trajectory figure as a PDF (Latin Modern via matplotlib)."""
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from openscvx.plotting.publication import (
        latin_modern_fontproperties as _latin_modern_fontproperties,
    )

    lm_fp = _latin_modern_fontproperties()
    if lm_fp is None:
        print("[plot] Latin Modern OTF not found; PDF will use matplotlib default serif.")

    fig, ax = plt.subplots(figsize=(10.0, 3.2), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    half_w = 0.5 * pipe_width + pipe_buffer
    pipe_color = (46 / 255, 168 / 255, 92 / 255, 0.45)
    for x_c, gap_c, gap_h in pipes:
        y_top = gap_c + gap_h - pipe_buffer
        y_bot = gap_c - gap_h + pipe_buffer
        ax.add_patch(
            Rectangle(
                (x_c - half_w, y_top),
                2 * half_w,
                world_y[1] - y_top,
                facecolor=pipe_color,
                edgecolor="none",
                zorder=1,
            )
        )
        ax.add_patch(
            Rectangle(
                (x_c - half_w, world_y[0]),
                2 * half_w,
                y_bot - world_y[0],
                facecolor=pipe_color,
                edgecolor="none",
                zorder=1,
            )
        )

    multishot_segments = _extract_multishot_position_segments(results)
    for seg_idx, seg in enumerate(multishot_segments):
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            color="darkorange",
            linewidth=2.0,
            linestyle=":",
            label="SCP multishot propagation" if seg_idx == 0 else None,
            zorder=3,
        )

    if results.trajectory and "position" in results.trajectory:
        pos_prop = np.asarray(results.trajectory["position"])
        ax.plot(
            pos_prop[:, 0],
            pos_prop[:, 1],
            color="limegreen",
            linewidth=2.5,
            label="Nonlinear propagation (post-process)",
            zorder=4,
        )

    pos_nodes = np.asarray(results.nodes["position"])
    ax.scatter(
        pos_nodes[:, 0],
        pos_nodes[:, 1],
        color="royalblue",
        s=35,
        label="SCP nodes",
        zorder=5,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$", fontproperties=lm_fp)
    ax.set_ylabel(r"$y$", fontproperties=lm_fp)
    if lm_fp is not None:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(lm_fp)

    leg = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False, prop=lm_fp
    )
    if lm_fp is not None:
        for text in leg.get_texts():
            text.set_fontproperties(lm_fp)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.98)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] Saved Flappy Bird figure to {out.resolve()}")


class FlappyBirdFigure:
    """Plotly figure with Latin Modern ``show()`` and matplotlib ``save_pdf()``."""

    __slots__ = ("_fig", "_results", "_pipes", "_pipe_width", "_pipe_buffer", "_world_y")

    def __init__(self, fig, results, pipes, pipe_width, pipe_buffer, world_y) -> None:
        self._fig = fig
        self._results = results
        self._pipes = pipes
        self._pipe_width = pipe_width
        self._pipe_buffer = pipe_buffer
        self._world_y = world_y

    def show(self, *args, **kwargs) -> None:
        from openscvx.plotting.publication import show_plotly_with_latin_modern

        show_plotly_with_latin_modern(self._fig)

    def save_pdf(self, path) -> None:
        save_flappy_bird_pdf(
            self._results,
            self._pipes,
            path,
            pipe_width=self._pipe_width,
            pipe_buffer=self._pipe_buffer,
            world_y=self._world_y,
        )

    def __getattr__(self, name: str):
        return getattr(self._fig, name)


def plot_flappy(
    results,
    pipes,
    pipe_width=pipe_width,
    pipe_buffer=pipe_buffer,
    world_y=world_y,
) -> FlappyBirdFigure:
    """Plot Flappy Bird trajectory with pipes (white theme, Latin Modern, LaTeX labels)."""
    fig = _plot_flappy(
        results,
        pipes,
        pipe_width=pipe_width,
        pipe_buffer=pipe_buffer,
        world_y=world_y,
    )
    return FlappyBirdFigure(fig, results, pipes, pipe_width, pipe_buffer, world_y)


def _xy_to_scene(xy: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Embed planar (x, y) trajectory in Viser 3D with gameplay in the x–y plane."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), z)])


def _add_flappy_pipes(
    server,
    pipes,
    pipe_width: float,
    pipe_buffer: float,
    world_y: tuple[float, float],
    pipe_depth: float = 0.22,
) -> None:
    """Add top/bottom pipe solids as boxes (buffered geometry)."""
    half_w = 0.5 * pipe_width + pipe_buffer
    pipe_color = (46, 168, 92)
    pipe_edge = (28, 110, 58)

    for i, (x_c, gap_c, gap_h) in enumerate(pipes):
        y_top = gap_c + gap_h - pipe_buffer
        y_bot = gap_c - gap_h + pipe_buffer
        x_lo, x_hi = x_c - half_w, x_c + half_w

        bottom_h = max(y_bot - world_y[0], 1e-3)
        server.scene.add_box(
            f"/pipes/{i}/bottom",
            dimensions=(x_hi - x_lo, bottom_h, pipe_depth),
            position=((x_lo + x_hi) / 2, world_y[0] + bottom_h / 2, 0.0),
            color=pipe_color,
        )
        server.scene.add_line_segments(
            f"/pipes/{i}/bottom_edge",
            points=np.array(
                [
                    [[x_lo, y_bot, 0], [x_hi, y_bot, 0]],
                    [[x_lo, world_y[0], 0], [x_hi, world_y[0], 0]],
                ],
                dtype=np.float32,
            ),
            colors=pipe_edge,
            line_width=2.0,
        )

        top_h = max(world_y[1] - y_top, 1e-3)
        server.scene.add_box(
            f"/pipes/{i}/top",
            dimensions=(x_hi - x_lo, top_h, pipe_depth),
            position=((x_lo + x_hi) / 2, y_top + top_h / 2, 0.0),
            color=pipe_color,
        )
        server.scene.add_line_segments(
            f"/pipes/{i}/top_edge",
            points=np.array(
                [
                    [[x_lo, y_top, 0], [x_hi, y_top, 0]],
                    [[x_lo, world_y[1], 0], [x_hi, world_y[1], 0]],
                ],
                dtype=np.float32,
            ),
            colors=pipe_edge,
            line_width=2.0,
        )


def _flappy_follow_camera_pose(
    t: float,
    *,
    x0: float = x0,
    v_x: float = v_x0,
    world_y: tuple[float, float] = world_y,
    cam_z: float = 7.0,
    cam_y_offset: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Side-scroll cam: constant ``v_x``, fixed height, looking into the plane.

    Camera x = ``x0 + v_x * t`` (no vertical motion). Look-at shares that x at a
    fixed mid-world y so orientation stays level while the bird flaps independently.
    """
    examples_dir = os.path.dirname(current_dir)
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    from animations._camera import look_at_wxyz

    cam_x = float(x0 + v_x * t)
    cam_y = 0.5 * (world_y[0] + world_y[1]) + cam_y_offset
    look_y = 0.5 * (world_y[0] + world_y[1])
    cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float64)
    look_at = np.array([cam_x, look_y, 0.0], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    wxyz = look_at_wxyz(cam_pos, look_at, up)
    return cam_pos, wxyz, look_at


def create_flappy_bird_viser_server(
    results,
    pipes=pipes,
    pipe_width: float = pipe_width,
    pipe_buffer: float = pipe_buffer,
    world_y: tuple[float, float] = world_y,
    x_final: float = x_final,
    flap_magnitude: float = flap_magnitude,
    loop_animation: bool = True,
):
    """Interactive Viser animation: bird, pipes, growing path trace, and flap bursts.

    Playback follows the dense ``post_process`` trajectory. A static polyline shows the
    full route immediately; the cyan trace and point cloud grow during playback (press
    Play in the Animation folder). The camera scrolls at constant ``v_x`` with fixed height.
    """
    import viser
    import viser.transforms as vtf

    from openscvx.algorithms import OptimizationResults
    from openscvx.plotting.viser import (
        add_animated_trail,
        add_animation_controls,
        compute_velocity_colors,
    )

    if not isinstance(results, OptimizationResults):
        raise TypeError("results must be an OptimizationResults instance (call post_process first)")

    if not results.trajectory or "position" not in results.trajectory:
        raise ValueError("results.trajectory['position'] missing; call post_process() first")

    t_arr = np.asarray(results.t_full, dtype=np.float64).flatten()
    if t_arr.size < 2:
        t_arr = np.linspace(
            0.0, float(results.t_final), max(len(results.trajectory["position"]), 2)
        )

    pos_xy = np.asarray(results.trajectory["position"], dtype=np.float64)
    vel_xy = np.asarray(results.trajectory.get("velocity"), dtype=np.float64)
    if vel_xy.shape != pos_xy.shape:
        vel_xy = np.gradient(pos_xy, t_arr, axis=0)

    pos = _xy_to_scene(pos_xy).astype(np.float32)
    vel3 = _xy_to_scene(vel_xy)
    colors = compute_velocity_colors(vel3)

    server = viser.ViserServer()
    server.scene.set_up_direction("+y")
    server.gui.configure_theme(dark_mode=True, titlebar_content=None)

    # Sky gradient backdrop (large thin quads)
    sky_w = max(x_final + 6.0, 40.0)
    server.scene.add_box(
        "/backdrop/sky",
        dimensions=(sky_w, world_y[1] - world_y[0], 0.02),
        position=(sky_w / 2 - 2.0, (world_y[0] + world_y[1]) / 2, -0.5),
        color=(120, 190, 255),
    )

    _add_flappy_pipes(server, pipes, pipe_width, pipe_buffer, world_y)

    # Finish line
    server.scene.add_box(
        "/goal",
        dimensions=(0.08, world_y[1] - world_y[0], 0.12),
        position=(x_final, (world_y[0] + world_y[1]) / 2, 0.0),
        color=(255, 220, 60),
    )
    server.scene.add_label(
        "/goal/label",
        text="FINISH",
        position=(x_final, world_y[1] - 0.08, 0.15),
    )

    # Full route visible before playback (frame 0 otherwise shows only one trail point)
    if len(pos) >= 2:
        full_path_segments = np.stack([pos[:-1], pos[1:]], axis=1)
        server.scene.add_line_segments(
            "/trajectory_full",
            points=full_path_segments,
            colors=np.array([70, 160, 220], dtype=np.uint8),
            line_width=2.5,
        )

    _, update_trail = add_animated_trail(server, pos, colors, point_size=0.04)

    # Brighter polyline drawn under the bird as the animation plays
    trace_line = server.scene.add_line_segments(
        "/trajectory_trace",
        points=np.zeros((1, 2, 3), dtype=np.float32),
        colors=np.array([100, 230, 255], dtype=np.uint8),
        line_width=4.0,
    )
    bird = server.scene.add_icosphere(
        "/bird",
        radius=0.09,
        color=(255, 230, 60),
        position=tuple(pos[0]),
    )
    bird_beak = server.scene.add_icosphere(
        "/bird/beak",
        radius=0.035,
        color=(255, 140, 40),
        position=tuple(pos[0] + np.array([0.11, 0.0, 0.0])),
    )

    flap_handle = server.scene.add_line_segments(
        "/flap_impulse",
        points=np.zeros((1, 2, 3), dtype=np.float32),
        colors=np.array([255, 80, 80], dtype=np.uint8),
        line_width=5.0,
    )

    # Precompute nodal flap times for impulse flashes
    n_nodes = len(results.nodes["position"])
    t_nodes = np.linspace(float(t_arr[0]), float(t_arr[-1]), n_nodes)
    dv_nodes = np.asarray(results.nodes.get("delta_v_y", np.zeros((n_nodes, 1)))).reshape(-1)

    def _frame_to_node(frame_idx: int) -> int:
        return int(
            np.clip(np.searchsorted(t_nodes, t_arr[frame_idx], side="right") - 1, 0, n_nodes - 1)
        )

    def update_trace(frame_idx: int) -> None:
        idx = min(frame_idx + 1, len(pos))
        if idx < 2:
            trace_line.points = np.zeros((1, 2, 3), dtype=np.float32)
            return
        trace_line.points = np.stack([pos[: idx - 1], pos[1:idx]], axis=1).astype(np.float32)

    def update_bird(frame_idx: int) -> None:
        p = pos[frame_idx]
        v = vel3[frame_idx]
        # Bank slightly into velocity (rotation about z / view axis)
        pitch = float(np.clip(np.arctan2(v[1], v[0]), -0.7, 0.7))
        rot = vtf.SO3.from_z_radians(pitch * 0.35)
        bird.position = tuple(p)
        bird.wxyz = rot.wxyz
        beak_offset = rot @ np.array([0.14, 0.02, 0.0])
        bird_beak.position = tuple(p + beak_offset)

        node_i = _frame_to_node(frame_idx)
        dv = float(dv_nodes[node_i]) if node_i < len(dv_nodes) else 0.0
        flap_scale = 0.12
        if abs(dv) > 1e-3:
            flap_vec = np.array([0.0, dv * flap_scale, 0.0], dtype=np.float32)
            flap_handle.points = np.array([[p, p + flap_vec]], dtype=np.float32)
            bird.color = (255, 255, 120)
        else:
            flap_handle.points = np.zeros((1, 2, 3), dtype=np.float32)
            bird.color = (255, 230, 60)

    def update_follow_camera(frame_idx: int) -> None:
        cam_pos, cam_wxyz, look_at = _flappy_follow_camera_pose(
            float(t_arr[frame_idx]), world_y=world_y
        )
        for client in server.get_clients().values():
            client.camera.position = tuple(float(x) for x in cam_pos)
            client.camera.wxyz = tuple(float(x) for x in cam_wxyz)
            client.camera.look_at = tuple(float(x) for x in look_at)

    callbacks = [update_bird, update_trace, update_trail, update_follow_camera]
    add_animation_controls(server, t_arr, callbacks, loop=loop_animation)
    update_bird(0)
    update_trace(0)
    update_trail(0)
    update_follow_camera(0)

    # Constant-vx side-scroll: fixed height/orientation, no bobbing with the bird
    cam0, wxyz0, look0 = _flappy_follow_camera_pose(float(t_arr[0]), world_y=world_y)
    server.initial_camera.position = tuple(float(x) for x in cam0)
    server.initial_camera.wxyz = tuple(float(x) for x in wxyz0)
    server.initial_camera.look_at = tuple(float(x) for x in look0)
    server.initial_camera.up = (0.0, 1.0, 0.0)

    @server.on_client_connect
    def _on_client_connect(client) -> None:
        cam_pos, cam_wxyz, look_at = _flappy_follow_camera_pose(
            float(t_arr[0]), world_y=world_y
        )
        client.camera.position = tuple(float(x) for x in cam_pos)
        client.camera.wxyz = tuple(float(x) for x in cam_wxyz)
        client.camera.look_at = tuple(float(x) for x in look_at)

    with server.gui.add_folder("Flappy SCP"):
        server.gui.add_markdown(
            f"**Nodes:** {n_nodes}  \n"
            f"**Flap |Δv_y|:** {flap_magnitude:.2f} m/s  \n"
            f"**Horizon:** {t_arr[-1]:.2f} s  \n"
            f"**Pipes:** {len(pipes)}  \n"
            "**Trail:** post-process trajectory (press Play to animate)  \n"
            f"**Camera:** constant v_x = {v_x0:.2f} m/s, fixed height"
        )

    print(
        "Viser trajectory animation: open the URL above (not the SCP iteration viewer). "
        "Press Play in the Animation folder — camera scrolls at constant v_x."
    )
    return server


def create_flappy_bird_scp_viser_server(
    results,
    pipes=pipes,
    pipe_width: float = pipe_width,
    pipe_buffer: float = pipe_buffer,
    world_y: tuple[float, float] = world_y,
    x_final: float = x_final,
    frame_duration_ms: int = 250,
    propagation_line_width: float = 2.5,
    node_point_size: float = 0.12,
    cmap_name: str = "viridis",
) -> "viser.ViserServer":
    """Viser animation of SCP iteration convergence (nodes + multishot propagation).

    Steps through SCP iterations to show how the discrete trajectory and integrated
    segments converge. Uses a separate Viser port from ``create_flappy_bird_viser_server``.
    """
    import viser

    from openscvx.algorithms import OptimizationResults
    from openscvx.plotting.viser import (
        add_scp_animation_controls,
        add_scp_ghost_iterations,
        add_scp_iteration_nodes,
        add_scp_propagation_lines,
        extract_propagation_positions,
    )

    if not isinstance(results, OptimizationResults):
        raise TypeError("results must be an OptimizationResults instance")

    if not results.X:
        raise ValueError("No SCP iteration history in results.X")

    position_slice = None
    for state in results._states:
        if state.name == "position":
            position_slice = state._slice
            break
    if position_slice is None:
        position_slice = slice(0, 2)

    X_history = [np.asarray(X) for X in results.X]
    positions = [_xy_to_scene(X[:, position_slice]) for X in X_history]
    n_iterations = len(positions)

    server = viser.ViserServer()
    server.scene.set_up_direction("+y")
    server.gui.configure_theme(dark_mode=True, titlebar_content=None)

    sky_w = max(x_final + 6.0, 40.0)
    server.scene.add_box(
        "/backdrop/sky",
        dimensions=(sky_w, world_y[1] - world_y[0], 0.02),
        position=(sky_w / 2 - 2.0, (world_y[0] + world_y[1]) / 2, -0.5),
        color=(120, 190, 255),
    )
    _add_flappy_pipes(server, pipes, pipe_width, pipe_buffer, world_y)

    update_callbacks = []

    _, update_ghosts = add_scp_ghost_iterations(server, positions, cmap_name=cmap_name)
    update_callbacks.append(update_ghosts)

    if results.discretization_history:
        n_x = results.X[0].shape[1]
        n_u = results.U[0].shape[1]
        propagations = extract_propagation_positions(
            results.discretization_history,
            n_x=n_x,
            n_u=n_u,
            position_slice=position_slice,
            scene_scale=1.0,
        )
        propagations = [
            [_xy_to_scene(seg) if seg.shape[-1] == 2 else seg for seg in iter_segs]
            for iter_segs in propagations
        ]
        _, update_propagation = add_scp_propagation_lines(
            server,
            propagations,
            line_width=propagation_line_width,
            cmap_name=cmap_name,
        )
        update_callbacks.append(update_propagation)

    _, update_nodes = add_scp_iteration_nodes(
        server,
        positions,
        point_size=node_point_size,
        cmap_name=cmap_name,
    )
    update_callbacks.append(update_nodes)

    add_scp_animation_controls(
        server,
        n_iterations,
        update_callbacks,
        frame_duration_ms=frame_duration_ms,
        folder_name="SCP Iterations",
    )

    scene_center = np.array([0.5 * x_final, 0.5 * (world_y[0] + world_y[1]), 0.0])
    server.initial_camera.position = tuple(scene_center + np.array([-10.0, 2.0, 16.0]))
    server.initial_camera.look_at = tuple(scene_center)
    server.initial_camera.up = (0.0, 1.0, 0.0)

    with server.gui.add_folder("Flappy SCP"):
        server.gui.add_markdown(
            f"**SCP iterations:** {n_iterations}  \n"
            "Use the slider to scrub convergence.  \n"
            "Orange dotted curves are multishot propagated segments."
        )

    print("Viser SCP: open the local URL above to view iteration convergence.")
    return server


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()
    results.update(plotting_dict)

    plot_states(results).show()
    plot_controls(results).show()
    flappy_fig = plot_flappy(
        results,
        pipes,
        pipe_width=pipe_width,
        pipe_buffer=pipe_buffer,
        world_y=world_y,
    )
    flappy_fig.show()
    flappy_fig.save_pdf("figures/flappy_bird.pdf")

    scp_viser_server = create_flappy_bird_scp_viser_server(
        results,
        pipes=pipes,
        pipe_width=pipe_width,
        pipe_buffer=pipe_buffer,
        world_y=world_y,
        x_final=x_final,
    )
    traj_viser_server = create_flappy_bird_viser_server(
        results,
        pipes=pipes,
        pipe_width=pipe_width,
        pipe_buffer=pipe_buffer,
        world_y=world_y,
        x_final=x_final,
        flap_magnitude=flap_magnitude,
    )
    # Keep both Viser servers alive (separate browser tabs / ports).
    traj_viser_server.sleep_forever()
