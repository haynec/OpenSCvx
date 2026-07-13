"""Planar double integrator obstacle avoidance with Vmap for parallel constraints.

2D version of ``obstacle_avoidance_vmap.py``: a point mass moves in the x–y plane
with acceleration control and must stay outside many circular obstacles. Obstacle
distances are evaluated in parallel via ``ox.Vmap``.

Compare with:
  - ``obstacle_avoidance_vmap.py`` (3D double integrator with gravity)
  - ``obstacle_avoidance.py`` (manual loop over a few 3D ellipsoids, 6-DOF)
"""

import os
import sys

import numpy as np

# Add grandparent directory to path to import examples.plotting
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem

n = 50  # Number of nodes
total_time = 220.0  # Total time for the simulation

# =============================================================================
# State and Control Definitions (planar double integrator)
# =============================================================================

position = ox.State("position", shape=(2,))
position.max = np.array([150.0, 150.0])
position.min = np.array([-15.0, -15.0])
position.initial = np.array([-10.0, -10.0])
position.final = np.array([100.0, 100.0])

velocity = ox.State("velocity", shape=(2,))
velocity.max = np.array([10.0, 10.0])
velocity.min = np.array([-10.0, -10.0])
velocity.initial = np.array([0.0, 0.0])
velocity.final = [("free", 0.0), ("free", 0.0)]

force = ox.Control("force", shape=(2,))
a_max = 20.0
force.max = np.array([a_max, a_max])
force.min = np.array([-a_max, -a_max])

m = 1.0  # Mass

# =============================================================================
# Obstacle Configuration (2D grid of disks)
# =============================================================================

obstacle_radius_min, obstacle_radius_max = 1.0, 2.5

np.random.seed(42)
obstacle_centers = []

n_rows = 20
n_cols = 20

for i in range(n_rows):
    for j in range(n_cols):
        x = -6.0 + i * 6.0
        y = -7.5 + j * 5.0
        x += np.random.uniform(-1.0, 1.0)
        y += np.random.uniform(-1.0, 1.0)
        obstacle_centers.append([x, y])

n_obstacles = len(obstacle_centers)
obstacle_centers = np.array(obstacle_centers)  # Shape: (n_obstacles, 2)
obstacle_radii = np.random.uniform(obstacle_radius_min, obstacle_radius_max, size=n_obstacles)

print(f"Created {n_obstacles} obstacles")
print(f"Obstacle centers shape: {obstacle_centers.shape}")

# =============================================================================
# Dynamics (planar double integrator: no gravity)
# =============================================================================

dynamics = {
    "position": velocity,
    "velocity": (1.0 / m) * force,
}

# =============================================================================
# Constraints
# =============================================================================

states = [position, velocity]
controls = [force]
constraints = []

for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max),
            ox.ctcs(state.min <= state),
        ]
    )

constraints.extend(
    [
        force <= force.max,
        force.min <= force,
    ]
)

obstacle_avoidance = ox.ctcs(
    obstacle_radii
    <= ox.Vmap(
        lambda obs_center: ox.linalg.Norm(position - obs_center),
        batch=obstacle_centers,
    )
)
constraints.append(obstacle_avoidance)

# =============================================================================
# Initial Guesses
# =============================================================================

straight_line = np.linspace(position.initial, position.final, n)
position.guess = straight_line
velocity.guess = np.zeros((n, 2))
force.guess = np.zeros((n, 2))

# =============================================================================
# Problem Setup
# =============================================================================

time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
)

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
    float_dtype="float64",
)


def plot_obstacle_avoidance_vmap_2d(
    results,
    obstacle_centers=obstacle_centers,
    obstacle_radii=obstacle_radii,
    *,
    width: int = 640,
    height: int = 640,
):
    """Static Plotly view of the planar trajectory and circular obstacles."""
    import plotly.graph_objects as go

    from openscvx.plotting.publication import LM_PLOTLY_FONT as _LM_PLOTLY_FONT
    from openscvx.plotting.publication import LM_PLOTLY_TICK_FONT as _LM_PLOTLY_TICK_FONT

    fig = go.Figure()
    centers = np.asarray(obstacle_centers, dtype=np.float64)
    radii = np.asarray(obstacle_radii, dtype=np.float64).reshape(-1)

    for center, radius in zip(centers, radii):
        xc, yc = center
        fig.add_shape(
            type="circle",
            x0=xc - radius,
            y0=yc - radius,
            x1=xc + radius,
            y1=yc + radius,
            fillcolor="rgba(220, 100, 100, 0.35)",
            line={"color": "rgba(255, 150, 150, 0.55)", "width": 1},
            layer="below",
        )

    position = np.asarray(results.trajectory["position"], dtype=np.float64)
    velocity = np.asarray(results.trajectory.get("velocity"), dtype=np.float64)
    speed = np.linalg.norm(velocity, axis=1) if velocity.size else None

    if speed is not None:
        fig.add_trace(
            go.Scatter(
                x=position[:, 0],
                y=position[:, 1],
                mode="lines+markers",
                showlegend=False,
                line={"color": "rgba(200, 200, 200, 0.35)", "width": 0.5},
                marker={
                    "color": speed,
                    "colorscale": "Viridis",
                    "size": 4,
                    "colorbar": {
                        "title": {"text": r"$\|v\|_2$", "font": _LM_PLOTLY_FONT},
                        "tickfont": _LM_PLOTLY_TICK_FONT,
                    },
                    "showscale": True,
                },
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=position[:, 0],
                y=position[:, 1],
                mode="lines",
                showlegend=False,
                line={"color": "royalblue", "width": 0.5},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[position[0, 0]],
            y=[position[0, 1]],
            mode="markers",
            showlegend=False,
            marker={"color": "white", "size": 10, "symbol": "circle"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[position[-1, 0]],
            y=[position[-1, 1]],
            mode="markers",
            showlegend=False,
            marker={"color": "white", "size": 10, "symbol": "x"},
        )
    )

    fig.update_layout(
        xaxis_title=r"$x$",
        yaxis_title=r"$y$",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font=_LM_PLOTLY_FONT,
        autosize=False,
        width=width,
        height=height,
        margin={"l": 60, "r": 90, "t": 24, "b": 60},
    )
    fig.update_xaxes(
        title_font=_LM_PLOTLY_FONT,
        tickfont=_LM_PLOTLY_TICK_FONT,
        gridcolor="#2c2e33",
        zerolinecolor="#2c2e33",
    )
    fig.update_yaxes(
        title_font=_LM_PLOTLY_FONT,
        tickfont=_LM_PLOTLY_TICK_FONT,
        gridcolor="#2c2e33",
        zerolinecolor="#2c2e33",
    )
    return fig


def save_obstacle_avoidance_vmap_2d_pdf(
    results,
    path,
    obstacle_centers=obstacle_centers,
    obstacle_radii=obstacle_radii,
    *,
    width: int = 640,
    height: int = 640,
) -> None:
    """Save the planar obstacle-avoidance figure as a PDF (Latin Modern via matplotlib)."""
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    from openscvx.plotting.publication import (
        latin_modern_fontproperties as _latin_modern_fontproperties,
    )

    lm_fp = _latin_modern_fontproperties()
    if lm_fp is None:
        print("[plot] Latin Modern OTF not found; PDF will use matplotlib default serif.")

    bg = "#111111"
    fg = "#e0e0e0"
    grid = "#2c2e33"

    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    centers = np.asarray(obstacle_centers, dtype=np.float64)
    radii = np.asarray(obstacle_radii, dtype=np.float64).reshape(-1)
    for center, radius in zip(centers, radii):
        ax.add_patch(
            Circle(
                center,
                radius,
                facecolor=(220 / 255, 100 / 255, 100 / 255, 0.35),
                edgecolor=(255 / 255, 150 / 255, 150 / 255, 0.55),
                linewidth=1.0,
                zorder=1,
            )
        )

    position = np.asarray(results.trajectory["position"], dtype=np.float64)
    velocity = np.asarray(results.trajectory.get("velocity"), dtype=np.float64)
    speed = np.linalg.norm(velocity, axis=1) if velocity.size else None

    if speed is not None:
        ax.plot(
            position[:, 0],
            position[:, 1],
            color=(200 / 255, 200 / 255, 200 / 255, 0.35),
            linewidth=0.5,
            zorder=2,
        )
        sc = ax.scatter(
            position[:, 0],
            position[:, 1],
            c=speed,
            cmap="viridis",
            s=8,
            linewidths=0,
            zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\|v\|_2$", fontproperties=lm_fp, color=fg)
        cbar.ax.yaxis.set_tick_params(color=fg)
        cbar.outline.set_edgecolor(grid)
        plt.setp(cbar.ax.get_yticklabels(), color=fg)
        if lm_fp is not None:
            for lbl in cbar.ax.get_yticklabels():
                lbl.set_fontproperties(lm_fp)
    else:
        ax.plot(
            position[:, 0],
            position[:, 1],
            color="royalblue",
            linewidth=0.5,
            zorder=3,
        )

    ax.plot(
        position[0, 0],
        position[0, 1],
        marker="o",
        color="white",
        markersize=6,
        linestyle="None",
        zorder=4,
    )
    ax.plot(
        position[-1, 0],
        position[-1, 1],
        marker="x",
        color="white",
        markersize=7,
        linestyle="None",
        zorder=4,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$", fontproperties=lm_fp, color=fg)
    ax.set_ylabel(r"$y$", fontproperties=lm_fp, color=fg)
    ax.tick_params(colors=fg)
    ax.spines[:].set_color(grid)
    if lm_fp is not None:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(lm_fp)

    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.10, top=0.98)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight", facecolor=bg, edgecolor=bg)
    plt.close(fig)
    print(f"[plot] Saved 2D obstacle avoidance figure to {out.resolve()}")


class ObstacleAvoidanceVmap2dFigure:
    """Plotly figure with Latin Modern ``show()`` and matplotlib ``save_pdf()``."""

    __slots__ = ("_fig", "_results", "_obstacle_centers", "_obstacle_radii", "_width", "_height")

    def __init__(self, fig, results, obstacle_centers, obstacle_radii, width, height) -> None:
        self._fig = fig
        self._results = results
        self._obstacle_centers = obstacle_centers
        self._obstacle_radii = obstacle_radii
        self._width = width
        self._height = height

    def show(self, *args, **kwargs) -> None:
        from openscvx.plotting.publication import show_plotly_with_latin_modern

        show_plotly_with_latin_modern(self._fig)

    def save_pdf(self, path) -> None:
        save_obstacle_avoidance_vmap_2d_pdf(
            self._results,
            path,
            obstacle_centers=self._obstacle_centers,
            obstacle_radii=self._obstacle_radii,
            width=self._width,
            height=self._height,
        )

    def __getattr__(self, name: str):
        return getattr(self._fig, name)


def plot_obstacle_avoidance_vmap_2d_figure(
    results,
    obstacle_centers=obstacle_centers,
    obstacle_radii=obstacle_radii,
    *,
    width: int = 640,
    height: int = 640,
) -> ObstacleAvoidanceVmap2dFigure:
    """Build the static Plotly figure wrapper for show/save_pdf."""
    fig = plot_obstacle_avoidance_vmap_2d(
        results,
        obstacle_centers=obstacle_centers,
        obstacle_radii=obstacle_radii,
        width=width,
        height=height,
    )
    return ObstacleAvoidanceVmap2dFigure(
        fig, results, obstacle_centers, obstacle_radii, width, height
    )


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    vmap_fig = plot_obstacle_avoidance_vmap_2d_figure(results)
    vmap_fig.show()
    vmap_fig.save_pdf("figures/obstacle_avoidance_vmap_2d.pdf")
