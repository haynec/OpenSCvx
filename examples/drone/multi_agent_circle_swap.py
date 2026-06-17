"""Multi-agent planar double integrator: circle swap with collision avoidance.

Ten point-mass agents start evenly spaced on a circle and must reach the
antipodal points (swap sides) while maintaining a minimum separation. Per-agent
double-integrator dynamics are evaluated in parallel via ``ox.Vmap``.

Compare with:
  - ``obstacle_avoidance_vmap_2d.py`` (single agent, Vmap over obstacles)
  - ``2d_obstacle_avoidance_batched_ic.py`` (batched initial conditions)
"""

import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem

# =============================================================================
# Problem configuration
# =============================================================================

N_AGENTS = 20
n = 5
total_time = 5.0
circle_radius = 5.0
safe_distance = 0.25
a_max = 5.0
m = 1.0
pos_bound = 12.0
vel_bound = 8.0

angles = np.linspace(0.0, 2.0 * np.pi, N_AGENTS, endpoint=False)
initial_x = circle_radius * np.cos(angles)
initial_y = circle_radius * np.sin(angles)
final_x = -initial_x
final_y = -initial_y

# =============================================================================
# States and controls (one row per agent; Vmap batches over axis 0)
# =============================================================================

pos_x = ox.State("pos_x", shape=(N_AGENTS,))
pos_x.min = np.full(N_AGENTS, -pos_bound)
pos_x.max = np.full(N_AGENTS, pos_bound)
pos_x.initial = initial_x.tolist()
pos_x.final = final_x.tolist()

pos_y = ox.State("pos_y", shape=(N_AGENTS,))
pos_y.min = np.full(N_AGENTS, -pos_bound)
pos_y.max = np.full(N_AGENTS, pos_bound)
pos_y.initial = initial_y.tolist()
pos_y.final = final_y.tolist()

vel_x = ox.State("vel_x", shape=(N_AGENTS,))
vel_x.min = np.full(N_AGENTS, -vel_bound)
vel_x.max = np.full(N_AGENTS, vel_bound)
vel_x.initial = np.zeros(N_AGENTS).tolist()
vel_x.final = [ox.Free(0.0)] * N_AGENTS

vel_y = ox.State("vel_y", shape=(N_AGENTS,))
vel_y.min = np.full(N_AGENTS, -vel_bound)
vel_y.max = np.full(N_AGENTS, vel_bound)
vel_y.initial = np.zeros(N_AGENTS).tolist()
vel_y.final = [ox.Free(0.0)] * N_AGENTS

force_x = ox.Control("force_x", shape=(N_AGENTS,), parameterization="ZOH")
force_x.min = np.full(N_AGENTS, -a_max)
force_x.max = np.full(N_AGENTS, a_max)

force_y = ox.Control("force_y", shape=(N_AGENTS,), parameterization="ZOH")
force_y.min = np.full(N_AGENTS, -a_max)
force_y.max = np.full(N_AGENTS, a_max)

states = [pos_x, pos_y, vel_x, vel_y]
controls = [force_x, force_y]

# =============================================================================
# Vmapped double-integrator dynamics (one integrator per agent)
# =============================================================================

dynamics = {
    "pos_x": ox.Vmap(lambda vx: vx, batch=vel_x),
    "pos_y": ox.Vmap(lambda vy: vy, batch=vel_y),
    "vel_x": ox.Vmap(lambda fx: (1.0 / m) * fx, batch=force_x),
    "vel_y": ox.Vmap(lambda fy: (1.0 / m) * fy, batch=force_y),
}

# =============================================================================
# Constraints
# =============================================================================

constraints = []
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max),
            ox.ctcs(state.min <= state),
        ]
    )

for i in range(N_AGENTS):
    for j in range(i + 1, N_AGENTS):
        separation = ox.Concat(pos_x[i], pos_y[i]) - ox.Concat(pos_x[j], pos_y[j])
        constraints.append(ox.ctcs(safe_distance <= ox.linalg.Norm(separation)))

# =============================================================================
# Initial guesses (straight-line swap per agent)
# =============================================================================

# Rotate each agent π radians around the circle (preserves pairwise spacing).
t_vals = np.linspace(0.0, 1.0, n)
pos_x.guess = np.zeros((n, N_AGENTS))
pos_y.guess = np.zeros((n, N_AGENTS))
for k, t in enumerate(t_vals):
    angle = angles + np.pi * t
    pos_x.guess[k] = circle_radius * np.cos(angle)
    pos_y.guess[k] = circle_radius * np.sin(angle)
vel_x.guess = np.zeros((n, N_AGENTS))
vel_y.guess = np.zeros((n, N_AGENTS))
force_x.guess = np.zeros((n, N_AGENTS))
force_y.guess = np.zeros((n, N_AGENTS))

# =============================================================================
# Problem
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
    # float_dtype="float64",
    algorithm={"lam_prox": 2e0},
    discretizer=ox.LinearizeDiscretizeSparse(),
    solver=ox.MoreauPTRSolver(),
)


def plot_multi_agent_circle_swap(
    results,
    *,
    circle_radius=circle_radius,
    safe_distance=safe_distance,
    width: int = 720,
    height: int = 720,
):
    """Plot all agent trajectories in the plane."""
    import plotly.graph_objects as go

    from openscvx.plotting.publication import LM_PLOTLY_FONT as _LM_PLOTLY_FONT
    from openscvx.plotting.publication import LM_PLOTLY_TICK_FONT as _LM_PLOTLY_TICK_FONT

    px = np.asarray(results.trajectory["pos_x"], dtype=np.float64)
    py = np.asarray(results.trajectory["pos_y"], dtype=np.float64)
    vx = np.asarray(results.trajectory.get("vel_x"), dtype=np.float64)
    vy = np.asarray(results.trajectory.get("vel_y"), dtype=np.float64)
    speed = np.hypot(vx, vy) if vx.size and vy.size else None

    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    cx = circle_radius * np.cos(theta)
    cy = circle_radius * np.sin(theta)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cx,
            y=cy,
            mode="lines",
            showlegend=False,
            line={"color": "rgba(160, 160, 160, 0.55)", "width": 1, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=-cx,
            y=-cy,
            mode="lines",
            showlegend=False,
            line={"color": "rgba(160, 160, 160, 0.35)", "width": 1, "dash": "dot"},
        )
    )

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for agent in range(N_AGENTS):
        color = colors[agent % len(colors)]
        if speed is not None:
            fig.add_trace(
                go.Scatter(
                    x=px[:, agent],
                    y=py[:, agent],
                    mode="lines+markers",
                    name=f"agent {agent}",
                    line={"color": color, "width": 1.5},
                    marker={
                        "color": speed[:, agent],
                        "colorscale": "Viridis",
                        "size": 4,
                        "showscale": agent == 0,
                        "colorbar": {
                            "title": {"text": r"$\|v\|_2$", "font": _LM_PLOTLY_FONT},
                            "tickfont": _LM_PLOTLY_TICK_FONT,
                        },
                    },
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=px[:, agent],
                    y=py[:, agent],
                    mode="lines",
                    name=f"agent {agent}",
                    line={"color": color, "width": 1.5},
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[px[0, agent]],
                y=[py[0, agent]],
                mode="markers",
                showlegend=False,
                marker={"color": color, "size": 9, "symbol": "circle"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[px[-1, agent]],
                y=[py[-1, agent]],
                mode="markers",
                showlegend=False,
                marker={"color": color, "size": 9, "symbol": "x"},
            )
        )

    fig.update_layout(
        title=f"Multi-agent circle swap (d_min = {safe_distance:g})",
        xaxis_title=r"$x$",
        yaxis_title=r"$y$",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_LM_PLOTLY_FONT,
        autosize=False,
        width=width,
        height=height,
        margin={"l": 60, "r": 90, "t": 48, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    fig.update_xaxes(title_font=_LM_PLOTLY_FONT, tickfont=_LM_PLOTLY_TICK_FONT)
    fig.update_yaxes(title_font=_LM_PLOTLY_FONT, tickfont=_LM_PLOTLY_TICK_FONT)
    return fig


if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    fig = plot_multi_agent_circle_swap(results).show()
