"""Plotting helpers for double-integrator examples."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline

from openscvx.algorithms import OptimizationResults


def _ball_center_at(
    t: float,
    t_knots: np.ndarray,
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    *,
    method: str = "cubic",
) -> np.ndarray:
    if method == "pchip":
        from scipy.interpolate import PchipInterpolator

        cx = PchipInterpolator(t_knots, x_knots)(t)
        cy = PchipInterpolator(t_knots, y_knots)(t)
    else:
        cx = CubicSpline(t_knots, x_knots)(t)
        cy = CubicSpline(t_knots, y_knots)(t)
    return np.array([cx, cy])


def _circle_polyline(
    center: np.ndarray, radius: float, n_pts: int = 80
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts)
    return center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)


def _frame_args(duration_ms: int) -> dict:
    return {
        "frame": {"duration": duration_ms, "redraw": True},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": 0},
    }


def plot_moving_safe_zones(results: OptimizationResults, *, animate: bool = True) -> go.Figure:
    """Plot the solved trajectory with moving safe-zone balls.

    When ``animate`` is True (default), the figure includes a Play/Pause control
    and slider that advance the vehicle along the trajectory while the safe-zone
    balls move along their spline paths.
    """
    traj = results.trajectory
    pos = traj["position"]
    t = traj["time"].reshape(-1)
    radius = float(results.plotting_data["ball_radius"])
    t_knots = np.asarray(results.plotting_data["t_knots"], dtype=float)
    ball_paths = results.plotting_data["ball_paths"]
    interp_method = results.plotting_data.get("ball_interp_method", "cubic")
    start = np.asarray(results.plotting_data["start"], dtype=float)
    goal = np.asarray(results.plotting_data["goal"], dtype=float)

    n_balls = len(ball_paths)
    palette = [
        "#2ca02c",
        "#1f77b4",
        "#ff7f0e",
        "#9467bd",
        "#d62728",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    colors = [palette[i % len(palette)] for i in range(n_balls)]
    names = [f"Safe zone {i}" for i in range(n_balls)]
    n_animated_traces = 3 + 2 * n_balls

    # Subsample dense propagated trajectories for a manageable animation length
    n_pts = len(t)
    stride = max(1, n_pts // 60)
    frame_indices = list(range(0, n_pts, stride))
    if frame_indices[-1] != n_pts - 1:
        frame_indices.append(n_pts - 1)

    def _ball_circle_trace(idx: int, center: np.ndarray) -> go.Scatter:
        cx, cy = _circle_polyline(center, radius)
        return go.Scatter(
            x=cx,
            y=cy,
            mode="lines",
            line={"color": colors[idx], "width": 2.5},
            fill="toself",
            fillcolor=_rgba(colors[idx], 0.12),
            name=names[idx],
            hovertemplate=f"{names[idx]}<br>(%{{x:.2f}}, %{{y:.2f}})<extra></extra>",
        )

    # Frame 0 geometry
    centers_0 = [
        _ball_center_at(float(t[0]), t_knots, np.asarray(xk), np.asarray(yk), method=interp_method)
        for xk, yk in ball_paths
    ]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=pos[:1, 0],
                y=pos[:1, 1],
                mode="lines",
                line={"color": "#111111", "width": 3},
                name="Trajectory",
            ),
            go.Scatter(
                x=[pos[0, 0]],
                y=[pos[0, 1]],
                mode="markers",
                marker={"size": 14, "color": "#111111", "line": {"color": "white", "width": 2}},
                name="Vehicle",
            ),
            go.Scatter(
                x=[start[0], goal[0]],
                y=[start[1], goal[1]],
                mode="markers+text",
                marker={"size": 12, "symbol": ["circle", "star"], "color": ["#9467bd", "#d62728"]},
                text=["start", "goal"],
                textposition="top center",
                name="Endpoints",
            ),
            *[_ball_circle_trace(i, centers_0[i]) for i in range(n_balls)],
            *[
                go.Scatter(
                    x=np.asarray(x_knots),
                    y=np.asarray(y_knots),
                    mode="lines",
                    line={"color": colors[idx], "width": 1.5, "dash": "dash"},
                    showlegend=False,
                    hoverinfo="skip",
                    name=f"{names[idx]} path",
                )
                for idx, (x_knots, y_knots) in enumerate(ball_paths)
            ],
        ]
    )

    if animate:
        frames = []
        for frame_no, i in enumerate(frame_indices):
            centers_i = [
                _ball_center_at(
                    float(t[i]), t_knots, np.asarray(xk), np.asarray(yk), method=interp_method
                )
                for xk, yk in ball_paths
            ]
            frame_data = [
                go.Scatter(x=pos[: i + 1, 0], y=pos[: i + 1, 1]),
                go.Scatter(x=[pos[i, 0]], y=[pos[i, 1]]),
                go.Scatter(x=[start[0], goal[0]], y=[start[1], goal[1]]),
                *[
                    go.Scatter(
                        x=_circle_polyline(centers_i[j], radius)[0],
                        y=_circle_polyline(centers_i[j], radius)[1],
                    )
                    for j in range(n_balls)
                ],
                *[
                    go.Scatter(x=np.asarray(x_knots), y=np.asarray(y_knots))
                    for x_knots, y_knots in ball_paths
                ],
            ]
            frames.append(
                go.Frame(name=str(frame_no), data=frame_data, traces=list(range(n_animated_traces)))
            )

        fig.frames = frames

        frame_ms = 80
        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.05,
                    "y": -0.08,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, _frame_args(frame_ms)],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], _frame_args(0)],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "x": 0.12,
                    "y": -0.12,
                    "len": 0.82,
                    "pad": {"b": 10, "t": 40},
                    "currentvalue": {"prefix": "t = ", "suffix": " s", "visible": True},
                    "steps": [
                        {
                            "label": f"{float(t[i]):.2f}",
                            "method": "animate",
                            "args": [[str(frame_no)], _frame_args(0)],
                        }
                        for frame_no, i in enumerate(frame_indices)
                    ],
                }
            ],
        )
    else:
        # Static snapshot: full trajectory plus balls at a few sample times
        fig.data[0].x = pos[:, 0]
        fig.data[0].y = pos[:, 1]
        fig.data[0].mode = "lines+markers"
        fig.data[0].marker = {
            "size": 5,
            "color": t,
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": "time [s]"},
        }
        fig.data[1].visible = False

        sample_times = np.linspace(t[0], t[-1], 5)[1:-1]
        for ti in sample_times:
            for idx, (x_knots, y_knots) in enumerate(ball_paths):
                center = _ball_center_at(
                    ti, t_knots, np.asarray(x_knots), np.asarray(y_knots), method=interp_method
                )
                cx, cy = _circle_polyline(center, radius)
                fig.add_trace(
                    go.Scatter(
                        x=cx,
                        y=cy,
                        mode="lines",
                        line={"color": colors[idx], "width": 1, "dash": "dot"},
                        opacity=0.35,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    fig.update_layout(
        title="Moving Safe Zones — 2D Double Integrator",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        template="plotly_white",
        legend={"x": 0.01, "y": 0.99},
        margin={"b": 120 if animate else 80},
    )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
