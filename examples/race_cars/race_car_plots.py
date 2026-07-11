"""Shared Plotly builders for the race-car examples.

The race-car examples draw in two places, and one family lives here for
each, so every example composes its figures from the same small vocabulary
instead of re-scaffolding Plotly by hand:

* Standalone figures. :func:`track_figure` is the bird's-eye track
  projection — centreline, lane boundaries, distance markers — that each
  example decorates with its own traces (a speed-coloured lap, MPC horizon
  roll-outs, multishot segments). :func:`friction_ellipse` is the dashed
  tyre-grip limit that anchors a g-g diagram.

* Viser telemetry panels. Compact dark-styled figures with one live marker
  per car, animated by the comparison server's playback clock (the
  ``plot_panels`` argument of ``create_race_car_comparison_viser_server``).
  :func:`stripline_panel` runs any per-car signal against any per-car
  x axis — lap distance, race laps — and :func:`gg_panel` plays the cars'
  tyre utilisation live on the friction ellipse.

A panel is a plain dict — ``{"figure", "update", "aspect"}`` — matching the
contract documented on ``create_race_car_comparison_viser_server``. The
``update`` closure takes the playback time in seconds and moves the live
markers; interpolation clamps at each car's final sample, so a car that
finishes early parks its markers just as its 3D model parks at the line.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import plotly.graph_objects as go

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from tracks.readDataFcn import getTrack

# ── Standalone figures ─────────────────────────────────────────────────────────


def track_figure(
    track_file: str,
    *,
    lane_width: float = 0.12,
    distance_marker_step: float | str | None = "auto",
    title: str | None = None,
) -> go.Figure:
    """Bird's-eye scaffold of the track, ready for trajectory traces on top.

    Draws the dashed centreline, the lane boundaries at ``±lane_width``, and
    "x m" arc-length markers, on equal-aspect axes. ``distance_marker_step``
    spaces the markers as in the Viser scene: a number is the spacing in
    metres, ``"auto"`` picks roughly nine per lap, and ``None`` hides them.
    """
    sref, xref, yref, psiref, _ = getTrack(track_file)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xref,
            y=yref,
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            name="centreline",
        )
    )
    for sign in (-1.0, 1.0):
        fig.add_trace(
            go.Scatter(
                x=xref + sign * lane_width * np.sin(psiref),
                y=yref - sign * lane_width * np.cos(psiref),
                mode="lines",
                line=dict(color="black", width=1.5),
                showlegend=False,
            )
        )
    if distance_marker_step is not None:
        if distance_marker_step == "auto":
            step = max(1.0, round(sref[-1] / 9.0))
        else:
            step = float(distance_marker_step)
        for si in np.arange(0.0, sref[-1], step):
            k = int(np.argmin(np.abs(sref - si)))
            fig.add_annotation(
                x=xref[k], y=yref[k], text=f"{si:g}m", showarrow=False, font=dict(size=10)
            )
    fig.update_layout(
        title=title,
        xaxis=dict(title="x [m]", scaleanchor="y"),
        yaxis=dict(title="y [m]"),
        height=600,
    )
    return fig


def friction_ellipse(a_max: float, *, color: str = "black") -> go.Scatter:
    """Dashed grip-limit circle for a g-g diagram, as a single trace."""
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    return go.Scatter(
        x=a_max * np.cos(theta),
        y=a_max * np.sin(theta),
        mode="lines",
        line=dict(color=color, dash="dash", width=1),
        name="friction ellipse",
    )


# ── Viser telemetry panels ─────────────────────────────────────────────────────

# Dark styling matched to viser's control-panel grey so the panels read as
# part of the sidebar rather than white cutouts.
_PANEL_BG = "#1a1b1e"
_PANEL_GRID = "#2c2e33"


def _style_panel(fig: go.Figure, title: str, xaxis: str, yaxis: str) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_PANEL_BG,
        plot_bgcolor=_PANEL_BG,
        title=dict(text=title, font=dict(size=13, color="#c1c2c5")),
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        margin=dict(l=45, r=10, t=30, b=35),
        font=dict(size=10, color="#909296"),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=_PANEL_GRID, zerolinecolor=_PANEL_GRID)
    fig.update_yaxes(gridcolor=_PANEL_GRID, zerolinecolor=_PANEL_GRID)


def _add_live_markers(fig: go.Figure, x: list, y: list, css: list[str]) -> None:
    """One one-point marker trace per car — the dots ``update`` moves."""
    for xi, yi, c in zip(x, y, css):
        fig.add_trace(
            go.Scatter(
                x=xi[:1],
                y=yi[:1],
                mode="markers",
                marker=dict(color=c, size=10, line=dict(color="white", width=1)),
            )
        )


def stripline_panel(
    x: list[np.ndarray],
    y: list[np.ndarray],
    t: list[np.ndarray],
    colors: list[tuple[int, int, int]],
    *,
    title: str,
    xaxis: str,
    yaxis: str,
    aspect: float = 1.9,
) -> dict:
    """Per-car signal lines with a live playback marker each.

    Args:
        x: Per-car x series (e.g. lap distance), one array per car. Lines
            are decimated to ~500 points so panel pushes stay cheap.
        y: Per-car signal sampled on ``x``.
        t: Per-car playback time base aligned with ``x``/``y``; a car whose
            time base ends early holds its final sample.
        colors: Per-car RGB tuples, as passed to the Viser server.
    """
    css = [f"rgb{tuple(c)}" for c in colors]
    fig = go.Figure()
    for xi, yi, c in zip(x, y, css):
        stride = max(1, len(xi) // 500)
        fig.add_trace(
            go.Scatter(x=xi[::stride], y=yi[::stride], mode="lines", line=dict(color=c, width=1.5))
        )
    _add_live_markers(fig, x, y, css)
    _style_panel(fig, title, xaxis, yaxis)

    n_cars = len(css)

    def update(t_now: float) -> None:
        for i in range(n_cars):
            fig.data[n_cars + i].x = (float(np.interp(t_now, t[i], x[i])),)
            fig.data[n_cars + i].y = (float(np.interp(t_now, t[i], y[i])),)

    return {"figure": fig, "update": update, "aspect": aspect}


def gg_panel(
    a_lat: list[np.ndarray],
    a_long: list[np.ndarray],
    t: list[np.ndarray],
    colors: list[tuple[int, int, int]],
    *,
    a_max: float,
    aspect: float = 1.0,
) -> dict:
    """g-g diagram panel: friction circle, a faint cloud per car, one live dot each.

    Args mirror :func:`stripline_panel`, with accelerations in place of the
    x/y series; clouds are decimated to ~200 points per car.
    """
    css = [f"rgb{tuple(c)}" for c in colors]
    fig = go.Figure()
    fig.add_trace(friction_ellipse(a_max, color="gray"))
    for ai, bi, c in zip(a_lat, a_long, css):
        stride = max(1, len(ai) // 200)
        fig.add_trace(
            go.Scatter(
                x=ai[::stride],
                y=bi[::stride],
                mode="markers",
                marker=dict(color=c, size=3, opacity=0.2),
            )
        )
    _add_live_markers(fig, a_lat, a_long, css)
    _style_panel(fig, "g-g vs friction ellipse", "a_lat [m/s²]", "a_long [m/s²]")
    fig.update_yaxes(scaleanchor="x")

    n_cars = len(css)

    def update(t_now: float) -> None:
        for i in range(n_cars):
            fig.data[1 + n_cars + i].x = (float(np.interp(t_now, t[i], a_lat[i])),)
            fig.data[1 + n_cars + i].y = (float(np.interp(t_now, t[i], a_long[i])),)

    return {"figure": fig, "update": update, "aspect": aspect}
