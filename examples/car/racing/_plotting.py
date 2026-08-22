"""Shared Plotly figures for the race-car examples.

Every example in this directory draws the same furniture: the LMS track
outline the car is confined to, the friction ellipse it is grip-limited by,
and — for the multi-car races — the dark telemetry panels that play back in
the Viser sidebar. The helpers here own that furniture, so each example is
left with only what makes it different: its own overlay on the track, its own
tyre-force model behind the g-g points, and its own choice of telemetry.

Nothing here belongs in ``openscvx.plotting``: every function is bound to the
LMS path-parametric track, the friction ellipse, or race telemetry, while the
library deliberately ships domain-generic primitives instead (see
``docs/UsersGuide/05_visualization.md``).

The two ``*_panel`` builders return the ``{"figure", "update", "aspect"}``
dicts that :func:`examples.car.racing._viser.create_race_car_comparison_viser_server`
accepts as ``plot_panels``. They import nothing from that module, so the
dependency between the 2D figures and the 3D scene runs one way.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from examples.car.racing._tracks.readDataFcn import getTrack

# Figure heights. Equal-aspect figures need the extra room; a plain signal
# against time does not, and a stack of them grows by a band per row until it
# fills a screen.
TRACE_HEIGHT = 400
MAP_HEIGHT = 600
MAX_HEIGHT = 1000

# Viser's control-panel greys, so the sidebar panels read as part of the GUI
# rather than as white cutouts.
PANEL_BG = "#1a1b1e"
PANEL_GRID = "#2c2e33"
PANEL_TITLE_FG = "#c1c2c5"
PANEL_FG = "#909296"

# A live panel re-serializes its whole figure on every playback tick, so each
# car's dense log is strided down to at most this many points before it is
# ever drawn. The static figures are not animated and keep every sample.
STRIPLINE_MAX_POINTS = 500
GG_CLOUD_MAX_POINTS = 200


def stack_height(rows: int) -> int:
    """Height of a shared-x stack of ``rows`` signals, capped at a screenful."""
    return min(MAX_HEIGHT, TRACE_HEIGHT + 200 * (rows - 1))


def track_figure(
    track_file: str,
    lane_half_width: float,
    *,
    title: str,
    distance_marker_step: float | None = None,
) -> go.Figure:
    """Empty figure carrying the LMS centreline, both kerbs, and equal axes.

    Callers add their own trajectory overlay on top — the overlay is what
    distinguishes the examples, the outline is what they share. Pass the state
    bound the solver actually enforced as ``lane_half_width`` (e.g.
    ``n.max[0]``) so the drawn kerbs can never drift from the constraint.

    ``distance_marker_step`` spaces the "x m" arc-length labels along the
    centreline in metres, mirroring the argument of the same name on
    :func:`examples.car.racing._viser._add_lms_track_scene`; ``None`` hides them.
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
                x=xref + sign * lane_half_width * np.sin(psiref),
                y=yref - sign * lane_half_width * np.cos(psiref),
                mode="lines",
                line=dict(color="black", width=1.5),
                showlegend=False,
            )
        )

    if distance_marker_step is not None:
        for si in np.arange(0.0, sref[-1], distance_marker_step):
            k = int(np.argmin(np.abs(sref - si)))
            fig.add_annotation(
                x=xref[k], y=yref[k], text=f"{si:g} m", showarrow=False, font=dict(size=10)
            )

    fig.update_layout(
        title=title,
        xaxis=dict(title="x [m]", scaleanchor="y"),
        yaxis=dict(title="y [m]"),
        height=MAP_HEIGHT,
    )
    return fig


def gg_figure(
    a_lat: np.ndarray,
    a_long: np.ndarray,
    speed: np.ndarray,
    *,
    a_max: float,
    title: str = "OpenSCvx — g-g diagram",
) -> go.Figure:
    """The lap's accelerations against the friction circle of radius ``a_max``.

    Points are coloured by speed, so how much of the grip envelope the car
    actually used — and where in the lap it used it — is visible at a glance.
    The tyre-force model behind ``a_lat``/``a_long`` differs per power unit
    and stays in the example that owns it.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=a_max * np.cos(theta),
            y=a_max * np.sin(theta),
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            name="friction ellipse",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=a_lat,
            y=a_long,
            mode="markers",
            marker=dict(
                color=speed,
                colorscale="Rainbow",
                size=4,
                colorbar=dict(title="v [m/s]"),
                showscale=True,
            ),
            name="trajectory",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(title="a_lat [m/s²]", scaleanchor="y"),
        yaxis=dict(title="a_long [m/s²]"),
        height=MAP_HEIGHT,
    )
    return fig


def acceleration_figure(
    t: np.ndarray,
    a_lat: np.ndarray,
    a_long: np.ndarray,
    *,
    a_max: float,
    title: str,
) -> go.Figure:
    """Lateral and longitudinal acceleration vs time, with the ±``a_max`` bounds.

    The time view of :func:`gg_figure`: it shows *when* the box constraints
    the solver enforced are active, where the g-g diagram shows *how far* into
    the envelope the car went.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a_lat, name="a_lat", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=t, y=a_long, name="a_long", line=dict(color="orange")))
    for sign, show in [(1, True), (-1, False)]:
        fig.add_hline(
            y=sign * a_max,
            line=dict(color="black", dash="dash", width=1),
            annotation_text="±bound" if show else None,
        )
    fig.update_layout(
        title=title,
        xaxis_title="t [s]",
        yaxis_title="acceleration [m/s²]",
        height=TRACE_HEIGHT,
    )
    return fig


def style_panel(fig: go.Figure, *, title: str, xaxis: str, yaxis: str) -> go.Figure:
    """Apply the dark viser-sidebar theme so a figure reads as part of the GUI."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        title=dict(text=title, font=dict(size=13, color=PANEL_TITLE_FG)),
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        margin=dict(l=45, r=10, t=30, b=35),
        font=dict(size=10, color=PANEL_FG),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=PANEL_GRID, zerolinecolor=PANEL_GRID)
    fig.update_yaxes(gridcolor=PANEL_GRID, zerolinecolor=PANEL_GRID)
    return fig


def stripline_panel(
    lap_x: list[np.ndarray],
    signal: list[np.ndarray],
    t: list[np.ndarray],
    colors: list[str],
    *,
    title: str,
    yaxis: str,
) -> dict:
    """One telemetry panel: a stripline per car plus a live marker on each.

    The striplines run against race distance in laps (``lap_x``) rather than
    time, so the same corner lines up vertically across cars and laps; the
    ``update(t)`` closure interpolates each car's marker onto its own clock
    (``t``), so a car that has taken the flag parks its marker at the line.
    Every list is indexed by car and must be trimmed identically.
    """
    n_cars = len(signal)
    fig = go.Figure()
    for i in range(n_cars):
        stride = max(1, len(signal[i]) // STRIPLINE_MAX_POINTS)
        fig.add_trace(
            go.Scatter(
                x=lap_x[i][::stride],
                y=signal[i][::stride],
                mode="lines",
                line=dict(color=colors[i], width=1.5),
            )
        )
    for i in range(n_cars):
        fig.add_trace(
            go.Scatter(
                x=lap_x[i][:1],
                y=signal[i][:1],
                mode="markers",
                marker=dict(color=colors[i], size=10, line=dict(color="white", width=1)),
            )
        )
    style_panel(fig, title=title, xaxis="race distance [laps]", yaxis=yaxis)

    def update(now: float) -> None:
        for i in range(n_cars):
            fig.data[n_cars + i].x = (float(np.interp(now, t[i], lap_x[i])),)
            fig.data[n_cars + i].y = (float(np.interp(now, t[i], signal[i])),)

    return {"figure": fig, "update": update, "aspect": 1.9}


def gg_panel(
    gg: list[tuple[np.ndarray, np.ndarray]],
    t: list[np.ndarray],
    colors: list[str],
    *,
    a_max: float,
) -> dict:
    """The g-g diagram as a live panel: friction circle, cloud, and one dot per car.

    The faint cloud is the whole race, the dot is the current instant, so a
    car's cornering and braking style is visible as the race plays back. The
    static twin is :func:`gg_figure`.
    """
    n_cars = len(gg)
    theta = np.linspace(0.0, 2.0 * np.pi, 90)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=a_max * np.cos(theta),
            y=a_max * np.sin(theta),
            mode="lines",
            line=dict(color="gray", dash="dash", width=1),
        )
    )
    for i in range(n_cars):
        a_lat, a_long = gg[i]
        stride = max(1, len(a_lat) // GG_CLOUD_MAX_POINTS)
        fig.add_trace(
            go.Scatter(
                x=a_lat[::stride],
                y=a_long[::stride],
                mode="markers",
                marker=dict(color=colors[i], size=3, opacity=0.2),
            )
        )
    for i in range(n_cars):
        fig.add_trace(
            go.Scatter(
                x=gg[i][0][:1],
                y=gg[i][1][:1],
                mode="markers",
                marker=dict(color=colors[i], size=10, line=dict(color="white", width=1)),
            )
        )
    style_panel(fig, title="g-g vs friction ellipse", xaxis="a_lat [m/s²]", yaxis="a_long [m/s²]")
    fig.update_yaxes(scaleanchor="x")

    def update(now: float) -> None:
        for i in range(n_cars):
            fig.data[1 + n_cars + i].x = (float(np.interp(now, t[i], gg[i][0])),)
            fig.data[1 + n_cars + i].y = (float(np.interp(now, t[i], gg[i][1])),)

    return {"figure": fig, "update": update, "aspect": 1.0}
