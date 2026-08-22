"""Per-example matplotlib/plotly figures — **not** part of the ``openscvx`` package.

These are the reference 2D figures for three example families that need a picture
the library's generic ``plot_states`` / ``plot_controls`` cannot draw:

- **Dubins car** (``plot_dubins_car*``, ``plot_velocity_vs_*``) — planar paths with
  waypoint STL windows, plus the publication PDF twin of the waypoint figure.
- **Viewplanning camera** (``plot_camera_view`` and the ``_subject_*`` / ``_camera_*``
  helpers) — subject keypoints projected into the sensor frame.
- **Brachistochrone** (``plot_brachistochrone_*``) — position and speed against the
  analytic cycloid.

Copy what you need into your own project rather than importing from here; nothing in
this module is API-stable. See ``docs/UsersGuide/05_visualization.md``.
"""

import random
from os import PathLike
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from openscvx.algorithms import OptimizationResults
from openscvx.config import Config
from openscvx.plotting.publication import (
    LM_PLOTLY_FONT as _LM_PLOTLY_FONT,
)
from openscvx.plotting.publication import (
    LM_PLOTLY_TICK_FONT as _LM_PLOTLY_TICK_FONT,
)
from openscvx.plotting.publication import (
    latin_modern_fontproperties as _latin_modern_fontproperties,
)
from openscvx.plotting.publication import (
    show_plotly_with_latin_modern,
)
from openscvx.utils import get_kp_pose


def qdcm(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a direction cosine matrix (DCM).

    Args:
        q: Quaternion array [w, x, y, z] where w is the scalar part

    Returns:
        3x3 rotation matrix (direction cosine matrix)
    """
    q_norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
    w, x, y, z = q / q_norm
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def generate_subject_colors(result_or_count, min_rgb=0, max_rgb=255):
    """Generate random RGB colors for subjects/keypoints.

    Args:
        result_or_count: either a result dictionary (checks for 'init_poses') or an integer count
        min_rgb: minimum RGB value (0-255)
        max_rgb: maximum RGB value (0-255)

    Returns:
        List of RGB color strings
    """
    if isinstance(result_or_count, int):
        n_subjects = result_or_count
    else:
        n_subjects = len(result_or_count["init_poses"]) if "init_poses" in result_or_count else 1
    return [
        f"rgb({random.randint(min_rgb, max_rgb)}, {random.randint(min_rgb, max_rgb)}, "
        f"{random.randint(min_rgb, max_rgb)})"
        for _ in range(n_subjects)
    ]


def plot_dubins_car(results: OptimizationResults, params: Config):
    # Plot the trajectory of the Dubins car in 3d as an animaiton
    fig = go.Figure()

    position = results.trajectory["position"]
    x = position[:, 0]
    y = position[:, 1]

    obs_center = results.plotting_data["obs_center"]
    obs_radius = results.plotting_data["obs_radius"]

    # Create a 2D scatter plot
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", line={"color": "blue", "width": 2}, name="Trajectory")
    )

    # Plot the circular obstacle
    fig.add_trace(
        go.Scatter(
            x=obs_center[0] + obs_radius * np.cos(np.linspace(0, 2 * np.pi, 100)),
            y=obs_center[1] + obs_radius * np.sin(np.linspace(0, 2 * np.pi, 100)),
            mode="lines",
            line={"color": "red", "width": 2},
            name="Obstacle",
        )
    )

    fig.update_layout(title="Dubins Car Trajectory", title_x=0.5, template="plotly_dark")

    # Set axis to be equal
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig


class DubinsWaypointStlFigure:
    """Plotly figure with Latin Modern ``show()`` and matplotlib ``save_pdf()``."""

    __slots__ = ("_fig", "_results", "_params")

    def __init__(
        self,
        fig: go.Figure,
        results: OptimizationResults,
        params: Config | None,
    ) -> None:
        self._fig = fig
        self._results = results
        self._params = params

    def show(self, *args, **kwargs) -> None:
        show_plotly_with_latin_modern(self._fig)

    def save_pdf(self, path: str | PathLike[str]) -> None:
        save_dubins_car_waypoint_stl_pdf(self._results, path, self._params)

    def __getattr__(self, name: str):
        return getattr(self._fig, name)


def save_dubins_car_waypoint_stl_pdf(
    results: OptimizationResults,
    path: str | PathLike[str],
    params: Config | None = None,
) -> None:
    """Save the Dubins STL waypoint figure as a PDF (Latin Modern via matplotlib)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    position = np.asarray(results.trajectory["position"], dtype=np.float64)
    x = position[:, 0]
    y = position[:, 1]

    obs_center = np.asarray(results.plotting_data["obs_center"], dtype=np.float64).flatten()
    waypoint_radius = float(np.asarray(results.plotting_data["obs_radius"]).item())
    safety_radius = float(results.plotting_data.get("safety_threshold", waypoint_radius))

    speed = np.asarray(results.trajectory.get("speed"), dtype=np.float64).reshape(-1)
    theta = np.asarray(results.trajectory.get("theta"), dtype=np.float64).reshape(-1)
    vel_x = speed * np.sin(theta)
    vel_y = speed * np.cos(theta)
    vel_norm = np.linalg.norm(np.stack([vel_x, vel_y], axis=1), axis=1)

    lm_fp = _latin_modern_fontproperties()
    if lm_fp is None:
        print("[plot] Latin Modern OTF not found; PDF will use matplotlib default serif.")

    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    center_xy = (float(obs_center[0]), float(obs_center[1]))
    ax.add_patch(
        Circle(
            center_xy,
            safety_radius,
            fill=True,
            facecolor=(0.86, 0.31, 0.31, 0.12),
            edgecolor=(0.86, 0.31, 0.31, 0.9),
            linewidth=2.0,
            label=f"Safety region (r={safety_radius:.2f})",
            zorder=1,
        )
    )
    ax.add_patch(
        Circle(
            center_xy,
            waypoint_radius,
            fill=True,
            facecolor=(0.12, 0.55, 0.24, 0.18),
            edgecolor=(0.12, 0.55, 0.24, 0.95),
            linewidth=2.0,
            label=f"Waypoint ball (r={waypoint_radius:.2f})",
            zorder=2,
        )
    )

    sc = ax.scatter(
        x,
        y,
        c=vel_norm,
        cmap="viridis",
        s=28,
        linewidths=0,
        zorder=4,
        label="Trajectory",
    )
    ax.plot(x, y, color=(0.35, 0.35, 0.35, 0.35), linewidth=0.8, zorder=3)
    ax.plot(
        center_xy[0],
        center_xy[1],
        marker="x",
        color="black",
        markersize=8,
        linestyle="None",
        label="Waypoint center",
        zorder=5,
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("‖v‖₂ (m/s)", fontproperties=lm_fp)
    if lm_fp is not None:
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontproperties(lm_fp)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)", fontproperties=lm_fp)
    ax.set_ylabel("y (m)", fontproperties=lm_fp)
    if lm_fp is not None:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(lm_fp)

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.42, -0.10),
        ncol=2,
        frameon=False,
        prop=lm_fp,
    )
    if lm_fp is not None:
        for text in leg.get_texts():
            text.set_fontproperties(lm_fp)

    ax.grid(True, color=(0.85, 0.85, 0.85), linewidth=0.6)
    fig.subplots_adjust(left=0.12, right=0.82, bottom=0.20, top=0.98)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] Saved Dubins STL figure to {out.resolve()}")


def plot_dubins_car_waypoint_stl(results: OptimizationResults, params: Config):
    """Plot Dubins trajectory for STL waypoint examples (white theme, Latin Modern).

    Call ``.show()`` to open in the browser with Latin Modern on axes, legend,
    and colorbar. Call ``.save_pdf(path)`` for a publication PDF (matplotlib).

    Shows the propagated path colored by the 2-norm of world-frame velocity,
    the waypoint 2-norm ball (``obs_radius``), and the larger safety envelope
    (``safety_threshold``) used for the conditional speed constraint.
    """
    fig = go.Figure()

    position = np.asarray(results.trajectory["position"], dtype=np.float64)
    x = position[:, 0]
    y = position[:, 1]

    obs_center = np.asarray(results.plotting_data["obs_center"], dtype=np.float64).flatten()
    waypoint_radius = float(np.asarray(results.plotting_data["obs_radius"]).item())
    safety_radius = float(results.plotting_data.get("safety_threshold", waypoint_radius))

    speed = np.asarray(results.trajectory.get("speed"), dtype=np.float64).reshape(-1)
    theta = np.asarray(results.trajectory.get("theta"), dtype=np.float64).reshape(-1)
    vel_x = speed * np.sin(theta)
    vel_y = speed * np.cos(theta)
    vel_norm = np.linalg.norm(np.stack([vel_x, vel_y], axis=1), axis=1)

    theta_circle = np.linspace(0, 2 * np.pi, 100)

    def _add_disk(radius: float, line_color: str, fill_color: str, name: str) -> None:
        cx = obs_center[0] + radius * np.cos(theta_circle)
        cy = obs_center[1] + radius * np.sin(theta_circle)
        fig.add_trace(
            go.Scatter(
                x=cx,
                y=cy,
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line={"color": line_color, "width": 2},
                name=name,
                hoverinfo="skip",
            )
        )

    _add_disk(
        safety_radius,
        line_color="rgba(220, 80, 80, 0.9)",
        fill_color="rgba(220, 80, 80, 0.12)",
        name=f"Safety region (r={safety_radius:.2f})",
    )
    _add_disk(
        waypoint_radius,
        line_color="rgba(30, 140, 60, 0.95)",
        fill_color="rgba(30, 140, 60, 0.18)",
        name=f"Waypoint ball (r={waypoint_radius:.2f})",
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line={"color": "rgba(80, 80, 80, 0.35)", "width": 1},
            marker={
                "color": vel_norm,
                "colorscale": "Viridis",
                "size": 7,
                "colorbar": {
                    "title": {"text": "‖v‖₂ (m/s)", "font": _LM_PLOTLY_FONT},
                    "tickfont": _LM_PLOTLY_TICK_FONT,
                },
                "showscale": True,
            },
            name="Trajectory",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[obs_center[0]],
            y=[obs_center[1]],
            mode="markers",
            marker={"color": "black", "size": 10, "symbol": "x"},
            name="Waypoint center",
        )
    )

    _square_px = 640
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_LM_PLOTLY_FONT,
        autosize=False,
        width=_square_px,
        height=_square_px,
        margin={"l": 60, "r": 90, "t": 24, "b": 88},
        legend={
            "orientation": "h",
            "xref": "paper",
            "yref": "paper",
            "x": 0.42,
            "xanchor": "center",
            "y": 0.03,
            "yanchor": "bottom",
            "font": _LM_PLOTLY_FONT,
        },
    )
    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        constrain="domain",
        domain=[0.0, 0.82],
        title_text="x (m)",
        title_font=_LM_PLOTLY_FONT,
        tickfont=_LM_PLOTLY_TICK_FONT,
    )
    fig.update_yaxes(
        constrain="domain",
        domain=[0.10, 1.0],
        title_text="y (m)",
        title_font=_LM_PLOTLY_FONT,
        tickfont=_LM_PLOTLY_TICK_FONT,
    )
    return DubinsWaypointStlFigure(fig, results, params)


def plot_velocity_vs_distance(results: OptimizationResults, params: Config):
    """Plot velocity against distance to obstacle.

    This plot demonstrates how the conditional velocity constraint works,
    showing how velocity changes based on proximity to the obstacle.
    """
    fig = go.Figure()

    # Extract position and velocity along the propagated trajectory
    position = results.trajectory["position"]
    velocity = results.trajectory.get("speed")

    if velocity is None:
        # If speed is not available, try to get it from controls
        velocity = results.controls.get("speed")

    if velocity is None:
        raise ValueError("Velocity data not found in results")

    # Flatten velocity to 1D array
    velocity = np.asarray(velocity).flatten()

    # Get obstacle center and radius
    obs_center = results.plotting_data["obs_center"]
    _ = results.plotting_data["obs_radius"]

    # Calculate distance to obstacle center for each point
    # Distance = ||position - obs_center||
    distance_from_center = np.linalg.norm(position - obs_center, axis=1)

    # Plot velocity vs distance for the full trajectory
    fig.add_trace(
        go.Scatter(
            x=distance_from_center,
            y=velocity,
            mode="lines+markers",
            line={"color": "blue", "width": 2},
            marker={"size": 5},
            name="Velocity (trajectory)",
        )
    )

    # If node-level data is available, overlay the nodes as separate markers
    node_position = results.nodes.get("position")
    node_velocity = results.nodes.get("speed")

    if node_position is not None and node_velocity is not None:
        node_velocity = np.asarray(node_velocity).flatten()
        node_distance_from_center = np.linalg.norm(node_position - obs_center, axis=1)

        fig.add_trace(
            go.Scatter(
                x=node_distance_from_center,
                y=node_velocity,
                mode="markers",
                marker={"size": 9, "color": "cyan", "symbol": "x"},
                name="Velocity (nodes)",
            )
        )

    # Add vertical line at safety threshold if available
    if "safety_threshold" in results.plotting_data:
        safety_threshold = results.plotting_data["safety_threshold"]
        fig.add_vline(
            x=safety_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Safety threshold ({safety_threshold:.2f})",
            annotation_position="top",
        )

    # Add horizontal lines for max velocities
    fig.add_hline(
        y=5.0,
        line_dash="dot",
        line_color="orange",
        annotation_text="Max velocity (near): 5.0",
        annotation_position="right",
    )
    fig.add_hline(
        y=10.0,
        line_dash="dot",
        line_color="green",
        annotation_text="Max velocity (far): 10.0",
        annotation_position="right",
    )

    fig.update_layout(
        title="Velocity vs Distance to Obstacle",
        xaxis_title="Distance from Obstacle Center",
        yaxis_title="Velocity",
        template="plotly_dark",
        title_x=0.5,
    )

    return fig


def plot_velocity_vs_waypoint(results: OptimizationResults, params: Config):
    """Plot velocity against distance to the waypoint ball, using stored parameters.

    This is tailored for STL waypoint examples where a 2-norm ball acts as a
    waypoint and a reduced speed limit is imposed (e.g., inside a safety radius).
    """
    fig = go.Figure()

    # Extract position and velocity along the propagated trajectory
    position = results.trajectory["position"]
    velocity = results.trajectory.get("speed")

    if velocity is None:
        # If speed is not available, try to get it from controls
        velocity = results.controls.get("speed")

    if velocity is None:
        raise ValueError("Velocity data not found in results")

    # Flatten velocity to 1D array
    velocity = np.asarray(velocity).flatten()

    # Get waypoint / obstacle center and radius from plotting data
    obs_center = results.plotting_data["obs_center"]
    obs_radius = results.plotting_data["obs_radius"]

    # Optional reduced-speed level (e.g., inside safety radius). If not provided,
    # fall back to the minimum of the speed data for a reasonable default.
    reduced_speed = results.plotting_data.get("reduced_speed", float(np.min(velocity)))

    # Optional safety radius at which reduced speed begins to apply
    safety_radius = results.plotting_data.get("safety_threshold", float(obs_radius))

    # Distance to waypoint center for each point
    distance_from_center = np.linalg.norm(position - obs_center, axis=1)

    # Plot velocity vs distance for the full trajectory
    fig.add_trace(
        go.Scatter(
            x=distance_from_center,
            y=velocity,
            mode="lines+markers",
            line={"color": "blue", "width": 2},
            marker={"size": 5},
            name="Velocity (trajectory)",
        )
    )

    # Overlay node data if available
    node_position = results.nodes.get("position")
    node_velocity = results.nodes.get("speed")

    if node_position is not None and node_velocity is not None:
        node_velocity = np.asarray(node_velocity).flatten()
        node_distance_from_center = np.linalg.norm(node_position - obs_center, axis=1)

        fig.add_trace(
            go.Scatter(
                x=node_distance_from_center,
                y=node_velocity,
                mode="markers",
                marker={"size": 9, "color": "cyan", "symbol": "x"},
                name="Velocity (nodes)",
            )
        )

    # Mark the safety radius where reduced speed applies
    fig.add_vline(
        x=safety_radius,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Safety radius ({safety_radius:.2f})",
        annotation_position="top",
    )

    # Mark the reduced speed level
    fig.add_hline(
        y=reduced_speed,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Reduced speed ({reduced_speed:.2f})",
        annotation_position="right",
    )

    # Also show the global maximum speed if available from control/state bounds
    global_speed_max = None
    if hasattr(results, "controls") and "speed" in results.controls:
        # Try to infer a reasonable max from data
        global_speed_max = float(np.max(velocity))
    if global_speed_max is not None and global_speed_max > reduced_speed:
        fig.add_hline(
            y=global_speed_max,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Global speed (max ≈ {global_speed_max:.2f})",
            annotation_position="right",
        )

    fig.update_layout(
        title="Velocity vs Distance to Waypoint",
        xaxis_title="Distance from Waypoint Center",
        yaxis_title="Velocity",
        template="plotly_dark",
        title_x=0.5,
    )

    return fig


def plot_dubins_car_disjoint(results: OptimizationResults, params: Config):
    # Plot the trajectory of the Dubins car, but show wp1 and wp2 as circles with centers and radii
    fig = go.Figure()

    position = results.trajectory["position"]
    x = position[:, 0]
    y = position[:, 1]
    # Use the forward velocity from the control input
    velocity = results.trajectory.get("speed")
    if velocity is not None:
        # Flatten to 1D array for Plotly color mapping
        velocity = np.asarray(velocity).flatten()
    else:
        velocity = np.zeros_like(x)

    # Plot the trajectory colored by velocity
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line={"color": "rgba(0,0,0,0)"},  # Hide default line
            marker={
                "color": velocity,
                "colorscale": "Viridis",
                "size": 6,
                "colorbar": {"title": "Velocity"},
                "showscale": True,
            },
            name="Trajectory (velocity)",
        )
    )

    # Plot waypoints wp1 and wp2 as circles and their centers
    # Handle 0, 1, or 2 waypoints
    # Handle wp1 (optional)
    if "wp1_center" in results and "wp1_radius" in results:
        wp1_center = results.get("wp1_center")
        wp1_radius = results.get("wp1_radius")

        # Extract values if they are Parameter objects or other non-array types
        if hasattr(wp1_center, "value"):
            wp1_center = np.asarray(wp1_center.value)
        else:
            wp1_center = np.asarray(wp1_center)

        if hasattr(wp1_radius, "value"):
            wp1_radius = np.asarray(wp1_radius.value)
        else:
            wp1_radius = np.asarray(wp1_radius)

        # Ensure they are scalars/arrays
        wp1_center = np.asarray(wp1_center).flatten()
        wp1_radius = float(np.asarray(wp1_radius).item())

        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = wp1_center[0] + wp1_radius * np.cos(theta)
        circle_y = wp1_center[1] + wp1_radius * np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode="lines",
                line={"color": "green", "width": 2, "dash": "dash"},
                name="Waypoint 1 Area",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[wp1_center[0]],
                y=[wp1_center[1]],
                mode="markers",
                marker={"color": "green", "size": 12, "symbol": "x"},
                name="Waypoint 1 Center",
            )
        )

    # Handle wp2 (optional)
    if "wp2_center" in results and "wp2_radius" in results:
        wp2_center = results.get("wp2_center")
        wp2_radius = results.get("wp2_radius")

        # Extract values if they are Parameter objects or other non-array types
        if hasattr(wp2_center, "value"):
            wp2_center = np.asarray(wp2_center.value)
        else:
            wp2_center = np.asarray(wp2_center)

        if hasattr(wp2_radius, "value"):
            wp2_radius = np.asarray(wp2_radius.value)
        else:
            wp2_radius = np.asarray(wp2_radius)

        # Ensure they are scalars/arrays
        wp2_center = np.asarray(wp2_center).flatten()
        wp2_radius = float(np.asarray(wp2_radius).item())

        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = wp2_center[0] + wp2_radius * np.cos(theta)
        circle_y = wp2_center[1] + wp2_radius * np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode="lines",
                line={"color": "orange", "width": 2, "dash": "dash"},
                name="Waypoint 2 Area",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[wp2_center[0]],
                y=[wp2_center[1]],
                mode="markers",
                marker={"color": "orange", "size": 12, "symbol": "x"},
                name="Waypoint 2 Center",
            )
        )

    fig.update_layout(
        title="Dubins Car Trajectory with Waypoints", title_x=0.5, template="plotly_dark"
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig


def _results_has_moving_subject(results) -> bool:
    """True when keypoints vary in time (parametric, callback, or precomputed trajectory)."""
    if results.get("moving_subject"):
        return True
    if "get_kp_pose" in results:
        return True
    init_poses = results.get("init_poses")
    if init_poses is None:
        return False
    t_ref = results.trajectory.get("time")
    n_ref = len(np.asarray(t_ref).flatten()) if t_ref is not None else None
    for pose in init_poses:
        pose = np.asarray(pose)
        if pose.ndim == 2 and pose.shape[1] == 3 and n_ref is not None and pose.shape[0] == n_ref:
            return True
    return False


def _poses_at_times(pose: np.ndarray, t_samples: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    """World-frame subject positions (len(t_samples), 3) from a static or time-varying pose."""
    pose = np.asarray(pose)
    t_samples = np.asarray(t_samples).flatten()
    t_ref = np.asarray(t_ref).flatten()

    if pose.ndim == 1:
        return np.repeat(pose.reshape(1, 3), len(t_samples), axis=0)

    if pose.ndim == 2 and pose.shape[1] == 3:
        if pose.shape[0] == len(t_samples):
            return pose
        if pose.shape[0] == len(t_ref):
            return np.column_stack(
                [
                    np.interp(t_samples, t_ref, pose[:, i], left=pose[0, i], right=pose[-1, i])
                    for i in range(3)
                ]
            )
        if pose.shape[0] > 1:
            idx = np.linspace(0, pose.shape[0] - 1, len(t_samples)).astype(int)
            return pose[idx]

    raise ValueError(f"Unsupported pose shape {pose.shape} for time sampling.")


def _subject_world_trajectories(results, t_samples: np.ndarray) -> list[np.ndarray]:
    """World-frame (N, 3) trajectories per subject sampled at ``t_samples``."""
    t_samples = np.asarray(t_samples).flatten()
    t_ref = np.asarray(results.trajectory["time"]).flatten()
    subs_traj: list[np.ndarray] = []

    if "get_kp_pose" in results:
        fn = results["get_kp_pose"]
        total_time = results.get("total_time")
        if total_time is None:
            total_time = float(t_ref[-1]) if len(t_ref) else 1.0
        samples = [
            np.asarray(fn(float(t / total_time) if total_time else float(t))).reshape(3)
            for t in t_samples
        ]
        subs_traj.append(np.stack(samples, axis=0))
        return subs_traj

    if "moving_subject" in results and "init_poses" in results:
        init_poses = results.plotting_data["init_poses"]
        raw = init_poses[0] if isinstance(init_poses, list) else init_poses
        offset = np.asarray(raw).reshape(3)
        traj = np.asarray(get_kp_pose(t_samples, offset))
        if traj.ndim == 1:
            traj = traj.reshape(-1, 3)
        subs_traj.append(traj)
        return subs_traj

    if "init_poses" not in results:
        raise ValueError("No valid method to get keypoint poses.")

    init_poses = results.get("init_poses")
    if isinstance(init_poses, np.ndarray) and init_poses.ndim == 2 and init_poses.shape[1] == 3:
        init_poses = [init_poses[i] for i in range(init_poses.shape[0])]
    for pose in init_poses:
        subs_traj.append(_poses_at_times(pose, t_samples, t_ref))

    return subs_traj


def _uses_manipulator_camera_pose(results) -> bool:
    """True when the camera is mounted on an EE with ``ee_position`` in the trajectory."""
    return "ee_position" in results.trajectory


def _wxyz_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    R = np.asarray(R, dtype=np.float64)
    if R.shape == (4, 4):
        R = R[:3, :3]
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def _get_ee_quaternion_trajectory(results) -> np.ndarray:
    """EE attitude (wxyz) on the propagation time grid."""
    stored = results.get("ee_attitude")
    if stored is not None:
        return np.asarray(stored, dtype=np.float64)
    if "ee_attitude" in results.trajectory:
        return np.asarray(results.trajectory["ee_attitude"], dtype=np.float64)
    if "T_j7" not in results.trajectory:
        raise ValueError(
            "Manipulator camera view requires ee_attitude in results/trajectory "
            "or T_j7 + T_home to reconstruct EE orientation."
        )
    t_home = np.asarray(results.get("T_home", np.eye(4)), dtype=np.float64)
    t_j7 = np.asarray(results.trajectory["T_j7"], dtype=np.float64)
    return np.array([_wxyz_from_rotation_matrix(t_j7[i] @ t_home) for i in range(len(t_j7))])


def _camera_poses_at_times(results, t_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera mount position and attitude (wxyz) sampled at ``t_samples``."""
    t_samples = np.asarray(t_samples, dtype=np.float64).flatten()
    t_ref = np.asarray(results.trajectory["time"], dtype=np.float64).flatten()

    if _uses_manipulator_camera_pose(results):
        cam_pos = np.asarray(results.trajectory["ee_position"], dtype=np.float64)
        cam_quat = _get_ee_quaternion_trajectory(results)
    else:
        x_full = results.x_full
        if x_full is None or x_full.shape[1] < 10:
            raise ValueError(
                "Camera pose requires ee_position (manipulator) or x_full with "
                "position + attitude (aerial)."
            )
        cam_pos = np.asarray(x_full[:, 0:3], dtype=np.float64)
        cam_quat = np.asarray(x_full[:, 6:10], dtype=np.float64)

    if len(t_samples) == len(t_ref) and np.allclose(t_samples, t_ref, rtol=0.0, atol=1e-9):
        return cam_pos, cam_quat

    pos = np.column_stack(
        [
            np.interp(t_samples, t_ref, cam_pos[:, i], left=cam_pos[0, i], right=cam_pos[-1, i])
            for i in range(3)
        ]
    )
    idx = np.clip(np.searchsorted(t_ref, t_samples, side="left"), 0, len(t_ref) - 1)
    return pos, cam_quat[idx]


def _project_subjects_to_sensor(
    results, subs_traj: list[np.ndarray], t_samples: np.ndarray
) -> list[np.ndarray]:
    """Map world-frame subject trajectories into the wrist/sensor frame."""
    R_sb = np.asarray(results.plotting_data["R_sb"], dtype=np.float64)
    cam_pos, cam_quat = _camera_poses_at_times(results, t_samples)
    subs_traj_sen = []
    for sub_traj in subs_traj:
        sen = []
        for i, sub_pose in enumerate(sub_traj):
            r_ee = qdcm(cam_quat[i])
            sen.append(R_sb @ r_ee.T @ (np.asarray(sub_pose) - cam_pos[i]))
        subs_traj_sen.append(np.array(sen, dtype=np.float64))
    return subs_traj_sen


def full_subject_traj_time(results: OptimizationResults, params: Config):
    t_nodes = results.nodes["time"]
    t_full = results.trajectory["time"]
    subs_traj = _subject_world_trajectories(results, t_full)
    subs_traj_node = _subject_world_trajectories(results, t_nodes)
    subs_traj_sen = []
    subs_traj_sen_node = []

    if "R_sb" in results:
        subs_traj_sen = _project_subjects_to_sensor(results, subs_traj, t_full)
        subs_traj_sen_node = _project_subjects_to_sensor(results, subs_traj_node, t_nodes)
        return subs_traj, subs_traj_sen, subs_traj_node, subs_traj_sen_node
    else:
        raise ValueError("`R_sb` not found in results. Cannot compute sensor frame.")


def _camera_cone_outline_xy(result, n_grid: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Closed (x, y) polyline for the red camera-frame outline in 2D sensor view."""
    if "alpha_x" not in result or "alpha_y" not in result:
        raise ValueError("`alpha_x` and `alpha_y` not found in results.")
    if "norm_type" not in result:
        raise ValueError("`norm_type` not found in results.")

    A = np.diag([1 / np.tan(np.pi / result["alpha_y"]), 1 / np.tan(np.pi / result["alpha_x"])])
    range_limit = 10 if _results_has_moving_subject(result) else 80
    norm_type = result["norm_type"]
    ord_ = np.inf if norm_type == "inf" else norm_type

    x = np.linspace(-range_limit, range_limit, n_grid)
    y = np.linspace(-range_limit, range_limit, n_grid)
    X, Y = np.meshgrid(x, y)
    X, Y, Z = (
        X.flatten(),
        Y.flatten(),
        np.array(
            [
                np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord=ord_)
                for x_val in x
                for y_val in y
            ]
        ),
    )
    X, Y = X / Z, Y / Z
    order = np.argsort(np.arctan2(Y, X))
    X, Y = X[order], Y[order]
    return np.append(X, X[0]), np.append(Y, Y[0])


def _project_sensor_xy(positions: np.ndarray) -> np.ndarray:
    """Perspective-project sensor-frame (x, y, z) onto the image plane."""
    projected = np.asarray(positions, dtype=np.float64).copy()
    if projected.size == 0:
        return projected
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    return projected


def _camera_subject_traces(
    sub_traj_sen: np.ndarray,
    sub_traj_sen_node: np.ndarray,
    color: str,
    *,
    traj_end: int | None = None,
    n_nodes_visible: int | None = None,
) -> list[go.Scatter]:
    """Build trajectory line + node marker traces for 2D camera plots."""
    sub_traj = _project_sensor_xy(np.asarray(sub_traj_sen))
    if traj_end is not None:
        sub_traj = sub_traj[:traj_end]

    sub_nodes = _project_sensor_xy(np.asarray(sub_traj_sen_node))
    if n_nodes_visible is not None:
        sub_nodes = sub_nodes[:n_nodes_visible]

    return [
        go.Scatter(
            x=sub_traj[:, 0],
            y=sub_traj[:, 1],
            mode="lines",
            line={"color": color, "width": 3},
            showlegend=False,
        ),
        go.Scatter(
            x=sub_nodes[:, 0],
            y=sub_nodes[:, 1],
            mode="markers",
            marker={"color": color, "size": 10},
            showlegend=False,
        ),
    ]


def _add_camera_frame_outline(fig: go.Figure, result) -> None:
    cone_x, cone_y = _camera_cone_outline_xy(result)
    fig.add_trace(
        go.Scatter(
            x=cone_x,
            y=cone_y,
            mode="lines",
            line={"color": "red", "width": 5},
            name=r"$\text{Camera Frame}$",
            showlegend=False,
        )
    )


def _apply_camera_view_layout(
    fig: go.Figure,
    title: str,
    *,
    width: int | None = None,
    height: int | None = None,
    template: str = "plotly_dark",
) -> None:
    """Shared axis styling for static and animated 2D camera views."""
    layout_kwargs: dict = {
        "title": title,
        "title_x": 0.5,
        "title_y": 0.9,
        "template": template,
        "title_font_size": 20,
        "legend_font_size": 15,
        "margin": {"l": 0, "r": 0, "b": 0, "t": 0},
    }
    if template == "simple_white":
        layout_kwargs["paper_bgcolor"] = "white"
        layout_kwargs["plot_bgcolor"] = "white"
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="outside",
        tickwidth=0,
        tickcolor="black",
        range=[-1.1, 1.1],
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="outside",
        tickwidth=0,
        tickcolor="black",
        range=[-1.1, 1.1],
    )
    if width is not None and height is not None:
        fig.update_layout(autosize=False, width=width, height=height)


def plot_camera_view(result: OptimizationResults, params: Config | dict | None = None) -> go.Figure:
    """Static 2D camera view with full keypoint trajectories and SCP nodes.

    Requires viewplanning plotting data on ``result``: ``init_poses``, ``R_sb``,
    ``alpha_x``, ``alpha_y``, and ``norm_type``. Pass a dict as ``params`` to
    call ``update_plotting_data`` (manipulator VP, drone VP). For arms, also set
    ``T_home`` or ``ee_attitude`` when EE orientation is not in ``x_full``.
    """
    if isinstance(params, dict):
        result.update_plotting_data(**params)
    if "init_poses" not in result:
        raise ValueError(
            "plot_camera_view requires viewplanning results (init_poses, R_sb, alpha_x/y, "
            "norm_type). Manipulator examples should use create_snapshot_plotting_server "
            "instead."
        )
    title = r"$\text{Camera View}$"
    _, sub_positions_sen, _, sub_positions_sen_node = full_subject_traj_time(result, params)
    fig = go.Figure()

    _add_camera_frame_outline(fig, result)

    colors = generate_subject_colors(len(sub_positions_sen), min_rgb=10, max_rgb=255)
    for sub_idx, sub_traj in enumerate(sub_positions_sen):
        for trace in _camera_subject_traces(
            sub_traj,
            sub_positions_sen_node[sub_idx],
            colors[sub_idx],
        ):
            fig.add_trace(trace)

    _apply_camera_view_layout(fig, title, width=800, height=800, template="simple_white")

    fig.write_image("figures/camera_view.svg")

    return fig


def plot_brachistochrone_position(result: OptimizationResults, params=None):
    # Plot the position of the brachistochrone problem
    fig = go.Figure()

    position = result.trajectory["position"]
    x = position[:, 0]
    y = position[:, 1]

    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", line={"color": "blue", "width": 2}, name="Position")
    )
    fig.add_trace(
        go.Scatter(
            x=[x[0]], y=[y[0]], mode="markers", marker={"color": "green", "size": 10}, name="Start"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x[-1]], y=[y[-1]], mode="markers", marker={"color": "red", "size": 10}, name="End"
        )
    )

    fig.update_layout(title="Brachistochrone Position", title_x=0.5, template="plotly_dark")
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig


def plot_brachistochrone_velocity(results: OptimizationResults, params=None):
    # Plot the velocity of the brachistochrone problem
    fig = go.Figure()

    tof = results.t_final
    t_full = results.t_full

    v = results.trajectory["velocity"].squeeze()  # scalar velocity

    fig.add_trace(
        go.Scatter(x=t_full, y=v, mode="lines", line={"color": "blue", "width": 2}, name="Velocity")
    )

    fig.update_layout(
        title=f"Brachistochrone Velocity: {tof} seconds", title_x=0.5, template="plotly_dark"
    )
    return fig
