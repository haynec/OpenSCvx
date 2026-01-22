import random

import numpy as np
import plotly.graph_objects as go

# Optional pyqtgraph imports
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from PyQt5 import QtWidgets

    PYQTPHOT_AVAILABLE = True
except ImportError:
    PYQTPHOT_AVAILABLE = False
    pg = None
    gl = None
    QtWidgets = None

from openscvx.algorithms import OptimizationResults
from openscvx.config import Config
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


# Helper functions for plotting polytope connections
def create_connection_line_3d(positions, i, node_a, node_b, color="red", width=3):
    """Create a 3D line connecting two nodes."""
    return go.Scatter3d(
        x=[positions[node_a][i][0], positions[node_b][i][0]],
        y=[positions[node_a][i][1], positions[node_b][i][1]],
        z=[positions[node_a][i][2], positions[node_b][i][2]],
        mode="lines",
        line={"color": color, "width": width},
        showlegend=False,
    )


def create_connection_line_2d_projected(positions, i, node_a, node_b, color="red", width=3):
    """Create a 2D line connecting two nodes with z-normalization (perspective projection)."""
    return go.Scatter(
        x=[
            positions[node_a][i][0] / positions[node_a][i][2],
            positions[node_b][i][0] / positions[node_b][i][2],
        ],
        y=[
            positions[node_a][i][1] / positions[node_a][i][2],
            positions[node_b][i][1] / positions[node_b][i][2],
        ],
        mode="lines",
        line={"color": color, "width": width},
        showlegend=False,
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


def compute_cone_projection(values, A, norm_type, fixed_axis_value=0, axis_index=0):
    """Compute 1D projection of conic constraint.

    Args:
        values: array of values to evaluate along one axis
        A: conic constraint matrix (2x2)
        norm_type: "inf" or numeric norm order
        fixed_axis_value: value for the fixed axis (default 0)
        axis_index: 0 for x-projection, 1 for y-projection

    Returns:
        Array of z-values defining the cone boundary
    """
    z = []
    for val in values:
        vector = [0, 0]
        vector[axis_index] = val
        vector[1 - axis_index] = fixed_axis_value

        if norm_type == "inf":
            z.append(np.linalg.norm(A @ np.array(vector), axis=0, ord=np.inf))
        else:
            z.append(np.linalg.norm(A @ np.array(vector), axis=0, ord=norm_type))
    return np.array(z)


def frame_args(duration):
    """Create frame arguments for plotly animations.

    Args:
        duration: duration in milliseconds

    Returns:
        Dictionary of frame arguments
    """
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


# Modular component functions for building plots
def add_animation_controls(
    fig, slider_x=0.15, slider_y=0.32, play_speed=50, frame_speed=500, button_x=None, button_y=None
):
    """Add animation slider and play/pause controls to a plotly figure.

    Args:
        fig: plotly figure with frames already set
        slider_x: x position of slider (0-1)
        slider_y: y position of slider (0-1)
        play_speed: play button frame duration in ms
        frame_speed: slider frame duration in ms
        button_x: optional x position of buttons (defaults to slider_x)
        button_y: optional y position of buttons (defaults to slider_y)

    Returns:
        fig: modified figure
    """
    # Default button position to slider position if not specified
    if button_x is None:
        button_x = slider_x
    if button_y is None:
        button_y = slider_y

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.8,
            "x": slider_x,
            "y": slider_y,
            "steps": [
                {
                    "args": [[f.name], frame_args(frame_speed)],
                    "label": f.name,
                    "method": "animate",
                }
                for f in fig.frames
            ],
        }
    ]

    updatemenus = [
        {
            "buttons": [
                {
                    "args": [None, frame_args(play_speed)],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": button_x,
            "y": button_y,
        }
    ]

    fig.update_layout(updatemenus=updatemenus, sliders=sliders)
    return fig


def add_cone_surface(fig, A, norm_type, x_range, y_range, opacity=0.25, n_points=20):
    """Add a second-order cone surface to a plotly figure.

    Args:
        fig: plotly figure
        A: 2x2 conic constraint matrix
        norm_type: "inf" or numeric norm order
        x_range: (min, max) tuple for x values
        y_range: (min, max) tuple for y values
        opacity: surface opacity (0-1)
        n_points: mesh resolution

    Returns:
        fig: modified figure
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)

    z = []
    for x_val in x:
        for y_val in y:
            if norm_type == "inf":
                z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord=np.inf))
            else:
                z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord=norm_type))
    z = np.array(z)

    fig.add_trace(
        go.Surface(x=X, y=Y, z=z.reshape(n_points, n_points), opacity=opacity, showscale=False)
    )
    return fig


def add_cone_projections(
    fig, A, norm_type, x_vals, y_vals, x_range, y_range, color="grey", width=3
):
    """Add x-z and y-z plane projections of a cone to a plotly figure.

    Args:
        fig: plotly figure
        A: 2x2 conic constraint matrix
        norm_type: "inf" or numeric norm order
        x_vals: x-coordinate(s) for y-z projection plane
        y_vals: y-coordinate(s) for x-z projection plane
        x_range: (min, max) tuple for x values
        y_range: (min, max) tuple for y values
        color: line color
        width: line width

    Returns:
        fig: modified figure
    """
    x = np.linspace(x_range[0], x_range[1], 20)
    y = np.linspace(y_range[0], y_range[1], 20)

    # X-Z plane projection (y fixed)
    z_x = compute_cone_projection(x, A, norm_type, fixed_axis_value=0, axis_index=0)
    fig.add_trace(
        go.Scatter3d(
            y=x,
            x=y_vals,
            z=z_x,
            mode="lines",
            showlegend=False,
            line={"color": color, "width": width},
        )
    )

    # Y-Z plane projection (x fixed)
    z_y = compute_cone_projection(y, A, norm_type, fixed_axis_value=0, axis_index=1)
    fig.add_trace(
        go.Scatter3d(
            y=x_vals,
            x=y,
            z=z_y,
            mode="lines",
            showlegend=False,
            line={"color": color, "width": width},
        )
    )

    return fig


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


def plot_velocity_vs_distance(results: OptimizationResults, params: Config):
    """Plot velocity against distance to obstacle.

    This plot demonstrates how the conditional velocity constraint works,
    showing how velocity changes based on proximity to the obstacle.
    """
    fig = go.Figure()

    # Extract position and velocity
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

    # Plot velocity vs distance
    fig.add_trace(
        go.Scatter(
            x=distance_from_center,
            y=velocity,
            mode="lines+markers",
            line={"color": "blue", "width": 2},
            marker={"size": 5},
            name="Velocity",
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


def full_subject_traj_time(results: OptimizationResults, params: Config):
    x_full = results.x_full
    x_nodes = results.x.guess
    t_nodes = x_nodes[:, params.sim.time_slice]
    t_full = results.t_full
    subs_traj = []
    subs_traj_node = []
    subs_traj_sen = []
    subs_traj_sen_node = []

    # if hasattr(params.dyn, 'get_kp_pose'):
    if "moving_subject" in results and "init_poses" in results:
        init_poses = results.plotting_data["init_poses"]
        subs_traj.append(get_kp_pose(t_full, init_poses))
        subs_traj_node.append(get_kp_pose(t_nodes, init_poses))
        subs_traj_node[0] = subs_traj_node[0].squeeze()
    elif "init_poses" in results:
        init_poses = results.plotting_data["init_poses"]
        for pose in init_poses:
            # repeat the pose for all time steps
            pose_full = np.repeat(pose[:, np.newaxis], x_full.shape[0], axis=1).T
            subs_traj.append(pose_full)

            pose_node = np.repeat(pose[:, np.newaxis], x_nodes.shape[0], axis=1).T
            subs_traj_node.append(pose_node)
    else:
        raise ValueError("No valid method to get keypoint poses.")

    if "R_sb" in results:
        R_sb = results.plotting_data["R_sb"]
        for sub_traj in subs_traj:
            sub_traj_sen = []
            for i in range(x_full.shape[0]):
                sub_pose = sub_traj[i]
                sub_traj_sen.append(R_sb @ qdcm(x_full[i, 6:10]).T @ (sub_pose - x_full[i, 0:3]))
            subs_traj_sen.append(np.array(sub_traj_sen).squeeze())

        for sub_traj_node in subs_traj_node:
            sub_traj_sen_node = []
            for i in range(x_nodes.shape[0]):
                sub_pose = sub_traj_node[i]
                sub_traj_sen_node.append(
                    R_sb @ qdcm(x_nodes[i, 6:10]).T @ (sub_pose - x_nodes[i, 0:3]).T
                )
            subs_traj_sen_node.append(np.array(sub_traj_sen_node).squeeze())
        return subs_traj, subs_traj_sen, subs_traj_node, subs_traj_sen_node
    else:
        raise ValueError("`R_sb` not found in results. Cannot compute sensor frame.")


def plot_camera_view(result: OptimizationResults, params: Config) -> None:
    title = r"$\text{Camera View}$"
    _, sub_positions_sen, _, sub_positions_sen_node = full_subject_traj_time(result, params)
    fig = go.Figure()

    # Create a cone plot
    A = np.diag(
        [
            1 / np.tan(np.pi / result.plotting_data["alpha_y"]),
            1 / np.tan(np.pi / result.plotting_data["alpha_x"]),
        ]
    )  # Conic Matrix

    # Meshgrid
    if "moving_subject" in result:
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        z = np.linspace(-10, 10, 100)
    else:
        x = np.linspace(-80, 80, 100)
        y = np.linspace(-80, 80, 100)
        z = np.linspace(-80, 80, 100)

    X, Y = np.meshgrid(x, y)

    # Define the condition for the second order cone
    z = []
    for x_val in x:
        for y_val in y:
            if result.plotting_data["norm_type"] == "inf":
                z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord=np.inf))
            else:
                z.append(
                    np.linalg.norm(
                        A @ np.array([x_val, y_val]), axis=0, ord=result.plotting_data["norm_type"]
                    )
                )
    z = np.array(z)

    # Extract the points from the meshgrid
    X = X.flatten()
    Y = Y.flatten()
    Z = z.flatten()

    # Normalize the coordinates by the Z value
    X = X / Z
    Y = Y / Z

    # Order the points so they are connected in radial order about the origin
    order = np.argsort(np.arctan2(Y, X))
    X = X[order]
    Y = Y[order]

    # Repeat the first point to close the cone
    X = np.append(X, X[0])
    Y = np.append(Y, Y[0])

    # Plot the points on a red scatter plot
    fig.add_trace(
        go.Scatter(
            x=X, y=Y, mode="lines", line={"color": "red", "width": 5}, name=r"$\text{Camera Frame}$"
        )
    )

    sub_idx = 0
    for sub_traj in sub_positions_sen:
        color = (
            f"rgb({random.randint(10, 255)}, {random.randint(10, 255)}, {random.randint(10, 255)})"
        )
        sub_traj = np.array(sub_traj)
        sub_traj[:, 0] = sub_traj[:, 0] / sub_traj[:, 2]
        sub_traj[:, 1] = sub_traj[:, 1] / sub_traj[:, 2]
        fig.add_trace(
            go.Scatter(
                x=sub_traj[:, 0],
                y=sub_traj[:, 1],
                mode="lines",
                line={"color": color, "width": 3},
                name=r"$\text{Subject }" + str(sub_idx) + "$",
            )
        )

        sub_traj_nodal = np.array(sub_positions_sen_node[sub_idx])
        sub_traj_nodal[:, 0] = sub_traj_nodal[:, 0] / sub_traj_nodal[:, 2]
        sub_traj_nodal[:, 1] = sub_traj_nodal[:, 1] / sub_traj_nodal[:, 2]
        fig.add_trace(
            go.Scatter(
                x=sub_traj_nodal[:, 0],
                y=sub_traj_nodal[:, 1],
                mode="markers",
                marker={"color": color, "size": 20},
                name=r"$\text{Subject }" + str(sub_idx) + r"\text{ Node}$",
            )
        )
        sub_idx += 1

    # Center the title for the plot
    fig.update_layout(title=title, title_x=0.5)
    fig.update_layout(template="simple_white")

    # Increase title size
    fig.update_layout(title_font_size=20)

    # Increase legend size
    fig.update_layout(legend_font_size=15)

    # fig.update_yaxes(scaleanchor="x", scaleratio=1,)
    fig.update_layout(height=600)

    # Set x axis and y axis limits
    fig.update_xaxes(range=[-1.0, 1.0])
    fig.update_yaxes(range=[-1.0, 1.0])
    # Set aspect ratio to be equal
    fig.update_layout(autosize=False, width=800, height=800)

    # Save figure as svg
    fig.write_image("figures/camera_view.svg")

    return fig


def plot_camera_animation(result: dict, params: Config, path="") -> None:
    title = r"$\text{Camera Animation}$"
    _, subs_positions_sen, _, subs_positions_sen_node = full_subject_traj_time(result, params)
    fig = go.Figure()

    # Add blank plots for the subjects
    for _ in range(50):
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "blue", "width": 2})
        )

    # Create a cone plot
    if "alpha_x" in result and "alpha_y" in result:
        A = np.diag(
            [1 / np.tan(np.pi / result["alpha_y"]), 1 / np.tan(np.pi / result["alpha_x"])]
        )  # Conic Matrix
    else:
        raise ValueError("`alpha_x` and `alpha_y` not found in result dictionary.")

    # Meshgrid
    range_limit = 10 if "moving_subject" in result else 80
    x = np.linspace(-range_limit, range_limit, 50)
    y = np.linspace(-range_limit, range_limit, 50)
    X, Y = np.meshgrid(x, y)

    # Define the condition for the second order cone
    if "norm_type" in result:
        z = np.array(
            [
                np.linalg.norm(
                    A @ np.array([x_val, y_val]),
                    axis=0,
                    ord=(np.inf if result["norm_type"] == "inf" else result["norm_type"]),
                )
                for x_val in x
                for y_val in y
            ]
        )
    else:
        raise ValueError("`norm_type` not found in result dictionary.")

    # Extract the points from the meshgrid
    X, Y, Z = X.flatten(), Y.flatten(), z.flatten()

    # Normalize the coordinates by the Z value
    X, Y = X / Z, Y / Z

    # Order the points so they are connected in radial order about the origin
    order = np.argsort(np.arctan2(Y, X))
    X, Y = X[order], Y[order]

    # Repeat the first point to close the cone
    X, Y = np.append(X, X[0]), np.append(Y, Y[0])

    # Plot the points on a red scatter plot
    fig.add_trace(
        go.Scatter(
            x=X,
            y=Y,
            mode="lines",
            line={"color": "red", "width": 5},
            name=r"$\text{Camera Frame}$",
            showlegend=False,
        )
    )

    # Choose a random color for each subject
    colors = generate_subject_colors(len(subs_positions_sen), min_rgb=10, max_rgb=255)

    frames = []
    # Animate the subjects along their trajectories
    for i in range(0, len(subs_positions_sen[0]), 2):
        frame_data = []
        for sub_idx, sub_traj in enumerate(subs_positions_sen):
            color = colors[sub_idx]
            sub_traj = np.array(sub_traj)
            sub_traj_nodal = np.array(subs_positions_sen_node[sub_idx])
            sub_traj[:, 0] /= sub_traj[:, 2]
            sub_traj[:, 1] /= sub_traj[:, 2]
            frame_data.append(
                go.Scatter(
                    x=sub_traj[: i + 1, 0],
                    y=sub_traj[: i + 1, 1],
                    mode="lines",
                    line={"color": color, "width": 3},
                    showlegend=False,
                )
            )

            # Add in node when loop has reached point where node is present
            scaled_index = int((i // (sub_traj.shape[0] / sub_traj_nodal.shape[0])) + 1)
            sub_node_plot = sub_traj_nodal[:scaled_index]
            sub_node_plot[:, 0] /= sub_node_plot[:, 2]
            sub_node_plot[:, 1] /= sub_node_plot[:, 2]
            frame_data.append(
                go.Scatter(
                    x=sub_node_plot[:, 0],
                    y=sub_node_plot[:, 1],
                    mode="markers",
                    marker={"color": color, "size": 10},
                    showlegend=False,
                )
            )

        frames.append(go.Frame(name=str(i), data=frame_data))

    fig.frames = frames

    # Add animation controls using modular component
    add_animation_controls(fig, slider_x=0.15, slider_y=0.15, play_speed=50, frame_speed=500)

    # Center the title for the plot
    fig.update_layout(title=title, title_x=0.5)
    fig.update_layout(template="plotly_dark")
    # Remove grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Remove center line
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)

    # Increase title size
    fig.update_layout(title_font_size=20)

    # Increase legend size
    fig.update_layout(legend_font_size=15)

    # Remove the axis numbers
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # Remove ticks enrtirely
    fig.update_xaxes(ticks="outside", tickwidth=0, tickcolor="black")
    fig.update_yaxes(ticks="outside", tickwidth=0, tickcolor="black")

    # Set x axis and y axis limits
    fig.update_xaxes(range=[-1.1, 1.1])
    fig.update_yaxes(range=[-1.1, 1.1])

    # Move Title down
    fig.update_layout(title_y=0.9)

    # Set aspect ratio to be equal
    # fig.update_layout(autosize=False, width=650, height=650)
    # Remove marigns
    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

    return fig


def plot_camera_polytope_animation(result: dict, params: Config, path="") -> None:
    sub_positions_sen, _, sub_positions_sen_node = full_subject_traj_time(
        result["x_full"], params, False
    )
    fig = go.Figure()

    # Add blank plots for the subjects
    for _ in range(500):
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "blue", "width": 2})
        )

    # Create a cone plot
    A = np.diag(
        [1 / np.tan(np.pi / params.vp.alpha_y), 1 / np.tan(np.pi / params.vp.alpha_x)]
    )  # Conic Matrix

    # Meshgrid
    range_limit = 10 if params.vp.tracking else 80
    x = np.linspace(-range_limit, range_limit, 50)
    y = np.linspace(-range_limit, range_limit, 50)
    X, Y = np.meshgrid(x, y)

    # Define the condition for the second order cone
    z = np.array(
        [
            np.linalg.norm(
                A @ np.array([x_val, y_val]),
                axis=0,
                ord=(np.inf if params.vp.norm == "inf" else params.vp.norm),
            )
            for x_val in x
            for y_val in y
        ]
    )

    # Extract the points from the meshgrid
    X, Y, Z = X.flatten(), Y.flatten(), z.flatten()

    # Normalize the coordinates by the Z value
    X, Y = X / Z, Y / Z

    # Order the points so they are connected in radial order about the origin
    order = np.argsort(np.arctan2(Y, X))
    X, Y = X[order], Y[order]

    # Repeat the first point to close the cone
    X, Y = np.append(X, X[0]), np.append(Y, Y[0])

    # Plot the points on a red scatter plot
    fig.add_trace(
        go.Scatter(
            x=X,
            y=Y,
            mode="lines",
            line={"color": "red", "width": 5},
            name=r"$\text{Camera Frame}$",
            showlegend=False,
        )
    )

    # Choose a random color for each subject
    [
        f"rgb({random.randint(10, 255)}, {random.randint(10, 255)}, {random.randint(10, 255)})"
        for _ in sub_positions_sen
    ]

    frames = []
    # Animate the subjects along their trajectories
    for i in range(0, len(sub_positions_sen[0]), 2):
        frame_data = []
        for sub_idx, sub_traj in enumerate(sub_positions_sen):
            sub_traj = np.array(sub_traj)
            sub_traj_nodal = np.array(sub_positions_sen_node[sub_idx])
            sub_traj[:, 0] /= sub_traj[:, 2]
            sub_traj[:, 1] /= sub_traj[:, 2]
            frame_data.append(
                go.Scatter(
                    x=sub_traj[: i + 1, 0],
                    y=sub_traj[: i + 1, 1],
                    mode="lines",
                    line={"color": "darkblue", "width": 3},
                    showlegend=False,
                )
            )

            # Add in node when loop has reached point where node is present
            scaled_index = int((i // (sub_traj.shape[0] / sub_traj_nodal.shape[0])) + 1)
            sub_node_plot = sub_traj_nodal[:scaled_index]
            sub_node_plot[:, 0] /= sub_node_plot[:, 2]
            sub_node_plot[:, 1] /= sub_node_plot[:, 2]
            frame_data.append(
                go.Scatter(
                    x=sub_node_plot[:, 0],
                    y=sub_node_plot[:, 1],
                    mode="markers",
                    marker={"color": "darkblue", "size": 10},
                    showlegend=False,
                )
            )

        # Polytope connection topology: node -> [connected nodes]
        polytope_connections = {
            0: [16, 8, 12],
            1: [17, 9, 12],
            2: [16, 13, 10],
            3: [17, 11, 13],
            4: [18, 14, 8],
            5: [19, 9, 14],
            6: [18, 15, 10],
            7: [19, 11, 15],
            8: [0, 4, 10],
            9: [1, 5, 11],
            10: [8, 2, 6],
            11: [3, 7, 9],
            12: [0, 1, 14],
            13: [2, 3, 15],
            14: [4, 5, 12],
            15: [13, 6, 7],
            16: [0, 2, 17],
            17: [1, 3, 16],
            18: [4, 6, 19],
            19: [5, 7, 18],
        }
        # Connect polytope vertices using helper function and topology dictionary
        frame_data.extend(
            [
                create_connection_line_2d_projected(sub_positions_sen, i, node_a, node_b)
                for node_a, connections in polytope_connections.items()
                for node_b in connections
            ]
        )
        frames.append(go.Frame(name=str(i), data=frame_data))

    fig.frames = frames

    # Add animation controls using modular component
    add_animation_controls(fig, slider_x=0.15, slider_y=0.15, play_speed=50, frame_speed=500)

    # Center the title for the plot
    # fig.update_layout(title=title, title_x=0.5)
    fig.update_layout(template="plotly_dark")
    # Remove grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Remove center line
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)

    # Increase title size
    fig.update_layout(title_font_size=20)

    # Increase legend size
    fig.update_layout(legend_font_size=15)

    # Remove the axis numbers
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # Remove ticks enrtirely
    fig.update_xaxes(ticks="outside", tickwidth=0, tickcolor="black")
    fig.update_yaxes(ticks="outside", tickwidth=0, tickcolor="black")

    # Set x axis and y axis limits
    fig.update_xaxes(range=[-1.1, 1.1])
    fig.update_yaxes(range=[-1.1, 1.1])

    # Move Title down
    fig.update_layout(title_y=0.9)

    # Set aspect ratio to be equal
    # fig.update_layout(autosize=False, width=650, height=650)
    # Remove marigns
    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

    return fig


def plot_conic_view_animation(result: dict, params: Config, path="") -> None:
    sub_positions_sen, _, sub_positions_sen_node = full_subject_traj_time(
        result["x_full"], params, False
    )
    fig = go.Figure()
    for i in range(100):
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "blue", "width": 2})
        )

    # Create a cone plot
    if "alpha_x" in result and "alpha_y" in result:
        A = np.diag(
            [1 / np.tan(np.pi / result["alpha_y"]), 1 / np.tan(np.pi / result["alpha_x"])]
        )  # Conic Matrix
    else:
        raise ValueError("`alpha_x` and `alpha_y` not found in result dictionary.")

    # Meshgrid
    if "moving_subject" in result:
        x = np.linspace(-6, 6, 20)
        y = np.linspace(-6, 6, 20)
    else:
        x = np.linspace(-80, 80, 20)
        y = np.linspace(-80, 80, 20)

    X, Y = np.meshgrid(x, y)

    if "norm_type" in result:
        # Add cone surface using helper function
        x_range = (x[0], x[-1])
        y_range = (y[0], y[-1])
        add_cone_surface(fig, A, result["norm_type"], x_range, y_range, opacity=0.25, n_points=20)
        frames = []

        if "moving_subject" in result:
            x_vals = 12 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])
            y_vals = 12 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])
        else:
            x_vals = 110 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])
            y_vals = 110 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])

        # Add cone projections using helper function
        x_range = (x[0], x[-1])
        y_range = (y[0], y[-1])
        add_cone_projections(fig, A, result["norm_type"], x_vals, y_vals, x_range, y_range)
    else:
        raise ValueError("`norm_type` not found in result dictionary.")

    # Choose a random color for each subject
    colors = generate_subject_colors(len(sub_positions_sen), min_rgb=10, max_rgb=255)

    sub_node_plot = []
    for i in range(0, len(sub_positions_sen[0]), 4):
        frame = go.Frame(name=str(i))
        data = []
        sub_idx = 0

        for sub_traj in sub_positions_sen:
            sub_traj = np.array(sub_traj)
            sub_traj_nodal = np.array(sub_positions_sen_node[sub_idx])

            if "moving_subject" in result:
                x_vals = 12 * np.ones_like(sub_traj[: i + 1, 0])
                y_vals = 12 * np.ones_like(sub_traj[: i + 1, 0])
            else:
                x_vals = 110 * np.ones_like(sub_traj[: i + 1, 0])
                y_vals = 110 * np.ones_like(sub_traj[: i + 1, 0])

            data.append(
                go.Scatter3d(
                    x=sub_traj[: i + 1, 0],
                    y=y_vals,
                    z=sub_traj[: i + 1, 2],
                    mode="lines",
                    showlegend=False,
                    line={"color": "grey", "width": 4},
                )
            )
            data.append(
                go.Scatter3d(
                    x=x_vals,
                    y=sub_traj[: i + 1, 1],
                    z=sub_traj[: i + 1, 2],
                    mode="lines",
                    showlegend=False,
                    line={"color": "grey", "width": 4},
                )
            )

            # Add subject position to data
            sub_traj = np.array(sub_traj)
            data.append(
                go.Scatter3d(
                    x=sub_traj[: i + 1, 0],
                    y=sub_traj[: i + 1, 1],
                    z=sub_traj[: i + 1, 2],
                    mode="lines",
                    line={"color": colors[sub_idx], "width": 3},
                    showlegend=False,
                )
            )

            # Add in node when loop has reached point where node is present
            scaled_index = int((i // (sub_traj.shape[0] / sub_traj_nodal.shape[0])) + 1)
            sub_node_plot = sub_traj_nodal[:scaled_index]

            data.append(
                go.Scatter3d(
                    x=sub_node_plot[:, 0],
                    y=sub_node_plot[:, 1],
                    z=sub_node_plot[:, 2],
                    mode="markers",
                    marker={"color": colors[sub_idx], "size": 5},
                    showlegend=False,
                )
            )

            sub_idx += 1

        frame.data = data
        frames.append(frame)

    fig.frames = frames

    # Add animation controls using modular component
    add_animation_controls(fig, slider_x=0.15, slider_y=0.32, play_speed=50, frame_speed=500)

    # Set camera position
    fig.update_layout(
        scene_camera={
            "up": {"x": 0, "y": 0, "z": 10},
            "center": {"x": -2, "y": 0, "z": -3},
            "eye": {"x": -28, "y": -22, "z": 15},
        }
    )

    # Set axis labels
    fig.update_layout(
        scene={"xaxis_title": "x (m)", "yaxis_title": "y (m)", "zaxis_title": "z (m)"}
    )

    fig.update_layout(template="plotly_dark")

    # Make only the grid lines thicker in the template
    fig.update_layout(
        scene={
            "xaxis": {"showgrid": True, "gridwidth": 5},
            "yaxis": {"showgrid": True, "gridwidth": 5},
            "zaxis": {"showgrid": True, "gridwidth": 5},
        }
    )

    fig.update_layout(scene={"aspectmode": "manual", "aspectratio": {"x": 20, "y": 20, "z": 20}})
    # fig.update_layout(autosize=False, width=600, height=600)

    # Remove marigns
    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

    return fig


def plot_conic_view_polytope_animation(result: dict, params: Config, path="") -> None:
    sub_positions_sen, _, sub_positions_sen_node = full_subject_traj_time(
        result["x_full"], params, False
    )
    fig = go.Figure()
    for i in range(500):
        fig.add_trace(
            go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line={"color": "blue", "width": 2})
        )

    # Create a cone plot
    A = np.diag(
        [1 / np.tan(np.pi / params.vp.alpha_y), 1 / np.tan(np.pi / params.vp.alpha_x)]
    )  # Conic Matrix

    # Meshgrid
    if params.vp.tracking:
        x = np.linspace(-6, 6, 20)
        y = np.linspace(-6, 6, 20)
    else:
        x = np.linspace(-80, 80, 20)
        y = np.linspace(-80, 80, 20)

    X, Y = np.meshgrid(x, y)

    # Add cone surface using helper function
    x_range = (x[0], x[-1])
    y_range = (y[0], y[-1])
    add_cone_surface(fig, A, params.vp.norm, x_range, y_range, opacity=0.25, n_points=20)
    frames = []

    if params.vp.tracking:
        x_vals = 12 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])
        y_vals = 12 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])
    else:
        x_vals = 110 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])
        y_vals = 110 * np.ones_like(np.array(sub_positions_sen[0])[:, 0])

    # Add cone projections using helper function
    x_range = (x[0], x[-1])
    y_range = (y[0], y[-1])
    add_cone_projections(fig, A, params.vp.norm, x_vals, y_vals, x_range, y_range)

    for i in range(0, len(sub_positions_sen[0]), 4):
        frame = go.Frame(name=str(i))
        data = []
        sub_idx = 0

        for sub_traj in sub_positions_sen:
            sub_traj = np.array(sub_traj)
            sub_traj_nodal = np.array(sub_positions_sen_node[sub_idx])

            if params.vp.tracking:
                x_vals = 12 * np.ones_like(sub_traj[: i + 1, 0])
                y_vals = 12 * np.ones_like(sub_traj[: i + 1, 0])
            else:
                x_vals = 110 * np.ones_like(sub_traj[: i + 1, 0])
                y_vals = 110 * np.ones_like(sub_traj[: i + 1, 0])

            data.append(
                go.Scatter3d(
                    x=sub_traj[: i + 1, 0],
                    y=y_vals,
                    z=sub_traj[: i + 1, 2],
                    mode="lines",
                    showlegend=False,
                    line={"color": "grey", "width": 4},
                )
            )
            data.append(
                go.Scatter3d(
                    x=x_vals,
                    y=sub_traj[: i + 1, 1],
                    z=sub_traj[: i + 1, 2],
                    mode="lines",
                    showlegend=False,
                    line={"color": "grey", "width": 4},
                )
            )

            # Add subject position to data
            sub_traj = np.array(sub_traj)
            data.append(
                go.Scatter3d(
                    x=sub_traj[: i + 1, 0],
                    y=sub_traj[: i + 1, 1],
                    z=sub_traj[: i + 1, 2],
                    mode="lines",
                    line={"color": "darkblue", "width": 3},
                    showlegend=False,
                )
            )

            # Add in node when loop has reached point where node is present
            scaled_index = int((i // (sub_traj.shape[0] / sub_traj_nodal.shape[0])) + 1)
            sub_traj_nodal[:scaled_index]

            sub_idx += 1

        # Polytope connection topology: node -> [connected nodes]
        polytope_connections = {
            0: [16, 8, 12],
            1: [17, 9, 12],
            2: [16, 13, 10],
            3: [17, 11, 13],
            4: [18, 14, 8],
            5: [19, 9, 14],
            6: [18, 15, 10],
            7: [19, 11, 15],
            8: [0, 4, 12],
            9: [1, 5, 11],
            10: [8, 2, 6],
            11: [3, 7, 9],
            12: [0, 1, 14],
            13: [2, 3, 15],
            14: [4, 5, 12],
            15: [13, 6, 7],
            16: [0, 2, 17],
            17: [1, 3, 16],
            18: [4, 6, 19],
            19: [5, 7, 18],
        }

        # Connect polytope vertices using helper function and topology dictionary
        data.extend(
            [
                create_connection_line_3d(sub_positions_sen, i, node_a, node_b)
                for node_a, connections in polytope_connections.items()
                for node_b in connections
            ]
        )

        frame.data = data
        frames.append(frame)

    fig.frames = frames

    # Add animation controls using modular component
    add_animation_controls(fig, slider_x=0.15, slider_y=0.32, play_speed=50, frame_speed=500)

    # Set camera position
    fig.update_layout(
        scene_camera={
            "up": {"x": 0, "y": 0, "z": 10},
            "center": {"x": -2, "y": 0, "z": -3},
            "eye": {"x": -28, "y": -22, "z": 15},
        }
    )

    # Set axis labels
    fig.update_layout(
        scene={"xaxis_title": "x (m)", "yaxis_title": "y (m)", "zaxis_title": "z (m)"}
    )

    fig.update_layout(template="plotly_dark")

    # Make only the grid lines thicker in the template
    fig.update_layout(
        scene={
            "xaxis": {"showgrid": True, "gridwidth": 5},
            "yaxis": {"showgrid": True, "gridwidth": 5},
            "zaxis": {"showgrid": True, "gridwidth": 5},
        }
    )

    fig.update_layout(scene={"aspectmode": "manual", "aspectratio": {"x": 20, "y": 20, "z": 20}})
    # fig.update_layout(autosize=False, width=600, height=600)

    # Remove marigns
    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

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


def scp_traj_interp(scp_trajs, params: Config):
    scp_prop_trajs = []
    for traj in scp_trajs:
        states = []
        for k in range(params.scp.n):
            traj_temp = np.repeat(
                np.expand_dims(traj[k], axis=1), params.prp.inter_sample - 1, axis=1
            )
            for i in range(1, params.prp.inter_sample - 1):
                states.append(traj_temp[:, i])
        scp_prop_trajs.append(np.array(states))
    return scp_prop_trajs
