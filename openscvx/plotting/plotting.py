from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from openscvx.algorithms import OptimizationResults

from .publication import (
    PlotStyle,
    VarSpec,
    apply_publication_plotly_layout,
    expand_var_components,
    latex_component_label,
    publication_trace_colors,
    save_timeseries_pdf,
    wrap_publication_figure,
)


def _get_var(result: OptimizationResults, var_name: str, var_list: list):
    """Get a variable object by name from the metadata list."""
    for var in var_list:
        if var.name == var_name:
            return var
    raise ValueError(f"Variable '{var_name}' not found")


def _get_var_dim(result: OptimizationResults, var_name: str, var_list: list) -> int:
    """Get dimensionality of a variable from the metadata."""
    var = _get_var(result, var_name, var_list)
    s = var._slice
    if isinstance(s, slice):
        return (s.stop or 1) - (s.start or 0)
    return 1


def _is_impulsive_control(result: OptimizationResults, control_name: str) -> bool:
    """Return True if the control uses impulsive parameterization."""
    var = _get_var(result, control_name, result._controls)
    return getattr(var, "parameterization", None) == "impulsive"


def _has_impulsive_controls(result: OptimizationResults) -> bool:
    """Return True if any control is marked as impulsive."""
    for control in result._controls:
        if _is_impulsive_control(result, control.name):
            return True
    return False


def _add_component_traces(
    fig: go.Figure,
    result: OptimizationResults,
    var_name: str,
    component_idx: int,
    row: int,
    col: int,
    show_legend: bool,
    min_val: float | None = None,
    max_val: float | None = None,
    impulsive: bool = False,
    plot_trajectory: bool = True,
    split_nodes: bool = False,
    colors: dict[str, str] | None = None,
):
    """Add traces for a single component of a variable to a subplot.

    Args:
        fig: Plotly figure to add traces to
        result: Optimization results
        var_name: Name of the variable
        component_idx: Index of the component to plot
        row: Subplot row
        col: Subplot column
        show_legend: Whether to show legend entries
        min_val: Optional minimum bound to show as horizontal line
        max_val: Optional maximum bound to show as horizontal line
    """
    import numpy as np

    if colors is None:
        colors = {
            "trajectory": "green",
            "nodes": "cyan",
            "nodes_prior": "#D98B8B",
            "bounds": "red",
            "impulses": "orange",
        }

    t_nodes = result.nodes["time"].flatten()
    has_trajectory = bool(result.trajectory) and var_name in result.trajectory
    t_full = result.trajectory["time"].flatten() if has_trajectory else None

    # Plot propagated trajectory if available
    if has_trajectory and plot_trajectory:
        data = result.trajectory[var_name]
        y = data if data.ndim == 1 else data[:, component_idx]
        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=y,
                mode="lines",
                name="Trajectory",
                showlegend=show_legend,
                legendgroup="trajectory",
                line={"color": colors["trajectory"], "width": 2},
            ),
            row=row,
            col=col,
        )

    # Plot optimization nodes
    if var_name in result.nodes:
        data = result.nodes[var_name]
        y = data if data.ndim == 1 else data[:, component_idx]
        if split_nodes and len(t_nodes) > 0:
            fig.add_trace(
                go.Scatter(
                    x=t_nodes[:1],
                    y=y[:1],
                    mode="markers",
                    name="Prior state",
                    showlegend=show_legend,
                    legendgroup="nodes_prior",
                    marker={"color": colors["nodes_prior"], "size": 7, "symbol": "diamond"},
                ),
                row=row,
                col=col,
            )
            if len(t_nodes) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=t_nodes[1:],
                        y=y[1:],
                        mode="markers",
                        name="Posterior state",
                        showlegend=show_legend,
                        legendgroup="nodes_posterior",
                        marker={"color": colors["nodes"], "size": 6, "symbol": "circle"},
                    ),
                    row=row,
                    col=col,
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=t_nodes,
                    y=y,
                    mode="markers",
                    name="Nodes",
                    showlegend=show_legend,
                    legendgroup="nodes",
                    marker={"color": colors["nodes"], "size": 6, "symbol": "circle"},
                ),
                row=row,
                col=col,
            )

        if impulsive:
            x_imp = []
            y_imp = []
            for k in range(len(t_nodes)):
                if np.abs(y[k]) > 0:
                    x_imp.extend([t_nodes[k], t_nodes[k], None])
                    y_imp.extend([0.0, y[k], None])
            if x_imp:
                fig.add_trace(
                    go.Scatter(
                        x=x_imp,
                        y=y_imp,
                        mode="lines",
                        name="Impulses",
                        showlegend=show_legend,
                        legendgroup="impulses",
                        line={"color": colors["impulses"], "width": 2, "dash": "dash"},
                    ),
                    row=row,
                    col=col,
                )

    # Add horizontal bound lines if provided
    # Only add if finite (skip -inf/inf bounds)
    if min_val is not None and np.isfinite(min_val):
        fig.add_hline(
            y=min_val,
            line={"color": colors["bounds"], "width": 1.5, "dash": "dash"},
            row=row,
            col=col,
        )
    if max_val is not None and np.isfinite(max_val):
        fig.add_hline(
            y=max_val,
            line={"color": colors["bounds"], "width": 1.5, "dash": "dash"},
            row=row,
            col=col,
        )


# =============================================================================
# State Plotting
# =============================================================================


def plot_state_component(
    result: OptimizationResults,
    state_name: str,
    component: int = 0,
) -> go.Figure:
    """Plot a single component of a state variable vs time.

    This is the low-level function for plotting one scalar value over time.
    For plotting all components of a state, use plot_states().

    Args:
        result: Optimization results containing state trajectories
        state_name: Name of the state variable
        component: Component index (0-indexed). For scalar states, use 0.

    Returns:
        Plotly figure with single plot

    Example:
        >>> plot_state_component(result, "position", 2)  # Plot z-component
    """
    available = {s.name for s in result._states}
    if state_name not in available:
        raise ValueError(f"State '{state_name}' not found. Available: {sorted(available)}")

    dim = _get_var_dim(result, state_name, result._states)
    if component < 0 or component >= dim:
        raise ValueError(f"Component {component} out of range for '{state_name}' (dim={dim})")

    t_nodes = result.nodes["time"].flatten()
    has_trajectory = bool(result.trajectory) and state_name in result.trajectory
    t_full = result.trajectory["time"].flatten() if has_trajectory else None

    label = f"{state_name}_{component}" if dim > 1 else state_name

    fig = go.Figure()
    fig.update_layout(title_text=label, template="plotly_dark")

    if has_trajectory:
        data = result.trajectory[state_name]
        y = data if data.ndim == 1 else data[:, component]
        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=y,
                mode="lines",
                name="Trajectory",
                line={"color": "green", "width": 2},
            )
        )

    if state_name in result.nodes:
        data = result.nodes[state_name]
        y = data if data.ndim == 1 else data[:, component]
        if _has_impulsive_controls(result) and len(t_nodes) > 0:
            fig.add_trace(
                go.Scatter(
                    x=t_nodes[:1],
                    y=y[:1],
                    mode="markers",
                    name="Prior state",
                    marker={"color": "#D98B8B", "size": 7, "symbol": "diamond"},
                )
            )
            if len(t_nodes) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=t_nodes[1:],
                        y=y[1:],
                        mode="markers",
                        name="Posterior state",
                        marker={"color": "cyan", "size": 6, "symbol": "circle"},
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=t_nodes,
                    y=y,
                    mode="markers",
                    name="Nodes",
                    marker={"color": "cyan", "size": 6},
                )
            )

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text=label)
    return fig


def plot_states(
    result: OptimizationResults,
    state_names: list[VarSpec] | None = None,
    include_private: bool = False,
    cols: int = 4,
    style: PlotStyle = "dark",
    pdf_path: str | Path | None = None,
) -> go.Figure:
    """Plot state variables in a subplot grid.

    Each component of each state gets its own subplot with individual y-axis
    scaling. This is the primary function for visualizing state trajectories.

    Args:
        result: Optimization results containing state trajectories
        state_names: List of state names to plot. If None, plots all states.
            For multidimensional states, pass a single component with
            ``("position", 0)``, ``"position:0"``, or ``"position[0]"``.
        include_private: Whether to include private states (names starting with '_')
        cols: Maximum number of columns in subplot grid
        style: ``"dark"`` for the default Plotly theme, ``"publication"`` for a
            white theme with Latin Modern fonts, LaTeX labels, and automatic PDF
            export, or ``"publication_dark"`` for the same figure on the dark
            palette (the exported PDF stays white for print).
        pdf_path: Output path for the PDF when ``style="publication"``. Defaults
            to ``figures/state_trajectories.pdf``.

    Returns:
        Plotly figure, or a :class:`~openscvx.plotting.publication.PublicationFigure`
        wrapper when ``style="publication"``.

    Examples:
        >>> plot_states(result, ["position"])  # 3 subplots for x, y, z
        >>> plot_states(result, ["position:2"])  # z component only
        >>> plot_states(result, [("position", 0), "velocity"])  # x and all velocity
        >>> plot_states(result, style="publication", pdf_path="figures/states.pdf")
    """

    components = expand_var_components(
        result,
        state_names,
        result._states,
        include_private=include_private,
    )

    publication = style in ("publication", "publication_dark")
    colors = publication_trace_colors() if publication else None

    n_cols = min(cols, len(components))
    n_rows = (len(components) + n_cols - 1) // n_cols

    if publication:
        subplot_titles = [""] * len(components)
    else:
        subplot_titles = [c[0] for c in components]

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
    if publication:
        apply_publication_plotly_layout(
            fig, n_rows=n_rows, n_cols=n_cols, dark=style == "publication_dark"
        )
    else:
        fig.update_layout(title_text="State Trajectories", template="plotly_dark")

    has_impulsive_controls = _has_impulsive_controls(result)

    for idx, (_, var_name, comp_idx) in enumerate(components):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        # Get bounds for this component
        var = _get_var(result, var_name, result._states)
        min_val = var.min[comp_idx] if var.min is not None else None
        max_val = var.max[comp_idx] if var.max is not None else None
        _add_component_traces(
            fig,
            result,
            var_name,
            comp_idx,
            row,
            col,
            show_legend=(idx == 0),
            min_val=min_val,
            max_val=max_val,
            split_nodes=has_impulsive_controls,
            colors=colors,
        )

        if publication:
            fig.update_yaxes(
                title_text=latex_component_label(
                    var_name,
                    comp_idx,
                    dim=_get_var_dim(result, var_name, result._states),
                ),
                row=row,
                col=col,
            )

    # Add x-axis labels to bottom row
    x_label = r"$t\,\mathrm{(s)}$" if publication else "Time (s)"
    for col_idx in range(1, n_cols + 1):
        fig.update_xaxes(title_text=x_label, row=n_rows, col=col_idx)

    if publication:

        def _save(path: str | Path) -> None:
            save_timeseries_pdf(
                result,
                components,
                var_list=result._states,
                path=path,
                suptitle="",
                cols=cols,
                split_nodes=has_impulsive_controls,
            )

        return wrap_publication_figure(
            fig,
            pdf_path=pdf_path,
            default_pdf_name="state_trajectories.pdf",
            save_fn=_save,
        )

    return fig


# =============================================================================
# Control Plotting
# =============================================================================


def plot_control_component(
    result: OptimizationResults,
    control_name: str,
    component: int = 0,
) -> go.Figure:
    """Plot a single component of a control variable vs time.

    This is the low-level function for plotting one scalar control over time.
    For plotting all components of a control, use plot_controls().

    Args:
        result: Optimization results containing control trajectories
        control_name: Name of the control variable
        component: Component index (0-indexed). For scalar controls, use 0.

    Returns:
        Plotly figure with single plot

    Example:
        >>> plot_control_component(result, "thrust", 0)  # Plot thrust_x
    """
    available = {c.name for c in result._controls}
    if control_name not in available:
        raise ValueError(f"Control '{control_name}' not found. Available: {sorted(available)}")

    dim = _get_var_dim(result, control_name, result._controls)
    if component < 0 or component >= dim:
        raise ValueError(f"Component {component} out of range for '{control_name}' (dim={dim})")

    t_nodes = result.nodes["time"].flatten()
    has_trajectory = bool(result.trajectory) and control_name in result.trajectory
    t_full = result.trajectory["time"].flatten() if has_trajectory else None

    label = f"{control_name}_{component}" if dim > 1 else control_name

    fig = go.Figure()
    fig.update_layout(title_text=label, template="plotly_dark")

    if has_trajectory:
        data = result.trajectory[control_name]
        y = data if data.ndim == 1 else data[:, component]
        if not _is_impulsive_control(result, control_name):
            fig.add_trace(
                go.Scatter(
                    x=t_full,
                    y=y,
                    mode="lines",
                    name="Trajectory",
                    line={"color": "green", "width": 2},
                )
            )

    if control_name in result.nodes:
        data = result.nodes[control_name]
        y = data if data.ndim == 1 else data[:, component]
        fig.add_trace(
            go.Scatter(
                x=t_nodes,
                y=y,
                mode="markers",
                name="Nodes",
                marker={"color": "cyan", "size": 6},
            )
        )
        if _is_impulsive_control(result, control_name):
            x_imp = []
            y_imp = []
            for k in range(len(t_nodes)):
                if abs(y[k]) > 0:
                    x_imp.extend([t_nodes[k], t_nodes[k], None])
                    y_imp.extend([0.0, y[k], None])
            if x_imp:
                fig.add_trace(
                    go.Scatter(
                        x=x_imp,
                        y=y_imp,
                        mode="lines",
                        name="Impulses",
                        line={"color": "orange", "width": 2, "dash": "dash"},
                    )
                )

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text=label)
    return fig


def plot_controls(
    result: OptimizationResults,
    control_names: list[VarSpec] | None = None,
    include_private: bool = False,
    cols: int = 3,
    style: PlotStyle = "dark",
    pdf_path: str | Path | None = None,
) -> go.Figure:
    """Plot control variables in a subplot grid.

    Each component of each control gets its own subplot with individual y-axis
    scaling. This is the primary function for visualizing control trajectories.

    Args:
        result: Optimization results containing control trajectories
        control_names: List of control names to plot. If None, plots all controls.
            For multidimensional controls, pass a single component with
            ``("thrust_force", 2)``, ``"thrust_force:2"``, or ``"thrust_force[2]"``.
        include_private: Whether to include private controls (names starting with '_')
        cols: Maximum number of columns in subplot grid
        style: ``"dark"`` for the default Plotly theme, ``"publication"`` for a
            white theme with Latin Modern fonts, LaTeX labels, and automatic PDF
            export, or ``"publication_dark"`` for the same figure on the dark
            palette (the exported PDF stays white for print).
        pdf_path: Output path for the PDF when ``style="publication"``. Defaults
            to ``figures/control_trajectories.pdf``.

    Returns:
        Plotly figure, or a :class:`~openscvx.plotting.publication.PublicationFigure`
        wrapper when ``style="publication"``.

    Examples:
        >>> plot_controls(result, ["thrust_force"])  # 3 subplots for x, y, z
        >>> plot_controls(result, ["thrust_force:2"])  # fz only
        >>> plot_controls(result, style="publication")
    """

    components = expand_var_components(
        result,
        control_names,
        result._controls,
        include_private=include_private,
    )

    publication = style in ("publication", "publication_dark")
    colors = publication_trace_colors() if publication else None

    n_cols = min(cols, len(components))
    n_rows = (len(components) + n_cols - 1) // n_cols

    if publication:
        subplot_titles = [""] * len(components)
    else:
        subplot_titles = [c[0] for c in components]

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
    if publication:
        apply_publication_plotly_layout(
            fig, n_rows=n_rows, n_cols=n_cols, dark=style == "publication_dark"
        )
    else:
        fig.update_layout(title_text="Control Trajectories", template="plotly_dark")

    for idx, (_, var_name, comp_idx) in enumerate(components):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        # Get bounds for this component
        var = _get_var(result, var_name, result._controls)
        min_val = var.min[comp_idx] if var.min is not None else None
        max_val = var.max[comp_idx] if var.max is not None else None
        is_impulsive = _is_impulsive_control(result, var_name)

        _add_component_traces(
            fig,
            result,
            var_name,
            comp_idx,
            row,
            col,
            show_legend=(idx == 0),
            min_val=min_val,
            max_val=max_val,
            impulsive=is_impulsive,
            plot_trajectory=not is_impulsive,
            colors=colors,
        )

        if publication:
            fig.update_yaxes(
                title_text=latex_component_label(
                    var_name,
                    comp_idx,
                    dim=_get_var_dim(result, var_name, result._controls),
                ),
                row=row,
                col=col,
            )

    # Add x-axis labels to bottom row
    x_label = r"$t\,\mathrm{(s)}$" if publication else "Time (s)"
    for col_idx in range(1, n_cols + 1):
        fig.update_xaxes(title_text=x_label, row=n_rows, col=col_idx)

    if publication:

        def _save(path: str | Path) -> None:
            save_timeseries_pdf(
                result,
                components,
                var_list=result._controls,
                path=path,
                suptitle="",
                cols=cols,
                impulsive_fn=_is_impulsive_control,
                plot_trajectory_fn=lambda name: not _is_impulsive_control(result, name),
            )

        return wrap_publication_figure(
            fig,
            pdf_path=pdf_path,
            default_pdf_name="control_trajectories.pdf",
            save_fn=_save,
        )

    return fig


def plot_trust_region_heatmap(result: OptimizationResults):
    """Plot heatmap of the final trust-region deltas (TR_history[-1])."""
    if not result.TR_history:
        raise ValueError("Result has no TR_history to plot")

    tr_mat = result.TR_history[-1]

    # Build variable names list
    var_names = []
    for var_list in [result._states, result._controls]:
        for var in var_list:
            dim = _get_var_dim(result, var.name, var_list)
            if dim == 1:
                var_names.append(var.name)
            else:
                var_names.extend(f"{var.name}_{i}" for i in range(dim))

    # TR matrix is (n_states+n_controls, n_nodes): rows = variables, cols = nodes
    if tr_mat.shape[0] == len(var_names):
        z = tr_mat
    elif tr_mat.shape[1] == len(var_names):
        z = tr_mat.T
    else:
        raise ValueError("TR matrix dimensions do not align with state/control components")

    x_len = z.shape[1]
    t_nodes = result.nodes["time"].flatten()
    x_labels = t_nodes if len(t_nodes) == x_len else list(range(x_len))

    fig = go.Figure(data=go.Heatmap(z=z, x=x_labels, y=var_names, colorscale="Viridis"))
    fig.update_layout(
        title="Trust Region Delta Magnitudes (last iteration)", template="plotly_dark"
    )
    fig.update_xaxes(title_text="Node / Time", side="bottom")
    fig.update_yaxes(title_text="State / Control component", side="left")
    return fig


def plot_projections_2d(
    result: OptimizationResults,
    var_name: str = "position",
    velocity_var_name: str | None = None,
    cmap: str = "viridis",
) -> go.Figure:
    """Plot XY, XZ, YZ projections of a 3D variable.

    Useful for visualizing 3D trajectories in 2D plane views.

    Args:
        result: Optimization results containing trajectories
        var_name: Name of the 3D variable to plot (default: "position")
        velocity_var_name: Optional name of velocity variable for coloring by speed.
            If provided, trajectory points are colored by velocity magnitude.
        cmap: Matplotlib colormap name for velocity coloring (default: "viridis")

    Returns:
        Plotly figure with three subplots (XY, XZ, YZ planes)
    """
    import numpy as np

    has_trajectory = bool(result.trajectory) and var_name in result.trajectory
    has_nodes = var_name in result.nodes

    if not has_trajectory and not has_nodes:
        available_traj = set(result.trajectory.keys()) if result.trajectory else set()
        available_nodes = set(result.nodes.keys())
        raise ValueError(
            f"Variable '{var_name}' not found. "
            f"Available in trajectory: {sorted(available_traj)}, nodes: {sorted(available_nodes)}"
        )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("XY Plane", "XZ Plane", "YZ Plane"),
        specs=[[{}, {}], [{}, None]],
    )

    # Subplot positions: (x_idx, y_idx, row, col)
    subplots = [(0, 1, 1, 1), (0, 2, 1, 2), (1, 2, 2, 1)]

    # Compute velocity norms if velocity variable is provided
    traj_vel_norm = None
    node_vel_norm = None
    if velocity_var_name is not None:
        if has_trajectory and velocity_var_name in result.trajectory:
            traj_vel_norm = np.linalg.norm(result.trajectory[velocity_var_name], axis=1)
        if has_nodes and velocity_var_name in result.nodes:
            node_vel_norm = np.linalg.norm(result.nodes[velocity_var_name], axis=1)

    # Colorbar config (only shown once)
    colorbar_cfg = {"title": "‖velocity‖", "x": 1.02, "y": 0.5, "len": 0.9}

    # Plot trajectory if available
    if has_trajectory:
        data = result.trajectory[var_name]
        for i, (xi, yi, row, col) in enumerate(subplots):
            if traj_vel_norm is not None:
                marker = {
                    "size": 4,
                    "color": traj_vel_norm,
                    "colorscale": cmap,
                    "showscale": (i == 0),
                    "colorbar": colorbar_cfg if i == 0 else None,
                }
                fig.add_trace(
                    go.Scatter(
                        x=data[:, xi],
                        y=data[:, yi],
                        mode="markers",
                        marker=marker,
                        name="Trajectory",
                        legendgroup="trajectory",
                        showlegend=(i == 0),
                    ),
                    row=row,
                    col=col,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=data[:, xi],
                        y=data[:, yi],
                        mode="lines",
                        line={"color": "green", "width": 2},
                        name="Trajectory",
                        legendgroup="trajectory",
                        showlegend=(i == 0),
                    ),
                    row=row,
                    col=col,
                )

    # Plot nodes if available
    if has_nodes:
        data = result.nodes[var_name]
        # Only show colorbar on nodes if trajectory doesn't have one
        show_node_colorbar = (traj_vel_norm is None) and (node_vel_norm is not None)
        for i, (xi, yi, row, col) in enumerate(subplots):
            if node_vel_norm is not None:
                marker = {
                    "size": 8,
                    "color": node_vel_norm,
                    "colorscale": cmap,
                    "showscale": show_node_colorbar and (i == 0),
                    "colorbar": colorbar_cfg if (show_node_colorbar and i == 0) else None,
                    "line": {"color": "white", "width": 1},
                }
            else:
                marker = {"color": "cyan", "size": 6}
            fig.add_trace(
                go.Scatter(
                    x=data[:, xi],
                    y=data[:, yi],
                    mode="markers",
                    marker=marker,
                    name="Nodes",
                    legendgroup="nodes",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

    # Set axis titles
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_yaxes(title_text="Z", row=1, col=2)
    fig.update_xaxes(title_text="Y", row=2, col=1)
    fig.update_yaxes(title_text="Z", row=2, col=1)

    # Set equal aspect ratio for each subplot
    layout_opts = {
        "title": f"{var_name} - XY, XZ, YZ Projections",
        "template": "plotly_dark",
        "xaxis": {"scaleanchor": "y"},
        "xaxis2": {"scaleanchor": "y2"},
        "xaxis3": {"scaleanchor": "y3"},
    }
    # Move legend to bottom-right when using colorbar to avoid overlap
    if velocity_var_name is not None:
        layout_opts["legend"] = {
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
        }
    fig.update_layout(**layout_opts)

    return fig


def plot_vector_norm(
    result: OptimizationResults,
    var_name: str,
    bounds: tuple[float, float] | None = None,
) -> go.Figure:
    """Plot the 2-norm of a vector variable over time.

    Useful for visualizing thrust magnitude, velocity magnitude, etc.

    Args:
        result: Optimization results containing trajectories
        var_name: Name of the vector variable (state or control)
        bounds: Optional (min, max) bounds to show as horizontal dashed lines

    Returns:
        Plotly figure
    """
    import numpy as np

    has_trajectory = bool(result.trajectory) and var_name in result.trajectory
    has_nodes = var_name in result.nodes

    if not has_trajectory and not has_nodes:
        available_traj = set(result.trajectory.keys()) if result.trajectory else set()
        available_nodes = set(result.nodes.keys())
        raise ValueError(
            f"Variable '{var_name}' not found. "
            f"Available in trajectory: {sorted(available_traj)}, nodes: {sorted(available_nodes)}"
        )

    fig = go.Figure()

    # Plot trajectory norm if available
    if has_trajectory:
        t_full = result.trajectory["time"].flatten()
        data = result.trajectory[var_name]
        norm = np.linalg.norm(data, axis=1)
        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=norm,
                mode="lines",
                line={"color": "green", "width": 2},
                name="Trajectory",
                legendgroup="trajectory",
            )
        )

    # Plot node norms if available
    if has_nodes:
        t_nodes = result.nodes["time"].flatten()
        data = result.nodes[var_name]
        norm = np.linalg.norm(data, axis=1)
        fig.add_trace(
            go.Scatter(
                x=t_nodes,
                y=norm,
                mode="markers",
                marker={"color": "cyan", "size": 6},
                name="Nodes",
                legendgroup="nodes",
            )
        )

    # Add bounds if provided
    if bounds is not None:
        min_bound, max_bound = bounds
        fig.add_hline(
            y=min_bound,
            line={"color": "red", "width": 2, "dash": "dash"},
            annotation_text="Min",
            annotation_position="right",
        )
        fig.add_hline(
            y=max_bound,
            line={"color": "red", "width": 2, "dash": "dash"},
            annotation_text="Max",
            annotation_position="right",
        )

    fig.update_layout(
        title=f"‖{var_name}‖₂",
        xaxis_title="Time (s)",
        yaxis_title="Norm",
        template="plotly_dark",
    )

    return fig


def plot_virtual_control_heatmap(result: OptimizationResults):
    """Plot heatmap of the final virtual control magnitudes (VC_history[-1])."""
    if not result.VC_history:
        raise ValueError("Result has no VC_history to plot")

    vc_mat = result.VC_history[-1]

    # Build state names list
    state_names = []
    for var in result._states:
        dim = _get_var_dim(result, var.name, result._states)
        if dim == 1:
            state_names.append(var.name)
        else:
            state_names.extend(f"{var.name}_{i}" for i in range(dim))

    # Align so rows = states, cols = nodes
    if vc_mat.shape[1] == len(state_names):
        z = vc_mat.T
    elif vc_mat.shape[0] == len(state_names):
        z = vc_mat
    else:
        raise ValueError("VC matrix shape does not align with state components")

    x_len = z.shape[1]
    t_nodes = result.nodes["time"].flatten()

    # Virtual control uses N-1 intervals
    if len(t_nodes) == x_len + 1:
        x_labels = t_nodes[:-1]
    elif len(t_nodes) == x_len:
        x_labels = t_nodes
    else:
        x_labels = list(range(x_len))

    fig = go.Figure(data=go.Heatmap(z=z, x=x_labels, y=state_names, colorscale="Magma"))
    fig.update_layout(title="Virtual Control Magnitudes (last iteration)", template="plotly_dark")
    fig.update_xaxes(title_text="Node Interval (N-1)")
    fig.update_yaxes(title_text="State component")
    return fig
