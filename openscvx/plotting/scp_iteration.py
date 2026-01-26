import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from openscvx.algorithms import OptimizationResults

from .plotting import _get_var


def plot_scp_iterations(
    result: OptimizationResults,
    state_names: list[str] | None = None,
    control_names: list[str] | None = None,
    cmap_name: str = "viridis",
    show_propagation: bool = True,
) -> go.Figure:
    """Plot all SCP iterations overlaid with colormap-based coloring.

    Shows the evolution of states and controls across SCP iterations. Early
    iterations are dark, later iterations are bright (following the colormap).
    This makes convergence visible at a glance.

    Args:
        result: Optimization results containing iteration history
        state_names: Optional list of state names to include. If None, plots all states.
        control_names: Optional list of control names to include. If None, plots all controls.
        cmap_name: Matplotlib colormap name (default: "viridis")
        show_propagation: If True, show multi-shot propagation lines (default: True)

    Returns:
        Plotly figure with all iterations overlaid

    Example:
        >>> results = problem.solve()
        >>> plot_scp_iterations(results, ["position", "velocity"]).show()
    """
    import matplotlib.pyplot as plt

    if not result.X:
        raise ValueError("No iteration history available in result.X")

    # Derive dimensions from result data
    n_x = result.X[0].shape[1]
    n_u = result.U[0].shape[1]

    # Find time slice by looking for "time" state
    time_slice = None
    for state in result._states:
        if state.name.lower() == "time":
            time_slice = state._slice
            break

    # Extract multi-shot propagation trajectories
    V_history = result.discretization_history if result.discretization_history else []
    X_prop_history = []
    if V_history and show_propagation:
        i4 = n_x + n_x * n_x + 2 * n_x * n_u
        for V in V_history:
            pos_traj = []
            for i_multi in range(V.shape[1]):
                pos_traj.append(V[:, i_multi].reshape(-1, i4)[:, :n_x])
            X_prop_history.append(np.array(pos_traj))

    n_iterations = len(result.X)
    if X_prop_history:
        n_iterations = min(n_iterations, len(X_prop_history))

    # Filter states and controls (exclude ctcs_aug and time)
    states = [
        s for s in result._states if "ctcs_aug" not in s.name.lower() and s.name.lower() != "time"
    ]
    controls = list(result._controls) if result._controls else []

    state_filter = set(state_names) if state_names else None
    control_filter = set(control_names) if control_names else None

    if state_filter and control_filter is None:
        controls = []
    if control_filter and state_filter is None:
        states = []
    if state_filter:
        states = [s for s in states if s.name in state_filter]
        if not states:
            available = {s.name for s in result._states if "ctcs_aug" not in s.name.lower()}
            raise ValueError(
                f"No states matched filter {state_names}. Available: {sorted(available)}"
            )
    if control_filter:
        controls = [c for c in controls if c.name in control_filter]
        if not controls:
            available = {c.name for c in result._controls}
            raise ValueError(
                f"No controls matched filter {control_names}. Available: {sorted(available)}"
            )

    if not states and not controls:
        raise ValueError("No states or controls to plot")

    # Expand multi-dimensional variables to individual components
    def expand_variables(variables):
        expanded = []
        for var in variables:
            s = var._slice
            start = s.start if isinstance(s, slice) else s
            stop = s.stop if isinstance(s, slice) else start + 1
            n_comp = (stop or start + 1) - (start or 0)

            for i in range(n_comp):
                expanded.append(
                    {
                        "name": f"{var.name}_{i}" if n_comp > 1 else var.name,
                        "idx": start + i,
                        "parent": var.name,
                        "comp": i,
                    }
                )
        return expanded

    expanded_states = expand_variables(states)
    expanded_controls = expand_variables(controls)

    # Grid layout
    n_states = len(expanded_states)
    n_controls = len(expanded_controls)
    n_state_cols = min(7, n_states) if n_states > 0 else 1
    n_control_cols = min(3, n_controls) if n_controls > 0 else 1
    n_state_rows = (n_states + n_state_cols - 1) // n_state_cols if n_states > 0 else 0
    n_control_rows = (n_controls + n_control_cols - 1) // n_control_cols if n_controls > 0 else 0
    total_rows = n_state_rows + n_control_rows
    max_cols = max(n_state_cols, n_control_cols)

    subplot_titles = [s["name"] for s in expanded_states] + [c["name"] for c in expanded_controls]
    fig = make_subplots(
        rows=total_rows,
        cols=max_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    # Get colormap
    cmap = plt.get_cmap(cmap_name)

    def iter_color(iter_idx):
        rgba = cmap(iter_idx / max(n_iterations - 1, 1))
        return f"rgb({int(rgba[0] * 255)},{int(rgba[1] * 255)},{int(rgba[2] * 255)})"

    # Plot all iterations
    for iter_idx in range(n_iterations):
        X_nodes = result.X[iter_idx]
        U_iter = result.U[iter_idx]
        color = iter_color(iter_idx)
        legend_group = f"iter_{iter_idx}"
        show_legend_for_iter = True  # Show legend for first trace of this iteration

        t_nodes = (
            X_nodes[:, time_slice].flatten()
            if time_slice is not None
            else np.linspace(0, result.t_final, X_nodes.shape[0])
        )

        # States
        for state_idx, state in enumerate(expanded_states):
            row = (state_idx // n_state_cols) + 1
            col = (state_idx % n_state_cols) + 1
            idx = state["idx"]

            # Multi-shot propagation lines
            if X_prop_history and iter_idx < len(X_prop_history):
                pos_traj = X_prop_history[iter_idx]
                for j in range(pos_traj.shape[1]):
                    segment_times = pos_traj[:, j, time_slice].flatten()
                    segment_states = pos_traj[:, j, idx]
                    fig.add_trace(
                        go.Scatter(
                            x=segment_times,
                            y=segment_states,
                            mode="lines",
                            line={"color": color, "width": 1.5},
                            legendgroup=legend_group,
                            showlegend=show_legend_for_iter,
                            name=f"Iter {iter_idx}" if show_legend_for_iter else None,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )
                    show_legend_for_iter = False

            # Nodes
            fig.add_trace(
                go.Scatter(
                    x=t_nodes,
                    y=X_nodes[:, idx],
                    mode="markers",
                    marker={"color": color, "size": 5},
                    legendgroup=legend_group,
                    showlegend=show_legend_for_iter,
                    name=f"Iter {iter_idx}" if show_legend_for_iter else None,
                    hovertemplate=f"iter {iter_idx}<br>t=%{{x:.2f}}<br>y=%{{y:.3g}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            show_legend_for_iter = False

        # Controls
        for control_idx, control in enumerate(expanded_controls):
            row = n_state_rows + (control_idx // n_control_cols) + 1
            col = (control_idx % n_control_cols) + 1
            idx = control["idx"]

            fig.add_trace(
                go.Scatter(
                    x=t_nodes,
                    y=U_iter[:, idx],
                    mode="markers",
                    marker={"color": color, "size": 5},
                    legendgroup=legend_group,
                    showlegend=show_legend_for_iter,
                    name=f"Iter {iter_idx}" if show_legend_for_iter else None,
                    hovertemplate=f"iter {iter_idx}<br>t=%{{x:.2f}}<br>y=%{{y:.3g}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            show_legend_for_iter = False

    # Add bounds (once, using final iteration's time range)
    t_nodes_final = (
        result.X[-1][:, time_slice].flatten()
        if time_slice is not None
        else np.linspace(0, result.t_final, result.X[-1].shape[0])
    )
    t_min, t_max = t_nodes_final.min(), t_nodes_final.max()

    for state_idx, state in enumerate(expanded_states):
        row = (state_idx // n_state_cols) + 1
        col = (state_idx % n_state_cols) + 1
        parent = _get_var(result, state["parent"], result._states)
        comp_idx = state["comp"]

        for bound_val, bound_attr in [(parent.min, "min"), (parent.max, "max")]:
            if bound_val is not None and np.isfinite(bound_val[comp_idx]):
                fig.add_trace(
                    go.Scatter(
                        x=[t_min, t_max],
                        y=[bound_val[comp_idx], bound_val[comp_idx]],
                        mode="lines",
                        line={"color": "red", "width": 1.5, "dash": "dot"},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

    for control_idx, control in enumerate(expanded_controls):
        row = n_state_rows + (control_idx // n_control_cols) + 1
        col = (control_idx % n_control_cols) + 1
        parent = _get_var(result, control["parent"], result._controls)
        comp_idx = control["comp"]

        for bound_val in [parent.min, parent.max]:
            if bound_val is not None and np.isfinite(bound_val[comp_idx]):
                fig.add_trace(
                    go.Scatter(
                        x=[t_min, t_max],
                        y=[bound_val[comp_idx], bound_val[comp_idx]],
                        mode="lines",
                        line={"color": "red", "width": 1.5, "dash": "dot"},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

    # Layout
    fig.update_layout(
        title_text="SCP Iterations",
        template="plotly_dark",
        showlegend=True,
        legend={
            "title": "Iterations",
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 1.02,
            "bgcolor": "rgba(0, 0, 0, 0.5)",
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
    )

    for col_idx in range(1, max_cols + 1):
        fig.update_xaxes(title_text="Time (s)", row=total_rows, col=col_idx)

    return fig
