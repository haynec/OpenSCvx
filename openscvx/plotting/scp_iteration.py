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


def plot_scp_convergence_histories(result: OptimizationResults) -> go.Figure:
    """Plot SCP convergence histories: trust region weight, reduction histories,
    and acceptance ratio.

    Creates three separate plots:
    1. Trust region weight (lam_prox) history
    2. Actual and predicted reduction histories (overlaid)
    3. Acceptance ratio history

    Args:
        result: OptimizationResults containing the convergence histories.

    Returns:
        Plotly figure with three subplots

    Example:
        >>> problem.initialize()
        >>> result = problem.solve()
        >>> plot_scp_convergence_histories(result).show()
    """
    if not isinstance(result, OptimizationResults):
        raise TypeError(f"Expected OptimizationResults, got {type(result)}")

    # Create subplots: 3 rows, 1 column
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Trust Region Weight History",
            "Actual vs Predicted Reduction History",
            "Acceptance Ratio History",
        ),
        vertical_spacing=0.12,
    )

    # Prepare iteration indices (0-indexed for plotting)
    iterations_lam_prox = np.arange(len(result.lam_prox_history))
    iterations_reduction = np.arange(len(result.actual_reduction_history))
    iterations_ratio = np.arange(len(result.acceptance_ratio_history))

    # Plot 1: Trust region weight history
    if len(result.lam_prox_history) > 0:
        fig.add_trace(
            go.Scatter(
                x=iterations_lam_prox,
                y=result.lam_prox_history,
                mode="lines+markers",
                name="lam_prox",
                line={"color": "cyan", "width": 2},
                marker={"size": 6},
                hovertemplate="Iteration: %{x}<br>lam_prox: %{y:.3g}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_annotation(
            text="No trust region weight history available",
            xref="x1",
            yref="y1",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=1,
            col=1,
        )

    # Plot 2: Actual and predicted reduction histories
    if len(result.actual_reduction_history) > 0 and len(result.pred_reduction_history) > 0:
        # Ensure both histories have the same length
        min_len = min(len(result.actual_reduction_history), len(result.pred_reduction_history))
        actual_reduction = result.actual_reduction_history[:min_len]
        predicted_reduction = result.pred_reduction_history[:min_len]
        iterations_reduction = np.arange(min_len)

        fig.add_trace(
            go.Scatter(
                x=iterations_reduction,
                y=actual_reduction,
                mode="lines+markers",
                name="Actual Reduction",
                line={"color": "green", "width": 2},
                marker={"size": 6},
                hovertemplate="Iteration: %{x}<br>Actual Reduction: %{y:.3g}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=iterations_reduction,
                y=predicted_reduction,
                mode="lines+markers",
                name="Predicted Reduction",
                line={"color": "orange", "width": 2},
                marker={"size": 6},
                hovertemplate="Iteration: %{x}<br>Predicted Reduction: %{y:.3g}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    else:
        fig.add_annotation(
            text="No reduction history available",
            xref="x2",
            yref="y2",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=2,
            col=1,
        )

    # Plot 3: Acceptance ratio history
    if len(result.acceptance_ratio_history) > 0:
        fig.add_trace(
            go.Scatter(
                x=iterations_ratio,
                y=result.acceptance_ratio_history,
                mode="lines+markers",
                name="Acceptance Ratio",
                line={"color": "magenta", "width": 2},
                marker={"size": 6},
                hovertemplate="Iteration: %{x}<br>Acceptance Ratio: %{y:.3g}<extra></extra>",
            ),
            row=3,
            col=1,
        )

        # Add reference lines at typical thresholds (eta_1=1e-6, eta_2=0.9)
        fig.add_hline(
            y=1e-6,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            annotation_text="η₁ = 1e-6",
            row=3,
            col=1,
        )
        fig.add_hline(
            y=0.9,
            line_dash="dash",
            line_color="yellow",
            opacity=0.5,
            annotation_text="η₂ = 0.9",
            row=3,
            col=1,
        )
    else:
        fig.add_annotation(
            text="No acceptance ratio history available",
            xref="x3",
            yref="y3",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=3,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title_text="SCP Convergence Histories",
        template="plotly_dark",
        showlegend=True,
        height=900,
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 1.02,
            "bgcolor": "rgba(0, 0, 0, 0.5)",
        },
    )

    # Update axes labels
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="lam_prox", type="log", row=1, col=1)

    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="Reduction", row=2, col=1)  # Linear scale

    fig.update_xaxes(title_text="Iteration", row=3, col=1)
    fig.update_yaxes(title_text="Acceptance Ratio (ρ)", row=3, col=1, range=[-0.5, 1.5])

    return fig
