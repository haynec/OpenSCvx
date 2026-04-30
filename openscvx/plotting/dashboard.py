"""Constraint-activity & convergence dashboard plots.

Built on top of the per-iteration instrumentation that PTR/PenalizedTrustRegion
records onto :class:`~openscvx.algorithms.optimization_results.OptimizationResults`
(see ``J_tr_history``, ``J_vb_history``, ``J_vc_history``, ``nu_history``,
``nu_vb_history``, ``nu_vb_cross_history``, ``constraint_names``).

All functions return :class:`plotly.graph_objects.Figure` so they can be
displayed in any Plotly-compatible viewer (notebooks, browser via ``.show()``,
embedded into Viser via ``openscvx.plotting.viser.dashboard``).

Example:
    Run a solve and pop up the dashboard panels::

        results = problem.solve()
        results = problem.post_process()

        import openscvx.plotting.dashboard as dash

        dash.plot_convergence(results).show()
        dash.plot_constraint_activity(results).show()
        dash.plot_active_set(results).show()
        dash.plot_ctcs_timeseries(problem, results).show()
"""

from typing import TYPE_CHECKING, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from openscvx.algorithms.optimization_results import OptimizationResults
    from openscvx.problem import Problem


# =============================================================================
# Convergence (4-panel)
# =============================================================================


def plot_convergence(results: "OptimizationResults") -> go.Figure:
    """4-panel convergence plot: ``J_tr``, ``J_vb``, ``J_vc``, and total cost.

    Each panel shows the value at every accepted SCP iteration, on a log y-axis
    when all entries are strictly positive (which they typically are for these
    penalty terms).

    Args:
        results: Results from :meth:`~openscvx.problem.Problem.solve`.

    Returns:
        Plotly figure with four stacked subplots.
    """
    j_tr = np.asarray(results.J_tr_history, dtype=float)
    j_vb = np.asarray(results.J_vb_history, dtype=float)
    j_vc = np.asarray(results.J_vc_history, dtype=float)

    # Cost is not stored as a scalar history; compute it from the final-state
    # objective contributions per iteration as a best-effort proxy.
    cost = _compute_cost_history(results)

    panels = [
        ("J_tr (trust region)", j_tr),
        ("J_vb (virtual buffer)", j_vb),
        ("J_vc (virtual control)", j_vc),
        ("Cost", cost),
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[name for name, _ in panels],
    )

    for i, (name, vals) in enumerate(panels):
        row = (i // 2) + 1
        col = (i % 2) + 1
        if vals is None or len(vals) == 0:
            continue
        x = np.arange(1, len(vals) + 1)
        is_log = bool(np.all(np.asarray(vals) > 0))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=vals,
                mode="lines+markers",
                name=name,
                line={"width": 2},
                marker={"size": 6},
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        if is_log:
            fig.update_yaxes(type="log", row=row, col=col)
        fig.update_xaxes(title_text="Iteration", row=row, col=col)

    fig.update_layout(
        title="SCP convergence",
        template="plotly_dark",
        height=600,
    )
    return fig


def _compute_cost_history(results: "OptimizationResults") -> np.ndarray:
    """Best-effort per-iteration cost from minimize/maximize boundary terms.

    OpenSCvx does not currently store the total objective value at each
    accepted iteration as a scalar list; it does store the full state
    trajectory, so we can reconstruct the contribution of any
    ``Minimize``/``Maximize`` boundary conditions. For more complex
    objectives the caller should track cost separately.
    """
    if not results.X:
        return np.array([])

    final_types = []
    initial_types = []
    for s in results._states:
        ft = getattr(s, "final_type", None)
        it = getattr(s, "initial_type", None)
        if ft is not None:
            for t in np.asarray(ft).reshape(-1):
                final_types.append(str(t))
        if it is not None:
            for t in np.asarray(it).reshape(-1):
                initial_types.append(str(t))

    if not final_types and not initial_types:
        return np.array([])

    cost = []
    for X in results.X:
        c = 0.0
        n_x = X.shape[1]
        for i in range(min(n_x, len(final_types))):
            t = final_types[i]
            if t == "Minimize":
                c += float(X[-1, i])
            elif t == "Maximize":
                c -= float(X[-1, i])
        for i in range(min(n_x, len(initial_types))):
            t = initial_types[i]
            if t == "Minimize":
                c += float(X[0, i])
            elif t == "Maximize":
                c -= float(X[0, i])
        cost.append(c)
    return np.array(cost, dtype=float)


# =============================================================================
# Constraint activity heatmap
# =============================================================================


def plot_constraint_activity(
    results: "OptimizationResults",
    iteration: int = -1,
    *,
    include_dynamics: bool = True,
    log_scale: bool = True,
) -> go.Figure:
    """Heatmap of per-node constraint slacks at a given SCP iteration.

    Rows correspond to constraints (one per nonconvex nodal constraint, one
    per cross-node constraint, plus per-state dynamics defects when
    ``include_dynamics`` is True). Columns are trajectory nodes.

    Args:
        results: Results from :meth:`~openscvx.problem.Problem.solve`.
        iteration: SCP iteration to display. Defaults to ``-1`` (the final
            accepted iterate).
        include_dynamics: Whether to include dynamics defects (``nu``) at the
            top of the heatmap.
        log_scale: Apply ``log10(|.| + eps)`` to the slack magnitudes for
            better dynamic range. Set to False for raw magnitudes.

    Returns:
        Plotly heatmap figure.
    """
    if not results.nu_vb_history and not results.nu_history:
        return _empty_figure(
            "No constraint-activity data was recorded. Make sure you ran "
            "Problem.solve() with PenalizedTrustRegion + PTRSolver."
        )

    rows = []
    row_labels = []

    n_nodes = results.X[-1].shape[0] if results.X else None

    if include_dynamics and results.nu_history:
        nu = np.abs(np.asarray(results.nu_history[iteration]))
        # nu has shape (N-1, n_x). Pad column to align with the N-node grid.
        for j in range(nu.shape[1]):
            row = np.concatenate([[0.0], nu[:, j]])
            rows.append(row)
            row_labels.append(f"dyn[{j}]")

    if results.nu_vb_history:
        per_iter = results.nu_vb_history[iteration]
        names = (
            results.constraint_names
            if results.constraint_names and len(results.constraint_names) >= len(per_iter)
            else [f"nodal[{i}]" for i in range(len(per_iter))]
        )
        for i, slack in enumerate(per_iter):
            arr = np.abs(np.asarray(slack)).reshape(-1)
            if n_nodes is not None and arr.size != n_nodes:
                # Pad/truncate as needed.
                tmp = np.zeros(n_nodes)
                tmp[: min(arr.size, n_nodes)] = arr[: min(arr.size, n_nodes)]
                arr = tmp
            rows.append(arr)
            row_labels.append(names[i])

    if results.nu_vb_cross_history:
        per_iter = results.nu_vb_cross_history[iteration]
        cross_names = (
            results.cross_node_constraint_names
            if results.cross_node_constraint_names
            and len(results.cross_node_constraint_names) >= len(per_iter)
            else [f"cross[{i}]" for i in range(len(per_iter))]
        )
        for i, val in enumerate(per_iter):
            arr = np.full(n_nodes if n_nodes is not None else 1, abs(float(val)))
            rows.append(arr)
            row_labels.append(cross_names[i])

    if not rows:
        return _empty_figure(
            "No nonconvex constraint slacks at this iteration. Convex "
            "constraints are handled by CVXPy directly and have no slacks."
        )

    Z = np.vstack(rows)
    if log_scale:
        eps = 1e-12
        Z = np.log10(Z + eps)
        colorbar_title = "log10|slack|"
    else:
        colorbar_title = "|slack|"

    x_axis = np.arange(Z.shape[1])
    if "time" in results.nodes:
        t_nodes = np.asarray(results.nodes["time"]).reshape(-1)
        if t_nodes.size == Z.shape[1]:
            x_axis = t_nodes

    fig = go.Figure(
        data=go.Heatmap(
            z=Z,
            x=x_axis,
            y=row_labels,
            colorscale="Magma",
            colorbar={"title": colorbar_title},
        )
    )
    fig.update_layout(
        title=f"Constraint activity (iteration {iteration})",
        template="plotly_dark",
        xaxis_title="Node / Time",
        yaxis_title="Constraint",
        height=max(300, 30 * len(row_labels) + 150),
    )
    return fig


# =============================================================================
# CTCS time-series
# =============================================================================


def plot_ctcs_timeseries(
    problem: "Problem",
    results: "OptimizationResults",
) -> go.Figure:
    """Continuous-time CTCS violation time series.

    Plots each augmented CTCS state along the propagated trajectory. CTCS
    augmented states accumulate the integral of the constraint penalty over
    time, so a strictly increasing curve indicates ongoing violation; a flat
    curve indicates the constraint is satisfied.

    Args:
        problem: The problem instance (needed for the CTCS slice metadata).
        results: Results from
            :meth:`~openscvx.problem.Problem.post_process`. Must contain
            ``x_full`` and ``t_full`` (i.e. ``post_process`` was called).

    Returns:
        Plotly figure with one line per CTCS group.
    """
    if results.x_full is None or results.t_full is None:
        return _empty_figure(
            "results.x_full / results.t_full are missing. Call problem.post_process()."
        )

    sim = problem.settings.sim
    ctcs_slice = sim.ctcs_slice_prop if hasattr(sim, "ctcs_slice_prop") else sim.ctcs_slice
    if ctcs_slice is None:
        return _empty_figure("This problem has no CTCS constraints.")

    x_full = np.asarray(results.x_full)
    t_full = np.asarray(results.t_full).reshape(-1)
    ctcs_block = x_full[:, ctcs_slice]
    n_groups = ctcs_block.shape[1] if ctcs_block.ndim > 1 else 1

    fig = go.Figure()
    for g in range(n_groups):
        y = ctcs_block[:, g] if n_groups > 1 else ctcs_block
        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=y,
                mode="lines",
                name=f"CTCS[{g}]",
            )
        )

    fig.update_layout(
        title="CTCS augmented state (continuous-time violation accumulator)",
        template="plotly_dark",
        xaxis_title="Time (s)",
        yaxis_title="Accumulated penalty",
        height=400,
    )
    return fig


# =============================================================================
# Active-set bar chart
# =============================================================================


def plot_active_set(
    results: "OptimizationResults",
    iteration: int = -1,
    *,
    tol: float = 1e-4,
) -> go.Figure:
    """Bar chart showing how many nodes each (nonconvex) constraint is active at.

    A constraint is considered "active" at a node when its virtual-buffer slack
    magnitude exceeds ``tol``. Useful for spotting which constraints actually
    drove the SCP iterate (versus those that were satisfied with margin and
    never bound).

    Args:
        results: Results from :meth:`~openscvx.problem.Problem.solve`.
        iteration: SCP iteration to display. Defaults to ``-1``.
        tol: Activation threshold on slack magnitude.

    Returns:
        Plotly bar chart.
    """
    if not results.nu_vb_history:
        return _empty_figure(
            "No nonconvex nodal constraints to display in the active set."
        )

    per_iter = results.nu_vb_history[iteration]
    counts = []
    labels = []
    for i, slack in enumerate(per_iter):
        arr = np.abs(np.asarray(slack)).reshape(-1)
        counts.append(int(np.sum(arr > tol)))
        if results.constraint_names and i < len(results.constraint_names):
            labels.append(results.constraint_names[i])
        else:
            labels.append(f"nodal[{i}]")

    fig = go.Figure(
        data=go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker_color="#7e57c2",
        )
    )
    fig.update_layout(
        title=f"Active-set size per constraint (iteration {iteration}, tol={tol:g})",
        template="plotly_dark",
        xaxis_title="# nodes where slack > tol",
        yaxis_title="Constraint",
        height=max(300, 25 * len(labels) + 150),
    )
    return fig


# =============================================================================
# Helpers
# =============================================================================


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14},
    )
    fig.update_layout(template="plotly_dark", height=200)
    return fig
