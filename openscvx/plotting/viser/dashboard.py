"""Viser overlay variant of the constraint-activity dashboard.

Adds two pieces of constraint-activity instrumentation to a Viser scene:

* Per-node coloring of a 3D trajectory by which (nonconvex) constraint is
  most active at each node, with a discrete colorbar legend.
* A side-panel plotly figure (the per-iteration heatmap from
  :mod:`openscvx.plotting.dashboard`) that is synchronized with the existing
  Viser time slider via a vertical line.

Designed to compose with the rest of :mod:`openscvx.plotting.viser`. See
:func:`add_constraint_activity_overlay` for the simplest entry point and
:func:`add_dashboard_panel` for sidebar-only usage.
"""

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..dashboard import plot_constraint_activity
from .plotly_integration import add_animated_plotly_vline

if TYPE_CHECKING:
    import viser

    from openscvx.algorithms.optimization_results import OptimizationResults


_AnimUpdate = Callable[[int], None]


def compute_constraint_activity_per_node(
    results: "OptimizationResults",
    iteration: int = -1,
    *,
    include_dynamics: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Return the index of the most-active (nonconvex) constraint at each node.

    Args:
        results: Results from :meth:`~openscvx.problem.Problem.solve`.
        iteration: SCP iteration to inspect.
        include_dynamics: When True, dynamics defects (``nu``) are included
            in the comparison; otherwise only nodal nonconvex constraints.

    Returns:
        Tuple ``(activity, labels)`` where ``activity`` is an array of length
        ``N`` with integer indices into ``labels`` (or ``-1`` if no constraint
        information is available at that node), and ``labels`` is the human-
        readable name list aligned to those indices.
    """
    rows: List[Tuple[str, np.ndarray]] = []

    n_nodes = results.X[-1].shape[0] if results.X else None
    if n_nodes is None:
        return np.array([], dtype=int), []

    if include_dynamics and results.nu_history:
        nu = np.abs(np.asarray(results.nu_history[iteration]))
        # Aggregate dynamics defects to a single row (max over states).
        agg = np.concatenate([[0.0], nu.max(axis=1)])
        rows.append(("dynamics", agg))

    if results.nu_vb_history:
        per_iter = results.nu_vb_history[iteration]
        names = (
            results.constraint_names
            if results.constraint_names and len(results.constraint_names) >= len(per_iter)
            else [f"nodal[{i}]" for i in range(len(per_iter))]
        )
        for label, slack in zip(names, per_iter):
            arr = np.abs(np.asarray(slack)).reshape(-1)
            if arr.size != n_nodes:
                tmp = np.zeros(n_nodes)
                tmp[: min(arr.size, n_nodes)] = arr[: min(arr.size, n_nodes)]
                arr = tmp
            rows.append((label, arr))

    if not rows:
        return np.full(n_nodes, -1, dtype=int), []

    labels = [name for name, _ in rows]
    Z = np.vstack([row for _, row in rows])
    # Mask zero columns: when no constraint is active, leave -1.
    activity = np.full(n_nodes, -1, dtype=int)
    nonzero_cols = np.any(Z > 0, axis=0)
    if np.any(nonzero_cols):
        activity[nonzero_cols] = np.argmax(Z[:, nonzero_cols], axis=0)
    return activity, labels


def activity_to_colors(
    activity: np.ndarray,
    n_labels: int,
    cmap_name: str = "tab20",
    inactive_color: Tuple[int, int, int] = (110, 110, 110),
) -> np.ndarray:
    """Map per-node constraint indices to RGB colors (uint8).

    Args:
        activity: Output of :func:`compute_constraint_activity_per_node`.
        n_labels: Number of distinct labels.
        cmap_name: Matplotlib qualitative colormap.
        inactive_color: Color used when ``activity[i] == -1`` (no active
            constraint at that node).
    """
    if activity.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    cmap = plt.get_cmap(cmap_name, max(n_labels, 1))
    out = np.zeros((activity.size, 3), dtype=np.uint8)
    for i, idx in enumerate(activity):
        if idx < 0:
            out[i] = inactive_color
        else:
            r, g, b = cmap(idx % cmap.N)[:3]
            out[i] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def add_constraint_activity_overlay(
    server: "viser.ViserServer",
    results: "OptimizationResults",
    *,
    position_state: str = "position",
    iteration: int = -1,
    include_dynamics: bool = False,
    point_size: float = 0.12,
    cmap_name: str = "tab20",
) -> Tuple[object, List[str]]:
    """Render the trajectory's nodes colored by most-active constraint.

    Args:
        server: A :class:`viser.ViserServer`.
        results: SCP results object.
        position_state: Name of a 2D or 3D position-like state.
        iteration: SCP iteration (default: final).
        include_dynamics: Whether to include dynamics defects in the
            argmax color choice.
        point_size: Marker size in scene units.
        cmap_name: Matplotlib qualitative colormap.

    Returns:
        Tuple ``(point_cloud_handle, labels)`` for downstream use (e.g. a
        legend).
    """
    pos = np.asarray(results.nodes[position_state], dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 1)
    if pos.shape[1] == 2:
        z = np.zeros((pos.shape[0], 1))
        pos = np.concatenate([pos, z], axis=1)
    elif pos.shape[1] not in (2, 3):
        raise ValueError(
            f"position_state {position_state!r} must be 2D or 3D, got shape {pos.shape}"
        )

    activity, labels = compute_constraint_activity_per_node(
        results, iteration=iteration, include_dynamics=include_dynamics
    )
    colors = activity_to_colors(activity, len(labels), cmap_name=cmap_name)

    handle = server.scene.add_point_cloud(
        "/dashboard/activity_nodes",
        points=pos.astype(np.float32),
        colors=colors,
        point_size=point_size,
    )

    if labels:
        # Build a small markdown legend so the user can map color -> label.
        cmap = plt.get_cmap(cmap_name, max(len(labels), 1))
        legend_lines = ["**Constraint legend**"]
        for i, name in enumerate(labels):
            r, g, b = cmap(i % cmap.N)[:3]
            hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            legend_lines.append(
                f'<span style="color:{hex_color}">●</span> {name}'
            )
        with server.gui.add_folder("Constraint Activity"):
            server.gui.add_markdown("\n\n".join(legend_lines))

    return handle, labels


def add_dashboard_panel(
    server: "viser.ViserServer",
    results: "OptimizationResults",
    *,
    iteration: int = -1,
    include_dynamics: bool = True,
    log_scale: bool = True,
    folder_name: str = "Constraint Dashboard",
    aspect: float = 1.4,
    time_array: Optional[np.ndarray] = None,
) -> Tuple[object, Optional[_AnimUpdate]]:
    """Embed the per-iteration constraint heatmap in the Viser sidebar.

    When ``time_array`` is provided (typically the propagated trajectory time
    grid), the panel adds a synchronized vertical line that follows whatever
    time slider drives the rest of the scene. Use the returned update
    callback exactly like the ones from
    :mod:`openscvx.plotting.viser.plotly_integration`.

    Args:
        server: A :class:`viser.ViserServer`.
        results: SCP results object.
        iteration: SCP iteration to display.
        include_dynamics: Whether to include dynamics defects in the heatmap.
        log_scale: Apply ``log10`` to the slack magnitudes.
        folder_name: GUI folder to put the panel under.
        aspect: Width/height ratio for the panel.
        time_array: Optional time array enabling the synchronized vertical
            line. When omitted, no line is drawn and ``update`` is ``None``.

    Returns:
        Tuple ``(panel_handle, update_callback_or_none)``.
    """
    fig = plot_constraint_activity(
        results, iteration=iteration, include_dynamics=include_dynamics, log_scale=log_scale
    )

    if time_array is not None:
        return add_animated_plotly_vline(
            server,
            fig,
            time_array=np.asarray(time_array, dtype=float).reshape(-1),
            line_color="cyan",
            line_dash="dash",
            annotation_text="t",
            folder_name=folder_name,
            aspect=aspect,
        )

    with server.gui.add_folder(folder_name):
        plot_handle = server.gui.add_plotly(figure=fig, aspect=aspect)
    return plot_handle, None
