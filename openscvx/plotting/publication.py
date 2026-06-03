"""Publication-style plotting helpers (white theme, Latin Modern, PDF export)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go

from openscvx.algorithms import OptimizationResults

PlotStyle = Literal["dark", "publication"]
VarSpec = str | tuple[str, int]

_LM_PLOTLY_FAMILY = "Latin Modern Roman"
LM_PLOTLY_FONT = {"family": f"{_LM_PLOTLY_FAMILY}, LM Roman 10, serif", "size": 12}
LM_PLOTLY_TICK_FONT = {"family": f"{_LM_PLOTLY_FAMILY}, LM Roman 10, serif", "size": 11}

_TIME_LABEL = r"$t\,\mathrm{(s)}$"

_COMPONENT_LATEX: dict[str, list[str]] = {
    "position": [r"$x$", r"$y$", r"$z$"],
    "qpos": [r"$x$", r"$y$", r"$z$", r"$q_w$", r"$q_x$", r"$q_y$", r"$q_z$"],
    "velocity": [r"$v_x$", r"$v_y$", r"$v_z$"],
    "angular_velocity": [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"],
    "attitude": [r"$q_w$", r"$q_x$", r"$q_y$", r"$q_z$"],
    "thrust_force": [r"$f_x$", r"$f_y$", r"$f_z$"],
    "thrust": [r"$f_x$", r"$f_y$", r"$f_z$"],
    "torque": [r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"],
}

_PUBLICATION_COLORS = {
    "trajectory": "#228833",
    "nodes": "#4477AA",
    "nodes_prior": "#CC6677",
    "bounds": "#CC3311",
    "impulses": "#EE7733",
}

# Fixed publication geometry: each subplot panel is the same size in px and inches.
_PUBLICATION_PANEL_W_PX = 320
_PUBLICATION_PANEL_H_PX = 280
_PUBLICATION_DPI = 100
_PUBLICATION_MARGIN = {"l": 64, "r": 64, "t": 56, "b": 64}
PUBLICATION_LEGEND_EXTRA_W = 140


def publication_grid_size(
    n_rows: int,
    n_cols: int,
    *,
    extra_legend_width: int = 0,
) -> tuple[int, int]:
    """Return fixed Plotly figure ``(width, height)`` in pixels for a subplot grid."""
    width = (
        n_cols * _PUBLICATION_PANEL_W_PX
        + _PUBLICATION_MARGIN["l"]
        + _PUBLICATION_MARGIN["r"]
        + extra_legend_width
    )
    height = (
        n_rows * _PUBLICATION_PANEL_H_PX
        + _PUBLICATION_MARGIN["t"]
        + _PUBLICATION_MARGIN["b"]
    )
    return width, height


def publication_mpl_figsize(n_rows: int, n_cols: int) -> tuple[float, float]:
    """Return fixed matplotlib ``figsize`` in inches for a subplot grid."""
    return (
        n_cols * _PUBLICATION_PANEL_W_PX / _PUBLICATION_DPI,
        n_rows * _PUBLICATION_PANEL_H_PX / _PUBLICATION_DPI,
    )


def parse_var_spec(spec: VarSpec) -> tuple[str, int | None]:
    """Parse a variable spec into ``(name, component)`` where component is None for all."""
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise ValueError(f"Variable tuple spec must be (name, component), got {spec!r}")
        name, component = spec
        if not isinstance(name, str):
            raise TypeError(f"Variable name must be str, got {type(name).__name__}")
        if not isinstance(component, int):
            raise TypeError(f"Component index must be int, got {type(component).__name__}")
        return name, component

    if not isinstance(spec, str):
        raise TypeError(f"Variable spec must be str or (str, int), got {type(spec).__name__}")

    if ":" in spec:
        name, comp_str = spec.rsplit(":", 1)
        if not name:
            raise ValueError(f"Invalid variable spec {spec!r}")
        return name, int(comp_str)

    if "[" in spec and spec.endswith("]"):
        name, comp_str = spec[:-1].split("[", 1)
        if not name:
            raise ValueError(f"Invalid variable spec {spec!r}")
        return name, int(comp_str)

    return spec, None


def _get_var_dim_from_obj(var) -> int:
    s = var._slice
    if isinstance(s, slice):
        return (s.stop or 1) - (s.start or 0)
    return 1


def expand_var_components(
    result: OptimizationResults,
    specs: list[VarSpec] | None,
    variables: list,
    *,
    include_private: bool = False,
) -> list[tuple[str, str, int]]:
    """Expand variable specs into ``(display_name, var_name, component_idx)`` entries."""
    from .plotting import _get_var, _get_var_dim

    filtered = list(variables)
    if not include_private:
        filtered = [v for v in filtered if not v.name.startswith("_")]

    if specs is None:
        components: list[tuple[str, str, int]] = []
        for var in filtered:
            dim = _get_var_dim(result, var.name, variables)
            if dim == 1:
                components.append((var.name, var.name, 0))
            else:
                for i in range(dim):
                    components.append((f"{var.name}_{i}", var.name, i))
        if not components:
            raise ValueError("No variable components to plot")
        return components

    available = {v.name for v in filtered}
    components = []
    for spec in specs:
        var_name, comp = parse_var_spec(spec)
        if var_name not in available:
            raise ValueError(f"Variable '{var_name}' not found. Available: {sorted(available)}")

        dim = _get_var_dim(result, var_name, variables)
        if comp is not None:
            if comp < 0 or comp >= dim:
                raise ValueError(
                    f"Component {comp} out of range for '{var_name}' (dim={dim})"
                )
            display = latex_component_label(var_name, comp) if dim > 1 else var_name
            components.append((display, var_name, comp))
        elif dim == 1:
            components.append((var_name, var_name, 0))
        else:
            for i in range(dim):
                components.append((f"{var_name}_{i}", var_name, i))

    if not components:
        raise ValueError("No variable components to plot")
    return components


def latex_component_label(var_name: str, component: int, *, dim: int | None = None) -> str:
    """Return a LaTeX axis label for a state/control component."""
    labels = _COMPONENT_LATEX.get(var_name)
    if labels is not None and 0 <= component < len(labels):
        return labels[component]
    if dim == 1:
        escaped = var_name.replace("_", r"\ ")
        return rf"$\mathrm{{{escaped}}}$"
    escaped = var_name.replace("_", r"\ ")
    return rf"$\mathrm{{{escaped}}}_{{{component}}}$"


def _find_latin_modern_otf() -> Path | None:
    """Locate Latin Modern Roman regular OTF (Font Book or MacTeX / TeX Live)."""
    for path in (
        Path.home() / "Library/Fonts/lmroman10-regular.otf",
        Path.home() / "Library/Fonts/lmroman12-regular.otf",
    ):
        if path.is_file():
            return path

    for name in ("lmroman10-regular.otf", "lmroman12-regular.otf"):
        try:
            proc = subprocess.run(
                ["kpsewhich", name],
                check=True,
                capture_output=True,
                text=True,
            )
            path = Path(proc.stdout.strip())
            if path.is_file():
                return path
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            pass

        for root in (
            Path("/Library/TeX/texmf-dist/fonts/opentype/public/lm"),
            Path("/usr/local/texlive"),
        ):
            if root.is_file() and root.name == name:
                return root
            if root.is_dir():
                hit = root / name
                if hit.is_file():
                    return hit
            if root.exists():
                for hit in root.rglob(name):
                    return hit
    return None


def latin_modern_fontproperties():
    """Matplotlib FontProperties for Latin Modern Roman (PDF export)."""
    from matplotlib import font_manager

    otf = _find_latin_modern_otf()
    if otf is None:
        return None
    font_manager.fontManager.addfont(str(otf))
    return font_manager.FontProperties(fname=str(otf))


def _latin_modern_plotly_font_css() -> str:
    """Embed Latin Modern OTF so Plotly's browser renderer can use it."""
    import base64

    otf = _find_latin_modern_otf()
    if otf is None:
        return ""
    data = base64.b64encode(otf.read_bytes()).decode("ascii")
    family = _LM_PLOTLY_FAMILY  # noqa: keep private here; it's only the font name string
    return f"""
@font-face {{
  font-family: '{family}';
  src: url(data:font/opentype;base64,{data}) format('opentype');
  font-weight: normal;
  font-style: normal;
}}
.js-plotly-plot .plotly .main-svg {{
  font-family: '{family}', serif !important;
}}
"""


def apply_publication_plotly_layout(
    fig: go.Figure,
    *,
    n_rows: int = 1,
    n_cols: int = 1,
    extra_legend_width: int = 0,
) -> None:
    """Apply white theme, Latin Modern fonts, LaTeX-friendly defaults, and fixed size."""
    width, height = publication_grid_size(
        n_rows,
        n_cols,
        extra_legend_width=extra_legend_width,
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=LM_PLOTLY_FONT,
        autosize=False,
        width=width,
        height=height,
        margin=_PUBLICATION_MARGIN,
    )
    fig.update_xaxes(
        title_font=LM_PLOTLY_FONT,
        tickfont=LM_PLOTLY_TICK_FONT,
        gridcolor="rgba(0,0,0,0.08)",
        zerolinecolor="rgba(0,0,0,0.15)",
    )
    fig.update_yaxes(
        title_font=LM_PLOTLY_FONT,
        tickfont=LM_PLOTLY_TICK_FONT,
        gridcolor="rgba(0,0,0,0.08)",
        zerolinecolor="rgba(0,0,0,0.15)",
    )


def show_plotly_with_latin_modern(fig: go.Figure) -> None:
    """Open a Plotly figure in the browser with Latin Modern for axes and legend."""
    from plotly.io import to_html
    from plotly.io._base_renderers import open_html_in_browser

    css = _latin_modern_plotly_font_css()
    if not css:
        print(
            "[plot] Latin Modern OTF not found; opening plot with default Plotly fonts."
        )
        fig.show()
        return

    html = to_html(fig, include_plotlyjs=True, full_html=True)
    html = html.replace("</head>", f"<style>{css}</style></head>", 1)
    open_html_in_browser(html)


def publication_trace_colors() -> dict[str, str]:
    return dict(_PUBLICATION_COLORS)


class PublicationFigure:
    """Plotly figure wrapper with Latin Modern ``show()`` and matplotlib ``save_pdf()``."""

    __slots__ = ("_fig", "_save_pdf_fn")

    def __init__(self, fig: go.Figure, save_pdf_fn) -> None:
        self._fig = fig
        self._save_pdf_fn = save_pdf_fn

    def show(self, *args, **kwargs) -> None:
        show_plotly_with_latin_modern(self._fig)

    def save_pdf(self, path: str | Path | None = None) -> None:
        self._save_pdf_fn(path)

    def __getattr__(self, name: str):
        return getattr(self._fig, name)


def _resolve_pdf_path(path: str | Path | None, default_name: str) -> Path:
    out = Path(path) if path is not None else Path("figures") / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _apply_lm_to_matplotlib_axis(ax, lm_fp) -> None:
    if lm_fp is None:
        return
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(lm_fp)
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=lm_fp)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=lm_fp)
    title = ax.get_title()
    if title:
        ax.set_title(title, fontproperties=lm_fp)


def save_timeseries_pdf(
    result: OptimizationResults,
    components: list[tuple[str, str, int]],
    *,
    var_list: list,
    path: str | Path,
    suptitle: str,
    cols: int,
    split_nodes: bool = False,
    impulsive_fn=None,
    plot_trajectory_fn=None,
) -> None:
    """Save state/control time-series subplots as a PDF via matplotlib."""
    import matplotlib.pyplot as plt

    from .plotting import _get_var

    lm_fp = latin_modern_fontproperties()
    if lm_fp is None:
        print(
            "[plot] Latin Modern OTF not found; PDF will use matplotlib default serif."
        )

    n_panels = len(components)
    n_cols = min(cols, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=publication_mpl_figsize(n_rows, n_cols),
        dpi=_PUBLICATION_DPI,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    t_nodes = result.nodes["time"].flatten()
    has_time_traj = bool(result.trajectory) and "time" in result.trajectory
    t_full = result.trajectory["time"].flatten() if has_time_traj else None

    for idx, (display_name, var_name, comp_idx) in enumerate(components):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        ax.set_facecolor("white")

        dim = _get_var_dim_from_obj(_get_var(result, var_name, var_list))
        ylabel = latex_component_label(var_name, comp_idx, dim=dim)

        has_trajectory = bool(result.trajectory) and var_name in result.trajectory
        plot_trajectory = plot_trajectory_fn(var_name) if plot_trajectory_fn else True

        if has_trajectory and plot_trajectory and t_full is not None:
            data = result.trajectory[var_name]
            y = data if data.ndim == 1 else data[:, comp_idx]
            ax.plot(
                t_full,
                y,
                color=_PUBLICATION_COLORS["trajectory"],
                linewidth=1.8,
                label="Trajectory",
            )

        if var_name in result.nodes:
            data = result.nodes[var_name]
            y = data if data.ndim == 1 else data[:, comp_idx]
            impulsive = impulsive_fn(var_name) if impulsive_fn else False

            if split_nodes and len(t_nodes) > 0:
                ax.plot(
                    t_nodes[:1],
                    y[:1],
                    linestyle="None",
                    marker="D",
                    markersize=5,
                    color=_PUBLICATION_COLORS["nodes_prior"],
                    label="Prior state",
                )
                if len(t_nodes) > 1:
                    ax.plot(
                        t_nodes[1:],
                        y[1:],
                        linestyle="None",
                        marker="o",
                        markersize=4,
                        color=_PUBLICATION_COLORS["nodes"],
                        label="Posterior state",
                    )
            else:
                ax.plot(
                    t_nodes,
                    y,
                    linestyle="None",
                    marker="o",
                    markersize=4,
                    color=_PUBLICATION_COLORS["nodes"],
                    label="Nodes",
                )

            if impulsive:
                for k in range(len(t_nodes)):
                    if np.abs(y[k]) > 0:
                        ax.plot(
                            [t_nodes[k], t_nodes[k]],
                            [0.0, y[k]],
                            color=_PUBLICATION_COLORS["impulses"],
                            linewidth=1.2,
                            linestyle="--",
                        )

        var = _get_var(result, var_name, var_list)
        min_val = var.min[comp_idx] if var.min is not None else None
        max_val = var.max[comp_idx] if var.max is not None else None
        for bound_val in (min_val, max_val):
            if bound_val is not None and np.isfinite(bound_val):
                ax.axhline(
                    bound_val,
                    color=_PUBLICATION_COLORS["bounds"],
                    linewidth=1.0,
                    linestyle="--",
                )

        ax.set_xlabel(_TIME_LABEL, fontproperties=lm_fp)
        ax.set_ylabel(ylabel, fontproperties=lm_fp)
        _apply_lm_to_matplotlib_axis(ax, lm_fp)

        if idx == 0:
            leg = ax.legend(loc="best", frameon=False, fontsize=8, prop=lm_fp)
            if lm_fp is not None and leg is not None:
                for text in leg.get_texts():
                    text.set_fontproperties(lm_fp)

    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontproperties=lm_fp, y=1.02)

    out = _resolve_pdf_path(path, "timeseries.pdf")
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] Saved figure to {out.resolve()}")


def save_scp_iterations_pdf(
    result: OptimizationResults,
    *,
    expanded_states: list[dict],
    expanded_controls: list[dict],
    n_state_cols: int,
    n_control_cols: int,
    n_state_rows: int,
    n_control_rows: int,
    n_iterations: int,
    time_slice,
    X_prop_history: list | None,
    path: str | Path,
    cmap_name: str = "viridis",
) -> None:
    """Save SCP iteration overlay plot as a PDF via matplotlib."""
    import matplotlib.pyplot as plt

    from .plotting import _get_var

    lm_fp = latin_modern_fontproperties()
    if lm_fp is None:
        print(
            "[plot] Latin Modern OTF not found; PDF will use matplotlib default serif."
        )

    n_states = len(expanded_states)
    n_controls = len(expanded_controls)
    total_rows = n_state_rows + n_control_rows
    max_cols = max(n_state_cols, n_control_cols, 1)

    fig, axes = plt.subplots(
        total_rows,
        max_cols,
        figsize=publication_mpl_figsize(total_rows, max_cols),
        dpi=_PUBLICATION_DPI,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    cmap = plt.get_cmap(cmap_name)

    def iter_color(iter_idx: int) -> tuple[float, float, float, float]:
        return cmap(iter_idx / max(n_iterations - 1, 1))

    for iter_idx in range(n_iterations):
        X_nodes = result.X[iter_idx]
        U_iter = result.U[iter_idx]
        color = iter_color(iter_idx)

        t_nodes = (
            X_nodes[:, time_slice].flatten()
            if time_slice is not None
            else np.linspace(0, result.t_final, X_nodes.shape[0])
        )

        for state_idx, state in enumerate(expanded_states):
            row = state_idx // n_state_cols
            col = state_idx % n_state_cols
            ax = axes[row][col]
            idx = state["idx"]

            if X_prop_history and iter_idx < len(X_prop_history):
                pos_traj = X_prop_history[iter_idx]
                for j in range(pos_traj.shape[1]):
                    segment_times = pos_traj[:, j, time_slice].flatten()
                    segment_states = pos_traj[:, j, idx]
                    ax.plot(segment_times, segment_states, color=color, linewidth=1.0, alpha=0.7)

            ax.plot(
                t_nodes,
                X_nodes[:, idx],
                linestyle="None",
                marker="o",
                markersize=3,
                color=color,
            )

        for control_idx, control in enumerate(expanded_controls):
            row = n_state_rows + (control_idx // n_control_cols)
            col = control_idx % n_control_cols
            ax = axes[row][col]
            idx = control["idx"]
            ax.plot(
                t_nodes,
                U_iter[:, idx],
                linestyle="None",
                marker="o",
                markersize=3,
                color=color,
            )

    t_nodes_final = (
        result.X[-1][:, time_slice].flatten()
        if time_slice is not None
        else np.linspace(0, result.t_final, result.X[-1].shape[0])
    )
    t_min, t_max = t_nodes_final.min(), t_nodes_final.max()

    for state_idx, state in enumerate(expanded_states):
        row = state_idx // n_state_cols
        col = state_idx % n_state_cols
        ax = axes[row][col]
        parent = _get_var(result, state["parent"], result._states)
        comp_idx = state["comp"]
        dim = _get_var_dim_from_obj(parent)
        ax.set_ylabel(latex_component_label(state["parent"], comp_idx, dim=dim), fontproperties=lm_fp)
        for bound_val in (parent.min, parent.max):
            if bound_val is not None and np.isfinite(bound_val[comp_idx]):
                ax.plot(
                    [t_min, t_max],
                    [bound_val[comp_idx], bound_val[comp_idx]],
                    color=_PUBLICATION_COLORS["bounds"],
                    linewidth=1.0,
                    linestyle=":",
                )
        if row == total_rows - 1:
            ax.set_xlabel(_TIME_LABEL, fontproperties=lm_fp)
        _apply_lm_to_matplotlib_axis(ax, lm_fp)

    for control_idx, control in enumerate(expanded_controls):
        row = n_state_rows + (control_idx // n_control_cols)
        col = control_idx % n_control_cols
        ax = axes[row][col]
        parent = _get_var(result, control["parent"], result._controls)
        comp_idx = control["comp"]
        dim = _get_var_dim_from_obj(parent)
        ax.set_ylabel(latex_component_label(control["parent"], comp_idx, dim=dim), fontproperties=lm_fp)
        for bound_val in (parent.min, parent.max):
            if bound_val is not None and np.isfinite(bound_val[comp_idx]):
                ax.plot(
                    [t_min, t_max],
                    [bound_val[comp_idx], bound_val[comp_idx]],
                    color=_PUBLICATION_COLORS["bounds"],
                    linewidth=1.0,
                    linestyle=":",
                )
        if row == total_rows - 1:
            ax.set_xlabel(_TIME_LABEL, fontproperties=lm_fp)
        _apply_lm_to_matplotlib_axis(ax, lm_fp)

    used = n_states + n_controls
    for idx in range(used, total_rows * max_cols):
        row, col = divmod(idx, max_cols)
        if row < total_rows and col < max_cols:
            axes[row][col].set_visible(False)

    out = _resolve_pdf_path(path, "scp_iterations.pdf")
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] Saved figure to {out.resolve()}")


def wrap_publication_figure(
    fig: go.Figure,
    *,
    pdf_path: str | Path | None,
    default_pdf_name: str,
    save_fn,
) -> PublicationFigure | go.Figure:
    """Return a publication wrapper that auto-saves PDF, or the raw figure for dark style."""
    resolved = _resolve_pdf_path(pdf_path, default_pdf_name)

    def _save(path: str | Path | None = None) -> None:
        save_fn(path or resolved)

    _save(resolved)
    return PublicationFigure(fig, _save)
