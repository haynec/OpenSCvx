"""Trajectory visualization and plotting utilities.

This module provides reusable building blocks for visualizing trajectory
optimization results. It is intentionally minimal - we provide common utilities
that can be composed together, not a complete solution that tries to do
everything for you.

**2D Plots** (plotly-based):
    Two-layer API for time series visualization::

        from openscvx.plotting import plot_states, plot_controls, plot_vector_norm

        # High-level: subplot grid with individual scaling per component
        plot_states(results, ["position", "velocity"]).show()
        plot_controls(results, ["thrust"]).show()

        # Low-level: single component
        plot_state_component(results, "position", component=2).show()  # z only

        # Specialized plots
        plot_vector_norm(results, "thrust", bounds=(rho_min, rho_max)).show()
        plot_projections_2d(results, velocity_var_name="velocity").show()

**Publication styling**:
    ``plot_states``, ``plot_controls``, and ``plot_scp_iterations`` accept
    ``style="publication"`` (white) or ``style="publication_dark"``. Figures you
    assemble yourself get the same look from ``apply_publication_plotly_layout``
    for Plotly and ``apply_latin_modern_to_axis`` for matplotlib, with the
    palettes from ``publication_trace_colors`` / ``publication_dark_colors``.

**3D Visualization** (viser-based):
    The ``viser`` submodule provides composable primitives for building
    interactive 3D visualizations. See ``openscvx.plotting.viser`` for details::

        from openscvx.plotting import viser
        server = viser.create_server(positions)
        viser.add_gates(server, gate_vertices)
        server.sleep_forever()

For problem-specific visualization examples (drones, rockets, etc.), see
``examples/plotting_viser.py``.
"""

try:
    from . import viser
except ModuleNotFoundError as e:
    # Make viser an optional dependency so 2D plotting works without it.
    # Accessing openscvx.plotting.viser will raise a helpful error message.
    class _MissingOptionalDependency:
        def __init__(self, *, package: str, exc: Exception):
            self._package = package
            self._exc = exc

        def __getattr__(self, name: str):
            raise ModuleNotFoundError(
                f"Optional dependency '{self._package}' is required for 3D visualization.\n"
                f"Install it with: pip install {self._package}\n"
                f"Original error: {self._exc}"
            ) from self._exc

        def __repr__(self) -> str:  # pragma: no cover
            return f"<missing optional dependency '{self._package}'>"

    viser = _MissingOptionalDependency(package="viser", exc=e)  # type: ignore[assignment]
from .plotting import (
    plot_control_component,
    plot_controls,
    plot_projections_2d,
    plot_state_component,
    plot_states,
    plot_trust_region_heatmap,
    plot_vector_norm,
    plot_virtual_control_heatmap,
)
from .publication import (
    LM_PLOTLY_FONT,
    LM_PLOTLY_TICK_FONT,
    PlotStyle,
    PublicationFigure,
    VarSpec,
    apply_latin_modern_to_axis,
    apply_publication_plotly_layout,
    latin_modern_fontproperties,
    publication_dark_colors,
    publication_trace_colors,
    show_plotly_with_latin_modern,
)
from .scp_iteration import plot_scp_convergence_histories, plot_scp_iterations

__all__ = [
    # 2D plotting functions (plotly)
    "plot_state_component",
    "plot_states",
    "plot_control_component",
    "plot_controls",
    "plot_projections_2d",
    "plot_vector_norm",
    "plot_trust_region_heatmap",
    "plot_virtual_control_heatmap",
    "plot_scp_iterations",
    "plot_scp_convergence_histories",
    "PublicationFigure",
    "PlotStyle",
    "VarSpec",
    "LM_PLOTLY_FONT",
    "LM_PLOTLY_TICK_FONT",
    "latin_modern_fontproperties",
    "show_plotly_with_latin_modern",
    "apply_publication_plotly_layout",
    "apply_latin_modern_to_axis",
    "publication_trace_colors",
    "publication_dark_colors",
    # 3D visualization submodule (viser)
    "viser",
]
